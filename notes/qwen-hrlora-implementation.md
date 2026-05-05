# HR-LoRA on Qwen1.5-MoE-A2.7B — Full Implementation Notes

## Model identity

`Qwen/Qwen1.5-MoE-A2.7B` — 2.7B active / 14.3B total parameters, 28.65 GB on-GPU (bfloat16).

| Property | Value |
|---|---|
| Layers | 24 |
| Routing experts per layer | 60 |
| Shared experts per layer | 1 (always-on) |
| Top-k routing | 4 |
| Model dim | 2048 |
| FFN dim (routing experts) | 1408 |
| FFN dim (shared expert) | 5632 |
| Uniform routing baseline | 4/60 ≈ 6.7% |

---

## Architecture inspection findings

These were confirmed in `qwen-architecture-inspection.ipynb` before writing any adapter code.

**Expert container:** `model.model.layers[i].mlp.experts` — class `Qwen2MoeExperts`.
Packed tensor layout, same 3D structure as OLMoE:
```
down_proj: [60, 2048, 1408]   # [num_routing_experts, out_f=model_dim, in_f=ffn_dim]
gate_up_proj: [60, 1408, 4096]
```

**Calling convention:** `experts(hidden_states, top_k_index, top_k_weights)` — identical to OLMoE. Drop-in replacement works directly.

**Shared expert:** `model.model.layers[i].mlp.shared_expert` — class `Qwen2MoeMLP`.
Gate: `model.model.layers[i].mlp.shared_expert_gate` — shape `[1, 2048]`, applied as
`torch.sigmoid(gate(h)) * shared_expert(h)`.

The MoE block calls these AFTER returning from `experts(...)`:
```python
# inside Qwen2MoeSparseMoeBlock.forward():
expert_output  = self.experts(hidden_states, top_k_index, top_k_weights)   # ← our wrapper intercepts here
shared_output  = torch.sigmoid(self.shared_expert_gate(hidden_states)) * self.shared_expert(hidden_states)
output = expert_output + shared_output
```
We never see the shared expert path. It remains frozen and outside our wrapper.

**Router logits:** `output_router_logits=True` returns a tuple of 24 tensors, each `[T, 60]`.
This covers routing experts only — shared expert is NOT included in the logits output.

**LoRA dimension:** Must be `out_f = 2048` (model_dim), NOT `in_f = 1408` (ffn_dim).
Same rule as the Phi→OLMoE port: LoRA correction lives in the model-dim residual stream.

---

## Jaccard diagnostic results

`qwen-jaccard-diagnostic.ipynb` — run on Colab A100. Config saved to `notebook_outputs/qwen_jaccard-test/qwen_experiment_config.json`.

### Calibration

Qwen uses top-4 of 60 routing experts (6.7% uniform baseline vs 12.5% for Phi and OLMoE).
Top-N for Jaccard set to 15 (top quartile: 15/60 = 25%, matching OLMoE's 16/64 = 25%).

### Results

| Domain Pair | Phi-3.5-MoE | OLMoE-1B-7B | Qwen1.5-MoE |
|---|---|---|---|
| Python vs Medical | 0.056 | 0.103 | **0.071** ← chosen |
| Math vs Creative | 0.069 | 0.053 | 0.132 |
| Null (same domain) | ~1.0 | 0.840 | **0.510** |
| Separation ratio (chosen) | 0.056x | 0.063x | **0.140x** |

Qwen's null baseline (0.510) is lower than OLMoE's (0.840) because fewer experts fire per token
(4/60 vs 8/64), so same-domain examples are less likely to share the same top-15 experts by chance.

**Python vs Medical chosen** (Option A): larger gap from null (0.510 − 0.071 = 0.439)
vs Math vs Creative (0.510 − 0.132 = 0.378). Python/Medical shows stronger domain separation.

### Per-layer Jaccard (Python vs Medical)

| Layer | Jaccard | Notes |
|---|---|---|
| 0 | 0.1538 | **TARGET** ← highest |
| 2 | 0.1111 | top-3 |
| 3 | 0.1111 | top-3 |
| 6 | 0.1111 | |
| 11 | 0.1111 | |
| 16 | 0.1111 | |
| … | | |
| Mean | 0.0714 | |

**Target layer: 0 | Target expert: 5**

Expert 5 geometric mean conflict score at Layer 0: 0.0236
(Freq_python = 0.0230, Freq_medical = 0.0242)

**Interpretation:** Qwen's routing is strongly domain-separated (0.14× null). Gradient
entanglement is a gradient-space problem, not a routing-space problem. HR-LoRA spawning
targets gradient-space conflict — the method generalises directly.

---

## What changed from the OLMoE implementation

### HierarchicalQwenExperts wrapper

| Change | Detail |
|---|---|
| Class name | `HierarchicalOLMoEExperts` → `HierarchicalQwenExperts` |
| `num_experts` | 64 → 60 (routing only; shared expert excluded) |
| `lora_dim` | 2048 → 2048 (numerically identical — no actual change) |
| Domain pair | Math/Creative → Python/Medical |
| `AlignDevicesHook` transfer | Removed — not needed on single A100 |

### What did NOT change

| Component | Status |
|---|---|
| `HierarchicalExpert` (LoRA sub-adapter) | Zero changes — architecture-agnostic |
| `ConflictSaturationMonitor` | Zero changes — reads only LoRA norms and loss values |
| Calling convention | `experts(h, idx, wts)` — identical in OLMoE and Qwen |
| Packed tensor layout | 3D structure `[N, out_f, in_f]` — same in both models |
| `lora_dim = out_f = 2048` | Same value — no numerical change |
| B=0 zero-spawn guarantee | Unchanged |
| SVD initialisation of spawned adapters | Unchanged |

### Patch site

```python
# Freeze everything
for p in model.parameters():
    p.requires_grad = False

# Wrap routing expert container only
original_experts = model.model.layers[TARGET_LAYER].mlp.experts
hier = HierarchicalQwenExperts(original_experts, base_rank=16, lora_alpha=32)
model.model.layers[TARGET_LAYER].mlp.experts = hier

# shared_expert and shared_expert_gate are NOT touched
# They remain frozen inside Qwen2MoeSparseMoeBlock and fire independently
```

**Idempotency check:** wrapper includes an unwrap guard so re-running the patch cell
does not double-wrap.

**Patch zero-output verification:** `‖orig − patched‖_max = 0.00000000` — confirmed B=0 init.

---

## device_map issue (resolved)

During Jaccard diagnostic development, `device_map="auto"` on a machine with insufficient
VRAM triggered disk offloading for the packed MoE tensors. Accelerate cannot re-save packed
3D expert tensors in their original format, raising:

```
ValueError: The current device_map had weights offloaded to the disk...
because the model uses an internal weight format different than the one saved (i.e. most MoE models)
```

**Resolution:** The issue did not recur on the Colab A100 (40 GB VRAM — model fits at 28.65 GB).
`device_map="auto"` works correctly when the full model fits on a single GPU with no disk offload.
Do not use `device_map={"": 0}` — the original `device_map="auto"` is correct and should be kept.

---

## Smoke test results

`qwen-smoke-test.ipynb` — 300 steps, Layer 0 Expert 5, 80% Python / 20% Medical interleaved.

| Check | Result |
|---|---|
| Trigger fired | ✅ YES — 7 spawns in 300 steps |
| No crash after spawn | ✅ YES — all 7 spawns clean |
| SVD init | ✅ Successful on all 7 spawns |
| \|ΔLoss\| at spawn | ⚠️ 0.007–0.059 (see note below) |

Spawn steps: ~116, 145, 174, 207, 236, 265, 294.

**|ΔLoss| measurement note:** The threshold check (`< 0.01`) shows ⚠️ for most spawns.
This is a measurement artifact: `loss_before_spawn` is recorded before `optimizer.step()`,
but `loss_after` is measured after the optimizer has already updated the base LoRA weights.
The delta includes the optimizer step's contribution, not just the spawn.
The B=0 guarantee is a mathematical certainty from the code — not an empirical claim.

**7 spawns in 300 steps is aggressive.** The EMA divergence grew large
(EMA_code ≈ 8–14, EMA_medical dropped to ≈ 2.7 → gap > 5.6, well above threshold 1.0).
For full training experiments, tune `delta_threshold` upward (e.g. 2.0–3.0) to reduce
spawn frequency. For the smoke test, frequent spawning confirms the monitor is working.

---

## Comparison table: all three models

| Component | Phi-3.5-MoE | OLMoE-1B-7B | Qwen1.5-MoE |
|---|---|---|---|
| Routing experts | 16 | 64 | 60 |
| Shared expert | None | None | 1 (always-on) |
| Top-k routing | 2 | 8 | 4 |
| Uniform baseline | 12.5% | 12.5% | 6.7% |
| Layers | 32 | 16 | 24 |
| Jaccard method | raw set | top-16 | top-15 |
| Chosen domain pair | Python/Medical | Math/Creative | Python/Medical |
| Chosen Jaccard | 0.056 | 0.053 | 0.071 |
| Null baseline | ~1.0 | 0.840 | 0.510 |
| Separation ratio | 0.056x | 0.063x | 0.140x |
| Target layer | — | 6 | 0 |
| Target expert | — | 18 | 5 |
| `HierarchicalExpert` | original | unchanged | unchanged |
| `ConflictSaturationMonitor` | original | unchanged | unchanged |
| dtype issue | none | float16→bf16 | none (bf16 throughout) |
| Multi-GPU hook | N/A | AlignDevicesHook | not needed (single A100) |

---

## Notebook run order

1. `qwen-architecture-inspection.ipynb` — inspect model, confirm dimensions (local or Colab, no GPU needed for inspection)
2. `qwen-jaccard-diagnostic.ipynb` — routing overlap diagnostic (A100, ~15 min) → saves `qwen_experiment_config.json`, `qwen_freq_cache.pkl`
3. `qwen-smoke-test.ipynb` — 300-step training verification (A100, ~20 min) → reads config, saves `qwen_smoke_test_loss.png`
4. `qwen-task3-results-table.ipynb` — format results tables for thesis (CPU, ~5 sec) → reads config + freq_cache

All outputs stored in `notebook_outputs/qwen_jaccard-test/`.

---

## Thesis Section 4.2 — Qwen paragraph

> Qwen1.5-MoE's routing is strongly domain-separated (top-K Jaccard 0.071, separation ratio 0.14×). This is consistent with Phi-3.5-MoE (0.056) and OLMoE (0.053), confirming that domain specialisation in expert routing is a structural property shared across sparse MoE architectures. The Hierarchical ReLU-LoRA spawning mechanism, which targets gradient-space conflict rather than routing-space overlap, generalises cleanly to Qwen's distinct routing configuration (top-4 of 60, 6.7% uniform baseline).

**Footnote / methodological note:**
> Unlike Phi-3.5-MoE and OLMoE, Qwen1.5-MoE includes a permanently-active shared expert that fires on every token alongside the top-4 routing experts. Jaccard analysis targets only the 60 routing experts (shared expert excluded), consistent with the routing analysis in the other two models. The HR-LoRA wrapper patches only the routing expert container; the shared expert path is transparent to the adapter and remains frozen throughout.

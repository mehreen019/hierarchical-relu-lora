# Task 5 — Architectural Differences: Phi-3.5-MoE → OLMoE Port

## Thesis write-up (copy into Section 4.x / Generalization)

Porting the Hierarchical ReLU-LoRA spawning method from Phi-3.5-MoE to
OLMoE-1B-7B required four concrete code changes, none of which touched the
core algorithm.

**Expert tensor layout.** Both models pack all expert weights into a single
fused tensor rather than storing each expert as a separate module. The shape
convention differs: Phi-3.5-MoE uses `[num_experts, ffn_dim, model_dim]`
(16 experts, 6400 × 4096) while OLMoE uses `[num_experts, out_dim, in_dim]`
(64 experts, 2048 × 1024). In both cases the LoRA dimension must be set to
`out_dim` (the model dimension), not `in_dim`. This was the original source of
a shape mismatch bug in the Phi prototype and the same rule was applied
directly to OLMoE.

**Gate and up projection.** Phi-3.5-MoE exposes `gate_proj` and `up_proj` as
separate attributes on the expert block. OLMoE fuses these into a single
`gate_up_proj` tensor of shape `[64, 2048, 2048]` that is chunked internally
with `.chunk(2, dim=-1)`. The wrapper does not interact with these projections
directly — it intercepts only the final `down_proj` output — so no change was
required beyond confirming the attribute names during the architecture
inspection step.

**Routing width.** Phi-3.5-MoE uses top-2 routing over 16 experts per layer;
OLMoE uses top-8 over 64 experts. The broader routing required recalibrating
the Jaccard diagnostic. The original concentration-threshold approach (designed
for Phi's sparse, unbalanced routing) produced zero domain-specific experts on
OLMoE because OLMoE's load-balancing auxiliary loss forces near-uniform
activation across all 64 experts. The fix was to replace the threshold filter
with a top-K Jaccard metric (top-16 of 64 per domain), which correctly captures
domain preference even under flat routing. The diagnostic identified
Math vs Creative as the higher-conflict pair (gap from null = 0.788) with
Layer 6, Expert 18 as the target.

**dtype.** The Phi experiments used `bfloat16` throughout. The initial OLMoE
smoke test loaded the model in `float16`, which caused NaN loss during
backpropagation due to float16's limited dynamic range. Switching to `bfloat16`
resolved the NaN immediately. No other numerical changes were needed.

**ConflictSaturationMonitor domain names.** The monitor was originally written
with hardcoded domain keys (`"code"` and `"medical"`). When the OLMoE
experiment switched to the Math vs Creative domain pair, the primary-domain
check was parameterised so the training loop can pass `primary_domain=DOMAIN_NAMES[0]`
at runtime. The monitor logic itself — dual EMA tracking, OLS plateau
detection, 15-step conflict window — was unchanged.

Everything else transferred without modification: the `HierarchicalExpert`
LoRA sub-adapter, the ReLU gate, the B=0 zero-loss spawn guarantee, the SVD
initialisation of spawned adapters, and the `AlignDevicesHook` transfer
pattern for multi-GPU compatibility.

---

## Smoke test results (Task 4 confirmation)

| Check | Result |
|---|---|
| Trigger fired | ✅ YES — 5 spawns at steps 29, 72, 117, 167, 196 |
| No crash after spawn | ✅ YES — all 5 spawns clean |
| \|ΔLoss\| < 0.01 at spawn | ✅ YES — final spawn \|ΔLoss\| = 0.00927 |

The |ΔLoss| decreased monotonically across spawns (0.032 → 0.025 → 0.019 →
0.014 → 0.009), confirming that each successive sub-adapter is initialised
closer to a neutral perturbation as the base adapter converges.

---

## Summary table

| Component | Changed? | What changed |
|---|---|---|
| `HierarchicalExpert` (LoRA sub-adapter) | No | — |
| `ConflictSaturationMonitor` | Minor | `primary_domain` param added to `update()` |
| `HierarchicalOLMoEExperts` wrapper | Yes | `lora_dim=2048`; no `gate_proj`/`up_proj` refs; 64 experts |
| Jaccard diagnostic | Yes | top-K metric replaces concentration threshold |
| Model dtype | Yes | `float16` → `bfloat16` |
| Expert calling convention | No | `experts(h, idx, wts)` identical in both models |
| AlignDevicesHook transfer | No | Same pattern, same code |
| B=0 spawn guarantee | No | Verified: \|ΔLoss\| = 0.000000 in Phi, 0.009 in OLMoE |

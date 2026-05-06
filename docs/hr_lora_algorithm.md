# HR-LoRA: Hierarchical ReLU-LoRA with Conflict-Triggered Sub-Adapter Spawning

## Overview

HR-LoRA (Hierarchical ReLU-LoRA) is a method for continual multi-domain fine-tuning of
MoE models. When a model trained on domain A begins training on conflicting domain B, a
single LoRA adapter tries to represent both domains with the same weights — causing
*negative transfer*: performance on A degrades as B is learned.

HR-LoRA responds to detected conflict by spawning new, independently-gated sub-adapters.
Each sub-adapter specialises for a subset of the input distribution via a learned ReLU
gate, allowing the model to partition conflicting domains without touching the frozen base
weights.

---

## 1. Architecture

For expert k in layer l, the full output is:

```
E_k(x) = W_k x  +  L_{k,0}(x)  +  Σ_j  ReLU(w_{k,j}^T x) · L_{k,j}(x)
          ───────   ───────────     ──────────────────────────────────────
          frozen    base LoRA       spawned sub-adapters (appear over time)
```

where:
- `W_k` — frozen pre-trained expert weight
- `L_{k,0}` — base LoRA (always present, trained from step 0)
- `L_{k,j}` — j-th spawned LoRA sub-adapter (added during training when conflict is
  detected)
- `w_{k,j}` — ReLU gating vector for sub-adapter j (learned, shape = hidden_size)

Each LoRA sub-adapter `L_{k,j}` is a `HierarchicalExpert`:

```python
class HierarchicalExpert(nn.Module):
    def __init__(self, in_features, out_features, base_rank, lora_alpha):
        self.A = nn.Parameter(torch.randn(base_rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, base_rank))
        self.scaling = lora_alpha / base_rank

    def forward(self, x):
        return (x @ A.T @ B.T) * self.scaling
```

B is always zero-initialised. At the moment a sub-adapter is spawned, its output is
exactly zero — no loss spike.

---

## 2. Forward Pass in Detail

The full wrapper `HierarchicalOLMoEExperts` intercepts the packed expert call:

```python
def forward(self, hidden_states, router_top_k_indices, router_top_k_weights):
    orig_out   = self.original.forward(hidden_states, ...)   # frozen experts
    correction = torch.zeros_like(orig_out)

    for k in range(self.num_experts):
        mask  = (router_top_k_indices == k)
        eff_w = (mask.float() * router_top_k_weights).sum(dim=1)  # [T]

        # Base LoRA
        base_out   = self.base_loras[k](hidden_states)
        correction = correction + base_out * eff_w.unsqueeze(-1)

        # Spawned sub-adapters
        for gate_vec, sub_lora in zip(self.spawn_gates[k], self.spawn_loras[k]):
            g       = F.relu(hidden_states @ gate_vec)               # [T] scalar gate
            sub_out = sub_lora(hidden_states)                         # [T, d]
            correction = correction + sub_out * (g * eff_w).unsqueeze(-1)

    return (orig_out + correction).to(hidden_states.dtype)
```

**Routing weight `eff_w`:** For a token not routed to expert k, `eff_w[token] = 0`, so
the correction is zero for that token regardless of the LoRA output. No masking needed —
zero-weight tokens contribute nothing automatically.

**ReLU gate `g`:** For each token, `g = ReLU(w_{k,j}^T x)`. This is a scalar that
scales the sub-adapter's output for that token. If `w_{k,j}^T x ≤ 0`, the sub-adapter
is completely silent for that token. This is the partitioning mechanism: the gate learns
to activate only for tokens from the domain it specialised for.

**Numerical example.** Hidden_size d = 2048. Expert 0 handles T_k = 10 tokens in this
batch. Suppose one spawned sub-adapter exists with gate vector w_0.

```
Token 1 (code):    w_0^T x_1 = +0.8  →  g_1 = 0.8  (sub-adapter active)
Token 2 (medical): w_0^T x_2 = -0.3  →  g_2 = 0.0  (sub-adapter silent)
Token 3 (code):    w_0^T x_3 = +0.5  →  g_3 = 0.5  (sub-adapter active)
Token 4 (medical): w_0^T x_4 = -0.1  →  g_4 = 0.0  (sub-adapter silent)
```

The sub-adapter effectively handles only code tokens. The base LoRA handles all tokens.
Medical tokens see only the base LoRA. Code tokens see base LoRA + scaled sub-adapter.

---

## 3. The Conflict-Saturation Monitor

Spawning is not manual — it is triggered automatically when the monitor detects that
the base LoRA has both **plateaued** (stopped learning) and **conflict** (two domains
have diverged in loss).

```python
class ConflictSaturationMonitor:
    def __init__(self, tau_plateau=5e-4, delta_threshold=0.15,
                 window=8, beta=0.9):
        self._ri_history      = []   # rank importance over time
        self._ema_code        = None
        self._ema_medical     = None
        self._plateau_window  = []   # booleans: is slope flat?
        self._conflict_window = []   # booleans: are domains diverged?
```

### 3.1 Domain-Separated Loss EMAs

```
ema_code^(t)    = β · ema_code^(t-1)    + (1-β) · loss_t    [when domain == "code"]
ema_medical^(t) = β · ema_medical^(t-1) + (1-β) · loss_t    [when domain == "medical"]
```

These track the smoothed loss for each domain independently. When the model is asked to
learn medical text, `ema_medical` rises. When it forgets code, `ema_code` rises.

### 3.2 Rank Importance (Plateau Detection)

The rank importance ri is computed as the mean Frobenius contribution of A and B:

```python
col_norms = lora_B.detach().float().norm(dim=0)   # (rank,)
row_norms = lora_A.detach().float().norm(dim=1)   # (rank,)
ri = (col_norms * row_norms).mean().item()
```

This is the mean of the product of norms of corresponding columns/rows of B and A.
If the LoRA adapter is actively learning, ri grows. When it plateaus, ri flattens.

### 3.3 Plateau Condition (OLS Slope)

Over the last `window` steps of ri history, compute the ordinary-least-squares slope:

```python
x = torch.arange(window, dtype=torch.float32)
y = torch.tensor(ri_history[-window:], dtype=torch.float32)
slope = ((x*y).mean() - x.mean()*y.mean()) / (x.var(unbiased=False) + 1e-12)
plateau = abs(slope) < tau_plateau   # tau_plateau = 5e-4
```

**Numerical example.** window = 8, tau_plateau = 5e-4.

```
ri history (last 8 steps): [0.0412, 0.0414, 0.0413, 0.0415, 0.0414, 0.0413, 0.0414, 0.0414]

x̄ = 3.5,  ȳ = 0.04138
(x·y).mean() = (0·0.0412 + 1·0.0414 + ... + 7·0.0414) / 8 = 0.14476
slope = (0.14476 - 3.5·0.04138) / var(x) = (0.14476 - 0.14483) / 5.25 = -0.000013
|slope| = 1.3e-5 < 5e-4   →   plateau = True
```

The rank importance has barely moved over 8 steps. The base LoRA has stopped learning.

### 3.4 Conflict Condition

```python
conflict = abs(ema_medical - ema_code) > delta_threshold   # delta_threshold = 0.15
```

**Numerical example.** After 400 steps of mixed code/medical training:

```
ema_code    = 3.82   (code loss: steady, not degrading yet)
ema_medical = 4.21   (medical loss: higher, model still learning this domain)
|4.21 - 3.82| = 0.39 > 0.15   →   conflict = True
```

The two domains have diverged in loss: the model cannot represent both well simultaneously
with the current adapter.

### 3.5 Fire Condition

The monitor fires (returns True) only when **both** conditions hold for **all** `window`
consecutive steps:

```python
if (len(plateau_window) == window
        and all(plateau_window)
        and all(conflict_window)):
    plateau_window.clear()
    conflict_window.clear()
    return True
```

This double-condition requirement prevents false triggers:
- Conflict alone (domains diverged but LoRA still learning) → wait, learning may resolve it
- Plateau alone (LoRA flat but domains aligned) → no conflict to resolve
- Both together → LoRA is stuck, two domains are entangled, spawn needed

**Numerical example.** window = 8. Steps 450–457:

```
Step 450: plateau=T, conflict=T  → windows: P=[T], C=[T]
Step 451: plateau=T, conflict=T  → P=[T,T], C=[T,T]
...
Step 457: plateau=T, conflict=T  → P=[T,T,T,T,T,T,T,T], C=[T,T,T,T,T,T,T,T]
all(P) = True, all(C) = True, len=8 = window   →   SPAWN TRIGGERED
```

---

## 4. Spawn Procedure

When the monitor fires, a new sub-adapter is created for the highest-norm expert:

```python
# Select expert with highest base_lora B norm (most loaded)
expert_norms = [wrapper.base_loras[e].B.detach().float().norm().item()
                for e in range(wrapper.num_experts)]
spawn_expert = argmax(expert_norms)
```

Then `wrapper.spawn(spawn_expert, rank=8, weight_grad=...)` runs:

### 4.1 Gradient-Informed A Initialisation (LoRA-GA)

If gradients are available, initialise A from the SVD of the outer product of B and A
gradients:

```python
weight_grad = grad_B.float() @ grad_A.float()   # [out, in] — residual gradient
U, S, Vh = torch.linalg.svd(weight_grad, full_matrices=False)
lora.A.copy_(Vh[:rank].to(dtype))   # top-r rows of V^H
```

The SVD of the gradient gives the directions in input space along which the loss changes
most. Initialising A to align with these directions means the new sub-adapter starts
learning the most important missing signal immediately.

**Numerical example.** rank = 8, d = 2048. grad_B @ grad_A ∈ ℝ^(2048 × 2048).

```
SVD: top singular values S = [0.84, 0.71, 0.65, 0.58, 0.49, 0.41, 0.35, 0.28, ...]
lora.A ← Vh[:8, :]   (8 rows, each a 2048-dim direction in input space)
lora.B ← zeros       (output contribution = 0 at spawn, always)
```

If SVD fails (e.g. gradient is None), A falls back to small random initialisation.

### 4.2 Gate Initialisation

```python
sigma = 1e-3 * self.original.down_proj[expert_id].float().var().item()
gate = nn.Parameter(torch.randn(self.lora_dim) * sigma)
```

**Why `var()` not `std()`?** The gate must break symmetry (all-zero gate → all tokens
get zero gate score → sub-adapter never activates) but must not disturb existing routing.
`var()` produces a much smaller sigma than `std()`:

```
Typical down_proj std ≈ 0.05   →   var ≈ 0.0025
sigma_correct  = 1e-3 × 0.0025 = 2.5e-6   (tiny — nearly zero gate at spawn)
sigma_wrong    = 1e-2 × 0.050  = 5.0e-4   (200× larger — disrupts routing immediately)
```

The tiny initial gate means the new sub-adapter starts nearly silent and must earn its
activation by gradient descent — a form of importance-weighted competition.

### 4.3 Adding to Optimizer

```python
new_params = list(lora.parameters()) + [gate]
optimizer.add_param_group({'params': new_params, 'lr': lr})
monitor.reset_after_spawn()
```

The new parameters are handed directly to AdamW. The optimizer creates fresh momentum
and variance buffers for them, initialised to zero. No existing parameter state is
disturbed.

**After spawn, the monitor resets:**
```
ri_history.clear()
plateau_window.clear()
conflict_window.clear()
```
This gives the new sub-adapter a clean observation window. The next spawn can only happen
after another `min_spawn_interval` steps (50 in the notebook config) have passed —
a cooldown to prevent cascade spawning.

---

## 5. Multi-Layer Extension

In the notebook, three layers are wrapped simultaneously: TARGET_LAYERS = [4, 8, 12].

```python
for target_layer in [4, 8, 12]:
    original_experts = get_clean_original_experts(model, target_layer)
    wrapper = HierarchicalOLMoEExperts(original_experts, base_rank, base_rank * 2)
    model.model.layers[target_layer].mlp.experts = wrapper
    wrappers.append(wrapper)

all_params = [p for w in wrappers for p in w.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(all_params, lr=lr)
```

All three layers share one optimizer but have independent monitors. The first wrapper
(layer 4) is the primary spawn target; its conflict signal is used to trigger spawning.
Spawned sub-adapters are added to the shared optimizer via `add_param_group`.

---

## 6. Aggregate Rank Importance

Rather than monitoring a single expert's LoRA, the implementation aggregates across the
top-4 most-loaded experts:

```python
ri_vals = []
for eid in range(min(4, wrapper.num_experts)):
    bl = wrapper.base_loras[eid]
    col_norms = bl.B.detach().float().norm(dim=0)
    row_norms = bl.A.detach().float().norm(dim=1)
    ri_vals.append((col_norms * row_norms).mean().item())
agg_ri = sum(ri_vals) / len(ri_vals)
```

**Numerical example.** 4 experts, ri values = [0.041, 0.038, 0.043, 0.036]:
```
agg_ri = (0.041 + 0.038 + 0.043 + 0.036) / 4 = 0.0395
```

This aggregate is passed to the monitor as `override_ri`. Using aggregate ri prevents
noise from a single expert dominating the plateau signal — if expert 0 is flat but
experts 1–3 are still learning, the aggregate correctly reflects ongoing learning.

---

## 7. Spawn Cooldown and Cap

Two safeguards prevent runaway spawning:

**Cooldown:** `min_spawn_interval = 50`. After a spawn at step s, no spawn is allowed
until step s+50. This gives the new sub-adapter time to start learning before the monitor
can fire again.

**Cap:** `max_sub_adapters = 10` per expert. After 10 spawns on a single expert, further
spawns for that expert are suppressed regardless of the monitor signal.

---

## 8. Complete Algorithm Summary

```
Initialise: base_loras[k] for each expert k in target layers
            spawn_loras[k] = [],  spawn_gates[k] = [],  last_spawn_step = -999
            monitor = ConflictSaturationMonitor(tau=5e-4, delta=0.15, window=8)

For each training step t:
  1. Sample batch; identify domain ("code" or "medical")
  2. Forward: loss = model(batch)
  3. loss.backward()
  4. Compute agg_ri from base_loras[0..3]
  5. monitor.update(lora_A, lora_B, loss, domain, override_ri=agg_ri)
  6. If monitor fires AND (t - last_spawn_step) ≥ 50:
       spawn_expert ← argmax_k( ||B_k||_F )
       If len(spawn_loras[spawn_expert]) < 10:
         weight_grad ← B.grad @ A.grad  (or None)
         new_params ← wrapper.spawn(spawn_expert, rank=8, weight_grad)
         optimizer.add_param_group({'params': new_params})
         monitor.reset_after_spawn()
         last_spawn_step ← t
  7. optimizer.step()
```

---

## 9. Key Properties

| Property | Value |
|---|---|
| base_rank | 8 |
| lora_alpha (base) | 16 (= base_rank × 2) |
| lora_alpha (spawned) | 16 (= spawn_rank × 2) |
| spawn_rank | 8 |
| tau_plateau | 5e-4 |
| delta_threshold | 0.15 |
| window | 8 consecutive steps |
| EMA decay β | 0.9 |
| min_spawn_interval | 50 steps |
| max_sub_adapters | 10 per expert |
| Gate init sigma | 1e-3 × Var(W_k) |

**Key guarantee:** B = 0 at every spawn. The output of any newly-spawned sub-adapter is
exactly zero at the moment of creation. The model's predictions are unchanged by the
spawn event — there is no loss spike and no need for a learning rate warmup for the new
parameters.

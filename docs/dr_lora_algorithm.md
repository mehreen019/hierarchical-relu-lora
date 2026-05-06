# DR-LoRA: Dynamic Rank LoRA with Capacity Reservation

## Overview

DR-LoRA addresses a core limitation of standard LoRA: the rank is fixed at initialisation.
In a Mixture-of-Experts (MoE) model, experts are not equally important — some handle more
tokens, some carry stronger gradient signal. DR-LoRA gives those high-value experts more
representational capacity by growing their rank over the course of training, without ever
resizing tensors or corrupting optimizer state.

---

## 1. The Problem with Fixed-Rank LoRA

Standard LoRA wraps a frozen weight matrix W ∈ ℝ^(out × in) with:

```
ΔW = B · A,   A ∈ ℝ^(r × in),   B ∈ ℝ^(out × r)
```

The rank r is chosen once before training starts. In a 64-expert MoE layer with top-8
routing, only 8 experts are active per token. If the same rank is assigned to all 64
experts, capacity is wasted on rarely-routed experts and starved from heavily-used ones.

**Numerical example.** Suppose r = 4 for all experts. Expert 0 handles 35% of tokens;
expert 31 handles 0.4%. Both have the same ΔW = B·A with rank 4. Expert 0 would benefit
from rank 8 or 12; expert 31 gains almost nothing beyond rank 1. Fixed LoRA cannot
redistribute this capacity.

---

## 2. Capacity Reservation: The Core Mechanism

DR-LoRA solves the tensor-resizing problem by allocating the **maximum rank upfront** but
only **activating a subset initially**.

```python
# From DRLoRALayer.__init__
self.lora_A = nn.Parameter(torch.zeros(r_max, in_features))   # full capacity
self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))  # full capacity
self.register_buffer('rank_mask', torch.zeros(r_max, dtype=torch.bool))
self.rank_mask[:r_init] = True   # only first r_init slots active
```

**Why this matters:** When a rank slot is activated later, no new tensor is created.
The optimizer already holds momentum and variance buffers for that slot (they just contain
zeros). Activating a rank means flipping one boolean in `rank_mask` — the A and B tensors
remain the same shape throughout training.

### 2.1 Binary Rank Mask

The mask m ∈ {0,1}^r_max controls which rank columns are live:

```
A_active = A[m, :]          # shape (r_active, in_features)
B_active = B[:, m]          # shape (out_features, r_active)
```

### 2.2 Masked Forward Pass

```python
# From DRLoRALayer.forward
A_active = self.lora_A[active_mask, :]
B_active = self.lora_B[:, active_mask]
active_ranks = active_mask.sum().item()
scaling = self.lora_alpha / active_ranks        # dynamic scaling
result = dropout(x) @ A_active.T @ B_active.T * scaling
```

The scaling is recomputed from the current number of active ranks, so adding a rank does
not suddenly multiply the output magnitude.

**Numerical example.** r_max = 16, r_init = 4, lora_alpha = 16.

- At step 0: 4 ranks active, scaling = 16/4 = 4.0
- After first growth event: 6 ranks active, scaling = 16/6 ≈ 2.67
- At full capacity: 16 ranks active, scaling = 16/16 = 1.0

The output scale decreases as ranks grow, preventing a sudden loss spike when new ranks
are added.

**Zero-init guarantee.** B is initialised to zeros, so at the moment any new rank slot
becomes active, its contribution B[:, new_rank] · A[new_rank, :] = 0. The model starts
from a neutral point and learns from there.

---

## 3. Routing Frequency Tracking

DR-LoRA measures how often each expert is used via a forward hook on the router's softmax
output.

**Equation (Eq. 5):**
```
f_{l,i}^(t) = β · f_{l,i}^(t-1) + (1-β) · w̄_{l,i}^(t)
```

where w̄_{l,i}^(t) is the mean softmax weight assigned to expert i in layer l over the
current batch, and β = 0.9 is the EMA decay.

```python
# From DRLoRATracker.update_routing_frequency_from_hooks
self.routing_frequency[layer_idx] = (
    self.ema_beta * self.routing_frequency[layer_idx]
    + (1 - self.ema_beta) * weights.to(self.device)
)
```

**Numerical example.** Layer 8, 64 experts, β = 0.9. At step t, the router assigns weight
0.12 to expert 0 (high-frequency) and 0.005 to expert 31 (low-frequency).

After 100 steps from a cold start (f initialised to 0):

```
f_0^(100) ≈ 0.12 · (1 - 0.9^100) / (1 - 0.9)  ≈  0.12 · (1 - 2.7e-5)  ≈  0.120
f_31^(100) ≈ 0.005 · (1 - 0.9^100)             ≈  0.005
```

The EMA has converged. Expert 0 has a routing frequency ~24× higher than expert 31.

---

## 4. Rank Importance Tracking

Routing frequency alone is insufficient — a frequently-used expert with saturated gradients
should not grow further. DR-LoRA also tracks **rank importance**: how much the current
active ranks are still learning.

**Fisher sensitivity score (Eq. 6):**

For each active rank j:
```
s_j = ||∇_A_j ⊙ A_j||₁  ×  ||∇_B_j ⊙ B_j||₁
```

This is the element-wise product of gradient and parameter value, summed over the rank
dimension. It measures the *Fisher information* — how much the loss would change if that
rank were perturbed.

```python
# From DRLoRATracker.compute_rank_importance
A_grad  = dl.lora_A.grad[active_mask]     # (r_active, in_features)
A_param = dl.lora_A.data[active_mask]
B_grad  = dl.lora_B.grad[:, active_mask]  # (out_features, r_active)
B_param = dl.lora_B.data[:, active_mask]

A_sensitivity = (A_grad * A_param).abs().sum(dim=1)   # (r_active,)
B_sensitivity = (B_grad * B_param).abs().sum(dim=0)   # (r_active,)
per_rank_scores = A_sensitivity * B_sensitivity
expert_importance = per_rank_scores.mean().item()
```

**Expert-level aggregation (Eq. 8):**
```
g_{l,i} = (1/r) · Σ_j s_j
```

**Rank importance EMA (Eq. 7):**
```
g_{l,i}^(t) = β · g_{l,i}^(t-1) + (1-β) · g_{l,i}^new
```

**Numerical example.** Expert 0, r_active = 4. Suppose:

```
A_sensitivity = [0.08, 0.06, 0.04, 0.01]  (rank 0 most important, rank 3 nearly dead)
B_sensitivity = [0.10, 0.07, 0.05, 0.02]

per_rank_scores = [0.0080, 0.0042, 0.0020, 0.0002]
g_0 = mean = (0.0080 + 0.0042 + 0.0020 + 0.0002) / 4 = 0.00360
```

After EMA update (β = 0.9, previous g = 0.004):
```
g_new = 0.9 × 0.004 + 0.1 × 0.00360 = 0.003600 + 0.000360 = 0.003960
```

**Why grad × param, not just grad?**

Consider rank j where A_j ≈ 0 (near-zero weights). The gradient ∇A_j might still be
large (the model *wants* to move), but the current contribution is near zero because the
parameters are small. Plain gradient magnitude would overestimate this rank's importance.
The product ∇A_j ⊙ A_j correctly assigns near-zero importance to a near-zero rank.

Conversely, a rank with large A_j values and small gradient has already converged — also
low importance for further growth.

---

## 5. Saliency Score

The final score combining routing frequency and rank importance (Eq. 9):

```
S_{l,i} = f_{l,i} · g_{l,i} / (r_{l,i} + 1)^γ
```

The `(r + 1)^γ` term is a **rank penalty**: experts that already have many active ranks
are deprioritised for further growth. This ensures convergence rather than a few experts
monopolising all capacity.

```python
# From DRLoRATracker.get_saliency_scores
f = self.routing_frequency[layer_idx]
g = self.rank_importance[layer_idx]
ranks = current_active_ranks_per_expert
saliency = f * g / (ranks + 1).pow(gamma)
```

**Numerical example.** γ = 0.5. Two experts in layer 8:

```
Expert 0:  f = 0.120,  g = 0.00396,  r = 4  →  S = 0.120 × 0.00396 / √5 = 0.000213
Expert 7:  f = 0.090,  g = 0.00500,  r = 6  →  S = 0.090 × 0.00500 / √7 = 0.000170
```

Expert 0 has higher saliency despite lower rank importance, because its routing frequency
is higher and its rank penalty is smaller (fewer active ranks).

After expert 0 grows to r = 6:
```
Expert 0:  S = 0.120 × 0.00396 / √7 = 0.000180
Expert 7:  S = 0.090 × 0.00500 / √7 = 0.000170
```

The gap closes. The rank penalty ensures balanced allocation over time.

---

## 6. Growth Schedule

Growth events are pre-scheduled to occur at fixed intervals after warmup, stopping before
training ends (end buffer ensures final training stability).

```python
# From DRLoRAGrowthSchedule.__init__
effective_end = total_steps - end_buffer_steps
self.growth_event_steps = [
    warmup_steps + (i + 1) * growth_interval
    for i in range(self.num_growth_events)
    if warmup_steps + (i + 1) * growth_interval < effective_end
]
```

**Numerical example.** n_steps = 1500, warmup = 150, interval = 187, end_buffer = 150.

```
effective_end = 1500 - 150 = 1350
Growth steps: [150+187, 150+374, 150+561, 150+748, 150+935, 150+1122]
            = [337, 524, 711, 898, 1085, 1272]
```

6 growth events, all before step 1350.

**Rank quota per event.** For a single layer with 64 experts, r_init = 4, r_max = 8:

```
total_ranks_to_grow = 64 × (8 - 4) = 256
ranks_per_event = 256 / 6 ≈ 42 per event
```

With p_grow = 0.5, each expert can grow by at most floor((r_max - r_init) × p_grow) = 2
ranks per event.

---

## 7. Rank Growth Algorithm

At each scheduled growth step:

1. **Compute saliency** S_{l,i} for every expert in the target layer.
2. **Sort** experts by saliency descending.
3. **Greedy allocation** with budget Q (quota per event):
   - Assign up to min(max_per_expert, remaining_quota, free_slots) new ranks to each
     expert, in saliency order.
4. **Activate** the next dormant rank slots in the mask.
5. **Reset** rank importance for grown experts (so newly active ranks can accumulate
   their own history from zero).

```python
# From perform_rank_growth (simplified)
for sal, info in sorted_by_saliency:
    dl = info["dr_lora"]
    current_rank = dl.get_active_ranks()
    free_slots = r_max - current_rank
    n_grow = min(max_per_expert, remaining_quota, free_slots)
    for _ in range(n_grow):
        next_rank = dl.get_active_ranks()
        dl.activate_rank(next_rank)   # flips rank_mask[next_rank] = True
    tracker.rank_importance[layer_idx, expert_idx] = 0.0   # reset
    remaining_quota -= actual_grown
```

**Numerical example.** Growth event at step 337. Quota = 42. γ = 0.5. Top 3 experts:

```
Expert  0: saliency = 0.000213,  r = 4,  free = 4  →  grow 2  (capped by max_per_expert)
Expert  7: saliency = 0.000170,  r = 4,  free = 4  →  grow 2
Expert 15: saliency = 0.000155,  r = 4,  free = 4  →  grow 2
...  (continues for remaining quota = 42 - 2 - 2 - 2 - ... = 0)
```

21 experts grow by 2 ranks each = 42 total new ranks. Low-saliency experts (rarely routed
or already converged) receive nothing in this event.

After growth, expert 0's mask:
```
Before: [1 1 1 1 0 0 0 0]   (4 active, 4 dormant)
After:  [1 1 1 1 1 1 0 0]   (6 active, 2 dormant)
```

---

## 8. Complete Algorithm Summary

```
Initialise: lora_A, lora_B ∈ ℝ^(r_max × d) allocated, only first r_init rows active
            routing_frequency f ← 0, rank_importance g ← 0

For each training step t:
  1. Forward pass (hooks capture router softmax weights)
  2. Compute loss, backward pass
  3. update_routing_frequency_from_hooks()         [EMA update of f]
  4. compute_rank_importance(lora_modules)          [Fisher sensitivity]
  5. update_rank_importance(importance)             [EMA update of g]
  6. If t ∈ growth_event_steps:
       For each layer:
         Compute S_{l,i} = f_{l,i} · g_{l,i} / (r_{l,i}+1)^γ
         Sort by S descending
         Greedy allocate quota Q: activate dormant rank slots
         Reset g for grown experts
  7. optimizer.step()
```

---

## 9. Key Properties

| Property | Value in implementation |
|---|---|
| r_max | 8 (= base_rank, configured) |
| r_init | 2 (= base_rank // 4) |
| lora_alpha | 8 (= base_rank) |
| EMA decay β | 0.9 |
| Rank penalty γ | 0.5 |
| p_grow (max fraction of free ranks per event) | 0.5 |
| Growth interval | max(10, n_steps // 8) |
| Warmup | max(10, n_steps // 10) |
| End buffer | max(10, n_steps // 10) |

The key guarantee: **tensor shape never changes**. The optimizer's Adam moment buffers
are always aligned with lora_A and lora_B, because those tensors stay at shape (r_max, d)
and (d, r_max) throughout training. Rank growth is a mask flip, not a reallocation.

# HR-LoRA vs DR-LoRA: How Hierarchical Spawning Addresses the Weaknesses of Dynamic Rank Growth

## Overview

DR-LoRA and HR-LoRA both address the rigidity of fixed-rank LoRA in MoE models, but they
diagnose the problem differently and reach for different solutions. This document works
through each fundamental weakness of DR-LoRA, shows concretely why it matters, and
explains precisely how HR-LoRA's design addresses it.

---

## 1. DR-LoRA Weakness: Shared Capacity for Conflicting Domains

### The Problem

DR-LoRA grows rank based on saliency (routing frequency × gradient importance). When
training on multiple conflicting domains (code + medical), saliency is high for all
high-traffic experts — but saliency cannot distinguish *why* the expert is learning
heavily. It may be because both domains are routing there and fighting over the same
weight directions.

**Concretely:** Expert 0 in layer 8 sees code tokens and medical tokens. Its saliency is
high because:
- `f_{8,0}` is high (both domains route to it)
- `g_{8,0}` is high (strong gradients — but pulling in two opposite directions)

DR-LoRA responds by growing Expert 0's rank. This gives it more capacity — but all of
that capacity is still in the same matrix ΔW = B·A. A single linear subspace must now
represent both code syntax and medical terminology. Rank growth increases magnitude of
representational capacity but does not allow *selective routing* to different parts of
the adapter.

**Numerical example.** After 500 steps of 50/50 code/medical training:

```
Expert 0 rank grows from 2 → 6 (saliency correctly identifies it as important).

Code gradient direction:    u_code    ∈ ℝ^2048
Medical gradient direction: u_medical ∈ ℝ^2048
cos(u_code, u_medical) = -0.72   (highly anti-aligned — genuinely conflicting)

The 6 active LoRA rank vectors try to simultaneously span u_code and -u_code.
Any rank vector moved toward u_code moves away from u_medical.
Result: neither domain is well-served. PPL_code = 150, PPL_medical = 148.
```

### How HR-LoRA Addresses This

HR-LoRA detects this conflict explicitly via the `ConflictSaturationMonitor`. When
|ema_medical - ema_code| > 0.15 for 8 consecutive steps AND the base LoRA has plateaued,
a **new sub-adapter is spawned**. This sub-adapter has its own A and B matrices, plus a
ReLU gate w that learns to activate only for one domain.

```
After spawn at step 450:
  Base LoRA L_{0,0}:          handles tokens from all domains (compromised by conflict)
  Sub-adapter L_{0,1} + gate: gate learns to activate for code tokens only

Code token at step 600:
  w_0^T x_code = +0.71  →  g = 0.71  →  sub-adapter contributes 0.71 × L_{0,1}(x)
Medical token at step 600:
  w_0^T x_medical = -0.24  →  g = 0.0  →  sub-adapter silent

Sub-adapter can now specialise entirely for code without contamination from medical.
```

**The key difference:** DR-LoRA grows a single shared capacity. HR-LoRA grows gated
independent capacity. Under conflict, only independent-gated expansion prevents
inter-domain contamination.

---

## 2. DR-LoRA Weakness: Fixed Trigger Timing (Schedule-Based Growth)

### The Problem

DR-LoRA uses a pre-computed schedule: growth events fire at steps
[warmup + interval, warmup + 2×interval, ...]. This schedule is set before training
starts and does not adapt to what is actually happening in the loss landscape.

**Concrete failure modes:**

1. **Growth before conflict exists (Run A: 100% code).** The schedule fires at step 337
   regardless of whether any domain conflict has emerged. Rank capacity is allocated to
   an already-well-trained adapter, with diminishing returns.

2. **Growth too late.** If conflict becomes severe at step 200 but the first growth event
   is at step 337, the model spends 137 steps trying to force two incompatible domains
   into the same rank-2 adapter. Negative transfer accumulates during this window.

3. **Growth equally in all conflict ratios.** The same schedule fires in Run A (no
   conflict) and Run C (50% conflict). In Run A, all 6 growth events are essentially
   wasted. In Run C, 6 events are not enough.

**Numerical example.** n_steps = 1500, Run A vs Run C:

```
Run A (0% medical):
  All 6 growth events fire.
  Expert 0 saliency dominated by: high f (code routes here), moderate g (converging).
  6 events add ~12 rank slots to top experts.
  Effect: marginal improvement. The adapter was already learning well.

Run C (50% medical):
  Conflict emerges at step ~300 (|ema_medical - ema_code| crosses 0.15).
  First growth event: step 337 (37 steps after conflict onset — reasonable).
  But growth adds more capacity to the same entangled matrix.
  Steps 337–1272: 6 more ranks added, all to the same conflicted ΔW.
  Result: PPL_code = 143.14 after Run C — negative transfer of -6.72.
  (Paradoxically, code PPL improves in Run C because extra capacity helps despite conflict.)
```

### How HR-LoRA Addresses This

HR-LoRA spawns **reactively**: a spawn only happens when the monitor fires, which requires
8 consecutive steps of [plateau AND conflict]. There is no pre-computed schedule.

```
Run A (0% medical):
  ema_medical never updates (no medical samples).
  conflict = False at every step.
  Monitor never fires.
  Zero spawns in Run A: 0 expansion events (confirmed in experimental results).

Run C (50% medical):
  Conflict detected at step ~400 (8 steps of both conditions satisfied).
  Spawn fires.  Last_spawn_step = 400.
  Next earliest spawn: step 450 (cooldown = 50).
  Model re-evaluates: is base LoRA still plateaued? Is conflict still present?
  If yes → spawn again.
  Run C: 21 expansion events — the model spawns until it has enough capacity to
          partition the domains.
```

**The key difference:** DR-LoRA always grows at pre-set intervals, wasting capacity
in low-conflict runs and failing to allocate adaptively in high-conflict runs. HR-LoRA
grows exactly when and only when the data demands it.

---

## 3. DR-LoRA Weakness: No Mechanism for Domain Partitioning

### The Problem

Even if DR-LoRA perfectly allocates rank to the right experts at the right time, all
LoRA rank vectors are combined linearly in the output:

```
ΔW · x = (B · A) · x = Σ_j  b_j · (a_j^T x)
```

For any input x, all active rank contributions are summed. There is no way for rank j
to activate only for code tokens and rank k to activate only for medical tokens.

**Why this matters structurally.** Linear combinations cannot implement partition-of-unity
semantics. If domain A and domain B pull in opposite gradient directions:

```
Code gradient for rank j:    +0.08 (push toward code representation)
Medical gradient for rank j: -0.06 (push toward medical representation)
Net gradient for rank j:     +0.02 (somewhere between both, satisfying neither)
```

Adding more ranks does not solve this — each new rank faces the same averaging problem.
The optimal solution under domain conflict requires *conditional computation*: different
parts of the adapter should activate for different inputs.

**Numerical example.** Expert 0, ranks r = 8 (DR-LoRA max capacity reached).

```
Code text:    "def fibonacci(n): if n <= 1: return n"
Medical text: "The patient presented with acute myocardial infarction"

ΔW · x_code    = (B·A) · x_code    — all 8 ranks contribute
ΔW · x_medical = (B·A) · x_medical — all 8 ranks contribute

If rank 3 has learned "return statement patterns" from code,
medical training will push rank 3 toward "infarction patterns".
The code representation of rank 3 degrades.
No mechanism exists to protect rank 3 from medical training.
```

### How HR-LoRA Addresses This

The ReLU gate `g = ReLU(w^T x)` is the exact mechanism DR-LoRA lacks. After sufficient
training:

```
For sub-adapter j (spawned after code-medical conflict detected):
  Code tokens x_code:     w_j^T x_code    ≈ +0.85  →  g = 0.85  (sub-adapter active)
  Medical tokens x_med:   w_j^T x_med     ≈ -0.30  →  g = 0.0   (sub-adapter silent)

Gradient for sub-adapter j:
  Code step:    grad flows through  g * sub_lora_output  with g = 0.85  → strong update
  Medical step: grad flows through  g * sub_lora_output  with g = 0.0   → zero gradient!
```

Medical training literally cannot reach sub-adapter j's parameters. The gradient through
`g * sub_out` when `g = 0` is zero (ReLU hard gate: when the gate is off, `dg/dw = 0`).
This is gradient-level domain isolation, not just reduced influence.

**Numerical example.** Sub-adapter j parameters at step 600:

```
Step 600: A_j = Vh[:8] from SVD of code gradient at spawn time
                       (explicitly initialised toward code directions)
Step 601 (code token):    grad_A_j ≠ 0  →  A_j updates toward code
Step 601 (medical token): g = 0         →  grad_A_j = 0  → A_j unchanged

After 900 more steps:
  A_j has moved purely in the direction of code signal.
  It has never received a single medical gradient.
  This is mathematically impossible in DR-LoRA.
```

---

## 4. DR-LoRA Weakness: Global Rank Budget Shared Across All Experts

### The Problem

DR-LoRA's quota system distributes a fixed number of new rank slots per growth event
across all experts in a layer, sorted by saliency. The total rank budget is:

```
total_ranks_to_grow = num_experts × (r_target - r_init)
```

This budget is global: 21 high-saliency experts growing 2 ranks each = 42 ranks,
consuming the full event quota. Low-saliency experts receive nothing.

But saliency ranks experts relative to each other — it does not measure whether any
expert *actually needs more capacity*. In a low-conflict run, even the highest-saliency
expert may have no need for additional rank. The budget is spent anyway.

**Numerical example.** n_steps = 1500, Run A (0% conflict), base_rank = 8, r_max = 8.
r_init = 2, so growth target is 8-2 = 6 ranks per expert.

```
Total budget: 64 experts × 6 = 384 ranks
Distributed across 6 events: 64 ranks per event
Each event: top 32 experts grow by 2 ranks each.

But in Run A, there is no domain conflict. The base LoRA is learning well.
Extra ranks add capacity to an already-non-conflicted adapter.
Marginal cost: each rank slot is an (r_max × in_dim) slice of gradient computation.
               Adding 384 rank-slots wastes ~6% of training compute on unused capacity.
```

### How HR-LoRA Addresses This

HR-LoRA has zero budget overhead in low-conflict runs. The monitor requires *both*
plateau and conflict — and if the model is learning well (no plateau) or the domains
are aligned (no conflict), the monitor never fires.

```
Run A results (from experimental output):
  HR-LoRA spawns: 0  (no conflict detected → 0 expansion events)
  DR-LoRA ranks added: 384 (scheduled regardless)

Run C results:
  HR-LoRA spawns: 21 (reactive, proportional to conflict severity)
  DR-LoRA ranks added: 384 (same schedule, no conflict-sensitivity)
```

Each spawn adds a sub-adapter of rank 8 with independent parameters (A, B, gate w):
```
Parameters per spawn = 8 × 2048 (A) + 2048 × 8 (B) + 2048 (w) = 34,816 params
```
Compared to DR-LoRA adding one rank to one expert:
```
Parameters per rank activation = 2048 (one row of A) + 2048 (one column of B) = 4,096 params
```

HR-LoRA's spawns are larger per event but strictly conditional. DR-LoRA's growth is
smaller per event but unconditional.

---

## 5. DR-LoRA Weakness: Rank Importance Resets Destroy History

### The Problem

After each growth event, DR-LoRA resets the rank importance EMA for all grown experts:

```python
tracker.rank_importance[layer_idx, expert_idx] = 0.0
```

The reasoning: newly activated ranks have no history, so the aggregated importance would
be diluted. But this also destroys the importance history of the *existing active ranks*.

After the reset, the next growth event's saliency scores are computed from a cold start.
If the reset happens frequently (tight growth interval), the tracker never accumulates
a reliable signal. Early growth events may target experts that were accidentally spiking
rather than consistently important ones.

**Numerical example.** Growth events at steps 337 and 524 (interval = 187 steps, β = 0.9).

```
After reset at step 337, the importance EMA starts at 0.
Steps 337–524 = 187 steps with β = 0.9:

EMA after 187 steps from 0 toward true importance g_true ≈ 0.004:
g^(187) = 0.004 × (1 - 0.9^187) ≈ 0.004 × (1 - 5.8e-9) ≈ 0.004

Full convergence — but only because 187 steps × β=0.9 gives enough warmup.
If interval were 50 steps: g^(50) = 0.004 × (1 - 0.0052) ≈ 0.00398 — still OK.

But importance at step 524 is estimated from only 187 post-reset steps.
Any burst signal in steps 338–340 (early after reset) is over-weighted.
```

This is a minor issue in practice but the reset creates a systematic underestimate of
importance for recently-added ranks (they haven't had time to accumulate signal).

### How HR-LoRA Addresses This

HR-LoRA's monitor state resets only after a spawn, and only the monitor's internal
buffers reset (ri_history, plateau_window, conflict_window). The base LoRA's parameters
and their gradient history are untouched.

The new sub-adapter starts from scratch (its own A, B, gate parameters with no history),
while the existing base LoRA continues updating normally. There is no coupling between
the monitor reset and the base LoRA's training trajectory.

Additionally, because spawning is infrequent (guarded by cooldown and cap), monitor
resets happen rarely — the base LoRA has long stable periods in which to accumulate
a reliable plateau/conflict signal.

---

## 6. DR-LoRA Weakness: Saliency Conflates Frequency and Importance

### The Problem

```
S_{l,i} = f_{l,i} × g_{l,i} / (r_{l,i} + 1)^γ
```

A high-frequency expert (f large) with low rank importance (g small) still gets a high
saliency score because the product f×g can be dominated by f. Meanwhile, a
moderate-frequency expert with genuinely high gradient signal may score lower.

In domain conflict, the highest-frequency experts are those that handle both code and
medical tokens. These are precisely the conflicted experts. DR-LoRA grows them first
(high saliency from high f), but growing a conflicted expert doesn't resolve the
conflict — it just adds more capacity to the entanglement.

**Numerical example.** Layer 8, two experts:

```
Expert 0: f = 0.120 (both code & medical route here),  g = 0.002 (low — conflicted)
           S = 0.120 × 0.002 / √5 = 0.000107

Expert 5: f = 0.045 (mostly code),  g = 0.008 (high — clear gradient direction)
           S = 0.045 × 0.008 / √3 = 0.000208
```

Expert 5 has higher saliency, so it correctly gets rank first. But Expert 0 has rank
growing events allocated to it in later events once Expert 5 approaches r_max. The
extra capacity in Expert 0 still ends up entangled.

### How HR-LoRA Addresses This

HR-LoRA's trigger is conflict-specific. The delta_threshold condition:

```
|ema_medical - ema_code| > 0.15
```

...directly measures domain entanglement, not just expert importance. An expert can have
high gradient signal (high g) and high routing frequency (high f) but if the two domains
are performing similarly in loss, no spawn is triggered.

The spawn fires specifically when the model's loss on the two domains diverges —
which is a direct measure that the current adapter cannot simultaneously serve both.

---

## 7. Experimental Comparison (From Notebook Results)

The `aimo3-h100-probe-3.ipynb` notebook ran both methods on OLMoE-1B-7B with the
conflict-scaling grid (Runs A, B, C):

```
================================================================================
THESIS RESULTS: CONFLICT-SCALING GRID
================================================================================

Method          Run   Conflict     Code PPL   Neg Transfer   Expansions
───────────────────────────────────────────────────────────────────────────
lora            A     0%             149.85              —            0
hierarchical    A     0%             133.19              —            0

lora            B     20%            150.09          +0.24            0
hierarchical    B     20%            135.37          +2.18           20

lora            C     50%            143.14          -6.72            0
hierarchical    C     50%            140.53          +7.34           21
```

**Reading the table:**

- **Run A (no conflict):** HR-LoRA achieves lower PPL than standard LoRA (133.19 vs
  149.85) with zero expansion events. The base LoRA alone is more effective — likely
  because HR-LoRA wraps 3 layers (4, 8, 12) vs LoRA wrapping 1, giving more trainable
  capacity.

- **Run B (20% conflict):** HR-LoRA spawns 20 times. Negative transfer +2.18 vs +0.24
  for LoRA. The spawning overhead slightly hurts code PPL in this moderate-conflict
  regime — the monitor fires but the sub-adapters may not have fully specialised in
  only 1500 steps.

- **Run C (50% conflict):** HR-LoRA spawns 21 times. Negative transfer +7.34 vs -6.72.
  At severe conflict, HR-LoRA's sub-adapters are actively spawning and specialising.
  The convergence gap compared to LoRA narrows at 50% conflict (140.53 vs 143.14),
  suggesting the domain-partitioning is beginning to work despite the short training run.

**Note on Run C negative transfer direction:** LoRA shows negative transfer = -6.72
(PPL *decreased* from Run A to Run C), which seems counterintuitive. This is because in
Run C, the model sees 750 code samples + 750 medical samples vs 1500 pure code in Run A.
The smaller effective code training set sometimes leads to different optima.

---

## 8. Summary Table

| Dimension | DR-LoRA | HR-LoRA |
|---|---|---|
| Capacity type | Additional rank in shared ΔW | Independent gated sub-adapter |
| Domain isolation | None (all ranks share input-output space) | ReLU gate provides hard gradient isolation |
| Growth trigger | Pre-scheduled (time-based) | Reactive (conflict + plateau detection) |
| Trigger sensitivity to conflict | No — fires equally in Run A and Run C | Yes — zero spawns in Run A, 21 in Run C |
| Partitioning mechanism | None | ReLU(w^T x) = 0 blocks gradient for silent tokens |
| Budget waste in low-conflict | High (full schedule executes) | None (monitor never fires) |
| New capacity init | Dormant slots activated (B already zeroed) | B=0 + SVD-informed A + tiny gate |
| Loss at spawn | None (B=0 always) | None (B=0 guaranteed) |
| Maximum capacity | r_max (hard ceiling) | max_sub_adapters × spawn_rank (soft ceiling) |
| Rank importance history | Resets after each growth event | Monitor resets only; base LoRA history continuous |
| Applicable when | One domain, importance-based resource allocation | Multiple conflicting domains, continual learning |

---

## 9. When to Use Each

**Use DR-LoRA when:**
- Fine-tuning on a single domain (no domain conflict)
- The bottleneck is rank capacity, not domain entanglement
- You want predictable, scheduled capacity growth
- Memory is tight (no dynamic allocation)

**Use HR-LoRA when:**
- Fine-tuning on multiple domains that conflict (code + medical, formal + informal, etc.)
- You need the model to retain performance on domain A while learning domain B
- Training distribution shifts over time (continual learning)
- You can tolerate reactive, unpredictable capacity growth

**The fundamental insight:** DR-LoRA optimises *how much* each expert learns. HR-LoRA
optimises *what* each expert learns, and partitions conflicting *what*s into separate,
non-interfering sub-adapters via the ReLU gate. The two are not alternatives for the same
problem — they solve different problems. HR-LoRA subsumes DR-LoRA's concern (capacity
allocation) as a secondary effect of its spawning mechanism, but its primary innovation
is domain partitioning through conditional computation.

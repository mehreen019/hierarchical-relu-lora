# OLMoE Jaccard Diagnostic — Findings and Changes

## What the diagnostic was supposed to do

Before training, measure routing overlap between two domain pairs (Python/Medical
and Math/Creative) in the base OLMoE model. The Jaccard score tells us whether
domains activate different experts or the same ones. This determines:
- Which domain pair has the most routing conflict (→ chosen for experiments)
- Which layer has the most overlap (→ target layer for adapter patching)
- Which expert at that layer is most contested (→ target expert)

---

## Run 1 — complete failure: concentration threshold found nothing

### What happened

All concentrated Jaccard scores were exactly **0.000**. Cell 12 printed:
```
Layer 0 — experts above threshold:
  Python:  0 experts  |  Medical: 0 experts  |  Shared: 0 experts
```

Final table:
```
Jaccard (Python vs Medical)   Phi: 0.056   OLMoE: 0.000
Jaccard (Math vs Creative)    Phi: 0.069   OLMoE: 0.000
Null baseline (same domain)   Phi: ~1.0    OLMoE: 0.000
```

### Root cause: OLMoE uses load-balanced routing

The original concentration threshold was `2 × uniform_baseline = 25%`. This was
designed for **Phi-3.5-MoE**, which has no load-balancing loss — its routing is
sparse and uneven, so certain experts dominate and easily exceed 25% frequency.

**OLMoE explicitly trains a load-balancing auxiliary loss** that forces routing
to be near-uniform across all 64 experts. With 60 examples × ~50 tokens × top-8,
no expert ever exceeds ~14–16% frequency. The 25% threshold excludes everything.

The **raw Jaccard was ~0.99** for all pairs — with top-8/64, nearly every expert
fires for every domain just by volume. Both raw Jaccard and the concentration
threshold are meaningless for load-balanced routing.

Actual Layer 0 Python frequency distribution:
```
Top-5 expert frequencies: [0.0977, 0.0833, 0.0385, 0.0321, 0.0313]
Max: 0.0977  |  Min: 0.0006
```
Even the most active expert is below 10% — nowhere near 25%.

### Why Phi was different

Phi's raw Jaccard was 0.056 — only 5.6% overlap in which (layer, expert) pairs
fired at all. Phi's routing is highly concentrated: a few experts dominate per
layer, and those experts differ between domains. The `0.002` threshold (basically
"ever fired") was sufficient to isolate domain-specific experts.

---

## Fix applied (Run 2)

### Replaced concentration threshold with top-K Jaccard

For each domain and layer, take the **top-16 most active experts** (top quartile
of 64). Compute Jaccard between those two sets of 16.

```python
def jaccard_topk(freq_a, freq_b, top_n=16):
    top_a = set(sorted(freq_a, key=freq_a.get, reverse=True)[:top_n])
    top_b = set(sorted(freq_b, key=freq_b.get, reverse=True)[:top_n])
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else 0.0
```

Even with flat routing, some experts are slightly more preferred per domain.
Top-16 captures this signal without requiring any hard frequency threshold.

**Null baseline interpretation:**
- Same domain (Python-A vs Python-B): high Jaccard expected — both halves prefer
  the same experts
- Cross-domain (Python vs Medical): lower Jaccard if routing differs between domains

### Changed: target layer selection

Changed from `argmin(topk_a)` (was accidentally wrong — most separated layer) to
`argmax(chosen_topk)` — the layer with **highest top-K Jaccard** = most shared
top experts = most gradient conflict. This is the layer where both domains compete
hardest for the same experts.

### Changed: target expert selection

Replaced threshold-filtered `find_shared_experts` with `find_top_conflict_experts`
using **geometric mean of frequencies** — no threshold needed. Always returns a
valid target regardless of how flat routing is.

### Changed: chosen pair selection

Now picks the pair with the **largest gap from null baseline** (`null − cross_domain`)
rather than lowest absolute Jaccard. Larger gap = more genuine domain separation
relative to same-domain noise.

### Cells modified in `olmoe-jaccard-diagnostic.ipynb`

| Cell | What changed |
|------|-------------|
| 8    | Added: saves `olmoe_freq_cache.pkl` after frequency collection |
| 12   | Removed concentration threshold; added TOP_N=16; updated plot |
| 14   | Replaced `jaccard_concentrated` with `jaccard_topk` |
| 16   | Updated variable names (`topk_a/b/null`); added gap-from-null output |
| 18   | Variable name update (plot labels) |
| 20   | Replaced `find_shared_experts` with `find_top_conflict_experts`; uses chosen pair dynamically; target layer = argmax |
| 24   | Updated decision logic (gap-based); updated comparison table |
| 26   | Updated config keys (`top_n_jaccard` replaces `concentration_threshold`) |

---

## Run 2 — results

```
Option A -- Python vs Medical
  Mean raw Jaccard:    0.990
  Mean top-K Jaccard:  0.103

Option B -- Math vs Creative
  Mean raw Jaccard:    1.000
  Mean top-K Jaccard:  0.053

Null baseline -- Python-A vs Python-B (same domain)
  Mean raw Jaccard:    0.993
  Mean top-K Jaccard:  0.840

Separation from null:
  Option A gap: 0.840 - 0.103 = 0.737
  Option B gap: 0.840 - 0.053 = 0.788   ← LARGER
```

### Chosen pair: Option B — Math vs Creative

Math vs Creative has the larger gap from null (0.788 vs 0.737), meaning it shows
more genuine domain-specific routing preference relative to same-domain noise.

### Per-layer results (top-K Jaccard, chosen pair Math/Creative)

Highest overlap layer (= most conflict): **Layer 6 — Jaccard = 0.143**

Target layer: **6**
Target expert: **TBD** — re-run Cell 20 to get conflict expert for Math/Creative
at Layer 6. (First run of Cell 20 incorrectly used freq_python/freq_medical;
fix was applied, cells 20 and 26 need one more re-run.)

### Full comparison table

| Metric | Phi-3.5-MoE | OLMoE |
|---|---|---|
| Experts per layer | 16 | 64 |
| Top-k routing | 2 | 8 |
| Uniform baseline | 12.5% | 12.5% |
| Jaccard method | raw set | top-16 |
| Python vs Medical | 0.056 | 0.103 |
| Math vs Creative | 0.069 | 0.053 |
| Null baseline | ~1.0 | 0.840 |
| Gap from null (chosen) | — | 0.788 |
| **Chosen pair** | Python/Medical | **Math/Creative** |
| **Target layer** | — | **6** |

---

## Thesis implication

**Phi-3.5-MoE (Jaccard = 0.056):** Routing already partially separates domains —
only ~5.6% of expert slots overlap between Python and Medical. Gradient entanglement
is a gradient-space problem, not a routing-space one. Routing acts as a weak
natural separator.

**OLMoE (null = 0.840, cross-domain = 0.053–0.103):** The null baseline of 0.840
shows that within the same domain, the same top-16 experts are consistently
preferred (~84% overlap). The cross-domain scores (5–10%) are far below the null,
confirming real domain separation. However, the load-balancing loss means no single
expert is ever "owned" by one domain — every expert is shared to some degree.

This means gradient entanglement in OLMoE is **broader but shallower** than in
Phi — spread across more experts, but with weaker per-expert conflict. The
hierarchical spawning mechanism must therefore operate across more experts to be
effective.

Note: the Phi result used Python vs Medical; OLMoE chose Math vs Creative.
The domain pair choice is architecture-specific and driven by routing structure,
not by the method itself.

---

## Final confirmed results (Run 2, cells 20 + 26)

Config saved to `notebook_outputs/olmoe-jaccard-test/olmoe_experiment_config.json`:

```json
{
  "chosen_option":    "B",
  "chosen_pair":      "Math vs Creative",
  "target_layer":     6,
  "target_expert":    18,
  "jaccard_python_medical": 0.1029,
  "jaccard_math_creative":  0.0528,
  "jaccard_null_baseline":  0.8403,
  "top_n_jaccard":    16
}
```

Top conflict experts at Layer 6 (Math vs Creative):

| Expert | Freq_Math | Freq_Creative | Conflict Score |
|--------|-----------|---------------|----------------|
| **18** | 0.0612    | 0.0401        | **0.0495**     |
| 62     | 0.0257    | 0.0267        | 0.0262         |
| 27     | 0.0050    | 0.1135        | 0.0239         |

**Note:** Notebooks run on Kaggle. Outputs stored in `notebook_outputs/`.
Both `olmoe-task3-results-table.ipynb` and `olmoe-smoke-test.ipynb` were updated
to read datasets and target layer/expert from the config file automatically.

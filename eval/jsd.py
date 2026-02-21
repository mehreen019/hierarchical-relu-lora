# eval/jsd.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_jsd(hier_model, probe_primary, probe_conflict, tokenizer,
                max_length=256, n_bins=50):
    """
    Collect ReLU gate activation distributions on held-out probe data.
    Compute JSD between primary and conflict domain distributions.

    Args:
        probe_primary: list of {"text": ..., "domain": ...} dicts
        probe_conflict: same for conflict domain

    Returns:
        {"mean_jsd": float, "per_adapter": list, "claim_3_pass": bool}
    """
    # Register hooks on all sub-adapter w_gate params
    activation_store = {}   # (layer_key, sub_idx) -> {"primary": [], "conflict": []}

    hooks = []

    def make_hook(layer_key, sub_idx, domain):
        def hook_fn(module, input, output):
            x = input[0]  # (batch, seq, features)
            gate = F.relu(x @ module.w_gate)  # (batch, seq)
            activations = gate.flatten().detach().cpu().tolist()
            key = (layer_key, sub_idx)
            if key not in activation_store:
                activation_store[key] = {"primary": [], "conflict": []}
            activation_store[key][domain].extend(activations)
        return hook_fn

    # Phase 1: collect primary domain activations
    for layer_key, layer in hier_model.hier_layers.items():
        for sub_idx, sub in enumerate(layer.sub_adapters):
            h = sub.register_forward_hook(make_hook(layer_key, sub_idx, "primary"))
            hooks.append(h)

    hier_model.model.eval()
    device = next(hier_model.model.parameters()).device

    with torch.no_grad():
        for ex in probe_primary:
            inputs = tokenizer(
                ex["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(device)
            _ = hier_model.model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()
    hooks = []

    # Phase 2: collect conflict domain activations
    for layer_key, layer in hier_model.hier_layers.items():
        for sub_idx, sub in enumerate(layer.sub_adapters):
            h = sub.register_forward_hook(make_hook(layer_key, sub_idx, "conflict"))
            hooks.append(h)

    with torch.no_grad():
        for ex in probe_conflict:
            inputs = tokenizer(
                ex["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(device)
            _ = hier_model.model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute JSD for each sub-adapter
    jsd_list = []
    per_adapter_details = []

    for (layer_key, sub_idx), acts in activation_store.items():
        p_acts = np.array(acts["primary"])
        c_acts = np.array(acts["conflict"])

        if len(p_acts) < 10 or len(c_acts) < 10:
            continue

        # Build histograms
        all_acts = np.concatenate([p_acts, c_acts])
        lo, hi = np.percentile(all_acts, [1, 99])
        bins = np.linspace(lo, hi, n_bins + 1)

        p_hist, _ = np.histogram(p_acts, bins=bins, density=True)
        c_hist, _ = np.histogram(c_acts, bins=bins, density=True)

        # Add epsilon to avoid div/0
        p_hist = p_hist + 1e-10
        c_hist = c_hist + 1e-10
        p_hist = p_hist / p_hist.sum()
        c_hist = c_hist / c_hist.sum()

        jsd = jensenshannon(p_hist, c_hist) ** 2
        jsd_list.append(jsd)
        per_adapter_details.append({
            "layer_key": layer_key,
            "sub_idx": sub_idx,
            "jsd": jsd,
            "n_primary": len(p_acts),
            "n_conflict": len(c_acts)
        })

    mean_jsd = float(np.mean(jsd_list)) if jsd_list else 0.0
    claim_3_pass = mean_jsd > 0.3

    print("\n=== JSD Domain Specialization Report ===")
    for d in per_adapter_details:
        print(f"  {d['layer_key']} sub#{d['sub_idx']}: JSD={d['jsd']:.4f}")
    print(f"Mean JSD: {mean_jsd:.4f} | {'CLAIM 3 PASS (>0.3)' if claim_3_pass else 'CLAIM 3 FAIL (<0.3)'}")

    return {
        "mean_jsd": mean_jsd,
        "per_adapter": per_adapter_details,
        "claim_3_pass": claim_3_pass
    }

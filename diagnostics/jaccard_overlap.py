# diagnostics/jaccard_overlap.py
"""
Run this script BEFORE any training.
It determines whether Python/Medical or Math/Creative is the better conflict pair.
Usage: python diagnostics/jaccard_overlap.py
"""

import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from configs.track_a_config import MODEL_NAME, DATASET_OPTION_A, DATASET_OPTION_B


def get_top_k_experts_for_domain(model, tokenizer, examples, n_examples=50, max_length=256):
    """
    Feed examples through base OLMoE (no adapters), collect which experts fire.
    Returns a set of (layer_idx, expert_idx) tuples weighted by frequency.
    """
    expert_counts = defaultdict(int)
    model.eval()

    with torch.no_grad():
        for text in examples[:n_examples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(model.device)

            outputs = model(
                **inputs,
                output_router_logits=True,
                return_dict=True
            )

            # OLMoE router_logits: list of tensors, one per layer
            # Each tensor shape: (batch_size * seq_len, n_experts)
            if outputs.router_logits is None:
                raise ValueError("router_logits is None — check that OLMoE supports output_router_logits=True")

            for layer_idx, router_logits in enumerate(outputs.router_logits):
                # router_logits shape: (B*S, n_experts) or (B, S, n_experts)
                if router_logits.dim() == 3:
                    router_logits = router_logits.view(-1, router_logits.shape[-1])

                # OLMoE uses top-8 routing
                top_experts = router_logits.topk(8, dim=-1).indices  # (B*S, 8)
                for expert_idx in top_experts.flatten().tolist():
                    expert_counts[(layer_idx, expert_idx)] += 1

    return expert_counts


def compute_jaccard(counts_a: dict, counts_b: dict) -> float:
    """Jaccard similarity between two sets of active experts."""
    set_a = set(counts_a.keys())
    set_b = set(counts_b.keys())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def load_text_examples(dataset_name, text_col, config_name=None, n=60, split="train"):
    """Load raw text examples from a dataset."""
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    texts = [str(ex[text_col]) for ex in ds.select(range(min(n, len(ds))))]
    return texts


def run_diagnostic():
    print("=" * 60)
    print("JACCARD OVERLAP DIAGNOSTIC")
    print("Determines best domain conflict pair for Track A")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load BASE model only — no adapters
    print(f"\nLoading base model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Test Option A: Python vs Medical ---
    print("\n--- Testing Option A: Python vs Medical ---")
    py_texts = load_text_examples(
        DATASET_OPTION_A["primary"],
        DATASET_OPTION_A["primary_text_col"],
        n=60
    )
    med_texts = load_text_examples(
        DATASET_OPTION_A["conflict"],
        DATASET_OPTION_A["conflict_text_col"],
        DATASET_OPTION_A["conflict_config"],
        n=60
    )
    py_counts = get_top_k_experts_for_domain(model, tokenizer, py_texts)
    med_counts = get_top_k_experts_for_domain(model, tokenizer, med_texts)
    jaccard_a = compute_jaccard(py_counts, med_counts)
    print(f"Python vs Medical Jaccard overlap: {jaccard_a:.3f}")

    # --- Test Option B: Math vs Creative ---
    print("\n--- Testing Option B: Math vs Creative Fiction ---")
    math_texts = load_text_examples(
        DATASET_OPTION_B["primary"],
        DATASET_OPTION_B["primary_text_col"],
        n=60,
        split="train"
    )
    creative_texts = load_text_examples(
        DATASET_OPTION_B["conflict"],
        DATASET_OPTION_B["conflict_text_col"],
        DATASET_OPTION_B["conflict_config"],
        n=60
    )
    math_counts = get_top_k_experts_for_domain(model, tokenizer, math_texts)
    creative_counts = get_top_k_experts_for_domain(model, tokenizer, creative_texts)
    jaccard_b = compute_jaccard(math_counts, creative_counts)
    print(f"Math vs Creative Jaccard overlap: {jaccard_b:.3f}")

    # --- Decision ---
    print("\n" + "=" * 60)
    print("DECISION:")
    if jaccard_a >= 0.6:
        print(f"  Use OPTION A (Python/Medical) — overlap={jaccard_a:.3f} > 0.6")
        print("   Update track_a_config.py: set ACTIVE_DATASET = DATASET_OPTION_A")
        chosen = "A"
    elif jaccard_b >= 0.6:
        print(f"  Use OPTION B (Math/Creative) — overlap={jaccard_b:.3f} > 0.6")
        print("   Update track_a_config.py: set ACTIVE_DATASET = DATASET_OPTION_B")
        chosen = "B"
    else:
        best = "A" if jaccard_a > jaccard_b else "B"
        best_val = max(jaccard_a, jaccard_b)
        print(f"  Neither pair has overlap > 0.6. Best is Option {best} ({best_val:.3f}).")
        print("   Proceed with the higher-overlap pair but note this in the thesis.")
        print("   Consider that OLMoE's base routing may already be well-separated.")
        chosen = best

    print(f"\nChosen dataset option: {chosen}")
    print("=" * 60)
    return chosen, jaccard_a, jaccard_b


if __name__ == "__main__":
    run_diagnostic()

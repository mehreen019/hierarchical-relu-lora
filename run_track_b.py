# run_track_b.py
"""
Orchestrates 3 Track B jobs (one per method) on the OLMoE SFT mix.
Run: python run_track_b.py
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from configs.track_a_config import MODEL_NAME, STANDARD_LORA, DR_LORA, HIERARCHICAL
from configs.track_b_config import (
    TOTAL_STEPS, MAX_SEQ_LENGTH, LEARNING_RATE, WARMUP_STEPS,
    DR_LORA_TRACK_B_OVERRIDES, HIERARCHICAL_TRACK_B_OVERRIDES, EVAL_TASKS
)
from data.dataset_builder import build_track_b_dataloader
from baselines.standard_lora import get_standard_lora_model
from baselines.dr_lora import DRLoRAModel
from method.hierarchical_spawning import HierarchicalSpawningModel
from train import train
from eval.stability import compute_stability


def load_base_model_4bit():
    """Load base OLMoE in 4-bit for custom methods."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model


def run_method(method_name: str, tokenizer, output_base: str):
    """Run a single method on Track B (OLMoE SFT mix)."""
    output_dir = os.path.join(output_base, method_name)
    print(f"\n{'='*60}")
    print(f"Running: {method_name} | Track B (SFT Mix)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Build dataloader
    dataloader = build_track_b_dataloader(
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        batch_size=1
    )

    # Load model based on method
    dr_lora_model = None
    hier_model = None

    if method_name == "standard_lora":
        model = get_standard_lora_model(MODEL_NAME, STANDARD_LORA)
    elif method_name == "dr_lora":
        # Merge Track B overrides
        dr_cfg = {**DR_LORA, **DR_LORA_TRACK_B_OVERRIDES}
        base = load_base_model_4bit()
        dr_lora_model = DRLoRAModel(base, dr_cfg)
        model = dr_lora_model.model
    elif method_name == "hierarchical":
        # Merge Track B overrides
        hier_cfg = {**HIERARCHICAL, **HIERARCHICAL_TRACK_B_OVERRIDES}
        base = load_base_model_4bit()
        hier_model = HierarchicalSpawningModel(base, hier_cfg)
        model = hier_model.model
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Train
    model, metrics_log, growth_steps = train(
        method_name=method_name,
        model=model,
        dataloader=dataloader,
        total_steps=TOTAL_STEPS,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        output_dir=output_dir,
        dr_lora_model=dr_lora_model,
        hier_model=hier_model
    )

    # Compute stability
    stability_report, stability_summary = compute_stability(metrics_log, growth_steps)
    with open(os.path.join(output_dir, "stability.json"), "w") as f:
        json.dump({"report": stability_report, "summary": stability_summary}, f, indent=2)

    return {
        "method": method_name,
        "final_loss": metrics_log[-1]["loss"],
        "stability": stability_summary
    }


def main():
    print("="*80)
    print("TRACK B: OLMoE SFT Mix Experiment (5000 steps)")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    output_base = "results/track_b"
    os.makedirs(output_base, exist_ok=True)

    all_results = []
    methods = ["standard_lora", "dr_lora", "hierarchical"]

    for method in methods:
        result = run_method(
            method_name=method,
            tokenizer=tokenizer,
            output_base=output_base
        )
        all_results.append(result)
        torch.cuda.empty_cache()

    # Summary
    with open(os.path.join(output_base, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("TRACK B COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_base}/all_results.json")
    print("\nNext: Run lm-eval harness on saved checkpoints for benchmark eval.")
    print(f"  Tasks: {EVAL_TASKS}")


if __name__ == "__main__":
    main()

# Hierarchical ReLU-LoRA Spawning for Parameter-Efficient MoE Adaptation

This repository contains the implementation for comparing three parameter-efficient fine-tuning methods on Mixture-of-Experts models:

1. **Standard LoRA** - Fixed-rank baseline (entanglement)
2. **DR-LoRA** - Dynamic rank growth (state-of-the-art)
3. **Hierarchical ReLU-LoRA Spawning** - Our proposed method (domain decoupling)

## 🎯 Research Goal

Prove that **Hierarchical ReLU-LoRA Spawning eliminates parameter entanglement** better than DR-LoRA's rank growth approach by spawning multiple independent sub-adapters with ReLU gates instead of growing a single monolithic adapter.

## 📊 Experimental Setup

Following [DR-LoRA paper](https://arxiv.org/abs/2601.04823) experimental protocol:

- **Model:** OLMoE-1B-7B-0924 (6.9B params, 1.3B activated)
- **Dataset:** OLMoE SFT Mix (38,000 steps, 1 epoch)
- **Benchmarks:** MMLU, HellaSwag, BBH, GSM8k, ARC-Challenge, HumanEval, IFEval
- **Hardware:** 15.6GB VRAM GPU (A100/L40S equivalent)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository (if applicable)
cd hierarchical-relu-lora

# Install dependencies
pip install -r requirements.txt
```

**Note:** Flash Attention 2 is optional but recommended for memory efficiency:
```bash
pip install flash-attn --no-build-isolation
```

### 2. Quick Test (Recommended)

Run 100 steps to verify everything works before full training:

```bash
# Test Standard LoRA
python train.py --method standard_lora --total_steps 100 --save_steps 50

# Test DR-LoRA
python train.py --method dr_lora --total_steps 100 --save_steps 50

# Test Hierarchical Spawning
python train.py --method hierarchical --total_steps 100 --save_steps 50
```

Check for:
- ✅ No CUDA OOM errors
- ✅ DR-LoRA triggers growth (should see 1-2 events in 100 steps)
- ✅ Hierarchical triggers spawn (may or may not spawn in 100 steps - depends on data)

### 3. Full Training

**Option A: Run all methods sequentially (recommended)**

Linux/Mac:
```bash
bash run_training.sh
```

Windows:
```bash
run_training.bat
```

**Option B: Run methods individually**

```bash
# Standard LoRA (~40 hours)
python train.py --method standard_lora

# DR-LoRA (~45 hours)
python train.py --method dr_lora

# Hierarchical Spawning (~45 hours)
python train.py --method hierarchical
```

**Total training time:** ~130-150 GPU-hours (~6 days on single GPU)

### 4. Evaluation

After training completes, evaluate each method:

```bash
# Standard LoRA
python eval/evaluate.py \
    --method standard_lora \
    --checkpoint_path ./results/standard_lora/checkpoint-38000

# DR-LoRA (with stability metrics)
python eval/evaluate.py \
    --method dr_lora \
    --checkpoint_path ./results/dr_lora/checkpoint-38000 \
    --compute_stability

# Hierarchical Spawning (with stability + JSD metrics)
python eval/evaluate.py \
    --method hierarchical \
    --checkpoint_path ./results/hierarchical/checkpoint-38000 \
    --compute_stability \
    --compute_jsd
```

### 5. Compare Results

Generate comprehensive comparison report:

```bash
python eval/compare_methods.py --eval_results_dir ./eval_results
```

This creates `comparison_report.txt` with:
- DR-LoRA Table 1 style benchmark comparison
- Stability metrics (zero-loss spawn validation)
- JSD metrics (domain specialization proof)
- Statistical significance

## 📁 Project Structure

```
hierarchical-relu-lora/
├── data/
│   └── dataset_builder.py          # OLMoE SFT Mix loader
├── baselines/
│   ├── standard_lora.py             # Fixed-rank LoRA (PEFT)
│   └── dr_lora.py                   # Dynamic rank growth
├── method/
│   └── hierarchical_spawning.py     # Our proposed method
├── eval/
│   ├── evaluate.py                  # LM Eval Harness wrapper
│   └── compare_methods.py           # Cross-method comparison
├── train.py                          # Universal training loop
├── config.py                         # Centralized hyperparameters
├── run_training.sh                   # Bash script (all methods)
├── run_training.bat                  # Windows batch script
├── requirements.txt                  # Dependencies
├── IMPLEMENTATION_STATUS.md          # Detailed status
└── README.md                         # This file
```

## 🔬 Method Details

### Standard LoRA (Baseline)

**Configuration:**
- Rank: 64
- Alpha: 128 (2×rank)
- Targets: Expert FFN layers (up_proj, down_proj)

**Expected Performance (DR-LoRA paper):** ~40.8% average accuracy

**Why it fails:** Forces all task knowledge into single uniform adapter → entanglement

---

### DR-LoRA (State-of-the-Art Competitor)

**Configuration:**
- Initial rank: 32
- Max rank: 128
- Target avg rank: 64
- Growth interval: Every 200 steps
- Saliency: S_i = (f_i × g_i) / (r_i + 1)^1.2

**Expected Performance (DR-LoRA paper):** ~41.5% average accuracy (+0.7 over Standard)

**How it works:**
1. Start with rank=32 for all experts
2. Track routing frequency (f_i) and rank importance (g_i) per expert
3. Compute saliency score every 200 steps
4. Grow high-saliency experts up to rank=128
5. Goal: Allocate more capacity to heavily-used experts

**Limitation:** Still monolithic adapter per expert → partial entanglement

---

### Hierarchical ReLU-LoRA Spawning (Our Method)

**Configuration:**
- Base rank: 16
- Max sub-adapters: 3 per expert
- Spawn threshold: alpha=1.3, tau_plateau=1e-4
- Saturation window: 50 steps

**Expected Performance (hypothesis):** ~42-43% average accuracy (+1.5 to +2.0 over Standard)

**How it works:**
1. Start with base LoRA (rank=16) per expert
2. Monitor bivariate saturation:
   - Learning plateau: |Δ importance| < 1e-4
   - High residual error: expert_loss > 1.3 × batch_loss
3. When both sustained for 50 steps → **SPAWN** new sub-adapter
4. Sub-adapter has independent ReLU gate: gate(x) = ReLU(w^T x)
5. Output: E_k(x) = base(x) + base_lora(x) + Σ gate_j(x) × lora_j(x)

**Key advantages:**
- **Zero-loss spawn:** New B matrices initialized to zero → no output change
- **Domain specialization:** ReLU gates learn to partition tasks
- **Independent learning:** Each sub-adapter updates without interfering
- **Provable with JSD:** High JSD between Python/Medical gate activations

---

## 📈 Success Criteria

### Minimum Viable Success
1. ✅ Hierarchical avg > DR-LoRA avg > Standard LoRA avg
2. ✅ Zero-loss spawn stability (|Δ loss| < 0.01)
3. ✅ At least 8 sub-adapters spawned across all experts

### Strong Success (Publication-Worthy)
1. ✅ Hierarchical avg ≥ DR-LoRA avg + 1.0 points
2. ✅ Statistically significant improvement (p < 0.05, 3 seeds)
3. ✅ Mean JSD > 0.3 proving domain partitioning
4. ✅ Visualizations match DR-LoRA paper quality

---

## 🛠️ Configuration

All hyperparameters are centralized in [config.py](config.py). To modify:

```python
from config import get_hierarchical_config

config = get_hierarchical_config()
config['hierarchical'].spawn_alpha = 1.5  # Adjust threshold
config['training'].learning_rate = 1e-5   # Adjust LR
```

Or use command-line arguments:

```bash
python train.py --method hierarchical --spawn_alpha 1.5 --learning_rate 1e-5
```

---

## 📊 Expected Outputs

### Training Outputs (per method)

```
results/
└── {method}/
    ├── args.json                    # Training arguments
    ├── checkpoint-5000/             # Intermediate checkpoints
    ├── checkpoint-38000/            # Final checkpoint
    │   ├── model.pt                 # Model state dict
    │   └── metrics.json             # Training metrics log
    ├── growth_events.json           # DR-LoRA only
    └── spawn_events.json            # Hierarchical only
```

### Evaluation Outputs

```
eval_results/
├── standard_lora/
│   └── benchmark_results.json
├── dr_lora/
│   ├── benchmark_results.json
│   └── stability_metrics.json
└── hierarchical/
    ├── benchmark_results.json
    ├── stability_metrics.json
    └── jsd_metrics.json
```

### Comparison Report

```
comparison_report.txt
```

Example content:
```
====================================================================================================
BENCHMARK COMPARISON (DR-LoRA Table 1 Style)
====================================================================================================
Task                 |  Standard Lora |        Dr Lora | Hierarchical | Δ (Hier - DR)
----------------------------------------------------------------------------------------------------
mmlu                 |        50.30   |        50.60   |        50.70 |        +0.10
hellaswag            |        55.90   |        55.80   |        56.10 |        +0.30
gsm8k                |        30.20   |        31.60   |        33.50 |        +1.90
...
----------------------------------------------------------------------------------------------------
Average              |        40.80   |        41.50   |        42.80 |        +1.30
====================================================================================================
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce max_seq_length: `--max_seq_length 2048` (from 4096)
2. Ensure gradient checkpointing is enabled (default)
3. Use Flash Attention 2: `pip install flash-attn`
4. Reduce micro_batch_size to 1 (already default)

### DR-LoRA Not Growing

**Symptom:** No growth events after 1000 steps

**Diagnosis:**
- Print routing frequency: Should show some experts >0.1
- Print rank importance: Should be >0 after backward pass
- Check growth window: Must be between [warmup_end, total_steps-200]

**Fix:**
```bash
python train.py --method dr_lora --logging_steps 10  # Frequent logging
```

### Hierarchical Not Spawning

**Symptom:** Zero spawns after 5000 steps

**Diagnosis:**
- Check expert_loss/batch_loss ratio: Should exceed alpha occasionally
- Check importance delta: Should drop below tau_plateau

**Fix:**
```bash
python train.py --method hierarchical --spawn_alpha 1.1 --tau_plateau 1e-3
```

### Low JSD (<0.1)

**Symptom:** JSD metrics show weak domain specialization

**Possible causes:**
1. Not enough sub-adapters spawned (need >5)
2. Sub-router sigma too small (gates don't activate)
3. Probe dataset too similar

**Fix:**
- Lower spawn thresholds to get more sub-adapters
- Check gate activation statistics in logs

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{hierarchical-relu-lora-2026,
  title={Hierarchical ReLU-LoRA Spawning for Parameter-Efficient MoE Adaptation},
  author={[Your Name]},
  year={2026}
}
```

And the DR-LoRA paper:

```bibtex
@article{dr-lora-2025,
  title={DR-LoRA: Efficient Fine-Tuning of Mixture-of-Experts via Dynamic Rank Allocation},
  author={[DR-LoRA Authors]},
  journal={arXiv preprint arXiv:2601.04823},
  year={2025}
}
```

---

## 📧 Support

For questions or issues:
1. Check [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed status
2. Review training logs in `results/{method}/`
3. Open an issue with error messages and logs

---

## 🎉 Acknowledgments

- OLMoE team for the excellent MoE model
- DR-LoRA authors for the baseline implementation
- HuggingFace for PEFT and Transformers libraries
- EleutherAI for LM Evaluation Harness

---

**Good luck with your experiments! 🚀**

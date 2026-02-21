#!/bin/bash

# Training script for all three methods
# Follows DR-LoRA experimental setup (Table 6)

set -e

# Configuration
BASE_MODEL="allenai/OLMoE-1B-7B-0924"
OUTPUT_DIR="./results"
SEED=42

# Training hyperparameters (matching DR-LoRA)
TOTAL_STEPS=38000
MICRO_BATCH=1
GRAD_ACCUM=16
MAX_SEQ_LEN=4096
LR=2e-5
WARMUP_RATIO=0.03

echo "=========================================="
echo "Hierarchical ReLU-LoRA Training Pipeline"
echo "=========================================="
echo ""
echo "Model: $BASE_MODEL"
echo "Total steps: $TOTAL_STEPS"
echo "Effective batch size: $((MICRO_BATCH * GRAD_ACCUM))"
echo "Learning rate: $LR"
echo ""

# Method 1: Standard LoRA
echo "=========================================="
echo "Method 1: Standard LoRA (rank=64)"
echo "=========================================="
python train.py \
    --method standard_lora \
    --base_model $BASE_MODEL \
    --total_steps $TOTAL_STEPS \
    --micro_batch_size $MICRO_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_seq_length $MAX_SEQ_LEN \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --save_steps 5000 \
    --seed $SEED

echo ""
echo "Standard LoRA training complete!"
echo ""

# Method 2: DR-LoRA
echo "=========================================="
echo "Method 2: DR-LoRA (rinit=32, rmax=128, rtarget=64)"
echo "=========================================="
python train.py \
    --method dr_lora \
    --base_model $BASE_MODEL \
    --total_steps $TOTAL_STEPS \
    --micro_batch_size $MICRO_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_seq_length $MAX_SEQ_LEN \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --rank_init 32 \
    --rank_max 128 \
    --rank_target 64 \
    --lora_alpha 128 \
    --dr_beta 0.9 \
    --dr_gamma 1.2 \
    --growth_interval 200 \
    --p_grow 0.1 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --save_steps 5000 \
    --seed $SEED

echo ""
echo "DR-LoRA training complete!"
echo ""

# Method 3: Hierarchical Spawning
echo "=========================================="
echo "Method 3: Hierarchical ReLU-LoRA Spawning (base_rank=16)"
echo "=========================================="
python train.py \
    --method hierarchical \
    --base_model $BASE_MODEL \
    --total_steps $TOTAL_STEPS \
    --micro_batch_size $MICRO_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_seq_length $MAX_SEQ_LEN \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --base_rank 16 \
    --lora_alpha 32 \
    --spawn_alpha 1.3 \
    --tau_plateau 1e-4 \
    --saturation_window 50 \
    --max_sub_adapters 3 \
    --dr_beta 0.9 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --save_steps 5000 \
    --seed $SEED

echo ""
echo "Hierarchical Spawning training complete!"
echo ""

echo "=========================================="
echo "All training runs complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Run evaluation: python eval/evaluate.py"
echo "2. Compare results across methods"
echo "3. Generate visualizations"

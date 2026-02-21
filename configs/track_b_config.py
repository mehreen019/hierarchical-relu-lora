# configs/track_b_config.py

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
DATASET = "allenai/tulu-v3.1-mix-preview-4096-OLMoE"

TOTAL_STEPS = 5000
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
MAX_SEQ_LENGTH = 2048           # reduced from 4096 for memory safety
LEARNING_RATE = 2e-5
LR_SCHEDULE = "linear"
WARMUP_STEPS = 150
WEIGHT_DECAY = 0.0

# Same method configs as Track A — same hyperparams, different data/duration
# Import from track_a_config: STANDARD_LORA, DR_LORA, HIERARCHICAL
# Override only DR_LORA T_grow for 5000-step run
DR_LORA_TRACK_B_OVERRIDES = {
    "T_grow": 250,              # back to ~5% of run length
    "warmup_steps": 150,
    "cooldown_steps": 250,
}
HIERARCHICAL_TRACK_B_OVERRIDES = {
    "window": 50,               # back to standard window
    "tau_plateau": 1e-4,        # back to standard sensitivity
    "max_spawn_step": 4500,     # don't spawn in last 500 steps
}

EVAL_TASKS = ["mmlu", "hellaswag", "bbh", "gsm8k", "arc_challenge", "humaneval", "ifeval"]

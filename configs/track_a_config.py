# configs/track_a_config.py

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

# Dataset options - chosen AFTER Jaccard diagnostic
# If Python/Medical Jaccard overlap > 0.6: use OPTION_A
# If overlap < 0.3: use OPTION_B
DATASET_OPTION_A = {
    "primary": "iamtarun/python_code_instructions_18k_alpaca",
    "primary_text_col": "output",   # column that contains the text
    "conflict": "qiaojin/PubMedQA",
    "conflict_text_col": "question",
    "conflict_config": "pqa_labeled",
    "domain_names": ("python", "medical")
}
DATASET_OPTION_B = {
    "primary": "openai/gsm8k",
    "primary_text_col": "question",
    "conflict": "euclaise/writingprompts",
    "conflict_text_col": "text",
    "conflict_config": None,
    "domain_names": ("math", "creative")
}

# Training
TOTAL_STEPS = 1000
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16       # effective batch = 16
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
LR_SCHEDULE = "linear"
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.0
PRECISION = "bf16"               # QLoRA uses 4-bit base + bf16 adapters

# Standard LoRA config
STANDARD_LORA = {
    "r": 64,
    "lora_alpha": 128,
    "target_modules": ["up_proj", "down_proj"],
    "lora_dropout": 0.0,
    "bias": "none"
}

# DR-LoRA config — temporally scaled for 1000 steps
DR_LORA = {
    "r_init": 32,
    "r_max": 128,
    "r_target": 64,
    "beta": 0.9,                 # EMA decay
    "gamma": 1.2,                # rank penalty exponent
    "T_grow": 50,                # SCALED from 200 → 50 (proportional to run length)
    "p_grow": 0.1,               # max 10% of free capacity per event
    "warmup_steps": 100,         # don't grow before step 100
    "cooldown_steps": 50,        # don't grow in last 50 steps
    "target_modules": ["up_proj", "down_proj"]
}

# Hierarchical Spawning config — temporally scaled for 1000 steps
HIERARCHICAL = {
    "base_rank": 16,
    "alpha_ratio": 1.2,          # expert_loss > 1.2 * global_loss triggers
    "tau_plateau": 1e-3,         # more aggressive than default for short runs
    "window": 20,                # shorter window for faster trigger
    "max_spawn_step": 700,       # CRITICAL: no spawning after step 700
    "max_sub_adapters": 3,       # cap per expert
    "sigma_scale": 1e-4,         # sub-router init noise scale
    "target_modules": ["up_proj", "down_proj"]
}

# Conflict grid
RUNS = {
    "run_a": {"conflict_ratio": 0.0, "label": "0% Conflict"},
    "run_b": {"conflict_ratio": 0.2, "label": "20% Conflict"},
    "run_c": {"conflict_ratio": 0.5, "label": "50% Conflict"},
}

# Probe dataset sizes (for JSD eval — held out, never seen in training)
PROBE_N_PER_DOMAIN = 150

# Evaluation
EVAL_N_PYTHON = 50      # examples from HumanEval or held-out CodeAlpaca
EVAL_N_MEDICAL = 100    # held-out PubMedQA examples

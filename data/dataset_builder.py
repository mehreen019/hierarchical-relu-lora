# data/dataset_builder.py

import torch
from datasets import load_dataset, interleave_datasets, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def _load_and_format(dataset_name, text_col, config_name, split, n_skip=0, n_take=None):
    """Load a dataset and return it with a unified 'text' column."""
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    # Rename target column to 'text' if needed
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    # Keep only 'text' column
    cols_to_remove = [c for c in ds.column_names if c != "text"]
    ds = ds.remove_columns(cols_to_remove)

    # Handle probe split: skip first n_skip, take n_take
    if n_skip > 0 or n_take is not None:
        end = (n_skip + n_take) if n_take else len(ds)
        ds = ds.select(range(n_skip, min(end, len(ds))))

    return ds


def _tokenize(ds, tokenizer, max_length, domain_label=None):
    """Tokenize dataset. Optionally attach a domain label (stored but not as tensor)."""
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        if domain_label is not None:
            out["domain"] = [domain_label] * len(examples["text"])
        return out

    remove_cols = ["text"]
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    return tokenized


def build_conflict_dataloader(
    dataset_config: dict,
    tokenizer,
    conflict_ratio: float,
    max_length: int = 2048,
    batch_size: int = 1,
    n_training_examples: int = 2000,
    seed: int = 42
) -> DataLoader:
    """
    Build a blended dataloader for Track A conflict grid.

    Args:
        dataset_config: DATASET_OPTION_A or DATASET_OPTION_B from track_a_config
        conflict_ratio: 0.0, 0.2, or 0.5
        n_training_examples: total examples to draw (caps the dataset size)
        seed: fixed seed for interleaving — MUST be same across all 3 methods
    """
    primary_raw = _load_and_format(
        dataset_config["primary"],
        dataset_config["primary_text_col"],
        config_name=None,
        split="train",
        n_take=n_training_examples
    )
    conflict_raw = _load_and_format(
        dataset_config["conflict"],
        dataset_config["conflict_text_col"],
        config_name=dataset_config.get("conflict_config"),
        split="train",
        n_take=n_training_examples
    )

    primary_tok = _tokenize(primary_raw, tokenizer, max_length)
    conflict_tok = _tokenize(conflict_raw, tokenizer, max_length)

    if conflict_ratio == 0.0:
        mixed = primary_tok
    else:
        mixed = interleave_datasets(
            [primary_tok, conflict_tok],
            probabilities=[1.0 - conflict_ratio, conflict_ratio],
            seed=seed,
            stopping_strategy="first_exhausted"
        )

    tensor_cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in mixed.column_names:
        tensor_cols.append("token_type_ids")
    mixed.set_format(type="torch", columns=tensor_cols)

    # Labels = input_ids shifted (handled by DataCollatorForLanguageModeling)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    loader = DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=False,   # interleave_datasets controls ordering
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    return loader


def build_probe_datasets(dataset_config: dict, tokenizer, n_per_domain: int = 150,
                          max_length: int = 512, skip: int = 2000):
    """
    Build held-out probe datasets for JSD evaluation.
    Skips the first `skip` examples (which were used for training) to prevent leakage.
    Returns: (primary_probe_list, conflict_probe_list)
    Each is a list of tokenized dicts with domain labels.
    """
    primary_raw = _load_and_format(
        dataset_config["primary"],
        dataset_config["primary_text_col"],
        config_name=None,
        split="train",
        n_skip=skip,
        n_take=n_per_domain
    )
    conflict_raw = _load_and_format(
        dataset_config["conflict"],
        dataset_config["conflict_text_col"],
        config_name=dataset_config.get("conflict_config"),
        split="train",
        n_skip=skip,
        n_take=n_per_domain
    )

    # Return as simple lists of raw text (JSD eval tokenizes on the fly)
    primary_texts = [ex["text"] for ex in primary_raw]
    conflict_texts = [ex["text"] for ex in conflict_raw]

    domain_names = dataset_config["domain_names"]
    return (
        [{"text": t, "domain": domain_names[0]} for t in primary_texts],
        [{"text": t, "domain": domain_names[1]} for t in conflict_texts]
    )


def build_track_b_dataloader(tokenizer, max_length=2048, batch_size=1, seed=42):
    """Dataloader for Track B using the OLMoE SFT mix."""
    from configs.track_b_config import DATASET
    ds = load_dataset(DATASET, split="train")

    # Tulu dataset uses 'messages' column in chat format — extract to text
    def format_tulu(examples):
        texts = []
        for msgs in examples["messages"]:
            text = ""
            for msg in msgs:
                role = msg.get("role", "")
                content = msg.get("content", "")
                text += f"<|{role}|>\n{content}\n"
            texts.append(text.strip())
        return {"text": texts}

    ds = ds.map(format_tulu, batched=True, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=seed)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False,
                      collate_fn=collator, num_workers=2, pin_memory=True)

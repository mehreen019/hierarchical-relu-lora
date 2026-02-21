# train.py

import torch
import json
import os
from transformers import get_linear_schedule_with_warmup


def get_expert_loss_map(router_logits_list, global_loss, n_experts=64):
    """
    Approximate per-expert loss by weighting global loss by routing probability.
    Returns dict: "L{i}_E{j}" -> approximate loss float.
    """
    expert_loss_map = {}
    if not router_logits_list:
        return expert_loss_map

    for layer_idx, rl in enumerate(router_logits_list):
        if rl is None:
            continue
        if rl.dim() == 3:
            rl = rl.view(-1, rl.shape[-1])
        probs = torch.softmax(rl.float(), dim=-1).mean(0)  # (n_experts,)

        for eid in range(probs.shape[0]):
            # Experts used more heavily get a higher proxy loss
            key = f"L{layer_idx}_E{eid}"
            expert_loss_map[key] = global_loss * (1.0 + probs[eid].item())

    return expert_loss_map


def train(
    method_name: str,
    model,
    dataloader,
    total_steps: int,
    lr: float,
    warmup_steps: int,
    output_dir: str,
    dr_lora_model=None,       # pass if method == "dr_lora"
    hier_model=None,          # pass if method == "hierarchical"
    log_every: int = 20,
    eval_fn=None,             # optional callable(model, step) -> dict
    eval_every: int = 200
):
    os.makedirs(output_dir, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0,
                                   betas=(0.9, 0.999), eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Propagate total_steps to DR-LoRA for growth window calculation
    if dr_lora_model is not None:
        dr_lora_model.total_steps = total_steps

    # Pass lr to hierarchical model for new param groups
    if hier_model is not None:
        hier_model.cfg["lr"] = lr

    metrics_log = []
    step = 0
    data_iter = iter(dataloader)

    model.train()

    # Track loss just before and after spawn/growth for stability metric
    pre_event_loss = None
    growth_steps = set()

    while step < total_steps:
        # Refresh iterator if exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_router_logits=True,
            return_dict=True
        )

        loss = outputs.loss

        # DR-LoRA: update routing stats before backward
        if dr_lora_model is not None and outputs.router_logits:
            dr_lora_model.update_routing_stats(outputs.router_logits)

        # Backward
        loss.backward()

        # DR-LoRA: update rank importance from gradients
        if dr_lora_model is not None:
            dr_lora_model.update_rank_importance()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )

        # DR-LoRA: maybe grow ranks (zero-loss: new B cols are zeroed)
        if dr_lora_model is not None:
            prev_growth_count = len(dr_lora_model.growth_log)
            dr_lora_model.maybe_grow(step)
            if len(dr_lora_model.growth_log) > prev_growth_count:
                growth_steps.add(step)

        # Hierarchical: check saturation and spawn
        if hier_model is not None and outputs.router_logits:
            expert_loss_map = get_expert_loss_map(outputs.router_logits, loss.item())
            prev_spawns = hier_model.total_spawns
            hier_model.check_and_spawn(expert_loss_map, loss.item(), step, optimizer)
            if hier_model.total_spawns > prev_spawns:
                growth_steps.add(step)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        record = {
            "step": step,
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        }
        if hier_model is not None:
            record["total_spawns"] = hier_model.total_spawns
        if dr_lora_model is not None:
            record["total_growths"] = len(dr_lora_model.growth_log)

        metrics_log.append(record)

        if step % log_every == 0:
            extra = f"spawns={record.get('total_spawns', record.get('total_growths', 0))}"
            print(f"  [{method_name}] step={step:5d} | loss={loss.item():.4f} | {extra}")

        if eval_fn is not None and step % eval_every == 0 and step > 0:
            eval_results = eval_fn(model, step)
            record["eval"] = eval_results
            print(f"  [{method_name}] EVAL at step {step}: {eval_results}")

        step += 1

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Save adapter weights only
    adapter_state = {
        k: v for k, v in model.state_dict().items()
        if any(tag in k for tag in ["lora", "sub_adapter", "w_gate"])
    }
    torch.save(adapter_state, os.path.join(output_dir, "adapter_weights.pt"))

    print(f"\n  [{method_name}] Training complete. {total_steps} steps.")
    print(f"   Metrics: {metrics_path}")

    return model, metrics_log, list(growth_steps)

# baselines/dr_lora.py

import torch
import torch.nn as nn
import math
import numpy as np
from collections import deque
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


class DynamicRankLoRALinear(nn.Module):
    """
    A LoRA linear layer with pre-allocated rmax capacity.
    Only the masked ranks are active at any time.
    Zero-init of new B columns guarantees zero-loss growth events.
    """
    def __init__(self, base_layer: nn.Linear, r_init: int, r_max: int, alpha: int):
        super().__init__()
        self.base_layer = base_layer
        self.r_max = r_max
        self.alpha = alpha
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Pre-allocate full rmax space
        self.lora_A = nn.Parameter(torch.empty(r_max, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r_max))

        # Kaiming init for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Binary mask: True = active rank
        self.register_buffer("mask", torch.zeros(r_max, dtype=torch.bool))
        self.mask[:r_init] = True

    @property
    def active_rank(self):
        return self.mask.sum().item()

    def forward(self, x):
        base_out = self.base_layer(x)
        scaling = self.alpha / max(self.active_rank, 1)

        # Apply mask to select only active ranks
        active_A = self.lora_A[self.mask]       # (active_r, in_features)
        active_B = self.lora_B[:, self.mask]    # (out_features, active_r)

        lora_out = (x @ active_A.T) @ active_B.T
        return base_out + scaling * lora_out

    def grow(self, n_new_ranks: int):
        """Activate n_new_ranks additional ranks. Zero-init their B columns."""
        inactive = (~self.mask).nonzero(as_tuple=True)[0]
        if len(inactive) == 0:
            return 0

        n_actual = min(n_new_ranks, len(inactive))
        new_indices = inactive[:n_actual]

        self.mask[new_indices] = True

        # CRITICAL: zero-init B for new ranks to guarantee zero-loss growth
        with torch.no_grad():
            self.lora_B[:, new_indices] = 0.0

        return n_actual


class DRLoRAModel(nn.Module):
    def __init__(self, base_model, cfg: dict):
        super().__init__()
        self.model = base_model
        self.r_init = cfg["r_init"]
        self.r_max = cfg["r_max"]
        self.r_target = cfg["r_target"]
        self.beta = cfg["beta"]
        self.gamma = cfg["gamma"]
        self.T_grow = cfg["T_grow"]
        self.p_grow = cfg["p_grow"]
        self.warmup_steps = cfg["warmup_steps"]
        self.cooldown_steps = cfg["cooldown_steps"]
        self.total_steps = None  # set before training

        # State
        self.routing_freq_ema = {}   # expert_key -> float
        self.rank_importance_ema = {}  # expert_key -> float
        self.growth_log = []
        self.lora_layers = {}

        self._inject(cfg["target_modules"])

    def _get_module(self, name):
        parts = name.split(".")
        mod = self.model
        for p in parts:
            mod = getattr(mod, p)
        return mod

    def _inject(self, target_modules):
        replacements = {}
        for name, module in self.model.named_modules():
            if (isinstance(module, nn.Linear) and
                    any(name.endswith(t) for t in target_modules) and
                    "experts" in name):

                parts = name.split(".")
                try:
                    layer_id = int(parts[parts.index("layers") + 1])
                    expert_id = int(parts[parts.index("experts") + 1])
                except (ValueError, IndexError):
                    continue

                parent_name = ".".join(parts[:-1])
                attr = parts[-1]
                key = f"L{layer_id}_E{expert_id}_{attr}"

                lora_layer = DynamicRankLoRALinear(
                    module, self.r_init, self.r_max, alpha=self.r_init * 2
                )
                replacements[(parent_name, attr, key)] = lora_layer

        for (parent_name, attr, key), lora_layer in replacements.items():
            parent = self._get_module(parent_name)
            setattr(parent, attr, lora_layer)
            self.lora_layers[key] = lora_layer

        print(f"DR-LoRA: injected into {len(self.lora_layers)} expert linear layers")

    def update_routing_stats(self, router_logits_list):
        """
        Called each step with the list of router logits.
        Updates routing frequency EMA per expert.
        router_logits_list: list of (B*S, n_experts) tensors, one per layer.
        """
        for layer_idx, rl in enumerate(router_logits_list):
            if rl is None:
                continue
            if rl.dim() == 3:
                rl = rl.view(-1, rl.shape[-1])
            probs = torch.softmax(rl.float(), dim=-1).mean(0)  # (n_experts,)

            for expert_id in range(probs.shape[0]):
                key_prefix = f"L{layer_idx}_E{expert_id}"
                f_new = probs[expert_id].item()
                for suffix in ["_up_proj", "_down_proj"]:
                    key = key_prefix + suffix
                    if key not in self.routing_freq_ema:
                        self.routing_freq_ema[key] = f_new
                        self.rank_importance_ema[key] = 0.0
                    else:
                        self.routing_freq_ema[key] = (
                            self.beta * self.routing_freq_ema[key] +
                            (1 - self.beta) * f_new
                        )

    def update_rank_importance(self):
        """
        Called after backward. Computes rank importance from gradient magnitudes.
        g_i = mean(|grad_A| * |grad_B|) across active ranks — proxy for sensitivity.
        """
        for key, layer in self.lora_layers.items():
            if layer.lora_A.grad is None or layer.lora_B.grad is None:
                continue

            active_mask = layer.mask
            grad_A = layer.lora_A.grad[active_mask].abs().mean().item()
            grad_B = layer.lora_B.grad[:, active_mask].abs().mean().item()
            importance = grad_A * grad_B

            self.rank_importance_ema[key] = (
                self.beta * self.rank_importance_ema.get(key, importance) +
                (1 - self.beta) * importance
            )

    def compute_saliency(self, key):
        f_i = self.routing_freq_ema.get(key, 0.0)
        g_i = self.rank_importance_ema.get(key, 0.0)
        r_i = self.lora_layers[key].active_rank
        return (f_i * g_i) / ((r_i + 1) ** self.gamma)

    def maybe_grow(self, step: int):
        """
        Called every T_grow steps. Grows ranks for the highest-saliency experts.
        Respects warmup and cooldown windows.
        """
        if self.total_steps is None:
            raise ValueError("Set dr_lora_model.total_steps before training.")

        if step < self.warmup_steps:
            return
        if step > self.total_steps - self.cooldown_steps:
            return
        if step % self.T_grow != 0:
            return

        # Compute quota: how many total new ranks to distribute this event
        n_growth_events = (self.total_steps - self.warmup_steps - self.cooldown_steps) // self.T_grow
        total_new_ranks_budget = len(self.lora_layers) * (self.r_target - self.r_init)
        quota_per_event = max(1, total_new_ranks_budget // max(n_growth_events, 1))

        # Sort by saliency descending
        saliencies = {k: self.compute_saliency(k) for k in self.lora_layers}
        sorted_keys = sorted(saliencies, key=saliencies.get, reverse=True)

        remaining = quota_per_event
        events = []

        for key in sorted_keys:
            if remaining <= 0:
                break
            layer = self.lora_layers[key]
            free = self.r_max - layer.active_rank
            n_grow = min(int(free * self.p_grow) + 1, remaining, free)
            if n_grow > 0:
                actual = layer.grow(n_grow)
                # Reset rank importance after growth
                self.rank_importance_ema[key] = 0.0
                remaining -= actual
                events.append((key, actual, layer.active_rank))

        if events:
            self.growth_log.append({"step": step, "events": events})
            print(f"  [DR-LoRA] Step {step}: grew {len(events)} layers, "
                  f"quota_used={quota_per_event - remaining}/{quota_per_event}")

# method/hierarchical_spawning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd as scipy_svd
from collections import deque
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


class SubAdapter(nn.Module):
    """
    One spawned ReLU-gated LoRA sub-adapter.
    Forward: ReLU(x @ w_gate) * (x @ A^T @ B^T)
    B is zero-initialized → zero contribution at spawn time.
    """
    def __init__(self, in_features: int, out_features: int, rank: int,
                 sigma: float, device, dtype):
        super().__init__()
        self.rank = rank

        # Sub-router (symmetry breaking init)
        self.w_gate = nn.Parameter(
            torch.randn(in_features, device=device, dtype=dtype) * sigma
        )

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def init_from_svd(self, grad_matrix: torch.Tensor):
        """LoRA-GA style: A ← top-r right singular vectors of residual gradient."""
        try:
            g = grad_matrix.float().cpu().numpy()
            # Clip for numerical stability
            g = np.clip(g, -1e4, 1e4)
            _, S, Vt = scipy_svd(g, full_matrices=False)
            r = self.rank
            top_Vt = torch.tensor(Vt[:r], dtype=self.lora_A.dtype,
                                   device=self.lora_A.device)
            self.lora_A.data.copy_(top_Vt)
            self.lora_B.data.zero_()  # zero-init B preserves zero-loss spawn
            print(f"    [spawn] SVD success, top singular value = {S[0]:.4f}")
        except Exception as e:
            print(f"    [spawn] SVD failed ({e}), keeping kaiming init")

    def forward(self, x):
        # Gate: scalar per token position
        gate = F.relu(x @ self.w_gate)          # (..., )
        gate = gate.unsqueeze(-1)                # (..., 1)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T   # (..., out_features)
        return gate * lora_out


class HierarchicalLoRALinear(nn.Module):
    """
    Replaces a frozen expert Linear layer.
    Output = base(x) + scaling * base_lora(x) + scaling * sum_j sub_j(x)
    """
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int,
                 sigma_scale: float, layer_id: int, expert_id: int, proj_name: str):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.sigma_scale = sigma_scale
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.proj_name = proj_name
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.scaling = alpha / rank

        for p in self.base_layer.parameters():
            p.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Base LoRA (L_{k,0})
        self.base_lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.base_lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.base_lora_A, a=np.sqrt(5))

        self.sub_adapters = nn.ModuleList()
        self._spawn_steps = []

    def forward(self, x):
        out = self.base_layer(x)
        out = out + self.scaling * ((x @ self.base_lora_A.T) @ self.base_lora_B.T)
        for sub in self.sub_adapters:
            out = out + self.scaling * sub(x)
        return out

    def spawn(self, step: int, residual_grad: torch.Tensor = None):
        """
        Spawn one sub-adapter. Zero-loss guaranteed by B=0 init.
        """
        w = self.base_layer.weight
        sigma = self.sigma_scale * (w.float().var().item() ** 0.5)
        sigma = max(sigma, 1e-7)   # floor to avoid dead gate

        sub = SubAdapter(
            self.in_features, self.out_features, self.rank,
            sigma=sigma,
            device=w.device,
            dtype=w.dtype
        )

        if residual_grad is not None:
            sub.init_from_svd(residual_grad)

        self.sub_adapters.append(sub)
        self._spawn_steps.append(step)
        print(f"    [spawn] L{self.layer_id} E{self.expert_id} {self.proj_name} "
              f"sub-adapter #{len(self.sub_adapters)} at step {step} "
              f"| sigma={sigma:.2e}")
        return sub

    def new_parameters(self):
        """Return parameters of the most recently spawned sub-adapter."""
        if self.sub_adapters:
            return list(self.sub_adapters[-1].parameters())
        return []


class SaturationMonitor:
    """
    Bivariate trigger: fires when BOTH conditions hold for `window` consecutive steps.
      1. Learning plateau: |EMA delta of rank importance| < tau_plateau
      2. High residual error: expert_loss > alpha_ratio * global_loss
    """
    def __init__(self, alpha_ratio: float, tau_plateau: float, window: int, beta: float = 0.9):
        self.alpha_ratio = alpha_ratio
        self.tau_plateau = tau_plateau
        self.window = window
        self.beta = beta

        self.importance_ema = None
        self.prev_importance_ema = None
        self._plateau_window = deque(maxlen=window)
        self._error_window = deque(maxlen=window)

    def update(self, expert_loss: float, global_loss: float,
               grad_A: torch.Tensor, grad_B: torch.Tensor) -> bool:
        """Returns True if saturation detected (spawn should be triggered)."""
        # Rank importance = mean(|grad_A| * |grad_B|)
        importance = (grad_A.abs().mean() * grad_B.abs().mean()).item()

        if self.importance_ema is None:
            self.importance_ema = importance
            self.prev_importance_ema = importance
            return False

        self.prev_importance_ema = self.importance_ema
        self.importance_ema = self.beta * self.importance_ema + (1 - self.beta) * importance

        delta = abs(self.importance_ema - self.prev_importance_ema)
        plateau = delta < self.tau_plateau
        high_error = (global_loss > 0) and (expert_loss > self.alpha_ratio * global_loss)

        self._plateau_window.append(plateau)
        self._error_window.append(high_error)

        if (len(self._plateau_window) == self.window and
                all(self._plateau_window) and all(self._error_window)):
            # Reset after firing to prevent immediate re-trigger
            self._plateau_window.clear()
            self._error_window.clear()
            return True

        return False


class HierarchicalSpawningModel(nn.Module):
    def __init__(self, base_model, cfg: dict):
        super().__init__()
        self.model = base_model
        self.cfg = cfg
        self.max_spawn_step = cfg["max_spawn_step"]
        self.max_sub_adapters = cfg["max_sub_adapters"]

        self.hier_layers = {}     # key -> HierarchicalLoRALinear
        self.monitors = {}        # key -> SaturationMonitor
        self.spawn_log = []
        self.total_spawns = 0

        self._inject(cfg["target_modules"])
        self._init_monitors()

    def _get_module(self, name):
        parts = name.split(".")
        mod = self.model
        for p in parts:
            mod = getattr(mod, p)
        return mod

    def _inject(self, target_modules):
        replacements = []
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
                replacements.append((name, parent_name, attr, key, layer_id, expert_id, module))

        for name, parent_name, attr, key, layer_id, expert_id, module in replacements:
            hier = HierarchicalLoRALinear(
                module,
                rank=self.cfg["base_rank"],
                alpha=self.cfg["base_rank"] * 2,
                sigma_scale=self.cfg["sigma_scale"],
                layer_id=layer_id,
                expert_id=expert_id,
                proj_name=attr
            )
            parent = self._get_module(parent_name)
            setattr(parent, attr, hier)
            self.hier_layers[key] = hier

        print(f"Hierarchical Spawning: injected into {len(self.hier_layers)} layers")

    def _init_monitors(self):
        for key in self.hier_layers:
            self.monitors[key] = SaturationMonitor(
                alpha_ratio=self.cfg["alpha_ratio"],
                tau_plateau=self.cfg["tau_plateau"],
                window=self.cfg["window"]
            )

    def check_and_spawn(self, expert_loss_map: dict, global_loss: float, step: int,
                        optimizer: torch.optim.Optimizer):
        """
        Called after backward() each step.
        expert_loss_map: dict mapping expert_key_prefix -> approximate loss
        """
        if step > self.max_spawn_step:
            return

        for key, layer in self.hier_layers.items():
            if len(layer.sub_adapters) >= self.max_sub_adapters:
                continue

            expert_key = f"L{layer.layer_id}_E{layer.expert_id}"
            expert_loss = expert_loss_map.get(expert_key, global_loss)

            if layer.base_lora_A.grad is None or layer.base_lora_B.grad is None:
                continue

            should_spawn = self.monitors[key].update(
                expert_loss=expert_loss,
                global_loss=global_loss,
                grad_A=layer.base_lora_A.grad,
                grad_B=layer.base_lora_B.grad
            )

            if should_spawn:
                residual_grad = layer.base_lora_B.grad.data.clone()
                sub = layer.spawn(step=step, residual_grad=residual_grad)

                # THE OPTIMIZER FIX: add new params without resetting existing momentum
                new_params = layer.new_parameters()
                optimizer.add_param_group({
                    "params": new_params,
                    "lr": self.cfg.get("lr", 2e-5),
                    "betas": (0.9, 0.999),
                    "weight_decay": 0.0,
                    "eps": 1e-8
                })

                self.total_spawns += 1
                self.spawn_log.append({
                    "step": step,
                    "key": key,
                    "expert_loss": expert_loss,
                    "global_loss": global_loss,
                    "n_sub_adapters": len(layer.sub_adapters)
                })

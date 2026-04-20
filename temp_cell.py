import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalPhimoeExperts(nn.Module):
    """
    Drop-in replacement for Phi-MoE's packed expert block.

    The SparseMoeBlock calls: self.experts(hidden_states, top_k_ids, top_k_weights)
    where hidden_states is [T, model_dim].  We intercept this, compute the
    original output, and add per-expert LoRA corrections:

        E_k(x) = W_k x + L_{k,0}(x) + Σ_j ReLU(w_{k,j}^T x) L_{k,j}(x)

    NOTE on dimensions
    ------------------
    down_proj.shape = [num_experts, model_dim, ffn_dim]
                                   ^^^^^^^^^  ^^^^^^^^
                                   out_f      in_f (internal ffn dim)

    hidden_states arriving at the experts is [T, model_dim].
    Our LoRA correction therefore lives in model_dim → model_dim space,
    i.e. lora_in = lora_out = out_f = model_dim.
    Using in_f (ffn_dim) here was the critical shape-mismatch bug.
    """

    def __init__(self, original_experts, base_rank: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.original = original_experts

        # Freeze every original expert parameter
        for p in self.original.parameters():
            p.requires_grad = False

        # Infer packed-weight dimensions
        # down_proj: [num_experts, model_dim, ffn_dim]
        self.num_experts, self.out_f, self.in_f = self.original.down_proj.shape
        self.dtype = self.original.down_proj.dtype

        # LoRA correction lives in model_dim space (input to & output from experts)
        # FIX: was using self.in_f (ffn_dim) — wrong dimension for hidden_states
        self.lora_dim = self.out_f   # = model_dim

        # Base LoRA per expert — registered as nn.ModuleList so optimizer finds them
        dev   = self.original.down_proj.device
        dtype = self.original.down_proj.dtype

        self.base_loras = nn.ModuleList([
            HierarchicalExpert(
                in_features=self.lora_dim,
                out_features=self.lora_dim,
                base_rank=base_rank,
                lora_alpha=lora_alpha,
            ).to(dev).to(dtype)
            for _ in range(self.num_experts)
        ])

        # Spawned adapters: plain Python lists, added to optimizer manually on spawn.
        # spawn_loras[k] = [HierarchicalExpert, ...]
        # spawn_gates[k] = [nn.Parameter of shape [lora_dim], ...]
        self.spawn_loras: list[list] = [[] for _ in range(self.num_experts)]
        self.spawn_gates: list[list] = [[] for _ in range(self.num_experts)]

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.original.down_proj.device

    # ------------------------------------------------------------------
    def spawn(self, expert_id: int, rank: int = 8,
              weight_grad: torch.Tensor | None = None) -> list:
        """
        Spawn a new ReLU-gated LoRA sub-adapter inside expert `expert_id`.

        Initialization protocol
        -----------------------
        • A  ← top-r rows of V^H from SVD of grad (residual-gradient init, LoRA-GA style)
                Falls back to small random if grad is None or SVD fails.
        • B  ← zero   (guarantees exact zero output at spawn → no loss spike, Check 2)
        • w  ← N(0, σ²) where σ = 1e-3 · Var(W_k)   (per proposal §3.3)

        Returns list of new nn.Parameters to add to the optimizer.
        """
        dev, dtype = self.device, self.dtype

        lora = HierarchicalExpert(
            in_features=self.lora_dim,
            out_features=self.lora_dim,
            base_rank=rank,
            lora_alpha=2 * rank,
        ).to(dev).to(dtype)

        # Gradient-informed A init, B always stays zero
        if weight_grad is not None:
            try:
                # weight_grad: [lora_dim, lora_dim]
                U, S, Vh = torch.linalg.svd(weight_grad.float(), full_matrices=False)
                with torch.no_grad():
                    lora.A.copy_(Vh[:rank].to(dtype))   # top-r rows of V^H
                    # lora.B stays zero → output = 0 at spawn (Check 2 guaranteed)
            except Exception as e:
                print(f"  [spawn] SVD failed ({e}), using default random A init.")

        # Symmetry-breaking gate: σ = c · Var(W_k)  (proposal eq. 3.3)
        # FIX: original used sqrt(var) inconsistently — using var() matches proposal
        sigma = 1e-3 * self.original.down_proj[expert_id].float().var().item()
        gate = nn.Parameter(
            torch.randn(self.lora_dim, device=dev, dtype=dtype) * sigma
        )

        self.spawn_loras[expert_id].append(lora)
        self.spawn_gates[expert_id].append(gate)

        # Return ALL new leaf parameters so the caller can add them to the optimizer
        return list(lora.parameters()) + [gate]

    # ------------------------------------------------------------------
    def forward(self,
                hidden_states: torch.Tensor,   # [T, model_dim]
                top_k_indices: torch.Tensor,   # [T, top_k]
                top_k_weights: torch.Tensor,   # [T, top_k]
                ) -> torch.Tensor:

        orig_out   = self.original.forward(hidden_states, top_k_indices, top_k_weights)
        # FIX: zeros_like inherits device/dtype automatically — avoids multi-GPU mismatch
        correction = torch.zeros_like(orig_out)

        for k in range(self.num_experts):
            mask = (top_k_indices == k)      # [T, top_k]  bool
            if not mask.any():
                continue

            # Effective routing weight for expert k per token (sum over top_k slots)
            # top_k_indices: [T, top_k], top_k_weights: [T, top_k]
            # Find which top_k slot(s) selected expert k, sum their weights
            eff_w = top_k_weights[:, k]

            # Base LoRA correction
            base_out   = self.base_loras[k](hidden_states)              # [T, lora_dim]
            correction = correction + base_out * eff_w.unsqueeze(-1)

            # Spawned sub-adapter corrections
            for gate_vec, sub_lora in zip(self.spawn_gates[k], self.spawn_loras[k]):
                # ReLU scalar gate per token (proposal §3.1)
                g = F.relu(hidden_states @ gate_vec)                     # [T]
                sub_out    = sub_lora(hidden_states)                     # [T, lora_dim]
                correction = correction + sub_out * (g * eff_w).unsqueeze(-1)

        # Last line of forward(), replacing the current return:
        return (orig_out + correction).to(hidden_states.dtype)
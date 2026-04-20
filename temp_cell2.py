import torch
import torch.nn as nn

class HierarchicalExpert(nn.Module):
    """
    Low-rank LoRA sub-adapter.  Computes: L(x) = (x A^T B^T) * scaling
    
    Key fix: B is zero-initialized by default. This guarantees that at the
    moment of spawn, the sub-adapter's contribution is exactly zero, which
    is what makes Check 2 (zero-loss-spike) provably pass.
    """
    def __init__(self, in_features, out_features, base_rank=16, lora_alpha=32):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank    = base_rank
        self.scaling = lora_alpha / base_rank

        # Standard LoRA init: A ~ N(0, 0.01), B = 0
        # B=0 means output is zero at init → no loss spike on insertion
        self.A = nn.Parameter(torch.randn(base_rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, base_rank))   # FIX: was randn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast A and B to match x's dtype at runtime.
        # This handles bf16/fp16/fp32 transparently regardless of how
        # the parameters were initialized.
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        return (x @ A.t() @ B.t()) * self.scaling
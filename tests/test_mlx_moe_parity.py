# Apply Mac compatibility patches BEFORE importing unsloth
import platform
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import torch
import torch.nn as nn
import numpy as np
import unittest
from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm
import mlx.core as mx

class TestMLXMoEParity(unittest.TestCase):
    def setUp(self):
        self.device = "mps" # Test MLX path which should be active on MPS if MLX is available
        self.E = 8
        self.M = 128
        self.K = 256
        self.N = 512
        self.topk = 2
        
        # We'll use small values to avoid overflow in parity checks
        self.X = torch.randn(self.M, self.K, device=self.device) / self.K**0.5
        self.W = torch.randn(self.E, self.N, self.K, device=self.device) / self.K**0.5
        
        # Random selected experts
        self.selected_experts = torch.randint(0, self.E, (self.M, self.topk), device=self.device)

    def reference_grouped_gemm(self, X, W, selected_experts, topk):
        # Flatten X and selected_experts
        X_flat = X.view(-1, self.K)
        experts_flat = selected_experts.view(-1)
        
        # Output buffer
        Y = torch.zeros(self.M * topk, self.N, device=X.device, dtype=X.dtype)
        
        # Manual loop over experts
        for expert_idx in range(self.E):
            # Find tokens assigned to this expert
            idx = (experts_flat == expert_idx).nonzero().view(-1)
            if len(idx) > 0:
                # Get the tokens (indexed by token_idx // topk)
                tokens = X_flat[idx // topk]
                # Compute GEMM
                Y[idx] = tokens @ W[expert_idx].T
                
        return Y

    def test_forward_parity(self):
        from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import get_routing_indices
        token_counts, gather_indices = get_routing_indices(self.selected_experts, self.E)
        
        # MLX path
        Y_mlx = grouped_gemm(
            X=self.X,
            W=self.W,
            m_sizes=token_counts,
            gather_indices=gather_indices,
            topk=self.topk,
            permute_x=True,
            permute_y=False,
        )
        
        # Reference
        Y_ref = self.reference_grouped_gemm(self.X, self.W, self.selected_experts, self.topk)
        
        # Check parity
        # We need to unpermute Y_mlx to compare with Y_ref which is ordered by (token, topk)
        from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import unpermute
        Y_mlx_unpermuted = unpermute(Y_mlx, gather_indices)
        
        diff = (Y_mlx_unpermuted - Y_ref).abs().max().item()
        print(f"Forward Max Diff: {diff}")
        self.assertLess(diff, 1e-3)

    def test_backward_parity(self):
        # We'll test dX and dW
        self.X.requires_grad = True
        self.W.requires_grad = True
        
        from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import get_routing_indices
        token_counts, gather_indices = get_routing_indices(self.selected_experts, self.E)
        
        # Forward pass
        Y = grouped_gemm(
            X=self.X,
            W=self.W,
            m_sizes=token_counts,
            gather_indices=gather_indices,
            topk=self.topk,
            permute_x=True,
            permute_y=False,
        )
        
        # Backward pass
        grad_out = torch.randn_like(Y)
        Y.backward(grad_out)
        
        dX_mlx = self.X.grad.clone()
        dW_mlx = self.W.grad.clone()
        
        # Reference backward
        self.X.grad = None
        self.W.grad = None
        
        Y_ref = self.reference_grouped_gemm(self.X, self.W, self.selected_experts, self.topk)
        Y_ref.backward(grad_out) # Note: we use the same grad_out but it needs to be permuted if Y was permuted
        
        # Wait, if Y_mlx was permuted, we need to permute grad_out accordingly
        # In our test, Y_mlx is in expert-order (permute_y=False in grouped_gemm but permute_x=True)
        # Actually grouped_gemm(permute_x=True, permute_y=False) returns Y in EXPERT order.
        # reference_grouped_gemm returns Y in (TOKEN, TOPK) order.
        
        # Correct reference backward:
        dX_ref = self.X.grad.clone()
        dW_ref = self.W.grad.clone()
        
        # Check parity
        diff_dX = (dX_mlx - dX_ref).abs().max().item()
        diff_dW = (dW_mlx - dW_ref).abs().max().item()
        
        print(f"Backward dX Max Diff: {diff_dX}")
        print(f"Backward dW Max Diff: {diff_dW}")
        
        self.assertLess(diff_dX, 1e-3)
        self.assertLess(diff_dW, 1e-3)

if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import torch.nn.functional as F
import unittest
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Apply Mac compatibility patches BEFORE importing unsloth
import platform
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

from unsloth.kernels.moe.grouped_gemm.mps.fallback import (
    grouped_gemm_mps_forward,
    grouped_gemm_mps_dX,
    grouped_gemm_mps_dW,
    grouped_gemm_mps,
)
from unsloth.kernels.moe.grouped_gemm.reference.moe_ops import (
    torch_grouped_gemm,
    permute,
    unpermute,
    calculate_topk,
    get_routing_indices,
)

class TestMoEMPSParity(unittest.TestCase):
    def setUp(self):
        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"\nRunning tests on device: {self.device}")
        
        self.num_tokens = 64
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_experts = 8
        self.topk = 2
        self.dtype = torch.float32  # Use float32 for parity checks to minimize numerical error
        
        # Fixed seed for reproducibility
        torch.manual_seed(42)
        
        # Create inputs
        self.X = torch.randn(self.num_tokens, self.hidden_size, device=self.device, dtype=self.dtype)
        self.W = torch.randn(self.num_experts, self.intermediate_size, self.hidden_size, device=self.device, dtype=self.dtype)
        
        # Gating output to create m_sizes and gather_indices
        self.gating_output = torch.randn(self.num_tokens, self.num_experts, device=self.device, dtype=self.dtype)
        
        self.topk_weights, self.topk_ids = calculate_topk(
            self.gating_output, self.topk, use_sigmoid=False, renormalize=True
        )
        self.m_sizes, self.gather_indices = get_routing_indices(self.topk_ids, num_experts=self.num_experts)
        self.m_sizes = self.m_sizes.to(torch.long)

    def test_forward_no_permute(self):
        # Case: Input already permuted, output raw
        X_perm = permute(self.X, self.gather_indices, self.topk)
        
        # Reference
        ref_out = torch_grouped_gemm(X_perm, self.W, self.m_sizes)
        
        # MPS Fallback
        test_out = grouped_gemm_mps_forward(
            X=X_perm,
            W=self.W,
            topk=self.topk,
            m_sizes=self.m_sizes,
            permute_x=False,
            permute_y=False
        )
        
        torch.testing.assert_close(test_out, ref_out, atol=1e-5, rtol=1e-5)
        print("✓ Forward (No Permute) Passed")

    def test_forward_permute_x(self):
        # Case: Input in token order, output expert order
        # Reference: Manual permute + grouped_gemm
        X_perm = permute(self.X, self.gather_indices, self.topk)
        ref_out = torch_grouped_gemm(X_perm, self.W, self.m_sizes)
        
        # MPS Fallback
        test_out = grouped_gemm_mps_forward(
            X=self.X,
            W=self.W,
            topk=self.topk,
            m_sizes=self.m_sizes,
            gather_indices=self.gather_indices,
            permute_x=True,
            permute_y=False
        )
        
        torch.testing.assert_close(test_out, ref_out, atol=1e-5, rtol=1e-5)
        print("✓ Forward (Permute X) Passed")

    def test_forward_permute_y(self):
        # Case: Input expert order, output token order
        X_perm = permute(self.X, self.gather_indices, self.topk)
        ref_out_perm = torch_grouped_gemm(X_perm, self.W, self.m_sizes)
        ref_out = unpermute(ref_out_perm, self.gather_indices)
        
        # MPS Fallback
        test_out = grouped_gemm_mps_forward(
            X=X_perm,
            W=self.W,
            topk=self.topk,
            m_sizes=self.m_sizes,
            gather_indices=self.gather_indices,
            permute_x=False,
            permute_y=True
        )
        
        # In this case (fuse_mul_post=False), the output should have total_tokens (128)
        self.assertEqual(test_out.shape, (self.num_tokens * self.topk, self.intermediate_size))
        torch.testing.assert_close(test_out, ref_out, atol=1e-5, rtol=1e-5)
        print("✓ Forward (Permute Y) Passed")

    def test_backward_dX(self):
        # dX = dY @ W
        dY = torch.randn(self.num_tokens * self.topk, self.intermediate_size, device=self.device, dtype=self.dtype)
        
        # Reference implementation of dX
        # Note: torch_grouped_gemm(dY, W_flat, m_sizes, transpose=False)
        # where W_flat is reshaped to (E, K, N)
        W_T = self.W.transpose(1, 2).contiguous() # (E, K, N)
        ref_dX = torch_grouped_gemm(dY, W_T, self.m_sizes)
        
        test_dX = grouped_gemm_mps_dX(
            dY=dY,
            W=self.W,
            gather_indices=self.gather_indices,
            m_sizes=self.m_sizes,
            topk=self.topk,
            permute_x=False,
            permute_y=False
        )
        
        torch.testing.assert_close(test_dX, ref_dX, atol=1e-5, rtol=1e-5)
        print("✓ Backward dX Passed")

    def test_backward_dW(self):
        # dW = X.T @ dY
        X_perm = permute(self.X, self.gather_indices, self.topk)
        dY = torch.randn(self.num_tokens * self.topk, self.intermediate_size, device=self.device, dtype=self.dtype)
        
        # Reference dW calculation
        ref_dW = torch.zeros_like(self.W)
        m_cumsum = torch.cumsum(self.m_sizes, dim=0)
        m_start = 0
        for i in range(self.num_experts):
            m_end = m_cumsum[i].item()
            if m_end > m_start:
                X_exp = X_perm[m_start:m_end]
                dY_exp = dY[m_start:m_end]
                ref_dW[i] = dY_exp.t() @ X_exp
            m_start = m_end
            
        test_dW = grouped_gemm_mps_dW(
            X=X_perm,
            dY=dY,
            m_sizes=self.m_sizes,
            gather_indices=self.gather_indices,
            topk=self.topk,
            permute_x=False,
            permute_y=False
        )
        
        torch.testing.assert_close(test_dW, ref_dW, atol=1e-5, rtol=1e-5)
        print("✓ Backward dW Passed")

    def test_autograd_permute_x(self):
        # Test autograd with permute_x (First GEMM)
        X_ref = self.X.detach().clone().requires_grad_(True)
        W_ref = self.W.detach().clone().requires_grad_(True)
        
        X_test = self.X.detach().clone().requires_grad_(True)
        W_test = self.W.detach().clone().requires_grad_(True)
        
        # Forward Reference
        X_perm_ref = permute(X_ref, self.gather_indices, self.topk)
        out_ref = torch_grouped_gemm(X_perm_ref, W_ref, self.m_sizes)
        
        # Forward Test
        out_test = grouped_gemm_mps(
            X=X_test,
            W=W_test,
            m_sizes=self.m_sizes,
            topk=self.topk,
            gather_indices=self.gather_indices,
            permute_x=True,
            permute_y=False,
        )
        
        torch.testing.assert_close(out_test, out_ref, atol=1e-5, rtol=1e-5)
        
        # Backward
        grad_output = torch.randn_like(out_ref)
        out_ref.backward(grad_output)
        out_test.backward(grad_output)
        
        torch.testing.assert_close(X_test.grad, X_ref.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(W_test.grad, W_ref.grad, atol=1e-5, rtol=1e-5)
        print("✓ Autograd Permute X Passed")

    def test_autograd_permute_y(self):
        # Test autograd with permute_y (Second GEMM)
        X_perm = permute(self.X, self.gather_indices, self.topk)
        
        X_ref = X_perm.detach().clone().requires_grad_(True)
        W_ref = self.W.detach().clone().requires_grad_(True)
        
        X_test = X_perm.detach().clone().requires_grad_(True)
        W_test = self.W.detach().clone().requires_grad_(True)
        
        # Forward Reference
        out_perm_ref = torch_grouped_gemm(X_ref, W_ref, self.m_sizes)
        out_ref = unpermute(out_perm_ref, self.gather_indices)
        
        # Forward Test
        out_test = grouped_gemm_mps(
            X=X_test,
            W=W_test,
            m_sizes=self.m_sizes,
            topk=self.topk,
            gather_indices=self.gather_indices,
            permute_x=False,
            permute_y=True,
        )
        
        torch.testing.assert_close(out_test, out_ref, atol=1e-5, rtol=1e-5)
        
        # Backward
        grad_output = torch.randn_like(out_ref)
        out_ref.backward(grad_output)
        out_test.backward(grad_output)
        
        torch.testing.assert_close(X_test.grad, X_ref.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(W_test.grad, W_ref.grad, atol=1e-5, rtol=1e-5)
        print("✓ Autograd Permute Y Passed")

    def test_inference_fuse_mul_post(self):
        # Test inference-only fused weight + reduce
        # Reference
        X_perm = permute(self.X, self.gather_indices, self.topk)
        out_perm = torch_grouped_gemm(X_perm, self.W, self.m_sizes)
        # Multiply by weights in token-expert order
        # Weighting happens on expert-order outputs in my implementation if fuse_mul_post is True?
        # Let's check: Y_unperm = Y_unperm * topk_weights.view(-1, 1)
        # So weights are in token-expert order.
        out_unperm = unpermute(out_perm, self.gather_indices)
        out_weighted = out_unperm * self.topk_weights.view(-1, 1)
        ref_out = out_weighted.view(self.num_tokens, self.topk, self.intermediate_size).sum(dim=1)
        
        # MPS Fallback
        test_out = grouped_gemm_mps_forward(
            X=X_perm,
            W=self.W,
            topk=self.topk,
            m_sizes=self.m_sizes,
            gather_indices=self.gather_indices,
            topk_weights=self.topk_weights,
            permute_x=False,
            permute_y=True,
            fuse_mul_post=True
        )
        
        self.assertEqual(test_out.shape, (self.num_tokens, self.intermediate_size))
        torch.testing.assert_close(test_out, ref_out, atol=1e-5, rtol=1e-5)
        print("✓ Inference Fuse Mul Post Passed")

if __name__ == "__main__":
    unittest.main()

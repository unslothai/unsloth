"""
MoE Grouped GEMM Parity Test for MPS.
Verifies that the MPS (MLX/fallback) implementation produces correct results.
"""
import sys
import os
import torch
import platform
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMoEMPS(unittest.TestCase):
    def setUp(self):
        if platform.system() != "Darwin":
            self.skipTest("Not running on macOS")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")
            
    def test_grouped_gemm_forward_parity(self):
        """Test that MPS grouped GEMM forward matches a reference implementation."""
        print("\n" + "=" * 60)
        print("Running MoE Grouped GEMM Forward Parity Test")
        print("=" * 60)
        
        try:
            from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm_forward
        except ImportError as e:
            self.skipTest(f"Could not import grouped_gemm: {e}")
        
        # Setup: 2 experts, 4 tokens, topk=1 (simple case)
        num_experts = 2
        num_tokens = 4
        topk = 1
        K = 16  # input dim
        N = 32  # output dim per expert
        
        device = torch.device("mps")
        dtype = torch.float16
        
        # Weights: (E, N, K)
        W = torch.randn(num_experts * N, K, device=device, dtype=dtype) / (K**0.5)
        
        # Input X: (total_tokens, K), total_tokens = num_tokens * topk
        total_tokens = num_tokens * topk
        X = torch.randn(total_tokens, K, device=device, dtype=dtype)
        
        # m_sizes: tokens per expert. Simple: 2 tokens each.
        m_sizes = torch.tensor([2, 2], device=device, dtype=torch.int32)
        
        # gather_indices: maps each sorted position to original token index
        # Tokens 0, 1 -> Expert 0; Tokens 2, 3 -> Expert 1
        gather_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        
        # Run grouped_gemm_forward
        print("Running grouped_gemm_forward on MPS...")
        Y = grouped_gemm_forward(
            X=X,
            W=W,
            topk=topk,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            permute_x=False,
            permute_y=False,
        )
        print(f"Output Y shape: {Y.shape}")
        self.assertEqual(Y.shape, (total_tokens, N))
        
        # Reference (manual loop)
        # Expert 0: X[0:2] @ W[0:N].T
        # Expert 1: X[2:4] @ W[N:2N].T
        W_reshaped = W.view(num_experts, N, K)
        Y_ref = torch.zeros_like(Y)
        m_cum = 0
        for i in range(num_experts):
            m_end = m_cum + m_sizes[i].item()
            X_exp = X[m_cum:m_end]
            W_exp = W_reshaped[i]  # (N, K)
            Y_ref[m_cum:m_end] = X_exp @ W_exp.T
            m_cum = m_end
            
        # Compare
        error = (Y - Y_ref).abs().max().item()
        print(f"Max error vs reference: {error}")
        self.assertLess(error, 1e-2, f"Error too high: {error}")
        print("PASS!")

if __name__ == "__main__":
    unittest.main()

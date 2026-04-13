"""Test numerical stability with extreme log-probability values."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

# Very small logps (large negative values)
new1 = torch.randn(B, S, requires_grad=True) - 20.0
ref1 = torch.randn(B, S) - 20.0
old1 = torch.randn(B, S) - 20.0
r1 = grpo_compute_loss(ref1, new1, old1, None, input_ids, mask, beta, advantages)
assert torch.isfinite(r1[0]), f"Small logps: loss not finite: {r1[0]}"
print(f"PASS: small logps, loss={r1[0].item():.4f}")

# Logps near zero
new2 = torch.randn(B, S, requires_grad=True) * 0.01
ref2 = torch.randn(B, S) * 0.01
old2 = torch.randn(B, S) * 0.01
r2 = grpo_compute_loss(ref2, new2, old2, None, input_ids, mask, beta, advantages)
assert torch.isfinite(r2[0]), f"Near-zero logps: loss not finite: {r2[0]}"
print(f"PASS: near-zero logps, loss={r2[0].item():.4f}")

# Identical ref and new (KL should be ~0)
shared = torch.randn(B, S)
new3 = shared.clone().requires_grad_(True)
r3 = grpo_compute_loss(shared, new3, shared.clone(), None, input_ids, mask, beta, advantages)
assert torch.isfinite(r3[0]), f"Identical logps: loss not finite"
assert abs(r3[2].item()) < 0.01, f"Expected ~0 KL for identical ref/new, got {r3[2].item()}"
print(f"PASS: identical ref/new, mean_kl={r3[2].item():.6f}")

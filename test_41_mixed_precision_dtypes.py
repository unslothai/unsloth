"""Test with float16 and bfloat16 inputs (common in mixed-precision training)."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    ref = torch.randn(B, S, dtype=dtype)
    new = torch.randn(B, S, dtype=dtype, requires_grad=True)
    old = torch.randn(B, S, dtype=dtype)
    result = grpo_compute_loss(ref, new, old, None, input_ids, mask, beta, advantages)
    loss = result[0]
    assert torch.isfinite(loss), f"{dtype}: loss not finite"
    loss.backward()
    assert new.grad is not None, f"{dtype}: no gradient"
    print(f"PASS: dtype={dtype}, loss={loss.item():.4f}")

print("PASS: All dtypes work correctly")

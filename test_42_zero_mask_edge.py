"""Test edge case: single row with all-zero mask (no completion tokens for that sample)."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad = True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
beta = 0.1
advantages = torch.randn(B)

# Mask where one row has all zeros (empty completion)
mask = torch.ones(B, S)
mask[0, :] = 0.0  # first sample has no completion tokens

result = grpo_compute_loss(ref, new, old, None, input_ids, mask, beta, advantages)
loss = result[0]
# Loss should still be finite (div by clamp(min=1.0) prevents NaN)
assert torch.isfinite(loss), f"Loss not finite with partial-zero mask: {loss}"
print(f"PASS: partial-zero mask, loss={loss.item():.4f}")

"""Test with partial completion mask (zeros in mask), not all-ones."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 16
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad = True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
beta = 0.1
advantages = torch.randn(B)

# Mask with varying completion lengths per sample
mask = torch.zeros(B, S)
mask[0, :4] = 1.0
mask[1, :8] = 1.0
mask[2, :12] = 1.0
mask[3, :16] = 1.0

result = grpo_compute_loss(ref, new, old, None, input_ids, mask, beta, advantages)
loss, comp_len, mean_kl, _, _, coef_1, out_mask = result

assert torch.isfinite(loss), f"Loss not finite: {loss}"
assert torch.isfinite(mean_kl), f"mean_kl not finite: {mean_kl}"
# Completion length should be mean of [4, 8, 12, 16] = 10.0
assert (
    abs(comp_len.item() - 10.0) < 0.01
), f"Expected comp_len=10.0, got {comp_len.item()}"

loss.backward()
assert new.grad is not None
print(f"PASS: partial mask, loss={loss.item():.4f}, comp_len={comp_len.item():.1f}")

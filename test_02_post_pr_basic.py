"""Confirm the post-PR call with correct positional args succeeds and returns valid output."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad = True)
old = torch.randn(B, S)
sampling = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

result = grpo_compute_loss(
    ref,
    new,
    old,
    sampling,
    input_ids,
    mask,
    beta,
    advantages,
    loss_type = "grpo",
)
assert len(result) == 7, f"Expected 7-tuple, got {len(result)}"
loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, out_mask = result
assert torch.isfinite(loss), f"Loss is not finite: {loss}"
assert torch.isfinite(mean_kl), f"mean_kl is not finite: {mean_kl}"
print(f"PASS: loss={loss.item():.4f}, mean_kl={mean_kl.item():.4f}")

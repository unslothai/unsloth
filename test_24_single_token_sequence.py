"""Test edge case: single token sequence (S=1) and single sample (B=1)."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

# B=1, S=1: minimal possible input
ref = torch.randn(1, 1)
new = torch.randn(1, 1, requires_grad = True)
old = torch.randn(1, 1)
input_ids = torch.randint(0, 100, (1, 1))
mask = torch.ones(1, 1)
beta = 0.1
advantages = torch.randn(1)

result = grpo_compute_loss(ref, new, old, None, input_ids, mask, beta, advantages)
assert len(result) == 7
loss = result[0]
assert torch.isfinite(loss), f"B=1,S=1: loss not finite: {loss}"
loss.backward()
assert new.grad is not None
print(f"PASS: B=1,S=1, loss={loss.item():.4f}")

# Test with sequence-level IS on single token
new2 = torch.randn(1, 1, requires_grad = True)
r2 = grpo_compute_loss(
    ref,
    new2,
    old,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    importance_sampling_level = "sequence",
)
assert torch.isfinite(r2[0]), f"Sequence IS with S=1: loss not finite"
print(f"PASS: sequence IS with S=1, loss={r2[0].item():.4f}")

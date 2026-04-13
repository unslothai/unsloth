"""Test with all-negative and all-positive advantages (clipping edge cases)."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
ref = torch.randn(B, S)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1

# All-negative advantages
new1 = torch.randn(B, S, requires_grad=True)
adv_neg = -torch.abs(torch.randn(B)) - 0.1  # guarantee negative
r1 = grpo_compute_loss(ref, new1, old, None, input_ids, mask, beta, adv_neg)
assert torch.isfinite(r1[0]), f"All-negative adv: loss not finite"

# All-positive advantages
new2 = torch.randn(B, S, requires_grad=True)
adv_pos = torch.abs(torch.randn(B)) + 0.1  # guarantee positive
r2 = grpo_compute_loss(ref, new2, old, None, input_ids, mask, beta, adv_pos)
assert torch.isfinite(r2[0]), f"All-positive adv: loss not finite"

# Zero advantages
new3 = torch.randn(B, S, requires_grad=True)
adv_zero = torch.zeros(B)
r3 = grpo_compute_loss(ref, new3, old, None, input_ids, mask, beta, adv_zero)
assert torch.isfinite(r3[0]), f"Zero adv: loss not finite"

print(f"PASS: neg_adv loss={r1[0].item():.4f}, pos_adv loss={r2[0].item():.4f}, zero_adv loss={r3[0].item():.4f}")

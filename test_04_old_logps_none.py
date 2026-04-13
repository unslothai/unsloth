"""Test old_logps=None (dapo path where old is not available)."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad = True)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

# old=None triggers log_ratio = new - new.detach()
result = grpo_compute_loss(
    ref,
    new,
    None,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    loss_type = "dapo",
    num_items_in_batch = B,
    num_processes = 1,
)
assert len(result) == 7
loss = result[0]
assert torch.isfinite(loss), f"Loss not finite: {loss}"
print(f"PASS: old=None (dapo), loss={loss.item():.4f}")

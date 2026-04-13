"""Test all supported loss types produce finite results."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad = True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

loss_types = {
    "grpo": {},
    "bnpo": {},
    "dr_grpo": {"max_completion_length": S},
    "dapo": {"num_items_in_batch": B, "num_processes": 1},
    "cispo": {"num_items_in_batch": B, "num_processes": 1},
}

for lt, extra_kwargs in loss_types.items():
    new_t = torch.randn(B, S, requires_grad = True)
    result = grpo_compute_loss(
        ref,
        new_t,
        old,
        None,
        input_ids,
        mask,
        beta,
        advantages,
        loss_type = lt,
        **extra_kwargs,
    )
    assert len(result) == 7, f"{lt}: expected 7-tuple"
    assert torch.isfinite(result[0]), f"{lt}: loss not finite: {result[0]}"
    print(f"PASS: loss_type={lt}, loss={result[0].item():.4f}")

print("PASS: All loss types OK")

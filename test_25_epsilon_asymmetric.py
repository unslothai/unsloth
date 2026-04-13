"""Test asymmetric epsilon_low != epsilon_high clipping."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
ref = torch.randn(B, S)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

configs = [
    ("tight_low", 0.05, 0.3),
    ("tight_high", 0.3, 0.05),
    ("symmetric", 0.2, 0.2),
    ("zero_low", 0.0, 0.2),
    ("zero_high", 0.2, 0.0),
]

for label, eps_low, eps_high in configs:
    new = torch.randn(B, S, requires_grad = True)
    r = grpo_compute_loss(
        ref,
        new,
        old,
        None,
        input_ids,
        mask,
        beta,
        advantages,
        loss_type = "grpo",
        epsilon_low = eps_low,
        epsilon_high = eps_high,
    )
    assert torch.isfinite(r[0]), f"{label}: loss not finite"
    print(
        f"PASS: {label} (eps_low={eps_low}, eps_high={eps_high}), loss={r[0].item():.4f}"
    )

print("PASS: All asymmetric epsilon configs OK")

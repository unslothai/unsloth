"""Test token vs sequence importance sampling levels."""

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

for level in ["token", "sequence"]:
    new = torch.randn(B, S, requires_grad = True)
    result = grpo_compute_loss(
        ref,
        new,
        old,
        None,
        input_ids,
        mask,
        beta,
        advantages,
        loss_type = "grpo",
        importance_sampling_level = level,
    )
    assert len(result) == 7
    assert torch.isfinite(result[0]), f"{level}: loss not finite"
    print(f"PASS: importance_sampling_level={level}, loss={result[0].item():.4f}")

# Invalid level should raise
new = torch.randn(B, S, requires_grad = True)
try:
    grpo_compute_loss(
        ref,
        new,
        old,
        None,
        input_ids,
        mask,
        beta,
        advantages,
        importance_sampling_level = "invalid",
    )
    print("FAIL: Expected ValueError for invalid importance_sampling_level")
    sys.exit(1)
except ValueError:
    print("PASS: Invalid importance_sampling_level raises ValueError")

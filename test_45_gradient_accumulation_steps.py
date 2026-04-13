"""Test current_gradient_accumulation_steps scaling for GRPO/BNPO loss types."""

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
new_base = torch.randn(B, S)

# GRPO/BNPO divide loss by current_gradient_accumulation_steps
for gas in [1, 2, 4]:
    new = new_base.clone().requires_grad_(True)
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
        current_gradient_accumulation_steps = gas,
    )
    assert torch.isfinite(result[0]), f"gas={gas}: loss not finite"
    print(f"PASS: gradient_accumulation_steps={gas}, loss={result[0].item():.6f}")

# Verify: loss with gas=2 should be half of gas=1
new1 = new_base.clone().requires_grad_(True)
r1 = grpo_compute_loss(
    ref,
    new1,
    old,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    current_gradient_accumulation_steps = 1,
)
new2 = new_base.clone().requires_grad_(True)
r2 = grpo_compute_loss(
    ref,
    new2,
    old,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    current_gradient_accumulation_steps = 2,
)
ratio = r1[0].item() / r2[0].item() if r2[0].item() != 0 else float("inf")
assert abs(ratio - 2.0) < 0.01, f"Expected loss ratio of 2.0, got {ratio:.4f}"
print(f"PASS: gas=1 loss / gas=2 loss = {ratio:.4f} (expected 2.0)")

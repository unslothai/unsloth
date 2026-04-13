"""Test KL divergence properties: KL >= 0, KL=0 when ref==new."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 16
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
advantages = torch.randn(B)

# When ref == new, KL should be exactly 0 (reverse KL: exp(ref-new) - (ref-new) - 1 = 0)
shared = torch.randn(B, S)
new = shared.clone().requires_grad_(True)
result = grpo_compute_loss(
    shared.clone(),
    new,
    shared.clone(),
    None,
    input_ids,
    mask,
    0.1,
    advantages,
)
mean_kl = result[2].item()
assert abs(mean_kl) < 1e-5, f"Expected KL~0 when ref==new, got {mean_kl}"
print(f"PASS: KL={mean_kl:.8f} when ref==new (expected ~0)")

# KL should be non-negative (reverse KL: exp(x) - x - 1 >= 0 for all x)
ref = torch.randn(B, S)
new2 = torch.randn(B, S, requires_grad = True)
result2 = grpo_compute_loss(
    ref,
    new2,
    torch.randn(B, S),
    None,
    input_ids,
    mask,
    0.1,
    advantages,
)
mean_kl2 = result2[2].item()
assert mean_kl2 >= -1e-6, f"KL should be non-negative, got {mean_kl2}"
print(f"PASS: KL={mean_kl2:.6f} >= 0 (reverse KL non-negativity)")

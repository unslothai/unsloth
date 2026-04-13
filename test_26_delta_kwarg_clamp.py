"""Test delta kwarg which clamps coef_1 max in GRPO loss."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
ref = torch.randn(B, S)
# Make new very different from old to produce large coef_1
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

# With delta=None (no clamp on coef_1)
new1 = torch.randn(B, S, requires_grad = True)
r_none = grpo_compute_loss(
    ref,
    new1,
    old,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    loss_type = "grpo",
    delta = None,
)
assert torch.isfinite(r_none[0])

# With delta=1.5 (clamp coef_1 at 1.5)
new2 = new1.detach().clone().requires_grad_(True)
r_delta = grpo_compute_loss(
    ref, new2, old, None, input_ids, mask, beta, advantages, loss_type = "grpo", delta = 1.5
)
assert torch.isfinite(r_delta[0])

# With delta=1.0 (tight clamp)
new3 = new1.detach().clone().requires_grad_(True)
r_tight = grpo_compute_loss(
    ref, new3, old, None, input_ids, mask, beta, advantages, loss_type = "grpo", delta = 1.0
)
assert torch.isfinite(r_tight[0])

print(
    f"PASS: delta=None loss={r_none[0].item():.4f}, delta=1.5 loss={r_delta[0].item():.4f}, delta=1.0 loss={r_tight[0].item():.4f}"
)

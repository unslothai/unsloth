"""Verify BNPO and GRPO produce different losses due to different normalization."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

torch.manual_seed(99)
B, S = 4, 8
ref = torch.randn(B, S)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)
new_base = torch.randn(B, S)

# GRPO: mean over samples of (per-sample mean)
new1 = new_base.clone().requires_grad_(True)
r_grpo = grpo_compute_loss(
    ref, new1, old, None, input_ids, mask, beta, advantages, loss_type = "grpo"
)

# BNPO: sum / total_mask_count (global mean, not per-sample then mean)
new2 = new_base.clone().requires_grad_(True)
r_bnpo = grpo_compute_loss(
    ref, new2, old, None, input_ids, mask, beta, advantages, loss_type = "bnpo"
)

assert torch.isfinite(r_grpo[0]) and torch.isfinite(r_bnpo[0])

# With uniform mask they should actually be the same, test with non-uniform
mask2 = torch.zeros(B, S)
mask2[0, :2] = 1.0
mask2[1, :4] = 1.0
mask2[2, :6] = 1.0
mask2[3, :8] = 1.0

new3 = new_base.clone().requires_grad_(True)
r_grpo2 = grpo_compute_loss(
    ref, new3, old, None, input_ids, mask2, beta, advantages, loss_type = "grpo"
)

new4 = new_base.clone().requires_grad_(True)
r_bnpo2 = grpo_compute_loss(
    ref, new4, old, None, input_ids, mask2, beta, advantages, loss_type = "bnpo"
)

assert torch.isfinite(r_grpo2[0]) and torch.isfinite(r_bnpo2[0])
# With non-uniform mask, GRPO and BNPO should differ
assert (
    r_grpo2[0].item() != r_bnpo2[0].item()
), "GRPO and BNPO should differ with non-uniform mask"
print(
    f"PASS: grpo={r_grpo2[0].item():.4f}, bnpo={r_bnpo2[0].item():.4f} (differ with non-uniform mask)"
)

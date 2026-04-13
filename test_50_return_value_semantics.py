"""Verify return value semantics: completion_length, mask casting, delta/flat_is_ratio emptiness."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 16
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad=True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
beta = 0.1
advantages = torch.randn(B)

# Mask with known completion lengths
mask = torch.zeros(B, S)
mask[0, :4] = 1
mask[1, :8] = 1
mask[2, :12] = 1
mask[3, :16] = 1

loss, comp_len, mean_kl, delta, flat_is_ratio, coef_1, out_mask = grpo_compute_loss(
    ref, new, old, None, input_ids, mask, beta, advantages,
)

# completion_length should be mean of [4, 8, 12, 16] = 10.0
assert abs(comp_len.item() - 10.0) < 0.01, f"comp_len={comp_len.item()}, expected 10.0"
print(f"PASS: completion_length={comp_len.item():.1f} (expected 10.0)")

# Without use_vllm, delta and flat_is_ratio should be empty tensors
assert delta.numel() == 0, f"delta should be empty without vllm, got {delta.shape}"
assert flat_is_ratio.numel() == 0, f"flat_is_ratio should be empty without vllm, got {flat_is_ratio.shape}"
print(f"PASS: delta and flat_is_ratio empty without vllm")

# Output mask should be float32 (cast inside the function)
assert out_mask.dtype == torch.float32, f"mask should be float32, got {out_mask.dtype}"
print(f"PASS: output mask dtype={out_mask.dtype}")

# coef_1 should be positive (exp of something)
assert (coef_1 > 0).all(), "coef_1 should be all positive (exp values)"
print(f"PASS: coef_1 all positive, range [{coef_1.min().item():.4f}, {coef_1.max().item():.4f}]")

# mean_kl should be non-negative (reverse KL property)
assert mean_kl.item() >= -1e-6, f"mean_kl should be non-negative, got {mean_kl.item()}"
print(f"PASS: mean_kl={mean_kl.item():.6f} >= 0")

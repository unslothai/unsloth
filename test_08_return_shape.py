"""Verify return tuple shape, types, and gradient flow."""
import sys, torch
sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad=True)
old = torch.randn(B, S)
sampling = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, out_mask = grpo_compute_loss(
    ref, new, old, sampling, input_ids, mask, beta, advantages,
)

# Check types
assert loss.dim() == 0, f"loss should be scalar, got dim={loss.dim()}"
assert completion_length.dim() == 0, f"completion_length should be scalar"
assert mean_kl.dim() == 0, f"mean_kl should be scalar"
assert out_mask.shape == (B, S), f"mask shape mismatch: {out_mask.shape}"
assert coef_1.shape == (B, S), f"coef_1 shape mismatch: {coef_1.shape}"

# Check gradient flows through loss
loss.backward()
assert new.grad is not None, "Gradient did not flow to new logps"
assert torch.isfinite(new.grad).all(), "Gradients contain non-finite values"
print(f"PASS: Return shapes correct, gradients flow properly")

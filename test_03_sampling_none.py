"""Test sampling_per_token_logps=None (most common production path)."""
import sys, torch
sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad=True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

result = grpo_compute_loss(
    ref, new, old, None, input_ids, mask, beta, advantages,
    loss_type="grpo",
)
assert len(result) == 7
loss = result[0]
assert torch.isfinite(loss), f"Loss not finite: {loss}"
print(f"PASS: sampling=None, loss={loss.item():.4f}")

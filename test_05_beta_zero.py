"""Test beta=0 path (no KL divergence term)."""
import sys, torch
sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad=True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
advantages = torch.randn(B)

result = grpo_compute_loss(
    ref, new, old, None, input_ids, mask, 0.0, advantages,
    loss_type="grpo",
)
assert len(result) == 7
loss, completion_length, mean_kl = result[0], result[1], result[2]
assert torch.isfinite(loss), f"Loss not finite: {loss}"
assert mean_kl.item() == 0.0, f"Expected mean_kl=0 with beta=0, got {mean_kl}"
print(f"PASS: beta=0, loss={loss.item():.4f}, mean_kl={mean_kl.item():.4f}")

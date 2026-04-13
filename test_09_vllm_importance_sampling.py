"""Test use_vllm=True with sampling_per_token_logps (importance sampling correction)."""
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

# With vllm and sampling logps
result = grpo_compute_loss(
    ref, new, old, sampling, input_ids, mask, beta, advantages,
    use_vllm=True, vllm_importance_sampling_cap=2.0,
)
assert len(result) == 7
loss, _, _, delta, flat_is_ratio, _, _ = result
assert torch.isfinite(loss), f"Loss not finite: {loss}"
assert delta.numel() > 0, "delta should be non-empty with use_vllm=True"
assert flat_is_ratio.numel() > 0, "flat_is_ratio should be non-empty with use_vllm=True"
print(f"PASS: use_vllm=True with sampling, loss={loss.item():.4f}")

# With vllm but sampling=None (should skip IS correction)
new2 = torch.randn(B, S, requires_grad=True)
result2 = grpo_compute_loss(
    ref, new2, old, None, input_ids, mask, beta, advantages,
    use_vllm=True,
)
assert len(result2) == 7
assert torch.isfinite(result2[0]), "Loss not finite with vllm+sampling=None"
assert result2[3].numel() == 0, "delta should be empty when sampling=None"
print(f"PASS: use_vllm=True, sampling=None, loss={result2[0].item():.4f}")

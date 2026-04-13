"""Verify grpo_compute_loss works on CPU without requiring CUDA."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 2, 8

# Explicitly create CPU tensors
device = torch.device("cpu")
ref = torch.randn(B, S, device=device)
new = torch.randn(B, S, device=device, requires_grad=True)
old = torch.randn(B, S, device=device)
sampling = torch.randn(B, S, device=device)
input_ids = torch.randint(0, 100, (B, S), device=device)
mask = torch.ones(B, S, device=device)
beta = 0.1
advantages = torch.randn(B, device=device)

# Test all loss types on CPU
for lt, extra in [("grpo", {}), ("bnpo", {}), ("dapo", {"num_items_in_batch": B, "num_processes": 1})]:
    new_t = torch.randn(B, S, device=device, requires_grad=True)
    result = grpo_compute_loss(
        ref, new_t, old, None, input_ids, mask, beta, advantages,
        loss_type=lt, **extra,
    )
    assert result[0].device.type == "cpu", f"{lt}: result not on CPU"
    assert torch.isfinite(result[0]), f"{lt}: loss not finite on CPU"
    result[0].backward()
    assert new_t.grad is not None, f"{lt}: no gradient on CPU"
    print(f"PASS: {lt} on CPU, loss={result[0].item():.4f}")

# Test with sampling on CPU
new_s = torch.randn(B, S, device=device, requires_grad=True)
result_s = grpo_compute_loss(ref, new_s, old, sampling, input_ids, mask, beta, advantages)
assert torch.isfinite(result_s[0])
print(f"PASS: CPU with sampling_per_token_logps, loss={result_s[0].item():.4f}")

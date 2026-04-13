"""Test DR-GRPO normalization with varying max_completion_length values."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 16
ref = torch.randn(B, S)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)
new_base = torch.randn(B, S)

# DR-GRPO divides by (batch_size * max_completion_length), so different
# max_completion_length should produce different losses
results = {}
for mcl in [S, S * 2, S * 4]:
    new = new_base.clone().requires_grad_(True)
    result = grpo_compute_loss(
        ref, new, old, None, input_ids, mask, beta, advantages,
        loss_type="dr_grpo", max_completion_length=mcl,
    )
    assert torch.isfinite(result[0]), f"mcl={mcl}: loss not finite"
    results[mcl] = result[0].item()
    print(f"PASS: dr_grpo max_completion_length={mcl}, loss={result[0].item():.6f}")

# Larger max_completion_length should produce smaller loss magnitude (larger denominator)
assert abs(results[S]) > abs(results[S * 4]) or abs(results[S]) == 0.0, \
    f"Expected |loss(mcl={S})| > |loss(mcl={S*4})|, got {abs(results[S]):.6f} vs {abs(results[S*4]):.6f}"
print("PASS: DR-GRPO normalization scales correctly with max_completion_length")

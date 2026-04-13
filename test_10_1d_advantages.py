"""Test 1D advantages auto-unsqueeze and various batch/sequence sizes."""
import sys, torch
sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

# Test 1D advantages (should be auto-unsqueezed to 2D)
B, S = 4, 16
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad=True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages_1d = torch.randn(B)

result = grpo_compute_loss(
    ref, new, old, None, input_ids, mask, beta, advantages_1d,
)
assert len(result) == 7
assert torch.isfinite(result[0]), f"1D advantages: loss not finite"
print(f"PASS: 1D advantages, loss={result[0].item():.4f}")

# Test 2D advantages (already correct shape)
new2 = torch.randn(B, S, requires_grad=True)
advantages_2d = torch.randn(B, 1)
result2 = grpo_compute_loss(
    ref, new2, old, None, input_ids, mask, beta, advantages_2d,
)
assert len(result2) == 7
assert torch.isfinite(result2[0]), f"2D advantages: loss not finite"
print(f"PASS: 2D advantages, loss={result2[0].item():.4f}")

# Test various batch/seq sizes
for b, s in [(1, 1), (1, 32), (8, 4)]:
    new_t = torch.randn(b, s, requires_grad=True)
    r = grpo_compute_loss(
        torch.randn(b, s), new_t, torch.randn(b, s), None,
        torch.randint(0, 100, (b, s)), torch.ones(b, s),
        beta, torch.randn(b),
    )
    assert torch.isfinite(r[0]), f"B={b},S={s}: loss not finite"
    print(f"PASS: B={b}, S={s}, loss={r[0].item():.4f}")

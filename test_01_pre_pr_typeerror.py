"""Reproduce the pre-PR TypeError: grpo_compute_loss gets multiple values for sampling_per_token_logps."""
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

# Pre-PR call: sampling_per_token_logps missing from positional args, passed as kwarg
try:
    grpo_compute_loss(
        ref, new, old,
        input_ids,        # WRONG: should be sampling_per_token_logps
        mask,             # WRONG: should be input_ids
        beta,             # WRONG: should be mask
        advantages,       # WRONG: should be beta
        sampling_per_token_logps=sampling,  # duplicate kwarg
    )
    print("FAIL: Expected TypeError was not raised")
    sys.exit(1)
except TypeError as e:
    assert "sampling_per_token_logps" in str(e) or "multiple values" in str(e), f"Wrong TypeError: {e}"
    print(f"PASS: Pre-PR call correctly raises TypeError: {e}")

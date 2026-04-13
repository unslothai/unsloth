"""Verify grpo_compute_loss function signature has sampling_per_token_logps as 4th param."""

import sys, inspect

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

sig = inspect.signature(grpo_compute_loss)
params = list(sig.parameters.keys())

expected_order = [
    "ref",
    "new",
    "old",
    "sampling_per_token_logps",
    "input_ids",
    "mask",
    "beta",
    "advantages",
]

for i, (expected, actual) in enumerate(zip(expected_order, params)):
    assert expected == actual, f"Param {i}: expected '{expected}', got '{actual}'"

assert params[-1] == "kwargs", f"Last param should be **kwargs, got '{params[-1]}'"

# Verify sampling_per_token_logps is POSITIONAL_OR_KEYWORD (not keyword-only)
sp = sig.parameters["sampling_per_token_logps"]
assert (
    sp.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
), f"sampling_per_token_logps should be POSITIONAL_OR_KEYWORD, got {sp.kind}"

print(f"PASS: Function signature correct: {', '.join(params[:8])}, **kwargs")

"""Verify the PR call site passes exactly 8 positional args matching the function signature.
Parse the actual source file to check arg count at the call site."""

import sys, ast

sys.path.insert(0, "unsloth_zoo_repo")

# Read the PR-fixed source
with open("unsloth/models/rl_replacements.py", "r") as f:
    source = f.read()

tree = ast.parse(source)

# Find all calls to grpo_compute_loss_slow
calls_found = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "grpo_compute_loss_slow":
            calls_found.append(node)

assert len(calls_found) >= 1, "Expected at least 1 call to grpo_compute_loss_slow"

for i, call in enumerate(calls_found):
    n_positional = len(call.args)
    kwarg_names = [kw.arg for kw in call.keywords]

    # Must have exactly 8 positional args
    assert (
        n_positional == 8
    ), f"Call {i}: expected 8 positional args, got {n_positional}"

    # sampling_per_token_logps must NOT be in kwargs (it's positional now)
    assert (
        "sampling_per_token_logps" not in kwarg_names
    ), f"Call {i}: sampling_per_token_logps should not be a kwarg"

    print(
        f"PASS: Call {i} at line {call.lineno}: {n_positional} positional args, "
        f"{len(kwarg_names)} kwargs, no duplicate sampling_per_token_logps"
    )

print(
    f"PASS: All {len(calls_found)} grpo_compute_loss_slow calls have correct arg structure"
)

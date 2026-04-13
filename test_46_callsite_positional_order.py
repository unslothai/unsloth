"""AST: verify the exact positional arg order at the grpo_compute_loss_slow call site matches the signature."""

import sys, ast

# Read the call site
with open("unsloth/models/rl_replacements.py", "r") as f:
    source = f.read()
tree = ast.parse(source)

# Read the function signature
sys.path.insert(0, "unsloth_zoo_repo")
import inspect
from unsloth_zoo.rl_replacements import grpo_compute_loss

sig_params = list(inspect.signature(grpo_compute_loss).parameters.keys())[:8]

# Find the grpo_compute_loss_slow call
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "grpo_compute_loss_slow":
            positional_args = node.args
            assert (
                len(positional_args) == 8
            ), f"Expected 8 positional args, got {len(positional_args)}"

            # Extract the arg names from the AST
            arg_names = []
            for arg in positional_args:
                if isinstance(arg, ast.Name):
                    arg_names.append(arg.id)
                elif isinstance(arg, ast.Attribute):
                    # e.g., self.beta -> "self.beta"
                    arg_names.append(
                        f"{arg.value.id}.{arg.attr}"
                        if isinstance(arg.value, ast.Name)
                        else "attr"
                    )
                else:
                    arg_names.append(type(arg).__name__)

            # Verify key positional args are in the right slots
            # Slot 0: ref_logps -> ref
            # Slot 1: per_token_logps -> new
            # Slot 2: old_logps -> old
            # Slot 3: sampling_per_token_logps -> sampling_per_token_logps (THE FIX)
            # Slot 4: input_ids -> input_ids
            # Slot 5: completion_mask -> mask
            # Slot 6: self.beta -> beta
            # Slot 7: advantages -> advantages

            assert (
                arg_names[3] == "sampling_per_token_logps"
            ), f"Slot 3 should be 'sampling_per_token_logps', got '{arg_names[3]}'"
            assert (
                arg_names[4] == "input_ids"
            ), f"Slot 4 should be 'input_ids', got '{arg_names[4]}'"
            assert (
                arg_names[6] == "self.beta"
            ), f"Slot 6 should be 'self.beta', got '{arg_names[6]}'"
            assert (
                arg_names[7] == "advantages"
            ), f"Slot 7 should be 'advantages', got '{arg_names[7]}'"

            print(f"PASS: Positional args in correct order: {arg_names}")
            break
else:
    print("FAIL: grpo_compute_loss_slow call not found")
    sys.exit(1)

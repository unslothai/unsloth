"""CPO shares ORPO's row-tokenization replacements (issue #4952).

CPOTrainer reuses ORPO's tokenize/init code, so the ORPO rewriters must also be
registered for cpo_trainer. The rewriters themselves are covered by
test_orpo_processor_text_tokenizer.py; here we just check cpo mirrors orpo.
Static, CPU-only, no torch.
"""

import ast
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RL_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _registrations(source):
    """Map each RL_FUNCTIONS[key] target to the appended function names."""
    out = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr != "append":
            continue
        sub = call.func.value
        if not (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)):
            continue
        if sub.value.id != "RL_FUNCTIONS":
            continue
        key = sub.slice
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        arg = call.args[0]
        if isinstance(arg, ast.Name):
            out.setdefault(key.value, []).append(arg.id)
    return out


def test_cpo_registration_matches_orpo():
    regs = _registrations(open(RL_PATH).read())
    shared = {"orpo_trainer_text_tokenizer", "orpo_trainer_processor_pad_token"}
    assert shared <= set(regs.get("orpo_trainer", []))
    assert shared <= set(regs.get("cpo_trainer", []))

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


def _load_pad_rewriter():
    """Exec orpo_trainer_processor_pad_token (+ _PAD_FALLBACK) without importing unsloth."""
    tree = ast.parse(open(RL_PATH).read())
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", None) == "_PAD_FALLBACK" for t in n.targets
        ):
            nodes.append(n)
        elif (
            isinstance(n, ast.FunctionDef)
            and n.name == "orpo_trainer_processor_pad_token"
        ):
            nodes.append(n)
    import re as _re

    ns = {"re": _re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), RL_PATH, "exec"), ns)
    return ns["orpo_trainer_processor_pad_token"]


def test_pad_token_default_routed_through_inner_tokenizer():
    # TRL 1.x CPO/ORPO __init__ defaults pad_token from eos_token before
    # tokenizing; on a multimodal processor those live on `.tokenizer`. The
    # rewrite must route both the default and pad_token_id through the inner
    # tokenizer so a processor without bare pad_token does not AttributeError.
    rewrite = _load_pad_rewriter()
    init_src = (
        "def __init__(self, model, args, processing_class):\n"
        "    if processing_class.pad_token is None:\n"
        "        processing_class.pad_token = processing_class.eos_token\n"
        "    self.pad_token_id = processing_class.pad_token_id\n"
    )
    out = rewrite("__init__", init_src)
    assert "if processing_class.pad_token is None:" not in out
    assert "processing_class.pad_token = processing_class.eos_token" not in out
    assert (
        "_unsloth_proc_tok = getattr(processing_class, 'tokenizer', processing_class)"
        in out
    )
    # bare pad_token_id must be routed through the getattr fallback, not left raw
    assert "= processing_class.pad_token_id\n" not in out
    ast.parse(out)  # rewritten source still compiles


def test_pad_rewrite_noop_without_bare_pad_block():
    # Older TRL (the pinned <=0.24.0 range) has no bare pad_token block; the
    # rewrite must only touch pad_token_id and leave everything else intact.
    rewrite = _load_pad_rewriter()
    init_src = (
        "def __init__(self, model, args, processing_class):\n"
        "    self.pad_token_id = processing_class.pad_token_id\n"
    )
    out = rewrite("__init__", init_src)
    assert "_unsloth_proc_tok" not in out
    assert "= processing_class.pad_token_id\n" not in out  # still routed via fallback
    ast.parse(out)

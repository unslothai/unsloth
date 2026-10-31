"""CPOTrainer reuses ORPO's tokenize code, so cpo_trainer must register the same ORPO rewriters."""

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
    regs = _registrations(open(RL_PATH, encoding = "utf-8").read())
    shared = {"orpo_trainer_text_tokenizer", "orpo_trainer_processor_pad_token"}
    assert shared <= set(regs.get("orpo_trainer", []))
    assert shared <= set(regs.get("cpo_trainer", []))


def _load_pad_rewriter():
    """Exec orpo_trainer_processor_pad_token (+ _PAD_FALLBACK) without importing unsloth."""
    tree = ast.parse(open(RL_PATH, encoding = "utf-8").read())
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", None) == "_PAD_FALLBACK" for t in n.targets
        ):
            nodes.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name == "orpo_trainer_processor_pad_token":
            nodes.append(n)
    import re as _re

    ns = {"re": _re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), RL_PATH, "exec"), ns)
    return ns["orpo_trainer_processor_pad_token"]


def test_pad_token_default_routed_through_inner_tokenizer():
    # Multimodal processors keep pad/eos on .tokenizer, so the rewrite must route through it.
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
    assert "_unsloth_proc_tok = getattr(processing_class, 'tokenizer', processing_class)" in out
    assert "= processing_class.pad_token_id\n" not in out
    ast.parse(out)


def test_pad_rewrite_noop_without_bare_pad_block():
    # Older TRL (<=0.24.0) has no bare pad_token block; only pad_token_id may change.
    rewrite = _load_pad_rewriter()
    init_src = (
        "def __init__(self, model, args, processing_class):\n"
        "    self.pad_token_id = processing_class.pad_token_id\n"
    )
    out = rewrite("__init__", init_src)
    assert "_unsloth_proc_tok" not in out
    assert "= processing_class.pad_token_id\n" not in out
    ast.parse(out)

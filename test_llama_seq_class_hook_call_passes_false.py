import ast
from pathlib import Path


def _find_llama():
    for p in [
        Path(__file__).resolve().parent / "unsloth" / "models" / "llama.py",
        Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("llama.py not found")


def _collect_hook_calls_under(node):
    """Yield every Call node invoking _attach_bnb_multidevice_hooks within the
    AST subtree rooted at `node`."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and getattr(sub.func, "id", None) == "_attach_bnb_multidevice_hooks"
        ):
            yield sub


def _kwarg_literal(call, name):
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
        if kw.arg == name:
            return kw.value
    return None


def test_seq_class_branch_passes_fast_inference_false():
    """The AutoModelForSequenceClassification branch (gated by
    `if num_labels is not None:`) must pass fast_inference=False to
    _attach_bnb_multidevice_hooks. That branch never reaches vLLM, so
    forwarding a truthy fast_inference would short-circuit hook install
    and regress multi-GPU bnb seq-class inference."""
    tree = ast.parse(_find_llama().read_text())

    hook_calls = []
    for if_node in ast.walk(tree):
        if not isinstance(if_node, ast.If):
            continue
        test = if_node.test
        is_num_labels_not_none = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "num_labels"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )
        if not is_num_labels_not_none:
            continue
        for body_node in if_node.body:
            hook_calls.extend(_collect_hook_calls_under(body_node))

    assert (
        hook_calls
    ), "No _attach_bnb_multidevice_hooks call found under `if num_labels is not None:`"
    for call in hook_calls:
        v = _kwarg_literal(call, "fast_inference")
        assert (
            v is False
        ), f"seq-class hook call must use fast_inference=False (got {ast.dump(call)})"

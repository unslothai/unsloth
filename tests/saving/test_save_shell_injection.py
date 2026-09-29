from __future__ import annotations

import ast
from pathlib import Path


SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"


def _get_function(source: str, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(source, filename = str(SAVE_PY))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Function {function_name} not found in save.py")


def _popen_calls(node: ast.AST) -> list[ast.Call]:
    calls = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "Popen"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "subprocess"
        ):
            calls.append(child)
    return calls


def _list_assignments(node: ast.AST, target: str) -> list[ast.List]:
    lists = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.List):
            if any(isinstance(t, ast.Name) and t.id == target for t in child.targets):
                lists.append(child.value)
    return lists


def test_lora_gguf_conversion_does_not_use_shell() -> None:
    """The LoRA -> GGUF conversion must pass argv as a list (no shell=True), so a crafted
    save path cannot inject shell commands. The conversion lives in the shared helper now."""
    helper = _get_function(SAVE_PY.read_text(encoding = "utf-8"), "_unsloth_save_lora_gguf")
    popen_calls = _popen_calls(helper)
    assert popen_calls, "Expected at least one subprocess.Popen call in _unsloth_save_lora_gguf"

    for call in popen_calls:
        shell = [
            kw
            for kw in call.keywords
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        ]
        assert not shell, "subprocess.Popen must not use shell=True"

        assert call.args, "subprocess.Popen must receive argv as a positional argument"
        argv = call.args[0]
        if isinstance(argv, ast.List):
            elts = argv.elts
        else:
            # argv is built as a list variable (cmd = [...]) and passed positionally.
            assert isinstance(argv, ast.Name), "argv must be a list or a list-built variable"
            assigned = _list_assignments(helper, argv.id)
            assert assigned, f"argv variable '{argv.id}' must be assigned a list literal"
            elts = assigned[0].elts

        assert len(elts) >= 2, "argv must include the interpreter and the converter script"
        first = elts[0]
        assert (
            isinstance(first, ast.Attribute) and first.attr == "executable"
        ), "argv[0] should be sys.executable, not a shell string"


def test_legacy_ggml_wrappers_delegate_safely() -> None:
    """The legacy ggml entry points must delegate to the shared helper and not build their
    own subprocess invocation."""
    source = SAVE_PY.read_text(encoding = "utf-8")
    for function_name in (
        "unsloth_convert_lora_to_ggml_and_push_to_hub",
        "unsloth_convert_lora_to_ggml_and_save_locally",
    ):
        node = _get_function(source, function_name)
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
        assert any(
            isinstance(c.func, ast.Name) and c.func.id == "_unsloth_save_lora_gguf" for c in calls
        ), f"{function_name} should delegate to _unsloth_save_lora_gguf"
        assert not _popen_calls(
            node
        ), f"{function_name} should not call subprocess.Popen directly anymore"

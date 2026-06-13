from __future__ import annotations

import ast
from pathlib import Path


SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"


def _function_calls(source: str, function_name: str) -> list[ast.Call]:
    tree = ast.parse(source, filename = str(SAVE_PY))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return [child for child in ast.walk(node) if isinstance(child, ast.Call)]
    raise AssertionError(f"Function {function_name} not found in save.py")


def _assert_safe_ggml_calls(calls: list[ast.Call]) -> None:
    popen_calls = []
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "Popen":
            if (
                isinstance(call.func.value, ast.Name)
                and call.func.value.id == "subprocess"
            ):
                popen_calls.append(call)

    assert popen_calls, "Expected at least one subprocess.Popen call"

    ggml_calls = []
    for call in popen_calls:
        if not call.args:
            continue
        argv = call.args[0]
        if isinstance(argv, ast.List) and len(argv.elts) >= 2:
            second_arg = argv.elts[1]
            if (
                isinstance(second_arg, ast.Constant)
                and second_arg.value == "llama.cpp/convert-lora-to-ggml.py"
            ):
                ggml_calls.append(call)

    assert ggml_calls, "Expected the GGML conversion subprocess call"

    for call in ggml_calls:
        shell_kwargs = [
            keyword
            for keyword in call.keywords
            if keyword.arg == "shell"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
        ]
        assert not shell_kwargs, "subprocess.Popen must not use shell=True"

        assert call.args, "subprocess.Popen must receive argv as a positional argument"
        argv = call.args[0]
        assert isinstance(
            argv, ast.List
        ), "subprocess.Popen must be called with an argv list"
        assert len(argv.elts) == 5, "GGML conversion argv should have five elements"

        second_arg = argv.elts[1]
        assert isinstance(second_arg, ast.Constant)
        assert second_arg.value == "llama.cpp/convert-lora-to-ggml.py"


def test_ggml_conversion_paths_do_not_use_shell() -> None:
    source = SAVE_PY.read_text(encoding = "utf-8")
    for function_name in (
        "unsloth_convert_lora_to_ggml_and_push_to_hub",
        "unsloth_convert_lora_to_ggml_and_save_locally",
    ):
        calls = _function_calls(source, function_name)
        _assert_safe_ggml_calls(calls)

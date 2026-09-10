"""Static guards that FP8/FP4 export requires explicit consent before installing llm-compressor (#8904)."""

from __future__ import annotations

import ast
from pathlib import Path

SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"

_MERGED_ENTRYPOINTS = (
    "unsloth_save_pretrained_merged",
    "unsloth_push_to_hub_merged",
    "unsloth_generic_save_pretrained_merged",
    "unsloth_generic_push_to_hub_merged",
)


def _module() -> ast.Module:
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"), filename = str(SAVE_PY))


def _get_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found in save.py")


def _first_lineno(fn: ast.AST, predicate) -> int | None:
    lines = [n.lineno for n in ast.walk(fn) if predicate(n) and hasattr(n, "lineno")]
    return min(lines) if lines else None


def test_manual_install_helper_is_exposed() -> None:
    names = {n.name for n in ast.walk(_module()) if isinstance(n, ast.FunctionDef)}
    assert "llm_compressor_manual_install_command" in names
    assert "_llm_compressor_missing_error" in names


def test_install_missing_dependencies_parameter_is_declared() -> None:
    fn = _get_function("install_llm_compressor")
    arg_names = [a.arg for a in fn.args.args]
    assert "install_missing_dependencies" in arg_names


def test_consent_gate_precedes_subprocess_install() -> None:
    fn = _get_function("install_llm_compressor")
    consent_line = _first_lineno(
        fn,
        lambda n: isinstance(n, ast.Name) and n.id == "install_missing_dependencies",
    )
    assert consent_line is not None, "install_missing_dependencies must gate installation"

    def _is_check_call(n: ast.AST) -> bool:
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "check_call"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "subprocess"
        )

    install_line = _first_lineno(fn, _is_check_call)
    assert install_line is not None
    assert consent_line < install_line, "consent must be checked before subprocess.check_call"


def test_compressed_export_entrypoints_declare_consent_kwarg() -> None:
    for name in _MERGED_ENTRYPOINTS:
        fn = _get_function(name)
        arg_names = [a.arg for a in fn.args.args]
        assert "install_missing_dependencies" in arg_names, f"{name} must expose the consent kwarg"


def test_compressed_export_forwards_consent_to_helper() -> None:
    fn = _get_function("_unsloth_save_compressed_tensors")
    src = ast.get_source_segment(SAVE_PY.read_text(encoding = "utf-8"), fn) or ""
    assert "install_llm_compressor(install_missing_dependencies" in src


def test_arguments_dict_strips_consent_kwarg() -> None:
    src = SAVE_PY.read_text(encoding = "utf-8")
    assert src.count('del arguments["install_missing_dependencies"]') >= 4

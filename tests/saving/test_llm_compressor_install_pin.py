"""Guard that the automatic first-use install of llm-compressor stays version-pinned.

``install_llm_compressor()`` auto-installs llm-compressor when a user requests an FP8/FP4
compressed export and it is not already present. A bare ``pip install llmcompressor`` would resolve
to whatever the configured package index serves, so a compromised, dependency-confused, or
inflated-version ("999.0.0") release could run under the Unsloth process at install/import time.

These are static checks (no import, no network, no GPU), mirroring test_save_shell_injection.py:
they keep the install command bounded to a vetted range and keep the opt-out env gate in place so
locked-down environments can forbid the automatic install entirely.
"""

from __future__ import annotations

import ast
from pathlib import Path

SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"

_ENV_FLAG = "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL"


def _module() -> ast.Module:
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"), filename = str(SAVE_PY))


def _get_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found in save.py")


def _spec_value():
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(
                isinstance(t, ast.Name) and t.id == "_LLM_COMPRESSOR_SPEC" for t in node.targets
            ):
                return node.value.value
    return None


def _first_lineno(fn: ast.AST, predicate) -> int | None:
    lines = [n.lineno for n in ast.walk(fn) if predicate(n) and hasattr(n, "lineno")]
    return min(lines) if lines else None


def test_spec_is_a_bounded_pin() -> None:
    spec = _spec_value()
    assert spec is not None, "_LLM_COMPRESSOR_SPEC must be defined at module scope"
    assert "llmcompressor" in spec, f"spec must name llmcompressor, got {spec!r}"
    # A lower and an upper bound: pip cannot jump to an arbitrary (e.g. inflated) future release.
    assert ">=" in spec and "<" in spec, f"spec must have lower and upper bounds, got {spec!r}"


def test_install_command_uses_pinned_spec_not_bare_name() -> None:
    fn = _get_function("install_llm_compressor")
    # No argv list may pass the bare, unpinned package literal "llmcompressor".
    for node in ast.walk(fn):
        if isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and elt.value == "llmcompressor":
                    raise AssertionError(
                        "install command must not pass an unpinned 'llmcompressor' literal; "
                        "use the bounded _LLM_COMPRESSOR_SPEC"
                    )
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "_LLM_COMPRESSOR_SPEC" in names, "install command must reference _LLM_COMPRESSOR_SPEC"


def test_optout_env_gate_precedes_subprocess_install() -> None:
    fn = _get_function("install_llm_compressor")
    env_line = _first_lineno(fn, lambda n: isinstance(n, ast.Constant) and n.value == _ENV_FLAG)
    assert env_line is not None, f"{_ENV_FLAG} opt-out must be checked in install_llm_compressor"

    def _is_check_call(n: ast.AST) -> bool:
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "check_call"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "subprocess"
        )

    install_line = _first_lineno(fn, _is_check_call)
    assert install_line is not None, "expected a subprocess.check_call install in the function"
    assert (
        env_line < install_line
    ), "the auto-install opt-out must be evaluated before any package install runs"

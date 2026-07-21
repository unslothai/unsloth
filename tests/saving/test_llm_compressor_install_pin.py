"""Static guards (no import/network/GPU, like test_save_shell_injection.py) that
install_llm_compressor()'s first-use auto-install of llm-compressor stays version-pinned to a vetted
range and keeps its opt-out env gate, so a compromised/inflated release can't be auto-pulled."""

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
                isinstance(t, ast.Name) and t.id == "_LLM_COMPRESSOR_SPEC"
                for t in node.targets
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
    assert (
        ">=" in spec and "<" in spec
    ), f"spec must have lower and upper bounds, got {spec!r}"


def test_ceiling_blocks_inflated_versions() -> None:
    """Cap to the exact vetted patch: block an inflated 0.x, a new major, and any higher in-range patch."""
    from packaging.requirements import Requirement

    spec = Requirement(_spec_value()).specifier
    assert spec.contains("0.12.0"), "the current vetted release must resolve"
    assert not spec.contains("0.999.0"), "an inflated 0.x must be blocked"
    assert not spec.contains("1.0.0"), "a new major must not be auto-installed"
    assert not spec.contains(
        "0.12.1"
    ), "a higher in-range patch must be blocked (cap to the vetted patch)"
    assert not spec.contains(
        "0.12.999"
    ), "a crafted higher in-range patch (e.g. on a mirror) must be blocked"


def test_floor_stays_compatible_with_supported_torch() -> None:
    """Floor must stay <=0.6.0: 0.7+ need torch>=2.7, but the pinned torch can be as old as 2.4."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    req = Requirement(_spec_value())
    lowers = [
        Version(s.version) for s in req.specifier if s.operator in (">=", "==", "~=")
    ]
    assert lowers, "spec must declare a lower bound"
    assert max(lowers) <= Version("0.6.0"), (
        f"floor {max(lowers)} requires a torch newer than Unsloth's minimum (2.4); "
        "llm-compressor >0.6.0 needs torch>=2.7. Keep the floor <= 0.6.0."
    )


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
    assert (
        "_LLM_COMPRESSOR_SPEC" in names
    ), "install command must reference _LLM_COMPRESSOR_SPEC"


def test_optout_env_gate_precedes_subprocess_install() -> None:
    fn = _get_function("install_llm_compressor")
    env_line = _first_lineno(
        fn, lambda n: isinstance(n, ast.Constant) and n.value == _ENV_FLAG
    )
    assert (
        env_line is not None
    ), f"{_ENV_FLAG} opt-out must be checked in install_llm_compressor"

    def _is_check_call(n: ast.AST) -> bool:
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "check_call"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "subprocess"
        )

    install_line = _first_lineno(fn, _is_check_call)
    assert (
        install_line is not None
    ), "expected a subprocess.check_call install in the function"
    assert (
        env_line < install_line
    ), "the auto-install opt-out must be evaluated before any package install runs"

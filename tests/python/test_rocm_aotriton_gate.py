"""Unsloth and Studio both open the ROCm AOTriton SDPA gate, whatever loaded first.

Torch reads TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL late: into a function-local
`static const bool` inside `check_flash_attention_hardware_support` and
`check_mem_efficient_hardware_support`
(`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, unchanged from 2.8 through 2.11
and main), so it is fixed at the first ROCm SDPA capability probe, not while the C++
extension loads. `import torch` before `import unsloth` is therefore still in time, and
that is the order `studio/backend/core/training/trainer.py` uses. Skipping the write there
leaves the gate shut and finetuning back on the quadratic MATH path #8819 measured.

The write is also a bare `os.environ.setdefault` in both files, asserted below. Deciding
from the torch build or the GPU architecture would mean importing torch: on the Studio side
that lands on the app-import path `utils/torch_warmup.py` exists to keep clear, and on
either side it would shadow a matrix that lives in the AOTriton binary inside the wheel.
That matrix moves between AOTriton releases and differs between the v2 and v3 builds of the
same release, so no allowlist written here can track it. Torch already does the filtering
that matters: it asks AOTriton for a kernel for this GPU, then asks whether the
architecture is experimental, and only then reads the variable.
"""

import ast
import functools
import os
import pathlib
import subprocess
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_INIT = _ROOT / "unsloth" / "__init__.py"
_STUDIO_MAIN = _ROOT / "studio" / "backend" / "main.py"
_STUDIO_RUN = _ROOT / "studio" / "backend" / "run.py"
_GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


@functools.cache
def _source(path):
    return path.read_text(encoding = "utf-8")


def _gate_statement(path):
    """The top-level statement that opens the gate, with any block it sits in."""
    for node in ast.parse(_source(path)).body:
        if _GATE in ast.unparse(node):
            return node
    raise AssertionError(f"no top-level statement in {path} sets {_GATE}")


def _open_gate(path, modules, environ):
    """Run that statement against a synthetic `sys.modules` / environment. Running it
    rather than grepping its source is what keeps a re-added guard visible."""
    scope = {
        "os": types.SimpleNamespace(environ = environ),
        "sys": types.SimpleNamespace(modules = modules),
    }
    exec(ast.unparse(_gate_statement(path)), scope)


def _first_torch_import(path):
    """Where the file first imports torch, at any nesting depth."""
    return min(
        (
            node.lineno
            for node in ast.walk(ast.parse(_source(path)))
            if (
                isinstance(node, ast.Import)
                and any(alias.name.split(".")[0] == "torch" for alias in node.names)
            )
            or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
        ),
        default = 10**9,
    )


_GATE_FILES = pytest.mark.parametrize("path", [_INIT, _STUDIO_MAIN], ids = ["unsloth", "studio"])


@_GATE_FILES
def test_an_unset_gate_is_opened(path):
    environ = {}
    _open_gate(path, {}, environ)
    assert environ.get(_GATE) == "1"


@_GATE_FILES
def test_an_already_imported_torch_does_not_skip_the_write(path):
    """The regression a `"torch" not in sys.modules` guard reintroduces. Torch has not read
    the variable yet at that point, so the write still decides the gate."""
    environ = {}
    _open_gate(path, {"torch": types.ModuleType("torch")}, environ)
    assert environ.get(_GATE) == "1"


@_GATE_FILES
@pytest.mark.parametrize("value", ["0", "1"])
def test_an_explicit_value_is_never_overwritten(path, value):
    """`c10::utils::check_env` reads "0" as false and "1" as true, so "0" is the opt-out and
    it has to survive both orderings."""
    for modules in ({}, {"torch": types.ModuleType("torch")}):
        environ = {_GATE: value}
        _open_gate(path, modules, environ)
        assert environ[_GATE] == value


@_GATE_FILES
def test_the_gate_is_a_bare_setdefault_that_cannot_reach_torch(path):
    """No helper, no probe, no version or architecture allowlist: the statement is exactly
    `os.environ.setdefault(<name>, "1")` with two literals, so nothing it runs can import
    torch, read the wheel, or grow a policy that has to track AOTriton's experimental set."""
    node = _gate_statement(path)
    assert isinstance(node, ast.Expr), ast.unparse(node)
    call = node.value
    assert isinstance(call, ast.Call), ast.unparse(node)
    assert ast.unparse(call.func) == "os.environ.setdefault"
    assert not call.keywords
    assert [getattr(arg, "value", None) for arg in call.args] == [_GATE, "1"]


@_GATE_FILES
def test_the_gate_precedes_the_files_own_torch_import(path):
    """The variable is only read at the first SDPA probe, but staying above every torch
    import keeps it that way even if something imported here probes on the way in."""
    assert _gate_statement(path).lineno < _first_torch_import(path)


def test_studio_opens_the_gate_before_route_and_hardware_imports():
    source = _source(_STUDIO_MAIN)
    gate = _gate_statement(_STUDIO_MAIN).lineno
    for marker in ("from routes import (", "from utils.hardware import ("):
        assert source.index(marker) > 0, marker
        assert source[: source.index(marker)].count("\n") + 1 > gate, marker


def test_all_studio_launches_converge_on_the_main_gate():
    """`run.py` execs the app rather than setting anything itself, so the launcher and a
    direct `uvicorn main:app` cannot disagree about the gate."""
    source = _source(_STUDIO_RUN)
    assert _GATE not in source
    assert "from main import app" in source


def test_installers_never_persist_the_gate():
    """The AOTriton opt-in belongs to Unsloth processes, not system or user config."""
    for name in (
        "install.sh",
        "install.ps1",
        "scripts/install_rocm_wsl_strixhalo.sh",
        "scripts/uninstall.sh",
        "scripts/uninstall.ps1",
    ):
        assert _GATE not in _source(_ROOT / name), name


def _run(code, **env):
    """Run `code` in a fresh interpreter, so no module state leaks between cases."""
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    # Importing Unsloth sets the gate; each case supplies its own starting value.
    clean = {k: v for k, v in os.environ.items() if k != _GATE}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = dict(clean, PYTHONPATH = os.pathsep.join(path), **env),
        timeout = 900,
    )


@functools.cache
def _unsloth_is_importable():
    return _run("import unsloth").returncode == 0


def _needs_unsloth():
    if not _unsloth_is_importable():
        pytest.skip("unsloth is not importable in this environment")


_REPORT = "import os, unsloth\nprint('GATE', os.environ.get({name!r}))".format(name = _GATE)


@pytest.mark.parametrize(
    "prologue",
    ["", "import torch\n"],
    ids = ["unsloth_first", "torch_first"],
)
def test_importing_unsloth_opens_the_gate(prologue):
    """The end-to-end shape of #8819, both ways round."""
    _needs_unsloth()
    out = _run(prologue + _REPORT)
    assert out.returncode == 0, out.stderr
    assert "GATE 1" in out.stdout, out.stdout


def test_an_explicit_opt_out_survives_the_import():
    _needs_unsloth()
    out = _run(_REPORT, **{_GATE: "0"})
    assert out.returncode == 0, out.stderr
    assert "GATE 0" in out.stdout, out.stdout

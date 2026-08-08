# SPDX-License-Identifier: AGPL-3.0-only

"""Regression tests for the CUDA-vs-MLX dispatch gates Unsloth relies on.

Two gates: (1) ``unsloth._IS_MLX`` (import-time, delegates to the zoo MLX
runtime gate behind a local precheck barrier); (2)
``utils.hardware.detect_hardware()`` (runtime, CUDA->XPU->MLX->CPU). These
are the canaries against "MLX support accidentally hijacks CUDA/AMD/Intel
users": we check the _IS_MLX helper structure, flip both gates True under a
spoofed Darwin+arm64 with a fake mlx module, and confirm both stay CUDA-side
on the real host. No real MLX install needed.
"""

import ast
import importlib
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UNSLOTH_INIT = REPO_ROOT / "unsloth" / "__init__.py"
MODELS_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"


# 1. Source-level structure check on _IS_MLX (no platform dependencies).


def test_is_mlx_gate_uses_three_required_predicates():
    """_IS_MLX must AND Darwin+arm64+importable-mlx; dropping any breaks dispatch."""
    tree = ast.parse(UNSLOTH_INIT.read_text(encoding = "utf-8"))

    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_IS_MLX"
        ):
            target = node.value
            break
    assert target is not None, "_IS_MLX assignment not found in unsloth/__init__.py"
    assert isinstance(target, ast.Call), "_IS_MLX must call the shared MLX helper"
    expr_src = ast.unparse(target)
    assert expr_src == "_is_mlx_available()", "_IS_MLX must delegate to the shared MLX runtime gate"

    helper = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_is_mlx_available":
            helper = node
            break
    assert helper is not None, "_is_mlx_available helper not found"

    helper_src = ast.unparse(helper)
    assert (
        "platform.system()" in helper_src
        and "'Darwin'" in helper_src
        and "platform.machine()" in helper_src
        and "'arm64'" in helper_src
        and "find_spec" in helper_src
        and "'mlx'" in helper_src
        and "from unsloth_zoo.mlx import is_mlx_available" in helper_src
    ), "_IS_MLX helper must precheck local MLX predicates before importing zoo"
    assert (
        "from unsloth_zoo.mlx import is_mlx_available" in helper_src
        and "return is_mlx_available()" in helper_src
    ), "_IS_MLX helper must delegate final detection to the shared zoo MLX runtime gate"
    assert helper_src.index("UNSLOTH_FORCE_GPU_PATH") < helper_src.index(
        "from unsloth_zoo.mlx import is_mlx_available"
    ), "_IS_MLX helper must run the local MLX precheck before importing zoo"


# 2. Runtime gate behavior with platform spoofed to Apple Silicon + fake mlx.
#    Re-evaluates the expression rather than reloading unsloth (avoids a torch
#    cascade-reload).


def _evaluate_is_mlx_precheck(platform_module, importlib_util, os_module):
    """Re-evaluate the local _is_mlx_available precheck with injected deps."""
    return (
        os_module.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") != "1"
        and platform_module.system() == "Darwin"
        and platform_module.machine() == "arm64"
        and importlib_util.find_spec("mlx") is not None
    )


def test_is_mlx_gate_true_on_apple_silicon_with_mlx_present(monkeypatch):
    import platform
    import importlib.util

    # Fake mlx so find_spec returns a non-None ModuleSpec.
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader = None)
    fake_mlx.__path__ = []
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    import os

    assert _evaluate_is_mlx_precheck(platform, importlib.util, os) is True


def test_is_mlx_gate_false_when_mlx_missing(monkeypatch):
    import platform
    import importlib.util

    # Apple Silicon but no mlx -> gate must be False.
    monkeypatch.delitem(sys.modules, "mlx", raising = False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    real_find_spec = importlib.util.find_spec

    def _no_mlx(name, *args, **kwargs):
        if name == "mlx":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _no_mlx)

    import os

    assert _evaluate_is_mlx_precheck(platform, importlib.util, os) is False


def test_is_mlx_gate_false_on_non_apple_silicon():
    """On the real Linux+CUDA / AMD / Intel test host, the gate stays False."""
    import platform
    import importlib.util

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import pytest
        pytest.skip("Test host is Apple Silicon; CUDA-side canary doesn't apply.")

    import os

    assert _evaluate_is_mlx_precheck(platform, importlib.util, os) is False


# ---------------------------------------------------------------------------
# 3. detect_hardware() picks MLX only when CUDA+XPU are both unavailable AND
#    the host is Apple Silicon AND mlx is importable.
# ---------------------------------------------------------------------------


def _import_studio_hardware():
    """Lazy import of the Unsloth hardware module (studio/backend on sys.path)."""
    studio_backend = REPO_ROOT / "studio" / "backend"
    if str(studio_backend) not in sys.path:
        sys.path.insert(0, str(studio_backend))
    from utils.hardware import hardware as hw  # type: ignore

    return hw


def test_detect_hardware_picks_mlx_when_only_apple_silicon_available(monkeypatch):
    hw = _import_studio_hardware()

    # Force CUDA + XPU off so detect_hardware falls through to MLX.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    # Spoof Apple Silicon + importable mlx.core for _has_mlx().
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    # detect_hardware now gates MLX on the full stack via _has_usable_mlx_stack()
    # (utils.mlx_repair.mlx_stack_available imports mlx_lm/mlx_vlm and checks
    # versions); faking mlx.core alone no longer satisfies it. This test asserts the
    # dispatch decision when the stack IS usable, so model that directly.
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: True)

    detected = hw.detect_hardware()
    assert detected == hw.DeviceType.MLX, f"expected MLX, got {detected!r}"


def test_detect_hardware_picks_cuda_on_real_host():
    """Canary: a real CUDA host must dispatch to CUDA even if mlx is importable."""
    import torch

    if not torch.cuda.is_available():
        import pytest
        pytest.skip("No CUDA available on this host; canary not applicable.")

    hw = _import_studio_hardware()
    detected = hw.detect_hardware()
    assert detected == hw.DeviceType.CUDA, f"CUDA host must dispatch to CUDA, got {detected!r}"


# ---------------------------------------------------------------------------
# 4. The MLX branch must report unsloth's OWN version, not unsloth-zoo's.
#    unsloth-zoo is a separately released distribution pinned with `>=`, so its
#    version routinely trails the core's; aliasing it made `unsloth.__version__`
#    disagree with `pip show unsloth` on every Apple Silicon install (#8171).
#    Executed straight out of the source so the test exercises the real
#    statements, not a reimplementation, and needs no torch/mlx to run.
# ---------------------------------------------------------------------------


_ZOO_SENTINEL = "0.0.0+zoo-sentinel"


def _mlx_branch(tree):
    """The body of the module-level ``if _IS_MLX:`` dispatch branch."""
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "_IS_MLX"
        ):
            return node
    raise AssertionError("`if _IS_MLX:` dispatch branch not found in unsloth/__init__.py")


def _assigns_version(node):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
    )


def _resolve_mlx_version(zoo_version):
    """Run the MLX branch's ``__version__`` resolution with a spoofed zoo.

    Executes the real source statements (plus any module-level helper they
    call) in an isolated namespace whose ``unsloth_zoo.__version__`` is a
    sentinel, so borrowing the zoo's version is directly observable.
    """
    tree = ast.parse(UNSLOTH_INIT.read_text(encoding = "utf-8"))

    version_stmts = [node for node in _mlx_branch(tree).body if _assigns_version(node)]
    assert version_stmts, "MLX branch does not assign __version__"

    called = {
        child.func.id
        for stmt in version_stmts
        for child in ast.walk(stmt)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in called
    ]

    fake_zoo = types.ModuleType("unsloth_zoo")
    fake_zoo.__version__ = zoo_version
    namespace = {"__file__": str(UNSLOTH_INIT), "unsloth_zoo": fake_zoo}
    for node in helpers + version_stmts:
        exec(compile(ast.Module(body = [node], type_ignores = []), "<mlx-branch>", "exec"), namespace)
    return namespace["__version__"]


def _packaging_version_literal():
    """The literal pyproject.toml resolves the distribution version from."""
    tree = ast.parse(MODELS_UTILS.read_text(encoding = "utf-8"))
    for node in tree.body:
        if _assigns_version(node) and isinstance(node.value, ast.Constant):
            return node.value.value
    raise AssertionError("__version__ literal not found in unsloth/models/_utils.py")


def _installed_unsloth_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("unsloth")
    except PackageNotFoundError:
        return None


def test_mlx_branch_reports_unsloth_version_not_zoo():
    """Apple Silicon must report unsloth's version, not whatever zoo resolved to."""
    resolved = _resolve_mlx_version(zoo_version = _ZOO_SENTINEL)

    assert resolved != _ZOO_SENTINEL, (
        "unsloth.__version__ is aliased to unsloth_zoo.__version__ on the MLX path; "
        "unsloth-zoo is pinned '>=' and trails the core, so this misreports the "
        "installed unsloth (issue #8171)"
    )
    expected = _installed_unsloth_version() or _packaging_version_literal()
    assert resolved == expected, f"MLX reported {resolved!r}, expected {expected!r}"


def test_gpu_branch_still_sources_version_from_gpu_init():
    """Canary: the GPU path was already correct; don't 'fix' the MLX path by breaking it."""
    tree = ast.parse(UNSLOTH_INIT.read_text(encoding = "utf-8"))

    gpu_branch = ast.Module(body = _mlx_branch(tree).orelse, type_ignores = [])
    assert "from ._gpu_init import __version__" in ast.unparse(gpu_branch), (
        "GPU path must keep resolving __version__ through _gpu_init -> models._utils"
    )

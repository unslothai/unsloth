# SPDX-License-Identifier: AGPL-3.0-only

"""
Regression tests for the CUDA-vs-MLX dispatch gates Studio relies on.

Two gates drive every dispatch decision in Studio's MLX path:

  1. ``unsloth._IS_MLX`` at the top of ``unsloth/__init__.py`` -- evaluated
     once at import time and read by Studio worker code to choose between
     the GPU and MLX trainer / inference / export paths. Defined as
     ``Darwin AND arm64 AND find_spec("mlx") is not None``.

  2. ``utils.hardware.detect_hardware()`` -- runtime probe in the Studio
     backend. Priority order: CUDA -> XPU -> MLX -> CPU. The MLX branch is
     reached only when both CUDA and XPU are unavailable AND the host is
     Apple Silicon AND mlx is importable.

These gates are the canaries for "MLX support accidentally hijacks
CUDA/AMD/Intel users". The tests here:

  * verify the source-level structure of the ``_IS_MLX`` expression so an
    accidental rewrite (e.g. dropping the ``arm64`` check) is caught,
  * exercise the runtime gate logic under a spoofed Darwin+arm64 platform
    with a fake ``mlx`` module in ``sys.modules`` to confirm both gates
    flip True together,
  * confirm that on the actual Linux+CUDA test host both gates remain in
    their CUDA-side state.

No real MLX install is required; uses the same ``monkeypatch.setitem``
fake-mlx pattern as ``test_mlx_inference_backend.py``.
"""

import ast
import importlib
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UNSLOTH_INIT = REPO_ROOT / "unsloth" / "__init__.py"


# ---------------------------------------------------------------------------
# 1. Source-level structure check on _IS_MLX (no platform dependencies).
# ---------------------------------------------------------------------------


def test_is_mlx_gate_uses_three_required_predicates():
    """The _IS_MLX assignment must AND together exactly the three checks
    that Studio depends on: Darwin OS, arm64 machine, and an importable
    mlx package. Dropping any one of them silently breaks dispatch.
    """
    tree = ast.parse(UNSLOTH_INIT.read_text())

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
    assert isinstance(target, ast.BoolOp) and isinstance(
        target.op, ast.And
    ), "_IS_MLX must be a BoolOp(And) of platform + mlx checks"

    expr_src = ast.unparse(target)
    assert (
        "platform.system()" in expr_src and "Darwin" in expr_src
    ), "_IS_MLX must check platform.system() == 'Darwin'"
    assert (
        "platform.machine()" in expr_src and "arm64" in expr_src
    ), "_IS_MLX must check platform.machine() == 'arm64'"
    assert (
        "find_spec" in expr_src and "'mlx'" in expr_src
    ), "_IS_MLX must check importlib.util.find_spec('mlx')"


# ---------------------------------------------------------------------------
# 2. Runtime gate behavior with the platform spoofed to Apple Silicon and a
#    fake mlx module in sys.modules.  Re-evaluates the same expression
#    rather than reloading unsloth (which would cascade-reload torch).
# ---------------------------------------------------------------------------


def _evaluate_is_mlx_gate(platform_module, importlib_util):
    """Re-evaluate the _IS_MLX expression using injected dependencies.

    Mirrors the assignment in unsloth/__init__.py exactly.
    """
    return (
        platform_module.system() == "Darwin"
        and platform_module.machine() == "arm64"
        and importlib_util.find_spec("mlx") is not None
    )


def test_is_mlx_gate_true_on_apple_silicon_with_mlx_present(monkeypatch):
    import platform
    import importlib.util

    # Inject a fake mlx package so find_spec returns a non-None ModuleSpec.
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader = None)
    fake_mlx.__path__ = []
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    assert _evaluate_is_mlx_gate(platform, importlib.util) is True


def test_is_mlx_gate_false_when_mlx_missing(monkeypatch):
    import platform
    import importlib.util

    # Apple Silicon platform but no mlx package -> gate must be False.
    monkeypatch.delitem(sys.modules, "mlx", raising = False)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    real_find_spec = importlib.util.find_spec

    def _no_mlx(name, *args, **kwargs):
        if name == "mlx":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _no_mlx)

    assert _evaluate_is_mlx_gate(platform, importlib.util) is False


def test_is_mlx_gate_false_on_non_apple_silicon():
    """On the real Linux+CUDA / AMD / Intel test host, the gate stays False."""
    import platform
    import importlib.util

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # On a Mac CI runner this assertion would not apply; skip there.
        import pytest

        pytest.skip("Test host is Apple Silicon; CUDA-side canary doesn't apply.")

    assert _evaluate_is_mlx_gate(platform, importlib.util) is False


# ---------------------------------------------------------------------------
# 3. Studio's runtime detect_hardware() picks MLX only when CUDA + XPU are
#    both unavailable AND the host is Apple Silicon AND mlx is importable.
# ---------------------------------------------------------------------------


def _import_studio_hardware():
    """Lazy import for the Studio hardware module, with the bare-imports
    convention that Studio uses (studio/backend on sys.path).
    """
    studio_backend = REPO_ROOT / "studio" / "backend"
    if str(studio_backend) not in sys.path:
        sys.path.insert(0, str(studio_backend))
    from utils.hardware import hardware as hw  # type: ignore

    return hw


def test_detect_hardware_picks_mlx_when_only_apple_silicon_available(monkeypatch):
    hw = _import_studio_hardware()

    # Force CUDA + XPU paths off so detect_hardware falls through to MLX.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    # Spoof Apple Silicon and provide an importable mlx.core for _has_mlx().
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    detected = hw.detect_hardware()
    assert detected == hw.DeviceType.MLX, f"expected MLX, got {detected!r}"


def test_detect_hardware_picks_cuda_on_real_host():
    """Canary: on a real CUDA host the MLX branch must NOT be taken even
    if mlx happens to be importable. Protects CUDA/AMD/Intel users from
    accidental MLX dispatch when MLX support is added.
    """
    import torch

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("No CUDA available on this host; canary not applicable.")

    hw = _import_studio_hardware()
    detected = hw.detect_hardware()
    assert (
        detected == hw.DeviceType.CUDA
    ), f"CUDA host must dispatch to CUDA, got {detected!r}"

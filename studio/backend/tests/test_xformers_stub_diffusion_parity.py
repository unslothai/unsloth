# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: the diffusion paths must stub xformers and torchao before importing diffusers.

The Windows xformers pin is CUDA-only. Against a ROCm torch, which has no distributed
backend, ``import xformers.ops`` dies inside torch.distributed, and diffusers imports
xformers on sight -- so a Windows ROCm host cannot load any image or video model, and
the error names neither xformers nor the real cause. diffusers reaches torchao the same
way through its quantizers. Both stubs are no-ops on every other runtime.

Every ``import diffusers`` in the diffusion modules is lazy, so a module-scope install is
what puts the stubs in place first; asserting module scope is what stops a later edit from
tucking one inside a function that a load path can skip.

The diffusion modules are not enough on their own: a stub only seeds names nothing has
imported yet, and by the time the server imports a diffusion module it has already pulled
torchao in through the route tree. So run.py installs both before its own first import too.

CPU-only: source is parsed with ``ast``, and the behaviour tests fake the platform probe.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from core import _torchao_stub
from core._torchao_stub import (
    install_torchao_windows_rocm_stub,
    install_xformers_windows_rocm_stub,
)

_BACKEND = Path(__file__).resolve().parent.parent  # studio/backend
_CORE = _BACKEND / "core"
# Renaming either installer breaks the import above, loudly, rather than these assertions.
_INSTALLS = frozenset(
    {install_xformers_windows_rocm_stub.__name__, install_torchao_windows_rocm_stub.__name__}
)

# Where diffusers gets imported: the loader, and the trainers' shared module (they run in a
# spawned child, so the loader's install does not carry over).
_DIFFUSION_MODULES = [
    _CORE / "inference" / "diffusion.py",
    _CORE / "training" / "diffusion_train_common.py",
]

_ENTRY_POINT = _BACKEND / "run.py"
_STUB_MODULE = "core._torchao_stub"
_ML_ROOTS = frozenset({"diffusers", "peft", "torch", "torchao", "transformers", "xformers"})
# Must run BEFORE the installers, not after: these set the env vars torch reads when it sizes
# its OpenMP/BLAS pools, so nothing heavy may precede them. Imports stdlib only.
_PRE_STUB = frozenset({"utils.cpu_threads"})


def _import_roots(node) -> set[str]:
    """Top-level package names a module-scope import statement pulls in."""
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module and node.module != _STUB_MODULE:
        if node.module in _PRE_STUB:
            return set()
        return {node.module.split(".")[0]}
    return set()


def _reaches_torch(root: str) -> bool:
    # Anything in the backend tree can, transitively; probing the tree keeps this honest as
    # modules come and go. stdlib roots fall through both checks.
    return root in _ML_ROOTS or (_BACKEND / root).is_dir() or (_BACKEND / f"{root}.py").is_file()


def _module_level_installs(tree) -> set[str]:
    """Names of the stub installers called as bare module-scope statements."""
    found = set()
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name) and func.id in _INSTALLS:
            found.add(func.id)
    return found


@pytest.mark.parametrize("path", _DIFFUSION_MODULES, ids = lambda p: p.name)
def test_diffusion_modules_install_both_stubs_at_module_scope(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))

    assert _module_level_installs(tree) == _INSTALLS, (
        f"{path.relative_to(_BACKEND)} must call both stub installers at module scope, before the "
        "lazy `import diffusers` calls below them."
    )


@pytest.mark.parametrize("path", _DIFFUSION_MODULES, ids = lambda p: p.name)
def test_diffusion_modules_install_before_any_torch_reaching_import(path):
    """Module scope alone is not the invariant: a sibling imported above the installs can pull
    torchao in first, and a stub only seeds names nothing has imported yet."""
    installed: set[str] = set()
    for node in ast.parse(path.read_text(encoding = "utf-8")).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id in _INSTALLS:
                installed.add(func.id)
            continue
        for root in _import_roots(node):
            if root in _ML_ROOTS:
                assert installed == _INSTALLS, (
                    f"{path.relative_to(_BACKEND)}:{node.lineno} imports {root} before installing "
                    f"{sorted(_INSTALLS - installed)}."
                )


def test_the_entry_point_installs_both_stubs_before_its_first_heavy_import():
    installed: set[str] = set()
    for node in ast.parse(_ENTRY_POINT.read_text(encoding = "utf-8")).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id in _INSTALLS:
                installed.add(func.id)
            continue
        for root in _import_roots(node):
            if _reaches_torch(root):
                assert installed == _INSTALLS, (
                    f"run.py:{node.lineno} imports {root} before installing "
                    f"{sorted(_INSTALLS - installed)}; a stub set after the first import that "
                    "reaches torchao or xformers is a no-op."
                )

    assert installed == _INSTALLS, "run.py must call both stub installers at module scope"


def _is_stub_key(name: str) -> bool:
    # Both packages, not just xformers: a torchao stub left behind turns a later
    # importorskip("torchao.quantization") into a silent no-op instead of a skip.
    return any(name == p or name.startswith(p + ".") for p in ("xformers", "torchao"))


@pytest.fixture
def on_windows_rocm(monkeypatch):
    """Force the Windows-ROCm probe on, and leave sys.modules / sys.meta_path as found."""
    monkeypatch.setattr(_torchao_stub, "_is_windows_rocm", lambda: True)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    saved = {k: v for k, v in sys.modules.items() if _is_stub_key(k)}
    for name in saved:
        del sys.modules[name]
    yield
    for name in [k for k in sys.modules if _is_stub_key(k)]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_xformers_is_stubbed_on_windows_rocm(on_windows_rocm):
    install_xformers_windows_rocm_stub()

    # The names diffusers imports, plus a deeper one nothing seeded: the finder covers it.
    import xformers  # noqa: F401
    import xformers.ops  # noqa: F401
    import xformers.ops.fmha  # noqa: F401

    for name in ("xformers", "xformers.ops", "xformers.ops.fmha"):
        assert sys.modules[name]._unsloth_stub is _torchao_stub._STUB_SENTINEL, name


def test_a_real_xformers_is_left_alone(on_windows_rocm, monkeypatch):
    import types

    real = types.ModuleType("xformers")
    monkeypatch.setitem(sys.modules, "xformers", real)

    install_xformers_windows_rocm_stub()

    assert sys.modules["xformers"] is real


def test_no_stub_off_windows_rocm(monkeypatch):
    # A CUDA or Linux host has a working xformers; shadowing it would cost real attention kernels.
    monkeypatch.setattr(_torchao_stub, "_is_windows_rocm", lambda: False)
    monkeypatch.delitem(sys.modules, "xformers", raising = False)

    install_xformers_windows_rocm_stub()

    assert "xformers" not in sys.modules


def test_the_finder_is_registered_once(on_windows_rocm):
    install_xformers_windows_rocm_stub()
    install_torchao_windows_rocm_stub()
    install_xformers_windows_rocm_stub()

    finders = [f for f in sys.meta_path if isinstance(f, _torchao_stub._StubSubpackageFinder)]
    assert len(finders) == 1


def test_no_stub_survives_this_module():
    """The fixture must hand sys.modules back clean: a leaked torchao stub is silently
    accepted by a later pytest.importorskip("torchao.quantization")."""
    for name in ("xformers", "torchao", "torchao.quantization"):
        assert not _torchao_stub.is_stubbed(name), name


def _cuda_bf16_target():
    import torch
    return type("T", (), {"device": "cuda", "dtype": torch.bfloat16})()


def test_dense_quant_declines_a_stubbed_torchao(on_windows_rocm):
    """The stub's quantize_ is a no-op, so the smoke probe passes on a still-dense Linear and
    the transformer gets MARKED quantised without being quantised. Decline before that."""
    from core.inference.diffusion_transformer_quant import dense_transformer_supported

    # The fixture cleared torchao out of sys.modules, so nothing is stubbed yet.
    target = _cuda_bf16_target()
    assert dense_transformer_supported(target) is True

    install_torchao_windows_rocm_stub()
    assert dense_transformer_supported(target) is False


@pytest.mark.parametrize("mode", ["int8", "fp8_dynamic", "nvfp4"])
def test_te_quant_declines_a_stubbed_torchao(on_windows_rocm, mode):
    """Same no-op, same false report, on the text encoders. ROCm answers device "cuda" and a
    capability pair, so nothing else in te_quant_supported catches it."""
    from core.inference.diffusion_precision import te_quant_supported

    target = _cuda_bf16_target()
    install_torchao_windows_rocm_stub()
    assert te_quant_supported(target, mode) is False


def test_layerwise_fp8_te_still_works_under_the_stub(on_windows_rocm):
    """Plain fp8 is a torch cast with no torchao in it, so the guard must not take it away."""
    from core.inference.diffusion_precision import te_quant_supported

    target = _cuda_bf16_target()
    install_torchao_windows_rocm_stub()
    import torch

    assert te_quant_supported(target, "fp8") is hasattr(torch, "float8_e4m3fn")


@pytest.mark.parametrize("mode", ["fp8", "mxfp8"])
def test_dit_training_refuses_dense_precision_under_the_stub(on_windows_rocm, mode):
    """_apply_fp8_training / _apply_mxfp8_training call the stub, get None, raise nothing and
    return True, so the run would report fp8 while training bf16. Fail fast instead."""
    from core.training.diffusion_dit_trainer import _resolve_base_precision

    install_torchao_windows_rocm_stub()
    cfg = type("C", (), {"base_precision": mode, "mixed_precision": "bf16"})()
    with pytest.raises(ValueError, match = "Windows-ROCm stub"):
        _resolve_base_precision(cfg, None, "cuda")


def test_xformers_is_never_selected_on_a_rocm_target(monkeypatch):
    """The one check standing between an API-supplied attention_backend="xformers" and a stub
    that returns None mid-denoise. diffusers' own probe is metadata-based, so with xformers
    installed it believes the stub is usable and will not refuse the backend itself."""
    import torch

    from core.inference.diffusion_attention import select_attention_backend

    monkeypatch.setattr(torch.version, "hip", "7.2.1", raising = False)  # ROCm build
    rocm = type("T", (), {"device": "cuda"})()
    for speed in (True, False):
        assert select_attention_backend(rocm, "xformers", speed_active = speed) is None
        assert select_attention_backend(rocm, "auto", speed_active = speed) != "xformers"

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


def _import_roots(node) -> set[str]:
    """Top-level package names a module-scope import statement pulls in."""
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module and node.module != _STUB_MODULE:
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


@pytest.fixture
def on_windows_rocm(monkeypatch):
    """Force the Windows-ROCm probe on, and leave sys.modules / sys.meta_path as found."""
    monkeypatch.setattr(_torchao_stub, "_is_windows_rocm", lambda: True)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    saved = {k: v for k, v in sys.modules.items() if k == "xformers" or k.startswith("xformers.")}
    for name in saved:
        del sys.modules[name]
    yield
    for name in [k for k in sys.modules if k == "xformers" or k.startswith("xformers.")]:
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

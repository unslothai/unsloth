# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _CUDA_TORCH_PKG_SPEC in install_python_stack.py.

The CUDA repair path installs the torch trio from an exclusive --index-url (no
PyPI fallback), so these pinned ranges decide which torch the venv gets. The
upper bound is locked to the 2.11.x family to match the base image and rocm7.2
spec and to keep the companions off a torch-2.12 wheel that would ABI-mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

# install_python_stack.py lives at repo_root/studio/install_python_stack.py
_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"


def _load_module(monkeypatch):
    """(Re-)import and return install_python_stack (mirrors test_torchao_select)."""
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


def _spec_of(pkg_spec: str):
    """Parse 'torch>=2.4,<2.12.0' into a packaging SpecifierSet."""
    return Requirement(pkg_spec).specifier


@pytest.mark.parametrize(
    "index, allowed, rejected",
    [
        # torch: 2.11.x allowed (matches base image); 2.12.x excluded.
        (0, ["2.11.0", "2.11.2", "2.10.0", "2.4.0"], ["2.12.0", "2.3.0", "1.13.1"]),
        # torchvision: 0.26.x (torch 2.11 companion) allowed; 0.27.x (torch 2.12) out.
        (1, ["0.26.0", "0.26.1", "0.19.0"], ["0.27.0", "0.18.0"]),
        # torchaudio: same 2.11.x window as torch.
        (2, ["2.11.0", "2.10.0", "2.4.0"], ["2.12.0", "2.3.0"]),
    ],
)
def test_cuda_spec_bounds(monkeypatch, index, allowed, rejected):
    mod = _load_module(monkeypatch)
    spec = _spec_of(mod._CUDA_TORCH_PKG_SPEC[index])
    for v in allowed:
        assert spec.contains(v, prereleases = True), f"{v} should satisfy {spec}"
    for v in rejected:
        assert not spec.contains(v, prereleases = True), f"{v} should not satisfy {spec}"


def test_cuda_spec_matches_rocm72_upper_bound(monkeypatch):
    """CUDA and rocm7.2 target the same torch 2.11.x family, so their upper
    bounds must stay in lockstep (bump both together at 2.12.x)."""
    mod = _load_module(monkeypatch)
    rocm72 = mod._ROCM_TORCH_PKG_SPECS["rocm7.2"]

    def _upper(pkg_spec: str) -> str:
        for clause in _spec_of(pkg_spec):
            if clause.operator == "<":
                return clause.version
        raise AssertionError(f"no upper bound in {pkg_spec!r}")

    for cuda_pkg, rocm_pkg in zip(mod._CUDA_TORCH_PKG_SPEC, rocm72, strict = True):
        assert _upper(cuda_pkg) == _upper(
            rocm_pkg
        ), f"CUDA {cuda_pkg!r} upper bound must match rocm7.2 {rocm_pkg!r}"

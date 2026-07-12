"""Regression guard for the CUDA torch2110 optional-dependency extras.

torch 2.11's DEFAULT PyPI wheel is CUDA 13.0 (torch 2.10 defaulted to CUDA 12.x).
So the CUDA-12 `cuXXXonlytorch2110` extras must pin the torch trio to the matching
`+cuXXX` local build; a bare `torch>=2.11` there would resolve a cu130 torch from
PyPI alongside the cu126/cu128 xformers wheel and fail at import. The cu130 extra
needs no local pin because the bare default already lands on cu130.

Hermetic: only parses pyproject.toml, no network or install.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:  # tomllib is stdlib on Python 3.11+; older interpreters need the tomli backport.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 / 3.10
    tomllib = pytest.importorskip("tomli")

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_TORCH_TRIO = ("torch", "torchvision", "torchaudio")


def _extra(name: str) -> list[str]:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"][name]


def _reqs(specs: list[str]) -> dict[str, Requirement]:
    out = {}
    for spec in specs:
        r = Requirement(spec)
        out[r.name.lower()] = r
    return out


@pytest.mark.parametrize("cuda", ["cu126", "cu128"])
def test_cuda12_torch2110_pins_matching_local_build(cuda: str):
    # Each of torch/torchvision/torchaudio must pin the exact +cuXXX local build
    # so it can only resolve from the matching PyTorch CUDA index, never the
    # CUDA-13 default on PyPI.
    reqs = _reqs(_extra(f"{cuda}onlytorch2110"))
    for pkg in _TORCH_TRIO:
        spec = str(reqs[pkg].specifier)
        assert (
            spec == f"=={('2.11.0' if pkg != 'torchvision' else '0.26.0')}+{cuda}"
        ), f"{cuda}onlytorch2110: {pkg} pinned as '{spec}', expected the +{cuda} local build"
    # xformers must come from the same CUDA index.
    xf = reqs["xformers"]
    assert xf.url and f"/whl/{cuda}/" in xf.url, f"xformers not on the {cuda} index: {xf.url}"


def test_cu130_torch2110_uses_arbitrary_equality():
    # cu130 matches torch 2.11's PyPI default, so no +cu130 local pin is needed
    # (it could fail to resolve from PyPI's unlabelled default wheel). But a
    # RANGE is not enough either: PEP 440 ranks a local build (2.11.0+cu126)
    # above the unlabelled 2.11.0, so with any CUDA-12 extra index configured a
    # range silently pairs a CUDA-12 trio with the cu130 xformers. Arbitrary
    # equality (===) matches only the unlabelled release, excluding every
    # +cuXXX local candidate.
    expected = {"torch": "2.11.0", "torchvision": "0.26.0", "torchaudio": "2.11.0"}
    reqs = _reqs(_extra("cu130onlytorch2110"))
    for pkg in _TORCH_TRIO:
        spec = str(reqs[pkg].specifier)
        assert spec == f"==={expected[pkg]}", (
            f"cu130onlytorch2110: {pkg} must pin ==={expected[pkg]} "
            f"(excludes +cuXXX local builds), got '{spec}'"
        )

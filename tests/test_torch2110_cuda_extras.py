"""Regression guard for the CUDA torch2110 optional-dependency extras.

torch 2.11's DEFAULT PyPI wheel is CUDA 13.0 (torch 2.10 defaulted to CUDA 12.x).
So the CUDA-12 `cuXXXonlytorch2110` extras must pin the torch trio to the matching
`+cuXXX` local build; a bare `torch>=2.11` there would resolve a cu130 torch from
PyPI alongside the cu126/cu128 xformers wheel and fail at import. The cu130 extra
needs no local pin because the bare default already lands on cu130.

Hermetic: only parses pyproject.toml, no network or install.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

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
        assert spec == f"=={('2.11.0' if pkg != 'torchvision' else '0.26.0')}+{cuda}", (
            f"{cuda}onlytorch2110: {pkg} pinned as '{spec}', expected the +{cuda} local build"
        )
    # xformers must come from the same CUDA index.
    xf = reqs["xformers"]
    assert xf.url and f"/whl/{cuda}/" in xf.url, f"xformers not on the {cuda} index: {xf.url}"


def test_cu130_torch2110_stays_bare():
    # cu130 matches torch 2.11's PyPI default, so no local pin is needed (and a
    # +cu130 pin could fail to resolve from PyPI's unlabelled default wheel).
    reqs = _reqs(_extra("cu130onlytorch2110"))
    for pkg in _TORCH_TRIO:
        spec = str(reqs[pkg].specifier)
        assert "+cu" not in spec, f"cu130onlytorch2110: {pkg} should stay bare, got '{spec}'"
        assert ">=2.11" in spec or ">=0.26" in spec, f"cu130: {pkg} lost its 2.11 floor: '{spec}'"

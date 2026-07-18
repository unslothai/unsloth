# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression guard for the CUDA torch2110 optional-dependency extras.

The cuXXXonlytorch2110 extras must pin the torch trio to the matching +cuXXX local
build (torch 2.11 defaults to a CUDA-13 PyPI wheel), or resolution mismatches the
xformers wheel. Hermetic: only parses pyproject.toml, no network or install.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:  # tomllib is stdlib on 3.11+; older interpreters need the tomli backport.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 / 3.10
    tomllib = pytest.importorskip("tomli")

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_TORCH_TRIO = ("torch", "torchvision", "torchaudio")


def _extra(name: str) -> list[str]:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"][name]


def _reqs(specs: list[str]) -> dict[str, list[Requirement]]:
    # name -> reqs (one Linux + one Windows xformers per extra)
    out: dict[str, list[Requirement]] = {}
    for spec in specs:
        r = Requirement(spec)
        out.setdefault(r.name.lower(), []).append(r)
    return out


@pytest.mark.parametrize("cuda", ["cu126", "cu128", "cu130"])
def test_cuda12_torch2110_pins_matching_local_build(cuda: str):
    reqs = _reqs(_extra(f"{cuda}onlytorch2110"))
    for pkg in _TORCH_TRIO:
        (req,) = reqs[pkg]
        spec = str(req.specifier)
        assert (
            spec == f"=={('2.11.0' if pkg != 'torchvision' else '0.26.0')}+{cuda}"
        ), f"{cuda}onlytorch2110: {pkg} pinned as '{spec}', expected the +{cuda} local build"
    xformers = reqs["xformers"]
    assert len(xformers) == 2, f"expected Linux + Windows xformers wheels, got {xformers}"
    linux = [r for r in xformers if r.url and r.url.endswith("manylinux_2_28_x86_64.whl")]
    windows = [r for r in xformers if r.url and r.url.endswith("win_amd64.whl")]
    assert len(linux) == 1 and len(windows) == 1, f"unexpected xformers wheels: {xformers}"
    for r in linux + windows:
        assert (
            f"/whl/{cuda}/xformers-0.0.35-" in r.url
        ), f"xformers not on the {cuda} index: {r.url}"
        # markers must exclude aarch64 / ARM64
        assert r.marker is not None
        assert not r.marker.evaluate({"sys_platform": "linux", "platform_machine": "aarch64"})
        assert not r.marker.evaluate({"sys_platform": "win32", "platform_machine": "ARM64"})
    assert linux[0].marker.evaluate({"sys_platform": "linux", "platform_machine": "x86_64"})
    assert windows[0].marker.evaluate({"sys_platform": "win32", "platform_machine": "AMD64"})


@pytest.mark.parametrize("cuda", ["cu126", "cu128", "cu130"])
@pytest.mark.parametrize("variant", ["", "ampere-"])
def test_torch2110_wrapper_references_matching_leaf(cuda: str, variant: str):
    specs = _extra(f"{cuda}-{variant}torch2110")
    assert specs == [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
        f"unsloth[{cuda}onlytorch2110]",
    ]


@pytest.mark.parametrize("cuda", ["cu126", "cu128", "cu130"])
def test_cuda12_torch2100_keeps_torch_pinned_off_x86(cuda: str):
    # xformers wheels now carry x86-64 markers, so the leaf must pin torch for ARM64.
    reqs = _reqs(_extra(f"{cuda}onlytorch2100"))
    (torch_req,) = reqs["torch"]
    assert str(torch_req.specifier) == "==2.10.0", (
        f"{cuda}onlytorch2100 must pin torch==2.10.0 for machines where the "
        f"x86-64-only xformers wheel (and its transitive pin) is skipped"
    )
    assert torch_req.marker is None, "the torch pin must apply on every machine"

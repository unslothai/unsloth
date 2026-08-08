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

"""Regression guard for the CUDA torch2110 and torch212x optional-dependency extras.

The cuXXXonlytorch2110 / cuXXXonlytorch212X extras must pin the torch trio to the
matching +cuXXX local build (these releases default to a CUDA-13 PyPI wheel, and
xformers 0.0.35 depends on torch without pinning it), or resolution walks torch up
to the newest release on the index and mismatches the xformers wheel. Hermetic:
only parses pyproject.toml and _auto_install.py, no network or install.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:  # tomllib is stdlib on 3.11+; older interpreters need the tomli backport.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9 / 3.10
    tomllib = pytest.importorskip("tomli")

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
AUTO_INSTALL = REPO / "unsloth" / "_auto_install.py"
_TORCH_TRIO = ("torch", "torchvision", "torchaudio")
# torchaudio has no 2.12 release, so the 2.12 leaves keep the unpinned 2.11.0 audio wheel.
_TORCH212_TRIO = {
    "torch2120": {"torch": "2.12.0", "torchvision": "0.27.0", "torchaudio": "2.11.0"},
    "torch2121": {"torch": "2.12.1", "torchvision": "0.27.1", "torchaudio": "2.11.0"},
}
# torch 2.12 is absent from the cu128 index, so only these two flavors get 2.12 extras.
_TORCH212_CUDA = ("cu126", "cu130")


def _extras() -> dict[str, list[str]]:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]


def _extra(name: str) -> list[str]:
    return _extras()[name]


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


@pytest.mark.parametrize("cuda", _TORCH212_CUDA)
@pytest.mark.parametrize("series", sorted(_TORCH212_TRIO))
def test_cuda12_torch212_pins_matching_local_build(cuda: str, series: str):
    reqs = _reqs(_extra(f"{cuda}only{series}"))
    for pkg, want in _TORCH212_TRIO[series].items():
        (req,) = reqs[pkg]
        spec = str(req.specifier)
        assert spec == f"=={want}+{cuda}", (
            f"{cuda}only{series}: {pkg} pinned as '{spec}', "
            f"expected the =={want}+{cuda} local build"
        )
        assert req.marker is None, f"the {pkg} pin must apply on every machine"
    xformers = reqs["xformers"]
    linux = [r for r in xformers if r.url and r.url.endswith("manylinux_2_28_x86_64.whl")]
    windows = [r for r in xformers if r.url and r.url.endswith("win_amd64.whl")]
    assert len(linux) == 1 and len(windows) == 1, f"unexpected xformers wheels: {xformers}"
    for r in linux + windows:
        assert (
            f"/whl/{cuda}/xformers-0.0.35-" in r.url
        ), f"xformers not on the {cuda} index: {r.url}"
        assert r.marker is not None
        assert not r.marker.evaluate({"sys_platform": "linux", "platform_machine": "aarch64"})
        assert not r.marker.evaluate({"sys_platform": "win32", "platform_machine": "ARM64"})
    assert linux[0].marker.evaluate({"sys_platform": "linux", "platform_machine": "x86_64"})
    assert windows[0].marker.evaluate({"sys_platform": "win32", "platform_machine": "AMD64"})


@pytest.mark.parametrize("cuda", _TORCH212_CUDA)
@pytest.mark.parametrize("series", sorted(_TORCH212_TRIO))
@pytest.mark.parametrize("variant", ["", "ampere-"])
def test_torch212_wrapper_references_matching_leaf(cuda: str, series: str, variant: str):
    specs = _extra(f"{cuda}-{variant}{series}")
    assert specs == [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
        f"unsloth[{cuda}only{series}]",
    ]


@pytest.mark.parametrize("series", sorted(_TORCH212_TRIO))
def test_no_cu128_torch212_extras(series: str):
    # torch 2.12 is not published on the cu128 index; a cu128 leaf would be unresolvable.
    names = _extras()
    for name in (f"cu128only{series}", f"cu128-{series}", f"cu128-ampere-{series}"):
        assert name not in names, f"{name} cannot resolve: no torch 2.12 on the cu128 index"


@pytest.mark.parametrize("series", sorted(_TORCH212_TRIO))
def test_auto_install_maps_torch212_to_defined_extras(series: str):
    # The printed command must name extras that exist, and must add the index that
    # serves the +cuNNN local builds those extras pin.
    source = AUTO_INSTALL.read_text()
    assert f"'cu{{}}{{}}-{series}'" in source, f"_auto_install.py never selects {series}"
    assert f"'-{series}'" in source, f"{series} missing from the extra-index-url gate"
    names = _extras()
    for cuda in _TORCH212_CUDA:
        for variant in ("", "-ampere"):
            assert f"cu{cuda[2:]}{variant}-{series}" in names


def test_auto_install_rejects_cuda128_on_torch212():
    # cu128 tops out at torch 2.11, so 2.12 on that flavor must fail loudly rather
    # than print an install command for an extra that does not exist.
    source = AUTO_INSTALL.read_text()
    assert 'if v >= V(\'2.12.0\') and cuda not in ("12.6", "13.0")' in source


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

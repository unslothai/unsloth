"""Regression guard for the CUDA torch2110 optional-dependency extras.

torch 2.11's DEFAULT PyPI wheel is CUDA 13.0 (torch 2.10 defaulted to CUDA 12.x).
So the CUDA-12 `cuXXXonlytorch2110` extras must pin the torch trio to the matching
`+cuXXX` local build; a bare `torch>=2.11` there would resolve a cu130 torch from
PyPI alongside the cu126/cu128 xformers wheel and fail at import. The cu130 extra is
pinned to +cu130 too: a bare or ===-pinned spec either lets a foreign CUDA index
outrank the intended wheel or force-replaces official cu130-index installs.

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


def _reqs(specs: list[str]) -> dict[str, list[Requirement]]:
    # Keyed by name -> list: each extra carries one Linux and one Windows
    # xformers requirement, so a plain name -> Requirement dict would silently
    # drop the Linux entry.
    out: dict[str, list[Requirement]] = {}
    for spec in specs:
        r = Requirement(spec)
        out.setdefault(r.name.lower(), []).append(r)
    return out


@pytest.mark.parametrize("cuda", ["cu126", "cu128", "cu130"])
def test_cuda12_torch2110_pins_matching_local_build(cuda: str):
    # Each of torch/torchvision/torchaudio must pin the exact +cuXXX local build
    # so it can only resolve from the matching PyTorch CUDA index, never the
    # CUDA-13 default on PyPI.
    reqs = _reqs(_extra(f"{cuda}onlytorch2110"))
    for pkg in _TORCH_TRIO:
        (req,) = reqs[pkg]
        spec = str(req.specifier)
        assert (
            spec == f"=={('2.11.0' if pkg != 'torchvision' else '0.26.0')}+{cuda}"
        ), f"{cuda}onlytorch2110: {pkg} pinned as '{spec}', expected the +{cuda} local build"
    # Both xformers wheels (Linux and Windows) must come from the same CUDA index.
    xformers = reqs["xformers"]
    assert len(xformers) == 2, f"expected Linux + Windows xformers wheels, got {xformers}"
    linux = [r for r in xformers if r.url and r.url.endswith("manylinux_2_28_x86_64.whl")]
    windows = [r for r in xformers if r.url and r.url.endswith("win_amd64.whl")]
    assert len(linux) == 1 and len(windows) == 1, f"unexpected xformers wheels: {xformers}"
    for r in linux + windows:
        assert (
            f"/whl/{cuda}/xformers-0.0.35-" in r.url
        ), f"xformers not on the {cuda} index: {r.url}"
        # The wheels are x86-64 only, so the markers must exclude other machines
        # (e.g. Linux aarch64 such as GB200/DGX Spark, Windows ARM64) where the
        # torch trio resolves fine but these wheels would abort the install.
        assert r.marker is not None
        assert not r.marker.evaluate({"sys_platform": "linux", "platform_machine": "aarch64"})
        assert not r.marker.evaluate({"sys_platform": "win32", "platform_machine": "ARM64"})
    assert linux[0].marker.evaluate({"sys_platform": "linux", "platform_machine": "x86_64"})
    assert windows[0].marker.evaluate({"sys_platform": "win32", "platform_machine": "AMD64"})


@pytest.mark.parametrize("cuda", ["cu126", "cu128", "cu130"])
@pytest.mark.parametrize("variant", ["", "ampere-"])
def test_torch2110_wrapper_references_matching_leaf(cuda: str, variant: str):
    # The six public wrappers must pull in the usual huggingface + bitsandbytes
    # pair and reference the internal leaf of the SAME CUDA version.
    specs = _extra(f"{cuda}-{variant}torch2110")
    assert specs == [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
        f"unsloth[{cuda}onlytorch2110]",
    ]


@pytest.mark.parametrize("cuda", ["cu126", "cu128", "cu130"])
def test_cuda12_torch2100_keeps_torch_pinned_off_x86(cuda: str):
    # The torch2100 leaves used to rely on the xformers 0.0.34 wheel's transitive
    # torch==2.10.0 pin. Now that the x86-64-only wheels carry platform_machine
    # markers, the leaf must pin torch explicitly so an ARM64 install stays on
    # torch 2.10 (Linux aarch64 wheels exist) or fails loudly (Windows ARM64)
    # instead of resolving an unpinned newer torch.
    reqs = _reqs(_extra(f"{cuda}onlytorch2100"))
    (torch_req,) = reqs["torch"]
    assert str(torch_req.specifier) == "==2.10.0", (
        f"{cuda}onlytorch2100 must pin torch==2.10.0 for machines where the "
        f"x86-64-only xformers wheel (and its transitive pin) is skipped"
    )
    assert torch_req.marker is None, "the torch pin must apply on every machine"

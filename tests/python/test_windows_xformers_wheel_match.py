# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""xformers/_C.pyd is linked to one exact torch/CUDA pair; a mismatch only warns and drops kernels."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

WHEEL_INDEX_BASE = "https://download.pytorch.org/whl"

# (CUDA family, torch release) -> xFormers version published by that index.
# Keep in step with wheel_utils._XFORMERS_WHEEL_VERSIONS and install.ps1 (enforced below).
XFORMERS_WHEEL_MATRIX: dict[tuple[str, str], str] = {
    ("cu126", "290"): "0.0.33.post1",
    ("cu128", "290"): "0.0.33.post1",
    ("cu130", "290"): "0.0.33.post1",
    ("cu126", "291"): "0.0.33.post2",
    ("cu128", "291"): "0.0.33.post2",
    ("cu130", "291"): "0.0.33.post2",
    ("cu126", "2100"): "0.0.34",
    ("cu128", "2100"): "0.0.34",
    ("cu130", "2100"): "0.0.34",
}

# Empty: 0.0.35 is stable-ABI and loads under 2.11 and up.
TORCH_RELEASES_WITHOUT_XFORMERS_WHEELS: tuple[str, ...] = ()


def _tomllib():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    return pytest.importorskip("tomli")


def _extras() -> dict[str, list[str]]:
    tomllib = _tomllib()
    return tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))["project"]["optional-dependencies"]


def _windows_xformers(deps: list[str]) -> list[str]:
    """Every xformers dep in ``deps``, URL or version pin, whose marker holds on Win x64."""
    markers = pytest.importorskip("packaging.markers")
    env = {
        "sys_platform": "win32",
        "platform_machine": "AMD64",
        "platform_system": "Windows",
        "os_name": "nt",
        "python_version": "3.13",
        "python_full_version": "3.13.2",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }
    found: list[str] = []
    for dep in deps:
        spec, _, marker_text = dep.partition(";")
        spec = spec.strip()
        if not spec.startswith("xformers"):
            continue
        if marker_text.strip() and not markers.Marker(marker_text.strip()).evaluate(env):
            continue
        found.append(spec)
    return found


@pytest.mark.parametrize(("family", "torch_tag"), sorted(XFORMERS_WHEEL_MATRIX))
def test_windows_resolves_a_cuda_matched_wheel(family: str, torch_tag: str):
    """`unsloth[cu128-torch2100]` / `unsloth[cu130-torch2100]` and friends must land on
    a win_amd64 wheel served by their OWN CUDA index -- never PyPI, never a neighbour's."""
    version = XFORMERS_WHEEL_MATRIX[(family, torch_tag)]
    extra = f"{family}onlytorch{torch_tag}"
    specs = _windows_xformers(_extras()[extra])

    assert (
        len(specs) == 1
    ), f"{extra} must resolve exactly one xformers wheel on Windows, got {specs}"
    spec = specs[0]
    if "@" in spec:
        url = spec.split("@", 1)[1].strip()
        assert url == f"{WHEEL_INDEX_BASE}/{family}/xformers-{version}-cp39-abi3-win_amd64.whl"
    else:
        # Index is the install-time --index-url; the pin must be the version it publishes for this torch.
        assert (
            spec == f"xformers=={version}"
        ), f"{extra} must pin the {family} wheel version for torch {torch_tag}, got {spec!r}"


@pytest.mark.parametrize(("family", "torch_tag"), sorted(XFORMERS_WHEEL_MATRIX))
def test_aggregate_extra_pulls_in_the_matched_wheel(family: str, torch_tag: str):
    """The user-facing `cuXXX-torchYYY` extra must reference the `only` extra that
    carries the win_amd64 row, else the Windows route documented in pyproject.toml
    silently degrades to whatever `unsloth[huggingface]` drags in."""
    aggregate = _extras()[f"{family}-torch{torch_tag}"]
    assert f"unsloth[{family}onlytorch{torch_tag}]" in aggregate


@pytest.mark.parametrize("torch_tag", TORCH_RELEASES_WITHOUT_XFORMERS_WHEELS)
def test_no_extras_invented_for_torch_without_xformers_wheels(torch_tag: str):
    extras = _extras()
    # CUDA extras only; the XPU extras carry no xformers row.
    pattern = re.compile(rf"^cu\d+(?:only)?-?torch{torch_tag}$")
    offenders = [n for n in extras if pattern.match(n)]
    assert offenders == [], (
        f"no xFormers wheel is published for torch {torch_tag}; extras {offenders} "
        "would resolve a wheel built for a different torch minor"
    )


def test_windows_extra_xformers_spec_is_a_version_range():
    """The windows extra is deliberately uncapped, since a cap would strand torch 2.10.1 users."""
    deps = _extras()["windows"]
    specs = [d for d in deps if d.split(";")[0].strip().startswith("xformers")]
    assert len(specs) == 1, f"expected one xformers spec in the windows extra, got {specs}"
    spec = specs[0].split(";")[0].strip()
    assert "xformers @ " not in spec, (
        "the windows extra is the CUDA-agnostic fallback and must stay a version range; "
        "a direct URL here hard-pins torch for every Windows user"
    )


def test_windows_extra_documents_the_cuda_matched_route():
    """The comment block is load bearing: it is the only place a Windows user is told
    that `unsloth[windows]` cannot pick a CUDA-matched wheel and `unsloth[cu130-torch2100]`
    can. Losing it is how this regressed the first time."""
    text = PYPROJECT.read_text(encoding = "utf-8")
    header = text.split("\nwindows = [", 1)[0]
    assert "unsloth[cu130-torch2100]" in header
    assert "unsloth[cu128-torch2100]" in header


def test_xformers_matrix_agrees_with_wheel_utils():
    """One matrix, three consumers (pyproject, wheel_utils, install.ps1). Drift here is
    exactly the bug: a runtime resolver that disagrees with the packaged pin."""
    source = (REPO_ROOT / "studio" / "backend" / "utils" / "wheel_utils.py").read_text(
        encoding = "utf-8"
    )
    body = re.search(
        r"_XFORMERS_WHEEL_VERSIONS[^=]*=\s*\{(.*?)^\}", source, re.DOTALL | re.MULTILINE
    )
    assert body, "could not find _XFORMERS_WHEEL_VERSIONS in wheel_utils.py"
    for (family, torch_tag), version in XFORMERS_WHEEL_MATRIX.items():
        release = f"{torch_tag[0]}.{torch_tag[1:-1]}.{torch_tag[-1]}"  # "2100" -> "2.10.0"
        row = re.search(rf'^\s*"{re.escape(release)}":\s*\{{(.*?)\}}', body.group(1), re.MULTILINE)
        assert row, f"wheel_utils has no row for torch {release}"
        assert f'"{family}": "{version}"' in row.group(1), (
            f"wheel_utils torch {release} row must map {family} -> {version}, got "
            f"{row.group(1)!r}"
        )


def test_install_ps1_matrix_agrees_with_pyproject():
    source = (REPO_ROOT / "install.ps1").read_text(encoding = "utf-8")
    body = re.search(
        r"\$script:XformersWheelVersions\s*=\s*@\{(.*?)^\s*\}", source, re.DOTALL | re.MULTILINE
    )
    assert body, "could not find $script:XformersWheelVersions in install.ps1"
    for (family, torch_tag), version in XFORMERS_WHEEL_MATRIX.items():
        release = f"{torch_tag[0]}.{torch_tag[1:-1]}.{torch_tag[-1]}"
        row = re.search(
            rf'^\s*"{re.escape(release)}"\s*=\s*@\{{(.*?)\}}', body.group(1), re.MULTILINE
        )
        assert row, f"install.ps1 has no row for torch {release}"
        assert f'"{family}" = "{version}"' in row.group(1), (
            f"install.ps1 torch {release} row must map {family} -> {version}, got "
            f"{row.group(1)!r}"
        )

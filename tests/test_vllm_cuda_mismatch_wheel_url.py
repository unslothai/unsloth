# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""The CUDA-mismatch hint used to hand-build a wheel filename as
``vllm-{v}+cu{system_cuda}-cp38-abi3-manylinux_2_35_{arch}.whl``. No vLLM release
has ever published a ``+cu128`` asset and the manylinux tag moved
manylinux1 -> 2_31 -> 2_35 -> 2_34 -> 2_24 -> 2_28, so on a CUDA 12.8 box the hint
was a 404 (the same broken pattern as vllm-project/vllm#37847).

Every URL the hint prints is checked against ``_RELEASE_ASSETS`` below, a snapshot of
the real GitHub release assets (``gh api repos/vllm-project/vllm/releases/tags/vX -q
'.assets[].name'``) that is independent of the table in ``import_fixes.py``. GPU-free.
"""

from __future__ import annotations

import os
import re
import urllib.request

import pytest

from unsloth.import_fixes import (
    _get_vllm_cuda_mismatch_message,
    _get_vllm_wheel_url,
    _VLLM_WHEEL_ASSETS,
)


# Real CUDA wheel assets per release (the ``+cpu`` and macOS assets are dropped).
_RELEASE_ASSETS = {
    "0.11.0": (
        "vllm-0.11.0+cu129-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.0-cp38-abi3-manylinux2014_aarch64.whl",
    ),
    "0.11.1": (
        "vllm-0.11.1+cu129-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.1+cu130-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.1-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.1-cp38-abi3-manylinux2014_aarch64.whl",
    ),
    "0.11.2": (
        "vllm-0.11.2+cu129-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.2+cu130-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.2-cp38-abi3-manylinux1_x86_64.whl",
        "vllm-0.11.2-cp38-abi3-manylinux2014_aarch64.whl",
    ),
    "0.12.0": (
        "vllm-0.12.0+cu130-cp38-abi3-manylinux_2_31_x86_64.whl",
        "vllm-0.12.0-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.12.0-cp38-abi3-manylinux_2_31_x86_64.whl",
    ),
    "0.13.0": (
        "vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.13.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl",
        "vllm-0.13.0-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl",
    ),
    "0.14.1": (
        "vllm-0.14.1+cu130-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.14.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl",
        "vllm-0.14.1-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.14.1-cp38-abi3-manylinux_2_31_x86_64.whl",
    ),
    "0.17.1": (
        "vllm-0.17.1+cu130-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.17.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl",
        "vllm-0.17.1-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.17.1-cp38-abi3-manylinux_2_31_x86_64.whl",
    ),
    "0.18.0": (
        "vllm-0.18.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.18.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl",
        "vllm-0.18.0-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.18.0-cp38-abi3-manylinux_2_31_x86_64.whl",
    ),
    "0.19.1": (
        "vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.19.1+cu130-cp38-abi3-manylinux_2_35_x86_64.whl",
        "vllm-0.19.1-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl",
    ),
    "0.20.0": (
        "vllm-0.20.0+cu129-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.20.0+cu129-cp38-abi3-manylinux_2_31_x86_64.whl",
        "vllm-0.20.0-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.20.0-cp38-abi3-manylinux_2_35_x86_64.whl",
    ),
    "0.20.2": (
        "vllm-0.20.2+cu129-cp38-abi3-manylinux_2_31_aarch64.whl",
        "vllm-0.20.2+cu129-cp38-abi3-manylinux_2_31_x86_64.whl",
        "vllm-0.20.2-cp38-abi3-manylinux_2_35_aarch64.whl",
        "vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl",
    ),
    "0.21.0": (
        "vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_aarch64.whl",
        "vllm-0.21.0+cu129-cp38-abi3-manylinux_2_34_x86_64.whl",
        "vllm-0.21.0-cp38-abi3-manylinux_2_24_aarch64.whl",
        "vllm-0.21.0-cp38-abi3-manylinux_2_24_x86_64.whl",
    ),
    "0.22.0": (
        "vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.22.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
        "vllm-0.22.0-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl",
    ),
    "0.23.0": (
        "vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
        "vllm-0.23.0-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.23.0-cp38-abi3-manylinux_2_28_x86_64.whl",
    ),
    "0.25.1": (
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
        "vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl",
    ),
    "0.26.0": (
        "vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
        "vllm-0.26.0-cp38-abi3-manylinux_2_28_aarch64.whl",
        "vllm-0.26.0-cp38-abi3-manylinux_2_28_x86_64.whl",
    ),
}

# Which CUDA major each snapshot asset is built for.
# the unsuffixed default wheel is CUDA 13 from 0.20.0 on ("CUDA 13.0 default" in the v0.20.0 release notes) and CUDA 12
# before that.
_LOCAL_TAG_RE = re.compile(r"^vllm-[0-9.]+\+cu(\d\d)\d")

_ARCHES = ("x86_64", "aarch64")
_CUDA_MAJORS = (12, 13)


def _cuda_major_of(asset, version):
    match = _LOCAL_TAG_RE.match(asset)
    if match:
        return int(match.group(1))
    from packaging.version import Version

    return 13 if Version(version) >= Version("0.20.0") else 12


def _assets_for(version, cuda_major, arch):
    return {
        asset
        for asset in _RELEASE_ASSETS[version]
        if asset.endswith(f"_{arch}.whl") and _cuda_major_of(asset, version) == cuda_major
    }


@pytest.mark.parametrize("version", sorted(_RELEASE_ASSETS))
@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("cuda_major", _CUDA_MAJORS)
def test_wheel_url_names_a_published_asset(version, arch, cuda_major):
    """Any URL we print must be a file that actually exists in that release."""
    url = _get_vllm_wheel_url(version, cuda_major, arch)
    published = _assets_for(version, cuda_major, arch)
    if url is None:
        return  # Not naming a wheel is always allowed; naming a wrong one is not.
    assert url.startswith(
        f"https://github.com/vllm-project/vllm/releases/download/v{version}/"
    ), url
    filename = url.rsplit("/", 1)[-1]
    assert filename in published, (
        f"vLLM {version} CUDA {cuda_major} {arch}: {filename} is not a published "
        f"release asset. Published: {sorted(published) or 'none'}"
    )


@pytest.mark.parametrize("version", sorted(_RELEASE_ASSETS))
def test_cuda12_wheel_is_resolved_for_every_known_release(version):
    """The table must actually resolve, or the check above passes vacuously."""
    url = _get_vllm_wheel_url(version, 12, "x86_64")
    assert url is not None, f"no CUDA 12 x86_64 wheel resolved for vLLM {version}"
    assert url.endswith(".whl")


def test_no_release_ever_published_a_cu128_wheel():
    """The old code built '+cu128' from torch.version.cuda; that asset never existed."""
    for version, assets in _RELEASE_ASSETS.items():
        for asset in assets:
            assert "+cu128" not in asset, (version, asset)


def _mismatch_message(
    monkeypatch,
    *,
    cuda = "12.8",
    vllm = "0.23.0",
    machine = "x86_64",
    system = "Linux",
    soname = 13,
):
    import platform as _platform
    import unsloth.import_fixes as import_fixes

    monkeypatch.setattr(_platform, "machine", lambda: machine)
    monkeypatch.setattr(_platform, "system", lambda: system)
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: vllm)

    class _FakeTorchVersion:
        pass

    _FakeTorchVersion.cuda = cuda
    fake_torch = type("torch", (), {"version": _FakeTorchVersion})
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    error = ImportError(
        f"libcudart.so.{soname}: cannot open shared object file: No such file or directory"
    )
    return _get_vllm_cuda_mismatch_message(error)


def test_message_recommends_the_real_cu129_wheel_on_a_cuda12_system(monkeypatch):
    message = _mismatch_message(monkeypatch)
    assert (
        "https://github.com/vllm-project/vllm/releases/download/v0.23.0/"
        "vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" in message
    ), message
    assert "cu128" not in message
    assert "manylinux_2_35" not in message


def test_message_is_arch_correct_on_aarch64(monkeypatch):
    message = _mismatch_message(monkeypatch, machine = "aarch64")
    assert "manylinux_2_28_aarch64.whl" in message, message
    assert "x86_64" not in message


def test_message_recommends_the_default_wheel_on_a_cuda13_system(monkeypatch):
    """Reverse direction: CUDA 13 host, vLLM built for CUDA 12."""
    message = _mismatch_message(monkeypatch, cuda = "13.0", soname = 12)
    assert "vllm-0.23.0-cp38-abi3-manylinux_2_28_x86_64.whl" in message, message
    assert "+cu" not in message


def test_older_release_uses_its_own_manylinux_tag(monkeypatch):
    message = _mismatch_message(monkeypatch, vllm = "0.18.0")
    assert "vllm-0.18.0-cp38-abi3-manylinux_2_31_x86_64.whl" in message, message


def test_unmapped_newer_release_points_at_the_release_page(monkeypatch):
    message = _mismatch_message(monkeypatch, vllm = "0.99.0")
    assert ".whl" not in message, message
    assert "https://github.com/vllm-project/vllm/releases/tag/v0.99.0" in message
    assert "+cu129" in message  # still says which variant to pick


def test_non_release_version_never_fabricates_a_filename(monkeypatch):
    message = _mismatch_message(monkeypatch, vllm = "0.24.0rc1")
    assert ".whl" not in message, message
    assert "https://github.com/vllm-project/vllm/releases" in message


@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Darwin", "arm64")])
def test_non_linux_never_recommends_a_manylinux_wheel(monkeypatch, system, machine):
    message = _mismatch_message(monkeypatch, system = system, machine = machine)
    assert "manylinux" not in message, message
    assert ".whl" not in message, message


def test_matching_cuda_is_not_reported_as_a_mismatch(monkeypatch):
    assert _mismatch_message(monkeypatch, cuda = "12.8", soname = 12) is None


def test_unrelated_error_is_not_reported_as_a_mismatch(monkeypatch):
    import unsloth.import_fixes as import_fixes
    assert (
        import_fixes._get_vllm_cuda_mismatch_message(
            ImportError("vllm._C: undefined symbol: _ZN3c108ListType3ofTsEv")
        )
        is None
    )


def test_table_ranges_are_ordered_and_disjoint():
    from packaging.version import Version
    previous_high = None
    for low, high, by_cuda in _VLLM_WHEEL_ASSETS:
        assert Version(low) <= Version(high), (low, high)
        if previous_high is not None:
            assert Version(previous_high) < Version(low), (previous_high, low)
        previous_high = high
        assert by_cuda, (low, high)


@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_NETWORK", "0") not in ("1", "true", "True"),
    reason = "set UNSLOTH_TEST_NETWORK=1 to check the URLs against GitHub",
)
@pytest.mark.parametrize("version", sorted(_RELEASE_ASSETS))
def test_wheel_urls_resolve_live(version):
    for cuda_major in _CUDA_MAJORS:
        for arch in _ARCHES:
            url = _get_vllm_wheel_url(version, cuda_major, arch)
            if url is None:
                continue
            request = urllib.request.Request(url, method = "HEAD")
            with urllib.request.urlopen(request, timeout = 30) as response:
                assert response.status == 200, (url, response.status)

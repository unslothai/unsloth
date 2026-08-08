# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""wheel_utils must resolve a CUDA-matched xFormers wheel, on Windows too.

The platform-tag helper returned None for Windows, so nothing in the backend could
resolve a Windows wheel URL at all -- which is why the on-demand xFormers install fell
back to an unpinned ``pip install xformers`` and landed the PyPI CUDA-12.8 build next to
a cu130 torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import wheel_utils  # noqa: E402


def _env(**overrides) -> dict[str, str]:
    env = {
        "python_tag": "cp313",
        "torch_mm": "2.10",
        "torch_version": "2.10.0+cu130",
        "cuda_version": "13.0",
        "cuda_major": "13",
        "hip_version": "",
        "cxx11abi": "TRUE",
        "platform_tag": "win_amd64",
    }
    env.update(overrides)
    return env


class TestWheelPlatformTag:
    def test_windows_x64_emits_win_amd64(self, monkeypatch):
        monkeypatch.setattr(wheel_utils.sys, "platform", "win32")
        monkeypatch.setattr(wheel_utils.platform, "machine", lambda: "AMD64")
        assert wheel_utils.wheel_platform_tag() == "win_amd64"

    def test_windows_on_arm_has_no_wheel(self, monkeypatch):
        """No CUDA and no win_arm64 asset on any index -- must stay None."""
        monkeypatch.setattr(wheel_utils.sys, "platform", "win32")
        monkeypatch.setattr(wheel_utils.platform, "machine", lambda: "ARM64")
        assert wheel_utils.wheel_platform_tag() is None

    @pytest.mark.parametrize(
        ("machine", "expected"),
        [("x86_64", "linux_x86_64"), ("aarch64", "linux_aarch64")],
    )
    def test_linux_tags_are_unchanged(self, monkeypatch, machine, expected):
        monkeypatch.setattr(wheel_utils.sys, "platform", "linux")
        monkeypatch.setattr(wheel_utils.platform, "machine", lambda: machine)
        assert wheel_utils.wheel_platform_tag() == expected

    def test_macos_still_has_no_wheel(self, monkeypatch):
        monkeypatch.setattr(wheel_utils.sys, "platform", "darwin")
        monkeypatch.setattr(wheel_utils.platform, "machine", lambda: "arm64")
        assert wheel_utils.wheel_platform_tag() is None

    def test_probe_stays_linux_only_by_default(self, monkeypatch):
        """flash-attn / causal-conv1d / mamba-ssm publish no win_amd64 assets, so the
        existing callers must keep getting None on Windows even though the platform tag
        now resolves. Guarded here because the probe shells out otherwise."""
        monkeypatch.setattr(wheel_utils, "wheel_platform_tag", lambda: "win_amd64")
        assert wheel_utils.probe_torch_wheel_env() is None
        assert wheel_utils.probe_torch_wheel_env(timeout = 5) is None


class TestXformersCudaFamily:
    @pytest.mark.parametrize(
        ("cuda_version", "expected"),
        [
            ("13.0", "cu130"),
            ("12.8", "cu128"),
            ("12.6", "cu126"),
            ("12.9", "cu129"),
            ("11.8", "cu118"),
        ],
    )
    def test_maps_torch_cuda_to_the_index_leaf(self, cuda_version, expected):
        assert wheel_utils.xformers_cuda_family(cuda_version) == expected

    @pytest.mark.parametrize("value", ["", None, "rocm6.4", "abc"])
    def test_non_cuda_builds_have_no_family(self, value):
        assert wheel_utils.xformers_cuda_family(value) is None


class TestXformersWheelUrl:
    @pytest.mark.parametrize(
        ("torch_version", "cuda_version", "expected_family", "expected_version"),
        [
            ("2.10.0+cu130", "13.0", "cu130", "0.0.34"),
            ("2.10.0+cu128", "12.8", "cu128", "0.0.34"),
            ("2.10.0+cu126", "12.6", "cu126", "0.0.34"),
            ("2.9.1+cu130", "13.0", "cu130", "0.0.33.post2"),
            ("2.9.0+cu128", "12.8", "cu128", "0.0.33.post1"),
            ("2.8.0+cu129", "12.9", "cu129", "0.0.32.post2"),
        ],
    )
    def test_windows_url_matches_the_resident_cuda_build(
        self, torch_version, cuda_version, expected_family, expected_version
    ):
        url = wheel_utils.xformers_wheel_url(
            _env(torch_version = torch_version, cuda_version = cuda_version)
        )
        assert url == (
            f"https://download.pytorch.org/whl/{expected_family}"
            f"/xformers-{expected_version}-cp39-abi3-win_amd64.whl"
        )

    def test_linux_keeps_its_own_platform_leaf(self):
        url = wheel_utils.xformers_wheel_url(_env(platform_tag = "linux_x86_64"))
        assert url == (
            "https://download.pytorch.org/whl/cu130"
            "/xformers-0.0.34-cp39-abi3-manylinux_2_28_x86_64.whl"
        )

    def test_url_is_interpreter_independent(self):
        """0.0.31..0.0.34 ship a single cp39-abi3 wheel, so the interpreter must not leak
        into the URL. (0.0.30 and earlier did ship one wheel per cpXY, which is exactly why
        torch 2.7.0 is not in the matrix.)"""
        urls = {
            wheel_utils.xformers_wheel_url(_env(python_tag = tag))
            for tag in ("cp39", "cp310", "cp311", "cp312", "cp313", "cp314")
        }
        assert len(urls) == 1 and None not in urls

    @pytest.mark.parametrize(
        ("torch_version", "cuda_version", "why"),
        [
            ("2.8.0+cu130", "13.0", "no cu130 xFormers build exists for torch 2.8"),
            ("2.7.0+cu128", "12.8", "xFormers 0.0.30 predates abi3 and stops at cp312"),
            ("2.9.0+cu118", "11.8", "cu118 stops before the abi3 switch, so its wheels "
                                    "are per-interpreter and unnameable here"),
            ("2.10.0+cu124", "12.4", "cu124 stopped at xFormers 0.0.29"),
            ("2.11.0+cu130", "13.0", "torch 2.11 has no xFormers wheel on any index"),
            ("2.10.0.dev20260101+cu130", "13.0", "a nightly torch has no matching wheel"),
            ("2.6.0+cu126", "12.6", "below the oldest row"),
        ],
    )
    def test_unmatched_pairs_resolve_to_nothing(self, torch_version, cuda_version, why):
        """The whole point: no match must mean "install nothing", never "install the
        closest thing" -- a neighbouring CUDA family is exactly the reported bug."""
        assert (
            wheel_utils.xformers_wheel_url(
                _env(torch_version = torch_version, cuda_version = cuda_version)
            )
            is None
        ), why

    @pytest.mark.parametrize(
        "overrides",
        [
            {"cuda_version": "", "cuda_major": ""},  # CPU / ROCm torch
            {"platform_tag": "linux_aarch64"},  # no aarch64 xFormers wheels
            {"platform_tag": ""},
            {"torch_version": ""},
        ],
    )
    def test_missing_inputs_resolve_to_nothing(self, overrides):
        assert wheel_utils.xformers_wheel_url(_env(**overrides)) is None

    def test_none_env_is_tolerated(self):
        assert wheel_utils.xformers_wheel_url(None) is None


def test_flash_attn_resolution_is_untouched():
    """direct_wheel_url / flash_attn_wheel_url share the env dict; the two new keys must
    not change a single flash-attn URL."""
    env = _env(platform_tag = "linux_x86_64", python_tag = "cp312", cuda_major = "12")
    assert wheel_utils.flash_attn_wheel_url(env) == (
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1"
        "/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )


def test_filename_python_tag_is_ranged_not_open_ended():
    """xFormers has changed its wheel filename tag twice. Guessing the tag for an unreleased
    version is how a resolver starts emitting URLs that 404, so an unknown release must
    resolve to nothing. Verified against the real WHEEL metadata: 0.0.34 is
    cp39-abi3-win_amd64, 0.0.35 is py39-none-win_amd64."""
    assert wheel_utils.xformers_filename_python_tag("0.0.31.post1") == "cp39-abi3"
    assert wheel_utils.xformers_filename_python_tag("0.0.34") == "cp39-abi3"
    assert wheel_utils.xformers_filename_python_tag("0.0.35") == "py39-none"
    for unknown in ("0.0.30", "0.0.29.post3", "0.0.36", "0.1.0", "", "nonsense"):
        assert wheel_utils.xformers_filename_python_tag(unknown) is None, unknown


def test_every_matrix_version_has_a_known_filename_tag():
    for families in wheel_utils._XFORMERS_WHEEL_VERSIONS.values():
        for version in families.values():
            assert wheel_utils.xformers_filename_python_tag(version) is not None, version


@pytest.mark.parametrize("platform_tag", ["win_amd64", "linux_x86_64"])
def test_every_url_the_matrix_can_produce_is_live(platform_tag):
    """The matrix is a hand-maintained list of URLs; the only way to know a row is real is
    to ask. A 404 fails (that row is wrong); a network outage skips, matching
    tests/version_compat/_fetch.py."""
    import urllib.error
    import urllib.request

    cuda_for_family = {"cu126": "12.6", "cu128": "12.8", "cu129": "12.9", "cu130": "13.0"}
    urls: set[str] = set()
    for release, families in wheel_utils._XFORMERS_WHEEL_VERSIONS.items():
        for family, _version in families.items():
            assert family in cuda_for_family, f"unmapped CUDA family {family}"
            for python_tag in ("cp39", "cp310", "cp311", "cp312", "cp313", "cp314"):
                url = wheel_utils.xformers_wheel_url(
                    _env(
                        torch_version = f"{release}+{family}",
                        cuda_version = cuda_for_family[family],
                        python_tag = python_tag,
                        platform_tag = platform_tag,
                    )
                )
                if url is not None:
                    urls.add(url)
    assert urls, "the matrix produced no URLs at all"

    dead = []
    for url in sorted(urls):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method = "HEAD"), timeout = 30):
                pass
        except urllib.error.HTTPError as exc:
            dead.append(f"{exc.code} {url}")
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"download.pytorch.org unreachable: {exc}")
    assert dead == [], "matrix rows pointing at wheels that do not exist:\n" + "\n".join(dead)

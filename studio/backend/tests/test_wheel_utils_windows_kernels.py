# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows kernel wheel resolution: our win_amd64 builds on prebuilt-wheels-cu13, NVIDIA's win_arm64
flash-attn, nothing upstream, and Linux URLs exactly as before."""

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import kernel_install, wheel_utils  # noqa: E402

OURS = "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
NVIDIA = "https://github.com/NVIDIA/pytorch-windows-ci/releases/download/Flash-Attention-for-WoA/"


def env(**overrides):
    base = {
        "platform_tag": "win_amd64",
        "torch_mm": "2.13",
        "cuda_major": "13",
        # What the Windows cu130 torch 2.13 / 2.14 wheels report (measured on windows-2022).
        "cxx11abi": "TRUE",
        "python_tag": "cp313",
        "abi_tag": "cp313",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "hip_version": "",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("torch_mm", ["2.13", "2.14"])
def test_win_amd64_resolves_every_kernel_to_our_release(torch_mm):
    e = env(torch_mm = torch_mm)
    suffix = f"+cu13torch{torch_mm}cxx11abiTRUE-cp313-cp313-win_amd64.whl"
    assert wheel_utils.flash_attn_wheel_url(e) == f"{OURS}flash_attn-2.8.4{suffix}"
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) == f"{OURS}causal_conv1d-1.7.0{suffix}"
    assert kernel_install.MAMBA_SSM.wheel_url(e) == f"{OURS}mamba_ssm-2.3.2.post1{suffix}"


@pytest.mark.parametrize("torch_mm", ["2.10", "2.11", "2.12"])
@pytest.mark.parametrize("python_tag", ["cp311", "cp312", "cp313"])
def test_win_amd64_torch_210_to_212_resolve_to_our_release(torch_mm, python_tag):
    # Existing installs keep their torch minor; upstream's torch2.10 wheels have no Windows asset.
    e = env(torch_mm = torch_mm, python_tag = python_tag, abi_tag = python_tag)
    suffix = f"+cu13torch{torch_mm}cxx11abiTRUE-{python_tag}-{python_tag}-win_amd64.whl"
    assert wheel_utils.flash_attn_wheel_url(e) == f"{OURS}flash_attn-2.8.4{suffix}"
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) == f"{OURS}causal_conv1d-1.7.0{suffix}"
    assert kernel_install.MAMBA_SSM.wheel_url(e) == f"{OURS}mamba_ssm-2.3.2.post1{suffix}"
    # They are win_amd64 only.
    assert (
        wheel_utils.flash_attn_wheel_url(env(torch_mm = torch_mm, platform_tag = "win_arm64")) is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"python_tag": "cp312", "abi_tag": "cp312"},  # 2.13 / 2.14 publish cp313 only
        {"abi_tag": "cp313t"},  # a free-threaded build cannot load the GIL wheel
        {"cxx11abi": "FALSE"},  # not what the Windows cu130 torch reports
        {"torch_mm": "2.12", "python_tag": "cp310", "abi_tag": "cp310"},
        {"torch_mm": "2.11", "python_tag": "cp314", "abi_tag": "cp314"},
        {"torch_mm": "2.11", "abi_tag": "cp313t"},
        {"torch_mm": "2.10", "python_tag": "cp310", "abi_tag": "cp310"},
        {"torch_mm": "2.10", "abi_tag": "cp313t"},
        {"torch_mm": "2.9"},
        {"cuda_major": "12"},
        {"cuda_major": ""},
    ],
)
def test_win_amd64_outside_the_published_cells_resolves_to_nothing(overrides):
    e = env(**overrides)
    assert wheel_utils.flash_attn_wheel_url(e) is None
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) is None
    assert kernel_install.MAMBA_SSM.wheel_url(e) is None


def test_no_windows_url_ever_names_an_upstream_release():
    for torch_mm, cuda, py, abi in itertools.product(
        ["2.4", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.14"],
        ["12", "13"],
        ["cp311", "cp312", "cp313", "cp314"],
        ["TRUE", "FALSE"],
    ):
        for platform_tag in ("win_amd64", "win_arm64"):
            e = env(
                platform_tag = platform_tag,
                torch_mm = torch_mm,
                cuda_major = cuda,
                python_tag = py,
                abi_tag = py,
                cxx11abi = abi,
            )
            for url in (
                wheel_utils.flash_attn_wheel_url(e),
                kernel_install.CAUSAL_CONV1D.wheel_url(e),
                kernel_install.MAMBA_SSM.wheel_url(e),
            ):
                assert url is None or url.startswith((OURS, NVIDIA)), url
                assert url is None or "Dao-AILab" not in url and "state-spaces" not in url


@pytest.mark.parametrize(
    ("python_tag", "abi_tag"),
    [
        ("cp311", "cp311"),
        ("cp312", "cp312"),
        ("cp313", "cp313"),
        ("cp314", "cp314"),
        ("cp314", "cp314t"),
    ],
)
def test_win_arm64_flash_attn_is_nvidias_wheel_with_its_sha256(python_tag, abi_tag):
    e = env(
        platform_tag = "win_arm64",
        torch_mm = "2.14",
        python_tag = python_tag,
        abi_tag = abi_tag,
        torch_version = "2.14.0+cu134",
        cuda_version = "13.4",
    )
    url = wheel_utils.flash_attn_wheel_url(e)
    digest = wheel_utils.NVIDIA_WOA_FLASH_ATTN_SHA256[abi_tag]
    assert url == (
        f"{NVIDIA}flash_attn-2.8.4+cu13torch2.14-{python_tag}-{abi_tag}-win_arm64.whl#sha256={digest}"
    )
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert "#" not in wheel_utils.redact_url_credentials(url)


NVIDIA_TORCH215 = "https://github.com/NVIDIA/pytorch-windows-ci/releases/download/Flash-Attention-for-WoA-torch2.15/"
# The release's asset digests, copied here so a table edit cannot pass by changing both sides.
TORCH215_SHA256 = {
    "cp311": "aea522334852454885af363360da7a38471a29c229ef14dd1dae2e27faa87fed",
    "cp312": "52d7c6d05b6857384336a1356504f34ec7022ec51dffaa822f09b4564fbc35f4",
    "cp313": "a108a9f39c622d21d35a4d94ec7c3095d89718bda7878616ec31534d00e826eb",
    "cp314": "51b34de575067b1fe6044cb924a37ff9c4e05ce60a3fb88914ad4cb2ad1fb7b8",
    "cp314t": "f78dfbbe84e7016da1028480f4b216ec58c274f80f00639e91e2ce78d538cad7",
    "cp315": "90b90880e663289ef935cc1bd959578d18607c9214947a818be0ba82cc706314",
    "cp315t": "37794b35f1c32edf5c5e074b2391fc678f433390589b5efc4761426a137d7d02",
}


@pytest.mark.parametrize("abi_tag", sorted(TORCH215_SHA256))
@pytest.mark.parametrize(
    "torch_version, cuda_version", [("2.15.0+cu135", "13.5"), ("2.15.0+cu134", "13.4")]
)
def test_win_arm64_flash_attn_for_torch_2_15(abi_tag, torch_version, cuda_version):
    python_tag = abi_tag.rstrip("t")
    e = env(
        platform_tag = "win_arm64",
        torch_mm = "2.15",
        python_tag = python_tag,
        abi_tag = abi_tag,
        torch_version = torch_version,
        cuda_version = cuda_version,
    )
    assert wheel_utils.flash_attn_wheel_url(e) == (
        f"{NVIDIA_TORCH215}flash_attn-2.8.4+cu13torch2.15-{python_tag}-{abi_tag}-win_arm64.whl"
        f"#sha256={TORCH215_SHA256[abi_tag]}"
    )
    # Our own kernels are built for torch 2.14 only, so 2.15 gets none of them yet.
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) is None
    assert kernel_install.MAMBA_SSM.wheel_url(e) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"torch_mm": "2.13"},
        {"torch_mm": "2.16", "torch_version": "2.16.0+cu135", "cuda_version": "13.5"},
        {"torch_mm": "2.15", "torch_version": "2.15.0+cu135", "cuda_version": "13.0"},
        {"torch_mm": "2.15", "python_tag": "cp310", "abi_tag": "cp310"},
        {"cuda_major": "12", "cuda_version": "12.8"},
        {"cuda_version": "13.0"},
        {"python_tag": "cp310", "abi_tag": "cp310"},
    ],
)
def test_win_arm64_flash_attn_only_for_the_published_pair(overrides):
    e = woa_env()
    e.update(overrides)
    assert wheel_utils.flash_attn_wheel_url(e) is None


def woa_env(**overrides):
    """What NVIDIA's torch 2.14.0+cu134 reports on Windows on ARM."""
    return env(
        **{
            "platform_tag": "win_arm64",
            "torch_mm": "2.14",
            "torch_version": "2.14.0+cu134",
            "cuda_version": "13.4",
            **overrides,
        }
    )


@pytest.mark.parametrize("python_tag", ["cp311", "cp312"])
def test_win_arm64_unpublished_interpreters_resolve_to_nothing(python_tag):
    # The release carries cp313 only for win_arm64; a cp311 / cp312 URL would be a 404.
    e = woa_env(python_tag = python_tag, abi_tag = python_tag)
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) is None
    assert kernel_install.MAMBA_SSM.wheel_url(e) is None
    # flash-attn stays NVIDIA's, which publishes every interpreter.
    assert wheel_utils.flash_attn_wheel_url(e).startswith(NVIDIA)


@pytest.mark.parametrize("python_tag", ["cp313"])
def test_win_arm64_causal_conv1d_and_mamba_are_ours(python_tag):
    e = woa_env(python_tag = python_tag, abi_tag = python_tag)
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) == (
        f"{OURS}causal_conv1d-1.7.0+cu13torch2.14cxx11abiTRUE-{python_tag}-{python_tag}-win_arm64.whl"
    )
    assert kernel_install.MAMBA_SSM.wheel_url(e) == (
        f"{OURS}mamba_ssm-2.3.2.post1+cu13torch2.14cxx11abiTRUE-{python_tag}-{python_tag}-win_arm64.whl"
    )
    # flash-attn stays NVIDIA's even though the platform is in our table.
    assert wheel_utils.flash_attn_wheel_url(e).startswith(NVIDIA)


@pytest.mark.parametrize(
    "overrides",
    [
        {"torch_mm": "2.13", "torch_version": "2.13.0+cu134"},
        {"cuda_version": "13.0"},
        {"cuda_version": "", "cuda_major": ""},
        {"python_tag": "cp314", "abi_tag": "cp314"},
        {"python_tag": "cp314", "abi_tag": "cp314t"},
    ],
)
def test_win_arm64_kernels_only_for_the_cells_we_build(overrides):
    e = woa_env(**overrides)
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) is None
    assert kernel_install.MAMBA_SSM.wheel_url(e) is None


def test_an_env_without_abi_tag_is_read_as_a_gil_build():
    e = woa_env()
    del e["abi_tag"]
    assert wheel_utils.flash_attn_wheel_url(e).endswith(
        "-cp313-cp313-win_arm64.whl#sha256=" + wheel_utils.NVIDIA_WOA_FLASH_ATTN_SHA256["cp313"]
    )
    e = env()
    del e["abi_tag"]
    assert wheel_utils.flash_attn_wheel_url(e) is not None


@pytest.mark.parametrize(
    ("torch_mm", "python_tag", "expected"),
    [
        (
            "2.13",
            "cp313",
            f"{OURS}flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        ),
        # Built by the workflow on request, but not published: no URL rather than a 404.
        ("2.14", "cp312", None),
        (
            "2.14",
            "cp313",
            f"{OURS}flash_attn-2.8.4+cu13torch2.14cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        ),
        (
            "2.11",
            "cp313",
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/"
            "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        ),
    ],
)
def test_linux_flash_attn_urls_unchanged(torch_mm, python_tag, expected):
    e = env(
        platform_tag = "linux_x86_64",
        cxx11abi = "TRUE",
        torch_mm = torch_mm,
        python_tag = python_tag,
        abi_tag = python_tag,
    )
    assert wheel_utils.flash_attn_wheel_url(e) == expected


def test_linux_free_threaded_is_not_served_the_gil_wheel():
    # The release carries cp313-cp313 only; a free-threaded cp313t interpreter cannot install it.
    e = env(platform_tag = "linux_x86_64", cxx11abi = "TRUE", abi_tag = "cp313t")
    assert wheel_utils.flash_attn_wheel_url(e) is None
    e = env(platform_tag = "linux_x86_64", cxx11abi = "TRUE", abi_tag = "cp313")
    assert wheel_utils.flash_attn_wheel_url(e) == (
        f"{OURS}flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    )


def test_windows_probe_is_opt_in_for_both_windows_tags(monkeypatch):
    calls = []
    monkeypatch.setattr(
        wheel_utils.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 1, ""),
    )
    for tag in ("win_amd64", "win_arm64"):
        monkeypatch.setattr(wheel_utils, "wheel_platform_tag", lambda tag = tag: tag)
        assert wheel_utils.probe_torch_wheel_env(timeout = 5) is None
    assert calls == []  # never even spawned the probe without the opt-in
    wheel_utils.probe_torch_wheel_env(timeout = 5, include_windows = True)
    assert len(calls) == 1


def test_probe_reports_abi_tag():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, sysconfig; print(f'cp{sys.version_info.major}{sys.version_info.minor}' + "
            "('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))",
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    tag = probe.stdout.strip()
    assert tag.startswith(f"cp{sys.version_info.major}{sys.version_info.minor}")
    source = Path(wheel_utils.__file__).read_text(encoding = "utf-8")
    assert (
        "'abi_tag': f'cp{sys.version_info.major}{sys.version_info.minor}' + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else '')"
        in source
    )


def test_cli_resolves_windows_kernels():
    e = env()
    assert kernel_install.resolve_wheel_url("flash_attn", e).endswith("-cp313-cp313-win_amd64.whl")
    assert kernel_install.resolve_wheel_url("causal_conv1d", e).endswith(
        "-cp313-cp313-win_amd64.whl"
    )
    assert kernel_install.resolve_wheel_url("mamba_ssm", e).endswith("-cp313-cp313-win_amd64.whl")
    assert kernel_install.resolve_wheel_url("flash_attn", env(torch_mm = "2.9")) is None
    assert (
        kernel_install.resolve_wheel_url("flash_attn", env(platform_tag = "macosx_14_0_arm64"))
        is None
    )

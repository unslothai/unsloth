# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""linux_aarch64 kernel wheel resolution: upstream's aarch64 wheels stop at torch 2.10, so torch
2.13 / 2.14 cp313 resolve to our builds on prebuilt-wheels-cu13, and every other cell resolves
exactly as it did before."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import kernel_install, wheel_utils  # noqa: E402

OURS = "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"


def env(**overrides):
    base = {
        "platform_tag": "linux_aarch64",
        "torch_mm": "2.13",
        "cuda_major": "13",
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
def test_all_three_resolve_to_our_builds(torch_mm):
    e = env(torch_mm = torch_mm, torch_version = f"{torch_mm}.0+cu130")
    tail = f"+cu13torch{torch_mm}cxx11abiTRUE-cp313-cp313-linux_aarch64.whl"
    assert wheel_utils.flash_attn_wheel_url(e) == f"{OURS}flash_attn-2.8.4{tail}"
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) == f"{OURS}causal_conv1d-1.7.0{tail}"
    assert kernel_install.MAMBA_SSM.wheel_url(e) == f"{OURS}mamba_ssm-2.3.2.post1{tail}"


@pytest.mark.parametrize(
    "overrides",
    [
        {"python_tag": "cp312", "abi_tag": "cp312"},
        {"python_tag": "cp314", "abi_tag": "cp314t"},
        {"cuda_major": "12", "cuda_version": "12.8"},
        {"cxx11abi": "FALSE"},
        {"torch_mm": "2.10", "python_tag": "cp310", "abi_tag": "cp310"},
        {"torch_mm": "2.11", "cuda_major": "12", "cuda_version": "12.8"},
    ],
)
def test_cells_we_do_not_build_never_name_our_release(overrides):
    e = env(**overrides)
    for url in (
        wheel_utils.flash_attn_wheel_url(e),
        kernel_install.CAUSAL_CONV1D.wheel_url(e),
        kernel_install.MAMBA_SSM.wheel_url(e),
    ):
        assert url is None or not url.startswith(OURS), url


@pytest.mark.parametrize("torch_mm", ["2.10", "2.11", "2.12"])
def test_older_torches_take_our_causal_conv1d_and_mamba_ssm_but_upstream_flash_attn(torch_mm):
    e = env(torch_mm = torch_mm, torch_version = f"{torch_mm}.0+cu130")
    tail = f"+cu13torch{torch_mm}cxx11abiTRUE-cp313-cp313-linux_aarch64.whl"
    assert kernel_install.CAUSAL_CONV1D.wheel_url(e) == f"{OURS}causal_conv1d-1.7.0{tail}"
    assert kernel_install.MAMBA_SSM.wheel_url(e) == f"{OURS}mamba_ssm-2.3.2.post1{tail}"
    flash = wheel_utils.flash_attn_wheel_url(e)
    assert flash is None or not flash.startswith(OURS), flash


def test_torch_2_10_cu12_still_resolves_upstream():
    e = env(torch_mm = "2.10", torch_version = "2.10.0+cu128", cuda_major = "12", cuda_version = "12.8")
    assert "Dao-AILab/causal-conv1d" in kernel_install.CAUSAL_CONV1D.wheel_url(e)

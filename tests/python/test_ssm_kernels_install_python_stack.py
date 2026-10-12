# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup installs causal-conv1d and mamba-ssm prebuilt wheels next to flash-attn, so a Mamba /
hybrid model does not wait on a download at load time."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest import mock

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))
sys.path.insert(0, str(STUDIO_DIR / "backend"))

import install_python_stack as ips  # noqa: E402

SOURCE = (STUDIO_DIR / "install_python_stack.py").read_text(encoding = "utf-8")
ENV = {"platform_tag": "linux_x86_64", "torch_mm": "2.13", "torch_version": "2.13.0+cu130"}


def _run(
    *,
    install_kernel,
    env = ENV,
    no_torch = False,
    macos = False,
    disabled = False,
):
    steps, notes = [], []
    with (
        mock.patch.object(ips, "NO_TORCH", no_torch),
        mock.patch.object(ips, "IS_MACOS", macos),
        mock.patch.object(ips, "IS_WINDOWS", False),
        mock.patch.object(ips, "probe_torch_wheel_env", return_value = env) as probe,
        mock.patch.object(ips, "install_kernel", side_effect = install_kernel) as kernel,
        mock.patch.object(ips, "_count_install_action") as counted,
        mock.patch.object(ips, "_step", side_effect = lambda label, value, *_: steps.append(value)),
        mock.patch.object(ips, "_note", side_effect = lambda message, *_: notes.append(message)),
        mock.patch.dict(
            "os.environ", {"UNSLOTH_STUDIO_SKIP_SSM_KERNELS_INSTALL": "1" if disabled else ""}
        ),
    ):
        ips._ensure_ssm_kernels()
    return kernel, probe, counted, steps, notes


def _installs(name, env):
    print(f"Unsloth: installed {name} ({name}-x.whl).")
    return 0


def test_both_kernels_install_for_the_resident_torch():
    kernel, probe, counted, steps, notes = _run(install_kernel = _installs)
    assert [c.args for c in kernel.call_args_list] == [("causal_conv1d", ENV), ("mamba_ssm", ENV)]
    assert probe.call_args.kwargs["include_windows"] is True
    assert counted.call_count == 2
    assert steps == []
    assert notes == [
        "installed causal_conv1d (causal_conv1d-x.whl).",
        "installed mamba_ssm (mamba_ssm-x.whl).",
    ]


def test_an_already_loading_kernel_counts_no_install():
    def loads(name, env):
        print(f"Unsloth: {name} already installed and loads.")
        return 0

    _, _, counted, steps, _ = _run(install_kernel = loads)
    assert counted.call_count == 0 and steps == []


def test_a_failure_warns_and_the_next_kernel_still_runs():
    def first_fails(name, env):
        if name == "causal_conv1d":
            raise OSError("disk full")
        return _installs(name, env)

    kernel, _, _, steps, notes = _run(install_kernel = first_fails)
    assert kernel.call_count == 2
    assert steps == ["Continuing without causal_conv1d"]
    assert "OSError: disk full" in notes


@pytest.mark.parametrize(
    "kwargs",
    [{"no_torch": True}, {"macos": True}, {"disabled": True}, {"env": None}],
    ids = ["no-torch", "macos", "opted-out", "no-cuda-torch"],
)
def test_nothing_is_installed_when_it_cannot_apply(kwargs):
    kernel, _, _, _, _ = _run(install_kernel = _installs, **kwargs)
    assert kernel.call_count == 0


def test_every_flash_attn_site_also_resyncs_the_ssm_kernels():
    calls = [m.start() for m in re.finditer(r"\n\s+_ensure_flash_attn\(\)\n", SOURCE)]
    assert len(calls) == 3
    for start in calls:
        following = SOURCE[start : start + 200]
        assert "_ensure_ssm_kernels()" in following, following
    assert "base_total += 1  # causal-conv1d + mamba-ssm, same gate" in SOURCE

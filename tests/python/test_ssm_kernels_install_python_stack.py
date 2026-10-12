# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup installs causal-conv1d and mamba-ssm prebuilt wheels next to flash-attn, so a Mamba /
hybrid model does not wait on a download at load time."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))
sys.path.insert(0, str(STUDIO_DIR / "backend"))

import install_python_stack as ips  # noqa: E402

SOURCE = (STUDIO_DIR / "install_python_stack.py").read_text(encoding = "utf-8")
ENV = {"platform_tag": "linux_x86_64", "torch_mm": "2.13", "torch_version": "2.13.0+cu130"}
DISTS = {"causal_conv1d": "causal-conv1d", "mamba_ssm": "mamba-ssm"}


class _Host:
    """Which kernels are installed and whether they import; the fake install_kernel edits it."""

    def __init__(
        self,
        installed = (),
        loading = (),
        hangs = (),
        unlaunchable = (),
    ):
        self.installed = set(installed)
        self.loading = set(loading)
        self.hangs = set(hangs)
        self.unlaunchable = set(unlaunchable)
        self.timeouts = []

    def version(self, distribution):
        return "1.0" if distribution in self.installed else None

    def run(self, cmd, *args, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        for name, distribution in DISTS.items():
            if cmd[-1] == ips.KERNELS[name][1]:
                if distribution in self.hangs:
                    raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
                if distribution in self.unlaunchable:
                    raise PermissionError(13, "Access is denied")
                return SimpleNamespace(returncode = 0 if distribution in self.loading else 1)
        return SimpleNamespace(returncode = 0)


def _run(
    host,
    install_kernel,
    *,
    env = ENV,
    no_torch = False,
    macos = False,
    disabled = False,
    offline = False,
):
    steps, notes = [], []
    with (
        mock.patch.object(ips, "NO_TORCH", no_torch),
        mock.patch.object(ips, "IS_MACOS", macos),
        mock.patch.object(ips, "IS_WINDOWS", False),
        mock.patch.object(ips, "_uv_is_offline", return_value = offline),
        mock.patch.object(ips, "probe_torch_wheel_env", return_value = env) as probe,
        mock.patch.object(ips, "install_kernel", side_effect = install_kernel) as kernel,
        mock.patch.object(ips, "_installed_distribution_version", side_effect = host.version),
        mock.patch.object(ips.subprocess, "run", side_effect = host.run),
        mock.patch.object(
            ips, "_uninstall_distribution", side_effect = lambda d: host.installed.discard(d) or True
        ) as uninstall,
        mock.patch.object(ips, "_count_install_action") as counted,
        mock.patch.object(ips, "_step", side_effect = lambda label, value, *_: steps.append(value)),
        mock.patch.object(ips, "_note", side_effect = lambda message, *_: notes.append(message)),
        mock.patch.dict(
            "os.environ", {"UNSLOTH_STUDIO_SKIP_SSM_KERNELS_INSTALL": "1" if disabled else ""}
        ),
    ):
        ips._ensure_ssm_kernels()
    return SimpleNamespace(
        kernel = kernel,
        probe = probe,
        uninstall = uninstall,
        counted = counted,
        steps = steps,
        notes = notes,
    )


def _installer(host):
    def install(name, env, **kwargs):
        print(f"Unsloth: installed {name} ({name}-x.whl).")
        host.installed.add(DISTS[name])
        host.loading.add(DISTS[name])
        return 0

    return install


def test_both_kernels_install_for_the_resident_torch():
    host = _Host()
    r = _run(host, _installer(host))
    assert [c.args for c in r.kernel.call_args_list] == [("causal_conv1d", ENV), ("mamba_ssm", ENV)]
    for call in r.kernel.call_args_list:
        assert call.kwargs["install_without_gpu"] is True
    assert r.probe.call_args.kwargs["include_windows"] is True
    assert r.counted.call_count == 2
    assert r.steps == [] and r.uninstall.call_count == 0
    assert r.notes == [
        "installed causal_conv1d (causal_conv1d-x.whl).",
        "installed mamba_ssm (mamba_ssm-x.whl).",
    ]


def test_a_kernel_that_already_loads_is_not_looked_up_online():
    host = _Host(installed = DISTS.values(), loading = DISTS.values())
    r = _run(host, _installer(host))
    assert r.kernel.call_count == 0 and r.counted.call_count == 0
    assert r.notes == [
        "causal_conv1d already installed and loads.",
        "mamba_ssm already installed and loads.",
    ]


def test_a_resident_build_that_no_longer_loads_is_removed_when_nothing_replaces_it():
    host = _Host(installed = DISTS.values())

    def no_wheel(name, env, **kwargs):
        print(f"Unsloth: no prebuilt {name} for torch 2.15.0; using the torch fallback.")
        return 0

    r = _run(host, no_wheel)
    assert [c.args[0] for c in r.uninstall.call_args_list] == ["causal-conv1d", "mamba-ssm"]
    assert r.steps == [
        "causal_conv1d does not load with this torch; removed it",
        "mamba_ssm does not load with this torch; removed it",
    ]


def test_offline_mode_neither_looks_up_nor_installs_but_keeps_what_loads():
    host = _Host(installed = ["causal-conv1d"], loading = ["causal-conv1d"])
    r = _run(host, _installer(host), offline = True)
    assert r.kernel.call_count == 0 and r.uninstall.call_count == 0
    assert r.notes == [
        "causal_conv1d already installed and loads.",
        "mamba_ssm not installed: UV_OFFLINE is set; run `unsloth install-kernels` online",
    ]


def test_offline_still_removes_a_resident_build_that_no_longer_loads():
    host = _Host(installed = ["mamba-ssm"])
    r = _run(host, _installer(host), offline = True)
    assert r.kernel.call_count == 0
    assert [c.args[0] for c in r.uninstall.call_args_list] == ["mamba-ssm"]


def test_a_probe_that_cannot_be_launched_is_inconclusive_not_fatal():
    host = _Host(installed = ["mamba-ssm"], unlaunchable = ["mamba-ssm"])
    r = _run(host, lambda name, env, **kwargs: 0)
    assert r.kernel.call_count == 2 and r.uninstall.call_count == 0


def test_the_torch_probe_is_bounded_everywhere_and_a_launch_failure_skips_the_step():
    host = _Host()
    r = _run(host, _installer(host))
    assert r.probe.call_args.kwargs["timeout"] == ips._FLASH_ATTN_IMPORT_PROBE_TIMEOUT
    with mock.patch.object(ips, "probe_torch_wheel_env", side_effect = OSError("no fork")):
        with mock.patch.object(ips, "install_kernel") as kernel:
            ips._ensure_ssm_kernels()
    assert kernel.call_count == 0


def test_a_failure_warns_and_the_next_kernel_still_runs():
    host = _Host()
    install = _installer(host)

    def first_fails(name, env, **kwargs):
        if name == "causal_conv1d":
            raise OSError("disk full")
        return install(name, env, **kwargs)

    r = _run(host, first_fails)
    assert r.kernel.call_count == 2
    assert r.steps == ["Continuing without causal_conv1d"]
    assert "OSError: disk full" in r.notes


def test_a_hung_import_probe_is_bounded_and_never_read_as_broken():
    host = _Host(installed = ["mamba-ssm"], hangs = ["mamba-ssm"])
    r = _run(host, lambda name, env, **kwargs: 0)
    assert r.uninstall.call_count == 0
    assert host.timeouts
    assert all(t == ips._FLASH_ATTN_IMPORT_PROBE_TIMEOUT for t in host.timeouts)


def test_each_call_gets_a_fresh_runner_so_a_repaired_torch_is_probed_again():
    runners = []

    def record(name, env, **kwargs):
        runners.append(kwargs["run"])
        return 0

    _run(_Host(), record)
    _run(_Host(), record)
    assert runners[0] is runners[1] and runners[2] is runners[3]
    assert runners[0] is not runners[2]


@pytest.mark.parametrize(
    "kwargs",
    [{"no_torch": True}, {"macos": True}, {"disabled": True}, {"env": None}],
    ids = ["no-torch", "macos", "opted-out", "no-cuda-torch"],
)
def test_nothing_is_installed_when_it_cannot_apply(kwargs):
    host = _Host()
    r = _run(host, _installer(host), **kwargs)
    assert r.kernel.call_count == 0


def test_every_flash_attn_site_also_resyncs_the_ssm_kernels():
    calls = [m.start() for m in re.finditer(r"\n\s+_ensure_flash_attn\(\)\n", SOURCE)]
    assert len(calls) == 3
    for start in calls:
        following = SOURCE[start : start + 200]
        assert "_ensure_ssm_kernels()" in following, following
    assert "base_total += 1  # causal-conv1d + mamba-ssm, same gate" in SOURCE

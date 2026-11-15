# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""flash-attn's Triton AMD backend: installed by setup on Linux ROCm RDNA3/4 only, best effort, and
its import switch defaulted only for a Triton-only build on ROCm torch."""

from __future__ import annotations

import importlib.machinery
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from utils import rocm_flash_attn as rfa

_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"

_ROCM_ENV = {
    "python_tag": "cp312",
    "torch_mm": "2.11",
    "torch_version": "2.11.0+rocm7.13.0",
    "cuda_version": "",
    "cuda_major": "",
    "hip_version": "7.13.0",
    "cxx11abi": "TRUE",
    "platform_tag": "linux_x86_64",
}
_CUDA_ENV = {**_ROCM_ENV, "hip_version": "", "cuda_major": "13", "cuda_version": "13.0"}


@pytest.fixture
def ips(monkeypatch):
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    monkeypatch.delenv("UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL", raising = False)
    monkeypatch.delenv(rfa.SKIP_INSTALL_ENV, raising = False)
    return install_python_stack


class _Host:
    """install_python_stack with every host fact and side effect replaced."""

    def __init__(
        self,
        ips,
        monkeypatch,
        *,
        env,
        gfx = ("gfx1151",),
        windows = False,
        install_ok = True,
        importable_after = True,
        installed_version = None,
    ):
        self.calls: list[tuple] = []
        self.notes: list[str] = []
        self.steps: list[str] = []
        self.removed = 0
        self.wheel_urls: list[str] = []
        self.installed = False
        self.env_during_install: dict = {}
        monkeypatch.setattr(ips, "NO_TORCH", False)
        monkeypatch.setattr(ips, "IS_MACOS", False)
        monkeypatch.setattr(ips, "IS_WINDOWS", windows)
        monkeypatch.setattr(ips.sys, "platform", "win32" if windows else "linux")
        monkeypatch.setattr(ips, "probe_torch_wheel_env", lambda **_: dict(env))
        monkeypatch.setattr(ips, "_detect_amd_gfx_codes", lambda *a, **k: list(gfx))
        monkeypatch.setattr(ips, "_flash_attn_gpu_below_floor", lambda: False)
        monkeypatch.setattr(
            ips, "_flash_attn_importable", lambda: self.installed and importable_after
        )
        monkeypatch.setattr(
            ips,
            "_installed_distribution_version",
            lambda name: "2.8.4" if self.installed else installed_version,
        )
        monkeypatch.setattr(ips, "_remove_stale_flash_attn", self._remove)
        monkeypatch.setattr(ips, "_note", lambda message, *a, **k: self.notes.append(message))
        monkeypatch.setattr(ips, "_step", lambda label, value, *a, **k: self.steps.append(value))
        monkeypatch.setattr(ips, "url_exists", lambda url: self.wheel_urls.append(url) or False)
        self.install_ok = install_ok
        monkeypatch.setattr(ips, "pip_install_try", self._pip_install_try)

    def _remove(self):
        self.removed += 1

    def _pip_install_try(self, label, *args, **kwargs):
        self.calls.append(args)
        if rfa.FLASH_ATTN_TRITON_AMD_SPEC in args:
            self.env_during_install = {
                name: os.environ.get(name) for name in rfa.FLASH_ATTN_TRITON_AMD_BUILD_ENV
            }
            self.installed = self.install_ok
            return self.install_ok
        return True

    @property
    def fa_install_args(self):
        return [args for args in self.calls if rfa.FLASH_ATTN_TRITON_AMD_SPEC in args]


# --- installer ---------------------------------------------------------------------------------


@pytest.mark.parametrize("gfx", ["gfx1151", "gfx1100", "gfx1150", "gfx1201"])
def test_installs_on_linux_rocm_rdna(ips, monkeypatch, gfx):
    host = _Host(ips, monkeypatch, env = _ROCM_ENV, gfx = (gfx,))
    ips._ensure_flash_attn()
    assert len(host.fa_install_args) == 1
    args = host.fa_install_args[0]
    assert "--no-build-isolation" in args and "--no-deps" in args
    # Build deps first, in a separate call.
    assert set(rfa.FLASH_ATTN_TRITON_AMD_BUILD_DEPS) <= set(host.calls[0])
    # The Triton switch and FORCE_BUILD reach the build, then leave the installer's environment.
    assert host.env_during_install == {
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
    }
    assert host.removed == 0
    assert not host.wheel_urls  # no CUDA wheel lookup on ROCm


def test_spec_is_a_pinned_commit_archive():
    assert rfa.FLASH_ATTN_TRITON_AMD_SPEC == (
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/archive/"
        "bbe25bacff9f337568b66e69950e549df9839068.tar.gz"
    )


def test_stale_copy_is_reinstalled(ips, monkeypatch):
    host = _Host(ips, monkeypatch, env = _ROCM_ENV, installed_version = "2.8.4")
    ips._ensure_flash_attn()
    assert "--force-reinstall" in host.fa_install_args[0]


def test_build_env_restored_after_install(ips, monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE")
    monkeypatch.delenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", raising = False)
    _Host(ips, monkeypatch, env = _ROCM_ENV)
    ips._ensure_flash_attn()
    assert os.environ["FLASH_ATTENTION_FORCE_BUILD"] == "FALSE"
    assert "FLASH_ATTENTION_TRITON_AMD_ENABLE" not in os.environ


@pytest.mark.parametrize(
    "case",
    ["unsupported_gfx", "cdna", "no_gfx", "opt_out", "global_opt_out"],
)
def test_skipped(ips, monkeypatch, case):
    gfx = {"unsupported_gfx": ("gfx1030",), "cdna": ("gfx942",), "no_gfx": ()}.get(
        case, ("gfx1151",)
    )
    if case == "opt_out":
        monkeypatch.setenv(rfa.SKIP_INSTALL_ENV, "1")
    if case == "global_opt_out":
        monkeypatch.setenv("UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL", "1")
    host = _Host(ips, monkeypatch, env = _ROCM_ENV, gfx = gfx)
    ips._ensure_flash_attn()
    assert host.calls == []
    assert not host.wheel_urls


def test_skipped_rocm_removes_an_unimportable_copy(ips, monkeypatch):
    # As before: an installed flash-attn that does not import breaks diffusers / xformers imports.
    host = _Host(ips, monkeypatch, env = _ROCM_ENV, gfx = ("gfx1030",), installed_version = "2.8.3")
    ips._ensure_flash_attn()
    assert host.calls == [] and host.removed == 1


def test_nvidia_keeps_the_wheel_route(ips, monkeypatch):
    host = _Host(ips, monkeypatch, env = _CUDA_ENV, gfx = ())
    ips._ensure_flash_attn()
    assert host.fa_install_args == []
    assert host.wheel_urls  # the CUDA prebuilt-wheel lookup ran as before


def test_cpu_torch_installs_nothing(ips, monkeypatch):
    cpu = {**_ROCM_ENV, "hip_version": "", "cuda_major": "", "cuda_version": ""}
    host = _Host(ips, monkeypatch, env = cpu)
    ips._ensure_flash_attn()
    assert host.fa_install_args == []


def test_windows_rocm_never_builds(ips, monkeypatch):
    host = _Host(ips, monkeypatch, env = {**_ROCM_ENV, "platform_tag": "win_amd64"}, windows = True)
    ips._ensure_flash_attn()
    assert host.fa_install_args == []


def test_skip_reason_is_linux_only():
    assert rfa.install_skip_reason(platform = "win32", hip_version = "7.1", gfx_codes = ["gfx1151"])
    assert rfa.install_skip_reason(platform = "darwin", hip_version = "7.1", gfx_codes = ["gfx1151"])
    assert (
        rfa.install_skip_reason(platform = "linux", hip_version = "7.1", gfx_codes = ["gfx1151"]) is None
    )
    # A mixed host installs when any visible card is RDNA3/4.
    assert (
        rfa.install_skip_reason(
            platform = "linux", hip_version = "7.1", gfx_codes = ["gfx942", "gfx1100"]
        )
        is None
    )


@pytest.mark.parametrize("failure", ["install_fails", "not_importable", "deps_fail"])
def test_failure_is_not_fatal(ips, monkeypatch, failure):
    host = _Host(
        ips,
        monkeypatch,
        env = _ROCM_ENV,
        install_ok = failure != "install_fails",
        importable_after = failure != "not_importable",
    )
    if failure == "deps_fail":
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: False)
    ips._ensure_flash_attn()  # never raises or exits
    assert any("Continuing without flash-attn" in step for step in host.steps)
    if failure == "not_importable":
        assert host.removed == 1  # an unimportable copy is never left behind


def test_import_probe_enables_triton_only_on_rocm():
    probe = rfa.IMPORT_PROBE
    assert "flash_attn_2_cuda" in probe and "torch.version, 'hip'" in probe
    assert "setdefault('FLASH_ATTENTION_TRITON_AMD_ENABLE', 'TRUE')" in probe
    assert probe.rstrip().endswith("import flash_attn")
    compile(probe, "<probe>", "exec")


def test_installer_import_probe_uses_it(ips, monkeypatch):
    seen = []
    monkeypatch.setattr(
        ips.subprocess,
        "run",
        lambda cmd, **kwargs: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    assert ips._flash_attn_importable() is True
    assert seen[0][1:] == ["-c", rfa.IMPORT_PROBE]


# --- runtime env default -------------------------------------------------------------------------


def _fake_specs(
    monkeypatch,
    *,
    flash_attn = True,
    extension = False,
    hip = True,
):
    specs = {
        "flash_attn": flash_attn,
        "flash_attn_2_cuda": extension,
    }
    real = rfa._find_spec

    def find_spec(name):
        if name in specs:
            return importlib.machinery.ModuleSpec(name, None) if specs[name] else None
        return real(name)

    monkeypatch.setattr(rfa, "_find_spec", find_spec)
    monkeypatch.setattr(rfa, "_torch_is_rocm", lambda: hip)
    monkeypatch.setattr(rfa.sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "flash_attn", raising = False)


def test_env_default_set_on_rocm_triton_only_build(monkeypatch):
    _fake_specs(monkeypatch)
    env: dict = {}
    assert rfa.default_triton_amd_env(env) is True
    assert env == {"FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE", rfa.DIFFUSION_ONLY_ENV: "1"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hip": False},  # NVIDIA / CPU torch: would misroute a CUDA flash_attn
        {"extension": True},  # CK or CUDA build: has its compiled extension
        {"flash_attn": False},  # not installed
    ],
)
def test_env_default_not_set_elsewhere(monkeypatch, kwargs):
    _fake_specs(monkeypatch, **kwargs)
    env: dict = {}
    assert rfa.default_triton_amd_env(env) is False
    assert env == {}


def test_env_default_not_set_on_windows(monkeypatch):
    _fake_specs(monkeypatch)
    monkeypatch.setattr(rfa.sys, "platform", "win32")
    env: dict = {}
    assert rfa.default_triton_amd_env(env) is False and env == {}


@pytest.mark.parametrize("value", ["TRUE", "FALSE"])
def test_explicit_value_wins_and_opts_training_in(monkeypatch, value):
    _fake_specs(monkeypatch)
    env = {"FLASH_ATTENTION_TRITON_AMD_ENABLE": value}
    assert rfa.default_triton_amd_env(env) is False
    assert env == {"FLASH_ATTENTION_TRITON_AMD_ENABLE": value}  # no diffusion-only marker


def test_explicit_diffusion_only_zero_is_kept(monkeypatch):
    _fake_specs(monkeypatch)
    env = {rfa.DIFFUSION_ONLY_ENV: "0"}
    assert rfa.default_triton_amd_env(env) is True
    assert env[rfa.DIFFUSION_ONLY_ENV] == "0"


def test_torch_is_rocm_reads_version_file_without_importing(monkeypatch, tmp_path):
    for hip_line, expected in (
        ("hip: Optional[str] = '7.13.0-abc'", True),
        ("hip: Optional[str] = None", False),
    ):
        pkg = tmp_path / ("rocm" if expected else "cuda") / "torch"
        pkg.mkdir(parents = True)
        (pkg / "version.py").write_text(f"__version__ = '2.11.0'\n{hip_line}\n")
        spec = importlib.machinery.ModuleSpec("torch", None, is_package = True)
        spec.submodule_search_locations = [str(pkg)]
        monkeypatch.setattr(rfa, "_find_spec", lambda name, spec = spec: spec)
        monkeypatch.delitem(sys.modules, "torch", raising = False)
        assert rfa._torch_is_rocm() is expected


def test_torch_is_rocm_prefers_imported_torch(monkeypatch):
    fake = types.SimpleNamespace(version = types.SimpleNamespace(hip = "7.1"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert rfa._torch_is_rocm() is True
    fake.version.hip = None
    assert rfa._torch_is_rocm() is False


def test_unsloth_gates_llm_flash_attention_on_the_marker():
    # The training side reads the marker in its hip branch, before probing flash_attn.
    source = (Path(__file__).resolve().parents[3] / "unsloth" / "models" / "_utils.py").read_text(
        encoding = "utf-8"
    )
    hip_branch = source.split('elif DEVICE_TYPE == "hip":', 1)[1].split("elif DEVICE_TYPE ==", 1)[0]
    gate = hip_branch.index(f'os.environ.get("{rfa.DIFFUSION_ONLY_ENV}") == "1"')
    assert gate < hip_branch.index('_package_available("flash_attn")')

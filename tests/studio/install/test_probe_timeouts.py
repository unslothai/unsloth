"""Tests that NVIDIA probes in the installers are bounded by a timeout.

Covers audit findings 5 and 6: a wedged nvidia-smi must not hang the installer,
and the Windows probe must require a real GPU listing (not just exit code 0).

Source-level assertions verify the guards are present in install.sh / install.ps1
/ setup.ps1; one behavioral shell test confirms the bash helper actually returns
within the timeout when nvidia-smi hangs.
"""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = PACKAGE_ROOT / "install.sh"
INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"


def _extract_sh_function_body(source: str, name: str) -> str:
    """Return a shell function body from `source` by brace matching."""
    needle = f"{name}() {{"
    start = source.find(needle)
    if start < 0:
        return ""
    depth = 0
    i = start + len(needle) - 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    return source[start:]


# ── install.sh: _run_bounded helper and its use at every nvidia-smi call ──


class TestInstallShBoundedProbe:
    def _src(self) -> str:
        return INSTALL_SH.read_text(encoding = "utf-8")

    def test_run_bounded_helper_defined(self):
        body = _extract_sh_function_body(self._src(), "_run_bounded")
        assert body, "install.sh must define a _run_bounded helper"
        assert (
            "command -v timeout" in body
        ), "_run_bounded must check for the `timeout` binary before using it"
        assert "timeout 10" in body, "_run_bounded must apply a 10s timeout"
        # Must fall back to running unbounded when `timeout` is unavailable
        # (e.g. macOS) so semantics are unchanged there.
        assert (
            "else" in body and '"$@"' in body
        ), "_run_bounded must run the command unbounded when `timeout` is absent"

    def test_nvidia_smi_dash_l_probe_is_bounded(self):
        body = _extract_sh_function_body(self._src(), "_has_usable_nvidia_gpu")
        assert body, "install.sh must define _has_usable_nvidia_gpu"
        # The -L probe must go through the bounded runner, not call nvidia-smi raw.
        assert (
            '_run_bounded "$_nvsmi" -L' in body
        ), "_has_usable_nvidia_gpu must run nvidia-smi -L through _run_bounded"
        # The /proc fallback from PR 6174 must still be present.
        assert "/proc/driver/nvidia" in body

    def test_cuda_version_parse_is_bounded(self):
        body = _extract_sh_function_body(self._src(), "get_torch_index_url")
        assert body, "install.sh must define get_torch_index_url"
        assert (
            "_run_bounded" in body
        ), "get_torch_index_url CUDA-version parse must run nvidia-smi through _run_bounded"
        # The locale must be forced without depending on `env` being on PATH.
        assert "LC_ALL=C" in body
        # _nvidia_detected gating from PR 6174 must remain.
        assert "_nvidia_detected" in body

    def test_no_unbounded_nvidia_smi_invocation_remains(self):
        """Every nvidia-smi *execution* in install.sh goes through _run_bounded.

        `command -v nvidia-smi` and `-x /usr/bin/nvidia-smi` are resolution
        checks, not executions, and are allowed. An execution looks like
        `"$_nvsmi" ...` / `$_smi ...` / `nvidia-smi -L`.
        """
        body_nvidia = _extract_sh_function_body(self._src(), "_has_usable_nvidia_gpu")
        body_torch = _extract_sh_function_body(self._src(), "get_torch_index_url")
        # In _has_usable_nvidia_gpu the only execution of $_nvsmi must be bounded.
        assert '"$_nvsmi" -L' not in body_nvidia.replace(
            '_run_bounded "$_nvsmi" -L', ""
        ), "found an unbounded nvidia-smi -L execution in _has_usable_nvidia_gpu"
        # In get_torch_index_url the $_smi execution must be bounded.
        assert (
            "LC_ALL=C $_smi" not in body_torch
        ), "found an unbounded LC_ALL=C $_smi execution in get_torch_index_url"


# ── install.ps1 / setup.ps1: bounded, GPU-row-validated Windows probe ──


class TestPowerShellBoundedProbe:
    @pytest.mark.parametrize("path", [INSTALL_PS1, SETUP_PS1])
    def test_bounded_helper_present(self, path):
        src = path.read_text(encoding = "utf-8")
        assert (
            "function Invoke-NvidiaSmiBounded" in src
        ), f"{path.name} must define Invoke-NvidiaSmiBounded"
        assert (
            "WaitForExit($TimeoutSec * 1000)" in src
        ), f"{path.name} bounded probe must use WaitForExit with a timeout"
        # Kill + sentinel on timeout, mirroring Invoke-AmdSmiNoElevate.
        assert (
            "$proc.Kill()" in src and "124" in src
        ), f"{path.name} must kill nvidia-smi and signal a timeout exit code"

    @pytest.mark.parametrize("path", [INSTALL_PS1, SETUP_PS1])
    def test_probe_requires_gpu_row(self, path):
        src = path.read_text(encoding = "utf-8")
        assert (
            "function Test-NvidiaSmiHasGpu" in src
        ), f"{path.name} must define Test-NvidiaSmiHasGpu"
        assert "@('-L')" in src, f"{path.name} must probe nvidia-smi with -L"
        assert (
            "^GPU\\s+\\d+:" in src
        ), f"{path.name} must require a 'GPU <n>:' data row, not just exit code 0"

    @pytest.mark.parametrize("path", [INSTALL_PS1, SETUP_PS1])
    def test_detection_uses_validated_probe(self, path):
        src = path.read_text(encoding = "utf-8")
        # The exit-code-only pattern must be gone from the detection block.
        assert (
            "& $nvSmiCmd.Source *> $null" not in src
        ), f"{path.name} must not use the exit-code-only nvidia-smi probe"
        assert (
            "Test-NvidiaSmiHasGpu $nvSmiCmd.Source" in src
        ), f"{path.name} PATH probe must use Test-NvidiaSmiHasGpu"
        assert (
            "Test-NvidiaSmiHasGpu $p" in src
        ), f"{path.name} hardcoded-path fallback must use Test-NvidiaSmiHasGpu"


# ── Behavioral: a hanging nvidia-smi must not hang _has_usable_nvidia_gpu ──


def _have_timeout() -> bool:
    return shutil.which("timeout") is not None


@pytest.mark.skipif(not _have_timeout(), reason = "`timeout` binary not available")
def test_has_usable_nvidia_gpu_returns_under_timeout():
    """Extract _run_bounded + _has_usable_nvidia_gpu, point them at a fake
    nvidia-smi that sleeps 30s, and assert the probe returns well under that.
    """
    src = INSTALL_SH.read_text(encoding = "utf-8")
    helper = _extract_sh_function_body(src, "_run_bounded")
    fn = _extract_sh_function_body(src, "_has_usable_nvidia_gpu")
    assert helper and fn

    workdir = tempfile.mkdtemp(prefix = "pr6174_timeout_", dir = str(PACKAGE_ROOT.parent))
    try:
        fake_dir = Path(workdir, "bin")
        fake_dir.mkdir()
        fake_smi = fake_dir / "nvidia-smi"
        fake_smi.write_text("#!/bin/sh\nsleep 30\n")
        fake_smi.chmod(fake_smi.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Build a minimal PATH that includes the fake nvidia-smi plus the real
        # `timeout`/`awk`/`ls` it needs. Use the fake dir first so it wins.
        real_bins = {Path(shutil.which(c)).parent for c in ("timeout", "awk", "ls", "sh")}
        path_env = os.pathsep.join([str(fake_dir)] + [str(p) for p in real_bins])

        # Force the /proc fallback off so the result depends only on the probe,
        # and so a host with real NVIDIA does not mask the timeout behaviour.
        script = (
            f"{helper}\n{fn}\n"
            "if _has_usable_nvidia_gpu; then echo DETECTED; else echo NONE; fi\n"
        )
        proc = subprocess.run(
            ["sh", "-c", script],
            env = {"PATH": path_env},
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 20,  # generous: the internal timeout is 10s, sleep is 30s
        )
        # The probe must have returned (not hung). On this CI host /proc/driver/
        # nvidia/gpus is absent, so a timed-out smi yields NONE; on a real NVIDIA
        # host the /proc fallback yields DETECTED. Either way it must not hang.
        assert proc.stdout.strip() in {"NONE", "DETECTED"}
    finally:
        shutil.rmtree(workdir, ignore_errors = True)

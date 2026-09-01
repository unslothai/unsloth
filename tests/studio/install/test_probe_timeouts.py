"""NVIDIA installer probes must be timeout-bounded (audit findings 5 and 6): a wedged nvidia-smi
must not hang the installer, and the Windows probe must require a real GPU listing (not exit code 0).

Source-level asserts check the guards in install.sh / install.ps1 / setup.ps1; one behavioral
shell test confirms the bash helper returns within the timeout when nvidia-smi hangs.
"""

import os
import re
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
        # Falls back to unbounded when `timeout` is absent (e.g. macOS), keeping semantics there.
        assert (
            "else" in body and '"$@"' in body
        ), "_run_bounded must run the command unbounded when `timeout` is absent"

    def test_nvidia_smi_dash_l_probe_is_bounded(self):
        body = _extract_sh_function_body(self._src(), "_has_usable_nvidia_gpu")
        assert body, "install.sh must define _has_usable_nvidia_gpu"
        # The -L probe must go through the bounded runner.
        assert (
            '_run_bounded "$_nvsmi" -L' in body
        ), "_has_usable_nvidia_gpu must run nvidia-smi -L through _run_bounded"
        # The /proc fallback from PR 6174 must remain.
        assert "/proc/driver/nvidia" in body

    def test_cuda_version_parse_is_bounded(self):
        body = _extract_sh_function_body(self._src(), "get_torch_index_url")
        assert body, "install.sh must define get_torch_index_url"
        assert (
            "_run_bounded" in body
        ), "get_torch_index_url CUDA-version parse must run nvidia-smi through _run_bounded"
        # Locale forced without depending on `env` being on PATH.
        assert "LC_ALL=C" in body
        # _nvidia_detected gating from PR 6174 must remain.
        assert "_nvidia_detected" in body

    def test_no_unbounded_nvidia_smi_invocation_remains(self):
        """Every nvidia-smi execution goes through _run_bounded (resolution checks are allowed)."""
        body_nvidia = _extract_sh_function_body(self._src(), "_has_usable_nvidia_gpu")
        body_torch = _extract_sh_function_body(self._src(), "get_torch_index_url")
        # The only $_nvsmi execution in _has_usable_nvidia_gpu must be bounded.
        assert '"$_nvsmi" -L' not in body_nvidia.replace(
            '_run_bounded "$_nvsmi" -L', ""
        ), "found an unbounded nvidia-smi -L execution in _has_usable_nvidia_gpu"
        # The $_smi execution in get_torch_index_url must be bounded.
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
        # Kill + sentinel on timeout (mirrors Invoke-AmdSmiNoElevate).
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
        # The exit-code-only probe must be gone from the detection block.
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
    """Point _has_usable_nvidia_gpu at a fake nvidia-smi that sleeps 30s; the probe must return early."""
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

        # PATH with the fake nvidia-smi first plus the real timeout/awk/ls it needs.
        real_bins = {Path(shutil.which(c)).parent for c in ("timeout", "awk", "ls", "sh")}
        path_env = os.pathsep.join([str(fake_dir)] + [str(p) for p in real_bins])

        # Force /proc fallback off so the result depends only on the probe (real NVIDIA host won't mask it).
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
        # The probe must have returned (not hung): NONE without /proc, DETECTED via /proc fallback.
        assert proc.stdout.strip() in {"NONE", "DETECTED"}
    finally:
        shutil.rmtree(workdir, ignore_errors = True)


# Highest-wins runs every ROCm version source on every AMD host, so a hang in any one of
# them hangs the installer: rpm was bounded when that landed, dpkg-query, hipconfig and
# amd-smi were not, and dpkg-query decides on Debian, which ships no rocm-core.

# Matched against the EXECUTION line only, so a helper stays free to resolve the tool some
# other way (a $ROCM_PATH/bin path, a variable) without the assertion going stale.
_ROCM_SOURCE_PROBES = [
    ("_rocm_tag_from_amd_smi", r"amd-smi\s+version\b"),
    ("_rocm_tag_from_hipconfig", r"--version\b"),
    ("_rocm_tag_from_dpkg", r"dpkg-query\s+-W\b"),
    ("_rocm_tag_from_rpm", r"\brpm\s+-q\b"),
]


def _exec_lines(body: str, pattern: str) -> list:
    """Lines that run the probe: matches `pattern` and is not a comment or a lookup guard."""
    return [
        line
        for line in body.splitlines()
        if re.search(pattern, line)
        and not line.lstrip().startswith("#")
        and "command -v" not in line
        and not re.search(r"\[\s*-x\s", line)
    ]


class TestRocmVersionSourcesBounded:
    def _src(self) -> str:
        return INSTALL_SH.read_text(encoding = "utf-8")

    @pytest.mark.parametrize("fn_name,pattern", _ROCM_SOURCE_PROBES)
    def test_source_probe_is_bounded(self, fn_name, pattern):
        body = _extract_sh_function_body(self._src(), fn_name)
        assert body, f"install.sh must define {fn_name}"
        lines = _exec_lines(body, pattern)
        assert lines, f"{fn_name} no longer runs a probe matching {pattern}"
        for line in lines:
            assert "_run_bounded" in line, f"{fn_name} runs an unbounded probe: {line.strip()}"

    @pytest.mark.parametrize("pattern", [r"amd-smi\s+version\b", r"--version\b"])
    def test_radeon_wheel_url_probes_are_bounded(self, pattern):
        body = _extract_sh_function_body(self._src(), "get_radeon_wheel_url")
        assert body, "install.sh must define get_radeon_wheel_url"
        lines = _exec_lines(body, pattern)
        assert lines, f"get_radeon_wheel_url no longer runs a probe matching {pattern}"
        for line in lines:
            assert (
                "_run_bounded" in line
            ), f"get_radeon_wheel_url runs an unbounded probe: {line.strip()}"


@pytest.mark.skipif(not _have_timeout(), reason = "`timeout` binary not available")
@pytest.mark.parametrize("hanging_tool", ["dpkg-query", "hipconfig", "amd-smi"])
def test_detect_rocm_version_tag_returns_when_a_source_hangs(hanging_tool):
    """A wedged version source must make _detect_rocm_version_tag decline, not hang the install."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    parts = [
        _extract_sh_function_body(src, name)
        for name in (
            "_run_bounded",
            "_rocm_tag_from_amd_smi",
            "_rocm_tag_from_version_file",
            "_rocm_tag_from_hipconfig",
            "_rocm_tag_from_dpkg",
            "_rocm_tag_from_rpm",
            "_highest_rocm_tag",
            "_detect_rocm_version_tag",
        )
    ]
    assert all(parts)

    workdir = tempfile.mkdtemp(prefix = "rocm_probe_timeout_")
    try:
        fake_dir = Path(workdir, "bin")
        fake_dir.mkdir()
        hanging = fake_dir / hanging_tool
        hanging.write_text("#!/bin/sh\nsleep 30\n")
        hanging.chmod(hanging.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        real_bins = {
            Path(shutil.which(c)).parent for c in ("timeout", "awk", "grep", "sort", "tr", "sh")
        }
        path_env = os.pathsep.join([str(fake_dir)] + [str(p) for p in real_bins])

        script = "\n".join(parts) + '\n_detect_rocm_version_tag\necho "RC=$?"\n'
        proc = subprocess.run(
            ["sh", "-c", script],
            env = {"PATH": path_env},
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 25,  # internal bound is 10s per probe, the fake sleeps 30s
        )
        assert "RC=0" in proc.stdout
    finally:
        shutil.rmtree(workdir, ignore_errors = True)


# ── The aarch64 + NVIDIA gates added for the Spark/WoA path ──
# These live in the main script bodies rather than inside a helper function, so
# the function-scoped assertions above cannot see them. They run nvidia-smi on
# exactly the hosts whose driver is most likely to be wedged (WSL2 GPU-PV), so
# they must be bounded like every other probe in these scripts.

_NEW_SMI_CALL_SITES = [
    # (file, resolver variable, bounded wrapper that must appear on the line)
    ("studio/setup.sh", "_NVSMI_GATE", "_setup_run_smi"),
    ("install.sh", "_bnb_nvsmi", "_run_bounded"),
]


@pytest.mark.parametrize("rel,resolver,wrapper", _NEW_SMI_CALL_SITES)
def test_aarch64_gate_probes_are_bounded(rel, resolver, wrapper):
    """A wedged nvidia-smi must not hang the aarch64 gates.

    Measured before this assertion existed: a fake nvidia-smi that sleeps 60s
    held the WSL deferral gate for the full 60s. With the wrapper it returns in
    ~10s, matching every other probe in these scripts.
    """
    src = (PACKAGE_ROOT / rel).read_text(encoding = "utf-8")
    offenders = [
        line.strip()
        for line in src.splitlines()
        if re.search(rf'"\${resolver}"\s+-L', line)
        and wrapper not in line
        and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        f"{rel}: nvidia-smi executed without {wrapper}:\n" + "\n".join(offenders)
    )


def test_setup_sh_aarch64_gates_respect_hidden_devices():
    """CUDA_VISIBLE_DEVICES=""/-1 means "no usable NVIDIA" everywhere else here.

    All three aarch64 gates must consult _setup_nvidia_usable, or a user who
    deliberately hid the GPU still gets CUDA llama.cpp provisioned for it.
    """
    src = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    gates = [
        "WSL2 aarch64 + NVIDIA, no nvcc yet: defer to the background CUDA build",
        "Native Linux aarch64 + NVIDIA, no nvcc yet: skip the CPU build too",
        "llama.cpp when the source build above could not",
    ]
    for marker in gates:
        start = src.index(marker)
        cond = src[src.index("\nif ", start) : src.index("; then", start)]
        assert '_setup_nvidia_usable' in cond, f"gate {marker!r} ignores hidden NVIDIA devices"


def test_woa_reroute_does_not_run_wmi_on_x64_installs():
    """Get-HostMachineArch already distinguishes an x64-emulated shell on ARM64.

    A Win32_Processor CIM query on every install is slow and is the kind of WMI
    probe the desktop installers moved away from (#8586), so it may only run when
    all three architecture signals were unreadable.
    """
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "$_hostArch = Get-HostMachineArch" in src
    cim = src.find("Get-CimInstance Win32_Processor")
    if cim >= 0:
        guard = src.find("-ieq 'unknown'")
        assert 0 <= guard < cim, "Win32_Processor probe is not gated behind an unknown arch"


def test_woa_reroute_honours_an_explicit_wheel_index_pin():
    """An explicit pin is authoritative for the ROCm and XPU reroutes beside it.

    Without this the WSL reroute overrides a deliberate cpu / mirror pin on a
    Windows-on-ARM host that happens to have an NVIDIA GPU.
    """
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "-and (-not $TorchIndexPinned) -and ($env:UNSLOTH_NO_WSL_FALLBACK" in src

"""End-to-end GPU-offload spoof: run the real validate_server (real subprocess
+ real HTTP + real log classifier) against a fake llama-server, no GPU needed.

Unlike test_validate_server_gpu_offload.py (which mocks subprocess/urlopen),
this launches an actual process that "starts and serves HTTP 200" while its log
reports CPU-only or GPU offload, reproducing #5807 / #5830 end to end. POSIX
only: validate_server execs the binary path directly, which needs a shebang
wrapper; the Windows equivalent runs in the studio-gpu-offload-smoke workflow
via a .bat shim.
"""

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest


if sys.platform == "win32":
    pytest.skip("POSIX-only (Windows covered by the spoof CI workflow)", allow_module_level = True)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
FAKE_SERVER = Path(__file__).resolve().parent / "fake_llama_server.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_e2e", MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

HostInfo = M.HostInfo


def linux_cuda_host(**overrides):
    defaults = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = (13, 0),
        compute_caps = ["120"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
        has_rocm = False,
    )
    defaults.update(overrides)
    return HostInfo(**defaults)


@pytest.fixture
def fake_server_binary(tmp_path):
    """A `llama-server` that execs the fake server, so validate_server runs it
    exactly as it would a real prebuilt binary."""
    binary = tmp_path / "llama-server"
    binary.write_text(
        "#!/bin/sh\n"
        f'exec "{sys.executable}" "{FAKE_SERVER}" "$@"\n'
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return binary


def _validate(binary, tmp_path, host, install_kind, mode, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", mode)
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF\x00fake")
    M.validate_server(binary, probe, host, tmp_path, install_kind = install_kind)


def test_cpu_only_binary_tagged_cuda_is_rejected(fake_server_binary, tmp_path, monkeypatch):
    # The #5807 case: a binary that serves 200 but loaded the model on CPU.
    with pytest.raises(M.GpuOffloadFailure):
        _validate(fake_server_binary, tmp_path, linux_cuda_host(), "linux-cuda", "cpu", monkeypatch)


def test_offloaded_zero_binary_tagged_cuda_is_rejected(fake_server_binary, tmp_path, monkeypatch):
    with pytest.raises(M.GpuOffloadFailure):
        _validate(
            fake_server_binary, tmp_path, linux_cuda_host(), "linux-cuda", "offloaded_zero", monkeypatch
        )


def test_gpu_binary_tagged_cuda_passes(fake_server_binary, tmp_path, monkeypatch):
    _validate(fake_server_binary, tmp_path, linux_cuda_host(), "linux-cuda", "cuda", monkeypatch)


def test_gpu_buffer_format_passes(fake_server_binary, tmp_path, monkeypatch):
    _validate(fake_server_binary, tmp_path, linux_cuda_host(), "linux-cuda", "cuda_buffer", monkeypatch)


def test_cpu_only_binary_tagged_cpu_is_accepted(fake_server_binary, tmp_path, monkeypatch):
    # A linux-cpu bundle is the intentional fallback; never GPU-gated.
    _validate(fake_server_binary, tmp_path, linux_cuda_host(), "linux-cpu", "cpu", monkeypatch)


def test_no_signal_binary_tagged_cuda_is_accepted(fake_server_binary, tmp_path, monkeypatch):
    # No offload evidence -> conservative: do not reject on no signal.
    _validate(fake_server_binary, tmp_path, linux_cuda_host(), "linux-cuda", "no_signal", monkeypatch)


def test_smoke_test_cli_exit_codes(fake_server_binary, tmp_path, monkeypatch):
    # The contract setup.sh / setup.ps1 depend on, exercised end to end through
    # the real --smoke-test CLI: CPU-only -> 2, GPU -> 0.
    probe = tmp_path / "probe.gguf"
    probe.write_bytes(b"GGUF\x00fake")
    monkeypatch.setattr(M, "detect_host", lambda: linux_cuda_host())

    def run(mode):
        monkeypatch.setenv("FAKE_LLAMA_MODE", mode)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "install_llama_prebuilt.py",
                "--smoke-test", str(fake_server_binary),
                "--probe", str(probe),
                "--install-kind", "linux-cuda",
            ],
        )
        return M.main()

    assert run("cpu") == M.EXIT_FALLBACK
    assert run("cuda") == M.EXIT_SUCCESS

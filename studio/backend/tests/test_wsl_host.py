# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Windows engine host, driven through a fake wsl.exe; the guest runner runs for real."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.inference import engine_install as install
from core.inference import managed_engine
from core.inference import wsl_host

FAKE = Path(__file__).with_name("fake_wsl.py")


@pytest.fixture
def wsl(tmp_path, monkeypatch):
    """Windows host with a fake wsl.exe; guest paths live under tmp_path."""
    log = tmp_path / "wsl.log"
    log.touch()
    exe = tmp_path / "wsl.exe"
    exe.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} "$@"\n')
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_WSL_LOG", str(log))
    monkeypatch.setattr(wsl_host, "active", lambda: True)
    monkeypatch.setattr(wsl_host, "wsl_exe", lambda: str(exe))
    monkeypatch.setattr(wsl_host, "GUEST_ROOT", str(tmp_path / "guest"))
    monkeypatch.setattr(wsl_host, "distro_name", lambda: "Unsloth-Engines-test")
    monkeypatch.setattr(install, "engine_root", lambda: tmp_path / "engines")
    monkeypatch.setattr(wsl_host, "boot_id", lambda: 1000)
    monkeypatch.setattr(install, "_jobs", {})

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return calls


def test_to_guest_path():
    assert wsl_host.to_guest_path(r"C:\Users\A B\model") == "/mnt/c/Users/A B/model"
    assert wsl_host.to_guest_path("D:\\") == "/mnt/d"
    with pytest.raises(ValueError):
        wsl_host.to_guest_path(r"\\server\share\model")


def test_decode_reads_wsl_utf16_and_guest_utf8():
    assert wsl_host.decode("Default Version: 2".encode("utf-16-le")) == "Default Version: 2"
    assert wsl_host.decode("\ufeffok".encode("utf-16-le")) == "ok"
    assert wsl_host.decode("naïve".encode()) == "naïve"


def test_blocker_names_virtualization_fix():
    message = wsl_host.blocker("Error code: Wsl/Service/CreateInstance/CreateVm/0x80370102")
    assert "virtualization" in message
    assert wsl_host.blocker("some other failure") is None


def test_wsl_state_across_a_restart(wsl, monkeypatch):
    monkeypatch.setenv("FAKE_WSL_STATUS", "1")
    assert wsl_host.wsl_state() == "missing"
    wsl_host.write_state(state = "restart_required", boot = 1000)
    # A Studio restart in the same Windows boot is not the restart WSL needs.
    assert wsl_host.wsl_state() == "restart_required"
    monkeypatch.setattr(wsl_host, "boot_id", lambda: 2000)
    assert wsl_host.wsl_state().startswith("blocked: ")
    monkeypatch.setenv("FAKE_WSL_STATUS", "0")
    assert wsl_host.wsl_state() == "ready"
    assert wsl_host.read_state()["state"] == "ready"


def test_missing_wsl_is_not_unsupported(wsl, monkeypatch):
    monkeypatch.setattr(wsl_host, "native_machine", lambda: "x86_64")
    monkeypatch.setattr(wsl_host, "windows_build", lambda: 26100)
    monkeypatch.setattr(install, "_driver_rows", lambda gpu_id, wait = True: [["580.10", "9.0"]])
    assert install.support_reason("vllm") is None
    monkeypatch.setattr(wsl_host, "native_machine", lambda: "arm64")
    assert "x64" in install.support_reason("vllm")
    monkeypatch.setattr(wsl_host, "native_machine", lambda: "x86_64")
    monkeypatch.setattr(wsl_host, "windows_build", lambda: 19041)
    assert "19044" in install.support_reason("vllm")


def test_restart_leaves_a_waiting_job_not_an_interrupted_one(wsl, monkeypatch):
    monkeypatch.setattr(install, "support_reason", lambda *a, **k: None)
    monkeypatch.setattr(install, "_record_manifest", lambda engine: None)

    def prepare(progress = None):
        raise wsl_host.Waiting(
            "Restart Windows to finish installing WSL, then click Install again."
        )

    monkeypatch.setattr(wsl_host, "prepare", prepare)
    install.start_install("vllm")
    deadline = time.monotonic() + 30
    while install._jobs["vllm"]["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    job = install.status("vllm")["job"]
    assert job["state"] == "waiting"
    assert "Restart Windows" in job["message"]
    assert install.status("vllm")["host"] == "wsl"


def test_secrets_cross_through_wslenv_not_argv(wsl, monkeypatch):
    monkeypatch.setenv("WSLENV", "USERPROFILE/p")
    command, env = wsl_host.guest_command(
        ["python", "-V"], env = {"CUDA_VISIBLE_DEVICES": "GPU-1"}, secrets = {"HF_TOKEN": "hf_secret"}
    )
    assert "hf_secret" not in " ".join(command)
    assert command[1:8] == ["-d", "Unsloth-Engines-test", "-u", "root", "--cd", "/root", "--"]
    assert "CUDA_VISIBLE_DEVICES=GPU-1" in command
    assert env["HF_TOKEN"] == "hf_secret"
    assert env["WSLENV"] == "USERPROFILE/p:HF_TOKEN/u"
    subprocess.run(command, env = env, check = True, capture_output = True)
    assert wsl()[-1]["shared"]["HF_TOKEN"] == "hf_secret"


def test_gpus_are_selected_by_uuid(monkeypatch):
    rows = "0, GPU-aaaa\n1, GPU-bbbb\n"
    monkeypatch.setattr(
        wsl_host.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout = rows, stderr = ""),
    )
    assert wsl_host.gpu_uuids([1, 0]) == ["GPU-bbbb", "GPU-aaaa"]
    with pytest.raises(RuntimeError):
        wsl_host.gpu_uuids([2])
    # The guest enumerates the same two cards the other way round.
    monkeypatch.setattr(wsl_host, "guest", lambda *a, **k: "0, GPU-bbbb\n1, GPU-aaaa\n")
    assert wsl_host.guest_gpu_indices([0, 1]) == [1, 0]


def test_installed_wsl_record_never_boots_the_distro(wsl, tmp_path):
    root = tmp_path / "engines" / "vllm"
    root.mkdir(parents = True)
    (root / "active.json").write_text(
        json.dumps({"directory": "env-abc", "host": "wsl", "profile_digest": "x"})
    )
    info = install.installed("vllm")
    assert info["path"] == f"{wsl_host.GUEST_ROOT}/engines/vllm/env-abc"
    assert wsl() == []
    (root / "active.json").write_text(json.dumps({"directory": "../x", "host": "wsl"}))
    assert install.installed("vllm") is None


def _runner(tmp_path) -> Path:
    runner = tmp_path / "run-engine"
    runner.write_text(wsl_host.RUNNER)
    runner.chmod(0o755)
    return runner


def _gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_runner_stops_the_engine_group_when_studio_closes_the_pipe(tmp_path):
    pids = tmp_path / "pids"
    engine = f"import os, subprocess, time; child = subprocess.Popen(['sleep', '300']); open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}'); time.sleep(300)"
    proc = subprocess.Popen(
        [str(_runner(tmp_path)), sys.executable, "-c", engine], stdin = subprocess.PIPE
    )
    for _ in range(100):
        if pids.exists() and pids.read_text():
            break
        time.sleep(0.05)
    engine_pid, grandchild = map(int, pids.read_text().split())
    proc.stdin.close()
    assert proc.wait(timeout = 15) != 0
    # The engine and what it spawned both go: the whole process group is signalled.
    assert _gone(engine_pid, 5) and _gone(grandchild, 5)


def test_runner_survives_idle_and_reports_the_engine_exit_code(tmp_path):
    proc = subprocess.Popen(
        [
            str(_runner(tmp_path)),
            sys.executable,
            "-c",
            "import time, sys; time.sleep(1); sys.exit(7)",
        ],
        stdin = subprocess.PIPE,
    )
    assert proc.wait(timeout = 15) == 7


def test_runner_stops_the_engine_when_studio_is_killed(tmp_path):
    """Studio dying closes its end of the pipe too, so vLLM never outlives its owner."""
    pids = tmp_path / "pids"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; p = subprocess.Popen(sys.argv[1:], stdin = subprocess.PIPE); time.sleep(300)",
            str(_runner(tmp_path)),
            sys.executable,
            "-c",
            f"import os, time; open({str(pids)!r}, 'w').write(str(os.getpid())); time.sleep(300)",
        ],
        start_new_session = True,
    )
    for _ in range(100):
        if pids.exists() and pids.read_text():
            break
        time.sleep(0.05)
    engine_pid = int(pids.read_text())
    os.kill(owner.pid, signal.SIGKILL)
    owner.wait()
    assert _gone(engine_pid, 15)


def test_wsl_launch_command(wsl, monkeypatch):
    # Windows GPU 1 and 3 are guest GPUs 0 and 2.
    monkeypatch.setattr(wsl_host, "guest_gpu_indices", lambda ids: [{1: 0, 3: 2}[i] for i in ids])
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *_: 0.5)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    guest = Path(wsl_host.GUEST_ROOT) / "engines" / "vllm" / "env-abc" / "bin"
    guest.mkdir(parents = True)
    (guest / "python").write_text("")
    (guest / "python").chmod(0o755)
    engine = managed_engine.ManagedEngine("vllm")
    engine.context = 2048
    info = {
        "path": str(guest.parent),
        "host": "wsl",
        "profile_digest": "d",
        "deep_gemm_unloadable": True,
        "cuda_environment": {"CUDA_HOME": "/env/cuda", "CPATH": "/env/include"},
    }
    command, env = engine._wsl_command(
        info,
        {"HF_TOKEN": "hf_secret", "HF_HUB_CACHE": r"C:\Users\a\.cache", "PATH": r"C:\Windows"},
        [1, 3],
        None,
        False,
        "unsloth/Qwen3-0.6B",
        None,
        8123,
    )
    joined = " ".join(command)
    assert command[command.index("--") + 2].endswith("/bin/run-engine")
    assert "CUDA_VISIBLE_DEVICES=0,2" in command and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in command
    assert "VLLM_USE_DEEP_GEMM=0" in command and "CUDA_HOME=/env/cuda" in command
    assert f"HF_HOME={wsl_host.GUEST_ROOT}/hf" in command
    # Windows paths and secrets never reach the guest command line.
    assert "hf_secret" not in joined and "C:\\" not in joined
    assert env["HF_TOKEN"] == "hf_secret" and "HF_TOKEN/u" in env["WSLENV"]
    assert "unsloth/Qwen3-0.6B" in command and "--port" in command


def test_sglang_launcher_is_read_through_mnt(wsl, monkeypatch):
    monkeypatch.setattr(wsl_host, "guest_gpu_indices", lambda ids: [0])
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *_: 0.5)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    guest = Path(wsl_host.GUEST_ROOT) / "engines" / "sglang" / "env-abc" / "bin"
    guest.mkdir(parents = True)
    (guest / "python").write_text("")
    (guest / "python").chmod(0o755)
    engine = managed_engine.ManagedEngine("sglang")
    engine.context = 2048
    command, _ = engine._wsl_command(
        {"path": str(guest.parent), "host": "wsl"}, {}, [0], None, False, "m", None, 8123
    )
    assert "/mnt/c/sglang_server.py" in command
    assert str(Path(managed_engine.__file__).with_name("sglang_server.py")) not in command


def test_linux_launch_is_unchanged_without_wsl(monkeypatch):
    assert wsl_host.active() is (sys.platform == "win32")
    assert install._host_status() == {"host": "local"}

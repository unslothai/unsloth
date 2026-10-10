# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Windows engine host, driven through a fake wsl.exe; the guest runner runs for real."""

import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from core.inference import engine_install as install
from core.inference import managed_engine
from core.inference import wsl_host

FAKE = Path(__file__).with_name("fake_wsl.py")


@pytest.fixture
def wsl(tmp_path, monkeypatch):
    """Windows host with a fake wsl.exe; guest paths live under tmp_path."""
    if sys.platform == "win32":
        pytest.skip("the fake wsl.exe is a POSIX shell script")
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

    def prepare(
        progress = None,
        cancel = None,
        platform = "cuda",
    ):
        raise wsl_host.Waiting(
            "Restart Windows to finish installing WSL, then click Install again."
        )

    monkeypatch.setattr(wsl_host, "prepare", prepare)
    install.start_install("vllm")
    deadline = time.monotonic() + 30
    # status() reads the job file, written just after the in-memory state.
    while install.status("vllm")["job"]["state"] == "running" and time.monotonic() < deadline:
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
    assert command[1:8] == ["-d", "Unsloth-Engines-test", "-u", "root", "--cd", "/root", "--exec"]
    assert "CUDA_VISIBLE_DEVICES=GPU-1" in command
    assert env["HF_TOKEN"] == "hf_secret"
    assert env["WSLENV"] == "USERPROFILE/p:HF_TOKEN/u"
    subprocess.run(command, env = env, check = True, capture_output = True)
    assert wsl()[-1]["shared"]["HF_TOKEN"] == "hf_secret"


def test_an_anonymous_load_withholds_a_token_the_user_shares_through_wslenv(wsl, monkeypatch):
    monkeypatch.setenv("WSLENV", "USERPROFILE/p:HF_TOKEN/u")
    monkeypatch.setenv("HF_TOKEN", "hf_ambient")
    command, env = wsl_host.guest_command(["python", "-V"], withhold = ("HF_TOKEN",))
    assert "HF_TOKEN" not in env
    assert env["WSLENV"] == "USERPROFILE/p"


def test_cancel_stops_a_download_and_leaves_no_partial(tmp_path, monkeypatch):
    import contextlib
    import threading

    cancel = threading.Event()
    monkeypatch.setattr(wsl_host, "host_dir", lambda: tmp_path)

    class Response:
        def raise_for_status(self):
            pass

        def iter_bytes(self, size):
            for _ in range(1000):
                cancel.set()
                yield b"x" * 16

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(stream = lambda *a, **k: contextlib.nullcontext(Response())),
    )
    spec = {"url": "https://example.invalid/r", "sha256": "0" * 64, "size": 16000}
    with pytest.raises(RuntimeError, match = "cancelled"):
        wsl_host.download(spec, "rootfs.tar.gz", cancel = cancel)
    assert list((tmp_path / "downloads").iterdir()) == []


def test_localized_utf16_messages_decode_and_still_name_the_blocker():
    message = "请启用虚拟机平台 Windows 功能并确保在 BIOS 中启用虚拟化。错误代码: Wsl/0x80370102"
    assert wsl_host.decode(message.encode("utf-16-le")) == message
    assert "BIOS" in wsl_host.blocker(wsl_host.decode(message.encode("utf-16-le")))
    assert wsl_host.decode("请启用虚拟化".encode("utf-16-le")) == "请启用虚拟化"


def test_concurrent_installs_prepare_the_distro_one_at_a_time(monkeypatch):
    import threading

    inside, peak = [0], [0]

    def ensure_distro(
        progress = None,
        cancel = None,
        platform = "cuda",
    ):
        inside[0] += 1
        peak[0] = max(peak[0], inside[0])
        time.sleep(0.3)
        inside[0] -= 1

    monkeypatch.setattr(wsl_host, "wsl_state", lambda: "ready")
    monkeypatch.setattr(wsl_host, "ensure_distro", ensure_distro)
    threads = [threading.Thread(target = wsl_host.prepare) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 1


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


# The runner is the WSL guest's bash script (setsid, Linux process groups); it never runs elsewhere.
_GUEST_RUNNER = pytest.mark.skipif(sys.platform != "linux", reason = "WSL guest runner is Linux only")


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


@_GUEST_RUNNER
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


@_GUEST_RUNNER
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


@_GUEST_RUNNER
def test_runner_reaps_workers_left_behind_when_the_engine_exits(tmp_path):
    pids = tmp_path / "pids"
    engine = f"import subprocess, sys; child = subprocess.Popen(['sleep', '300']); open({str(pids)!r}, 'w').write(str(child.pid)); sys.exit(3)"
    proc = subprocess.Popen(
        [str(_runner(tmp_path)), sys.executable, "-c", engine], stdin = subprocess.PIPE
    )
    assert proc.wait(timeout = 20) == 3
    assert _gone(int(pids.read_text()), 5)


@_GUEST_RUNNER
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
    (guest / "python").write_text("", encoding = "utf-8")
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
    assert command[command.index("--exec") + 2].endswith("/bin/run-engine")
    assert "CUDA_VISIBLE_DEVICES=0,2" in command and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in command
    assert "LIBRARY_PATH=/usr/lib/wsl/lib" in command
    assert "VLLM_USE_DEEP_GEMM=0" in command and "CUDA_HOME=/env/cuda" in command
    assert f"HF_HOME={wsl_host.GUEST_ROOT}/hf" in command
    # Windows paths and secrets never reach the guest command line.
    assert "hf_secret" not in joined and "C:\\" not in joined
    assert env["HF_TOKEN"] == "hf_secret" and "HF_TOKEN/u" in env["WSLENV"]
    assert "unsloth/Qwen3-0.6B" in command and "--port" in command
    # The engine key rides WSLENV too: on the guest command line any local user could read it.
    assert engine.key not in joined
    assert env["VLLM_API_KEY"] == engine.key and "VLLM_API_KEY/u" in env["WSLENV"]
    # vLLM's route-guarding launcher is a Studio source file the guest reads through /mnt.
    assert "/mnt/c/vllm_server.py" in command


def test_a_logged_in_token_file_reaches_the_guest(wsl, monkeypatch, tmp_path):
    monkeypatch.setattr(wsl_host, "guest_gpu_indices", lambda ids: [0])
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *_: 0.5)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    guest = Path(wsl_host.GUEST_ROOT) / "engines" / "vllm" / "env-abc" / "bin"
    guest.mkdir(parents = True)
    (guest / "python").write_text("", encoding = "utf-8")
    (guest / "python").chmod(0o755)
    (tmp_path / "token").write_text("hf_from_login\n", encoding = "utf-8")
    engine = managed_engine.ManagedEngine("vllm")
    engine.context = 2048
    info = {"path": str(guest.parent), "host": "wsl"}
    # `hf auth login` stores the token under the host HF_HOME, which the guest does not share.
    command, env = engine._wsl_command(
        info, {"HF_HOME": str(tmp_path)}, [0], None, False, "m", None, 8123
    )
    assert env["HF_TOKEN"] == "hf_from_login" and "HF_TOKEN/u" in env["WSLENV"]
    assert "hf_from_login" not in " ".join(command)
    # An explicit token wins, and an anonymous load sends none.
    _, env = engine._wsl_command(
        info,
        {"HF_HOME": str(tmp_path), "HF_TOKEN": "hf_explicit"},
        [0],
        None,
        False,
        "m",
        None,
        8123,
    )
    assert env["HF_TOKEN"] == "hf_explicit"
    _, env = engine._wsl_command(
        info,
        {"HF_HOME": str(tmp_path), "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"},
        [0],
        None,
        False,
        "m",
        None,
        8123,
    )
    assert "HF_TOKEN" not in env or env["HF_TOKEN"] != "hf_from_login"


def test_offline_mode_reaches_the_guest(wsl, monkeypatch):
    monkeypatch.setattr(wsl_host, "guest_gpu_indices", lambda ids: [0])
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *_: 0.5)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    guest = Path(wsl_host.GUEST_ROOT) / "engines" / "vllm" / "env-abc" / "bin"
    guest.mkdir(parents = True)
    (guest / "python").write_text("", encoding = "utf-8")
    (guest / "python").chmod(0o755)
    engine = managed_engine.ManagedEngine("vllm")
    engine.context = 2048
    info = {"path": str(guest.parent), "host": "wsl"}
    command, _ = engine._wsl_command(info, {}, [0], None, False, "m", None, 8123)
    assert "HF_HUB_OFFLINE=1" not in command
    command, _ = engine._wsl_command(
        info, {"TRANSFORMERS_OFFLINE": "1"}, [0], None, False, "m", None, 8123
    )
    assert "HF_HUB_OFFLINE=1" in command


def test_a_failed_unregister_keeps_the_distro_recorded(wsl, monkeypatch):
    monkeypatch.setattr(wsl_host, "distro_ready", lambda: False)
    monkeypatch.setattr(wsl_host, "host_dir", lambda: Path(wsl_host.GUEST_ROOT).parent / "host")
    monkeypatch.setenv("FAKE_WSL_UNREGISTER", "1")
    monkeypatch.setenv("FAKE_WSL_DISTROS", "Ubuntu,Unsloth-Engines-test")
    with pytest.raises(RuntimeError, match = "Could not remove"):
        wsl_host.unregister()
    assert wsl_host.read_state().get("state") != "removed"
    # Already gone: a failing exit code is not an error.
    monkeypatch.setenv("FAKE_WSL_DISTROS", "Ubuntu")
    wsl_host.unregister()
    assert wsl_host.read_state()["state"] == "removed"


def test_an_orphaned_import_directory_is_cleared_before_importing(wsl, monkeypatch, tmp_path):
    host = tmp_path / "host"
    (host / "distro").mkdir(parents = True)
    (host / "distro" / "ext4.vhdx").write_text("half an import")
    monkeypatch.setattr(wsl_host, "host_dir", lambda: host)
    monkeypatch.setattr(wsl_host, "distro_ready", lambda: False)
    monkeypatch.setattr(wsl_host, "download", lambda *a, **k: tmp_path / "rootfs.tar.gz")
    seen = []
    real_run = wsl_host.run

    def run(args, **kwargs):
        if args[:1] == ["--import"]:
            seen.append(sorted(p.name for p in (host / "distro").iterdir()))
            raise KeyboardInterrupt
        return real_run(args, **kwargs)

    monkeypatch.setattr(wsl_host, "run", run)
    with pytest.raises(KeyboardInterrupt):
        wsl_host.ensure_distro()
    assert seen == [[]]


def test_host_probes_find_nvidia_smi_off_path(monkeypatch):
    import utils.hardware.nvidia as nvidia

    monkeypatch.setattr(nvidia, "_nvidia_smi_executable", lambda: r"C:\NVSMI\nvidia-smi.exe")
    argv = []

    def fake_run(args, *a, **k):
        argv.append(args[0])
        return subprocess.CompletedProcess(args, 0, stdout = "0, GPU-aaaa\n", stderr = "")

    monkeypatch.setattr(wsl_host.subprocess, "run", fake_run)
    wsl_host.gpu_uuids([0])
    monkeypatch.setattr(install.subprocess, "run", fake_run)
    install._probe_rows(0)
    from core.inference import engine_adapters

    monkeypatch.setattr(engine_adapters.subprocess, "run", fake_run)
    try:
        engine_adapters.gpu_memory_fraction([0], 1024)
    except Exception:
        pass
    assert argv == [r"C:\NVSMI\nvidia-smi.exe"] * 3


def test_removing_a_wsl_engine_also_drops_its_compile_cache(wsl):
    guest = Path(wsl_host.GUEST_ROOT)
    guest.mkdir(parents = True)
    (guest / "owner.json").write_text("{}", encoding = "utf-8")
    for folder in ("engines/vllm/env-abc", "cache/vllm/k", "engines/sglang", "cache/sglang/k"):
        (guest / folder).mkdir(parents = True)
    install.remove("vllm")
    assert not (guest / "engines" / "vllm").exists() and not (guest / "cache" / "vllm").exists()
    assert (guest / "engines" / "sglang").exists() and (guest / "cache" / "sglang").exists()


def test_sglang_launcher_is_read_through_mnt(wsl, monkeypatch):
    monkeypatch.setattr(wsl_host, "guest_gpu_indices", lambda ids: [0])
    monkeypatch.setattr(managed_engine, "gpu_memory_fraction", lambda *_: 0.5)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    guest = Path(wsl_host.GUEST_ROOT) / "engines" / "sglang" / "env-abc" / "bin"
    guest.mkdir(parents = True)
    (guest / "python").write_text("", encoding = "utf-8")
    (guest / "python").chmod(0o755)
    engine = managed_engine.ManagedEngine("sglang")
    engine.context = 2048
    command, env = engine._wsl_command(
        {"path": str(guest.parent), "host": "wsl"}, {}, [0], None, False, "m", None, 8123
    )
    assert "/mnt/c/sglang_server.py" in command
    assert engine.key not in " ".join(command)
    assert env["UNSLOTH_ENGINE_API_KEY"] == engine.key
    assert "UNSLOTH_ENGINE_API_KEY/u" in env["WSLENV"]
    assert str(Path(managed_engine.__file__).with_name("sglang_server.py")) not in command


def test_linux_launch_is_unchanged_without_wsl(monkeypatch):
    assert wsl_host.active() is (sys.platform == "win32")
    if sys.platform == "win32":
        assert install._host_status()["host"] == "wsl"
    else:
        assert install._host_status() == {"host": "local"}


def test_a_failed_reset_never_deletes_the_disk_of_a_registered_distro(wsl, monkeypatch, tmp_path):
    host = tmp_path / "host"
    (host / "distro").mkdir(parents = True)
    (host / "distro" / "ext4.vhdx").write_text("a live distro's disk")
    monkeypatch.setattr(wsl_host, "host_dir", lambda: host)
    monkeypatch.setattr(wsl_host, "distro_ready", lambda: False)  # e.g. the probe timed out
    monkeypatch.setattr(wsl_host, "download", lambda *a, **k: tmp_path / "rootfs.tar.gz")
    monkeypatch.setenv("FAKE_WSL_UNREGISTER", "1")
    monkeypatch.setenv("FAKE_WSL_DISTROS", "Unsloth-Engines-test")
    with pytest.raises(RuntimeError, match = "Could not reset"):
        wsl_host.ensure_distro()
    assert (host / "distro" / "ext4.vhdx").exists()


def test_removal_waits_for_an_unresponsive_but_registered_distro(wsl, monkeypatch):
    root = install.engine_root() / "vllm"
    root.mkdir(parents = True)
    (root / "active.json").write_text("{}")
    monkeypatch.setattr(wsl_host, "distro_ready", lambda: False)
    monkeypatch.setenv("FAKE_WSL_DISTROS", "Unsloth-Engines-test")
    with pytest.raises(RuntimeError, match = "not responding"):
        install.remove("vllm")
    assert (root / "active.json").exists()
    monkeypatch.setenv("FAKE_WSL_DISTROS", "Ubuntu")
    install.remove("vllm")
    assert not root.exists()


def test_host_gpu_probes_hide_their_console_window(monkeypatch):
    from core.inference import engine_adapters

    seen = []

    def fake_run(args, *a, **k):
        seen.append(k.get("creationflags"))
        return subprocess.CompletedProcess(args, 0, stdout = "0, GPU-aaaa\n", stderr = "")

    hidden = lambda: {"creationflags": 0x08000000}  # noqa: E731
    for module in (install, engine_adapters):
        monkeypatch.setattr(module, "windows_hidden_subprocess_kwargs", hidden)
        monkeypatch.setattr(module.subprocess, "run", fake_run)
    install._probe_rows(0)
    try:
        engine_adapters.gpu_memory_fraction([0], 1024)
    except Exception:
        pass
    assert seen == [0x08000000] * 2


@pytest.mark.parametrize("present", [True, False])
def test_wsl_build_tools_are_provisioned_only_when_missing(present):
    calls = []
    messages = []

    def run_guest(argv):
        calls.append(argv)
        if len(calls) == 1:
            return "UNSLOTH_BUILD_TOOLS_READY" if present else "UNSLOTH_BUILD_TOOLS_MISSING"
        return ""

    wsl_host.ensure_build_tools(run_guest, messages.append)
    assert calls[0][:2] == ["sh", "-c"]
    if present:
        assert len(calls) == 1 and not messages
    else:
        assert messages == ["Installing WSL build tools"]
        assert calls[1:] == [
            ["dpkg", "--configure", "-a"],
            ["apt-get", "update"],
            ["apt-get", "install", "-y", "--no-install-recommends", "build-essential"],
        ]


def test_wsl_build_tool_probe_failure_does_not_install_packages():
    calls = []
    error = RuntimeError("WSL could not start")

    def run_guest(argv):
        calls.append(argv)
        raise error

    with pytest.raises(RuntimeError) as caught:
        wsl_host.ensure_build_tools(run_guest)
    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.parametrize("output", ["", "unexpected WSL output"])
def test_wsl_build_tool_probe_requires_an_explicit_result(output):
    calls = []

    def run_guest(argv):
        calls.append(argv)
        return output

    with pytest.raises(RuntimeError, match = "Could not check WSL build tools"):
        wsl_host.ensure_build_tools(run_guest)
    assert len(calls) == 1


def test_wsl_build_tool_failure_is_reported():
    calls = []

    def run_guest(argv):
        calls.append(argv)
        if len(calls) == 1:
            return "UNSLOTH_BUILD_TOOLS_MISSING"
        raise RuntimeError("package repair failed")

    with pytest.raises(RuntimeError, match = "package repair failed"):
        wsl_host.ensure_build_tools(run_guest)
    assert len(calls) == 2


@_GUEST_RUNNER
@pytest.mark.parametrize("cancel_source", ["event", "file"])
def test_build_tool_cancellation_reaps_the_guest_process_group(
    tmp_path, monkeypatch, cancel_source
):
    import threading

    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    monkeypatch.setattr(install, "_update", lambda *args, **kwargs: None)
    pids = tmp_path / "apt-pids"
    code = (
        "import os, subprocess, time; child = subprocess.Popen(['sleep', '300']); "
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}'); time.sleep(300)"
    )
    cancel = threading.Event()
    calls, errors = [], []

    def run_guest(argv):
        calls.append(argv)
        if argv[0] == "sh":
            return "UNSLOTH_BUILD_TOOLS_MISSING"
        if argv[0] == "dpkg":
            return ""
        return install._run(
            "vllm",
            [str(_runner(tmp_path)), sys.executable, "-I", "-S", "-u", "-c", code],
            cancel,
            stdin_pipe = True,
        )

    def provision():
        try:
            wsl_host.ensure_build_tools(run_guest)
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target = provision)
    thread.start()
    try:
        # Python start-up on a loaded -n 4 runner, not the cancellation, sets this wait.
        deadline = time.monotonic() + 60
        while (
            not (pids.exists() and pids.read_text())
            and thread.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert pids.exists() and pids.read_text(), errors
        apt_pid, child_pid = map(int, pids.read_text().split())
        if cancel_source == "file":
            (tmp_path / "vllm.cancel").touch()
        else:
            cancel.set()
        thread.join(20)
        assert not thread.is_alive()
        assert errors and "cancelled" in str(errors[0]).lower()
        assert _gone(apt_pid, 5) and _gone(child_pid, 5)
        assert [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "build-essential",
        ] not in calls
    finally:
        cancel.set()
        thread.join(20)


def test_cancelled_preparation_does_not_start_a_guest_command(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    (tmp_path / "vllm.cancel").touch()
    started = tmp_path / "started"
    command = [sys.executable, "-c", f"open({str(started)!r}, 'w').close()"]
    with pytest.raises(RuntimeError, match = "cancelled"):
        install._run("vllm", command, threading.Event(), stdin_pipe = True)
    assert not started.exists()


def test_preparation_pipe_close_failure_still_cleans_up(tmp_path, monkeypatch):
    import io
    import threading
    from utils import process_lifetime

    class Stdin:
        def close(self):
            raise OSError("pipe already disconnected")

    class Process:
        pid = 123456
        returncode = 0
        stdin = Stdin()
        stdout = io.StringIO()

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    proc, forgotten = Process(), []
    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    monkeypatch.setattr(process_lifetime, "spawn_on_lifetime_thread", lambda factory: proc)
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(process_lifetime, "forget_pid", forgotten.append)
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: False)
    assert install._run("vllm", ["unused"], threading.Event(), stdin_pipe = True) == ""
    assert forgotten == [proc.pid] and proc.stdout.closed


@pytest.fixture
def amd(wsl, monkeypatch):
    """A Windows host whose Studio PyTorch is a ROCm build."""
    monkeypatch.setattr(install, "gpu_platform", lambda: "rocm")

    def no_nvidia(*_a, **_k):
        raise AssertionError("nvidia-smi is not an AMD tool")

    monkeypatch.setattr(install, "_driver_rows", no_nvidia)
    monkeypatch.setattr(wsl_host, "guest_gpu_indices", no_nvidia)
    return wsl


def test_amd_on_windows_installs_rocm_inside_wsl(amd, monkeypatch):
    from utils.hardware import hardware

    monkeypatch.setattr(wsl_host, "native_machine", lambda: "x86_64")
    monkeypatch.setattr(wsl_host, "windows_build", lambda: 26200)
    inventory = {"devices": [{"vendor": "amd", "name": "AMD Radeon(TM) 8060S Graphics"}]}
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda block = True: inventory)
    # The driver may not name the architecture; the install checks it inside WSL.
    assert install.support_reason("vllm") is None
    inventory["devices"][0]["gfx"] = "gfx1151"
    assert install.support_reason("vllm") is None
    # A card the Windows driver already names is refused before the ROCm download.
    inventory["devices"][0]["gfx"] = "gfx1030"
    assert install.support_reason("vllm", wait = False).endswith("(found gfx1030).")
    # With a supported and an unsupported card, the selected GPU's own target decides.
    inventory["devices"].append(
        {"vendor": "amd", "name": "AMD Radeon RX 7900 XTX", "gfx": "gfx1100"}
    )
    monkeypatch.setattr(install, "_rocm_gpu_arches", lambda: {0: "gfx1030", 1: "gfx1100"})
    assert install.support_reason("vllm", wait = False) is None
    assert install.support_reason("vllm", 1) is None
    assert install.support_reason("vllm", 0).endswith("(found gfx1030).")
    assert (
        install.support_reason("sglang") == "SGLang requires an NVIDIA GPU. Use vLLM on AMD GPUs."
    )
    assert install.profile("vllm")["lock"] == "vllm-linux-rocm723"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_nvidia_wsl_venv_never_gets_the_windows_interpreter(wsl, monkeypatch, engine):
    import threading

    # Studio's Windows Python matching the engine's version must not reach the Linux uv.
    windows_python = r"C:\Unsloth\Scripts\python.exe"
    monkeypatch.setattr(install, "gpu_platform", lambda: "cuda")
    monkeypatch.setattr(install, "_python", lambda engine: sys.version_info[:2])
    monkeypatch.setattr(install.sys, "executable", windows_python)
    monkeypatch.setattr(install, "_record_manifest", lambda engine: None)
    monkeypatch.setattr(wsl_host, "prepare", lambda progress, cancel, platform: None)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    monkeypatch.setattr(wsl_host, "put", lambda path, text, mode = "644": None)
    monkeypatch.setattr(
        wsl_host,
        "guest",
        lambda argv, **_: json.dumps({"cuda_environment": {}, "deep_gemm_unloadable": False})
        if argv[-2].endswith("finalize.py")
        else "",
    )
    commands = []
    monkeypatch.setattr(
        install,
        "_run",
        lambda engine, argv, cancel, env = None, **_: commands.append(argv)
        or "UNSLOTH_BUILD_TOOLS_READY",
    )
    install._install_wsl(engine, threading.Event())
    venv = next(c for c in commands if "venv" in c)
    assert venv[-3:-1] == ["--python", "{}.{}".format(*sys.version_info[:2])]
    assert install._venv_python_args(engine) == ["--python", windows_python]


def test_rocm_repository_key_ships_with_studio():
    # repo.radeon.com serves the key at a mutable URL; Studio's copy is what the distro trusts.
    assert wsl_host._sha256(wsl_host.ROCM_APT_KEY) == wsl_host.ROCM_APT_KEY_SHA256
    assert wsl_host.ROCM_APT_KEY.read_text(encoding = "utf-8").startswith(
        "-----BEGIN PGP PUBLIC KEY BLOCK-----"
    )


@pytest.mark.parametrize(("agents", "supported"), [("gfx1151", True), ("gfx1030", False)])
def test_amd_wsl_install_sets_up_rocm_before_the_engine(
    amd, monkeypatch, tmp_path, agents, supported
):
    import threading

    monkeypatch.setattr(install, "_record_manifest", lambda engine: None)
    prepared, commands = [], []
    monkeypatch.setattr(
        wsl_host, "prepare", lambda progress, cancel, platform: prepared.append(platform)
    )
    monkeypatch.setattr(wsl_host, "download", lambda spec, name, *a: tmp_path / name)
    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    monkeypatch.setattr(wsl_host, "put", lambda path, text, mode = "644": None)

    def guest(
        argv,
        env = None,
        timeout = 600,
        input = None,
    ):
        if argv == ["/opt/rocm/bin/rocminfo"]:
            assert env == {**wsl_host.ROCM_ENVIRONMENT, "HSA_OVERRIDE_GFX_VERSION": "11.5.1"}
            return f"  Name:                    {agents}\n  Name:   amdgcn-amd-amdhsa--{agents}\n"
        if argv[-2].endswith("finalize.py"):
            return json.dumps({"cuda_environment": {}, "deep_gemm_unloadable": False})
        return ""

    monkeypatch.setattr(wsl_host, "guest", guest)
    # Studio's torch reported the overridden target, so the guest has to see the same one.
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
    monkeypatch.setattr(
        install,
        "_run",
        lambda engine, argv, cancel, env = None, **_: commands.append(argv)
        or "UNSLOTH_BUILD_TOOLS_READY",
    )
    if supported:
        install._install_wsl("vllm", threading.Event())
    else:
        with pytest.raises(RuntimeError, match = r"Ryzen AI Max or AI 300 GPU \(found gfx1030\)"):
            install._install_wsl("vllm", threading.Event())
    assert prepared == ["rocm"]
    setup = next(i for i, c in enumerate(commands) if c[-4].endswith("/bin/setup-rocm"))
    assert commands[setup][-3:] == [
        "/mnt/c/rocm.gpg.key",
        "/mnt/c/rocdxg-roct_1.2.2_amd64.deb",
        "/mnt/c/rocdxg-amd-smi-lib_1.2.2_amd64.deb",
    ]
    if not supported:
        assert not any("venv" in c for c in commands)
        return
    venv = next(i for i, c in enumerate(commands) if "venv" in c)
    assert setup < venv and commands[venv][-5:-1] == [
        "--python",
        "3.12",
        "--python-preference",
        "only-managed",
    ]
    sync = next(c for c in commands if "sync" in c)
    assert sync[sync.index("--extra-index-url") + 1] == install.profile("vllm")["index"]
    assert "HSA_OVERRIDE_GFX_VERSION=11.5.1" in sync
    info = json.loads((install.engine_root() / "vllm" / "active.json").read_text())
    assert info["platform"] == "rocm" and info["host"] == "wsl"
    # Recorded, so the next install no longer prices the ROCm download.
    assert wsl_host.summary()["rocm"] == wsl_host.ROCM_RELEASE


def test_amd_wsl_price_includes_rocm_until_it_is_set_up(amd, monkeypatch):
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    sizes = install._compat_file("vllm")["sizes"]
    engine = sum(size or 0 for size in sizes.values())
    rocm = wsl_host.ROCM_APT_BYTES + sum(
        spec["size"] for spec in (wsl_host.ROCDXG, wsl_host.ROCDXG_SMI)
    )
    distro = {"state": "ready", "distro": "Unsloth-Engines-test"}
    monkeypatch.setattr(wsl_host, "summary", lambda: {**distro, "rocm": None})
    assert install.download_bytes("vllm") == engine + rocm
    monkeypatch.setattr(wsl_host, "summary", lambda: {**distro, "rocm": wsl_host.ROCM_RELEASE})
    assert install.download_bytes("vllm") == engine


def test_amd_wsl_launch_uses_one_hip_device_and_the_dxg_bridge(amd, monkeypatch):
    from utils import vram_budget_settings

    monkeypatch.setattr(wsl_host, "to_guest_path", lambda path: "/mnt/c/" + Path(path).name)
    monkeypatch.setattr(vram_budget_settings, "get_vram_budget_fraction", lambda: 0.97)
    measured = []

    def rows(info, env, gpu_ids):
        # The selected GPUs reach the capacity check, which reads them from Windows.
        assert gpu_ids == [0]
        measured.append(env)
        return [(65536.0, 60000.0)]

    monkeypatch.setattr(managed_engine, "_engine_memory_rows", rows)
    guest = Path(wsl_host.GUEST_ROOT) / "engines" / "vllm" / "env-rocm" / "bin"
    guest.mkdir(parents = True)
    (guest / "python").write_text("", encoding = "utf-8")
    (guest / "python").chmod(0o755)
    engine = managed_engine.ManagedEngine("vllm")
    engine.context = 2048
    info = {"path": str(guest.parent), "host": "wsl", "platform": "rocm", "profile_digest": "d"}
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising = False)
    command, _ = engine._wsl_command(info, {}, [0], None, False, "unsloth/Qwen3-0.6B", None, 8123)
    assert not any(part.startswith("HSA_OVERRIDE_GFX_VERSION=") for part in command)
    # Eligibility accepted the target Studio's override presents; the engine must present it too.
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
    command, _ = engine._wsl_command(info, {}, [0], None, False, "unsloth/Qwen3-0.6B", None, 8123)
    assert "HSA_OVERRIDE_GFX_VERSION=11.5.1" in command
    assert "CUDA_VISIBLE_DEVICES=0" in command and "HIP_VISIBLE_DEVICES=0" in command
    assert (
        "HSA_ENABLE_DXG_DETECTION=1" in command and "LD_LIBRARY_PATH=/opt/rocm-wsl/lib" in command
    )
    assert measured and measured[0]["HSA_ENABLE_DXG_DETECTION"] == "1"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.855"


def test_amd_on_windows_loads_on_one_gpu(amd, monkeypatch):
    from models.inference import LoadRequest

    info = {"path": "/env", "version": "0.30.0", "profile_digest": install.profile_digest("vllm")}
    monkeypatch.setattr(managed_engine, "installed", lambda _: info)
    monkeypatch.setattr(managed_engine, "support_reason", lambda *a: None)
    monkeypatch.setattr(managed_engine, "resolve_requested_gpu_ids", lambda ids: list(ids or [0]))
    request = LoadRequest(model_path = "m", engine = "vllm", gpu_ids = [0, 1])
    with pytest.raises(ValueError, match = "one AMD GPU"):
        managed_engine.validate_load("vllm", request)
    assert managed_engine.validate_load("vllm", request.model_copy(update = {"gpu_ids": [0]})) == [0]


def test_an_amd_distro_checks_for_the_dxg_device(wsl, monkeypatch, tmp_path):
    monkeypatch.setattr(wsl_host, "host_dir", lambda: tmp_path / "host")
    monkeypatch.setattr(wsl_host, "distro_ready", lambda: False)
    monkeypatch.setattr(wsl_host, "download", lambda *a, **k: tmp_path / "rootfs.tar.gz")
    monkeypatch.setattr(wsl_host, "put", lambda *a, **k: None)
    probed = []

    def guest(argv, **kwargs):
        probed.append(argv)
        raise RuntimeError("missing")

    monkeypatch.setattr(wsl_host, "guest", guest)
    with pytest.raises(RuntimeError, match = "AMD Adrenalin driver"):
        wsl_host.ensure_distro(platform = "rocm")
    assert probed == [["test", "-e", "/dev/dxg"]]
    with pytest.raises(RuntimeError, match = "NVIDIA Windows driver"):
        wsl_host.ensure_distro()
    assert probed[-1] == ["test", "-e", "/usr/lib/wsl/lib/libcuda.so"]


def test_rocm_setup_script_is_idempotent_and_pinned():
    script = wsl_host.ROCM_SETUP
    assert "https://repo.radeon.com/rocm/apt/7.2.1 noble main" in script
    assert "signed-by=/etc/apt/keyrings/rocm.asc" in script
    for package in ("rocm-libs", "rocprofiler-sdk", "hsa-amd-aqlprofile", "libopenmpi3t64", "gcc"):
        assert package in script
    assert "7.2.1|7.2.1-*) installed=1" in script and 'apt-get install -y "$2" "$3"' in script
    # A retry after a cancelled install repairs dpkg before apt-get runs again.
    assert script.index("dpkg --configure -a") < script.index("apt-get update")
    assert "{release}" not in script


def test_amd_wsl_budget_stays_inside_what_the_host_can_back(monkeypatch):
    # DXG reported 102 GiB on a 128 GB Strix Halo; an 84.5 GiB KV cache sized from that hung.
    def guest(
        argv,
        env = None,
        timeout = 600,
        input = None,
    ):
        if argv == ["cat", "/proc/meminfo"]:
            return "MemTotal:       65400496 kB\nMemAvailable:   62914560 kB\n"
        assert argv[-1] == managed_engine._DEVICE_MEMORY
        # wsl.exe returns stderr with stdout, and torch can warn after the measurement.
        return json.dumps([[102 * 2**30, 102 * 2**30]]) + "\nUserWarning: amdsmi\n"

    monkeypatch.setattr(wsl_host, "guest", guest)
    info = {"path": "/env", "host": "wsl"}
    # Windows' figure first: dedicated memory, plus 80% of the host's available RAM on an APU.
    monkeypatch.setattr(managed_engine, "_wsl_amd_usable_mib", lambda ids: [(80 * 1024.0, True)])
    assert managed_engine._engine_memory_rows(info, {}, [0]) == [(102 * 1024, 80 * 1024.0)]
    # A discrete card: 8 GiB already allocated in a 40 GiB pool leaves 8 of its 16 GiB.
    monkeypatch.setattr(
        wsl_host,
        "guest",
        lambda argv, **k: json.dumps([[32 * 2**30, 40 * 2**30]]),
    )
    monkeypatch.setattr(managed_engine, "_wsl_amd_usable_mib", lambda ids: [(16 * 1024.0, False)])
    assert managed_engine._engine_memory_rows(info, {}, [0]) == [(40 * 1024, 8 * 1024.0)]
    monkeypatch.setattr(wsl_host, "guest", guest)
    # Windows cannot say: the VM's available memory is the bound.
    monkeypatch.setattr(managed_engine, "_wsl_amd_usable_mib", lambda ids: None)
    assert managed_engine._engine_memory_rows(info, {}, [0]) == [(102 * 1024, 61440.0)]
    monkeypatch.setattr(wsl_host, "guest", lambda argv, **k: "" if argv[0] == "cat" else "[[1, 2]]")
    with pytest.raises(ValueError, match = "memory of the WSL environment"):
        managed_engine._engine_memory_rows(info, {}, [0])


def test_amd_wsl_capacity_follows_the_windows_adapter(monkeypatch):
    import types
    from utils.hardware import hardware

    props = types.SimpleNamespace(name = "AMD Radeon(TM) 8060S Graphics", gcnArchName = "gfx1151")
    torch = types.SimpleNamespace(
        cuda = types.SimpleNamespace(device_count = lambda: 1, get_device_properties = lambda i: props)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hardware, "_torch_ordinal_physical_ids", lambda count: [0])
    monkeypatch.setattr(hardware, "_props_gfx_arch", lambda p: p.gcnArchName)
    record = {"name": props.name, "gfx": "gfx1151", "dedicated_memory_bytes": 512 * 2**20}
    monkeypatch.setattr(hardware, "_windows_amd_adapter_records_or_none", lambda: {1: record})
    psutil = types.SimpleNamespace(
        virtual_memory = lambda: types.SimpleNamespace(available = 100 * 2**30)
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    # An APU adds 80% of the RAM Windows has available to its dedicated memory.
    monkeypatch.setattr(hardware, "_rocm_props_are_positively_unified", lambda p: True)
    assert managed_engine._wsl_amd_usable_mib([0]) == [(512 + 0.8 * 100 * 1024, True)]
    monkeypatch.setattr(hardware, "_rocm_props_are_positively_unified", lambda p: False)
    assert managed_engine._wsl_amd_usable_mib([0]) == [(512.0, False)]
    # No registry record that names this GPU: Windows cannot say.
    monkeypatch.setattr(hardware, "_windows_amd_adapter_records_or_none", lambda: None)
    assert managed_engine._wsl_amd_usable_mib([0]) is None

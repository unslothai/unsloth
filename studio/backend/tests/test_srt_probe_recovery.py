# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic concurrency controls; native execution is tested separately."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import socket
import json

import pytest
from core.inference import srt_probe
from core.inference.srt_diagnostics import ProbeReason


@pytest.fixture(autouse = True)
def clean(monkeypatch):
    monkeypatch.setattr(srt_probe, "setup_in_progress", threading.Event())
    monkeypatch.setattr(srt_probe, "_cache", {})
    monkeypatch.setattr(srt_probe, "_setup_conflict_identity", None)
    monkeypatch.setattr(srt_probe, "_flights", {}, raising = False)
    monkeypatch.setattr(srt_probe.srt_adapter, "installation_identity", lambda: "fixture")


def test_forced_concurrent_checks_share_one_probe(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def native(**kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return False, ProbeReason("dependency_missing", "dependency", "fixture")

    monkeypatch.setattr(srt_probe, "_native_probe", native)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(srt_probe.probe, force = True)
        assert entered.wait(2)
        second = pool.submit(srt_probe.probe, force = True)
        time.sleep(0.05)
        release.set()
        assert first.result() == second.result()
    assert len(calls) == 1


def test_recent_cache_then_changed_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: (calls.append(kw) or False, "unavailable")
    )
    srt_probe.probe()
    start = time.monotonic()
    srt_probe.probe()
    assert time.monotonic() - start < 1
    assert len(calls) == 1
    monkeypatch.setattr(srt_probe.srt_adapter, "installation_identity", lambda: "repaired")
    srt_probe.probe()
    assert len(calls) == 2


def test_setup_invalidation_does_not_cache_old_inflight_success(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def native(**kwargs):
        entered.set()
        assert release.wait(3)
        return True, "old success"

    monkeypatch.setattr(srt_probe, "_native_probe", native)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(srt_probe.probe)
        assert entered.wait(2)
        srt_probe.invalidate_cache()
        release.set()
        assert pending.result()[0] is False
    assert not srt_probe._cache


@pytest.mark.parametrize("platform,budget", [("win32", 60), ("darwin", 60)])
def test_native_budget_includes_startup(monkeypatch, platform, budget):
    from types import SimpleNamespace
    from core.inference import tools

    now = [0.0]
    monkeypatch.setattr(srt_probe.sys, "platform", platform)
    monkeypatch.setattr(srt_probe, "_windows_proxy_port_available", lambda request: True)
    monkeypatch.setattr(srt_probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tools, "_build_safe_env", lambda work: {"PATH": "fixture"})
    monkeypatch.setattr(srt_probe.shutil, "which", lambda *a, **kw: "selected bash")

    def request_for(argv, work, env, timeout, **kwargs):
        assert timeout == min(30, budget)
        return {}

    monkeypatch.setattr(srt_probe.srt_adapter, "request_for", request_for)

    def communicate(timeout):
        assert timeout == budget - 4
        raise srt_probe.subprocess.TimeoutExpired("fixture", timeout)

    killed, released = [], []
    proc = SimpleNamespace(
        communicate = communicate, kill = lambda: killed.append(1), wait = lambda **kw: None
    )

    def spawn(*a, **kw):
        assert kw["launch_deadline"] == budget
        now[0] += 4
        return proc

    monkeypatch.setattr(srt_probe.srt_adapter, "spawn", spawn)
    monkeypatch.setattr(srt_probe.srt_adapter, "release_control", lambda p: released.append(p))
    with pytest.raises(srt_probe.srt_adapter.SrtError) as error:
        srt_probe._supported_platform_probe()
    assert error.value.diagnostic.code == "probe_timeout"
    assert killed == [1] and released == [proc]


@pytest.mark.parametrize("outcome", [0, 1, "timeout"])
def test_windows_setup_preserves_interpreter_and_fails_closed(monkeypatch, outcome):
    from types import SimpleNamespace
    from core.inference import srt_setup

    monkeypatch.setattr(srt_setup, "_setup_uncertain", False)
    monkeypatch.setattr(srt_setup.sys, "platform", "win32")
    selected = srt_setup.sys.executable
    calls = []

    def run(command, **kwargs):
        assert command[0] == selected
        assert command[-1] == "--windows-install"
        assert "--force" not in command
        assert srt_probe.setup_in_progress.is_set()
        assert not srt_probe.probe()[0]
        calls.append(1)
        if outcome == "timeout":
            raise srt_setup.subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode = outcome)

    monkeypatch.setattr(srt_setup.subprocess, "run", run)
    result = srt_setup.install_windows_sandbox()
    assert result["status"] == ({0: "installed", 1: "failed"}.get(outcome, "timeout"))
    assert calls == [1]
    assert srt_probe.setup_in_progress.is_set() is (outcome == "timeout")
    if outcome == "timeout":
        assert srt_setup.install_windows_sandbox()["status"] == "timeout"
        assert calls == [1]


def test_windows_setup_conflict_has_explicit_repair(monkeypatch):
    from types import SimpleNamespace
    from core.inference import srt_setup

    monkeypatch.setattr(srt_setup, "_setup_uncertain", False)
    monkeypatch.setattr(srt_setup.sys, "platform", "win32")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--windows-force" not in command:
            kwargs["stdout"].write(
                b"filters already exist under this sublayer with a different port range or sandbox-user name"
            )
            return SimpleNamespace(returncode = 1)
        return SimpleNamespace(returncode = 0)

    monkeypatch.setattr(srt_setup.subprocess, "run", run)
    assert srt_setup.install_windows_sandbox()["status"] == "conflict"
    assert len(calls) == 1 and "--windows-force" not in calls[0]
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: pytest.fail("known conflict launched a probe")
    )
    assert srt_probe.probe()[1].code == "setup_conflict"
    assert srt_setup.install_windows_sandbox(repair_existing = True)["status"] == "installed"
    assert srt_probe._setup_conflict_identity is None
    assert len(calls) == 2 and "--windows-force" in calls[1]


def test_default_runtime_binary_replacement_invalidates_cache(monkeypatch, tmp_path):
    node = tmp_path / "selected node.exe"
    node.write_bytes(b"first")
    monkeypatch.setattr(
        srt_probe.shutil, "which", lambda name: str(node) if name == "node" else None
    )
    calls = []
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: (calls.append(1) or False, "unavailable")
    )
    srt_probe.probe()
    srt_probe.probe()
    assert len(calls) == 1
    node.write_bytes(b"replacement binary")
    srt_probe.probe()
    assert len(calls) == 2


@pytest.mark.parametrize("recovery", ["force", "runtime", "setup"])
def test_confirmed_setup_conflict_is_immediate_until_recheck(monkeypatch, recovery):
    calls = []
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: (calls.append(1) or True, "verified")
    )
    srt_probe.invalidate_cache(setup_conflict = True)
    start = time.monotonic()
    for _ in range(3):
        available, reason = srt_probe.probe()
        assert not available and reason.code == "setup_conflict"
    assert time.monotonic() - start < 1
    assert calls == []
    if recovery == "runtime":
        monkeypatch.setattr(srt_probe.srt_adapter, "installation_identity", lambda: "changed")
    elif recovery == "setup":
        srt_probe.invalidate_cache()
    assert srt_probe.probe(force = recovery == "force")[0]
    assert calls == [1]
    assert srt_probe.probe()[0]
    assert calls == [1]


def test_unavailable_windows_proxy_ports_fail_before_spawn(monkeypatch):
    from core.inference import tools

    monkeypatch.setattr(srt_probe.sys, "platform", "win32")
    monkeypatch.setattr(tools, "_build_safe_env", lambda work: {"PATH": "fixture"})
    monkeypatch.setattr(srt_probe.shutil, "which", lambda *a, **kw: "selected bash")
    monkeypatch.setattr(
        srt_probe.srt_adapter,
        "spawn",
        lambda *a, **kw: pytest.fail("blocked port launched sandbox"),
    )
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        request = {"windowsProxyPortRange": [port, port]}
        monkeypatch.setattr(srt_probe.srt_adapter, "request_for", lambda *a, **kw: request)
        start = time.monotonic()
        available, reason = srt_probe._supported_platform_probe()
        assert not available and reason.code == "proxy_port_unavailable"
        assert time.monotonic() - start < 1
    assert srt_probe._windows_proxy_port_available(request)


def test_windows_terminal_reuses_only_the_shell_already_probed(monkeypatch, tmp_path):
    monkeypatch.setattr(srt_probe.sys, "platform", "win32")
    shell = str(tmp_path / "bash.exe")
    monkeypatch.setattr(srt_probe, "_windows_probe_shell", lambda: shell)
    calls = []
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: (calls.append(kw) or True, "verified")
    )
    srt_probe.probe()
    srt_probe.probe(execution_kind = "terminal", selected_executable = shell)
    assert len(calls) == 1
    srt_probe.probe(execution_kind = "terminal", selected_executable = str(tmp_path / "other.exe"))
    assert len(calls) == 2


@pytest.mark.parametrize(
    "name,args",
    [
        ("cmd.exe", ["/d", "/c", "echo shell-ok"]),
        ("bash.exe", ["--noprofile", "--norc", "-c", "printf shell-ok"]),
    ],
)
def test_windows_probe_uses_selected_shell_command_language(monkeypatch, tmp_path, name, args):
    from core.inference import tools

    monkeypatch.setattr(srt_probe.sys, "platform", "win32")
    shell = str(tmp_path / name)
    monkeypatch.setattr(srt_probe, "_windows_probe_shell", lambda: shell)
    monkeypatch.setattr(tools, "_build_safe_env", lambda work: {"PATH": "fixture"})
    monkeypatch.setattr(srt_probe, "_windows_proxy_port_available", lambda req: False)
    captured = []
    monkeypatch.setattr(
        srt_probe.srt_adapter, "request_for", lambda argv, *a, **kw: captured.append(argv) or {}
    )
    srt_probe._supported_platform_probe()
    assert json.loads(captured[0][-1]) == [shell, *args]


@pytest.mark.parametrize("platform,available,ttl", [("win32", False, 60), ("darwin", True, 60)])
def test_successful_windows_capability_survives_a_minute_without_rechecking(
    monkeypatch, platform, available, ttl
):
    now = [0.0]
    calls = []
    monkeypatch.setattr(srt_probe.sys, "platform", platform)
    monkeypatch.setattr(srt_probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: (calls.append(1) or available, "result")
    )
    assert srt_probe.probe()[0] is available
    now[0] = ttl - 1
    assert srt_probe.probe()[0] is available
    assert calls == [1]
    now[0] = ttl
    srt_probe.probe()
    assert calls == [1, 1]
    srt_probe.probe(force = True)
    assert calls == [1, 1, 1]
    monkeypatch.setattr(srt_probe.srt_adapter, "installation_identity", lambda: "changed runtime")
    srt_probe.probe()
    assert calls == [1, 1, 1, 1]


def test_windows_success_lasts_for_process_but_configuration_changes_invalidate(monkeypatch):
    now = [0.0]
    calls = []
    monkeypatch.setattr(srt_probe.sys, "platform", "win32")
    monkeypatch.setattr(srt_probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        srt_probe, "_native_probe", lambda **kw: (calls.append(1) or True, "verified")
    )
    assert srt_probe.probe()[0]
    for age in (301, 3600, 86400, 604800):
        now[0] = age
        assert srt_probe.probe()[0]
    assert len(calls) == 1
    srt_probe.probe(force = True)
    assert len(calls) == 2
    srt_probe.invalidate_cache()
    srt_probe.probe()
    assert len(calls) == 3
    monkeypatch.setattr(srt_probe.srt_adapter, "installation_identity", lambda: "changed")
    srt_probe.probe()
    assert len(calls) == 4

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.os_sandbox import ToolLaunchPlan, PreparedSandboxLaunch
from core.inference.windows_sandbox import backend, probe, terminal_qualification
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE, WindowsRuntimeError


@pytest.fixture
def broker(tmp_path, monkeypatch):
    executable = tmp_path / "selected-python.exe"
    executable.write_bytes(b"selected executable bytes")
    monkeypatch.setattr(backend, "_os_identity", lambda: ("native-windows", 1))
    return backend.WindowsBootstrapBackend(tmp_path / "store"), executable


def observations():
    return probe.ProbeObservations(
        "11" * 32,
        "22" * 32,
        "33" * 32,
        PYTHON_PROFILE.digest,
        (3, 12, 10),
        probe.CORE_CHECKS + probe.HOST_CHECKS,
        1.0,
        probe.DnsContextObservations(1, 1),
    )


def test_core_observations_never_enable_python(broker, monkeypatch):
    instance, executable = broker
    calls = []
    monkeypatch.setattr(
        probe, "run_python_probe", lambda *args, **kwargs: (calls.append(args), observations())[1]
    )
    result = instance.probe_for_kind("python", str(executable))
    assert result.available is False and result.qualified is False
    assert "complete isolation" in result.reason
    assert len(calls) == 1


def test_failed_probe_is_not_cached_by_path_or_os(broker, monkeypatch):
    instance, executable = broker
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise WindowsRuntimeError("WINDOWS_SANDBOX_STARTUP_TIMEOUT", "timeout")

    monkeypatch.setattr(probe, "run_python_probe", fail)
    for _ in range(2):
        capability = instance.probe_for_kind("python", str(executable))
        assert not capability.available and capability.transient
    assert len(calls) == 2


def test_terminal_qualification_is_independent_and_selected(broker, monkeypatch):
    instance, executable = broker
    calls = []

    def qualify(selected, **kwargs):
        calls.append(selected)
        return terminal_qualification.TerminalQualification(
            selected,
            False,
            terminal_qualification.TERMINAL_PROFILE_ID,
            "Native children denied",
            "CHILD_DENIED",
        )

    monkeypatch.setattr(terminal_qualification, "qualify_terminal_runtime", qualify)
    monkeypatch.setattr(
        probe, "run_python_probe", lambda *a, **k: pytest.fail("wrong execution kind")
    )
    result = instance.probe_for_kind("terminal", str(executable))
    assert not result.available and "Native children denied" in result.reason
    assert calls == [str(executable)]
    assert not instance.probe_for_kind("terminal").available


@pytest.fixture
def complete_python(monkeypatch):
    # This isolated contract fixture is not native qualification evidence.
    original = probe.ProbeObservations

    @dataclass(frozen = True)
    class CompleteObservations(original):
        qualification_complete: bool = True
        catalog_binding_digest: str = "55" * 32

    measured = CompleteObservations(
        "11" * 32,
        "22" * 32,
        "33" * 32,
        PYTHON_PROFILE.digest,
        (3, 12, 10),
        ("complete-host-matrix",),
        1.0,
        probe.DnsContextObservations(1, 1),
    )
    monkeypatch.setattr(probe, "ProbeObservations", CompleteObservations)
    monkeypatch.setattr(
        probe, "REQUIRED_QUALIFICATION_CHECKS", ("complete-host-matrix",), raising = False
    )
    monkeypatch.setattr(probe, "run_python_probe", lambda *a, **k: measured)
    return measured


def test_complete_contract_requires_all_checks_and_valid_digests(
    broker, complete_python, monkeypatch
):
    instance, executable = broker
    assert instance.probe_for_kind("python", str(executable)).available
    for measured in (
        replace(complete_python, checks = ()),
        replace(complete_python, artifact_digest = ""),
        replace(complete_python, qualification_complete = False),
        replace(complete_python, profile_digest = "00" * 32),
    ):
        monkeypatch.setattr(probe, "run_python_probe", lambda *a, **k: measured)
        assert not instance.probe_for_kind("python", str(executable)).available


def test_selected_executable_changed_during_probe_is_rejected(broker, complete_python, monkeypatch):
    instance, executable = broker

    def probe_and_replace(*args, **kwargs):
        executable.write_bytes(b"changed executable")
        return complete_python

    monkeypatch.setattr(probe, "run_python_probe", probe_and_replace)
    result = instance.probe_for_kind("python", str(executable))
    assert not result.available and "changed during" in result.reason


def prepared_launch(
    spec,
    measured,
    *,
    content = None,
):
    class Owner:
        closed = False
        published = SimpleNamespace(
            core = SimpleNamespace(digest = measured.runtime_digest),
            content_digest = content or measured.content_digest,
            artifacts = SimpleNamespace(digest = measured.artifact_digest),
        )

        def spawn(self, *args):
            raise AssertionError("qualification must not spawn user payload")

        def cleanup(self):
            self.closed = True

    owner = Owner()
    prepared = PreparedSandboxLaunch(
        spec.argv,
        spec.workdir,
        {},
        None,
        "windows-lpac",
        spawn_callback = owner.spawn,
        cleanup_callbacks = [owner.cleanup],
    )
    return prepared, owner


def test_python_prepare_calls_real_api_and_binds_final_generation(
    broker, complete_python, monkeypatch, tmp_path
):
    instance, executable = broker
    spec = ToolLaunchPlan(
        (str(executable), "-u", str(tmp_path / "tool.py")),
        str(tmp_path),
        {},
        execution_kind = "python",
    )
    prepared, owner = prepared_launch(spec, complete_python)
    calls = []
    monkeypatch.setattr(
        backend.launch, "prepare_python_launch", lambda *a, **k: (calls.append((a, k)), prepared)[1]
    )
    assert instance.prepare_for_profile(spec, PYTHON_PROFILE.profile_id) is prepared
    assert calls[0][0] == (spec, instance.store_root)
    assert not owner.closed
    assert owner.expected_catalog_binding == complete_python.catalog_binding_digest
    prepared.cleanup()
    assert owner.closed


def test_changed_prepared_generation_is_cleaned_and_rejected(
    broker, complete_python, monkeypatch, tmp_path
):
    instance, executable = broker
    spec = ToolLaunchPlan(
        (str(executable), "-u", str(tmp_path / "tool.py")),
        str(tmp_path),
        {},
        execution_kind = "python",
    )
    prepared, owner = prepared_launch(spec, complete_python, content = "00" * 32)
    monkeypatch.setattr(backend.launch, "prepare_python_launch", lambda *a, **k: prepared)
    with pytest.raises(WindowsRuntimeError, match = "differs"):
        instance.prepare(spec)
    assert owner.closed


def test_profile_mismatch_and_limited_mode_never_dispatch(broker, monkeypatch, tmp_path):
    instance, executable = broker
    monkeypatch.setattr(
        instance, "_qualify", lambda *a: pytest.fail("invalid plan must not qualify")
    )
    spec = ToolLaunchPlan(
        (str(executable), "-u", str(tmp_path / "tool.py")),
        str(tmp_path),
        {},
        execution_kind = "python",
    )
    with pytest.raises(WindowsRuntimeError, match = "profile"):
        instance.prepare_for_profile(spec, terminal_qualification.TERMINAL_PROFILE_ID)
    with pytest.raises(WindowsRuntimeError, match = "isolated execution"):
        instance.prepare(replace(spec, requested_mode = "limited"))


def test_success_is_not_cached_and_generation_ignores_elapsed_time(
    broker, complete_python, monkeypatch
):
    instance, executable = broker
    calls = []

    def qualify(*args, **kwargs):
        calls.append(args)
        return replace(complete_python, elapsed_seconds = float(len(calls)))

    monkeypatch.setattr(probe, "run_python_probe", qualify)
    first = instance.probe_for_kind("python", str(executable))
    second = instance.probe_for_kind("python", str(executable))
    assert first.available and second.available
    assert len(calls) == 2
    assert first.probe_generation == second.probe_generation


def test_os_changed_during_qualification_is_rejected(broker, complete_python, monkeypatch):
    instance, executable = broker
    versions = iter((("native-windows", 1), ("native-windows", 2)))
    monkeypatch.setattr(backend, "_os_identity", lambda: next(versions))
    assert not instance.probe_for_kind("python", str(executable)).available


def test_terminal_prepare_dispatch_is_bound_to_selected_shell(broker, monkeypatch, tmp_path):
    from core.inference.windows_sandbox import terminal_probe

    instance, executable = broker
    measured = terminal_qualification.TerminalQualification(
        str(executable),
        True,
        terminal_qualification.TERMINAL_PROFILE_ID,
        "contract fixture",
        "",
        checks = ("complete-terminal-matrix",),
        runtime_digest = "22" * 32,
        content_digest = "11" * 32,
    )
    monkeypatch.setattr(
        terminal_qualification, "REQUIRED_QUALIFICATION_CHECKS", measured.checks, raising = False
    )
    monkeypatch.setattr(
        terminal_qualification, "qualify_terminal_runtime", lambda *a, **k: measured
    )
    spec = ToolLaunchPlan(
        (str(executable), "/c", "echo test"), str(tmp_path), {}, execution_kind = "terminal"
    )
    prepared, owner = prepared_launch(
        spec,
        SimpleNamespace(
            runtime_digest = measured.runtime_digest,
            content_digest = measured.content_digest,
            artifact_digest = "33" * 32,
        ),
    )
    calls = []
    monkeypatch.setattr(
        backend.terminal_launch,
        "prepare_terminal_launch",
        lambda *a, **k: (calls.append(a), prepared)[1],
    )
    monkeypatch.setattr(
        terminal_probe,
        "_runtime_identity",
        lambda owner: (str(executable), (), measured.runtime_digest, measured.content_digest),
    )
    assert (
        instance.prepare_for_profile(spec, terminal_qualification.TERMINAL_PROFILE_ID) is prepared
    )
    assert calls == [(spec,)]
    prepared.cleanup()
    assert owner.closed


def test_expected_capability_generation_change_refuses_before_preparing(
    broker, complete_python, monkeypatch, tmp_path
):
    instance, executable = broker
    initial = instance.probe_for_kind("python", str(executable))
    assert initial.qualification_generation == initial.probe_generation
    monkeypatch.setattr(
        probe,
        "run_python_probe",
        lambda *a, **k: replace(complete_python, content_digest = "44" * 32),
    )
    monkeypatch.setattr(
        backend.launch,
        "prepare_python_launch",
        lambda *a, **k: pytest.fail("prepared changed generation"),
    )
    spec = ToolLaunchPlan(
        (str(executable), "-u", str(tmp_path / "tool.py")),
        str(tmp_path),
        {},
        execution_kind = "python",
    )
    with pytest.raises(WindowsRuntimeError, match = "qualification changed"):
        instance.prepare_for_profile(
            spec, PYTHON_PROFILE.profile_id, expected_generation = initial.qualification_generation
        )


def test_missing_private_catalog_binding_cannot_qualify(broker, complete_python, monkeypatch):
    instance, executable = broker
    monkeypatch.setattr(
        probe,
        "run_python_probe",
        lambda *a, **k: replace(complete_python, catalog_binding_digest = ""),
    )
    assert not instance.probe_for_kind("python", str(executable)).available


def test_failed_capability_generation_binds_selected_executable_identity(broker, monkeypatch):
    instance, executable = broker
    monkeypatch.setattr(probe, "run_python_probe", lambda *a, **k: observations())
    first = instance.probe_for_kind("python", str(executable))
    again = instance.probe_for_kind("python", str(executable))
    executable.write_bytes(b"changed selected executable")
    changed = instance.probe_for_kind("python", str(executable))
    assert not first.available and not changed.available
    assert first.probe_generation == again.probe_generation
    assert first.probe_generation != changed.probe_generation

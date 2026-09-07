# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cleanup remains operational without the Studio application logging stack."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_launch import LAUNCH, installed_runtime, run_harness, runtime_wheel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import prepared as prepared_module


def prepared(tmp_path, **kwargs):
    return prepared_module.PreparedPythonLaunch((), str(tmp_path), {}, None, "test", **kwargs)


def test_successful_cleanup_needs_no_application_logger(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "loggers", None)
    events = []
    stream = io.BytesIO()
    private = tmp_path / "private"
    private.mkdir()
    launch = prepared(
        tmp_path,
        cleanup_callbacks = [lambda: events.append(1), lambda: events.append(2)],
        owned_files = [stream],
        cleanup_paths = [str(private)],
    )
    launch.cleanup()
    launch.cleanup()
    assert events == [2, 1] and stream.closed and not private.exists()
    assert not launch.cleanup_diagnostics
    assert not launch.cleanup_callbacks and not launch.owned_files and not launch.cleanup_paths


@pytest.mark.parametrize("failure", ["import", "factory", "handler"])
def test_logging_failure_cannot_interrupt_cleanup(monkeypatch, tmp_path, failure):
    def broken(*args, **kwargs):
        assert stream.closed and not private.exists(), "Logging ran before cleanup completed"
        raise RuntimeError("logging unavailable")

    logger = SimpleNamespace(warning = broken)
    factory = broken if failure == "factory" else lambda name: logger
    monkeypatch.setitem(
        sys.modules, "loggers", None if failure == "import" else SimpleNamespace(get_logger = factory)
    )
    events = []

    def failed_callback():
        events.append("failed callback")
        raise OSError("callback unavailable")

    class FailedFile:
        def close(self):
            events.append("failed file")
            raise OSError("file unavailable")

    stream = io.BytesIO()
    private = tmp_path / "private"
    private.mkdir()
    blocked = tmp_path / "blocked"
    original = prepared_module.shutil.rmtree

    def remove(path):
        events.append(str(path))
        if path == str(blocked):
            raise PermissionError("path unavailable")
        original(path)

    monkeypatch.setattr(prepared_module.shutil, "rmtree", remove)
    launch = prepared(
        tmp_path,
        cleanup_callbacks = [lambda: events.append("last callback"), failed_callback],
        owned_files = [stream, FailedFile()],
        cleanup_paths = [str(private), str(blocked)],
    )
    launch.cleanup()
    before = list(launch.cleanup_diagnostics)
    launch.cleanup()
    assert launch.cleanup_diagnostics == before
    assert events == ["failed callback", "last callback", "failed file", str(blocked), str(private)]
    assert stream.closed and not private.exists()
    assert "OSError: callback unavailable" in before
    assert "OSError: file unavailable" in before
    assert f"could not remove private sandbox path: {blocked}" in before
    assert len([value for value in before if value.startswith("Cleanup logging failed:")]) == 3


def test_cleanup_logs_original_exception_after_resources_close(monkeypatch, tmp_path):
    stream = io.BytesIO()
    failure = OSError("owned callback failed")
    records = []

    def warning(message, diagnostic, *, exc_info):
        assert stream.closed
        assert exc_info[0] is OSError and exc_info[1] is failure and exc_info[2] is not None
        records.append(diagnostic)

    monkeypatch.setitem(
        sys.modules,
        "loggers",
        SimpleNamespace(get_logger = lambda name: SimpleNamespace(warning = warning)),
    )

    def fail():
        raise failure

    launch = prepared(tmp_path, cleanup_callbacks = [fail], owned_files = [stream])
    launch.cleanup()
    assert records == launch.cleanup_diagnostics == ["OSError: owned callback failed"]


@pytest.mark.skipif(sys.platform != "win32", reason = "Native installed Windows launch")
def test_installed_launch_public_cleanup_without_logger(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import identity as identities, launch
from core.inference.os_sandbox import PreparedSandboxLaunch
script.write_text("print('NATIVE_CLEANUP_CONTROL', flush=True)",encoding='utf-8')
{LAUNCH}
assert isinstance(prepared, PreparedSandboxLaunch)
assert prepared.network_audit is None
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert process.wait(timeout=10) == 0
    assert process.stdout.read().strip() == 'NATIVE_CLEANUP_CONTROL'
    sys.modules['loggers'] = None
    prepared.cleanup()
    prepared.cleanup()
    assert not prepared.cleanup_diagnostics
    assert owner.closed and owner.started and not owner.handles
    assert not owner.file_pins.handles and not owner.pins.handles
    assert not manifest.exists() and not prepared.cleanup_callbacks
    with identities._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
        assert identities._profile_path(sid) is None
    assert not list((root/'cache'/'.readers').iterdir()) and not launch._pending_cleanup
    print('PUBLIC_CLEANUP_WITHOUT_LOGGER_OK')
finally:
    owner.cleanup()
""",
    )
    assert "PUBLIC_CLEANUP_WITHOUT_LOGGER_OK" in output

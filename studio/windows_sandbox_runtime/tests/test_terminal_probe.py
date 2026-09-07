# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed Terminal probe observations and required native compatibility."""

from dataclasses import fields
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import terminal_probe as probe
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.mark.parametrize(
    "path,family",
    [
        (r"C:\Windows\System32\cmd.exe", "cmd"),
        (r"C:\Program Files\Git\bin\bash.exe", "bash"),
        (r"C:\tools\bash", "bash"),
    ],
)
def test_selected_shell_validation_is_structural(path, family):
    assert probe._selected(path) == family


@pytest.mark.parametrize(
    "path",
    [
        "cmd.exe",
        r"C:\Windows\System32\powershell.exe",
        r"C:\Windows\System32\cmd.exe\..\other.exe",
        r"\\server\share\cmd.exe",
        "",
        None,
    ],
)
def test_unknown_or_nonlocal_shell_fails_closed_before_fixture(path, monkeypatch):
    monkeypatch.setattr(probe, "_private_fixture", lambda: pytest.fail("fixture created"))
    with pytest.raises(WindowsRuntimeError) as caught:
        probe.run_terminal_probe(path)
    assert caught.value.code == "WINDOWS_SANDBOX_TERMINAL_PROBE_INVALID"


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 121, "30"])
def test_invalid_timeout_fails_before_fixture(timeout, monkeypatch):
    monkeypatch.setattr(probe, "_private_fixture", lambda: pytest.fail("fixture created"))
    with pytest.raises(WindowsRuntimeError) as caught:
        probe.run_terminal_probe(r"C:\Windows\System32\cmd.exe", timeout = timeout)
    assert caught.value.code == "WINDOWS_SANDBOX_TERMINAL_PROBE_INVALID"


def test_cancelled_probe_starts_nothing(monkeypatch):
    import threading

    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(probe, "_private_fixture", lambda: pytest.fail("fixture created"))
    with pytest.raises(WindowsRuntimeError) as caught:
        probe.run_terminal_probe(r"C:\Windows\System32\cmd.exe", cancel = cancel)
    assert caught.value.code == "WINDOWS_SANDBOX_CANCELLED"


@pytest.mark.parametrize("token", ["INLINE_OK", "BATCH_OK", "CHILD_OK"])
def test_fixed_output_parser_accepts_only_its_exact_token(token):
    assert probe._parse_output((token + "\r\n").encode(), token) is None


@pytest.mark.parametrize(
    "data",
    [b"", b"INLINE_OK\nEXTRA\n", b"wrong\n", b"\xff", b"x" * (16 * 1024 + 1)],
)
def test_fixed_output_parser_rejects_malformed_or_unexpected_data(data):
    with pytest.raises(WindowsRuntimeError) as caught:
        probe._parse_output(data, "INLINE_OK")
    assert caught.value.code == "WINDOWS_SANDBOX_TERMINAL_PROBE_FAILED"


def test_output_waits_for_pipe_eof_after_leader_exit(monkeypatch):
    class Api:
        calls = 0

        def PeekNamedPipe(self, handle, buffer, size, read, available, remaining):
            self.calls += 1
            available._obj.value = 0
            return self.calls == 1

    api = Api()
    process = SimpleNamespace(
        stdout = SimpleNamespace(buffer = SimpleNamespace(raw = SimpleNamespace(_handle = 123))),
        poll = lambda: 0,
    )
    monkeypatch.setattr(probe, "pipe_api", lambda: api)
    monkeypatch.setattr(probe.ctypes, "get_last_error", lambda: 109, raising = False)
    assert probe._collect_output(process, time.monotonic() + 1, None) == b""
    assert api.calls == 2


def test_plans_are_fixed_to_the_selected_shell_and_private_workdir(tmp_path):
    cmd = r"C:\Windows\System32\cmd.exe"
    bash = r"C:\Program Files\Git\bin\bash.exe"
    for shell, family, prefix in (
        (cmd, "cmd", ("/d", "/s", "/c")),
        (bash, "bash", ("--noprofile", "--norc", "-c")),
    ):
        plans = [probe._plan(family, shell, tmp_path, index) for index in range(3)]
        assert [item[0].argv[0] for item in plans] == [shell] * 3
        assert all(item[0].argv[1:-1] == prefix for item in plans)
        assert all(item[0].workdir == str(tmp_path) for item in plans)
        assert all(item[0].execution_kind == "terminal" for item in plans)
        assert plans[0][1] is None and plans[2][1] is None
        assert plans[1][1] is not None and plans[1][1].parent == tmp_path


def test_cmd_batch_fixture_is_quoted_and_exact_crlf(tmp_path):
    spec, script = probe._plan("cmd", r"C:\Windows\System32\cmd.exe", tmp_path, 1)
    assert script is not None and spec.argv[-1] == f'"{script}"'
    probe._write_script(script, "cmd")
    assert script.read_bytes() == b"@echo off\r\necho BATCH_OK\r\n"


@pytest.mark.skipif(sys.platform != "win32", reason = "Native private Terminal workdir")
def test_private_fixture_keeps_inheritable_user_acl_and_real_cmd_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMP", str(tmp_path))
    root, workdir, parent = probe._private_fixture()
    api = probe.native_files()
    shell = str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe")
    try:
        handle = api.open(workdir, directory = True)
        try:
            with api.security_attributes(
                f"O:{api.owner}D:P(A;OICI;FA;;;{api.owner})(A;OICI;FA;;;SY)"
            ) as attributes:
                assert api.security_text(handle) == api._sddl(attributes.descriptor)
        finally:
            assert api.kernel.CloseHandle(handle)
        # Exercise real preparation, LPAC spawn, output parsing and cleanup.
        probe._run_check(
            "cmd",
            shell,
            workdir,
            0,
            probe.TERMINAL_CHECKS[0],
            "INLINE_OK",
            time.monotonic() + 30,
            None,
        )
        assert (workdir / "inline io.txt").read_bytes() == b"INLINE_OK\r\n"
    finally:
        probe._remove_fixture(root, parent)
    assert not root.exists()


def test_runtime_digest_uses_pinned_root_file_identity():
    root = Path(r"C:\Program Files\Git\bin")
    executable = root / "bash.exe"
    selected = SimpleNamespace(argv = (str(executable),), runtime_roots = (str(root),))
    dir_info = SimpleNamespace(volume = 7, index_high = 8, index_low = 9)
    binary = SimpleNamespace(volume = 7, index_high = 10, index_low = 11, size = 12)
    pins = SimpleNamespace(
        handles = {root: 123, executable: 456},
        api = SimpleNamespace(info = lambda handle, directory = False: dir_info if directory else binary),
    )
    executable, roots, digest, content_digest = probe._runtime_identity(
        SimpleNamespace(selected = selected, pins = pins)
    )
    assert executable == selected.argv[0] and roots == selected.runtime_roots
    assert content_digest == ""
    assert (
        len(digest) == 64
        and digest == probe._runtime_identity(SimpleNamespace(selected = selected, pins = pins))[2]
    )


def test_copied_runtime_identity_binds_content_without_changing_selected_shell():
    root, copied = Path("C:/private/runtime/bin"), Path("C:/private/runtime/bin/bash.exe")
    selected = SimpleNamespace(argv = (str(copied),), runtime_roots = (str(root),))
    info = SimpleNamespace(volume = 7, index_high = 8, index_low = 9, size = 12)
    pins = SimpleNamespace(
        handles = {root: 123, copied: 456},
        api = SimpleNamespace(info = lambda *_args, **_kwargs: info),
    )
    snapshot = SimpleNamespace(
        digest = "a" * 64, source = SimpleNamespace(argv = ("C:/Program Files/Git/bin/bash.exe",))
    )
    owner = SimpleNamespace(selected = selected, pins = pins, snapshot = snapshot)
    first = probe._runtime_identity(owner)
    snapshot.digest = "b" * 64
    second = probe._runtime_identity(owner)
    assert first[:2] == second[:2] == (snapshot.source.argv[0], selected.runtime_roots)
    assert first[2] != second[2]
    assert first[3] == "a" * 64 and second[3] == "b" * 64


def test_retained_fixture_owner_retries_launch_before_removal(tmp_path, monkeypatch):
    root = tmp_path / "unsloth-terminal-probe-fixed"
    root.mkdir()
    launch = SimpleNamespace(closed = False)
    launch.cleanup = lambda: setattr(launch, "closed", True)
    removed = []
    monkeypatch.setattr(
        probe, "_remove_fixture", lambda path, parent: removed.append((path, parent))
    )
    owner = probe._FixtureOwner(root, tmp_path)
    error = RuntimeError("retained")
    owner.retain(error, launch)
    assert error.retained_fixture_owner is owner and owner in probe._pending_fixtures
    owner.cleanup()
    assert removed == [(root, tmp_path)]
    assert owner.closed and owner not in probe._pending_fixtures


def _git_bash():
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Git", "bin", "bash.exe"),
        shutil.which("bash.exe"),
    ]
    return next(
        (
            os.path.realpath(path)
            for path in candidates
            if path
            and os.path.isfile(path)
            and Path(path).parent.name.lower() == "bin"
            and Path(path).parent.parent.name.lower() == "git"
        ),
        None,
    )


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows Terminal probe")
@pytest.mark.parametrize(
    "shell",
    [
        pytest.param(
            str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"),
            id = "cmd",
        ),
        pytest.param(
            _git_bash() or "",
            id = "git-bash",
            marks = pytest.mark.skipif(_git_bash() is None, reason = "Git Bash is not installed"),
        ),
    ],
)
@pytest.mark.skipif(
    os.environ.get("UNSLOTH_TEST_TERMINAL_COMPATIBILITY") != "1",
    reason = "Opt-in native qualification diagnostic; this OS failed Terminal compatibility",
)
def test_actual_selected_terminal_passes_every_required_probe(shell, tmp_path, monkeypatch):
    # Isolate the real backend's content cache and journals between test runs.
    # Keep the production path; no backend result or payload launch is mocked.
    # A short independent cache avoids pytest's nested paths exceeding the
    # selected Git wrapper's legacy filename limit.
    with tempfile.TemporaryDirectory(prefix = "us-tp-") as local:
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("LOCALAPPDATA", local)
        observations = probe.run_terminal_probe(shell, store_root = os.path.join(local, "store"))
    assert observations.selected_executable == os.path.realpath(shell)
    assert observations.runtime_roots
    assert len(observations.runtime_digest) == 64
    assert observations.checks == probe.TERMINAL_CHECKS
    assert observations.elapsed_seconds > 0
    names = {field.name for field in fields(observations)}
    assert not names & {
        "available",
        "qualified",
        "capability",
        "execution_record",
        "protection_state",
    }
    assert not list(tmp_path.iterdir())

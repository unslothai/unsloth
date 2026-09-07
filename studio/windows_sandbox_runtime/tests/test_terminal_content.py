# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static Terminal content-copy contracts, not backend qualification."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from core.inference.windows_sandbox.content import ContentGeneration, RuntimeContentStore
from core.inference.windows_sandbox.content_files import PathLease, native_files
from core.inference.windows_sandbox.native_plan import ScanBounds
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
from core.inference.windows_sandbox import terminal_content, terminal_runtime
from core.inference.windows_sandbox.terminal_runtime import TerminalRuntimeRoots
from core.inference.os_sandbox import ToolLaunchPlan


@pytest.fixture
def selected_bash(tmp_path, monkeypatch):
    installation = tmp_path / "portable-bash"
    binary = installation / "bin"
    library = installation / "usr" / "bin"
    excluded = installation / "share"
    workdir = tmp_path / "work"
    for directory in (binary, library, excluded, workdir):
        directory.mkdir(parents = True)
    executable = binary / "bash.exe"
    files = {
        executable: b"fixture bash payload",
        binary / "support.dll": b"fixture support",
        library / "runtime.dll": b"fixture runtime",
    }
    for path, data in files.items():
        path.write_bytes(data)
    (excluded / "not-selected.dat").write_bytes(b"excluded sibling")
    argv = (str(executable), "--noprofile", "-c", "printf untouched")
    roots = (str(binary), str(library))
    source = TerminalRuntimeRoots(argv, str(workdir), roots, roots)
    monkeypatch.setattr(terminal_content.runtime, "_inspect", lambda _value: source)
    return source, installation, files


def snapshot_names(snapshot):
    return tuple(item.relative_path for item in snapshot.spec.files)


def test_snapshot_preserves_selected_layout_and_excludes_siblings(selected_bash):
    source, _installation, _files = selected_bash
    snapshot = terminal_content.plan_terminal_content(source)

    assert snapshot_names(snapshot) == (
        "shell/bin/bash.exe",
        "shell/bin/support.dll",
        "shell/usr/bin/runtime.dll",
    )
    assert snapshot.executable == "shell/bin/bash.exe"
    assert snapshot.roots == ("shell/bin", "shell/usr/bin")
    assert "not-selected.dat" not in snapshot.spec.manifest().decode("utf-8")
    assert snapshot.source.argv == source.argv


def test_same_size_same_mtime_content_change_changes_digest(selected_bash):
    source, _installation, files = selected_bash
    target = Path(source.runtime_roots[1]) / "runtime.dll"
    before = terminal_content.plan_terminal_content(source)
    timestamps = target.stat()
    replacement = b"changed runtime"
    assert len(replacement) == len(files[target])
    target.write_bytes(replacement)
    os.utime(target, ns = (timestamps.st_atime_ns, timestamps.st_mtime_ns))

    after = terminal_content.plan_terminal_content(source)
    assert target.stat().st_size == len(files[target])
    assert target.stat().st_mtime_ns == timestamps.st_mtime_ns
    assert after.digest != before.digest


def test_layout_marker_copies_only_empty_directory_and_changes_digest(selected_bash, tmp_path):
    source, installation, _files = selected_bash
    before = terminal_content.plan_terminal_content(source)
    marker = installation / "mingw64/bin"
    marker.mkdir(parents = True)
    (marker / "unselected-secret.dll").write_bytes(b"must not be copied")
    snapshot = terminal_content.plan_terminal_content(source)
    assert snapshot.spec.directories == ("shell/mingw64/bin",)
    assert snapshot.digest != before.digest
    assert snapshot.spec.files == before.spec.files
    assert snapshot.source == before.source
    root = tmp_path / "generation"
    generation = ContentGeneration(
        snapshot.digest,
        root,
        tuple(root / "files" / name for name in snapshot_names(snapshot)),
    )
    with pytest.raises(WindowsRuntimeError, match = "different directory inventory"):
        snapshot.relocated(generation)


@pytest.mark.parametrize("entry_type", ["symlink", "hardlink", "special"])
def test_snapshot_rejects_links_and_special_files(selected_bash, entry_type):
    source, _installation, _files = selected_bash
    root = Path(source.runtime_roots[0])
    target = root / "support.dll"
    hostile = root / "hostile"
    try:
        if entry_type == "symlink":
            hostile.symlink_to(target)
        elif entry_type == "hardlink":
            os.link(target, hostile)
        elif hasattr(os, "mkfifo"):
            os.mkfifo(hostile)
        else:
            pytest.skip("This platform cannot create a portable special-file fixture")
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"This host cannot create the {entry_type} fixture: {error}")

    with pytest.raises(WindowsRuntimeError, match = "reparse point|special or hardlinked"):
        terminal_content.plan_terminal_content(source)


def test_scan_bounds_and_cancellation_fail_closed(selected_bash):
    source, _installation, _files = selected_bash
    with pytest.raises(WindowsRuntimeError, match = "entry limit"):
        terminal_content.plan_terminal_content(source, bounds = ScanBounds(entries = 1))

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(WindowsRuntimeError, match = "WINDOWS_SANDBOX_CANCELLED"):
        terminal_content.plan_terminal_content(source, cancel = cancelled)


def test_validation_precedes_store_creation(selected_bash, tmp_path):
    source, _installation, _files = selected_bash
    store_root = tmp_path / "must-not-exist"

    with pytest.raises(WindowsRuntimeError, match = "Invalid Terminal content scan bounds"):
        terminal_content.publish_terminal_content(
            source,
            store_root,
            bounds = ScanBounds(entries = 0),
        )
    assert not store_root.exists()


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows path-overlap policy")
def test_store_cannot_overlap_selected_runtime(selected_bash):
    source, installation, _files = selected_bash
    store_root = installation / "bin" / "nested-store"

    with pytest.raises(WindowsRuntimeError, match = "store must be separate"):
        terminal_content.publish_terminal_content(source, store_root)
    assert not store_root.exists()


def test_relocation_requires_exact_generation_inventory_and_preserves_arguments(
    selected_bash, tmp_path
):
    source, _installation, _files = selected_bash
    snapshot = terminal_content.plan_terminal_content(source)
    generation_root = tmp_path / "generation"
    copied = tuple(generation_root / "files" / name for name in snapshot_names(snapshot))
    generation = ContentGeneration(snapshot.digest, generation_root, copied)

    relocated = snapshot.relocated(generation)
    assert relocated.argv[1:] == source.argv[1:]
    assert relocated.argv[0] == str(generation_root / "files" / snapshot.executable)
    assert relocated.runtime_roots == tuple(
        str(generation_root / "files" / root) for root in snapshot.roots
    )
    assert source.argv[1:] == ("--noprofile", "-c", "printf untouched")

    extra = ContentGeneration(
        snapshot.digest,
        generation_root,
        (*copied, generation_root / "files" / "shell/uninventoried.dll"),
    )
    with pytest.raises(WindowsRuntimeError, match = "different file inventory"):
        snapshot.relocated(extra)


def response_bytes(
    snapshot,
    request,
    *,
    pid = 73,
    nonce = "a" * 64,
):
    return json.dumps(
        {
            "schema": 1,
            "nonce": nonce,
            "request_digest": terminal_runtime._digest(request),
            "pid": pid,
            "snapshot": asdict(snapshot),
            "error": None,
        },
        separators = (",", ":"),
    ).encode()


@pytest.mark.parametrize("field", ["schema", "nonce", "request_digest", "pid", "missing"])
def test_snapshot_response_rejects_malformed_provenance(selected_bash, field):
    source, _installation, _files = selected_bash
    snapshot = terminal_content.plan_terminal_content(source)
    request = {"input": {}, "store_root": "unit-only"}
    value = json.loads(response_bytes(snapshot, request))
    if field == "missing":
        value.pop("error")
    elif field == "schema":
        value[field] = 2
    elif field == "pid":
        value[field] = 74
    else:
        value[field] = "b" * 64

    with pytest.raises(WindowsRuntimeError, match = "belongs to another preparation"):
        terminal_content._snapshot_response(json.dumps(value).encode(), 73, "a" * 64, request)


@pytest.mark.skipif(sys.platform != "win32", reason = "Full Windows response path validation")
def test_snapshot_response_accepts_exact_worker_inventory(selected_bash, tmp_path):
    source, _installation, _files = selected_bash
    snapshot = terminal_content.plan_terminal_content(source)
    request = {
        "input": terminal_runtime._input(
            {"argv": list(source.argv), "workdir": source.workdir, "env": {}}
        ),
        "store_root": str(tmp_path / "store"),
    }

    decoded = terminal_content._snapshot_response(
        response_bytes(snapshot, request), 73, "a" * 64, request
    )
    assert decoded == snapshot


@pytest.mark.skipif(sys.platform != "win32", reason = "Full Windows response path validation")
@pytest.mark.parametrize("change", ["outside", "layout", "inventory", "policy", "directory"])
def test_snapshot_response_rejects_changed_inventory_layout_or_policy(
    selected_bash, tmp_path, change
):
    source, _installation, _files = selected_bash
    snapshot = terminal_content.plan_terminal_content(source)
    request = {
        "input": terminal_runtime._input(
            {"argv": list(source.argv), "workdir": source.workdir, "env": {}}
        ),
        "store_root": str(tmp_path / "store"),
    }
    value = json.loads(response_bytes(snapshot, request))
    files = value["snapshot"]["spec"]["files"]
    if change == "outside":
        files[0]["source"]["path"] = str(tmp_path / "outside.dll")
    elif change == "layout":
        files[0]["relative_path"] = "shell/usr/bin/renamed.dll"
    elif change == "inventory":
        files[:] = [item for item in files if item["relative_path"] != snapshot.executable]
    elif change == "directory":
        value["snapshot"]["spec"]["directories"] = ["shell/unselected/bin"]
    else:
        value["snapshot"]["spec"]["profile_digest"] = "b" * 64
    if change != "policy":
        value["snapshot"]["spec"]["dependency_digest"] = terminal_runtime._digest(
            {"files": files, "directories": value["snapshot"]["spec"]["directories"]}
        )

    with pytest.raises(
        WindowsRuntimeError, match = "outside|relative source layout|omitted|copy policy"
    ):
        terminal_content._snapshot_response(json.dumps(value).encode(), 73, "a" * 64, request)


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows fixed-worker preparation")
def test_failed_preparation_never_falls_back_to_host_scan(monkeypatch, tmp_path):
    executable = str(tmp_path / "bash.exe")
    spec = ToolLaunchPlan(
        (executable, "-c", "printf untouched"),
        str(tmp_path),
        {},
        execution_kind = "terminal",
    )
    worker_calls = []
    monkeypatch.setattr(terminal_runtime, "_capture_broker_runtime", lambda: object())
    monkeypatch.setattr(terminal_runtime, "_scanner_executable", lambda _broker: sys.executable)
    monkeypatch.setattr(terminal_runtime, "_profile_environment", lambda: {})

    def fail_worker(*_args, **_kwargs):
        worker_calls.append(True)
        raise WindowsRuntimeError("WINDOWS_SANDBOX_PREPARATION_FAILED", "unit worker failure")

    monkeypatch.setattr(terminal_content, "_run_worker", fail_worker)
    monkeypatch.setattr(
        terminal_content,
        "plan_terminal_content",
        lambda *_args, **_kwargs: pytest.fail("host fallback"),
    )
    with pytest.raises(WindowsRuntimeError, match = "unit worker failure"):
        terminal_content.prepare_terminal_content(spec, str(tmp_path / "store"))
    assert worker_calls == [True]


@pytest.mark.skipif(sys.platform != "win32", reason = "Native fixed-worker content-store contract")
def test_fixed_worker_publishes_selected_git_bash_without_source_acl_mutation(tmp_path):
    from core.inference.tools import _get_shell_cmd

    argv = tuple(_get_shell_cmd(""))
    if Path(argv[0]).name.lower() not in ("bash", "bash.exe"):
        pytest.skip("Studio did not select an installed Git Bash")
    workdir = tmp_path / "work"
    workdir.mkdir()
    spec = ToolLaunchPlan(argv, str(workdir), {}, execution_kind = "terminal")
    source = terminal_runtime.inspect_terminal_runtime(spec)
    api = native_files()
    directories = tuple(Path(path) for path in source.runtime_roots)
    files = {Path(source.argv[0])}
    for directory in directories:
        candidates = sorted(
            path for path in directory.iterdir() if path.is_file() and not path.is_symlink()
        )
        files.update(candidates[:3])
    files = tuple(sorted(files))
    assert len(files) > 1

    def security():
        with PathLease() as pins:
            return {
                **{path: api.security_text(pins.directory(path)) for path in directories},
                **{path: api.security_text(pins.file(path)) for path in files},
            }

    before = security()
    store_root = tmp_path / "terminal-store"
    snapshot = terminal_content.prepare_terminal_content(spec, str(store_root))
    assert security() == before

    store = RuntimeContentStore(store_root)
    with store.lease(snapshot.digest) as generation:
        assert generation.directories
        for directory in generation.directories:
            assert directory.is_dir() and not list(directory.iterdir())
        assert {
            path.relative_to(generation.directory / "files").as_posix() for path in generation.files
        } == set(snapshot_names(snapshot))
        for item in snapshot.spec.files:
            copied = generation.directory / "files" / item.relative_path
            assert copied.read_bytes() == Path(item.source.path).read_bytes()
        relocated = snapshot.relocated(generation)
        assert relocated.argv[1:] == source.argv[1:]
    assert security() == before

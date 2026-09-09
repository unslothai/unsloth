# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused security contract for confined file editing.

These tests are deliberately independent of model inference. Portable tests
exercise the Windows fail-closed branch on every host, while the workflow also
runs the same contract on native Linux, macOS, and Windows filesystems.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from core.agent_workspace import common, mutation
from core.agent_workspace.common import AgentWorkspaceError, ProjectWorkspace
from core.agent_workspace.mutation import (
    ProjectFileMutation,
    WindowsMutationRejected,
)
from core.inference import tools


def _workspace(root: Path, project_id: str = "secure-tools") -> ProjectWorkspace:
    metadata = root.stat(follow_symlinks = False)
    return ProjectWorkspace(
        project_id = project_id,
        root = root.resolve(strict = True),
        kind = "managed",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
    )


@pytest.mark.skipif(os.name != "posix", reason = "Native POSIX symlink contract")
@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("parent", [False, True])
def test_sandboxed_edit_refuses_symlinks_even_when_the_target_is_inside(
    tmp_path, monkeypatch, outside, parent
):
    root = tmp_path / "workspace"
    root.mkdir()
    target_dir = tmp_path / "outside" if outside else root / "real"
    target_dir.mkdir()
    target = target_dir / "value.txt"
    target.write_text("original")
    link = root / "link"
    link.symlink_to(target_dir if parent else target, target_is_directory = parent)
    monkeypatch.setattr(tools, "_get_workdir", lambda _session = None: str(root))
    result = tools.execute_tool(
        "edit_file",
        {
            "path": "link/value.txt" if parent else "link",
            "edits": [{"old_string": "original", "new_string": "changed"}],
        },
        session_id = "conversation",
    )
    assert result.startswith("Error:"), result
    assert target.read_text() == "original"
    assert link.is_symlink()


def _bind_project(
    monkeypatch,
    root: Path,
    project_id: str = "secure-tools",
) -> str:
    workspace = _workspace(root, project_id)
    session_id = tools.project_session_id(project_id)
    monkeypatch.setattr(tools, "_get_project_workdir", lambda _session_id: str(root))
    monkeypatch.setattr(tools, "_get_workdir", lambda _session_id = None: str(root))
    monkeypatch.setattr(common, "project_workspace", lambda _project_id: workspace)

    @contextlib.contextmanager
    def access(received_project_id):
        assert received_project_id == project_id
        yield workspace

    monkeypatch.setattr(common, "project_workspace_access", access)
    return session_id


def test_project_execution_id_rejects_a_changed_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    replacement = tmp_path / "replacement"
    root.mkdir()
    replacement.mkdir()
    session_id = tools.project_session_id("secure-tools")
    monkeypatch.setattr(tools, "_get_project_workdir", lambda _session_id: str(root))

    with pytest.raises(RuntimeError, match = "workspace changed"):
        tools._project_execution_id(
            session_id,
            str(replacement),
            disable_sandbox = False,
        )


def test_stored_chat_with_project_prefix_keeps_its_conversation_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_thread_exists", lambda _session_id: True)
    monkeypatch.setattr(
        tools,
        "_get_project_workdir",
        lambda _session_id: pytest.fail("a stored chat was resolved as a project"),
    )
    assert tools._project_execution_id("project-chat", str(tmp_path), disable_sandbox = False) is None
    workspace = tools._edit_file_workspace("project-chat", str(tmp_path))
    assert workspace.kind == "conversation"
    assert workspace.root == tmp_path.resolve()


def test_hardlinked_file_is_rejected_before_content_is_read(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"private outside content")
    os.link(outside, root / "linked.txt")
    with ProjectFileMutation.open(_workspace(root), "linked.txt") as boundary:
        with pytest.raises((AgentWorkspaceError, WindowsMutationRejected), match = "Hard-linked"):
            boundary.read(1024)
    assert outside.read_bytes() == b"private outside content"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_round_trip_is_exact_and_atomic(tmp_path):
    root = tmp_path / "repository"
    target = root / "src" / "module.py"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"before\n")
    workspace = _workspace(root)

    with ProjectFileMutation.open(workspace, "src/module.py") as boundary:
        before, mode, identity = boundary.read(1024)
        assert before == b"before\n"
        assert (
            boundary.replace(
                b"after\n",
                expect = before,
                mode = mode,
                identity = identity,
            )
            is None
        )

    assert target.read_bytes() == b"after\n"
    assert not list(target.parent.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name == "nt", reason = "POSIX mode semantics")
def test_posix_project_create_applies_the_process_umask(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "created.txt"
    previous_umask = os.umask(0o027)
    try:
        with ProjectFileMutation.open(_workspace(root), "created.txt") as boundary:
            assert boundary.create(b"created") is None
    finally:
        os.umask(previous_umask)

    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.skipif(os.name == "nt", reason = "POSIX mode semantics")
def test_posix_project_replace_preserves_the_exact_existing_mode(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "executable.sh"
    target.write_bytes(b"before\n")
    target.chmod(0o751)

    with ProjectFileMutation.open(_workspace(root), "executable.sh") as boundary:
        before, mode, identity = boundary.read(1024)
        assert mode == 0o751
        assert (
            boundary.replace(
                b"after\n",
                expect = before,
                mode = mode,
                identity = identity,
            )
            is None
        )

    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_refuses_parent_symlink_escape(tmp_path):
    root = tmp_path / "repository"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"preserve")
    os.symlink(outside, root / "linked")

    with ProjectFileMutation.open(_workspace(root), "linked/secret.txt") as boundary:
        with pytest.raises((AgentWorkspaceError, OSError)):
            boundary.read(1024)

    assert secret.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_detects_stale_expected_content(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "state.txt"
    target.write_bytes(b"first")

    with ProjectFileMutation.open(_workspace(root), "state.txt") as boundary:
        before, mode, identity = boundary.read(1024)
        target.write_bytes(b"second")
        assert (
            boundary.replace(
                b"third",
                expect = before,
                mode = mode,
                identity = identity,
            )
            == "changed"
        )

    assert target.read_bytes() == b"second"
    assert not list(root.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name == "nt", reason = "POSIX descriptor mutation")
def test_posix_project_mutation_enforces_payload_bound_without_residue(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    with ProjectFileMutation.open(_workspace(root), "bounded.txt", max_bytes = 4) as boundary:
        with pytest.raises(OverflowError, match = "configured limit"):
            boundary.create(b"12345")

    assert not (root / "bounded.txt").exists()
    assert not list(root.glob(".unsloth_edit_*"))


@pytest.mark.parametrize("creating", [False, True])
def test_sandboxed_edit_file_dispatches_through_project_mutation(creating, tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "module.py"
    if not creating:
        target.write_bytes(b"value = 1\n")
    session_id = _bind_project(monkeypatch, root)
    real_open = ProjectFileMutation.open
    observed = []

    def tracking_open(
        workspace,
        path,
        *,
        max_bytes,
        cancel_event = None,
    ):
        observed.append((workspace, path, max_bytes))
        return real_open(workspace, path, max_bytes = max_bytes)

    monkeypatch.setattr(ProjectFileMutation, "open", tracking_open)
    edit = (
        {"old_string": "", "new_string": "value = 1\n"}
        if creating
        else {"old_string": "value = 1", "new_string": "value = 2"}
    )

    result = tools.execute_tool(
        "edit_file",
        {"path": "module.py", "edits": [edit]},
        session_id = session_id,
    )

    assert not result.startswith("Error:"), result
    assert len(observed) == 1
    assert observed[0][0].root == root
    assert observed[0][1] == "module.py"
    assert target.read_bytes() == (b"value = 1\n" if creating else b"value = 2\n")


def test_full_access_edit_file_keeps_the_explicit_outside_escape_hatch(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"before\n")
    session_id = _bind_project(monkeypatch, root)
    monkeypatch.setattr(
        ProjectFileMutation,
        "open",
        lambda *_args, **_kwargs: pytest.fail("opened a confined mutation in Full access"),
    )

    result = tools.execute_tool(
        "edit_file",
        {
            "path": str(outside),
            "edits": [{"old_string": "before", "new_string": "after"}],
        },
        session_id = session_id,
        disable_sandbox = True,
    )

    assert not result.startswith("Error:"), result
    assert outside.read_bytes() == b"after\n"


class _UnusedWindowsOps:
    def open_existing(self, *_args, **_kwargs):
        pytest.fail("opened a path that lexical validation should reject")

    def close(self, _handle):
        pass


def _windows_workspace() -> ProjectWorkspace:
    return ProjectWorkspace(
        project_id = "secure-tools",
        root = Path(r"C:\project"),
        kind = "managed",
        device_id = 11,
        file_id = 22,
    )


def test_windows_containment_preserves_a_drive_root():
    assert mutation._windows_within(r"C:\project\inside.txt", "C:\\")
    assert not mutation._windows_within(r"D:\project\inside.txt", "C:\\")


@pytest.mark.parametrize(
    "target",
    [
        r"..\outside.txt",
        r"C:\other\outside.txt",
        r"\\server\share\outside.txt",
        r"\\?\C:\other\outside.txt",
        r"inside.txt:stream",
        r"NUL",
        r"con.txt",
        "trailing.",
        "trailing ",
        "control\x01.txt",
    ],
)
def test_windows_project_mutation_rejects_ambiguous_or_escaping_paths_portably(target):
    with pytest.raises((AgentWorkspaceError, WindowsMutationRejected, OSError, ValueError)):
        with ProjectFileMutation.open(
            _windows_workspace(),
            target,
            _windows_ops = _UnusedWindowsOps(),
        ):
            pass


class _ReparseWindowsOps(_UnusedWindowsOps):
    def __init__(self):
        self.closed = []

    def open_existing(self, path, **_kwargs):
        assert mutation._windows_key(path) == mutation._windows_key(r"C:\project")
        return 7

    def info(self, handle):
        assert handle == 7
        return mutation._WindowsHandleInfo(
            attributes = mutation._FILE_ATTRIBUTE_DIRECTORY | mutation._FILE_ATTRIBUTE_REPARSE_POINT,
            identity_options = ((11, 22),),
            size = 0,
            modified_ns = 0,
            final_path = r"C:\project",
        )

    def close(self, handle):
        self.closed.append(handle)


def test_windows_project_mutation_rejects_a_reparse_root_portably():
    operations = _ReparseWindowsOps()

    with pytest.raises(WindowsMutationRejected, match = "reparse"):
        ProjectFileMutation.open(
            _windows_workspace(),
            "inside.txt",
            _windows_ops = operations,
        )

    assert operations.closed == [7]


class _ReparseParentWindowsOps(_UnusedWindowsOps):
    def __init__(self):
        self.paths = {}
        self.closed = []

    def open_existing(self, path, **_kwargs):
        handle = len(self.paths) + 1
        self.paths[handle] = mutation._normalize_windows_path(path)
        return handle

    def info(self, handle):
        path = self.paths[handle]
        reparse = mutation._windows_key(path) == mutation._windows_key(r"C:\project\linked")
        return mutation._WindowsHandleInfo(
            attributes = mutation._FILE_ATTRIBUTE_DIRECTORY
            | (mutation._FILE_ATTRIBUTE_REPARSE_POINT if reparse else 0),
            identity_options = ((11, 33 if reparse else 22),),
            size = 0,
            modified_ns = 0,
            final_path = path,
        )

    def close(self, handle):
        self.closed.append(handle)


def test_windows_project_mutation_rejects_a_reparse_parent_portably():
    operations = _ReparseParentWindowsOps()

    with ProjectFileMutation.open(
        _windows_workspace(),
        r"linked\inside.txt",
        _windows_ops = operations,
    ) as boundary:
        with pytest.raises(WindowsMutationRejected, match = "reparse"):
            boundary.read(1024)

    assert operations.closed


class _ReplacementMetadataWindowsOps:
    def __init__(
        self,
        *,
        attributes = 0x20,
        streams = ("::$DATA",),
    ):
        self.attributes = attributes
        self.streams = tuple(streams)
        self.target_basic = mutation._WindowsBasicMetadata(1, attributes)
        self.target_dacl = mutation._WindowsDacl(True, True, b"stable-dacl")
        self.temp_basic = mutation._WindowsBasicMetadata(10, 0x20)
        self.temp_dacl = mutation._WindowsDacl(True, False, b"inherited-dacl")
        self.applied = []
        self.closed = []

    def open_existing(self, path, **kwargs):
        assert mutation._windows_key(path) == mutation._windows_key(r"C:\project\inside.txt")
        assert kwargs["read_control"] is True
        return 7

    def info(self, handle):
        if handle == 7:
            return mutation._WindowsHandleInfo(
                attributes = self.attributes,
                identity_options = ((11, 33),),
                size = len(b"before"),
                modified_ns = 99,
                final_path = r"C:\project\inside.txt",
            )
        assert handle == 8
        return mutation._WindowsHandleInfo(
            attributes = self.temp_basic.attributes,
            identity_options = ((11, 44),),
            size = len(b"after"),
            modified_ns = 100,
            final_path = r"C:\project\.unsloth_edit_temp",
        )

    def read(self, handle, _limit):
        assert handle == 7
        return b"before"

    def stream_names(self, handle):
        return self.streams if handle == 7 else ("::$DATA",)

    def basic_metadata(self, handle):
        return self.target_basic if handle == 7 else self.temp_basic

    def dacl(self, handle):
        return self.target_dacl if handle == 7 else self.temp_dacl

    def apply_dacl(self, handle, dacl):
        assert handle == 8
        self.applied.append("dacl")
        self.temp_dacl = dacl

    def apply_basic_metadata(self, handle, basic):
        assert handle == 8
        self.applied.append("basic")
        self.temp_basic = basic

    def close(self, handle):
        self.closed.append(handle)


def _portable_windows_replacement_backend(operations):
    backend = object.__new__(mutation._WindowsVerifiedMutation)
    backend.path = r"C:\project"
    backend.parts = ("inside.txt",)
    backend._ops = operations
    backend._closed = False
    return backend


def test_windows_replacement_copies_verified_dacl_and_basic_metadata_portably():
    operations = _ReplacementMetadataWindowsOps()
    backend = _portable_windows_replacement_backend(operations)

    assert backend._copy_verified_replacement_metadata(8, b"before", (11, 33)) is True
    assert operations.applied == ["dacl", "basic"]
    assert operations.temp_dacl.comparison_key == operations.target_dacl.comparison_key
    assert operations.temp_basic == operations.target_basic
    assert operations.closed == [7]


@pytest.mark.parametrize(
    "attributes",
    [
        mutation._FILE_ATTRIBUTE_COMPRESSED,
        mutation._FILE_ATTRIBUTE_ENCRYPTED,
        mutation._FILE_ATTRIBUTE_SPARSE_FILE,
    ],
)
def test_windows_replacement_rejects_unsupported_storage_attributes_portably(attributes):
    operations = _ReplacementMetadataWindowsOps(attributes = attributes)
    backend = _portable_windows_replacement_backend(operations)

    with pytest.raises(WindowsMutationRejected, match = "Compressed, encrypted, and sparse"):
        backend._copy_verified_replacement_metadata(8, b"before", (11, 33))

    assert operations.applied == []
    assert operations.closed == [7]


def test_windows_replacement_rejects_named_streams_portably():
    operations = _ReplacementMetadataWindowsOps(streams = ("::$DATA", ":secret:$DATA"))
    backend = _portable_windows_replacement_backend(operations)

    with pytest.raises(WindowsMutationRejected, match = "named streams"):
        backend._copy_verified_replacement_metadata(8, b"before", (11, 33))

    assert operations.applied == []
    assert operations.closed == [7]


@pytest.mark.skipif(os.name != "nt", reason = "native Win32 read-only metadata")
def test_native_windows_read_only_replacement_preserves_target_and_cleans_temp(tmp_path):
    target = tmp_path / "read-only.txt"
    target.write_bytes(b"before")
    target.chmod(0o444)
    try:
        with ProjectFileMutation.open(_workspace(tmp_path), target.name) as boundary:
            before, mode, identity = boundary.read(1024)
            with pytest.raises(WindowsMutationRejected, match = "Read-only"):
                boundary.replace(b"after", expect = before, mode = mode, identity = identity)
        assert target.read_bytes() == b"before"
        assert not list(tmp_path.glob(".unsloth_edit_*"))
    finally:
        target.chmod(0o666)


@pytest.mark.skipif(os.name != "nt", reason = "native Win32 mutation")
@pytest.mark.parametrize("filename", ["a", "a.py", "created.txt", "created-\U0001f9ea.txt"])
def test_native_windows_project_mutation_round_trip(tmp_path, filename):
    root = tmp_path / "repository"
    root.mkdir()
    workspace = _workspace(root)

    with ProjectFileMutation.open(workspace, filename) as boundary:
        assert boundary.create(b"first\r\n") is None

    target = root / filename
    operations = mutation._NativeWindowsMutationOps()
    before_handle = operations.open_existing(
        str(target),
        read_control = True,
        share_write = False,
        share_delete = False,
    )
    try:
        before_basic = operations.basic_metadata(before_handle)
        before_dacl = operations.dacl(before_handle)
    finally:
        operations.close(before_handle)

    with ProjectFileMutation.open(workspace, filename) as boundary:
        before, mode, identity = boundary.read(1024)
        assert before == b"first\r\n"
        assert (
            boundary.replace(
                b"second\r\n",
                expect = before,
                mode = mode,
                identity = identity,
            )
            is None
        )

    assert (root / filename).read_bytes() == b"second\r\n"
    after_handle = operations.open_existing(
        str(target),
        read_control = True,
        share_write = False,
        share_delete = False,
    )
    try:
        after_basic = operations.basic_metadata(after_handle)
        after_dacl = operations.dacl(after_handle)
    finally:
        operations.close(after_handle)
    assert after_basic.attributes == before_basic.attributes
    assert after_basic.creation_time == before_basic.creation_time
    assert after_dacl.comparison_key == before_dacl.comparison_key
    assert not list(root.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name != "nt", reason = "native NTFS junction")
def test_native_windows_project_mutation_rejects_junction_escape(tmp_path):
    root = tmp_path / "repository"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"preserve")
    linked = root / "linked"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(linked), str(outside)],
        check = False,
        capture_output = True,
        text = True,
    )
    assert created.returncode == 0, created.stderr or created.stdout

    with ProjectFileMutation.open(_workspace(root), "linked/secret.txt") as boundary:
        with pytest.raises(WindowsMutationRejected, match = "reparse"):
            boundary.read(1024)

    assert secret.read_bytes() == b"preserve"


@pytest.mark.parametrize("tool_name", ["edit_file", "direct"])
def test_cancelled_edit_does_not_wait_for_or_release_another_commands_slot(
    tool_name, tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    session_id = _bind_project(monkeypatch, tmp_path)
    identity = (workspace.device_id, workspace.file_id)
    assert mutation.acquire_workspace_mutation_slot(identity)
    cancelled = threading.Event()
    cancelled.set()
    try:
        if tool_name == "edit_file":
            result = tools.execute_tool(
                "edit_file",
                {"path": "new.txt", "edits": [{"old_string": "", "new_string": "unsafe"}]},
                session_id = session_id,
                cancel_event = cancelled,
            )
            assert "cancelled" in result.casefold()
        else:
            with pytest.raises(AgentWorkspaceError, match = "cancelled"):
                ProjectFileMutation.open(workspace, "new.txt", cancel_event = cancelled)
        assert identity in mutation._ACTIVE_MUTATION_ROOTS
        assert not (tmp_path / "new.txt").exists()
    finally:
        mutation.release_workspace_mutation_slot(identity)


def test_persisted_managed_workspace_routes_edits_and_holds_deletion_lease(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    project = studio_db.upsert_chat_project(
        {"id": "persisted-secure", "name": "Secure project", "createdAt": 1, "updatedAt": 1}
    )
    session_id = tools.project_session_id(project["id"])
    with common.project_workspace_access(project["id"]) as workspace:
        assert workspace.kind == "managed"
        assert workspace.root == Path(project["sandboxPath"])
        assert tools.wait_for_sessions_idle([session_id], timeout = 0) is False
        result = tools.execute_tool(
            "edit_file",
            {"path": "src/value.py", "edits": [{"old_string": "", "new_string": "VALUE = 1\n"}]},
            session_id = session_id,
        )
        assert result.startswith("Created"), result
        assert (workspace.root / "src/value.py").read_text(encoding = "utf-8") == "VALUE = 1\n"
    assert tools.wait_for_sessions_idle([session_id], timeout = 0) is True

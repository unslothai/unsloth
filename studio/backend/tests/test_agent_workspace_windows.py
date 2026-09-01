# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import ntpath
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.agent_workspace import windows_traversal as windows_traversal_module
from core.agent_workspace.common import AgentWorkspaceError, _terminate_bounded_process
from core.agent_workspace.discovery import (
    build_repository_map,
    secure_repository_traversal_supported,
)
from core.agent_workspace.execution import execution_boundary_status
from core.agent_workspace.instructions import (
    resolve_agents_instructions,
    resolve_repository_instructions,
    resolve_targeted_repository_instructions,
    secure_instruction_traversal_supported,
)
from core.agent_workspace.verification import _shell_argv
from core.agent_workspace.windows_traversal import (
    WindowsTraversalRejected,
    WindowsVerifiedMutation,
    WindowsVerifiedRoot,
    normalize_windows_path,
    windows_path_is_within,
    windows_path_key,
)


class _FakeMutationNode:
    def __init__(
        self,
        identity: int,
        *,
        directory: bool = False,
        reparse: bool = False,
        content: bytes = b"",
    ) -> None:
        self.identity = identity
        self.directory = directory
        self.reparse = reparse
        self.content = content
        self.modified_ns = identity


class _FakeMutationOps:
    def __init__(self) -> None:
        self.nodes: dict[str, tuple[str, _FakeMutationNode]] = {}
        self.handles: dict[int, tuple[str, _FakeMutationNode]] = {}
        self.opens: list[dict[str, object]] = []
        self.writes: list[str] = []
        self.moves: list[tuple[str, str]] = []
        self.replacements: list[tuple[str, str]] = []
        self.next_handle = 100
        self.next_identity = 10
        self.fail_replace = False
        self.add_directory(r"C:\Repo", identity = 1)

    @staticmethod
    def _key(path: str) -> str:
        return windows_path_key(path)

    def add_directory(
        self,
        path: str,
        *,
        identity: int | None = None,
        reparse: bool = False,
    ) -> None:
        assigned = identity if identity is not None else self._allocate_identity()
        normalized = normalize_windows_path(path)
        self.nodes[self._key(path)] = (
            normalized,
            _FakeMutationNode(assigned, directory = True, reparse = reparse),
        )

    def add_file(
        self,
        path: str,
        content: bytes,
        *,
        identity: int | None = None,
        reparse: bool = False,
    ) -> _FakeMutationNode:
        assigned = identity if identity is not None else self._allocate_identity()
        normalized = normalize_windows_path(path)
        node = _FakeMutationNode(
            assigned,
            directory = False,
            reparse = reparse,
            content = content,
        )
        self.nodes[self._key(path)] = (normalized, node)
        return node

    def _allocate_identity(self) -> int:
        value = self.next_identity
        self.next_identity += 1
        return value

    def _new_handle(self, path: str, node: _FakeMutationNode) -> int:
        handle = self.next_handle
        self.next_handle += 1
        self.handles[handle] = (normalize_windows_path(path), node)
        return handle

    def open_existing(
        self,
        path: str,
        *,
        read: bool = False,
        delete: bool = False,
        share_write: bool = True,
        share_delete: bool = False,
    ) -> int:
        entry = self.nodes.get(self._key(path))
        if entry is None:
            raise FileNotFoundError(path)
        canonical, node = entry
        self.opens.append(
            {
                "path": canonical,
                "read": read,
                "delete": delete,
                "share_write": share_write,
                "share_delete": share_delete,
                "directory": node.directory,
            }
        )
        return self._new_handle(canonical, node)

    def create_temp(self, path: str) -> int:
        if self._key(path) in self.nodes:
            raise FileExistsError(path)
        node = self.add_file(path, b"")
        return self._new_handle(path, node)

    def close(self, handle: int) -> None:
        self.handles.pop(handle, None)

    def info(self, handle: int):
        path, node = self.handles[handle]
        attributes = 0
        if node.directory:
            attributes |= windows_traversal_module._FILE_ATTRIBUTE_DIRECTORY
        if node.reparse:
            attributes |= windows_traversal_module._FILE_ATTRIBUTE_REPARSE_POINT
        return windows_traversal_module._HandleInfo(
            attributes = attributes,
            identity_options = ((1, node.identity),),
            size = len(node.content),
            modified_ns = node.modified_ns,
            final_path = path,
        )

    def read(self, handle: int, limit: int) -> bytes:
        return self.handles[handle][1].content[:limit]

    def write_and_flush(self, handle: int, payload: bytes) -> None:
        path, node = self.handles[handle]
        assert ntpath.basename(path).startswith(".unsloth_edit_")
        self.writes.append(path)
        node.content = payload
        node.modified_ns += 1

    def create_directory(self, path: str) -> None:
        if self._key(path) not in self.nodes:
            self.add_directory(path)

    def move_new(self, source: str, target: str) -> bool:
        self.moves.append((source, target))
        if self._key(target) in self.nodes:
            return False
        self.nodes[self._key(target)] = (
            normalize_windows_path(target),
            self.nodes.pop(self._key(source))[1],
        )
        return True

    def replace(self, source: str, target: str) -> None:
        self.replacements.append((source, target))
        if self.fail_replace:
            raise OSError("replace failed")
        if self._key(target) not in self.nodes:
            raise FileNotFoundError(target)
        self.nodes[self._key(target)] = (
            normalize_windows_path(target),
            self.nodes.pop(self._key(source))[1],
        )

    def mark_delete(self, handle: int) -> None:
        path, node = self.handles[handle]
        current = self.nodes.get(self._key(path))
        if current is not None and current[1] is node:
            del self.nodes[self._key(path)]

    def content(self, path: str) -> bytes:
        return self.nodes[self._key(path)][1].content

    def temporary_paths(self) -> list[str]:
        return [
            canonical
            for canonical, _node in self.nodes.values()
            if ntpath.basename(canonical).startswith(".unsloth_edit_")
        ]


def test_windows_path_comparison_is_case_insensitive_and_component_aware():
    assert normalize_windows_path(r"\\?\C:\Repo\src\..\tests") == r"C:\Repo\tests"
    assert windows_path_key(r"C:\Repo\SRC") == windows_path_key(r"c:/repo/src")
    assert windows_path_is_within(r"c:\repo\src\file.py", r"C:\Repo")
    assert not windows_path_is_within(r"C:\Repository\file.py", r"C:\Repo")
    assert not windows_path_is_within(r"D:\Repo\file.py", r"C:\Repo")


def test_windows_shell_contract_is_explicit_and_execution_stays_fail_closed():
    expected = (
        ["cmd.exe", "/d", "/s", "/c", 'echo "%PATH%" & exit /b 7']
        if os.name == "nt"
        else ["/bin/sh", "-c", 'echo "%PATH%" & exit /b 7']
    )
    assert _shell_argv('echo "%PATH%" & exit /b 7') == expected
    windows = execution_boundary_status("win32", probe = False)
    assert windows.available is False
    assert windows.backend is None
    assert "Windows" in str(windows.reason)


def test_windows_mutation_uses_delete_deny_guards_and_atomic_commits():
    ops = _FakeMutationOps()
    ops.add_directory(r"C:\Repo\src", identity = 2)
    original = ops.add_file(r"C:\Repo\src\main.py", b"VALUE = 1\r\n", identity = 3)

    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"C:\Repo\src\main.py",
        (1, 1),
        _ops = ops,
    ) as mutation:
        raw, _attributes, identity = mutation.read(1024)
        assert raw == b"VALUE = 1\r\n"
        assert identity == (1, original.identity)
        assert (
            mutation.replace(
                b"VALUE = 2\r\n",
                expect = raw,
                mode = 0o644,
                identity = identity,
            )
            is None
        )

    assert ops.content(r"C:\Repo\src\main.py") == b"VALUE = 2\r\n"
    assert len(ops.replacements) == 1
    assert all(ntpath.basename(path).startswith(".unsloth_edit_") for path in ops.writes)
    assert not ops.temporary_paths()
    directory_opens = [call for call in ops.opens if call["directory"]]
    assert {windows_path_key(str(call["path"])) for call in directory_opens} >= {
        windows_path_key(r"C:\Repo"),
        windows_path_key(r"C:\Repo\src"),
    }
    assert all(call["share_delete"] is False for call in directory_opens)


def test_windows_mutation_create_is_no_clobber_and_cleans_temporary_files():
    ops = _FakeMutationOps()
    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"src\new.py",
        (1, 1),
        _ops = ops,
    ) as mutation:
        assert mutation.create(b"print('new')\n", 0o666) is None

    assert ops.content(r"C:\Repo\src\new.py") == b"print('new')\n"
    assert len(ops.moves) == 1
    assert not ops.temporary_paths()

    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"src\new.py",
        (1, 1),
        _ops = ops,
    ) as mutation:
        assert mutation.create(b"print('clobber')\n", 0o666) == "exists"

    assert ops.content(r"C:\Repo\src\new.py") == b"print('new')\n"
    assert not ops.temporary_paths()


@pytest.mark.parametrize("stale_kind", ["content", "identity"])
def test_windows_mutation_detects_stale_file_without_committing(stale_kind):
    ops = _FakeMutationOps()
    original = ops.add_file(r"C:\Repo\main.py", b"VALUE = 1\n", identity = 3)
    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"C:\Repo\main.py",
        (1, 1),
        _ops = ops,
    ) as mutation:
        raw, _attributes, identity = mutation.read(1024)
        if stale_kind == "content":
            original.content = b"VALUE = external\n"
            original.modified_ns += 1
        else:
            ops.add_file(r"C:\Repo\main.py", raw, identity = 99)
        assert (
            mutation.replace(
                b"VALUE = 2\n",
                expect = raw,
                mode = 0o644,
                identity = identity,
            )
            == "changed"
        )

    assert not ops.replacements
    assert not ops.temporary_paths()


def test_windows_mutation_rejects_reparse_parent_before_any_write():
    ops = _FakeMutationOps()
    ops.add_directory(r"C:\Repo\linked", identity = 2, reparse = True)
    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"C:\Repo\linked\main.py",
        (1, 1),
        _ops = ops,
    ) as mutation:
        with pytest.raises(WindowsTraversalRejected, match = "reparse"):
            mutation.create(b"VALUE = 1\n")
    assert not ops.writes

    leaf_ops = _FakeMutationOps()
    leaf_ops.add_file(r"C:\Repo\linked.py", b"", identity = 3, reparse = True)
    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"C:\Repo\linked.py",
        (1, 1),
        _ops = leaf_ops,
    ) as mutation:
        with pytest.raises(WindowsTraversalRejected, match = "reparse"):
            mutation.create(b"VALUE = 1\n")
    assert not leaf_ops.writes


def test_windows_mutation_cleans_temp_when_atomic_replace_fails():
    ops = _FakeMutationOps()
    original = ops.add_file(r"C:\Repo\main.py", b"before\n", identity = 3)
    ops.fail_replace = True
    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"C:\Repo\main.py",
        (1, 1),
        _ops = ops,
    ) as mutation:
        with pytest.raises(OSError, match = "replace failed"):
            mutation.replace(
                b"after\n",
                expect = b"before\n",
                mode = 0o644,
                identity = (1, original.identity),
            )
    assert ops.content(r"C:\Repo\main.py") == b"before\n"
    assert not ops.temporary_paths()


def test_windows_mutation_enforces_utf8_and_size_bounds_before_writing():
    ops = _FakeMutationOps()
    with WindowsVerifiedMutation.open(
        r"C:\Repo",
        r"C:\Repo\main.py",
        (1, 1),
        max_bytes = 4,
        _ops = ops,
    ) as mutation:
        with pytest.raises(OverflowError, match = "limit"):
            mutation.create(b"12345")
        with pytest.raises(ValueError, match = "UTF-8"):
            mutation.create(b"\xff")
        with pytest.raises(ValueError, match = "NUL"):
            mutation.create(b"a\x00b")
    assert not ops.writes


@pytest.mark.skipif(os.name != "nt", reason = "Windows handle traversal integration")
def test_windows_handle_traversal_supports_scoped_instructions_and_discovery(tmp_path):
    root = tmp_path / "Workspace with spaces Ω"
    nested = root / "src" / "feature"
    sibling = root / "src" / "sibling"
    nested.mkdir(parents = True)
    sibling.mkdir(parents = True)
    (root / "AGENTS.md").write_text("root rules", encoding = "utf-8")
    (root / "src" / "AGENTS.md").write_text("src rules", encoding = "utf-8")
    (nested / "AGENTS.override.md").write_text("feature rules", encoding = "utf-8")
    (sibling / "AGENTS.md").write_text("sibling rules", encoding = "utf-8")
    (nested / "module.py").write_text("VALUE = 1\n", encoding = "utf-8")
    (root / "ignored.txt").write_text("ignored\n", encoding = "utf-8")
    (root / ".gitignore").write_text("ignored.txt\n", encoding = "utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01\x02")
    identity = (root.stat().st_dev, root.stat().st_ino)

    assert secure_instruction_traversal_supported()
    assert secure_repository_traversal_supported()

    resolved = resolve_agents_instructions(
        root,
        "src/feature/module.py",
        expected_identity = identity,
    )
    assert [layer["scope"] for layer in resolved["layers"]] == [".", "src", "src/feature"]
    assert [layer["content"] for layer in resolved["layers"]] == [
        "root rules",
        "src rules",
        "feature rules",
    ]

    targeted = resolve_targeted_repository_instructions(
        root,
        ["src/feature/module.py"],
        expected_identity = identity,
    )
    assert [layer["scope"] for layer in targeted["layers"]] == [".", "src", "src/feature"]
    assert all(layer["scope"] != "src/sibling" for layer in targeted["layers"])

    repository_wide = resolve_repository_instructions(root, expected_identity = identity)
    assert {layer["scope"] for layer in repository_wide["layers"]} == {
        ".",
        "src",
        "src/feature",
        "src/sibling",
    }

    repository_map = build_repository_map(root, expected_identity = identity)
    paths = {entry["path"] for entry in repository_map["entries"]}
    assert "src/feature/module.py" in paths
    assert "ignored.txt" not in paths
    assert "binary.bin" not in paths
    assert repository_map["skipped"]["binary"] == 1


@pytest.mark.skipif(os.name != "nt", reason = "Windows handle traversal integration")
def test_windows_traversal_accepts_case_variants_and_long_paths(tmp_path):
    root = tmp_path / "Long path workspace"
    root.mkdir()
    identity = (root.stat().st_dev, root.stat().st_ino)
    current = root
    for index in range(7):
        current = current / f"segment-{index}-{'x' * 28}"
        current.mkdir()
    (root / "AGENTS.md").write_text("root", encoding = "utf-8")
    (current / "AGENTS.md").write_text("deep", encoding = "utf-8")
    target = current / "module.py"
    target.write_text("pass\n", encoding = "utf-8")
    assert len(str(target)) > 260

    case_variant = Path(str(root).swapcase())
    result = resolve_agents_instructions(
        case_variant,
        target.relative_to(root).as_posix(),
        expected_identity = identity,
    )
    assert [layer["content"] for layer in result["layers"]] == ["root", "deep"]


@pytest.mark.skipif(os.name != "nt", reason = "Windows handle traversal integration")
def test_windows_root_identity_and_reparse_escape_fail_closed(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding = "utf-8")
    identity = (root.stat().st_dev, root.stat().st_ino)

    with pytest.raises(AgentWorkspaceError, match = "identity"):
        WindowsVerifiedRoot.open(root, (identity[0], identity[1] + 1))

    junction = root / "escape"
    fixture = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        timeout = 10,
        check = False,
    )
    assert fixture.returncode == 0, fixture.stdout
    with pytest.raises(AgentWorkspaceError, match = "reparse"):
        resolve_agents_instructions(
            root,
            "escape/secret.txt",
            expected_identity = identity,
        )
    repository_map = build_repository_map(root, expected_identity = identity)
    assert "escape/secret.txt" not in {entry["path"] for entry in repository_map["entries"]}
    assert repository_map["skipped"]["symlink"] >= 1


@pytest.mark.skipif(os.name != "nt", reason = "Windows mutation integration")
def test_windows_mutation_real_create_replace_stale_and_cleanup(tmp_path):
    root = tmp_path / "Mutation workspace Ω"
    root.mkdir()
    identity = (root.stat().st_dev, root.stat().st_ino)
    target = root / "nested" / "module.py"

    with WindowsVerifiedMutation.open(root, target, identity, max_bytes = 1024) as mutation:
        assert mutation.create(b"VALUE = 1\r\n") is None
        raw, mode, file_identity = mutation.read(1024)
        assert raw == b"VALUE = 1\r\n"
        assert (
            mutation.replace(
                b"VALUE = 2\r\n",
                expect = raw,
                mode = mode,
                identity = file_identity,
            )
            is None
        )
        stale_raw, stale_mode, stale_identity = mutation.read(1024)
        target.write_bytes(b"VALUE = external\r\n")
        assert (
            mutation.replace(
                b"VALUE = stale\r\n",
                expect = stale_raw,
                mode = stale_mode,
                identity = stale_identity,
            )
            == "changed"
        )

    assert target.read_bytes() == b"VALUE = external\r\n"
    assert not list(target.parent.glob(".unsloth_edit_*"))


@pytest.mark.skipif(os.name != "nt", reason = "Windows process-tree integration")
def test_windows_bounded_termination_kills_descendant_tree(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    child_source = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii'); "
        "time.sleep(120)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_source, str(child_pid_file)],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        deadline = time.monotonic() + 10
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text(encoding = "ascii"))
        _terminate_bounded_process(process, None)
        process.wait(timeout = 10)

        import psutil

        deadline = time.monotonic() + 5
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not psutil.pid_exists(child_pid)
    finally:
        if process.poll() is None:
            _terminate_bounded_process(process, None)

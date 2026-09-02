# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
    WindowsVerifiedRoot,
    normalize_windows_path,
    windows_path_is_within,
    windows_path_key,
)


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

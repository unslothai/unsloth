# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import importlib.util
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from core.agent_workspace import git_review
from core.agent_workspace.git_context import AgentWorkspaceError, ProjectWorkspace


def _git(
    root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd = root,
        check = check,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )


def _repository(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    # Stage exact fixture bytes on Windows too. The production Git session
    # intentionally does not inherit the runner's global autocrlf setting.
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "config", "user.email", "review@example.test")
    _git(root, "config", "user.name", "Review Test")
    return root


def _commit_all(root: Path, message: str = "snapshot") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)


def _workspace(path: Path, revision: int = 7) -> ProjectWorkspace:
    metadata = path.stat()
    return ProjectWorkspace(
        project_id = "project-one",
        root = path.resolve(),
        kind = "folder",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
        revision = revision,
    )


def _bind_workspace(monkeypatch: pytest.MonkeyPatch, workspace: ProjectWorkspace) -> None:
    @contextmanager
    def access(project_id: str):
        assert project_id == workspace.project_id
        yield workspace

    monkeypatch.setattr(git_review, "project_workspace_access", access)


def _file(manifest: dict, path: str) -> dict:
    return next(item for item in manifest["files"] if item["path"] == path)


@pytest.mark.skipif(
    os.name == "nt" and importlib.util.find_spec("core.agent_workspace.mutation") is None,
    reason = "Native Windows untracked reads require the secure-tools split",
)
def test_primary_status_and_three_diff_scopes_have_stable_display_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "repository")
    project = repository / "project"
    project.mkdir()
    (project / "tracked.txt").write_text("before\n", encoding = "utf-8")
    (repository / "sibling.txt").write_text("outside\n", encoding = "utf-8")
    _commit_all(repository)

    (project / "tracked.txt").write_text("after\n", encoding = "utf-8")
    (project / "staged.txt").write_text("staged\n", encoding = "utf-8")
    _git(repository, "add", "project/staged.txt")
    (project / "untracked.txt").write_text("untracked\n", encoding = "utf-8")
    (repository / "sibling.txt").write_text("must stay hidden\n", encoding = "utf-8")
    _bind_workspace(monkeypatch, _workspace(project))

    status = git_review.git_status("project-one")
    assert status["target"] == {"kind": "primary"}
    assert status["workspaceRevision"] == 7
    assert status["coherent"] is True
    assert status["counts"] == {"staged": 1, "unstaged": 1, "untracked": 1, "conflicted": 0}
    assert {item["path"] for item in status["files"]} == {
        "tracked.txt",
        "staged.txt",
        "untracked.txt",
    }

    head = git_review.build_diff_manifest("project-one", mode = "head")
    staged = git_review.build_diff_manifest("project-one", mode = "staged")
    unstaged = git_review.build_diff_manifest("project-one", mode = "unstaged")
    assert {item["path"] for item in head["files"]} == {
        "tracked.txt",
        "staged.txt",
        "untracked.txt",
    }
    assert {item["path"] for item in staged["files"]} == {"staged.txt"}
    assert {item["path"] for item in unstaged["files"]} == {"tracked.txt", "untracked.txt"}
    assert _file(head, "untracked.txt")["hunks"][0]["lines"][0]["text"] == "untracked"

    tracked_hunk_id = _file(head, "tracked.txt")["hunks"][0]["id"]
    previous_fingerprint = head["sourceFingerprint"]
    (project / "unrelated.txt").write_text("new\n", encoding = "utf-8")
    changed = git_review.build_diff_manifest("project-one", mode = "head")
    assert _file(changed, "tracked.txt")["hunks"][0]["id"] == tracked_hunk_id
    assert changed["sourceFingerprint"] != previous_fingerprint


@pytest.mark.skipif(os.name == "nt", reason = "POSIX byte filenames are not portable")
def test_pathological_and_invalid_utf8_filenames_are_visible_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "repository")
    project = repository / "project"
    project.mkdir()
    names = [b"line\nbreak.txt", b"literal\\slash.txt"]
    project_bytes = os.fsencode(project)
    for name in names:
        descriptor = os.open(project_bytes + b"/" + name, os.O_WRONLY | os.O_CREAT, 0o600)
        os.write(descriptor, b"before\n")
        os.close(descriptor)
    (repository / "outside.txt").write_text("before\n", encoding = "utf-8")
    _commit_all(repository)
    for name in names:
        descriptor = os.open(project_bytes + b"/" + name, os.O_WRONLY | os.O_TRUNC)
        os.write(descriptor, b"after\n")
        os.close(descriptor)
    (repository / "outside.txt").write_text("hidden\n", encoding = "utf-8")
    _bind_workspace(monkeypatch, _workspace(project))

    manifest = git_review.build_diff_manifest("project-one")
    paths = {item["path"]: item for item in manifest["files"]}
    assert "line\\nbreak.txt" in paths
    assert "literal\\\\slash.txt" in paths
    assert "outside.txt" not in paths
    assert git_review._visible_path(b"invalid-\xff.txt") == ("invalid-\\xFF.txt", "escaped")


def test_cross_boundary_renames_and_copies_render_as_scoped_deletes_or_adds(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    project = repository / "project"
    project.mkdir()
    (project / "inside.txt").write_text("same\n", encoding = "utf-8")
    (repository / "outside.txt").write_text("same\n", encoding = "utf-8")
    _commit_all(repository)
    identity = repository.stat()
    scoped = git_review._Repository(
        repository,
        (int(identity.st_dev), int(identity.st_ino)),
        b"project",
    )

    _git(repository, "mv", "project/inside.txt", "moved-out.txt")
    raw = _git(
        repository,
        "diff",
        "--cached",
        "--raw",
        "-z",
        "--no-abbrev",
        "--find-renames",
    ).stdout
    removed = git_review._parse_raw(scoped, raw)
    assert len(removed) == 1
    assert removed[0]["code"] == b"D"
    assert removed[0]["path"] == b"inside.txt"
    assert removed[0]["oldPath"] is None
    assert removed[0]["scopeBoundary"] is True

    _git(repository, "reset", "--hard", "-q", "HEAD")
    _git(repository, "mv", "outside.txt", "project/moved-in.txt")
    raw = _git(
        repository,
        "diff",
        "--cached",
        "--raw",
        "-z",
        "--no-abbrev",
        "--find-renames",
    ).stdout
    added = git_review._parse_raw(scoped, raw)
    assert len(added) == 1
    assert added[0]["code"] == b"A"
    assert added[0]["path"] == b"moved-in.txt"
    assert added[0]["oldPath"] is None
    assert added[0]["scopeBoundary"] is True
    boundary_file, boundary_hunks, boundary_lines = git_review._file_manifest(
        added[0],
        (
            b"diff --git a/outside.txt b/project/moved-in.txt\n"
            b"--- a/outside.txt\n"
            b"+++ b/project/moved-in.txt\n"
            b"@@ -1 +1 @@\n"
            b"-outside secret\n"
            b"+same\n"
        ),
        mode = "staged",
    )
    assert boundary_file["wholeFileOnly"] is True
    assert boundary_file["hunks"] == []
    assert boundary_file["additions"] == 0
    assert boundary_file["deletions"] == 0
    assert boundary_hunks == 0
    assert boundary_lines == 0

    _git(repository, "reset", "--hard", "-q", "HEAD")
    (project / "copied-in.txt").write_text("same\n", encoding = "utf-8")
    _git(repository, "add", "--", "project/copied-in.txt")
    raw = _git(
        repository,
        "diff",
        "--cached",
        "--raw",
        "-z",
        "--no-abbrev",
        "--find-copies-harder",
    ).stdout
    copied = git_review._parse_raw(scoped, raw)
    assert len(copied) == 1
    assert copied[0]["code"] == b"A"
    assert copied[0]["path"] == b"copied-in.txt"
    assert copied[0]["oldPath"] is None
    assert copied[0]["scopeBoundary"] is True


@pytest.mark.skipif(os.name == "nt", reason = "POSIX link and mode semantics")
def test_binary_invalid_content_rename_mode_and_symlink_are_metadata_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "repository")
    for name in ("binary.dat", "invalid.txt", "mode.txt", "rename.txt"):
        (repository / name).write_bytes(b"before\n")
    (repository / "target-one").write_text("one\n", encoding = "utf-8")
    (repository / "target-two").write_text("two\n", encoding = "utf-8")
    (repository / "link").symlink_to("target-one")
    _commit_all(repository)

    (repository / "binary.dat").write_bytes(b"after\0binary")
    (repository / "invalid.txt").write_bytes(b"after\xff\n")
    (repository / "mode.txt").chmod(0o755)
    (repository / "rename.txt").rename(repository / "renamed.txt")
    _git(repository, "add", "-A", "--", "rename.txt", "renamed.txt")
    (repository / "link").unlink()
    (repository / "link").symlink_to("target-two")
    _bind_workspace(monkeypatch, _workspace(repository))

    manifest = git_review.build_diff_manifest("project-one")
    assert manifest["selectable"] is True
    binary = _file(manifest, "binary.dat")
    invalid = _file(manifest, "invalid.txt")
    mode = _file(manifest, "mode.txt")
    link = _file(manifest, "link")
    renamed = _file(manifest, "renamed.txt")
    assert binary["binary"] is True and binary["wholeFileOnly"] is True
    assert invalid["encoding"] == "invalid-utf8" and invalid["hunks"] == []
    assert mode["modeChanged"] is True and mode["wholeFileOnly"] is True
    assert link["symlink"] is True and link["wholeFileOnly"] is True
    assert renamed["code"].startswith("R") and renamed["oldPath"] == "rename.txt"
    assert renamed["wholeFileOnly"] is True


def test_conflicts_block_hunks_but_keep_visible_conflict_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "repository")
    (repository / "conflict.txt").write_text("base\n", encoding = "utf-8")
    _commit_all(repository)
    _git(repository, "checkout", "-qb", "other")
    (repository / "conflict.txt").write_text("other\n", encoding = "utf-8")
    _commit_all(repository, "other")
    _git(repository, "checkout", "-q", "main")
    (repository / "conflict.txt").write_text("main\n", encoding = "utf-8")
    _commit_all(repository, "main")
    merge = _git(repository, "merge", "other", check = False)
    assert merge.returncode != 0
    _bind_workspace(monkeypatch, _workspace(repository))

    manifest = git_review.build_diff_manifest("project-one")
    assert manifest["selectable"] is False
    assert manifest["blockedReasons"] == ["repository-conflicts"]
    assert manifest["conflictedPaths"] == ["conflict.txt"]
    assert manifest["files"] == []


def test_repository_config_snapshot_retains_local_and_worktree_records_once() -> None:
    snapshot = git_review._repository_scoped_config(
        b"local\0filter.local.process\nfirst\0"
        b"worktree\0diff.worktree.command\nsecond\0"
        b"command\0filter.command.process\nignored\0"
        b"global\0diff.global.command\nignored\0"
    )
    assert snapshot == (
        b"local",
        b"filter.local.process\nfirst",
        b"worktree",
        b"diff.worktree.command\nsecond",
    )


def test_hostile_git_configuration_cannot_execute_hooks_or_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "repository")
    (repository / ".gitattributes").write_text(
        (
            "*.txt filter=evil diff=evil\n"
            "included.dat filter=included diff=included\n"
            "worktree.dat filter=worktree diff=worktree\n"
        ),
        encoding = "utf-8",
    )
    (repository / "tracked.txt").write_text("before\n", encoding = "utf-8")
    (repository / "included.dat").write_text("before\n", encoding = "utf-8")
    (repository / "worktree.dat").write_text("before\n", encoding = "utf-8")
    _commit_all(repository)
    marker = tmp_path / "executed"
    attacker = tmp_path / "attacker.sh"
    attacker.write_text(
        "#!/bin/sh\nprintf hit >> " + shlex.quote(marker.as_posix()) + "\ncat\n",
        encoding = "utf-8",
    )
    attacker.chmod(0o755)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "post-index-change").write_text(attacker.read_text(encoding = "utf-8"), encoding = "utf-8")
    (hooks / "post-index-change").chmod(0o755)
    hostile = attacker.as_posix()
    included_secret = "CONFIG_SECRET_MUST_NOT_ESCAPE"
    excludes = tmp_path / "external-excludes"
    excludes.write_text("untracked.txt\n", encoding = "utf-8")
    order = tmp_path / "external-order"
    order.write_text("worktree.dat\n", encoding = "utf-8")
    included_config = tmp_path / "included.config"
    included_config.write_text(
        '[filter "included"]\n'
        f"\tclean = {hostile}\n"
        f"\tsmudge = {hostile}\n"
        f"\tprocess = {hostile}\n"
        "\trequired = true\n"
        '[diff "included"]\n'
        f"\tcommand = {hostile}\n"
        f"\ttextconv = {hostile}\n"
        "[unsloth]\n"
        f"\tsecret = {included_secret}\n"
        "[core]\n"
        f"\taskPass = {hostile}\n"
        f"\texcludesFile = {excludes.as_posix()}\n"
        "[diff]\n"
        f"\torderFile = {order.as_posix()}\n",
        encoding = "utf-8",
    )
    _git(repository, "config", "include.path", str(included_config))
    hostile_global = tmp_path / "hostile-global.config"
    hostile_global.write_text(
        "[diff]\n\texternal = " + hostile + "\n[pager]\n\tdiff = " + hostile + "\n",
        encoding = "utf-8",
    )
    for name in (
        "GIT_CONFIG_GLOBAL",
        "GIT_EXTERNAL_DIFF",
        "GIT_PAGER",
        "GIT_EDITOR",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_SSH_COMMAND",
    ):
        monkeypatch.setenv(name, str(hostile_global) if name == "GIT_CONFIG_GLOBAL" else hostile)
    for key, value in (
        ("core.hooksPath", str(hooks)),
        ("core.fsmonitor", hostile),
        ("filter.evil.clean", hostile),
        ("filter.evil.smudge", hostile),
        ("filter.evil.process", hostile),
        ("filter.evil.required", "true"),
        ("diff.evil.command", hostile),
        ("diff.evil.textconv", hostile),
        ("diff.external", hostile),
        ("pager.diff", hostile),
        ("core.editor", hostile),
        ("credential.helper", hostile),
        ("gpg.program", hostile),
        ("maintenance.strategy", "incremental"),
    ):
        _git(repository, "config", key, value)
    _git(repository, "config", "extensions.worktreeConfig", "true")
    for key, value in (
        ("core.fsmonitor", hostile),
        ("filter.worktree.clean", hostile),
        ("filter.worktree.smudge", hostile),
        ("filter.worktree.process", hostile),
        ("filter.worktree.required", "true"),
        ("diff.worktree.command", hostile),
        ("diff.worktree.textconv", hostile),
    ):
        _git(repository, "config", "--worktree", key, value)
    (repository / "tracked.txt").write_text("after\n", encoding = "utf-8")
    (repository / "included.dat").write_text("after\n", encoding = "utf-8")
    (repository / "worktree.dat").write_text("after\n", encoding = "utf-8")
    (repository / "untracked.txt").write_text("visible\n", encoding = "utf-8")
    marker.unlink(missing_ok = True)
    _bind_workspace(monkeypatch, _workspace(repository))

    manifest = git_review.build_diff_manifest("project-one")
    can_read_untracked = (
        os.name != "nt" or importlib.util.find_spec("core.agent_workspace.mutation") is not None
    )
    assert manifest["selectable"] is can_read_untracked
    assert _file(manifest, "tracked.txt")["hunks"]
    assert _file(manifest, "included.dat")["hunks"]
    assert _file(manifest, "worktree.dat")["hunks"]
    assert bool(_file(manifest, "untracked.txt")["hunks"]) is can_read_untracked
    rendered = json.dumps(manifest)
    assert included_secret not in rendered
    assert str(included_config) not in rendered
    assert not marker.exists()


def test_submodule_change_is_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _repository(tmp_path / "submodule-source")
    (source / "value.txt").write_text("one\n", encoding = "utf-8")
    _commit_all(source)
    repository = _repository(tmp_path / "repository")
    _git(
        repository,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(source),
        "module",
    )
    _commit_all(repository)
    module = repository / "module"
    _git(module, "config", "user.email", "review@example.test")
    _git(module, "config", "user.name", "Review Test")
    (module / "value.txt").write_text("two\n", encoding = "utf-8")
    _commit_all(module, "advance")
    marker = tmp_path / "submodule-helper-ran"
    attacker = tmp_path / "submodule-attacker.sh"
    attacker.write_text(
        "#!/bin/sh\nprintf hit >> " + shlex.quote(str(marker)) + "\ncat\n",
        encoding = "utf-8",
    )
    attacker.chmod(0o755)
    for key in (
        "core.fsmonitor",
        "diff.external",
        "filter.evil.clean",
        "filter.evil.smudge",
        "filter.evil.process",
    ):
        _git(module, "config", key, str(attacker))
    _git(module, "config", "filter.evil.required", "true")
    (module / ".gitattributes").write_text("* filter=evil diff=evil\n", encoding = "utf-8")
    marker.unlink(missing_ok = True)
    _bind_workspace(monkeypatch, _workspace(repository))

    manifest = git_review.build_diff_manifest("project-one")
    module_file = _file(manifest, "module")
    assert module_file["submodule"] is True
    assert module_file["wholeFileOnly"] is True
    assert module_file["hunks"] == []
    assert not marker.exists()


def test_public_fingerprint_excludes_config_values_but_coherence_keeps_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_repository = _repository(tmp_path / "first")
    (first_repository / "file.txt").write_text("before\n", encoding = "utf-8")
    _commit_all(first_repository)
    second_repository = tmp_path / "second"
    _git(tmp_path, "clone", "-q", str(first_repository), str(second_repository))
    _git(first_repository, "config", "unsloth.secret", "first-secret")
    _git(second_repository, "config", "unsloth.secret", "second-secret")
    for repository_path in (first_repository, second_repository):
        (repository_path / "file.txt").write_text("after\n", encoding = "utf-8")

    _bind_workspace(monkeypatch, _workspace(first_repository))
    first_manifest = git_review.build_diff_manifest("project-one")
    _bind_workspace(monkeypatch, _workspace(second_repository))
    second_manifest = git_review.build_diff_manifest("project-one")
    assert first_manifest["sourceFingerprint"] == second_manifest["sourceFingerprint"]

    class ChangingConfigSession:
        def __init__(self) -> None:
            self.configs = iter((b"secret=first", b"secret=second"))

        def inspect_executable_config(self, repository: Path) -> bytes:
            return next(self.configs)

        def run(self, repository: Path, args, **kwargs):
            if args[0] == "rev-parse":
                output = b"1" * 40 + b"\n"
            elif args[0] == "symbolic-ref":
                output = b"main\n"
            else:
                output = b""
            return git_review._CommandResult(0, output, False, False)

    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    metadata = repository_path.stat()
    workspace = _workspace(repository_path)
    repository = git_review._Repository(
        repository_path,
        (int(metadata.st_dev), int(metadata.st_ino)),
        b"",
    )
    with pytest.raises(AgentWorkspaceError, match = "configuration changed"):
        git_review._capture(
            ChangingConfigSession(),
            workspace,
            repository,
            mode = None,
            max_bytes = 4_096,
        )


@pytest.mark.skipif(
    os.name == "nt" and importlib.util.find_spec("core.agent_workspace.mutation") is None,
    reason = "Native Windows untracked reads require the secure-tools split",
)
def test_truncation_and_incomplete_untracked_content_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "repository")
    (repository / "tracked.txt").write_text("before\n", encoding = "utf-8")
    _commit_all(repository)
    (repository / "tracked.txt").write_text("x" * 20_000 + "\n", encoding = "utf-8")
    _bind_workspace(monkeypatch, _workspace(repository))

    truncated = git_review.build_diff_manifest("project-one", max_bytes = 4_096)
    assert truncated["selectable"] is False
    assert truncated["fingerprintComplete"] is False
    assert truncated["truncated"] is True
    assert truncated["blockedReasons"] == ["git-output-limit"]

    (repository / "tracked.txt").write_text("before\n", encoding = "utf-8")
    (repository / "oversize.txt").write_bytes(b"x" * (git_review.MAX_UNTRACKED_FILE_BYTES + 1))
    (repository / "binary-untracked.dat").write_bytes(b"a\0b")
    (repository / "invalid-untracked.txt").write_bytes(b"a\xffb")
    if os.name != "nt":
        (repository / "untracked-link").symlink_to("tracked.txt")
    manifest = git_review.build_diff_manifest("project-one")
    assert manifest["selectable"] is False
    assert manifest["fingerprintComplete"] is False
    assert manifest["blockedReasons"] == ["untracked-content-incomplete"]
    assert _file(manifest, "oversize.txt")["truncated"] is True
    assert _file(manifest, "oversize.txt")["unavailableReason"] == "oversize"
    assert _file(manifest, "binary-untracked.dat")["binary"] is True
    assert _file(manifest, "binary-untracked.dat")["byteSize"] == 3
    assert _file(manifest, "invalid-untracked.txt")["encoding"] == "invalid-utf8"
    if os.name != "nt":
        link = _file(manifest, "untracked-link")
        assert link["symlink"] is True
        assert link["wholeFileOnly"] is True


def test_coherence_check_detects_concurrent_churn(monkeypatch: pytest.MonkeyPatch) -> None:
    capture_one = git_review._Capture(b"1" * 40, b"main", b"", b"", b"", (), b"")
    capture_two = git_review._Capture(b"1" * 40, b"main", b"x", b"", b"", (), b"")
    captures = iter((capture_one, capture_two))
    monkeypatch.setattr(git_review, "_capture", lambda *args, **kwargs: next(captures))
    identity_checks = {"workspace": 0, "repository": 0}

    def check_workspace(workspace: ProjectWorkspace) -> None:
        identity_checks["workspace"] += 1

    def check_repository(repository: git_review._Repository) -> None:
        identity_checks["repository"] += 1

    monkeypatch.setattr(git_review, "_assert_workspace_identity", check_workspace)
    monkeypatch.setattr(git_review, "_assert_repository_identity", check_repository)
    workspace = ProjectWorkspace("p", Path("/project"), "folder", 1, 2, 0)
    repository = git_review._Repository(Path("/repository"), (1, 3), b"")

    capture, coherent = git_review._coherent_capture(
        object(),
        workspace,
        repository,
        mode = "head",
        max_bytes = 4_096,
    )
    assert capture == capture_one
    assert coherent is False
    assert identity_checks == {"workspace": 3, "repository": 3}


def test_bounded_process_discards_excess_output() -> None:
    result = git_review._run_bounded(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"],
        cwd = Path.cwd(),
        env = {},
        output_limit = 4_096,
        timeout_seconds = 5,
    )
    assert result.overflowed is True
    assert len(result.output) == 4_096


def test_bounded_process_drains_stderr_without_parsing_it_as_stdout() -> None:
    result = git_review._run_bounded(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'valid\\0'); "
            "sys.stderr.buffer.write(b'xcode-select diagnostic\\n')",
        ],
        cwd = Path.cwd(),
        env = {},
        output_limit = 4_096,
        timeout_seconds = 5,
    )
    assert result.code == 0
    assert result.output == b"valid\0"
    assert result.overflowed is False
    assert result.timed_out is False


def test_bounded_process_applies_one_budget_across_stdout_and_stderr() -> None:
    result = git_review._run_bounded(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('o' * 3000); sys.stdout.flush(); "
            "sys.stderr.write('e' * 3000)",
        ],
        cwd = Path.cwd(),
        env = {},
        output_limit = 4_096,
        timeout_seconds = 5,
    )
    assert result.overflowed is True
    assert result.output == b"o" * 3_000


def test_trusted_git_ignores_process_path_and_windows_candidates_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attacker = tmp_path / "git"
    attacker.write_text("attacker", encoding = "utf-8")
    attacker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert git_review._trusted_git_executable() != attacker

    program_files = tmp_path / "Program Files"
    windows_git = program_files / "Git" / "cmd" / "git.exe"
    windows_git.parent.mkdir(parents = True)
    windows_git.write_bytes(b"trusted")
    assert (
        git_review._trusted_git_executable(
            _platform = "nt",
            _windows_roots = [program_files],
        )
        == windows_git.resolve()
    )

    windows_git.unlink()
    windows_git.symlink_to(attacker)
    with pytest.raises(AgentWorkspaceError, match = "trusted system Git"):
        git_review._trusted_git_executable(
            _platform = "nt",
            _windows_roots = [program_files],
        )

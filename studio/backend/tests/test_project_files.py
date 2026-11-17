# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core import project_files
from storage import studio_db
from utils.account_context import OWNER, AccountContext, as_api_caller, run_as

ALICE = AccountContext("a" * 32, "alice")
_OPT_IN = "UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason = "git is not installed")
needs_symlinks = pytest.mark.skipif(
    platform.system() == "Windows", reason = "symlinks need privileges on Windows"
)


@pytest.fixture
def projects_home(tmp_path):
    candidate = tmp_path / "Projects"
    resolved = str(candidate.resolve())
    check = os.path.normcase(resolved) if platform.system() == "Windows" else resolved
    if any(check == p or check.startswith(p + os.sep) for p in studio_db._denied_path_prefixes()):
        candidate = Path.home() / ".unsloth-studio-tests" / uuid.uuid4().hex
    candidate.mkdir(parents = True, exist_ok = True)
    try:
        yield candidate
    finally:
        if ".unsloth-studio-tests" in candidate.parts:
            shutil.rmtree(candidate, ignore_errors = True)


@pytest.fixture
def repo_parent(tmp_path):
    from hub.storage.scan_folders import is_denied_system_path

    if not is_denied_system_path(os.path.realpath(tmp_path)):
        yield tmp_path
        return
    candidate = Path.home() / ".unsloth-studio-tests" / uuid.uuid4().hex
    candidate.mkdir(parents = True)
    try:
        yield candidate
    finally:
        shutil.rmtree(candidate, ignore_errors = True)


@pytest.fixture
def env(tmp_path, monkeypatch, projects_home, repo_parent):
    from core.inference import tools

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects_home))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.delenv(_OPT_IN, raising = False)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    repo = repo_parent / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n")
    os.chmod(repo, 0o755)
    project = studio_db.upsert_chat_project(
        {"id": "p1", "name": "Repo", "createdAt": 1, "updatedAt": 1}
    )
    return SimpleNamespace(tools = tools, repo = repo, project = project, outside = tmp_path)


def _identity(path):
    st = os.stat(path)
    return (st.st_dev, st.st_ino)


def _use_repo(env):
    studio_db.set_chat_project_working_folder("p1", str(env.repo), _identity(env.repo))


def _names(listing):
    return [(e["name"], e["kind"]) for e in listing["entries"]]


def test_listing_puts_folders_first_hides_dotfiles_and_never_shows_git(tmp_path):
    (tmp_path / "b.txt").write_text("bb")
    (tmp_path / "A.md").write_text("a")
    (tmp_path / "zeta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".ssh").mkdir()
    listing = project_files.list_directory(str(tmp_path))
    assert _names(listing) == [
        ("Alpha", "dir"),
        ("zeta", "dir"),
        ("A.md", "file"),
        ("b.txt", "file"),
    ]
    assert listing["entries"][3]["size"] == 2
    assert "size" not in listing["entries"][0]
    assert listing["truncated"] is False

    shown = project_files.list_directory(str(tmp_path), show_hidden = True)
    assert [e["name"] for e in shown["entries"] if e["hidden"]] == [".env"]


def test_listing_caps_entries_and_visits(tmp_path, monkeypatch):
    for i in range(12):
        (tmp_path / f"f{i:02}.txt").write_text("x")
    monkeypatch.setattr(project_files, "LIST_MAX_ENTRIES", 5)
    listing = project_files.list_directory(str(tmp_path))
    assert [e["name"] for e in listing["entries"]] == [f"f{i:02}.txt" for i in range(5)]
    assert listing["truncated"] is True

    monkeypatch.setattr(project_files, "LIST_MAX_ENTRIES", 500)
    monkeypatch.setattr(project_files, "LIST_MAX_VISITED", 3)
    listing = project_files.list_directory(str(tmp_path))
    assert len(listing["entries"]) == 3
    assert listing["truncated"] is True


def test_listing_a_subfolder_normalises_the_path(tmp_path):
    (tmp_path / "src" / "app").mkdir(parents = True)
    (tmp_path / "src" / "app" / "main.py").write_text("x")
    listing = project_files.list_directory(str(tmp_path), "src\\app/")
    assert listing["path"] == "src/app"
    assert _names(listing) == [("main.py", "file")]


def test_missing_root_lists_nothing(tmp_path):
    assert project_files.list_directory(str(tmp_path / "nowhere"))["entries"] == []


@pytest.mark.parametrize(
    "path",
    [
        "..",
        "src/../..",
        "src/../../etc",
        "/etc",
        "\\etc",
        "C:/Windows",
        "c:secret",
        "src\\..\\..",
        "a\x00b",
        "a\nb",
        "x" * 256,
        ".ssh",
        "src/.aws",
        ".git",
        ".GIT/objects",
        "missing",
        "file.txt",
    ],
)
def test_traversal_and_bad_paths_are_refused(tmp_path, path):
    (tmp_path / "src").mkdir()
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".git" / "objects").mkdir(parents = True)
    with pytest.raises(project_files.ProjectPathError):
        project_files.list_directory(str(tmp_path), path)


@needs_symlinks
def test_symlinks_out_of_the_root_are_never_followed(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "inside").mkdir()
    (root / "inside" / "ok.txt").write_text("ok")
    os.symlink(outside, root / "escape")
    os.symlink(outside / "secret.txt", root / "secret-link.txt")
    os.symlink(root / "inside", root / "alias")
    listing = project_files.list_directory(str(root))
    assert _names(listing) == [("alias", "dir"), ("inside", "dir")]
    assert _names(project_files.list_directory(str(root), "alias")) == [("ok.txt", "file")]
    for path in ("escape", "escape/", "inside/../escape"):
        with pytest.raises(project_files.ProjectPathError):
            project_files.list_directory(str(root), path)


def _tree(root, paths):
    for rel in paths:
        target = root / rel
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text("x")


def test_search_ranks_name_matches_before_path_matches(tmp_path):
    _tree(
        tmp_path,
        [
            "main.py",
            "src/main.py",
            "src/main_window.py",
            "lib/domain.py",
            "main/readme.txt",
            "m/a/i/n.txt",
            "other.txt",
        ],
    )
    found = project_files.search_files(str(tmp_path), "main.py")
    assert [r["path"] for r in found["results"]] == [
        "main.py",
        "src/main.py",
        "lib/domain.py",
        "src/main_window.py",
    ]

    found = project_files.search_files(str(tmp_path), "main")
    paths = [r["path"] for r in found["results"]]
    assert paths == [
        "main.py",
        "src/main.py",
        "src/main_window.py",
        "lib/domain.py",
        "main/readme.txt",
        "m/a/i/n.txt",
    ]
    assert found["results"][1]["name"] == "main.py"
    assert project_files.search_files(str(tmp_path), "  ")["results"] == []
    assert [
        r["path"] for r in project_files.search_files(str(tmp_path), "src/main")["results"]
    ] == [
        "src/main.py",
        "src/main_window.py",
    ]


def test_search_limit_is_clamped(tmp_path):
    _tree(tmp_path, [f"f{i}.txt" for i in range(5)])
    assert len(project_files.search_files(str(tmp_path), "f", limit = 0)["results"]) == 1
    assert len(project_files.search_files(str(tmp_path), "f", limit = 10_000)["results"]) == 5


def test_search_skips_dot_dirs_vendored_trees_and_venvs(tmp_path):
    _tree(
        tmp_path,
        [
            "app/target.py",
            ".git/target.py",
            "node_modules/pkg/target.py",
            "__pycache__/target.py",
            "env/lib/target.py",
            ".ssh/target.py",
            "keychains/target.py",
        ],
    )
    (tmp_path / "env" / "pyvenv.cfg").write_text("home = /usr/bin\n")
    found = project_files.search_files(str(tmp_path), "target")
    assert [r["path"] for r in found["results"]] == ["app/target.py"]


@needs_symlinks
def test_search_never_descends_a_symlinked_folder(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    _tree(outside, ["leak.py"])
    _tree(root, ["inside/kept.py"])
    os.symlink(outside, root / "linked")
    os.symlink(root / "inside", root / "alias")
    os.symlink(outside / "leak.py", root / "leak.py")
    found = project_files.search_files(str(root), "py")
    assert [r["path"] for r in found["results"]] == ["inside/kept.py"]


def test_search_bounds_report_truncation(tmp_path, monkeypatch):
    _tree(tmp_path, [f"d{i}/f{i}.txt" for i in range(6)])
    monkeypatch.setattr(project_files, "SEARCH_MAX_DIRS", 3)
    assert project_files.search_files(str(tmp_path), "f")["truncated"] is True

    monkeypatch.setattr(project_files, "SEARCH_MAX_DIRS", 5000)
    monkeypatch.setattr(project_files, "SEARCH_MAX_FILES", 2)
    found = project_files.search_files(str(tmp_path), "f")
    assert found["truncated"] is True
    assert len(found["results"]) == 2

    monkeypatch.setattr(project_files, "SEARCH_MAX_FILES", 50000)
    ticks = iter(range(0, 100, 1))
    found = project_files.search_files(str(tmp_path), "f", clock = lambda: next(ticks) * 1.0)
    assert found["truncated"] is True
    assert project_files.search_files(str(tmp_path), "f")["truncated"] is False


def test_search_bounds_hold_inside_one_large_folder(tmp_path, monkeypatch):
    _tree(tmp_path, [f"f{i}.txt" for i in range(40)] + [f"d{i}/x.txt" for i in range(10)])
    monkeypatch.setattr(project_files, "SEARCH_MAX_ENTRIES", 5)
    found = project_files.search_files(str(tmp_path), "f")
    assert found["truncated"] is True
    assert len(found["results"]) <= 5

    monkeypatch.setattr(project_files, "SEARCH_MAX_ENTRIES", 200000)
    ticks = iter([0.0, 0.0] + [0.0] * 3 + [99.0] * 1000)
    found = project_files.search_files(str(tmp_path), "f", clock = lambda: next(ticks))
    assert found["truncated"] is True
    assert len(found["results"]) <= 3

    monkeypatch.setattr(project_files, "SEARCH_MAX_DIRS", 3)
    assert project_files.search_files(str(tmp_path), "x")["truncated"] is True


def test_porcelain_parsing_covers_every_status():
    data = b"\0".join(
        [
            b" M src/app.py",
            b"A  new.py",
            b" D gone.py",
            b"R  renamed.py",
            b"old name.py",
            b"?? notes/todo.md",
            b"UU both.py",
            b"AA added-twice.py",
            b"MM staged-and-not.py",
            b"!! ignored.log",
            b"",
        ]
    )
    assert project_files.parse_porcelain(data) == [
        {"path": "src/app.py", "status": "modified"},
        {"path": "new.py", "status": "added"},
        {"path": "gone.py", "status": "deleted"},
        {"path": "renamed.py", "status": "renamed"},
        {"path": "notes/todo.md", "status": "untracked"},
        {"path": "both.py", "status": "conflicted"},
        {"path": "added-twice.py", "status": "conflicted"},
        {"path": "staged-and-not.py", "status": "modified"},
    ]


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd = repo,
        check = True,
        capture_output = True,
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo)},
    )


def _init_repo(path):
    path.mkdir(parents = True, exist_ok = True)
    _git(path, "init", "-q", "-b", "main")
    for name in ("keep.py", "edit.py", "drop.py", "move.py"):
        (path / name).write_text(f"{name}\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")


@needs_git
def test_changes_of_a_real_repository(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "edit.py").write_text("changed\n")
    (repo / "drop.py").unlink()
    _git(repo, "mv", "move.py", "moved.py")
    (repo / "staged.py").write_text("new\n")
    _git(repo, "add", "staged.py")
    (repo / "dir").mkdir()
    (repo / "dir" / "loose.txt").write_text("x")
    out = project_files.git_changes(str(repo))
    assert out["repo"] is True
    assert out["branch"] == "main"
    assert sorted((c["path"], c["status"]) for c in out["changes"]) == [
        ("dir/loose.txt", "untracked"),
        ("drop.py", "deleted"),
        ("edit.py", "modified"),
        ("moved.py", "renamed"),
        ("staged.py", "added"),
    ]
    assert out["truncated"] is False


@needs_git
def test_changes_are_capped(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    for i in range(4):
        (repo / f"n{i}.txt").write_text("x")
    monkeypatch.setattr(project_files, "CHANGES_MAX", 2)
    out = project_files.git_changes(str(repo))
    assert len(out["changes"]) == 2
    assert out["truncated"] is True


def test_a_plain_folder_or_missing_git_is_not_a_repository(tmp_path, monkeypatch):
    none = {"repo": False, "branch": None, "changes": [], "truncated": False}
    assert project_files.git_changes(str(tmp_path)) == none
    assert project_files.git_changes(str(tmp_path / "missing")) == none
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(project_files.shutil, "which", lambda name: None)
    assert project_files.git_changes(str(tmp_path)) == none


@needs_git
def test_a_broken_repository_or_timeout_is_not_an_error(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    none = {"repo": False, "branch": None, "changes": [], "truncated": False}
    assert project_files.git_changes(str(tmp_path)) == none

    repo = tmp_path / "repo"
    _init_repo(repo)

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(project_files.subprocess, "run", slow)
    assert project_files.git_changes(str(repo)) == none


@needs_git
def test_git_runs_with_a_minimal_environment(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    seen = []
    real_run = subprocess.run

    def spy(argv, **kwargs):
        seen.append((argv, kwargs))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(project_files.subprocess, "run", spy)
    assert project_files.git_changes(str(repo))["repo"] is True
    for argv, kwargs in seen:
        assert kwargs["shell"] is False
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["timeout"] <= project_files.GIT_TIMEOUT_SECONDS
        assert "HF_TOKEN" not in kwargs["env"]
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert kwargs["env"]["GIT_CEILING_DIRECTORIES"] == os.path.dirname(os.path.realpath(repo))
        assert "core.fsmonitor=false" in argv
        assert "core.hooksPath=/dev/null" in argv


@needs_git
@pytest.mark.skipif(platform.system() == "Windows", reason = "uses a shell script")
def test_a_repositorys_config_cannot_make_git_run_a_program(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    marker = tmp_path / "ran"
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\ncat\n")
    script.chmod(0o755)
    _git(repo, "config", "core.fsmonitor", str(script))
    _git(repo, "config", "filter.evil.clean", str(script))
    _git(repo, "config", "filter.evil.process", str(script))
    (repo / ".gitattributes").write_text("*.py filter=evil\n")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok = True)
    for hook in ("post-index-change", "pre-commit"):
        (hooks / hook).write_text(f"#!/bin/sh\ntouch '{marker}'\n")
        (hooks / hook).chmod(0o755)
    os.utime(repo / "keep.py", (1, 1))
    (repo / "edit.py").write_text("changed\n")
    out = project_files.git_changes(str(repo))
    assert out["repo"] is True
    assert ("edit.py", "modified") in [(c["path"], c["status"]) for c in out["changes"]]
    assert not marker.exists()
    subprocess.run(["git", "status", "--porcelain"], cwd = repo, capture_output = True)
    assert marker.exists()


def _routes():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from routes import chat_history

    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[chat_history.get_current_subject] = lambda: "owner-test"
    return TestClient(app)


def test_routes_list_the_managed_folder_until_one_is_chosen(env):
    with _routes() as client:
        assert client.get("/api/chat/projects/nope/files").status_code == 404
        assert client.get("/api/chat/projects/nope/changes").status_code == 404
        assert (
            client.get("/api/chat/projects/nope/files/search", params = {"q": "x"}).status_code == 404
        )

        managed = client.get("/api/chat/projects/p1/files")
        assert managed.status_code == 200, managed.text
        assert managed.json()["chosen"] is False
        assert "main.py" not in [e["name"] for e in managed.json()["entries"]]

        _use_repo(env)
        chosen = client.get("/api/chat/projects/p1/files").json()
        assert chosen["chosen"] is True
        assert chosen["path"] == ""
        assert [e["name"] for e in chosen["entries"]] == ["main.py"]
        found = client.get("/api/chat/projects/p1/files/search", params = {"q": "main"}).json()
        assert found == {"results": [{"path": "main.py", "name": "main.py"}], "truncated": False}
        for path in ("..", "../..", "/etc", "..\\..", str(env.outside)):
            response = client.get("/api/chat/projects/p1/files", params = {"path": path})
            assert response.status_code == 404, path
        changes = client.get("/api/chat/projects/p1/changes").json()
        assert changes == {"repo": False, "branch": None, "changes": [], "truncated": False}


@pytest.mark.parametrize("missing", ["root", "sandbox"])
@pytest.mark.parametrize("endpoint", ["files", "files/search?q=keep", "changes"])
def test_routes_report_an_offline_managed_workspace_without_recreating_it(env, missing, endpoint):
    root = Path(env.project["rootPath"])
    marker = root / "sandbox" / "keep.txt"
    marker.write_text("original project contents")
    absent = root if missing == "root" else root / "sandbox"
    offline = absent.with_name(absent.name + "-offline")
    absent.rename(offline)
    before = studio_db.get_chat_project("p1")

    with _routes() as client:
        response = client.get(f"/api/chat/projects/p1/{endpoint}")

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert str(absent) in detail
        assert "unavailable" in detail.lower()
        assert "reconnect" in detail.lower() or "restore" in detail.lower()
        assert not absent.exists()
        assert studio_db.get_chat_project("p1") == before
        assert (offline / marker.relative_to(absent)).read_text() == "original project contents"

        offline.rename(absent)
        restored = client.get(f"/api/chat/projects/p1/{endpoint}")
        assert restored.status_code == 200, restored.text
        assert marker.read_text() == "original project contents"


@needs_git
def test_routes_keep_a_linked_folder_usable_when_the_managed_root_is_offline(env):
    _init_repo(env.repo)
    _use_repo(env)
    (env.repo / "main.py").write_text("print('changed')\n")
    root = Path(env.project["rootPath"])
    root.rename(root.with_name(root.name + "-offline"))

    with _routes() as client:
        listing = client.get("/api/chat/projects/p1/files")
        assert listing.status_code == 200, listing.text
        assert listing.json()["chosen"] is True
        assert "main.py" in [entry["name"] for entry in listing.json()["entries"]]
        found = client.get("/api/chat/projects/p1/files/search", params = {"q": "main"})
        assert found.status_code == 200, found.text
        assert found.json()["results"] == [{"path": "main.py", "name": "main.py"}]
        changes = client.get("/api/chat/projects/p1/changes")
        assert changes.status_code == 200, changes.text
        assert changes.json()["repo"] is True
        assert {"path": "main.py", "status": "modified"} in changes.json()["changes"]

    assert not root.exists()


def test_an_api_key_lists_the_managed_folder_not_the_chosen_one(env):
    from routes import chat_history

    _use_repo(env)
    owner = chat_history.list_project_files("p1", path = "", show_hidden = False, current_subject = "u")
    assert owner.chosen is True
    api = run_as(
        as_api_caller(OWNER),
        chat_history.list_project_files,
        "p1",
        path = "",
        show_hidden = False,
        current_subject = "u",
    )
    assert api.chosen is False
    assert "main.py" not in [e.name for e in api.entries]
    found = run_as(
        as_api_caller(OWNER),
        chat_history.search_project_files,
        "p1",
        q = "main",
        limit = 50,
        current_subject = "u",
    )
    assert found.results == []


def test_a_managed_account_never_reaches_the_owners_folder(env, monkeypatch):
    from routes import chat_history

    _use_repo(env)
    with pytest.raises(HTTPException) as caught:
        run_as(
            ALICE,
            chat_history.list_project_files,
            "p1",
            path = "",
            show_hidden = False,
            current_subject = "u",
        )
    assert caught.value.status_code == 404

    monkeypatch.setattr(env.tools, "is_owner_context", lambda: False)
    monkeypatch.setattr(studio_db, "is_owner_context", lambda: False)
    listing = chat_history.list_project_files("p1", path = "", show_hidden = False, current_subject = "u")
    assert listing.chosen is False
    assert "main.py" not in [e.name for e in listing.entries]


def _grant(path):
    return SimpleNamespace(canonical_path = path)


def _create(env, monkeypatch, project_id, reuse):
    from routes import chat_history
    from utils import native_path_leases

    monkeypatch.setattr(
        native_path_leases, "verify_native_path_lease", lambda lease, **checks: _grant(env.repo)
    )
    body = chat_history.ChatProjectCreate(
        id = project_id,
        name = "repo",
        createdAt = 2,
        updatedAt = 2,
        nativePathLease = "picked",
        reuseExisting = reuse,
    )
    return chat_history.save_project(body, current_subject = "u")


def test_open_folder_reuses_the_project_already_in_that_folder(env, monkeypatch):
    first = _create(env, monkeypatch, "p2", True)
    assert first.id == "p2"
    assert first.workingFolder == str(env.repo)
    again = _create(env, monkeypatch, "p3", True)
    assert again.id == "p2"
    assert again.name == "repo"
    assert studio_db.get_chat_project("p3") is None
    assert "reuseExisting" not in again.model_dump()


def test_without_reuse_a_second_project_is_created(env, monkeypatch):
    _create(env, monkeypatch, "p2", True)
    second = _create(env, monkeypatch, "p3", False)
    assert second.id == "p3"
    assert second.workingFolder == str(env.repo)


def test_reuse_ignores_archived_projects_and_other_folders(env, monkeypatch):
    other = env.repo.parent / "other"
    other.mkdir()
    studio_db.set_chat_project_working_folder("p1", str(other), _identity(other))
    archived = _create(env, monkeypatch, "p2", True)
    studio_db.update_chat_project("p2", {"archived": True})
    created = _create(env, monkeypatch, "p3", True)
    assert created.id == "p3"
    assert archived.id == "p2"


def test_reuse_repins_a_folder_recreated_at_the_same_path(env, monkeypatch):
    _create(env, monkeypatch, "p2", True)
    env.repo.rename(env.repo.with_name("repo-old"))
    env.repo.mkdir()
    assert studio_db.project_working_folder_status(studio_db.get_chat_project("p2")) == "replaced"
    again = _create(env, monkeypatch, "p3", True)
    assert again.id == "p2"
    assert again.workingFolderStatus == "ok"


def test_reuse_still_requires_the_owner(env, monkeypatch):
    from routes import chat_history

    monkeypatch.setattr(chat_history, "is_owner_context", lambda: False)
    with pytest.raises(HTTPException) as caught:
        _create(env, monkeypatch, "p2", True)
    assert caught.value.status_code == 403

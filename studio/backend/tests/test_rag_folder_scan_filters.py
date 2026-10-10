# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a linked-folder scan leaves out (.gitignore, build output, oversized text, the file cap),
and what each pass reports back."""

from __future__ import annotations

import json
import time
from contextlib import closing
from pathlib import Path

import pytest

from core.rag import config, folder_sync, gitignore
from storage import rag_db

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


def _ignored(
    rules_text: str,
    rel: str,
    is_dir: bool = False,
    base: str = "",
) -> bool:
    return gitignore.is_ignored(rel, is_dir, tuple(gitignore.parse(rules_text, base)))


@pytest.mark.parametrize(
    "rules, rel, is_dir, expected",
    [
        ("*.log", "app.log", False, True),
        ("*.log", "deep/nested/app.log", False, True),
        ("*.log", "app.log.md", False, False),
        ("/root.md", "root.md", False, True),
        ("/root.md", "sub/root.md", False, False),
        ("docs/*.md", "docs/a.md", False, True),
        ("docs/*.md", "docs/sub/a.md", False, False),
        ("docs/*.md", "other/docs/a.md", False, False),
        ("generated/", "generated", True, True),
        ("generated/", "generated", False, False),
        ("**/fixtures", "a/b/fixtures", True, True),
        ("a/**/b.md", "a/b.md", False, True),
        ("a/**/b.md", "a/x/y/b.md", False, True),
        ("logs/**", "logs/today.txt", False, True),
        ("*.md\n!keep.md", "keep.md", False, False),
        ("*.md\n!keep.md", "drop.md", False, True),
        ("# comment\n\n", "anything.md", False, False),
        ("\\#hash.md", "#hash.md", False, True),
        ("file?.md", "file1.md", False, True),
        ("file[0-9].md", "file7.md", False, True),
        ("file[!0-9].md", "file7.md", False, False),
        ("a[!x]b", "a/b", False, False),
        ("*a*b", "xaab", False, True),
        ("*a*b", "xaa/b", False, False),
        ("*.min.*", "app.min.js", False, True),
        ("*.min.*", "app.js", False, False),
        ("a*b*c", "abbc", False, True),
        ("a*b*c", "acbc", False, True),
        ("a*b*c", "acb", False, False),
        ("**/x/**/y/*.md", "p/x/q/r/y/a.md", False, True),
    ],
)
def test_gitignore_patterns(rules, rel, is_dir, expected):
    assert _ignored(rules, rel, is_dir) is expected


def test_a_malformed_class_drops_only_its_own_rule():
    rules = "[z-a].txt\n*.log"
    assert not _ignored(rules, "a.txt")
    assert _ignored(rules, "debug.log")


def test_hostile_rules_match_in_bounded_time():
    star_rule = gitignore.parse("*a" * 15 + "b", "")
    globstar_rule = gitignore.parse("a/**/" * 7 + "b", "")
    assert len(star_rule) == 1
    # Too many "**/" to match in bounded time: the rule is left out.
    assert globstar_rule == []
    started = time.monotonic()
    assert not gitignore.is_ignored("a" * 46, False, tuple(star_rule))
    assert time.monotonic() - started < 1


def test_nested_gitignore_applies_only_below_its_directory():
    assert _ignored("*.md", "sub/a.md", base = "sub")
    assert not _ignored("*.md", "a.md", base = "sub")
    assert not _ignored("*.md", "subway/a.md", base = "sub")


@pytest.fixture
def source(linkable_temp_base, tmp_path) -> Path:
    root = linkable_temp_base / f"scan-{tmp_path.name}"
    root.mkdir(parents = True)
    return root


def _write(
    root: Path,
    rel: str,
    text: str = "words",
) -> None:
    path = root / rel
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def test_scan_honours_root_and_nested_gitignore(source):
    _write(source, ".gitignore", "*.log\ngenerated/\n!important.log\n")
    _write(source, "notes.md")
    _write(source, "debug.log")
    _write(source, "important.log")
    _write(source, "generated/out.md")
    _write(source, "pkg/.gitignore", "local.md\n")
    _write(source, "pkg/local.md")
    _write(source, "pkg/kept.md")
    _write(source, "local.md")
    report: dict[str, int] = {}
    found, _ = folder_sync._scan(str(source), report = report)
    assert set(found) == {"notes.md", "important.log", "pkg/kept.md", "local.md"}
    # debug.log, the generated/ directory and pkg/local.md
    assert report["gitignored"] == 3


def test_an_unreadable_gitignore_fails_the_scan(source, monkeypatch):
    _write(source, ".gitignore", "secret.md\n")
    _write(source, "secret.md")
    real_open = open

    def locked_open(path, *args, **kwargs):
        if str(path).endswith(".gitignore"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", locked_open)
    with pytest.raises(RuntimeError, match = "Couldn't read .gitignore"):
        folder_sync._scan(str(source))


def test_a_bom_does_not_hide_the_first_gitignore_rule(source):
    (source / ".gitignore").write_bytes(b"\xef\xbb\xbfsecret.md\nother.md\n")
    _write(source, "secret.md")
    _write(source, "other.md")
    _write(source, "notes.md")
    found, _ = folder_sync._scan(str(source))
    assert set(found) == {"notes.md"}


def test_scan_skips_build_output_only_beside_a_project_manifest(source):
    _write(source, "app/package.json", "{}")
    _write(source, "app/dist/bundle.js", "bundled")
    _write(source, "app/src/index.js", "source")
    _write(source, "vault/build/plan.md", "the user's own build notes")
    report: dict[str, int] = {}
    found, _ = folder_sync._scan(str(source), report = report)
    assert set(found) == {"app/package.json", "app/src/index.js", "vault/build/plan.md"}
    assert report["build_dirs"] == 1


def test_scan_skips_oversized_plain_text_but_not_documents(source, monkeypatch):
    monkeypatch.setattr(config, "FOLDER_MAX_TEXT_BYTES", 10)
    _write(source, "dump.csv", "x" * 50)
    _write(source, "small.csv", "a,b")
    _write(source, "big.md", "x" * 50)
    report: dict[str, int] = {}
    found, _ = folder_sync._scan(str(source), report = report)
    # .md is a document type, not a SOURCE_TEXT_EXTS data dump.
    assert set(found) == {"small.csv", "big.md"}
    assert report["too_large"] == 1


def test_scan_skips_empty_files(source):
    _write(source, "pkg/__init__.py", "")
    _write(source, "pkg/core.py", "def core(): pass")
    found, _ = folder_sync._scan(str(source))
    assert set(found) == {"pkg/core.py"}


def test_scan_keeps_the_first_files_in_sorted_order_past_the_cap(source, monkeypatch):
    monkeypatch.setattr(config, "FOLDER_MAX_FILES", 3)
    for name in ("e.md", "a.md", "d.md", "b.md", "c.md"):
        _write(source, name)
    report: dict[str, int] = {}
    found, _ = folder_sync._scan(str(source), report = report)
    assert set(found) == {"a.md", "b.md", "c.md"}
    assert report["over_limit"] == 2


def test_scan_still_refuses_a_folder_far_past_the_cap(source, monkeypatch):
    monkeypatch.setattr(config, "FOLDER_MAX_FILES", 2)
    for index in range(5):
        _write(source, f"{index}.md")
    with pytest.raises(RuntimeError, match = "supported files limit"):
        folder_sync._scan(str(source))


def _folder_row(folder_id: str) -> dict:
    with closing(rag_db.get_connection()) as conn:
        return dict(
            conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
        )


def _run(folder_id: str) -> dict:
    job_id = folder_sync.request_sync(folder_id)
    folder_sync.reconcile_folder(job_id)
    return folder_sync.get_job(job_id)


@requires_sqlite_vec
def test_a_pass_that_changes_nothing_leaves_last_change_at_alone(rag_home, stub_embeddings):
    source = rag_home / "source"
    source.mkdir()
    (source / "notes.txt").write_text("alpha words", encoding = "utf-8")
    folder = folder_sync.create_folder(
        scope_type = "project", scope_id = "p1", path = str(source), name = "Docs"
    )
    assert _run(folder["id"])["status"] == "completed"
    first = _folder_row(folder["id"])
    assert first["last_change_at"]

    assert _run(folder["id"])["status"] == "completed"
    idle = _folder_row(folder["id"])
    assert idle["last_change_at"] == first["last_change_at"]

    (source / "more.txt").write_text("beta words", encoding = "utf-8")
    assert _run(folder["id"])["added"] == 1
    assert _folder_row(folder["id"])["last_change_at"] != first["last_change_at"]


@requires_sqlite_vec
def test_every_failed_file_is_listed_with_its_reason(rag_home, stub_embeddings, monkeypatch):
    source = rag_home / "source"
    source.mkdir()
    for index in range(5):
        (source / f"bad{index}.txt").write_text("words", encoding = "utf-8")
    folder = folder_sync.create_folder(
        scope_type = "project", scope_id = "p1", path = str(source), name = "Docs"
    )
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )
    job = _run(folder["id"])
    assert job["status"] == "failed"
    assert "and 2 more" in job["error"]

    report = json.loads(_folder_row(folder["id"])["scan_report"])
    assert report["failureCount"] == 5
    assert [entry["path"] for entry in report["failures"]] == [f"bad{i}.txt" for i in range(5)]
    assert all(entry["error"] == "embed unavailable" for entry in report["failures"])

    from routes.rag import _folder_view

    view = _folder_view(_folder_row(folder["id"]))
    assert view["failureCount"] == 5
    assert view["failures"][0] == {"path": "bad0.txt", "error": "embed unavailable"}

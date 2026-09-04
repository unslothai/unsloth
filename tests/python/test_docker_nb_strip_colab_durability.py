# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""A cleaned notebook must never outlive the record of what it was cleaned to.

The migration published every notebook and then wrote the state once at the end. Lose
that single write and every cleaned notebook no longer matches its record, so the
refresh reads a hash mismatch as a user edit, carries the stale record forward and
stops applying upstream updates to it for good, while the migration prints success and
exits 0. On the first boot after this ships the migration touches the whole set at
once, so a docker stop or an ENOSPC anywhere in that window stranded all of them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STRIP_PATH = REPO_ROOT / "docker" / "unsloth_nb_strip_colab.py"

INTRO = (
    'To run this, press "*Runtime*" and press "*Run all*" on a **free** '
    "Tesla T4 Google Colab instance!\n"
)
# json.dumps escapes the quotes in INTRO, so the raw sentence never appears in the file
# text; this fragment does, and only in the intro.
MARK = "Tesla T4 Google Colab instance"


@pytest.fixture(scope = "module")
def strip():
    assert STRIP_PATH.is_file(), f"missing {STRIP_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_nb_strip_durability", STRIP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook(path: Path, tag: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": [INTRO], "metadata": {}},
                    {
                        "cell_type": "code",
                        "source": [f"print({tag!r})\n"],
                        "metadata": {},
                        "outputs": [],
                        "execution_count": None,
                    },
                ],
                "metadata": {"widgets": {"stale": True}},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding = "utf-8",
    )


@pytest.fixture
def tree(tmp_path: Path):
    dest = tmp_path / "dest"
    names = [f"nb/n{i}.ipynb" for i in range(4)]
    lines = []
    for i, rel in enumerate(names):
        _notebook(dest / rel, f"cell-{i}")
        lines.append(f"{_sha256(dest / rel)}  {rel}")
    state = dest / ".unsloth_sync_state"
    state.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return dest, state, names


def _state(state: Path) -> dict:
    out = {}
    for line in state.read_text(encoding = "utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def _orphans(dest: Path, state: Path, names) -> list:
    """Notebooks on disk whose bytes no longer match what the state file records.
    Every one of these is permanently frozen from the refresh's point of view."""
    recorded = _state(state)
    return [rel for rel in names if recorded.get(rel) != _sha256(dest / rel)]


def test_the_migration_records_what_it_published(strip, tree):
    dest, state, names = tree
    assert strip.migrate(str(state), str(dest)) == 0
    assert _orphans(dest, state, names) == []
    for rel in names:
        assert MARK not in (dest / rel).read_text(encoding = "utf-8")


def test_a_failing_state_write_publishes_nothing(strip, tree, monkeypatch):
    """The whole point of the ordering: if the record cannot be written, the notebook
    must not be cleaned either, so the disk still matches and the next start retries."""
    dest, state, names = tree
    monkeypatch.setattr(strip, "_write_state", lambda *a, **k: False)

    assert strip.migrate(str(state), str(dest)) == 0

    assert _orphans(dest, state, names) == []
    for rel in names:
        assert MARK in (dest / rel).read_text(encoding = "utf-8"), rel
    assert not list((dest / "nb").glob("*.tmp")), "a staged copy was left behind"


def test_a_state_write_that_fails_partway_strands_nothing(strip, tree, monkeypatch):
    """The realistic shape: the disk fills, or the container is stopped, midway."""
    dest, state, names = tree
    real = strip._write_state
    calls = {"n": 0}

    def _fail_after_two(path, lines):
        calls["n"] += 1
        if calls["n"] > 2:
            return False
        return real(path, lines)

    monkeypatch.setattr(strip, "_write_state", _fail_after_two)

    assert strip.migrate(str(state), str(dest)) == 0

    assert _orphans(dest, state, names) == []
    cleaned = [rel for rel in names if MARK not in (dest / rel).read_text(encoding = "utf-8")]
    assert len(cleaned) == 2, cleaned
    assert not list((dest / "nb").glob("*.tmp"))


def test_the_rest_is_cleaned_on_the_next_start(strip, tree, monkeypatch):
    dest, state, names = tree
    real = strip._write_state
    calls = {"n": 0}

    def _fail_after_two(path, lines):
        calls["n"] += 1
        return real(path, lines) if calls["n"] <= 2 else False

    monkeypatch.setattr(strip, "_write_state", _fail_after_two)
    strip.migrate(str(state), str(dest))
    monkeypatch.setattr(strip, "_write_state", real)

    assert strip.migrate(str(state), str(dest)) == 0

    assert _orphans(dest, state, names) == []
    for rel in names:
        assert MARK not in (dest / rel).read_text(encoding = "utf-8"), rel


def test_a_notebook_that_fails_to_publish_gives_its_record_back(strip, tree, monkeypatch):
    """The inverted window. It is one notebook rather than all of them, and it is
    undone by rewriting that entry, which is why this ordering is the safe one."""
    dest, state, names = tree
    before = dict(_state(state))
    monkeypatch.setattr(strip, "_publish", lambda tmp, path, prev: strip._unlink(tmp) or False)

    assert strip.migrate(str(state), str(dest)) == 0

    assert _state(state) == before, "a record was left pointing at bytes never published"
    assert _orphans(dest, state, names) == []
    assert not list((dest / "nb").glob("*.tmp"))


def test_the_state_write_is_fsynced_before_the_rename(strip, tmp_path, monkeypatch):
    """A rename made visible before its content is durable strands the notebooks the
    same way a failed write does."""
    order = []
    real_fsync = os.fsync
    real_replace = os.replace

    monkeypatch.setattr(os, "fsync", lambda fd: order.append("fsync") or real_fsync(fd))
    monkeypatch.setattr(os, "replace", lambda a, b: order.append("replace") or real_replace(a, b))

    target = tmp_path / "state"
    assert strip._write_state(str(target), ["a  b"]) is True
    assert order == ["fsync", "replace"], order
    assert target.read_text(encoding = "utf-8") == "a  b\n"


def test_malformed_and_unmanaged_lines_survive_verbatim(strip, tmp_path):
    dest = tmp_path / "dest"
    _notebook(dest / "nb/x.ipynb", "x")
    state = dest / ".unsloth_sync_state"
    state.write_text(
        "not-a-record\n"
        f"{_sha256(dest / 'nb/x.ipynb')}  nb/x.ipynb\n"
        "deadbeef  nb/gone.ipynb\n",
        encoding = "utf-8",
    )

    assert strip.migrate(str(state), str(dest)) == 0

    lines = state.read_text(encoding = "utf-8").splitlines()
    assert lines[0] == "not-a-record"
    assert lines[2] == "deadbeef  nb/gone.ipynb"
    assert lines[1] == f"{_sha256(dest / 'nb/x.ipynb')}  nb/x.ipynb"

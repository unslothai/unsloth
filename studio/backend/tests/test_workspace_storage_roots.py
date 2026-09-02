# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compatibility and isolation contract for account-aware storage roots."""

import re

import pytest

from utils.paths import storage_roots
from utils.workspace_context import (
    LEGACY_WORKSPACE_SUBJECT,
    current_workspace_subject,
    reset_workspace_subject,
    run_in_workspace,
    set_workspace_subject,
    workspace_key,
    workspace_thread,
)


def test_legacy_subject_keeps_every_historical_root(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    documents_home = tmp_path / "documents"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(documents_home))

    assert current_workspace_subject() == LEGACY_WORKSPACE_SUBJECT
    assert storage_roots.workspace_root() == studio_home
    assert storage_roots.assets_root() == studio_home / "assets"
    assert storage_roots.outputs_root() == studio_home / "outputs"
    assert storage_roots.exports_root() == studio_home / "exports"
    assert storage_roots.studio_db_path() == studio_home / "studio.db"
    assert storage_roots.rag_root() == studio_home / "rag"
    assert storage_roots.tensorboard_root() == studio_home / "runs"
    assert storage_roots.project_workspaces_root() == documents_home / "Unsloth Studio" / "Projects"


def test_managed_subject_roots_share_one_stable_key(tmp_path, monkeypatch):
    studio_home = tmp_path / "studio"
    documents_home = tmp_path / "documents"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(documents_home))
    monkeypatch.setattr(storage_roots.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    token = set_workspace_subject("Alice.Example")
    try:
        key = workspace_key()
        private_root = studio_home / "workspaces" / key
        assert storage_roots.workspace_root() == private_root
        assert storage_roots.assets_root() == private_root / "assets"
        assert storage_roots.outputs_root() == private_root / "outputs"
        assert storage_roots.exports_root() == private_root / "exports"
        assert storage_roots.studio_db_path() == private_root / "studio.db"
        assert storage_roots.rag_root() == private_root / "rag"
        assert storage_roots.tensorboard_root() == private_root / "runs"
        assert storage_roots.project_workspaces_root() == (
            documents_home / "Unsloth Studio" / "Users" / key / "Projects"
        )
        assert storage_roots.tmp_root() == tmp_path / "tmp" / "unsloth-studio" / "workspaces" / key
    finally:
        reset_workspace_subject(token)


@pytest.mark.parametrize("subject", ["con.txt", "NUL.tar.gz", "a/b", "...", "Alice Example"])
def test_workspace_key_is_one_windows_safe_component(subject):
    key = workspace_key(subject)
    assert re.fullmatch(r"[a-z0-9_-]+-[0-9a-f]{12}", key)
    assert "." not in key
    assert "/" not in key
    assert "\\" not in key
    assert key == workspace_key(subject)


def test_sanitised_prefix_collisions_keep_distinct_digests():
    assert workspace_key("a.b") != workspace_key("a-b")


def test_nested_binding_restores_the_previous_subject():
    outer = set_workspace_subject("alice")
    try:
        assert run_in_workspace("bob", current_workspace_subject) == "bob"
        assert current_workspace_subject() == "alice"
    finally:
        reset_workspace_subject(outer)


def test_workspace_thread_carries_the_subject_without_leaking_it():
    seen = []
    token = set_workspace_subject("alice")
    try:
        thread = workspace_thread(target=lambda: seen.append(current_workspace_subject()))
    finally:
        reset_workspace_subject(token)
    thread.start()
    thread.join(timeout = 5)

    assert seen == ["alice"]
    assert current_workspace_subject() == LEGACY_WORKSPACE_SUBJECT

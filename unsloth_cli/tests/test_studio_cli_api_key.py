# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reuse the named CLI API key across `unsloth studio run` (#10595)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


class _FakeStorage:
    DEFAULT_ADMIN_USERNAME = "unsloth"

    def __init__(self):
        self.created: list[str] = []
        self.keys: dict[str, dict] = {}
        self.next_id = 1

    def create_api_key(self, username, name):
        raw = f"sk-unsloth-{self.next_id:032x}"
        row = {"id": self.next_id, "name": name, "is_active": 1}
        self.next_id += 1
        self.created.append(name)
        self.keys[raw] = row
        return raw, row

    def validate_api_key_with_credential(self, raw_key, touch = True):
        row = self.keys.get(raw_key)
        if row and row["is_active"]:
            return (self.DEFAULT_ADMIN_USERNAME, "secret")
        return None

    def list_api_keys(self, username, include_internal = False):
        return [dict(row) for row in self.keys.values()]

    def revoke_api_key(self, username, key_id):
        for row in self.keys.values():
            if row["id"] == key_id:
                row["is_active"] = 0
                return True
        return False


@pytest.fixture
def studio_home(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    return studio_mod, tmp_path


def test_second_run_reuses_the_same_cli_key(studio_home, monkeypatch):
    studio_mod, tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    first = studio_mod._create_api_key_inprocess("cli")
    second = studio_mod._create_api_key_inprocess("cli")

    assert first == second
    assert storage.created == ["cli"]
    secret = (tmp_path / "auth" / ".cli_api_key_cli").read_text(encoding = "utf-8")
    assert secret.strip() == first


def test_revoked_cached_key_mints_once_and_drops_the_old_row(studio_home, monkeypatch):
    studio_mod, _tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    first = studio_mod._create_api_key_inprocess("cli")
    storage.keys[first]["is_active"] = 0
    second = studio_mod._create_api_key_inprocess("cli")

    assert second != first
    assert storage.created == ["cli", "cli"]
    assert storage.keys[first]["is_active"] == 0
    assert storage.keys[second]["is_active"] == 1


def test_mint_revokes_leftover_same_name_keys(studio_home, monkeypatch):
    studio_mod, _tmp_path = studio_home
    storage = _FakeStorage()
    leftover, _row = storage.create_api_key("unsloth", "cli")
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    minted = studio_mod._create_api_key_inprocess("cli")

    assert minted != leftover
    assert storage.keys[leftover]["is_active"] == 0
    assert storage.keys[minted]["is_active"] == 1


def test_different_names_keep_separate_keys(studio_home, monkeypatch):
    studio_mod, _tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    cli = studio_mod._create_api_key_inprocess("cli")
    other = studio_mod._create_api_key_inprocess("other")

    assert cli != other
    assert storage.created == ["cli", "other"]

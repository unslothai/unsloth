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

    def validate_api_key_with_credential(
        self,
        raw_key,
        *,
        touch = True,
    ):
        row = self.keys.get(raw_key)
        if row and row["is_active"]:
            return (self.DEFAULT_ADMIN_USERNAME, "secret")
        return None


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
    secret = studio_mod._cli_api_key_secret_path("cli").read_text(encoding = "utf-8")
    assert secret.strip() == first
    assert (tmp_path / "auth") in studio_mod._cli_api_key_secret_path("cli").parents


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


def test_corrupt_secret_file_mints_instead_of_aborting(studio_home, monkeypatch):
    studio_mod, tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)
    path = studio_mod._cli_api_key_secret_path("cli")
    path.parent.mkdir(parents = True)
    path.write_bytes(b"\xff\xfe not utf-8")

    minted = studio_mod._create_api_key_inprocess("cli")

    assert minted.startswith("sk-unsloth-")
    assert storage.created == ["cli"]


def test_different_names_keep_separate_keys(studio_home, monkeypatch):
    studio_mod, _tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    cli = studio_mod._create_api_key_inprocess("cli")
    other = studio_mod._create_api_key_inprocess("other")

    assert cli != other
    assert storage.created == ["cli", "other"]


@pytest.mark.parametrize(
    "first_name, second_name",
    [
        ("foo/bar", "foo?bar"),  # punctuation collapses to the same "_"
        ("my key", "my_key"),  # space vs underscore
        ("k" * 70 + "A", "k" * 70 + "B"),  # differ only past the 64-char cut
        ("///", "cli"),  # sanitizes to empty, falls back to "cli"
        ("cli", "CLI"),  # collides on APFS / NTFS case folding
        ("中文", "日本"),  # non-ASCII stems both fold to "_", so only the digest separates them
    ],
)
def test_distinct_names_never_share_a_cache_file(studio_home, monkeypatch, first_name, second_name):
    """A run asking for one label must never be handed the key minted for another."""
    studio_mod, _tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    first_path = studio_mod._cli_api_key_secret_path(first_name)
    second_path = studio_mod._cli_api_key_secret_path(second_name)
    # .lower(): on a case-insensitive filesystem the two would be one file.
    assert first_path.name.lower() != second_path.name.lower()

    first = studio_mod._create_api_key_inprocess(first_name)
    second = studio_mod._create_api_key_inprocess(second_name)
    assert first != second
    assert storage.created == [first_name, second_name]


@pytest.mark.parametrize(
    "name",
    [
        "cli",
        "my key",
        "café",
        "k" * 200,
        "..",
        "-",
        "x/y",
        # isalnum() but multibyte: 64 chars was 282 bytes, over the 255-BYTE
        # NAME_MAX, so the cache missed with ENAMETOOLONG and re-minted every launch.
        "\U0001d7d8" * 64,
        "中文" * 100,
        "\U0001f600" * 80,
    ],
)
def test_cache_path_is_a_safe_filename_inside_auth(studio_home, name):
    studio_mod, tmp_path = studio_home
    path = studio_mod._cli_api_key_secret_path(name)

    assert path.parent == tmp_path / "auth"
    assert path.name.startswith(studio_mod.CLI_API_KEY_FILE_PREFIX)
    assert "/" not in path.name and ".." not in path.name
    assert len(path.name.encode("utf-8")) <= 255


@pytest.mark.parametrize(
    "exc",
    [
        OSError(30, "Read-only file system"),
        PermissionError(13, "locked by another process"),  # Windows AV / indexer
        IsADirectoryError(21, "Is a directory"),
    ],
)
def test_cache_write_failure_does_not_abort_the_launch(studio_home, monkeypatch, capsys, exc):
    """The key is committed before the cache write, and the caller shuts the server
    down on any exception, so a failed write must cost the NEXT launch, not this one."""
    studio_mod, _tmp_path = studio_home
    storage = _FakeStorage()
    monkeypatch.setattr(studio_mod, "_load_backend_auth_storage", lambda: storage)

    def _boom(path, secret):
        raise exc

    monkeypatch.setattr(studio_mod, "_write_auth_secret", _boom)

    minted = studio_mod._create_api_key_inprocess("cli")

    assert minted.startswith("sk-unsloth-")
    assert storage.created == ["cli"]
    err = capsys.readouterr().err
    assert "could not cache" in err
    assert minted not in err, "the warning must not print the key"

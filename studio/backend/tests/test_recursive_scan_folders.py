# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in recursive scan folders (#6371)."""

import os
import sqlite3
from pathlib import Path

import pytest

from hub.services.models import local_inventory
from storage import studio_db


@pytest.fixture(autouse = True)
def _registrable_tmp(monkeypatch):
    # macOS denies /private/var, where pytest keeps tmp_path.
    from hub.storage import scan_folders
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [])


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"GGUF" + b"\0" * 28)
    return path


def _library(root: Path) -> dict[str, Path]:
    """vendor/family/model nesting like the issue reporter's E:\\AI tree."""
    return {
        "flat": _gguf(root / "flat-Q4_K_M.gguf"),
        "deep": _gguf(root / "vendor" / "family" / "deep-model" / "deep-Q4_K_M.gguf"),
        "deeper": _gguf(root / "a" / "b" / "c" / "d" / "deeper-Q8_0.gguf"),
    }


def _paths(rows) -> set[str]:
    return {str(Path(row.path).resolve()) for row in rows}


def _scan(root: Path, recursive: bool):
    nested = tuple(local_inventory.nested_scan_roots(root)) if recursive else ()
    return local_inventory._scan_custom_folder(root, nested_roots = nested)


def test_walker_lists_plain_folders_and_stops_at_models(tmp_path):
    lib = _library(tmp_path)
    roots = {
        p.relative_to(tmp_path).as_posix() for p in local_inventory.nested_scan_roots(tmp_path)
    }
    assert roots == {"vendor", "vendor/family", "a", "a/b", "a/b/c"}
    assert lib["deep"].parent.relative_to(tmp_path).as_posix() not in roots


def test_walker_skips_hidden_hf_cache_repos_and_symlinks(tmp_path):
    _gguf(tmp_path / ".hidden" / "x" / "m.gguf")
    (tmp_path / "models--org--repo" / "snapshots" / "abc").mkdir(parents = True)
    (tmp_path / "ollama_links" / "x").mkdir(parents = True)
    real = tmp_path / "real"
    real.mkdir()
    try:
        (tmp_path / "loop").symlink_to(tmp_path, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    names = {p.relative_to(tmp_path).parts[0] for p in local_inventory.nested_scan_roots(tmp_path)}
    assert names == {"real"}


def test_walker_depth_cap_bounds_the_walk(tmp_path, monkeypatch):
    monkeypatch.setattr(local_inventory, "_MAX_NESTED_SCAN_DEPTH", 2)
    _gguf(tmp_path / "a" / "b" / "c" / "m.gguf")
    roots = {
        p.relative_to(tmp_path).as_posix() for p in local_inventory.nested_scan_roots(tmp_path)
    }
    assert roots == {"a", "a/b"}


def test_walker_on_a_model_folder_adds_nothing(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding = "utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 8)
    (tmp_path / "sub").mkdir()
    assert local_inventory.nested_scan_roots(tmp_path) == []


def test_recursive_folder_lists_nested_models_and_flat_stays_flat(tmp_path):
    lib = _library(tmp_path)
    flat = _paths(_scan(tmp_path, recursive = False))
    deep = _paths(_scan(tmp_path, recursive = True))
    assert str(lib["flat"].resolve()) in flat
    assert str(lib["deep"].parent.resolve()) not in flat
    assert {str(lib["flat"].resolve()), str(lib["deep"].parent.resolve())} <= deep
    assert str(lib["deeper"].parent.resolve()) in deep
    assert flat <= deep


def test_recursive_scan_does_not_relist_an_lmstudio_layout(tmp_path):
    _gguf(tmp_path / "publisher" / "repo-GGUF" / "repo-Q4_K_M.gguf")
    rows = _scan(tmp_path, recursive = True)
    target = str((tmp_path / "publisher" / "repo-GGUF").resolve())
    assert sum(1 for row in rows if str(Path(row.path).resolve()) == target) == 1


def test_recursive_scan_finds_a_nested_hf_cache(tmp_path):
    repo = tmp_path / "team" / "hub" / "models--org--tiny"
    snapshot = repo / "snapshots" / "abc123"
    _gguf(snapshot / "tiny-Q4_K_M.gguf")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123", encoding = "utf-8")
    ids = {row.model_id for row in _scan(tmp_path, recursive = True)}
    assert "org/tiny" in ids
    assert "org/tiny" not in {row.model_id for row in _scan(tmp_path, recursive = False)}


def test_compat_nested_rows_skip_rows_already_listed(tmp_path):
    from routes import models as models_routes

    lib = _library(tmp_path)
    existing = models_routes._scan_models_dir(tmp_path, limit = 200)
    nested = models_routes._scan_nested_compat_rows(tmp_path, existing, limit = 200)
    assert str(lib["deep"].parent.resolve()) in _paths(nested)
    assert not _paths(existing) & _paths(nested)
    assert models_routes._scan_nested_compat_rows(tmp_path, existing, limit = len(existing)) == []


def test_storage_round_trips_and_updates_the_flag(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    row, changed = studio_db.add_scan_folder_with_status(str(folder), True)
    assert changed and row["recursive"] == 1
    assert studio_db.list_scan_folders()[0]["recursive"] is True
    # Omitted keeps the stored flag: export re-registration must not reset it.
    row, changed = studio_db.add_scan_folder_with_status(str(folder))
    assert not changed and row["recursive"] == 1
    row, changed = studio_db.add_scan_folder_with_status(str(folder), False)
    assert changed and row["recursive"] == 0
    assert studio_db.list_scan_folders()[0]["recursive"] is False


def test_hub_storage_shares_the_flag(tmp_path):
    from hub.storage import scan_folders

    folder = tmp_path / "models"
    folder.mkdir()
    row, _ = scan_folders.add_scan_folder_with_status(str(folder), True)
    assert scan_folders.list_scan_folders()[0]["recursive"] is True
    assert studio_db.list_scan_folders()[0]["id"] == row["id"]


def test_an_existing_table_gains_the_column(tmp_path):
    db_path = studio_db.studio_db_path()
    db_path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE scan_folders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "path TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO scan_folders (path, created_at) VALUES (?, ?)", (str(tmp_path), "2026-01-01")
    )
    conn.commit()
    conn.close()
    rows = studio_db.list_scan_folders()
    assert rows == [
        {"id": 1, "path": str(tmp_path), "created_at": "2026-01-01", "recursive": False}
    ]


def test_add_endpoint_threads_the_flag(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    row = local_inventory.add_scan_folder_response(str(folder), True)
    assert row["recursive"] == 1
    assert os.path.samefile(row["path"], folder)


def test_a_config_only_folder_does_not_hide_the_model_below_it(tmp_path):
    from routes import models as models_routes

    family = tmp_path / "family"
    family.mkdir()
    (family / "config.json").write_text("{}", encoding = "utf-8")
    model = family / "variant" / "model"
    model.mkdir(parents = True)
    (model / "config.json").write_text("{}", encoding = "utf-8")
    (model / "model.safetensors").write_bytes(b"\0" * 8)
    assert str(model.resolve()) in _paths(_scan(tmp_path, recursive = True))
    nested = models_routes._scan_nested_compat_rows(tmp_path, [], limit = 200)
    assert str(model.resolve()) in _paths(nested)


def test_compat_nested_rows_skip_folders_with_nothing_to_load(tmp_path):
    from routes import models as models_routes

    app = tmp_path / "tools" / "some-app"
    app.mkdir(parents = True)
    (app / "config.json").write_text("{}", encoding = "utf-8")
    assert models_routes._scan_nested_compat_rows(tmp_path, [], limit = 200) == []


def test_compat_nested_rows_keep_the_cap_with_a_nested_hf_cache(tmp_path):
    from routes import models as models_routes

    hub = tmp_path / "team" / "hub"
    for i in range(5):
        repo = hub / f"models--org--tiny{i}"
        _gguf(repo / "snapshots" / "abc" / f"tiny{i}-Q4_K_M.gguf")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("abc", encoding = "utf-8")
    assert len(models_routes._scan_nested_compat_rows(tmp_path, [], limit = 5)) == 5
    assert len(models_routes._scan_nested_compat_rows(tmp_path, [], limit = 3)) == 3

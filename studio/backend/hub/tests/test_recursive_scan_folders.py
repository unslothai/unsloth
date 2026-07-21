# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sqlite3

from hub.services.models import local_inventory
from hub.storage import scan_folders


def _make_model_dir(path):
    path.mkdir(parents = True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"\0" * 8)


def test_walker_yields_nested_dirs_and_prunes_model_dirs(tmp_path):
    _make_model_dir(tmp_path / "a" / "b" / "model-x")
    (tmp_path / "plain" / "deeper").mkdir(parents = True)

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path))

    assert tmp_path / "a" in yielded
    assert tmp_path / "a" / "b" in yielded
    assert tmp_path / "plain" in yielded
    assert tmp_path / "plain" / "deeper" in yielded
    assert tmp_path / "a" / "b" / "model-x" not in yielded


def test_walker_skips_hidden_and_link_dirs(tmp_path):
    (tmp_path / ".studio_links" / "x").mkdir(parents = True)
    (tmp_path / "ollama_links" / "y").mkdir(parents = True)
    (tmp_path / "kept").mkdir()

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path))

    assert yielded == [tmp_path / "kept"]


def test_walker_respects_depth_cap(tmp_path):
    deep = tmp_path
    for i in range(12):
        deep = deep / f"d{i}"
    deep.mkdir(parents = True)

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path, max_depth = 3))

    assert tmp_path / "d0" / "d1" / "d2" in yielded
    assert all(len(p.parts) - len(tmp_path.parts) <= 3 for p in yielded)


def test_walker_respects_entry_limit(tmp_path):
    for i in range(30):
        (tmp_path / f"dir{i:02d}").mkdir()

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path, entry_limit = 10))

    assert len(yielded) <= 10


def test_walker_does_not_follow_symlinks(tmp_path):
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path, tmp_path / "real" / "loop")

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path))

    assert tmp_path / "real" in yielded
    assert tmp_path / "real" / "loop" not in yielded


def test_scan_custom_folder_finds_nested_models_only_when_recursive(tmp_path):
    _make_model_dir(tmp_path / "vendor" / "family" / "model-deep")
    _make_model_dir(tmp_path / "model-shallow")

    flat = local_inventory._scan_custom_folder(tmp_path)
    flat_paths = {m.path for m in flat}
    assert any("model-shallow" in p for p in flat_paths)
    assert not any("model-deep" in p for p in flat_paths)

    recursive = local_inventory._scan_custom_folder(tmp_path, recursive = True)
    recursive_paths = {m.path for m in recursive}
    assert any("model-shallow" in p for p in recursive_paths)
    assert any("model-deep" in p for p in recursive_paths)


def test_recursive_scan_descends_through_config_only_dirs(tmp_path):
    (tmp_path / "family").mkdir()
    (tmp_path / "family" / "config.json").write_text("{}")
    _make_model_dir(tmp_path / "family" / "variant" / "model-nested")

    recursive = local_inventory._scan_custom_folder(tmp_path, recursive = True)

    assert any("model-nested" in m.path for m in recursive)


def test_recursive_scan_ignores_symlinked_models_outside_folder(tmp_path):
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents = True)
    _make_model_dir(tmp_path / "outside" / "model-o")
    os.symlink(tmp_path / "outside" / "model-o", root / "a" / "b" / "link")

    recursive = local_inventory._scan_custom_folder(root, recursive = True)

    assert not any("model-o" in m.path or "link" in m.path for m in recursive)


def test_recursive_scan_descends_through_weight_only_dirs(tmp_path):
    (tmp_path / "family").mkdir()
    (tmp_path / "family" / "orphan.safetensors").write_bytes(b"\0" * 8)
    _make_model_dir(tmp_path / "family" / "variant" / "model-y")

    recursive = local_inventory._scan_custom_folder(tmp_path, recursive = True)

    assert any("model-y" in m.path for m in recursive)


def test_walker_prunes_gguf_leaf_dirs(tmp_path):
    (tmp_path / "gguf-model").mkdir()
    (tmp_path / "gguf-model" / "model.gguf").write_bytes(b"\0" * 8)
    (tmp_path / "plain").mkdir()

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path))

    assert tmp_path / "plain" in yielded
    assert tmp_path / "gguf-model" not in yielded


def test_walker_counts_files_toward_entry_limit(tmp_path):
    for i in range(60):
        (tmp_path / f"file{i:02d}.txt").write_text("x")
    (tmp_path / "sub").mkdir()

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path, entry_limit = 10))

    assert yielded == []


def test_recursive_scan_rejects_symlinked_weight_files(tmp_path):
    root = tmp_path / "root"
    (root / "x" / "a" / "model-s").mkdir(parents = True)
    (root / "x" / "a" / "model-s" / "config.json").write_text("{}")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.safetensors").write_bytes(b"\0" * 8)
    os.symlink(outside / "real.safetensors", root / "x" / "a" / "model-s" / "model.safetensors")

    recursive = local_inventory._scan_custom_folder(root, recursive = True)

    assert not any("model-s" in m.path for m in recursive)


def test_scan_result_within_folder_checks_weights_past_probe_window(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    model.mkdir(parents = True)
    (model / "config.json").write_text("{}")
    # Many non-weight files so the escaping weight sits well past the old probe cap.
    for i in range(250):
        (model / f"note{i:03d}.txt").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.safetensors").write_bytes(b"\0" * 8)
    os.symlink(outside / "real.safetensors", model / "model.safetensors")

    assert local_inventory.scan_result_within_folder(str(model), root) is False


def test_scan_result_within_folder_accepts_contained_weights(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    model.mkdir(parents = True)
    (model / "model.safetensors").write_bytes(b"\0" * 8)

    assert local_inventory.scan_result_within_folder(str(model), root) is True


def test_recursive_scan_does_not_surface_config_only_intermediate(tmp_path):
    (tmp_path / "family").mkdir()
    (tmp_path / "family" / "config.json").write_text("{}")
    _make_model_dir(tmp_path / "family" / "variant" / "model-real")

    recursive = local_inventory._scan_custom_folder(tmp_path, recursive = True)
    paths = {m.path for m in recursive}

    assert any("model-real" in p for p in paths)
    assert str(tmp_path / "family") not in paths


def test_scan_result_within_folder_rejects_symlinked_companion_gguf(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    model.mkdir(parents = True)
    (model / "model.gguf").write_bytes(b"\0" * 8)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "mmproj-model.gguf").write_bytes(b"\0" * 8)
    os.symlink(outside / "mmproj-model.gguf", model / "mmproj-model.gguf")

    assert local_inventory.scan_result_within_folder(str(model), root) is False


def test_scan_result_within_folder_rejects_any_symlinked_entry_outside(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    model.mkdir(parents = True)
    (model / "model.safetensors").write_bytes(b"\0" * 8)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    os.symlink(outside / "config.json", model / "config.json")

    assert local_inventory.scan_result_within_folder(str(model), root) is False


def test_scan_result_within_folder_allows_symlinks_inside_folder(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    model.mkdir(parents = True)
    (root / "shared.safetensors").write_bytes(b"\0" * 8)
    os.symlink(root / "shared.safetensors", model / "model.safetensors")

    assert local_inventory.scan_result_within_folder(str(model), root) is True


def test_dir_has_loadable_weights_rejects_config_and_companion_only(tmp_path):
    config_only = tmp_path / "config_only"
    config_only.mkdir()
    (config_only / "config.json").write_text("{}")
    assert local_inventory._dir_has_loadable_weights(config_only) is False

    companion_only = tmp_path / "companion"
    companion_only.mkdir()
    (companion_only / "mmproj-model.gguf").write_bytes(b"\0" * 8)
    assert local_inventory._dir_has_loadable_weights(companion_only) is False

    real = tmp_path / "real"
    real.mkdir()
    (real / "model.safetensors").write_bytes(b"\0" * 8)
    assert local_inventory._dir_has_loadable_weights(real) is True

    main_gguf = tmp_path / "gg"
    main_gguf.mkdir()
    (main_gguf / "model.gguf").write_bytes(b"\0" * 8)
    assert local_inventory._dir_has_loadable_weights(main_gguf) is True


def test_leaf_probe_prunes_gguf_dir_past_default_window(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for i in range(250):
        (model / f"note{i:03d}.txt").write_text("x")
    (model / "model.gguf").write_bytes(b"\0" * 8)
    (tmp_path / "plain").mkdir()

    yielded = list(local_inventory.iter_recursive_scan_dirs(tmp_path))

    # The main GGUF sorts after 200 entries; with the probe raised to the entry
    # limit the dir is still classified as a model leaf and pruned from descent.
    assert model not in yielded
    assert tmp_path / "plain" in yielded


def test_scan_custom_folder_recursive_does_not_duplicate(tmp_path):
    _make_model_dir(tmp_path / "sub" / "model-a")

    recursive = local_inventory._scan_custom_folder(tmp_path, recursive = True)

    matches = [m for m in recursive if "model-a" in m.path]
    assert len(matches) == 1


def _tmp_connection_factory(db_path):
    def _connect():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    return _connect


def test_add_scan_folder_round_trips_recursive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        scan_folders, "get_connection", _tmp_connection_factory(tmp_path / "studio.db")
    )
    monkeypatch.setattr(scan_folders, "_schema_ready", False)
    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [])
    target = tmp_path / "models"
    target.mkdir()

    folder = scan_folders.add_scan_folder(str(target), recursive = True)
    assert folder["recursive"] == 1

    listed = scan_folders.list_scan_folders()
    assert [f["recursive"] for f in listed] == [1]

    unchanged = scan_folders.add_scan_folder(str(target))
    assert unchanged["recursive"] == 1
    assert [f["recursive"] for f in scan_folders.list_scan_folders()] == [1]

    updated = scan_folders.add_scan_folder(str(target), recursive = False)
    assert updated["recursive"] == 0
    assert [f["recursive"] for f in scan_folders.list_scan_folders()] == [0]


def test_schema_migrates_existing_table(tmp_path, monkeypatch):
    db_path = tmp_path / "studio.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE scan_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO scan_folders (path, created_at) VALUES (?, ?)",
        ("/somewhere", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(scan_folders, "get_connection", _tmp_connection_factory(db_path))
    monkeypatch.setattr(scan_folders, "_schema_ready", False)

    listed = scan_folders.list_scan_folders()
    assert listed[0]["recursive"] == 0

    monkeypatch.setattr(scan_folders, "_schema_ready", False)
    listed_again = scan_folders.list_scan_folders()
    assert listed_again[0]["recursive"] == 0


def test_recursive_scan_descends_through_file_heavy_dirs(tmp_path):
    # A directory padded with many files must still be descended into to reach a
    # nested model. Guards the scandir walker (which counts files toward the cap)
    # against regressing the file-heavy case.
    padded = tmp_path / "padded"
    padded.mkdir()
    for i in range(50):
        (padded / f"note{i:03d}.txt").write_text("x")
    _make_model_dir(padded / "nested" / "model-z")

    recursive = local_inventory._scan_custom_folder(tmp_path, recursive = True)

    assert "model-z" in {m.display_name for m in recursive}

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""oMLX (https://github.com/jundot/omlx) model roots reach both local inventories."""

import asyncio
import json
from pathlib import Path

import pytest

import hub.services.models.local_inventory as local_inventory
import hub.utils.paths as hub_paths
import routes.models as models_route
import utils.paths.storage_roots as storage_roots


def _write_mlx_model(
    root: Path,
    org: str,
    name: str,
    *,
    shards: int = 1,
) -> Path:
    model_dir = root / org / name
    model_dir.mkdir(parents = True)
    config = {"model_type": "qwen3", "quantization": {"group_size": 64, "bits": 4}}
    (model_dir / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    if shards == 1:
        (model_dir / "model.safetensors").write_bytes(b"weights")
    else:
        names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
        index = {"weight_map": {f"l{i}": n for i, n in enumerate(names)}}
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding = "utf-8")
        for n in names:
            (model_dir / n).write_bytes(b"weights")
    return model_dir


def _write_hf_cache_repo(cache: Path, org: str, name: str) -> Path:
    repo = cache / f"models--{org}--{name}"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "abc123"
    blobs.mkdir(parents = True)
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123", encoding = "utf-8")
    contents = {
        "config.json": json.dumps({"model_type": "qwen3"}),
        "tokenizer_config.json": "{}",
        "model.safetensors": "weights",
    }
    for index, (filename, body) in enumerate(contents.items()):
        blob = blobs / f"blob{index}"
        blob.write_text(body, encoding = "utf-8")
        (snapshot / filename).symlink_to(blob)
    return snapshot


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_roots.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("OMLX_BASE_PATH", raising = False)
    monkeypatch.delenv("OMLX_MODEL_DIR", raising = False)
    return tmp_path


def _write_settings(base: Path, model: dict) -> None:
    base.mkdir(parents = True, exist_ok = True)
    (base / "settings.json").write_text(json.dumps({"model": model}), encoding = "utf-8")


def test_hub_paths_reexports_the_one_resolver():
    assert hub_paths.omlx_model_dirs is storage_roots.omlx_model_dirs


def test_default_models_dir_without_settings(home):
    models = home / ".omlx" / "models"
    models.mkdir(parents = True)
    assert storage_roots.omlx_model_dirs() == [models]


def test_no_omlx_install_yields_nothing(home):
    assert storage_roots.omlx_model_dirs() == []


def test_model_dirs_replace_the_default(home):
    first, second = home / "vol" / "one", home / "vol" / "two"
    first.mkdir(parents = True)
    second.mkdir(parents = True)
    (home / ".omlx" / "models").mkdir(parents = True)
    _write_settings(home / ".omlx", {"model_dirs": [str(first), str(second)]})
    assert storage_roots.omlx_model_dirs() == [first, second]


def test_legacy_model_dir_used_when_model_dirs_empty(home):
    configured = home / "vol" / "single"
    configured.mkdir(parents = True)
    _write_settings(home / ".omlx", {"model_dirs": [], "model_dir": str(configured)})
    assert storage_roots.omlx_model_dirs() == [configured]


def test_unreadable_settings_fall_back_to_default(home):
    models = home / ".omlx" / "models"
    models.mkdir(parents = True)
    (home / ".omlx" / "settings.json").write_text("{not json", encoding = "utf-8")
    assert storage_roots.omlx_model_dirs() == [models]


def test_duplicate_and_missing_roots_are_dropped(home):
    models = home / ".omlx" / "models"
    models.mkdir(parents = True)
    _write_settings(home / ".omlx", {"model_dirs": [str(models), str(home / "gone"), str(models)]})
    assert storage_roots.omlx_model_dirs() == [models]


def test_lmstudio_root_listed_by_omlx_is_not_returned(home):
    lmstudio = home / ".lmstudio" / "models"
    lmstudio.mkdir(parents = True)
    omlx = home / ".omlx" / "models"
    omlx.mkdir(parents = True)
    _write_settings(home / ".omlx", {"model_dirs": [str(omlx), str(lmstudio)]})
    assert storage_roots.omlx_model_dirs() == [omlx]


def test_base_path_env_overrides_bootstrap_and_default(home, monkeypatch):
    (home / ".omlx" / "models").mkdir(parents = True)
    moved = home / "moved"
    (moved / "models").mkdir(parents = True)
    monkeypatch.setenv("OMLX_BASE_PATH", str(moved))
    assert storage_roots.omlx_model_dirs() == [moved / "models"]


def test_model_dir_env_overrides_settings(home, monkeypatch):
    first, second, saved = home / "a", home / "b", home / "saved"
    for d in (first, second, saved):
        d.mkdir()
    _write_settings(home / ".omlx", {"model_dirs": [str(saved)]})
    monkeypatch.setenv("OMLX_MODEL_DIR", f"{first}, {second}")
    assert storage_roots.omlx_model_dirs() == [first, second]


def test_macos_bootstrap_file_moves_the_base(home):
    moved = home / "Volumes" / "ext" / "omlx"
    configured = home / "Volumes" / "ext" / "weights"
    configured.mkdir(parents = True)
    _write_settings(moved, {"model_dirs": [str(configured)]})
    bootstrap = home / "Library" / "Application Support" / "oMLX" / "base-path"
    bootstrap.parent.mkdir(parents = True)
    bootstrap.write_text(f"{moved}\n", encoding = "utf-8")
    assert storage_roots.omlx_model_dirs() == [configured]


def _hub_rows(cache: Path, empty: Path, omlx_root: Path):
    sources = local_inventory._LocalInventorySources(
        cache, empty, empty, (), (), (), (), omlx_dirs = (omlx_root,)
    )
    return asyncio.run(local_inventory._scan_local_models_response(str(empty), [], sources)).models


def _compat_rows(cache: Path, empty: Path, omlx_root: Path):
    sources = models_route._CompatLocalInventorySources(
        cache, empty, empty, (), (), omlx_dirs = (omlx_root,)
    )
    return models_route.collect_local_models(empty, custom_folders = [], sources = sources)


@pytest.fixture(params = ["hub", "compat"])
def collector(request, tmp_path):
    empty = tmp_path / "_empty"
    empty.mkdir()
    scan = _hub_rows if request.param == "hub" else _compat_rows
    return lambda cache, root: scan(cache, empty, root)


def test_hub_sources_include_omlx_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(local_inventory, "omlx_model_dirs", lambda: [tmp_path])
    assert local_inventory._local_inventory_sources().omlx_dirs == (tmp_path,)


def test_compat_sources_include_omlx_roots(monkeypatch, tmp_path):
    import utils.paths as utils_paths
    monkeypatch.setattr(utils_paths, "omlx_model_dirs", lambda: [tmp_path])
    assert models_route._compat_local_inventory_sources().omlx_dirs == (tmp_path,)


def test_omlx_rows_report_the_omlx_source(collector, tmp_path):
    cache = tmp_path / "hf"
    cache.mkdir()
    _write_mlx_model(tmp_path / "omlx", "mlx-community", "Qwen3-4bit", shards = 2)

    rows = collector(cache, tmp_path / "omlx")

    assert [(m.source, m.model_id) for m in rows] == [("omlx", "mlx-community/Qwen3-4bit")]


@pytest.mark.parametrize(
    "scan", [local_inventory._scan_lmstudio_dir, models_route._scan_lmstudio_dir]
)
def test_source_is_the_only_difference_from_an_lmstudio_root(scan, tmp_path):
    _write_mlx_model(tmp_path, "mlx-community", "Same-4bit", shards = 2)

    (as_omlx,) = [m.model_dump() for m in scan(tmp_path, source = "omlx")]
    (as_lmstudio,) = [m.model_dump() for m in scan(tmp_path)]

    differing = {k for k in as_omlx if as_omlx[k] != as_lmstudio.get(k)}
    assert as_lmstudio["source"] == "lmstudio"
    assert differing <= {"source", "inventory_id"} and "source" in differing


def test_folder_linking_into_hf_cache_is_listed_with_the_repo(collector, tmp_path):
    cache = tmp_path / "hf" / "hub"
    snapshot = _write_hf_cache_repo(cache, "mlx-community", "Shared-4bit")
    model = tmp_path / "omlx" / "mlx-community" / "Shared-4bit"
    model.mkdir(parents = True)
    for entry in snapshot.iterdir():
        (model / entry.name).symlink_to(entry)

    rows = collector(cache, tmp_path / "omlx")

    assert [Path(m.path).name for m in rows if m.source == "omlx"] == ["Shared-4bit"]
    assert any(m.source == "hf_cache" for m in rows)


def test_flat_folder_inside_an_hf_cache_root_is_found(collector, tmp_path):
    # oMLX can list the HF cache itself; its flat <publisher>__<model> folders are invisible to
    # the models--* walk.
    cache = tmp_path / "hub"
    snapshot = _write_hf_cache_repo(cache, "mlx-community", "Shared-4bit")
    flat = cache / "mlx-community__Shared-4bit"
    flat.mkdir()
    for entry in snapshot.iterdir():
        (flat / entry.name).symlink_to(entry)

    rows = collector(cache, cache)

    assert [Path(m.path).name for m in rows if m.source == "omlx"] == ["mlx-community__Shared-4bit"]


def test_omlx_root_registered_as_a_scan_folder_is_listed_once(monkeypatch, tmp_path):
    # Registering ~/.omlx/models as a custom folder was the workaround before this scan.
    root = tmp_path / ".omlx" / "models"
    model = _write_mlx_model(root, "mlx-community", "Qwen3-4bit")
    monkeypatch.setattr(local_inventory, "note_scan_folder_scanned", lambda *_a, **_k: None)
    monkeypatch.setattr(local_inventory, "record_scan_failure", lambda *_a, **_k: None)

    rows = asyncio.run(
        local_inventory._collect_models_from_default_sources(
            tmp_path / "models",
            tmp_path / "hf",
            tmp_path / "legacy",
            tmp_path / "default",
            (),
            (),
            (),
            (),
            [{"path": str(root)}],
            omlx_dirs = (root,),
        )
    )
    rows = local_inventory._filter_and_dedupe_local_models(rows)

    # Trainable, so the custom twin wins: the train picker refuses oMLX rows.
    assert [(row.source, row.path) for row in rows] == [("custom", str(model))]


def test_compat_omlx_root_registered_as_a_scan_folder_is_listed_once(tmp_path):
    root = tmp_path / ".omlx" / "models"
    model = _write_mlx_model(root, "mlx-community", "Qwen3-4bit")
    empty = tmp_path / "_empty"
    empty.mkdir()
    sources = models_route._CompatLocalInventorySources(
        empty, empty, empty, (), (), omlx_dirs = (root,)
    )

    rows = models_route.collect_local_models(
        empty, custom_folders = [{"path": str(root)}], sources = sources
    )

    assert [(row.source, row.path) for row in rows] == [("custom", str(model))]


@pytest.mark.parametrize("link_publisher", [False, True])
def test_compat_custom_symlink_alias_into_an_omlx_root_is_kept(tmp_path, link_publisher):
    root = tmp_path / "omlx"
    model = _write_mlx_model(root, "mlx-community", "Qwen3-4bit")
    custom = tmp_path / "custom"
    custom.mkdir()
    if link_publisher:
        (custom / "mlx-community").symlink_to(model.parent, target_is_directory = True)
    else:
        (custom / "Qwen3-4bit-alias").symlink_to(model, target_is_directory = True)
    empty = tmp_path / "_empty"
    empty.mkdir()
    sources = models_route._CompatLocalInventorySources(
        empty, empty, empty, (), (), omlx_dirs = (root,)
    )

    rows = models_route.collect_local_models(
        empty, custom_folders = [{"path": str(custom)}], sources = sources
    )

    assert sorted(row.source for row in rows) == ["custom", "omlx"]


def test_resolver_index_scans_omlx_roots(monkeypatch, tmp_path):
    # MLX weights are servable only on Apple Silicon, so assert the scan feeds the index.
    import core.inference.local_model_resolver as resolver
    import utils.paths as utils_paths

    root = tmp_path / "omlx"
    model = _write_mlx_model(root, "mlx-community", "OnlyInOmlx-4bit")
    monkeypatch.setattr(utils_paths, "omlx_model_dirs", lambda: [root])
    seen = []
    monkeypatch.setattr(
        resolver, "_local_servable_entry", lambda _id, info: seen.append(info) and None
    )

    resolver._build_index()

    assert ("omlx", str(model)) in [(i.source, i.path) for i in seen]


def test_hf_cache_repos_under_an_omlx_root_are_listed(collector, tmp_path):
    active = tmp_path / "active"
    active.mkdir()
    external = tmp_path / "external-hub"
    _write_hf_cache_repo(external, "mlx-community", "External-4bit")

    rows = collector(active, external)

    assert [(m.source, m.model_id) for m in rows] == [("hf_cache", "mlx-community/External-4bit")]

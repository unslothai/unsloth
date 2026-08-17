# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""oMLX (app.omlx) local model discovery.

oMLX keeps models in a ``<org>/<model>/`` tree, the same shape LM Studio uses, and
declares its roots in ``~/.omlx/settings.json``. Studio scanned neither, so a model
installed only through oMLX was invisible in the local inventory.
"""

import json
from pathlib import Path

import pytest

import hub.services.models.local_inventory as local_inventory
import hub.utils.paths as hub_paths
import utils.paths.storage_roots as storage_roots


def _write_mlx_model(
    root: Path,
    org: str,
    name: str,
    *,
    shards: int = 1,
) -> Path:
    """An MLX model as oMLX stores it: publisher/model with an MLX quantization block."""
    model_dir = root / org / name
    model_dir.mkdir(parents = True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            }
        ),
        encoding = "utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    if shards == 1:
        (model_dir / "model.safetensors").write_bytes(b"weights")
    else:
        names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"l{i}": n for i, n in enumerate(names)}}),
            encoding = "utf-8",
        )
        for n in names:
            (model_dir / n).write_bytes(b"weights")
    return model_dir


def _write_hf_cache_repo(cache: Path, org: str, name: str) -> Path:
    repo = cache / f"models--{org}--{name}"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "abc123"
    blobs.mkdir(parents = True)
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir(parents = True)
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


def _write_settings(home: Path, payload: dict) -> None:
    omlx = home / ".omlx"
    omlx.mkdir(parents = True, exist_ok = True)
    (omlx / "settings.json").write_text(json.dumps(payload), encoding = "utf-8")


@pytest.fixture(params = ["hub", "compat"])
def collector(request, tmp_path):
    empty = tmp_path / "_empty"
    empty.mkdir(exist_ok = True)

    def _hub(cache: Path, omlx_root: Path):
        import asyncio
        sources = local_inventory._LocalInventorySources(
            cache, empty, empty, (), (omlx_root,), (), ()
        )
        return asyncio.run(
            local_inventory._scan_local_models_response(str(empty), [], sources)
        ).models

    def _compat(cache: Path, omlx_root: Path):
        import routes.models as models_route
        sources = models_route._CompatLocalInventorySources(
            hf_cache_dir = cache,
            legacy_hf = empty,
            hf_default = empty,
            lm_dirs = (),
            omlx_dirs = (omlx_root,),
            known_hf_caches = (),
        )
        return models_route.collect_local_models(empty, custom_folders = [], sources = sources)

    return _hub if request.param == "hub" else _compat


def _link_model(root: Path, name: str, snapshot: Path) -> Path:
    model = root / "mlx-community" / name
    model.mkdir(parents = True)
    for entry in snapshot.iterdir():
        (model / entry.name).symlink_to(entry)
    return model


@pytest.fixture(params = [hub_paths, storage_roots], ids = ["hub", "storage_roots"])
def paths_module(request, monkeypatch, tmp_path):
    """Both path modules carry a duplicated copy of the host-app root resolvers.

    They must stay in sync, so every rule below is asserted against both.
    """
    monkeypatch.setattr(request.param.Path, "home", staticmethod(lambda: tmp_path))
    return request.param


def test_default_models_dir_is_found_without_settings(paths_module, tmp_path):
    models = tmp_path / ".omlx" / "models"
    models.mkdir(parents = True)

    assert paths_module.omlx_model_dirs() == [models]


def test_settings_model_dirs_are_honoured(paths_module, tmp_path):
    """oMLX declares its own roots; read them rather than guessing paths."""
    first = tmp_path / "vol" / "one"
    second = tmp_path / "vol" / "two"
    first.mkdir(parents = True)
    second.mkdir(parents = True)
    _write_settings(tmp_path, {"model": {"model_dirs": [str(first), str(second)]}})

    assert paths_module.omlx_model_dirs() == [first, second]


def test_settings_single_model_dir_is_honoured(paths_module, tmp_path):
    configured = tmp_path / "vol" / "single"
    configured.mkdir(parents = True)
    _write_settings(tmp_path, {"model": {"model_dir": str(configured)}})

    assert paths_module.omlx_model_dirs() == [configured]


def test_a_configured_dir_that_does_not_exist_is_dropped(paths_module, tmp_path):
    real = tmp_path / ".omlx" / "models"
    real.mkdir(parents = True)
    _write_settings(tmp_path, {"model": {"model_dirs": [str(tmp_path / "gone")]}})

    assert paths_module.omlx_model_dirs() == [real]


def test_unreadable_settings_falls_back_to_the_default_dir(paths_module, tmp_path):
    """A corrupt settings file must not cost the user the default root."""
    models = tmp_path / ".omlx" / "models"
    models.mkdir(parents = True)
    (tmp_path / ".omlx" / "settings.json").write_text("{not json", encoding = "utf-8")

    assert paths_module.omlx_model_dirs() == [models]


def test_the_default_dir_is_kept_alongside_a_configured_one(paths_module, tmp_path):
    configured = tmp_path / "vol" / "extra"
    configured.mkdir(parents = True)
    default = tmp_path / ".omlx" / "models"
    default.mkdir(parents = True)
    _write_settings(tmp_path, {"model": {"model_dir": str(configured)}})

    assert paths_module.omlx_model_dirs() == [configured, default]


def test_a_root_is_never_returned_twice(paths_module, tmp_path):
    models = tmp_path / ".omlx" / "models"
    models.mkdir(parents = True)
    _write_settings(
        tmp_path,
        {"model": {"model_dir": str(models), "model_dirs": [str(models)]}},
    )

    assert paths_module.omlx_model_dirs() == [models]


def test_an_lmstudio_root_oMLX_also_declares_is_not_returned(paths_module, tmp_path):
    """oMLX lists LM Studio's directory among its own roots. Studio already scans
    that root as ``source="lmstudio"``, so returning it here would scan it twice."""
    lmstudio = tmp_path / ".lmstudio" / "models"
    lmstudio.mkdir(parents = True)
    omlx = tmp_path / ".omlx" / "models"
    omlx.mkdir(parents = True)
    _write_settings(
        tmp_path,
        {"model": {"model_dirs": [str(omlx), str(lmstudio)]}},
    )

    assert paths_module.omlx_model_dirs() == [omlx]


class TestInventoryWiring:
    """oMLX roots must reach the scan, and the rows must say where they came from."""

    def test_the_omlx_root_is_one_of_the_inventory_sources(self, monkeypatch, tmp_path):
        root = tmp_path / ".omlx" / "models"
        root.mkdir(parents = True)
        monkeypatch.setattr(local_inventory, "omlx_model_dirs", lambda: [root])

        sources = local_inventory._local_inventory_sources()

        assert root in sources.omlx_dirs

    def test_a_scanned_omlx_model_reports_the_omlx_source(self, tmp_path):
        _write_mlx_model(tmp_path, "mlx-community", "Qwen3.8-27B-4bit")

        found = local_inventory._scan_lmstudio_dir(tmp_path, source = "omlx")

        assert [m.source for m in found] == ["omlx"]

    def test_the_same_scanner_still_reports_lmstudio_by_default(self, tmp_path):
        _write_mlx_model(tmp_path, "mlx-community", "Qwen3.8-27B-4bit")

        found = local_inventory._scan_lmstudio_dir(tmp_path)

        assert [m.source for m in found] == ["lmstudio"]


class TestClassification:
    """An oMLX row must be classified exactly like the same tree under LM Studio.

    Sharing the walk is the point: whatever the publisher scan already decides about a
    model, it must not start deciding differently because the root belongs to oMLX.
    """

    def test_a_sharded_mlx_model_is_classified_as_safetensors(self, tmp_path):
        _write_mlx_model(tmp_path, "mlx-community", "Complete-4bit", shards = 2)

        found = local_inventory._scan_lmstudio_dir(tmp_path, source = "omlx")

        assert [(m.model_format, m.partial) for m in found] == [("safetensors", False)]

    def test_the_source_is_the_only_difference_from_an_lmstudio_root(self, tmp_path):
        _write_mlx_model(tmp_path, "mlx-community", "Same-4bit", shards = 2)

        as_omlx = local_inventory._scan_lmstudio_dir(tmp_path, source = "omlx")
        as_lmstudio = local_inventory._scan_lmstudio_dir(tmp_path)

        assert len(as_omlx) == len(as_lmstudio) == 1
        omlx_row = as_omlx[0].model_dump()
        lmstudio_row = as_lmstudio[0].model_dump()
        differing = {key for key in omlx_row if omlx_row[key] != lmstudio_row.get(key)}
        # inventory_id is "<source>:<format>:<semantic id>", so it moves with the source
        # by construction -- that is what gives an oMLX row its own dedup identity.
        assert differing == {"source", "inventory_id"}
        assert omlx_row["inventory_id"].startswith("omlx:")
        assert lmstudio_row["inventory_id"].startswith("lmstudio:")


class TestHfCacheSymlinks:
    """A folder of links and the cache repo it points at are both listed.

    Suppressing the folder needs proof another row represents it, and every proxy for that --
    cache root, ``models--*`` path, enumerated repos, even a row's own path -- is wrong on some
    input, leaving a duplicate or, worse, dropping a model nothing else reports.
    """

    def test_a_model_symlinked_into_the_hf_cache_is_listed_with_the_repo(self, collector, tmp_path):
        cache = tmp_path / "hf" / "hub"
        snapshot = _write_hf_cache_repo(cache, "mlx-community", "Shared-4bit")
        _link_model(tmp_path / "omlx", "Shared-4bit", snapshot)

        rows = collector(cache, tmp_path / "omlx")

        assert [Path(m.path).name for m in rows if m.source == "omlx"] == ["Shared-4bit"]
        assert any(m.source in ("hf_cache", "custom") for m in rows)

    def test_a_model_with_real_files_is_still_listed(self, collector, tmp_path):
        cache = tmp_path / "hf" / "hub"
        _write_hf_cache_repo(cache, "mlx-community", "Other-4bit")
        _write_mlx_model(tmp_path / "omlx", "mlx-community", "Own-4bit")

        rows = collector(cache, tmp_path / "omlx")

        assert [Path(m.path).name for m in rows if m.source == "omlx"] == ["Own-4bit"]

    def test_every_model_under_the_root_is_reported(self, collector, tmp_path):
        cache = tmp_path / "hf" / "hub"
        snapshot = _write_hf_cache_repo(cache, "mlx-community", "Shared-4bit")
        loose = cache / "beside-the-repos.safetensors"
        loose.write_bytes(b"weights")

        root = tmp_path / "omlx"
        _link_model(root, "Linked-4bit", snapshot)
        _write_mlx_model(root, "mlx-community", "Own-4bit")
        stray = root / "mlx-community" / "Stray-4bit"
        stray.mkdir(parents = True)
        (stray / "config.json").write_text("{}", encoding = "utf-8")
        (stray / "model.safetensors").symlink_to(loose)

        rows = collector(cache, root)

        assert {Path(m.path).name for m in rows if m.source == "omlx"} == {
            "Linked-4bit",
            "Own-4bit",
            "Stray-4bit",
        }


class TestCompatInventory:
    """``routes.models`` carries a second, older copy of the inventory scan. The two are
    deliberately kept in step, so oMLX has to reach both."""

    def test_the_compat_scanner_reports_the_omlx_source(self, tmp_path):
        import routes.models as models_route

        _write_mlx_model(tmp_path, "mlx-community", "Qwen3.8-27B-4bit")

        found = models_route._scan_lmstudio_dir(tmp_path, source = "omlx")

        assert [m.source for m in found] == ["omlx"]

    def test_the_compat_scanner_still_defaults_to_lmstudio(self, tmp_path):
        import routes.models as models_route

        _write_mlx_model(tmp_path, "mlx-community", "Qwen3.8-27B-4bit")

        found = models_route._scan_lmstudio_dir(tmp_path)

        assert [m.source for m in found] == ["lmstudio"]

    def test_the_compat_collector_scans_the_omlx_root(self, tmp_path):
        import routes.models as models_route

        omlx_root = tmp_path / "omlx"
        _write_mlx_model(omlx_root, "mlx-community", "OnlyInOmlx-4bit")
        empty = tmp_path / "empty"
        empty.mkdir()
        sources = models_route._CompatLocalInventorySources(
            hf_cache_dir = empty,
            legacy_hf = empty,
            hf_default = empty,
            lm_dirs = (),
            omlx_dirs = (omlx_root,),
            known_hf_caches = (),
        )

        models = models_route.collect_local_models(
            empty,
            custom_folders = [],
            sources = sources,
        )

        assert [(m.source, m.model_id) for m in models] == [
            ("omlx", "mlx-community/OnlyInOmlx-4bit")
        ]

    def test_a_flat_publisher_folder_inside_the_cache_root_is_found(self, tmp_path):
        """A real install keeps flat ``<publisher>__<model>`` folders inside the cache root,
        where ``models--*`` scanning cannot see them."""
        import routes.models as models_route

        cache = tmp_path / "hub"
        snapshot = _write_hf_cache_repo(cache, "mlx-community", "Shared-4bit")
        flat = cache / "mlx-community__Shared-4bit"
        flat.mkdir()
        for entry in snapshot.iterdir():
            (flat / entry.name).symlink_to(entry)
        empty = tmp_path / "empty"
        empty.mkdir()
        sources = models_route._CompatLocalInventorySources(
            hf_cache_dir = cache,
            legacy_hf = empty,
            hf_default = empty,
            lm_dirs = (),
            omlx_dirs = (cache,),
            known_hf_caches = (),
        )

        models = models_route.collect_local_models(
            empty,
            custom_folders = [],
            sources = sources,
        )

        assert [Path(m.path).name for m in models if m.source == "omlx"] == [
            "mlx-community__Shared-4bit"
        ]

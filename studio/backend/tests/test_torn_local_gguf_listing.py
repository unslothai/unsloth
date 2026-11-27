# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF in a scan folder that is missing a shard, or has an empty one, is partial in the local
scan, so /v1/models does not offer it and a request naming it does not resolve: the treatment an
HF-cache repo with no whole quant already gets. Beside a whole quant, the torn one is not offered
either, and a Hermes split with an empty part is not listed."""

import json
from pathlib import Path

import pytest

import routes.inference as inference_route
import routes.models as models_route
from core.inference import local_model_resolver as resolver
from hub.services.models import local_inventory

_TORN = ("torn", "empty-shard", "Loose-Q8_0-00001-of-00002", "org/Torn-GGUF")


def _gguf(path: Path, size: int = 64) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4) if size else b"")


def _real_gguf(path: Path, split_count: int = 0) -> Path:
    import numpy as np
    from gguf import GGUFWriter

    path.parent.mkdir(parents = True, exist_ok = True)
    writer = GGUFWriter(str(path), "llama")
    if split_count:
        writer.add_uint16("split.count", split_count)
    writer.add_tensor("token_embd.weight", np.zeros((2, 2), dtype = np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _hf_repo_with_one_of_two_shards(hub: Path) -> None:
    repo = hub / "models--org--Torn-GGUF"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("a" * 40)
    _gguf(repo / "snapshots" / ("a" * 40) / "Torn-Q4_K_M-00001-of-00002.gguf")


@pytest.fixture
def sources(tmp_path, monkeypatch):
    folder = tmp_path / "scan"
    _gguf(folder / "torn" / "Torn-Q4_K_M-00001-of-00003.gguf")
    _gguf(folder / "empty-shard" / "Empty-Q4_K_M-00001-of-00002.gguf")
    _gguf(folder / "empty-shard" / "Empty-Q4_K_M-00002-of-00002.gguf", size = 0)
    _gguf(folder / "Loose-Q8_0-00001-of-00002.gguf")
    _gguf(folder / "Loose-Q8_0-00002-of-00002.gguf", size = 0)
    _gguf(folder / "whole" / "Whole-Q4_K_M.gguf")
    _gguf(folder / "mixed" / "Mixed-Q8_0.gguf")
    _gguf(folder / "mixed" / "Mixed-Q6_K.gguf", size = 0)
    _gguf(folder / "mixed" / "Mixed-Q4_K_M-00001-of-00002.gguf")
    _gguf(folder / "mixed" / "Mixed-Q4_K_M-00002-of-00002.gguf", size = 0)
    _gguf(folder / "hybrid" / "Hybrid-Q4_K_M-00001-of-00002.gguf")
    _gguf(folder / "hybrid" / "Hybrid-Q4_K_M-00002-of-00002.gguf", size = 0)
    (folder / "hybrid" / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (folder / "hybrid" / "model.safetensors").write_bytes(b"\0" * 64)
    hermes = tmp_path / "hermes"
    _gguf(hermes / "Hermes-Q4_K_M-00001-of-00002.gguf")
    _gguf(hermes / "Hermes-Q4_K_M-00002-of-00002.gguf", size = 0)
    hub = tmp_path / "hub"
    _hf_repo_with_one_of_two_shards(hub)
    empty = tmp_path / "empty"
    roots = local_inventory._LocalInventorySources(
        hf_cache_dir = hub,
        legacy_hf = empty,
        hf_default = empty,
        lm_dirs = (),
        ollama_dirs = (),
        hermes_dirs = (hermes,),
        known_hf_caches = (),
    )
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: roots)
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(folder)}])
    # GGUF only, so the hybrid folder's safetensors cannot stand in for its torn quant.
    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: False)
    monkeypatch.chdir(tmp_path)
    resolver._scan = (0.0, {})
    yield roots
    resolver._scan = (0.0, {})


def _name(row) -> str:
    return row.model_id or row.display_name


def test_a_torn_local_gguf_is_partial_unlisted_and_unresolved(sources):
    rows = models_route.collect_local_models(
        Path("models").resolve(), sources = sources, materialize_ollama_links = False
    )
    partial = {_name(row): row.partial for row in rows}
    assert {name: partial.get(name) for name in (*_TORN, "whole")} == {
        **{name: True for name in _TORN},
        "whole": False,
    }

    served = {
        _name(info)
        for info, *_ in inference_route._servable_catalog_rows(
            inference_route._classified_catalog(rows)
        )
    }
    assert served == {"whole", "mixed"}

    assert [name for name in _TORN if resolver.resolve_local_gguf(name) is not None] == []
    assert resolver.resolve_local_gguf("whole")[1] == "Q4_K_M"


def test_a_torn_quant_beside_whole_weights_is_not_offered(sources):
    rows = models_route.collect_local_models(
        Path("models").resolve(), sources = sources, materialize_ollama_links = False
    )
    offered = {
        _name(info): quants
        for info, _is_gguf, quants, _resident in inference_route._servable_catalog_rows(
            inference_route._classified_catalog(rows)
        )
    }
    assert offered == {"whole": ("Q4_K_M",), "mixed": ("Q8_0",)}
    assert resolver.resolve_local_gguf("mixed")[1] == "Q8_0"
    # hermes_cli stages a split once every part exists; an empty part still fails the load.
    assert "Hermes-Q4_K_M" not in {_name(row) for row in rows}
    torn = ("mixed:Q4_K_M", "mixed:Q6_K", "hybrid", "Hermes-Q4_K_M")
    assert [name for name in torn if resolver.resolve_local_gguf(name) is not None] == []


def test_a_whole_quant_folder_beside_torn_safetensors_still_resolves(sources, tmp_path):
    model = tmp_path / "scan" / "nested"
    (model / "Q4_K_M").mkdir(parents = True)
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    (model / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 64)
    _gguf(model / "Q4_K_M" / "Nested-Q4_K_M.gguf")
    _gguf(tmp_path / "scan" / "torn" / "Q8_0" / "Torn-Q8_0-00001-of-00002.gguf")

    assert resolver.resolve_local_gguf("nested")[1] == "Q4_K_M"
    assert resolver.resolve_local_gguf("torn") is None


def test_a_shard_symlink_is_checked_against_the_set_the_loader_opens(sources, tmp_path):
    hermes = sources.hermes_dirs[0]
    for name in ("Linked-Q4_K_M", "Followed-Q4_K_M"):
        for part in (1, 2):
            _gguf(tmp_path / "backing" / f"{name}-0000{part}-of-00002.gguf")
        first = f"{name}-00001-of-00002.gguf"
        (hermes / first).symlink_to(tmp_path / "backing" / first)
    # Both names exist beside this link, so the loader opens them and the empty second shard.
    _gguf(hermes / "Linked-Q4_K_M-00002-of-00002.gguf", size = 0)

    rows = models_route.collect_local_models(
        Path("models").resolve(), sources = sources, materialize_ollama_links = False
    )
    names = {_name(row) for row in rows}
    assert "Followed-Q4_K_M" in names and "Linked-Q4_K_M" not in names
    assert resolver.resolve_local_gguf("Linked-Q4_K_M") is None
    assert resolver.resolve_local_gguf("Followed-Q4_K_M") is not None


def test_a_torn_build_does_not_hide_a_whole_one_in_its_folder(sources, tmp_path):
    models = tmp_path / "models"
    _gguf(models / "pair" / "alpha-Q4_K_M-00001-of-00002.gguf")
    _gguf(models / "pair" / "beta-Q4_K_M.gguf")
    for part in (1, 2):
        _gguf(tmp_path / "backing" / f"Link-Q8_0-0000{part}-of-00002.gguf")
    first = "Link-Q8_0-00001-of-00002.gguf"
    (models / "linked").mkdir()
    (models / "linked" / first).symlink_to(tmp_path / "backing" / first)
    # A split-named link to one whole single-file GGUF.
    _gguf(tmp_path / "backing" / "whole-Q6_K.gguf")
    (models / "alias").mkdir()
    (models / "alias" / "alias-Q6_K-00001-of-00002.gguf").symlink_to(
        tmp_path / "backing" / "whole-Q6_K.gguf"
    )

    rows = models_route.collect_local_models(
        models.resolve(), sources = sources, materialize_ollama_links = False
    )
    partial = {_name(row): row.partial for row in rows}
    assert (partial.get("pair"), partial.get("linked"), partial.get("alias")) == (
        False,
        False,
        False,
    )
    offered = {
        _name(info): quants
        for info, _is_gguf, quants, _resident in inference_route._servable_catalog_rows(
            inference_route._classified_catalog(rows)
        )
    }
    assert (offered.get("pair"), offered.get("linked")) == (("beta-Q4_K_M",), ("Q8_0",))
    assert resolver.resolve_local_gguf("pair")[1] == "beta-Q4_K_M"
    assert resolver.resolve_local_gguf("pair:Q4_K_M") is None
    assert resolver.resolve_local_gguf("linked")[1] == "Q8_0"
    assert offered.get("alias") == ("Q6_K",)
    assert resolver.resolve_local_gguf("alias")[1] == "Q6_K"


def test_a_shard_link_target_is_judged_by_its_split_metadata(sources, tmp_path):
    # An HF blob has no name to read; a renamed first shard keeps split.count and llama.cpp rejects it.
    models = tmp_path / "models"
    blob = _real_gguf(tmp_path / "blobs" / ("b" * 64))
    renamed = _real_gguf(tmp_path / "backing" / "renamed-Q6_K.gguf", split_count = 2)
    for name, target in (("blob", blob), ("renamed", renamed)):
        (models / name).mkdir(parents = True)
        (models / name / f"{name}-Q6_K-00001-of-00002.gguf").symlink_to(target)

    rows = models_route.collect_local_models(
        models.resolve(), sources = sources, materialize_ollama_links = False
    )
    partial = {_name(row): row.partial for row in rows}
    assert (partial.get("blob"), partial.get("renamed")) == (False, True)
    offered = {
        _name(info): quants
        for info, _is_gguf, quants, _resident in inference_route._servable_catalog_rows(
            inference_route._classified_catalog(rows)
        )
    }
    assert offered.get("blob") == ("Q6_K",) and "renamed" not in offered
    assert resolver.resolve_local_gguf("blob")[1] == "Q6_K"
    assert resolver.resolve_local_gguf("renamed") is None


def test_split_shards_must_match_the_loader_filenames(tmp_path):
    from utils.models.model_config import split_set_is_loadable

    first = tmp_path / "Model-Q4_K_M-00001-of-00002.gguf"
    second = tmp_path / "model-Q4_K_M-00002-of-00002.gguf"
    _gguf(first)
    _gguf(second)
    expected = tmp_path / "Model-Q4_K_M-00002-of-00002.gguf"
    if expected.is_file():
        pytest.skip("filesystem resolves case variants")
    assert not split_set_is_loadable(first)
    second.rename(expected)
    assert split_set_is_loadable(first)


def test_mlx_index_free_shards_are_servable_only_when_whole(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file
    from utils.hardware import hardware

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    for index in (1, 2):
        save_file(
            {f"tensor.{index}": np.array([index], dtype = np.float32)},
            str(tmp_path / f"model-{index:05d}-of-00002.safetensors"),
        )
    monkeypatch.setattr(hardware, "DEVICE", hardware.DeviceType.CUDA)
    assert not local_inventory._mlx_has_complete_index_free_shards(tmp_path)
    monkeypatch.setattr(hardware, "DEVICE", hardware.DeviceType.MLX)
    assert local_inventory._mlx_has_complete_index_free_shards(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()
    assert not local_inventory._mlx_has_complete_index_free_shards(tmp_path)

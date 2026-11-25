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

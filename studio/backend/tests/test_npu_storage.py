# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU models in Settings > Storage: sized, listed, and deleted only through lemond."""

import json
import sys
from pathlib import Path

import pytest

import core.inference.npu_backend as nb
from core.inference.npu_backend import LemonadeNpuBackend, NpuError, NpuModel
from utils import cache_inventory
from utils.cache_inventory import cache_breakdown, definition_for, describe_cache, purge_cache


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"x" * size)
    return path


def _model(model_id: str, checkpoint: str, downloaded: bool = True) -> NpuModel:
    return NpuModel(
        id = model_id,
        checkpoint = checkpoint,
        size_gb = None,
        downloaded = downloaded,
        labels = ("chat",),
        max_context_length = None,
    )


class _Lemond(LemonadeNpuBackend):
    """Stands in for lemond's catalog and /v1/delete; a delete removes the folder as lemond would."""

    def __init__(self, root: Path, models: list[NpuModel], folders: dict[str, Path]):
        super().__init__(root)
        self.models = models
        self.folders = folders
        self.deleted: list[str] = []

    def catalog(self):
        return list(self.models)

    def delete(self, model_id: str) -> None:
        if model_id == "loaded-FLM":
            raise NpuError("Unload the model before deleting it.")
        self.deleted.append(model_id)
        for path in sorted(self.folders[model_id].rglob("*"), reverse = True):
            path.rmdir() if path.is_dir() else path.unlink()
        self.folders[model_id].rmdir()


@pytest.fixture
def npu_home(tmp_path, monkeypatch):
    root = tmp_path / "studio" / "lemonade"
    monkeypatch.setattr(nb, "_npu_root", lambda: root)
    monkeypatch.setattr(cache_inventory, "_size_cache", {})
    monkeypatch.setattr(cache_inventory, "_size_epochs", {})
    binary = root / "cache" / "bin" / "flm" / "npu" / "v1"
    binary /= "flm.exe" if sys.platform == "win32" else "flm"
    _write(binary, 1)
    (binary.parent / "model_list.json").write_text(
        json.dumps(
            {
                "model_path": "models",
                "models": {
                    "qwen3": {
                        "0.6b": {"name": "Qwen3-0.6B-NPU2"},
                        "4b": {"name": "Qwen3-4B-NPU2"},
                    }
                },
            }
        ),
        encoding = "utf-8",
    )
    models = root / "flm" / "models"
    _write(models / "Qwen3-0.6B-NPU2" / "model.q4nx", 600)
    _write(models / "Qwen3-4B-NPU2" / "model.q4nx", 4000)
    _write(models / "Stray-NPU2" / "model.q4nx", 30)
    _write(root / "flm" / "notes.txt", 5)
    backend = _Lemond(
        root,
        [_model("qwen3-0.6b-FLM", "qwen3:0.6b"), _model("qwen3-4b-FLM", "qwen3:4b")],
        {"qwen3-0.6b-FLM": models / "Qwen3-0.6B-NPU2", "qwen3-4b-FLM": models / "Qwen3-4B-NPU2"},
    )
    monkeypatch.setattr(nb, "get_npu_backend", lambda: backend)
    return backend


def test_npu_models_are_a_storage_source_with_their_size(npu_home):
    entry = describe_cache(definition_for("npu_models"))
    assert entry["present"] is True
    assert entry["group"] == "models"
    assert entry["opt_in"] is True
    assert entry["purgeable"] is True
    assert entry["paths"] == [str(npu_home.root / "flm")]
    assert entry["size_bytes"] == 600 + 4000 + 30 + 5


def test_no_npu_folder_means_no_storage_row(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "_npu_root", lambda: tmp_path / "missing")
    entry = describe_cache(definition_for("npu_models"))
    assert entry["present"] is False
    assert entry["size_bytes"] == 0


def test_the_breakdown_names_each_model_folder_by_checkpoint(npu_home):
    rows = {row["name"]: row for row in cache_breakdown("npu_models")["entries"]}
    assert set(rows) == {"Qwen3-0.6B-NPU2", "Qwen3-4B-NPU2", "Stray-NPU2", "notes.txt"}
    assert rows["Qwen3-4B-NPU2"]["size_bytes"] == 4000
    assert rows["Qwen3-4B-NPU2"]["checkpoint"] == "qwen3:4b"
    assert rows["Qwen3-4B-NPU2"]["delete_allowed"] is True
    # Nothing maps these to a lemond model, so the UI offers no delete for them.
    for name in ("Stray-NPU2", "notes.txt"):
        assert rows[name]["delete_allowed"] is False
        assert "checkpoint" not in rows[name]
    assert [row["name"] for row in cache_breakdown("npu_models")["entries"]][0] == "Qwen3-4B-NPU2"


def test_clearing_npu_models_deletes_through_lemond_not_the_disk(npu_home):
    result = purge_cache("npu_models")
    assert npu_home.deleted == ["qwen3-0.6b-FLM", "qwen3-4b-FLM"]
    assert result["removed_entries"] == 2
    assert result["freed_bytes"] == 4600
    assert result["errors"] == []
    # What lemond does not own stays: an rmtree of the folder would have taken these too.
    flm = npu_home.root / "flm"
    assert (flm / "models" / "Stray-NPU2" / "model.q4nx").exists()
    assert (flm / "notes.txt").exists()


def test_a_model_lemond_refuses_to_delete_is_reported_and_kept(npu_home):
    npu_home.models.append(_model("loaded-FLM", "qwen3:8b"))
    npu_home.models.append(_model("not-downloaded-FLM", "qwen3:14b", downloaded = False))
    result = purge_cache("npu_models")
    assert npu_home.deleted == ["qwen3-0.6b-FLM", "qwen3-4b-FLM"]
    assert result["errors"] == ["Unload the model before deleting it."]


def test_clearing_waits_for_a_running_npu_download(npu_home, monkeypatch):
    import routes.npu as npu_routes

    monkeypatch.setattr(npu_routes, "downloads_running", lambda: True)
    result = purge_cache("npu_models")
    assert npu_home.deleted == []
    assert result["errors"] == [cache_inventory._PURGE_BUSY]


def test_a_delete_drops_the_remembered_size(npu_home, monkeypatch):
    import asyncio

    import routes.npu as npu_routes

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: npu_home)
    describe = describe_cache(definition_for("npu_models"))
    cache_inventory._size_cache["npu_models"] = (0.0, 0.0, describe, 0.0)
    asyncio.run(npu_routes.delete_npu_model("qwen3-4b-FLM"))
    assert "npu_models" not in cache_inventory._size_cache

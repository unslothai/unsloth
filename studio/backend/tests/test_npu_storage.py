# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU models in Settings > Storage: sized, listed, and deleted only through lemond."""

import json
import sys
from pathlib import Path

import pytest

import core.inference.npu_backend as nb
from core.inference.npu_backend import LemonadeNpuBackend, NpuError
from utils import cache_inventory
from utils.cache_inventory import cache_breakdown, definition_for, describe_cache, purge_cache


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"x" * size)
    return path


def _model(
    model_id: str,
    checkpoint: str,
    downloaded: bool = True,
    labels = ("chat",),
) -> dict:
    return {
        "id": model_id,
        "checkpoint": checkpoint,
        "recipe": "flm",
        "downloaded": downloaded,
        "labels": list(labels),
    }


class _Lemond(LemonadeNpuBackend):
    """Stands in for lemond's /v1/models and /v1/delete; a delete removes the folder as lemond would."""

    def __init__(self, root: Path, models: list[dict], folders: dict[str, Path]):
        super().__init__(root)
        self.models = models
        self.folders = folders
        self.deleted: list[str] = []

    def _flm_rows(self):
        return list(self.models)

    def delete(self, model_id: str) -> None:
        if model_id == "loaded-FLM":
            raise NpuError("Unload the model before deleting it.")
        self.deleted.append(model_id)
        for path in sorted(self.folders[model_id].rglob("*"), reverse = True):
            path.rmdir() if path.is_dir() else path.unlink()
        self.folders[model_id].rmdir()
        for row in self.models:
            if row["id"] == model_id:
                row["downloaded"] = False


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
                    },
                    "embed-gemma": {"300m": {"name": "EmbedGemma-300M-NPU2"}},
                },
            }
        ),
        encoding = "utf-8",
    )
    models = root / "flm" / "models"
    _write(models / "Qwen3-0.6B-NPU2" / "model.q4nx", 600)
    _write(models / "Qwen3-4B-NPU2" / "model.q4nx", 4000)
    _write(models / "EmbedGemma-300M-NPU2" / "model.q4nx", 300)
    _write(models / "Stray-NPU2" / "model.q4nx", 30)
    _write(root / "flm" / "notes.txt", 5)
    backend = _Lemond(
        root,
        [
            _model("qwen3-0.6b-FLM", "qwen3:0.6b"),
            _model("qwen3-4b-FLM", "qwen3:4b"),
            # Not a chat model, so the Hub's catalog hides it; its bytes still count here.
            _model("embed-gemma-300m-FLM", "embed-gemma:300m", labels = ("embeddings",)),
        ],
        {
            "qwen3-0.6b-FLM": models / "Qwen3-0.6B-NPU2",
            "qwen3-4b-FLM": models / "Qwen3-4B-NPU2",
            "embed-gemma-300m-FLM": models / "EmbedGemma-300M-NPU2",
        },
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
    assert entry["size_bytes"] == 600 + 4000 + 300 + 30 + 5


def test_no_npu_folder_means_no_storage_row(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "_npu_root", lambda: tmp_path / "missing")
    entry = describe_cache(definition_for("npu_models"))
    assert entry["present"] is False
    assert entry["size_bytes"] == 0


def test_the_breakdown_names_each_model_folder_by_checkpoint(npu_home):
    rows = {row["name"]: row for row in cache_breakdown("npu_models")["entries"]}
    assert set(rows) == {
        "Qwen3-0.6B-NPU2",
        "Qwen3-4B-NPU2",
        "EmbedGemma-300M-NPU2",
        "Stray-NPU2",
        "notes.txt",
    }
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
    assert npu_home.deleted == ["qwen3-0.6b-FLM", "qwen3-4b-FLM", "embed-gemma-300m-FLM"]
    assert result["removed_entries"] == 3
    assert result["freed_bytes"] == 4900
    assert result["errors"] == []
    # What lemond does not own stays: an rmtree of the folder would have taken these too.
    flm = npu_home.root / "flm"
    assert (flm / "models" / "Stray-NPU2" / "model.q4nx").exists()
    assert (flm / "notes.txt").exists()


def test_a_model_lemond_refuses_to_delete_is_reported_and_kept(npu_home):
    npu_home.models.append(_model("loaded-FLM", "qwen3:8b"))
    npu_home.models.append(_model("not-downloaded-FLM", "qwen3:14b", downloaded = False))
    result = purge_cache("npu_models")
    assert npu_home.deleted == ["qwen3-0.6b-FLM", "qwen3-4b-FLM", "embed-gemma-300m-FLM"]
    assert result["errors"] == ["Unload the model before deleting it."]


def test_clearing_takes_a_partial_pull_lemond_calls_not_downloaded(npu_home):
    npu_home.models[1]["downloaded"] = False
    result = purge_cache("npu_models")
    assert npu_home.deleted == ["qwen3-0.6b-FLM", "qwen3-4b-FLM", "embed-gemma-300m-FLM"]
    assert not (npu_home.root / "flm" / "models" / "Qwen3-4B-NPU2").exists()
    assert result["errors"] == []


def test_a_folder_deletes_by_checkpoint_even_for_a_model_the_catalog_hides(npu_home):
    assert "embed-gemma-300m-FLM" not in [model.id for model in npu_home.catalog()]
    assert npu_home.delete_checkpoint("embed-gemma:300m") == "embed-gemma-300m-FLM"
    assert npu_home.deleted == ["embed-gemma-300m-FLM"]
    with pytest.raises(NpuError, match = "no FastFlowLM model for gone:1b"):
        npu_home.delete_checkpoint("gone:1b")
    assert npu_home.deleted == ["embed-gemma-300m-FLM"]


def test_clearing_waits_for_a_running_npu_download(npu_home, monkeypatch):
    import routes.npu as npu_routes

    monkeypatch.setitem(npu_routes._downloads, "qwen3-4b-FLM", npu_routes._Download())
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


def test_the_storage_delete_route_resolves_the_checkpoint(npu_home, monkeypatch):
    import asyncio

    import routes.npu as npu_routes

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: npu_home)
    cache_inventory._size_cache["npu_models"] = (0.0, 0.0, {}, 0.0)
    result = asyncio.run(npu_routes.delete_npu_checkpoint("qwen3:0.6b"))
    assert result == {"status": "deleted", "model": "qwen3-0.6b-FLM"}
    assert "npu_models" not in cache_inventory._size_cache


def test_a_folder_still_downloading_is_not_deleted(npu_home, monkeypatch):
    import asyncio

    import routes.npu as npu_routes

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: npu_home)
    job = npu_routes._Download()
    monkeypatch.setitem(npu_routes._downloads, "qwen3-4b-FLM", job)
    with pytest.raises(npu_routes.HTTPException, match = "Cancel the download of qwen3-4b-FLM"):
        asyncio.run(npu_routes.delete_npu_checkpoint("qwen3:4b"))
    assert npu_home.deleted == []
    job.finished = True
    asyncio.run(npu_routes.delete_npu_checkpoint("qwen3:4b"))
    assert npu_home.deleted == ["qwen3-4b-FLM"]


def test_a_pull_asked_for_during_a_storage_delete_is_refused_without_waiting(npu_home, monkeypatch):
    import asyncio
    import threading

    import routes.npu as npu_routes

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: npu_home)
    monkeypatch.setattr(npu_routes, "_downloads", {})
    started = []

    class _Npu:
        def download(self, model_id):
            started.append(model_id)
            return iter(())

    def start(model_id):
        # A thread with a timeout: a delete holding _downloads_lock would hang this, not fail it.
        outcome = []

        def _run():
            try:
                npu_routes._start_download(_Npu(), model_id)
                outcome.append("started")
            except NpuError as exc:
                outcome.append(str(exc))

        worker = threading.Thread(target = _run, daemon = True)
        worker.start()
        worker.join(timeout = 5)
        return outcome[0] if outcome else "blocked"

    seen = []
    delete = npu_home.delete

    def watched(model_id):
        seen.append((start("qwen3-4b-FLM"), start("qwen3-0.6b-FLM")))
        delete(model_id)

    npu_home.delete = watched
    asyncio.run(npu_routes.delete_npu_checkpoint("qwen3:4b"))
    # Only the model being deleted is held back.
    assert seen == [
        ("qwen3-4b-FLM is being deleted. Download it again once that finishes.", "started")
    ]
    assert npu_routes._deleting == []

    npu_routes._downloads.clear()
    seen.clear()
    purge_cache("npu_models")
    # Clear holds back every model.
    assert seen and all(
        pair[0].endswith("is being deleted. Download it again once that finishes.")
        and pair[1].endswith("is being deleted. Download it again once that finishes.")
        for pair in seen
    )
    assert npu_routes._deleting == []
    assert start("qwen3-4b-FLM") == "started"

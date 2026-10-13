# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reveal and delete for LM Studio / custom-folder / ./models rows: only listed paths, never a
scan root, never a loaded model."""

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from hub.schemas.inventory import LocalModelInfo, LocalModelListResponse
from hub.services.models import account_access, local_inventory, local_model_actions as actions

REAL_GUARD = actions._load_state_blocks_delete


def _row(path: Path, source: str) -> LocalModelInfo:
    return LocalModelInfo(id = str(path), display_name = path.name, path = str(path), source = source)


@pytest.fixture
def lm_root(tmp_path, monkeypatch):
    root = tmp_path / "lmstudio"
    model = root / "unsloth" / "gemma-GGUF"
    model.mkdir(parents = True)
    (model / "gemma-Q4.gguf").write_bytes(b"GGUF")
    folder = tmp_path / "custom"
    (folder / "solo").mkdir(parents = True)
    rows = [
        _row(model, "lmstudio"),
        _row(folder, "custom"),
        _row(tmp_path / "hf" / "models--a--b", "hf_cache"),
        _row(tmp_path / "ollama" / "blob", "ollama"),
    ]

    async def listing(models_dir: str = "./models"):
        return LocalModelListResponse(models_dir = "", hf_cache_dir = "", models = rows)

    sources = local_inventory._LocalInventorySources(
        tmp_path / "hf", tmp_path / "legacy", tmp_path / "default", (root,), (), (), (), ()
    )
    monkeypatch.setattr(local_inventory, "list_local_models_response", listing)
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)

    async def folders():
        return [{"path": str(folder)}]

    monkeypatch.setattr(local_inventory, "_load_custom_folders", folders)
    monkeypatch.setattr(account_access, "require_installation_owner", lambda: None)
    monkeypatch.setattr(actions, "_load_state_blocks_delete", lambda paths: None)
    return root, model, folder


def test_delete_removes_listed_lmstudio_model_and_empty_publisher(lm_root):
    root, model, _ = lm_root
    asyncio.run(actions.delete_local_model_response(str(model)))
    assert not model.exists()
    assert not model.parent.exists()
    assert root.is_dir()


def test_delete_refuses_unlisted_path_and_hf_cache_rows(lm_root, tmp_path):
    root, model, _ = lm_root
    stray = root / "unsloth" / "other"
    stray.mkdir()
    for path in (stray, tmp_path / "hf" / "models--a--b", tmp_path / "ollama" / "blob"):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(actions.delete_local_model_response(str(path)))
        assert caught.value.status_code == 404
    assert stray.is_dir() and model.is_dir()


def test_delete_refuses_a_row_that_is_the_scan_folder_itself(lm_root):
    _, _, folder = lm_root
    with pytest.raises(HTTPException) as caught:
        asyncio.run(actions.delete_local_model_response(str(folder)))
    assert caught.value.status_code == 400
    assert (folder / "solo").is_dir()


def test_delete_refuses_a_loaded_model(lm_root, monkeypatch):
    _, model, _ = lm_root
    monkeypatch.setattr(
        actions, "_load_state_blocks_delete", lambda paths: "Unload the model before deleting"
    )
    with pytest.raises(HTTPException) as caught:
        asyncio.run(actions.delete_local_model_response(str(model)))
    assert caught.value.status_code == 400
    assert model.is_dir()


def test_load_guard_matches_a_gguf_loaded_from_inside_the_folder(lm_root, monkeypatch):
    from hub.services.models import deletion

    _, model, _ = lm_root
    loaded = str(model / "gemma-Q4.gguf")
    monkeypatch.setattr(
        deletion,
        "_llama_cpp_blocks_delete",
        lambda path, variant: deletion._loaded_id_matches_repo(loaded, path),
    )
    monkeypatch.setattr(deletion, "_inference_backend_blocks_delete", lambda path: False)
    monkeypatch.setattr(deletion, "_diffusion_blocks_delete", lambda path: None)
    monkeypatch.setattr(deletion, "_video_blocks_delete", lambda path: None)
    assert REAL_GUARD((str(model),)) == "Unload the model before deleting"


def test_reveal_accepts_listed_ollama_row_and_rejects_hf_cache(lm_root, tmp_path, monkeypatch):
    import utils.paths.path_utils as path_utils

    target = tmp_path / "ollama" / "blob"
    target.parent.mkdir(parents = True)
    target.write_bytes(b"x")
    seen = []
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", lambda path: seen.append(path))
    asyncio.run(actions.reveal_local_model_response(str(target)))
    assert seen == [target]
    with pytest.raises(HTTPException) as caught:
        asyncio.run(actions.reveal_local_model_response(str(tmp_path / "hf" / "models--a--b")))
    assert caught.value.status_code == 404

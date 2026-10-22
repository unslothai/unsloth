# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.inference as inference_route
import routes.models as models_route
from utils.hidden_models import is_hidden_model

CLEF_GGUF = "ggml-org/Clef-Flash-GGUF"
CLEF_SNAPSHOT_FILE = "/root/.cache/huggingface/hub/models--ggml-org--Clef-Flash-GGUF/snapshots/4a19/Clef-Flash-Q8_0.gguf"


def _repo(repo_id, files, repo_path):
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = [SimpleNamespace(files = files)],
    )


def _file(name, size_on_disk):
    return SimpleNamespace(file_name = name, size_on_disk = size_on_disk, blob_path = None)


@pytest.mark.parametrize(
    "value",
    [
        CLEF_GGUF,
        "ggml-org/clef-flash-gguf",
        "ggml-org/Kev-0.8B-GGUF",
        "Cloudflare/clef-flash",
        "convaiinnovations/laya",
        "jaredpalmer/kev-0.8b",
        "openjev/openjev-MLX",
        CLEF_SNAPSHOT_FILE,
    ],
)
def test_decision_api_models_are_hidden_from_chat(value):
    assert is_hidden_model(value) is True


def test_chat_models_with_similar_names_stay_visible():
    assert is_hidden_model("unsloth/Qwen3-0.6B-GGUF") is False
    assert is_hidden_model("user/Clef-Flash-GGUF") is False
    assert is_hidden_model("ggml-org/Clef-Flash-GGUF-finetune") is False
    assert is_hidden_model("Qwen/Qwen3.5-0.8B-Base") is False


def test_cached_gguf_list_drops_a_decision_api_download(monkeypatch, tmp_path):
    clef = _repo(
        CLEF_GGUF,
        [_file("Clef-Flash-Q8_0.gguf", 9_000), _file("mmproj-Clef-Flash-Q8_0.gguf", 600)],
        tmp_path / "models--ggml-org--Clef-Flash-GGUF",
    )
    chat = _repo(
        "unsloth/Qwen3-0.6B-GGUF",
        [_file("Qwen3-0.6B-Q4_K_M.gguf", 400)],
        tmp_path / "models--unsloth--Qwen3-0.6B-GGUF",
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [clef, chat])]
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert [row["repo_id"] for row in result["cached"]] == ["unsloth/Qwen3-0.6B-GGUF"]


def test_the_chat_picker_hides_decision_api_repos_by_exact_id():
    _needles, exact_ids, _paths = models_route.hidden_model_matchers()

    assert "ggml-org/clef-flash-gguf" in exact_ids
    assert "cloudflare/clef-flash" in exact_ids
    assert "unsloth/qwen3-0.6b-gguf" not in exact_ids


def test_the_model_hub_keeps_a_decision_api_download_as_a_non_chat_row(monkeypatch, tmp_path):
    from hub.services.models import cache_inventory

    snapshot = tmp_path / "models--ggml-org--Clef-Flash-GGUF" / "snapshots" / "4a19"
    repo = SimpleNamespace(
        repo_id = CLEF_GGUF, repo_type = "model", repo_path = snapshot.parents[1], revisions = []
    )
    monkeypatch.setattr(cache_inventory, "_repo_gguf_size_bytes", lambda _repo: 10)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_last_modified", lambda _repo: 1.0)
    monkeypatch.setattr(cache_inventory, "_cached_model_snapshot_path", lambda _path: None)
    monkeypatch.setattr(
        cache_inventory,
        "_repo_gguf_payload_snapshots",
        lambda _repo: (snapshot, frozenset({str(snapshot)})),
    )
    monkeypatch.setattr(cache_inventory, "_gguf_variant_state_summary", lambda *a, **k: (False, 0))
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan, "is_gguf_repo_partial", lambda *a, **k: False
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest, "build_variant_state_index", lambda *a, **k: None
    )
    monkeypatch.setattr(cache_inventory, "_cached_row_task", lambda *a, **k: "text-generation")
    monkeypatch.setattr(
        cache_inventory,
        "_cache_inventory_fields",
        lambda *a, hidden_infra = False, **k: {"capabilities": {"can_chat": not hidden_infra}},
    )

    rows = cache_inventory._scan_cached_gguf(
        cache_scans = [SimpleNamespace(repos = [repo])], active_hub_cache = tmp_path
    )

    assert [row["repo_id"] for row in rows] == [CLEF_GGUF]
    assert rows[0]["capabilities"]["can_chat"] is False


def _patch_load(monkeypatch):
    from contextlib import asynccontextmanager

    from core.inference import llama_keepwarm

    @asynccontextmanager
    async def _gate():
        yield

    loaded = []

    async def _impl(request, *_args, **_kwargs):
        loaded.append(request.model_path)
        return {"status": "loaded"}

    monkeypatch.setattr(inference_route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(inference_route, "_pending_load_attempts", {})
    monkeypatch.setattr(llama_keepwarm, "inference_lifecycle_gate", _gate)
    monkeypatch.setattr(inference_route, "_run_tracked_load_model_impl", _impl)
    return loaded


@pytest.mark.parametrize("model_path", [CLEF_GGUF, "Cloudflare/clef-flash", CLEF_SNAPSHOT_FILE])
def test_loading_a_decision_api_model_as_chat_is_refused(monkeypatch, model_path):
    loaded = _patch_load(monkeypatch)
    request = inference_route.LoadRequest(model_path = model_path, gguf_variant = "Q8_0")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.load_model_gated(request, object(), "test"))

    assert exc.value.status_code == 400
    assert "Decision API" in exc.value.detail
    assert "/v1/systemone" in exc.value.detail
    assert loaded == []


def test_a_chat_model_still_loads(monkeypatch):
    loaded = _patch_load(monkeypatch)
    request = inference_route.LoadRequest(model_path = "unsloth/Qwen3-0.6B-GGUF")

    asyncio.run(inference_route.load_model_gated(request, object(), "test"))

    assert loaded == ["unsloth/Qwen3-0.6B-GGUF"]

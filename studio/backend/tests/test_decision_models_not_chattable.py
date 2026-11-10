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
from utils.account_context import AccountContext, arun_as, run_as
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


def _cache_repo(hub, repo_id, files):
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "4a19"
    snapshot.mkdir(parents = True)
    (repo / "blobs").mkdir()
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("4a19")
    for index, (name, data) in enumerate(files.items()):
        blob = repo / "blobs" / f"blob{index}"
        blob.write_bytes(data)
        (snapshot / name).symlink_to(f"../../blobs/blob{index}")
    return snapshot


def test_model_hub_and_train_keep_decision_api_downloads_as_non_chat_rows(tmp_path):
    from huggingface_hub import scan_cache_dir

    from hub.services.models import cache_inventory

    gguf = b"GGUF\x03\x00\x00\x00" + b"\x00" * 4096
    config = b'{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}'
    weights = {"config.json": config, "model.safetensors": b"\x00" * 4096}
    _cache_repo(tmp_path, CLEF_GGUF, {"Clef-Flash-Q8_0.gguf": gguf})
    _cache_repo(tmp_path, "unsloth/Qwen3-0.6B-GGUF", {"Qwen3-0.6B-Q4_K_M.gguf": gguf})
    _cache_repo(tmp_path, "Cloudflare/clef-flash", weights)
    _cache_repo(tmp_path, "unsloth/Qwen3-0.6B", weights)
    scans = [scan_cache_dir(tmp_path)]

    rows = {
        row["repo_id"]: row["capabilities"]
        for scan in (cache_inventory._scan_cached_gguf, cache_inventory._scan_cached_models)
        for row in scan(cache_scans = scans, active_hub_cache = tmp_path)
    }

    assert rows[CLEF_GGUF]["can_chat"] is False
    assert rows["Cloudflare/clef-flash"]["can_chat"] is False
    assert rows["Cloudflare/clef-flash"]["can_train"] is True
    assert rows["unsloth/Qwen3-0.6B-GGUF"]["can_chat"] is True
    assert rows["unsloth/Qwen3-0.6B"]["can_chat"] is True


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


@pytest.mark.parametrize("model_path", ["unsloth/Qwen3-0.6B-GGUF", f"{CLEF_GGUF}-finetune"])
def test_a_chat_model_still_loads(monkeypatch, model_path):
    loaded = _patch_load(monkeypatch)
    request = inference_route.LoadRequest(model_path = model_path)

    asyncio.run(inference_route.load_model_gated(request, object(), "test"))

    assert loaded == [model_path]


# Decision models that are not built-in repos: Studio's own decision runs, a downloaded one in outputs, a
# third-party one in the HF cache. Only their files say what they are.
THIRD_PARTY_CLEF = "someone/clef-finetune"
ALICE = AccountContext("a" * 32, "alice")
_QWEN3_CONFIG = b'{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}'
CHAT_FILES = {"config.json": _QWEN3_CONFIG, "model.safetensors": b"\x00" * 4096}
CLEF_FILES = {
    **CHAT_FILES,
    "joint_head.safetensors": b"\x00" * 64,
    "joint_head_config.json": b"{}",
}
CLEF_ADAPTER_FILES = {
    "adapter_config.json": b'{"base_model_name_or_path": "unsloth/Qwen3-0.6B"}',
    "adapter_model.safetensors": b"\x00" * 4096,
    "joint_head.safetensors": b"\x00" * 64,
    "joint_head_config.json": b"{}",
}
LAYA_FILES = {
    "rl_agent_config.json": b"{}",
    "model.safetensors": b"\x00" * 4096,
    "encoder/config.json": b"{}",
    "tokenizer/tokenizer.json": b"{}",
}


def _write(folder, files):
    for name, data in files.items():
        (folder / name).parent.mkdir(parents = True, exist_ok = True)
        (folder / name).write_bytes(data)
    return folder


class _LoadStarted(Exception):
    """The load got past _load_model_impl's checks."""


def _patch_load_past_the_checks(monkeypatch):
    """load_model_gated and _load_model_impl run for real; the first load step after the impl's checks raises."""
    real_tracked_impl = inference_route._run_tracked_load_model_impl
    _patch_load(monkeypatch)
    monkeypatch.setattr(inference_route, "_run_tracked_load_model_impl", real_tracked_impl)
    started = []

    def _start(model_ref):
        started.append(model_ref)
        raise _LoadStarted(model_ref)

    monkeypatch.setattr(inference_route, "_repo_is_in_the_hub_cache", _start)
    return started


def test_decision_runs_in_outputs_are_flagged_in_the_fine_tune_list():
    from utils.paths import exports_root, outputs_root

    _write(outputs_root() / "clef-run", CLEF_FILES)
    _write(outputs_root() / "clef-lora-run", CLEF_ADAPTER_FILES)
    _write(outputs_root() / "qwen3-run", CHAT_FILES)

    response = asyncio.run(
        models_route.scan_loras(
            outputs_dir = str(outputs_root()),
            exports_dir = str(exports_root()),
            hf_token = None,
            current_subject = "test-user",
        )
    )

    # Still listed, as the Library and the Audio page read this list too; Chat skips the flagged rows.
    rows = {row.display_name: (row.export_type, row.is_decision) for row in response.loras}
    assert rows == {
        "clef-run": ("merged", True),
        "clef-lora-run": ("lora", True),
        "qwen3-run": ("merged", False),
    }


@pytest.mark.parametrize(
    ("files", "refused"),
    [(CLEF_FILES, True), (CLEF_ADAPTER_FILES, True), (LAYA_FILES, True), (CHAT_FILES, False)],
    ids = ["clef", "clef-adapter", "laya", "chat-fine-tune"],
)
def test_only_a_chat_fine_tune_in_outputs_loads_as_chat(monkeypatch, files, refused):
    from utils.paths import outputs_root

    folder = _write(outputs_root() / "run", files)
    started = _patch_load_past_the_checks(monkeypatch)
    request = inference_route.LoadRequest(model_path = str(folder))

    if not refused:
        with pytest.raises(_LoadStarted):
            asyncio.run(inference_route.load_model_gated(request, object(), "test"))
        assert started == [str(folder)]
        return
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.load_model_gated(request, object(), "test"))
    assert exc.value.status_code == 400
    assert "Decision API" in exc.value.detail
    assert started == []


def test_a_cached_third_party_decision_repo_is_a_non_chat_row(tmp_path):
    from huggingface_hub import scan_cache_dir

    from hub.services.models import cache_inventory

    _cache_repo(tmp_path, THIRD_PARTY_CLEF, CLEF_FILES)
    _cache_repo(tmp_path, "unsloth/Qwen3-0.6B", CHAT_FILES)

    rows = {
        row["repo_id"]: row["capabilities"]
        for row in cache_inventory._scan_cached_models(
            cache_scans = [scan_cache_dir(tmp_path)], active_hub_cache = tmp_path
        )
    }

    assert rows[THIRD_PARTY_CLEF]["can_chat"] is False
    assert rows[THIRD_PARTY_CLEF]["can_train"] is True
    assert rows["unsloth/Qwen3-0.6B"]["can_chat"] is True


def test_the_chat_picker_hides_a_cached_third_party_decision_repo(monkeypatch, tmp_path):
    from huggingface_hub import scan_cache_dir

    _cache_repo(tmp_path, THIRD_PARTY_CLEF, CLEF_FILES)
    _cache_repo(tmp_path, "unsloth/Qwen3-0.6B", CHAT_FILES)
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan_cache_dir(tmp_path)])

    _needles, exact_ids, _paths = models_route.hidden_model_matchers()

    assert THIRD_PARTY_CLEF in exact_ids
    assert "unsloth/qwen3-0.6b" not in exact_ids


def test_loading_a_cached_third_party_decision_repo_as_chat_is_refused(monkeypatch, tmp_path):
    _cache_repo(tmp_path, THIRD_PARTY_CLEF, CLEF_FILES)
    _cache_repo(tmp_path, "unsloth/Qwen3-0.6B", CHAT_FILES)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    started = _patch_load_past_the_checks(monkeypatch)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.load_model_gated(
                inference_route.LoadRequest(model_path = THIRD_PARTY_CLEF), object(), "test"
            )
        )
    # Auto-switch and preview call the impl directly.
    with pytest.raises(HTTPException) as direct:
        asyncio.run(
            inference_route._load_model_impl(
                inference_route.LoadRequest(model_path = THIRD_PARTY_CLEF), None, "test"
            )
        )
    with pytest.raises(_LoadStarted):
        asyncio.run(
            inference_route.load_model_gated(
                inference_route.LoadRequest(model_path = "unsloth/Qwen3-0.6B"), object(), "test"
            )
        )

    assert exc.value.status_code == direct.value.status_code == 400
    assert "Decision API" in exc.value.detail
    assert started == ["unsloth/Qwen3-0.6B"]


def test_a_managed_account_without_a_token_cannot_load_a_cached_decision_repo_as_chat(
    monkeypatch, tmp_path
):
    from hub.services.models import account_access

    _cache_repo(tmp_path, THIRD_PARTY_CLEF, CLEF_FILES)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    checked = []
    monkeypatch.setattr(
        account_access, "require_model_access", lambda reference, *a, **k: checked.append(reference)
    )
    started = _patch_load_past_the_checks(monkeypatch)
    request = inference_route.LoadRequest(model_path = THIRD_PARTY_CLEF)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(arun_as(ALICE, inference_route._load_model_impl(request, None, "alice")))

    # The access check runs first; its token is the anonymous sentinel, which alone would not read the cache.
    assert checked == [THIRD_PARTY_CLEF]
    assert exc.value.status_code == 400
    assert started == []


def test_a_layout_check_that_raises_never_fails_a_load(monkeypatch, tmp_path):
    # Path.is_file() raises PermissionError or ENAMETOOLONG for some paths; the load goes on and reports its own error.
    from utils.models import model_config

    folder = _write(tmp_path / "run", CHAT_FILES)
    checked = []

    def _raises(model_name, *_args, **_kwargs):
        checked.append(model_name)
        raise PermissionError(13, "Permission denied", model_name)

    monkeypatch.setattr(model_config, "is_decision_model", _raises)
    started = _patch_load_past_the_checks(monkeypatch)

    with pytest.raises(_LoadStarted):
        asyncio.run(
            inference_route.load_model_gated(
                inference_route.LoadRequest(model_path = str(folder)), object(), "test"
            )
        )

    assert checked == [str(folder)]
    assert started == [str(folder)]


def test_a_fifo_named_config_json_in_the_cache_does_not_block_the_hidden_model_list(
    monkeypatch, tmp_path
):
    """read_text() on a FIFO with no writer blocks forever, and the chat picker waits on this list."""
    import os
    import threading

    from huggingface_hub import scan_cache_dir

    if not hasattr(os, "mkfifo"):
        pytest.skip("no mkfifo on this platform")
    _cache_repo(tmp_path, THIRD_PARTY_CLEF, CLEF_FILES)
    odd_repo = _cache_repo(tmp_path, "someone/odd-repo", {"model.safetensors": b"\x00" * 16})
    fifo = odd_repo / "config.json"
    os.mkfifo(fifo)
    scans = [scan_cache_dir(tmp_path)]
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: scans)
    result = {}
    worker = threading.Thread(
        target = lambda: result.update(ids = models_route.hidden_model_matchers()[1]), daemon = True
    )

    worker.start()
    worker.join(10)
    blocked = worker.is_alive()
    if blocked:
        try:  # a writer's EOF lets the blocked read return
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        worker.join(10)

    assert not blocked
    assert THIRD_PARTY_CLEF in result["ids"]
    assert "someone/odd-repo" not in result["ids"]


def test_a_managed_account_is_not_told_which_decision_repos_the_shared_cache_holds(
    monkeypatch, tmp_path
):
    from huggingface_hub import scan_cache_dir

    _cache_repo(tmp_path, THIRD_PARTY_CLEF, CLEF_FILES)
    scans = [scan_cache_dir(tmp_path)]
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: scans)

    owner_ids = models_route.hidden_model_matchers()[1]
    managed_ids = run_as(ALICE, lambda: models_route.hidden_model_matchers()[1])

    assert THIRD_PARTY_CLEF in owner_ids
    assert THIRD_PARTY_CLEF not in managed_ids
    # The built-in ids name no one's cache, so every account still gets them.
    assert "cloudflare/clef-flash" in managed_ids


def test_auto_switch_and_v1_models_do_not_serve_a_cached_clef(monkeypatch, tmp_path):
    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: True)

    def servable(repo_id, files):
        snapshot = _cache_repo(tmp_path, repo_id, files)
        info = SimpleNamespace(
            id = repo_id, model_id = repo_id, display_name = None, path = str(snapshot), partial = False
        )
        return resolver.local_servable_model(info)

    # Its backbone config says Qwen3ForCausalLM, which alone reads as a chat model.
    assert servable(THIRD_PARTY_CLEF, CLEF_FILES) is None
    assert servable("unsloth/Qwen3-0.6B", CHAT_FILES) is not None

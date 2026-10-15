# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import os
import sys
import threading
import types

import pytest
from datasets import Dataset
from fastapi import HTTPException

from core.training import token_preview, token_preview_worker
from models.training import TokenPreviewRequest


def _call_route(request: TokenPreviewRequest):
    from routes import training as training_routes
    return asyncio.run(
        training_routes.token_preview(request, hf_token = None, current_subject = "owner")
    )


@pytest.fixture
def captured(monkeypatch):
    from hub.services.datasets import formatting

    calls = []
    rows = Dataset.from_list([{"text": f"row {i}"} for i in range(100)])
    monkeypatch.setattr(formatting, "load_preview_slice", lambda *args, **kwargs: (rows, 64))
    monkeypatch.setattr(
        token_preview,
        "run_token_preview",
        lambda key, config, payload: calls.append((key, config, payload)) or {"samples": []},
    )
    return calls


def test_a_request_without_a_dataset_or_text_is_refused(captured):
    with pytest.raises(HTTPException) as excinfo:
        _call_route(TokenPreviewRequest(model_name = "unsloth/Qwen3-0.6B"))
    assert excinfo.value.status_code == 400
    assert captured == []


def test_the_pool_and_the_run_settings_reach_the_worker(captured):
    _call_route(
        TokenPreviewRequest(
            model_name = "unsloth/Qwen3-0.6B",
            dataset_name = "org/data",
            num_samples = 3,
            seed = 1,
            train_on_completions = True,
            max_seq_length = 512,
        )
    )
    payload = captured[-1][2]
    assert len(payload["rows"]) == 100
    assert (payload["num_samples"], payload["seed"]) == (3, 1)
    assert payload["train_on_completions"] is True
    assert payload["raw_text"] is False
    assert payload["max_seq_length"] == 512
    assert captured[-1][1]["model_name"] == "unsloth/Qwen3-0.6B"


def test_samples_are_drawn_by_seed():
    texts = [f"row {i}" for i in range(64)]
    first = token_preview_worker._sample(texts, 3, 1)
    assert len(first) == 3
    assert token_preview_worker._sample(texts, 3, 1) == first
    assert token_preview_worker._sample(texts, 3, 2) != first
    assert token_preview_worker._sample(texts[:2], 3, 1) == texts[:2]


def test_continued_pretraining_trains_every_token(captured):
    _call_route(
        TokenPreviewRequest(
            model_name = "unsloth/Qwen3-0.6B",
            dataset_name = "org/data",
            is_cpt = True,
            train_on_completions = True,
        )
    )
    payload = captured[-1][2]
    assert payload["raw_text"] is True
    assert payload["train_on_completions"] is False


def test_typed_text_reads_no_dataset(captured, monkeypatch):
    from hub.services.datasets import formatting

    monkeypatch.setattr(
        formatting,
        "load_preview_slice",
        lambda *args, **kwargs: pytest.fail("typed text must not read the dataset"),
    )
    _call_route(
        TokenPreviewRequest(model_name = "unsloth/Qwen3-0.6B", dataset_name = "org/data", text = "hi")
    )
    assert captured[-1][2]["rows"] is None
    assert captured[-1][2]["text"] == "hi"


class _ByteTokenizer:
    def decode(self, ids):
        return "�"


def test_a_partial_byte_level_piece_shows_its_bytes():
    # Llama 3 splits " ű" into "ĠÅ" (space + 0xC5) and "±" (0xB1).
    assert token_preview_worker._token_text(_ByteTokenizer(), 1, "ĠÅ") == " <0xC5>"
    assert token_preview_worker._token_text(_ByteTokenizer(), 2, "±") == "<0xB1>"
    # SentencePiece byte fallback already names the byte.
    assert token_preview_worker._token_text(_ByteTokenizer(), 3, "<0xC5>") == "<0xC5>"


def test_masking_goes_through_apply_completion_masking(monkeypatch):
    def fake_train_on_responses_only(
        trainer,
        instruction_part = None,
        response_part = None,
        tokenizer = None,
        return_function = False,
    ):
        assert trainer is None and return_function
        return lambda batch: {"labels": [[-100, -100] + ids[2:] for ids in batch["input_ids"]]}

    monkeypatch.setitem(
        sys.modules,
        "unsloth.chat_templates",
        types.SimpleNamespace(train_on_responses_only = fake_train_on_responses_only),
    )
    tokenizer = types.SimpleNamespace(
        _unsloth_input_part = "<user>", _unsloth_output_part = "<assistant>"
    )
    labels, masking = token_preview_worker._mask(
        tokenizer, "org/model", {"input_ids": [[1, 2, 3, 4]]}, "chatml_messages"
    )
    assert labels == [[-100, -100, 3, 4]]
    assert masking["applied"] is True
    assert masking["response_part"] == "<assistant>"


def test_stop_does_not_wait_behind_a_preview_in_flight():
    service = token_preview._TokenPreviewService()
    terminated = threading.Event()
    service._proc = types.SimpleNamespace(is_alive = lambda: True, terminate = terminated.set)
    with service._lock:
        service.stop()
    assert terminated.is_set()


_TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture
def real_worker():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("needs the Hub for the tiny tokenizer")
    config = {"model_name": _TINY, "tokenizer_source": _TINY, "hf_token": None}

    def run(**payload):
        return token_preview.run_token_preview(("owner", _TINY, _TINY, None), config, payload)

    yield run
    token_preview.stop_token_preview_worker()


def test_the_subprocess_tokenizes_and_masks_typed_text(real_worker):
    text = "Árvíztűrő ű ő"
    plain = real_worker(text = text, train_on_completions = False, max_seq_length = 128)
    sample = plain["samples"][0]
    assert sample["num_tokens"] == sample["num_trained"] > 0
    assert all(isinstance(token[0], int) for token in sample["tokens"])

    if importlib.util.find_spec("torch") is None:
        pytest.skip("completion masking needs torch, as it does for training")
    chat = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
    masked = real_worker(text = chat, train_on_completions = True, max_seq_length = 128)
    assert masked["masking"]["applied"] is True
    trained = "".join(
        piece for _, piece, is_trained in masked["samples"][0]["tokens"] if is_trained
    )
    assert trained.startswith("hello") and "user" not in trained


def test_typed_text_carries_the_dataset_format(captured):
    _call_route(
        TokenPreviewRequest(model_name = "unsloth/Qwen3-0.6B", text = "hi", final_format = "alpaca")
    )
    assert captured[-1][2]["final_format"] == "alpaca"


def test_a_tokenizer_load_failure_ends_the_worker(monkeypatch):
    service = token_preview._TokenPreviewService()
    stopped = []
    replies = [{"id": 1, "ok": False, "error": "offline", "fatal": True}]

    def fake_start(key, config):
        service._proc = types.SimpleNamespace(is_alive = lambda: True)
        service._cmd_queue = types.SimpleNamespace(put = lambda item: None)
        service._resp_queue = types.SimpleNamespace(get = lambda timeout: replies.pop(0))
        service._key, service._served = key, False

    monkeypatch.setattr(service, "_start_locked", fake_start)
    monkeypatch.setattr(service, "_stop_locked", lambda: stopped.append(True))
    with pytest.raises(token_preview.TokenPreviewError, match = "offline"):
        service.preview(("k",), {}, {"text": "hi"})
    assert stopped[-1] is True


def test_a_supplied_cache_folder_is_never_the_tokenizer_source(captured):
    _call_route(
        TokenPreviewRequest.model_validate(
            {
                "model_name": "unsloth/Qwen3-0.6B",
                "text": "hi",
                "model_known_cached": True,
                "model_local_path": "/somewhere/else",
            }
        )
    )
    assert "/somewhere" not in captured[-1][1]["tokenizer_source"]


class _LegacySentencePiece:
    def decode(self, ids):
        return "quick"


def test_a_sentencepiece_word_keeps_its_leading_space():
    assert token_preview_worker._token_text(_LegacySentencePiece(), 1, "▁quick") == " quick"


def test_audio_models_train_every_token_as_training_does(captured):
    _call_route(
        TokenPreviewRequest(
            model_name = "unsloth/orpheus-3b-0.1-ft",
            model_type = "audio",
            text = "hi",
            train_on_completions = True,
        )
    )
    assert captured[-1][2]["train_on_completions"] is False


def test_retiring_an_account_stops_its_preview_worker():
    from core.training import account_jobs
    from utils.account_context import AccountContext

    service = token_preview._TokenPreviewService()
    stopped = threading.Event()
    service._proc = types.SimpleNamespace(is_alive = lambda: True, terminate = stopped.set)
    account = AccountContext(account_id = "alice", username = "alice")
    service._result_account = account
    service.stop = lambda: stopped.set()
    service._account_cancel = service.stop
    assert service in set(account_jobs._services)
    try:
        account_jobs.retire_account_jobs(account)
    except RuntimeError:
        pass  # the stand-in process never reports itself stopped
    finally:
        account_jobs._retired.discard("alice")
    assert stopped.is_set()


def test_an_adapter_base_is_never_fetched(monkeypatch):
    import transformers
    from utils.models import model_config

    tried = []

    def from_pretrained(source, **kwargs):
        tried.append(source)
        raise OSError("no tokenizer files")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        model_config, "get_base_model_from_lora_identifier", lambda *args: "org/private-base"
    )
    with pytest.raises(OSError):
        token_preview_worker._load_tokenizer({"model_name": "org/adapter", "hf_token": None})
    assert set(tried) == {"org/adapter"}


def test_inventory_handles_are_resolved(monkeypatch):
    import models.inference as inference

    monkeypatch.setattr(
        inference, "resolve_inventory_handle", lambda v: v.replace("ref:", "/real/")
    )
    request = TokenPreviewRequest(model_name = "ref:abc", dataset_name = "ref:def", text = "hi")
    assert (request.model_name, request.dataset_name) == ("/real/abc", "/real/def")


def test_a_stop_during_a_preview_ends_the_worker_it_started(monkeypatch):
    service = token_preview._TokenPreviewService()
    stopped = []

    def fake_start(key, config):
        service._proc = types.SimpleNamespace(
            is_alive = lambda: False,
            terminate = lambda: None,
            join = lambda timeout: None,
            exitcode = 0,
            pid = None,
        )
        service._cmd_queue = types.SimpleNamespace(put = lambda item: service.stop())
        service._resp_queue = types.SimpleNamespace(
            get = lambda timeout: {"id": 1, "ok": True, "result": {}}
        )
        service._key, service._served = key, False

    real_stop_locked = service._stop_locked
    monkeypatch.setattr(service, "_start_locked", fake_start)
    monkeypatch.setattr(service, "_stop_locked", lambda: (stopped.append(True), real_stop_locked()))
    service.preview(("k",), {}, {"text": "hi"})
    assert len(stopped) == 2 and service._proc is None


def _viewer(
    monkeypatch,
    body,
    status = 200,
):
    import httpx

    from hub.services.datasets import formatting
    from hub.schemas.datasets import CheckFormatRequest

    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return types.SimpleNamespace(status_code = status, json = lambda: body)

    monkeypatch.setattr(httpx, "get", fake_get)
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_SERVER", "HF_ENDPOINT"):
        monkeypatch.delenv(var, raising = False)
    request = CheckFormatRequest(dataset_name = "org/data", train_split = "train")
    return lambda: formatting._viewer_rows(request, "hf_x", 100), calls


def _rows(n, **extra):
    return [
        {"row_idx": i, "row": {"text": f"row {i}"}, "truncated_cells": [], **extra}
        for i in range(n)
    ]


def test_viewer_rows_stand_in_only_when_complete(monkeypatch):
    features = [{"name": "text", "type": {"dtype": "string", "_type": "Value"}}]
    rows, calls = _viewer(
        monkeypatch, {"rows": _rows(100), "features": features, "num_rows_total": 5000}
    )
    assert rows() == [{"text": f"row {i}"} for i in range(100)]
    assert calls[0][0].endswith("/rows") and calls[0][1]["params"]["length"] == 100
    assert calls[0][1]["headers"] == {"Authorization": "Bearer hf_x"}
    # A dataset smaller than the pool is complete at its own size.
    rows, _ = _viewer(monkeypatch, {"rows": _rows(7), "features": features, "num_rows_total": 7})
    assert len(rows()) == 7
    # Short, cut or media answers fall back to streaming.
    for body in (
        {"rows": _rows(28), "features": features, "num_rows_total": 5000},
        {"rows": _rows(100, truncated_cells = ["text"]), "features": features},
        {"rows": _rows(100), "features": [{"name": "image", "type": {"_type": "Image"}}]},
        {"error": "not ready"},
    ):
        rows, _ = _viewer(monkeypatch, body)
        assert rows() is None
    rows, _ = _viewer(monkeypatch, {"rows": _rows(100), "features": features}, status = 404)
    assert rows() is None


def test_viewer_is_not_asked_offline_or_behind_a_mirror(monkeypatch):
    from utils import hf_endpoint

    rows, calls = _viewer(monkeypatch, {"rows": _rows(100)})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert rows() is None
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setattr(hf_endpoint, "browser_hf_endpoint", lambda: "https://hf-mirror.com")
    assert rows() is None
    assert calls == []


def test_only_the_token_preview_asks_the_viewer(monkeypatch):
    import inspect

    from hub.services.datasets import formatting
    from routes import training as training_routes

    assert "viewer_rows = True" in inspect.getsource(training_routes._token_preview_rows)
    assert "viewer_rows" not in inspect.getsource(formatting.check_format_response)

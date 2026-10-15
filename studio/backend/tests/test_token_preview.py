# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
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
    rows = Dataset.from_list([{"text": f"row {i}"} for i in range(64)])
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


def test_rows_are_sampled_by_seed_and_carry_the_run_settings(captured):
    def rows_for(seed):
        _call_route(
            TokenPreviewRequest(
                model_name = "unsloth/Qwen3-0.6B",
                dataset_name = "org/data",
                num_samples = 3,
                seed = seed,
                train_on_completions = True,
                max_seq_length = 512,
            )
        )
        return captured[-1][2]

    first = rows_for(1)
    assert len(first["rows"]) == 3
    assert rows_for(1)["rows"] == first["rows"]
    assert rows_for(2)["rows"] != first["rows"]
    assert first["train_on_completions"] is True
    assert first["raw_text"] is False
    assert first["max_seq_length"] == 512
    assert captured[-1][1]["model_name"] == "unsloth/Qwen3-0.6B"


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

    chat = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
    masked = real_worker(text = chat, train_on_completions = True, max_seq_length = 128)
    assert masked["masking"]["applied"] is True
    trained = "".join(
        piece for _, piece, is_trained in masked["samples"][0]["tokens"] if is_trained
    )
    assert trained.startswith("hello") and "user" not in trained

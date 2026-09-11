# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

try:
    import core.inference.inference  # noqa: E402,F401
except ImportError:
    pass

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _Batch(dict):
    def to(self, *_args, **_kwargs):
        return self


class _Tokenizer:
    all_special_tokens: list = []
    eos_token_id = 1
    pad_token_id = None
    chat_template = "{{ messages }}"

    def __call__(self, *_args, **_kwargs):
        torch = pytest.importorskip("torch")
        return _Batch({"input_ids": torch.zeros((1, 1), dtype = torch.long)})

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(int(i), "") for i in ids)


class _Processor:
    chat_template = None
    tokenizer = _Tokenizer()


class _Model:
    device = "cpu"
    generation_config = type("Cfg", (), {"eos_token_id": 1})()
    config = generation_config

    def __init__(self, pieces):
        self.pieces = pieces
        self.sent = []

    def generate(self, streamer, stopping_criteria, max_new_tokens, **_kwargs):
        torch = pytest.importorskip("torch")
        ids = torch.zeros((1, 1), dtype = torch.long)
        streamer.put(ids)
        for index, piece in enumerate(self.pieces[:max_new_tokens], 2):
            ids = torch.cat([ids, torch.tensor([[index]])], dim = 1)
            streamer.put(torch.tensor([index]))
            self.sent.append(piece)
            if stopping_criteria is not None and bool(stopping_criteria(ids, None).all()):
                break
        streamer.end()
        return ids


def _backend(pieces, tokenizer):
    inf = pytest.importorskip("core.inference.inference")
    model = _Model(pieces)
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    raw_tokenizer.pieces = dict(enumerate(pieces, 2))
    streamer = inf.TextIteratorStreamer(raw_tokenizer, skip_prompt = True, timeout = 0.2)
    backend = inf.InferenceBackend.__new__(inf.InferenceBackend)
    backend.active_model_name = "stop-test"
    backend._generation_lock = threading.Lock()
    backend.models = {"stop-test": {"model": model, "tokenizer": tokenizer, "processor": tokenizer}}
    backend._make_text_streamer = lambda *_args, **_kwargs: streamer
    return backend, model


class _Responses(list):
    def put(self, response):
        self.append(response)


def test_worker_forwards_stop_and_the_reply_ends_before_it(monkeypatch):
    from core.inference import worker

    backend, model = _backend(["Hello ", "ST", "OP", " world"], _Tokenizer())
    monkeypatch.setattr(
        backend, "_apply_chat_template_for_generation", lambda *a, **k: "PROMPT", raising = False
    )
    responses = _Responses()
    cmd = {
        "request_id": "r",
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": 3,
        "stop": ["STOP"],
    }

    worker._handle_generate(backend, cmd, responses, threading.Event())

    assert [r["text"] for r in responses if r["type"] == "token"] == ["Hello"]
    assert model.sent == ["Hello ", "ST", "OP"]
    done = next(r for r in responses if r["type"] == "gen_done")
    assert done["stats"]["truncated"] is False


def test_vision_reply_ends_before_a_stop_sequence():
    backend, model = _backend(["Hello ", "ST", "OP", " world"], _Processor())
    backend.format_chat_prompt = lambda *_args, **_kwargs: "PROMPT"

    snapshots = list(
        backend._generate_vision_response(
            messages = [{"role": "user", "content": "hi"}],
            system_prompt = "",
            image = None,
            temperature = 0.0,
            top_p = 1.0,
            top_k = 0,
            min_p = 0.0,
            max_new_tokens = 3,
            repetition_penalty = 1.0,
            stop = ["STOP"],
        )
    )

    assert snapshots == ["Hello"]
    assert model.sent == ["Hello ", "ST", "OP"]
    assert backend.last_generation_stats["truncated"] is False


def test_a_partial_stop_sequence_is_released_when_the_reply_ends():
    backend, _ = _backend(["Hello ", "ST"], _Tokenizer())

    snapshots = list(backend.generate_stream("PROMPT", max_new_tokens = 8, stop = ["STOP"]))

    assert snapshots == ["Hello", "Hello ST"]


@pytest.mark.parametrize("vision", [False, True])
def test_compact_json_stops_in_producer_before_word_flush(vision):
    pieces = ['{"id":1}', ',{"id":2}'] * 100
    backend, model = _backend(pieces, _Processor() if vision else _Tokenizer())
    if vision:
        backend.format_chat_prompt = lambda *a, **k: "PROMPT"
        output = list(
            backend._generate_vision_response(
                messages = [{"role": "user", "content": "hi"}],
                system_prompt = "",
                image = None,
                temperature = 0.0,
                top_p = 1.0,
                top_k = 0,
                min_p = 0.0,
                max_new_tokens = 200,
                repetition_penalty = 1.0,
                stop = ["}"],
            )
        )
    else:
        output = list(backend.generate_stream("PROMPT", max_new_tokens = 200, stop = ["}"]))
    assert model.sent == [pieces[0]]
    assert output == ['{"id":1']
    assert backend.last_generation_stats["truncated"] is False


@pytest.mark.parametrize(
    "kind,raw,stop",
    [
        (
            "harmony",
            "<|channel|>analysis<|message|>Reasoning<|channel|>final<|message|>Answer",
            "<think>",
        ),
        ("reasoning", "<|channel>thoughtReasoning <channel|>Answer ", "</think>"),
    ],
)
def test_synthetic_reasoning_tags_do_not_match_stops(kind, raw, stop):
    inf = pytest.importorskip("core.inference.inference")
    torch = pytest.importorskip("torch")
    tokenizer = _Tokenizer()
    tokenizer.pieces = {2: raw}
    if kind == "harmony":
        streamer = inf.HarmonyTextStreamer(tokenizer, skip_prompt = False)
    else:
        streamer = inf.ReasoningTextIteratorStreamer(
            tokenizer,
            markers = ("<|channel>thought", "<channel|>"),
            skip_prompt = False,
        )
    wrapped = inf._StopSequenceStreamer(streamer, [stop])
    # Use the production model.generate interface on both sides of the fix.
    producer = wrapped if hasattr(wrapped, "put") else streamer
    producer.put(torch.tensor([2]))
    producer.end()
    output = []
    while True:
        try:
            output.append(next(wrapped))
        except StopIteration:
            break
    assert not wrapped.matched.is_set()
    assert "Answer" in "".join(output)


@pytest.mark.parametrize("stop", [None, [], ["missing"]])
def test_requests_without_a_matching_stop_keep_all_text(stop):
    backend, model = _backend(["Hello ", "world"], _Tokenizer())
    output = list(backend.generate_stream("PROMPT", max_new_tokens = 8, stop = stop))
    assert output[-1] == "Hello world"
    assert model.sent == ["Hello ", "world"]


@pytest.mark.parametrize("abort", [False, True])
def test_stop_wrapper_finishes_reasoning_once(abort):
    inf = pytest.importorskip("core.inference.inference")
    torch = pytest.importorskip("torch")
    tokenizer = _Tokenizer()
    tokenizer.pieces = {2: "<|channel>thoughtReasoning ST"}
    streamer = inf.ReasoningTextIteratorStreamer(
        tokenizer,
        markers = ("<|channel>thought", "<channel|>"),
        skip_prompt = False,
    )
    wrapped = inf._StopSequenceStreamer(streamer, ["STOP"])
    wrapped.put(torch.tensor([2]))
    if abort:
        wrapped.abort()
    wrapped.end()
    wrapped.end()
    output = "".join(wrapped)
    assert "Reasoning ST" in output
    assert output.count("</think>") == (0 if abort else 1)


def test_stop_matching_decodes_split_unicode_without_leaking_partial_bytes():
    inf = pytest.importorskip("core.inference.inference")
    torch = pytest.importorskip("torch")

    class Tokenizer(_Tokenizer):
        def decode(self, ids, **kwargs):
            return "Hello �" if len(ids) == 1 else "Hello éEND"

    streamer = inf.TextIteratorStreamer(Tokenizer(), skip_prompt = False)
    wrapped = inf._StopSequenceStreamer(streamer, ["éEND"])
    wrapped.put(torch.tensor([2]))
    assert not wrapped.matched.is_set()
    wrapped.put(torch.tensor([3]))
    assert wrapped.matched.is_set()
    wrapped.end()
    assert "".join(wrapped) == "Hello "

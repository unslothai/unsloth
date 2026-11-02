# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import inspect
import types
from pathlib import Path

import pytest
import torch

# Loaded from source: importing core.inference.inference needs unsloth, which the backend matrix lacks.
_SRC = Path(__file__).resolve().parents[1] / "core" / "inference" / "inference.py"
_WANTED = {
    "_CUSTOM_GENERATE_REJECTED_KWARGS",
    "_supported_generate_kwargs",
    "_processor_takes_text_first",
    "_GenerationHalted",
    "_HaltingStreamer",
    "_generate_supported",
    "_refuse_without_generate",
    "_think_tag_ids",
}
_NS = {"inspect": inspect, "torch": torch}
exec(
    compile(
        ast.Module(
            [
                n
                for n in ast.parse(_SRC.read_text(encoding = "utf-8")).body
                if getattr(n, "name", None) in _WANTED
                or any(getattr(t, "id", None) in _WANTED for t in getattr(n, "targets", ()))
            ],
            [],
        ),
        str(_SRC),
        "exec",
    ),
    _NS,
)
_supported_generate_kwargs = _NS["_supported_generate_kwargs"]
_processor_takes_text_first = _NS["_processor_takes_text_first"]
_generate_supported = _NS.get("_generate_supported")
_refuse_without_generate = _NS.get("_refuse_without_generate")

_KWARGS = {"input_ids": 1, "logits_processor": [], "stopping_criteria": [], "streamer": object()}


def test_custom_generate_drops_rejected_kwargs():
    model = types.SimpleNamespace(_generation_fields_unsupported = {"cache_implementation": None})
    out = _supported_generate_kwargs(model, _KWARGS)
    assert "logits_processor" not in out and "stopping_criteria" not in out
    assert out["input_ids"] == 1 and "streamer" in out


def test_regular_model_keeps_all_kwargs():
    assert _supported_generate_kwargs(types.SimpleNamespace(), _KWARGS) is _KWARGS


@pytest.mark.parametrize("model_type", ["lfm3_omni", "lfm3_audio"])
def test_audio_chat_requests_text_instead_of_unstreamed_speech(model_type):
    model = types.SimpleNamespace(
        config = types.SimpleNamespace(model_type = model_type),
        _generation_fields_unsupported = {},
    )
    assert _supported_generate_kwargs(model, _KWARGS)["generation_mode"] == "text"
    model.config.model_type = "other_custom_model"
    assert "generation_mode" not in _supported_generate_kwargs(model, _KWARGS)


class _TextFirst:
    def __call__(
        self,
        text = None,
        audio = None,
        images = None,
        **kwargs,
    ):
        return text, images


class _ImagesFirst:
    def __call__(
        self,
        images = None,
        text = None,
        **kwargs,
    ):
        return images, text


def test_text_first_processor_detected():
    assert _processor_takes_text_first(_TextFirst()) is True
    assert _processor_takes_text_first(_ImagesFirst()) is False


def test_zoo_wrapped_call_still_reads_original_signature():
    class _Patched(_TextFirst):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

    assert _processor_takes_text_first(_Patched()) is True


class _Recorder:
    def __init__(self):
        self.tokens, self.ended = [], False

    def put(self, value):
        self.tokens.append(value)

    def end(self):
        self.ended = True


class _CustomGenerate:
    _generation_fields_unsupported = {"cache_implementation": None}

    def __init__(self, on_step = None):
        self.on_step = on_step
        self.steps = 0

    def generate(self, *, streamer, max_new_tokens, **kwargs):
        assert "stopping_criteria" not in kwargs
        try:
            for step in range(max_new_tokens):
                self.steps += 1
                if self.on_step:
                    self.on_step(step)
                streamer.put(step)
        finally:
            streamer.end()
        return "sequences"


def _criteria(*events):
    return [types.SimpleNamespace(ev = event) for event in events]


def test_cancel_ends_a_custom_generate_without_stopping_criteria():
    import threading

    cancel = threading.Event()
    model = _CustomGenerate(on_step = lambda step: step == 3 and cancel.set())
    streamer = _Recorder()
    out = _generate_supported(
        model, {"streamer": streamer, "max_new_tokens": 100, "stopping_criteria": _criteria(cancel)}
    )
    assert out is None
    assert model.steps == 4 and streamer.tokens == [0, 1, 2] and streamer.ended


def test_a_matched_stop_sequence_ends_it_too():
    import threading

    matched = threading.Event()
    model = _CustomGenerate(on_step = lambda step: step == 5 and matched.set())
    out = _generate_supported(
        model,
        {
            "streamer": _Recorder(),
            "max_new_tokens": 100,
            "stopping_criteria": _criteria(threading.Event(), matched),
        },
    )
    assert out is None and model.steps == 6


def test_an_uncancelled_custom_generate_returns_its_sequences():
    import threading

    model = _CustomGenerate()
    streamer = _Recorder()
    out = _generate_supported(
        model,
        {
            "streamer": streamer,
            "max_new_tokens": 4,
            "stopping_criteria": _criteria(threading.Event()),
        },
    )
    assert out == "sequences" and streamer.tokens == [0, 1, 2, 3] and streamer.ended


def test_a_text_load_without_generate_is_refused_by_name():
    class Lfm3AudioModel:
        pass

    with pytest.raises(ValueError, match = "Lfm3AudioModel, which cannot generate text"):
        _refuse_without_generate(Lfm3AudioModel(), "LiquidAI/LFM3-Audio-2B")
    _refuse_without_generate(types.SimpleNamespace(generate = lambda **kw: None), "any/causal-lm")


_OPEN, _CLOSE, _EOS = 901, 902, 900
_THINK_TOKENIZER = types.SimpleNamespace(
    unk_token_id = 0,
    convert_tokens_to_ids = lambda t: {"<think>": _OPEN, "</think>": _CLOSE}.get(t, 0),
)


class _ScriptedGenerate:
    _generation_fields_unsupported = {"cache_implementation": None}

    def __init__(self, reply):
        self.reply, self.steps = reply, 0

    def generate(self, *, input_ids, streamer, **kwargs):
        try:
            streamer.put(input_ids)
            for token in self.reply:
                self.steps += 1
                streamer.put(torch.tensor([[token]]))
        finally:
            streamer.end()
        return torch.cat((input_ids, torch.tensor([self.reply])), 1)


def _scripted(reply, **extra):
    model, streamer = _ScriptedGenerate(reply), _Recorder()
    kwargs = {
        "input_ids": torch.tensor([[1, 2]]),
        "streamer": streamer,
        "eos_token_id": [_EOS],
        **extra,
    }
    out = _generate_supported(model, kwargs, _THINK_TOKENIZER)
    return model, [int(t) for t in streamer.tokens[1:]], out


def test_a_second_think_close_ends_the_reply_before_its_restatement():
    model, streamed, out = _scripted([_OPEN, 10, _CLOSE, 11, _CLOSE, 11, _EOS])
    assert streamed == [_OPEN, 10, _CLOSE, 11]
    assert model.steps == 5
    assert out.tolist() == [[1, 2, _OPEN, 10, _CLOSE, 11, _EOS]]


def test_one_think_block_and_a_prefilled_close_stream_untouched():
    for reply in (
        [_OPEN, 10, _CLOSE, 11, _EOS],
        [10, _CLOSE, 11, _EOS],
        [_OPEN, 10, _CLOSE, _OPEN, 12, _CLOSE, 11, _EOS],
    ):
        model, streamed, out = _scripted(reply)
        assert streamed == reply and out.tolist() == [[1, 2, *reply]]


def test_cancel_still_returns_no_sequences_beside_the_think_guard():
    import threading

    cancel = threading.Event()
    cancel.set()
    _, streamed, out = _scripted([_OPEN, 10, _CLOSE, 11, _EOS], stopping_criteria = _criteria(cancel))
    assert out is None and streamed == []

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3-Omni's custom generate raises on logits_processor and stopping_criteria (#16054)."""

import ast
import inspect
import types
from pathlib import Path

# Loaded from source: importing core.inference.inference needs unsloth, which the backend matrix lacks.
_SRC = Path(__file__).resolve().parents[1] / "core" / "inference" / "inference.py"
_WANTED = {
    "_CUSTOM_GENERATE_REJECTED_KWARGS",
    "_supported_generate_kwargs",
    "_processor_takes_text_first",
}
_NS = {"inspect": inspect}
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

_KWARGS = {"input_ids": 1, "logits_processor": [], "stopping_criteria": [], "streamer": object()}


def test_custom_generate_drops_rejected_kwargs():
    model = types.SimpleNamespace(_generation_fields_unsupported = {"cache_implementation": None})
    out = _supported_generate_kwargs(model, _KWARGS)
    assert "logits_processor" not in out and "stopping_criteria" not in out
    assert out["input_ids"] == 1 and "streamer" in out


def test_regular_model_keeps_all_kwargs():
    assert _supported_generate_kwargs(types.SimpleNamespace(), _KWARGS) is _KWARGS


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
    """LFM3-style: rejects stopping_criteria, streams each token, ends its streamers on any exit."""

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
    from core.inference.inference import InferenceBackend
    return InferenceBackend._cancel_stopping_criteria(None, *events)


def test_cancel_ends_a_custom_generate_without_stopping_criteria():
    import threading
    from core.inference.inference import _generate_supported

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
    from core.inference.inference import _generate_supported

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
    from core.inference.inference import _generate_supported

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
    import pytest
    from core.inference.inference import _refuse_without_generate

    class Lfm3AudioModel:
        pass

    with pytest.raises(ValueError, match = "Lfm3AudioModel, which cannot generate text"):
        _refuse_without_generate(Lfm3AudioModel(), "LiquidAI/LFM3-Audio-2B")
    _refuse_without_generate(types.SimpleNamespace(generate = lambda **kw: None), "any/causal-lm")

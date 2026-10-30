# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3-Omni's custom generate raises on logits_processor and stopping_criteria (#16054)."""

import types

from core.inference.inference import _supported_generate_kwargs

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
    from core.inference.inference import _processor_takes_text_first
    assert _processor_takes_text_first(_TextFirst()) is True
    assert _processor_takes_text_first(_ImagesFirst()) is False


def test_zoo_wrapped_call_still_reads_original_signature():
    from core.inference.inference import _processor_takes_text_first
    class _Patched(_TextFirst):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

    assert _processor_takes_text_first(_Patched()) is True


def test_a_text_load_without_generate_is_refused_by_name():
    import pytest
    from core.inference.inference import _refuse_without_generate

    class Lfm3AudioModel:
        pass

    with pytest.raises(ValueError, match = "Lfm3AudioModel, which cannot generate text"):
        _refuse_without_generate(Lfm3AudioModel(), "LiquidAI/LFM3-Audio-2B")
    _refuse_without_generate(types.SimpleNamespace(generate = lambda **kw: None), "any/causal-lm")

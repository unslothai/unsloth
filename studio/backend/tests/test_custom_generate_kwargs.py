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

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

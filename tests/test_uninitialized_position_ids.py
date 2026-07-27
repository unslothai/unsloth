"""RaiseUninitialized must ignore a checkpoint that only re-initializes deterministic
position_ids buffers, but still raise when a real weight is missing -- even if the same
HF record also lists a benign position_ids buffer.
"""

from __future__ import annotations

import logging

import pytest

from unsloth.models._utils import (
    _all_missing_keys_are_position_ids,
    _RaiseUninitialized,
)

_TEMPLATE = (
    "Some weights of DeepseekOCRForCausalLM were not initialized from the model "
    "checkpoint at unsloth/DeepSeek-OCR and are newly initialized: {keys}\n"
    "You should probably TRAIN this model on a down-stream task."
)


def _record(keys_repr: str) -> logging.LogRecord:
    return logging.LogRecord(
        name = "transformers.modeling_utils",
        level = logging.WARNING,
        pathname = "modeling_utils.py",
        lineno = 1,
        msg = _TEMPLATE.format(keys = keys_repr),
        args = None,
        exc_info = None,
    )


@pytest.mark.parametrize(
    "keys_repr, expected",
    [
        ("['model.vision_model.embeddings.position_ids']", True),
        (
            "['model.vision_model.embeddings.position_ids', "
            "'vision_model.encoder.layers.0.position_ids']",
            True,
        ),
        # A real missing weight alongside position_ids must NOT be suppressed.
        (
            "['model.vision_model.embeddings.position_ids', 'model.layers.5.mlp.weight']",
            False,
        ),
        ("['model.layers.5.mlp.weight']", False),
        ("[]", False),
    ],
)
def test_all_missing_keys_are_position_ids(keys_repr, expected):
    assert _all_missing_keys_are_position_ids(_TEMPLATE.format(keys = keys_repr)) is expected


def test_emit_suppresses_position_ids_only_record():
    # A record listing only position_ids buffers loads cleanly (no raise).
    handler = _RaiseUninitialized()
    handler.emit(_record("['model.vision_model.embeddings.position_ids']"))


def test_emit_raises_when_real_weight_missing_alongside_position_ids():
    # The core fix: one benign position_ids key must not mask a real missing weight.
    handler = _RaiseUninitialized()
    with pytest.raises(Exception, match = "some weights are not initialized"):
        handler.emit(
            _record("['model.vision_model.embeddings.position_ids', 'model.layers.5.mlp.weight']")
        )


def test_emit_raises_on_real_missing_weight():
    handler = _RaiseUninitialized()
    with pytest.raises(Exception, match = "some weights are not initialized"):
        handler.emit(_record("['model.layers.5.mlp.weight']"))

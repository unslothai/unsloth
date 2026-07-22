# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pin TrainingStartRequest hyperparameter caps at the at-cap / over-cap boundary."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from models.training import (
    _MAX_BATCH_SIZE,
    _MAX_LORA_ALPHA,
    _MAX_LORA_R,
    _MAX_SEQ_LENGTH,
    _MAX_VISION_IMAGE_SIZE,
    _MIN_VISION_IMAGE_SIZE,
)


def _check_field(field_name: str, value):
    """Run the field validator without building a full TrainingStartRequest."""
    from models.training import TrainingStartRequest

    schema_field = TrainingStartRequest.model_fields[field_name]
    return TrainingStartRequest.__pydantic_validator__.validate_assignment(
        TrainingStartRequest.model_construct(),
        field_name,
        value,
    )


class TestSeqLengthCap:
    def test_at_cap_accepts(self):
        _check_field("max_seq_length", _MAX_SEQ_LENGTH)
        assert _MAX_SEQ_LENGTH == 2_000_000

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError) as exc:
            _check_field("max_seq_length", _MAX_SEQ_LENGTH + 1)
        assert "max_seq_length" in str(exc.value)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("max_seq_length", 0)


class TestBatchSizeCap:
    def test_at_cap_accepts(self):
        _check_field("batch_size", _MAX_BATCH_SIZE)
        assert _MAX_BATCH_SIZE == 4096

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("batch_size", _MAX_BATCH_SIZE + 1)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("batch_size", 0)


class TestVisionImageSizeCap:
    def test_none_accepts_model_default(self):
        _check_field("vision_image_size", None)

    @pytest.mark.parametrize(
        "value",
        [_MIN_VISION_IMAGE_SIZE, 640, 1000, _MAX_VISION_IMAGE_SIZE],
    )
    def test_in_range_accepts(self, value):
        _check_field("vision_image_size", value)
        assert _MIN_VISION_IMAGE_SIZE == 256
        assert _MAX_VISION_IMAGE_SIZE == 2048

    @pytest.mark.parametrize(
        "value",
        [_MIN_VISION_IMAGE_SIZE - 1, _MAX_VISION_IMAGE_SIZE + 1, 640.5, True],
    )
    def test_invalid_rejects(self, value):
        with pytest.raises(ValidationError):
            _check_field("vision_image_size", value)

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_error_says_integer_not_range(self, value):
        # Regression guard: bools say "integer or null", not "in [256, 2048]".
        with pytest.raises(ValidationError) as exc:
            _check_field("vision_image_size", value)
        assert "integer or null" in str(exc.value)

    @pytest.mark.parametrize("value", ["++512", "--256", "+-+512", "+", "-"])
    def test_multi_sign_string_says_integer_not_raw(self, value):
        # Regression guard: multi-sign strings say "integer or null", not int()'s raw message.
        with pytest.raises(ValidationError) as exc:
            _check_field("vision_image_size", value)
        assert "integer or null" in str(exc.value)
        assert "invalid literal" not in str(exc.value)

    @pytest.mark.parametrize("value", ["５１２", "٥١٢", "१०२४"])
    def test_unicode_digit_string_rejected(self, value):
        # Reject non-ASCII (full-width/Arabic-Indic/Devanagari) digits.
        with pytest.raises(ValidationError) as exc:
            _check_field("vision_image_size", value)
        assert "integer or null" in str(exc.value)


class TestLoraRCap:
    def test_at_cap_accepts(self):
        _check_field("lora_r", _MAX_LORA_R)
        assert _MAX_LORA_R == 16_384

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_r", _MAX_LORA_R + 1)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_r", 0)


class TestLoraAlphaCap:
    def test_at_cap_accepts(self):
        _check_field("lora_alpha", _MAX_LORA_ALPHA)
        assert _MAX_LORA_ALPHA == 32_768

    def test_over_cap_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_alpha", _MAX_LORA_ALPHA + 1)

    def test_below_min_rejects(self):
        with pytest.raises(ValidationError):
            _check_field("lora_alpha", 0)

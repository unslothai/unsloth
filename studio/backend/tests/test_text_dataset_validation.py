# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from utils.datasets.text_validation import validate_non_empty_text_field


class StreamingRows:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class SizedIterableRows(StreamingRows):
    def __len__(self):
        return len(self._rows)


def test_validate_non_empty_text_field_accepts_valid_text_dataset():
    dataset = [{"text": "hello"}, {"text": "world"}]

    validate_non_empty_text_field(dataset, split_name = "train")


def test_validate_non_empty_text_field_rejects_empty_text_with_row_context():
    dataset = [{"text": "hello"}, {"text": "   "}]

    with pytest.raises(ValueError) as exc:
        validate_non_empty_text_field(dataset, split_name = "train")

    message = str(exc.value)
    assert "Dataset validation failed" in message
    assert "train row 1 has an empty `text` field" in message
    assert "Remove blank rows" in message


def test_validate_non_empty_text_field_rejects_missing_text_field():
    dataset = [{"prompt": "hello"}]

    with pytest.raises(ValueError) as exc:
        validate_non_empty_text_field(dataset, split_name = "eval")

    assert "eval row 0 is missing the `text` field" in str(exc.value)


def test_validate_non_empty_text_field_rejects_streaming_empty_first_row():
    dataset = StreamingRows([{"text": ""}, {"text": "ok"}])

    with pytest.raises(ValueError) as exc:
        validate_non_empty_text_field(dataset, split_name = "train")

    assert "train row 0 has an empty `text` field" in str(exc.value)


def test_validate_non_empty_text_field_scans_sized_iterables_without_indexing():
    dataset = SizedIterableRows([{"text": "ok"}, {"text": ""}])

    with pytest.raises(ValueError) as exc:
        validate_non_empty_text_field(dataset, split_name = "train")

    assert "train row 1 has an empty `text` field" in str(exc.value)

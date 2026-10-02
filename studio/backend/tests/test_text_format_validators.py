# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.data_recipe.text_format_validators import (
    JSON_VALIDATION_FN_MARKER,
    MARKDOWN_VALIDATION_FN_MARKER,
    _validate_json_text,
    _validate_markdown_text,
    split_text_format_local_callable_validators,
)


def test_validate_json_text_accepts_object_payload():
    result = _validate_json_text('{"answer": "ok"}')
    assert result["is_valid"] is True


def test_validate_json_text_rejects_invalid_payload():
    result = _validate_json_text("{not json")
    assert result["is_valid"] is False
    assert result["error_message"]


def test_validate_markdown_text_rejects_unclosed_fence():
    result = _validate_markdown_text("```python\nprint('hi')")
    assert result["is_valid"] is False
    assert "fence" in result["error_message"].lower()


def test_validate_markdown_text_accepts_simple_markdown():
    result = _validate_markdown_text("# Title\n\nSome **bold** text.")
    assert result["is_valid"] is True


def test_split_text_format_local_callable_validators_extracts_json_and_markdown_specs():
    recipe = {
        "columns": [
            {
                "column_type": "validation",
                "name": "json_check",
                "target_columns": ["payload"],
                "validator_type": "local_callable",
                "validator_params": {
                    "validation_function": JSON_VALIDATION_FN_MARKER,
                },
            },
            {
                "column_type": "validation",
                "name": "md_check",
                "target_columns": ["body"],
                "validator_type": "local_callable",
                "validator_params": {
                    "validation_function": MARKDOWN_VALIDATION_FN_MARKER,
                },
            },
            {"column_type": "llm-text", "name": "payload"},
        ],
    }

    sanitized, specs = split_text_format_local_callable_validators(recipe)

    assert len(sanitized["columns"]) == 1
    assert [spec.format_kind for spec in specs] == ["json", "markdown"]


def test_validate_markdown_text_accepts_unmatched_brackets_and_parens():
    for text in ("1) first\n2) second", "Status (draft", "Smile :)", "see arr[0 here"):
        assert _validate_markdown_text(text)["is_valid"] is True, text


def test_validate_json_text_rejects_nan_and_accepts_native_values():
    assert _validate_json_text("NaN")["is_valid"] is False
    assert _validate_json_text({"a": True})["is_valid"] is True


def test_validate_markdown_text_follows_commonmark_fences():
    for text in (
        "````md\nuse ```python\n````",
        "Wrap code in ``` fences.",
        "```markdown\n# T\n```python\nx = 1\n```",
    ):
        assert _validate_markdown_text(text)["is_valid"] is True, text
    for text in ("```python\nprint(1)", "~~~\nx", "````\nx\n```"):
        assert _validate_markdown_text(text)["is_valid"] is False, text


def test_validate_json_text_accepts_native_scalars():
    for value in (True, False, 0, 2.5):
        assert _validate_json_text(value)["is_valid"] is True, value
    assert _validate_json_text(float("nan"))["is_valid"] is False


def test_validate_markdown_text_keeps_leading_indented_code_block():
    assert _validate_markdown_text("    ```\n    literal backticks")["is_valid"] is True


def test_missing_values_fail_both_checks():
    import math

    import pandas as pd

    from core.data_recipe.text_format_validators import _build_text_format_validation_function

    df = pd.DataFrame({"x": [math.nan, pd.NA, None, "# ok"]}, dtype = object)
    for kind, expected in (("markdown", [False, False, False, True]), ("json", [False] * 4)):
        result = _build_text_format_validation_function(kind)(df)
        assert result["is_valid"].tolist() == expected, kind

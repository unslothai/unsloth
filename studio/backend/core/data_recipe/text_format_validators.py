# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

JSON_VALIDATION_FN_MARKER = "unsloth_json_validator"
MARKDOWN_VALIDATION_FN_MARKER = "unsloth_markdown_validator"

_MARKDOWN_FENCE_RE = re.compile(r"```")


@dataclass(frozen = True)
class TextFormatLocalCallableValidatorSpec:
    name: str
    drop: bool
    target_columns: list[str]
    batch_size: int
    format_kind: str


def split_text_format_local_callable_validators(
    recipe_core: dict[str, Any],
) -> tuple[dict[str, Any], list[TextFormatLocalCallableValidatorSpec]]:
    columns = recipe_core.get("columns")
    if not isinstance(columns, list):
        return recipe_core, []

    sanitized = deepcopy(recipe_core)
    sanitized_columns = sanitized.get("columns")
    if not isinstance(sanitized_columns, list):
        return sanitized, []

    kept_columns: list[Any] = []
    specs: list[TextFormatLocalCallableValidatorSpec] = []

    for column in sanitized_columns:
        if not isinstance(column, dict):
            kept_columns.append(column)
            continue

        maybe_spec = _parse_text_format_spec(column = column)
        if maybe_spec is None:
            kept_columns.append(column)
            continue
        specs.append(maybe_spec)

    sanitized["columns"] = kept_columns
    return sanitized, specs


def register_text_format_local_callable_validators(
    *, builder, specs: list[TextFormatLocalCallableValidatorSpec]
) -> None:
    if not specs:
        return

    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import (
        LocalCallableValidatorParams,
        ValidatorType,
    )

    for spec in specs:
        validation_function = _build_text_format_validation_function(spec.format_kind)
        builder.add_column(
            ValidationColumnConfig(
                name = spec.name,
                drop = spec.drop,
                target_columns = spec.target_columns,
                validator_type = ValidatorType.LOCAL_CALLABLE,
                validator_params = LocalCallableValidatorParams(
                    validation_function = validation_function,
                ),
                batch_size = spec.batch_size,
            )
        )


def _parse_text_format_spec(
    *, column: dict[str, Any]
) -> TextFormatLocalCallableValidatorSpec | None:
    if str(column.get("column_type") or "").strip() != "validation":
        return None
    if str(column.get("validator_type") or "").strip() != "local_callable":
        return None

    params = column.get("validator_params")
    if not isinstance(params, dict):
        return None

    fn_raw = params.get("validation_function")
    fn_name = fn_raw.strip() if isinstance(fn_raw, str) else ""
    format_kind = _text_format_kind_from_marker(fn_name)
    if format_kind is None:
        return None

    name = str(column.get("name") or "").strip()
    if not name:
        return None

    target_columns_raw = column.get("target_columns")
    target_columns = (
        [value.strip() for value in target_columns_raw if isinstance(value, str) and value.strip()]
        if isinstance(target_columns_raw, list)
        else []
    )
    if not target_columns:
        return None

    return TextFormatLocalCallableValidatorSpec(
        name = name,
        drop = bool(column.get("drop") is True),
        target_columns = target_columns,
        batch_size = _parse_batch_size(column.get("batch_size")),
        format_kind = format_kind,
    )


def _text_format_kind_from_marker(fn_name: str) -> str | None:
    if fn_name == JSON_VALIDATION_FN_MARKER:
        return "json"
    if fn_name == MARKDOWN_VALIDATION_FN_MARKER:
        return "markdown"
    return None


def _parse_batch_size(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    return parsed if parsed >= 1 else 10


@lru_cache(maxsize = 4)
def _build_text_format_validation_function(format_kind: str):
    def _validator(df):
        import pandas as pd

        row_count = int(len(df.index))
        if row_count == 0:
            return pd.DataFrame({"is_valid": []})

        value_column = str(df.columns[0]) if len(df.columns) > 0 else ""
        values = (
            ["" for _ in range(row_count)]
            if not value_column
            else [_coerce_validation_value(value) for value in df[value_column].tolist()]
        )

        results = [
            _validate_text_format(value = value, format_kind = format_kind) for value in values
        ]
        return pd.DataFrame(results)

    _validator.__name__ = f"{format_kind}_format_validator"
    return _validator


def _coerce_validation_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    return str(value)


def _reject_json_constant(constant: str) -> float:
    raise ValueError(f"Invalid JSON constant: {constant}")


def _validate_text_format(*, value: Any, format_kind: str) -> dict[str, Any]:
    if format_kind == "json":
        return _validate_json_text(value)
    if format_kind == "markdown":
        return _validate_markdown_text(value)
    return {
        "is_valid": False,
        "error_count": 1,
        "error_message": f"Unsupported format validator: {format_kind}",
        "severity": None,
        "code": None,
        "labels": [],
        "codeframe": None,
        "warning_count": 0,
    }


def _validate_json_text(value: Any) -> dict[str, Any]:
    if isinstance(value, (dict, list)):
        payload = value
    else:
        stripped = str(value).strip()
        if not stripped:
            return _invalid_result("JSON value is empty.")
        try:
            payload = json.loads(stripped, parse_constant = _reject_json_constant)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            return _invalid_result(str(exc))
    try:
        json.dumps(payload, allow_nan = False)
    except (TypeError, ValueError) as exc:
        return _invalid_result(str(exc))
    return _valid_result()


def _markdown_segments_outside_fences(value: str) -> list[str]:
    segments: list[str] = []
    cursor = 0
    in_fence = False
    for match in _MARKDOWN_FENCE_RE.finditer(value):
        if not in_fence:
            segments.append(value[cursor : match.start()])
        in_fence = not in_fence
        cursor = match.end()
    if not in_fence:
        segments.append(value[cursor:])
    return segments


def _validate_markdown_text(value: Any) -> dict[str, Any]:
    stripped = str(value).strip()
    if not stripped:
        return _invalid_result("Markdown value is empty.")
    fence_count = len(_MARKDOWN_FENCE_RE.findall(stripped))
    if fence_count % 2 != 0:
        return _invalid_result("Markdown has an unclosed code fence.")
    for segment in _markdown_segments_outside_fences(stripped):
        if segment.count("[") != segment.count("]"):
            return _invalid_result("Markdown has unbalanced link brackets.")
        if segment.count("(") != segment.count(")"):
            return _invalid_result("Markdown has unbalanced parentheses.")
    return _valid_result()


def _valid_result() -> dict[str, Any]:
    return {
        "is_valid": True,
        "error_count": 0,
        "error_message": "",
        "severity": None,
        "code": None,
        "labels": [],
        "codeframe": None,
        "warning_count": 0,
    }


def _invalid_result(message: str) -> dict[str, Any]:
    return {
        "is_valid": False,
        "error_count": 1,
        "error_message": message,
        "severity": None,
        "code": None,
        "labels": [],
        "codeframe": None,
        "warning_count": 0,
    }

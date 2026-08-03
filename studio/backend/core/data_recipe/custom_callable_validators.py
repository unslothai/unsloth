# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import binascii
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from loggers import get_logger

logger = get_logger(__name__)

CUSTOM_VALIDATION_FN_MARKER = "unsloth_custom_validator"

CUSTOM_VALIDATION_FN_NAME = "validate"

CUSTOM_SOURCE_MAX_CHARS = 64 * 1024
CUSTOM_MARKER_MAX_CHARS = 128 * 1024
BATCH_SIZE_MAX = 512


@dataclass(frozen = True)
class CustomCallableValidatorSpec:
    name: str
    drop: bool
    target_columns: list[str]
    batch_size: int
    source: str


def encode_validation_source(source: str) -> str:
    """Encode custom validator source for embedding in a recipe marker string."""
    raw = source.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_validation_source(value: str) -> str:
    # Bound the decode input so a crafted marker cannot run padding math or
    # base64 work on an arbitrarily large string.
    if len(value) > CUSTOM_MARKER_MAX_CHARS:
        return ""
    padded = value + "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _reject_malformed_marker_column(column: dict[str, Any]) -> None:
    """Reject a marker-prefixed custom column that failed to parse.

    Leaving the marker string in place would hand data_designer a non-callable
    validation_function and fail later with an opaque error; fail early with a
    clear message instead. Columns whose marker is not ours are left alone.
    """
    params = column.get("validator_params")
    fn_raw = params.get("validation_function") if isinstance(params, dict) else None
    fn_name = fn_raw.strip() if isinstance(fn_raw, str) else ""
    if not fn_name.startswith(f"{CUSTOM_VALIDATION_FN_MARKER}:"):
        return
    name = str(column.get("name") or "").strip()
    raise ValueError(
        f"Column '{name or '?'}' has a malformed {CUSTOM_VALIDATION_FN_MARKER} marker.",
    )


def split_custom_callable_validators(
    recipe_core: dict[str, Any],
) -> tuple[dict[str, Any], list[CustomCallableValidatorSpec]]:
    columns = recipe_core.get("columns")
    if not isinstance(columns, list):
        return recipe_core, []

    sanitized = deepcopy(recipe_core)
    sanitized_columns = sanitized.get("columns")
    if not isinstance(sanitized_columns, list):
        return sanitized, []

    kept_columns: list[Any] = []
    custom_specs: list[CustomCallableValidatorSpec] = []

    for column in sanitized_columns:
        if not isinstance(column, dict):
            kept_columns.append(column)
            continue

        maybe_spec = _parse_custom_spec(column = column)
        if maybe_spec is None:
            _reject_malformed_marker_column(column)
            kept_columns.append(column)
            continue
        custom_specs.append(maybe_spec)

    sanitized["columns"] = kept_columns
    return sanitized, custom_specs


def _parse_custom_spec(*, column: dict[str, Any]) -> CustomCallableValidatorSpec | None:
    if str(column.get("column_type") or "").strip() != "validation":
        return None
    if str(column.get("validator_type") or "").strip() != "local_callable":
        return None

    params = column.get("validator_params")
    if not isinstance(params, dict):
        return None

    fn_raw = params.get("validation_function")
    fn_name = fn_raw.strip() if isinstance(fn_raw, str) else ""
    marker = f"{CUSTOM_VALIDATION_FN_MARKER}:"
    if not fn_name.startswith(marker):
        return None

    source = decode_validation_source(fn_name[len(marker) :].strip())
    if not source:
        return None
    if len(source) > CUSTOM_SOURCE_MAX_CHARS:
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

    return CustomCallableValidatorSpec(
        name = name,
        drop = bool(column.get("drop") is True),
        target_columns = target_columns,
        batch_size = _parse_batch_size(column.get("batch_size")),
        source = source,
    )


def _parse_batch_size(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    if parsed < 1:
        return 10
    return min(parsed, BATCH_SIZE_MAX)


def register_custom_callable_validators(
    *, builder, specs: list[CustomCallableValidatorSpec]
) -> None:
    if not specs:
        return

    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import (
        LocalCallableValidatorParams,
        ValidatorType,
    )

    for spec in specs:
        validation_function = _build_custom_validation_function(spec.source)
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


@lru_cache(maxsize = 64)
def _build_custom_validation_function(source: str):
    """Compile user-supplied validator source into a DataDesigner local callable.

    The source must define a callable named ``validate`` following the local
    callable contract: ``validate(df: pd.DataFrame) -> pd.DataFrame`` where the
    returned frame contains a boolean ``is_valid`` column. Any extra columns
    become per-row validation metadata.

    The namespace is pre-seeded with the common modules ``pd`` (pandas),
    ``subprocess``, ``tempfile`` and ``Path`` (pathlib.Path) so sources do not
    need to import them; imports are only required for additional modules the
    user's code relies on.

    Running user-supplied Python is arbitrary code execution. This is only ever
    reached for recipes that explicitly opt in through the "Advanced custom
    check" node (the UI requires a consent acknowledgement before a run can
    start) and executes inside the local job worker subprocess.
    """
    import pandas as pd
    import subprocess
    import tempfile
    from pathlib import Path

    namespace: dict[str, Any] = {
        "pd": pd,
        "subprocess": subprocess,
        "tempfile": tempfile,
        "Path": Path,
    }
    try:
        exec(source, namespace)  # noqa: S102 - user-authored validator source
    except Exception as exc:
        raise ValueError(f"Custom validator source failed to compile: {exc}") from exc

    fn = namespace.get(CUSTOM_VALIDATION_FN_NAME)
    if not callable(fn):
        raise ValueError(
            f"Custom validator must define a callable named '{CUSTOM_VALIDATION_FN_NAME}'.",
        )

    def _validator(df):
        input_row_count = int(len(df.index))
        try:
            result = fn(df)
        except Exception:
            # The exception text can leak local paths/env details into rows
            # that are persisted and shipped back to clients, so surface a
            # generic message and keep the full detail server-side only.
            logger.warning(
                "Custom validator raised during execution",
                exc_info = True,
            )
            return _error_results(
                input_row_count,
                "Custom validator raised an error.",
            )
        columns = getattr(result, "columns", None)
        if not hasattr(result, "columns") or "is_valid" not in columns:
            return _error_results(
                input_row_count,
                "Custom validator must return a DataFrame with an 'is_valid' column.",
            )
        result_row_count = int(len(result.index))
        if result_row_count != input_row_count:
            return _error_results(
                input_row_count,
                "Custom validator returned "
                f"{result_row_count} rows for {input_row_count} input rows; "
                "results must be one per input row.",
            )
        return result

    _validator.__name__ = f"{CUSTOM_VALIDATION_FN_MARKER}_{CUSTOM_VALIDATION_FN_NAME}"
    return _validator


def _error_results(row_count: int, message: str) -> Any:
    import pandas as pd  # lazy import
    return pd.DataFrame(
        {
            "is_valid": [False for _ in range(row_count)],
            "error_count": [1 for _ in range(row_count)],
            "error_message": [message for _ in range(row_count)],
        }
    )

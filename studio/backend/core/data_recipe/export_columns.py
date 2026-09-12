# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for reasoning about which recipe columns survive export."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Iterable

from data_designer.config.processors import ProcessorType  # pyright: ignore[reportMissingImports]


def _is_glob(pattern: str) -> bool:
    return "*" in pattern


def column_names_after_drop_processors(
    names: Iterable[str], processor_configs: list[Any]
) -> set[str]:
    """Return ``names`` minus columns removed by ``drop_columns`` processors."""
    remaining = set(names)
    for processor_config in processor_configs:
        if processor_config.processor_type != ProcessorType.DROP_COLUMNS:
            continue
        for pattern in processor_config.column_names:
            if _is_glob(pattern):
                matched = {name for name in remaining if fnmatch(name, pattern)}
                remaining -= matched
            elif pattern in remaining:
                remaining.remove(pattern)
    return remaining


def recipe_would_export_columns(columns: list[Any], processor_configs: list[Any] | None) -> bool:
    """True when at least one column would remain in the final dataset.

    Data Designer's ``validate_columns_not_all_dropped`` only counts generated
    columns (non-seed) with ``drop=False``. Seed columns kept for export — and
    only removed later by a ``drop_columns`` processor — still produce output
    but trigger that check incorrectly.
    """
    processors = processor_configs or []
    kept_names = [column.name for column in columns if not column.drop]
    return len(column_names_after_drop_processors(kept_names, processors)) > 0


def filter_studio_validation_violations(
    violations: list[Any], *, columns: list[Any], processor_configs: list[Any] | None
) -> list[Any]:
    if not recipe_would_export_columns(columns, processor_configs):
        return violations

    from data_designer.engine.validation import ViolationType  # pyright: ignore[reportMissingImports]
    return [
        violation for violation in violations if violation.type != ViolationType.ALL_COLUMNS_DROPPED
    ]

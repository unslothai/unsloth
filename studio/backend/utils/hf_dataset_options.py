# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation for Hugging Face dataset configuration and split selectors."""

from __future__ import annotations

import re
import unicodedata


MAX_HF_DATASET_OPTION_LENGTH = 128
HF_DATASET_SPLIT_NAME_PATTERN = re.compile(r"\w+(?:\.\w+)*")

_CONFIG_FORBIDDEN_PATTERN = re.compile(r"[<>:/\\|?*]")
_SPLIT_BOUNDARY = r"-?[0-9](?:_?[0-9])*%?"
_SPLIT_PART_PATTERN = re.compile(
    rf"({HF_DATASET_SPLIT_NAME_PATTERN.pattern})"
    rf"(?:\[({_SPLIT_BOUNDARY})?:({_SPLIT_BOUNDARY})?\])?"
    r"(?:\((closest|pct1_dropremainder)\))?"
)


def has_unsafe_hf_dataset_option_characters(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value)


def valid_hf_dataset_config_name(value: str, *, allow_empty: bool = False) -> bool:
    if not value:
        return allow_empty
    if len(value) > MAX_HF_DATASET_OPTION_LENGTH:
        return False
    if value in {".", ".."} or has_unsafe_hf_dataset_option_characters(value):
        return False
    return _CONFIG_FORBIDDEN_PATTERN.search(value) is None


def valid_hf_dataset_split_name(value: str) -> bool:
    return bool(
        value
        and len(value) <= MAX_HF_DATASET_OPTION_LENGTH
        and ".." not in value
        and HF_DATASET_SPLIT_NAME_PATTERN.fullmatch(value)
    )


def _valid_percent_boundary(value: str | None, percent_mode: bool) -> bool:
    if not value or not percent_mode:
        return True
    return abs(int(value.removesuffix("%").replace("_", ""))) <= 100


def hf_dataset_split_instruction_names(value: str) -> tuple[str, ...]:
    if (
        not value
        or len(value) > MAX_HF_DATASET_OPTION_LENGTH
        or ".." in value
        or "/" in value
        or "\\" in value
        or has_unsafe_hf_dataset_option_characters(value)
    ):
        return ()

    names: list[str] = []
    percent_rounding = None
    for part in re.split(r"\s*\+\s*", value):
        match = _SPLIT_PART_PATTERN.fullmatch(part)
        if match is None:
            return ()
        name, start, end, rounding = match.groups()
        uses_percent = bool((start and start.endswith("%")) or (end and end.endswith("%")))
        if (
            not _valid_percent_boundary(start, uses_percent)
            or not _valid_percent_boundary(end, uses_percent)
            or (rounding is not None and not uses_percent)
        ):
            return ()
        if uses_percent:
            effective_rounding = rounding or "closest"
            if percent_rounding is not None and percent_rounding != effective_rounding:
                return ()
            percent_rounding = effective_rounding
        names.append(name)
    return tuple(dict.fromkeys(names))


def valid_hf_dataset_split_instruction(value: str, *, allow_empty: bool = False) -> bool:
    if not value:
        return allow_empty
    return bool(hf_dataset_split_instruction_names(value))

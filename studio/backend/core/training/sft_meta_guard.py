# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for guarding Unsloth SFT tokenizer preflight on meta-backed models."""

from __future__ import annotations

import os
from typing import Iterable


def _get_model_names(model, requested_model_name: str | None = None) -> list[str]:
    names: list[str] = []
    if requested_model_name:
        names.append(requested_model_name)

    config = getattr(model, "config", None)
    config_name = getattr(config, "_name_or_path", None)
    if config_name:
        names.append(config_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        for candidate in (name, name.lower()):
            candidate = candidate.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
    return deduped


def extend_ignored_tokenizer_names(names: Iterable[str]) -> list[str]:
    existing = [
        item
        for item in os.environ.get("UNSLOTH_IGNORED_TOKENIZER_NAMES", "").split("\n")
        if item
    ]
    updated = list(existing)
    seen = set(existing)
    for name in names:
        if name and name not in seen:
            updated.append(name)
            seen.add(name)
    os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = "\n".join(updated)
    return updated


def inspect_meta_backed_embeddings(model) -> list[str]:
    reasons: list[str] = []
    checks = (
        ("input_embeddings", "get_input_embeddings"),
        ("output_embeddings", "get_output_embeddings"),
    )
    for label, getter_name in checks:
        getter = getattr(model, getter_name, None)
        if getter is None:
            continue
        try:
            module = getter()
            weight = getattr(module, "weight", None)
        except Exception as exc:
            reasons.append(f"{label}_access_error={type(exc).__name__}")
            continue
        if weight is None:
            continue
        if bool(getattr(weight, "is_meta", False)):
            reasons.append(f"{label}=meta")
    return reasons


def maybe_enable_sft_meta_guard(model, requested_model_name: str | None, logger) -> bool:
    reasons = inspect_meta_backed_embeddings(model)
    if not reasons:
        return False

    names = _get_model_names(model, requested_model_name)
    updated = extend_ignored_tokenizer_names(names)
    hf_device_map = getattr(model, "hf_device_map", None)
    logger.warning(
        "Skipping fix_untrained_tokens for meta-backed model before SFTTrainer init",
        model_name = requested_model_name,
        guard_model_names = names,
        reasons = reasons,
        hf_device_map_populated = bool(hf_device_map),
        hf_device_map_entries = len(hf_device_map) if isinstance(hf_device_map, dict) else None,
        ignored_tokenizer_name_count = len(updated),
    )
    return True

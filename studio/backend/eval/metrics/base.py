# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# A scorer compares one prediction string against a reference value.
Scorer = Callable[[str, Any], "MetricResult"]


@dataclass
class MetricResult:
    score: float                      # in [0, 1]
    breakdown: Optional[dict] = None  # e.g. serialized ScoreNode for JSON
    error: Optional[str] = None       # per-example issue (e.g. unparseable JSON)


@dataclass
class ConfigField:
    name: str
    type: str          # "bool" | "float" | "string" | "json"
    default: Any
    label: str
    options: Optional[list[str]] = None  # if set, the UI renders a dropdown


@dataclass
class MetricSpec:
    name: str
    label: str
    reference_kind: str            # "text" | "json"
    config_fields: list[ConfigField]
    build: Callable[[dict], Scorer]  # config -> scorer closure

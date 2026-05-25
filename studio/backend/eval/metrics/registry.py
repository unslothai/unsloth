# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import asdict

from .base import MetricSpec, Scorer
from . import exact_match, text_similarity, json_document

_SPECS: dict[str, MetricSpec] = {
    exact_match.SPEC.name: exact_match.SPEC,
    text_similarity.SPEC.name: text_similarity.SPEC,
    json_document.SPEC.name: json_document.SPEC,
}


def register(spec: MetricSpec) -> None:
    _SPECS[spec.name] = spec


def make_scorer(name: str, config: dict) -> Scorer:
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unknown metric {name!r}. Known: {sorted(_SPECS)}")
    return spec.build(config or {})


def list_metrics() -> list[dict]:
    out = []
    for spec in _SPECS.values():
        out.append({
            "name": spec.name,
            "label": spec.label,
            "reference_kind": spec.reference_kind,
            "config_fields": [asdict(f) for f in spec.config_fields],
        })
    return out

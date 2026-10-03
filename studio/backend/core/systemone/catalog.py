# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""System One decision models Studio can serve on ``POST /v1/systemone``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LAYA_REPO = "convaiinnovations/laya"


@dataclass(frozen = True)
class Checkpoint:
    name: str
    source: str
    subfolder: str | None
    description: str
    download_bytes: int = 0

    @property
    def is_local(self) -> bool:
        return Path(self.source).expanduser().is_dir()


CHECKPOINTS = {
    c.name: c
    for c in (
        Checkpoint(
            "laya-multilingual",
            LAYA_REPO,
            "multilingual",
            "Laya on mmBERT-base: 100+ languages, 1024-token context.",
            678_201_636,
        ),
        Checkpoint(
            "laya-english",
            LAYA_REPO,
            None,
            "Laya on ModernBERT-large: English, 512-token context.",
            846_195_574,
        ),
        Checkpoint(
            "laya-typed-decisions",
            LAYA_REPO,
            "typed-decisions",
            "Laya English fine-tuned on four typed-decision workflows, 1024-token context.",
            846_195_716,
        ),
    )
}

# Names TypeSafe's and OpenJev's SDKs send by default, so an unmodified client reaches the configured model.
DEFAULT_ALIASES = frozenset({"default", "laya", "jev-latest", "jev-preview", "openjev-latest"})
LOCAL_NAME = "laya-local"
CONNECTION_PREFIX = "connection:"


@dataclass(frozen = True)
class Connection:
    provider_id: str
    model: str

    @property
    def name(self) -> str:
        return f"{CONNECTION_PREFIX}{self.provider_id}:{self.model}"


def parse_connection(value: object) -> Connection | None:
    if not isinstance(value, str) or not value.startswith(CONNECTION_PREFIX):
        return None
    provider_id, _, model = value[len(CONNECTION_PREFIX) :].partition(":")
    return Connection(provider_id, model) if provider_id and model else None


LISTED_DECISION_MODELS: dict[tuple[str, str], tuple[float, list[str]]] = {}


def decision_models(row: dict) -> list[str] | None:
    from core.inference.providers import answers_decisions_only

    if answers_decisions_only(row["provider_type"], row.get("api_type")):
        return row["models"]
    if row["provider_type"] == "openrouter":
        return LISTED_DECISION_MODELS.get((row["id"], row["updated_at"]), (0.0, []))[1]
    return None


def decision_connections() -> list[tuple[dict, list[str]]]:
    # Connections are per account, the Decision API is installation-wide: read the owner's.
    from storage.providers_db import list_providers
    from utils.account_context import OWNER, run_as
    return [
        (row, models)
        for row in run_as(OWNER, list_providers)
        if row["is_enabled"] and (models := decision_models(row))
    ]


def default_checkpoint() -> Checkpoint | Connection:
    from utils.systemone_settings import get_model

    configured = get_model()
    if configured in CHECKPOINTS:
        return CHECKPOINTS[configured]
    if connection := parse_connection(configured):
        return connection
    subfolder = os.environ.get("UNSLOTH_SYSTEMONE_SUBFOLDER", "").strip() or None
    return Checkpoint(LOCAL_NAME, configured, subfolder, "Local Laya checkpoint.")


def resolve(model: str) -> Checkpoint | Connection | None:
    name = (model or "").strip()
    if name in DEFAULT_ALIASES:
        return default_checkpoint()
    if name == LOCAL_NAME or name.startswith(CONNECTION_PREFIX):
        checkpoint = default_checkpoint()
        return checkpoint if checkpoint.name == name else None
    return CHECKPOINTS.get(name)

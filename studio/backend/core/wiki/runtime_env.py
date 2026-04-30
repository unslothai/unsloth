# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime wiki environment metadata and override persistence helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional


WikiEnvKind = Literal["bool", "int", "float", "string"]


@dataclass(frozen = True)
class WikiEnvSpec:
    name: str
    kind: WikiEnvKind
    default: str
    description: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None


WIKI_ENV_SPECS: tuple[WikiEnvSpec, ...] = (
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_VAULT",
        kind = "string",
        default = "/tmp/unsloth_wiki",
        description = "Root wiki vault directory.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_WATCHER",
        kind = "bool",
        default = "true",
        description = "Enable background wiki raw-folder watcher.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_QUERY_ON_INGEST",
        kind = "bool",
        default = "true",
        description = "Run automatic wiki analysis after ingestion.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_QUERY_CHAT_HISTORY",
        kind = "bool",
        default = "false",
        description = "Include chat history files in auto-analysis.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_CHAT_HISTORY_FLUSH_SECONDS",
        kind = "int",
        default = "600",
        description = "Flush interval for batched chat history ingestion.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_LINT_EVERY",
        kind = "int",
        default = "10",
        description = "Maintenance cadence for lint/retry/enrichment workflows.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES",
        kind = "int",
        default = "24",
        description = "Max fallback pages scanned per maintenance run.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_PENDING_INGEST_INTERVAL_SECONDS",
        kind = "int",
        default = "45",
        description = "Minimum delay between pending ingest sweeps.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_PENDING_INGEST_MAX_FILES_PER_CHAT",
        kind = "int",
        default = "1",
        description = "Max pending raw files ingested per chat request.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_RAG_MAX_PAGES",
        kind = "int",
        default = "8",
        description = "Route-level RAG max pages.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE",
        kind = "int",
        default = "1800",
        description = "Route-level RAG max chars per selected page.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS",
        kind = "int",
        default = "12000",
        description = "Route-level RAG max total chars.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_RAG_INCLUDE_SOURCE_PAGES",
        kind = "bool",
        default = "true",
        description = "Include source pages in route-level retrieval.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_INDEX_INCLUDE_SOURCE_PAGES",
        kind = "bool",
        default = "true",
        description = "Include source pages in index-level retrieval.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_LOG_INJECTED_CONTEXT",
        kind = "bool",
        default = "true",
        description = "Log injected wiki context in backend logs.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_LOG_INJECTED_CONTEXT_MAX_CHARS",
        kind = "int",
        default = "12000",
        description = "Max chars logged for injected context.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_LLM_MAX_TOKENS",
        kind = "int",
        default = "1200",
        description = "Token budget for wiki-generated responses.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_MODEL_TOKEN_CAPACITY",
        kind = "int",
        default = "125000",
        description = "Optional model token context size used to derive character limits when per-knob overrides are unset.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_MODEL_SAFE_TOKEN_RATIO",
        kind = "float",
        default = "0.50",
        description = "Fraction of model token capacity reserved for safe wiki context budgeting.",
        minimum = 0.10,
        maximum = 0.95,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_MODEL_CHARS_PER_TOKEN",
        kind = "float",
        default = "4.0",
        description = "Character-per-token estimate used when converting token budgets to character defaults.",
        minimum = 1.0,
        maximum = 8.0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_EXTRACT_SOURCE_MAX_CHARS",
        kind = "int",
        default = "20000",
        description = "Max source chars used by extraction.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_SOURCE_EXCERPT_MAX_CHARS",
        kind = "int",
        default = "8000",
        description = "Max source excerpt chars persisted into pages.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_CHUNK_ANALYSIS_CONTEXT_WINDOW_CHARS",
        kind = "int",
        default = "125000",
        description = "Chunk planner context window in characters.",
        minimum = 1200,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_CHUNK_ANALYSIS_TARGET_RATIO",
        kind = "float",
        default = "0.70",
        description = "Target chunk-size ratio relative to chunk analysis context window.",
        minimum = 0.35,
        maximum = 0.95,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_CHUNK_ANALYSIS_OVERLAP_RATIO",
        kind = "float",
        default = "0.08",
        description = "Chunk overlap ratio relative to target chunk size.",
        minimum = 0.0,
        maximum = 0.40,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_CHUNK_ANALYSIS_MIN_CHARS",
        kind = "int",
        default = "1200",
        description = "Minimum chunk size in characters.",
        minimum = 300,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_CHUNK_ANALYSIS_MAX_CHARS",
        kind = "int",
        default = "125000",
        description = "Maximum chunk size in characters.",
        minimum = 1200,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_RANKING_MAX_CHARS",
        kind = "int",
        default = "24000",
        description = "Max chars read per page during ranking.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_MAX_CONTEXT_PAGES",
        kind = "int",
        default = "16",
        description = "Engine query max context pages.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_MAX_CHARS_PER_PAGE",
        kind = "int",
        default = "3500",
        description = "Engine query max chars per page.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_QUERY_CONTEXT_MAX_CHARS",
        kind = "int",
        default = "24000",
        description = "Engine query max total context chars.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_INCLUDE_ANALYSIS_IN_QUERY",
        kind = "bool",
        default = "true",
        description = "Include prior analysis pages in query retrieval.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_RANKING_ANALYSIS_FIRST",
        kind = "bool",
        default = "true",
        description = "Force analysis pages to rank before other page types.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_RANKING_LINK_DEPTH",
        kind = "int",
        default = "2",
        description = "Graph link-depth used during ranking expansion.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_RANKING_LINK_FANOUT",
        kind = "int",
        default = "8",
        description = "Graph link fanout used during ranking expansion.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_RANKING_LINK_LLM_SELECTOR_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM-based link selection during ranking expansion.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_RANKING_LINK_LLM_SELECTOR_MAX_CANDIDATES",
        kind = "int",
        default = "24",
        description = "Max outgoing links exposed to LLM link selector per source page.",
        minimum = 4,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_LLM_RERANK_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM reranking for wiki candidates.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_LLM_RERANK_CANDIDATES",
        kind = "int",
        default = "32",
        description = "Candidate pool size before reranking.",
        minimum = 3,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_LLM_RERANK_TOP_N",
        kind = "int",
        default = "12",
        description = "Top reranked candidates kept for context.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_LLM_RERANK_PREVIEW_CHARS",
        kind = "int",
        default = "420",
        description = "Preview chars per candidate in rerank prompt.",
        minimum = 80,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_LLM_RERANK_LOG_OUTPUT",
        kind = "bool",
        default = "true",
        description = "Log reranker outputs for diagnostics.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_LLM_RERANK_LOG_MAX_CHARS",
        kind = "int",
        default = "4000",
        description = "Max chars logged per reranker output entry.",
        minimum = 200,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_CONTEXT_FRACTION",
        kind = "float",
        default = "0.70",
        description = "Fraction of context budget for auto-analysis.",
        minimum = 0.0,
        maximum = 1.0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_CHARS_PER_TOKEN",
        kind = "int",
        default = "4",
        description = "Auto-analysis token-to-char conversion heuristic.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_ON_FALLBACK",
        kind = "bool",
        default = "true",
        description = "Retry auto-analysis when fallback quality gates trigger.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_MAX_RETRIES",
        kind = "int",
        default = "3",
        description = "Max retries for fallback-triggered auto-analysis.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_RETRY_REDUCTION",
        kind = "float",
        default = "0.5",
        description = "Context reduction factor per auto-analysis retry.",
        minimum = 0.0,
        maximum = 1.0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_MIN_CONTEXT_CHARS",
        kind = "int",
        default = "8000",
        description = "Minimum context chars for reduced retry attempts.",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY",
        kind = "bool",
        default = "false",
        description = "Use source-only context in auto-analysis mode.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_AUTO_ANALYSIS_SOURCE_ONLY_FINAL_RETRY",
        kind = "bool",
        default = "true",
        description = "Force source-only mode on final retry.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_LOW_UNIQUE_RATIO_MIN_TOKENS",
        kind = "int",
        default = "40",
        description = "Minimum token count before repetition gate applies.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_LOW_UNIQUE_RATIO_THRESHOLD",
        kind = "float",
        default = "0.25",
        description = "Unique-token threshold for repetition fallback gating.",
        minimum = 0.01,
        maximum = 1.0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_MERGE_MAINTENANCE_MAX_MERGES",
        kind = "int",
        default = "512",
        description = "Maximum merges per maintenance run.",
        minimum = 1,
        maximum = 512,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_KNOWLEDGE_MAX_INCREMENTAL_UPDATES",
        kind = "int",
        default = "48",
        description = "Max retained incremental updates per knowledge page.",
        minimum = 1,
        maximum = 256,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_FILL_GAPS_FROM_WEB",
        kind = "bool",
        default = "false",
        description = "Enable web gap-filling during wiki enrichment.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_QUERIES",
        kind = "int",
        default = "4",
        description = "Max web gap-fill queries per enrich pass.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_RESULTS",
        kind = "int",
        default = "3",
        description = "Max web results considered per gap-fill query.",
        minimum = 1,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_MAX_SNIPPET_CHARS",
        kind = "int",
        default = "280",
        description = "Snippet chars used when drafting gap-fill pages.",
        minimum = 80,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_LLM_PLANNER_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM planner for web gap-fill query planning.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_WEB_GAP_LLM_SELECTOR_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM selector for semantic web result selection in gap fill.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_LLM_SELECTOR_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM-based selection of Enrichment links for analysis pages.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_LLM_SELECTOR_MAX_CANDIDATES",
        kind = "int",
        default = "48",
        description = "Max candidate links considered by the enrichment LLM selector per group.",
        minimum = 8,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_INDEX_LLM_TITLE_ON_REBUILD",
        kind = "bool",
        default = "false",
        description = "Generate analysis index titles with LLM during index rebuild.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_REFRESH_OLDEST_NON_FALLBACK_PAGES",
        kind = "int",
        default = "0",
        description = "Refresh oldest non-fallback analysis pages before enrichment (0 disables).",
        minimum = 0,
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENRICH_REPAIR_ANSWER_LINKS",
        kind = "bool",
        default = "false",
        description = "Repair unresolved wiki links inside analysis Answer sections during enrichment.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_MERGE_LLM_CANDIDATE_PLANNER_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM semantic candidate planning for merge maintenance.",
    ),
    WikiEnvSpec(
        name = "UNSLOTH_WIKI_ENGINE_ENTITY_QUERY_FOCUS_LLM_ENABLED",
        kind = "bool",
        default = "true",
        description = "Enable LLM entity-intent parsing during retrieval ranking.",
    ),
)


_WIKI_ENV_SPECS_BY_NAME = {spec.name: spec for spec in WIKI_ENV_SPECS}
WIKI_ENV_NAMES = frozenset(_WIKI_ENV_SPECS_BY_NAME.keys())


def wiki_env_overrides_file() -> Path:
    configured = os.getenv("UNSLOTH_WIKI_ENV_OVERRIDES_FILE")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path.home() / ".unsloth" / "studio" / "wiki_env_overrides.json"


def _validate_numeric_bounds(spec: WikiEnvSpec, value: float) -> None:
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"Value must be >= {spec.minimum}.")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"Value must be <= {spec.maximum}.")


def _normalize_bool(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "true"
    if value in {"0", "false", "no", "off"}:
        return "false"
    raise ValueError("Expected a boolean value (true/false).")


def _validate_value(spec: WikiEnvSpec, raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        raise ValueError("Value cannot be empty.")

    if spec.kind == "string":
        return value
    if spec.kind == "bool":
        return _normalize_bool(value)
    if spec.kind == "int":
        try:
            numeric = int(value)
        except ValueError as exc:
            raise ValueError("Expected an integer value.") from exc
        _validate_numeric_bounds(spec, float(numeric))
        return str(numeric)
    if spec.kind == "float":
        try:
            numeric = float(value)
        except ValueError as exc:
            raise ValueError("Expected a numeric value.") from exc
        _validate_numeric_bounds(spec, numeric)
        return str(numeric)
    raise ValueError(f"Unsupported variable type: {spec.kind}")


def load_wiki_env_overrides() -> dict[str, str]:
    path = wiki_env_overrides_file()
    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    cleaned: dict[str, str] = {}
    for raw_name, raw_value in payload.items():
        if not isinstance(raw_name, str):
            continue
        spec = _WIKI_ENV_SPECS_BY_NAME.get(raw_name)
        if spec is None:
            continue
        normalized_input = str(raw_value)
        try:
            cleaned[raw_name] = _validate_value(spec, normalized_input)
        except ValueError:
            continue
    return cleaned


def persist_wiki_env_overrides(overrides: dict[str, str]) -> Path:
    path = wiki_env_overrides_file()
    if not overrides:
        try:
            path.unlink(missing_ok = True)
        except OSError:
            pass
        return path

    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(overrides, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8",
    )
    return path


def apply_wiki_env_overrides(
    overrides: dict[str, str],
    override_existing: bool = True,
) -> dict[str, str]:
    applied: dict[str, str] = {}
    for name, value in overrides.items():
        if name not in WIKI_ENV_NAMES:
            continue
        if override_existing:
            os.environ[name] = value
            applied[name] = value
        elif name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def apply_stored_wiki_env_overrides(
    override_existing: bool = False,
) -> dict[str, str]:
    overrides = load_wiki_env_overrides()
    return apply_wiki_env_overrides(overrides, override_existing = override_existing)


def collect_wiki_env_state() -> list[dict[str, Any]]:
    overrides = load_wiki_env_overrides()
    state: list[dict[str, Any]] = []
    for spec in WIKI_ENV_SPECS:
        current = os.getenv(spec.name)
        current_value = current if current is not None else spec.default
        state.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "description": spec.description,
                "default_value": spec.default,
                "current_value": current_value,
                "source": "environment" if current is not None else "default",
                "has_override": spec.name in overrides,
                "override_value": overrides.get(spec.name),
                "minimum": spec.minimum,
                "maximum": spec.maximum,
            }
        )
    return state


def update_wiki_env_values(values: dict[str, Optional[str]]) -> dict[str, Any]:
    overrides = load_wiki_env_overrides()
    updated: list[str] = []
    cleared: list[str] = []
    invalid: dict[str, str] = {}

    for name, incoming in values.items():
        spec = _WIKI_ENV_SPECS_BY_NAME.get(name)
        if spec is None:
            invalid[name] = "Unknown wiki environment variable."
            continue

        if incoming is None or incoming.strip() == "":
            had_value = name in os.environ or name in overrides
            overrides.pop(name, None)
            os.environ.pop(name, None)
            if had_value:
                cleared.append(name)
            continue

        try:
            normalized = _validate_value(spec, incoming)
        except ValueError as exc:
            invalid[name] = str(exc)
            continue

        overrides[name] = normalized
        os.environ[name] = normalized
        updated.append(name)

    overrides_path = persist_wiki_env_overrides(overrides)
    return {
        "updated": sorted(set(updated)),
        "cleared": sorted(set(cleared)),
        "invalid": invalid,
        "overrides_file": str(overrides_path),
        "overrides_count": len(overrides),
    }

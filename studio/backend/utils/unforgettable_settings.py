# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted Unforgettable episode defaults and supervisor knobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from unforgettable.rims.plugin import coerce_twin_plugin
from unforgettable.supervisor import (
    DEFAULT_SUPERVISOR_TIMEOUT,
    VOTER_MODES,
    config_from_env,
    config_from_mapping,
)

UNFORGETTABLE_SETTING_KEY = "unforgettable"
PLANNER_VALUES = frozenset({"on", "off"})
FILTER_VALUES = frozenset({"on", "off"})
STAKES_VALUES = frozenset({"high"})


def memory_db_path() -> Path:
    from utils.paths import studio_root
    return studio_root() / "memory" / "memory.db"


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("Expected a boolean or null.")


def _coerce_optional_int(value: Any, *, minimum: int) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Expected a number, got a boolean.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"Expected an integer >= {minimum}.")
    return parsed


def _coerce_timeout(value: Any) -> float:
    if value is None or value == "":
        return DEFAULT_SUPERVISOR_TIMEOUT
    if isinstance(value, bool):
        raise ValueError("Expected a number, got a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a timeout in seconds.") from exc
    if parsed <= 0:
        raise ValueError("Timeout must be greater than 0.")
    return parsed


def normalize_unforgettable_settings(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    planner = source.get("planner")
    if planner is None or planner == "":
        planner_value = None
    else:
        planner_value = str(planner).strip().lower()
        if planner_value not in PLANNER_VALUES:
            raise ValueError("Planner must be on or off.")
    filter_flag = source.get("filter")
    if filter_flag is None or filter_flag == "":
        filter_value = None
    else:
        filter_value = str(filter_flag).strip().lower()
        if filter_value not in FILTER_VALUES:
            raise ValueError("Filter must be on or off.")
    stakes = source.get("stakes")
    if stakes is None or stakes == "":
        stakes_value = None
    else:
        stakes_value = str(stakes).strip().lower()
        if stakes_value not in STAKES_VALUES:
            raise ValueError("Stakes must be high or unset.")
    voter = source.get("voter")
    if voter is None or voter == "":
        voter_value = None
    else:
        voter_value = str(voter).strip().lower()
        if voter_value not in VOTER_MODES:
            raise ValueError("Voter must be off, advisory, or binding.")
    twin = source.get("twin_plugin")
    if twin is None or twin == "":
        twin_value = None
    else:
        try:
            twin_value = coerce_twin_plugin(twin)
        except ValueError as exc:
            raise ValueError("Twin plugin must be fs.copy or none.") from exc
    return {
        "planner": planner_value,
        "planner_model": _coerce_optional_str(source.get("planner_model")),
        "filter": filter_value,
        "filter_model": _coerce_optional_str(source.get("filter_model")),
        "judge_model": _coerce_optional_str(source.get("judge_model")),
        "user_label": _coerce_optional_str(source.get("user_label")),
        "stakes": stakes_value,
        "confirm_retry": _coerce_optional_bool(source.get("confirm_retry")),
        "skip_standing": bool(source.get("skip_standing") or False),
        "adapter_id": _coerce_optional_str(source.get("adapter_id")),
        "test_command": _coerce_optional_str(source.get("test_command")),
        "max_clones": _coerce_optional_int(source.get("max_clones"), minimum = 1),
        "max_sim_turns": _coerce_optional_int(source.get("max_sim_turns"), minimum = 1),
        "twin_plugin": twin_value,
        "voter": voter_value,
        "voter_model": _coerce_optional_str(source.get("voter_model")),
        "supervisor_url": _coerce_optional_str(source.get("supervisor_url")),
        "supervisor_timeout": _coerce_timeout(source.get("supervisor_timeout")),
    }


def _stored_settings() -> dict[str, Any]:
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(UNFORGETTABLE_SETTING_KEY, {})
    except Exception:
        stored = {}
    return stored if isinstance(stored, dict) else {}


def get_unforgettable_settings() -> dict[str, Any]:
    stored = normalize_unforgettable_settings(_stored_settings())
    env = config_from_env()
    planner = stored["planner"] if stored["planner"] is not None else (env.planner or "off")
    filter_flag = stored["filter"] if stored.get("filter") is not None else (env.filter or "on")
    voter = stored["voter"] if stored["voter"] is not None else env.voter
    return {
        **stored,
        "planner": planner or "off",
        "planner_model": stored["planner_model"] or env.planner_model,
        "filter": filter_flag or "on",
        "filter_model": stored.get("filter_model") or env.filter_model,
        "judge_model": stored.get("judge_model") or env.judge_model,
        "voter": voter or "off",
        "voter_model": stored["voter_model"] or env.voter_model,
        "supervisor_url": stored["supervisor_url"] or env.url,
        "supervisor_timeout": stored["supervisor_timeout"] or env.timeout,
        "db_path": str(memory_db_path()),
        "namespace": "default",
    }


def set_unforgettable_settings(payload: Any) -> dict[str, Any]:
    current = normalize_unforgettable_settings(_stored_settings())
    incoming = payload if isinstance(payload, dict) else {}
    merged = {**current, **incoming}
    normalized = normalize_unforgettable_settings(merged)
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({UNFORGETTABLE_SETTING_KEY: normalized})
    return get_unforgettable_settings()


def supervisor_config_from_settings():
    return config_from_mapping(get_unforgettable_settings())


def episode_extras_from_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fields to copy onto a virtual-model chat completion."""
    data = settings if settings is not None else get_unforgettable_settings()
    extras: dict[str, Any] = {}
    if data.get("planner"):
        extras["planner"] = data["planner"]
    if data.get("planner_model"):
        extras["planner_model"] = data["planner_model"]
    if data.get("filter"):
        extras["filter"] = data["filter"]
    if data.get("filter_model"):
        extras["filter_model"] = data["filter_model"]
    if data.get("judge_model"):
        extras["judge_model"] = data["judge_model"]
    if data.get("user_label"):
        extras["user_label"] = data["user_label"]
    if data.get("stakes"):
        extras["stakes"] = data["stakes"]
    if data.get("confirm_retry") is not None:
        extras["confirm_retry"] = data["confirm_retry"]
    if data.get("skip_standing"):
        extras["skip_standing"] = True
    if data.get("adapter_id"):
        extras["adapter_id"] = data["adapter_id"]
    if data.get("test_command"):
        extras["test_command"] = data["test_command"]
    if data.get("max_clones") is not None:
        extras["max_clones"] = data["max_clones"]
    if data.get("max_sim_turns") is not None:
        extras["max_sim_turns"] = data["max_sim_turns"]
    if data.get("twin_plugin"):
        extras["twin_plugin"] = data["twin_plugin"]
    if data.get("voter_model"):
        extras["voter_model"] = data["voter_model"]
    return extras

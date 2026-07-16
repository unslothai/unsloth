# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Canonical validation for account-scoped assets; errors never expose values."""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, overload


MAX_ID_CHARS = 128
MAX_NAME_CHARS = 200
_POLICY_PATH = Path(__file__).resolve().parents[2] / "user_assets_persistence_policy.json"
with _POLICY_PATH.open(encoding = "utf-8") as _policy_file:
    _PERSISTENCE_POLICY = json.load(_policy_file)

MAX_RECIPE_JSON_BYTES = int(_PERSISTENCE_POLICY["maxRecipeJsonBytes"])
MAX_EXECUTION_JSON_BYTES = int(_PERSISTENCE_POLICY["maxExecutionJsonBytes"])
MAX_EXECUTION_ERROR_BYTES = 4 * 1024
MAX_COMPLETED_COLUMNS = 1000
MAX_LEGACY_RECIPES = 100
MAX_LEGACY_EXECUTIONS = 500
MAX_LEGACY_BATCH_JSON_BYTES = int(_PERSISTENCE_POLICY["maxLegacyBatchJsonBytes"])

EXECUTION_METADATA_FIELDS = frozenset(
    {
        "jobId",
        "kind",
        "run_name",
        "status",
        "rows",
        "recipeSignature",
        "stage",
        "current_column",
        "completed_columns",
        "progress",
        "column_progress",
        "batch",
        "source_progress",
        "model_usage",
        "lastEventId",
        "datasetTotal",
        "analysis",
        "error",
        "createdAt",
        "finishedAt",
    }
)

_DENIED_SECRET_KEYS = frozenset(_PERSISTENCE_POLICY["deniedSecretKeys"])
_SAFE_SECRET_LOOKING_KEYS = frozenset(_PERSISTENCE_POLICY["safeSecretLookingKeys"])
_MCP_ENV_DENIED_KEY_PARTS = tuple(_PERSISTENCE_POLICY["mcpEnvDeniedKeyParts"])
_MCP_ENV_DENIED_KEY_SUFFIXES = tuple(_PERSISTENCE_POLICY["mcpEnvDeniedKeySuffixes"])
_MCP_ENV_DENIED_EXACT_KEYS = frozenset(_PERSISTENCE_POLICY["mcpEnvDeniedExactKeys"])
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_FIRST_CAMEL_BOUNDARY_RE = re.compile(r"(.)([A-Z][a-z]+)")
_SECOND_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")

JsonPathPart: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathPart, ...]
T = TypeVar("T")


@dataclass(frozen = True)
class UserAssetValidationError(ValueError):
    """Structured, value-free validation error shared by storage and API layers."""

    code: str
    message: str
    paths: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.message

    @property
    def field_paths(self) -> tuple[str, ...]:
        """Alias used by API detail builders."""

        return self.paths

    def to_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.paths:
            detail["paths"] = list(self.paths)
        return detail


def _normalize_key(key: str) -> str:
    value = _FIRST_CAMEL_BOUNDARY_RE.sub(r"\1_\2", key)
    value = _SECOND_CAMEL_BOUNDARY_RE.sub(r"\1_\2", value).lower()
    return _NON_WORD_RE.sub("_", value).strip("_")


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, Mapping, Sequence)):
        return len(value) > 0
    return True


def _looks_like_secret_key(normalized_key: str) -> bool:
    if normalized_key in _SAFE_SECRET_LOOKING_KEYS:
        return False
    if normalized_key in _DENIED_SECRET_KEYS:
        return True
    return normalized_key.endswith(
        (
            "_api_key",
            "_token",
            "_password",
            "_secret",
            "_credential",
            "_credentials",
            "_private_key",
            "_access_key",
            "_access_key_id",
        )
    )


def _is_mcp_env(path: JsonPath) -> bool:
    string_parts = [_normalize_key(part) for part in path if isinstance(part, str)]
    return bool(
        string_parts and string_parts[-1] == "env" and any("mcp" in p for p in string_parts[:-1])
    )


def _is_seed_source(path: JsonPath) -> bool:
    """Match only ``seed_config.source.token``, not same-named data fields."""

    string_parts = [_normalize_key(part) for part in path if isinstance(part, str)]
    return len(string_parts) >= 2 and string_parts[-2:] == ["seed_config", "source"]


def _is_structured_output_schema_property(path: JsonPath, value: Any) -> bool:
    """Allow credential-like names only in valid ``output_format.properties`` schemas."""

    string_parts = [_normalize_key(part) for part in path if isinstance(part, str)]
    return bool(
        string_parts
        and string_parts[-1] == "properties"
        and "output_format" in string_parts
        and (isinstance(value, Mapping) or isinstance(value, bool))
    )


def _is_secret_entry(path: JsonPath, key: str, value: Any) -> bool:
    if not _has_value(value):
        return False
    normalized = _normalize_key(key)
    if normalized in _SAFE_SECRET_LOOKING_KEYS:
        return False
    if _is_structured_output_schema_property(path, value):
        return False
    if normalized == "token" and _is_seed_source(path):
        return True
    if _looks_like_secret_key(normalized):
        return True
    # Deny provider-specific secret names only inside typed MCP env blocks.
    if _is_mcp_env(path):
        return (
            normalized in _MCP_ENV_DENIED_EXACT_KEYS
            or normalized.endswith(_MCP_ENV_DENIED_KEY_SUFFIXES)
            or any(part in normalized for part in _MCP_ENV_DENIED_KEY_PARTS)
        )
    # Hyphenated header names normalize into the exact denylist.
    return False


def format_json_path(path: JsonPath) -> str:
    """Format JSON paths without losing indices or unusual keys."""

    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif _IDENTIFIER_RE.fullmatch(part):
            rendered += f".{part}"
        else:
            rendered += f"[{json.dumps(part, ensure_ascii = False)}]"
    return rendered


def _secret_path_parts(value: Any, path: JsonPath = ()) -> list[JsonPath]:
    found: list[JsonPath] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                continue
            child_path = (*path, raw_key)
            if _is_secret_entry(path, raw_key, child):
                found.append(child_path)
            else:
                found.extend(_secret_path_parts(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_secret_path_parts(child, (*path, index)))
    return found


def find_secret_paths(value: Any) -> list[str]:
    """Return exact, value-free paths for non-empty secret fields."""

    return [format_json_path(path) for path in _secret_path_parts(value)]


def _redact(value: Any, path: JsonPath = ()) -> tuple[Any, list[JsonPath]]:
    redacted_paths: list[JsonPath] = []
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                # Validation rejects non-string keys later.
                result[raw_key] = child  # type: ignore[index]
                continue
            child_path = (*path, raw_key)
            if _is_secret_entry(path, raw_key, child):
                redacted_paths.append(child_path)
                continue
            clean_child, child_paths = _redact(child, child_path)
            result[raw_key] = clean_child
            redacted_paths.extend(child_paths)
        return result, redacted_paths
    if isinstance(value, (list, tuple)):
        result_list: list[Any] = []
        for index, child in enumerate(value):
            clean_child, child_paths = _redact(child, (*path, index))
            result_list.append(clean_child)
            redacted_paths.extend(child_paths)
        return result_list, redacted_paths
    return value, redacted_paths


def redact_secret_fields(value: T) -> tuple[T, list[str]]:
    """Copy ``value``, remove secrets, and report their paths."""

    clean, paths = _redact(value)
    return clean, [format_json_path(path) for path in paths]


def canonical_json(value: Any, max_bytes: int, field_name: str) -> str:
    """Serialize JSON deterministically and enforce the UTF-8 byte limit."""

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii = False,
            allow_nan = False,
            separators = (",", ":"),
            sort_keys = True,
        )
    except (TypeError, ValueError):
        raise UserAssetValidationError(
            "invalid_json",
            f"{field_name} must contain only finite JSON values",
        ) from None
    size = len(encoded.encode("utf-8"))
    if size > max_bytes:
        raise UserAssetValidationError(
            "size_limit_exceeded",
            f"{field_name} exceeds its {max_bytes}-byte canonical JSON limit",
        )
    return encoded


def _canonical_object(value: Any, max_bytes: int, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UserAssetValidationError(
            "invalid_json_object",
            f"{field_name} must be a JSON object",
        )
    encoded = canonical_json(value, max_bytes, field_name)
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise UserAssetValidationError("invalid_json_object", f"{field_name} must be a JSON object")
    return result


def _validate_secret_policy(
    value: Any, *, legacy: bool, max_bytes: int, field_name: str
) -> dict[str, Any] | tuple[dict[str, Any], list[str]]:
    if legacy:
        clean, redacted_paths = redact_secret_fields(value)
        return _canonical_object(clean, max_bytes, field_name), redacted_paths
    paths = find_secret_paths(value)
    if paths:
        raise UserAssetValidationError(
            "secret_fields_present",
            f"{field_name} contains secret fields",
            tuple(paths),
        )
    return _canonical_object(value, max_bytes, field_name)


@overload
def validate_recipe_payload(value: Any, legacy: bool = False) -> dict[str, Any]: ...


@overload
def validate_recipe_payload(
    value: Any, legacy: bool
) -> dict[str, Any] | tuple[dict[str, Any], list[str]]: ...


def validate_recipe_payload(
    value: Any, legacy: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], list[str]]:
    """Validate a recipe; legacy mode removes secrets and returns their paths."""

    return _validate_secret_policy(
        value,
        legacy = legacy,
        max_bytes = MAX_RECIPE_JSON_BYTES,
        field_name = "recipe payload",
    )


def validate_id(value: Any, field_name: str = "id") -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS:
        raise UserAssetValidationError(
            "invalid_id",
            f"{field_name} must be a non-empty string of at most {MAX_ID_CHARS} characters",
        )
    if "/" in value or "\\" in value:
        raise UserAssetValidationError(
            "invalid_id",
            f"{field_name} must be safe to use as one URL path segment",
        )
    return value


def validate_name(value: Any, field_name: str = "name") -> str:
    if not isinstance(value, str):
        raise UserAssetValidationError("invalid_name", f"{field_name} must be a string")
    trimmed = value.strip()
    if not trimmed or len(trimmed) > MAX_NAME_CHARS:
        raise UserAssetValidationError(
            "invalid_name",
            f"{field_name} must be non-empty and at most {MAX_NAME_CHARS} characters",
        )
    return trimmed


def validate_timestamp(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UserAssetValidationError(
            "invalid_timestamp",
            f"{field_name} must be a non-negative Unix epoch millisecond integer",
        )
    return value


def _require_string(
    value: Any,
    field: str,
    max_bytes: int,
    *,
    nullable: bool = False,
) -> Any:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise UserAssetValidationError("invalid_execution_metadata", f"{field} must be a string")
    if len(value.encode("utf-8")) > max_bytes:
        raise UserAssetValidationError("field_limit_exceeded", f"{field} exceeds its byte limit")
    return value


def _require_nonnegative_int(
    value: Any,
    field: str,
    *,
    nullable: bool = False,
) -> Any:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UserAssetValidationError(
            "invalid_execution_metadata", f"{field} must be a non-negative integer"
        )
    return value


def _require_object(
    value: Any,
    field: str,
    *,
    nullable: bool = True,
) -> Any:
    if value is None and nullable:
        return None
    if not isinstance(value, Mapping):
        raise UserAssetValidationError("invalid_execution_metadata", f"{field} must be an object")
    return value


def project_execution_metadata(value: Any) -> dict[str, Any]:
    """Allowlist durable execution metadata and omit UI-only fields."""

    if not isinstance(value, Mapping):
        raise UserAssetValidationError(
            "invalid_execution_metadata", "execution metadata must be a JSON object"
        )

    projected: dict[str, Any] = {}
    string_fields = {
        "jobId": (MAX_ID_CHARS, True),
        "kind": (32, False),
        "run_name": (MAX_NAME_CHARS, True),
        "status": (32, False),
        "recipeSignature": (4096, False),
        "stage": (MAX_NAME_CHARS, True),
        "current_column": (MAX_NAME_CHARS, True),
        "error": (MAX_EXECUTION_ERROR_BYTES, True),
    }
    integer_fields = {
        "rows": False,
        "lastEventId": True,
        "datasetTotal": False,
        "createdAt": False,
        "finishedAt": True,
    }
    object_fields = {
        "progress",
        "column_progress",
        "batch",
        "source_progress",
        "model_usage",
        "analysis",
    }

    for field, (limit, nullable) in string_fields.items():
        if field in value:
            projected[field] = _require_string(value[field], field, limit, nullable = nullable)
    for field, nullable in integer_fields.items():
        if field in value:
            projected[field] = _require_nonnegative_int(value[field], field, nullable = nullable)
    for field in object_fields:
        if field in value:
            projected[field] = _require_object(value[field], field)

    if "completed_columns" in value:
        columns = value["completed_columns"]
        if not isinstance(columns, list):
            raise UserAssetValidationError(
                "invalid_execution_metadata", "completed_columns must be an array"
            )
        if len(columns) > MAX_COMPLETED_COLUMNS:
            raise UserAssetValidationError(
                "field_limit_exceeded",
                f"completed_columns exceeds its {MAX_COMPLETED_COLUMNS}-entry limit",
            )
        projected["completed_columns"] = [
            _require_string(column, f"completed_columns[{index}]", MAX_NAME_CHARS)
            for index, column in enumerate(columns)
        ]

    paths = find_secret_paths(projected)
    if paths:
        raise UserAssetValidationError(
            "secret_fields_present",
            "execution metadata contains secret fields",
            tuple(paths),
        )
    return _canonical_object(projected, MAX_EXECUTION_JSON_BYTES, "execution metadata")


def validate_legacy_batch_size(recipes: Any, executions: Any) -> None:
    """Enforce legacy batch count and byte limits."""

    if not isinstance(recipes, list) or not isinstance(executions, list):
        raise UserAssetValidationError(
            "invalid_legacy_batch", "legacy recipes and executions must be arrays"
        )
    if len(recipes) > MAX_LEGACY_RECIPES or len(executions) > MAX_LEGACY_EXECUTIONS:
        raise UserAssetValidationError(
            "legacy_batch_limit_exceeded", "legacy batch item limit exceeded"
        )
    canonical_json(
        {"recipes": recipes, "executions": executions},
        MAX_LEGACY_BATCH_JSON_BYTES,
        "legacy batch",
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import structlog
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from loggers import get_logger
from utils.node_runtime import resolve_node_executable
from utils.paths import ensure_dir, oxc_validator_tmp_root

logger = get_logger(__name__)

OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator"
TOOL_VALIDATION_FN_MARKER = "unsloth_tool_validator"

_TOOL_FILE_EXT_RE = re.compile(r"^[A-Za-z0-9.+-]{1,20}$")
_TOOL_RUN_TIMEOUT_SECONDS = 60

_OXC_LANG_TO_NODE_LANG = {
    "javascript": "js",
    "typescript": "ts",
    "jsx": "jsx",
    "tsx": "tsx",
}
_OXC_VALIDATION_MODES = {"syntax", "lint", "syntax+lint"}
_OXC_CODE_SHAPES = {"auto", "module", "snippet"}

_OXC_TOOL_DIR = Path(__file__).resolve().parent / "oxc-validator"
_OXC_RUNNER_PATH = _OXC_TOOL_DIR / "validate.mjs"


from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)


@dataclass(frozen = True)
class OxcLocalCallableValidatorSpec:
    name: str
    drop: bool
    target_columns: list[str]
    batch_size: int
    code_lang: str
    validation_mode: str
    code_shape: str


@dataclass(frozen = True)
class ToolLocalCallableValidatorSpec:
    name: str
    drop: bool
    target_columns: list[str]
    batch_size: int
    file_ext: str
    command: str


def split_tool_local_callable_validators(
    recipe_core: dict[str, Any],
) -> tuple[dict[str, Any], list[ToolLocalCallableValidatorSpec]]:
    columns = recipe_core.get("columns")
    if not isinstance(columns, list):
        return recipe_core, []

    sanitized = deepcopy(recipe_core)
    sanitized_columns = sanitized.get("columns")
    if not isinstance(sanitized_columns, list):
        return sanitized, []

    kept_columns: list[Any] = []
    tool_specs: list[ToolLocalCallableValidatorSpec] = []

    for column in sanitized_columns:
        if not isinstance(column, dict):
            kept_columns.append(column)
            continue

        maybe_spec = _parse_tool_spec(column = column)
        if maybe_spec is None:
            kept_columns.append(column)
            continue
        tool_specs.append(maybe_spec)

    sanitized["columns"] = kept_columns
    return sanitized, tool_specs


def register_tool_local_callable_validators(
    *, builder, specs: list[ToolLocalCallableValidatorSpec]
) -> None:
    if not specs:
        return

    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import (
        LocalCallableValidatorParams,
        ValidatorType,
    )

    for spec in specs:
        validation_function = _build_tool_validation_function(
            spec.file_ext,
            spec.command,
        )
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


def split_oxc_local_callable_validators(
    recipe_core: dict[str, Any],
) -> tuple[dict[str, Any], list[OxcLocalCallableValidatorSpec]]:
    columns = recipe_core.get("columns")
    if not isinstance(columns, list):
        return recipe_core, []

    sanitized = deepcopy(recipe_core)
    sanitized_columns = sanitized.get("columns")
    if not isinstance(sanitized_columns, list):
        return sanitized, []

    kept_columns: list[Any] = []
    oxc_specs: list[OxcLocalCallableValidatorSpec] = []

    for column in sanitized_columns:
        if not isinstance(column, dict):
            kept_columns.append(column)
            continue

        maybe_spec = _parse_oxc_spec(column = column)
        if maybe_spec is None:
            kept_columns.append(column)
            continue
        oxc_specs.append(maybe_spec)

    sanitized["columns"] = kept_columns
    return sanitized, oxc_specs


def register_oxc_local_callable_validators(
    *, builder, specs: list[OxcLocalCallableValidatorSpec]
) -> None:
    if not specs:
        return

    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import (
        LocalCallableValidatorParams,
        ValidatorType,
    )

    for spec in specs:
        validation_function = _build_oxc_validation_function(
            spec.code_lang,
            spec.validation_mode,
            spec.code_shape,
        )
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


def _parse_oxc_spec(*, column: dict[str, Any]) -> OxcLocalCallableValidatorSpec | None:
    if str(column.get("column_type") or "").strip() != "validation":
        return None
    if str(column.get("validator_type") or "").strip() != "local_callable":
        return None

    params = column.get("validator_params")
    if not isinstance(params, dict):
        return None

    fn_raw = params.get("validation_function")
    fn_name = fn_raw.strip() if isinstance(fn_raw, str) else ""
    if not fn_name.startswith(OXC_VALIDATION_FN_MARKER):
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

    code_lang, validation_mode, code_shape = _parse_oxc_validation_marker(fn_name)
    batch_size = _parse_batch_size(column.get("batch_size"))
    drop = bool(column.get("drop") is True)

    return OxcLocalCallableValidatorSpec(
        name = name,
        drop = drop,
        target_columns = target_columns,
        batch_size = batch_size,
        code_lang = code_lang,
        validation_mode = validation_mode,
        code_shape = code_shape,
    )


def _parse_batch_size(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    return parsed if parsed >= 1 else 10


def _parse_oxc_validation_marker(fn_name: str) -> tuple[str, str, str]:
    marker = f"{OXC_VALIDATION_FN_MARKER}:"
    if not fn_name.startswith(marker):
        return "javascript", "syntax", "auto"
    suffix = fn_name[len(marker) :]
    parts = [part.strip() for part in suffix.split(":") if part.strip()]
    if len(parts) < 2:
        return "javascript", "syntax", "auto"
    code_lang = parts[0] if parts[0] in _OXC_LANG_TO_NODE_LANG else "javascript"
    mode = parts[1] if parts[1] in _OXC_VALIDATION_MODES else "syntax"
    code_shape = parts[2] if len(parts) >= 3 and parts[2] in _OXC_CODE_SHAPES else "auto"
    return code_lang, mode, code_shape


def _decode_base64url(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _parse_tool_spec(*, column: dict[str, Any]) -> ToolLocalCallableValidatorSpec | None:
    if str(column.get("column_type") or "").strip() != "validation":
        return None
    if str(column.get("validator_type") or "").strip() != "local_callable":
        return None

    params = column.get("validator_params")
    if not isinstance(params, dict):
        return None

    fn_raw = params.get("validation_function")
    fn_name = fn_raw.strip() if isinstance(fn_raw, str) else ""
    marker = f"{TOOL_VALIDATION_FN_MARKER}:"
    if not fn_name.startswith(marker):
        return None

    encoded = fn_name[len(marker) :].strip()
    if not encoded:
        return None
    decoded = _decode_base64url(encoded)
    try:
        spec = json.loads(decoded)
    except (TypeError, ValueError):
        return None
    if not isinstance(spec, dict):
        return None

    file_ext = str(spec.get("ext") or "").strip().lstrip(".")
    command = str(spec.get("command") or "").strip()
    if not file_ext or not _TOOL_FILE_EXT_RE.fullmatch(file_ext):
        return None
    if not command:
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

    return ToolLocalCallableValidatorSpec(
        name = name,
        drop = bool(column.get("drop") is True),
        target_columns = target_columns,
        batch_size = _parse_batch_size(column.get("batch_size")),
        file_ext = file_ext,
        command = command,
    )


@lru_cache(maxsize = 8)
def _build_tool_validation_function(file_ext: str, command: str):
    normalized_ext = file_ext
    normalized_command = command

    def _validator(df):
        import pandas as pd  # lazy import for local callable runtime

        row_count = int(len(df.index))
        if row_count == 0:
            return pd.DataFrame({"is_valid": []})

        code_column = str(df.columns[0]) if len(df.columns) > 0 else ""
        code_values = (
            ["" for _ in range(row_count)]
            if not code_column
            else ["" if value is None else str(value) for value in df[code_column].tolist()]
        )

        results = _run_tool_batch(
            file_ext = normalized_ext,
            command = normalized_command,
            code_values = code_values,
        )
        if len(results) != row_count:
            results = _fallback_results(
                row_count,
                "Tool validator returned mismatched result size.",
            )
        return pd.DataFrame(results)

    _validator.__name__ = f"{TOOL_VALIDATION_FN_MARKER}_{normalized_ext}"
    return _validator


def _run_tool_batch(*, file_ext: str, command: str, code_values: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for code_value in code_values:
        results.append(
            _run_tool_single(
                file_ext = file_ext,
                command = command,
                code_value = code_value,
            )
        )
    return results


def _run_tool_single(*, file_ext: str, command: str, code_value: str) -> dict[str, Any]:
    tmp_root = ensure_dir(oxc_validator_tmp_root())
    try:
        with tempfile.TemporaryDirectory(dir = str(tmp_root), prefix = "tool-") as raw_dir:
            run_dir = Path(raw_dir)
            if file_ext == "go":
                (run_dir / "go.mod").write_text(
                    "module example.com/check\n\ngo 1.21\n",
                    encoding = "utf-8",
                )
            source_path = run_dir / f"main.{file_ext}"
            source_path.write_text(code_value, encoding = "utf-8")

            substituted = command.replace("{file}", str(source_path)).replace("{dir}", str(run_dir))
            env = child_env_without_native_path_secret()
            proc = subprocess.run(
                substituted,
                cwd = str(run_dir),
                shell = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                capture_output = True,
                check = False,
                timeout = _TOOL_RUN_TIMEOUT_SECONDS,
                env = env,
                **_windows_hidden_subprocess_kwargs(),
            )
    except subprocess.TimeoutExpired:
        return _tool_result(
            is_valid = False,
            error_count = 1,
            error_message = f"Tool check timed out after {_TOOL_RUN_TIMEOUT_SECONDS}s.",
            tool_output = "",
        )
    except (OSError, ValueError) as exc:
        return _tool_result(
            is_valid = False,
            error_count = 1,
            error_message = f"Tool check launch failed: {exc}",
            tool_output = "",
        )

    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    output = output.strip()
    if proc.returncode == 0:
        return _tool_result(
            is_valid = True,
            error_count = 0,
            error_message = "",
            tool_output = output,
        )
    message = output or "Tool check failed."
    if len(message) > 300:
        message = f"{message[:300]}..."
    return _tool_result(
        is_valid = False,
        error_count = 1,
        error_message = message,
        tool_output = output,
    )


def _tool_result(
    *, is_valid: bool, error_count: int, error_message: str, tool_output: str
) -> dict[str, Any]:
    return {
        "is_valid": bool(is_valid),
        "error_count": int(error_count),
        "error_message": str(error_message),
        "severity": None,
        "code": None,
        "labels": [],
        "codeframe": None,
        "warning_count": 0,
        "tool_output": str(tool_output),
    }


@lru_cache(maxsize = 8)
def _build_oxc_validation_function(lang: str, validation_mode: str, code_shape: str):
    node_lang = _OXC_LANG_TO_NODE_LANG.get(lang, "js")
    mode = validation_mode if validation_mode in _OXC_VALIDATION_MODES else "syntax"
    normalized_code_shape = code_shape if code_shape in _OXC_CODE_SHAPES else "auto"

    def _validator(df):
        import pandas as pd  # lazy import for local callable runtime

        row_count = int(len(df.index))
        if row_count == 0:
            return pd.DataFrame({"is_valid": []})

        code_column = str(df.columns[0]) if len(df.columns) > 0 else ""
        code_values = (
            ["" for _ in range(row_count)]
            if not code_column
            else ["" if value is None else str(value) for value in df[code_column].tolist()]
        )

        results = _run_oxc_batch(
            node_lang = node_lang,
            validation_mode = mode,
            code_shape = normalized_code_shape,
            code_values = code_values,
        )
        if len(results) != row_count:
            results = _fallback_results(
                row_count,
                "OXC validator returned mismatched result size.",
            )
        return pd.DataFrame(results)

    _validator.__name__ = (
        f"{OXC_VALIDATION_FN_MARKER}_{node_lang}_{mode.replace('+', '_')}_{normalized_code_shape}"
    )
    return _validator


def _run_oxc_batch(
    *, node_lang: str, validation_mode: str, code_shape: str, code_values: list[str]
) -> list[dict[str, Any]]:
    if not _OXC_RUNNER_PATH.exists():
        return _fallback_results(
            len(code_values),
            f"OXC runner missing at {_OXC_RUNNER_PATH}",
        )

    payload = {
        "lang": node_lang,
        "mode": validation_mode,
        "code_shape": code_shape,
        "codes": code_values,
    }
    # Resolve a usable Node (system or the isolated install, which is not on the
    # user's PATH); a bare "node" would fail for isolated-Node users.
    node_executable = resolve_node_executable()
    if not node_executable:
        return _fallback_results(
            len(code_values),
            "Node.js not found (install Node >= 20.19, or re-run Unsloth setup to provision it).",
        )
    try:
        tmp_dir = ensure_dir(oxc_validator_tmp_root())
        env = child_env_without_native_path_secret()
        tmp_dir_str = str(tmp_dir)
        env["TMPDIR"] = tmp_dir_str
        env["TMP"] = tmp_dir_str
        env["TEMP"] = tmp_dir_str
        # Resolved node's dir first on the child PATH so it finds its own npm/npx.
        node_bin_dir = os.path.dirname(node_executable)
        if node_bin_dir:
            env["PATH"] = node_bin_dir + os.pathsep + env.get("PATH", "")
        env.pop("NODE_PATH", None)
        proc = subprocess.run(
            [node_executable, str(_OXC_RUNNER_PATH)],
            cwd = str(_OXC_TOOL_DIR),
            input = json.dumps(payload),
            text = True,
            encoding = "utf-8",
            errors = "replace",
            capture_output = True,
            check = False,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, ValueError) as exc:
        logger.warning("OXC subprocess launch failed: %s", exc)
        return _fallback_results(len(code_values), f"OXC launch failed: {exc}")

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "unknown error").strip()
        if len(message) > 300:
            message = f"{message[:300]}..."
        return _fallback_results(len(code_values), f"OXC failed: {message}")

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _fallback_results(len(code_values), "OXC output parse failed.")

    if not isinstance(raw, list):
        return _fallback_results(len(code_values), "OXC output must be an array.")

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            out.append(
                {
                    "is_valid": False,
                    "error_count": 1,
                    "error_message": "Invalid OXC result entry.",
                    "severity": None,
                    "code": None,
                    "labels": [],
                    "codeframe": None,
                    "warning_count": 0,
                }
            )
            continue
        is_valid_raw = item.get("is_valid")
        error_count_raw = item.get("error_count")
        message_raw = item.get("error_message")
        severity_raw = item.get("severity")
        code_raw = item.get("code")
        labels_raw = item.get("labels")
        codeframe_raw = item.get("codeframe")
        warning_count_raw = item.get("warning_count")
        out.append(
            {
                "is_valid": bool(is_valid_raw) if isinstance(is_valid_raw, bool) else False,
                "error_count": int(error_count_raw) if isinstance(error_count_raw, int) else 0,
                "error_message": str(message_raw or ""),
                "severity": str(severity_raw) if isinstance(severity_raw, str) else None,
                "code": str(code_raw) if isinstance(code_raw, str) else None,
                "labels": labels_raw if isinstance(labels_raw, list) else [],
                "codeframe": str(codeframe_raw) if isinstance(codeframe_raw, str) else None,
                "warning_count": int(warning_count_raw)
                if isinstance(warning_count_raw, int)
                else 0,
            }
        )
    return out


def _fallback_results(row_count: int, message: str) -> list[dict[str, Any]]:
    return [
        {
            "is_valid": False,
            "error_count": 1,
            "error_message": message,
            "severity": None,
            "code": None,
            "labels": [],
            "codeframe": None,
            "warning_count": 0,
        }
        for _ in range(row_count)
    ]

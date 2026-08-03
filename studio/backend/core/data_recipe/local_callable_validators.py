# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import signal
import structlog
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Thread
from typing import Any

from loggers import get_logger
from utils.node_runtime import resolve_node_executable
from utils.paths import ensure_dir, oxc_validator_tmp_root

logger = get_logger(__name__)

OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator"
TOOL_VALIDATION_FN_MARKER = "unsloth_tool_validator"

_TOOL_FILE_EXT_RE = re.compile(r"^[A-Za-z0-9.+-]{1,20}$")
_TOOL_SCAFFOLD_PATH_RE = re.compile(r"^[A-Za-z0-9._+-]+(?:/[A-Za-z0-9._+-]+)*$")
_TOOL_SCAFFOLD_MAX_ROWS = 10
_TOOL_SCAFFOLD_MAX_TOTAL_CHARS = 32 * 1024
_TOOL_RUN_TIMEOUT_SECONDS = 60
# Fixed in-memory capture guard per stream; the value stored per row is the
# user-configurable _TOOL_OUTPUT_MAX_CHARS_DEFAULT (clamped to this ceiling).
_TOOL_OUTPUT_CAPTURE_MAX_CHARS = 256 * 1024
_TOOL_OUTPUT_MAX_CHARS_DEFAULT = 8 * 1024
_TOOL_OUTPUT_MAX_CHARS_MIN = 1 * 1024
_TOOL_OUTPUT_MAX_CHARS_MAX = 256 * 1024
_TOOL_SOURCE_FILE_MAX_CHARS_DEFAULT = 32 * 1024
_TOOL_SOURCE_FILE_MAX_CHARS_MIN = 1 * 1024
_TOOL_SOURCE_FILE_MAX_CHARS_MAX = 64 * 1024
_TOOL_COMMAND_MAX_CHARS = 8 * 1024
_TOOL_MARKER_MAX_CHARS = 128 * 1024
BATCH_SIZE_MAX = 512

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
    scaffold: tuple[tuple[str, str], ...] = ()
    output_max_chars: int = _TOOL_OUTPUT_MAX_CHARS_DEFAULT
    source_file_max_chars: int = _TOOL_SOURCE_FILE_MAX_CHARS_DEFAULT


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
            spec.scaffold,
            spec.output_max_chars,
            spec.source_file_max_chars,
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
    if parsed < 1:
        return 10
    return min(parsed, BATCH_SIZE_MAX)


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
    # Bound the decode input so a crafted marker cannot run padding math or
    # base64 work on an arbitrarily large string.
    if len(value) > _TOOL_MARKER_MAX_CHARS:
        return ""
    padded = value + "=" * (-len(value) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError):
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _parse_tool_char_cap(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


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
    if not encoded or len(encoded) > _TOOL_MARKER_MAX_CHARS:
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
    if len(command) > _TOOL_COMMAND_MAX_CHARS:
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

    scaffold = _parse_tool_scaffold(spec)
    if scaffold is None:
        return None

    return ToolLocalCallableValidatorSpec(
        name = name,
        drop = bool(column.get("drop") is True),
        target_columns = target_columns,
        batch_size = _parse_batch_size(column.get("batch_size")),
        file_ext = file_ext,
        command = command,
        scaffold = scaffold,
        output_max_chars = _parse_tool_char_cap(
            spec.get("output_max_chars"),
            default = _TOOL_OUTPUT_MAX_CHARS_DEFAULT,
            minimum = _TOOL_OUTPUT_MAX_CHARS_MIN,
            maximum = _TOOL_OUTPUT_MAX_CHARS_MAX,
        ),
        source_file_max_chars = _parse_tool_char_cap(
            spec.get("source_file_max_chars"),
            default = _TOOL_SOURCE_FILE_MAX_CHARS_DEFAULT,
            minimum = _TOOL_SOURCE_FILE_MAX_CHARS_MIN,
            maximum = _TOOL_SOURCE_FILE_MAX_CHARS_MAX,
        ),
    )


def _parse_tool_scaffold(spec: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    """Parse and validate the optional ``scaffold`` rows from a tool spec.

    Returns an empty tuple when no scaffold is present, a tuple of
    ``(path, content)`` rows when valid, or None when the scaffold is present
    but malformed (the column is then rejected like any other bad marker).
    """
    raw = spec.get("scaffold")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        return None

    rows: list[tuple[str, str]] = []
    total_chars = 0
    for entry in raw:
        if not isinstance(entry, dict):
            return None
        path = str(entry.get("path") or "").strip()
        content = entry.get("content")
        if not isinstance(content, str):
            return None
        if not path:
            continue
        if not _TOOL_SCAFFOLD_PATH_RE.fullmatch(path):
            return None
        if any(segment in (".", "..") for segment in path.split("/")):
            return None
        rows.append((path, content))
        total_chars += len(path) + len(content)

    if len(rows) > _TOOL_SCAFFOLD_MAX_ROWS:
        return None
    if total_chars > _TOOL_SCAFFOLD_MAX_TOTAL_CHARS:
        return None
    return tuple(rows)


@lru_cache(maxsize = 8)
def _build_tool_validation_function(
    file_ext: str,
    command: str,
    scaffold: tuple[tuple[str, str], ...] = (),
    output_max_chars: int = _TOOL_OUTPUT_MAX_CHARS_DEFAULT,
    source_file_max_chars: int = _TOOL_SOURCE_FILE_MAX_CHARS_DEFAULT,
):
    normalized_ext = file_ext
    normalized_command = command
    normalized_scaffold = scaffold
    normalized_output_max_chars = output_max_chars
    normalized_source_file_max_chars = source_file_max_chars

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
            scaffold = normalized_scaffold,
            code_values = code_values,
            output_max_chars = normalized_output_max_chars,
            source_file_max_chars = normalized_source_file_max_chars,
        )
        if len(results) != row_count:
            results = _fallback_results(
                row_count,
                "Tool validator returned mismatched result size.",
            )
        return pd.DataFrame(results)

    _validator.__name__ = f"{TOOL_VALIDATION_FN_MARKER}_{normalized_ext}"
    return _validator


def _run_tool_batch(
    *,
    file_ext: str,
    command: str,
    scaffold: tuple[tuple[str, str], ...],
    code_values: list[str],
    max_workers: int | None = None,
    output_max_chars: int = _TOOL_OUTPUT_MAX_CHARS_DEFAULT,
    source_file_max_chars: int = _TOOL_SOURCE_FILE_MAX_CHARS_DEFAULT,
) -> list[dict[str, Any]]:
    """Run the tool command for every code cell in a batch.

    Cells are checked in parallel, up to the batch size (rows per
    invocation) and the detected CPU count; ``max_workers`` overrides the
    CPU cap for tests. Each cell runs in its own unique temp dir, so
    concurrent runs never share files. Results keep the input row order.
    """
    if not code_values:
        return []
    worker_count = _tool_max_workers() if max_workers is None else max(1, max_workers)
    worker_count = min(worker_count, len(code_values))
    if worker_count <= 1:
        return [
            _run_tool_single(
                file_ext = file_ext,
                command = command,
                scaffold = scaffold,
                code_value = code_value,
                output_max_chars = output_max_chars,
                source_file_max_chars = source_file_max_chars,
            )
            for code_value in code_values
        ]
    with ThreadPoolExecutor(max_workers = worker_count) as executor:
        return list(
            executor.map(
                lambda code_value: _run_tool_single(
                    file_ext = file_ext,
                    command = command,
                    scaffold = scaffold,
                    code_value = code_value,
                    output_max_chars = output_max_chars,
                    source_file_max_chars = source_file_max_chars,
                ),
                code_values,
            )
        )


def _tool_max_workers() -> int:
    """Detected CPU count (floor of 1), used to cap parallel tool checks."""
    return max(1, os.cpu_count() or 1)


def _tool_launch_kwargs() -> dict[str, Any]:
    """Subprocess kwargs that run the tool command in its own process group.

    The command runs as the leader of a new session (POSIX) or a new process
    group (Windows), so a timeout can kill the entire tree instead of leaving
    orphaned children behind.
    """
    kwargs: dict[str, Any] = _windows_hidden_subprocess_kwargs()
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | create_new_process_group
    return kwargs


def _terminate_tool_process_tree(proc: subprocess.Popen[str]) -> None:
    """Best-effort kill of the tool command and every process it spawned."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (OSError, ValueError):
            pass
    else:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output = True,
                check = False,
                timeout = 15,
            )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    try:
        proc.kill()
    except (OSError, ValueError):
        pass


class _CappedOutputReader(Thread):
    """Read a text stream in chunks, keeping at most ``limit`` characters.

    Reading stops at the cap; if the child keeps writing after that its pipe
    fills and it blocks until the run timeout kills it, bounding memory
    instead of buffering unbounded output for every row in a batch.
    ``truncated`` is set when the stream still had output left at the cap.
    """

    def __init__(self, stream, limit: int) -> None:
        super().__init__(daemon = True)
        self._stream = stream
        self._limit = limit
        self.output = ""
        self.truncated = False

    def run(self) -> None:
        chunks: list[str] = []
        total = 0
        stream = self._stream
        while total < self._limit:
            remaining = self._limit - total
            chunk = stream.read(remaining + 1)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if len(chunk) > remaining:
                self.truncated = True
                break
        self.output = "".join(chunks)


def _run_tool_single(
    *,
    file_ext: str,
    command: str,
    scaffold: tuple[tuple[str, str], ...],
    code_value: str,
    output_max_chars: int = _TOOL_OUTPUT_MAX_CHARS_DEFAULT,
    source_file_max_chars: int = _TOOL_SOURCE_FILE_MAX_CHARS_DEFAULT,
) -> dict[str, Any]:
    tmp_root = ensure_dir(oxc_validator_tmp_root())
    try:
        with tempfile.TemporaryDirectory(dir = str(tmp_root), prefix = "tool-") as raw_dir:
            run_dir = Path(raw_dir)
            source_path: Path | None = None
            for path, content in scaffold:
                target = run_dir / path
                resolved = target.resolve()
                if not resolved.is_relative_to(run_dir.resolve()):
                    return _tool_result(
                        is_valid = False,
                        error_count = 1,
                        error_message = "Tool check scaffold path escapes the temp folder.",
                        tool_output = "",
                    )
                target.parent.mkdir(parents = True, exist_ok = True)
                if "{source}" in content:
                    content = content.replace("{source}", code_value)
                    if source_path is None:
                        source_path = target
                # The pre-substitution scaffold is bounded; a large generated
                # cell must not be able to expand a file past the same bound.
                if len(content) > source_file_max_chars:
                    return _tool_result(
                        is_valid = False,
                        error_count = 1,
                        error_message = (
                            "Generated code exceeds the "
                            f"{source_file_max_chars // 1024} KiB scaffold file limit."
                        ),
                        tool_output = "",
                    )
                target.write_text(content, encoding = "utf-8")
            if source_path is None:
                source_path = run_dir / f"main.{file_ext}"
                if len(code_value) > source_file_max_chars:
                    return _tool_result(
                        is_valid = False,
                        error_count = 1,
                        error_message = (
                            "Generated code exceeds the "
                            f"{source_file_max_chars // 1024} KiB file limit."
                        ),
                        tool_output = "",
                    )
                source_path.write_text(code_value, encoding = "utf-8")

            substituted = command.replace("{file}", str(source_path)).replace("{dir}", str(run_dir))
            env = child_env_without_native_path_secret()
            proc = subprocess.Popen(
                substituted,
                cwd = str(run_dir),
                shell = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                env = env,
                **_tool_launch_kwargs(),
            )
            out_reader = _CappedOutputReader(proc.stdout, _TOOL_OUTPUT_CAPTURE_MAX_CHARS)
            err_reader = _CappedOutputReader(proc.stderr, _TOOL_OUTPUT_CAPTURE_MAX_CHARS)
            out_reader.start()
            err_reader.start()
            timed_out = False
            try:
                returncode = proc.wait(timeout = _TOOL_RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_tool_process_tree(proc)
                # Reap the (now killed) command so it does not linger as a zombie.
                try:
                    proc.wait(timeout = 10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            out_reader.join(timeout = 10)
            err_reader.join(timeout = 10)
            stdout = out_reader.output
            stderr = err_reader.output
            if timed_out:
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

    output = (stdout or "") + (("\n" + stderr) if stderr else "")
    output = output.strip()
    output_truncated = out_reader.truncated or err_reader.truncated
    if returncode == 0:
        return _tool_result(
            is_valid = True,
            error_count = 0,
            error_message = "",
            tool_output = output,
            output_max_chars = output_max_chars,
            output_truncated = output_truncated,
        )
    message = output or "Tool check failed."
    if len(message) > 300:
        message = f"{message[:300]}..."
    return _tool_result(
        is_valid = False,
        error_count = 1,
        error_message = message,
        tool_output = output,
        output_max_chars = output_max_chars,
        output_truncated = output_truncated,
    )


def _tool_result(
    *,
    is_valid: bool,
    error_count: int,
    error_message: str,
    tool_output: str,
    output_max_chars: int = _TOOL_OUTPUT_MAX_CHARS_DEFAULT,
    output_truncated: bool = False,
) -> dict[str, Any]:
    stored_output = str(tool_output)
    truncated = bool(output_truncated)
    if len(stored_output) > output_max_chars:
        stored_output = stored_output[:output_max_chars]
        truncated = True
    result = {
        "is_valid": bool(is_valid),
        "error_count": int(error_count),
        "error_message": str(error_message),
        "severity": None,
        "code": None,
        "labels": [],
        "codeframe": None,
        "warning_count": 0,
        "tool_output": stored_output,
    }
    if truncated:
        result["tool_output_truncated"] = True
        result["tool_output_message"] = f"Tool output truncated at {output_max_chars} characters."
    return result


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

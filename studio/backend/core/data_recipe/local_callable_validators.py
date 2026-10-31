# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import queue
import structlog
import subprocess
import threading
import time
from contextlib import contextmanager
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
_OXC_TIMEOUT_S = 30


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


@lru_cache(maxsize = 8)
def _build_oxc_validation_function(lang: str, validation_mode: str, code_shape: str):
    node_lang = _OXC_LANG_TO_NODE_LANG.get(lang, "js")
    mode = validation_mode if validation_mode in _OXC_VALIDATION_MODES else "syntax"
    normalized_code_shape = code_shape if code_shape in _OXC_CODE_SHAPES else "auto"

    def _validator(df):
        import pandas as pd

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


# One ``validate.mjs --serve`` per job, not per batch: Node startup was ~99% of a syntax batch.
_OXC_WORKER_MAX_LINE_BYTES = 64 * 1024 * 1024
_OXC_WORKER_MAX_FAILURES = 3
_OXC_WORKER_ENV = "UNSLOTH_OXC_PERSISTENT_WORKER"
_WORKER_UNUSED = object()
_WORKER_TIMED_OUT = object()


class _OxcWorkerFailed(Exception):
    pass


class _OxcWorker:
    def __init__(self, node_executable: str, env: dict[str, str], key: Any) -> None:
        self.key = key
        self._next_id = 0
        self._replies: queue.Queue = queue.Queue()
        self.proc = subprocess.Popen(
            [node_executable, str(_OXC_RUNNER_PATH), "--serve"],
            cwd = str(_OXC_TOOL_DIR),
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
        self._reader = threading.Thread(
            target = self._read_replies, name = "oxc-validator-reader", daemon = True
        )
        self._reader.start()

    def _read_replies(self) -> None:
        stdout = self.proc.stdout
        try:
            while True:
                line = stdout.readline(_OXC_WORKER_MAX_LINE_BYTES + 1)
                if not line.endswith(b"\n"):
                    break
                self._replies.put(line)
        except (OSError, ValueError):
            pass
        self._replies.put(None)

    def _write_request(self, line: bytes, errors: list[BaseException]) -> None:
        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            errors.append(exc)

    def request(self, request_text: str) -> Any:
        self._next_id += 1
        request_id = self._next_id
        line = f'{{"id": {request_id}, "payload": {request_text}}}\n'.encode("utf-8")
        # Bounded by the deadline: a Node that never reads would hold the session lock forever.
        deadline = time.monotonic() + _OXC_TIMEOUT_S
        write_errors: list[BaseException] = []
        writer = threading.Thread(
            target = self._write_request,
            args = (line, write_errors),
            name = "oxc-validator-writer",
            daemon = True,
        )
        writer.start()
        writer.join(_OXC_TIMEOUT_S)
        if writer.is_alive():
            return _WORKER_TIMED_OUT
        if write_errors:
            raise write_errors[0]
        try:
            reply = self._replies.get(timeout = max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            return _WORKER_TIMED_OUT
        if reply is None:
            raise _OxcWorkerFailed("OXC worker exited")
        message = json.loads(reply.decode("utf-8", errors = "replace"))
        if (
            not isinstance(message, dict)
            or message.get("id") != request_id
            or "result" not in message
        ):
            raise _OxcWorkerFailed("OXC worker could not answer")
        return message["result"]

    def close(self) -> None:
        proc = self.proc
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout = 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        self._reader.join(timeout = 2)
        if proc.stdout is not None and not self._reader.is_alive():
            proc.stdout.close()

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.close()


class _OxcWorkerSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: _OxcWorker | None = None
        self._failures = 0
        self._closed = False

    def run(self, node_executable: str, env: dict[str, str], request_text: str) -> Any:
        if (
            self._closed
            or self._failures >= _OXC_WORKER_MAX_FAILURES
            or len(request_text) > _OXC_WORKER_MAX_LINE_BYTES // 2
        ):
            return _WORKER_UNUSED
        # Concurrent callers run one-shot instead of queueing.
        if not self._lock.acquire(blocking = False):
            return _WORKER_UNUSED
        try:
            if self._closed:
                return _WORKER_UNUSED
            # The env carries the account's temp dir.
            key = (node_executable, tuple(sorted(env.items())))
            if self._worker is not None and self._worker.key != key:
                self._drop_worker(kill = False)
            if self._worker is None:
                self._worker = _OxcWorker(node_executable, env, key)
            result = self._worker.request(request_text)
            if result is _WORKER_TIMED_OUT:
                self._drop_worker(kill = True)
                return _WORKER_TIMED_OUT
            self._failures = 0
            return result
        except (OSError, ValueError, _OxcWorkerFailed) as exc:
            self._failures += 1
            logger.debug("OXC worker unavailable, running the batch one-shot: %s", exc)
            self._drop_worker(kill = True)
            return _WORKER_UNUSED
        finally:
            self._lock.release()

    def _drop_worker(self, *, kill: bool) -> None:
        worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            worker.kill() if kill else worker.close()
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._drop_worker(kill = False)


_OXC_SESSION: _OxcWorkerSession | None = None


@contextmanager
def oxc_validator_session():
    """OXC batches inside this block share one Node process (off: UNSLOTH_OXC_PERSISTENT_WORKER=0).

    A killed caller closes the worker's stdin, which ends it on every platform.
    """
    global _OXC_SESSION
    if os.environ.get(_OXC_WORKER_ENV, "1").strip() == "0":
        yield
        return
    session = _OxcWorkerSession()
    previous, _OXC_SESSION = _OXC_SESSION, session
    try:
        yield
    finally:
        _OXC_SESSION = previous
        session.close()


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
        # The wrapper bounds oxlint with this; a kill here would leave the grandchild.
        "timeout_ms": int(_OXC_TIMEOUT_S * 1000),
    }
    # Resolve a usable Node: a bare "node" fails for isolated-Node users, whose install is not on PATH.
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
        request_text = json.dumps(payload)
        session = _OXC_SESSION
        raw = _WORKER_UNUSED if session is None else session.run(node_executable, env, request_text)
        if raw is _WORKER_TIMED_OUT:
            raise subprocess.TimeoutExpired(cmd = node_executable, timeout = _OXC_TIMEOUT_S)
        if raw is _WORKER_UNUSED:
            raw = _run_oxc_one_shot(node_executable, env, request_text)
    except subprocess.TimeoutExpired:
        logger.warning("OXC validation timed out after %ss", _OXC_TIMEOUT_S)
        return _fallback_results(len(code_values), "OXC validation timed out")
    except (OSError, ValueError) as exc:
        logger.warning("OXC subprocess launch failed: %s", exc)
        return _fallback_results(len(code_values), f"OXC launch failed: {exc}")

    if isinstance(raw, _OneShotFailure):
        return _fallback_results(len(code_values), raw.message)
    return _normalize_oxc_results(raw, len(code_values))


class _OneShotFailure:
    def __init__(self, message: str) -> None:
        self.message = message


def _run_oxc_one_shot(node_executable: str, env: dict[str, str], request_text: str) -> Any:
    proc = subprocess.run(
        [node_executable, str(_OXC_RUNNER_PATH)],
        cwd = str(_OXC_TOOL_DIR),
        input = request_text,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        capture_output = True,
        check = False,
        env = env,
        timeout = _OXC_TIMEOUT_S,
        **_windows_hidden_subprocess_kwargs(),
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "unknown error").strip()
        if len(message) > 300:
            message = f"{message[:300]}..."
        return _OneShotFailure(f"OXC failed: {message}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _OneShotFailure("OXC output parse failed.")


def _normalize_oxc_results(raw: Any, row_count: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return _fallback_results(row_count, "OXC output must be an array.")

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

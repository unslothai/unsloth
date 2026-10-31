# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Studio HTTP client and polling predicates; callers must scrub the token and password from output."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# The field is `phase`, not `status`; only `completed` means the adapter exists.
TRAINING_TERMINAL = frozenset({"completed", "error", "stopped"})
TRAINING_OK = "completed"

EXPORT_OK = "success"

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")

MIN_ADAPTER_BYTES = 4096


class StudioError(RuntimeError):
    """An HTTP call to Unsloth that did not do what the payload needed."""


class Studio:
    """Bearer-authenticated JSON calls against a local Unsloth."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None
        self.password: str | None = None

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout: float | None = None,
        auth: bool = True,
    ) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data = data, headers = headers, method = method)
        try:
            with urllib.request.urlopen(req, timeout = timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors = "replace")
                status = resp.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors = "replace")
            status = exc.code
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, raw

    def get(self, path: str, **kw) -> tuple[int, Any]:
        return self.request("GET", path, None, **kw)

    def post(
        self,
        path: str,
        body: dict | None = None,
        **kw,
    ) -> tuple[int, Any]:
        return self.request("POST", path, body, **kw)

    def expect(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        **kw,
    ) -> Any:
        status, payload = self.request(method, path, body, **kw)
        if status != 200:
            detail = payload if isinstance(payload, str) else json.dumps(payload)[:600]
            raise StudioError(f"{method} {path} -> HTTP {status}: {detail}")
        return payload

    def login(
        self,
        password: str,
        *,
        username: str = "unsloth",
    ) -> None:
        """Retires the bootstrap password, else other routes 403; the new one is kept on self.password."""
        status, payload = self.post(
            "/api/auth/login",
            {"username": username, "password": password},
            auth = False,
        )
        if status != 200 or not isinstance(payload, dict):
            raise StudioError(f"login failed with HTTP {status}")
        token = payload.get("access_token")
        if not token:
            raise StudioError("login returned no access_token")
        self.token = str(token)
        self.password = password
        if payload.get("must_change_password"):
            self._retire_bootstrap_password(password)

    def _retire_bootstrap_password(self, current_password: str) -> None:
        """Replace the bootstrap password so the session can reach the real routes."""
        import secrets

        # token_urlsafe never yields whitespace, which change-password rejects.
        replacement = secrets.token_urlsafe(24)
        status, payload = self.post(
            "/api/auth/change-password",
            {"current_password": current_password, "new_password": replacement},
        )
        if status != 200 or not isinstance(payload, dict):
            raise StudioError(f"forced password change failed with HTTP {status}")
        token = payload.get("access_token")
        if not token:
            raise StudioError("forced password change returned no access_token")
        self.token = str(token)
        self.password = replacement


def health_is_ready(payload: Any) -> bool:
    """Status healthy is not enough: during hardware detection Unsloth refuses train and export."""
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "healthy":
        return False
    if payload.get("hardware_detecting"):
        return False
    return True


def wait_for(
    probe: Callable[[], Any],
    accept: Callable[[Any], bool],
    *,
    deadline_s: float,
    interval_s: float = 2.0,
    alive: Callable[[], bool] | None = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, Any, str]:
    """Poll probe until accept, the deadline, or a dead process, so a crash is not reported as slow."""
    started = now()
    last: Any = None
    while True:
        if alive is not None and not alive():
            return False, last, "the process being waited on exited"
        try:
            last = probe()
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        else:
            if accept(last):
                return True, last, ""
        elapsed = now() - started
        if elapsed >= deadline_s:
            return False, last, f"timed out after {elapsed:.0f}s (deadline {deadline_s:.0f}s)"
        sleep(interval_s)


def training_verdict(status: Any) -> tuple[bool, str]:
    """A run that ends in a non-completed phase is terminal with its reason, so polling stops at once."""
    if not isinstance(status, dict):
        return False, ""
    phase = status.get("phase")
    if phase not in TRAINING_TERMINAL:
        return False, ""
    if phase == TRAINING_OK:
        return True, ""
    error = status.get("error") or status.get("message") or ""
    return True, f"training ended in phase {phase!r}: {error}"


def export_verdict(status: Any, baseline_seq: int) -> tuple[bool, str]:
    """Judge the export by last_op_seq moving past baseline_seq; is_export_active alone is ambiguous."""
    if not isinstance(status, dict):
        return False, ""
    seq = status.get("last_op_seq")
    if not isinstance(seq, int) or seq <= baseline_seq:
        return False, ""
    if status.get("is_export_active"):
        return False, ""
    result = status.get("last_op_status")
    if result == EXPORT_OK:
        return True, ""
    error = status.get("last_op_error") or ""
    return True, f"export ended with last_op_status={result!r}: {error}"


def adapter_verdict(output_dir: str | Path | None) -> tuple[bool, list[str], dict]:
    """Check the LoRA adapter files on disk; a completed status is only the worker's own bookkeeping."""
    detail: dict = {"output_dir": str(output_dir) if output_dir else None}
    if not output_dir:
        return False, ["training reported no output_dir, so nothing can be checked"], detail

    root = Path(output_dir)
    if not root.is_dir():
        return False, [f"training output_dir does not exist: {root}"], detail

    failures: list[str] = []
    config = root / ADAPTER_CONFIG
    detail["adapter_config_present"] = config.is_file()
    if not config.is_file():
        failures.append(f"no {ADAPTER_CONFIG} in {root}")

    weights = None
    for name in ADAPTER_WEIGHTS:
        candidate = root / name
        if candidate.is_file():
            weights = candidate
            break
    if weights is None:
        failures.append(f"no adapter weights ({' or '.join(ADAPTER_WEIGHTS)}) in {root}")
    else:
        size = weights.stat().st_size
        detail["adapter_weights"] = weights.name
        detail["adapter_bytes"] = size
        if size < MIN_ADAPTER_BYTES:
            failures.append(
                f"{weights.name} is {size} bytes, below the {MIN_ADAPTER_BYTES}-byte floor, "
                f"so the save wrote a stub rather than an adapter"
            )

    return not failures, failures, detail


def _loss_values(status: Any) -> list:
    if not isinstance(status, dict):
        return []
    history = status.get("metric_history")
    if not isinstance(history, dict):
        return []
    losses = history.get("loss")
    return losses if isinstance(losses, list) else []


def _is_finite_loss(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def trained_steps(status: Any) -> int:
    """Count finite logged losses only: a diverged fp16 run logs NaN or inf yet still reaches completed."""
    return len([value for value in _loss_values(status) if _is_finite_loss(value)])


def nonfinite_losses(status: Any) -> list:
    """The logged losses that are NaN, infinite, or not a number at all."""
    return [
        value for value in _loss_values(status) if not _is_finite_loss(value) and value is not None
    ]


def newest_gguf(root: str | Path) -> Path | None:
    """Excludes mmproj sidecars: llama.cpp accepts a projector as a model but offloads nothing to
    the GPU."""
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = sorted(
        (p for p in root.rglob("*.gguf") if "mmproj" not in p.name.lower()),
        key = lambda p: p.stat().st_mtime,
        reverse = True,
    )
    return candidates[0] if candidates else None

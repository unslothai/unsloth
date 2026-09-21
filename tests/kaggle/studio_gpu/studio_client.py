# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A small HTTP client for Unsloth, and the state machines the payload polls.

Split out of ``run_studio_gpu.py`` so the parts that can be wrong in
interesting ways can be tested without a GPU, a browser or a server. The
polling predicates in particular are where a green result gets fabricated:
"the training job finished" and "the training job never started" produce the
same ``phase`` for the first few seconds, and an export that failed reports
the same ``is_export_active: false`` as one that succeeded.

Nothing here prints. ``Studio.token`` is set from the bootstrap password and is
never logged, echoed or written to a report, and neither is ``Studio.password``
-- which IS held for the run, because Unsloth forces a password change on the
bootstrap account and the repo's Playwright driver needs whatever the current
password is. Scrubbing it out of anything that leaves the machine is the
caller's job.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# Terminal phases of GET /api/train/status.
# From studio/backend/models/training.py: the field is `phase`, not `status`, and `completed` is the only one of the
# three that means the adapter exists.
TRAINING_TERMINAL = frozenset({"completed", "error", "stopped"})
TRAINING_OK = "completed"

# Terminal values of GET /api/export/status last_op_status.
EXPORT_OK = "success"

# What a saved PEFT adapter directory has to contain before this payload will call a training run complete.
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")

# A LoRA adapter for the smallest model this payload will ever train is still tens of MiB.
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
        # The password this session is currently authenticated by. See login().
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
        """Exchange the bootstrap password for a bearer token, retiring it if Unsloth insists.

        A bootstrap account is created with ``must_change_password`` set, and
        ``get_current_subject`` turns that into

            HTTP 403 {"detail": "Password change required"}

        on every route except the password-change one. Logging in therefore
        yields a token that authenticates and can do nothing: the first
        hardware run of this payload reached ``POST /api/inference/load`` and
        ``POST /api/train/start`` and got 403 from both, so inference, tool
        calling, training and export were all unmeasured while the login step
        itself reported success. The token is only useful once the change is
        done, so it is done here rather than left for each caller to discover.

        The replacement is random per run and is left on ``self.password``,
        because something DOES need it again: the repo's Playwright chat driver
        rotates the password itself as its first phase and asserts the old one
        stops working, so it has to be handed whatever the current password is.
        The first version of this change dropped the replacement on the floor
        and the driver failed with "the bootstrap password is gone, so the
        driver cannot log in" -- three assertions fixed and a fourth broken.

        ``self.password`` is therefore a credential held for the run, unlike
        every other value here. The caller is responsible for adding it to
        whatever scrubs the logs.

        The passwords reach this function and Unsloth and go nowhere else. A
        StudioError from here is raised with the status code only.
        """
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

        # token_urlsafe never yields whitespace, which change-password rejects,
        # and never collides with the bootstrap value it has to differ from.
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
    """Is this ``/api/health`` body an Unsloth that is done starting up?

    Two conditions, and the second is the one that matters. ``status ==
    "healthy"`` is what the repo's own wait-for-health.sh checks, but Unsloth
    answers healthy while hardware detection is still running, and during that
    window it reports itself chat-only and refuses to start a training run or
    an export. A payload that raced that window would fail on the Train and
    Export gates having proved nothing about them.
    """
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
    """Poll ``probe`` until ``accept``, the deadline, or the process dies.

    Returns ``(ok, last_value, reason)``. ``alive`` is what turns a ten-minute
    timeout into a two-second one when the thing being waited on has already
    exited: without it, every crash at startup costs the full deadline and
    reports itself as "slow" rather than "dead".
    """
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
    """Is this ``/api/train/status`` body a finished, successful run?

    Returns ``(terminal, reason)``. A non-terminal phase returns
    ``(False, "")``; a terminal one that is not ``completed`` returns
    ``(True, why)`` so the caller stops polling and reports the real cause
    rather than waiting out its deadline on a job that already failed.
    """
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
    """Has an export finished, and did it succeed?

    ``baseline_seq`` is ``last_op_seq`` sampled before the export was
    requested. Checking it is what separates "this export finished" from "a
    previous operation finished and this one has not started yet" -- the
    export API has no job id, and ``is_export_active`` is false in both cases.
    """
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
    """Did the training run leave a real LoRA adapter on disk?

    This is the assertion that separates "the run reported completed" from
    "the run produced something". Unsloth reports ``completed`` from the
    worker's own bookkeeping; only the files say whether a save happened.
    """
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
    """How many steps the run actually logged a USABLE loss for.

    A run can reach ``completed`` having trained nothing -- a dataset that
    formatted to zero usable rows is the way it happens -- and the phase alone
    does not say. The loss history does.

    Non-finite entries do not count. A T4 has no bf16, so training runs in
    fp16, and an fp16 run that diverges logs ``NaN`` or ``inf`` for every step
    while still reaching ``completed`` and still saving an adapter. Those
    entries are not ``None``, so counting mere list occupancy scored a
    numerically broken run as a full-length one and turned the CUDA training
    assertion green.
    """
    return len([value for value in _loss_values(status) if _is_finite_loss(value)])


def nonfinite_losses(status: Any) -> list:
    """The logged losses that are NaN, infinite, or not a number at all."""
    return [
        value for value in _loss_values(status) if not _is_finite_loss(value) and value is not None
    ]


def newest_gguf(root: str | Path) -> Path | None:
    """The most recently written MODEL ``.gguf`` under ``root``, if any.

    ``mmproj`` sidecars are excluded, and that is not a tidy-up. A vision export
    writes two files -- ``Qwen3.5-2B.Q8_0.gguf`` and
    ``Qwen3.5-2B.F16-mmproj.gguf`` -- and the projector is often the newer of
    the two. Handing it to llama.cpp as a model is not an error: the server
    starts, reports ``gpu_layers=-1``, offloads nothing and still returns text,
    so the run reads as a GPU failure in the export assertion. Measured on
    kernel unsloth-probe-studio-full2-815a0c, where exactly that happened.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = sorted(
        (p for p in root.rglob("*.gguf") if "mmproj" not in p.name.lower()),
        key = lambda p: p.stat().st_mtime,
        reverse = True,
    )
    return candidates[0] if candidates else None

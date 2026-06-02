# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio REST client for the deploy bootstrap chain.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from unsloth_cli.deploy import DeployError


# Cloudflare fronts *.proxy.runpod.net and blocks bot-flagged User-Agents
# ("Error 1010: browser signature banned"); posing as curl gets us through.
_USER_AGENT = "curl/8.7.1"

_HEALTH_POLL_S = 3
_DEFAULT_TIMEOUT_S = 30
_LOAD_TIMEOUT_S = 900          # total budget for a model load to finish
_LOAD_POLL_S = 5               # how often to poll load status while waiting
# Proxies cut a request off before a big model finishes loading; these codes mean
# "still working", not "failed".
_GATEWAY_TIMEOUT_CODES = (504, 524)


def _is_gateway_timeout(msg: str) -> bool:
    """True if an error looks like a proxy/gateway cut-off (Cloudflare 524 and
    friends, or a read/connection timeout) rather than a real failure."""
    if any(f"-> {code}:" in msg for code in _GATEWAY_TIMEOUT_CODES):
        return True
    # A transport-level timeout/reset surfaces as "<verb> <path> failed: <err>";
    # match those words only within that suffix, never inside a response body that
    # could legitimately contain "timeout" and mask a genuine load failure.
    marker = "failed: "
    if marker not in msg:
        return False
    transport = msg.split(marker, 1)[1].lower()
    return any(s in transport for s in ("timed out", "timeout", "connection reset"))


def _parse_json(text: str, where: str) -> dict:
    """Parse a response body, turning a non-JSON body (e.g. a Cloudflare HTML error
    page) into a DeployError instead of a raw ValueError."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except ValueError as e:
        raise DeployError(f"{where} returned a non-JSON response: {text[:200]}") from e


class StudioClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._token: Optional[str] = None

    @property
    def token(self) -> str:
        if not self._token:
            raise DeployError("StudioClient is not authenticated; call login() first.")
        return self._token

    def wait_healthy(self, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                req = urllib.request.Request(
                    self.base_url + "/api/health",
                    headers = {"User-Agent": _USER_AGENT},
                    method = "GET",
                )
                with urllib.request.urlopen(req, timeout = 5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(_HEALTH_POLL_S)
        raise DeployError(
            f"Studio /api/health did not return 200 within {timeout_s}s"
        )

    def login(self, username: str, password: str) -> dict:
        body = self._post(
            "/api/auth/login",
            {"username": username, "password": password},
            auth = False,
        )
        self._token = body["access_token"]
        return body

    def change_password(self, current: str, new: str) -> dict:
        body = self._post(
            "/api/auth/change-password",
            {"current_password": current, "new_password": new},
            auth = True,
        )
        # A 2xx means the password is already changed server-side. Adopt a re-issued
        # token if present; otherwise keep the current one rather than raising
        # KeyError, which would strand the user on a billing instance whose password
        # has rotated without ever surfacing it.
        token = body.get("access_token")
        if token:
            self._token = token
        return body

    def create_api_key(self, name: str) -> str:
        body = self._post("/api/auth/api-keys", {"name": name}, auth = True)
        return body["key"]

    def load_model(self, model_path: str, **kwargs) -> dict:
        """Load a model on the pod. A load can outlast the ~100s a single request
        survives behind Cloudflare (524), so a gateway timeout here is not a
        failure: fall back to polling /api/inference/status until the model is live."""
        payload: dict[str, Any] = {"model_path": model_path}
        payload.update(kwargs)
        try:
            return self._post(
                "/api/inference/load", payload, auth = True, timeout = _LOAD_TIMEOUT_S,
            )
        except DeployError as e:
            # A non-gateway error (bad path, OOM, auth) is a real failure -- let it
            # propagate. Only a proxy cut-off falls through to polling.
            if not _is_gateway_timeout(str(e)):
                raise
        return self._wait_until_loaded(payload, timeout_s = _LOAD_TIMEOUT_S)

    def _wait_until_loaded(self, payload: dict, *, timeout_s: int) -> dict:
        """Poll /api/inference/status until a model is active, after the load was
        cut off by the proxy. If the load stops without a model becoming active,
        re-issue it so the caller sees Studio's real error (re-loading is a no-op)."""
        deadline = time.time() + timeout_s
        saw_loading = False
        while time.time() < deadline:
            try:
                status = self._get("/api/inference/status", auth = True)
            except DeployError:
                time.sleep(_LOAD_POLL_S)
                continue  # transient blip mid-load; keep waiting
            if status.get("active_model"):
                return status
            if status.get("loading"):
                saw_loading = True
            elif saw_loading:
                # Was loading, now neither loading nor active: the load failed.
                # Re-issue so Studio returns the actual error (or 200 if it raced).
                return self._post(
                    "/api/inference/load", payload, auth = True, timeout = _LOAD_TIMEOUT_S,
                )
            time.sleep(_LOAD_POLL_S)
        raise DeployError(
            f"Model '{payload['model_path']}' did not finish loading within {timeout_s}s."
        )

    def _get(self, path: str, *, auth: bool, timeout: int = _DEFAULT_TIMEOUT_S) -> dict:
        headers = {"User-Agent": _USER_AGENT}
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.base_url + path, headers = headers, method = "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout = timeout) as resp:
                text = resp.read().decode(errors = "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors = "replace")
            raise DeployError(f"GET {path} -> {e.code}: {detail[:400]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise DeployError(f"GET {path} failed: {e}") from e
        return _parse_json(text, f"GET {path}")

    def _post(
        self, path: str, json_body: dict, *, auth: bool, timeout: int = _DEFAULT_TIMEOUT_S,
    ) -> dict:
        headers = {"User-Agent": _USER_AGENT, "Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            self.base_url + path,
            data = json.dumps(json_body).encode(),
            headers = headers,
            method = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout = timeout) as resp:
                text = resp.read().decode(errors = "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors = "replace")
            raise DeployError(f"POST {path} -> {e.code}: {detail[:400]}") from e
        except (urllib.error.URLError, OSError) as e:
            # Surface timeouts/resets as DeployError so the caller still prints the
            # stop-the-pod hint.
            raise DeployError(f"POST {path} failed: {e}") from e
        return _parse_json(text, f"POST {path}")

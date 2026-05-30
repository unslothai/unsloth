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
# (`Python-urllib/*`, `python-requests/*`) with "Error 1010: browser signature
# banned". Posing as curl gets us through.
_USER_AGENT = "curl/8.7.1"

_HEALTH_POLL_S = 3
_DEFAULT_TIMEOUT_S = 30
_LOAD_TIMEOUT_S = 900


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
        self._token = body["access_token"]
        return body

    def create_api_key(self, name: str) -> str:
        body = self._post("/api/auth/api-keys", {"name": name}, auth = True)
        return body["key"]

    def load_model(self, model_path: str, **kwargs) -> dict:
        payload: dict[str, Any] = {"model_path": model_path}
        payload.update(kwargs)
        return self._post(
            "/api/inference/load", payload, auth = True, timeout = _LOAD_TIMEOUT_S,
        )

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
            # Timeouts and connection resets land here; surface them as
            # DeployError so the caller still prints the stop-the-pod hint.
            raise DeployError(f"POST {path} failed: {e}") from e
        return json.loads(text) if text else {}

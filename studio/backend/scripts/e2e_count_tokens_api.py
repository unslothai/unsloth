#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API-stack E2E: load a small GGUF and POST /api/inference/chat/count_tokens."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from utils.paths.storage_roots import cache_root  # noqa: E402

BASE = os.environ.get("E2E_BACKEND_URL", "http://127.0.0.1:8888")
MODEL_PATH = os.environ.get("E2E_GGUF_PATH", str(cache_root() / "stories260K.gguf"))


def _request(
    method: str,
    path: str,
    *,
    token: str,
    body: dict | None = None,
    timeout: float = 300.0,
) -> tuple[int, object]:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{BASE}{path}",
        data = data,
        headers = headers,
        method = method,
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors = "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload


def main() -> int:
    from auth import storage
    from auth.authentication import create_access_token

    if not Path(MODEL_PATH).exists():
        print(f"GGUF not found at {MODEL_PATH}; set E2E_GGUF_PATH")
        return 1

    storage.ensure_default_admin()
    # Seeded account carries must_change_password, so mint with the desktop exemption (as
    # the Playwright issuer does) or every protected route answers 403.
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME, desktop = True)

    print("== 1. Auth status ==")
    code, status = _request("GET", "/api/auth/status", token = token)
    print(f"HTTP {code}: {status}")
    if code != 200:
        return 1

    print("\n== 2. Inference status (before load) ==")
    code, before = _request("GET", "/api/inference/status", token = token)
    print(f"HTTP {code}: loaded={before.get('loaded') if isinstance(before, dict) else before}")

    print(f"\n== 3. Load GGUF: {MODEL_PATH} ==")
    load_body = {
        "model_path": MODEL_PATH,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "is_lora": False,
        "gpu_layers": 0,
    }
    code, load_resp = _request(
        "POST",
        "/api/inference/load",
        token = token,
        body = load_body,
        timeout = 600.0,
    )
    print(f"HTTP {code}")
    if code not in (200, 201):
        print(json.dumps(load_resp, indent = 2))
        return 1
    print(json.dumps(load_resp, indent = 2)[:800])

    print("\n== 4. Wait for loaded status ==")
    loaded = False
    for attempt in range(60):
        code, st = _request("GET", "/api/inference/status", token = token)
        if code == 200 and isinstance(st, dict) and st.get("loaded"):
            loaded = True
            print(f"Loaded after {attempt + 1} poll(s): {st.get('model_path') or st.get('model')}")
            break
        time.sleep(2)
    if not loaded:
        print("Model never reported loaded=true")
        return 1

    print("\n== 5. POST /api/inference/chat/count_tokens ==")
    count_body = {
        "model": MODEL_PATH,
        "messages": [
            {"role": "user", "content": "Once upon a time"},
            {"role": "assistant", "content": "there was a brave fox."},
            {"role": "user", "content": "Tell me more."},
        ],
        "enable_tools": True,
        "enabled_tools": ["web_search"],
    }
    code, count_resp = _request(
        "POST",
        "/api/inference/chat/count_tokens",
        token = token,
        body = count_body,
    )
    print(f"HTTP {code}: {count_resp}")
    if code != 200:
        return 1

    input_tokens = count_resp.get("input_tokens") if isinstance(count_resp, dict) else None
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        print(f"FAIL: expected positive input_tokens, got {input_tokens!r}")
        return 1

    print(f"\nPASS: input_tokens={input_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

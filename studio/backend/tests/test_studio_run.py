# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
End-to-end tests for ``unsloth studio run`` and API key authentication.

Starts a Studio server via the ``run`` subcommand, then exercises the
four usage examples shown on the API Keys page:

    1. curl -- basic chat completions (non-streaming)
    2. curl -- streaming chat completions
    3. Python OpenAI SDK -- streaming completions
    4. curl -- with tools (web_search + python)

The test also validates the ``--help`` output and the server banner.

Usage:
    python test_studio_run.py                        # default model
    python test_studio_run.py --model unsloth/...    # custom model

Requires a GPU and ~2 GB of disk for the GGUF download.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────

DEFAULT_MODEL = "unsloth/Qwen3-1.7B-GGUF"
DEFAULT_VARIANT = "UD-Q4_K_XL"
PORT = 18222  # high port unlikely to collide
HOST = "127.0.0.1"
STARTUP_TIMEOUT = 120  # seconds to wait for banner
LOG_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "temp"
    / "test_studio_run.log"
)


# ── Helpers ──────────────────────────────────────────────────────────


def _http(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 60,
) -> tuple[int, str]:
    """Minimal stdlib HTTP helper.  Returns (status_code, body_text)."""
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data = data, headers = headers or {}, method = method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors = "replace")


def _stream_http(
    url: str,
    *,
    body: dict,
    headers: dict,
    timeout: int = 60,
) -> tuple[int, list[dict]]:
    """POST a streaming request and collect SSE chunks."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data = data, headers = headers, method = "POST")
    req.add_header("Content-Type", "application/json")
    chunks: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            status = resp.status
            for raw_line in resp:
                line = raw_line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunks.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
            return status, chunks
    except urllib.error.HTTPError as exc:
        return exc.code, []


# ── Test functions ───────────────────────────────────────────────────


def test_help_output():
    """``unsloth studio run --help`` should show all documented options."""
    result = subprocess.run(
        ["unsloth", "studio", "run", "--help"],
        capture_output = True,
        text = True,
        timeout = 15,
    )
    out = result.stdout
    assert result.returncode == 0, f"--help exited with {result.returncode}"

    for flag in [
        "--model",
        "--gguf-variant",
        "--max-seq-length",
        "--load-in-4bit",
        "--api-key-name",
        "--port",
        "--host",
        "--frontend",
        "--silent",
    ]:
        assert flag in out, f"Missing flag {flag!r} in --help output"
    print("  PASS  --help shows all flags")


def test_curl_basic(base_url: str, api_key: str):
    """Example 1: basic non-streaming chat completion via HTTP."""
    status, text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Say just the word hello"}],
            "stream": False,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
    )
    assert status == 200, f"Expected 200, got {status}: {text[:300]}"
    data = json.loads(text)
    assert "choices" in data, f"Missing 'choices' in response: {text[:300]}"
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Empty assistant content"
    print(f"  PASS  curl basic: {content[:80]!r}")


def _collect_streamed_content(chunks: list[dict]) -> str:
    """Extract text from SSE chunks, skipping role-only and usage chunks."""
    parts = []
    for c in chunks:
        choices = c.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        part = delta.get("content")
        if part:
            parts.append(part)
    return "".join(parts)


def test_curl_streaming(base_url: str, api_key: str):
    """Example 2: streaming chat completion via HTTP SSE."""
    status, chunks = _stream_http(
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Count from 1 to 3"}],
            "stream": True,
        },
        headers = {"Authorization": f"Bearer {api_key}"},
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(chunks) > 0, "No SSE chunks received"
    full = _collect_streamed_content(chunks)
    assert len(full) > 0, "Streamed content is empty"
    print(f"  PASS  curl streaming: got {len(chunks)} chunks, {len(full)} chars")


def test_openai_sdk(base_url: str, api_key: str):
    """Example 3: OpenAI Python SDK streaming completion."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP  openai SDK not installed")
        return

    client = OpenAI(base_url = f"{base_url}/v1", api_key = api_key)
    response = client.chat.completions.create(
        model = "current",
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        stream = True,
    )
    content_parts = []
    for chunk in response:
        if not chunk.choices:
            continue
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            content_parts.append(delta_content)
    full = "".join(content_parts)
    assert len(full) > 0, "OpenAI SDK returned empty content"
    print(f"  PASS  OpenAI SDK streaming: {full.strip()[:80]!r}")


def test_curl_with_tools(base_url: str, api_key: str):
    """Example 4: chat completion with tool calling enabled.

    Note: when ``enable_tools`` is set the server always returns SSE
    streaming regardless of the ``stream`` flag, so we parse SSE chunks.
    The model may or may not produce visible content -- tool orchestration
    can intercept the response -- so we only assert the endpoint succeeds.
    """
    status, chunks = _stream_http(
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 123 * 456? Use code to compute it.",
                }
            ],
            "stream": True,
            "enable_tools": True,
            "enabled_tools": ["python"],
            "session_id": "test-session",
        },
        headers = {"Authorization": f"Bearer {api_key}"},
        timeout = 120,
    )
    assert status == 200, f"Expected 200, got {status}"
    assert len(chunks) > 0, "No SSE chunks received for tools request"

    # Check that at least one chunk has the expected shape
    has_valid_chunk = any("choices" in c or "type" in c for c in chunks)
    assert has_valid_chunk, "No valid chunks in tools response"
    full = _collect_streamed_content(chunks)
    print(f"  PASS  curl with tools: {len(chunks)} chunks, {len(full)} chars content")


def test_invalid_key_rejected(base_url: str):
    """Requests with a bad API key should be rejected."""
    status, _text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        headers = {"Authorization": "Bearer sk-unsloth-boguskey123"},
    )
    assert status == 401, f"Expected 401 for invalid key, got {status}"
    print("  PASS  invalid API key rejected (401)")


def test_no_key_rejected(base_url: str):
    """Requests without any auth header should be rejected."""
    status, _text = _http(
        "POST",
        f"{base_url}/v1/chat/completions",
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert status == 401 or status == 403, f"Expected 401/403 for no key, got {status}"
    print(f"  PASS  no API key rejected ({status})")


# ── Server lifecycle ─────────────────────────────────────────────────


def _start_server(model: str, variant: str | None) -> tuple[subprocess.Popen, str]:
    """Launch ``unsloth studio run`` and parse the API key from its banner.

    Returns (process, api_key).
    """
    cmd = [
        "unsloth",
        "studio",
        "run",
        "--model",
        model,
        "--port",
        str(PORT),
        "--host",
        HOST,
        "--api-key-name",
        "test",
    ]
    if variant:
        cmd.extend(["--gguf-variant", variant])

    LOG_FILE.parent.mkdir(parents = True, exist_ok = True)
    log_fh = open(LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout = log_fh,
        stderr = subprocess.STDOUT,
        preexec_fn = os.setsid,
    )

    # Wait for the banner containing the API key
    api_key = None
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(2)
        if proc.poll() is not None:
            log_fh.flush()
            log_text = LOG_FILE.read_text()
            raise RuntimeError(
                f"Server exited early (code {proc.returncode}):\n{log_text[-2000:]}"
            )
        log_text = LOG_FILE.read_text()
        m = re.search(r"API Key:\s+(sk-unsloth-[a-f0-9]+)", log_text)
        if m:
            api_key = m.group(1)
            break

    if not api_key:
        log_text = LOG_FILE.read_text()
        _kill_server(proc)
        raise RuntimeError(
            f"Timed out waiting for API key in server output:\n{log_text[-2000:]}"
        )

    # Wait a moment for the model to be fully loaded
    time.sleep(2)
    return proc, api_key


def _kill_server(proc: subprocess.Popen):
    """Send SIGTERM to the process group and wait for cleanup."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout = 10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout = 5)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description = "End-to-end tests for unsloth studio run"
    )
    parser.add_argument(
        "--model",
        default = DEFAULT_MODEL,
        help = f"Model to test with (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--gguf-variant",
        default = DEFAULT_VARIANT,
        help = f"GGUF variant (default: {DEFAULT_VARIANT})",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    def run_test(fn, *a, **kw):
        nonlocal passed, failed, skipped
        try:
            fn(*a, **kw)
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")

    # ── 1. Test --help (no server needed) ────────────────────────────
    print("\n[1/7] Testing --help output")
    run_test(test_help_output)

    # ── 2-7. Start server and run API tests ──────────────────────────
    print(
        f"\nStarting server: {args.model} (variant={args.gguf_variant}) on port {PORT}..."
    )
    proc = None
    try:
        proc, api_key = _start_server(args.model, args.gguf_variant)
        base_url = f"http://{HOST}:{PORT}"
        print(f"Server ready.  API Key: {api_key[:20]}...\n")

        print("[2/7] Testing curl basic (non-streaming)")
        run_test(test_curl_basic, base_url, api_key)

        print("[3/7] Testing curl streaming")
        run_test(test_curl_streaming, base_url, api_key)

        print("[4/7] Testing OpenAI Python SDK (streaming)")
        run_test(test_openai_sdk, base_url, api_key)

        print("[5/7] Testing curl with tools")
        run_test(test_curl_with_tools, base_url, api_key)

        print("[6/7] Testing invalid API key rejection")
        run_test(test_invalid_key_rejected, base_url)

        print("[7/7] Testing no API key rejection")
        run_test(test_no_key_rejected, base_url)

    except RuntimeError as exc:
        print(f"\nFATAL: Server failed to start: {exc}")
        failed += 7  # count remaining tests as failed
    finally:
        if proc:
            print("\nStopping server...")
            _kill_server(proc)
            print("Server stopped.")

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"Log: {LOG_FILE}")
    print(f"{'=' * 40}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

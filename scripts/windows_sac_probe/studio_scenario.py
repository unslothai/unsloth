# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Exercise a running Studio and time every request.

Driven entirely over HTTP with the standard library, so it runs under whatever
Python is on the machine and needs no browser and no pip install. That matters
because this is meant to be run by someone reproducing a bug, not by CI.

Two things are being measured at once.

Whether the local runtime works at all: load a GGUF, generate, run a web search
turn and a tool call turn. On a machine where Smart App Control is blocking
llama.cpp, the load is what fails, and the backend log will carry the code
integrity reason.

How long the status route takes: a background poller hits /api/inference/status
throughout and records every duration. Diagnostics from the field show that
route taking over 80 seconds, which starves /api/health and gets the backend
killed by the desktop watchdog at its ~75 second budget, reported as "Server
stopped unexpectedly". The stalls were observed with no model loaded, so the
cause is not yet established, and the point of the poller is to measure it
rather than to assume it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


DEFAULT_PORTS = list(range(8888, 8896))


def split_model_ref(ref: str) -> tuple[str, Optional[str]]:
    """`repo:variant` into the two fields /api/inference/load takes.

    The backend has no shorthand parser: `model_path` is the repository or
    path and `gguf_variant` selects the quant, and a colon left in `model_path`
    reaches the hub as an invalid repository id. Only the CLI splits the
    shorthand, so this does the same. A colon followed by a slash is part of a
    path (`C:\\models\\x.gguf`), not a variant.
    """
    repo, sep, variant = ref.rpartition(":")
    if not sep or not repo or "/" in variant or "\\" in variant:
        return ref, None
    return repo, variant


class Timed:
    """Every HTTP call this script makes, with how long it took."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(
        self,
        method: str,
        path: str,
        ms: float,
        status: Any,
        note: str = "",
    ) -> None:
        with self._lock:
            self.calls.append(
                {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "method": method,
                    "path": path,
                    "ms": round(ms, 1),
                    "status": status,
                    "note": note,
                }
            )

    def slowest(self, n: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self.calls, key = lambda c: -c["ms"])[:n]


TIMED = Timed()


def _request(
    base_url: str,
    method: str,
    path: str,
    payload: Optional[dict] = None,
    token: Optional[str] = None,
    timeout: int = 900,
) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base_url + path, data = data, headers = headers, method = method)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout = timeout) as response:
            body = response.read()
            ms = (time.monotonic() - start) * 1000.0
            TIMED.record(method, path, ms, response.status)
            try:
                return response.status, json.loads(body)
            except ValueError:
                return response.status, body.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        ms = (time.monotonic() - start) * 1000.0
        body = exc.read().decode("utf-8", "replace")
        TIMED.record(method, path, ms, exc.code, body[:400])
        return exc.code, body
    except Exception as exc:  # noqa: BLE001 - a transport failure is a result here
        ms = (time.monotonic() - start) * 1000.0
        TIMED.record(method, path, ms, "error", str(exc)[:400])
        return 0, str(exc)


def discover_port(explicit: Optional[int]) -> int:
    ports = [explicit] if explicit else DEFAULT_PORTS
    for port in ports:
        status, _ = _request(f"http://127.0.0.1:{port}", "GET", "/api/liveness", timeout = 5)
        if status == 200:
            return port
    raise SystemExit(
        "no Studio backend answered /api/liveness on "
        + ", ".join(str(p) for p in ports)
        + ". Start Unsloth Studio first, or pass --port."
    )


def authenticate(base_url: str, home: Path, password: Optional[str]) -> str:
    """Log in, rotating the bootstrap credential when this home has never been used.

    Studio deletes auth/.bootstrap_password once it has been rotated, so on a
    desktop install that has already been opened the file is gone and the only
    credential left is the one the app set. There is no way to recover it from
    disk, which is why --password exists.

    The rotation is permanent and `revert` does not undo it, so the new
    password is always the operator's own: a default here would leave every
    probed machine on a published credential, and printing the operator's
    choice would put it in studio-scenario.log inside the evidence zip.
    """
    boot_file = home / "auth" / ".bootstrap_password"
    if not password:
        raise SystemExit(
            "no password given. Pass --password (or set UNSLOTH_STUDIO_PASSWORD): on a "
            "Studio that has never been opened it becomes the account password, on one "
            "that has it must be the password you sign in with."
        )
    if boot_file.exists():
        secret = boot_file.read_text(encoding = "utf-8").strip()
        rotate = True
    else:
        secret = password
        rotate = False

    status, body = _request(
        base_url, "POST", "/api/auth/login", {"username": "unsloth", "password": secret}
    )
    if status != 200:
        raise SystemExit(f"login failed ({status}): {str(body)[:300]}")
    token = body["access_token"]

    if rotate:
        status, body = _request(
            base_url,
            "POST",
            "/api/auth/change-password",
            {"current_password": secret, "new_password": password},
            token = token,
        )
        if status != 200:
            raise SystemExit(f"password rotation failed ({status}): {str(body)[:300]}")
        token = body["access_token"]
        print("  rotated the bootstrap password to the one given with --password")
    return token


class StatusPoller(threading.Thread):
    """Poll /api/inference/status the way the frontend does, and time it."""

    def __init__(
        self,
        base_url: str,
        token: str,
        interval: float = 5.0,
    ) -> None:
        super().__init__(daemon = True)
        self.base_url = base_url
        self.token = token
        self.interval = interval
        # Not `_stop`: Thread.join() calls its own internal `_stop()` method,
        # and an Event assigned over it raised "'Event' object is not callable"
        # out of the finally block before the results were written.
        self._stop_event = threading.Event()
        self.durations: list[float] = []
        self.in_flight_since: Optional[float] = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()
            self.in_flight_since = start
            _request(self.base_url, "GET", "/api/inference/status", token = self.token, timeout = 300)
            self.in_flight_since = None
            self.durations.append((time.monotonic() - start) * 1000.0)
            # Interval from the last start, not the last finish, so a slow poll
            # does not quietly stretch the cadence and hide the pile-up.
            self._stop_event.wait(max(0.0, self.interval - (time.monotonic() - start)))

    def stop(self) -> None:
        self._stop_event.set()

    def in_flight_ms(self) -> Optional[float]:
        """How long the current poll has been waiting, if one is."""
        since = self.in_flight_since
        return None if since is None else (time.monotonic() - since) * 1000.0


def chat(
    base_url: str,
    token: str,
    model: str,
    prompt: str,
    *,
    tools: bool = False,
    enabled_tools: Optional[list[str]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": False,
    }
    if tools:
        payload["enable_tools"] = True
        payload["enabled_tools"] = enabled_tools or ["web_search"]
    status, body = _request(base_url, "POST", "/v1/chat/completions", payload, token = token)
    text = ""
    tool_calls: list[Any] = []
    if status == 200 and isinstance(body, dict):
        for choice in body.get("choices", []) or []:
            message = choice.get("message") or {}
            text += message.get("content") or ""
            tool_calls.extend(message.get("tool_calls") or [])
    return {
        "status": status,
        "ok": status == 200 and bool(text.strip() or tool_calls),
        "chars": len(text),
        "tool_calls": len(tool_calls),
        "text": text[:600],
        "error": None if status == 200 else str(body)[:400],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", default = "unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL")
    parser.add_argument("--out", default = ".", help = "directory for scenario-results.json")
    parser.add_argument("--port", type = int, default = None)
    parser.add_argument("--password", default = os.environ.get("UNSLOTH_STUDIO_PASSWORD"))
    parser.add_argument(
        "--home",
        default = (
            os.environ.get("UNSLOTH_STUDIO_HOME")
            or os.environ.get("STUDIO_HOME")
            or str(Path.home() / ".unsloth" / "studio")
        ),
        help = "Studio home (where auth/ lives); follows UNSLOTH_STUDIO_HOME like Studio does",
    )
    parser.add_argument(
        "--poll-seconds",
        type = float,
        default = 5.0,
        help = "status poll cadence; 5s matches the frontend",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents = True, exist_ok = True)

    port = discover_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    print(f"Studio on {base_url}")

    token = authenticate(base_url, Path(args.home), args.password)
    print("authenticated")

    poller = StatusPoller(base_url, token, args.poll_seconds)
    poller.start()

    repo, variant = split_model_ref(args.model)
    results: dict[str, Any] = {
        "model": args.model,
        "model_path": repo,
        "gguf_variant": variant,
        "port": port,
        "steps": {},
    }
    load_payload: dict[str, Any] = {"model_path": repo}
    if variant:
        load_payload["gguf_variant"] = variant
    try:
        print(f"loading {args.model} ...")
        status, body = _request(
            base_url,
            "POST",
            "/api/inference/load",
            load_payload,
            token = token,
            timeout = 1800,
        )
        results["steps"]["load"] = {
            "status": status,
            "ok": status == 200,
            "error": None if status == 200 else str(body)[:800],
        }
        if status != 200:
            # This is the interesting failure. A blocked llama-server shows up
            # here, and the backend log carries the code integrity reason.
            print(f"  load FAILED ({status}): {str(body)[:400]}")
        else:
            print("  loaded")

            print("inference ...")
            results["steps"]["inference"] = chat(
                base_url,
                token,
                repo,
                "Reply with exactly one short sentence about the Antarctic.",
            )
            print(f"  {results['steps']['inference']['chars']} chars")

            print("web search ...")
            results["steps"]["web_search"] = chat(
                base_url,
                token,
                repo,
                "Search the web for today's date in Reykjavik and tell me what you found.",
                tools = True,
                enabled_tools = ["web_search"],
            )
            print(f"  tool calls: {results['steps']['web_search']['tool_calls']}")

            print("tool calls ...")
            results["steps"]["tool_calls"] = chat(
                base_url,
                token,
                repo,
                "Use your tools to look up what the tallest building in the world is.",
                tools = True,
                enabled_tools = ["web_search"],
            )
            print(f"  tool calls: {results['steps']['tool_calls']['tool_calls']}")

            # UnloadRequest.model_path is required; an empty body is a 422 that
            # leaves the runtime resident, and the next matrix cell would then
            # reuse this llama-server instead of loading its PE files again.
            status, body = _request(
                base_url,
                "POST",
                "/api/inference/unload",
                {"model_path": repo},
                token = token,
                timeout = 300,
            )
            results["steps"]["unload"] = {
                "status": status,
                "ok": status == 200,
                "error": None if status == 200 else str(body)[:400],
            }
            print(
                "  unloaded" if status == 200 else f"  unload FAILED ({status}): {str(body)[:200]}"
            )
    finally:
        poller.stop()
        # A poll stalled for the 75 to 80 seconds this exists to catch is still
        # in flight here. Wait it out (its own timeout is 300s) rather than
        # discard the daemon thread with the one duration that matters.
        poller.join(timeout = 330)

    durations = poller.durations
    in_flight = poller.in_flight_ms() if poller.is_alive() else None
    results["status_poll"] = {
        "count": len(durations),
        "in_flight_ms": None if in_flight is None else round(in_flight, 1),
        "max_ms": round(max(durations), 1) if durations else None,
        "median_ms": round(statistics.median(durations), 1) if durations else None,
        # The watchdog kills the backend after roughly 75s of unanswered health
        # checks, so anything at or past that is the reported failure mode.
        "over_10s": sum(1 for d in durations if d > 10_000)
        + (1 if in_flight and in_flight > 10_000 else 0),
        "over_75s": sum(1 for d in durations if d > 75_000)
        + (1 if in_flight and in_flight > 75_000 else 0),
    }
    results["slowest_calls"] = TIMED.slowest(15)
    results["all_calls"] = TIMED.calls

    path = out_dir / "scenario-results.json"
    path.write_text(json.dumps(results, indent = 2), encoding = "utf-8")

    print()
    print("=" * 60)
    for name, step in results["steps"].items():
        print(f"{name:12s} {'ok' if step.get('ok') else 'FAILED'}  ({step.get('status')})")
    poll = results["status_poll"]
    print(
        f"status polls  {poll['count']}, median {poll['median_ms']} ms, max {poll['max_ms']} ms, "
        f"{poll['over_10s']} over 10s, {poll['over_75s']} over 75s"
    )
    print(f"written to {path}")

    # Nonzero when the runtime did not work, so the caller can tell the cells of
    # the matrix apart without parsing the JSON.
    return 0 if all(s.get("ok") for s in results["steps"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

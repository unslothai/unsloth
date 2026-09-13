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


# /api/inference/load and /unload commit a 200 after about 15 seconds and pad
# the body so a proxy cannot time the call out; a failure found after that
# travels only under this key, and a proxy that gives up mid-pad leaves an
# empty body. The same reading unsloth_cli/_inference.py applies.
_DEFERRED_ERROR_KEY = "_deferred_error"


def padded_route_failure(status: int, body: Any) -> Optional[str]:
    """Why a /load or /unload reply is not a success, or None when it is."""
    if status != 200:
        return str(body)[:800]
    if not isinstance(body, dict) or not body:
        return "the connection closed before the server's reply arrived (empty or truncated body)"
    deferred = body.get(_DEFERRED_ERROR_KEY)
    if isinstance(deferred, dict):
        return f"deferred {deferred.get('status_code')}: {str(deferred.get('detail'))[:700]}"
    return None


def _stream_events(
    base_url: str,
    path: str,
    payload: dict,
    token: Optional[str],
    timeout: int = 900,
) -> tuple[int, list[dict], Optional[str]]:
    """POST a streaming request and return every parsed `data:` object.

    Tool execution is visible only here: the non-streaming route drains the
    tool loop and returns the final text alone, so a turn that never ran a
    tool is indistinguishable from one that did.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        # tool_start / tool_end carry no `choices`, so /v1/chat/completions
        # suppresses them for external clients and emits a clean OpenAI stream;
        # the Studio frontend opts back in with this header
        # (chat-api.ts, _ui_stream_events_enabled in routes/inference.py).
        # Without it the tool turns below see an empty `finished` list, report
        # "no tool_end event: the turn executed no tool", and the scenario exits
        # non-zero on every real Studio, so the probe cannot measure the tool
        # behaviour it drives the stream for.
        "X-Unsloth-Events": "1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        base_url + path, data = json.dumps(payload).encode(), headers = headers, method = "POST"
    )
    start = time.monotonic()
    events: list[dict] = []
    error: Optional[str] = None
    done = False
    try:
        with urllib.request.urlopen(req, timeout = timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    parsed = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                events.append(parsed)
                # A failure after the status line went out arrives in band,
                # as `{"error": ...}` with the 200 kept, and the stream ends
                # there without [DONE].
                if "error" in parsed and error is None:
                    detail = parsed["error"]
                    message = detail.get("message") if isinstance(detail, dict) else detail
                    error = f"stream error: {str(message)[:300]}"
            if error is None and not done:
                error = "stream ended without [DONE]: the turn did not complete"
            TIMED.record("POST", path, (time.monotonic() - start) * 1000.0, response.status)
            return response.status, events, error
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        TIMED.record("POST", path, (time.monotonic() - start) * 1000.0, exc.code, body[:400])
        return exc.code, events, body[:400]
    except Exception as exc:  # noqa: BLE001 - a transport failure is a result here
        TIMED.record("POST", path, (time.monotonic() - start) * 1000.0, "error", str(exc)[:400])
        return 0, events, str(exc)[:400]


def discover_port(explicit: Optional[int]) -> int:
    ports = [explicit] if explicit else DEFAULT_PORTS
    for port in ports:
        status, body = _request(f"http://127.0.0.1:{port}", "GET", "/api/liveness", timeout = 5)
        # The identity marker, not the status code. The next thing this script
        # does with the port it picks is post the operator's Studio password to
        # it, and these are default ports shared with other software (8888 is
        # Jupyter's), so a catch-all server answering 200 would have been sent
        # the credential. Same check as Test-StudioResponding in sac-probe.ps1;
        # `service` has been in this route since the route existed.
        if status == 200 and isinstance(body, dict) and body.get("service") == "Unsloth UI Backend":
            return port
    raise SystemExit(
        "no Studio backend answered /api/liveness on "
        + ", ".join(str(p) for p in ports)
        + ". Start Unsloth Studio first, or pass --port."
    )


def default_studio_home() -> str:
    """Studio's own precedence (utils/paths/storage_roots.studio_root):
    UNSLOTH_STUDIO_HOME, then the STUDIO_HOME alias, then the legacy home. A
    value that is only whitespace counts as unset there, so it does here."""
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    return override or str(Path.home() / ".unsloth" / "studio")


def resolve_studio_home(value: str) -> Path:
    """The same normalization Studio applies: strip, expanduser, resolve.

    A supported override like ``UNSLOTH_STUDIO_HOME=~\\my-studio`` is a
    cwd-relative directory named ``~`` to pathlib, so the bootstrap credential
    was looked for somewhere the running Studio never wrote it: an installation
    that had never been opened then read as already rotated and the login was
    attempted with the operator's replacement password instead.
    """
    home = Path(value.strip()).expanduser()
    try:
        return home.resolve()
    except (OSError, ValueError):
        return home


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
    # An EMPTY file is a rotated installation: on Windows Studio truncates the
    # bootstrap file when it cannot delete it (auth/storage.py), which is the
    # state of exactly the locked-down machines this probe targets.
    bootstrap = boot_file.read_text(encoding = "utf-8").strip() if boot_file.exists() else ""
    if bootstrap:
        secret = bootstrap
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


# The frontend's numbers (studio/frontend/src/features/loaded-models): a tick
# every 5 s, each status read abandoned after 10 s, and a tick that arrived
# while a read was in flight runs as soon as that read ends. Under a stall the
# pattern is therefore a stream of 10 s abandoned requests, not one long one,
# and reproducing the load means reproducing that.
STATUS_INTERVAL_S = 5.0
STATUS_READ_TIMEOUT_S = 10.0


class StatusPoller(threading.Thread):
    """Poll /api/inference/status the way the frontend does, and time it."""

    def __init__(
        self,
        base_url: str,
        token: str,
        interval: float = STATUS_INTERVAL_S,
        read_timeout: float = STATUS_READ_TIMEOUT_S,
    ) -> None:
        super().__init__(daemon = True)
        self.base_url = base_url
        self.token = token
        self.interval = interval
        self.read_timeout = read_timeout
        # (started_at, duration_ms, timed_out) per poll, in order.
        self.polls: list[tuple[float, float, bool]] = []
        # Not `_stop`: Thread.join() calls its own internal `_stop()` method,
        # and an Event assigned over it raised "'Event' object is not callable"
        # out of the finally block before the results were written.
        self._stop_event = threading.Event()
        self.in_flight_since: Optional[float] = None

    @property
    def durations(self) -> list[float]:
        return [ms for _, ms, _ in self.polls]

    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()
            self.in_flight_since = start
            status, _ = _request(
                self.base_url,
                "GET",
                "/api/inference/status",
                token = self.token,
                timeout = self.read_timeout,
            )
            self.in_flight_since = None
            elapsed = time.monotonic() - start
            timed_out = status == 0 and elapsed >= self.read_timeout
            self.polls.append((start, elapsed * 1000.0, timed_out))
            # A read that ran past the tick is followed at once by the tick it
            # blocked, as the frontend's queued refresh does; otherwise the
            # interval runs from the last start, so a slow poll does not
            # quietly stretch the cadence and hide the pile-up.
            self._stop_event.wait(max(0.0, self.interval - elapsed))

    def stalls_ms(self) -> list[float]:
        """Length of each run of consecutive abandoned reads, first start to last end."""
        runs: list[float] = []
        run_start: Optional[float] = None
        run_end = 0.0
        for start, ms, timed_out in self.polls:
            if timed_out:
                run_start = start if run_start is None else run_start
                run_end = start + ms / 1000.0
            elif run_start is not None:
                runs.append((run_end - run_start) * 1000.0)
                run_start = None
        if run_start is not None:
            runs.append((run_end - run_start) * 1000.0)
        return runs

    def stop(self) -> None:
        self._stop_event.set()

    def in_flight_ms(self) -> Optional[float]:
        """How long the current poll has been waiting, if one is."""
        since = self.in_flight_since
        return None if since is None else (time.monotonic() - since) * 1000.0


# studio/backend/state/tool_approvals.py; the loop puts it in `result` when a
# call is declined before running.
TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."
# studio_tool_loop.py closes a card it did not execute (budget exhausted,
# disabled, truncated by the provider, cancelled, duplicate) with a result
# that starts one of these ways.
TOOL_NOT_RUN_PREFIXES = ("Unsloth did not ", "Unsloth stopped this tool call")


def tool_end_failure(result: Any) -> Optional[str]:
    """Why a tool_end did not come from an executed tool, or None if it did."""
    if not isinstance(result, str) or not result.strip():
        return "empty result"
    text = result.strip()
    if text == TOOL_REJECTED_MESSAGE:
        return "declined before running"
    if text.startswith("Error:") or text.startswith(TOOL_NOT_RUN_PREFIXES):
        return text[:160]
    return None


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
        # Streamed, because tool_start / tool_end ride the stream and nothing
        # else says a tool ran: the non-streaming route returns the final
        # text alone, and a model answering from memory would pass as a tool
        # turn. tool_choice pins the tool so an ignored instruction is a
        # server finding rather than a model's mood.
        names = enabled_tools or ["web_search"]
        payload["stream"] = True
        payload["enable_tools"] = True
        payload["enabled_tools"] = names
        payload["tool_choice"] = {"type": "function", "function": {"name": names[0]}}
        # Nobody is at the keyboard. An unset permission_mode is read as "auto",
        # and under auto web_search prompts as soon as the model supplies a url,
        # since the tool then fetches that page (routes/inference.py,
        # _confirm_gate_needs_stream). The approval would be written to the
        # stream and waited on for _DECISION_TIMEOUT, an hour, against this
        # script's 900s read timeout, so the turn would hang and then fail as a
        # transport error. "off" disables the confirmation gate only; the
        # python/terminal sandbox stays on, unlike "full" / bypass_permissions.
        payload["permission_mode"] = "off"
        status, events, error = _stream_events(
            base_url, "/v1/chat/completions", payload, token = token
        )
        text = ""
        started: list[str] = []
        finished: list[str] = []
        failed: list[dict[str, str]] = []
        for event in events:
            kind = event.get("type")
            if kind == "tool_start":
                started.append(str(event.get("tool_name") or ""))
            elif kind == "tool_end":
                # A tool_end is also what a refusal, a lost runtime and an
                # interrupted call emit, with the reason in `result`. Only a
                # non-empty result that is not one of those is an execution.
                name = str(event.get("tool_name") or "")
                why = tool_end_failure(event.get("result"))
                if why is None:
                    finished.append(name)
                else:
                    failed.append({"tool": name, "why": why})
            for choice in event.get("choices", []) or []:
                text += (choice.get("delta") or {}).get("content") or ""
        if error:
            reason = error
        elif finished:
            reason = None
        elif failed:
            reason = "tool ended without executing: " + "; ".join(f["why"] for f in failed)[:300]
        else:
            reason = "no tool_end event: the turn executed no tool"
        return {
            "status": status,
            "ok": status == 200 and bool(finished) and error is None,
            "chars": len(text),
            "tool_calls": len(finished),
            "tools_started": started,
            "tools_run": finished,
            "tools_failed": failed,
            "text": text[:600],
            "error": reason,
        }
    status, body = _request(base_url, "POST", "/v1/chat/completions", payload, token = token)
    text = ""
    if status == 200 and isinstance(body, dict):
        for choice in body.get("choices", []) or []:
            message = choice.get("message") or {}
            text += message.get("content") or ""
    return {
        "status": status,
        "ok": status == 200 and bool(text.strip()),
        "chars": len(text),
        "tool_calls": 0,
        "text": text[:600],
        "error": None if status == 200 else str(body)[:400],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", default = "unsloth/Qwen3.5-2B-MTP-GGUF:UD-Q4_K_XL")
    parser.add_argument("--out", default = ".", help = "directory for scenario-results.json")
    parser.add_argument("--port", type = int, default = None)
    # SAC_PROBE_STUDIO_PASSWORD, not UNSLOTH_STUDIO_PASSWORD: unsloth_cli claims
    # that name and treats it as set-the-initial-password, so a Studio that
    # already has one hard-errors on launch. The probe sets this variable for
    # this child alone and removes it afterwards. --password is still accepted
    # for a hand-run, but the probe does not use it: an argv secret is readable
    # from the process table and is captured by process-creation auditing.
    parser.add_argument("--password", default = os.environ.get("SAC_PROBE_STUDIO_PASSWORD"))
    parser.add_argument(
        "--home",
        default = default_studio_home(),
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

    token = authenticate(base_url, resolve_studio_home(args.home), args.password)
    print("authenticated")

    # Which llama-server Studio will run, in its own precedence: a folder
    # selected in Studio's settings is visible only here, and the inventory
    # must be of the build the scenario drives.
    status, body = _request(base_url, "GET", "/api/settings/llama-cpp-path", token = token)
    runtime = (
        body
        if status == 200 and isinstance(body, dict)
        else {"error": f"{status}: {str(body)[:200]}"}
    )
    (out_dir / "runtime-selection.json").write_text(json.dumps(runtime, indent = 2), encoding = "utf-8")
    print(
        f"runtime: {runtime.get('resolved_binary') or runtime.get('error') or 'unresolved'} ({runtime.get('source')})"
    )

    poller = StatusPoller(base_url, token, args.poll_seconds)
    poller.start()

    repo, variant = split_model_ref(args.model)
    results: dict[str, Any] = {
        "model": args.model,
        "model_path": repo,
        "gguf_variant": variant,
        "port": port,
        "runtime": runtime,
        "steps": {},
    }
    load_payload: dict[str, Any] = {"model_path": repo}
    if variant:
        load_payload["gguf_variant"] = variant
    try:
        # A model already resident makes /load a no-op ("already_loaded"), and
        # then no PE is loaded inside the evidence window: the absence of
        # events would read as an allow. Evict first; a 4xx here just means
        # nothing was resident.
        status, body = _request(
            base_url,
            "POST",
            "/api/inference/unload",
            {"model_path": repo},
            token = token,
            timeout = 300,
        )
        results["evicted_before_load"] = {"status": status, "body": str(body)[:200]}

        print(f"loading {args.model} ...")
        status, body = _request(
            base_url,
            "POST",
            "/api/inference/load",
            load_payload,
            token = token,
            timeout = 1800,
        )
        load_error = padded_route_failure(status, body)
        if load_error is None and isinstance(body, dict) and body.get("status") == "already_loaded":
            load_error = (
                "the model was already resident, so this load started no llama-server and "
                "loaded no PE; the evidence window contains nothing for it"
            )
        results["steps"]["load"] = {
            "status": status,
            "ok": load_error is None,
            "error": load_error,
        }
        if load_error is not None:
            # This is the interesting failure. A blocked llama-server shows up
            # here, and the backend log carries the code integrity reason. A
            # slow download followed by the refusal arrives as a 200 whose
            # body carries the error, which is why the status alone is not read.
            print(f"  load FAILED ({status}): {load_error[:400]}")
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
            unload_error = padded_route_failure(status, body)
            results["steps"]["unload"] = {
                "status": status,
                "ok": unload_error is None,
                "error": unload_error,
            }
            print(
                "  unloaded"
                if unload_error is None
                else f"  unload FAILED ({status}): {unload_error[:200]}"
            )
    finally:
        poller.stop()
        # A read still in flight is abandoned at STATUS_READ_TIMEOUT_S like the
        # frontend's; wait that out rather than discard the daemon thread with
        # its record.
        poller.join(timeout = STATUS_READ_TIMEOUT_S + 15)

    durations = poller.durations
    in_flight = poller.in_flight_ms() if poller.is_alive() else None
    stalls = poller.stalls_ms()
    results["status_poll"] = {
        "count": len(durations),
        "in_flight_ms": None if in_flight is None else round(in_flight, 1),
        "max_ms": round(max(durations), 1) if durations else None,
        "median_ms": round(statistics.median(durations), 1) if durations else None,
        # Reads the frontend would have abandoned, and how long each run of
        # them lasted. The watchdog kills the backend after roughly 75s of
        # unanswered health checks, so a stall that long is the reported
        # failure mode.
        "abandoned_reads": sum(1 for _, _, t in poller.polls if t),
        "stalls_ms": [round(x, 1) for x in stalls],
        "over_10s": sum(1 for d in durations if d >= 10_000),
        "over_75s": sum(1 for x in stalls if x >= 75_000),
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
        f"{poll['abandoned_reads']} abandoned at {STATUS_READ_TIMEOUT_S:.0f}s, "
        f"{poll['over_75s']} stall(s) over 75s"
    )
    print(f"written to {path}")

    # Nonzero when the runtime did not work, so the caller can tell the cells of
    # the matrix apart without parsing the JSON.
    return 0 if all(s.get("ok") for s in results["steps"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

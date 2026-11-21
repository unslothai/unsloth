# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The slice of lemond's API Studio uses, served from Studio's Python over AMD-signed flm.exe
(lemond.exe is unsigned, so Smart App Control may refuse it). Started with lemond's arguments."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs

import httpx

# FastFlowLM modes with no chat endpoint; lemond lists every other model as chat too.
_NOT_CHAT = frozenset({"embeddings", "transcription", "realtime-transcription"})
_HIDDEN = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # CREATE_NO_WINDOW
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "host",
    "authorization",
}


def model_id(tag: str) -> str:
    return tag.replace(":", "-") + "-FLM"


def _context_window(folder: Optional[Path]) -> Optional[int]:
    # lemond reports the downloaded config's max_position_embeddings, else nothing.
    try:
        value = json.loads((folder / "config.json").read_text(encoding = "utf-8"))
        value = value.get("max_position_embeddings")
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return value if isinstance(value, int) and value > 0 else None


def catalog_row(entry: dict[str, Any], models_dir: Optional[Path] = None) -> dict[str, Any]:
    tag = str(entry.get("name") or entry.get("model") or "")
    labels = [str(label) for label in entry.get("label") or []]
    if not _NOT_CHAT.intersection(labels):
        labels.append("chat")
    folder = str(entry.get("url") or "").rstrip("/").rpartition("/")[2]
    installed = bool(entry.get("installed"))
    return {
        "id": model_id(tag),
        "checkpoint": tag,
        "labels": labels,
        "size": entry.get("footprint"),
        "downloaded": installed,
        "recipe": "flm",
        "max_context_window": _context_window(models_dir / folder)
        if installed and folder and models_dir
        else None,
    }


def json_object(output: str, key: str) -> Optional[dict[str, Any]]:
    """The JSON object holding ``key`` in flm's output. ``Flm.run`` joins stdout and stderr,
    so a line flm prints before or after the object is skipped instead of failing the parse."""
    decoder = json.JSONDecoder()
    first = output.find("{")
    # The object starts at the first brace or, after a status line, at the start of a line.
    starts = sorted({first, *(m.start() for m in re.finditer(r"^\{", output, re.M))} - {-1})
    for start in starts:
        try:
            value, _end = decoder.raw_decode(output, start)
        except ValueError:
            continue
        if isinstance(value, dict) and key in value:
            return value
    return None


def log(message: str, tag: str = "FlmHost") -> None:
    """A line in lemond's format. Studio keeps (Process) lines only while a load is open, as
    flm echoes prompts."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    try:
        print(f"{stamp} [Info] ({tag}) {message}", flush = True)
    except (OSError, ValueError):
        pass


def _die_with_spawner() -> None:
    """Linux: SIGKILL flm serve when its spawning thread's process (flm_host) dies."""
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno = True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


class _Spawner:
    """Starts flm serve from one thread that lives as long as flm_host. Linux delivers
    PDEATHSIG when the spawning thread exits, so a request thread would kill it at once."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target = self._loop, daemon = True, name = "flm-spawner").start()

    def _loop(self) -> None:
        while True:
            fn, box, done = self._jobs.get()
            try:
                box.append((True, fn()))
            except BaseException as exc:  # noqa: BLE001 -- re-raised in the caller
                box.append((False, exc))
            done.set()

    def popen(self, cmd: list[str], **kwargs: Any) -> subprocess.Popen:
        if sys.platform != "linux":
            return subprocess.Popen(cmd, **kwargs)
        box: list = []
        done = threading.Event()
        self._jobs.put(
            (lambda: subprocess.Popen(cmd, preexec_fn = _die_with_spawner, **kwargs), box, done)
        )
        done.wait()
        ok, value = box[0]
        if not ok:
            raise value
        return value


class Flm:
    """flm under ``cache_dir``, run with the FLM_* environment Studio set for lemond."""

    def __init__(self, cache_dir: Path) -> None:
        # Linux ships an `flm` wrapper that puts the bundled XRT libraries on the library path.
        binary = (
            cache_dir / "bin" / "flm" / "npu" / ("flm.exe" if sys.platform == "win32" else "flm")
        )
        if not binary.is_file():
            raise SystemExit(f"FastFlowLM is not installed under {cache_dir}.")
        self.binary = binary
        self.models_dir = Path(os.environ.get("FLM_MODEL_PATH") or cache_dir / "flm") / "models"

    def run(
        self,
        *args: str,
        timeout: float = 600.0,
        stdin: str = "",
    ) -> tuple[int, str]:
        try:
            done = subprocess.run(
                [str(self.binary), *args],
                input = stdin,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = timeout,
                **_HIDDEN,
            )
        except OSError as exc:
            # Windows refusing flm.exe (Smart App Control) is a failed run with its reason.
            return 1, f"[ERROR] {exc}"
        output = (done.stdout or "") + (done.stderr or "")
        if done.returncode and not output.strip():
            # A refused DLL (Smart App Control) exits 0xC0E90002 before printing anything.
            output = f"[ERROR] exit code {done.returncode} (0x{done.returncode & 0xFFFFFFFF:08X})"
        # flm exits 0 after printing [ERROR], so the text decides.
        return (done.returncode or (1 if "[ERROR]" in output else 0)), output

    def catalog(self) -> list[dict[str, Any]]:
        code, output = self.run("list", "--json", timeout = 120.0)
        report = None if code else json_object(output, "models")
        if report is None:
            raise RuntimeError(f"flm list failed: {output.strip()[-300:]}")
        models = report.get("models") or []
        return [catalog_row(entry, self.models_dir) for entry in models if isinstance(entry, dict)]


class Serving:
    """The one `flm serve` child, as lemond holds one model (max_loaded_models = 1)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process: Optional[subprocess.Popen] = None
        self.model_id: Optional[str] = None
        self.tag: Optional[str] = None
        self.ctx: Optional[int] = None
        self.port: Optional[int] = None

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        with self.lock:
            process, self.process = self.process, None
            self.model_id = self.tag = self.ctx = self.port = None
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout = 15)
            except subprocess.TimeoutExpired:
                pass


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _relay(process: subprocess.Popen, tail: list[str]) -> None:
    """flm's output on ours, where Studio reads the reason a load failed. Never stops reading:
    a full pipe would block flm serve mid-reply."""
    for line in process.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        tail.append(line)
        del tail[:-80]
        log(line, "Process")


def make_handler(
    flm: Flm,
    serving: Serving,
    api_key: str,
    spawner: Optional[_Spawner] = None,
):
    spawner = spawner or _Spawner()
    # Pooled, not per request: a new client (SSL context, connect) delayed every first token.
    # trust_env off keeps loopback calls away from HTTP_PROXY and the Windows system proxy.
    client = httpx.Client(trust_env = False, timeout = httpx.Timeout(600.0, connect = 10.0))

    def find(model_name: str) -> dict[str, Any]:
        for row in flm.catalog():
            if model_name in (row["id"], row["checkpoint"]):
                return row
        raise KeyError(f"{model_name} is not a FastFlowLM model.")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True  # small SSE chunks must not wait on delayed ACKs

        def log_message(self, fmt, *args) -> None:
            pass

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "server_error"}})

        def _authorized(self) -> bool:
            if self.path == "/live":
                return True
            sent = self.headers.get("Authorization", "").encode()
            if api_key and not hmac.compare_digest(sent, f"Bearer {api_key}".encode()):
                self._error(401, "Invalid API key.")
                return False
            return True

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw) if raw else {}

        def do_GET(self) -> None:
            if not self._authorized():
                return
            path = self.path.split("?", 1)[0]
            if path == "/live":
                self._json(200, {"status": "ok"})
            elif path == "/v1/health":
                loaded = []
                with serving.lock:
                    if serving.alive():
                        loaded.append({
                            "model_name": serving.model_id, "checkpoint": serving.tag, "device": "npu",
                            "recipe": "flm", "loaded": True, "recipe_options": {"ctx_size": serving.ctx},
                        })  # fmt: skip
                self._json(200, {"status": "ok", "all_models_loaded": loaded})
            elif path == "/v1/models":
                query = parse_qs(self.path.partition("?")[2])
                show_all = (query.get("show_all") or [""])[0].lower() in ("true", "1")
                try:
                    rows = flm.catalog()
                    # Like lemond: downloaded models only unless show_all.
                    rows = rows if show_all else [r for r in rows if r["downloaded"]]
                    self._json(200, {"object": "list", "data": rows})
                except Exception as exc:  # noqa: BLE001 -- reported to Studio
                    self._error(500, str(exc))
            elif path.startswith("/v1/models/") and path.endswith("/options"):
                # Studio reads flm_args only; loads never save options.
                self._json(200, {"effective": {"flm_args": "", "ctx_size": -1}, "saved": {}})
            else:
                self._proxy("GET")

        def do_POST(self) -> None:
            if not self._authorized():
                return
            path = self.path.split("?", 1)[0]
            routes = {
                "/internal/set": lambda body: self._json(200, {}),
                "/v1/install": self._install,
                "/v1/pull": self._pull,
                "/v1/load": self._load,
                "/v1/unload": self._unload,
                "/v1/delete": self._delete,
            }
            if path not in routes:
                self._proxy("POST")
                return
            try:
                body = self._body()
            except ValueError:
                self._error(400, "Invalid JSON.")
                return
            routes[path](body if isinstance(body, dict) else {})

        def _install(self, body: dict[str, Any]) -> None:
            # Studio installs the pinned, signed FastFlowLM itself; only report what is here.
            code, output = flm.run("version", "--json", timeout = 30.0)
            if code:
                self._error(500, f"FastFlowLM does not run: {output.strip()[-300:]}")
            else:
                self._json(200, {"status": "success"})

        def _pull(self, body: dict[str, Any]) -> None:
            try:
                row = find(str(body.get("model_name") or ""))
            except Exception as exc:  # noqa: BLE001
                self._error(404, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            code, output = flm.run("pull", row["checkpoint"], timeout = 24 * 3600.0)
            for line in output.splitlines()[-20:]:
                log(line, "Process")
            if code:
                event = {
                    "error": output.strip().splitlines()[-1]
                    if output.strip()
                    else "flm pull failed"
                }
                self.wfile.write(f"event: error\ndata: {json.dumps(event)}\n\n".encode())
            else:
                self.wfile.write(
                    f"event: complete\ndata: {json.dumps({'status': 'success', 'model': row['id']})}\n\n".encode()
                )
            self.wfile.flush()
            self.close_connection = True

        def _load(self, body: dict[str, Any]) -> None:
            name = str(body.get("model_name") or "")
            try:
                row = find(name)
            except Exception as exc:  # noqa: BLE001
                self._error(404, str(exc))
                return
            ctx = (
                body.get("ctx_size")
                if isinstance(body.get("ctx_size"), int)
                else row.get("max_context_window")
            )
            port = _free_port()
            cmd = [
                str(flm.binary),
                "serve",
                row["checkpoint"],
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                "--quiet",
            ]
            if ctx:
                cmd += ["--ctx-len", str(ctx)]
            extra = body.get("flm_args")
            if isinstance(extra, str) and extra.strip():
                cmd += shlex.split(extra, posix = sys.platform != "win32")
            serving.stop()
            log(f"Starting: {' '.join(cmd)}")
            try:
                process = spawner.popen(
                    cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.DEVNULL,
                    text = True, encoding = "utf-8", errors = "replace", **_HIDDEN,
                )  # fmt: skip
            except OSError as exc:
                self._error(
                    500, f"Failed to load model '{row['id']}': FastFlowLM failed to start: {exc}"
                )
                return
            tail: list[str] = []
            threading.Thread(target = _relay, args = (process, tail), daemon = True).start()
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    time.sleep(0.2)
                    code = process.returncode
                    reason = next(
                        (l for l in reversed(tail) if l.strip()),
                        f"exit code {code} (0x{code & 0xFFFFFFFF:08X})",
                    )
                    self._error(
                        500,
                        f"Failed to load model '{row['id']}': FastFlowLM failed to start: {reason}",
                    )
                    return
                try:
                    if (
                        client.get(f"http://127.0.0.1:{port}/v1/models", timeout = 2.0).status_code
                        == 200
                    ):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                process.kill()
                self._error(
                    500, f"Failed to load model '{row['id']}': FastFlowLM did not start in time"
                )
                return
            with serving.lock:
                serving.process, serving.model_id, serving.tag = (
                    process,
                    row["id"],
                    row["checkpoint"],
                )
                serving.ctx, serving.port = ctx, port
            self._json(200, {"status": "success", "model_name": row["id"]})

        def _unload(self, body: dict[str, Any]) -> None:
            serving.stop()
            self._json(200, {"status": "success"})

        def _delete(self, body: dict[str, Any]) -> None:
            try:
                row = find(str(body.get("model_name") or ""))
            except Exception as exc:  # noqa: BLE001
                self._error(422, str(exc))
                return
            if serving.model_id == row["id"]:
                serving.stop()
            code, output = flm.run("remove", row["checkpoint"], stdin = "y\n", timeout = 300.0)
            if code:
                self._error(500, output.strip()[-300:] or "flm remove failed")
            else:
                self._json(200, {"status": "success"})

        def _proxy(self, method: str) -> None:
            with serving.lock:
                port, tag = (serving.port, serving.tag) if serving.alive() else (None, None)
            if port is None:
                self._error(
                    404 if method == "GET" else 503,
                    f"No route {self.path}, and no model is loaded.",
                )
                return
            raw = (
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if method == "POST"
                else b""
            )
            if raw and "json" in (self.headers.get("Content-Type") or ""):
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and "model" in payload:
                        payload["model"] = tag  # flm serve names the model by its tag
                        raw = json.dumps(payload).encode()
                except ValueError:
                    pass
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
            try:
                with client.stream(method, f"http://127.0.0.1:{port}{self.path}", content = raw or None,
                                   headers = headers) as upstream:  # fmt: skip
                    self.send_response(upstream.status_code)
                    for key, value in upstream.headers.items():
                        if key.lower() not in _HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in upstream.iter_raw():
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except httpx.HTTPError as exc:
                self._error(502, f"FastFlowLM did not answer: {exc}")
            self.close_connection = True

    return Handler


def exit_with_parent(parent_pid: int, serving: Serving) -> None:
    """Stop flm serve and exit if Studio dies without stopping us. Linux needs no watcher:
    Studio's PDEATHSIG kills flm_host, and _Spawner's kills flm serve."""
    if sys.platform != "win32" or parent_pid <= 0:
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE

    def wait() -> None:
        if handle:
            kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        serving.stop()
        os._exit(0)

    threading.Thread(target = wait, daemon = True, name = "parent-watch").start()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("cache_dir", type = Path)
    parser.add_argument("config_dir", type = Path)
    parser.add_argument("--port", type = int, required = True)
    parser.add_argument("--host", default = "127.0.0.1")
    args, _ = parser.parse_known_args(
        argv
    )  # lemond-only flags (--no-broadcast, --log-file) are ignored
    # flm prints emoji; a cp1252 pipe would raise on them.
    sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
    flm = Flm(args.cache_dir)
    serving = Serving()
    exit_with_parent(os.getppid(), serving)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(flm, serving, os.environ.get("LEMONADE_API_KEY", ""))
    )
    server.daemon_threads = True
    log(f"FastFlowLM host (no Lemonade) on 127.0.0.1:{args.port} using {flm.binary}")
    try:
        server.serve_forever()
    finally:
        serving.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

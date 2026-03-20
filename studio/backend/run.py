# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Run script for Unsloth UI Backend.
Works independently and can be moved to any directory.
"""

import os
import sys

# Fix for Anaconda Python: its sys.version contains "| packaged by Anaconda, Inc. |"
# which breaks platform._sys_version() regex parsing. This affects attrs -> rich ->
# structlog import chain. CPython won't fix this (issue #102396), so we patch here
# before any library imports. See: https://github.com/python/cpython/issues/102396
if "|" in sys.version:
    import re
    import platform
    try:
        _clean = re.sub(r"\s*\|[^|]*\|\s*", " ", sys.version).strip()
        if "|" in _clean:
            # Unpaired pipes -- keep version number + everything from "(" onward
            _m = re.match(r"([\w.+]+)\s*", _clean)
            _p = _clean.find("(")
            if _m and _p > 0:
                _clean = _m.group(0) + _clean[_p:]
        _result = platform._sys_version(_clean)
        platform._sys_version_cache[sys.version] = _result
        del _clean, _result
    except Exception:
        pass

# Suppress annoying C-level dependency warnings globally (e.g. SwigPyPacked)
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from loggers import get_logger

logger = get_logger(__name__)


def _resolve_external_ip() -> str:
    """
    Resolve the machine's external IP address.

    Tries (in order):
    1. GCE metadata server (instant, works on Google Cloud VMs)
    2. ifconfig.me (works anywhere with internet)
    3. LAN IP via UDP socket trick (fallback)
    """
    import urllib.request
    import socket

    # 1. Try GCE metadata server (responds in <10ms on GCE, times out fast elsewhere)
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
            headers = {"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout = 1) as resp:
            ip = resp.read().decode().strip()
            if ip:
                return ip
    except Exception:
        pass

    # 2. Try public IP service
    try:
        with urllib.request.urlopen("https://ifconfig.me", timeout = 3) as resp:
            ip = resp.read().decode().strip()
            if ip:
                return ip
    except Exception:
        pass

    # 3. Fallback: LAN IP via UDP socket trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def _is_port_free(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port(host: str, start: int, max_attempts: int = 20) -> int:
    """Find a free port starting from `start`, trying up to max_attempts ports."""
    for offset in range(max_attempts):
        candidate = start + offset
        if _is_port_free(host, candidate):
            return candidate
    raise RuntimeError(
        f"Could not find a free port in range {start}-{start + max_attempts - 1}"
    )


def _graceful_shutdown(server = None):
    """Explicitly shut down all subprocess backends and the uvicorn server.

    Called from signal handlers to ensure child processes are cleaned up
    before the parent exits. This is critical on Windows where atexit
    handlers are unreliable after Ctrl+C.
    """
    logger.info("Graceful shutdown initiated — cleaning up subprocesses...")

    # 1. Shut down uvicorn server (releases the listening socket)
    if server is not None:
        server.should_exit = True

    # 2. Clean up inference subprocess (if instantiated)
    try:
        from core.inference.orchestrator import _inference_backend

        if _inference_backend is not None:
            _inference_backend._shutdown_subprocess(timeout = 5.0)
    except Exception as e:
        logger.warning("Error shutting down inference subprocess: %s", e)

    # 3. Clean up export subprocess (if instantiated)
    try:
        from core.export.orchestrator import _export_backend

        if _export_backend is not None:
            _export_backend._shutdown_subprocess(timeout = 5.0)
    except Exception as e:
        logger.warning("Error shutting down export subprocess: %s", e)

    # 4. Clean up training subprocess (if active)
    try:
        from core.training.training import _training_backend

        if _training_backend is not None:
            _training_backend.force_terminate()
    except Exception as e:
        logger.warning("Error shutting down training subprocess: %s", e)

    # 5. Kill llama-server subprocess (if loaded)
    try:
        from routes.inference import _llama_cpp_backend

        if _llama_cpp_backend is not None:
            _llama_cpp_backend._kill_process()
    except Exception as e:
        logger.warning("Error shutting down llama-server: %s", e)

    logger.info("All subprocesses cleaned up")


# The uvicorn server instance — set by run_server(), used by callers
# that need to tell the server to exit (e.g. signal handlers).
_server = None

# Shutdown event — used to wake the main loop on signal
_shutdown_event = None


def run_server(
    host: str = "0.0.0.0",
    port: int = 8888,
    frontend_path: Path = Path(__file__).resolve().parent.parent / "frontend" / "dist",
    silent: bool = False,
):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to (auto-increments if in use)
        frontend_path: Path to frontend build directory (optional)
        silent: Suppress startup messages

    Note:
        Signal handlers are NOT registered here so that embedders
        (e.g. Colab notebooks) keep their own interrupt semantics.
        Standalone callers should register handlers after calling this.
    """
    global _server, _shutdown_event

    import nest_asyncio

    nest_asyncio.apply()

    import asyncio
    from threading import Thread, Event
    import time
    import uvicorn

    from main import app, setup_frontend
    from utils.paths import ensure_studio_directories

    # Create all standard directories on startup
    ensure_studio_directories()

    # Auto-find free port if requested port is in use
    if not _is_port_free(host, port):
        original_port = port
        port = _find_free_port(host, port)
        if not silent:
            print(f"Port {original_port} is in use, using port {port} instead")

    # Setup frontend if path provided
    if frontend_path:
        if setup_frontend(app, frontend_path):
            if not silent:
                print(f"✅ Frontend loaded from {frontend_path}")
        else:
            if not silent:
                print(f"⚠️ Frontend not found at {frontend_path}")

    # Create the uvicorn server and expose it for signal handlers
    config = uvicorn.Config(
        app, host = host, port = port, log_level = "info", access_log = False
    )
    _server = uvicorn.Server(config)
    _shutdown_event = Event()

    # Run server in a daemon thread
    def _run():
        asyncio.run(_server.serve())

    thread = Thread(target = _run, daemon = True)
    thread.start()
    time.sleep(3)

    if not silent:
        display_host = _resolve_external_ip() if host == "0.0.0.0" else host

        print("")
        print("=" * 50)
        print(f"🦥 Open your web browser, and enter http://localhost:{port}")
        print("=" * 50)
        print("")
        print("=" * 50)
        print(f"🦥 Unsloth Studio is running on port {port}")
        print(f"   Local Access:          http://localhost:{port}")
        print(f"   Worldwide Web Address: http://{display_host}:{port}")
        print(f"   API:                   http://{display_host}:{port}/api")
        print(f"   Health:                http://{display_host}:{port}/api/health")
        print("=" * 50)

    return app


# For direct execution (also invoked by CLI via os.execvp / subprocess)
if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description = "Run Unsloth UI Backend server")
    parser.add_argument("--host", default = "0.0.0.0", help = "Host to bind to")
    parser.add_argument("--port", type = int, default = 8888, help = "Port to bind to")
    parser.add_argument(
        "--frontend",
        type = str,
        default = Path(__file__).resolve().parent.parent / "frontend" / "dist",
        help = "Path to frontend build",
    )
    parser.add_argument("--silent", action = "store_true", help = "Suppress output")

    args = parser.parse_args()

    kwargs = dict(host = args.host, port = args.port, silent = args.silent)
    if args.frontend is not None:
        kwargs["frontend_path"] = Path(args.frontend)
    run_server(**kwargs)

    # ── Signal handler — ensures subprocess cleanup on Ctrl+C ────
    def _signal_handler(signum, frame):
        _graceful_shutdown(_server)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # On Windows, some terminals send SIGBREAK for Ctrl+C / Ctrl+Break
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    # Keep running until shutdown signal.
    # NOTE: Event.wait() without a timeout blocks at the C level on Linux,
    # which prevents Python from delivering SIGINT (Ctrl+C).  Using a
    # short timeout in a loop lets the interpreter process pending signals.
    while not _shutdown_event.is_set():
        _shutdown_event.wait(timeout = 1)

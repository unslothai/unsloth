# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Run script for Unsloth UI Backend.
Works independently and can be moved to any directory.
"""

import os
import sys
from pathlib import Path

# Suppress annoying C-level dependency warnings globally (e.g. SwigPyPacked)
os.environ["PYTHONWARNINGS"] = "ignore"

# Add the backend directory to Python path early so local modules are importable
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Fix for Anaconda/conda-forge Python: seed platform._sys_version_cache before
# any library imports that trigger attrs -> rich -> structlog -> platform crash.
# See: https://github.com/python/cpython/issues/102396
import _platform_compat  # noqa: F401

from loggers import get_logger
from startup_banner import print_studio_access_banner

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


def _get_pid_on_port(port: int) -> "tuple[int, str] | None":
    """Return (pid, process_name) of the process listening on *port*, or None.

    Uses psutil when available.  Falls back gracefully to None so callers
    can still report the port conflict without process details.

    Works on Windows, macOS, and Linux wherever psutil is installed.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        for conn in psutil.net_connections(kind = "tcp"):
            if conn.status == "LISTEN" and conn.laddr.port == port:
                if conn.pid is None:
                    return None
                try:
                    proc = psutil.Process(conn.pid)
                    return (conn.pid, proc.name())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return (conn.pid, "<unknown>")
    except (psutil.AccessDenied, OSError) as e:
        # psutil.net_connections() needs elevated privileges on some platforms
        logger.debug("Failed to scan network connections for port %s: %s", port, e)
    return None


def _is_port_free(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    When *host* is ``0.0.0.0`` (wildcard), we also check whether anything
    is already listening on ``127.0.0.1`` (and ``::1`` when IPv6 is
    available).  An SSH tunnel or similar process may hold the loopback
    address while our wildcard bind still succeeds, making Unsloth Studio
    unreachable via ``localhost``.

    Works on Windows, macOS, and Linux.
    """
    import socket

    # 1. Can we bind to the requested address?
    #    Use getaddrinfo so both IPv4 ("0.0.0.0") and IPv6 ("::") hosts
    #    resolve to the correct address family automatically.
    try:
        addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        family, socktype, proto, _, sockaddr = addr_info[0]
        with socket.socket(family, socktype, proto) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
    except OSError:
        return False

    # 2. When binding to all interfaces, verify that localhost is not
    #    already claimed by another process (e.g. an SSH -L tunnel).
    #    We attempt a TCP connect -- if it succeeds something is listening.
    if host in ("0.0.0.0", "::"):
        for loopback, family in [
            ("127.0.0.1", socket.AF_INET),
            ("::1", socket.AF_INET6),
        ]:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    if s.connect_ex((loopback, port)) == 0:
                        # Connection succeeded -- port is taken on loopback
                        return False
            except OSError:
                # IPv6 disabled or other OS-level restriction -- skip
                continue

    return True


def _find_free_port(host: str, start: int, max_attempts: int = 20) -> int:
    """Find a free port starting from `start`, trying up to max_attempts ports."""
    for offset in range(max_attempts):
        candidate = start + offset
        if _is_port_free(host, candidate):
            return candidate
    raise RuntimeError(
        f"Could not find a free port in range {start}-{start + max_attempts - 1}"
    )


from utils.paths.storage_roots import studio_root as _studio_root

_PID_FILE = _studio_root() / "studio.pid"

# Direct backend launches bypass the CLI's env re-export; do it here for
# real custom roots so unsloth-zoo's import-time LLAMA_CPP_DEFAULT_DIR
# picks up the custom build. Skip for legacy-default to avoid flipping
# default-mode installs into env-override.
try:
    _LEGACY_STUDIO_ROOT = (Path.home() / ".unsloth" / "studio").resolve()
except (OSError, ValueError):
    _LEGACY_STUDIO_ROOT = Path.home() / ".unsloth" / "studio"
try:
    _STUDIO_ROOT_RESOLVED = _studio_root().resolve()
except (OSError, ValueError):
    _STUDIO_ROOT_RESOLVED = _studio_root()
if _STUDIO_ROOT_RESOLVED != _LEGACY_STUDIO_ROOT:
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(_STUDIO_ROOT_RESOLVED)
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_STUDIO_ROOT_RESOLVED / "llama.cpp")


def _write_pid_file():
    """Write the current process PID to the studio PID file."""
    try:
        _PID_FILE.parent.mkdir(parents = True, exist_ok = True)
        _PID_FILE.write_text(str(os.getpid()))
    except OSError:
        pass


def _remove_pid_file():
    """Remove the PID file if it belongs to this process."""
    try:
        if _PID_FILE.is_file():
            stored = _PID_FILE.read_text().strip()
            if stored == str(os.getpid()):
                _PID_FILE.unlink(missing_ok = True)
    except OSError:
        pass


def _graceful_shutdown(server = None):
    """Explicitly shut down all subprocess backends and the uvicorn server.

    Called from signal handlers to ensure child processes are cleaned up
    before the parent exits. This is critical on Windows where atexit
    handlers are unreliable after Ctrl+C.
    """
    _remove_pid_file()
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


# The uvicorn server instance -- set by run_server(), used by callers
# that need to tell the server to exit (e.g. signal handlers).
_server = None

# Shutdown event -- used to wake the main loop on signal
_shutdown_event = None


def run_server(
    host: str = "127.0.0.1",
    port: int = 8888,
    frontend_path: Path = Path(__file__).resolve().parent.parent / "frontend" / "dist",
    silent: bool = False,
    api_only: bool = False,
    llama_parallel_slots: int = 1,
):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to (auto-increments if in use)
        frontend_path: Path to frontend build directory (optional)
        silent: Suppress startup messages
        api_only: Run API server only, no frontend serving (for Tauri desktop app)
        llama_parallel_slots: Number of parallel slots for llama-server

    Note:
        Signal handlers are NOT registered here so that embedders
        (e.g. Colab notebooks) keep their own interrupt semantics.
        Standalone callers should register handlers after calling this.
    """
    global _server, _shutdown_event

    # On Windows the default console encoding (cp1252) cannot encode emoji.
    # Reconfigure stdout to UTF-8 so startup messages do not crash the server.
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass

    # Set env var BEFORE importing main so CORS middleware picks it up
    if api_only:
        os.environ["UNSLOTH_API_ONLY"] = "1"

    import nest_asyncio

    nest_asyncio.apply()

    import asyncio
    from threading import Thread, Event
    import uvicorn

    from main import app, setup_frontend
    from utils.paths import ensure_studio_directories

    # Create all standard directories on startup
    ensure_studio_directories()

    # Auto-find free port if requested port is in use
    if not _is_port_free(host, port):
        original_port = port
        blocker = _get_pid_on_port(port)
        port = _find_free_port(host, port + 1)
        if not silent:
            print("")
            print("=" * 50)
            if blocker:
                pid, name = blocker
                print(
                    f"Port {original_port} is already in use by " f"{name} (PID {pid})."
                )
            else:
                print(f"Port {original_port} is already in use.")
            print(f"Unsloth Studio will use port {port} instead.")
            print(f"Open http://localhost:{port} in your browser.")
            print("=" * 50)
            print("")

    # Setup frontend if path provided (skip in api-only mode)
    if frontend_path and not api_only:
        if setup_frontend(app, frontend_path):
            if not silent:
                print(f"[OK] Frontend loaded from {frontend_path}")
        else:
            if not silent:
                print(f"[WARNING] Frontend not found at {frontend_path}")

    ready_event = Event()
    startup_failed = Event()
    startup_errors = []

    class _ReadyServer(uvicorn.Server):
        async def startup(self, *args, **kwargs):
            await super().startup(*args, **kwargs)
            if getattr(self, "started", False) and not self.should_exit:
                ready_event.set()

    # server_header=False suppresses uvicorn's "Server: uvicorn"; SecurityHeadersMiddleware sets its own.
    config = uvicorn.Config(
        app,
        host = host,
        port = port,
        log_level = "info",
        access_log = False,
        server_header = False,
    )
    _server = _ReadyServer(config)
    _shutdown_event = Event()

    # Expose the actual bound port so request-handling code can build
    # loopback URLs that point at the real backend, not whatever port a
    # reverse proxy or tunnel exposed in the request URL. Only publish
    # an explicit value when we know the concrete port; for ephemeral
    # binds (port==0) leave it unset and let request handlers fall back
    # to the ASGI request scope or request.base_url.
    app.state.server_port = port if port and port > 0 else None
    app.state.llama_parallel_slots = llama_parallel_slots

    # Expose a shutdown callable via app.state before the server can accept
    # requests so /api/shutdown is available as soon as readiness is published.
    def _trigger_shutdown():
        _graceful_shutdown(_server)
        if _shutdown_event is not None:
            _shutdown_event.set()

    app.state.trigger_shutdown = _trigger_shutdown

    # Run server in a daemon thread
    def _run():
        try:
            asyncio.run(_server.serve())
        except BaseException as exc:
            startup_errors.append(exc)
            startup_failed.set()
        finally:
            if not ready_event.is_set():
                startup_failed.set()

    thread = Thread(target = _run, daemon = True)
    thread.start()

    # Wait until uvicorn has completed lifespan startup and bound sockets, or
    # until the server exits/fails before startup. This intentionally has no
    # correctness deadline: a slow but live startup should remain in progress.
    try:
        while not ready_event.is_set():
            if startup_failed.is_set() or not thread.is_alive():
                if startup_errors:
                    raise RuntimeError(
                        "Uvicorn server failed before startup completed"
                    ) from startup_errors[0]
                raise RuntimeError("Uvicorn server exited before startup completed")
            ready_event.wait(timeout = 0.1)
    except KeyboardInterrupt:
        _graceful_shutdown(_server)
        _shutdown_event.set()
        raise

    _write_pid_file()
    import atexit

    atexit.register(_remove_pid_file)

    # Output port for Tauri to parse when in api-only mode. Emit only after
    # uvicorn sockets are bound and FastAPI lifespan/startup has completed.
    if api_only:
        print(f"TAURI_PORT={port}", flush = True)

    if not silent:
        display_host = _resolve_external_ip() if host == "0.0.0.0" else host
        print_studio_access_banner(
            port = port,
            bind_host = host,
            display_host = display_host,
        )

    return app


# For direct execution (also invoked by CLI via os.execvp / subprocess)
if __name__ == "__main__":
    import argparse
    import signal
    import traceback

    # Ensure stderr can handle Unicode on Windows (tracebacks with non-ASCII paths)
    if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description = "Run Unsloth UI Backend server")
    parser.add_argument(
        "--host",
        default = "127.0.0.1",
        help = "Host to bind to (default: 127.0.0.1; use 0.0.0.0 for network/cloud access)",
    )
    parser.add_argument("--port", type = int, default = 8888, help = "Port to bind to")
    parser.add_argument(
        "--frontend",
        type = str,
        default = Path(__file__).resolve().parent.parent / "frontend" / "dist",
        help = "Path to frontend build",
    )
    parser.add_argument("--silent", action = "store_true", help = "Suppress output")
    parser.add_argument(
        "--api-only",
        action = "store_true",
        help = "API server only, no frontend (for Tauri)",
    )

    args = parser.parse_args()

    kwargs = dict(
        host = args.host, port = args.port, silent = args.silent, api_only = args.api_only
    )
    if args.frontend is not None:
        kwargs["frontend_path"] = Path(args.frontend)

    try:
        run_server(**kwargs)
    except Exception:
        sys.stderr.write("\n")
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write("ERROR: Unsloth Studio failed to start.\n")
        sys.stderr.write("=" * 60 + "\n")
        traceback.print_exc(file = sys.stderr)
        sys.stderr.write("\n")
        sys.stderr.write(
            "If a package is missing, try re-running: unsloth studio setup\n"
        )
        sys.stderr.flush()
        sys.exit(1)

    # Signal handler -- ensures subprocess cleanup on Ctrl+C
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

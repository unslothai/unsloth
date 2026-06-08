# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run script for Unsloth UI Backend.

Self-contained; can be moved to any directory.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Suppress C-level dependency warnings globally (e.g. SwigPyPacked).
os.environ["PYTHONWARNINGS"] = "ignore"

# Add the backend dir to sys.path early so local modules import.
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.cpu_threads import configure_cpu_threads

try:
    configure_cpu_threads()
except ValueError as exc:
    configured = os.environ.get("UNSLOTH_CPU_THREADS")
    raise SystemExit(f"Error: Invalid UNSLOTH_CPU_THREADS value {configured!r}: {exc}") from None

# Anaconda/conda-forge Python: seed platform._sys_version_cache before
# imports that trigger attrs -> rich -> structlog -> platform crash.
# See: https://github.com/python/cpython/issues/102396
import _platform_compat  # noqa: F401

from loggers import get_logger
from startup_banner import print_studio_access_banner, print_studio_stop_hint

logger = get_logger(__name__)


def _resolve_external_ip() -> str:
    """Resolve the machine's external IP address.

    Tries, in order:
    1. GCE metadata server (instant on Google Cloud VMs)
    2. ifconfig.me (anywhere with internet)
    3. LAN IP via UDP socket trick (fallback)
    """
    import urllib.request
    import socket

    # 1. GCE metadata server (<10ms on GCE, times out fast elsewhere).
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

    # 2. Public IP service.
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


def _install_uvicorn_startup_log_rewrite(bind_host: str, display_host: str) -> None:
    """Rewrite Uvicorn's startup log line: swap the wildcard bind for the
    externally-reachable address, replace the CTRL+C suffix with our
    Mac-aware stop hint, and rename the prefix to "Unsloth Studio running
    on"."""
    import logging
    import re

    rewrite_host = (
        bind_host in ("0.0.0.0", "::") and bool(display_host) and display_host != bind_host
    )
    new_suffix = "(To stop: press Ctrl+C -- on macOS, Control+C not Command+C)"
    old_suffix_re = re.compile(r"\(Press CTRL\+C to quit\)")
    old_prefix = "Uvicorn running on "
    new_prefix = "Unsloth Studio running on "

    def _rewrite(text: str) -> str:
        if text.startswith(old_prefix):
            text = new_prefix + text[len(old_prefix) :]
        return old_suffix_re.sub(new_suffix, text)

    class _UvicornStartupRewrite(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.msg if isinstance(record.msg, str) else ""
                if (
                    msg.startswith(old_prefix)
                    and isinstance(record.args, tuple)
                    and len(record.args) >= 3
                ):
                    if rewrite_host and record.args[1] == bind_host:
                        record.args = (
                            record.args[0],
                            display_host,
                            record.args[2],
                            *record.args[3:],
                        )
                    record.msg = _rewrite(msg)
                    cmsg = getattr(record, "color_message", None)
                    if isinstance(cmsg, str):
                        record.color_message = _rewrite(cmsg)
            except Exception:
                pass
            return True

    f = _UvicornStartupRewrite()
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).addFilter(f)


def _local_port_open(
    host: str,
    port: int,
    timeout: float = 1.0,
) -> bool:
    """True iff a TCP connection to (host, port) succeeds within timeout."""
    import socket
    try:
        with socket.create_connection((host, port), timeout = timeout):
            return True
    except OSError:
        return False


def _working_local_url(port: int) -> "str | None":
    """A working loopback URL on this machine, or None if neither
    127.0.0.1 nor ::1 responds. Fallback when external reachability fails."""
    if _local_port_open("127.0.0.1", port):
        return f"http://127.0.0.1:{port}"
    if _local_port_open("::1", port):
        return f"http://[::1]:{port}"
    return None


def _localhost_ipv6_mismatch_url(bind_host: str, port: int) -> "str | None":
    """Return the IPv4 loopback URL when localhost won't reach 127.0.0.1.

    Local Studio binds to 127.0.0.1. Where localhost resolves to IPv6
    only (::1), http://localhost:<port> fails -- or worse, hits a
    different process on ::1 -- even though http://127.0.0.1:<port> works.
    Return the IPv4 URL so the caller can tell the user what to open.
    """
    import socket

    if bind_host != "127.0.0.1" or not port or port <= 0:
        return None

    ipv4_url = f"http://127.0.0.1:{port}"

    # Only warn once Studio is confirmed answering on IPv4 loopback.
    if _working_local_url(port) != ipv4_url:
        return None

    try:
        addr_info = socket.getaddrinfo("localhost", port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except Exception:
        return None

    if not addr_info:
        return None

    has_ipv4_loopback = False
    has_ipv6_loopback = False
    for family, _, _, _, sockaddr in addr_info:
        if family == socket.AF_INET and sockaddr and sockaddr[0] == "127.0.0.1":
            has_ipv4_loopback = True
        elif family == socket.AF_INET6 and sockaddr:
            host = sockaddr[0].split("%", 1)[0]
            if host == "::1":
                has_ipv6_loopback = True

    # A connection to ::1 is NOT evidence Studio is reachable there:
    # Studio binds 127.0.0.1 only, so anything on ::1 is a different
    # process -- exactly when to steer the user to 127.0.0.1. Dual-stack
    # localhost is fine (browsers fall back to 127.0.0.1 when ::1
    # refuses), so only the IPv6-only case strands the user.
    if has_ipv6_loopback and not has_ipv4_loopback:
        return ipv4_url
    return None


def _stdout_color_ok() -> bool:
    """Whether to emit ANSI color codes on stdout. Mirrors startup_banner."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _print_localhost_ipv6_mismatch_warning(local_url: str, port: int) -> None:
    """Warn that localhost points at ::1 while Studio is bound to 127.0.0.1."""
    use_color = _stdout_color_ok()
    warn_c = "\033[38;5;215;1m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    print(
        f"{warn_c}  Warning: localhost resolves to IPv6 (::1), but Unsloth "
        f"Studio is listening on 127.0.0.1 only. Open {local_url} instead of "
        f"http://localhost:{port}.{reset}",
        flush = True,
    )


def _verify_global_reachability(display_host: str, port: int) -> None:
    """Probe check-host.net to confirm display_host:port is reachable
    from the public internet. Synchronous so the caller can render output
    between the banner URL section and the trailing stop hint. Bounded at
    ~15s; failures are swallowed (verifier failing != Studio failing).
    Only meaningful when bound to a wildcard host."""
    import ipaddress
    import json
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    if not display_host or display_host in ("0.0.0.0", "::"):
        return

    use_color = _stdout_color_ok()
    dim = "\033[38;5;245m" if use_color else ""
    ok_c = "\033[38;5;120;1m" if use_color else ""
    err_c = "\033[38;5;203;1m" if use_color else ""
    warn_c = "\033[38;5;215;1m" if use_color else ""
    local_url_c = "\033[38;5;108;1m" if use_color else ""  # matches banner's URL color
    reset = "\033[0m" if use_color else ""

    url = f"http://{display_host}:{port}"

    # Private/loopback/link-local addresses aren't globally routable.
    try:
        addr = ipaddress.ip_address(display_host)
        if addr.is_loopback or addr.is_private or addr.is_link_local:
            print(
                f"{dim}  Note: {display_host} is a private/LAN address -- "
                f"reachable on this network only, not from the public internet."
                f"{reset}",
                flush = True,
            )
            return
    except ValueError:
        # Not an IP literal; probe by hostname.
        pass

    try:
        qs = urllib.parse.urlencode({"host": f"{display_host}:{port}", "max_nodes": 3})
        req = urllib.request.Request(
            f"https://check-host.net/check-tcp?{qs}",
            headers = {
                "Accept": "application/json",
                "User-Agent": "unsloth-studio-reachability/1",
            },
        )
        with urllib.request.urlopen(req, timeout = 5) as resp:
            init = json.loads(resp.read().decode("utf-8", errors = "replace"))
        req_id = init.get("request_id")
        if not req_id:
            return

        results = {}
        deadline = time.monotonic() + 15.0
        poll_req = urllib.request.Request(
            f"https://check-host.net/check-result/{req_id}",
            headers = {
                "Accept": "application/json",
                "User-Agent": "unsloth-studio-reachability/1",
            },
        )
        while time.monotonic() < deadline:
            time.sleep(1.5)
            try:
                with urllib.request.urlopen(poll_req, timeout = 5) as resp:
                    results = json.loads(resp.read().decode("utf-8", errors = "replace"))
            except Exception:
                continue
            if results and all(v is not None for v in results.values()):
                break
            # Two decisive nodes is enough; stop early.
            decisive = [
                v
                for v in results.values()
                if isinstance(v, list)
                and v
                and isinstance(v[0], dict)
                and ("time" in v[0] or "error" in v[0])
            ]
            if len(decisive) >= 2:
                break

        ok_nodes = err_nodes = 0
        for v in results.values():
            if not isinstance(v, list) or not v or not isinstance(v[0], dict):
                continue
            if "time" in v[0]:
                ok_nodes += 1
            elif "error" in v[0]:
                err_nodes += 1
        total = ok_nodes + err_nodes

        print("", flush = True)
        if ok_nodes:
            print(
                f"{ok_c}  Reachability check: {url}/ is reachable from the "
                f"public internet ({ok_nodes}/{total} probe nodes connected).{reset}",
                flush = True,
            )
        elif err_nodes:
            print(
                f"{err_c}  Reachability check: {url}/ is NOT reachable from "
                f"the public internet ({err_nodes}/{total} probe nodes failed).{reset}",
                flush = True,
            )
            print(f"{dim}    Common causes:{reset}", flush = True)
            print(
                f"{dim}      * AWS  -- the instance's Security Group doesn't "
                f"allow inbound TCP {port}.{reset}",
                flush = True,
            )
            print(
                f"{dim}      * GCP  -- no firewall rule allowing TCP {port} "
                f"for the instance's network tag.{reset}",
                flush = True,
            )
            print(
                f"{dim}      * Azure / other clouds -- equivalent NSG / "
                f"firewall rule missing.{reset}",
                flush = True,
            )
            print(
                f"{dim}      * Home -- your router isn't port-forwarding "
                f"{port} to this machine.{reset}",
                flush = True,
            )
            print(
                f"{dim}    Workaround that needs no firewall changes -- "
                f"SSH local-forward from your laptop:{reset}",
                flush = True,
            )
            print(
                f"{dim}        ssh -L {port}:localhost:{port} " f"<user>@{display_host}{reset}",
                flush = True,
            )
            print(
                f"{dim}    then open http://localhost:{port}/ in your browser.{reset}",
                flush = True,
            )
            # Only offer the local URL if loopback answers.
            local_url = _working_local_url(port)
            if local_url:
                print(
                    f"{local_url_c}  You can access Unsloth Studio locally "
                    f"in the meantime: {local_url}{reset}",
                    flush = True,
                )
        else:
            print(
                f"{warn_c}  Reachability check: probe nodes did not respond "
                f"in time -- could not verify {url}/.{reset}",
                flush = True,
            )
    except urllib.error.URLError:
        # Outbound HTTPS blocked; skip.
        pass
    except Exception:
        pass


def _emit_startup_output(host: str, port: int, display_host: str) -> None:
    """Print the access banner plus any post-startup warnings.

    Extracted from ``_run`` so the banner/warning wiring is testable. The
    ``localhost``-to-::1 mismatch warning and the wildcard reachability
    check are mutually exclusive (the mismatch helper returns None for any
    non-127.0.0.1 bind, and wildcard binds are never 127.0.0.1), so the
    trailing stop hint is emitted exactly once.
    """
    wildcard_bind = host in ("0.0.0.0", "::")
    localhost_mismatch_url = _localhost_ipv6_mismatch_url(host, port)
    # For wildcard binds, run the reachability check between the URL
    # section and the stop hint so the stop hint stays last.
    print_studio_access_banner(
        port = port,
        bind_host = host,
        display_host = display_host,
        include_stop_hint = not wildcard_bind and not localhost_mismatch_url,
    )
    if localhost_mismatch_url:
        _print_localhost_ipv6_mismatch_warning(localhost_mismatch_url, port)
        print_studio_stop_hint()
    elif wildcard_bind:
        _verify_global_reachability(display_host, port)
        print_studio_stop_hint()


def _get_pid_on_port(port: int) -> "tuple[int, str] | None":
    """Return (pid, process_name) listening on *port*, or None.

    Uses psutil when available, falling back to None so callers can still
    report the conflict without process details. Works on Windows, macOS,
    and Linux wherever psutil is installed.
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
        # net_connections() needs elevated privileges on some platforms.
        logger.debug("Failed to scan network connections for port %s: %s", port, e)
    return None


def _is_port_free(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    When *host* is ``0.0.0.0`` (wildcard), also check whether anything is
    already listening on ``127.0.0.1`` (and ``::1`` when IPv6 exists). An
    SSH tunnel may hold the loopback address while our wildcard bind
    succeeds, making Studio unreachable via ``localhost``.

    Works on Windows, macOS, and Linux.
    """
    import socket

    # 1. Can we bind to the requested address? getaddrinfo resolves both
    #    IPv4 ("0.0.0.0") and IPv6 ("::") to the right address family.
    try:
        addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        family, socktype, proto, _, sockaddr = addr_info[0]
        with socket.socket(family, socktype, proto) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
    except OSError:
        return False

    # 2. When binding to all interfaces, verify localhost isn't already
    #    claimed by another process (e.g. an SSH -L tunnel). A successful
    #    TCP connect means something is listening.
    if host in ("0.0.0.0", "::"):
        for loopback, family in [
            ("127.0.0.1", socket.AF_INET),
            ("::1", socket.AF_INET6),
        ]:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    if s.connect_ex((loopback, port)) == 0:
                        # Port is taken on loopback.
                        return False
            except OSError:
                # IPv6 disabled or other OS-level restriction -- skip.
                continue

    return True


def _find_free_port(
    host: str,
    start: int,
    max_attempts: int = 20,
) -> int:
    """Find a free port from `start`, trying up to max_attempts ports."""
    for offset in range(max_attempts):
        candidate = start + offset
        if _is_port_free(host, candidate):
            return candidate
    raise RuntimeError(f"Could not find a free port in range {start}-{start + max_attempts - 1}")


from utils.paths.storage_roots import studio_root as _studio_root

_PID_FILE = _studio_root() / "studio.pid"

# Direct backend launches bypass the CLI's env re-export; do it here for
# real custom roots so unsloth-zoo's import-time LLAMA_CPP_DEFAULT_DIR
# picks up the custom build. Skip legacy-default to avoid flipping
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
    """Shut down all subprocess backends and the uvicorn server.

    Called from signal handlers to clean up child processes before the
    parent exits. Critical on Windows where atexit handlers are
    unreliable after Ctrl+C.
    """
    _remove_pid_file()
    logger.info("Graceful shutdown initiated — cleaning up subprocesses...")

    # 1. Shut down uvicorn (releases the listening socket).
    if server is not None:
        server.should_exit = True

    # 2. Clean up inference subprocess (if instantiated).
    try:
        from core.inference.orchestrator import _inference_backend
        if _inference_backend is not None:
            _inference_backend._shutdown_subprocess(timeout = 5.0)
    except Exception as e:
        logger.warning("Error shutting down inference subprocess: %s", e)

    # 3. Clean up export subprocess (if instantiated).
    try:
        from core.export.orchestrator import _export_backend
        if _export_backend is not None:
            _export_backend._shutdown_subprocess(timeout = 5.0)
    except Exception as e:
        logger.warning("Error shutting down export subprocess: %s", e)

    # 4. Clean up training subprocess (if active).
    try:
        from core.training.training import _training_backend
        if _training_backend is not None:
            _training_backend.force_terminate()
    except Exception as e:
        logger.warning("Error shutting down training subprocess: %s", e)

    # 5. Kill llama-server subprocess (if loaded).
    try:
        from routes.inference import _llama_cpp_backend
        if _llama_cpp_backend is not None:
            _llama_cpp_backend._kill_process()
    except Exception as e:
        logger.warning("Error shutting down llama-server: %s", e)

    logger.info("All subprocesses cleaned up")


# The uvicorn server instance -- set by run_server(), used by callers
# that tell the server to exit (e.g. signal handlers).
_server = None

# Shutdown event -- wakes the main loop on signal.
_shutdown_event = None


_DEFAULT_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _iter_frontend_fallback_candidates() -> "list[Path]":
    """Yield `studio/frontend/dist` paths to try when the default is missing.

    Covers PATH-shadowed binaries whose __file__ resolves into a
    site-packages tree with no vite build (e.g. plain `pip install
    unsloth` from PyPI).
    """
    import ast
    import re

    out: list[Path] = []
    home_str = (
        os.environ.get("UNSLOTH_STUDIO_HOME")
        or os.environ.get("STUDIO_HOME")
        or str(Path.home() / ".unsloth" / "studio")
    )
    venv_dir = Path(home_str).expanduser() / "unsloth_studio"
    # Installer venv site-packages.
    for pattern in (
        "lib/python*/site-packages/studio/frontend/dist",
        "Lib/site-packages/studio/frontend/dist",
    ):
        out.extend(venv_dir.glob(pattern))
    # Editable source roots referenced from the installer venv.
    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for finder in sp.glob("__editable___*_finder.py"):
                try:
                    src = finder.read_text(encoding = "utf-8")
                except OSError:
                    continue
                # Tolerate single- or multi-line dict literals; [^}]*
                # still rejects nested dicts, which the setuptools
                # template never emits for editable installs.
                m = re.search(r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S)
                if not m:
                    continue
                try:
                    mapping = ast.literal_eval(m.group(1))
                except (SyntaxError, ValueError):
                    continue
                # Defensive: literal_eval can return a set/list/None if
                # the matched `{...}` literal isn't a dict.
                if not isinstance(mapping, dict):
                    continue
                studio_pkg = mapping.get("studio")
                if studio_pkg:
                    out.append(Path(studio_pkg) / "frontend" / "dist")
    return out


def _resolve_frontend_path(frontend_path: Path) -> tuple[Optional[Path], list[Path]]:
    """Pick a frontend dir that contains `index.html`.

    Returns (chosen, attempted). `chosen` is None if nothing servable was
    found; `attempted` is the ordered list for diagnostics.
    """
    attempted: list[Path] = []
    seen: set[Path] = set()

    def _try(p: Path) -> bool:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key in seen:
            return False
        seen.add(key)
        attempted.append(p)
        return (p / "index.html").is_file()

    if _try(Path(frontend_path)):
        return attempted[-1], attempted
    for alt in _iter_frontend_fallback_candidates():
        if _try(alt):
            return attempted[-1], attempted
    return None, attempted


def run_server(
    host: str = "127.0.0.1",
    port: int = 8888,
    frontend_path: Path = _DEFAULT_FRONTEND_PATH,
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
        api_only: API server only, no frontend (for Tauri desktop app)
        llama_parallel_slots: parallel slots for llama-server

    Note:
        Signal handlers are NOT registered here so embedders (e.g. Colab
        notebooks) keep their own interrupt semantics. Standalone callers
        should register handlers after calling this.
    """
    global _server, _shutdown_event

    # Windows console encoding (cp1252) can't encode emoji. Reconfigure
    # stdout to UTF-8 so startup messages don't crash the server.
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass

    # Set env var BEFORE importing main so CORS middleware picks it up.
    if api_only:
        os.environ["UNSLOTH_API_ONLY"] = "1"

    import nest_asyncio

    nest_asyncio.apply()

    import asyncio
    from threading import Thread, Event
    import uvicorn

    from main import app, setup_frontend, _IS_COLAB
    from utils.paths import ensure_studio_directories

    # Create all standard directories on startup.
    ensure_studio_directories()

    # Auto-find a free port if the requested one is in use.
    if not _is_port_free(host, port):
        original_port = port
        blocker = _get_pid_on_port(port)
        port = _find_free_port(host, port + 1)
        if not silent:
            print("")
            print("=" * 50)
            if blocker:
                pid, name = blocker
                print(f"Port {original_port} is already in use by " f"{name} (PID {pid}).")
            else:
                print(f"Port {original_port} is already in use.")
            print(f"Unsloth Studio will use port {port} instead.")
            print(f"Open http://localhost:{port} in your browser.")
            print("=" * 50)
            print("")

    # Setup frontend if path provided (skip in api-only mode). Falls back
    # through alternate locations if the default lacks a built dist;
    # errors loudly rather than serving 404 on `/`.
    if frontend_path and not api_only:
        chosen, attempted = _resolve_frontend_path(Path(frontend_path))
        if chosen is not None and setup_frontend(app, chosen):
            if not silent:
                # Resolve so logs show an absolute path for support.
                try:
                    display = chosen.resolve()
                except OSError:
                    display = chosen
                print(f"[OK] Frontend loaded from {display}")
        else:
            home_str = (
                os.environ.get("UNSLOTH_STUDIO_HOME")
                or os.environ.get("STUDIO_HOME")
                or str(Path.home() / ".unsloth" / "studio")
            )
            # Windows ships the shim at $STUDIO_HOME/bin/unsloth.exe (a
            # hardlink to the venv exe); Linux/macOS use the venv binary
            # at $STUDIO_HOME/unsloth_studio/bin/unsloth.
            home = Path(home_str).expanduser()
            if sys.platform == "win32":
                installer_bin = home / "bin" / "unsloth.exe"
            else:
                installer_bin = home / "unsloth_studio" / "bin" / "unsloth"
            tried_lines = "\n".join(f"  - {p}" for p in attempted) or "  (none)"
            raise SystemExit(
                "[ERROR] Studio frontend build not found.\n"
                f"Tried:\n{tried_lines}\n"
                "\n"
                "Likely cause: another 'unsloth' on PATH is shadowing the "
                "installer's binary and points at a site-packages tree with "
                "no built dist.\n"
                "\n"
                "Fix one of:\n"
                f"  - run the installer's binary directly: {installer_bin} studio\n"
                "  - pass --frontend <path/to/studio/frontend/dist>\n"
                "  - pass --api-only to skip serving the web UI\n"
                "  - reinstall: curl -fsSL https://unsloth.ai/install.sh | sh"
            )

    # Resolve once; shared by the log rewrite and banner.
    display_host = _resolve_external_ip() if host == "0.0.0.0" else host
    _install_uvicorn_startup_log_rewrite(host, display_host)

    ready_event = Event()
    startup_failed = Event()
    startup_errors = []

    class _ReadyServer(uvicorn.Server):
        async def startup(self, *args, **kwargs):
            await super().startup(*args, **kwargs)
            if getattr(self, "started", False) and not self.should_exit:
                ready_event.set()

    # server_header=False suppresses uvicorn's "Server: uvicorn"; SecurityHeadersMiddleware sets its own.
    config_kwargs = dict(
        host = host,
        port = port,
        log_level = "info",
        access_log = False,
        server_header = False,
    )
    # Colab only: trust X-Forwarded-* from Colab's reverse proxy so the
    # app sees the real https origin. forwarded_allow_ips="*" is fine in
    # Colab's single-user sandbox but an unwanted relaxation for a normal
    # local/standalone Studio, so leave uvicorn's safe defaults
    # (forwarded headers trusted from loopback only) elsewhere.
    if _IS_COLAB:
        config_kwargs["proxy_headers"] = True
        config_kwargs["forwarded_allow_ips"] = "*"
    config = uvicorn.Config(app, **config_kwargs)
    _server = _ReadyServer(config)
    _shutdown_event = Event()

    # Expose the actual bound port so request handlers build loopback
    # URLs pointing at the real backend, not whatever port a proxy/tunnel
    # exposed in the request URL. Only publish a concrete port; for
    # ephemeral binds (port==0) leave it unset so handlers fall back to
    # the ASGI request scope or request.base_url.
    app.state.server_port = port if port and port > 0 else None
    app.state.llama_parallel_slots = llama_parallel_slots

    # Expose a shutdown callable via app.state before the server accepts
    # requests so /api/shutdown is ready as soon as readiness publishes.
    def _trigger_shutdown():
        _graceful_shutdown(_server)
        if _shutdown_event is not None:
            _shutdown_event.set()

    app.state.trigger_shutdown = _trigger_shutdown

    # Run server in a daemon thread. Use explicit new_event_loop() +
    # run_until_complete() rather than asyncio.run() so nest_asyncio's
    # global patches to asyncio.run don't interfere when called from a
    # thread while Colab/IPython already runs a loop on the main thread.
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_server.serve())
        except BaseException as exc:
            startup_errors.append(exc)
            startup_failed.set()
        finally:
            loop.close()
            if not ready_event.is_set():
                startup_failed.set()

    thread = Thread(target = _run, daemon = True)
    thread.start()

    # Wait until uvicorn finishes lifespan startup and binds sockets, or
    # until the server exits/fails before startup. No correctness
    # deadline: a slow but live startup should remain in progress.
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

    # Output port for Tauri to parse in api-only mode. Emit only after
    # uvicorn sockets are bound and FastAPI startup completed.
    if api_only:
        print(f"TAURI_PORT={port}", flush = True)

    if not silent:
        _emit_startup_output(host, port, display_host)

    return app


# For direct execution (also invoked by CLI via os.execvp / subprocess).
if __name__ == "__main__":
    import argparse
    import signal
    import traceback

    # Ensure stderr handles Unicode on Windows (non-ASCII path tracebacks).
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
        default = _DEFAULT_FRONTEND_PATH,
        help = "Path to frontend build",
    )
    parser.add_argument("--silent", action = "store_true", help = "Suppress output")
    parser.add_argument(
        "--api-only",
        action = "store_true",
        help = "API server only, no frontend (for Tauri)",
    )
    # Mirror unsloth_cli/commands/studio.py's _PARALLEL_*. Default 1
    # applies to direct backend launches only; `unsloth studio run`
    # always passes its own value (4) explicitly.
    _PARALLEL_MIN = 1
    _PARALLEL_MAX = 64
    _PARALLEL_DEFAULT_PLAIN = 1
    parser.add_argument(
        "--parallel",
        "--n-parallel",
        type = int,
        default = _PARALLEL_DEFAULT_PLAIN,
        help = (
            f"llama-server parallel decode slots ({_PARALLEL_MIN}..{_PARALLEL_MAX}). "
            f"Default {_PARALLEL_DEFAULT_PLAIN}; `unsloth studio run` uses 4."
        ),
    )

    args = parser.parse_args()
    if not _PARALLEL_MIN <= args.parallel <= _PARALLEL_MAX:
        parser.error(f"--parallel must be between {_PARALLEL_MIN} and {_PARALLEL_MAX}")

    kwargs = dict(
        host = args.host,
        port = args.port,
        silent = args.silent,
        api_only = args.api_only,
        llama_parallel_slots = args.parallel,
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
        sys.stderr.write("If a package is missing, try re-running: unsloth studio setup\n")
        sys.stderr.flush()
        sys.exit(1)

    # Signal handler -- ensures subprocess cleanup on Ctrl+C.
    def _signal_handler(signum, frame):
        _graceful_shutdown(_server)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # On Windows, some terminals send SIGBREAK for Ctrl+C / Ctrl+Break.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    # Keep running until shutdown signal.
    # NOTE: Event.wait() without a timeout blocks at the C level on Linux,
    # preventing Python from delivering SIGINT (Ctrl+C). A short timeout
    # in a loop lets the interpreter process pending signals.
    while not _shutdown_event.is_set():
        _shutdown_event.wait(timeout = 1)

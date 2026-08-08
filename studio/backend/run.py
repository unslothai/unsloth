# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run script for Unsloth UI Backend.

Self-contained; can be moved to any directory.
"""

import os
import sys
import time
from pathlib import Path
from typing import NoReturn, Optional, Sequence, Tuple


def _normalize_standard_streams():
    """Point missing std streams at the null device.

    sys.stdout/stderr/stdin are None in a process with no valid std handles (a
    Windows pythonw or detached launch), so every .write() / .isatty() on them
    raises AttributeError. A null-device stream answers like a real one, so
    nothing downstream needs its own None check.

    MUST run before the `from loggers import ...` below: structlog binds
    `from sys import stdout` at ITS import time and PrintLogger does
    `self._file = file or stdout`, so a None stdout is captured permanently.
    Normalizing inside run_server() is too late to undo that capture.
    """
    for name, mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
        if getattr(sys, name, None) is not None:
            continue
        try:
            stream = open(os.devnull, mode, encoding = "utf-8", errors = "replace")
        except Exception:
            # Normalizing must never itself be what kills startup.
            continue
        setattr(sys, name, stream)
        # sys.__stdout__ & co are None here too, and readers of those (rich reads
        # sys.__stdout__.fileno() at import) otherwise fall back to fds 0/1/2,
        # which this process does not have. Point them at the real null fd.
        if getattr(sys, f"__{name}__", None) is None:
            setattr(sys, f"__{name}__", stream)


_normalize_standard_streams()


def _fix_torch_cuda_ld_path():
    """Prepend torch's bundled CUDA libs to LD_LIBRARY_PATH.

    PyTorch wheels ship their own CUDA runtime (libcudart, libcublas, ...) in
    ``site-packages/nvidia/*/lib``. On Linux the dynamic linker reads
    LD_LIBRARY_PATH before the RUNPATH baked into torch's .so files, so a
    pre-existing LD_LIBRARY_PATH pointing at a different system CUDA (e.g.
    /usr/local/cuda-13/lib64 from conda or a Docker base image) shadows torch's
    libs and triggers "undefined symbol" errors when torch is imported. Detect
    torch's lib dirs (without importing torch) and prepend them. Returns True if
    LD_LIBRARY_PATH was changed.
    """
    if sys.platform != "linux":
        return False
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        return False
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return False
        torch_dir = os.path.dirname(spec.origin)
        site_pkgs = os.path.dirname(torch_dir)
        nvidia_dir = os.path.join(site_pkgs, "nvidia")

        lib_dirs = []
        torch_lib = os.path.join(torch_dir, "lib")
        if os.path.isdir(torch_lib):
            lib_dirs.append(torch_lib)
        if os.path.isdir(nvidia_dir):
            for sub in sorted(os.listdir(nvidia_dir)):
                lib = os.path.join(nvidia_dir, sub, "lib")
                if os.path.isdir(lib):
                    lib_dirs.append(lib)
        if not lib_dirs:
            return False

        existing = ld_path.split(":")
        if existing[: len(lib_dirs)] == lib_dirs:
            return False  # already at the front, nothing to do

        torch_set = set(lib_dirs)
        cleaned = [p for p in existing if p not in torch_set]
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + cleaned)
        return True
    except Exception:
        return False


_LD_FIXED_SENTINEL = "_UNSLOTH_STUDIO_LD_FIXED"


def _maybe_reexec_for_cuda_ld_path():
    """Re-exec once so the dynamic linker sees the corrected LD_LIBRARY_PATH.

    LD_LIBRARY_PATH is read at process start, so editing os.environ in-process
    cannot fix the running interpreter; a single re-exec is required. Call only
    from a true entry point (the ``if __name__ == "__main__"`` block), never at
    import time, because os.execv replaces the whole process (an embedder such
    as Colab that does ``from run import run_server`` must not be re-exec'd).
    """
    if _LD_FIXED_SENTINEL in os.environ:
        return
    if not _fix_torch_cuda_ld_path():
        return
    os.environ[_LD_FIXED_SENTINEL] = "1"
    argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    os.execv(sys.executable, argv)


# Suppress C-level dependency warnings globally (e.g. SwigPyPacked).
os.environ["PYTHONWARNINGS"] = "ignore"

# Add the backend dir to sys.path early so local modules import.
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# First, so these vars are in place before anything below can initialize an OpenMP/BLAS pool.
# utils.cpu_threads imports os and typing only.
from utils.cpu_threads import configure_cpu_threads

try:
    configure_cpu_threads()
except ValueError as exc:
    configured = os.environ.get("UNSLOTH_CPU_THREADS")
    raise SystemExit(f"Error: Invalid UNSLOTH_CPU_THREADS value {configured!r}: {exc}") from None

# Windows ROCm ships no distributed backend, so torchao and the CUDA-only xformers both die on
# import, taking anything that touches diffusers/transformers with them. A stub only seeds a name
# nothing has imported yet, so both must precede the first import below. No-op on other runtimes.
from core._torchao_stub import (
    install_torchao_windows_rocm_stub,
    install_xformers_windows_rocm_stub,
)

install_xformers_windows_rocm_stub()
install_torchao_windows_rocm_stub()

# Anaconda/conda-forge Python: seed platform._sys_version_cache before imports
# that trigger attrs -> rich -> structlog -> platform crash.
# See: https://github.com/python/cpython/issues/102396
import _platform_compat  # noqa: F401

from loggers import get_logger, install_uvicorn_duplicate_exception_filter
from startup_banner import print_studio_access_banner, print_studio_stop_hint

logger = get_logger(__name__)

DISABLE_PUBLIC_CHECK_ENV = "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK"


def public_check_disabled() -> bool:
    """True when the operator has turned off the third-party startup lookups.

    On a wildcard bind Unsloth asks ifconfig.me for the public IP and check-host.net
    whether the port is reachable. Both are useful for sharing a Studio but both tell
    an outside service this machine is running one, which lab and privacy-sensitive
    deployments do not want (#7307 Problem 8). Set the var to opt out.
    """
    return os.environ.get(DISABLE_PUBLIC_CHECK_ENV, "").strip().lower() in {"1", "true", "yes"}


def _resolve_external_ip() -> str:
    """Resolve the machine's external IP address.

    Tries, in order:
    1. GCE metadata server (instant on Google Cloud VMs)
    2. ifconfig.me (anywhere with internet, skipped by UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK)
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

    # 2. Public IP service. Third-party, so skippable; the LAN address below still works.
    if not public_check_disabled():
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
    """Rewrite Uvicorn's startup log line: swap wildcard bind for the
    externally-reachable address, use our Mac-aware stop hint, and rename the
    prefix to "Unsloth Studio running on"."""
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
    """A working loopback URL on this machine, or None if neither 127.0.0.1 nor
    ::1 responds. Fallback when external reachability fails."""
    if _local_port_open("127.0.0.1", port):
        return f"http://127.0.0.1:{port}"
    if _local_port_open("::1", port):
        return f"http://[::1]:{port}"
    return None


def _localhost_ipv6_mismatch_url(bind_host: str, port: int) -> "str | None":
    """Return the IPv4 loopback URL when localhost won't reach 127.0.0.1.

    Local Unsloth binds to 127.0.0.1. Where localhost resolves to IPv6 only (::1),
    http://localhost:<port> fails (or hits a different process on ::1) even though
    http://127.0.0.1:<port> works. Return the IPv4 URL for the caller to surface.
    """
    import socket

    if bind_host != "127.0.0.1" or not port or port <= 0:
        return None

    ipv4_url = f"http://127.0.0.1:{port}"

    # Only warn once Unsloth is confirmed answering on IPv4 loopback.
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

    # A connection to ::1 is NOT evidence Unsloth is reachable there: Unsloth binds
    # 127.0.0.1 only, so anything on ::1 is a different process. Dual-stack
    # localhost is fine (browsers fall back to 127.0.0.1), so only the IPv6-only
    # case strands the user.
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
    """Warn that localhost points at ::1 while Unsloth is bound to 127.0.0.1."""
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
    """Probe check-host.net to confirm display_host:port is reachable from the
    public internet. Synchronous so output lands between the banner URLs and the
    stop hint. Bounded at ~15s; failures swallowed (verifier failing != Unsloth
    failing). Only meaningful for a wildcard bind, and skipped entirely by
    UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK."""
    global _public_reachable
    # Reset to "unknown" each run; set True/False only when the probe decides.
    _public_reachable = None
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

    url = f"http://{_url_host(display_host)}:{port}"

    # Private/loopback/link-local addresses aren't globally routable.
    try:
        addr = ipaddress.ip_address(display_host)
        if addr.is_loopback or addr.is_private or addr.is_link_local:
            _public_reachable = False
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

    # The probe hands display_host:port to a third party and asks it to connect.
    if public_check_disabled():
        logger.debug("Skipping the check-host.net probe (%s).", DISABLE_PUBLIC_CHECK_ENV)
        return

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
            _public_reachable = True
            print(
                f"{ok_c}  Reachability check: {url}/ is reachable from the "
                f"public internet ({ok_nodes}/{total} probe nodes connected).{reset}",
                flush = True,
            )
        elif err_nodes:
            _public_reachable = False
            print(
                f"{err_c}  Reachability check: {url}/ is NOT reachable from "
                f"the public internet ({err_nodes}/{total} probe nodes failed).{reset}",
                flush = True,
            )
            print(
                f"{dim}    Usually a cloud firewall (AWS security group, "
                f"GCP firewall / Azure NSG rule) or home router isn't "
                f"allowing inbound TCP {port}.{reset}",
                flush = True,
            )
            print(
                f"{dim}    No firewall change needed -- SSH local-forward "
                f"from your own computer:{reset}",
                flush = True,
            )
            print(
                f"{dim}        ssh -L {port}:localhost:{port} <user>@{display_host}{reset}",
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


def _display_host_for_bind(host: str) -> str:
    return _resolve_external_ip() if host in ("0.0.0.0", "::") else host


def _loopback_bind_host_for(host: str) -> str:
    return "::1" if host == "::" else "127.0.0.1"


def _url_host(host: str) -> str:
    return (
        f"[{host}]" if ":" in host and not (host.startswith("[") and host.endswith("]")) else host
    )


def _tool_policy_notice(host: str, secure: bool, enable_tools: "Optional[bool]") -> str:
    """One-line tool-policy summary for the plain-server startup banner, so a
    network-reachable launch is never silent about code execution."""
    if enable_tools is False:
        return "Server-side tools are DISABLED (--disable-tools)."
    state = (
        "ENABLED (--enable-tools)"
        if enable_tools
        else "ENABLED by default (per-request setting honored)"
    )
    if secure:
        return (
            f"Server-side tools are {state}, reachable via the authenticated "
            "Cloudflare HTTPS tunnel. Anyone with the API key can run code on "
            "this machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    from utils.host_policy import is_external_host

    if host in ("0.0.0.0", "::") or is_external_host(host):
        return (
            f"Server-side tools are {state} and this port is network-reachable. "
            "Anyone who can reach it with the API key can run code on this "
            "machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    return f"Server-side tools are {state} for loopback. Pass --disable-tools to turn off."


def _emit_tool_policy_notice(host: str, secure: bool, enable_tools: "Optional[bool]") -> None:
    print(_tool_policy_notice(host, secure, enable_tools), flush = True)


def _emit_secure_startup_output(port: int, enable_tools: "Optional[bool]" = None) -> None:
    """Secure-mode banner: only the Cloudflare link (loopback has no public raw URL)."""
    print("")
    print("🦥 Unsloth Studio is running (secure)")
    print("─" * 52)
    _print_cloudflare_line(secure = True)
    print(f"  On this machine only: http://127.0.0.1:{port}/")
    print("─" * 52)
    _emit_tool_policy_notice("127.0.0.1", True, enable_tools)
    print_studio_stop_hint()


def _emit_startup_output(
    host: str,
    port: int,
    display_host: str,
    secure: bool = False,
    enable_tools: "Optional[bool]" = None,
) -> None:
    """Print the access banner, post-startup warnings, the tool-policy notice,
    then a single stop hint. Extracted from ``_run`` so the wiring is testable."""
    if secure:
        _emit_secure_startup_output(port, enable_tools)
        return
    wildcard_bind = host in ("0.0.0.0", "::")
    localhost_mismatch_url = _localhost_ipv6_mismatch_url(host, port)
    print_studio_access_banner(
        port = port,
        bind_host = host,
        display_host = display_host,
        include_stop_hint = False,
    )
    if localhost_mismatch_url:
        _print_localhost_ipv6_mismatch_warning(localhost_mismatch_url, port)
    elif wildcard_bind:
        _verify_global_reachability(display_host, port)
        _print_cloudflare_line(loopback_host = _loopback_bind_host_for(host))
    _emit_tool_policy_notice(host, False, enable_tools)
    print_studio_stop_hint()


def _print_cloudflare_line(secure: bool = False, loopback_host: str = "127.0.0.1") -> None:
    """Print Cloudflare tunnel state for startup banners."""
    from startup_banner import stdout_supports_color

    accent = "\033[38;5;150;1m"
    warn = "\033[38;5;215;1m"
    reset = "\033[0m"
    color = stdout_supports_color()

    def _emit(text: str, style: str = "") -> None:
        print(f"{style}{text}{reset}" if (color and style) else text)

    if _cloudflare_url:
        if _public_reachable is False:
            _emit(f"  Use the secure link access via Cloudflare instead: {_cloudflare_url}", accent)
        else:
            _emit(f"  Secure link access via Cloudflare: {_cloudflare_url}", accent)
        if not secure:
            if _public_reachable is True:
                _emit(
                    "  Cloudflare tunnel: ON. This Cloudflare URL is PUBLIC, and the "
                    "raw port is also publicly reachable. --no-cloudflare disables "
                    f"only the Cloudflare URL; bind {loopback_host} or close firewall "
                    "access to keep Unsloth private.",
                    warn,
                )
            else:
                _emit(
                    "  Cloudflare tunnel: ON. This is a PUBLIC internet URL: anyone "
                    "who has it can reach this Unsloth. Relaunch with --no-cloudflare "
                    f"to disable the Cloudflare URL; bind {loopback_host} or close "
                    "firewall access to keep Unsloth private.",
                    warn,
                )
        return
    if _cloudflare_requested:
        if _public_reachable is True:
            _emit(
                "  Cloudflare tunnel: requested but failed to start. The raw port is "
                "still reachable from the public internet (see the reachability check "
                "above): anyone who can reach it can access this Unsloth.",
                warn,
            )
        elif _public_reachable is False:
            _emit(
                "  Cloudflare tunnel: requested but failed to start. Unsloth is reachable "
                "on your local network only (no public link).",
                warn,
            )
        else:
            _emit(
                "  Cloudflare tunnel: requested but failed to start. There is no "
                "Cloudflare public link. Raw port reachability was not verified; "
                f"bind {loopback_host} or close firewall access to keep Unsloth private.",
                warn,
            )
    elif _cloudflare_flag:
        if _public_reachable is True:
            _emit(
                "  Cloudflare tunnel: OFF for this mode. The raw port is still "
                "reachable from the public internet (see the reachability check above): "
                "anyone who can reach it can access this Unsloth.",
                warn,
            )
        elif _public_reachable is False:
            _emit(
                "  Cloudflare tunnel: OFF for this mode. Unsloth is reachable on your "
                "local network only (no public link)."
            )
        else:
            _emit(
                "  Cloudflare tunnel: OFF for this mode. There is no Cloudflare public "
                "link. Raw port reachability was not verified; "
                f"bind {loopback_host} or close firewall access to keep Unsloth private.",
                warn,
            )
    elif _cloudflare_flag is False or _cloudflare_flag is None:
        # None = off by default (no flag); False = explicit --no-cloudflare.
        _reason = "default" if _cloudflare_flag is None else "--no-cloudflare"
        if _public_reachable is True:
            _emit(
                f"  Cloudflare tunnel: OFF ({_reason}). The raw port is still "
                "reachable from the public internet (see the reachability check above): "
                "pass --cloudflare to also expose a public Cloudflare HTTPS link, or "
                f"bind {loopback_host} to keep Unsloth private.",
                warn,
            )
        elif _public_reachable is False:
            _emit(
                f"  Cloudflare tunnel: OFF ({_reason}). Unsloth is reachable on your "
                "local network only. Pass --cloudflare to expose a public "
                "Cloudflare HTTPS link."
            )
        else:
            _emit(
                f"  Cloudflare tunnel: OFF ({_reason}). There is no Cloudflare "
                "public link. Raw port reachability was not verified; pass --cloudflare "
                "to expose a public Cloudflare HTTPS link, or "
                f"bind {loopback_host} or close firewall access to keep Unsloth private.",
                warn,
            )


def _get_pid_on_port(port: int) -> "tuple[int, str] | None":
    """Return (pid, process_name) listening on *port*, or None.

    Uses psutil when available, else None so callers can still report the conflict
    without process details.
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


def _bind_addresses(host: str, port: int) -> "set[str]":
    """Every address *host* resolves to. `localhost` is both 127.0.0.1 and ::1, and
    recording only the first lets a later launch on the other one miss us."""
    import socket

    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return {host}
    return {info[4][0] for info in infos} or {host}


def _addresses_collide(recorded: "str | None", host: str, port: int) -> bool:
    """Would a server bound to *recorded* block a bind to *host*?

    *recorded* may list several addresses. Unknown or wildcard on either side
    collides: refusing with a clear message beats silently starting a duplicate.
    """
    wildcards = ("0.0.0.0", "::", "")
    if not recorded or host in wildcards:
        return True
    listed = {a.strip() for a in recorded.split(",") if a.strip()}
    if not listed or listed & set(wildcards):
        return True
    return bool(listed & _bind_addresses(host, port))


def _is_port_free(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    For a ``0.0.0.0`` wildcard host, also check whether anything is listening on
    ``127.0.0.1`` (and ``::1`` when IPv6 exists): an SSH tunnel may hold loopback
    while the wildcard bind succeeds, making Unsloth unreachable via ``localhost``.
    """
    import socket

    # 1. Can we bind to the requested address? getaddrinfo resolves both
    #    IPv4 and IPv6 to the right address family.
    try:
        addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        family, socktype, proto, _, sockaddr = addr_info[0]
        with socket.socket(family, socktype, proto) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
    except OSError:
        return False

    # 2. On a wildcard bind, verify localhost isn't already claimed by another
    #    process (e.g. an SSH -L tunnel); a successful connect means it is.
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
    avoid_own_studio: bool = False,
) -> int:
    """Find a free port from `start`, trying up to max_attempts ports.

    ``avoid_own_studio`` aborts rather than skipping past one of our own servers
    in the fallback range, which would start a duplicate on a later port.
    """
    for offset in range(max_attempts):
        candidate = start + offset
        if _is_port_free(host, candidate):
            return candidate
        if avoid_own_studio:
            own = _own_studio_on_port(candidate, host)
            if own is not None:
                _abort_already_running(own, candidate)
    raise RuntimeError(f"Could not find a free port in range {start}-{start + max_attempts - 1}")


from utils.paths.storage_roots import studio_root as _studio_root

# Legacy single-instance file; still read so `stop` finds an older build's server.
_PID_FILE = _studio_root() / "studio.pid"
PID_FILE_GLOB = "studio-*.pid"


def _pid_file_for_port(port: int) -> Path:
    # PID in the name: 127.0.0.1 and ::1 can share a port, and one file per port
    # would let the second bind overwrite the first.
    return _studio_root() / f"studio-{port}-{os.getpid()}.pid"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if sys.platform == "win32":
        # os.kill(pid, 0) raises OSError for every pid on Windows, so tasklist is
        # the only usable probe here.
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
            ).stdout
        except Exception:
            # Unconfirmed means keep, matching the CLI's _pid_alive. Pruning a
            # live server's record is what lets the next launch fall back past it
            # and strand it, which is the bug this file exists to fix. A stale
            # record instead costs one clear "already running" message.
            return True
        return f'"{int(pid)}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _process_create_time(pid: int) -> "float | None":
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _read_pid_record(path: Path) -> "tuple[int, float | None, str | None] | None":
    """Parse ``pid`` / optional ``create_time`` / optional bind address."""
    try:
        lines = path.read_text(encoding = "utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or not lines[0].strip().isdigit():
        return None
    try:
        # isdigit() is not enough: a superscript two passes it but int() rejects it.
        pid = int(lines[0].strip())
    except ValueError:
        return None
    # kill(0) signals our whole process group; kill(1) is init. Never either.
    if pid < 2:
        return None
    created = None
    if len(lines) > 1:
        try:
            created = float(lines[1].strip())
        except ValueError:
            created = None
    address = lines[2].strip() if len(lines) > 2 and lines[2].strip() else None
    return pid, created, address


def _pid_is_studio_backend(pid: int, created_times: "Sequence[float | None]" = ()) -> bool:
    """False only when a recorded start time proves this PID is a different process.

    Any recorded time matching is enough -- a stale record must not veto a live
    server that reused the PID. Untimed records cannot be checked at all, so they
    are trusted: a legacy `python run.py` has no telltale argv, and guessing from
    the command line rejected real servers.
    """
    known = [c for c in created_times if c is not None]
    if not known:
        return True
    actual = _process_create_time(pid)
    if actual is None:
        return True
    return any(abs(actual - c) < 1.0 for c in known)


def _own_studio_on_port(port: int, host: str) -> "int | None":
    """PID of one of our own servers already bound to *port* for *host*.

    Reads our own records rather than enumerating listeners: psutil is optional,
    and without it a listener scan finds nothing and we silently start a duplicate.
    """
    try:
        paths = list(_studio_root().glob(f"studio-{port}-*.pid"))
    except OSError:
        return None
    for path in paths:
        record = _read_pid_record(path)
        if record is None:
            continue
        pid, created, address = record
        if not _pid_alive(pid):
            # Pruning is a courtesy; an undeletable record must not abort startup.
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
            continue
        if not _addresses_collide(address, host, port):
            continue
        if _pid_is_studio_backend(pid, [created]):
            return pid
    return _legacy_studio_on_port(port)


def _legacy_studio_on_port(port: int) -> "int | None":
    """A pre-upgrade server recorded only its PID, so match it to the listener.

    Falling back past one leaves it running while `_write_pid_file` overwrites the
    only record of it. When the listener is unknowable, assume it is ours.
    """
    record = _read_pid_record(_PID_FILE)
    if record is None:
        return None
    pid, created, _address = record
    if not _pid_alive(pid):
        return None
    # A current build writes a per-port file too, so its port is already known --
    # and this port's records were just checked. Only count a record that still
    # matches the live process: a stale one may just share a reused PID.
    for other in _per_port_records():
        if other and other[0] == pid and _pid_is_studio_backend(pid, [other[1]]):
            return None
    blocker = _get_pid_on_port(port)
    if blocker is not None and blocker[0] != pid:
        return None
    if not _pid_is_studio_backend(pid, [created]):
        return None
    return pid


def _per_port_records() -> "list[tuple[int, float | None, str | None] | None]":
    try:
        return [_read_pid_record(p) for p in _studio_root().glob(PID_FILE_GLOB)]
    except OSError:
        return []


def _resolve_port(
    host: str,
    port: int,
    avoid_own_studio: bool = True,
) -> int:
    """The requested port, or the next free one.

    With ``avoid_own_studio`` this aborts rather than falling back past one of our
    own servers, on *port* itself or anywhere in the fallback range: skipping one
    is what strands it. Callers that read the bound port back pass False and keep
    the plain fallback.
    """
    if _is_port_free(host, port):
        return port
    if avoid_own_studio:
        own = _own_studio_on_port(port, host)
        if own is not None:
            _abort_already_running(own, port)
    return _find_free_port(host, port + 1, avoid_own_studio = avoid_own_studio)


def _abort_already_running(pid: int, port: int) -> "NoReturn":
    print(
        f"Error: Unsloth Studio is already running on port {port} (PID {pid}). Run "
        "`unsloth studio stop` first, or start this one on a different --port.",
        file = sys.stderr,
        flush = True,
    )
    sys.exit(1)


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

# The studio bundles unsloth_zoo; declare unsloth present (as `import unsloth`
# does) so its lazy submodule imports (export, hardware, mlx) and the
# DiffusionGemma runner never trip the install guard on a clean install.
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")


_OWN_PID_FILE: "Path | None" = None


def _write_pid_file(port: int, host: str = ""):
    """Record this PID under its own port so `stop` can find every server."""
    global _OWN_PID_FILE
    path = _pid_file_for_port(port)
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
    except OSError:
        pass
    try:
        # Start time pins the record to this process; the bind address tells a
        # later launch whether this server would actually block it.
        created = _process_create_time(os.getpid())
        address = ",".join(sorted(_bind_addresses(host, port))) if host else ""
        body = f"{os.getpid()}\n{'' if created is None else repr(created)}\n{address}"
        # Write-then-rename: `stop` reads these concurrently, and a reader that
        # catches the truncate window sees a corrupt record and deletes it.
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(body, encoding = "utf-8")
            os.replace(tmp, path)
        finally:
            # A failed replace would otherwise leave the scratch file behind. It
            # does not end in .pid, so no glob picks it up either way.
            tmp.unlink(missing_ok = True)
    except OSError:
        pass
    else:
        _OWN_PID_FILE = path
    # An older CLI's `stop` only reads this one, and expects a bare PID. Written
    # independently of the per-port record: if that one failed, this is the only
    # thing keeping the server stoppable at all.
    try:
        # Never take it from a server that is still running. A pre-upgrade server
        # is recorded here and nowhere else, so overwriting its entry is exactly
        # what strands it -- the orphan this file exists to prevent.
        prior = _read_pid_record(_PID_FILE) if _PID_FILE.is_file() else None
        if prior is None or prior[0] == os.getpid() or not _pid_alive(prior[0]):
            _PID_FILE.write_text(str(os.getpid()), encoding = "utf-8")
    except OSError:
        pass


def _legacy_heir() -> "int | None":
    """Another live server's PID, to hand the legacy studio.pid over to.

    Only one server owns studio.pid at a time, so its exit would otherwise drop
    the single record an older CLI can read, stranding any sibling that is still
    serving.
    """
    try:
        paths = sorted(_studio_root().glob(PID_FILE_GLOB))
    except OSError:
        return None
    for path in paths:
        if _OWN_PID_FILE is not None and path == _OWN_PID_FILE:
            continue
        record = _read_pid_record(path)
        if record is None or record[0] == os.getpid():
            continue
        if _pid_alive(record[0]) and _pid_is_studio_backend(record[0], [record[1]]):
            return record[0]
    return None


def _remove_pid_file():
    """Remove the PID files that belong to this process.

    _PID_FILE is checked even when the per-port record was never written, since
    _write_pid_file writes the two independently.
    """
    # Nothing here may raise: _graceful_shutdown calls this at the end, and an
    # unreadable or undeletable record must not abandon the rest of the exit
    # path. _read_pid_record already swallows OSError/UnicodeDecodeError.
    if _OWN_PID_FILE is not None:
        try:
            record = _read_pid_record(_OWN_PID_FILE) if _OWN_PID_FILE.is_file() else None
            if record is not None and record[0] == os.getpid():
                _OWN_PID_FILE.unlink(missing_ok = True)
        except OSError:
            pass
    try:
        record = _read_pid_record(_PID_FILE) if _PID_FILE.is_file() else None
        if record is not None and record[0] == os.getpid():
            # Hand the pointer to a live sibling rather than deleting it. An
            # older CLI reads only this file, so dropping it while another
            # server is still up leaves that server unstoppable.
            heir = _legacy_heir()
            if heir is None:
                _PID_FILE.unlink(missing_ok = True)
            else:
                _PID_FILE.write_text(str(heir), encoding = "utf-8")
    except OSError:
        pass


def _graceful_shutdown(server = None):
    """Shut down all subprocess backends and the uvicorn server.

    Called from signal handlers to clean up children before exit. Critical on
    Windows where atexit handlers are unreliable after Ctrl+C.
    """
    logger.info("Graceful shutdown initiated -- cleaning up subprocesses...")

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

    # 6. Stop the Cloudflare tunnel (if started).
    try:
        from cloudflare_tunnel import close_studio_tunnel_lifecycle
        close_studio_tunnel_lifecycle()
    except Exception as e:
        logger.warning("Error stopping Cloudflare tunnel: %s", e)

    # 7. Backstop sweep for any adopted child the steps above missed.
    try:
        from utils.process_lifetime import terminate_all
        terminate_all()
    except Exception as e:
        logger.warning("Error in process-lifetime sweep: %s", e)

    # Last: while cleanup runs the server is still alive, and dropping the record
    # early leaves a retried `stop` or a new launch unable to find it.
    _remove_pid_file()
    logger.info("All subprocesses cleaned up")


# Bound the join so a stuck uvicorn shutdown cannot hang the terminal.
_SERVER_SHUTDOWN_JOIN_TIMEOUT = 5.0


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _wait_for_server_shutdown(timeout: Optional[float] = _SERVER_SHUTDOWN_JOIN_TIMEOUT) -> None:
    """Join the uvicorn thread so the prompt returns only after its shutdown logs
    flush. Skip the self-join when called from the server thread."""
    import threading

    thread = _server_thread
    if thread is None or thread is threading.current_thread():
        _flush_standard_streams()
        return
    thread.join(timeout = timeout)
    if thread.is_alive():
        logger.warning("Timed out waiting for uvicorn server thread to stop")
    _flush_standard_streams()


# The uvicorn server instance -- set by run_server(), used by callers
# that tell the server to exit (e.g. signal handlers).
_server = None
_server_thread = None

# Shutdown event -- wakes the main loop on signal.
_shutdown_event = None

# trycloudflare.com URL for wildcard binds (set by run_server, read by the banner);
# None when there is no tunnel (loopback, disabled, or a silently-ignored failure).
_cloudflare_url = None


def _publish_cloudflare_url(app_state, cloudflare_url: "Optional[str]") -> None:
    global _cloudflare_url
    _cloudflare_url = cloudflare_url
    app_state.cloudflare_url = cloudflare_url


# Public reachability from the last _verify_global_reachability run, read by the
# Cloudflare banner line. True when the public ip:port probe confirmed reachable,
# False when it confirmed NOT reachable, None when the probe did not run or could
# not decide (timeout, blocked, private address).
_public_reachable = None

_cloudflare_requested = False
# Opt-in tri-state (mirrors the CLI): None = off by default, True = on,
# False = explicit --no-cloudflare. run_server overwrites it before the banner.
_cloudflare_flag = None


_DEFAULT_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _iter_frontend_fallback_candidates() -> "list[Path]":
    """Yield `studio/frontend/dist` paths to try when the default is missing.

    Covers PATH-shadowed binaries whose __file__ resolves into a site-packages
    tree with no vite build (e.g. plain `pip install unsloth`).
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
                except (OSError, UnicodeDecodeError):
                    continue
                # Tolerate single/multi-line dict literals; [^}]* rejects nested
                # dicts, which the setuptools editable template never emits.
                m = re.search(r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S)
                if not m:
                    continue
                try:
                    mapping = ast.literal_eval(m.group(1))
                except (SyntaxError, ValueError):
                    continue
                # literal_eval can return a set/list/None if `{...}` isn't a dict.
                if not isinstance(mapping, dict):
                    continue
                studio_pkg = mapping.get("studio")
                if studio_pkg:
                    out.append(Path(studio_pkg) / "frontend" / "dist")
    return out


def _resolve_frontend_path(frontend_path: Path) -> tuple[Optional[Path], list[Path]]:
    """Pick a frontend dir that contains `index.html`.

    Returns (chosen, attempted). `chosen` is None if nothing servable was found;
    `attempted` is the ordered list for diagnostics.
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


def _frontend_serving_mode(*, api_only: bool, desktop_owned: bool) -> tuple[bool, bool]:
    tunnel_only = api_only and desktop_owned
    return not api_only or tunnel_only, tunnel_only


def _missing_frontend_is_fatal(*, tunnel_only: bool) -> bool:
    """Whether an unresolvable SPA build must abort startup.

    It must when the web UI is this launch's own surface: a 404 on / is worse
    than a loud error. It must not for the desktop, which passes --api-only with
    no --frontend and whose installer skips the frontend build outright; there
    the SPA only backs the optional remote web UI, so aborting would kill the
    local API before TAURI_PORT is ever emitted. Degrade to API-only instead."""
    return not tunnel_only


class _TeeStream:
    """Mirror writes to the original stream and a session log file.

    Console behavior is unchanged (writes/returns delegate to the original
    stream; Tauri's structured-stdout protocol and isatty probes see exactly
    what they saw before). The file copy is best-effort: a full disk or a
    closed handle must never break the console."""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        try:
            self._log_fh.write(data)
        except Exception:
            pass
        if self._stream is None:
            return len(data)
        return self._stream.write(data)

    def flush(self):
        try:
            self._log_fh.flush()
        except Exception:
            pass
        if self._stream is None:
            return
        try:
            self._stream.flush()
        except Exception:
            pass

    def close(self):
        # We do NOT own the console stream (it is the terminal / Jupyter kernel
        # stream we wrapped), so closing the tee must never take the server down.
        # Flush the log copy, then forward close() to the wrapped stream
        # best-effort: on Colab that stream is an ipykernel OutStream whose
        # close() can raise (see _harden_console_close / ipython/ipykernel#867).
        try:
            self._log_fh.flush()
        except Exception:
            pass
        if self._stream is None:
            return
        try:
            self._stream.close()
        except Exception:
            pass

    def __getattr__(self, name):
        if self._stream is None:
            raise AttributeError(name)
        return getattr(self._stream, name)


_WATCH_FD_THREAD_ATTR = "watch_fd_thread"


def _is_missing_watch_fd_thread(exc):
    """True only for ipython/ipykernel#867's missing-``watch_fd_thread`` error.

    ``AttributeError.name`` exists from Python 3.10; the message carries the
    attribute name on every version (possibly with a "Did you mean" tail), so
    check both and let every other AttributeError through.
    """
    if getattr(exc, "name", None) == _WATCH_FD_THREAD_ATTR:
        return True
    return _WATCH_FD_THREAD_ATTR in str(exc)


def _harden_console_close(stream):
    """Stop a displaced console stream's close() from aborting Studio startup.

    ``_setup_server_disk_logging`` replaces ``sys.stdout``/``sys.stderr`` with a
    tee. That changes the object identity of the console stream, so a third-party
    logging handler that captured the ORIGINAL stream (notably Colab's ``absl``
    logging handler, whose ``close()`` skips ``sys.stdout``/``sys.stderr`` but not
    a stream that is no longer either) treats it as an ordinary stream and calls
    ``close()`` on it during logging teardown -- ``uvicorn.Config()`` ->
    ``logging.config.dictConfig()`` -> ``logging.shutdown()``.

    A Jupyter/Colab ``ipykernel`` ``OutStream`` created with ``watchfd=False``
    (the Colab default, and every in-process kernel) never gains a
    ``watch_fd_thread``, yet the ``OutStream.close()`` shipped in the affected
    ipykernel versions joins that thread unconditionally and raises
    ``AttributeError: 'OutStream' object has no attribute 'watch_fd_thread'``
    (ipython/ipykernel#867). That AttributeError propagates out of
    ``uvicorn.Config(...)`` and aborts startup ("Unsloth Studio failed to start").

    Wrap the stream's ``close()`` in a transparent pass-through that swallows
    ONLY that specific teardown AttributeError. A healthy close() (a real console
    stream, or an OutStream with fd-watching on) runs to completion exactly as
    before and any other error still propagates, so nothing changes off Colab. A
    stream whose ``close`` cannot be reassigned keeps its original close().
    """
    if stream is None:
        return
    try:
        _orig_close = stream.close
    except Exception:
        return

    def _safe_close(*args, **kwargs):
        try:
            return _orig_close(*args, **kwargs)
        except AttributeError as exc:
            if not _is_missing_watch_fd_thread(exc):
                # A real teardown failure; never hide it.
                raise
            # ipython/ipykernel#867: watchfd=False OutStream.close() joins a
            # thread that was never created. Nothing to clean up; keep going.
            return None

    try:
        stream.close = _safe_close
    except (AttributeError, TypeError):
        # A stream that forbids setting instance attributes; leave it as-is.
        pass


def _setup_server_disk_logging():
    """Tee stdout/stderr to ~/.unsloth/studio/logs/server/ and aim
    faulthandler at the same file so hard crashes (access violations /
    SIGSEGV in the GPU runtime) leave a stack trace on disk.

    Also exports PYTHONFAULTHANDLER=1 so child Python processes (training
    workers) dump native-crash stacks to their captured stderr. Keeps the
    newest 20 session logs. Opt out with UNSLOTH_STUDIO_NO_FILE_LOG=1.
    Returns the log path, or None when disabled/unavailable.
    """
    if os.environ.get("UNSLOTH_STUDIO_NO_FILE_LOG") == "1":
        return None
    try:
        from utils.paths import studio_root
        log_dir = Path(studio_root()) / "logs" / "server"
    except Exception:
        home = (
            os.environ.get("UNSLOTH_STUDIO_HOME")
            or os.environ.get("STUDIO_HOME")
            or os.path.join(os.path.expanduser("~"), ".unsloth", "studio")
        )
        log_dir = Path(home) / "logs" / "server"
    try:
        log_dir.mkdir(parents = True, exist_ok = True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"server-{stamp}-pid{os.getpid()}.log"
        # Line-buffered so the tail survives a hard kill; errors="replace"
        # so a console encoding quirk can never take the server down.
        log_fh = open(log_path, "w", encoding = "utf-8", errors = "replace", buffering = 1)
    except Exception:
        return None

    import faulthandler

    try:
        faulthandler.enable(file = log_fh, all_threads = True)
    except Exception:
        pass
    # Children (training workers) inherit: their native-crash stacks land on
    # the stderr the server already captures.
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")

    # Replacing the console streams orphans them from third-party "is this the
    # live console?" checks, so guard their close() first (ipython/ipykernel#867).
    _harden_console_close(sys.stdout)
    _harden_console_close(sys.stderr)

    # _normalize_standard_streams() ran at import, so these are never None; the
    # tee's own None guards are defence-in-depth for _TeeStream(None, ...).
    sys.stdout = _TeeStream(sys.stdout, log_fh)
    sys.stderr = _TeeStream(sys.stderr, log_fh)

    # Best-effort retention: keep the newest 20 session logs.
    try:
        logs = sorted(log_dir.glob("server-*.log"), key = lambda p: p.stat().st_mtime)
        for old in logs[:-20]:
            old.unlink(missing_ok = True)
    except Exception:
        pass
    return log_path


def _cloudflare_tunnel_should_start(
    *, cloudflare: bool, host: str, secure: bool, api_only: bool, is_colab: bool
) -> bool:
    """Whether to start the Cloudflare tunnel. --secure exposes only the tunnel
    (loopback bind), so it tunnels even api-only (headless secure API serving);
    otherwise tunnel wildcard binds, never api-only (Tauri) or Colab."""
    if is_colab or not cloudflare:
        return False
    if secure:
        return True
    return host in ("0.0.0.0", "::") and not api_only


def _final_bound_port(server, requested_port: int) -> int:
    """Resolve Uvicorn's OS-assigned port after readiness for ``port=0``."""
    if requested_port > 0:
        return requested_port
    for listener in getattr(server, "servers", ()):
        for sock in getattr(listener, "sockets", ()) or ():
            address = sock.getsockname()
            if isinstance(address, tuple) and len(address) >= 2 and int(address[1]) > 0:
                return int(address[1])
    raise RuntimeError("Uvicorn did not expose its final bound port")


_CLOUDFLARE_INTENT_ENV = "_UNSLOTH_CLOUDFLARE_INTENT"


def _consume_cloudflare_intent(cloudflare: "Optional[bool]", secure: bool) -> str:
    """Resolve user intent without confusing a compatibility flag with opt-out.

    An explicit choice on THIS invocation wins: letting the inherited marker
    override it would let a stale export, Docker ENV or systemd Environment=
    re-enable a tunnel the user opted out of.
    """
    inherited = os.environ.pop(_CLOUDFLARE_INTENT_ENV, None)
    if secure or cloudflare is True:
        return "enabled"
    if cloudflare is False:
        # Only "the parent omitted the option" softens this into unselected.
        return "unset" if inherited == "unset" else "disabled"
    if inherited in {"unset", "enabled", "disabled"}:
        return inherited
    return "unset"


def _stream_isatty(stream) -> bool:
    """isatty() that treats broken streams as non-interactive.

    isatty() can raise under service wrappers (closed stdin -> ValueError;
    sys.stdin None in Windows GUI -> AttributeError); such a stream can't host a
    prompt, which is a fallback, not an error.
    """
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _terminal_password_gate(
    *,
    tunnel_will_start: bool,
    host: str,
    secure: bool,
    api_only: bool,
    frontend_served: bool,
    is_colab: bool = False,
) -> Tuple[bool, bool]:
    """Force a terminal password change before the public tunnel goes up.

    When the tunnel is about to publish Unsloth and the seeded admin password was
    never changed, ask for a new one (masked, confirmed) before any public URL
    exists. The CLI normally does this before re-exec'ing the backend; this is
    the backstop for direct `python run.py` launches and older-CLI installs.
    Must run BEFORE the uvicorn socket binds: on a wildcard bind the served HTML
    injects the bootstrap credential, so a pre-gate listener would hand the
    default password to anyone reaching the raw port while the operator types.

    Returns (proceed, drop_bootstrap_injection):
      proceed False -> abort the launch (interactive refusal, or a headless
        public launch nothing would protect); fail closed.
      drop_bootstrap_injection True -> caller must null
        app.state.bootstrap_password: the password just changed (stale), or a
        public URL is about to serve the default credential and must not leak it.

    Without a usable terminal the prompt is skipped: proceed if the bootstrap
    deadline (armed later) will protect the launch; if even that is disabled
    (api-only, timeout 0) nothing protects it, so refuse. NOT wrapped in a broad
    try/except: an auth storage failure must abort rather than expose the default.
    """
    if not tunnel_will_start:
        return True, False

    from auth import hashing as _auth_hashing
    from auth import storage as _auth_storage
    from auth.bootstrap_timeout import (
        bootstrap_timeout_seconds,
        should_arm_bootstrap_timeout,
    )
    from auth.terminal_prompt import (
        prompt_for_password_change,
        should_prompt_password_change,
    )

    _admin = _auth_storage.DEFAULT_ADMIN_USERNAME
    # Gate can run before lifespan: seed the admin row here (idempotent).
    _auth_storage.ensure_default_admin()
    requires_change = _auth_storage.requires_password_change(_admin)
    if not requires_change:
        return True, False

    if not should_prompt_password_change(
        tunnel_will_start = tunnel_will_start,
        requires_change = requires_change,
        stdin_isatty = _stream_isatty(sys.stdin),
        stderr_isatty = _stream_isatty(sys.stderr),
    ):
        # No terminal: only proceed if the bootstrap deadline will arm; api-only
        # and TIMEOUT=0 never arm it, leaving the default credential public.
        deadline_arms = should_arm_bootstrap_timeout(
            host = host,
            secure = secure,
            api_only = api_only,
            frontend_served = frontend_served,
            is_colab = is_colab,
            requires_change = True,
            timeout_seconds = bootstrap_timeout_seconds(),
        )
        if not deadline_arms:
            print(
                "Refusing to publish Unsloth on a public Cloudflare URL: the "
                "default admin password was never changed, no terminal is "
                "attached to change it here, and the bootstrap shutdown "
                "deadline does not apply to this launch (api-only, or "
                "UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0). Change the password "
                "first (run `unsloth studio` locally and log in, or re-run "
                "with a terminal attached), then retry.",
                file = sys.stderr,
                flush = True,
            )
            return False, False
        # The public page won't auto-fill the bootstrap credential (suppressed
        # below) and the seeded file may already be gone, so point recovery at a
        # terminal-attached run / reset-password instead of reading it from disk.
        print(
            "  WARNING: the default admin password is still active while "
            "Unsloth is about to be published on a public Cloudflare URL, and "
            "no terminal is attached to change it here. The public page will "
            "NOT auto-fill the bootstrap credential. Set a new password by "
            "running `unsloth studio` locally with a terminal attached, or "
            "`unsloth studio reset-password`. Unsloth shuts down after the "
            "bootstrap deadline (UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT, default 1h) "
            "unless the password is changed.",
            file = sys.stderr,
            flush = True,
        )
        # Never serve the default credential in HTML over a public URL.
        return True, True

    def _is_current_password(candidate: str) -> bool:
        record = _auth_storage.get_user_and_secret(_admin)
        if record is None:
            return False
        salt, pwd_hash, _jwt_secret, _must_change = record
        return _auth_hashing.verify_password(candidate, salt, pwd_hash)

    def _apply_change(new_password: str) -> None:
        # Same effects as routes/auth.py change_password: rehash, rotate the JWT
        # secret, revoke refresh tokens in the SAME transaction.
        _auth_storage.update_password(_admin, new_password, revoke_refresh_tokens = True)

    changed = prompt_for_password_change(
        min_length = _auth_storage.MIN_PASSWORD_LENGTH,
        is_current_password = _is_current_password,
        apply_change = _apply_change,
        out = sys.stderr,
    )
    return (True, True) if changed else (False, False)


def _apply_supplied_password(password_value: "Optional[str]") -> None:
    """Non-interactively set the INITIAL admin password before the socket binds,
    for a direct ``python run.py`` launch (the CLI does this in its own parent).
    Value comes from --password / UNSLOTH_STUDIO_PASSWORD / stdin.

    Only ever sets the FIRST password: an already-set one is a hard error, an
    invalid value fails closed. NOT wrapped in a broad try/except: an auth
    storage failure must abort rather than expose the default credential.
    """
    from auth import hashing as _auth_hashing
    from auth import storage as _auth_storage
    from auth.terminal_prompt import SUPPLIED_PASSWORD_ENV, resolve_supplied_password

    supplied = resolve_supplied_password(password_value)
    # Strip the env var once read so child subprocesses (cloudflared, llama-server,
    # code-exec tools) can't inherit the plaintext via /proc/PID/environ. Mirrors
    # the CLI. Unconditional: strips a leftover value even when a literal --password won.
    os.environ.pop(SUPPLIED_PASSWORD_ENV, None)
    if not supplied:
        return

    _admin = _auth_storage.DEFAULT_ADMIN_USERNAME
    _auth_storage.ensure_default_admin()
    if not _auth_storage.requires_password_change(_admin):
        print(
            "Error: an Unsloth admin password is already set; --password only sets "
            "the initial password. Change it in the UI, or run `unsloth studio "
            "reset-password` for a new one.",
            file = sys.stderr,
            flush = True,
        )
        sys.exit(1)

    def _is_current_password(candidate: str) -> bool:
        record = _auth_storage.get_user_and_secret(_admin)
        if record is None:
            return False
        salt, pwd_hash, _jwt_secret, _must_change = record
        return _auth_hashing.verify_password(candidate, salt, pwd_hash)

    if len(supplied) < _auth_storage.MIN_PASSWORD_LENGTH:
        print(
            f"Error: password must be at least {_auth_storage.MIN_PASSWORD_LENGTH} "
            "characters; not starting.",
            file = sys.stderr,
            flush = True,
        )
        sys.exit(1)
    if any(ch.isspace() for ch in supplied):
        print(
            "Error: password cannot contain spaces; not starting.",
            file = sys.stderr,
            flush = True,
        )
        sys.exit(1)
    if _is_current_password(supplied):
        print(
            "Error: the new password must differ from the current bootstrap "
            "password; not starting.",
            file = sys.stderr,
            flush = True,
        )
        sys.exit(1)
    _auth_storage.update_password(_admin, supplied, revoke_refresh_tokens = True)
    print(f"Password updated for '{_admin}'.", file = sys.stderr, flush = True)


def _apply_cli_tool_policy(enable_tools: "Optional[bool]") -> None:
    """Honor an explicit --enable-tools/--disable-tools; None leaves the policy
    unset (tools default on, per-request enable_tools honored). Host is never
    inspected here."""
    if enable_tools is None:
        return
    from state.tool_policy import set_tool_policy

    set_tool_policy(enable_tools)


# Mirror unsloth_cli/commands/studio.py's _PARALLEL_*: the admission queue caps concurrent
# chats at the slot count, so a direct launch matches the CLI (VRAM fit may still cut it
# back). Defined above run_server() so embedders that omit it do not serialise every chat.
_PARALLEL_MIN = 1
_PARALLEL_MAX = 64
_PARALLEL_DEFAULT_PLAIN = 4


def run_server(
    host: str = "127.0.0.1",
    port: int = 8888,
    frontend_path: Path = _DEFAULT_FRONTEND_PATH,
    silent: bool = False,
    api_only: bool = False,
    llama_parallel_slots: int = _PARALLEL_DEFAULT_PLAIN,
    cloudflare: "Optional[bool]" = None,
    secure: bool = False,
    enable_tools: "Optional[bool]" = None,
    password: "Optional[str]" = None,
    emit_tauri_port: bool = True,
    abort_if_own_studio: "Optional[bool]" = None,
):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to (auto-increments if in use)
        frontend_path: Path to frontend build directory (optional)
        silent: Suppress startup messages
        api_only: API server only, except that a Tauri-owned backend serves its
            packaged frontend exclusively through a live Cloudflare tunnel
        llama_parallel_slots: parallel slots for llama-server (default
            _PARALLEL_DEFAULT_PLAIN, matching the CLI entry points)
        cloudflare: opt in to the public Cloudflare HTTPS tunnel for a wildcard
            bind. Tri-state: None (unset) and False both mean off; True enables it.
            --secure implies it (True) and rejects an explicit False.
        enable_tools: explicit --enable-tools/--disable-tools policy; None leaves
            the default (tools on, per-request enable_tools honored)
        emit_tauri_port: print the machine-readable TAURI_PORT line the desktop
            app parses from stdout; the headless `run --api-only` path turns it
            off so it does not pollute the documented URL/API-key banner

    Note:
        Signal handlers are NOT registered here so embedders (e.g. Colab) keep
        their own interrupt semantics; standalone callers register them after.
    """
    global _server, _server_thread, _shutdown_event

    boot_started = time.perf_counter()
    logger.info("run_server startup begin api_only=%s host=%s port=%s", api_only, host, port)
    cloudflare_intent = _consume_cloudflare_intent(cloudflare, secure)

    # Reap every child if the parent dies abnormally (terminal close, Task
    # Manager kill, SIGKILL); must run before any child can spawn.
    from utils.process_lifetime import initialize_parent_lifetime

    initialize_parent_lifetime()

    # --secure exposes ONLY the Cloudflare link: reject --secure --no-cloudflare,
    # then force a loopback bind so the raw port is never public (even -H 0.0.0.0).
    # Otherwise keep the tri-state so the banner distinguishes "off by default"
    # from an explicit --no-cloudflare.
    if secure:
        if cloudflare is False:
            raise SystemExit(
                "--secure requires the Cloudflare tunnel; do not combine it with --no-cloudflare."
            )
        cloudflare = True
        host = "127.0.0.1"

    # `unsloth studio run` installs its own resolved policy and passes None here.
    _apply_cli_tool_policy(enable_tools)

    # Windows cp1252 can't encode emoji; reconfigure stdout to UTF-8.
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass

    # Persist a session log + native-crash stacks BEFORE importing main, so
    # even import-time failures leave evidence on disk. Field report: Unsloth
    # "terminates without a warning" -- a native crash in the GPU runtime
    # kills the process with no Python traceback, and a desktop-shortcut
    # console closes before anything can be read. Console-only logging made
    # that undiagnosable.
    _session_log = _setup_server_disk_logging()
    if _session_log is not None and not silent:
        print(f"Session log: {_session_log}")

    # Set env vars BEFORE importing main so CORS middleware picks them up.
    # secure api-only is a remote server behind Cloudflare, so it keeps the
    # any-origin CORS profile; plain api-only stays locked to the Tauri app.
    if api_only:
        os.environ["UNSLOTH_API_ONLY"] = "1"
    if secure:
        os.environ["UNSLOTH_SECURE"] = "1"

    import asyncio

    # nest_asyncio is for Colab/IPython, where the main thread already runs a loop
    # the blocking waits below would collide with. Apply it only with a loop running
    # (a plain CLI start has nothing to nest) and only on Python <= 3.13: on 3.14+
    # its global Task patch leaves asyncio.current_task() None (tracking moved into
    # C), which also breaks the background uvicorn loop and 500s every request. It
    # is archived upstream, so no 3.14 fix is coming; skip it there.
    if sys.version_info < (3, 14):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            import nest_asyncio
            nest_asyncio.apply()

    from threading import Thread, Event
    import uvicorn

    # `from main import app` below loads torch/unsloth/transformers (~2 min cold,
    # silent), so print a flushed heads-up (piped stdout is block-buffered).
    if not silent:
        print(
            "Loading Unsloth Studio, please wait... (this can take a few minutes)",
            flush = True,
        )
        print("  - loading PyTorch, Unsloth and Transformers...", flush = True)

    import_started = time.perf_counter()

    from main import app, setup_frontend, _desktop_owner, _IS_COLAB

    logger.info(
        "Imported FastAPI app in %.1fms",
        (time.perf_counter() - import_started) * 1000,
    )
    if not silent:
        print("  - Starting server...", flush = True)
    from utils.paths import ensure_studio_directories

    # Allow local stdio MCP servers on a loopback bind (the user's own machine),
    # but never on Colab, which is a hosted VM reachable through its proxy. The
    # gate reads the env var at request time, so this need not precede the import.
    from utils.host_policy import apply_stdio_mcp_loopback_default

    apply_stdio_mcp_loopback_default(host, is_colab = _IS_COLAB)

    # Create all standard directories on startup.
    ensure_studio_directories()

    logger.info(
        "Ensured Unsloth directories in %.1fms",
        (time.perf_counter() - boot_started) * 1000,
    )

    # Auto-find a free port if the requested one is in use.
    original_port = port
    # Refusing rather than falling back is for callers that cannot follow us to
    # the new port. `studio run` reads app.state.server_port back and the desktop
    # app reads TAURI_PORT, so both should keep the plain fallback; only the
    # bare launch, which has nothing but the banner, benefits from the refusal.
    if abort_if_own_studio is None:
        abort_if_own_studio = not api_only
    port = _resolve_port(host, port, avoid_own_studio = abort_if_own_studio)
    if port != original_port:
        blocker = _get_pid_on_port(original_port)
        if not silent:
            print("")
            print("=" * 50)
            if blocker:
                pid, name = blocker
                print(f"Port {original_port} is already in use by {name} (PID {pid}).")
            else:
                print(f"Port {original_port} is already in use.")
            print(f"Unsloth Studio will use port {port} instead.")
            print(f"Open http://localhost:{port} in your browser.")
            print("=" * 50)
            print("")

    _serve_frontend, _tunnel_only_frontend = _frontend_serving_mode(
        api_only = api_only,
        desktop_owned = _desktop_owner() is not None,
    )

    # A desktop-owned API-only backend mounts the packaged SPA behind a
    # Cloudflare-only request gate so Remote access can serve the WebUI without
    # changing the direct local API-only surface.
    if frontend_path and _serve_frontend:
        chosen, attempted = _resolve_frontend_path(Path(frontend_path))
        if chosen is not None and setup_frontend(app, chosen, tunnel_only = _tunnel_only_frontend):
            if not silent:
                # Resolve so logs show an absolute path for support.
                try:
                    display = chosen.resolve()
                except OSError:
                    display = chosen
                print(f"[OK] Frontend loaded from {display}")
        elif not _missing_frontend_is_fatal(tunnel_only = _tunnel_only_frontend):
            # Remote access serves nothing; the local API the desktop asked for
            # still comes up. The tunnel gate already 404s every other request.
            logger.warning(
                "No frontend build found, so Remote access will not serve the web UI. Tried: %s",
                ", ".join(str(p) for p in attempted) or "(none)",
            )
        else:
            home_str = (
                os.environ.get("UNSLOTH_STUDIO_HOME")
                or os.environ.get("STUDIO_HOME")
                or str(Path.home() / ".unsloth" / "studio")
            )
            # Windows shim: $STUDIO_HOME/bin/unsloth.exe; Linux/macOS venv binary:
            # $STUDIO_HOME/unsloth_studio/bin/unsloth.
            home = Path(home_str).expanduser()
            if sys.platform == "win32":
                installer_bin = home / "bin" / "unsloth.exe"
            else:
                installer_bin = home / "unsloth_studio" / "bin" / "unsloth"
            tried_lines = "\n".join(f"  - {p}" for p in attempted) or "  (none)"
            raise SystemExit(
                "[ERROR] Unsloth frontend build not found.\n"
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
    display_host = _display_host_for_bind(host)
    _install_uvicorn_startup_log_rewrite(host, display_host)
    # LoggingMiddleware already logs every unhandled request exception with its full
    # traceback as a structured event; without this uvicorn prints the same traceback
    # again on stderr and the desktop shell copies it into tauri.log line by line.
    install_uvicorn_duplicate_exception_filter()

    logger.info(
        "run_server pre-uvicorn setup completed in %.1fms",
        (time.perf_counter() - boot_started) * 1000,
    )

    ready_event = Event()
    startup_failed = Event()
    startup_errors = []

    class _ReadyServer(uvicorn.Server):
        async def startup(self, *args, **kwargs):
            await super().startup(*args, **kwargs)
            if getattr(self, "started", False) and not self.should_exit:
                logger.info(
                    "Uvicorn startup hook completed in %.1fms",
                    (time.perf_counter() - boot_started) * 1000,
                )
                ready_event.set()

    # server_header=False suppresses uvicorn's "Server: uvicorn"; SecurityHeadersMiddleware sets its own.
    config_kwargs = dict(
        host = host,
        port = port,
        log_level = "info",
        access_log = False,
        server_header = False,
    )
    # Colab only: trust X-Forwarded-* from Colab's reverse proxy so the app sees
    # the real https origin. forwarded_allow_ips="*" is safe in Colab's
    # single-user sandbox but too lax for local/standalone, so leave uvicorn's
    # loopback-only default elsewhere.
    if _IS_COLAB:
        config_kwargs["proxy_headers"] = True
        config_kwargs["forwarded_allow_ips"] = "*"
    config = uvicorn.Config(app, **config_kwargs)
    _server = _ReadyServer(config)
    _shutdown_event = Event()

    # Expose the actual bound port so handlers build loopback URLs at the real
    # backend, not whatever a proxy/tunnel exposed. For ephemeral binds (port==0)
    # leave it unset so handlers fall back to the request scope / base_url.
    app.state.server_port = port if port and port > 0 else None
    # Direct (non-tunnel) base for the API panel; resolve wildcard binds to the LAN IP.
    if port and port > 0:
        _direct_host = _display_host_for_bind(host)
        app.state.server_url = f"http://{_url_host(_direct_host)}:{port}"
    else:
        app.state.server_url = None
    app.state.secure = secure
    app.state.llama_parallel_slots = llama_parallel_slots

    global _cloudflare_url, _cloudflare_requested, _cloudflare_flag
    _cloudflare_url = None
    _cloudflare_flag = cloudflare
    app.state.cloudflare_url = None
    from utils.remote_access_settings import configure_remote_access

    _launch_tunnel_managed = _cloudflare_tunnel_should_start(
        cloudflare = cloudflare,
        host = host,
        secure = secure,
        api_only = api_only,
        is_colab = _IS_COLAB,
    )
    configure_remote_access(
        app.state,
        port = port,
        intent = cloudflare_intent,
        is_colab = _IS_COLAB,
        launch_managed = _launch_tunnel_managed,
    )

    # Expose a shutdown callable before the server accepts requests so
    # /api/shutdown is ready as soon as readiness publishes.
    def _trigger_shutdown():
        _graceful_shutdown(_server)
        if _shutdown_event is not None:
            _shutdown_event.set()

    app.state.trigger_shutdown = _trigger_shutdown

    # A supplied --password / UNSLOTH_STUDIO_PASSWORD / stdin sets the initial
    # admin password before the gate and socket bind (direct `python run.py`;
    # the CLI applies it in its own parent).
    _apply_supplied_password(password)

    # Never publish with the seeded default password active: prompt first (or
    # warn / fail closed headless; see _terminal_password_gate). Runs BEFORE the
    # socket binds so a pre-gate listener can't hand out the injected credential.
    _pw_proceed, _pw_drop_bootstrap = _terminal_password_gate(
        tunnel_will_start = _launch_tunnel_managed,
        host = host,
        secure = secure,
        api_only = api_only,
        frontend_served = bool(frontend_path) and not api_only,
        is_colab = _IS_COLAB,
    )
    if not _pw_proceed:
        print(
            "Not starting Unsloth; set a new admin password first, or launch "
            "without --secure/--cloudflare.",
            file = sys.stderr,
            flush = True,
        )
        sys.exit(1)
    if _pw_drop_bootstrap:
        # Password just changed (stale) or a public URL is about to serve the
        # default credential: don't leak it in the HTML. Lifespan runs AFTER this
        # and re-reads the bootstrap password, so the flag (not a plain None)
        # makes it skip that re-read.
        app.state.suppress_bootstrap_injection = True
        app.state.bootstrap_password = None

    from cloudflare_tunnel import open_studio_tunnel_lifecycle

    open_studio_tunnel_lifecycle()

    # Run server in a daemon thread with explicit new_event_loop() +
    # run_until_complete() (not asyncio.run) so nest_asyncio's patches don't
    # interfere when Colab/IPython already runs a loop on the main thread.
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
    _server_thread = thread
    thread.start()

    # Wait until uvicorn finishes lifespan startup and binds sockets, or until it
    # exits/fails first. No deadline: a slow but live startup stays in progress.
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

    logger.info(
        "run_server uvicorn ready after %.1fms",
        (time.perf_counter() - boot_started) * 1000,
    )

    port = _final_bound_port(_server, port)
    app.state.server_port = port
    app.state.server_url = f"http://{_url_host(_display_host_for_bind(host))}:{port}"
    app.state.remote_access_port = port

    _write_pid_file(port, host)
    import atexit

    atexit.register(_remove_pid_file)
    from utils.process_lifetime import terminate_all

    atexit.register(terminate_all)

    # Output port for Tauri (api-only), only after sockets bind and startup done.
    # The headless `run --api-only` path opts out so it does not leak this line.
    if api_only and emit_tauri_port:
        print(f"TAURI_PORT={port}", flush = True)
        # Desktop-owned backends only (the owner env handshake): a headless
        # `unsloth studio --api-only` has no app to bind its lifetime to and
        # must survive its terminal (e.g. nohup). If the app dies without
        # running its cleanup, exit instead of orphaning on the port.
        from main import _desktop_owner
        if _desktop_owner() is not None:
            from utils.parent_watchdog import start_parent_watchdog
            owner_pid = os.environ.pop("UNSLOTH_STUDIO_DESKTOP_OWNER_PID", "")
            start_parent_watchdog(
                _trigger_shutdown,
                parent_pid = int(owner_pid) if owner_pid.isdigit() else None,
            )

    from cloudflare_tunnel import (
        set_studio_tunnel_runtime_callback,
        set_studio_tunnel_url_callback,
    )
    from utils.host_policy import set_remote_connector_active

    set_studio_tunnel_runtime_callback(set_remote_connector_active)
    set_studio_tunnel_url_callback(lambda url: _publish_cloudflare_url(app.state, url))
    app.state.remote_access_ready = True

    # Free trycloudflare.com tunnel for wildcard binds (the raw ip:port is often
    # unreachable). Started pre-banner and even when silent so the CLI banner can
    # read app.state.cloudflare_url; torn down by _graceful_shutdown.
    _cloudflare_enabled = _launch_tunnel_managed
    _cloudflare_requested = _cloudflare_enabled

    if _cloudflare_enabled:
        try:  # best-effort: any failure must not block startup
            from cloudflare_tunnel import start_studio_tunnel
            start_studio_tunnel(port, managed_by = "launch")
        except Exception as e:
            logger.debug("Cloudflare tunnel skipped: %s", e)

    # Backstop for both launch- and settings-managed tunnels on abnormal exits.
    from cloudflare_tunnel import close_studio_tunnel_lifecycle

    atexit.register(close_studio_tunnel_lifecycle)

    # --secure fails closed: no tunnel means no public link, so exit rather than
    # silently fall back to a raw port.
    if secure and not _cloudflare_url:
        print(
            "A secure Cloudflare link is not allowed, use --no-secure which provides a 0.0.0.0 link",
            file = sys.stderr,
            flush = True,
        )
        _graceful_shutdown(_server)
        sys.exit(1)

    # Time-box a freshly-exposed web UI: if nobody changes the seeded admin
    # password within the deadline (default 1h), shut down rather than leave an
    # unsecured public instance running. No-op for loopback, --api-only, Colab,
    # an already-changed password, or UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0.
    try:
        from auth import storage as _auth_storage
        from auth.bootstrap_timeout import (
            arm_bootstrap_timeout,
            bootstrap_timeout_seconds,
            should_arm_bootstrap_timeout,
        )

        _bootstrap_timeout = bootstrap_timeout_seconds()
        if should_arm_bootstrap_timeout(
            host = host,
            secure = secure,
            api_only = api_only,
            frontend_served = bool(frontend_path) and not api_only,
            is_colab = _IS_COLAB,
            requires_change = _auth_storage.requires_password_change(
                _auth_storage.DEFAULT_ADMIN_USERNAME
            ),
            timeout_seconds = _bootstrap_timeout,
        ):
            arm_bootstrap_timeout(
                _auth_storage,
                _trigger_shutdown,
                timeout_seconds = _bootstrap_timeout,
                logger = logger,
            )
            logger.info(
                "Unsloth will shut down in %ds unless the default admin password is changed.",
                _bootstrap_timeout,
            )
    except Exception as e:  # best-effort: never block startup on the timeout
        logger.warning("Bootstrap timeout not armed: %s", e)

    from utils.remote_access_settings import maybe_auto_start_remote_access

    if maybe_auto_start_remote_access(app.state):
        logger.info("Remote access auto-start scheduled")

    if not silent:
        _emit_startup_output(host, port, display_host, secure = secure, enable_tools = enable_tools)

    return app


def _build_arg_parser():
    """Build the backend CLI argument parser.

    Extracted from the __main__ block so the flag wiring (notably the
    --secure/--no-secure polarity and its --not-secure alias) stays unit-testable.
    """
    import argparse

    parser = argparse.ArgumentParser(description = "Run Unsloth UI Backend server")
    parser.add_argument(
        "--host",
        default = "127.0.0.1",
        help = "Host to bind to (default: 127.0.0.1; use 0.0.0.0 for network/cloud access)",
    )
    parser.add_argument(
        "--password",
        default = None,
        help = "Set the INITIAL admin password non-interactively (headless), only when "
        "none is set yet. Also reads UNSLOTH_STUDIO_PASSWORD, or --password - for stdin. "
        "A literal value is visible in the process list. Rotate later via "
        "`unsloth studio reset-password`.",
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
    parser.add_argument(
        "--cloudflare",
        action = argparse.BooleanOptionalAction,
        default = None,
        help = "Expose Unsloth on a PUBLIC internet URL via a free Cloudflare HTTPS "
        "tunnel, for non-api-only wildcard binds (0.0.0.0 or ::). Off by default; "
        "pass --cloudflare to enable it (--secure implies it), --no-cloudflare to "
        "force it off. It does not change a raw wildcard bind. If the admin "
        "password was never changed, Unsloth asks for a new one in the terminal "
        "before publishing the URL.",
    )
    parser.add_argument(
        "--secure",
        action = argparse.BooleanOptionalAction,
        default = False,
        help = "Expose ONLY a Cloudflare HTTPS link: bind localhost and fail closed "
        "if the tunnel can't start. Without it, --no-secure also serves the raw "
        "0.0.0.0 port, which is reachable from anywhere on the network. If the "
        "admin password was never changed, Unsloth asks for a new one in the "
        "terminal before publishing the URL.",
    )
    # Back-compat: accept --not-secure as a hidden alias for --no-secure.
    parser.add_argument(
        "--not-secure",
        dest = "secure",
        action = "store_false",
        default = argparse.SUPPRESS,
        help = argparse.SUPPRESS,
    )
    # Tri-state tool policy: no flag -> None (tools on, per-request honored);
    # --enable-tools/--disable-tools force on/off.
    parser.add_argument(
        "--enable-tools",
        dest = "enable_tools",
        action = "store_true",
        default = None,
        help = "Force server-side tools (web search, code execution) on for "
        "every request. Default: on for every bind, per-request setting honored. "
        "/v1/messages takes the on direction per request (enable_tools) because it has "
        "no confirmation channel; the off direction still applies everywhere.",
    )
    parser.add_argument(
        "--disable-tools",
        dest = "enable_tools",
        action = "store_false",
        default = None,
        help = "Force server-side tools off for every request.",
    )
    parser.add_argument(
        "--disable-dns-pinning",
        action = "store_true",
        help = "Allow hostname-based web fetches for enterprise proxies. WARNING: weakens "
        "DNS-rebinding protection; hostname and redirect validation remain enabled.",
    )
    parser.add_argument(
        "--parallel",
        "--n-parallel",
        type = int,
        default = _PARALLEL_DEFAULT_PLAIN,
        help = (
            f"llama-server parallel decode slots ({_PARALLEL_MIN}..{_PARALLEL_MAX}). "
            f"Default {_PARALLEL_DEFAULT_PLAIN}. The Studio run settings "
            "(Parallel Slots) override it per load."
        ),
    )
    return parser


# For direct execution (also invoked by CLI via os.execvp / subprocess).
if __name__ == "__main__":
    # Correct a conflicting system CUDA on LD_LIBRARY_PATH before torch is
    # imported (below, via run_server). Re-execs once on Linux so the dynamic
    # linker uses torch's bundled CUDA libs; no-op on other platforms, when
    # LD_LIBRARY_PATH is unset or already correct, or after the single re-exec.
    _maybe_reexec_for_cuda_ld_path()

    import signal
    import traceback

    # Ensure stderr handles Unicode on Windows (non-ASCII path tracebacks).
    if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass

    parser = _build_arg_parser()
    args = parser.parse_args()
    if not _PARALLEL_MIN <= args.parallel <= _PARALLEL_MAX:
        parser.error(f"--parallel must be between {_PARALLEL_MIN} and {_PARALLEL_MAX}")
    if args.secure and args.cloudflare is False:
        parser.error(
            "--secure requires the Cloudflare tunnel; do not combine it with --no-cloudflare"
        )
    if args.disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    kwargs = dict(
        host = args.host,
        port = args.port,
        silent = args.silent,
        api_only = args.api_only,
        llama_parallel_slots = args.parallel,
        cloudflare = args.cloudflare,
        secure = args.secure,
        enable_tools = args.enable_tools,
        password = args.password,
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
        # Restore defaults so a second signal force-quits if shutdown stalls.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_DFL)
        _graceful_shutdown(_server)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # On Windows, some terminals send SIGBREAK for Ctrl+C / Ctrl+Break.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    # Keep running until shutdown signal. Event.wait() without a timeout blocks at
    # the C level on Linux, preventing SIGINT delivery; a short timeout in a loop
    # lets the interpreter process pending signals.
    while not _shutdown_event.is_set():
        _shutdown_event.wait(timeout = 1)
    _wait_for_server_shutdown()

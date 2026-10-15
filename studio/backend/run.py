# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import errno
import os
import sys
import time
from pathlib import Path
from typing import NoReturn, Optional, Sequence, Tuple


def _normalize_standard_streams():
    """Must run before the loggers import: structlog binds stdout at import, so a None stdout is kept."""
    for name, mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
        if getattr(sys, name, None) is not None:
            continue
        try:
            stream = open(os.devnull, mode, encoding = "utf-8", errors = "replace")
        except Exception:
            continue
        setattr(sys, name, stream)
        # sys.__stdout__ etc are None too; rich reads sys.__stdout__.fileno() at import.
        if getattr(sys, f"__{name}__", None) is None:
            setattr(sys, f"__{name}__", stream)


_normalize_standard_streams()


def _is_tegra():
    """utils.tegra.is_tegra, inlined: this runs before the backend is on sys.path."""
    try:
        if os.path.exists("/etc/nv_tegra_release"):
            return True
        with open("/proc/device-tree/compatible", "rb") as fh:
            return b"nvidia,tegra" in fh.read()
    except OSError:
        return False


def _fix_torch_cuda_ld_path():
    """Prepends torch's bundled CUDA libs; a foreign LD_LIBRARY_PATH shadows them and breaks import."""
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
        # Jetson: JetPack's CUDA stays first (#4862).
        if os.path.isdir(nvidia_dir) and not _is_tegra():
            for sub in sorted(os.listdir(nvidia_dir)):
                lib = os.path.join(nvidia_dir, sub, "lib")
                if os.path.isdir(lib):
                    lib_dirs.append(lib)
        if not lib_dirs:
            return False

        existing = ld_path.split(":")
        if existing[: len(lib_dirs)] == lib_dirs:
            return False

        torch_set = set(lib_dirs)
        cleaned = [p for p in existing if p not in torch_set]
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + cleaned)
        return True
    except Exception:
        return False


_LD_FIXED_SENTINEL = "_UNSLOTH_STUDIO_LD_FIXED"


def _maybe_reexec_for_cuda_ld_path():
    """Only from a true entry point: os.execv replaces the process, so embedders must not re-exec."""
    if _LD_FIXED_SENTINEL in os.environ:
        return
    if not _fix_torch_cuda_ld_path():
        return
    os.environ[_LD_FIXED_SENTINEL] = "1"
    argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    os.execv(sys.executable, argv)


os.environ["PYTHONWARNINGS"] = "ignore"

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# First, before anything sizes an OpenMP/BLAS pool. Imports stdlib only.
from utils.cpu_threads import configure_cpu_threads

try:
    configure_cpu_threads()
except ValueError as exc:
    configured = os.environ.get("UNSLOTH_CPU_THREADS")
    raise SystemExit(f"Error: Invalid UNSLOTH_CPU_THREADS value {configured!r}: {exc}") from None

# Windows ROCm has no distributed backend, so torchao and xformers die on import. Stubs must
# precede the first import below.
from core._torchao_stub import (
    alias_torchao_fqn_to_config,
    hide_xformers_built_for_another_torch,
    install_torchao_windows_rocm_stub,
    install_xformers_windows_rocm_stub,
)

install_xformers_windows_rocm_stub()
hide_xformers_built_for_another_torch()
install_torchao_windows_rocm_stub()
alias_torchao_fqn_to_config()

# Conda Python: seed platform._sys_version_cache (https://github.com/python/cpython/issues/102396).
import _platform_compat  # noqa: F401

from loggers import get_logger, install_uvicorn_duplicate_exception_filter
from startup_banner import print_studio_access_banner, print_studio_stop_hint
from utils.host_policy import (
    is_wildcard_host,
    normalize_wildcard_bind_host,
    resolved_bind_address_count,
    published_url_host as _url_host,
    wildcard_ip_versions,
    wildcard_loopback_host,
)

logger = get_logger(__name__)

DISABLE_PUBLIC_CHECK_ENV = "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK"


def public_check_disabled() -> bool:
    """True when the operator opts out of the ifconfig.me and check-host.net startup lookups."""
    return os.environ.get(DISABLE_PUBLIC_CHECK_ENV, "").strip().lower() in {"1", "true", "yes"}


def _resolve_lan_ip(ip_version: int = 4) -> str:
    """This machine's own LAN-facing address, with no third-party network call: a UDP route lookup plus
    the active interfaces, and the lookup only fixes the local end of the socket."""
    from lan_access import detect_lan_addresses

    try:
        addresses = detect_lan_addresses(ip_version)
    except Exception:
        addresses = []
    if addresses:
        return addresses[0]
    return "0.0.0.0" if ip_version == 4 else "::"


def _resolve_external_ip() -> str:
    """Internet-facing address, not the LAN one; the last resort skips _resolve_lan_ip's LAN filtering."""
    import socket
    import urllib.request

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

    if not public_check_disabled():
        try:
            with urllib.request.urlopen("https://ifconfig.me", timeout = 3) as resp:
                ip = resp.read().decode().strip()
                if ip:
                    return ip
        except Exception:
            pass

    # A UDP connect sends nothing; it only fixes the local end.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def _install_uvicorn_startup_log_rewrite(bind_host: str) -> None:
    """Resolves the network-share host here so no caller can pass in the internet-facing address."""
    import logging
    import re

    display_host = _network_share_host_for_bind(bind_host)
    rewrite_host = is_wildcard_host(bind_host) and bool(display_host) and display_host != bind_host
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
        return f"{_url_scheme}://127.0.0.1:{port}"
    if _local_port_open("::1", port):
        return f"{_url_scheme}://[::1]:{port}"
    return None


def _localhost_ipv6_mismatch_url(bind_host: str, port: int) -> "str | None":
    """The IPv4 loopback URL when localhost will not reach 127.0.0.1. Local Unsloth binds to 127.0.0.1,
    and where localhost resolves to IPv6 only (::1), http://localhost:<port> fails or hits a different
    process even though the IPv4 URL works."""
    import socket

    if bind_host != "127.0.0.1" or not port or port <= 0:
        return None

    ipv4_url = f"{_url_scheme}://127.0.0.1:{port}"

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

    # Unsloth binds 127.0.0.1 only, so anything on ::1 is another process; only IPv6-only strands.
    if has_ipv6_loopback and not has_ipv4_loopback:
        return ipv4_url
    return None


def _stdout_color_ok() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _print_localhost_ipv6_mismatch_warning(local_url: str, port: int) -> None:
    use_color = _stdout_color_ok()
    warn_c = "\033[38;5;215;1m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    print(
        f"{warn_c}  Warning: localhost resolves to IPv6 (::1), but Unsloth "
        f"Unsloth is listening on 127.0.0.1 only. Open {local_url} instead of "
        f"{_url_scheme}://localhost:{port}.{reset}",
        flush = True,
    )


def _verify_global_reachability(
    display_host: str,
    port: int,
    wsl_nat: bool = False,
) -> None:
    """Failures are swallowed: a broken check-host.net probe is not a broken Unsloth."""
    global _public_reachable
    _public_reachable = None
    import ipaddress
    import json
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    if not display_host or is_wildcard_host(display_host):
        return

    use_color = _stdout_color_ok()
    dim = "\033[38;5;245m" if use_color else ""
    ok_c = "\033[38;5;120;1m" if use_color else ""
    err_c = "\033[38;5;203;1m" if use_color else ""
    warn_c = "\033[38;5;215;1m" if use_color else ""
    local_url_c = "\033[38;5;108;1m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    url = f"{_url_scheme}://{_url_host(display_host)}:{port}"

    try:
        addr = ipaddress.ip_address(display_host)
        if addr.is_loopback or addr.is_private or addr.is_link_local:
            _public_reachable = False
            if wsl_nat:
                return
            print(
                f"{dim}  Note: {display_host} is a private/LAN address -- "
                f"reachable on this network only, not from the public internet."
                f"{reset}",
                flush = True,
            )
            return
    except ValueError:
        pass

    if public_check_disabled():
        logger.debug("Skipping the check-host.net probe (%s).", DISABLE_PUBLIC_CHECK_ENV)
        return

    try:
        qs = urllib.parse.urlencode({"host": f"{_url_host(display_host)}:{port}", "max_nodes": 3})
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
                f"{dim}    then open {_url_scheme}://localhost:{port}/ in your browser.{reset}",
                flush = True,
            )
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
        pass
    except Exception:
        pass


def _display_host_for_bind(host: str) -> str:
    wildcard_versions = wildcard_ip_versions(host)
    if not wildcard_versions:
        return host
    ipv4_display_host = None
    if 4 in wildcard_versions:
        ipv4_display_host = _resolve_external_ip()
        if not is_wildcard_host(ipv4_display_host):
            return ipv4_display_host
    if 6 in wildcard_versions:
        from lan_access import detect_lan_addresses
        addresses = detect_lan_addresses(6)
        if addresses:
            return addresses[0]
    return ipv4_display_host or "::"


def _network_share_host_for_bind(host: str) -> str:
    """Not the internet-facing address: a wildcard bind's public WAN IP is unroutable from the LAN."""
    wildcard_versions = wildcard_ip_versions(host)
    if not wildcard_versions:
        return host
    for ip_version in wildcard_versions:
        lan_host = _resolve_lan_ip(ip_version)
        if not is_wildcard_host(lan_host):
            return lan_host
    return host


def _is_wsl_nat() -> bool:
    from lan_access import _wsl_networking_mode

    # "unknown" = WSL too old for wslinfo, which is NAT.
    if _wsl_networking_mode() not in ("nat", "unknown"):
        return False
    # Docker Desktop containers also read "unknown".
    from utils.paths.file_manager import _in_container

    return not _in_container()


def _print_wsl_windows_hint(port: int) -> None:
    """WSL2 NAT: Windows reaches a wildcard bind via localhost forwarding (#11187)."""
    dim = "\033[38;5;245m" if _stdout_color_ok() else ""
    reset = "\033[0m" if dim else ""
    print(
        f"{dim}  WSL2: open {_url_scheme}://localhost:{port} in a Windows browser. Other devices on your "
        f"network can't reach WSL's NAT address; set networkingMode=mirrored in "
        f"%UserProfile%\\.wslconfig for LAN access.{reset}",
        flush = True,
    )


def _loopback_bind_host_for(host: str) -> str:
    return wildcard_loopback_host(host) or "127.0.0.1"


def _direct_server_url(host: str, port: int) -> "Optional[str]":
    """The API panel's direct (non-tunnel) base, or None when this launch has no address worth publishing. The
    frontend prefers any non-null server_url over the origin the client reached, so publishing
    ``http://0.0.0.0:<port>`` (a wildcard bind with no LAN address: WSL behind NAT, loopback-only) would
    put an unroutable address in the API examples, the desktop agent command and a copied preview link."""
    # A TLS certificate usually names a DNS host, not the LAN IP; the frontend then falls back to its own origin.
    if not port or port <= 0 or _url_scheme == "https":
        return None
    share_host = _network_share_host_for_bind(host)
    if is_wildcard_host(share_host):
        return None
    return f"{_url_scheme}://{_url_host(share_host)}:{port}"


def _tool_policy_notice(host: str, secure: bool, enable_tools: "Optional[bool]") -> str:
    """One-line tool-policy summary for the plain-server startup banner, so a
    network-reachable launch is never silent about code execution."""
    if enable_tools is False:
        return "Server-side tools are DISABLED (--disable-tools)."
    if enable_tools is None:
        return (
            "Server-side tools follow each request's enable_tools; the Unsloth UI's "
            "tool toggles decide. Pass --enable-tools to force them on for every "
            "request."
        )
    state = "ENABLED (--enable-tools)"
    if secure:
        return (
            f"Server-side tools are {state}, reachable via the authenticated "
            "Cloudflare HTTPS tunnel. Anyone with the API key can run code on "
            "this machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    from utils.host_policy import is_external_host

    if is_external_host(host):
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
    lan_addresses: "tuple[str, ...]" = (),
) -> None:
    """A loopback launch with LAN addresses bound is network-reachable, so the banner must say so."""
    if secure:
        _emit_secure_startup_output(port, enable_tools)
        return
    wildcard_bind = is_wildcard_host(host)
    localhost_mismatch_url = _localhost_ipv6_mismatch_url(host, port)
    print_studio_access_banner(
        port = port,
        bind_host = host,
        display_host = display_host,
        network_host = _network_share_host_for_bind(host),
        include_stop_hint = False,
        lan_addresses = lan_addresses,
        scheme = _url_scheme,
    )
    if localhost_mismatch_url:
        _print_localhost_ipv6_mismatch_warning(localhost_mismatch_url, port)
    elif wildcard_bind:
        wsl_nat = _is_wsl_nat()
        if wsl_nat:
            _print_wsl_windows_hint(port)
        _verify_global_reachability(display_host, port, wsl_nat = wsl_nat)
        if _url_scheme == "http":  # --cloudflare is refused alongside --ssl-certfile
            _print_cloudflare_line(loopback_host = _loopback_bind_host_for(host))
    _emit_tool_policy_notice(lan_addresses[0] if lan_addresses else host, False, enable_tools)
    print_studio_stop_hint()


def _print_cloudflare_line(secure: bool = False, loopback_host: str = "127.0.0.1") -> None:
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
    """(pid, process_name) listening on *port*, or None. Uses psutil when available, else None so
    callers can still report the conflict without process details."""
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
    """Would a server bound to *recorded* block a bind to *host*? *recorded* may list several addresses;
    unknown or wildcard on either side collides, since refusing with a clear message beats silently
    starting a duplicate."""
    if not recorded or is_wildcard_host(host):
        return True
    listed = {a.strip() for a in recorded.split(",") if a.strip()}
    if not listed or any(is_wildcard_host(address) for address in listed):
        return True
    return bool(listed & _bind_addresses(host, port))


def _is_port_free(host: str, port: int) -> bool:
    """Wildcard binds also check loopback: an SSH tunnel can hold 127.0.0.1 while 0.0.0.0 succeeds."""
    import socket

    sockets = []
    bound = []
    try:
        addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        seen = set()
        for family, socktype, proto, _, sockaddr in addr_info:
            key = (family, socktype, proto, sockaddr)
            if key in seen:
                continue
            seen.add(key)
            bound.append((family, sockaddr))
            probe = socket.socket(family, socktype, proto)
            sockets.append(probe)
            # Windows SO_REUSEADDR lets a second socket bind a listening port, so the probe would lie.
            # POSIX keeps it so a TIME_WAIT listener does not strand the port.
            if sys.platform != "win32":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            probe.bind(sockaddr)
        if not sockets:
            return False
    except OSError:
        return False
    finally:
        for probe in sockets:
            probe.close()

    # A successful bind can still sit beside a live listener, so a connect that lands means taken. Short
    # timeout: Windows only refuses a free port after ~2 s of SYN retries.
    if is_wildcard_host(host):
        targets = [
            (socket.AF_INET, ("127.0.0.1", port)),
            (socket.AF_INET6, ("::1", port)),
        ]
    else:
        targets = bound
    for family, sockaddr in targets:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.25)
                result = s.connect_ex(sockaddr)
                if result == 0:
                    return False
                # Windows times out on a full-backlog listener exactly as on a free port.
                if result not in _CONNECT_REFUSED and _listener_collides(sockaddr[0], port):
                    return False
        except OSError:
            continue

    return True


_CONNECT_REFUSED = {errno.ECONNREFUSED, 10061}  # WSAECONNREFUSED


def _listener_collides(address: str, port: int) -> bool:
    """A same-family listener on *port* at *address* or its wildcard; a v6-only ``::`` shares the port
    with IPv4. Best effort: no psutil means no."""
    import socket

    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    try:
        import psutil
        listeners = [
            c.laddr[0]
            for c in psutil.net_connections(kind = "tcp")
            if c.status == psutil.CONN_LISTEN
            and c.family == family
            and c.laddr
            and c.laddr[1] == port
        ]
    except Exception:
        return False
    return any(_addresses_collide(listener, address, port) for listener in listeners)


def _find_free_port(
    host: str,
    start: int,
    max_attempts: int = 20,
    avoid_own_studio: bool = False,
) -> int:
    """Find a free port from `start`, trying up to max_attempts ports. ``avoid_own_studio`` aborts rather
    than skipping past one of our own servers in the fallback range, which would start a duplicate."""
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
# Not a .pid: PID_FILE_GLOB readers expect a bound server with a port in the name.
STARTUP_MARKER_GLOB = "studio-starting-*.marker"
_OWN_STARTUP_MARKERS: "list[Path]" = []
_STARTUP_MARKER_HOOK_REGISTERED = False


def _pid_file_for_port(port: int) -> Path:
    # PID in the name: 127.0.0.1 and ::1 can share a port.
    return _studio_root() / f"studio-{port}-{os.getpid()}.pid"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if sys.platform == "win32":
        # os.kill(pid, 0) raises for every pid on Windows.
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
            # Unconfirmed means keep, matching the CLI's _pid_alive.
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
    try:
        lines = path.read_text(encoding = "utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or not lines[0].strip().isdigit():
        return None
    try:
        # isdigit() accepts superscripts that int() rejects.
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
    """Untimed records are trusted: command-line guessing wrongly rejected real legacy servers."""
    known = [c for c in created_times if c is not None]
    if not known:
        return True
    actual = _process_create_time(pid)
    if actual is None:
        return True
    return any(abs(actual - c) < 1.0 for c in known)


def _own_studio_on_port(port: int, host: str) -> "int | None":
    """PID of one of our own servers already bound to *port* for *host*. Reads our own records rather than
    enumerating listeners: psutil is optional, and without it a listener scan finds nothing and we silently
    start a duplicate."""
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
    """A pre-upgrade server recorded only its PID, so match it to the listener. Falling back past one leaves it
    running while `_write_pid_file` overwrites the only record of it. When the listener is unknowable,
    assume it is ours."""
    record = _read_pid_record(_PID_FILE)
    if record is None:
        return None
    pid, created, _address = record
    if not _pid_alive(pid):
        return None
    # Only count a record still matching the live process; a stale one may share a reused PID.
    for other in _per_port_records():
        if other and other[0] == pid and _pid_is_studio_backend(pid, [other[1]]):
            return None
    blocker = _get_pid_on_port(port)
    if blocker is not None and blocker[0] != pid:
        return None
    if not _pid_is_studio_backend(pid, [created]):
        return None
    return pid


def write_startup_marker() -> None:
    """Written before uvicorn starts, under the cache-clear lock, so a sibling cannot miss it."""
    me = os.getpid()
    created = _process_create_time(me)
    body = f"{me}\n{created if created is not None else ''}\n"
    try:
        from utils.cache_cleanup import compiled_cache_lock, LOCK_BUSY
    except Exception:
        import contextlib
        compiled_cache_lock = contextlib.nullcontext
        LOCK_BUSY = None
    # Wait out a holder mid-rmtree; publish anyway on timeout, or this backend stays invisible.
    with compiled_cache_lock(timeout = 30.0) as lock_state:
        published_unlocked = lock_state == LOCK_BUSY
        if published_unlocked:
            logger.warning(
                "Publishing the startup marker without the cache lock: another backend "
                "has held it for 30s"
            )
        directory = _studio_root()
        path = directory / f"studio-starting-{me}.marker"
        try:
            directory.mkdir(parents = True, exist_ok = True)
            # Same layout as a per-port record; unknown start time is a blank line.
            path.write_text(body, encoding = "utf-8")
            if path not in _OWN_STARTUP_MARKERS:
                _OWN_STARTUP_MARKERS.append(path)
        except OSError:
            pass
    if not _OWN_STARTUP_MARKERS:
        return
    import atexit

    # Register before the wait so an interrupt still removes the marker.
    global _STARTUP_MARKER_HOOK_REGISTERED
    if not _STARTUP_MARKER_HOOK_REGISTERED:
        atexit.register(_remove_startup_marker)
        _STARTUP_MARKER_HOOK_REGISTERED = True
    if published_unlocked:
        _wait_out_a_running_cache_clear(compiled_cache_lock, LOCK_BUSY)


# Not how long a clear takes: how long before starting beats refusing to start.
_CACHE_CLEAR_WAIT_SECONDS = 300.0


def _wait_out_a_running_cache_clear(compiled_cache_lock, lock_busy) -> None:
    """Taking and dropping the lock is the wait; the OS frees it when the clearing backend exits."""
    with compiled_cache_lock(timeout = _CACHE_CLEAR_WAIT_SECONDS) as state:
        if state == lock_busy:
            logger.warning(
                "Starting while another backend is still clearing the compiled cache: "
                "it has held the lock for %ss. Compiled modules may be removed underneath "
                "this one and recompiled.",
                int(30.0 + _CACHE_CLEAR_WAIT_SECONDS),
            )


# Outlast a typical Windows indexer/scanner holding the file.
_MARKER_REMOVAL_ATTEMPTS = 5
_MARKER_REMOVAL_BACKOFF_SECONDS = 0.2


def _remove_startup_marker() -> None:
    """A path leaves the list only once unlinked; a leftover marker would pin the compiled cache."""
    import time

    remaining = []
    while _OWN_STARTUP_MARKERS:
        path = _OWN_STARTUP_MARKERS.pop()
        for attempt in range(_MARKER_REMOVAL_ATTEMPTS):
            try:
                path.unlink(missing_ok = True)
                break
            except OSError:
                if attempt + 1 == _MARKER_REMOVAL_ATTEMPTS:
                    remaining.append(path)
                    break
                time.sleep(_MARKER_REMOVAL_BACKOFF_SECONDS)
    _OWN_STARTUP_MARKERS.extend(remaining)


def _record_written_at(path: Path) -> float:
    """When *path* was last written, or 0.0 when the filesystem will not say. 0.0 makes an unreadable time the
    oldest possible, so it never wins the "which statement about this PID came last" comparison below."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _timed_records() -> "list[tuple[tuple[int, float | None, str | None], float]]":
    """Every startup marker and per-port record, each with its write time."""
    paths: "list[Path]" = []
    for pattern in (STARTUP_MARKER_GLOB, PID_FILE_GLOB):
        try:
            paths.extend(_studio_root().glob(pattern))
        except OSError:
            continue
    found = []
    for path in paths:
        record = _read_pid_record(path)
        if record is not None:
            found.append((record, _record_written_at(path)))
    return found


def _legacy_record() -> "tuple[tuple[int, float | None, str | None], float] | None":
    try:
        if not _PID_FILE.is_file():
            return None
    except OSError:
        return None
    record = _read_pid_record(_PID_FILE)
    return None if record is None else (record, _record_written_at(_PID_FILE))


def _sibling_is_running(pid: int) -> bool:
    """Alive, and not an exited process nobody waited on. A zombie answers kill(0) and keeps its start time,
    and under a non-reaping parent (PID 1 in some containers) it can stay that way indefinitely, so a
    record pointing at one would pin the compiled cache on every later startup."""
    if not _pid_alive(pid):
        return False
    try:
        from utils.process_lifetime import _pid_is_zombie
    except Exception:  # noqa: BLE001
        return True
    return not _pid_is_zombie(pid)


def _live_sibling(records: "list", me: int, timed: "list") -> "int | None":
    for entry in records:
        if entry is None:
            continue
        record, written_at = entry
        pid, created, _address = record
        if pid == me or not _sibling_is_running(pid):
            continue
        # studio.pid has no start time, so corroborate against timestamped records for this PID; only
        # records no older than it count (a reused PID must not clear a live legacy backend's cache).
        corroborating = [
            other[1]
            for other, other_written in timed
            if other[0] == pid and other_written >= written_at
        ]
        if _pid_is_studio_backend(pid, [created] + corroborating):
            return pid
    return None


def live_sibling_backend() -> "int | None":
    """Reads the startup marker, per-port record and studio.pid, since any one may be the only record."""
    me = os.getpid()
    timed = _timed_records()
    return _live_sibling(timed + [_legacy_record()], me, timed)


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
    """The requested port, or the next free one. With ``avoid_own_studio`` this aborts rather than falling back
    past one of our own servers, on *port* itself or anywhere in the fallback range, since skipping one is
    what strands it."""
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


# Direct launches bypass the CLI's env re-export; export custom roots so unsloth-zoo's
# import-time LLAMA_CPP_DEFAULT_DIR sees them. Skip legacy-default.
try:
    _LEGACY_STUDIO_ROOT = (Path.home() / ".unsloth" / "studio").resolve()
except (OSError, ValueError):
    _LEGACY_STUDIO_ROOT = Path.home() / ".unsloth" / "studio"
try:
    _STUDIO_ROOT_RESOLVED = _studio_root().resolve()
except (OSError, ValueError):
    _STUDIO_ROOT_RESOLVED = _studio_root()
from utils.paths.storage_roots import unsloth_home as _unsloth_home

_MASTER_ROOT = _unsloth_home()
# A master root at the legacy path still owns runtimes beside it.
if _STUDIO_ROOT_RESOLVED != _LEGACY_STUDIO_ROOT or _MASTER_ROOT is not None:
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(_STUDIO_ROOT_RESOLVED)
    # Runtimes sit at the master root, beside studio/.
    _MANAGED_ROOT = _MASTER_ROOT or _STUDIO_ROOT_RESOLVED
    _MANAGED_LLAMA_CPP_PATH = _MANAGED_ROOT / "llama.cpp"
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_MANAGED_LLAMA_CPP_PATH)
    from utils.llama_cpp_path_settings import mark_managed_llama_cpp_path

    mark_managed_llama_cpp_path(_MANAGED_LLAMA_CPP_PATH)

# Declare unsloth present so unsloth_zoo's install guard passes on a clean install.
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
        created = _process_create_time(os.getpid())
        address = ",".join(sorted(_bind_addresses(host, port))) if host else ""
        body = f"{os.getpid()}\n{'' if created is None else repr(created)}\n{address}"
        # Write-then-rename: concurrent `stop` readers would delete a truncated record.
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(body, encoding = "utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok = True)
    except OSError:
        pass
    else:
        _OWN_PID_FILE = path
    # An older CLI's `stop` reads only this file and expects a bare PID.
    try:
        # Never take it from a running server: a pre-upgrade server is recorded only here.
        prior = _read_pid_record(_PID_FILE) if _PID_FILE.is_file() else None
        if prior is None or prior[0] == os.getpid() or not _pid_alive(prior[0]):
            _PID_FILE.write_text(str(os.getpid()), encoding = "utf-8")
    except OSError:
        pass


def _legacy_heir() -> "int | None":
    """Another live server's PID, to hand the legacy studio.pid over to. Only one server owns studio.pid
    at a time, so its exit would otherwise drop the single record an older CLI can read."""
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
    """Keeps the startup marker: a backend still finishing a request is importing from the cache."""
    # Nothing here may raise: _graceful_shutdown calls this last.
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
            # Hand off to a live sibling: older CLIs read only this file.
            heir = _legacy_heir()
            if heir is None:
                _PID_FILE.unlink(missing_ok = True)
            else:
                _PID_FILE.write_text(str(heir), encoding = "utf-8")
    except OSError:
        pass


# Windows terminates the process ~5s after a close event, so leave a margin.
_CONSOLE_SHUTDOWN_BUDGET = 4.5


# Ctrl+C / Ctrl+Break (0, 1) are excluded: Python delivers them as signals.
_CONSOLE_SHUTDOWN_EVENTS = (2, 5, 6)


def _console_event_is_shutdown(event: int) -> bool:
    return event in _CONSOLE_SHUTDOWN_EVENTS


def _run_console_shutdown(shutdown) -> None:
    try:
        shutdown()
    except Exception as error:
        logger.warning("Console-close cleanup failed: %s", error)


def _install_windows_console_handler(shutdown) -> bool:
    """Closing the console skips signal handlers and atexit, and Windows kills the process ~5s later."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        import threading

        def _on_console_event(event: int) -> bool:
            if _console_event_is_shutdown(event):
                worker = threading.Thread(
                    target = _run_console_shutdown, args = (shutdown,), daemon = True
                )
                worker.start()
                worker.join(timeout = _CONSOLE_SHUTDOWN_BUDGET)
                return True
            return False

        callback = HANDLER(_on_console_event)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.SetConsoleCtrlHandler.argtypes = [HANDLER, wintypes.BOOL]
        kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL
        if not kernel32.SetConsoleCtrlHandler(callback, True):
            logger.warning(
                "Could not install the console-close handler (WinError %s); closing the "
                "window will skip subprocess cleanup.",
                ctypes.get_last_error(),
            )
            return False
        # Hold a reference: a collected callback leaves Windows calling into freed memory.
        globals()["_WINDOWS_CONSOLE_HANDLER"] = callback
        logger.info("Console-close handler installed")
        return True
    except Exception as error:
        logger.warning("Could not install the console-close handler: %s", error)
        return False


def _note_quit_signal():
    from utils.process_lifetime import mark_quit_signal_received
    mark_quit_signal_received()


def _graceful_shutdown(server = None):
    """Shut down all subprocess backends and the uvicorn server. Called from signal handlers to clean up
    children before exit; critical on Windows where atexit handlers are unreliable after Ctrl+C."""
    logger.info("Graceful shutdown initiated -- cleaning up subprocesses...")

    # 0a. Latch "quitting" first: in-flight loads can reach spawners whose step already ran.
    try:
        from utils.process_lifetime import mark_process_shutting_down
        mark_process_shutting_down()
    except Exception as e:
        logger.warning("Could not latch the process shutdown flag: %s", e)

    # 0. Drop the LAN listener first: it shares the loop uvicorn is about to stop.
    try:
        from lan_access import close_lan_listener_lifecycle
        close_lan_listener_lifecycle()
    except Exception as e:
        logger.warning("Error stopping the LAN listener: %s", e)

    try:
        from core.training.training import _training_backend
        if _training_backend is not None:
            _training_backend.stop_for_shutdown()
    except Exception as e:
        logger.warning("Error stopping the training run for shutdown: %s", e)

    try:
        _diffusion = sys.modules.get("core.training.diffusion_training_service")
        if _diffusion is not None and _diffusion._service is not None:
            from core.training.training import _SHUTDOWN_STOP_TIMEOUT_S
            if not _diffusion._service.stop_for_shutdown(_SHUTDOWN_STOP_TIMEOUT_S):
                logger.warning("Shutdown: diffusion training did not finish saving in time")
    except Exception as e:
        logger.warning("Error stopping the diffusion training run for shutdown: %s", e)

    if server is not None:
        server.should_exit = True

    try:
        from core.inference.orchestrator import _inference_backend
        if _inference_backend is not None:
            _inference_backend._shutdown_subprocess(timeout = 5.0)
    except Exception as e:
        logger.warning("Error shutting down inference subprocess: %s", e)

    try:
        from core.export.orchestrator import _export_backend
        if _export_backend is not None:
            _export_backend._shutdown_subprocess(timeout = 5.0)
    except Exception as e:
        logger.warning("Error shutting down export subprocess: %s", e)

    try:
        from core.training.training import _training_backend
        if _training_backend is not None:
            _training_backend.force_terminate()
    except Exception as e:
        logger.warning("Error shutting down training subprocess: %s", e)

    try:
        from core.inference.model_slots import unload_extra_models
        from routes.inference import _llama_cpp_backend, cancel_pending_loads

        # Before the kill: a load in the lifecycle gate could spawn llama-server after this step.
        try:
            cancelled = cancel_pending_loads()
            if cancelled:
                logger.info("Cancelled %d in-flight model load(s) for shutdown", cancelled)
        except Exception as e:
            logger.warning("Could not cancel in-flight loads: %s", e)
        if _llama_cpp_backend is not None:
            # teardown = True: only an app-level stop may end an in-flight health wait.
            _llama_cpp_backend._kill_process(teardown = True)
        unload_extra_models()
    except Exception as e:
        logger.warning("Error shutting down llama-server: %s", e)

    try:
        from core.inference.npu_backend import peek_npu_backend
        _npu = peek_npu_backend()
        if _npu is not None:
            # Unload first: lemond then stops FastFlowLM itself, before the tree kill.
            _npu.shutdown()
    except Exception as e:
        logger.warning("Error shutting down the NPU runtime: %s", e)

    try:
        from cloudflare_tunnel import close_studio_tunnel_lifecycle
        close_studio_tunnel_lifecycle()
    except Exception as e:
        logger.warning("Error stopping Cloudflare tunnel: %s", e)

    try:
        from utils.process_lifetime import clear_breadcrumb, terminate_all
        terminate_all()
        clear_breadcrumb()
    except Exception as e:
        logger.warning("Error in process-lifetime sweep: %s", e)

    # Last: a retried `stop` or new launch must still find the server during cleanup.
    _remove_pid_file()
    logger.info("All subprocesses cleaned up")


_SERVER_SHUTDOWN_JOIN_TIMEOUT = 5.0


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _wait_for_server_shutdown(timeout: Optional[float] = _SERVER_SHUTDOWN_JOIN_TIMEOUT) -> None:
    """Join the uvicorn thread so the prompt returns only after its shutdown logs flush. Skip the
    self-join when called from the server thread."""
    import threading

    thread = _server_thread
    if thread is None or thread is threading.current_thread():
        _flush_standard_streams()
        return
    thread.join(timeout = timeout)
    if thread.is_alive():
        logger.warning("Timed out waiting for uvicorn server thread to stop")
    _flush_standard_streams()


_server = None
_server_thread = None
# "https" for an --ssl-certfile launch, used by every URL the launch prints for itself
_url_scheme = "http"

_shutdown_event = None

# Set by run_server for wildcard binds; None when there is no tunnel.
_cloudflare_url = None


def _publish_cloudflare_url(app_state, cloudflare_url: "Optional[str]") -> None:
    global _cloudflare_url
    _cloudflare_url = cloudflare_url
    app_state.cloudflare_url = cloudflare_url


# True/False when the probe decided, None otherwise.
_public_reachable = None

_cloudflare_requested = False
# None = off by default, False = explicit --no-cloudflare.
_cloudflare_flag = None


_DEFAULT_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _iter_frontend_fallback_candidates() -> "list[Path]":
    """Yield `studio/frontend/dist` paths to try when the default is missing, covering PATH-shadowed binaries
    whose __file__ resolves into a site-packages tree with no vite build (e.g. plain `pip install
    unsloth`)."""
    import ast
    import re

    out: list[Path] = []
    home_str = (
        os.environ.get("UNSLOTH_STUDIO_HOME")
        or os.environ.get("STUDIO_HOME")
        or str(Path.home() / ".unsloth" / "studio")
    )
    venv_dir = Path(home_str).expanduser() / "unsloth_studio"
    for pattern in (
        "lib/python*/site-packages/studio/frontend/dist",
        "Lib/site-packages/studio/frontend/dist",
    ):
        out.extend(venv_dir.glob(pattern))
    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for finder in sp.glob("__editable___*_finder.py"):
                try:
                    src = finder.read_text(encoding = "utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                # [^}]* rejects nested dicts, which the setuptools editable template never emits.
                m = re.search(r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S)
                if not m:
                    continue
                try:
                    mapping = ast.literal_eval(m.group(1))
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(mapping, dict):
                    continue
                studio_pkg = mapping.get("studio")
                if studio_pkg:
                    out.append(Path(studio_pkg) / "frontend" / "dist")
    return out


def _resolve_frontend_path(frontend_path: Path) -> tuple[Optional[Path], list[Path]]:
    """Pick a frontend dir that contains `index.html`. Returns (chosen, attempted); `chosen` is None if
    nothing servable was found."""
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
    """Fatal only for the web UI; the desktop's --api-only launch would die before TAURI_PORT."""
    return not tunnel_only


class _TeeStream:
    """Mirror writes to the original stream and a session log file. Console behavior is unchanged, and
    the file copy is best-effort: a full disk or a closed handle must never break the console.

    The file copy collapses carriage-return progress frames: a tqdm bar redraws "\\r<frame>" hundreds
    of times and a file keeps every one. Only frames are ever withheld: a partial line with no "\\r" (a
    prompt, a traceback torn by a hang) is written on arrival, and a held frame is closed off on its
    own line before the next record.

    Frames are picked exactly as the desktop reader picks them, so the session log and tauri.log stay
    interchangeable: strip the terminator (trim_line_endings), then take the last non-blank
    "\\r"-separated frame (collapse_progress_frames), both in src-tauri/src/process.rs. Reading the
    "\\r" of a CRLF as a redraw instead keeps the empty text after it and drops the line, which on
    Windows is every relayed child line there is."""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh
        self._pending_frame = ""

    @staticmethod
    def _last_frame(line):
        # A trailing "\r" terminates the line (CRLF, tqdm sign-off), not an empty redraw.
        line = line.rstrip("\r")
        if "\r" not in line:
            return line
        for frame in reversed(line.split("\r")):
            if frame.strip():
                return frame
        # Return a frame: a bare CR would land as CR CR LF.
        return line.rsplit("\r", 1)[-1]

    def _write_file(self, data):
        if not data:
            return
        if "\r" not in data and not self._pending_frame:
            self._log_fh.write(data)
            return

        if self._pending_frame and data[:1] not in ("\r", "\n"):
            # Close the held frame on its own line; concatenating breaks a JSON reader.
            self._log_fh.write(self._pending_frame + "\n")
            self._pending_frame = ""

        buf = self._pending_frame + data
        self._pending_frame = ""
        head, newline, tail = buf.rpartition("\n")
        if newline:
            complete = head + newline
            self._log_fh.write("\n".join(self._last_frame(line) for line in complete.split("\n")))
        if tail:
            # Hold an unterminated tail only if it is a redraw, so a hang cannot swallow real output.
            if "\r" in tail:
                self._pending_frame = self._last_frame(tail)
            else:
                self._log_fh.write(tail)

    def write(self, data):
        try:
            self._write_file(data)
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
        # We do not own the console stream; forward close() best-effort (ipython/ipykernel#867).
        try:
            if self._pending_frame:
                self._log_fh.write(self._pending_frame + "\n")
                self._pending_frame = ""
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
    """True only for ipython/ipykernel#867's missing-``watch_fd_thread`` error. ``AttributeError.name`` exists
    from Python 3.10; the message carries the attribute name on every version (possibly with a "Did you
    mean" tail), so check both and let every other AttributeError through."""
    if getattr(exc, "name", None) == _WATCH_FD_THREAD_ATTR:
        return True
    return _WATCH_FD_THREAD_ATTR in str(exc)


def _harden_console_close(stream):
    """Swallows only the AttributeError ipykernel raises from close() on a displaced stream."""
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
                raise
            # ipython/ipykernel#867: watchfd=False OutStream.close() joins a thread that was never created.
            return None

    try:
        stream.close = _safe_close
    except (AttributeError, TypeError):
        pass


def _setup_server_disk_logging():
    """Tees output to disk so hard crashes leave a stack; UNSLOTH_STUDIO_NO_FILE_LOG=1 opts out."""
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
        # Line-buffered so the tail survives a hard kill.
        log_fh = open(log_path, "w", encoding = "utf-8", errors = "replace", buffering = 1)
    except Exception:
        return None

    import faulthandler

    try:
        faulthandler.enable(file = log_fh, all_threads = True)
    except Exception:
        pass
    # Children inherit, so native-crash stacks land on the captured stderr.
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")

    # Guard the console streams' close() first (ipython/ipykernel#867).
    _harden_console_close(sys.stdout)
    _harden_console_close(sys.stderr)

    sys.stdout = _TeeStream(sys.stdout, log_fh)
    sys.stderr = _TeeStream(sys.stderr, log_fh)

    # Keep the newest 20; `protect` because same-second starts may not sort newest.
    try:
        from utils.log_retention import prune_log_dir
        prune_log_dir(log_dir, "server-*.log", protect = log_path)
    except Exception:
        pass
    return log_path


def _cloudflare_tunnel_should_start(
    *, cloudflare: bool, host: str, secure: bool, api_only: bool, is_colab: bool
) -> bool:
    """Whether to start the Cloudflare tunnel. --secure exposes only the tunnel (loopback bind), so it
    tunnels even api-only; otherwise tunnel wildcard binds, never api-only (Tauri) or Colab."""
    if is_colab or not cloudflare:
        return False
    if secure:
        return True
    return is_wildcard_host(host) and not api_only


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


def _bound_request_host(server) -> str:
    """Return a literal address from Uvicorn's active listener sockets."""
    for listener in getattr(server, "servers", ()):
        for sock in getattr(listener, "sockets", ()) or ():
            address = sock.getsockname()
            if not isinstance(address, tuple) or not address or not isinstance(address[0], str):
                continue
            host = address[0]
            loopback_host = wildcard_loopback_host(host)
            if loopback_host is not None:
                return loopback_host
            if len(address) >= 4 and address[3] and ":" in host and "%" not in host:
                return f"{host}%{address[3]}"
            return host
    raise RuntimeError("Uvicorn did not expose its bound address")


_CLOUDFLARE_INTENT_ENV = "_UNSLOTH_CLOUDFLARE_INTENT"


def _consume_cloudflare_intent(cloudflare: "Optional[bool]", secure: bool) -> str:
    """Resolve user intent without confusing a compatibility flag with opt-out. An explicit choice on THIS
    invocation wins: letting the inherited marker override it would let a stale export, Docker ENV or
    systemd Environment= re-enable a tunnel the user opted out of."""
    inherited = os.environ.pop(_CLOUDFLARE_INTENT_ENV, None)
    if secure or cloudflare is True:
        return "enabled"
    if cloudflare is False:
        return "unset" if inherited == "unset" else "disabled"
    if inherited in {"unset", "enabled", "disabled"}:
        return inherited
    return "unset"


def _stream_isatty(stream) -> bool:
    """isatty() that treats broken streams as non-interactive: it can raise under service wrappers (closed
    stdin gives ValueError; sys.stdin None in a Windows GUI gives AttributeError), and such a stream cannot
    host a prompt."""
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


# Wait for the FIRST keystroke only; detached ptys (`docker run -dt`, `tmux new -d`) never type,
# and 30s stays inside a default Docker HEALTHCHECK start period. Mirrored in the CLI.
_UNATTENDED_PROMPT_SECONDS = 30.0


def _prompt_owns_the_terminal() -> bool:
    """A backgrounded job sees a TTY, but tcsetattr there sends SIGTTOU, so it must not prompt."""
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except (AttributeError, OSError, ValueError):
        return True


def _exposure_phrase(*, tunnel_will_start: bool, host: str) -> str:
    """A tunnel is the public internet, a wildcard bind every interface; a concrete host is neither."""
    if tunnel_will_start:
        return "on the public internet"
    if is_wildcard_host(host):
        return "on every network interface"
    return f"at {host}, which other machines on the network can reach"


def _terminal_password_gate(
    *,
    tunnel_will_start: bool,
    host: str,
    secure: bool,
    api_only: bool,
    frontend_served: bool,
    is_colab: bool = False,
) -> Tuple[bool, bool]:
    """Must run before the socket binds: served HTML injects the default password on wildcard binds."""
    from auth.bootstrap_timeout import _is_exposed_bind

    # Raw non-loopback binds inject the bootstrap credential to the network too. Web UI only, never
    # api-only or Colab. secure = False: only socket reachability matters here.
    bind_is_exposed = (
        _is_exposed_bind(host, False) and frontend_served and not api_only and not is_colab
    )
    if not tunnel_will_start and not bind_is_exposed:
        return True, False

    # The parent already waited; consume (not peek) so it does not double the stall or leak.
    if os.environ.pop("UNSLOTH_STUDIO_UNATTENDED_PROMPT_DONE", "") and not tunnel_will_start:
        return True, False

    # Raw-bind-only launches decide promptability BEFORE opening auth storage.
    if not tunnel_will_start and not (
        _stream_isatty(sys.stdin) and _stream_isatty(sys.stderr) and _prompt_owns_the_terminal()
    ):
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
        bind_is_exposed = bind_is_exposed,
        requires_change = requires_change,
        stdin_isatty = _stream_isatty(sys.stdin),
        stderr_isatty = _stream_isatty(sys.stderr),
    ):
        # Headless raw bind: keep the old behaviour; refuse-and-strip would break -H 0.0.0.0 containers.
        if not tunnel_will_start:
            return True, False
        # api-only and TIMEOUT=0 never arm the deadline, leaving the default credential public.
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
        # Absolute command here (stderr on the host); the 401 body carries only the PATH form.
        from routes.auth import _reset_password_command

        print(
            "  WARNING: the default admin password is still active while "
            "Unsloth is about to be published on a public Cloudflare URL, and "
            "no terminal is attached to change it here. The public page will "
            "NOT auto-fill the bootstrap credential. Set a new password by "
            "running `unsloth studio` locally with a terminal attached, or "
            f"`{_reset_password_command()}`. Unsloth shuts down after the "
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
        # Same as routes/auth.py change_password: rotate JWT secret, revoke refresh tokens in one txn.
        _auth_storage.update_password(_admin, new_password, revoke_refresh_tokens = True)

    changed = prompt_for_password_change(
        min_length = _auth_storage.MIN_PASSWORD_LENGTH,
        is_current_password = _is_current_password,
        apply_change = _apply_change,
        out = sys.stderr,
        exposure = _exposure_phrase(tunnel_will_start = tunnel_will_start, host = host),
        # A detached pty passes isatty yet nobody types, so raw binds get a first-key timeout.
        # Ctrl+C / EOF fails closed. The tunnel waits forever.
        first_key_timeout = None if tunnel_will_start else _UNATTENDED_PROMPT_SECONDS,
    )
    if changed is True:
        return True, True
    if changed is False or tunnel_will_start:
        return False, False
    # TIMEOUT=0 never arms the deadline, so say what will actually happen.
    deadline_arms = should_arm_bootstrap_timeout(
        host = host,
        secure = secure,
        api_only = api_only,
        frontend_served = frontend_served,
        is_colab = is_colab,
        requires_change = True,
        timeout_seconds = bootstrap_timeout_seconds(),
    )
    if deadline_arms:
        tail = (
            "Unsloth shuts down after the bootstrap deadline "
            "(UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT, default 1h) unless the password "
            "is changed."
        )
    else:
        tail = (
            "The bootstrap shutdown deadline is DISABLED for this launch "
            "(UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0), so nothing will stop it "
            "serving that credential."
        )
    from routes.auth import _reset_password_command

    print(
        "  WARNING: continuing with the auto-generated admin password on a bind "
        f"that is reachable from the network. {tail} Change it by logging in, or "
        f"with `{_reset_password_command()}`.",
        file = sys.stderr,
        flush = True,
    )
    return True, False


def _apply_supplied_password(password_value: "Optional[str]") -> None:
    """Sets only the first password; a storage failure aborts rather than exposing the default."""
    from auth import hashing as _auth_hashing
    from auth import storage as _auth_storage
    from auth.terminal_prompt import SUPPLIED_PASSWORD_ENV, resolve_supplied_password

    supplied = resolve_supplied_password(password_value)
    # Strip so children cannot read the plaintext via /proc/PID/environ, even when --password won.
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
    """None leaves policy unset, so each request decides; a tools-on default would reach launches
    where omitted means off."""
    if enable_tools is None:
        return
    from state.tool_policy import set_tool_policy

    set_tool_policy(enable_tools)


# Mirror CLI _PARALLEL_*; define above run_server() so embedders do not serialise chats.
_PARALLEL_MIN = 1
_PARALLEL_MAX = 64
_PARALLEL_DEFAULT_PLAIN = 4


def _drops_its_marker_on_failure(start):
    """Removes the startup marker on failure: an embedded host keeps running and no exit hook fires."""
    import functools

    @functools.wraps(start)
    def started(*args, **kwargs):
        try:
            return start(*args, **kwargs)
        except BaseException:
            # Only with no server thread left: a Ctrl+C during readiness leaves uvicorn finishing lifespan,
            # and dropping the marker then would let a sibling clear the cache under a live backend.
            thread = _server_thread
            if thread is None or not thread.is_alive():
                _remove_startup_marker()
            raise

    return started


def _repair_pinned_diffusers(silent: bool) -> None:
    """Repair before importing the app; exit if packages may still be half replaced at timeout."""
    echo = (lambda _line: None) if silent else (lambda line: print(line, flush = True))
    try:
        from utils.diffusers_repair import (
            InstallInterrupted,
            PeerInstallInProgress,
            repair_diffusers_before_imports,
        )
    except Exception as exc:  # noqa: BLE001 -- a self-heal must never block startup
        echo(f"  - diffusers self-heal skipped: {exc}")
        return
    try:
        repair_diffusers_before_imports(echo)
    except (PeerInstallInProgress, InstallInterrupted) as exc:
        print(f"Error: {exc}", file = sys.stderr, flush = True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 -- a self-heal must never block startup
        echo(f"  - diffusers self-heal skipped: {exc}")


@_drops_its_marker_on_failure
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
    ssl_certfile: "Optional[str]" = None,
    ssl_keyfile: "Optional[str]" = None,
):
    """Start the FastAPI server. Public entry point: colab.py and other embedders import it.

    ``port`` auto-increments when the given one is in use. ``cloudflare`` is tri-state: None and
    False both mean off, True opts the public HTTPS tunnel in for a wildcard bind, and ``secure``
    implies True and rejects an explicit False. ``api_only`` still serves the packaged frontend
    when a Tauri-owned backend has a live tunnel. ``emit_tauri_port`` prints the machine-readable
    TAURI_PORT line the desktop app parses; ``run --api-only`` turns it off so it does not pollute
    the URL and API-key banner. ``enable_tools`` of None leaves the default on, honouring a
    request's own ``enable_tools: false``. ``ssl_certfile`` + ``ssl_keyfile`` serve HTTPS from that pair.

    Signal handlers are NOT registered here, so embedders keep their own interrupt semantics;
    standalone callers register them afterwards.
    """
    global _server, _server_thread, _shutdown_event, _url_scheme

    if not isinstance(host, str) or not host.strip():
        raise SystemExit("--host cannot be empty; use 0.0.0.0 to bind every IPv4 interface.")

    boot_started = time.perf_counter()

    # Reject before touching process globals (the tee replaces sys.stdout/stderr).
    if secure and cloudflare is False:
        raise SystemExit(
            "--secure requires the Cloudflare tunnel; do not combine it with --no-cloudflare."
        )
    from utils.studio_tls import resolve_studio_tls

    tls = resolve_studio_tls(ssl_certfile, ssl_keyfile)
    if tls is not None and (secure or cloudflare):
        # The tunnel dials the origin over plain HTTP.
        raise SystemExit(
            "--ssl-certfile/--ssl-keyfile serve HTTPS directly; do not combine them with --secure or --cloudflare."
        )
    _url_scheme = "https" if tls is not None else "http"
    if not secure:
        try:
            host = normalize_wildcard_bind_host(host)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if port == 0 and resolved_bind_address_count(host) > 1:
            raise SystemExit(
                "--port 0 cannot be used when --host resolves to multiple bind addresses; "
                "choose an explicit port."
            )

    # Before the tee, so it reaches the console stream.
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass

    # Session log + crash stacks BEFORE anything else, so import-time native crashes leave evidence.
    _session_log = _setup_server_disk_logging()
    if _session_log is not None and not silent:
        print(f"Session log: {_session_log}")

    # After the tee: PrintLoggerFactory captures sys.stdout at configure() time. Imported here
    # because tests stand a bare ModuleType in for `loggers`.
    from loggers.config import LogConfig

    LogConfig.setup_logging(
        service_name = "unsloth-studio-backend",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    logger.info(
        "run_server startup begin api_only=%s host=%s port=%s log_level=%s studio_home=%s "
        "python=%s pid=%s",
        api_only,
        host,
        port,
        os.getenv("LOG_LEVEL", "INFO"),
        _studio_root(),
        sys.version.split()[0],
        os.getpid(),
    )
    cloudflare_intent = _consume_cloudflare_intent(cloudflare, secure)

    # Must run before any child can spawn.
    from utils.process_lifetime import initialize_parent_lifetime, reap_recorded_children

    initialize_parent_lifetime()
    # macOS has no PR_SET_PDEATHSIG or job objects; sweep leftovers from a crash before spawning.
    try:
        reaped = reap_recorded_children()
        if reaped:
            logger.warning("Reaped %d orphan(s) from a previous Unsloth: %s", len(reaped), reaped)
    except Exception as e:
        logger.warning("Could not sweep orphans from a previous run: %s", e)

    # --secure forces loopback so the raw port is never public. Else keep the tri-state for the banner.
    if secure:
        cloudflare = True
        host = "127.0.0.1"

    _apply_cli_tool_policy(enable_tools)

    # Set env vars BEFORE importing main so CORS middleware picks them up. secure api-only keeps
    # any-origin CORS (remote behind Cloudflare); plain api-only stays locked to the Tauri app.
    if api_only:
        os.environ["UNSLOTH_API_ONLY"] = "1"
    if secure:
        os.environ["UNSLOTH_SECURE"] = "1"

    import asyncio

    # nest_asyncio only with a running loop and Python <= 3.13: on 3.14 it breaks
    # asyncio.current_task() and 500s every request (archived upstream).
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

    # `from main import app` takes ~2 min cold; flush a heads-up.
    if not silent:
        print(
            "Loading Unsloth Studio, please wait... (this can take a few minutes)",
            flush = True,
        )

    _repair_pinned_diffusers(silent)

    if not silent:
        print("  - loading PyTorch, Unsloth and Transformers...", flush = True)

    import_started = time.perf_counter()

    # Before the import, to shrink the window where two launches think they are alone. The cache env
    # must be seeded first: UNSLOTH_COMPILE_LOCATION is part of the coordination key.
    try:
        from utils.paths.storage_roots import setup_cache_env
        setup_cache_env()
    except Exception:  # noqa: BLE001
        pass
    write_startup_marker()

    from main import app, setup_frontend, _desktop_owner, _IS_COLAB

    # A callable, re-asked at shutdown. On app.state: main.py must not import this module.
    app.state.live_sibling_backend = live_sibling_backend

    logger.info(
        "Imported FastAPI app in %.1fms",
        (time.perf_counter() - import_started) * 1000,
    )
    if not silent:
        print("  - Starting server...", flush = True)
    from utils.paths import ensure_studio_directories

    # Local stdio MCP on loopback binds, never on Colab. Read at request time.
    from utils.host_policy import apply_stdio_mcp_loopback_default

    apply_stdio_mcp_loopback_default(host, is_colab = _IS_COLAB)

    ensure_studio_directories()

    logger.info(
        "Ensured Unsloth directories in %.1fms",
        (time.perf_counter() - boot_started) * 1000,
    )

    original_port = port
    # `studio run` and the desktop app follow port fallback; only the bare launch refuses.
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
            print(f"Open {_url_scheme}://localhost:{port} in your browser.")
            print("=" * 50)
            print("")

    _serve_frontend, _tunnel_only_frontend = _frontend_serving_mode(
        api_only = api_only,
        desktop_owned = _desktop_owner() is not None,
    )

    # desktop api-only serves its packaged SPA to remote callers only: tunnel or LAN, not loopback
    _frontend_mounted = False
    if frontend_path and _serve_frontend:
        chosen, attempted = _resolve_frontend_path(Path(frontend_path))
        if chosen is not None and setup_frontend(app, chosen, tunnel_only = _tunnel_only_frontend):
            _frontend_mounted = True
            if not silent:
                try:
                    display = chosen.resolve()
                except OSError:
                    display = chosen
                print(f"[OK] Frontend loaded from {display}")
        elif not _missing_frontend_is_fatal(tunnel_only = _tunnel_only_frontend):
            logger.warning(
                "No frontend build found, so Remote and LAN access will not serve the web UI. "
                "Tried: %s",
                ", ".join(str(p) for p in attempted) or "(none)",
            )
        else:
            home_str = (
                os.environ.get("UNSLOTH_STUDIO_HOME")
                or os.environ.get("STUDIO_HOME")
                or str(Path.home() / ".unsloth" / "studio")
            )
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
                + (
                    # App Control can block the generated unsloth.exe while the signed interpreter still runs.
                    f"  - or through the interpreter: {sys.executable} -I -m unsloth_cli studio\n"
                    if sys.platform == "win32"
                    else ""
                )
                + "  - pass --frontend <path/to/studio/frontend/dist>\n"
                "  - pass --api-only to skip serving the web UI\n"
                "  - reinstall: curl -fsSL https://unsloth.ai/install.sh | sh"
            )

    display_host = _display_host_for_bind(host)
    _install_uvicorn_startup_log_rewrite(host)
    # LoggingMiddleware already logs these with tracebacks; avoid duplicates in tauri.log.
    install_uvicorn_duplicate_exception_filter()

    logger.info(
        "run_server pre-uvicorn setup completed in %.1fms",
        (time.perf_counter() - boot_started) * 1000,
    )

    ready_event = Event()
    startup_failed = Event()
    startup_errors = []

    class _ReadyServer(uvicorn.Server):
        loopback_http = None

        async def startup(self, *args, **kwargs):
            await super().startup(*args, **kwargs)
            if tls is not None and getattr(self, "started", False) and not self.should_exit:
                await self._start_loopback_http()
            if getattr(self, "started", False) and not self.should_exit:
                logger.info(
                    "Uvicorn startup hook completed in %.1fms",
                    (time.perf_counter() - boot_started) * 1000,
                )
                ready_event.set()

        async def _start_loopback_http(self):
            # Plain HTTP on an ephemeral loopback port for Studio's own /v1 self-calls (utils/studio_tls.py). Kept out
            # of self.servers: _final_bound_port and _bound_request_host read the user's listener from there.
            config = self.config

            def _protocol(_loop = None):
                return config.http_protocol_class(
                    config = config,
                    server_state = self.server_state,
                    app_state = self.lifespan.state,
                    _loop = _loop,
                )

            self.loopback_http = await asyncio.get_running_loop().create_server(
                _protocol, host = "127.0.0.1", port = 0, backlog = config.backlog
            )
            app.state.loopback_http_port = self.loopback_http.sockets[0].getsockname()[1]

        async def shutdown(self, *args, **kwargs):
            if self.loopback_http is not None:
                self.loopback_http.close()
            await super().shutdown(*args, **kwargs)

    # server_header=False suppresses uvicorn's "Server: uvicorn"; SecurityHeadersMiddleware sets its own.
    # http=... is uvicorn's own h11 protocol unless we are on the plain-h11 path, where it becomes a subclass
    # that ignores socket reads delivered after the connection was already closed. Without it every clean
    # shutdown on Windows ends in an h11 LocalProtocolError traceback that reads as a crash; see
    # utils/uvicorn_h11_shutdown.py.
    from utils.uvicorn_h11_shutdown import uvicorn_http_protocol

    config_kwargs = dict(
        host = host,
        port = port,
        log_level = "info",
        access_log = False,
        server_header = False,
        http = uvicorn_http_protocol(),
    )
    # forwarded_allow_ips="*" is safe only in Colab's single-user sandbox.
    if _IS_COLAB:
        config_kwargs["proxy_headers"] = True
        config_kwargs["forwarded_allow_ips"] = "*"
    if tls is not None:
        from utils.studio_tls import StudioTLSConfig
        config = StudioTLSConfig(app, tls = tls, **config_kwargs)
    else:
        config = uvicorn.Config(app, **config_kwargs)
    app.state.studio_tls = tls
    app.state.loopback_http_port = None
    _server = _ReadyServer(config)
    _shutdown_event = Event()

    # Real bound port for loopback URLs; unset for ephemeral binds (port==0).
    app.state.server_port = port if port and port > 0 else None
    app.state.server_request_host = None
    app.state.server_url = _direct_server_url(host, port)
    app.state.bind_host = host
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

    from utils.lan_access_settings import configure_lan_access

    configure_lan_access(
        app.state,
        port = port,
        bind_host = host,
        secure = secure,
        is_colab = _IS_COLAB,
        frontend_served = _serve_frontend and _frontend_mounted,
    )

    def _trigger_shutdown():
        _graceful_shutdown(_server)
        if _shutdown_event is not None:
            _shutdown_event.set()

    app.state.trigger_shutdown = _trigger_shutdown

    _apply_supplied_password(password)

    # Per launch: an embedded host may call run_server() again with different flags.
    app.state.api_only = api_only
    app.state.suppress_bootstrap_injection = False

    # Runs BEFORE the socket binds so no listener hands out the injected default credential.
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
            "Not starting Unsloth; set a new admin password first, or pass one "
            "non-interactively with --password / UNSLOTH_STUDIO_PASSWORD. "
            + (
                "Launch without --secure/--cloudflare to stay off the public internet."
                if _launch_tunnel_managed
                else "Launch with -H 127.0.0.1 to keep Unsloth off the network."
            ),
            file = sys.stderr,
            flush = True,
        )
        sys.exit(1)
    if _pw_drop_bootstrap:
        # Lifespan runs AFTER this and re-reads the bootstrap password; the flag makes it skip that.
        app.state.suppress_bootstrap_injection = True
        app.state.bootstrap_password = None

    from cloudflare_tunnel import open_studio_tunnel_lifecycle

    open_studio_tunnel_lifecycle()

    # new_event_loop + run_until_complete, not asyncio.run, so nest_asyncio does not interfere.
    def _run():
        from utils.proactor_self_pipe import install_proactor_self_pipe_guard

        install_proactor_self_pipe_guard()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app.state.lan_access_loop = loop
        try:
            loop.run_until_complete(_server.serve())
        except BaseException as exc:
            startup_errors.append(exc)
            startup_failed.set()
        finally:
            loop.close()
            if not ready_event.is_set():
                startup_failed.set()
            # Embedded hosts outlive the server thread; this is where the backend truly stops serving.
            _remove_startup_marker()
            _remove_pid_file()
            from lan_access import close_lan_listener_lifecycle as _close_lan_listener

            _close_lan_listener()

    # Embedded hosts (colab.py) may rerun run_server; clear the process-wide shutdown flags here,
    # after `from main import app` and before uvicorn serves.
    try:
        from routes.inference import _llama_cpp_backend, begin_load_lifecycle
        from utils.process_lifetime import begin_process_lifecycle

        if _llama_cpp_backend is not None:
            _llama_cpp_backend._begin_server_lifecycle()
        begin_process_lifecycle()
        begin_load_lifecycle()
    except Exception as e:
        logger.warning("Could not reset llama-server shutdown state: %s", e)

    thread = Thread(target = _run, daemon = True)
    _server_thread = thread
    thread.start()

    # No deadline: a slow but live startup stays in progress.
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
    app.state.server_request_host = _bound_request_host(_server)
    app.state.server_url = _direct_server_url(host, port)
    app.state.remote_access_port = port
    app.state.lan_access_port = port

    _write_pid_file(port, host)
    import atexit

    atexit.register(_remove_pid_file)
    from utils.process_lifetime import mark_process_shutting_down, terminate_all

    atexit.register(terminate_all)
    # LAST so it runs FIRST (atexit is LIFO): the sweep must see the shutdown latch set.
    atexit.register(mark_process_shutting_down)

    # Tauri port: print only after sockets bind and startup; headless run --api-only opts out.
    if api_only and emit_tauri_port:
        print(f"TAURI_PORT={port}", flush = True)
        # Desktop-owned backends only: a headless --api-only must survive its terminal (nohup).
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
    app.state.lan_access_ready = True

    # Started even when silent so the CLI banner can read app.state.cloudflare_url.
    _cloudflare_enabled = _launch_tunnel_managed
    _cloudflare_requested = _cloudflare_enabled

    if _cloudflare_enabled:
        try:
            from cloudflare_tunnel import start_studio_tunnel
            start_studio_tunnel(
                port,
                managed_by = "launch",
                origin_host = app.state.server_request_host,
            )
        except Exception as e:
            logger.debug("Cloudflare tunnel skipped: %s", e)

    from cloudflare_tunnel import close_studio_tunnel_lifecycle

    atexit.register(close_studio_tunnel_lifecycle)

    # --secure fails closed: no tunnel, no raw-port fallback.
    if secure and not _cloudflare_url:
        print(
            "A secure Cloudflare link is not allowed, use --no-secure which provides a 0.0.0.0 link",
            file = sys.stderr,
            flush = True,
        )
        _graceful_shutdown(_server)
        sys.exit(1)

    # Shut down a freshly exposed web UI if the seeded password is unchanged by the deadline.
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
    except Exception as e:
        logger.warning("Bootstrap timeout not armed: %s", e)

    from utils.remote_access_settings import maybe_auto_start_remote_access

    if maybe_auto_start_remote_access(app.state):
        logger.info("Remote access auto-start scheduled")

    from lan_access import close_lan_listener_lifecycle, lan_listener_status
    from utils.lan_access_settings import maybe_auto_start_lan_access

    atexit.register(close_lan_listener_lifecycle)
    if maybe_auto_start_lan_access(app):
        logger.info("LAN access auto-started")

    if not silent:
        _emit_startup_output(
            host,
            port,
            display_host,
            secure = secure,
            enable_tools = enable_tools,
            lan_addresses = tuple(lan_listener_status()["addresses"]),
        )

    return app


def _build_arg_parser():
    """Build the backend CLI argument parser, extracted from the __main__ block so the flag wiring
    (notably the --secure/--no-secure polarity and its --not-secure alias) stays unit-testable."""
    import argparse

    parser = argparse.ArgumentParser(description = "Run Unsloth UI Backend server")
    parser.add_argument(
        "--host",
        default = "127.0.0.1",
        help = "Host to bind to (default: 127.0.0.1; use 0.0.0.0 for network/cloud access). "
        "On a shared host, set UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 to stop external "
        "provider connections from targeting private addresses.",
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
    parser.add_argument(
        "--not-secure",
        dest = "secure",
        action = "store_false",
        default = argparse.SUPPRESS,
        help = argparse.SUPPRESS,
    )
    # None = tools default on and a request's enable_tools: false is honored.
    parser.add_argument(
        "--enable-tools",
        dest = "enable_tools",
        action = "store_true",
        default = None,
        help = "Force server-side tools (web search, code execution) on for "
        "every request. Default: no server-wide policy, so each request's own "
        "enable_tools decides (`unsloth studio run` is the launcher that defaults "
        "them on). "
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
        help = "Send the hostname (not the validated IP) in web fetches that go through an "
        "explicitly configured HTTP(S)_PROXY, so the proxy can apply hostname policy and "
        "TLS interception. Direct fetches stay pinned to the validated IP.",
    )
    parser.add_argument(
        "--parallel",
        "--n-parallel",
        type = int,
        default = _PARALLEL_DEFAULT_PLAIN,
        help = (
            f"llama-server parallel decode slots ({_PARALLEL_MIN}..{_PARALLEL_MAX}). "
            f"Default {_PARALLEL_DEFAULT_PLAIN}. The Unsloth run settings "
            "(Parallel Slots) override it per load."
        ),
    )
    parser.add_argument(
        "--ssl-certfile",
        default = None,
        help = "Serve HTTPS with this certificate (PEM, full chain). Needs --ssl-keyfile.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default = None,
        help = "Private key (PEM) for --ssl-certfile.",
    )
    return parser


if __name__ == "__main__":
    # Before torch is imported: re-execs once on Linux so torch's bundled CUDA libs win.
    _maybe_reexec_for_cuda_ld_path()

    import signal
    import traceback

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
        ssl_certfile = args.ssl_certfile,
        ssl_keyfile = args.ssl_keyfile,
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

    def _signal_handler(signum, frame):
        # Restore defaults so a second signal force-quits if shutdown stalls.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_DFL)
        _note_quit_signal()
        _graceful_shutdown(_server)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    # Not _signal_handler: Windows runs this off the main thread, where signal.signal() raises.
    def _console_shutdown():
        _graceful_shutdown(_server)
        _shutdown_event.set()

    _install_windows_console_handler(_console_shutdown)

    # A timeout-less Event.wait() blocks SIGINT delivery on Linux.
    while not _shutdown_event.is_set():
        _shutdown_event.wait(timeout = 1)
    _wait_for_server_shutdown()

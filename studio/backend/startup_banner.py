# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Terminal banner for Studio startup.

Stdlib only — safe to import without the rest of the backend (no structlog/uvicorn).
"""

from __future__ import annotations

import os
import sys


def stdout_supports_color() -> bool:
    """True if we should emit ANSI colors."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def print_port_in_use_notice(original_port: int, new_port: int) -> None:
    """Message when the requested port is taken and another is chosen."""
    msg = f"Port {original_port} is in use, using port {new_port} instead."
    if stdout_supports_color():
        print(f"\033[38;5;245m{msg}\033[0m")
    else:
        print(msg)


def print_studio_stop_hint() -> None:
    """Print the trailing stop hint + closing divider. Separate from the main
    banner so callers can interleave content (e.g. a reachability check)."""
    use_color = stdout_supports_color()
    dim = "\033[38;5;245m"
    stop_hint_style = "\033[38;5;215;1m"
    reset = "\033[0m"

    def style(text: str, code: str) -> str:
        return f"{code}{text}{reset}" if use_color else text

    print(
        "\n".join(
            [
                "",
                style(
                    "  To stop Unsloth Studio: press Ctrl+C in this terminal.",
                    stop_hint_style,
                ),
                style("  (On macOS this is Control+C, not Command+C.)", dim),
                style("─" * 52, dim),
                "",
            ]
        )
    )


def print_studio_access_banner(
    *,
    port: int,
    bind_host: str,
    display_host: str,
    include_stop_hint: bool = True,
) -> None:
    """Pretty-print URLs after the server is listening. Set
    ``include_stop_hint=False`` to omit the trailing stop block; pair with
    :func:`print_studio_stop_hint` after inserting your own content."""
    use_color = stdout_supports_color()
    dim = "\033[38;5;245m"
    title = "\033[38;5;150m"
    local_url_style = "\033[38;5;108;1m"
    secondary = "\033[38;5;109m"
    stop_hint_style = "\033[38;5;215;1m"
    reset = "\033[0m"

    def style(text: str, code: str) -> str:
        return f"{code}{text}{reset}" if use_color else text

    ipv6_bind = bind_host in ("::", "::1")
    if ipv6_bind:
        loopback_url = f"http://[::1]:{port}"
        alt_local = f"http://localhost:{port}"
    else:
        loopback_url = f"http://127.0.0.1:{port}"
        alt_local = f"http://localhost:{port}"
    if ":" in display_host:
        external_url = f"http://[{display_host}]:{port}"
    else:
        external_url = f"http://{display_host}:{port}"

    listen_all = bind_host in ("0.0.0.0", "::")
    loopback_bind = bind_host in ("127.0.0.1", "localhost", "::1")

    # Use loopback URL only when the server is reachable on loopback;
    # otherwise show the actual bound address.
    primary_url = loopback_url if listen_all or loopback_bind else external_url
    tip_url = alt_local if listen_all or loopback_bind else external_url
    api_base = primary_url

    lines: list[str] = [
        "",
        style("🦥 Unsloth Studio is running", title),
        style("─" * 52, dim),
        style("  On this machine -- open this in your browser:", dim),
        style(f"    {primary_url}", local_url_style),
    ]

    if (listen_all or loopback_bind) and primary_url != alt_local:
        lines.append(style(f"    (same as {alt_local})", dim))

    if listen_all and display_host not in (
        "127.0.0.1",
        "localhost",
        "::1",
        "0.0.0.0",
        "::",
    ):
        lines.extend(
            [
                "",
                style("  From another device on your network / to share:", dim),
                style(f"    {external_url}", secondary),
            ]
        )
    elif not listen_all and not loopback_bind and external_url != primary_url:
        lines.extend(
            [
                "",
                style("  Bound address:", dim),
                style(f"    {external_url}", secondary),
            ]
        )

    lines.extend(
        [
            "",
            style("  API & health:", dim),
            style(f"    {api_base}/api", secondary),
            style(f"    {api_base}/api/health", secondary),
            style("─" * 52, dim),
            style(
                f"  Tip: if you are on this computer, open {tip_url}/ in your browser.",
                dim,
            ),
        ]
    )

    if loopback_bind and not listen_all:
        lines.extend(
            [
                "",
                style(
                    "  Studio is only reachable on this machine (bound to 127.0.0.1).",
                    secondary,
                ),
                style(
                    "  To deploy and access globally:",
                    secondary,
                ),
                style(
                    "    1. press Ctrl+C to stop Studio",
                    secondary,
                ),
                style(
                    f"    2. relaunch with:  unsloth studio -H 0.0.0.0 -p {port}",
                    secondary,
                ),
                style(
                    "  Only do this on trusted networks -- it exposes the API on every interface.",
                    secondary,
                ),
            ]
        )

    if include_stop_hint:
        lines.extend(
            [
                "",
                style(
                    "  To stop Unsloth Studio: press Ctrl+C in this terminal.",
                    stop_hint_style,
                ),
                style("  (On macOS this is Control+C, not Command+C.)", dim),
                style("─" * 52, dim),
                "",
            ]
        )

    print("\n".join(lines))

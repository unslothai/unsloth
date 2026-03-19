#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Print Studio terminal messages as users see them — no server, no extra deps.

Run from anywhere:

  python studio/backend/print_terminal_harness.py
  python studio/backend/print_terminal_harness.py --scenario loopback
  FORCE_COLOR=1 python studio/backend/print_terminal_harness.py | less -R

See also: studio/ITERATING.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from startup_banner import (  # noqa: E402
    print_port_in_use_notice,
    print_studio_access_banner,
)


MSG_REEXEC = "Launching with studio venv..."
MSG_START = "Starting Unsloth Studio (connection links print in a few seconds)..."
MSG_FRONTEND_OK = "✅ Frontend loaded from /path/to/frontend/dist"


def _apply_color_mode(mode: str) -> None:
    if mode == "off":
        os.environ["NO_COLOR"] = "1"
        os.environ.pop("FORCE_COLOR", None)
    elif mode == "on":
        os.environ.pop("NO_COLOR", None)
        os.environ["FORCE_COLOR"] = "1"
    else:
        os.environ.pop("FORCE_COLOR", None)
        os.environ.pop("NO_COLOR", None)


def _print_full_sequence(
    *,
    port: int,
    bind_host: str,
    display_host: str,
    include_reexec: bool,
    include_start: bool,
    include_frontend: bool,
    port_was_busy: bool,
    original_port: int,
) -> None:
    if include_reexec:
        print(MSG_REEXEC)
    if include_start:
        print(MSG_START)
    if port_was_busy:
        print_port_in_use_notice(original_port, port)
    if include_frontend:
        print(MSG_FRONTEND_OK)
    print_studio_access_banner(
        port = port,
        bind_host = bind_host,
        display_host = display_host,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Preview Studio terminal output (stdlib only).",
    )
    parser.add_argument(
        "--scenario",
        choices = ("default", "loopback", "lan-share", "custom-bind", "minimal"),
        default = "default",
        help = "Preset host/display combinations (default: 0.0.0.0 + LAN-style IP).",
    )
    parser.add_argument("--port", type = int, default = 8000)
    parser.add_argument("--host", default = None, help = "Override bind host")
    parser.add_argument(
        "--display-host",
        default = None,
        help = "Override resolved display host (share line)",
    )
    parser.add_argument(
        "--color",
        choices = ("auto", "on", "off"),
        default = "auto",
        help = "on = FORCE_COLOR (e.g. piping to less -R); off = NO_COLOR",
    )
    parser.add_argument(
        "--no-prelude",
        action = "store_true",
        help = "Banner only (skip CLI lines before the wait)",
    )
    parser.add_argument(
        "--with-reexec",
        action = "store_true",
        help = "Include 'Launching with studio venv...' (outer CLI)",
    )
    parser.add_argument(
        "--with-frontend",
        action = "store_true",
        help = "Include sample 'Frontend loaded' line",
    )
    parser.add_argument(
        "--port-was-busy",
        action = "store_true",
        help = "Print port-reassignment line before the banner",
    )
    parser.add_argument(
        "--original-port",
        type = int,
        default = 8000,
        help = "Original port when using --port-was-busy",
    )

    args = parser.parse_args()
    _apply_color_mode(args.color)

    scenarios = {
        "default": ("0.0.0.0", "192.168.1.42"),
        "loopback": ("127.0.0.1", "127.0.0.1"),
        "lan-share": ("0.0.0.0", "10.0.0.15"),
        "custom-bind": ("192.168.1.10", "192.168.1.10"),
        "minimal": ("0.0.0.0", "192.168.1.42"),
    }
    bind_host, display_host = scenarios[args.scenario]
    if args.host is not None:
        bind_host = args.host
    if args.display_host is not None:
        display_host = args.display_host

    include_reexec = args.with_reexec
    include_start = not args.no_prelude
    include_frontend = args.with_frontend
    if args.scenario == "minimal":
        include_start = False
        include_reexec = args.with_reexec
        include_frontend = args.with_frontend

    _print_full_sequence(
        port = args.port,
        bind_host = bind_host,
        display_host = display_host,
        include_reexec = include_reexec,
        include_start = include_start,
        include_frontend = include_frontend,
        port_was_busy = args.port_was_busy,
        original_port = args.original_port,
    )


if __name__ == "__main__":
    main()

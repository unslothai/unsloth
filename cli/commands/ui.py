# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
from pathlib import Path
from typing import Optional

import typer


def ui(
    port: int = typer.Option(
        8000, "--port", "-p", help = "Port to run the UI server on."
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", "-H", help = "Host address to bind to."
    ),
    frontend: Optional[Path] = typer.Option(
        None, "--frontend", "-f", help = "Path to frontend build directory."
    ),
    silent: bool = typer.Option(
        False, "--silent", "-q", help = "Suppress startup messages."
    ),
):
    """Launch the Unsloth web UI backend server."""
    from studio.backend.run import run_server

    if not silent:
        from studio.backend.run import _resolve_external_ip

        display_host = _resolve_external_ip() if host == "0.0.0.0" else host
        typer.echo(f"Starting Unsloth Studio on http://{display_host}:{port}")

    run_server(
        host = host,
        port = port,
        frontend_path = frontend,
        silent = silent,
    )

    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\nShutting down...")

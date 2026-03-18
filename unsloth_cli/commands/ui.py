# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
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
    """Launch the Unsloth web UI backend server (alias for 'unsloth studio')."""
    from unsloth_cli.commands.studio import (
        _studio_venv_python,
        _find_run_py,
        STUDIO_HOME,
    )

    # Re-execute in studio venv if available and not already inside it
    studio_venv_dir = STUDIO_HOME / ".venv"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))

    if not in_studio_venv:
        studio_python = _studio_venv_python()
        run_py = _find_run_py()
        if studio_python and run_py:
            if not silent:
                typer.echo("Launching with studio venv...")
            args = [
                str(studio_python),
                str(run_py),
                "--host",
                host,
                "--port",
                str(port),
            ]
            if frontend:
                args.extend(["--frontend", str(frontend)])
            if silent:
                args.append("--silent")
            # On Windows, os.execvp() spawns a child but the parent lingers,
            # so Ctrl+C only kills the parent leaving the child orphaned.
            # Use subprocess.run() on Windows so the parent waits for the child.
            if sys.platform == "win32":
                import subprocess as _sp

                proc = _sp.Popen(args)
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    # Child has its own signal handler — let it finish
                    rc = proc.wait()
                raise typer.Exit(rc)
            else:
                os.execvp(str(studio_python), args)
        else:
            typer.echo("Studio not set up. Run 'unsloth studio setup' first.")
            raise typer.Exit(1)

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

    from studio.backend.run import _shutdown_event

    try:
        if _shutdown_event is not None:
            _shutdown_event.wait()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        from studio.backend.run import _graceful_shutdown, _server

        _graceful_shutdown(_server)
        typer.echo("\nShutting down...")

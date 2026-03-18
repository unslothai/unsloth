# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import typer

studio_app = typer.Typer(help = "Unsloth Studio commands.")

STUDIO_HOME = Path.home() / ".unsloth" / "studio"

# __file__ is unsloth_cli/commands/studio.py -- two parents up is the package root
# (either site-packages or the repo root for editable installs).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _studio_venv_python() -> Optional[Path]:
    """Return the studio venv Python binary, or None if not set up."""
    if platform.system() == "Windows":
        p = STUDIO_HOME / ".venv" / "Scripts" / "python.exe"
    else:
        p = STUDIO_HOME / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def _find_run_py() -> Optional[Path]:
    """Find studio/backend/run.py.

    No CWD dependency — works from any directory.
    Since studio/ is now a proper package (has __init__.py), it lives in
    site-packages after pip install, right next to unsloth_cli/.
    """
    # 1. Relative to __file__ (site-packages or editable repo root)
    run_py = _PACKAGE_ROOT / "studio" / "backend" / "run.py"
    if run_py.is_file():
        return run_py
    # 2. Studio venv's site-packages (Linux + Windows layouts)
    for pattern in (
        "lib/python*/site-packages/studio/backend/run.py",
        "Lib/site-packages/studio/backend/run.py",
    ):
        for match in (STUDIO_HOME / ".venv").glob(pattern):
            return match
    return None


def _find_setup_script() -> Optional[Path]:
    """Find studio/setup.sh or studio/setup.ps1.

    No CWD dependency — works from any directory.
    """
    name = "setup.ps1" if platform.system() == "Windows" else "setup.sh"
    # 1. Relative to __file__ (site-packages or editable repo root)
    s = _PACKAGE_ROOT / "studio" / name
    if s.is_file():
        return s
    # 2. Studio venv's site-packages
    for pattern in (
        f"lib/python*/site-packages/studio/{name}",
        f"Lib/site-packages/studio/{name}",
    ):
        for match in (STUDIO_HOME / ".venv").glob(pattern):
            return match
    return None


# ── unsloth studio (server) ──────────────────────────────────────────


@studio_app.callback(invoke_without_command = True)
def studio_default(
    ctx: typer.Context,
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
):
    """Launch the Unsloth Studio server."""
    if ctx.invoked_subcommand is not None:
        return

    # Always use the studio venv if it exists and we're not already in it
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
            # NOTE: Event.wait() without a timeout blocks at the C level
            # on Linux, preventing Python from delivering SIGINT (Ctrl+C).
            while not _shutdown_event.is_set():
                _shutdown_event.wait(timeout = 1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        from studio.backend.run import _graceful_shutdown, _server

        _graceful_shutdown(_server)
        typer.echo("\nShutting down...")


# ── unsloth studio setup ─────────────────────────────────────────────


@studio_app.command()
def setup():
    """Run one-time Studio environment setup."""
    script = _find_setup_script()
    if not script:
        typer.echo("Error: Could not find setup script (setup.sh / setup.ps1).")
        raise typer.Exit(1)

    if platform.system() == "Windows":
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        )
    else:
        result = subprocess.run(["bash", str(script)])

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


# ── unsloth studio reset-password ────────────────────────────────────


@studio_app.command("reset-password")
def reset_password():
    """Reset the Studio admin password.

    Deletes the auth database so that a fresh admin account with a new
    random password is created on the next server start.  The Studio
    server must be restarted after running this command.
    """
    auth_dir = STUDIO_HOME / "auth"
    db_file = auth_dir / "auth.db"
    pw_file = auth_dir / ".bootstrap_password"

    if not db_file.exists():
        typer.echo("No auth database found -- nothing to reset.")
        raise typer.Exit(0)

    db_file.unlink(missing_ok = True)
    pw_file.unlink(missing_ok = True)

    typer.echo("Auth database deleted. Restart Unsloth Studio to get a new password.")

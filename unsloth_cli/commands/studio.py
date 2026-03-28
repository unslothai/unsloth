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
        p = STUDIO_HOME / "unsloth_studio" / "Scripts" / "python.exe"
    else:
        p = STUDIO_HOME / "unsloth_studio" / "bin" / "python"
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
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
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
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


# ── unsloth studio (server) ──────────────────────────────────────────


@studio_app.callback(invoke_without_command = True)
def studio_default(
    ctx: typer.Context,
    port: int = typer.Option(8888, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Launch the Unsloth Studio server."""
    if ctx.invoked_subcommand is not None:
        return

    if debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ.setdefault("ENVIRONMENT_TYPE", "development")

    # Always use the studio venv if it exists and we're not already in it
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))

    if not in_studio_venv:
        studio_python = _studio_venv_python()
        run_py = _find_run_py()
        if studio_python and run_py:
            if not silent:
                typer.echo("Launching Unsloth Studio... Please wait...")
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
            if debug:
                args.append("--debug")
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
                if rc != 0:
                    typer.echo(
                        f"\nError: Studio server exited unexpectedly (code {rc}).",
                        err = True,
                    )
                    typer.echo(
                        "Check the error above. If a package is missing, "
                        "re-run: unsloth studio setup",
                        err = True,
                    )
                raise typer.Exit(rc)
            else:
                os.execvp(str(studio_python), args)
        else:
            typer.echo("Studio not set up. Run install.sh first.")
            raise typer.Exit(1)

    from studio.backend.run import run_server

    if not silent:
        from studio.backend.run import _resolve_external_ip

        display_host = _resolve_external_ip() if host == "0.0.0.0" else host
        typer.echo(f"Starting Unsloth Studio on http://{display_host}:{port}")
        if debug:
            typer.echo("Debug logging enabled (LOG_LEVEL=DEBUG)")

    run_kwargs = dict(host = host, port = port, silent = silent)
    if frontend is not None:
        run_kwargs["frontend_path"] = frontend
    run_server(**run_kwargs)

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


# ── unsloth studio stop ───────────────────────────────────────────────

_PID_FILE = STUDIO_HOME / "studio.pid"


@studio_app.command()
def stop():
    """Stop a running Unsloth Studio server.

    Reads the PID from ~/.unsloth/studio/studio.pid and sends SIGTERM
    (or TerminateProcess on Windows) to shut it down gracefully.
    """
    import signal as _signal

    if not _PID_FILE.is_file():
        typer.echo("No running Studio server found (no PID file).")
        raise typer.Exit(0)

    pid_text = _PID_FILE.read_text().strip()
    if not pid_text.isdigit():
        typer.echo(f"Invalid PID file contents: {pid_text}")
        _PID_FILE.unlink(missing_ok = True)
        raise typer.Exit(1)

    pid = int(pid_text)

    # Check if the process is still alive
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        typer.echo(
            f"Studio server (PID {pid}) is not running. Cleaning up stale PID file."
        )
        _PID_FILE.unlink(missing_ok = True)
        raise typer.Exit(0)
    except PermissionError:
        pass  # process exists but we may not own it; try to signal anyway

    # Send SIGTERM (graceful shutdown) or TerminateProcess on Windows
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check = True)
        else:
            os.kill(pid, _signal.SIGTERM)
        typer.echo(f"Sent shutdown signal to Studio server (PID {pid}).")
    except ProcessLookupError:
        typer.echo(f"Studio server (PID {pid}) already exited.")
        _PID_FILE.unlink(missing_ok = True)
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Failed to stop Studio server (PID {pid}): {e}", err = True)
        raise typer.Exit(1)

    # Wait briefly for the process to exit and clean up
    for _ in range(10):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _PID_FILE.unlink(missing_ok = True)
            typer.echo("Studio server stopped.")
            raise typer.Exit(0)
        except PermissionError:
            break

    typer.echo("Studio server is shutting down (may take a few seconds).")


# ── unsloth studio setup / update ─────────────────────────────────────


def _run_setup_script(*, verbose: bool = False) -> None:
    """Find and run the studio setup/update script."""
    script = _find_setup_script()
    if not script:
        typer.echo("Error: Could not find setup script (setup.sh / setup.ps1).")
        raise typer.Exit(1)

    env = {**os.environ, "UNSLOTH_VERBOSE": "1"} if verbose else None

    if platform.system() == "Windows":
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            env = env,
        )
    else:
        result = subprocess.run(["bash", str(script)], env = env)

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@studio_app.command(hidden = True)
def setup(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Full pip/build output during setup for troubleshooting.",
    ),
):
    """Deprecated: use 'unsloth studio update' or re-run install.sh."""
    typer.echo(
        "Note: 'unsloth studio setup' is deprecated. Use 'unsloth studio update' or re-run install.sh."
    )
    _run_setup_script(verbose = verbose)


@studio_app.command()
def update(
    local: bool = typer.Option(
        False, "--local", help = "Install from local repo instead of PyPI"
    ),
    package: str = typer.Option(
        "unsloth", "--package", help = "Package name to install/update (for testing)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Full pip/build output during update for troubleshooting.",
    ),
):
    """Update Unsloth Studio dependencies and rebuild."""
    os.environ["STUDIO_LOCAL_INSTALL"] = "1" if local else "0"
    os.environ["STUDIO_PACKAGE_NAME"] = package
    if local:
        # Pass the repo root explicitly so install_python_stack.py doesn't
        # have to guess from SCRIPT_DIR (which may be inside site-packages).
        repo_root = Path(__file__).resolve().parents[2]
        os.environ["STUDIO_LOCAL_REPO"] = str(repo_root)
    _run_setup_script(verbose = verbose)


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

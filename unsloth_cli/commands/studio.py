# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import hashlib
import json
import os
import platform
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import typer

studio_app = typer.Typer(help = "Unsloth Studio commands.")

STUDIO_HOME = Path.home() / ".unsloth" / "studio"
BOOTSTRAP_PASSWORD_FILE = ".bootstrap_password"
DESKTOP_SECRET_FILE = ".desktop_secret"
DEFAULT_ADMIN_USERNAME = "unsloth"
DESKTOP_SECRET_PREFIX = "desktop-"
API_KEY_PBKDF2_SALT_KEY = "api_key_pbkdf2_salt"
DESKTOP_SECRET_HASH_KEY = "desktop_secret_hash"
DESKTOP_SECRET_CREATED_AT_KEY = "desktop_secret_created_at"
PBKDF2_ITERATIONS = 100_000

# __file__ is unsloth_cli/commands/studio.py -- two parents up is the package root
# (either site-packages or the repo root for editable installs).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _should_hide_windows_subprocesses() -> bool:
    """Hide child console windows only for non-interactive Windows launches."""
    if platform.system() != "Windows":
        return False
    try:
        return not sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return True


def _windows_hidden_subprocess_kwargs() -> dict[str, object]:
    """Return Windows-only Popen kwargs that suppress transient console windows."""
    if not _should_hide_windows_subprocesses():
        return {}

    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    sw_hide = getattr(subprocess, "SW_HIDE", 0)
    if startupinfo_factory is not None and startf_use_showwindow:
        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = sw_hide
        kwargs["startupinfo"] = startupinfo

    return kwargs


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


# ── helpers for `unsloth studio run` ────────────────────────────────


def _wait_for_server(port: int, timeout: int = 30) -> bool:
    """Poll ``GET /api/health`` until the server responds 200 or *timeout* expires."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout = 2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def _create_api_key_inprocess(name: str) -> str:
    """Create an API key via direct storage call (no HTTP needed).

    Bypasses the ``must_change_password`` gate that blocks HTTP
    ``POST /api/auth/api-keys`` on fresh installs.  Safe because the
    CLI already has filesystem access to ``~/.unsloth/studio``.
    """
    storage = _load_backend_auth_storage()

    raw_key, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = name,
    )
    return raw_key


def _load_backend_auth_storage():
    run_py = _find_run_py()
    backend_dir = (
        run_py.parent if run_py is not None else _PACKAGE_ROOT / "studio" / "backend"
    )
    if backend_dir.is_dir() and str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    auth_dir = backend_dir / "auth"
    storage_py = auth_dir / "storage.py"
    loaded = sys.modules.get("auth.storage")
    loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
    if loaded is not None and loaded_path == storage_py:
        return loaded

    package = sys.modules.get("auth")
    package_paths = [Path(path).resolve() for path in getattr(package, "__path__", [])]
    if package is None or auth_dir.resolve() not in package_paths:
        package = types.ModuleType("auth")
        package.__path__ = [str(auth_dir)]
        package.__package__ = "auth"
        package.__file__ = str(auth_dir / "__init__.py")
        sys.modules["auth"] = package

    spec = importlib.util.spec_from_file_location("auth.storage", storage_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend auth storage from {storage_py}")
    storage = importlib.util.module_from_spec(spec)
    sys.modules["auth.storage"] = storage
    spec.loader.exec_module(storage)

    return storage


def _write_auth_secret(path: Path, secret: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    fd, tmp_name = tempfile.mkstemp(prefix = f".{path.name}.", dir = path.parent)
    tmp_path = Path(tmp_name)
    try:
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w") as f:
            fd = -1
            f.write(secret)
        os.replace(tmp_path, path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok = True)
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _connect_auth_db() -> sqlite3.Connection:
    auth_dir = STUDIO_HOME / "auth"
    auth_dir.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(auth_dir / "auth.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            jwt_secret TEXT NOT NULL,
            must_change_password INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY,
            token_hash TEXT NOT NULL,
            username TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            is_desktop INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            key_prefix TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            expires_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_secrets (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    auth_columns = {row[1] for row in conn.execute("PRAGMA table_info(auth_user)")}
    if "must_change_password" not in auth_columns:
        conn.execute(
            "ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )
    refresh_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(refresh_tokens)")
    }
    if "is_desktop" not in refresh_columns:
        conn.execute(
            "ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()
    return conn


def _pbkdf2_hex(value: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        value.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    ).hex()


def _hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    pwd_hash = _pbkdf2_hex(password, salt.encode("utf-8"))
    return salt, pwd_hash


def _get_or_create_api_key_pbkdf2_salt(conn: sqlite3.Connection) -> bytes:
    row = conn.execute(
        "SELECT value FROM app_secrets WHERE key = ?",
        (API_KEY_PBKDF2_SALT_KEY,),
    ).fetchone()
    if row is None:
        salt_hex = secrets.token_hex(32)
        conn.execute(
            "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
            (API_KEY_PBKDF2_SALT_KEY, salt_hex),
        )
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (API_KEY_PBKDF2_SALT_KEY,),
        ).fetchone()
    return bytes.fromhex(row[0])


def _ensure_cli_default_admin(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT 1 FROM auth_user WHERE username = ?",
        (DEFAULT_ADMIN_USERNAME,),
    ).fetchone()
    if row is not None:
        return

    bootstrap_password = secrets.token_urlsafe(32)
    password_salt, password_hash = _hash_password(bootstrap_password)
    conn.execute(
        """
        INSERT INTO auth_user (
            username,
            password_salt,
            password_hash,
            jwt_secret,
            must_change_password
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            DEFAULT_ADMIN_USERNAME,
            password_salt,
            password_hash,
            secrets.token_urlsafe(64),
            1,
        ),
    )
    _write_auth_secret(
        STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE,
        bootstrap_password,
    )


def _create_desktop_secret_in_cli() -> str:
    raw_secret = DESKTOP_SECRET_PREFIX + secrets.token_urlsafe(48)
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect_auth_db()
    try:
        _ensure_cli_default_admin(conn)
        secret_hash = _pbkdf2_hex(raw_secret, _get_or_create_api_key_pbkdf2_salt(conn))
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (DESKTOP_SECRET_HASH_KEY, secret_hash),
        )
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (DESKTOP_SECRET_CREATED_AT_KEY, now),
        )
        conn.commit()
        return raw_secret
    finally:
        conn.close()


def _load_model_via_http(
    port: int,
    api_key: str,
    model: str,
    gguf_variant: Optional[str],
    max_seq_length: int,
    load_in_4bit: bool,
    timeout: int = 600,
) -> dict:
    """POST to ``/api/inference/load`` using the API key for auth."""
    import json
    import urllib.request
    import urllib.error

    payload: dict = {
        "model_path": model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
    }
    if gguf_variant:
        payload["gguf_variant"] = gguf_variant

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/inference/load",
        data = data,
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors = "replace")
        raise RuntimeError(f"Model load failed (HTTP {exc.code}): {body}") from exc


# ── unsloth studio (server) ──────────────────────────────────────────


@studio_app.callback(invoke_without_command = True)
def studio_default(
    ctx: typer.Context,
    port: int = typer.Option(8888, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
    api_only: bool = typer.Option(
        False,
        "--api-only",
        help = "Run API server only, no frontend serving (for Tauri desktop app)",
    ),
):
    """Launch the Unsloth Studio server."""
    if ctx.invoked_subcommand is not None:
        return

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
            if api_only:
                args.append("--api-only")
            # On Windows, os.execvp() spawns a child but the parent lingers,
            # so Ctrl+C only kills the parent leaving the child orphaned.
            # Use subprocess.run() on Windows so the parent waits for the child.
            if sys.platform == "win32":
                import subprocess as _sp

                proc = _sp.Popen(args, **_windows_hidden_subprocess_kwargs())
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

    run_kwargs = dict(host = host, port = port, silent = silent, api_only = api_only)
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


# ── unsloth studio run ───────────────────────────────────────────────


@studio_app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help = "Model path or HF repo"),
    gguf_variant: Optional[str] = typer.Option(
        None, "--gguf-variant", help = "GGUF quant variant (e.g. UD-Q4_K_XL)"
    ),
    max_seq_length: int = typer.Option(
        0, "--max-seq-length", help = "Max sequence length (0 = model default)"
    ),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
    api_key_name: str = typer.Option(
        "cli", "--api-key-name", help = "Label for the auto-generated API key"
    ),
    port: int = typer.Option(8888, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
):
    """Start Studio, load a model, and print an API key -- one-liner server.

    Example:
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --gguf-variant UD-Q4_K_XL
    """
    # ── 1. Venv re-exec (same pattern as studio_default) ──────────────
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))

    if not in_studio_venv:
        studio_python = _studio_venv_python()
        if not studio_python:
            typer.echo("Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # Re-exec into the studio venv via its `unsloth` entry point
        studio_bin = studio_python.parent / "unsloth"
        if not studio_bin.is_file():
            typer.echo(
                "Studio venv missing 'unsloth' entry point. Re-run: unsloth studio setup"
            )
            raise typer.Exit(1)
        args = [
            str(studio_bin),
            "studio",
            "run",
            "--model",
            model,
            "--max-seq-length",
            str(max_seq_length),
            "--api-key-name",
            api_key_name,
            "--port",
            str(port),
            "--host",
            host,
        ]
        if gguf_variant:
            args.extend(["--gguf-variant", gguf_variant])
        if not load_in_4bit:
            args.append("--no-load-in-4bit")
        if frontend:
            args.extend(["--frontend", str(frontend)])
        if silent:
            args.append("--silent")

        if sys.platform == "win32":
            proc = subprocess.Popen(args)
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                rc = proc.wait()
            raise typer.Exit(rc)
        else:
            os.execvp(str(studio_bin), args)

    # ── 2. Start server (always suppress built-in banner) ─────────────
    from studio.backend.run import run_server, _resolve_external_ip

    run_kwargs = dict(host = host, port = port, silent = True, llama_parallel_slots = 4)
    if frontend is not None:
        run_kwargs["frontend_path"] = frontend
    app = run_server(**run_kwargs)
    actual_port = getattr(app.state, "server_port", port) or port

    # ── 3. Wait for server health ─────────────────────────────────────
    if not silent:
        typer.echo("Starting Unsloth Studio...")
    if not _wait_for_server(actual_port):
        typer.echo("Error: server did not become healthy within 30 seconds.", err = True)
        raise typer.Exit(1)

    # ── 4. Create API key in-process ──────────────────────────────────
    api_key = _create_api_key_inprocess(api_key_name)

    # ── 5. Load model via HTTP ────────────────────────────────────────
    if not silent:
        typer.echo(f"Loading model: {model}...")
    try:
        result = _load_model_via_http(
            port = actual_port,
            api_key = api_key,
            model = model,
            gguf_variant = gguf_variant,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(1)

    loaded_model = result.get("model", model)
    display_variant = f" ({gguf_variant})" if gguf_variant else ""

    # ── 6. Print banner ───────────────────────────────────────────────
    display_host = _resolve_external_ip() if host == "0.0.0.0" else host
    base_url = f"http://{display_host}:{actual_port}"
    sdk_base_url = f"{base_url}/v1"

    if not silent:
        typer.echo("")
        typer.echo("=" * 56)
        typer.echo(f"  Unsloth Studio running at {base_url}")
        typer.echo(f"  Model loaded: {loaded_model}{display_variant}")
        typer.echo(f"  API Key:      {api_key}")
        typer.echo("")
        typer.echo("  OpenAI / Anthropic SDK base URL:")
        typer.echo(f"    {sdk_base_url}")
        typer.echo("=" * 56)
        typer.echo("")
        typer.echo("OpenAI Chat Completions:")
        typer.echo(f"  curl {sdk_base_url}/chat/completions \\")
        typer.echo(f'    -H "Authorization: Bearer {api_key}" \\')
        typer.echo('    -H "Content-Type: application/json" \\')
        typer.echo(
            """    -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'"""
        )
        typer.echo("")
        typer.echo("Anthropic Messages:")
        typer.echo(f"  curl {sdk_base_url}/messages \\")
        typer.echo(f'    -H "Authorization: Bearer {api_key}" \\')
        typer.echo('    -H "Content-Type: application/json" \\')
        typer.echo(
            """    -d '{"max_tokens": 256, "messages": [{"role": "user", "content": "Hello"}], "stream": true}'"""
        )
        typer.echo("")
        typer.echo("OpenAI Responses:")
        typer.echo(f"  curl {sdk_base_url}/responses \\")
        typer.echo(f'    -H "Authorization: Bearer {api_key}" \\')
        typer.echo('    -H "Content-Type: application/json" \\')
        typer.echo("""    -d '{"input": "Hello", "stream": true}'""")
        typer.echo("")

    # ── 7. Wait for Ctrl+C ────────────────────────────────────────────
    from studio.backend.run import _shutdown_event, _graceful_shutdown, _server

    try:
        if _shutdown_event is not None:
            while not _shutdown_event.is_set():
                _shutdown_event.wait(timeout = 1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
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
        powershell_args = ["powershell.exe"]
        if _should_hide_windows_subprocesses():
            powershell_args.extend(
                ["-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden"]
            )
        powershell_args.extend(["-ExecutionPolicy", "Bypass", "-File", str(script)])
        result = subprocess.run(
            powershell_args,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
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
    """Run Studio setup (called by install.ps1 / install.sh)."""
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
    # Ensure SKIP_STUDIO_BASE is not inherited from a parent install.ps1 session
    os.environ.pop("SKIP_STUDIO_BASE", None)
    os.environ["STUDIO_PACKAGE_NAME"] = package
    if local:
        os.environ["STUDIO_LOCAL_INSTALL"] = "1"
        # Pass the repo root explicitly so install_python_stack.py doesn't
        # have to guess from SCRIPT_DIR (which may be inside site-packages).
        repo_root = Path(__file__).resolve().parents[2]
        os.environ["STUDIO_LOCAL_REPO"] = str(repo_root)
    else:
        os.environ["STUDIO_LOCAL_INSTALL"] = "0"
        os.environ.pop("STUDIO_LOCAL_REPO", None)
    _run_setup_script(verbose = verbose)


# ── unsloth studio reset-password ────────────────────────────────────


@studio_app.command("desktop-capabilities", hidden = True)
def desktop_capabilities(
    json_output: bool = typer.Option(
        False,
        "--json",
        help = "Emit machine-readable JSON.",
    ),
):
    payload = {
        "desktop_protocol_version": 1,
        "supports_provision_desktop_auth": True,
        "supports_api_only": True,
        "version": "unknown",
    }
    try:
        from importlib.metadata import version as package_version

        payload["version"] = package_version("unsloth")
    except Exception:
        pass

    if json_output:
        typer.echo(json.dumps(payload, sort_keys = True))
        return

    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@studio_app.command("provision-desktop-auth", hidden = True)
def provision_desktop_auth():
    """Create/repair desktop auth state for the local machine."""
    auth_dir = STUDIO_HOME / "auth"
    secret = _create_desktop_secret_in_cli()
    _write_auth_secret(auth_dir / DESKTOP_SECRET_FILE, secret)
    typer.echo("Desktop auth ready.")


@studio_app.command("reset-password")
def reset_password():
    """Reset the Studio admin password.

    Deletes the auth database so that a fresh admin account with a new
    random password is created on the next server start.  The Studio
    server must be restarted after running this command.
    """
    auth_dir = STUDIO_HOME / "auth"
    db_file = auth_dir / "auth.db"
    stale_files = [
        auth_dir / BOOTSTRAP_PASSWORD_FILE,
        auth_dir / DESKTOP_SECRET_FILE,
    ]
    had_db = db_file.exists()

    db_file.unlink(missing_ok = True)
    for path in stale_files:
        path.unlink(missing_ok = True)

    if not had_db:
        typer.echo("No auth database found -- nothing to reset.")
        raise typer.Exit(0)

    typer.echo("Auth database deleted. Restart Unsloth Studio to get a new password.")

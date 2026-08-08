# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import importlib.util
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Sequence
import typer

from unsloth_cli import _studio_deps, _studio_runtime_gate
from unsloth_cli._inference import SpeculativeType
from unsloth_cli.commands import _password_prompt

studio_app = typer.Typer(help = "Unsloth Studio commands.")


def _enable_verbose_access_logs() -> None:
    """Restore every per-request access log by disabling the burst dedup and the
    quiet-poll heartbeat. Inherited by the spawned/re-exec'd server via the env."""
    os.environ["UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"] = "0"
    os.environ["UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"] = "0"


# Resolve install root: UNSLOTH_STUDIO_HOME, then STUDIO_HOME alias, then
# sys.prefix inference (so a direct call to <root>/bin/unsloth resolves after
# the installer's env var has expired), then legacy ~/.unsloth/studio.
# UNSLOTH_STUDIO_HOME wins when both env vars are set.
def _looks_like_installer_managed_studio_home(candidate: Path) -> bool:
    """Sentinel check (studio.conf or bin shim) so a dev venv named
    unsloth_studio is not misidentified as a custom Unsloth root.
    """
    shim_name = "unsloth.exe" if platform.system() == "Windows" else "unsloth"
    return (candidate / "share" / "studio.conf").is_file() or (
        candidate / "bin" / shim_name
    ).is_file()


def _resolve_studio_home() -> tuple[Path, bool]:
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        try:
            return Path(override).expanduser().resolve(), True
        except (OSError, ValueError):
            return Path(override).expanduser(), True
    try:
        prefix = Path(sys.prefix).resolve()
        if prefix.name == "unsloth_studio":
            inferred = prefix.parent
            legacy = (Path.home() / ".unsloth" / "studio").resolve()
            if inferred != legacy and _looks_like_installer_managed_studio_home(inferred):
                return inferred, True
    except (OSError, ValueError):
        pass
    return Path.home() / ".unsloth" / "studio", False


STUDIO_HOME, _STUDIO_HOME_IS_CUSTOM = _resolve_studio_home()


def _ensure_studio_env_exported() -> None:
    """Re-export UNSLOTH_STUDIO_HOME / UNSLOTH_LLAMA_CPP_PATH only for real
    custom roots so subprocesses inherit the right install. Called from each
    studio subcommand entry rather than at import time, to avoid leaking env
    state into unrelated importers (tests, --help, CLI introspection).
    """
    if not _STUDIO_HOME_IS_CUSTOM:
        return
    # Truthy-check (not setdefault) so a blank UNSLOTH_STUDIO_HOME= does not
    # suppress the inferred custom root.
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(STUDIO_HOME)
    # When override == legacy default, llama.cpp stays at ~/.unsloth/llama.cpp.
    try:
        _legacy_studio = (Path.home() / ".unsloth" / "studio").resolve()
        _is_legacy = STUDIO_HOME.resolve() == _legacy_studio
    except (OSError, ValueError):
        _is_legacy = STUDIO_HOME == (Path.home() / ".unsloth" / "studio")
    if _is_legacy:
        _llama_dir = Path.home() / ".unsloth" / "llama.cpp"
    else:
        _llama_dir = STUDIO_HOME / "llama.cpp"
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_llama_dir)


BOOTSTRAP_PASSWORD_FILE = ".bootstrap_password"
DESKTOP_SECRET_FILE = ".desktop_secret"
DEFAULT_ADMIN_USERNAME = "unsloth"
DESKTOP_SECRET_PREFIX = "desktop-"
API_KEY_PBKDF2_SALT_KEY = "api_key_pbkdf2_salt"
DESKTOP_SECRET_HASH_KEY = "desktop_secret_hash"
DESKTOP_SECRET_CREATED_AT_KEY = "desktop_secret_created_at"
PBKDF2_ITERATIONS = 100_000
_START_API_KEY_MARKER_ENV = "_UNSLOTH_START_API_KEY_MARKER"
_CLOUDFLARE_INTENT_ENV = "_UNSLOTH_CLOUDFLARE_INTENT"


def _consume_start_api_key_marker_env() -> bool:
    """Consume the one-shot readiness marker passed across a Studio re-exec."""
    return os.environ.pop(_START_API_KEY_MARKER_ENV, None) == "1"


def _preserve_cloudflare_intent(cloudflare: Optional[bool], secure: bool) -> None:
    """Carry the user's tri-state choice across compatibility re-execs."""
    if _CLOUDFLARE_INTENT_ENV in os.environ:
        return
    if secure or cloudflare is True:
        intent = "enabled"
    elif cloudflare is False:
        intent = "disabled"
    else:
        intent = "unset"
    os.environ[_CLOUDFLARE_INTENT_ENV] = intent


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


@contextlib.contextmanager
def _studio_runtime_launch_guard(*, inherited: bool = False):
    guard = _studio_runtime_gate.studio_runtime_launch_guard(
        STUDIO_HOME,
        inherited = inherited,
    )
    try:
        acquired = guard.__enter__()
    except _studio_runtime_gate.StudioRuntimeGateBusy:
        typer.echo(
            "Error: Unsloth installation is modifying the managed environment. "
            "Wait for it to finish, then try again.",
            err = True,
        )
        raise typer.Exit(1)
    except OSError as exc:
        typer.echo(f"Error: could not coordinate the Studio launch: {exc}", err = True)
        raise typer.Exit(1)

    try:
        yield acquired
    finally:
        guard.__exit__(None, None, None)


def _stream_for_subprocess(stream):
    """Return *stream* if it has a real OS file descriptor, else None.

    subprocess.run on Windows refuses to inherit std handles unless
    they're passed explicitly (otherwise close_fds=True forces
    bInheritHandles=False, and a CREATE_NO_WINDOW child ends up with
    no stdio at all). When sys.stdout / sys.stderr is a real fd-backed
    stream we want to hand it through; when it's been captured by a
    test harness (pytest's capsys, an in-memory wrapper, etc) we fall
    back to None so subprocess uses its default.
    """
    if stream is None:
        return None
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def _display_host_for_bind(run_mod, host: str) -> str:
    return run_mod._resolve_external_ip() if host in ("0.0.0.0", "::") else host


def _loopback_bind_host_for(host: str) -> str:
    return "::1" if host == "::" else "127.0.0.1"


def _url_host(host: str) -> str:
    return (
        f"[{host}]" if ":" in host and not (host.startswith("[") and host.endswith("]")) else host
    )


def _emit_run_cloudflare_notice(
    run_mod, host: str, display_host: str, actual_port: int, secure: bool
) -> None:
    from unsloth_cli._tool_policy import is_external_host

    if not is_external_host(host):
        return
    run_mod._verify_global_reachability(display_host, actual_port)
    run_mod._print_cloudflare_line(
        secure = secure,
        loopback_host = _loopback_bind_host_for(host),
    )


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
    # 2. Unsloth venv's site-packages (Linux + Windows layouts)
    for pattern in (
        "lib/python*/site-packages/studio/backend/run.py",
        "Lib/site-packages/studio/backend/run.py",
    ):
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


def _install_state() -> dict:
    """verify_install() result for this install root.

    STUDIO_HOME is an extra search root so a CLI installed outside the managed
    venv still inspects the venv the desktop app launches.
    """
    return _studio_deps.install_state(extra_roots = (STUDIO_HOME / "unsloth_studio",))


_RUN_MODULE = None


def _load_run_module():
    """Import studio.backend.run without relying on package resolution.

    `studio update` can leave a partial ``site-packages/studio/backend/``
    tree (plugin build artefacts only). That shadowed tree wins over an
    editable install and breaks ``from studio.backend.run import ...``.
    Loading by file path sidesteps the conflict.
    """
    global _RUN_MODULE
    if _RUN_MODULE is not None:
        return _RUN_MODULE

    run_py = _find_run_py()
    if run_py is None:
        raise ImportError("Could not find studio/backend/run.py. Re-run: unsloth studio setup")

    loaded = sys.modules.get("studio.backend.run")
    if loaded is not None:
        # __file__ can be None for namespace packages from partial trees.
        loaded_path = Path(getattr(loaded, "__file__", None) or "").resolve()
        if loaded_path == run_py.resolve():
            _RUN_MODULE = loaded
            return _RUN_MODULE

    spec = importlib.util.spec_from_file_location("studio.backend.run", run_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load studio backend from {run_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["studio.backend.run"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("studio.backend.run", None)
        raise
    _RUN_MODULE = module
    return _RUN_MODULE


def _find_setup_script(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Find studio/setup.sh or studio/setup.ps1.

    No CWD dependency — works from any directory.

    `repo_root` is the explicit --local checkout, when there is one. Its setup
    script has to win: the scripts build the frontend under their own
    $SCRIPT_DIR, and the editable install of `repo_root` removes the installed
    tree that the installed copy's script would have built into. studio/frontend
    /dist is gitignored, so a fresh checkout would then have no frontend at all.
    """
    name = "setup.ps1" if platform.system() == "Windows" else "setup.sh"
    # 0. The checkout the caller actually asked to install from. No fallback:
    #    dropping back to the installed copy's script is the exact behaviour
    #    this branch exists to stop, so a checkout without one is unusable
    #    rather than a reason to use somebody else's.
    if repo_root is not None:
        s = repo_root / "studio" / name
        return s if s.is_file() else None
    # 1. Relative to __file__ (site-packages or editable repo root)
    s = _PACKAGE_ROOT / "studio" / name
    if s.is_file():
        return s
    # 2. Unsloth venv's site-packages
    for pattern in (
        f"lib/python*/site-packages/studio/{name}",
        f"Lib/site-packages/studio/{name}",
    ):
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


# Mirror in studio/backend/run.py argparse + backend denylist test;
# bumping the cap in one place only desyncs.
_PARALLEL_MIN = 1
_PARALLEL_MAX = 64
_PARALLEL_DEFAULT_RUN = 4  # pre-PR hardcoded for `unsloth studio run`
# New Chat leaves the previous conversation generating and the admission queue caps decodes at
# the slot count, so at 1 every extra chat queues. _slots_that_fit_on_gpu() may cut it back.
_PARALLEL_DEFAULT_PLAIN = 4


def _resolve_secure(secure: bool, not_secure: bool) -> bool:
    """Reconcile the deprecated --not-secure alias with --secure/--no-secure.

    Typer parses --secure and --not-secure as independent options, so the alias
    cannot lean on Click's last-wins ordering the way --secure/--no-secure do.
    Restore that ordering from argv: --not-secure only forces secure off when it
    is the last of the secure flags on the command line, matching the backend's
    BooleanOptionalAction.
    """
    if not not_secure:
        return secure
    last_secure = max(
        (i for i, a in enumerate(sys.argv) if a in ("--secure", "--no-secure")),
        default = -1,
    )
    last_not_secure = max(
        (i for i, a in enumerate(sys.argv) if a == "--not-secure"),
        default = -1,
    )
    return secure if last_secure > last_not_secure else False


def _iter_editable_studio_source_roots(venv_dir: Path):
    """Yield repo roots from setuptools `__editable___*_finder.py` files in
    *venv_dir*'s site-packages whose MAPPING includes a `studio` entry.

    Returns the parent dir of the mapped `studio` package (i.e. the repo
    root), so callers can append `/studio/...` to reach any subdir.
    """
    import ast
    import re

    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for finder in sp.glob("__editable___*_finder.py"):
                try:
                    src = finder.read_text(encoding = "utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                # Tolerate single- or multi-line dict literals; [^}]* still
                # rejects nested dicts, which the setuptools template never
                # emits for editable installs.
                m = re.search(r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S)
                if not m:
                    continue
                try:
                    mapping = ast.literal_eval(m.group(1))
                except (SyntaxError, ValueError):
                    continue
                # Defensive: literal_eval can return a set / list / None if the
                # matched literal is not a dict (regex captures `{...}`).
                if not isinstance(mapping, dict):
                    continue
                studio_pkg = mapping.get("studio")
                if studio_pkg:
                    yield Path(studio_pkg).parent


def _find_frontend_dist() -> Optional[Path]:
    """Locate a built `studio/frontend/dist` (containing index.html).

    Probes (in order): package-local default, installer venv site-packages,
    editable source roots referenced from the installer venv. Returns None
    if nothing servable is found, so callers can decide to error or proceed
    in `--api-only` mode.

    Fixes the silent 404 when another `unsloth` on PATH shadows the
    installer's binary and points `_PACKAGE_ROOT` at a site-packages copy
    that never received a vite build.
    """
    candidates: List[Path] = [_PACKAGE_ROOT / "studio" / "frontend" / "dist"]
    venv_dir = STUDIO_HOME / "unsloth_studio"
    for pattern in (
        "lib/python*/site-packages/studio/frontend/dist",
        "Lib/site-packages/studio/frontend/dist",
    ):
        candidates.extend(venv_dir.glob(pattern))
    for repo_root in _iter_editable_studio_source_roots(venv_dir):
        candidates.append(repo_root / "studio" / "frontend" / "dist")
    seen: set[Path] = set()
    for c in candidates:
        try:
            resolved = c.resolve()
        except OSError:
            resolved = c
        if resolved in seen:
            continue
        seen.add(resolved)
        if (c / "index.html").is_file():
            return c
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
    backend_dir = run_py.parent if run_py is not None else _PACKAGE_ROOT / "studio" / "backend"
    if backend_dir.is_dir() and str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    auth_dir = backend_dir / "auth"
    storage_py = auth_dir / "storage.py"
    loaded = sys.modules.get("auth.storage")
    if loaded is not None:
        # __file__ can be None for namespace packages from partial trees.
        loaded_path = Path(getattr(loaded, "__file__", None) or "").resolve()
        if loaded_path == storage_py.resolve():
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
        # newline pins LF: text mode writes CRLF on Windows, and `$(cat ...)`
        # strips the LF but leaves the CR glued to the credential.
        with os.fdopen(fd, "w", encoding = "utf-8", newline = "\n") as f:
            fd = -1
            # Newline so `cat` doesn't run it into the shell prompt; readers strip.
            f.write(secret + "\n")
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
    # A live server writes this DB while the CLI runs; the default lock wait is zero.
    conn.execute("PRAGMA busy_timeout=5000")
    # Mirror backend storage.get_connection: this path can create auth/ and
    # auth.db (the pre-exposure gate writes here first), and sqlite3.connect
    # makes the DB 0644 under a 022 umask. Keep both private.
    for _path, _mode in ((auth_dir, 0o700), (auth_dir / "auth.db", 0o600)):
        try:
            os.chmod(_path, _mode)
        except OSError:
            pass
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
            is_desktop INTEGER NOT NULL DEFAULT 0,
            secret_gen TEXT
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
    refresh_columns = {row[1] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
    if "is_desktop" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0")
    if "secret_gen" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN secret_gen TEXT")
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


def _should_prompt_password_change(
    *, cloudflare: Optional[bool], host: str, secure: bool, api_only: bool
) -> bool:
    """Whether this launch will expose Unsloth through the Cloudflare tunnel.

    CLI mirror of run.py's _cloudflare_tunnel_should_start, minus the Colab
    case (Colab launches never come through this CLI path). --secure implies
    the tunnel; --cloudflare only tunnels non-api-only wildcard binds.
    """
    if secure:
        return True
    if cloudflare is not True:
        return False
    return host in ("0.0.0.0", "::") and not api_only


def _prompt_streams_interactive() -> bool:
    """The prompt needs a real terminal for input and for the masked echo."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _bootstrap_deadline_active() -> bool:
    """Whether the backend's bootstrap shutdown deadline will arm.

    Mirror of studio/backend/auth/bootstrap_timeout.py bootstrap_timeout_seconds:
    unset/blank/malformed UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT falls back to the 1h
    default (a typo must not remove protection); 0 or negative disables it.
    """
    raw = os.environ.get("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "").strip()
    if not raw:
        return True
    try:
        return int(raw) > 0
    except ValueError:
        return True


def _generate_reset_password() -> str:
    """Readable 4-word passphrase; the user has to type this one back in."""
    try:
        import diceware
        return diceware.get_passphrase(
            options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
        )
    except Exception:
        return secrets.token_urlsafe(24)


def _cli_update_password(
    conn: sqlite3.Connection,
    username: str,
    new_password: str,
    *,
    revoke_api_keys: bool = False,
) -> None:
    """CLI mirror of backend update_password + change-password route effects.

    One transaction: rehash, rotate the JWT secret, clear must_change_password,
    revoke refresh tokens (PR #6651 finding), drop the desktop secret, and (for a
    reset) the API keys the old credential could have minted. File cleanup happens
    after commit; a failed unlink must not roll the change back.
    """
    password_salt, password_hash = _hash_password(new_password)
    with conn:
        conn.execute(
            """
            UPDATE auth_user
            SET password_salt = ?, password_hash = ?, jwt_secret = ?, must_change_password = 0
            WHERE username = ?
            """,
            (password_salt, password_hash, secrets.token_urlsafe(64), username),
        )
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.execute(
            "DELETE FROM app_secrets WHERE key IN (?, ?)",
            (DESKTOP_SECRET_HASH_KEY, DESKTOP_SECRET_CREATED_AT_KEY),
        )
        if revoke_api_keys:
            conn.execute("DELETE FROM api_keys")
    for stale in (BOOTSTRAP_PASSWORD_FILE, DESKTOP_SECRET_FILE):
        stale_path = STUDIO_HOME / "auth" / stale
        try:
            stale_path.unlink(missing_ok = True)
        except OSError as exc:
            # The hash is already committed, so a failed unlink must NOT roll the
            # change back. But a locked-yet-writable file (Windows AV, read-only
            # auth dir) must be truncated: otherwise its stale plaintext survives
            # and generate_bootstrap_password() would re-validate this revoked
            # credential if auth.db is ever recreated. Mirrors backend
            # clear_bootstrap_password().
            try:
                stale_path.write_text("", encoding = "utf-8")
                cleared = True
            except OSError:
                cleared = False
            if cleared:
                typer.echo(
                    f"Warning: could not remove stale {stale} file ({exc}); cleared its "
                    "contents so the old credential cannot be reused.",
                    err = True,
                )
            else:
                typer.echo(
                    f"Warning: could not remove or clear stale {stale} file ({exc}); the "
                    "old credential is still on disk. Remove it manually to prevent reuse "
                    "after a reset.",
                    err = True,
                )


def _apply_supplied_password_before_launch(supplied_password: "str | None") -> None:
    """Non-interactively set the INITIAL admin password (from --password /
    UNSLOTH_STUDIO_PASSWORD / stdin) before the server binds, while the account
    still has its auto-generated bootstrap password.

    Only ever sets the FIRST password: an already-set one is a hard error (an
    override would be an auth bypass on a public launch), and an invalid value
    fails closed. Runs in the parent before any re-exec so the secret never
    crosses to the child argv.
    """
    if not supplied_password:
        return
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        typer.echo(
            f"Error: --password could not open the Unsloth auth database ({exc}); not starting.",
            err = True,
        )
        raise typer.Exit(1)
    try:
        _ensure_cli_default_admin(conn)
        conn.commit()
        row = conn.execute(
            "SELECT password_salt, password_hash, must_change_password "
            "FROM auth_user WHERE username = ?",
            (DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        if not row:
            typer.echo(
                "Error: --password could not initialize the admin account; not starting.",
                err = True,
            )
            raise typer.Exit(1)
        if not row[2]:
            typer.echo(
                "Error: an Unsloth admin password is already set; --password only sets "
                "the initial password. Change it in the UI, or run `unsloth studio "
                "reset-password` for a new one.",
                err = True,
            )
            raise typer.Exit(1)
        password_salt, password_hash = row[0], row[1]

        def _is_current_password(candidate: str) -> bool:
            return hmac.compare_digest(
                _pbkdf2_hex(candidate, password_salt.encode("utf-8")), password_hash
            )

        problem = _password_prompt.validate_new_password(supplied_password, _is_current_password)
        if problem is not None:
            typer.echo(f"Error: {problem} Not starting.", err = True)
            raise typer.Exit(1)
        _cli_update_password(conn, DEFAULT_ADMIN_USERNAME, supplied_password)
        typer.echo(f"Password updated for '{DEFAULT_ADMIN_USERNAME}'.", err = True)
    except (OSError, sqlite3.Error) as exc:
        # Any DB failure fails closed (typer.Exit is not caught here, so the
        # deliberate Exit(1) branches above propagate unchanged).
        typer.echo(
            f"Error: --password could not update the Unsloth auth database ({exc}); not starting.",
            err = True,
        )
        raise typer.Exit(1)
    finally:
        conn.close()


def _strip_seeded_bootstrap_password_or_exit(*, context: str) -> None:
    """Remove the seeded plaintext bootstrap password before a public re-exec.

    Version-independent protection: a re-exec'd child of ANY version (including an
    old studio-venv predating the pre-bind gate) then reads None instead of
    injecting the default credential into the public page. must_change_password
    stays set, so the login page still forces a change and the timer still arms.
    Removal IS the protection, so if it fails (locked file, read-only auth dir)
    fail closed rather than publish it.
    """
    bootstrap_file = STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE
    try:
        bootstrap_file.unlink(missing_ok = True)
    except OSError as exc:
        typer.echo(
            "Error: refusing to publish Unsloth on a public Cloudflare URL: "
            f"could not remove the seeded bootstrap password file ({exc}), so an "
            f"older Unsloth child could still serve the default credential ({context}). "
            "Delete it manually or change the admin password (run `unsloth studio` "
            "locally with a terminal attached, or `unsloth studio reset-password`), "
            "then retry.",
            err = True,
        )
        raise typer.Exit(1)


def _require_servable_frontend_or_exit(
    *, frontend: Optional[Path], api_only: bool, cloudflare: Optional[bool], host: str, secure: bool
) -> Optional[Path]:
    """Fail closed BEFORE the pre-exposure gate if a public UI launch has no
    login page to change the seeded password.

    The gate strips the seeded .bootstrap_password on a headless public launch,
    so if the child then cannot serve the login page the admin is locked out
    (must_change_password=1, no file, no UI) until `unsloth studio reset-password`.
    The login page is the ONLY in-band way to change the seeded password, so a
    public non-api-only launch must have a servable dist before the strip.

    Returns the dist to serve: a user-supplied --frontend (validated to contain
    index.html) or the auto-resolved built dist. Returns `frontend` unchanged for
    non-public or --api-only launches (no login page needed).
    """
    if api_only or not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return frontend
    if frontend is not None:
        # A user-supplied dist is not vetted by _find_frontend_dist, so verify it
        # can serve the login page; else `--frontend /bad/path` bypasses the guard.
        if (Path(frontend) / "index.html").is_file():
            return frontend
        typer.echo(
            "Error: --frontend points at a directory with no index.html, so a "
            "public Unsloth launch would have no login page to change the seeded "
            "admin password. Point --frontend at a built dist, rebuild it (re-run "
            "install.sh), or use --api-only.",
            err = True,
        )
        raise typer.Exit(1)
    # _find_frontend_dist only returns a path that already contains index.html.
    resolved = _find_frontend_dist()
    if resolved is not None:
        return resolved
    typer.echo(
        "Error: the Unsloth frontend is not built, so a public launch would have "
        "no login page to change the seeded admin password. Build it (re-run "
        "install.sh), pass --frontend PATH to a built dist, or use --api-only.",
        err = True,
    )
    raise typer.Exit(1)


def _validate_inproc_backend_before_strip(
    *, cloudflare: Optional[bool], host: str, secure: bool, api_only: bool
) -> None:
    """In-venv (in-process) analogue of the re-exec launcher check.

    In-venv there is no re-exec, so the backend is imported in-process only AFTER
    the gate. On the headless public path the gate strips the seeded
    .bootstrap_password, so a broken venv that fails at import would leave
    must_change_password=1 with no password to log in. Import the backend up front
    on that path and exit cleanly if broken, before anything is stripped.
    Headless-only so an interactive prompt is not delayed behind the import.
    """
    if not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return
    if _prompt_streams_interactive():
        return
    try:
        _load_run_module()
    except Exception as exc:
        typer.echo(
            f"Error: the Unsloth backend could not be loaded ({exc}); refusing to "
            "expose Unsloth publicly before it is confirmed runnable. Re-run: "
            "unsloth studio setup",
            err = True,
        )
        raise typer.Exit(1)


def _tunnel_binary_confirmed_unavailable() -> bool:
    """True only if cloudflared is provably unavailable (found nowhere on PATH or
    in the Unsloth cache AND the download failed), so the tunnel cannot start.

    Used on the --secure path (loopback bind, so the tunnel is the ONLY public
    exposure) to skip stripping the seeded recovery password before a public URL
    that will never come up. Loads the stdlib-only cloudflare_tunnel helper by
    file path so the check runs in the parent, before the strip.

    Returns False on ANY uncertainty: a possible credential leak outweighs a
    recoverable lockout, so the caller keeps the strip unless the tunnel is
    provably dead.
    """
    run_py = _find_run_py()
    if run_py is None:
        return False
    backend_dir = run_py.parent
    tunnel_py = backend_dir / "cloudflare_tunnel.py"
    if not tunnel_py.is_file():
        return False
    # ensure_cloudflared() lazily imports utils.paths.storage_roots to resolve the
    # Unsloth bin cache. The outer CLI hasn't added studio/backend to sys.path yet,
    # so that import would fail and return None (a false "unavailable" that wrongly
    # refuses --secure). Add the backend dir so the cache path resolves as in the child.
    added_backend_path = False
    try:
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
            added_backend_path = True
        spec = importlib.util.spec_from_file_location("studio.backend.cloudflare_tunnel", tunnel_py)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ensure_cloudflared() is None
    except Exception:
        return False
    finally:
        if added_backend_path:
            try:
                sys.path.remove(str(backend_dir))
            except ValueError:
                pass


def _child_self_suppresses(*, in_studio_venv: bool, child_run_py: Optional[Path]) -> bool:
    """True when the child that will serve Unsloth is provably THIS install's
    backend, whose pre-bind gate sets app.state.suppress_bootstrap_injection and
    so never serves the seeded credential publicly -- even with .bootstrap_password
    on disk. The parent-side strip is then unnecessary and can be skipped to avoid
    a lockout if the tunnel never comes up, keeping the file for LOCAL recovery.

    True iff we run in-process here, or the re-exec target is the outer install's
    own run.py (identity match). False on ANY doubt -- a studio-venv console script
    or a venv run.py that may predate the gate -- so the strip stays in force
    wherever an old child is possible.
    """
    if in_studio_venv:
        return True
    if child_run_py is None:
        return False
    try:
        outer_run_py = (_PACKAGE_ROOT / "studio" / "backend" / "run.py").resolve()
        return child_run_py.resolve() == outer_run_py
    except OSError:
        return False


def _enforce_password_change_before_exposure(
    *,
    cloudflare: Optional[bool],
    host: str,
    secure: bool,
    api_only: bool,
    child_self_suppresses: bool = False,
) -> None:
    """Force a terminal password change before the first public (tunnel) exposure.

    When the launch will start the tunnel and the admin still has its
    auto-generated bootstrap password, ask for a new one in the terminal (masked,
    confirmed) before any server or tunnel exists. Committing here, in the parent,
    keeps the password off argv/env and an older studio-venv child sees it
    immediately. Without a terminal, warn and fall back to the bootstrap shutdown
    timer (~1h, UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT).
    """
    if not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return
    # Before public exposure we must PROVE the admin password is no longer the
    # seeded default. If we cannot (auth DB won't open, or a fresh admin cannot be
    # seeded + committed below), an old studio-venv child could regenerate a fresh
    # bootstrap credential and serve it; stripping a file we can't vouch for cannot
    # stop a regeneration. So those cases fail closed, as does a failure after the
    # user typed a new password.
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        # Cannot open the auth DB, so cannot confirm a committed admin exists.
        # Refuse rather than risk a child serving the default login; a transient
        # lock clears on retry.
        typer.echo(
            "Error: refusing to publish Unsloth on a public Cloudflare URL: could "
            f"not open the Unsloth auth database ({exc}) to confirm the admin "
            "password was changed. Retry (a transient database lock clears), or "
            "change the password first (run `unsloth studio` locally with a "
            "terminal attached, or `unsloth studio reset-password`).",
            err = True,
        )
        raise typer.Exit(1)
    try:
        try:
            _ensure_cli_default_admin(conn)
            # Persist a freshly seeded admin before we might re-exec: the INSERT is
            # otherwise uncommitted and rolls back on conn.close(). If the seed or
            # commit fails, no admin is committed, so a re-exec'd OLD child finds
            # none, regenerates a fresh bootstrap password + file, and serves THAT
            # -- stripping cannot stop a regeneration. Can't prove a committed
            # admin, so fail closed.
            conn.commit()
        except (OSError, sqlite3.Error) as exc:
            # Best-effort remove any half-written seed file (its row rolled back);
            # the launch is refused regardless.
            try:
                (STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE).unlink(missing_ok = True)
            except OSError:
                pass
            typer.echo(
                "Error: refusing to publish Unsloth on a public Cloudflare URL: could "
                f"not initialize the admin account ({exc}), so a re-exec'd Unsloth "
                "child could regenerate and serve a default credential. Retry (a "
                "transient database lock clears), or change the password first (run "
                "`unsloth studio` locally with a terminal attached, or `unsloth "
                "studio reset-password`).",
                err = True,
            )
            raise typer.Exit(1)
        try:
            row = conn.execute(
                "SELECT password_salt, password_hash, must_change_password "
                "FROM auth_user WHERE username = ?",
                (DEFAULT_ADMIN_USERNAME,),
            ).fetchone()
        except (OSError, sqlite3.Error) as exc:
            if child_self_suppresses:
                # Could not read must_change back, but the child is this install's
                # own backend and suppresses the injection, so nothing serves the
                # seeded credential; proceed without stripping.
                return
            # The admin is committed above, so an old child finds it and won't
            # regenerate; we just couldn't read must_change back. Strip the seeded
            # file so nothing serves it, failing closed if the strip itself fails.
            typer.echo(
                f"Warning: could not read the Unsloth admin state back ({exc}); "
                "removing the seeded bootstrap password before public exposure.",
                err = True,
            )
            _strip_seeded_bootstrap_password_or_exit(context = "auth DB row unreadable")
            return
        if not row or not row[2]:
            return
        if not _prompt_streams_interactive():
            # Only proceed headless if the bootstrap shutdown deadline will protect
            # the launch: it never arms for api-only, and TIMEOUT=0 disables it.
            if api_only or not _bootstrap_deadline_active():
                typer.echo(
                    "Error: refusing to publish Unsloth on a public Cloudflare "
                    "URL: the default admin password was never changed, no "
                    "terminal is attached to change it here, and the bootstrap "
                    "shutdown deadline does not apply to this launch (api-only, "
                    "or UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0). Change the "
                    "password first (run `unsloth studio` locally and log in, "
                    "or re-run with a terminal attached), then retry.",
                    err = True,
                )
                raise typer.Exit(1)
            if child_self_suppresses:
                # The child is this install's own backend, whose pre-bind gate sets
                # app.state.suppress_bootstrap_injection, so the seeded credential
                # is never served publicly even with the file on disk. Skip the
                # strip: unnecessary here, and it would lock the user out if the
                # tunnel never comes up (e.g. a --secure loopback whose tunnel
                # fails). Keep the file for LOCAL recovery; must_change stays set
                # and the deadline arms.
                typer.echo(
                    "Warning: Unsloth is being exposed publicly while the admin "
                    "account still uses its auto-generated bootstrap password. The "
                    "login page forces a change and the credential is never served "
                    "on the public page. Set a new password by running `unsloth "
                    "studio` locally with a terminal attached, or `unsloth studio "
                    "reset-password`; Unsloth shuts down after ~1h if the password "
                    "stays unchanged (UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT).",
                    err = True,
                )
                return
            # The strip permanently removes the only plaintext recovery credential.
            # On --secure the bind is loopback, so the tunnel is the ONLY public
            # exposure: if cloudflared is provably unavailable no public URL can
            # start, so stripping would just lock the user out. Refuse with the
            # credential preserved. (A wildcard --cloudflare bind is public
            # regardless of the tunnel, so it still strips below, as does any
            # uncertainty.)
            if secure and _tunnel_binary_confirmed_unavailable():
                typer.echo(
                    "Error: refusing to expose Unsloth: the Cloudflare tunnel binary "
                    "(cloudflared) is unavailable and could not be downloaded, so no "
                    "public URL can start. The seeded bootstrap password is preserved "
                    "for recovery; fix connectivity and retry, or change the password "
                    "first (`unsloth studio` locally, or `unsloth studio "
                    "reset-password`).",
                    err = True,
                )
                raise typer.Exit(1)
            # Mixed-version safety: an OLD studio-venv child (predating this gate)
            # has no pre-bind suppression and would read the seeded credential back
            # from disk and inject it into the public HTML until the deadline.
            # Delete the file here, in the parent, so a fresh child of ANY version
            # reads None. must_change_password stays set, so the login page still
            # forces a change and the timer still arms; only the on-disk copy goes.
            _strip_seeded_bootstrap_password_or_exit(context = "no terminal to change it")
            typer.echo(
                "Warning: Unsloth is being exposed publicly while the admin account "
                "still uses its auto-generated bootstrap password. The seeded password "
                "file has been removed so it is not served on the public page. Set a new "
                "password by running `unsloth studio` locally with a terminal attached, "
                "or `unsloth studio reset-password`; Unsloth shuts down after ~1h if the "
                "password stays unchanged (UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT).",
                err = True,
            )
            return
        password_salt, password_hash = row[0], row[1]

        def _is_current_password(candidate: str) -> bool:
            return hmac.compare_digest(
                _pbkdf2_hex(candidate, password_salt.encode("utf-8")), password_hash
            )

        typer.echo(
            "Unsloth Studio will be exposed on the public internet, so set a "
            "password now. Ctrl+C to abort.",
            err = True,
        )
        try:
            new_password = _password_prompt.prompt_new_password(_is_current_password)
        except (KeyboardInterrupt, EOFError):
            typer.echo(
                "\nError: password change aborted; refusing to expose Unsloth "
                "with the default admin password. Re-run and set a password, "
                "or launch without --secure/--cloudflare.",
                err = True,
            )
            raise typer.Exit(1)
        _cli_update_password(conn, DEFAULT_ADMIN_USERNAME, new_password)
        typer.echo(f"Password updated for '{DEFAULT_ADMIN_USERNAME}'.", err = True)
    finally:
        conn.close()


def _load_model_via_http(
    port: int,
    api_key: str,
    model: str,
    gguf_variant: Optional[str],
    max_seq_length: int,
    load_in_4bit: bool,
    gpu_memory_mode: Literal["auto", "manual"] = "auto",
    tensor_parallel: bool = False,
    speculative_type: Optional[SpeculativeType] = None,
    spec_draft_n_max: Optional[int] = None,
    llama_extra_args: Optional[List[str]] = None,
    timeout: int = 600,
) -> dict:
    """POST to ``/api/inference/load`` using the API key for auth."""
    import json
    import urllib.request
    import urllib.error

    from unsloth_cli._inference import raise_for_deferred_error, require_completed_padded_body

    payload: dict = {
        "model_path": model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
    }
    if gguf_variant:
        payload["gguf_variant"] = gguf_variant
    if gpu_memory_mode == "manual":
        payload["gpu_memory_mode"] = "manual"
        payload["gpu_layers"] = -1
    if tensor_parallel:
        payload["tensor_parallel"] = True
    if speculative_type is not None:
        payload["speculative_type"] = speculative_type
    if spec_draft_n_max is not None:
        payload["spec_draft_n_max"] = spec_draft_n_max
    if llama_extra_args:
        payload["llama_extra_args"] = list(llama_extra_args)

    data = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{port}/api/inference/load"
    req = urllib.request.Request(
        url,
        data = data,
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            try:
                body = json.loads(resp.read())
            except ValueError:
                body = None  # truncated padded reply; rejected below
        # A slow load commits its 200 before it finishes and pads the body, so a late
        # failure arrives in-band; raise it as the HTTPError this function already turns
        # into the RuntimeError the caller reports. A truncated body is no report at all.
        return require_completed_padded_body(url, raise_for_deferred_error(url, body))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors = "replace")
        raise RuntimeError(f"Model load failed (HTTP {exc.code}): {body}") from exc


def _format_context_length_line(load_result: dict) -> Optional[str]:
    value = load_result.get("context_length")
    if isinstance(value, bool):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    if value_int <= 0:
        return None
    return f"  Context length: {value_int} tokens"


# ── unsloth studio (server) ──────────────────────────────────────────


@studio_app.callback(invoke_without_command = True)
def studio_default(
    ctx: typer.Context,
    port: int = typer.Option(8888, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
    api_only: bool = typer.Option(
        False,
        "--api-only",
        help = "Run API server only, no frontend serving (for Tauri desktop app)",
    ),
    parallel: int = typer.Option(
        _PARALLEL_DEFAULT_PLAIN,
        "--parallel",
        "--n-parallel",
        min = _PARALLEL_MIN,
        max = _PARALLEL_MAX,
        help = (
            f"llama-server parallel decode slots ({_PARALLEL_MIN}..{_PARALLEL_MAX}). "
            f"Default {_PARALLEL_DEFAULT_PLAIN}. The Studio run settings "
            "(Parallel Slots) override it per load."
        ),
    ),
    cloudflare: Optional[bool] = typer.Option(
        None,
        "--cloudflare/--no-cloudflare",
        help = "Expose Unsloth on a PUBLIC internet URL via a free Cloudflare HTTPS "
        "tunnel, for non-api-only wildcard binds (0.0.0.0 or ::). Off by default; "
        "pass --cloudflare to enable it (--secure implies it). --no-cloudflare forces "
        "it off but does not change a raw wildcard bind.",
    ),
    secure: bool = typer.Option(
        False,
        "--secure/--no-secure",
        help = "Expose ONLY a Cloudflare HTTPS link: bind localhost and fail closed "
        "if the tunnel can't start. Without it, --no-secure also serves the raw "
        "0.0.0.0 port, which is reachable from anywhere on the network.",
    ),
    not_secure: bool = typer.Option(
        False,
        "--not-secure",
        hidden = True,
        help = "Deprecated alias for --no-secure.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Log every API request, including the high-frequency polling that is "
        "deduplicated by default.",
    ),
    enable_tools: Optional[bool] = typer.Option(
        None,
        "--enable-tools/--disable-tools",
        help = "Force server-side tools (web search, code execution) on or off for "
        "every request. Default: on for every bind, with the per-chat UI toggle honored. "
        "/v1/messages takes the on direction per request (enable_tools) because it has no "
        "confirmation channel; the off direction still applies everywhere.",
    ),
    disable_dns_pinning: bool = typer.Option(
        False,
        "--disable-dns-pinning",
        help = "Allow hostname-based web fetches for enterprise proxies. WARNING: weakens "
        "DNS-rebinding protection; hostname and redirect validation remain enabled.",
    ),
    password: str = typer.Option(
        "",
        "--password",
        help = "Set the INITIAL admin password non-interactively (headless setups), "
        "only when none is set yet. Also reads the UNSLOTH_STUDIO_PASSWORD env var, or "
        "`--password -` to read one line from stdin. A literal value is visible in the "
        "process list and shell history. Rotate later with `unsloth studio reset-password`.",
    ),
):
    """Launch the Unsloth Studio server."""
    # Back-compat: --not-secure is a deprecated alias for --no-secure.
    secure = _resolve_secure(secure, not_secure)
    # Runs before every subcommand (run/setup/update/...).
    _ensure_studio_env_exported()
    if ctx.invoked_subcommand is not None:
        # Typer doesn't forward parent options to subcommands, so
        # `unsloth studio --parallel N run ...` would silently drop N.
        if parallel != _PARALLEL_DEFAULT_PLAIN:
            typer.echo(
                f"Error: --parallel on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put the flag after the "
                f"subcommand: `unsloth studio {ctx.invoked_subcommand} "
                f"--parallel {parallel} ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --cloudflare/--no-cloudflare: it would not reach the subcommand.
        if cloudflare is not None:
            _cf_flag = "--cloudflare" if cloudflare else "--no-cloudflare"
            typer.echo(
                f"Error: {_cf_flag} on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} {_cf_flag} ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --secure: it would not reach the subcommand.
        if secure:
            typer.echo(
                f"Error: --secure on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --secure ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --verbose: it would not reach the subcommand.
        if verbose:
            typer.echo(
                f"Error: --verbose on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --verbose ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --enable-tools/--disable-tools: it would not reach the subcommand.
        if enable_tools is not None:
            _tool_flag = "--enable-tools" if enable_tools else "--disable-tools"
            typer.echo(
                f"Error: {_tool_flag} on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} {_tool_flag} ...`",
                err = True,
            )
            raise typer.Exit(2)
        if disable_dns_pinning:
            typer.echo(
                "Error: --disable-dns-pinning on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio {ctx.invoked_subcommand}`, "
                f"put it after the subcommand: `unsloth studio {ctx.invoked_subcommand} "
                "--disable-dns-pinning ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --api-only: dropping it here would silently serve the UI.
        if api_only:
            typer.echo(
                f"Error: --api-only on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --api-only ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --password: it applies to the plain-server path only.
        if password:
            typer.echo(
                f"Error: --password on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --password ...`",
                err = True,
            )
            raise typer.Exit(2)
        return

    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    _preserve_cloudflare_intent(cloudflare, secure)

    # --secure requires the tunnel; force a loopback bind.
    if secure:
        if cloudflare is False:
            typer.echo(
                "Error: --secure requires the Cloudflare tunnel; do not combine it "
                "with --no-cloudflare.",
                err = True,
            )
            raise typer.Exit(2)
        if host not in ("127.0.0.1", "localhost", "::1"):
            typer.echo(
                "Note: --secure ignores -H (it binds loopback and serves only "
                "through the Cloudflare tunnel). Drop --secure to bind "
                f"{host} directly, or keep --secure for a tunnel-only public link.",
                err = True,
            )
        host = "127.0.0.1"

    # --verbose restores the per-request access logs that are suppressed by
    # default (plain-server path; the `run` subcommand has its own --verbose).
    if verbose:
        _enable_verbose_access_logs()
    if disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    # Use the studio venv if present and not already in it. Resolve the child
    # launcher BEFORE the gate: a headless gate strips the seeded
    # .bootstrap_password, so aborting afterward (venv/run.py missing) would leave
    # must_change_password=1 with no password to log in.
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))
    studio_python = run_py = None
    resolved_frontend = frontend
    if not in_studio_venv:
        studio_python = _studio_venv_python()
        run_py = _find_run_py()
        if not (studio_python and run_py):
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # A public UI launch must have a servable login page BEFORE the gate can
        # strip the seeded .bootstrap_password, or the child has no way to change
        # it. Also returns the resolved dist so the child serves a real build
        # regardless of where its __file__ lands (fixes the shadowed silent 404).
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = resolved_frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        # Non-public / api-only launches skip that validation but still forward an
        # explicitly resolved dist for the same silent-404 reason.
        if resolved_frontend is None and not api_only:
            resolved_frontend = _find_frontend_dist()
    else:
        # Already in the studio venv: no re-exec, served in-process below. On the
        # headless public path the gate strips the seeded .bootstrap_password, so
        # validate BOTH FIRST -- else a bad dist or broken venv fails only after
        # the strip (must_change_password=1, no password to log in). Frontend check
        # first (cheap); the backend import is headless-only so an interactive
        # prompt is not delayed behind it.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = resolved_frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        _validate_inproc_backend_before_strip(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )

    # A supplied --password / UNSLOTH_STUDIO_PASSWORD / stdin sets the initial
    # admin password here in the parent, before the gate and any re-exec, so the
    # secret never reaches the child argv; strip the env var so a re-exec'd child
    # can't re-read it. The interactive gate below then no-ops.
    _apply_supplied_password_before_launch(_password_prompt.resolve_supplied_password(password))
    os.environ.pop(_password_prompt.SUPPLIED_PASSWORD_ENV, None)

    # Public (tunnel) exposure with the seeded default password: force a terminal
    # password change first, before any re-exec or server exists. The child is
    # self-suppressing when we serve in-process or re-exec this install's own
    # run.py (its pre-bind gate suppresses the injection), so the gate can skip
    # the destructive strip.
    _enforce_password_change_before_exposure(
        cloudflare = cloudflare,
        host = host,
        secure = secure,
        api_only = api_only,
        child_self_suppresses = _child_self_suppresses(
            in_studio_venv = in_studio_venv, child_run_py = run_py
        ),
    )

    if not in_studio_venv:
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
                "--parallel",
                str(parallel),
            ]
            # Forward the frontend dist resolved before the gate (skipped in
            # --api-only, which serves no UI).
            if resolved_frontend is not None:
                args.extend(["--frontend", str(resolved_frontend)])
            if silent:
                args.append("--silent")
            if api_only:
                args.append("--api-only")
            # Forward polarity explicitly: _find_run_py can fall back to an older
            # run.py (--cloudflare defaulted on), so an unset default must not let a
            # mixed install silently re-enable the tunnel. --secure implies it, so
            # forward nothing then.
            if cloudflare is True:
                args.append("--cloudflare")
            elif not secure:
                args.append("--no-cloudflare")
            args.append("--secure" if secure else "--no-secure")
            # Forward an explicit tool policy; None -> run.py leaves it unset (tools on).
            if enable_tools is True:
                args.append("--enable-tools")
            elif enable_tools is False:
                args.append("--disable-tools")
            # On Windows os.execvp keeps the parent alive, so Ctrl+C
            # would orphan the child; use Popen+wait instead.
            if sys.platform == "win32":
                import subprocess as _sp

                # Hand our std handles to the child: without them CREATE_NO_WINDOW
                # gives the backend its own hidden console and `unsloth studio > log`
                # captures nothing -- the same trap noted at the setup.ps1 call below.
                # Omitting stdin does not withhold it (subprocess still fills it from
                # GetStdHandle); that would need stdin = DEVNULL.
                with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
                    proc = _sp.Popen(
                        args,
                        stdout = _stream_for_subprocess(sys.stdout),
                        stderr = _stream_for_subprocess(sys.stderr),
                        **_windows_hidden_subprocess_kwargs(),
                    )
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    # Child handles its own signal; let it finish.
                    rc = proc.wait()
                if rc != 0:
                    typer.echo(
                        f"\nError: Unsloth server exited unexpectedly (code {rc}).",
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
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)

    with _studio_deps.studio_backend_imports("unsloth studio"):
        run_mod = _load_run_module()
    run_server = run_mod.run_server

    if not silent:
        display_host = _display_host_for_bind(run_mod, host)
        typer.echo(f"Starting Unsloth Studio on http://{_url_host(display_host)}:{port}")

    run_kwargs = dict(
        host = host,
        port = port,
        silent = silent,
        api_only = api_only,
        llama_parallel_slots = parallel,
        cloudflare = cloudflare,
        secure = secure,
        enable_tools = enable_tools,
    )
    # Forward the frontend validated before the gate (in-venv path), so the
    # in-process server serves exactly the dist we vouched for.
    if resolved_frontend is not None:
        run_kwargs["frontend_path"] = resolved_frontend
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        run_server(**run_kwargs)

    try:
        if run_mod._shutdown_event is not None:
            # Event.wait() with no timeout blocks at C-level on Linux
            # and swallows SIGINT; loop with a 1s timeout instead.
            while not run_mod._shutdown_event.is_set():
                run_mod._shutdown_event.wait(timeout = 1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        run_mod._graceful_shutdown(run_mod._server)
        typer.echo("\nShutting down...")
    finally:
        getattr(run_mod, "_wait_for_server_shutdown", lambda: None)()


# ── unsloth studio run ───────────────────────────────────────────────


def _split_repo_variant(model_arg: str) -> tuple[str, Optional[str]]:
    """Split ``org/name:variant`` into ``(repo, variant)``; mirrors
    llama.cpp's ``-hf <repo>:<quant>``. Local paths, Windows drives,
    and ids without ``:`` pass through verbatim."""
    s = model_arg.strip()
    if not s:
        return s, None
    if s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    # Windows drive letter (e.g. "C:\path"): colon is a path separator.
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant:
        return s, None
    # Quant labels never contain a slash; `foo:bar/baz` isn't repo:variant.
    if "/" in variant:
        return s, None
    return repo, variant


def _expand_attached_np_short() -> None:
    # Click clusters `-np8` as `-n -p 8` (-p = --port), dropping the parallel
    # value. Split to `-np <N>` so typer's alias matches. Stops at `--`;
    # accepts signed/junk forms so typer reports a clean error against `-np`.
    # Kept in lockstep with the backend `_flag_name` recogniser.
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == "--":
            break
        if len(tok) > 3 and tok.startswith("-np") and tok[3] != "=":
            suffix = tok[3:]
            first_numeric = suffix[0].isdigit() or (
                len(suffix) > 1 and suffix[0] in {"-", "+"} and suffix[1].isdigit()
            )
            if first_numeric:
                sys.argv[i : i + 1] = ["-np", suffix]
                i += 2
                continue
        i += 1


def _consume_legacy_short_aliases(
    args: List[str], aliases: tuple[str, ...], current: Optional[str], canonical: str
) -> tuple[Optional[str], List[str]]:
    """Pop exact-match legacy shorts (`-m`/`-hfr`/`-f`) from args;
    leave clusters (`-mg`/`-fa`/...) for the llama-server tail. Inline
    `-x=value` form also accepted."""
    out: List[str] = []
    value = current
    i, n = 0, len(args)
    while i < n:
        tok = args[i]
        if tok == "--":  # end of options; tail is raw payload.
            out.extend(args[i:])
            break
        name, sep, inline = tok.partition("=")
        if name not in aliases:
            out.append(tok)
            i += 1
            continue
        if value is not None:
            raise typer.BadParameter(f"{name} conflicts with {canonical} already provided")
        if sep:
            if inline == "":  # `-m=` would become --model '' (Path('')='.').
                raise typer.BadParameter(f"{name} requires a non-empty value")
            value = inline
            i += 1
        elif i + 1 < n:
            nxt = args[i + 1]
            # `--long` is unambiguously a flag; single-dash `-x` may be a path.
            if nxt.startswith("--") and nxt != "--":
                raise typer.BadParameter(f"{name} expects a value but got the flag {nxt}")
            value = nxt
            i += 2
        else:
            raise typer.BadParameter(f"{name} requires a value")
    return value, out


# Help panels so `unsloth run --help` groups options instead of one long list.
_RUN_PANEL_MODEL = "Model"
_RUN_PANEL_SERVER = "Server & network"
_RUN_PANEL_TOOLS = "Tool calls"
_RUN_PANEL_SAMPLING = "Sampling"
_RUN_PANEL_ADVANCED = "Advanced"


@studio_app.command(
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-hf",
        "--hf-repo",
        # `-m` / `-hfr` removed (Click would cluster `-mg`/`-md`/...).
        # Exact-match `-m`/`-hfr` still work via the legacy shim below.
        # `-hf` stays (multi-char shorts don't cluster).
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "Model path or HF repo. Accepts llama.cpp-style "
            "`org/repo:variant` syntax. `-hf` / `--hf-repo` match "
            "llama-server's spelling."
        ),
    ),
    gguf_variant: Optional[str] = typer.Option(
        None,
        "--gguf-variant",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = "GGUF quant variant (e.g. UD-Q4_K_XL)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Log every API request, including the high-frequency polling that is "
        "deduplicated by default.",
    ),
    max_seq_length: int = typer.Option(
        0,
        "--max-seq-length",
        "--context-length",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = "Runtime context length in tokens (0 = model default for GGUF; 2048 for hub models)",
    ),
    gpu_memory_mode: Literal["auto", "manual"] = typer.Option(
        "auto",
        "--gpu-memory-mode",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "GPU memory strategy for GGUF models. Auto lets Unsloth select GPUs "
            "and cap context to fit VRAM. Manual with default layers and context "
            "delegates placement and sizing to llama.cpp --fit."
        ),
    ),
    speculative_type: Optional[SpeculativeType] = typer.Option(
        None,
        "--speculative-type",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "Speculative decoding mode for GGUF models. DSpark automatically uses a "
            "matching dspark-*.gguf sidecar when available. Default: unset (Studio auto)."
        ),
    ),
    spec_draft_n_max: Optional[int] = typer.Option(
        None,
        "--spec-draft-n-max",
        min = 1,
        max = 16,
        rich_help_panel = _RUN_PANEL_MODEL,
        help = "Maximum draft tokens per step for MTP or DSpark (1..16).",
    ),
    load_in_4bit: bool = typer.Option(
        True, "--load-in-4bit/--no-load-in-4bit", rich_help_panel = _RUN_PANEL_MODEL
    ),
    api_key_name: str = typer.Option(
        "cli",
        "--api-key-name",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Label for the auto-generated API key",
    ),
    port: int = typer.Option(8888, "--port", "-p", rich_help_panel = _RUN_PANEL_SERVER),
    host: str = typer.Option("127.0.0.1", "--host", "-H", rich_help_panel = _RUN_PANEL_SERVER),
    # `-f` removed (clustered `-fa`/`-fit*`); studio_default keeps it.
    frontend: Optional[Path] = typer.Option(None, "--frontend", rich_help_panel = _RUN_PANEL_SERVER),
    api_only: bool = typer.Option(
        False,
        "--api-only",
        rich_help_panel = _RUN_PANEL_SERVER,
        help = "Serve only the API (no web UI), for a headless model server. "
        "Pairs with --secure to expose the API over the Cloudflare link alone.",
    ),
    silent: bool = typer.Option(False, "--silent", "-q", rich_help_panel = _RUN_PANEL_ADVANCED),
    enable_tools: Optional[bool] = typer.Option(
        None,
        "--enable-tools/--disable-tools",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = (
            "Force server-side tools (web search, code execution) on or off for "
            "every request. Default: on for every bind. /v1/messages takes the on "
            "direction per request (enable_tools) because it has no confirmation "
            "channel; the off direction still applies everywhere."
        ),
    ),
    disable_dns_pinning: bool = typer.Option(
        False,
        "--disable-dns-pinning",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = "Allow hostname-based web fetches for enterprise proxies. WARNING: weakens "
        "DNS-rebinding protection; hostname and redirect validation remain enabled.",
    ),
    tool_call_healing: Optional[bool] = typer.Option(
        None,
        "--enable-tool-call-healing/--disable-tool-call-healing",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = (
            "Promote text-form tool calls (small GGUFs often emit <tool_call>...) "
            "back into structured calls on the client-tool passthrough. Default: on. "
            "An explicit --disable-tool-call-healing is an absolute server kill-switch."
        ),
    ),
    tool_call_nudging: Optional[bool] = typer.Option(
        None,
        "--enable-tool-call-nudging/--disable-tool-call-nudging",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = (
            "On the non-streaming client-tool passthrough, retry once with a short "
            "nudge when the model emitted a tool signal that healing could not repair. "
            "Default: on. No effect on streaming requests or the server-side agentic loop."
        ),
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        min = 0.0,
        max = 2.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = (
            "Pin the sampling temperature for every request that omits it, overriding the "
            "model's recommended value. Default: unset (use the per-model recommendation)."
        ),
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        min = 0.0,
        max = 1.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin top-p (nucleus) sampling. Default: unset (per-model recommendation).",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        min = -1,
        max = 100,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin top-k sampling. Default: unset (per-model recommendation).",
    ),
    min_p: Optional[float] = typer.Option(
        None,
        "--min-p",
        min = 0.0,
        max = 1.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin min-p sampling threshold. Default: unset (per-model recommendation).",
    ),
    repetition_penalty: Optional[float] = typer.Option(
        None,
        "--repetition-penalty",
        min = 1.0,
        max = 2.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin the repetition penalty. Default: unset (per-model recommendation).",
    ),
    presence_penalty: Optional[float] = typer.Option(
        None,
        "--presence-penalty",
        min = 0.0,
        max = 2.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin the presence penalty. Default: unset (per-model recommendation).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Accepted for backward compatibility; the tool policy no longer prompts.",
    ),
    parallel: int = typer.Option(
        _PARALLEL_DEFAULT_RUN,
        "--parallel",
        "--n-parallel",
        "-np",
        min = _PARALLEL_MIN,
        max = _PARALLEL_MAX,
        rich_help_panel = _RUN_PANEL_SERVER,
        help = (
            "llama-server parallel decode slots. N requests share one "
            "loaded model; each slot gets ctx/N KV cache. Default "
            f"{_PARALLEL_DEFAULT_RUN} (pre-PR hardcoded value). The Studio "
            "run settings (Parallel Slots) can override it per load."
        ),
    ),
    cloudflare: Optional[bool] = typer.Option(
        None,
        "--cloudflare/--no-cloudflare",
        rich_help_panel = _RUN_PANEL_SERVER,
        help = "Expose Unsloth on a PUBLIC internet URL via a free Cloudflare HTTPS "
        "tunnel, for non-api-only wildcard binds (0.0.0.0 or ::). Off by default; "
        "pass --cloudflare to enable it (--secure implies it). --no-cloudflare forces "
        "it off but does not change a raw wildcard bind.",
    ),
    secure: bool = typer.Option(
        False,
        "--secure/--no-secure",
        rich_help_panel = _RUN_PANEL_SERVER,
        help = "Expose ONLY a Cloudflare HTTPS link: bind localhost and fail closed "
        "if the tunnel can't start. Without it, --no-secure also serves the raw "
        "0.0.0.0 port, which is reachable from anywhere on the network.",
    ),
    not_secure: bool = typer.Option(
        False,
        "--not-secure",
        hidden = True,
        help = "Deprecated alias for --no-secure.",
    ),
    tensor_parallel: bool = typer.Option(
        False,
        "--tensor-parallel/--no-tensor-parallel",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "Split a GGUF across GPUs by tensor (--split-mode tensor) instead of "
            "by layer. Multi-GPU only (no effect on one GPU); dense models gain "
            "decode speed, MoE usually don't."
        ),
    ),
    start_api_key_marker: bool = typer.Option(
        False,
        "--start-api-key-marker",
        hidden = True,
        help = "Emit an early API key marker for the unsloth start parent process.",
    ),
    password: str = typer.Option(
        "",
        "--password",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Set the INITIAL admin password non-interactively (headless setups), "
        "only when none is set yet. Also reads the UNSLOTH_STUDIO_PASSWORD env var, or "
        "`--password -` to read one line from stdin. A literal value is visible in the "
        "process list and shell history. Rotate later with `unsloth studio reset-password`.",
    ),
):
    """Start Unsloth, load a model, print an API key -- one-liner server.

    Unknown flags pass through to llama-server (GGUF only). Unsloth
    rejects managed flags with HTTP 400: model identity, network
    (--host/--port/--path/--api-prefix/--reuse-port), auth/TLS
    (--api-key/--ssl-*), single-model UI (--ui/--models-*/--webui),
    and parallel slots (use --parallel above). Full denylist in
    studio/backend/core/inference/llama_server_args.py. Other knobs
    (-c, -ngl, --jinja, --flash-attn, -t, ...) pass through and
    last-wins-override Unsloth's auto-set value.

    Example:
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --gguf-variant UD-Q4_K_XL
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --temperature 0.7 --seed 42 --parallel 8
        unsloth studio run --model some-model --chat-template-file /path/to/tpl.jinja
        unsloth studio run --model unsloth/Qwen3-27B-GGUF --gguf-variant Q8_0 --tensor-parallel
    """
    # A newer outer CLI can re-exec into an older Studio venv; pass this signal via
    # env so an older child ignores it instead of treating it as a llama-server arg.
    inherited_start_api_key_marker = _consume_start_api_key_marker_env()
    start_api_key_marker = start_api_key_marker or inherited_start_api_key_marker
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()

    # Back-compat: --not-secure is a deprecated alias for --no-secure.
    secure = _resolve_secure(secure, not_secure)
    _preserve_cloudflare_intent(cloudflare, secure)
    extra_llama_args: List[str] = list(ctx.args) if ctx.args else []

    # Tool-call healing/nudging are read from the env at backend import. Resolve here
    # (before any re-exec/import) so the in-venv child inherits the decision. When the
    # flag is omitted, respect a value the parent already set (e.g. `unsloth start`
    # forwards its choice via the env) and otherwise apply the default: healing on,
    # nudging on for a CLI-launched server.
    _healing_disabled = (
        os.environ.get("UNSLOTH_DISABLE_TOOL_CALL_HEALING") == "1"
        if tool_call_healing is None
        else not tool_call_healing
    )
    os.environ["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] = "1" if _healing_disabled else "0"
    if tool_call_nudging is not None:
        os.environ["UNSLOTH_TOOL_CALL_NUDGE"] = "1" if tool_call_nudging else "0"
    elif "UNSLOTH_TOOL_CALL_NUDGE" not in os.environ:
        os.environ["UNSLOTH_TOOL_CALL_NUDGE"] = "1"

    # Sampling overrides: the backend resolver reads UNSLOTH_SAMPLING_* to hard-pin a field
    # (winning over both the client and the per-model recommendation). Only write a flag that
    # was set explicitly so an omitted flag inherits any value the parent forwarded (e.g.
    # `unsloth start`) and, when nothing is set, leaves the per-model recommendation in charge.
    for _sampling_env, _sampling_value in (
        ("UNSLOTH_SAMPLING_TEMPERATURE", temperature),
        ("UNSLOTH_SAMPLING_TOP_P", top_p),
        ("UNSLOTH_SAMPLING_TOP_K", top_k),
        ("UNSLOTH_SAMPLING_MIN_P", min_p),
        ("UNSLOTH_SAMPLING_REPETITION_PENALTY", repetition_penalty),
        ("UNSLOTH_SAMPLING_PRESENCE_PENALTY", presence_penalty),
    ):
        if _sampling_value is not None:
            os.environ[_sampling_env] = str(_sampling_value)

    # Set before any re-exec so the in-venv server inherits it via the env.
    # `run --verbose` used to pass through to llama-server (its own -v); keep
    # that by forwarding --log-verbose so we add Unsloth logs without dropping it.
    if verbose:
        _enable_verbose_access_logs()
        if not any(a in ("--verbose", "-v", "--log-verbose") for a in extra_llama_args):
            extra_llama_args.append("--log-verbose")
    if disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    # Promote legacy exact `-m`/`-hfr`/`-f` back into typer params;
    # clusters stay in extras.
    model, extra_llama_args = _consume_legacy_short_aliases(
        extra_llama_args,
        ("-m", "-hfr"),
        model,
        "--model",
    )
    legacy_frontend, extra_llama_args = _consume_legacy_short_aliases(
        extra_llama_args,
        ("-f",),
        str(frontend) if frontend is not None else None,
        "--frontend",
    )
    if legacy_frontend is not None and frontend is None:
        frontend = Path(legacy_frontend)

    if model is None:
        typer.echo(
            "Error: Missing option '--model' / '-hf' / '--hf-repo' "
            "(legacy aliases '-m' / '-hfr' are still accepted).",
            err = True,
        )
        raise typer.Exit(2)

    # 0. Parse llama.cpp `repo:variant` in --model; error if also paired
    # with --gguf-variant and they disagree.
    parsed_repo, embedded_variant = _split_repo_variant(model)
    if embedded_variant:
        if gguf_variant and gguf_variant != embedded_variant:
            typer.echo(
                f"Error: --model embeds variant '{embedded_variant}' but "
                f"--gguf-variant '{gguf_variant}' was also provided.",
                err = True,
            )
            raise typer.Exit(1)
        model = parsed_repo
        gguf_variant = gguf_variant or embedded_variant

    # --secure requires the tunnel; force a loopback bind so the raw port is never public.
    if secure:
        if cloudflare is False:
            typer.echo(
                "Error: --secure requires the Cloudflare tunnel; do not combine it "
                "with --no-cloudflare.",
                err = True,
            )
            raise typer.Exit(2)
        if host not in ("127.0.0.1", "localhost", "::1"):
            typer.echo(
                "Note: --secure ignores -H (it binds loopback and serves only "
                "through the Cloudflare tunnel). Drop --secure to bind "
                f"{host} directly, or keep --secure for a tunnel-only public link.",
                err = True,
            )
        host = "127.0.0.1"

    # Tool policy no longer depends on the bind: tools default on everywhere
    # (--secure is a loopback tunnel; the operator owns a raw bind). Resolve here
    # so the re-exec'd child inherits a concrete decision.
    from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy

    enable_tools = resolve_tool_policy(
        host = host,
        flag = enable_tools,
        yes = yes,
        silent = silent,
    )

    # 1. Re-exec into the studio venv (same pattern as studio_default). Resolve
    # the child launcher BEFORE the gate: a headless gate strips the seeded
    # .bootstrap_password, so aborting afterward (venv/entry point missing) would
    # leave must_change_password=1 with no password to log in.
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))
    studio_bin = None
    resolved_frontend = frontend
    if not in_studio_venv:
        studio_python = _studio_venv_python()
        if not studio_python:
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # Re-exec via the studio venv's `unsloth` console-script. Windows ships it as
        # unsloth.exe, so the bare name is never a file there and `unsloth run` aborted
        # with "venv missing 'unsloth' entry point" on a perfectly good install.
        studio_bin = studio_python.parent / (
            "unsloth.exe" if platform.system() == "Windows" else "unsloth"
        )
        if not studio_bin.is_file():
            typer.echo("Unsloth venv missing 'unsloth' entry point. Re-run: unsloth studio setup")
            raise typer.Exit(1)
        # `run` serves the same Unsloth UI (unless --api-only); a public launch must
        # have a servable login page BEFORE the gate strips the seeded password, or
        # the child has no way to change it. Validate here and forward the resolved
        # dist so a shadowed child that can't self-resolve one still serves it.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
    else:
        # In-venv (in-process) run: validate the servable frontend and importable
        # backend before the headless gate strips the seeded password. Frontend
        # check first (cheap); backend import is headless-only so a prompt isn't
        # delayed.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        _validate_inproc_backend_before_strip(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )

    # A supplied --password / UNSLOTH_STUDIO_PASSWORD / stdin sets the initial
    # admin password here in the parent, before the gate and any re-exec, so the
    # secret never reaches the child argv; strip the env var so a re-exec'd child
    # can't re-read it. The interactive gate below then no-ops.
    _apply_supplied_password_before_launch(_password_prompt.resolve_supplied_password(password))
    os.environ.pop(_password_prompt.SUPPLIED_PASSWORD_ENV, None)

    # Public (tunnel) exposure with the seeded default password: force a terminal
    # password change first, before any re-exec or server exists. The re-exec here
    # runs the studio venv's `unsloth` console script (a possibly-OLD child), so it
    # is NOT provably self-suppressing -- only the in-process case is, and the
    # strip stays in force otherwise.
    _enforce_password_change_before_exposure(
        cloudflare = cloudflare,
        host = host,
        secure = secure,
        api_only = api_only,
        child_self_suppresses = _child_self_suppresses(
            in_studio_venv = in_studio_venv, child_run_py = None
        ),
    )

    if not in_studio_venv:
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
        if gpu_memory_mode != "auto":
            args.extend(["--gpu-memory-mode", gpu_memory_mode])
        if gguf_variant:
            args.extend(["--gguf-variant", gguf_variant])
        if speculative_type is not None:
            args.extend(["--speculative-type", speculative_type])
        if spec_draft_n_max is not None:
            args.extend(["--spec-draft-n-max", str(spec_draft_n_max)])
        # Forward the explicit polarity; a future default flip on one
        # layer must not silently invert behaviour for the other.
        args.append("--load-in-4bit" if load_in_4bit else "--no-load-in-4bit")
        # Forward the frontend resolved before the gate, not just a user-supplied
        # one: the parent may have found a built dist the shadowed child cannot,
        # and stripping without forwarding it would abort the child at frontend
        # setup (lockout).
        if resolved_frontend is not None:
            args.extend(["--frontend", str(resolved_frontend)])
        if api_only:
            args.append("--api-only")
        if silent:
            args.append("--silent")
        # Forward the resolved tool policy so the child doesn't re-resolve.
        if enable_tools:
            args.append("--enable-tools")
        else:
            args.append("--disable-tools")
        # Forward --yes only if the user passed it; resolution no longer prompts.
        if yes:
            args.append("--yes")
        # Typer claims --parallel outside ctx.args; without this the
        # child reverts to its default and silently drops the value.
        args.extend(["--parallel", str(parallel)])
        # Always forward explicit polarity: a mixed-version studio venv whose old
        # default was --cloudflare-on must not silently re-enable the tunnel.
        # --secure implies it, so forward nothing then.
        if cloudflare is True:
            args.append("--cloudflare")
        elif not secure:
            args.append("--no-cloudflare")
        args.append("--secure" if secure else "--no-secure")
        args.append("--tensor-parallel" if tensor_parallel else "--no-tensor-parallel")
        if verbose:
            args.append("--verbose")
        # llama-server pass-through extras → child ctx.args → load payload.
        if extra_llama_args:
            args.extend(extra_llama_args)

        if start_api_key_marker:
            os.environ[_START_API_KEY_MARKER_ENV] = "1"
        try:
            if sys.platform == "win32":
                with _studio_runtime_launch_guard(inherited = runtime_gate_handoff) as gate_held:
                    popen_kwargs = {}
                    if gate_held:
                        popen_kwargs["env"] = _studio_runtime_gate.runtime_gate_child_environment()
                    proc = subprocess.Popen(args, **popen_kwargs)
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    rc = proc.wait()
                raise typer.Exit(rc)
            else:
                os.execvp(str(studio_bin), args)
        finally:
            # execvp doesn't return on success; restore env after a Windows wait or a failed launch.
            os.environ.pop(_START_API_KEY_MARKER_ENV, None)

    # ── 2. Start server (always suppress built-in banner) ─────────────
    with _studio_deps.studio_backend_imports("unsloth studio"):
        run_mod = _load_run_module()
    run_server = run_mod.run_server

    # Match the route handlers' import path: run.py adds studio/backend/ to
    # sys.path, so they import as `state.tool_policy`. Set this before
    # run_server() starts uvicorn; once sockets are bound, routes can be hit.
    from state.tool_policy import set_tool_policy

    set_tool_policy(enable_tools)

    run_kwargs = dict(
        host = host,
        port = port,
        silent = True,
        api_only = api_only,
        llama_parallel_slots = parallel,
        cloudflare = cloudflare,
        secure = secure,
        # Headless serving prints its own URL/API-key banner; the Tauri-only
        # TAURI_PORT line would corrupt that machine-parseable output.
        emit_tauri_port = False,
        # We read the bound port back below, so a fallback past another Studio is
        # safe here and keeps side-by-side model runs working.
        abort_if_own_studio = False,
    )
    # Forward the frontend validated before the gate (in-venv path).
    if resolved_frontend is not None:
        run_kwargs["frontend_path"] = resolved_frontend
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        app = run_server(**run_kwargs)
    actual_port = getattr(app.state, "server_port", port) or port

    # Steps 3-5 can abort (health timeout, model-load error, or Ctrl+C during the
    # slow load); tear the server and its children (llama-server, cloudflared) down
    # on any abort so they never orphan.
    from studio.backend.run import _graceful_shutdown, _server

    try:
        # 3. Wait for server health.
        if not silent:
            typer.echo("Starting Unsloth Studio...")
        if not _wait_for_server(actual_port):
            typer.echo("Error: server did not become healthy within 30 seconds.", err = True)
            raise typer.Exit(1)

        # 4. Create API key in-process.
        api_key = _create_api_key_inprocess(api_key_name)
        if start_api_key_marker:
            # `unsloth start` reads this key from a private 0600 log to authenticate
            # download-progress polling; the normal `unsloth run` output is unchanged.
            typer.echo(f"UNSLOTH_START_API_KEY: {api_key}")

        # 5. Load model via HTTP.
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
                gpu_memory_mode = gpu_memory_mode,
                tensor_parallel = tensor_parallel,
                speculative_type = speculative_type,
                spec_draft_n_max = spec_draft_n_max,
                llama_extra_args = extra_llama_args,
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err = True)
            raise typer.Exit(1)
    except BaseException:
        _graceful_shutdown(_server)
        getattr(run_mod, "_wait_for_server_shutdown", lambda: None)()
        raise

    loaded_model = result.get("model", model)
    display_variant = f" ({gguf_variant})" if gguf_variant else ""
    context_length_line = _format_context_length_line(result)

    # 6. Print banner.
    display_host = _display_host_for_bind(run_mod, host)
    base_url = f"http://{_url_host(display_host)}:{actual_port}"
    sdk_base_url = f"{base_url}/v1"
    # run_server started the tunnel during the silent run above (wildcard or --secure).
    _cf_url = getattr(app.state, "cloudflare_url", None)
    # --secure: examples must use the public tunnel URL, not the loopback address.
    if secure and _cf_url:
        sdk_base_url = f"{_cf_url}/v1"

    # Orange so the tool-policy notice stands out; printed under
    # --silent / --yes too so the policy is never invisible.
    _tool_notice_fg = (217, 119, 87)
    _is_external = is_external_host(host)
    if not enable_tools:
        _tool_notice = "Server-side tools are DISABLED (--disable-tools)."
    elif secure:
        _tool_notice = (
            "Server-side tools are ENABLED, reachable via the authenticated "
            "Cloudflare HTTPS tunnel. Anyone with the API key can run code on "
            "this machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    elif _is_external:
        _tool_notice = (
            "Server-side tools are ENABLED and this port is network-reachable. "
            "Anyone who can reach it with the API key can run code on this "
            "machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    else:
        _tool_notice = (
            "Server-side tools are ENABLED for loopback. Pass --disable-tools to turn off."
        )

    if not silent:
        typer.echo("")
        typer.echo("=" * 56)
        if secure and _cf_url:
            typer.echo(f"  Unsloth Studio running (secure) at {_cf_url}")
            typer.echo(f"  On this machine only: {base_url}")
        else:
            typer.echo(f"  Unsloth Studio running at {base_url}")
            _emit_run_cloudflare_notice(run_mod, host, display_host, actual_port, secure)
        typer.echo(f"  Model loaded: {loaded_model}{display_variant}")
        if context_length_line:
            typer.echo(context_length_line)
        typer.echo(f"  API Key:      {api_key}")
        typer.echo("")
        typer.echo("  OpenAI / Anthropic SDK base URL:")
        typer.echo(f"    {sdk_base_url}")
        typer.echo("=" * 56)
        typer.secho(_tool_notice, fg = _tool_notice_fg, bold = True)
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
    else:
        # Silent still prints URL + API key + tool-status policy.
        if secure and _cf_url:
            typer.echo(f"URL:     {_cf_url}")
            typer.echo(f"Local:   {base_url}")
        else:
            typer.echo(f"URL:     {base_url}")
            _emit_run_cloudflare_notice(run_mod, host, display_host, actual_port, secure)
        if context_length_line:
            typer.echo(context_length_line.strip())
        typer.echo(f"API Key: {api_key}")
        typer.secho(_tool_notice, fg = _tool_notice_fg, bold = True)

    # 7. Wait for Ctrl+C.
    try:
        if run_mod._shutdown_event is not None:
            while not run_mod._shutdown_event.is_set():
                run_mod._shutdown_event.wait(timeout = 1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        run_mod._graceful_shutdown(run_mod._server)
        typer.echo("\nShutting down...")
    finally:
        getattr(run_mod, "_wait_for_server_shutdown", lambda: None)()


# ── unsloth studio stop ───────────────────────────────────────────────

_PID_FILE = STUDIO_HOME / "studio.pid"
PID_FILE_GLOB = "studio-*.pid"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists.

    ``os.kill(pid, 0)`` raises OSError (WinError 87) for every pid on Windows,
    so use ``tasklist`` there and the signal-0 probe elsewhere.
    """
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                capture_output = True,
                text = True,
                timeout = 10,
            ).stdout
        except Exception:
            # Can't determine -- assume alive; taskkill no-ops if already gone.
            return True
        return f'"{int(pid)}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_pid_record(text: str) -> "tuple[int, float | None] | None":
    """Parse ``pid`` / optional ``create_time`` from PID file contents."""
    lines = text.splitlines()
    if not lines or not lines[0].strip().isdigit():
        return None
    try:
        # isdigit() is not enough: "²".isdigit() is True but int() rejects it.
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
    return pid, created


def _read_pid_record(path: Path) -> "tuple[int, float | None] | None":
    """Parse ``pid`` / optional ``create_time`` from a PID file."""
    try:
        text = path.read_text(encoding = "utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _parse_pid_record(text)


def _unlink_quietly(path: Path) -> None:
    """Drop a record without letting one bad file end the loop.

    An undeletable record must not stop us reaching the other servers -- that is
    the orphan this command exists to prevent.
    """
    try:
        path.unlink(missing_ok = True)
    except OSError as e:
        typer.echo(f"Could not remove PID file {path.name}: {e}", err = True)


def _report_unreadable(paths: "list[Path]") -> None:
    """Say which servers we could not reach, since `stop` is about to exit 1."""
    names = ", ".join(sorted(p.name for p in paths))
    typer.echo(
        f"Could not read {len(paths)} PID file(s): {names}. A server recorded "
        f"there may still be running; re-run with permission to read "
        f"{STUDIO_HOME} to stop it.",
        err = True,
    )


def _pid_file_entries(
    unreadable: "list[Path] | None" = None,
) -> "list[tuple[int, list[float | None], list[Path]]]":
    """(pid, create_times, files) per recorded server, including the legacy studio.pid.

    Paths that could not be read are appended to `unreadable` when given, so the
    caller can tell "nothing is running" apart from "something is running and we
    could not see it".

    Grouped by PID: a server writes both its per-port file and studio.pid, and
    signalling twice would hit the SIG_DFL the first SIGTERM installs, hard-killing
    it mid-shutdown. Every recorded time is kept -- a stale file and a live server
    can share a PID, and the stale one must not veto the live one.
    """
    by_pid: "dict[int, tuple[list[float | None], list[Path]]]" = {}
    try:
        paths = sorted(STUDIO_HOME.glob(PID_FILE_GLOB)) + [_PID_FILE]
    except OSError:
        paths = [_PID_FILE]
    seen = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding = "utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # Unreadable is not the same as invalid. A root-owned record, or one
            # caught mid-write, still belongs to a live server, and deleting it
            # strands that server -- the bug this command exists to fix.
            typer.echo(f"Cannot read PID file {path.name}: {e}", err = True)
            if unreadable is not None:
                unreadable.append(path)
            continue
        record = _parse_pid_record(text)
        if record is None:
            typer.echo(f"Ignoring invalid PID file {path.name}")
            _unlink_quietly(path)
            continue
        pid, created = record
        created_times, files = by_pid.setdefault(pid, ([], []))
        created_times.append(created)
        files.append(path)
    return [(pid, times, files) for pid, (times, files) in by_pid.items()]


def _pid_is_studio_server(pid: int, created_times: "Sequence[float | None]" = ()) -> bool:
    """False only when a recorded start time proves this PID is a different process.

    Any recorded time matching is enough -- a stale record must not veto a live
    server that reused the PID. Records with no time at all (a legacy studio.pid,
    or a server started without psutil) cannot be checked, so they are trusted:
    the old `stop` signalled with no checks at all, and skipping a live server is
    the orphan bug this exists to fix.

    An untimed record sitting *alongside* a timed one carries no information, so
    it must not cancel the timed one either. Every current server writes both a
    timed per-port record and an untimed studio.pid, so letting the untimed half
    win made this check inert exactly where it matters and let `stop` SIGTERM an
    unrelated process that had inherited the PID.
    """
    known = [c for c in created_times if c is not None]
    if not known:
        return True
    try:
        import psutil
        actual = psutil.Process(pid).create_time()
    except Exception:
        return True
    return any(abs(actual - c) < 1.0 for c in known)


def _signal_stop(pid: int) -> "str | None":
    """SIGTERM (or taskkill) the pid. Returns an error string, or None on success."""
    import signal as _signal

    if pid < 2:
        return f"refusing to signal PID {pid}"
    try:
        if sys.platform == "win32":
            # /T also stops llama-server children, which otherwise keep GPU and port.
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check = True)
        else:
            os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        return None
    except Exception as e:
        return str(e)
    return None


@studio_app.command()
def stop():
    """Stop every running Unsloth Studio server for this STUDIO_HOME.

    The port fallback can leave more than one running, so stop them all.
    """
    unreadable: "list[Path]" = []
    entries = _pid_file_entries(unreadable)
    if not entries:
        if unreadable:
            # Reporting success here would be a lie: the records we could not
            # read are kept, and the servers behind them are still serving.
            _report_unreadable(unreadable)
            raise typer.Exit(1)
        typer.echo("No running Unsloth server found (no PID file).")
        raise typer.Exit(0)

    signalled, failed = [], []
    for pid, created_times, paths in entries:
        if not _pid_alive(pid) or not _pid_is_studio_server(pid, created_times):
            for path in paths:
                _unlink_quietly(path)
            continue
        error = _signal_stop(pid)
        if error is not None:
            failed.append((pid, error))
            typer.echo(f"Failed to stop Unsloth server (PID {pid}): {error}", err = True)
            continue
        typer.echo(f"Sent shutdown signal to Unsloth server (PID {pid}).")
        signalled.append((pid, paths))

    if not signalled and not failed:
        if unreadable:
            _report_unreadable(unreadable)
            raise typer.Exit(1)
        typer.echo("No running Unsloth server found (cleaned up stale PID files).")
        raise typer.Exit(0)

    pending = list(signalled)
    for _ in range(10):
        if not pending:
            break
        time.sleep(0.5)
        for entry in list(pending):
            pid, paths = entry
            if not _pid_alive(pid):
                for path in paths:
                    _unlink_quietly(path)
                pending.remove(entry)

    stopped = len(signalled) - len(pending)
    if stopped:
        typer.echo(f"Unsloth server{'s' if stopped > 1 else ''} stopped ({stopped}).")
    for pid, _paths in pending:
        typer.echo(f"Unsloth server (PID {pid}) is shutting down (may take a few seconds).")
    if unreadable:
        _report_unreadable(unreadable)
    if failed or unreadable:
        raise typer.Exit(1)


# ── unsloth studio setup / update ─────────────────────────────────────


def _wait_for_windows_setup_process(process) -> int:
    """Reap setup and its descendants before a runtime-gate owner can unwind."""

    try:
        return process.wait()
    except BaseException:
        if process.poll() is not None:
            raise
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                check = False,
                **_windows_hidden_subprocess_kwargs(),
            )
        except BaseException:
            # taskkill interrupted or unavailable: hold the gate until setup
            # exits naturally rather than exposing a live mutator.
            pass
        while process.poll() is None:
            try:
                process.wait()
            except KeyboardInterrupt:
                continue
        raise


def _run_setup_script(*, verbose: bool = False, repo_root: Optional[Path] = None) -> None:
    """Find and run the studio setup/update script."""
    script = _find_setup_script(repo_root)
    if not script:
        if repo_root is not None:
            name = "setup.ps1" if platform.system() == "Windows" else "setup.sh"
            typer.echo(f"Error: {repo_root} has no studio/{name}.", err = True)
            typer.echo("  --local needs a complete checkout: the setup script builds", err = True)
            typer.echo("  the frontend into the tree that is installed editable.", err = True)
        else:
            typer.echo("Error: Could not find setup script (setup.sh / setup.ps1).")
        raise typer.Exit(1)

    env = {**os.environ, "UNSLOTH_VERBOSE": "1"} if verbose else None

    if platform.system() == "Windows":
        powershell_args = ["powershell.exe"]
        if _should_hide_windows_subprocesses():
            powershell_args.extend(
                ["-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden"]
            )
        # Use -Command + `*>&1` (not -File) so setup.ps1's Write-Host output
        # (Information stream #6) merges into stdout. -File drops it when
        # stdout is a pipe, e.g. `unsloth studio update --local 2>&1 | tee`.
        # Single-quote escaping handles paths containing apostrophes.
        script_pwsh_literal = str(script).replace("'", "''")
        powershell_args.extend(
            [
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"& '{script_pwsh_literal}' *>&1",
            ]
        )
        # Explicitly hand std handles to the child so CI tee sees setup.ps1's
        # output. On Windows, subprocess.Popen defaults to close_fds=True
        # (bInheritHandles=False); combined with CREATE_NO_WINDOW the child
        # has no console and no inherited handles, so Write-Host writes to
        # nothing. Passing stdout/stderr makes Python mark the std handles
        # inheritable via PROC_THREAD_ATTRIBUTE_HANDLE_LIST. Empty update.log
        # on windows-latest CI was the smoking gun (runs 25533694490/25534292239).
        process = subprocess.Popen(
            powershell_args,
            env = env,
            stdin = _stream_for_subprocess(sys.stdin),
            stdout = _stream_for_subprocess(sys.stdout),
            stderr = _stream_for_subprocess(sys.stderr),
            **_windows_hidden_subprocess_kwargs(),
        )
        returncode = _wait_for_windows_setup_process(process)
    else:
        result = subprocess.run(["bash", str(script)], env = env)
        returncode = result.returncode

    if returncode != 0:
        raise typer.Exit(returncode)


_INSTALLER_URL_BASH = "https://unsloth.ai/install.sh"
_INSTALLER_URL_PWSH = "https://unsloth.ai/install.ps1"


def _refresh_desktop_shortcuts(*, verbose: bool = False) -> None:
    """Re-run installer with --shortcuts-only to refresh launchers post-update."""
    env = {**os.environ}
    if verbose:
        env["UNSLOTH_VERBOSE"] = "1"

    is_windows = platform.system() == "Windows"
    installer_name = "install.ps1" if is_windows else "install.sh"
    installer_url = _INSTALLER_URL_PWSH if is_windows else _INSTALLER_URL_BASH

    # Prefer local checkout, fall back to package dir, then network fetch.
    local_repo = (os.environ.get("STUDIO_LOCAL_REPO") or "").strip()
    candidates: list[Path] = []
    if local_repo:
        candidates.append(Path(local_repo) / installer_name)
    candidates.append(_PACKAGE_ROOT / installer_name)

    args = ["--shortcuts-only"]
    if verbose:
        args.append("--verbose")

    if is_windows:
        ps_argv: list[str] = ["powershell.exe"]
        if _should_hide_windows_subprocesses():
            ps_argv.extend(["-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden"])

        for script in candidates:
            try:
                if script.is_file():
                    quoted = str(script).replace("'", "''")
                    argv = list(ps_argv)
                    argv.extend(
                        [
                            "-ExecutionPolicy",
                            "Bypass",
                            "-Command",
                            f"& '{quoted}' {' '.join(args)} *>&1",
                        ]
                    )
                    result = subprocess.run(
                        argv,
                        env = env,
                        check = False,
                        **_windows_hidden_subprocess_kwargs(),
                    )
                    if result.returncode != 0:
                        typer.echo(f"  refresh-launcher  install.ps1 exited {result.returncode}")
                    return
            except OSError:
                continue

        # PyPI installs lack install.ps1: fetch + pipe to powershell stdin.
        try:
            request = urllib.request.Request(
                installer_url, headers = {"User-Agent": "unsloth-studio-update"}
            )
            with urllib.request.urlopen(request, timeout = 30) as response:
                installer = response.read().decode("utf-8", errors = "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            typer.echo(f"  refresh-launcher  skipped: could not fetch {installer_url} ({exc})")
            return

        # install.ps1 auto-invokes `Install-UnslothStudio @args` at EOF; over
        # stdin `$args` is empty so that triggers the full installer flow
        # (deps, venv, prompts) before our shortcuts-only call. Strip it.
        installer = re.sub(
            r"(?m)^[ \t]*Install-UnslothStudio[ \t]+@args[ \t]*\r?\n?",
            "",
            installer,
        )
        # stdin-piped scripts have empty $args, so call Install-UnslothStudio explicitly.
        marker_args = " ".join(args)
        wrapper = installer + f"\nInstall-UnslothStudio {marker_args}\n"

        # Write to a UTF-8 BOM tempfile and use -File rather than -Command -.
        # `powershell.exe -Command -` reads stdin via [Console]::InputEncoding
        # (CP1252/OEM on most Windows boxes), which mangles box-drawing chars
        # in install.ps1. -File reads the BOM and decodes correctly. The
        # prefix gives AV/EDR engines (and grep'ing users) a clear identity.
        ps1_fd, ps1_path = tempfile.mkstemp(
            prefix = "unsloth-studio-refresh-",
            suffix = ".ps1",
        )
        try:
            with os.fdopen(ps1_fd, "wb") as fh:
                fh.write(b"\xef\xbb\xbf" + wrapper.encode("utf-8"))
            argv = list(ps_argv)
            argv.extend(["-ExecutionPolicy", "Bypass", "-File", ps1_path])
            try:
                result = subprocess.run(
                    argv,
                    env = env,
                    check = False,
                    **_windows_hidden_subprocess_kwargs(),
                )
                if result.returncode != 0:
                    typer.echo(
                        f"  refresh-launcher  fetched install.ps1 exited {result.returncode}"
                    )
            except OSError as exc:
                typer.echo(f"  refresh-launcher  skipped: powershell exec failed ({exc})")
        finally:
            try:
                os.unlink(ps1_path)
            except OSError:
                pass
        return

    for script in candidates:
        try:
            if script.is_file():
                result = subprocess.run(
                    ["bash", str(script), *args],
                    env = env,
                    check = False,
                )
                if result.returncode != 0:
                    typer.echo(f"  refresh-launcher  install.sh exited {result.returncode}")
                return
        except OSError:
            continue

    # PyPI installs lack install.sh: fetch upstream.
    try:
        request = urllib.request.Request(
            installer_url, headers = {"User-Agent": "unsloth-studio-update"}
        )
        with urllib.request.urlopen(request, timeout = 30) as response:
            installer = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        typer.echo(f"  refresh-launcher  skipped: could not fetch {installer_url} ({exc})")
        return

    try:
        result = subprocess.run(
            ["bash", "-s", "--", *args],
            input = installer,
            env = env,
            check = False,
        )
        if result.returncode != 0:
            typer.echo(f"  refresh-launcher  fetched install.sh exited {result.returncode}")
    except OSError as exc:
        typer.echo(f"  refresh-launcher  skipped: bash exec failed ({exc})")


@studio_app.command(hidden = True)
def setup(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Full pip/build output during setup for troubleshooting.",
    ),
):
    """Run Unsloth setup (called by install.ps1 / install.sh)."""
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        _studio_runtime_gate.ensure_managed_environment_is_idle(STUDIO_HOME)
        _run_setup_script(verbose = verbose)


def _fail_if_install_damaged() -> None:
    """Refuse to call an update successful when the tree it produced is damaged.

    pip considers a distribution with intact metadata already satisfied, so an
    update reinstalls nothing when a package's FILES are damaged: it prints
    "Unsloth Studio Installed", exits 0, and the backend then dies at boot. That
    is the shape behind "just re-run the installer", and it is only actionable
    if the update says so.
    """
    if _studio_deps.running_outside_managed_venv((STUDIO_HOME / "unsloth_studio",)):
        # This CLI does not live in the venv the update just wrote, so its own
        # file list describes the wrong tree. Silence beats a wrong answer.
        return
    damaged = _studio_deps.damaged_installed_files()
    if not damaged:
        return
    typer.echo("", err = True)
    typer.echo("Update finished, but some installed files are damaged:", err = True)
    for entry in damaged:
        typer.echo(f"  {entry}", err = True)
    typer.echo("", err = True)
    typer.echo("An update cannot repair these. pip sees intact package metadata and", err = True)
    typer.echo("reinstalls nothing, so Studio will keep failing to start. Reinstall", err = True)
    typer.echo("over the top:", err = True)
    # Carry a custom root into the command. The shim is a bare symlink and
    # _ensure_studio_env_exported only sets os.environ for this process, so the
    # shell that runs this line has no UNSLOTH_STUDIO_HOME: an unqualified
    # reinstall would build a fresh ~/.unsloth/studio and leave the damaged
    # install exactly as broken as it was.
    # And carry the recorded install mode. install.sh derives SKIP_TORCH only
    # from its own flag or UNSLOTH_NO_TORCH and passes that value into setup, so
    # a plain reinstall over a GGUF-only install downloads the whole PyTorch
    # stack. Only added when the record says True: recorded_no_torch() returns
    # None when nothing recorded the mode, and None must never be read as False.
    #
    # No root argument: the manifest and marker live in the VENV, not the
    # install root, and recorded_no_torch defaults to Path(sys.prefix). Passing
    # STUDIO_HOME would look one directory too high, find nothing, and silently
    # never fire. The early return above guarantees sys.prefix is that venv.
    no_torch = False
    try:
        _manifest = _studio_deps.load_install_manifest_module()
        no_torch = _manifest is not None and _manifest.recorded_no_torch() is True
    except Exception:
        no_torch = False
    if platform.system() == "Windows":
        prefix = ""
        if _STUDIO_HOME_IS_CUSTOM:
            prefix = "$env:UNSLOTH_STUDIO_HOME = '{}'; ".format(str(STUDIO_HOME).replace("'", "''"))
        if no_torch:
            prefix += "$env:UNSLOTH_NO_TORCH = '1'; "
        typer.echo(f"  {prefix}irm https://unsloth.ai/install.ps1 | iex", err = True)
    else:
        # The assignments go before `sh`, not before `curl`: that is the form
        # install.sh documents, and it is sh that reads them.
        env = ""
        if _STUDIO_HOME_IS_CUSTOM:
            env = f"UNSLOTH_STUDIO_HOME={shlex.quote(str(STUDIO_HOME))} "
        if no_torch:
            env += "UNSLOTH_NO_TORCH=1 "
        typer.echo(f"  curl -fsSL https://unsloth.ai/install.sh | {env}sh", err = True)
    typer.echo("", err = True)
    # The installer installs the current requirement sets; it does not prune or
    # reinstall anything outside them. So a package left over from an older
    # release, or added by hand, is not repaired by the command above and would
    # otherwise report the same damage forever. Say what to do in that case
    # rather than scoping the scan, which would risk passing over real damage.
    typer.echo("If a package above is still listed after that, the installer does not", err = True)
    typer.echo("manage it. Repair it directly, or remove it if nothing needs it:", err = True)
    # --no-deps: without it pip resolves the damaged package's dependency graph
    # and --force-reinstall would replace pinned runtime packages too, which can
    # swap the installed CUDA/ROCm torch build for a default one while repairing
    # an unrelated orphan. The installer's own targeted repairs pair the two for
    # the same reason. The interpreter path is quoted because a custom root may
    # contain spaces, and on Windows a quoted command needs the call operator.
    # <package>==<version> rather than a bare name: --force-reinstall reinstalls
    # even when the package is already up to date, so an unpinned name fetches
    # the newest release and silently upgrades an orphan whose consumers may
    # depend on the older one. --no-deps does not protect against that.
    _spec = "<package>==<installed version>"
    if platform.system() == "Windows":
        _py = str(Path(sys.executable)).replace("'", "''")
        typer.echo(f"  & '{_py}' -m pip install --force-reinstall --no-deps {_spec}", err = True)
    else:
        _py = shlex.quote(str(Path(sys.executable)))
        typer.echo(f"  {_py} -m pip install --force-reinstall --no-deps {_spec}", err = True)
    typer.echo("", err = True)
    typer.echo("To update anyway without this check: unsloth studio update --no-verify", err = True)
    raise typer.Exit(code = 1)


@studio_app.command()
def update(
    local: bool = typer.Option(False, "--local", help = "Install from local repo instead of PyPI"),
    package: str = typer.Option(
        "unsloth", "--package", help = "Package name to install/update (for testing)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Full pip/build output during update for troubleshooting.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help = "After updating, scan installed files for damage an update cannot repair.",
    ),
):
    """Update Unsloth Studio dependencies and rebuild."""
    # Re-export UNSLOTH_STUDIO_HOME for env-mode installs so the refresh
    # subprocess resolves the same install root the user originally chose.
    _ensure_studio_env_exported()
    # Ensure SKIP_STUDIO_BASE is not inherited from a parent install.ps1 session
    os.environ.pop("SKIP_STUDIO_BASE", None)
    os.environ["STUDIO_PACKAGE_NAME"] = package
    repo_root: Optional[Path] = None
    if local:
        os.environ["STUDIO_LOCAL_INSTALL"] = "1"
        # Pass the repo root explicitly so install_python_stack.py doesn't
        # have to guess from SCRIPT_DIR (which may be inside site-packages).
        # Deriving it from __file__ only holds while this CLI runs from a
        # checkout. Once an update has installed unsloth into the venv
        # non-editably, parents[2] IS site-packages, and uv rejects it with
        # "does not appear to be a Python project: neither 'setup.py' nor
        # 'pyproject.toml' found" -- which is what a second `update --local`
        # hit on Windows, where the first update replaces the editable install.
        # Absolutise the override: setup.sh does `cd "$SCRIPT_DIR"` before it
        # runs install_python_stack.py, so a relative STUDIO_LOCAL_REPO would
        # be re-resolved against studio/ (no pyproject.toml) and hand uv back
        # the very error this guard exists to replace. .strip()/.expanduser()
        # match the handling in _refresh_desktop_shortcuts.
        _explicit = (os.environ.get("STUDIO_LOCAL_REPO") or "").strip()
        repo_root = (
            Path(_explicit).expanduser().resolve()
            if _explicit
            else Path(__file__).resolve().parents[2]
        )
        if not (repo_root / "pyproject.toml").is_file():
            typer.echo("Error: --local needs an Unsloth checkout to install from.", err = True)
            typer.echo(f"  no pyproject.toml under: {repo_root}", err = True)
            typer.echo("  This CLI is running from an installed copy, not a source tree.", err = True)
            typer.echo("", err = True)
            typer.echo("  Point at a checkout:", err = True)
            if platform.system() == "Windows":
                # PowerShell has no `VAR=value command` prefix form: it parses
                # the assignment as a command name and fails to find it. This
                # guard fires on the Windows update path, so the POSIX spelling
                # would be unusable for most of the people who see it.
                typer.echo(
                    "    $env:STUDIO_LOCAL_REPO='C:\\path\\to\\unsloth'; "
                    "unsloth studio update --local",
                    err = True,
                )
            else:
                typer.echo(
                    "    STUDIO_LOCAL_REPO=/path/to/unsloth unsloth studio update --local",
                    err = True,
                )
            typer.echo("  Or update from PyPI:", err = True)
            typer.echo("    unsloth studio update", err = True)
            raise typer.Exit(2)
        os.environ["STUDIO_LOCAL_REPO"] = str(repo_root)
    else:
        os.environ["STUDIO_LOCAL_INSTALL"] = "0"
        os.environ.pop("STUDIO_LOCAL_REPO", None)
    # main gained a runtime gate around setup; this branch replaced the
    # rename-to-.deleteme helpers with the launcher transaction. Both apply:
    # the gate keeps a second Studio process off the venv, the transaction
    # keeps the launcher recoverable across the setup it wraps.
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        _studio_runtime_gate.ensure_managed_environment_is_idle(STUDIO_HOME)
        with _WindowsLauncherUpdateTransaction() as launcher_update:
            _run_setup_script(verbose = verbose, repo_root = repo_root)
            # This deliberately runs even with --no-verify: the broad package scan
            # is optional, but a successful update must leave its own launcher usable.
            launcher_update.validate_launcher()
            if verify:
                _fail_if_install_damaged()
    # Tauri desktop owns its own bundle entries; skip CLI launcher refresh
    # so a Tauri-initiated update doesn't create duplicate shortcuts.
    if os.environ.get("UNSLOTH_TAURI_UPDATE") == "1":
        if verbose:
            typer.echo("  refresh-launcher  skipped (Tauri update)")
        return
    _refresh_desktop_shortcuts(verbose = verbose)


class _WindowsLauncherUpdateTransaction:
    """Keep the managed Windows launcher recoverable during a Python update."""

    _VERSION_TIMEOUT_SECONDS = 10
    _RESTORE_ATTEMPTS = 3

    def __init__(self) -> None:
        self.enabled = platform.system() == "Windows"
        self.launcher: Optional[Path] = None
        self.backup: Optional[Path] = None
        self.legacy_backup: Optional[Path] = None
        self.stale: Optional[Path] = None
        self.shim: Optional[Path] = None
        self.lock_path: Optional[Path] = None
        self._lock_file = None
        self._validated = False

    @staticmethod
    def _is_valid_pe(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < 2:
                return False
            with path.open("rb") as handle:
                return handle.read(2) == b"MZ"
        except OSError:
            return False

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        """Publish a sibling copy without exposing a partial destination."""
        fd, temporary_name = tempfile.mkstemp(
            prefix = f".{destination.name}.",
            suffix = ".tmp",
            dir = str(destination.parent),
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as source_handle, os.fdopen(fd, "wb") as target_handle:
                fd = -1
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            os.replace(temporary, destination)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temporary.unlink(missing_ok = True)
            except OSError:
                pass

    def _acquire_lock(self) -> None:
        import msvcrt

        assert self.lock_path is not None
        try:
            self.lock_path.parent.mkdir(parents = True, exist_ok = True)
        except OSError:
            pass
        lock_file = self.lock_path.open("a+b")
        try:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock_file.close()
            typer.echo(
                "Error: another Unsloth Studio update is already running for this environment.",
                err = True,
            )
            raise typer.Exit(1)
        self._lock_file = lock_file

    def _release_lock(self) -> None:
        if self._lock_file is None:
            return
        try:
            import msvcrt
            self._lock_file.seek(0)
            msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            self._lock_file.close()
            self._lock_file = None

    def _recover_missing_launcher(self) -> None:
        assert self.launcher is not None
        # Validity, not existence: a truncated or quarantined launcher is just as
        # unusable, and the backup beside it can repair either.
        if self._is_valid_pe(self.launcher):
            return
        for recovery in (self.backup, self.stale, self.legacy_backup, self.shim):
            if recovery is not None and self._is_valid_pe(recovery):
                try:
                    self._atomic_copy(recovery, self.launcher)
                except OSError as exc:
                    typer.echo(
                        f"Error: could not recover {self.launcher} from {recovery}: {exc}",
                        err = True,
                    )
                    typer.echo(f"Manual recovery copy retained at: {recovery}", err = True)
                    raise typer.Exit(1)
                return

    @staticmethod
    def _files_match(left: Path, right: Path) -> bool:
        try:
            if left.stat().st_size != right.stat().st_size:
                return False
            with left.open("rb") as left_handle, right.open("rb") as right_handle:
                while True:
                    left_chunk = left_handle.read(1024 * 1024)
                    if left_chunk != right_handle.read(1024 * 1024):
                        return False
                    if not left_chunk:
                        return True
        except OSError:
            return False

    def _move_launcher_aside(self) -> None:
        """Free the canonical path so the installer can publish a replacement.

        uv only self-replaces its OWN executable, so it deletes a third-party
        console script outright and hard-errors when the file is in use, after
        which the pip fallback no-ops on the already-satisfied bare unsloth and
        the upgrade is silently skipped. Renaming a running image is allowed on
        Windows: verified on windows-latest that renaming a live console-script
        launcher succeeds and a replacement can then be written at the freed
        path. Non-fatal, since failing to move it aside costs only the upgrade.
        """
        assert self.launcher is not None and self.stale is not None
        if not self._is_valid_pe(self.launcher):
            return
        try:
            os.replace(self.launcher, self.stale)
        except OSError as exc:
            # Not fatal: an antivirus hold must not make the environment
            # unupdatable. But say what it costs, because uv cannot then replace
            # the launcher and the pip fallback drops --upgrade-package, so
            # unsloth is left at its old version while everything else updates.
            typer.echo(f"Warning: could not move the Unsloth launcher aside: {exc}", err = True)
            typer.echo(
                "  unsloth itself may not be upgraded. Close anything holding "
                f"{self.launcher} and re-run the update.",
                err = True,
            )

    def _retained_backup(self) -> Optional[Path]:
        """The backup, when it exists and is usable. Nothing to point users at otherwise."""
        if self.backup is not None and self._is_valid_pe(self.backup):
            return self.backup
        return None

    def _recovery_candidates(self) -> List[Path]:
        """Copies that could stand in for the launcher, best first.

        The backup is the last launcher known to run; the moved-aside copy is
        only this run's unvalidated canonical file; the legacy .deleteme and the
        PATH shim are what an install broken by the old updater still has. All
        of them are kept, because passing the two-byte header check does not
        make any one of them runnable and the next candidate has to be reachable.
        """
        seen: List[Path] = []
        for path in (self.backup, self.stale, self.legacy_backup, self.shim):
            if path is None or not self._is_valid_pe(path):
                continue
            if not any(os.path.normcase(str(path)) == os.path.normcase(str(p)) for p in seen):
                seen.append(path)
        return seen

    def _restore_from(self, source: Path) -> bool:
        assert self.launcher is not None
        # The common setup-failure case leaves the original executable exactly
        # where it was. Avoid replacing that running file on Windows when it
        # already is the source byte-for-byte.
        if self._is_valid_pe(self.launcher) and self._files_match(self.launcher, source):
            return True
        last_error: Optional[OSError] = None
        for attempt in range(self._RESTORE_ATTEMPTS):
            try:
                self._atomic_copy(source, self.launcher)
                return self._is_valid_pe(self.launcher)
            except OSError as exc:
                last_error = exc
                if attempt + 1 < self._RESTORE_ATTEMPTS:
                    time.sleep(0.1)
        if last_error is not None:
            typer.echo(f"Error: could not restore the Unsloth launcher: {last_error}", err = True)
        return False

    def _restore_runnable(self) -> bool:
        """Put back the first copy that actually runs.

        Passing the two-byte header check does not make a copy runnable, so a
        candidate that fails --version must not stop the next one being tried,
        and a launcher already in place and working must not be replaced by a
        candidate that is merely PE-shaped.
        """
        if self._launcher_health_error() is None:
            return True
        candidates = self._recovery_candidates()
        for source in candidates:
            if self._restore_from(source) and self._launcher_health_error() is None:
                return True
        # Nothing ran. Leave the best candidate in place rather than whichever
        # one happened to be tried last.
        if candidates:
            self._restore_from(candidates[0])
        return False

    def _launcher_health_error(self) -> Optional[str]:
        assert self.launcher is not None
        if not self._is_valid_pe(self.launcher):
            return "the updated launcher is missing or is not a non-empty PE file"
        try:
            result = subprocess.run(
                [str(self.launcher), "--version"],
                check = False,
                capture_output = True,
                timeout = self._VERSION_TIMEOUT_SECONDS,
                **_windows_hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return f"the updated launcher timed out after {self._VERSION_TIMEOUT_SECONDS} seconds"
        except OSError as exc:
            return f"the updated launcher could not run --version ({exc})"
        if result.returncode != 0:
            return f"the updated launcher returned {result.returncode} for --version"
        return None

    @staticmethod
    def _managed_scripts_dir() -> Path:
        """Scripts dir of the venv setup actually updates.

        setup.ps1 installs into STUDIO_HOME/unsloth_studio, which is not this
        interpreter when a pip-installed or checkout CLI drives the update. Same
        distinction _studio_deps._managed_root draws for the damage scan.
        """
        managed = STUDIO_HOME / "unsloth_studio"
        if (managed / "pyvenv.cfg").is_file():
            try:
                foreign = managed.resolve() != Path(sys.prefix).resolve()
            except OSError:
                foreign = True
            if foreign:
                return managed / "Scripts"
        return Path(sys.executable).resolve().parent

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            scripts = self._managed_scripts_dir()
        except (OSError, RuntimeError) as exc:
            typer.echo(f"Error: could not resolve the managed Python environment: {exc}", err = True)
            raise typer.Exit(1)
        self.launcher = scripts / "unsloth.exe"
        self.backup = scripts / "unsloth.exe.update-backup"
        self.legacy_backup = scripts / "unsloth.exe.deleteme"
        # Under the Studio home, not the venv: setup.ps1 removes the whole
        # $VenvDir to rebuild a stale torch, and an open handle inside it makes
        # Windows refuse the recursive delete. One lock per Studio home is the
        # right grain anyway, since that is what names the managed venv.
        self.lock_path = STUDIO_HOME / "unsloth.exe.update-lock"
        # install.ps1 hardlinks this to the launcher, so it survives the old
        # updater's .deleteme unlink and is a valid recovery source.
        self.shim = STUDIO_HOME / "bin" / "unsloth.exe"
        self.stale = scripts / "unsloth.exe.update-stale"
        self._acquire_lock()
        try:
            self._recover_missing_launcher()
            if not self._is_valid_pe(self.launcher):
                # Warn, do not exit. The previous updater could leave an install
                # with no launcher and no .deleteme, and refusing here would stop
                # exactly those users from ever updating again. Setup may well
                # write a new launcher; validate_launcher still judges the result.
                typer.echo(
                    f"Warning: the managed Unsloth launcher is missing or invalid: {self.launcher}",
                    err = True,
                )
                typer.echo("Continuing; setup may reinstall it.", err = True)
                if self._retained_backup() is None:
                    self.backup = None
            elif self._retained_backup() is None:
                # Only write a backup when there is no usable one already. A
                # backup outlives __enter__ only when a previous run died before
                # validating, so it holds the last launcher known to run, while
                # the canonical file has passed nothing but the two-byte header
                # check. Overwriting it here destroyed the only recovery copy.
                try:
                    self._atomic_copy(self.launcher, self.backup)
                except OSError as exc:
                    # A backup is a safety net, not a precondition. Antivirus or a
                    # locked-down Scripts dir must not abort the update outright.
                    typer.echo(f"Warning: could not back up the Unsloth launcher: {exc}", err = True)
                    self.backup = None
            self._move_launcher_aside()
        except BaseException:
            self._release_lock()
            raise
        return self

    def validate_launcher(self) -> None:
        if not self.enabled:
            return
        # Whether setup published anything decides how a bad result is read, so
        # it has to be sampled before any restore puts a launcher back.
        published = self.launcher.exists()
        error = self._launcher_health_error()
        if error is not None:
            restored = self._restore_runnable()
            # Setup publishing nothing is the case this transaction exists for:
            # a no-op pip update leaves the freed path empty, and the old
            # updater then deleted its own .deleteme, leaving no launcher at
            # all. Putting the previous one back is success, not failure. A
            # launcher setup DID write and that cannot run is still a failure,
            # even though the previous one goes back.
            if published or not restored:
                typer.echo(f"Error: Unsloth Studio update failed because {error}.", err = True)
                if restored:
                    typer.echo("The previous launcher was restored.", err = True)
                elif self._retained_backup() is not None:
                    typer.echo(f"Manual recovery copy retained at: {self.backup}", err = True)
                raise typer.Exit(1)
        self._validated = True
        for orphan in (self.stale, self.backup, self.legacy_backup):
            if orphan is None:
                continue
            try:
                orphan.unlink(missing_ok = True)
            except OSError:
                pass

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if self.enabled and exc_type is not None and not self._validated:
                if not self._restore_runnable() and self._retained_backup() is not None:
                    typer.echo(f"Manual recovery copy retained at: {self.backup}", err = True)
        finally:
            self._release_lock()
        return False


# ── unsloth studio reset-password ────────────────────────────────────


@studio_app.command("desktop-capabilities", hidden = True)
def desktop_capabilities(
    json_output: bool = typer.Option(
        False,
        "--json",
        help = "Emit machine-readable JSON.",
    ),
):
    state = _install_state()
    payload = {
        "desktop_protocol_version": 1,
        # 2 adds studio_install_ok; the desktop treats < 2 as stale rather than
        # guess at an absent field.
        "desktop_manageability_version": 2,
        "supports_provision_desktop_auth": True,
        "supports_api_only": True,
        "supports_desktop_backend_ownership": True,
        # Did the install finish and are the backend's boot deps still there.
        "studio_install_ok": bool(state["ok"]),
        "studio_install_reason": state["reason"],
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


@studio_app.command("verify-install")
def verify_install(
    json_output: bool = typer.Option(
        False,
        "--json",
        help = "Emit machine-readable JSON.",
    ),
):
    """Check that the Unsloth Studio dependency install completed.

    Exits 0 when complete, 1 otherwise. setup.sh / setup.ps1 use the exit code
    to decide whether the "already up to date" fast path may be taken.
    """
    state = _install_state()

    if json_output:
        typer.echo(json.dumps(state, sort_keys = True))
        raise typer.Exit(0 if state["ok"] else 1)

    if state["ok"]:
        typer.echo("Unsloth Studio install is complete.")
        raise typer.Exit(0)

    typer.echo(f"Unsloth Studio install is incomplete ({state['reason']}).")
    if state["missing"]:
        typer.echo(f"  missing packages: {', '.join(state['missing'])}")
    typer.echo("  repair with: unsloth studio update")
    raise typer.Exit(1)


@studio_app.command("provision-desktop-auth", hidden = True)
def provision_desktop_auth():
    """Create/repair desktop auth state for the local machine."""
    auth_dir = STUDIO_HOME / "auth"
    secret = _create_desktop_secret_in_cli()
    _write_auth_secret(auth_dir / DESKTOP_SECRET_FILE, secret)
    typer.echo("Desktop auth ready.")


@studio_app.command("reset-password")
def reset_password():
    """Reset the Unsloth admin password.

    Rotates the credential in place: a running Unsloth accepts the new password on
    its next request, so there is nothing to restart. Shared /p preview links are
    not revoked -- rotate those in Settings if the old password leaked.
    """
    new_password = _generate_reset_password()
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        typer.echo(
            f"Error: could not open the auth database ({exc}). Check that "
            f"{STUDIO_HOME / 'auth'} is writable; if auth.db itself is unreadable, stop "
            "Unsloth, delete it, and start again to re-seed.",
            err = True,
        )
        raise typer.Exit(1)

    try:
        _ensure_cli_default_admin(conn)
        _cli_update_password(conn, DEFAULT_ADMIN_USERNAME, new_password, revoke_api_keys = True)
    except (OSError, sqlite3.Error) as exc:
        typer.echo(f"Error: could not reset the password ({exc}).", err = True)
        raise typer.Exit(1)
    finally:
        conn.close()

    typer.echo(f"New password for '{DEFAULT_ADMIN_USERNAME}': {new_password}")
    typer.echo(
        "Sessions and API keys revoked. A running Unsloth takes it on the next request, "
        "though repeated failed logins can hold the rate limit shut for up to a minute."
    )

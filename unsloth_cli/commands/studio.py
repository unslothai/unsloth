# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import platform
import re
import secrets
import sqlite3
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import typer

studio_app = typer.Typer(help = "Unsloth Studio commands.")


# Resolve install root: UNSLOTH_STUDIO_HOME, then STUDIO_HOME alias, then
# sys.prefix inference (so a direct call to <root>/bin/unsloth resolves after
# the installer's env var has expired), then legacy ~/.unsloth/studio.
# UNSLOTH_STUDIO_HOME wins when both env vars are set.
def _looks_like_installer_managed_studio_home(candidate: Path) -> bool:
    """Sentinel check (studio.conf or bin shim) so a dev venv named
    unsloth_studio is not misidentified as a custom Studio root.
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
            if inferred != legacy and _looks_like_installer_managed_studio_home(
                inferred
            ):
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
STUDIO_HOME = Path.home() / ".unsloth" / "studio"

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


# Mirror in studio/backend/run.py argparse + backend denylist test;
# bumping the cap in one place only desyncs.
_PARALLEL_MIN = 1
_PARALLEL_MAX = 64
_PARALLEL_DEFAULT_RUN = 4  # pre-PR hardcoded for `unsloth studio run`
_PARALLEL_DEFAULT_PLAIN = 1  # pre-PR effective for plain `unsloth studio`


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
                except OSError:
                    continue
                # Tolerate single- or multi-line dict literals; [^}]* still
                # rejects nested dicts, which the setuptools template never
                # emits for editable installs.
                m = re.search(
                    r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S
                )
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
    from auth.storage import create_api_key, DEFAULT_ADMIN_USERNAME

    raw_key, _row = create_api_key(username = DEFAULT_ADMIN_USERNAME, name = name)
    return raw_key


def _load_model_via_http(
    port: int,
    api_key: str,
    model: str,
    gguf_variant: Optional[str],
    max_seq_length: int,
    load_in_4bit: bool,
    llama_extra_args: Optional[List[str]] = None,
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
    if llama_extra_args:
        payload["llama_extra_args"] = list(llama_extra_args)

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
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
<<<<<<< HEAD
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
            f"Default {_PARALLEL_DEFAULT_PLAIN}; `unsloth studio run` "
            f"defaults to {_PARALLEL_DEFAULT_RUN}."
        ),
    ),
=======
>>>>>>> 09566d730 (fix failed to start on docker)
):
    """Launch the Unsloth Studio server."""
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
        return

    # In Docker, packages live in /opt/conda — skip venv re-exec entirely.
    if not os.environ.get("UNSLOTH_DOCKER"):
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

    run_kwargs = dict(
        host = host,
        port = port,
        silent = silent,
        api_only = api_only,
        llama_parallel_slots = parallel,
    )
    if frontend is not None:
        run_kwargs["frontend_path"] = frontend
    run_server(**run_kwargs)

    from studio.backend.run import _shutdown_event

    try:
        if _shutdown_event is not None:
            # Event.wait() with no timeout blocks at C-level on Linux
            # and swallows SIGINT; loop with a 1s timeout instead.
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
    # Click clusters `-np8` as `-n -p 8` (-p = --port), dropping the
    # parallel value. Split to `-np <N>` so typer's alias matches.
    # Stops at `--`; accepts signed and digit-prefix-junk forms so
    # typer can report a clean error against `-np`. Kept in lockstep
    # with the backend `_flag_name` recogniser.
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
    args: List[str],
    aliases: tuple[str, ...],
    current: Optional[str],
    canonical: str,
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
            raise typer.BadParameter(
                f"{name} conflicts with {canonical} already provided"
            )
        if sep:
            if inline == "":  # `-m=` would become --model '' (Path('')='.').
                raise typer.BadParameter(f"{name} requires a non-empty value")
            value = inline
            i += 1
        elif i + 1 < n:
            nxt = args[i + 1]
            # `--long` is unambiguously a flag; single-dash `-x` may be a path.
            if nxt.startswith("--") and nxt != "--":
                raise typer.BadParameter(
                    f"{name} expects a value but got the flag {nxt}"
                )
            value = nxt
            i += 2
        else:
            raise typer.BadParameter(f"{name} requires a value")
    return value, out


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
        help = (
            "Model path or HF repo. Accepts llama.cpp-style "
            "`org/repo:variant` syntax. `-hf` / `--hf-repo` match "
            "llama-server's spelling."
        ),
    ),
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
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    # `-f` removed (clustered `-fa`/`-fit*`); studio_default keeps it.
    frontend: Optional[Path] = typer.Option(None, "--frontend"),
    silent: bool = typer.Option(False, "--silent", "-q"),
    enable_tools: Optional[bool] = typer.Option(
        None,
        "--enable-tools/--disable-tools",
        help = (
            "Force server-side tools on/off for all requests. "
            "Default: on for 127.0.0.1, off for 0.0.0.0."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help = "Skip the 0.0.0.0 + --enable-tools confirmation prompt.",
    ),
    parallel: int = typer.Option(
        _PARALLEL_DEFAULT_RUN,
        "--parallel",
        "--n-parallel",
        "-np",
        min = _PARALLEL_MIN,
        max = _PARALLEL_MAX,
        help = (
            "llama-server parallel decode slots. N requests share one "
            "loaded model; each slot gets ctx/N KV cache. Default "
            f"{_PARALLEL_DEFAULT_RUN} (pre-PR hardcoded value)."
        ),
    ),
):
    """Start Studio, load a model, print an API key -- one-liner server.

    Unknown flags pass through to llama-server (GGUF only). Studio
    rejects managed flags with HTTP 400: model identity, network
    (--host/--port/--path/--api-prefix/--reuse-port), auth/TLS
    (--api-key/--ssl-*), single-model UI (--ui/--models-*/--webui),
    and parallel slots (use --parallel above). Full denylist in
    studio/backend/core/inference/llama_server_args.py. Other knobs
    (-c, -ngl, --jinja, --flash-attn, -t, ...) pass through and
    last-wins-override Studio's auto-set value.

    Example:
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --gguf-variant UD-Q4_K_XL
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --top-k 20 --seed 42 --parallel 8
        unsloth studio run --model some-model --chat-template-file /path/to/tpl.jinja
    """
    extra_llama_args: List[str] = list(ctx.args) if ctx.args else []

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

    # Resolve tool policy here so the re-exec'd child inherits a
    # concrete decision and never re-prompts.
    from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy

    enable_tools = resolve_tool_policy(
        host = host,
        flag = enable_tools,
        yes = yes,
        silent = silent,
    )

    # 1. Re-exec into the studio venv (same pattern as studio_default).
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))

    if not in_studio_venv:
        studio_python = _studio_venv_python()
        if not studio_python:
            typer.echo("Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # Re-exec via the studio venv's `unsloth` console-script.
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
        # Forward the explicit polarity; a future default flip on one
        # layer must not silently invert behaviour for the other.
        args.append("--load-in-4bit" if load_in_4bit else "--no-load-in-4bit")
        if frontend:
            args.extend(["--frontend", str(frontend)])
        if silent:
            args.append("--silent")
        # Forward the resolved tool policy so the child doesn't re-resolve.
        if enable_tools:
            args.append("--enable-tools")
        else:
            args.append("--disable-tools")
        # Forward --yes if the parent already cleared the network-bind
        # prompt, else the child re-prompts.
        if yes or (enable_tools and is_external_host(host)):
            args.append("--yes")
        # Typer claims --parallel outside ctx.args; without this the
        # child reverts to its default and silently drops the value.
        args.extend(["--parallel", str(parallel)])
        # llama-server pass-through extras → child ctx.args → load payload.
        if extra_llama_args:
            args.extend(extra_llama_args)

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

    run_kwargs = dict(host = host, port = port, silent = True, llama_parallel_slots = parallel)
    if frontend is not None:
        run_kwargs["frontend_path"] = frontend
    app = run_server(**run_kwargs)
    actual_port = getattr(app.state, "server_port", port) or port

    # Match the route handlers' import path: run.py adds
    # studio/backend/ to sys.path, so they import as `state.tool_policy`.
    # Importing via `studio.backend.state.tool_policy` would cache a
    # second module object whose flag the gates can't see.
    from state.tool_policy import set_tool_policy

    set_tool_policy(enable_tools)

    # 3. Wait for server health.
    if not silent:
        typer.echo("Starting Unsloth Studio...")
    if not _wait_for_server(actual_port):
        typer.echo("Error: server did not become healthy within 30 seconds.", err = True)
        raise typer.Exit(1)

    # 4. Create API key in-process.
    api_key = _create_api_key_inprocess(api_key_name)

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
            llama_extra_args = extra_llama_args,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(1)

    loaded_model = result.get("model", model)
    display_variant = f" ({gguf_variant})" if gguf_variant else ""

    # 6. Print banner.
    display_host = _resolve_external_ip() if host == "0.0.0.0" else host
    base_url = f"http://{display_host}:{actual_port}"
    sdk_base_url = f"{base_url}/v1"

    # Orange so the tool-policy notice stands out; printed under
    # --silent / --yes too so the policy is never invisible.
    _tool_notice_fg = (217, 119, 87)
    _is_external = is_external_host(host)
    if _is_external and enable_tools:
        _tool_notice = (
            f"Server-side tools are ENABLED on {host} (network-reachable). "
            f"Anyone with the API key can run code on this machine. "
            f"Do not share the API key."
        )
    elif _is_external:
        _tool_notice = (
            f"Server-side tools are disabled by default on {host} "
            f"(network-reachable). Pass --enable-tools to turn on "
            f"(you will be warned about API-key risk)."
        )
    elif enable_tools:
        _tool_notice = (
            "Server-side tools are enabled by default for loopback. "
            "Pass --disable-tools to turn off."
        )
    else:
        _tool_notice = "Server-side tools are disabled."

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
        typer.echo(f"URL:     {base_url}")
        typer.echo(f"API Key: {api_key}")
        typer.secho(_tool_notice, fg = _tool_notice_fg, bold = True)

    # 7. Wait for Ctrl+C.
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
        # Use -Command + `*>&1` instead of -File so setup.ps1's
        # Write-Host output (PowerShell Information stream / #6) is
        # merged into the success stream and reaches the parent's
        # stdout. With -File, Information stream output is dropped
        # whenever stdout is a pipe, which is exactly the situation
        # CI hits with `unsloth studio update --local 2>&1 | tee
        # logs/update.log`. Single-quote escaping handles paths that
        # contain apostrophes.
        script_pwsh_literal = str(script).replace("'", "''")
        powershell_args.extend(
            [
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"& '{script_pwsh_literal}' *>&1",
            ]
        )
        # Explicitly hand stdin/stdout/stderr to the child so the
        # CI tee actually sees setup.ps1's output. Without this,
        # subprocess.run on Windows uses close_fds=True (default,
        # since Python 3.7) which sets bInheritHandles=False on
        # CreateProcess. With CREATE_NO_WINDOW also set (via
        # _windows_hidden_subprocess_kwargs in non-TTY runs), the
        # child has neither a console nor any inherited std
        # handles, so PowerShell's Write-Host -- and even
        # [Console]::Out.WriteLine -- writes to nothing. Passing
        # stdout=sys.stdout / stderr=sys.stderr makes Python set up
        # PROC_THREAD_ATTRIBUTE_HANDLE_LIST with the std handles
        # explicitly inheritable, which works alongside
        # CREATE_NO_WINDOW. Empty update.log on the windows-latest
        # CI was the smoking gun (run 25533694490 and 25534292239).
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            env = env,
            stdin = _stream_for_subprocess(sys.stdin),
            stdout = _stream_for_subprocess(sys.stdout),
            stderr = _stream_for_subprocess(sys.stderr),
            **_windows_hidden_subprocess_kwargs(),
        )
    else:
        result = subprocess.run(["bash", str(script)], env = env)

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


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
            ps_argv.extend(
                ["-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden"]
            )

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
                        typer.echo(
                            f"  refresh-launcher  install.ps1 exited {result.returncode}"
                        )
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
            typer.echo(
                f"  refresh-launcher  skipped: could not fetch {installer_url} ({exc})"
            )
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
                typer.echo(
                    f"  refresh-launcher  skipped: powershell exec failed ({exc})"
                )
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
                    typer.echo(
                        f"  refresh-launcher  install.sh exited {result.returncode}"
                    )
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
        typer.echo(
            f"  refresh-launcher  skipped: could not fetch {installer_url} ({exc})"
        )
        return

    try:
        result = subprocess.run(
            ["bash", "-s", "--", *args],
            input = installer,
            env = env,
            check = False,
        )
        if result.returncode != 0:
            typer.echo(
                f"  refresh-launcher  fetched install.sh exited {result.returncode}"
            )
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
    # Re-export UNSLOTH_STUDIO_HOME for env-mode installs so the refresh
    # subprocess resolves the same install root the user originally chose.
    _ensure_studio_env_exported()
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
    _release_self_exe_lock_windows()
    try:
        _run_setup_script(verbose = verbose)
    except BaseException:
        # Restore unsloth.exe from .deleteme if setup failed before pip
        # produced a replacement; otherwise the user has no CLI for recovery.
        _restore_self_exe_lock_windows()
        raise
    # On Windows clear the .deleteme orphan now that pip wrote a fresh
    # unsloth.exe; on next update os.replace would overwrite it anyway,
    # but leaving a stale binary around invites cross-version restore
    # confusion from _restore_self_exe_lock_windows.
    _cleanup_self_exe_lock_windows()
    # Tauri desktop owns its own bundle entries; skip CLI launcher refresh
    # so a Tauri-initiated update doesn't create duplicate shortcuts.
    if os.environ.get("UNSLOTH_TAURI_UPDATE") == "1":
        if verbose:
            typer.echo("  refresh-launcher  skipped (Tauri update)")
        return
    _refresh_desktop_shortcuts(verbose = verbose)


def _release_self_exe_lock_windows() -> None:
    """Rename running unsloth.exe so pip can replace it. setup.ps1 also retries."""
    if platform.system() != "Windows":
        return
    try:
        venv_scripts = Path(sys.executable).resolve().parent
    except OSError:
        return
    exe = venv_scripts / "unsloth.exe"
    if not exe.exists():
        return
    stale = exe.with_suffix(".exe.deleteme")
    try:
        # os.replace is atomic-overwrite on Windows; os.rename would raise
        # FileExistsError if a prior aborted update left a .deleteme behind.
        os.replace(exe, stale)
    except OSError as e:
        # Not fatal; setup.ps1 retries from a sibling process.
        print(f"[update] could not rename {exe.name} -> {stale.name}: {e}")


def _restore_self_exe_lock_windows() -> None:
    """If setup failed before pip wrote a working unsloth.exe, restore .deleteme."""
    if platform.system() != "Windows":
        return
    try:
        venv_scripts = Path(sys.executable).resolve().parent
    except OSError:
        return
    exe = venv_scripts / "unsloth.exe"
    stale = exe.with_suffix(".exe.deleteme")
    if not stale.exists():
        return
    # Treat a missing or zero-byte exe as "pip didn't produce a usable
    # replacement"; otherwise leave the new binary alone.
    if exe.exists():
        try:
            if exe.stat().st_size > 0:
                return
        except OSError:
            return
    try:
        os.replace(stale, exe)
    except OSError as e:
        print(f"[update] could not restore {stale.name} -> {exe.name}: {e}")


def _cleanup_self_exe_lock_windows() -> None:
    """Remove the .deleteme orphan after a successful update on Windows."""
    if platform.system() != "Windows":
        return
    try:
        venv_scripts = Path(sys.executable).resolve().parent
    except OSError:
        return
    stale = (venv_scripts / "unsloth.exe").with_suffix(".exe.deleteme")
    try:
        stale.unlink(missing_ok = True)
    except OSError:
        pass


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
        "desktop_manageability_version": 1,
        "supports_provision_desktop_auth": True,
        "supports_api_only": True,
        "supports_desktop_backend_ownership": True,
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
    pw_file = auth_dir / ".bootstrap_password"

    if not db_file.exists():
        typer.echo("No auth database found -- nothing to reset.")
        raise typer.Exit(0)

    db_file.unlink(missing_ok = True)
    pw_file.unlink(missing_ok = True)

    typer.echo("Auth database deleted. Restart Unsloth Studio to get a new password.")

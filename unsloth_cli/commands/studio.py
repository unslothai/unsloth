# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
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
):
    """Launch the Unsloth Studio server."""
    if ctx.invoked_subcommand is not None:
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


# ── unsloth studio run ───────────────────────────────────────────────


def _split_repo_variant(model_arg: str) -> tuple[str, Optional[str]]:
    """Split ``org/name:variant`` HF-style identifiers into (repo, variant).

    Mirrors llama.cpp's ``-hf <repo>:<quant>`` convention so users can
    write ``unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL`` instead of passing
    ``--gguf-variant`` separately. Local paths (absolute, ``./``,
    ``~/``, Windows drive letters) and identifiers without a ``:``
    suffix are returned verbatim.
    """
    s = model_arg.strip()
    if not s:
        return s, None
    if s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    # Windows drive letter (e.g. "C:\\path" or "C:/path") -- the colon
    # here is a path separator, not a variant suffix.
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant:
        return s, None
    # A real quant label has no slashes; ``foo:bar/baz`` is not
    # ``repo:variant`` syntax.
    if "/" in variant:
        return s, None
    return repo, variant


@studio_app.command(
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run(
    ctx: typer.Context,
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        "-hf",
        "-hfr",
        "--hf-repo",
        help = (
            "Model path or HF repo. Accepts llama.cpp-style "
            "`org/repo:variant` syntax. The `-hf` / `--hf-repo` aliases "
            "match llama-server's spelling."
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
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
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
):
    """Start Studio, load a model, and print an API key -- one-liner server.

    Any flag this command does not recognize is forwarded verbatim to
    the underlying llama-server (GGUF only). Studio-managed flags
    (--port, -c / --ctx-size, --api-key, -ngl, --jinja, --flash-attn,
    --no-context-shift, model-identity flags, ...) are rejected with
    HTTP 400.

    Example:
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --gguf-variant UD-Q4_K_XL
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --top-k 20 --seed 42
        unsloth studio run --model some-model --chat-template-file /path/to/tpl.jinja
    """
    extra_llama_args: List[str] = list(ctx.args) if ctx.args else []

    # ── 0. Parse llama.cpp-style ``repo:variant`` syntax in --model. ───
    # Lets users write ``--model unsloth/foo-GGUF:UD-Q4_K_XL`` instead
    # of pairing ``--model`` with ``--gguf-variant``. If both are given
    # and disagree, fail loudly instead of silently picking one.
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

    # ── Resolve the server-side tool policy. The y/N prompt (if any)
    # runs in the outer process so the re-exec'd child never re-prompts.
    from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy

    enable_tools = resolve_tool_policy(
        host = host,
        flag = enable_tools,
        yes = yes,
        silent = silent,
    )

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
        # Forward the resolved tool policy (always concrete True/False
        # at this point — the resolver above ran before the re-exec).
        if enable_tools:
            args.append("--enable-tools")
        else:
            args.append("--disable-tools")
        # Forward --yes whenever the parent already cleared the prompt
        # (either operator passed --yes, or the parent's resolver
        # accepted the network-bind confirmation). Otherwise the child
        # re-runs the resolver and prompts a second time.
        if yes or (enable_tools and is_external_host(host)):
            args.append("--yes")
        # Forward unknown args (llama-server pass-through) to the
        # re-exec'd command so the studio venv sees them in ctx.args
        # and the re-execed run() can include them in the load payload.
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

    run_kwargs = dict(host = host, port = port, silent = True, llama_parallel_slots = 4)
    if frontend is not None:
        run_kwargs["frontend_path"] = frontend
    app = run_server(**run_kwargs)
    actual_port = getattr(app.state, "server_port", port) or port

    # ── Apply the resolved tool policy as a process-level override.
    # Must use the same import path the route handlers use --
    # `studio/backend/run.py` adds `studio/backend/` to sys.path so the
    # routes import this module as top-level `state.tool_policy`. If we
    # imported via `studio.backend.state.tool_policy` instead, Python
    # would cache two different module objects with two different
    # `_tool_policy` globals, and the gates would never see our value.
    from state.tool_policy import set_tool_policy

    set_tool_policy(enable_tools)

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
            llama_extra_args = extra_llama_args,
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

    # Claude orange (Claude Code's brand color) for tool-policy notices
    # so they stand out from the surrounding banner. Always printed --
    # even under --silent / --yes -- so the operator never misses the
    # current tool-execution status.
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
        # Silent mode still prints the essentials (URL, API key) plus
        # the orange tool-status notice so the operator never loses
        # visibility into the security-relevant policy.
        typer.echo(f"URL:     {base_url}")
        typer.echo(f"API Key: {api_key}")
        typer.secho(_tool_notice, fg = _tool_notice_fg, bold = True)

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

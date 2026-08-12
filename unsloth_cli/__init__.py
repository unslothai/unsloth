# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os as _os
import sys as _sys

# Are we the `unsloth` console script, rather than a library import? Both the
# stream guard below and the `-np<N>` rewrite further down are entry-point
# behaviour and must not reach into a host application that imports us.
_entry_base = _os.path.basename(_sys.argv[0]).lower() if _sys.argv else ""
_is_entry_point = _entry_base in {"unsloth", "unsloth.exe"}

# Typer renders help via rich, whose box characters cp1252 and cp437 cannot encode,
# so `unsloth --help` dies once stdout is a pipe or a file. Windows gets UTF-8, as
# unsloth/__init__ already does; elsewhere the caller's encoding is kept and only
# the error handler is relaxed, so an explicit PYTHONIOENCODING still picks the
# bytes and only loses unencodable glyphs. Before typer, which binds the stream.
if _is_entry_point:
    _to_utf8 = _sys.platform == "win32"
    for _name in ("stdout", "stderr"):
        _stream = getattr(_sys, _name, None)
        try:
            if "utf" not in (_stream.encoding or "").lower():
                _stream.reconfigure(encoding = "utf-8" if _to_utf8 else None, errors = "replace")
        except Exception:
            pass
    del _name, _stream, _to_utf8

from unsloth_cli._system_dir_guard import check_working_directory as _check_working_directory

# Running from System32 or any subdir WILL cause errors if not prevented. A
# command that cannot be affected by the folder (the ones Unsloth Desktop spawns,
# see issue #8510) moves out of it; everything else stops in the callback below.
#
# This runs before the command imports because unsloth_cli.commands.studio
# resolves STUDIO_HOME at import time, and a relative UNSLOTH_STUDIO_HOME would
# otherwise be pinned to the folder we are about to leave. The message is held
# until typer exists to render it. A library import reaches the same check from
# the callback instead, where argv is still the only thing to go on.
_startup_guard = (
    _check_working_directory(_sys.argv[1:], _os.environ, _sys.platform) if _is_entry_point else None
)

import typer
from importlib.metadata import version as package_version, PackageNotFoundError


from unsloth_cli.commands.train import train
from unsloth_cli.commands.inference import inference
from unsloth_cli.commands.chat import chat
from unsloth_cli.commands.start import start_app
from unsloth_cli.commands.export import export, list_checkpoints
from unsloth_cli.commands.studio import (
    run as studio_run,
    studio_app,
    _expand_attached_np_short,
)


# Canonicalise `-np<N>` only under the `unsloth` console-script;
# third-party scripts that import unsloth_cli keep their argv intact.
if _is_entry_point:
    _expand_attached_np_short()
del _entry_base, _is_entry_point


def show_version(value: bool):
    if value:
        try:
            version = package_version("unsloth")
        except PackageNotFoundError:
            version = "unknown"
        typer.echo(f"unsloth {version}")
        raise typer.Exit()


app = typer.Typer(
    help = "Command-line interface for Unsloth training, inference, and export.",
    context_settings = {"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback = show_version,
        is_eager = True,
        help = "Show version and exit.",
    ),
):
    # Consume the import-time result once. A host that imports us and calls the
    # app more than once gets a fresh check each time, since it can chdir between
    # calls; the console script has already been checked before this point.
    global _startup_guard
    _guard, _startup_guard = _startup_guard, None
    if _guard is None:
        _guard = _check_working_directory(_sys.argv[1:], _os.environ, _sys.platform)
    _message, _colour, _fatal = _guard
    if _message is not None:
        typer.secho(_message, fg = _colour, err = True)
    if _fatal:
        raise typer.Exit(code = 1)


app.command()(train)
app.command()(inference)
app.command()(chat)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")
app.add_typer(
    start_app,
    name = "start",
    help = "Start a coding agent (Claude, Codex, OpenClaw, OpenCode, Hermes, Pi) against Unsloth.",
)
# Backwards-compatible hidden alias: `unsloth connect` routes to `unsloth start`.
app.add_typer(
    start_app,
    name = "connect",
    hidden = True,
    help = "Deprecated alias for `unsloth start`.",
)

# Top-level `unsloth run` aliases `unsloth studio run`; same context
# so unknown flags still pass through to llama-server.
app.command(
    "run",
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    help = "Alias for `unsloth studio run`.",
)(studio_run)

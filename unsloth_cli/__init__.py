# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os as _os
import sys as _sys

import typer
from importlib.metadata import version as package_version, PackageNotFoundError


from unsloth_cli.commands.train import train
from unsloth_cli.commands.inference import inference
from unsloth_cli.commands.chat import chat
from unsloth_cli.commands.start import start_app
from unsloth_cli.commands.export import export, list_checkpoints
from unsloth_cli.commands.eval import evaluate as eval_command
from unsloth_cli.commands.studio import (
    run as studio_run,
    studio_app,
    _expand_attached_np_short,
)


# Canonicalise `-np<N>` only under the `unsloth` console-script;
# third-party scripts that import unsloth_cli keep their argv intact.
_entry_base = _os.path.basename(_sys.argv[0]).lower() if _sys.argv else ""
if _entry_base in {"unsloth", "unsloth.exe"}:
    _expand_attached_np_short()
del _entry_base


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
    if (
        _sys.platform == "win32"
    ):  # this block catches unsloth running inside of System32 or any subdirs, this WILL cause errors if not prevented.
        _cwd = _os.path.normcase(_os.path.normpath(_os.getcwd()))
        _system32 = _os.path.normcase(
            _os.path.normpath(_os.path.join(_os.environ.get("WINDIR", r"C:\Windows"), "System32"))
        )
        if _cwd == _system32 or _cwd.startswith(_system32 + _os.sep):
            typer.secho(
                "Refusing to run Unsloth inside System32 as it will lead to Errors.\n"
                "cd to a normal working directory and try again.",
                fg = "red",
                err = True,
            )
            raise typer.Exit(code = 1)


app.command()(train)
app.command()(inference)
app.command()(chat)
app.command()(export)
app.command("eval")(eval_command)
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

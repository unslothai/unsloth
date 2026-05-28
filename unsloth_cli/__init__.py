# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os.path as _osp
import sys as _sys

import typer
from importlib.metadata import version as package_version, PackageNotFoundError


from unsloth_cli.commands.train import train
from unsloth_cli.commands.inference import inference
from unsloth_cli.commands.export import export, list_checkpoints
from unsloth_cli.commands.studio import (
    run as studio_run,
    studio_app,
    _expand_attached_np_short,
)


# Canonicalise `-np<N>` only under the `unsloth` console-script;
# third-party scripts that import unsloth_cli keep their argv intact.
_entry_base = _osp.basename(_sys.argv[0]).lower() if _sys.argv else ""
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
    help="Command-line interface for Unsloth training, inference, and export.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=show_version,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


app.command()(train)
app.command()(inference)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.add_typer(studio_app, name="studio", help="Unsloth Studio commands.")

# Top-level `unsloth run` aliases `unsloth studio run`; same context
# so unknown flags still pass through to llama-server.
app.command(
    "run",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    help="Alias for `unsloth studio run`.",
)(studio_run)

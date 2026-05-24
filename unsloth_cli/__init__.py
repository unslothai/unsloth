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


# Run the studio `-np<N>` argv canonicalisation only when invoked through
# a known entry-point launcher; tests and notebooks that import this
# module must not have their argv mutated. Exact-basename match (not
# endswith) so an unrelated third-party `mycli.py` that happens to
# import unsloth_cli isn't side-effected.
_entry_base = _osp.basename(_sys.argv[0]).lower() if _sys.argv else ""
if _entry_base in {
    "unsloth",
    "unsloth.exe",
    "unsloth-cli",
    "unsloth-cli.exe",
    "cli.py",
    "unsloth-cli.py",
}:
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
    pass


app.command()(train)
app.command()(inference)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")

# Top-level alias: `unsloth run ...` is equivalent to `unsloth studio run ...`.
# Same context_settings as the studio_app registration so unknown flags
# still pass through to llama-server.
app.command(
    "run",
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    help = "Alias for `unsloth studio run`.",
)(studio_run)

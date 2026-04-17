# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import typer

from unsloth_cli.commands.train import train
from unsloth_cli.commands.inference import inference
from unsloth_cli.commands.export import export, list_checkpoints
from unsloth_cli.commands.studio import studio_app

app = typer.Typer(
    help = "Command-line interface for Unsloth training, inference, and export.",
    context_settings = {"help_option_names": ["-h", "--help"]},
)

app.command()(train)
app.command()(inference)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")

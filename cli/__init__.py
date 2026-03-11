# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

import typer

from cli.commands.train import train
from cli.commands.inference import inference
from cli.commands.export import export, list_checkpoints
from cli.commands.ui import ui
from cli.commands.studio import studio_app

app = typer.Typer(
    help = "Command-line interface for Unsloth training, inference, and export.",
    context_settings = {"help_option_names": ["-h", "--help"]},
)

app.command()(train)
app.command()(inference)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.command()(ui)
app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")

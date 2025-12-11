import typer

from cli.commands.train import train
from cli.commands.inference import inference
from cli.commands.export import export, list_checkpoints

app = typer.Typer(
    help="Command-line interface for Unsloth training, inference, and export.",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command()(train)
app.command()(inference)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)

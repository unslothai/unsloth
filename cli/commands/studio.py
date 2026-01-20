import typer


def studio(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the UI server on."),
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Host address to bind to."),
    share: bool = typer.Option(True, "--share", "-s", help="Create a public Gradio share link."),
):
    """Launch the Unsloth web UI for training, inference, and export."""
    from app import demo, script_dir

    typer.echo(f"Starting Unsloth UI on http://{host}:{port}")

    demo.launch(
        share=share,
        server_port=port,
        server_name=host,
        favicon_path=f"{script_dir}/assets/favicon-32x32.png",
    )

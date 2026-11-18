# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import typer

from unsloth_cli._npu import NpuCliError, fetch_npu_models, format_npu_models, npu_status_problem


def _fail(message: str):
    typer.echo(f"Error: {message}", err = True)
    raise typer.Exit(code = 1)


def _local_npu_status() -> dict:
    """Studio's /api/npu/status computed in this process; it reads files and never starts the runtime."""
    from unsloth_cli._inference import ensure_studio_backend_path

    ensure_studio_backend_path(seed_cache_env = False)
    from core.inference.npu_backend import get_npu_backend

    return get_npu_backend().status()


def list_npu_models():
    """List the AMD NPU models, with whether each is downloaded and fits this PC's memory."""
    from unsloth_cli._inference import (
        HttpChatBackend,
        _studio_token,
        find_studio_server,
        is_loopback_url,
        verify_studio_identity,
    )

    base = find_studio_server()
    if not base:
        try:
            problem = npu_status_problem(_local_npu_status())
        except Exception:
            problem = None
        # The -FLM catalog comes from the NPU runtime, which only a running Studio starts.
        _fail(
            problem
            or "Unsloth Studio isn't running. Start it with `unsloth studio` and run this again."
        )
    if not is_loopback_url(base) or not verify_studio_identity(base):
        _fail(f"Can't read the NPU models from {base}: it isn't this machine's Unsloth Studio.")
    token = _studio_token()
    if not token:
        _fail("Couldn't sign in to Unsloth Studio (is Unsloth set up here?).")
    backend = HttpChatBackend(base, token)
    try:
        status, models = fetch_npu_models(
            lambda method, path, payload, timeout: backend._request(
                method, path, payload, timeout = timeout
            )
        )
    except NpuCliError as exc:
        _fail(str(exc))
    if not models:
        typer.echo("No NPU models found.")
        raise typer.Exit()
    for line in format_npu_models(models, status.get("memory_window_gb")):
        typer.echo(line)
    typer.echo("\nChat with one: unsloth chat <ID>")

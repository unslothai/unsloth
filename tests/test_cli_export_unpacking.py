# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for unsloth_cli.commands.export.

ExportOrchestrator.export_* now returns (success, message, output_path) so the
frontend can show the realpath, but the CLI still unpacked two values, crashing
every `unsloth export` with "too many values to unpack (expected 2)". These
tests pin the CLI to the 3-tuple contract via a fake ExportBackend injected into
sys.modules (the CLI's deferred import binds to it), asserting exit_code == 0
per --format."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Fake ExportBackend
# ---------------------------------------------------------------------------


class _FakeExportBackend:
    """Stand-in for ExportBackend: export_* return the new 3-tuple;
    load_checkpoint keeps its 2-tuple shape."""

    def __init__(self) -> None:
        self.loaded: str | None = None

    def load_checkpoint(self, **kwargs):
        self.loaded = kwargs.get("checkpoint_path")
        return True, f"Loaded {self.loaded}"

    def scan_checkpoints(self, **kwargs):
        return []

    def export_merged_model(self, **kwargs):
        return True, "merged ok", str(Path(kwargs["save_directory"]).resolve())

    def export_base_model(self, **kwargs):
        return True, "base ok", str(Path(kwargs["save_directory"]).resolve())

    def export_gguf(self, **kwargs):
        return True, "gguf ok", str(Path(kwargs["save_directory"]).resolve())

    def export_lora_adapter(self, **kwargs):
        return True, "lora ok", str(Path(kwargs["save_directory"]).resolve())


def _install_fake_studio_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake studio.backend.core.export into sys.modules. The CLI imports
    ExportBackend lazily, so this steers it at the fake; parent packages are
    stubbed too so import machinery skips the real structlog-dependent tree."""
    for name in ("studio", "studio.backend", "studio.backend.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    fake_mod = types.ModuleType("studio.backend.core.export")
    fake_mod.ExportBackend = _FakeExportBackend
    monkeypatch.setitem(sys.modules, "studio.backend.core.export", fake_mod)

    # Drop cached CLI module so export()'s deferred import re-resolves the fake.
    monkeypatch.delitem(sys.modules, "unsloth_cli.commands.export", raising = False)


@pytest.fixture
def cli_app(monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    """Typer app wrapping unsloth_cli.commands.export.export."""
    _install_fake_studio_backend(monkeypatch)
    from unsloth_cli.commands import export as export_cmd

    app = typer.Typer()
    app.command("export")(export_cmd.export)

    # Typer flattens a single-command app into that command, which would
    # make argv[0] ("export") look like an extra positional argument to
    # the test invocation. Register a harmless second command so Typer
    # keeps "export" as a real subcommand and the tests drive the
    # intended code path.
    @app.command("noop")
    def _noop() -> None:  # pragma: no cover - only exists to pin routing
        pass

    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# The actual regression tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "format_flag,quant_flag",
    [
        ("merged-16bit", None),
        ("merged-4bit", None),
        ("gguf", "q4_k_m"),
        ("lora", None),
    ],
)
def test_cli_export_unpacks_three_tuple(
    cli_app: typer.Typer,
    runner: CliRunner,
    tmp_path: Path,
    format_flag: str,
    quant_flag: str | None,
) -> None:
    """Each --format path must unpack (success, message, output_path)
    without raising ValueError. Pre-fix, every parametrized case fails
    with 'too many values to unpack (expected 2)'.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    out = tmp_path / "out"

    cli_args = ["export", str(ckpt), str(out), "--format", format_flag]
    if quant_flag is not None:
        cli_args += ["--quantization", quant_flag]

    result = runner.invoke(cli_app, cli_args)

    assert result.exit_code == 0, (
        f"CLI exited with code {result.exit_code} for --format {format_flag}.\n"
        f"Output:\n{result.output}\n"
        f"Exception: {result.exception!r}"
    )
    # Sanity: the success message from the fake backend should reach stdout.
    expected_prefix = format_flag.split("-")[0]
    assert f"{expected_prefix} ok" in result.output

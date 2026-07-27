# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for unsloth_cli.commands.export: pin the CLI to the export_* 3-tuple contract (was unpacking 2, crashing every `unsloth export`) via a fake ExportBackend in sys.modules."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner


class _FakeExportBackend:
    """Stand-in for ExportBackend: export_* return the 3-tuple, load_checkpoint stays a 2-tuple."""

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
    """Inject a fake studio.backend.core.export into sys.modules so the CLI's lazy import binds to it; parent packages stubbed to skip the structlog-dependent tree."""
    for name in ("studio", "studio.backend", "studio.backend.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    fake_mod = types.ModuleType("studio.backend.core.export")
    fake_mod.ExportBackend = _FakeExportBackend
    monkeypatch.setitem(sys.modules, "studio.backend.core.export", fake_mod)

    # Drop the cached CLI module so its deferred import re-resolves the fake.
    monkeypatch.delitem(sys.modules, "unsloth_cli.commands.export", raising = False)


@pytest.fixture
def cli_app(monkeypatch: pytest.MonkeyPatch) -> typer.Typer:
    """Typer app wrapping unsloth_cli.commands.export.export."""
    _install_fake_studio_backend(monkeypatch)
    from unsloth_cli.commands import export as export_cmd

    app = typer.Typer()
    app.command("export")(export_cmd.export)

    # Typer flattens a single-command app, making "export" look like a stray positional;
    # a harmless second command keeps "export" a real subcommand.
    @app.command("noop")
    def _noop() -> None:  # pragma: no cover - only exists to pin routing
        pass

    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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
    """Each --format path unpacks the 3-tuple without ValueError (pre-fix: 'too many values to unpack (expected 2)')."""
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
    # Fake backend's success message should reach stdout.
    expected_prefix = format_flag.split("-")[0]
    assert f"{expected_prefix} ok" in result.output

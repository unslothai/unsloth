# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import sys
import types

import pytest
import typer
from typer.testing import CliRunner

from unsloth_cli.commands import export as export_command


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], None),
        (["--format", "gguf"], None),
        (["--load-in-4bit"], True),
        (["--no-load-in-4bit"], False),
        (["--format", "merged-4bit"], True),
        (["--format", "merged-4bit", "--no-load-in-4bit"], False),
    ],
    ids = [
        "default",
        "gguf_default",
        "explicit_4bit",
        "explicit_16bit",
        "merged_4bit_default",
        "merged_4bit_explicit_16bit",
    ],
)
def test_export_leaves_load_in_4bit_to_the_backend_unless_set(
    monkeypatch, tmp_path, flags, expected
):
    calls = {}

    class ExportBackend:
        def load_checkpoint(self, **kwargs):
            calls.update(kwargs)
            return True, "loaded"

        def export_merged_model(self, **kwargs):
            return True, "exported", kwargs["save_directory"]

        def export_gguf(self, **kwargs):
            return True, "exported", kwargs["save_directory"]

    module = types.ModuleType("studio.backend.core.export")
    module.ExportBackend = ExportBackend
    monkeypatch.setitem(sys.modules, "studio.backend.core.export", module)
    monkeypatch.setattr(
        export_command, "studio_backend_imports", lambda *a, **k: contextlib.nullcontext()
    )

    app = typer.Typer()
    app.command()(export_command.export)
    result = CliRunner().invoke(app, [str(tmp_path), str(tmp_path / "out"), *flags])

    assert result.exit_code == 0, result.output
    assert calls["load_in_4bit"] is expected

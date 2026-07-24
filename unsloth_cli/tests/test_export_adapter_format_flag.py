# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mac pass-through and off-Mac rejection for --adapter-format."""

import sys
import types
from pathlib import Path

import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _FakeBackend:
    def __init__(self, lora_result = (True, "exported", "/tmp/out")):
        self.lora_result, self.lora_kwargs = lora_result, None

    def load_checkpoint(self, **kwargs):
        return True, "loaded"

    def export_lora_adapter(self, **kwargs):
        self.lora_kwargs = kwargs
        return self.lora_result


def _run(
    monkeypatch,
    args,
    is_mac,
    backend = None,
):
    from unsloth_cli.commands import export as export_mod

    backend = backend or _FakeBackend()
    monkeypatch.setattr(export_mod, "_is_apple_silicon", lambda: is_mac)
    fake_core = types.SimpleNamespace(ExportBackend = lambda: backend)
    monkeypatch.setitem(sys.modules, "studio.backend.core.export", fake_core)
    app = typer.Typer()
    app.command()(export_mod.export)
    result = CliRunner().invoke(app, ["/tmp/ckpt", "/tmp/out", "--format", "lora", *args])
    return result, backend


def test_flag_contract(monkeypatch):
    result, backend = _run(monkeypatch, ["--adapter-format", "peft"], is_mac = True)
    assert result.exit_code == 0, result.output
    assert backend.lora_kwargs["adapter_format"] == "peft"
    result, backend = _run(monkeypatch, ["--adapter-format", "mlx"], is_mac = False)
    assert (
        result.exit_code == 2 and "Apple-silicon" in result.stderr and backend.lora_kwargs is None
    )

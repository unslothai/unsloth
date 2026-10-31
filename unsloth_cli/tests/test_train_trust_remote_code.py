# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Models that ship custom modelling code (LiquidAI/LFM3-Omni-2B and friends) cannot be loaded
# without trust_remote_code, and the loader's own error tells the user to pass it. Until this flag
# existed the CLI had no way to answer that, so the model was unreachable from `unsloth train`.

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import typer
from typer.testing import CliRunner

from unsloth_cli.config import TrainingConfig
from unsloth_cli.options import add_options_from_config


def _app_for(captured: dict):
    """A one-command app carrying the real training options."""

    app = typer.Typer()

    @app.command()
    @add_options_from_config(TrainingConfig)
    def run(config_overrides: dict = None):
        captured.update(config_overrides or {})

    return app


def test_training_config_defaults_to_not_running_remote_code():
    assert TrainingConfig().trust_remote_code is False


def test_train_accepts_trust_remote_code():
    captured: dict = {}
    result = CliRunner().invoke(_app_for(captured), ["--trust-remote-code"])

    assert result.exit_code == 0, result.output
    assert captured.get("trust_remote_code") is True


def test_trust_remote_code_is_off_unless_asked():
    captured: dict = {}
    result = CliRunner().invoke(_app_for(captured), [])

    assert result.exit_code == 0, result.output
    # Omitted options are left out of the overrides, so the model default stands.
    assert not captured.get("trust_remote_code")

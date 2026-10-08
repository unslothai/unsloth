# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth train` flags must be validated like the same keys in a config file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.config import Config, ConfigError  # noqa: E402


def _train_app():
    from unsloth_cli.commands.train import train

    app = typer.Typer()
    app.command()(train)
    return app


def _dry_run(*args: str):
    return CliRunner().invoke(_train_app(), ["--model", "m", "--dataset", "d", "--dry-run", *args])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--training-type", "lroa"),
        ("--format-type", "chatlm"),
        ("--gradient-checkpointing", "maybe"),
    ],
)
def test_an_invalid_flag_value_exits_2_like_the_same_value_in_a_config(flag, value):
    result = _dry_run(flag, value)

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.output.startswith("Error: ")
    assert flag in result.output


def test_every_invalid_flag_is_reported_at_once():
    result = _dry_run(
        "--training-type", "lroa", "--format-type", "chatlm", "--gradient-checkpointing", "maybe"
    )

    assert result.exit_code == 2
    for flag in ("--training-type", "--format-type", "--gradient-checkpointing"):
        assert flag in result.output


def test_an_invalid_flag_is_rejected_over_a_valid_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  training_type: lora\n", encoding = "utf-8")

    result = _dry_run("--config", str(path), "--training-type", "lroa")

    assert result.exit_code == 2
    assert "--training-type" in result.output


def test_valid_flags_still_override_the_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  training_type: lora\n  num_epochs: 1\n", encoding = "utf-8")

    result = _dry_run(
        "--config",
        str(path),
        "--training-type",
        "full",
        "--format-type",
        "chatml",
        "--gradient-checkpointing",
        "none",
        "--num-epochs",
        "5",
    )

    assert result.exit_code == 0, result.output
    assert "training_type: full" in result.output
    assert "format_type: chatml" in result.output
    assert "gradient_checkpointing: none" in result.output
    assert "num_epochs: 5" in result.output


def test_apply_overrides_raises_config_error_and_names_the_flag():
    with pytest.raises(ConfigError) as excinfo:
        Config().apply_overrides(training_type = "lroa")

    message = str(excinfo.value)
    assert "--training-type" in message
    assert "'lora' or 'full'" in message


@pytest.mark.parametrize(
    ("flag", "value", "resolved"),
    [
        ("--training-type", "LoRA", "training_type: lora"),
        ("--format-type", "ChatML", "format_type: chatml"),
        ("--format-type", "raw", "format_type: raw"),
        ("--gradient-checkpointing", "yes", "gradient_checkpointing: 'true'"),
        ("--gradient-checkpointing", "false", "gradient_checkpointing: none"),
        ("--gradient-checkpointing", "off", "gradient_checkpointing: none"),
        ("--gradient-checkpointing", "", "gradient_checkpointing: unsloth"),
    ],
)
def test_spellings_the_trainer_already_accepted_still_work(flag, value, resolved):
    result = _dry_run(flag, value)

    assert result.exit_code == 0, result.output
    assert resolved in result.output


def test_a_yaml_boolean_gradient_checkpointing_is_accepted(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  gradient_checkpointing: false\n", encoding = "utf-8")

    result = _dry_run("--config", str(path))

    assert result.exit_code == 0, result.output
    assert "gradient_checkpointing: none" in result.output

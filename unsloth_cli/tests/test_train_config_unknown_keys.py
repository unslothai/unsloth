# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth train --config` used to drop every key it did not recognise, so a
config that put `learning_rate` at the top level (where the CLI flag lives) or
spelled a key CLI-style inside a section trained on defaults and said nothing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.config import ConfigError, load_config  # noqa: E402

_SHIPPED_CONFIGS = _REPO_ROOT / "studio" / "backend" / "assets" / "configs"


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding = "utf-8")
    return path


def _train_app():
    from unsloth_cli.commands.train import train

    app = typer.Typer()
    app.command()(train)
    return app


def test_top_level_flat_key_is_rejected_and_names_its_section(tmp_path):
    path = _write(tmp_path, "model: unsloth/Qwen2.5-0.5B\nlearning_rate: 5e-5\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "learning_rate" in message
    assert "training:" in message


def test_hyphenated_top_level_key_is_rejected_and_names_its_section(tmp_path):
    path = _write(tmp_path, "learning-rate: 5e-5\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "learning-rate" in message
    assert "training:" in message


def test_key_in_the_wrong_section_is_rejected_and_names_the_right_one(tmp_path):
    path = _write(tmp_path, "training:\n  lora_r: 8\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "lora_r" in message
    assert "'training'" in message
    assert "'lora:'" in message


def test_cli_spelling_inside_a_section_is_rejected_with_the_yaml_spelling(tmp_path):
    path = _write(tmp_path, "training:\n  num-epochs: 9\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "num-epochs" in message
    assert "num_epochs" in message


def test_valid_config_still_loads(tmp_path):
    path = _write(
        tmp_path,
        "model: unsloth/Qwen2.5-0.5B\n"
        "data:\n"
        "  dataset: tatsu-lab/alpaca\n"
        "training:\n"
        "  num_epochs: 9\n"
        "  learning_rate: 5e-5\n"
        "lora:\n"
        "  lora_r: 8\n",
    )

    cfg = load_config(path)

    assert cfg.model == "unsloth/Qwen2.5-0.5B"
    assert cfg.data.dataset == "tatsu-lab/alpaca"
    assert cfg.training.num_epochs == 9
    assert cfg.training.learning_rate == 5e-5
    assert cfg.lora.lora_r == 8


@pytest.mark.parametrize("name", ["full_finetune.yaml", "lora_text.yaml", "vision_lora.yaml"])
def test_shipped_example_configs_still_load(name):
    assert load_config(_SHIPPED_CONFIGS / name).model


def test_train_exits_2_instead_of_tracebacking_on_an_unknown_key(tmp_path):
    path = _write(tmp_path, "model: unsloth/Qwen2.5-0.5B\nlearning_rate: 5e-5\n")

    result = CliRunner().invoke(_train_app(), ["--config", str(path)])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "learning_rate" in result.output
    assert "training:" in result.output


def test_unparseable_yaml_reports_cleanly_instead_of_tracebacking(tmp_path):
    path = _write(tmp_path, "training:\n  num_epochs: 3\n   learning_rate: 1\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert "Could not parse config file" in str(excinfo.value)


def test_unparseable_json_reports_cleanly_instead_of_tracebacking(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{oops}", encoding = "utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert "Could not parse config file" in str(excinfo.value)


def test_a_top_level_list_reports_cleanly_instead_of_tracebacking(tmp_path):
    path = _write(tmp_path, "- model: unsloth/Qwen2.5-0.5B\n- model: unsloth/Qwen2.5-1.5B\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert "must be a mapping" in message
    assert "list" in message


def test_an_empty_config_still_loads_defaults(tmp_path):
    assert load_config(_write(tmp_path, "")).training.num_epochs == 3


@pytest.mark.parametrize(
    "body",
    [
        "training:\n  num_epochs: 3\n   learning_rate: 1\n",
        "- model: unsloth/Qwen2.5-0.5B\n",
    ],
)
def test_train_exits_2_on_an_unloadable_config(tmp_path, body):
    result = CliRunner().invoke(_train_app(), ["--config", str(_write(tmp_path, body))])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.output.startswith("Error: ")

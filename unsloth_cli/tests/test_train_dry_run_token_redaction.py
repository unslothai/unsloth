# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth train --config cfg.yaml --dry-run` dumped the resolved config straight to stdout,
so `logging.hf_token` and `logging.wandb_token` printed in cleartext into CI logs and notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FAKE_HF_TOKEN = "not-a-real-hf-token-0000"
_FAKE_WANDB_TOKEN = "not-a-real-wandb-token-1111"


@pytest.fixture(autouse=True)
def _no_ambient_tokens(monkeypatch):
    """HF_TOKEN/WANDB_API_KEY are envvars on --hf-token/--wandb-token, and the generated
    options feed them into cfg.logging, so a developer who exports one fails the unset cases."""
    for var in ("HF_TOKEN", "WANDB_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _train_app():
    from unsloth_cli.commands.train import train

    app = typer.Typer()
    app.command()(train)
    return app


def _dry_run(path: Path):
    return CliRunner().invoke(_train_app(), ["--config", str(path), "--dry-run"])


def test_dry_run_masks_tokens_set_in_the_config_file(tmp_path):
    path = _write(
        tmp_path,
        "model: unsloth/Qwen2.5-0.5B\n"
        "logging:\n"
        f"  hf_token: {_FAKE_HF_TOKEN}\n"
        f"  wandb_token: {_FAKE_WANDB_TOKEN}\n",
    )

    result = _dry_run(path)

    assert result.exit_code == 0
    assert _FAKE_HF_TOKEN not in result.output
    assert _FAKE_WANDB_TOKEN not in result.output

    dumped = yaml.safe_load(result.output)
    assert dumped["logging"]["hf_token"] == "[redacted]"
    assert dumped["logging"]["wandb_token"] == "[redacted]"


def test_dry_run_still_shows_an_unset_token_as_unset(tmp_path):
    path = _write(tmp_path, "model: unsloth/Qwen2.5-0.5B\n")

    result = _dry_run(path)

    assert result.exit_code == 0
    dumped = yaml.safe_load(result.output)
    assert dumped["logging"]["hf_token"] is None
    assert dumped["logging"]["wandb_token"] is None


@pytest.mark.parametrize("name", ["hf_token", "wandb_token"])
def test_dry_run_masks_one_token_without_inventing_the_other(tmp_path, name):
    path = _write(
        tmp_path,
        f"model: unsloth/Qwen2.5-0.5B\nlogging:\n  {name}: {_FAKE_HF_TOKEN}\n",
    )

    result = _dry_run(path)

    assert result.exit_code == 0
    assert _FAKE_HF_TOKEN not in result.output

    logging_section = yaml.safe_load(result.output)["logging"]
    other = "wandb_token" if name == "hf_token" else "hf_token"
    assert logging_section[name] == "[redacted]"
    assert logging_section[other] is None


def test_dry_run_does_not_mutate_the_loaded_config(tmp_path):
    """The masking has to happen on the dumped dict: training reads cfg.logging afterwards."""
    from unsloth_cli.config import load_config

    path = _write(
        tmp_path,
        "model: unsloth/Qwen2.5-0.5B\n"
        "logging:\n"
        f"  hf_token: {_FAKE_HF_TOKEN}\n"
        f"  wandb_token: {_FAKE_WANDB_TOKEN}\n",
    )

    assert _dry_run(path).exit_code == 0

    cfg = load_config(path)
    assert cfg.logging.hf_token == _FAKE_HF_TOKEN
    assert cfg.logging.wandb_token == _FAKE_WANDB_TOKEN

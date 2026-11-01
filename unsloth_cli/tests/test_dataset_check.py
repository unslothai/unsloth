# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from typer.testing import CliRunner

from unsloth_cli import app
from unsloth_cli.commands import dataset


@pytest.fixture(autouse = True)
def pytorch_host(monkeypatch):
    monkeypatch.setattr(dataset.platform, "system", lambda: "Linux")


def test_full_finetuning_does_not_select_a_quantized_tokenizer_mirror(tmp_path, monkeypatch):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "model": "org/model",
                "data": {"dataset": "org/data"},
                "training": {"training_type": "full", "load_in_4bit": True},
            }
        )
    )
    captured = []

    def run(request):
        captured.append(request)
        return {"status": "ok"}, 0

    monkeypatch.setattr(dataset, "_run_audit", run)
    result = CliRunner().invoke(app, ["dataset", "check", "--config", str(config), "--json"])
    assert result.exit_code == 0, result.output
    assert captured[0]["config"]["load_in_4bit"] is False


def test_mlx_host_does_not_report_pytorch_loss_statistics(monkeypatch):
    monkeypatch.setattr(dataset.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dataset.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(dataset, "_run_audit", lambda request: pytest.fail("must not start"))
    result = CliRunner().invoke(
        app, ["dataset", "check", "org/data", "--model", "org/model", "--json"]
    )
    assert result.exit_code == 2
    assert "MLX" in json.loads(result.output)["error"]


def test_config_and_overrides_reach_worker_without_training_imports(tmp_path, monkeypatch):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "model": "org/model",
                "data": {"dataset": "org/data"},
                "training": {"max_seq_length": 128, "train_on_completions": True},
            }
        )
    )
    captured = []

    def run(request):
        captured.append(request)
        return {"status": "ok", "coverage": {"complete": False}}, 0

    monkeypatch.setattr(dataset, "_run_audit", run)
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "check",
            "local.jsonl",
            "--config",
            str(config),
            "--max-seq-length",
            "64",
            "--max-rows",
            "7",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    request = captured[0]
    assert request["config"]["dataset_name"] is None
    assert request["config"]["local_datasets"] == ["local.jsonl"]
    assert request["config"]["max_seq_length"] == 64
    assert request["config"]["train_on_completions"] is True
    assert request["max_rows"] == 7 and not request["full"]
    assert not json.loads(result.output)["coverage"]["complete"]


def test_issues_found_exit_code_is_preserved(monkeypatch):
    monkeypatch.setattr(dataset, "_run_audit", lambda request: ({"status": "issues_found"}, 1))
    result = CliRunner().invoke(
        app, ["dataset", "check", "org/data", "--model", "org/model", "--json"]
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["status"] == "issues_found"


def test_packed_config_fails_before_worker_start(tmp_path, monkeypatch):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {"model": "org/model", "data": {"dataset": "org/data"}, "training": {"packing": True}}
        )
    )
    monkeypatch.setattr(
        dataset,
        "_run_audit",
        lambda request: (_ for _ in ()).throw(AssertionError("must not start")),
    )
    result = CliRunner().invoke(app, ["dataset", "check", "--config", str(config), "--json"])
    assert result.exit_code == 2
    assert "packing=False" in json.loads(result.output)["error"]


def test_help_explains_coverage():
    result = CliRunner().invoke(app, ["dataset", "check", "--help"])
    assert result.exit_code == 0
    assert "--full" in result.output and "--max-rows" in result.output

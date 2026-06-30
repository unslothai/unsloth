# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import typer
import pytest
import yaml
from typer.testing import CliRunner

import unsloth_cli.commands.eval as evalmod


def _eval_app():
    cli = typer.Typer()
    cli.command()(evalmod.evaluate)
    return cli


def test_resolve_base_model_reads_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B"})
    )
    assert evalmod.resolve_base_model(str(tmp_path)) == "unsloth/Llama-3.2-1B"


def test_resolve_base_model_none_for_plain_dir(tmp_path):
    assert evalmod.resolve_base_model(str(tmp_path)) is None


def test_resolve_base_model_none_for_missing_path():
    assert evalmod.resolve_base_model("/no/such/dir") is None


def test_make_jsonl_task_generates_expected_spec(tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"question": "1+1?", "answer": "2"}\n')
    out_dir = tmp_path / "tasks"

    name = evalmod.make_jsonl_task(data, "question", "answer", out_dir)

    assert name == "qa"
    spec = yaml.safe_load((out_dir / "qa.yaml").read_text())
    assert spec["task"] == "qa"
    assert spec["dataset_path"] == "json"
    assert spec["dataset_kwargs"]["data_files"] == str(data.resolve())
    assert spec["doc_to_text"] == "{{question}}"
    assert spec["doc_to_target"] == "{{answer}}"
    assert spec["metric_list"][0]["metric"] == "exact_match"


def test_make_jsonl_task_honours_custom_keys(tmp_path):
    data = tmp_path / "prompts.csv"
    data.write_text("prompt,label\nhi,hello\n")
    name = evalmod.make_jsonl_task(data, "prompt", "label", tmp_path / "t")

    spec = yaml.safe_load((tmp_path / "t" / "prompts.yaml").read_text())
    assert name == "prompts"
    assert spec["dataset_path"] == "csv"
    assert spec["doc_to_text"] == "{{prompt}}"
    assert spec["doc_to_target"] == "{{label}}"


def test_resolve_tasks_builtin_names(tmp_path):
    names, includes = evalmod.resolve_tasks("mmlu, gsm8k", "question", "answer", tmp_path)
    assert names == ["mmlu", "gsm8k"]
    assert includes == []


def test_resolve_tasks_custom_yaml(tmp_path):
    task_file = tmp_path / "custom.yaml"
    task_file.write_text(yaml.safe_dump({"task": "my_task", "output_type": "generate_until"}))

    names, includes = evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path)

    assert names == ["my_task"]
    assert includes == [str(tmp_path.resolve())]


def test_resolve_tasks_jsonl_generates_task(tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"question": "q", "answer": "a"}\n')
    gen_dir = tmp_path / "gen"

    names, includes = evalmod.resolve_tasks(str(data), "question", "answer", gen_dir)

    assert names == ["qa"]
    assert includes == [str(gen_dir.resolve())]
    assert (gen_dir / "qa.yaml").exists()


def test_resolve_tasks_yaml_without_task_name_raises(tmp_path):
    task_file = tmp_path / "bad.yaml"
    task_file.write_text(yaml.safe_dump({"output_type": "generate_until"}))
    with pytest.raises(ValueError, match = "missing a 'task:' name"):
        evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path)


def test_resolve_tasks_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evalmod.resolve_tasks("./nope.yaml", "question", "answer", tmp_path)


def test_resolve_tasks_empty_raises(tmp_path):
    with pytest.raises(ValueError, match = "No tasks provided"):
        evalmod.resolve_tasks("  , ", "question", "answer", tmp_path)


def test_render_results_renders_metric_row(capsys):
    evalmod._render_results(
        {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.5,
                    "exact_match_stderr,strict-match": 0.05,
                    "alias": "gsm8k",
                }
            }
        }
    )
    out = capsys.readouterr().out
    assert "gsm8k" in out
    assert "0.5000" in out
    assert "0.0500" in out


def test_eval_missing_lm_eval_shows_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    result = CliRunner().invoke(_eval_app(), ["fake/model", "--tasks", "gsm8k"])
    assert result.exit_code == 1, result.output
    assert "pip install unsloth[eval]" in result.output


@pytest.fixture
def fake_eval_env(monkeypatch):
    calls = {}

    class _FakeFLM:
        @classmethod
        def from_pretrained(cls, model_name = None, **kw):
            calls["model_name"] = model_name
            return SimpleNamespace(name = model_name), SimpleNamespace(name = "tok")

        @classmethod
        def for_inference(cls, model):
            calls["for_inference"] = True
            return model

    class _FakeHFLM:
        def __init__(self, pretrained = None, tokenizer = None, batch_size = None):
            calls["batch_size"] = batch_size

    class _FakeTaskManager:
        all_tasks = ["gsm8k", "qa", "mmlu", "hellaswag"]
        all_groups = ["mmlu"]
        all_tags = []

        def __init__(self, include_path = None):
            calls["include_path"] = include_path

    def _simple_evaluate(model = None, model_args = None, tasks = None, **kw):
        calls["model"] = model
        calls["model_args"] = model_args
        calls["tasks"] = tasks
        calls["simple_evaluate_kwargs"] = kw
        return {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.42,
                    "exact_match_stderr,strict-match": 0.01,
                    "alias": "gsm8k",
                }
            },
            "configs": {},
        }

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = _FakeFLM

    # deterministic device detection, no real torch needed
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = SimpleNamespace(is_available = lambda: False)
    torch_mod.backends = SimpleNamespace(
        mps = SimpleNamespace(is_available = lambda: False)
    )

    lm_eval_mod = types.ModuleType("lm_eval")
    lm_eval_mod.simple_evaluate = _simple_evaluate
    models_mod = types.ModuleType("lm_eval.models")
    hf_mod = types.ModuleType("lm_eval.models.huggingface")
    hf_mod.HFLM = _FakeHFLM
    tasks_mod = types.ModuleType("lm_eval.tasks")
    tasks_mod.TaskManager = _FakeTaskManager

    for name, mod in {
        "unsloth": unsloth_mod,
        "torch": torch_mod,
        "lm_eval": lm_eval_mod,
        "lm_eval.models": models_mod,
        "lm_eval.models.huggingface": hf_mod,
        "lm_eval.tasks": tasks_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return calls


def test_eval_success_writes_results(fake_eval_env, tmp_path):
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", "gsm8k", "--output-dir", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Saved results to" in result.output
    assert fake_eval_env["tasks"] == ["gsm8k"]
    assert fake_eval_env["simple_evaluate_kwargs"]["task_manager"] is not None
    assert fake_eval_env["simple_evaluate_kwargs"]["log_samples"] is False
    assert fake_eval_env["include_path"] is None

    saved = json.loads((out_dir / "results.json").read_text())
    assert saved["results"]["gsm8k"]["exact_match,strict-match"] == 0.42


def test_eval_jsonl_task_builds_task_manager(fake_eval_env, tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"question": "q", "answer": "a"}\n')

    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", str(data), "--output-dir", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    assert fake_eval_env["tasks"] == ["qa"]
    assert fake_eval_env["include_path"] is not None


def test_eval_mlx_falls_back_to_hf(fake_eval_env, tmp_path):
    sys.modules["unsloth"].DEVICE_TYPE = "mlx"
    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", "gsm8k", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "falling back" in result.output
    assert fake_eval_env["model"] == "hf"
    assert fake_eval_env["model_args"] == {"pretrained": "fake/model", "max_length": 2048}
    assert "model_name" not in fake_eval_env


def test_eval_hf_backend_skips_unsloth(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks", "gsm8k",
            "--backend", "hf",
            "--device", "cpu",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model"] == "hf"
    assert fake_eval_env["model_args"] == {"pretrained": "fake/model", "max_length": 2048}
    assert fake_eval_env["simple_evaluate_kwargs"]["device"] == "cpu"
    assert "model_name" not in fake_eval_env


def test_eval_rejects_nonpositive_batch_size(fake_eval_env, tmp_path):
    for bad in ["0", "-1", "abc"]:
        result = CliRunner().invoke(
            _eval_app(),
            [
                "fake/model", "--tasks", "gsm8k",
                "--backend", "hf", "--device", "cpu",
                "--batch-size", bad,
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 2, (bad, result.output)
        assert "positive integer or 'auto'" in result.output


def test_eval_hf_forwards_max_seq_length(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model", "--tasks", "gsm8k",
            "--backend", "hf", "--device", "cpu",
            "--max-seq-length", "1024",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_args"] == {"pretrained": "fake/model", "max_length": 1024}


def test_eval_hf_honors_base_model_for_remote_adapter(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "someuser/my-lora", "--tasks", "gsm8k",
            "--backend", "hf", "--device", "cpu",
            "--base-model", "meta-llama/Llama-2-7b",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_args"] == {
        "pretrained": "meta-llama/Llama-2-7b",
        "peft": "someuser/my-lora",
        "max_length": 2048,
    }


def test_eval_cuda_index_keeps_auto_batch_size(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model", "--tasks", "gsm8k",
            "--backend", "hf", "--device", "cuda:0",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    # 'auto' survives an explicit CUDA index (not downgraded to 1)
    assert fake_eval_env["simple_evaluate_kwargs"]["batch_size"] == "auto"
    assert fake_eval_env["model_args"]["load_in_4bit"] is True


def test_eval_unknown_task_errors(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", "notarealtask", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 2, result.output
    assert "unknown task" in result.output


def test_eval_hf_token_sets_env(fake_eval_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder")
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model", "--tasks", "gsm8k",
            "--backend", "hf", "--device", "cpu",
            "--hf-token", "hf_secret",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert os.environ.get("HF_TOKEN") == "hf_secret"

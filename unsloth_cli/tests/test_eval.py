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


def test_resolve_base_model_none_for_non_dict_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps(["not", "a", "dict"]))
    assert evalmod.resolve_base_model(str(tmp_path)) is None


def test_resolve_base_model_finds_hub_adapter(tmp_path, monkeypatch):
    remote_config = tmp_path / "adapter_config.json"
    remote_config.write_text(json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B"}))

    hub_mod = types.ModuleType("huggingface_hub")

    def fake_download(repo_id, filename, **kwargs):
        assert repo_id == "someuser/my-lora"
        assert filename == "adapter_config.json"
        return str(remote_config)

    hub_mod.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_mod)

    assert evalmod.resolve_base_model("someuser/my-lora") == "unsloth/Llama-3.2-1B"


def test_resolve_base_model_none_when_hub_lookup_fails(monkeypatch):
    hub_mod = types.ModuleType("huggingface_hub")

    def fake_download(*args, **kwargs):
        raise RuntimeError("no adapter_config.json in repo")

    hub_mod.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_mod)

    assert evalmod.resolve_base_model("someuser/full-model") is None


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
    assert spec["fewshot_split"] == "train"


def test_make_jsonl_task_honours_custom_keys(tmp_path):
    data = tmp_path / "prompts.csv"
    data.write_text("prompt,label\nhi,hello\n")
    name = evalmod.make_jsonl_task(data, "prompt", "label", tmp_path / "t")

    spec = yaml.safe_load((tmp_path / "t" / "prompts.yaml").read_text())
    assert name == "prompts"
    assert spec["dataset_path"] == "csv"
    assert spec["doc_to_text"] == "{{prompt}}"
    assert spec["doc_to_target"] == "{{label}}"


def test_make_jsonl_task_uses_raw_lookup_for_non_identifier_keys(tmp_path):
    data = tmp_path / "weird.jsonl"
    data.write_text('{"prompt-text": "1+1?", "expected answer": "2"}\n')
    evalmod.make_jsonl_task(data, "prompt-text", "expected answer", tmp_path / "t")

    spec = yaml.safe_load((tmp_path / "t" / "weird.yaml").read_text())
    # jinja can't parse these keys; lm-eval resolves raw column names directly
    assert spec["doc_to_text"] == "prompt-text"
    assert spec["doc_to_target"] == "expected answer"


def test_make_jsonl_task_avoids_reserved_names(tmp_path):
    data = tmp_path / "gsm8k.jsonl"
    data.write_text('{"question": "q", "answer": "a"}\n')

    name = evalmod.make_jsonl_task(
        data, "question", "answer", tmp_path / "t", reserved = frozenset({"gsm8k"})
    )

    assert name == "gsm8k_2"
    assert (tmp_path / "t" / "gsm8k_2.yaml").exists()


def test_has_tokenizer_files_checks_hub_repo(monkeypatch):
    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.list_repo_files = lambda repo_id: [
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_mod)

    assert evalmod._has_tokenizer_files("someuser/my-lora") is True


def test_has_tokenizer_files_false_when_hub_listing_fails(monkeypatch):
    hub_mod = types.ModuleType("huggingface_hub")

    def _fail(repo_id):
        raise RuntimeError("offline")

    hub_mod.list_repo_files = _fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_mod)

    assert evalmod._has_tokenizer_files("someuser/my-lora") is False


def test_resolve_tasks_builtin_names(tmp_path):
    names, includes = evalmod.resolve_tasks("mmlu, gsm8k", "question", "answer", tmp_path)
    assert names == ["mmlu", "gsm8k"]
    assert includes == []


def test_resolve_tasks_custom_yaml_copied_to_include_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    task_file = src / "custom.yaml"
    task_file.write_text(yaml.safe_dump({"task": "my_task", "output_type": "generate_until"}))
    # a broken sibling must not end up on the include path
    (src / "broken.yaml").write_text("task: [unclosed")
    tmp_dir = tmp_path / "gen"

    names, includes = evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_dir)

    custom_dir = tmp_dir / "custom"
    assert names == ["my_task"]
    assert includes == [str(custom_dir.resolve())]
    assert (custom_dir / "my_task.yaml").exists()
    assert not (custom_dir / "broken.yaml").exists()


def test_resolve_tasks_yml_normalised_to_yaml(tmp_path):
    task_file = tmp_path / "custom.yml"
    task_file.write_text(yaml.safe_dump({"task": "my_task", "output_type": "generate_until"}))
    tmp_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_dir)

    assert names == ["my_task"]
    # lm-eval only indexes .yaml files
    assert (tmp_dir / "custom" / "my_task.yaml").exists()


def test_resolve_tasks_include_yaml_keeps_parent_dir(tmp_path):
    task_file = tmp_path / "custom.yaml"
    task_file.write_text(yaml.safe_dump({"task": "my_task", "include": "base.yaml"}))

    names, includes = evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path / "gen")

    assert names == ["my_task"]
    # the config references a sibling file, so its directory stays included
    assert includes == [str(tmp_path.resolve())]


def test_resolve_tasks_yaml_with_function_tag_keeps_parent_dir(tmp_path):
    task_file = tmp_path / "custom.yaml"
    task_file.write_text(
        "task: fn_task\noutput_type: generate_until\n"
        "process_docs: !function utils.process_docs\n"
    )
    tmp_dir = tmp_path / "gen"

    names, includes = evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_dir)

    assert names == ["fn_task"]
    # !function imports resolve relative to the yaml, so utils.py must stay
    # next to it — no copy into the temp dir
    assert includes == [str(tmp_path.resolve())]
    assert not (tmp_dir / "custom" / "fn_task.yaml").exists()


def test_resolve_tasks_task_name_from_included_base(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"task": "from_base", "output_type": "generate_until"})
    )
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"include": "base.yaml", "dataset_path": "json"}))

    names, includes = evalmod.resolve_tasks(str(child), "question", "answer", tmp_path / "gen")

    # lm-eval resolves include: during indexing, so the name from the base
    # config counts
    assert names == ["from_base"]
    assert includes == [str(tmp_path.resolve())]


def test_resolve_tasks_rejects_yml_group_config(tmp_path):
    task_file = tmp_path / "suite.yml"
    task_file.write_text(yaml.safe_dump({"group": "my_suite", "task": ["task_a", "task_b"]}))
    with pytest.raises(ValueError, match = "only indexes .yaml"):
        evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path / "gen")


def test_resolve_tasks_jsonl_generates_task(tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"question": "q", "answer": "a"}\n')
    tmp_dir = tmp_path / "gen"

    names, includes = evalmod.resolve_tasks(str(data), "question", "answer", tmp_dir)

    gen_dir = tmp_dir / "generated"
    assert names == ["qa"]
    assert includes == [str(gen_dir.resolve())]
    assert (gen_dir / "qa.yaml").exists()


def test_resolve_tasks_uniquifies_colliding_dataset_stems(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "qa.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    (dir_b / "qa.jsonl").write_text('{"question": "q2", "answer": "a2"}\n')
    gen_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(
        f"{dir_a / 'qa.jsonl'},{dir_b / 'qa.jsonl'}", "question", "answer", gen_dir
    )

    assert names == ["qa", "qa_2"]
    spec_a = yaml.safe_load((gen_dir / "generated" / "qa.yaml").read_text())
    spec_b = yaml.safe_load((gen_dir / "generated" / "qa_2.yaml").read_text())
    assert spec_a["dataset_kwargs"]["data_files"] == str((dir_a / "qa.jsonl").resolve())
    assert spec_b["dataset_kwargs"]["data_files"] == str((dir_b / "qa.jsonl").resolve())
    assert spec_b["task"] == "qa_2"


def test_resolve_tasks_reserves_group_child_names_for_datasets(tmp_path):
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"group": "suite", "task": ["qa", {"task": "qa_inline"}]})
    )
    (tmp_path / "qa.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    tmp_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(
        f"{tmp_path / 'suite.yaml'},{tmp_path / 'qa.jsonl'}", "question", "answer", tmp_dir
    )

    # the dataset must not generate a task shadowing the suite's child 'qa'
    assert names == ["suite", "qa_2"]
    assert (tmp_dir / "generated" / "qa_2.yaml").exists()


def test_resolve_tasks_invalid_yaml_raises(tmp_path):
    task_file = tmp_path / "broken.yaml"
    task_file.write_text("task: [unclosed")
    with pytest.raises(ValueError, match = "Invalid YAML"):
        evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path)


def test_resolve_tasks_yaml_list_raises(tmp_path):
    task_file = tmp_path / "list.yaml"
    task_file.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(ValueError, match = "YAML mapping"):
        evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path)


def test_resolve_tasks_group_yaml_uses_group_name(tmp_path):
    task_file = tmp_path / "suite.yaml"
    task_file.write_text(yaml.safe_dump({"group": "my_suite", "task": ["task_a", "task_b"]}))

    names, includes = evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path)

    assert names == ["my_suite"]
    assert includes == [str(tmp_path.resolve())]


def test_resolve_tasks_group_yaml_without_group_raises(tmp_path):
    task_file = tmp_path / "suite.yaml"
    task_file.write_text(yaml.safe_dump({"task": ["task_a", "task_b"]}))
    with pytest.raises(ValueError, match = "no 'group:' name"):
        evalmod.resolve_tasks(str(task_file), "question", "answer", tmp_path)


def test_resolve_tasks_yaml_rejects_registered_name(tmp_path):
    task_file = tmp_path / "clash.yaml"
    task_file.write_text(yaml.safe_dump({"task": "gsm8k", "output_type": "generate_until"}))
    with pytest.raises(ValueError, match = "redefines 'gsm8k'"):
        evalmod.resolve_tasks(
            str(task_file), "question", "answer", tmp_path, reserved = frozenset({"gsm8k"})
        )


def test_resolve_tasks_rejects_duplicate_yaml_names(tmp_path):
    for stem in ("one", "two"):
        (tmp_path / f"{stem}.yaml").write_text(
            yaml.safe_dump({"task": "same_task", "output_type": "generate_until"})
        )
    with pytest.raises(ValueError, match = "Duplicate task name 'same_task'"):
        evalmod.resolve_tasks(
            f"{tmp_path / 'one.yaml'},{tmp_path / 'two.yaml'}",
            "question",
            "answer",
            tmp_path / "gen",
        )


def test_resolve_tasks_rejects_duplicate_builtins(tmp_path):
    with pytest.raises(ValueError, match = "Duplicate task 'gsm8k'"):
        evalmod.resolve_tasks("gsm8k,gsm8k", "question", "answer", tmp_path)


def test_resolve_tasks_renames_dataset_colliding_with_yaml_name(tmp_path):
    (tmp_path / "foo.yaml").write_text(
        yaml.safe_dump({"task": "foo", "output_type": "generate_until"})
    )
    (tmp_path / "foo.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    tmp_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(
        f"{tmp_path / 'foo.yaml'},{tmp_path / 'foo.jsonl'}", "question", "answer", tmp_dir
    )

    # the dataset must not silently shadow (or be shadowed by) the yaml task
    assert names == ["foo", "foo_2"]
    assert (tmp_dir / "generated" / "foo_2.yaml").exists()


def _fake_torch(
    monkeypatch,
    cuda_available = False,
    device_count = 0,
    mps_available = False,
    xpu_available = False,
    xpu_count = 0,
):
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = SimpleNamespace(
        is_available = lambda: cuda_available, device_count = lambda: device_count
    )
    torch_mod.backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: mps_available))
    torch_mod.xpu = SimpleNamespace(
        is_available = lambda: xpu_available, device_count = lambda: xpu_count
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def test_hf_device_error_validates_cuda_strings(monkeypatch):
    _fake_torch(monkeypatch, cuda_available = True, device_count = 2)
    assert evalmod._hf_device_error("cuda") is None
    assert evalmod._hf_device_error("cuda:0") is None
    assert evalmod._hf_device_error("cuda:1") is None
    # lm-eval only recognises canonical cuda:<i>; everything else falls back
    for bad in ("cuda0", "cuda:", "cuda:01", "cuda:-1", "cudax"):
        assert evalmod._hf_device_error(bad) is not None, bad
    assert "only 2 CUDA" in evalmod._hf_device_error("cuda:2")


def test_hf_device_error_validates_mps_strings(monkeypatch):
    _fake_torch(monkeypatch, mps_available = True)
    assert evalmod._hf_device_error("mps") is None
    assert evalmod._hf_device_error("mps:0") is None
    assert evalmod._hf_device_error("mps:1") is not None
    _fake_torch(monkeypatch, mps_available = False)
    assert "MPS is not available" in evalmod._hf_device_error("mps")


def test_hf_device_error_rejects_unknown_literals(monkeypatch):
    _fake_torch(monkeypatch)
    assert evalmod._hf_device_error("cpu") is None
    # typos would silently fall back to HFLM's default device
    for bad in ("cpuu", "cude", "gpu", "xpu", "npu"):
        assert "invalid --device" in evalmod._hf_device_error(bad), bad


def test_hf_device_error_validates_indexed_accelerators(monkeypatch):
    # an unavailable or out-of-range accelerator would also silently fall back
    _fake_torch(monkeypatch, xpu_available = True, xpu_count = 2)
    assert evalmod._hf_device_error("xpu:0") is None
    assert evalmod._hf_device_error("xpu:1") is None
    assert "only 2 XPU" in evalmod._hf_device_error("xpu:2")
    # this torch build has no npu/hpu module at all
    assert "NPU is not available" in evalmod._hf_device_error("npu:0")
    assert "HPU is not available" in evalmod._hf_device_error("hpu:0")
    _fake_torch(monkeypatch, xpu_available = False)
    assert "XPU is not available" in evalmod._hf_device_error("xpu:0")


def test_metric_number_unwraps_numpy_like_scalars():
    class _FakeScalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    assert evalmod._metric_number(0.5) == 0.5
    assert evalmod._metric_number(3) == 3
    assert evalmod._metric_number(_FakeScalar(0.25)) == 0.25
    assert evalmod._metric_number(_FakeScalar("not a number")) is None
    assert evalmod._metric_number("alias-ish string") is None


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


def test_render_results_includes_group_aggregates(capsys):
    evalmod._render_results(
        {
            "results": {
                "mmlu_abstract_algebra": {"acc,none": 0.30, "alias": " - abstract_algebra"},
            },
            "groups": {
                "mmlu": {"acc,none": 0.45, "alias": "mmlu"},
            },
        }
    )
    out = capsys.readouterr().out
    assert "0.3000" in out
    # the group aggregate must be shown, not just per-subtask rows
    assert "0.4500" in out


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
        def from_pretrained(
            cls,
            model_name = None,
            **kw,
        ):
            calls["model_name"] = model_name
            model = SimpleNamespace(
                name = model_name,
                get_input_embeddings = lambda: SimpleNamespace(
                    weight = SimpleNamespace(shape = (32000, 4096))
                ),
                resize_token_embeddings = lambda n: calls.setdefault("events", []).append(
                    ("resize", n)
                ),
            )
            return model, SimpleNamespace(name = "tok")

        @classmethod
        def for_inference(cls, model):
            calls["for_inference"] = True
            return model

    class _FakeHFLM:
        def __init__(
            self,
            pretrained = None,
            tokenizer = None,
            batch_size = None,
            max_length = None,
        ):
            calls["batch_size"] = batch_size
            calls["hflm_tokenizer"] = tokenizer
            calls["hflm_max_length"] = max_length

    class _FakeTaskManager:
        def __init__(self, include_path = None):
            calls["include_path"] = include_path
            self.all_tasks = ["gsm8k", "mmlu", "hellaswag"]
            self.all_groups = ["mmlu"]
            self.all_tags = []
            # mirror lm-eval: yaml tasks/groups under include paths get
            # registered under their task or group name
            for directory in include_path or []:
                for spec_file in sorted(Path(directory).glob("*.yaml")):
                    # like lm-eval, tolerate !function tags but not broken yaml
                    spec = yaml.load(spec_file.read_text(), Loader = evalmod._TaskYamlLoader)
                    if not isinstance(spec, dict):
                        continue
                    name = spec.get("task")
                    if isinstance(name, list):
                        if spec.get("group"):
                            self.all_groups.append(str(spec["group"]))
                    elif name:
                        self.all_tasks.append(str(name))

    def _simple_evaluate(
        model = None,
        model_args = None,
        tasks = None,
        **kw,
    ):
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
    torch_mod.cuda = SimpleNamespace(is_available = lambda: False, device_count = lambda: 0)
    torch_mod.backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: False))

    # no adapter_config.json on the fake Hub, and no network access in tests
    hub_mod = types.ModuleType("huggingface_hub")

    def _no_hub_download(*args, **kwargs):
        raise RuntimeError("adapter_config.json not found")

    hub_mod.hf_hub_download = _no_hub_download

    def _no_repo_files(*args, **kwargs):
        raise RuntimeError("repo not found")

    hub_mod.list_repo_files = _no_repo_files

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
        "huggingface_hub": hub_mod,
        "lm_eval": lm_eval_mod,
        "lm_eval.models": models_mod,
        "lm_eval.models.huggingface": hf_mod,
        "lm_eval.tasks": tasks_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # deterministic regardless of whether bitsandbytes is installed locally
    monkeypatch.setattr(evalmod, "_bitsandbytes_available", lambda: True)

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
    assert fake_eval_env["hflm_max_length"] == 2048
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
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
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
                "fake/model",
                "--tasks",
                "gsm8k",
                "--backend",
                "hf",
                "--device",
                "cpu",
                "--batch-size",
                bad,
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 2, (bad, result.output)
        assert "positive integer or 'auto'" in result.output


def test_eval_hf_forwards_max_seq_length(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--max-seq-length",
            "1024",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_args"] == {"pretrained": "fake/model", "max_length": 1024}


def test_eval_unsloth_forwards_max_seq_length_to_hflm(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--max-seq-length",
            "512",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["hflm_max_length"] == 512


def test_eval_hf_local_adapter_uses_adapter_tokenizer(fake_eval_env, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B"})
    )
    (adapter / "tokenizer_config.json").write_text("{}")

    result = CliRunner().invoke(
        _eval_app(),
        [
            str(adapter),
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_args"] == {
        "pretrained": "unsloth/Llama-3.2-1B",
        "peft": str(adapter),
        "tokenizer": str(adapter),
        "max_length": 2048,
    }


def _make_local_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B"})
    )
    (adapter / "tokenizer_config.json").write_text("{}")
    return adapter


def _install_adapter_stubs(monkeypatch, fake_eval_env, tokenizer_len):
    peft_mod = types.ModuleType("peft")

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path):
            fake_eval_env["peft_adapter"] = adapter_path
            fake_eval_env.setdefault("events", []).append(("peft", adapter_path))
            return model

    peft_mod.PeftModel = _FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    class _FakeTokenizer:
        name = "adapter-tok"

        def __len__(self):
            return tokenizer_len

    transformers_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            fake_eval_env["tokenizer_from"] = path
            return _FakeTokenizer()

    transformers_mod.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)


def test_eval_unsloth_adapter_prefers_adapter_tokenizer(fake_eval_env, tmp_path, monkeypatch):
    adapter = _make_local_adapter(tmp_path)
    # same vocab size as the fake base model: no resize expected
    _install_adapter_stubs(monkeypatch, fake_eval_env, tokenizer_len = 32000)

    result = CliRunner().invoke(
        _eval_app(),
        [str(adapter), "--tasks", "gsm8k", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_name"] == "unsloth/Llama-3.2-1B"
    assert fake_eval_env["peft_adapter"] == str(adapter)
    assert fake_eval_env["tokenizer_from"] == str(adapter)
    assert fake_eval_env["hflm_tokenizer"].name == "adapter-tok"
    assert fake_eval_env["events"] == [("peft", str(adapter))]


def test_eval_unsloth_adapter_resizes_embeddings_before_peft(fake_eval_env, tmp_path, monkeypatch):
    adapter = _make_local_adapter(tmp_path)
    # adapter tokenizer grew past the fake base vocab (32000)
    _install_adapter_stubs(monkeypatch, fake_eval_env, tokenizer_len = 32005)

    result = CliRunner().invoke(
        _eval_app(),
        [str(adapter), "--tasks", "gsm8k", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    # the resize must land before the adapter weights are applied
    assert fake_eval_env["events"] == [("resize", 32005), ("peft", str(adapter))]


def test_eval_hf_honors_base_model_for_remote_adapter(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "someuser/my-lora",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--base-model",
            "meta-llama/Llama-2-7b",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_args"] == {
        "pretrained": "meta-llama/Llama-2-7b",
        "peft": "someuser/my-lora",
        "max_length": 2048,
    }


def test_eval_cuda_index_keeps_auto_batch_size(fake_eval_env, tmp_path):
    sys.modules["torch"].cuda = SimpleNamespace(is_available = lambda: True, device_count = lambda: 1)
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cuda:0",
            "--output-dir",
            str(tmp_path / "out"),
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


def test_eval_dataset_shadowing_builtin_is_renamed(fake_eval_env, tmp_path):
    data = tmp_path / "gsm8k.jsonl"
    data.write_text('{"question": "q", "answer": "a"}\n')

    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", str(data), "--output-dir", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    # the built-in gsm8k benchmark must not shadow the user's dataset
    assert fake_eval_env["tasks"] == ["gsm8k_2"]
    assert "as 'gsm8k_2'" in result.output


def test_eval_custom_yaml_shadowing_builtin_errors(fake_eval_env, tmp_path):
    task_file = tmp_path / "clash.yaml"
    task_file.write_text(yaml.safe_dump({"task": "gsm8k", "output_type": "generate_until"}))

    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", str(task_file), "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 2, result.output
    assert "redefines 'gsm8k'" in result.output


def test_eval_fewshot_with_raw_key_dataset_errors(fake_eval_env, tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"expected answer": "2", "question": "1+1?"}\n')

    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            str(data),
            "--target-key",
            "expected answer",
            "--num-fewshot",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "plain-identifier column names" in result.output


def test_eval_custom_yaml_survives_broken_sibling(fake_eval_env, tmp_path):
    task_file = tmp_path / "good.yaml"
    task_file.write_text(yaml.safe_dump({"task": "good_task", "output_type": "generate_until"}))
    # the fake TaskManager (like lm-eval 0.4.4) chokes on unparseable yaml
    # in an include dir; the broken sibling must never reach it
    (tmp_path / "broken.yaml").write_text("task: [unclosed")

    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", str(task_file), "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["tasks"] == ["good_task"]


def test_eval_group_yaml_runs_under_group_name(fake_eval_env, tmp_path):
    task_file = tmp_path / "suite.yaml"
    task_file.write_text(yaml.safe_dump({"group": "my_suite", "task": ["task_a", "task_b"]}))

    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", str(task_file), "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["tasks"] == ["my_suite"]


def test_eval_unsloth_rejects_multi_process_launch(fake_eval_env, tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    result = CliRunner().invoke(
        _eval_app(),
        ["fake/model", "--tasks", "gsm8k", "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 2, result.output
    assert "multi-process launches" in result.output


def test_eval_hf_allows_multi_process_launch(fake_eval_env, tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_eval_rejects_nonpositive_limit_and_max_seq_length(fake_eval_env, tmp_path):
    for flag, bad, message in (
        ("--limit", "0", "--limit must be a positive integer"),
        ("--limit", "-5", "--limit must be a positive integer"),
        ("--max-seq-length", "0", "--max-seq-length must be a positive integer"),
        ("--max-seq-length", "-1", "--max-seq-length must be a positive integer"),
    ):
        result = CliRunner().invoke(
            _eval_app(),
            [
                "fake/model",
                "--tasks",
                "gsm8k",
                flag,
                bad,
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 2, (flag, bad, result.output)
        assert message in result.output, (flag, bad, result.output)


def test_eval_rejects_negative_num_fewshot(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--num-fewshot",
            "-1",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--num-fewshot must be >= 0" in result.output


def test_eval_rejects_unknown_backend(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hff",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--backend must be 'unsloth' or 'hf'" in result.output


def test_eval_hf_cuda_without_bnb_loads_full_precision(fake_eval_env, tmp_path, monkeypatch):
    sys.modules["torch"].cuda = SimpleNamespace(is_available = lambda: True, device_count = lambda: 1)
    monkeypatch.setattr(evalmod, "_bitsandbytes_available", lambda: False)
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cuda:0",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "load_in_4bit" not in fake_eval_env["model_args"]
    assert "bitsandbytes is not installed" in result.output


def test_eval_hf_hub_adapter_uses_hub_tokenizer(fake_eval_env, tmp_path, monkeypatch):
    remote_config = tmp_path / "adapter_config.json"
    remote_config.write_text(json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B"}))

    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.hf_hub_download = lambda repo_id, filename, **kwargs: str(remote_config)
    hub_mod.list_repo_files = lambda repo_id: ["adapter_config.json", "tokenizer.json"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_mod)

    result = CliRunner().invoke(
        _eval_app(),
        [
            "someuser/my-lora",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert fake_eval_env["model_args"] == {
        "pretrained": "unsloth/Llama-3.2-1B",
        "peft": "someuser/my-lora",
        "tokenizer": "someuser/my-lora",
        "max_length": 2048,
    }


def test_eval_hf_rejects_cuda_when_unavailable(fake_eval_env, tmp_path):
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cuda",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "CUDA is not available" in result.output


def test_eval_hf_rejects_out_of_range_cuda_index(fake_eval_env, tmp_path):
    sys.modules["torch"].cuda = SimpleNamespace(is_available = lambda: True, device_count = lambda: 1)
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cuda:1",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "only 1 CUDA device(s)" in result.output


def test_eval_worker_rank_exits_cleanly_on_none_results(fake_eval_env, tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    sys.modules["lm_eval"].simple_evaluate = lambda **kwargs: None

    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Saved results" not in result.output
    assert not (tmp_path / "out").exists()


def test_eval_none_results_errors_on_single_process(fake_eval_env, tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising = False)
    monkeypatch.delenv("LOCAL_RANK", raising = False)
    sys.modules["lm_eval"].simple_evaluate = lambda **kwargs: None

    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "no results" in result.output


def test_eval_hf_token_sets_env(fake_eval_env, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder")
    result = CliRunner().invoke(
        _eval_app(),
        [
            "fake/model",
            "--tasks",
            "gsm8k",
            "--backend",
            "hf",
            "--device",
            "cpu",
            "--hf-token",
            "hf_secret",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert os.environ.get("HF_TOKEN") == "hf_secret"


def test_resolve_tasks_dataset_before_group_still_avoids_child_names(tmp_path):
    # argument order must not decide the generated task's name
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"group": "suite", "task": ["qa", {"task": "qa_inline"}]})
    )
    (tmp_path / "qa.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    tmp_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(
        f"{tmp_path / 'qa.jsonl'},{tmp_path / 'suite.yaml'}", "question", "answer", tmp_dir
    )

    assert names == ["qa_2", "suite"]
    assert (tmp_dir / "generated" / "qa_2.yaml").exists()


def test_resolve_tasks_rejects_builtin_child_shadowed_by_sibling(tmp_path):
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump({"group": "suite", "task": ["gsm8k"]}))
    (tmp_path / "gsm8k.yaml").write_text(yaml.safe_dump({"task": "gsm8k", "dataset_path": "json"}))
    with pytest.raises(ValueError, match = "depends on the lm-eval version"):
        evalmod.resolve_tasks(
            str(tmp_path / "suite.yaml"),
            "question",
            "answer",
            tmp_path / "gen",
            reserved = frozenset({"gsm8k"}),
        )


def test_resolve_tasks_allows_group_of_builtins_without_siblings(tmp_path):
    # a suite that aggregates registered tasks is legitimate lm-eval usage
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"group": "suite", "task": ["gsm8k", "mmlu"]})
    )

    names, _ = evalmod.resolve_tasks(
        str(tmp_path / "suite.yaml"),
        "question",
        "answer",
        tmp_path / "gen",
        reserved = frozenset({"gsm8k", "mmlu"}),
    )

    assert names == ["suite"]


def test_resolve_tasks_rejects_include_order_dependent_name(tmp_path):
    # lm-eval versions disagree on include precedence, so a name that changes
    # with the merge order must be rejected
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"task": "name_a"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"task": "name_b"}))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"include": ["a.yaml", "b.yaml"], "dataset_path": "json"}))

    with pytest.raises(ValueError, match = "include order"):
        evalmod.resolve_tasks(str(child), "question", "answer", tmp_path / "gen")


def test_resolve_tasks_accepts_local_name_over_include_conflict(tmp_path):
    # a top-level task: settles the name on every lm-eval version
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"task": "name_a"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"task": "name_b"}))
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump({"include": ["a.yaml", "b.yaml"], "task": "mine", "dataset_path": "json"})
    )

    names, _ = evalmod.resolve_tasks(str(child), "question", "answer", tmp_path / "gen")

    assert names == ["mine"]


def test_load_task_spec_resolves_includes_against_parent_dir(tmp_path, monkeypatch):
    # lm-eval resolves relative includes against the including file, never cwd
    task_dir = tmp_path / "tasks"
    decoy_dir = tmp_path / "decoy"
    task_dir.mkdir()
    decoy_dir.mkdir()
    (task_dir / "base.yaml").write_text(yaml.safe_dump({"task": "right"}))
    (decoy_dir / "base.yaml").write_text(yaml.safe_dump({"task": "wrong"}))
    child = task_dir / "child.yaml"
    child.write_text(yaml.safe_dump({"include": "base.yaml", "dataset_path": "json"}))
    monkeypatch.chdir(decoy_dir)

    spec = evalmod._load_task_spec(child)

    assert spec["task"] == "right"


def test_json_default_preserves_numeric_scalars():
    class _FakeNumpyScalar:
        def tolist(self):
            return 3

    dumped = json.dumps({"n": _FakeNumpyScalar(), "s": {1, 2}}, default = evalmod._json_default)

    parsed = json.loads(dumped)
    assert parsed["n"] == 3
    assert isinstance(parsed["s"], str)


def test_resolve_tasks_rejects_builtin_child_shadowed_in_subdirectory(tmp_path):
    # lm-eval indexes include paths recursively, so a nested sibling shadows too
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump({"group": "suite", "task": ["gsm8k"]}))
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "gsm8k.yaml").write_text(yaml.safe_dump({"task": "gsm8k", "dataset_path": "json"}))
    with pytest.raises(ValueError, match = "depends on the lm-eval version"):
        evalmod.resolve_tasks(
            str(tmp_path / "suite.yaml"),
            "question",
            "answer",
            tmp_path / "gen",
            reserved = frozenset({"gsm8k"}),
        )


def test_hf_device_error_gates_xpu_hpu_on_lm_eval_version(monkeypatch):
    # HFLM only enumerated xpu/hpu from 0.4.10; older versions silently fall back
    _fake_torch(monkeypatch, xpu_available = True, xpu_count = 1)
    monkeypatch.setattr(evalmod, "_lm_eval_version", lambda: (0, 4, 4))
    assert "needs lm-eval >= 0.4.10" in evalmod._hf_device_error("xpu:0")
    assert "needs lm-eval >= 0.4.10" in evalmod._hf_device_error("hpu:0")
    # npu has been enumerated since 0.4.4
    assert "NPU is not available" in evalmod._hf_device_error("npu:0")
    monkeypatch.setattr(evalmod, "_lm_eval_version", lambda: (0, 4, 10))
    assert evalmod._hf_device_error("xpu:0") is None


def test_resolve_tasks_reserves_tag_aliases_for_datasets(tmp_path):
    # a tag: alias registers under that name, so a dataset must not take it
    (tmp_path / "custom.yaml").write_text(
        yaml.safe_dump({"task": "foo", "tag": "qa", "dataset_path": "json"})
    )
    (tmp_path / "qa.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    tmp_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(
        f"{tmp_path / 'qa.jsonl'},{tmp_path / 'custom.yaml'}", "question", "answer", tmp_dir
    )

    assert names == ["qa_2", "foo"]


def test_resolve_tasks_reserves_string_group_alias_for_datasets(tmp_path):
    # legacy string group: on a single task acts as a tag alias
    (tmp_path / "custom.yaml").write_text(
        yaml.safe_dump({"task": "foo", "group": "myalias", "dataset_path": "json"})
    )
    (tmp_path / "myalias.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    tmp_dir = tmp_path / "gen"

    names, _ = evalmod.resolve_tasks(
        f"{tmp_path / 'custom.yaml'},{tmp_path / 'myalias.jsonl'}", "question", "answer", tmp_dir
    )

    assert names == ["foo", "myalias_2"]

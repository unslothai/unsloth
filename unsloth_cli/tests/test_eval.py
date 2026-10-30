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

import pytest
import typer
import yaml
from typer.testing import CliRunner

import unsloth_cli.commands.eval as evalmod


def _eval_app():
    cli = typer.Typer()
    cli.command()(evalmod.evaluate)
    return cli


def _run(*args):
    return CliRunner().invoke(_eval_app(), list(args))


@pytest.mark.parametrize(
    "content, expected",
    [
        (json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B"}), "unsloth/Llama-3.2-1B"),
        ("not json", None),
        ("[1, 2]", None),
    ],
)
def test_resolve_base_model_from_local_adapter(tmp_path, content, expected):
    (tmp_path / "adapter_config.json").write_text(content)
    assert evalmod.resolve_base_model(str(tmp_path)) == expected


def test_resolve_base_model_plain_dir_and_non_hub_name(tmp_path):
    assert evalmod.resolve_base_model(str(tmp_path)) is None
    assert evalmod.resolve_base_model("not a repo id") is None


@pytest.mark.parametrize(
    "key, expected",
    [
        ("question", "{{question}}"),
        ("expected answer", "expected answer"),
        ("if", "if"),
        ("none", "none"),
    ],
)
def test_doc_column(key, expected):
    assert evalmod._doc_column(key) == expected


def test_resolve_tasks_plain_names():
    names, includes = evalmod.resolve_tasks("gsm8k, mmlu ,", "q", "a", Path("/unused"))
    assert names == ["gsm8k", "mmlu"]
    assert includes == []


@pytest.mark.parametrize("tasks", ["", " , ", "gsm8k,gsm8k"])
def test_resolve_tasks_rejects_empty_and_duplicates(tasks, tmp_path):
    with pytest.raises(ValueError):
        evalmod.resolve_tasks(tasks, "q", "a", tmp_path)


def test_resolve_tasks_missing_files(tmp_path):
    for missing in ("nope.jsonl", "nope.yaml"):
        with pytest.raises(FileNotFoundError):
            evalmod.resolve_tasks(str(tmp_path / missing), "q", "a", tmp_path)


def test_dataset_task_yaml(tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"question": "1+1?", "answer": "2"}\n')
    names, includes = evalmod.resolve_tasks(str(data), "question", "answer", tmp_path / "tmp")
    assert names == ["qa"]
    spec = yaml.safe_load(Path(includes[0], "qa.yaml").read_text())
    assert spec["dataset_path"] == "json"
    assert spec["dataset_kwargs"] == {"data_files": str(data.resolve())}
    assert spec["doc_to_text"] == "{{question}}"
    assert spec["fewshot_split"] == "train"
    assert spec["metric_list"][0]["metric"] == "exact_match"


def test_dataset_task_stays_clear_of_names_requested_alongside(tmp_path):
    (tmp_path / "gsm8k.jsonl").write_text("{}\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "gsm8k.csv").write_text("question,answer\n")
    names, _ = evalmod.resolve_tasks(
        f"gsm8k,{tmp_path / 'gsm8k.jsonl'},{tmp_path / 'sub' / 'gsm8k.csv'}",
        "q",
        "a",
        tmp_path / "tmp",
    )
    assert names == ["gsm8k", "gsm8k_2", "gsm8k_3"]


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
@pytest.mark.parametrize("dataset_first", [True, False])
def test_dataset_task_avoids_explicit_yaml_name(tmp_path, suffix, dataset_first):
    data = tmp_path / "qa.jsonl"
    data.write_text("{}\n")
    task = tmp_path / f"custom{suffix}"
    task.write_text("task: qa\ndataset_path: json\n")
    entries = [data, task] if dataset_first else [task, data]
    names, includes = evalmod.resolve_tasks(",".join(map(str, entries)), "q", "a", tmp_path / "tmp")
    assert names == (["qa_2", "qa"] if dataset_first else ["qa", "qa_2"])
    generated = (tmp_path / "tmp" / "generated").resolve()
    custom = (tmp_path / "tmp" / "custom").resolve()
    assert includes == list(map(str, [generated, custom] if dataset_first else [custom, generated]))
    spec = yaml.safe_load((generated / "qa_2.yaml").read_text())
    assert spec["task"] == "qa_2"
    assert spec["dataset_kwargs"] == {"data_files": str(data.resolve())}
    assert yaml.safe_load((custom / "qa.yaml").read_text())["task"] == "qa"


@pytest.mark.parametrize("dataset_first", [True, False])
def test_dataset_task_avoids_explicit_group_name(tmp_path, dataset_first):
    data = tmp_path / "suite.csv"
    data.write_text("q,a\n")
    group = tmp_path / "custom.yaml"
    group.write_text("group: suite\ntask: [a, b]\n")
    entries = [data, group] if dataset_first else [group, data]
    names, includes = evalmod.resolve_tasks(",".join(map(str, entries)), "q", "a", tmp_path / "tmp")
    assert names == (["suite_2", "suite"] if dataset_first else ["suite", "suite_2"])
    generated = (tmp_path / "tmp" / "generated").resolve()
    assert includes == list(
        map(
            str,
            [generated, tmp_path.resolve()] if dataset_first else [tmp_path.resolve(), generated],
        )
    )
    spec = yaml.safe_load((generated / "suite_2.yaml").read_text())
    assert spec["task"] == "suite_2"
    assert spec["dataset_kwargs"] == {"data_files": str(data.resolve())}


@pytest.mark.parametrize("dataset_first", [True, False])
def test_dataset_tasks_avoid_explicit_yaml_numeric_suffixes(tmp_path, dataset_first):
    data = [tmp_path / "qa.jsonl", tmp_path / "qa.csv"]
    data[0].write_text("{}\n")
    data[1].write_text("q,a\n")
    tasks = [tmp_path / "custom.yaml", tmp_path / "other.yaml"]
    tasks[0].write_text("task: qa\n")
    tasks[1].write_text("task: qa_2\n")
    entries = data + tasks if dataset_first else tasks + data
    names, _ = evalmod.resolve_tasks(",".join(map(str, entries)), "q", "a", tmp_path / "tmp")
    assert names == (
        ["qa_3", "qa_4", "qa", "qa_2"] if dataset_first else ["qa", "qa_2", "qa_3", "qa_4"]
    )
    for name, path in zip(["qa_3", "qa_4"], data):
        spec = yaml.safe_load((tmp_path / "tmp" / "generated" / f"{name}.yaml").read_text())
        assert spec["task"] == name
        assert spec["dataset_kwargs"] == {"data_files": str(path.resolve())}


@pytest.mark.parametrize("duplicate", ["yaml", "same_yaml", "plain_first", "plain_last"])
def test_explicit_yaml_names_still_reject_duplicates(tmp_path, duplicate):
    task = tmp_path / "custom.yaml"
    task.write_text("task: qa\n")
    other = tmp_path / "other.yaml"
    other.write_text("task: qa\n")
    entries = {
        "yaml": [str(task), str(other)],
        "same_yaml": [str(task), str(task)],
        "plain_first": ["qa", str(task)],
        "plain_last": [str(task), "qa"],
    }[duplicate]
    with pytest.raises(ValueError, match = "Duplicate task 'qa' in --tasks"):
        evalmod.resolve_tasks(",".join(entries), "q", "a", tmp_path / "tmp")


def test_standalone_yaml_is_copied_alone(tmp_path):
    (tmp_path / "broken.yaml").write_text("task: [unclosed\n")
    task = tmp_path / "mine.yml"
    task.write_text("task: my_task\ndataset_path: json\n")
    names, includes = evalmod.resolve_tasks(str(task), "q", "a", tmp_path / "tmp")
    assert names == ["my_task"]
    assert sorted(p.name for p in Path(includes[0]).iterdir()) == ["my_task.yaml"]


def test_group_yaml_puts_its_directory_on_the_include_path(tmp_path):
    group = tmp_path / "suite.yaml"
    group.write_text("group: suite\ntask: [a, b]\n")
    names, includes = evalmod.resolve_tasks(str(group), "q", "a", tmp_path / "tmp")
    assert names == ["suite"]
    assert includes == [str(tmp_path.resolve())]


def test_function_tag_yaml_uses_its_directory(tmp_path):
    task = tmp_path / "fn.yaml"
    task.write_text("task: fn_task\ndoc_to_text: !function utils.doc_to_text\n")
    names, includes = evalmod.resolve_tasks(str(task), "q", "a", tmp_path / "tmp")
    assert names == ["fn_task"]
    assert includes == [str(tmp_path.resolve())]


@pytest.mark.parametrize(
    "filename, content",
    [
        ("bad.yaml", "task: [unclosed\n"),
        ("list.yaml", "- a\n- b\n"),
        ("noname.yaml", "dataset_path: json\n"),
        ("group_no_name.yaml", "task: [a, b]\n"),
        ("sibling.yml", "group: g\ntask: [a]\n"),
    ],
)
def test_invalid_yaml_tasks(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content)
    with pytest.raises(ValueError):
        evalmod.resolve_tasks(str(path), "q", "a", tmp_path / "tmp")


def _fake_torch(
    cuda_count = 0,
    mps = False,
    xpu_count = 0,
):
    xpu = SimpleNamespace(is_available = lambda: xpu_count > 0, device_count = lambda: xpu_count)
    return SimpleNamespace(
        cuda = SimpleNamespace(is_available = lambda: cuda_count > 0, device_count = lambda: cuda_count),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: mps)),
        xpu = xpu,
    )


@pytest.mark.parametrize(
    "device, ok",
    [
        ("cpu", True),
        ("cuda", True),
        ("cuda:1", True),
        ("cuda:2", False),
        ("cuda:01", False),
        ("cpu:0", False),
        ("mps", False),
        ("xpu", False),
        ("xpu:0", True),
        ("npu:0", False),
        ("tpu", False),
    ],
)
def test_hf_device_error(monkeypatch, device, ok):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_count = 2, xpu_count = 1))
    assert (evalmod._hf_device_error(device) is None) is ok


def test_hflm_4bit_passes_a_quantization_config_not_load_in_4bit():
    # transformers without torch cannot build BitsAndBytesConfig (macOS runners)
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BitsAndBytesConfig

    class _Base:
        def _create_model(
            self,
            pretrained,
            quantization_config = None,
            **kwargs,
        ):
            self.seen = dict(kwargs, quantization_config = quantization_config)

    lm = evalmod._hflm_4bit(_Base)()
    lm._create_model(pretrained = "x", load_in_4bit = True, quantization_config = None)
    assert "load_in_4bit" not in lm.seen
    assert isinstance(lm.seen["quantization_config"], BitsAndBytesConfig)
    assert lm.seen["quantization_config"].load_in_4bit is True

    prequantized = object()
    lm._create_model(pretrained = "x", load_in_4bit = True, quantization_config = prequantized)
    assert lm.seen["quantization_config"] is prequantized


def test_metric_number_and_render(capsys):
    np = pytest.importorskip("numpy")
    assert evalmod._metric_number(np.float32(0.5)) == 0.5
    assert evalmod._metric_number("x") is None
    evalmod._render_results(
        {
            "results": {
                "gsm8k": {"exact_match,strict": np.float32(0.3), "exact_match_stderr,strict": 0.01}
            },
            "groups": {"mmlu": {"acc,none": 0.45, "alias": "mmlu"}},
        }
    )
    out = capsys.readouterr().out
    assert "0.3000" in out and "0.0100" in out and "0.4500" in out


def test_eval_missing_lm_eval_shows_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "lm_eval", raising = False)
    monkeypatch.setattr(evalmod, "find_spec", lambda name: None)
    result = _run("fake/model", "--tasks", "gsm8k")
    assert result.exit_code == 1, result.output
    assert "pip install unsloth[eval]" in result.output


@pytest.fixture
def env(monkeypatch):
    calls = {"task_managers": 0}

    class _FakeFLM:
        @classmethod
        def from_pretrained(
            cls,
            model_name = None,
            **kw,
        ):
            calls["model_name"] = model_name
            calls["load_kwargs"] = kw
            model = SimpleNamespace(
                get_input_embeddings = lambda: SimpleNamespace(weight = SimpleNamespace(shape = (10, 4))),
                resize_token_embeddings = lambda n: calls.__setitem__("resized", n),
            )
            return model, SimpleNamespace(name = "tok")

        @classmethod
        def for_inference(cls, model):
            calls["for_inference"] = True

    class _FakeHFLM:
        def __init__(
            self,
            pretrained = None,
            tokenizer = None,
            batch_size = None,
            max_length = None,
            **kw,
        ):
            calls["hflm"] = dict(
                kw, pretrained = pretrained, batch_size = batch_size, max_length = max_length
            )

    class _FakeTaskManager:
        def __init__(self, include_path = None):
            calls["task_managers"] += 1
            calls["include_path"] = include_path
            self.all_tasks = ["gsm8k", "mmlu", "mmlu_a", "mmlu_b"]
            for directory in include_path or []:
                self.all_tasks += [p.stem for p in Path(directory).glob("*.yaml")]

        def match_tasks(self, patterns):
            import fnmatch
            return [t for p in patterns for t in self.all_tasks if fnmatch.fnmatch(t, p)]

    def _simple_evaluate(
        model = None,
        model_args = None,
        tasks = None,
        confirm_run_unsafe_code = False,
        **kw,
    ):
        calls.update(model = model, model_args = model_args, tasks = tasks, kwargs = kw)
        calls["confirm_run_unsafe_code"] = confirm_run_unsafe_code
        np = pytest.importorskip("numpy")
        return {"results": {"gsm8k": {"exact_match,strict": np.float64(0.42)}}, "configs": {}}

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = _FakeFLM
    lm_eval_mod = types.ModuleType("lm_eval")
    lm_eval_mod.simple_evaluate = _simple_evaluate
    hf_mod = types.ModuleType("lm_eval.models.huggingface")
    hf_mod.HFLM = _FakeHFLM
    tasks_mod = types.ModuleType("lm_eval.tasks")
    tasks_mod.TaskManager = _FakeTaskManager
    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = SimpleNamespace(
        from_pretrained = lambda m, path: calls.__setitem__("peft", path) or m
    )
    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.hf_hub_download = hub_mod.list_repo_files = lambda *a, **k: (_ for _ in ()).throw(
        OSError()
    )
    for name, mod in {
        "unsloth": unsloth_mod,
        "torch": _fake_torch(),
        "huggingface_hub": hub_mod,
        "peft": peft_mod,
        "lm_eval": lm_eval_mod,
        "lm_eval.models": types.ModuleType("lm_eval.models"),
        "lm_eval.models.huggingface": hf_mod,
        "lm_eval.tasks": tasks_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(evalmod, "find_spec", lambda name: object())
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    return calls


def test_unsloth_backend_writes_results(env, tmp_path):
    result = _run("fake/model", "--tasks", "gsm8k", "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["model_name"] == "fake/model"
    assert env["load_kwargs"]["load_in_4bit"] is True
    assert env["for_inference"]
    assert env["hflm"]["max_length"] == 2048 and env["hflm"]["batch_size"] == "auto"
    assert env["tasks"] == ["gsm8k"] and env["task_managers"] == 1
    assert env["kwargs"]["log_samples"] is False
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["results"]["gsm8k"]["exact_match,strict"] == 0.42


def test_eval_dataset_before_yaml_with_same_task_name(env, tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text("{}\n")
    task = tmp_path / "custom.yaml"
    task.write_text("task: qa\ndataset_path: json\n")
    result = _run("fake/model", "--tasks", f"{data},{task}", "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["tasks"] == ["qa_2", "qa"]
    assert env["task_managers"] == 1


def test_default_output_keeps_every_run(env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evalmod, "time", SimpleNamespace(time = lambda: 1761480000), raising = False)
    for model in ("org/base", "org/tuned", "org/tuned"):
        result = _run(model, "--tasks", "gsm8k")
        assert result.exit_code == 0, result.output
    saved = sorted(p.parent.name for p in (tmp_path / "eval_results").rglob("results.json"))
    assert saved == ["org_base_1761480000", "org_tuned_1761480000", "org_tuned_1761480001"], saved


@pytest.mark.skipif(sys.version_info < (3, 10), reason = "the Studio backend needs Python 3.10+")
@pytest.mark.parametrize(
    "model", ["org/tuned", "./my model/ckpt-100", "/abs/lora/", "C:\\x\\y", "~/", "", "x" * 300]
)
def test_run_dir_name_matches_studio(model, monkeypatch):
    pytest.importorskip("structlog")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "studio" / "backend"))
    from utils.paths import default_run_dir_name

    assert evalmod._run_dir_name(model) == default_run_dir_name(model)


def _adapter(tmp_path, tokenizer = False):
    adapter = tmp_path / "lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    if tokenizer:
        (adapter / "tokenizer.json").write_text("{}")
    return adapter


def test_unsloth_backend_loads_adapter_through_unsloth(env, tmp_path):
    adapter = _adapter(tmp_path)
    result = _run(str(adapter), "--tasks", "gsm8k", "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["model_name"] == str(adapter)
    assert "peft" not in env and "resize_model_vocab" not in env["load_kwargs"]


def test_unsloth_backend_grows_vocab_for_added_tokens(env, tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, tokenizer = True)
    tok = types.ModuleType("transformers")
    tok.AutoTokenizer = SimpleNamespace(from_pretrained = lambda path: [0] * 12)
    monkeypatch.setitem(sys.modules, "transformers", tok)
    monkeypatch.setattr(evalmod, "_base_vocab_size", lambda model: 10)
    assert _run(str(adapter), "--tasks", "gsm8k", "-o", str(tmp_path / "out")).exit_code == 0
    assert env["load_kwargs"]["resize_model_vocab"] == 12
    monkeypatch.setattr(evalmod, "_base_vocab_size", lambda model: 16)
    assert _run(str(adapter), "--tasks", "gsm8k", "-o", str(tmp_path / "out")).exit_code == 0
    assert "resize_model_vocab" not in env["load_kwargs"]


def test_base_model_override_attaches_adapter_manually(env, tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, tokenizer = True)
    tok = types.ModuleType("transformers")
    tok.AutoTokenizer = SimpleNamespace(from_pretrained = lambda path: [0] * 8)
    monkeypatch.setitem(sys.modules, "transformers", tok)
    args = (
        str(adapter),
        "--tasks",
        "gsm8k",
        "--base-model",
        "org/other",
        "-o",
        str(tmp_path / "o"),
    )
    result = _run(*args)
    assert result.exit_code == 0, result.output
    assert env["model_name"] == "org/other" and env["peft"] == str(adapter)
    assert "resized" not in env
    tok.AutoTokenizer = SimpleNamespace(from_pretrained = lambda path: [0] * 11)
    assert _run(*args).exit_code == 0
    assert env["resized"] == 11


def test_unsloth_backend_reports_missing_bitsandbytes(env, tmp_path, monkeypatch):
    monkeypatch.setattr(
        evalmod, "find_spec", lambda name: None if name == "bitsandbytes" else object()
    )
    result = _run("fake/model", "--tasks", "gsm8k", "-o", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "bitsandbytes is not installed" in result.output
    assert env["load_kwargs"]["load_in_4bit"] is False


@pytest.mark.parametrize(
    "run_config, expected",
    [
        ({"unsloth_training_method": "lora", "unsloth_load_in_4bit": False}, False),
        ({"unsloth_training_method": "qlora", "unsloth_load_in_4bit": True}, True),
        (None, True),
    ],
    ids = ["lora_16bit_run", "qlora_run", "unfinished_run"],
)
def test_checkpoint_follows_its_run(env, tmp_path, monkeypatch, run_config, expected):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    run = tmp_path / "studio" / "outputs" / "run"
    checkpoint = run / "checkpoint-15"
    checkpoint.mkdir(parents = True)
    base = {"base_model_name_or_path": "org/base"}
    (checkpoint / "adapter_config.json").write_text(json.dumps(base))
    if run_config is not None:
        (run / "adapter_config.json").write_text(json.dumps({**base, **run_config}))
    result = _run(str(checkpoint), "--tasks", "gsm8k", "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["load_kwargs"]["load_in_4bit"] is expected


@pytest.mark.parametrize(
    "where, expected",
    [
        ("outputs", False),
        ("linked", False),
        ("accounts/acct1/outputs", False),
        ("accounts/acct1/exports", True),
    ],
)
def test_full_finetune_in_any_account_outputs_loads_in_16bit(
    env, tmp_path, monkeypatch, where, expected
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    if where == "linked":
        (tmp_path / "disk").mkdir()
        (tmp_path / "studio").mkdir()
        (tmp_path / "studio" / "outputs").symlink_to(tmp_path / "disk", target_is_directory = True)
        where = "outputs"
    run = tmp_path / "studio" / where / "run"
    run.mkdir(parents = True)
    (run / "config.json").write_text(json.dumps({"model_type": "llama"}))
    result = _run(str(run), "--tasks", "gsm8k", "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["load_kwargs"]["load_in_4bit"] is expected


@pytest.mark.parametrize(
    "files, flags, in_outputs, expected",
    [
        ({"config.json": {"model_type": "llama"}}, [], True, False),
        ({"config.json": {"model_type": "llama"}}, [], False, True),
        ({"config.json": {"text_config": {"quantization_config": {}}}}, [], False, True),
        ({"config.json": {"quantization_config": {"load_in_4bit": True}}}, [], True, True),
        ({"adapter_config.json": {"unsloth_load_in_4bit": False}}, [], False, False),
        ({"adapter_config.json": {"unsloth_training_method": "lora"}}, [], False, False),
        ({"adapter_config.json": {"unsloth_load_in_4bit": True}}, [], True, True),
        ({"adapter_config.json": {}}, [], False, True),
        ({"config.json": {"model_type": "llama"}}, ["--load-in-4bit"], True, True),
        (
            {"adapter_config.json": {"unsloth_load_in_4bit": True}},
            ["--no-load-in-4bit"],
            True,
            False,
        ),
    ],
    ids = [
        "full_finetune",
        "downloaded_full_model",
        "downloaded_nested_quantized",
        "prequantized",
        "lora_16bit",
        "lora_method_only",
        "qlora",
        "unknown_adapter",
        "explicit_4bit",
        "explicit_16bit",
    ],
)
def test_checkpoint_loads_in_the_precision_it_was_trained_in(
    env, tmp_path, monkeypatch, files, flags, in_outputs, expected
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    run = (tmp_path / "studio" / "outputs" if in_outputs else tmp_path) / "run"
    run.mkdir(parents = True)
    for name, data in files.items():
        (run / name).write_text(json.dumps({"base_model_name_or_path": "org/base", **data}))
    result = _run(str(run), "--tasks", "gsm8k", *flags, "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["load_kwargs"]["load_in_4bit"] is expected


def test_custom_dataset_builds_one_task_manager(env, tmp_path):
    data = tmp_path / "qa.jsonl"
    data.write_text('{"question": "q", "answer": "a"}\n')
    result = _run("fake/model", "--tasks", f"gsm8k,{data}", "-o", str(tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert env["tasks"] == ["gsm8k", "qa"]
    assert env["task_managers"] == 1 and len(env["include_path"]) == 1


def test_glob_expands_and_unknown_task_errors(env, tmp_path):
    assert _run("fake/model", "--tasks", "mmlu_*", "-o", str(tmp_path)).exit_code == 0
    assert env["tasks"] == ["mmlu_a", "mmlu_b"]
    for tasks in ("nope", "zzz_*"):
        result = _run("fake/model", "--tasks", tasks)
        assert result.exit_code == 2 and "Error:" in result.output


def test_hf_backend_cpu(env, tmp_path):
    result = _run("fake/model", "--tasks", "gsm8k", "--backend", "hf", "-o", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert env["model"] == "hf"
    assert env["model_args"] == {"pretrained": "fake/model", "max_length": 2048}
    assert env["kwargs"]["device"] == "cpu" and env["kwargs"]["batch_size"] == 1
    assert "model_name" not in env


def test_hf_backend_cuda_4bit_builds_quantizing_hflm(env, tmp_path):
    sys.modules["torch"].cuda = SimpleNamespace(is_available = lambda: True, device_count = lambda: 1)
    result = _run(
        "fake/model",
        "--tasks",
        "gsm8k",
        "--backend",
        "hf",
        "--device",
        "cuda:0",
        "-o",
        str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert type(env["model"]).__name__ == "_HFLM4bit"
    assert env["hflm"]["load_in_4bit"] is True and env["hflm"]["device"] == "cuda:0"


def test_hf_backend_adapter_uses_peft_arg(env, tmp_path):
    adapter = tmp_path / "lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    (adapter / "tokenizer.json").write_text("{}")
    result = _run(str(adapter), "--tasks", "gsm8k", "--backend", "hf", "-o", str(tmp_path / "o"))
    assert result.exit_code == 0, result.output
    assert env["model_args"]["pretrained"] == "org/base"
    assert env["model_args"]["peft"] == env["model_args"]["tokenizer"] == str(adapter)


def test_mlx_falls_back_to_hf(env, tmp_path):
    sys.modules["unsloth"].DEVICE_TYPE = "mlx"
    result = _run("fake/model", "--tasks", "gsm8k", "-o", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "falling back" in result.output and env["model"] == "hf"


def test_unsafe_code_flag_is_forwarded(env, tmp_path):
    assert (
        _run(
            "fake/model", "--tasks", "gsm8k", "--confirm-run-unsafe-code", "-o", str(tmp_path)
        ).exit_code
        == 0
    )
    assert env["confirm_run_unsafe_code"] is True


@pytest.mark.parametrize(
    "args",
    [
        ["--batch-size", "0"],
        ["--batch-size", "-1"],
        ["--batch-size", "big"],
        ["--backend", "tgi"],
        ["--backend", "vllm", "--gpu-memory-utilization", "1.5"],
        ["--num-fewshot", "-1"],
        ["--limit", "0"],
        ["--limit", "2.5"],
        ["--max-seq-length", "0"],
        ["--backend", "hf", "--device", "cuda:9"],
    ],
)
def test_invalid_arguments_exit_2(env, args):
    result = _run("fake/model", "--tasks", "gsm8k", *args)
    assert result.exit_code == 2, result.output
    assert "model" not in env


def test_fractional_and_whole_limits(env, tmp_path):
    assert (
        _run("fake/model", "--tasks", "gsm8k", "--limit", "0.25", "-o", str(tmp_path)).exit_code
        == 0
    )
    assert env["kwargs"]["limit"] == 0.25
    assert (
        _run("fake/model", "--tasks", "gsm8k", "--limit", "5", "-o", str(tmp_path)).exit_code == 0
    )
    assert env["kwargs"]["limit"] == 5 and isinstance(env["kwargs"]["limit"], int)


def test_unsloth_backend_refuses_multi_process(env, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    result = _run("fake/model", "--tasks", "gsm8k")
    assert result.exit_code == 2 and "--backend hf" in result.output


@pytest.fixture
def vllm_env(env, monkeypatch):
    sys.modules["torch"].cuda = SimpleNamespace(is_available = lambda: True, device_count = lambda: 1)
    monkeypatch.setattr(evalmod.sys, "platform", "linux")
    monkeypatch.setattr(evalmod, "version", lambda name: "0.4.13", raising = False)
    return env


def test_vllm_backend_builds_lm_eval_vllm(vllm_env, tmp_path):
    result = _run(
        "fake/model",
        "--tasks",
        "gsm8k",
        "--backend",
        "vllm",
        "--max-seq-length",
        "4096",
        "--gpu-memory-utilization",
        "0.5",
        "-o",
        str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert vllm_env["model"] == "vllm"
    assert vllm_env["model_args"] == {
        "pretrained": "fake/model",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.5,
    }
    assert vllm_env["kwargs"]["batch_size"] == "auto"
    assert "model_name" not in vllm_env and "hflm" not in vllm_env


def test_vllm_backend_serves_adapter_as_lora(vllm_env, tmp_path):
    adapter = tmp_path / "lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-8B-unsloth-bnb-4bit", "r": 64})
    )
    (adapter / "tokenizer.json").write_text("{}")
    result = _run(
        str(adapter), "--tasks", "gsm8k", "--backend", "vllm", "-b", "8", "-o", str(tmp_path / "o")
    )
    assert result.exit_code == 0, result.output
    args = vllm_env["model_args"]
    assert args["pretrained"] == "unsloth/Qwen3-8B"
    assert args["lora_local_path"] == args["tokenizer"] == str(adapter.resolve())
    assert args["max_lora_rank"] == 64
    assert vllm_env["kwargs"]["batch_size"] == 8


def test_vllm_backend_keeps_default_lora_rank_and_ignores_4bit(vllm_env, tmp_path):
    adapter = _adapter(tmp_path)
    result = _run(
        str(adapter),
        "--tasks",
        "gsm8k",
        "--backend",
        "vllm",
        "--load-in-4bit",
        "-o",
        str(tmp_path / "o"),
    )
    assert result.exit_code == 0, result.output
    assert vllm_env["model_args"]["pretrained"] == "org/base"
    assert (
        "max_lora_rank" not in vllm_env["model_args"] and "tokenizer" not in vllm_env["model_args"]
    )
    assert "16-bit" in result.output


def test_vllm_backend_refuses_multi_process(vllm_env, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    result = _run("fake/model", "--tasks", "gsm8k", "--backend", "vllm")
    assert result.exit_code == 2 and "--backend hf" in result.output
    assert "model" not in vllm_env


@pytest.mark.parametrize("installed, ok", [("0.4.11", False), ("0.4.12", True)])
def test_vllm_backend_needs_lm_eval_0_4_12(vllm_env, monkeypatch, tmp_path, installed, ok):
    monkeypatch.setattr(evalmod, "version", lambda name: installed)
    result = _run("fake/model", "--tasks", "gsm8k", "--backend", "vllm", "-o", str(tmp_path))
    if ok:
        assert result.exit_code == 0, result.output
    else:
        assert result.exit_code == 2 and "model" not in vllm_env
        assert installed in result.output and "pip install -U lm-eval" in result.output


def test_vllm_engine_failure_prints_the_engine_log(vllm_env, monkeypatch, tmp_path):
    def _fail_in_engine(**kwargs):
        os.write(2, b"ValueError: No available memory for the cache blocks.\n")
        raise RuntimeError("Engine core initialization failed. See root cause above.")

    monkeypatch.setattr(sys.modules["lm_eval"], "simple_evaluate", _fail_in_engine)
    result = _run("fake/model", "--tasks", "gsm8k", "--backend", "vllm", "-o", str(tmp_path))
    assert result.exit_code != 0
    assert "No available memory for the cache blocks" in result.output


def test_vllm_gpu_probe_leaves_cuda_uninitialised(vllm_env, monkeypatch, tmp_path):
    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising = False)
    seen = []
    sys.modules["torch"].cuda = SimpleNamespace(
        is_available = lambda: seen.append(os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")) or True,
        device_count = lambda: 1,
    )
    result = _run("fake/model", "--tasks", "gsm8k", "--backend", "vllm", "-o", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert seen == ["1"]


@pytest.mark.parametrize(
    "platform, cuda, vllm_installed, hint",
    [
        ("linux", True, False, "pip install vllm"),
        ("linux", False, True, "--backend hf"),
        ("darwin", True, True, "On a Mac use --backend hf"),
    ],
)
def test_vllm_backend_refuses_where_it_cannot_run(
    env, monkeypatch, platform, cuda, vllm_installed, hint
):
    sys.modules["torch"].cuda = SimpleNamespace(is_available = lambda: cuda, device_count = lambda: 1)
    monkeypatch.setattr(evalmod.sys, "platform", platform)
    monkeypatch.setattr(evalmod, "version", lambda name: "0.4.13", raising = False)
    monkeypatch.setattr(
        evalmod,
        "find_spec",
        lambda name: None if name == "vllm" and not vllm_installed else object(),
    )
    result = _run("fake/model", "--tasks", "gsm8k", "--backend", "vllm")
    assert result.exit_code == 2, result.output
    assert hint in result.output and "model" not in env


def test_silence_keeps_the_reconfigured_stdout_encoding(tmp_path):
    import subprocess

    script = tmp_path / "enc.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "from unsloth_cli.commands.eval import _silence\n"
        "with _silence() as c:\n"
        "    c.print('Evaluating gsm8k\\u2026')\n"
    )
    env = dict(os.environ, PYTHONPATH = str(_REPO_ROOT), PYTHONCOERCECLOCALE = "0", LC_ALL = "C")
    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", str(script)], capture_output = True, env = env
    )
    assert proc.returncode == 0, proc.stderr.decode(errors = "replace")
    assert "Evaluating gsm8k\u2026" in proc.stdout.decode("utf-8")


@pytest.mark.parametrize("backend", ["unsloth", "hf"])
def test_gpu_memory_utilization_says_it_only_applies_to_vllm(env, monkeypatch, tmp_path, backend):
    """Only the vllm branch forwards it, so a user capping memory on another backend would
    otherwise read the dropped cap as an OOM."""
    result = _run(
        "org/model",
        "--tasks",
        "gsm8k",
        "--backend",
        backend,
        "--gpu-memory-utilization",
        "0.5",
        "-o",
        str(tmp_path / "o"),
    )
    assert result.exit_code == 0, result.output
    assert "--gpu-memory-utilization applies to --backend vllm" in result.output
    assert "gpu_memory_utilization" not in (env["model_args"] or {})

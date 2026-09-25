# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the OpenVINO export in unsloth.save.

The merge and the optimum-cli subprocess are faked in all but the last test, so these run on
CPU-only CI without optimum-intel. The last one runs a real export of a tiny random Llama when
optimum-intel is installed.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import subprocess
import sys

import pytest
import torch.nn as nn

import unsloth.save as save_mod
from unsloth.save import (
    patch_saving_functions,
    unsloth_push_to_hub_openvino,
    unsloth_save_pretrained_openvino,
)

_REAL_CLI_PARSER = save_mod._openvino_cli_parser


class _FakeModel:
    config = type(
        "cfg", (), {"_name_or_path": "fake/model", "architectures": ["LlamaForCausalLM"]}
    )()


def _fake_cli_parser():
    """The optimum-cli options these tests use, so they run without optimum-intel."""
    parser = argparse.ArgumentParser(prog = "optimum-cli export openvino")
    parser.add_argument("output")
    parser.add_argument("-m", "--model", required = True)
    parser.add_argument("--weight-format", choices = ["fp32", "fp16", "int8", "int4", "nf4"])
    parser.add_argument("--sym", action = "store_true", default = None)
    parser.add_argument("--group-size", type = int)
    parser.add_argument("--ratio", type = float)
    parser.add_argument("--library")
    parser.add_argument("--task")
    parser.add_argument("--trust-remote-code", action = "store_true")
    return parser


@pytest.fixture
def export(monkeypatch, tmp_path):
    """Call _unsloth_save_openvino with the merge and the subprocess faked, recording both."""
    seen = {"merges": [], "cmds": [], "envs": [], "staging": []}
    monkeypatch.setattr(save_mod, "_openvino_cli_parser", _fake_cli_parser)
    monkeypatch.setattr(save_mod, "get_token", lambda: None)

    def fake_merge(**kwargs):
        seen["merges"].append(kwargs)
        os.makedirs(kwargs["save_directory"], exist_ok = True)
        with open(os.path.join(kwargs["save_directory"], "config.json"), "w") as f:
            f.write("{}")

    def fake_check_call(cmd, env = None):
        seen["cmds"].append(cmd)
        seen["envs"].append(env)
        staging = cmd[cmd.index("--model") + 1]
        seen["staging"].append((staging, os.path.isfile(os.path.join(staging, "config.json"))))
        os.makedirs(cmd[-1], exist_ok = True)
        for name in ("openvino_model.xml", "openvino_model.bin", "openvino_tokenizer.xml"):
            open(os.path.join(cmd[-1], name), "w").close()
        return 0

    monkeypatch.setattr(save_mod, "unsloth_generic_save", fake_merge)
    monkeypatch.setattr(save_mod.subprocess, "check_call", fake_check_call)

    def run(model = None, **kwargs):
        kwargs.setdefault("save_directory", str(tmp_path / "ov_out"))
        return save_mod._unsloth_save_openvino(model = model or _FakeModel(), **kwargs)

    run.seen = seen
    return run


def _leftover_staging(directory):
    return [name for name in os.listdir(directory) if "unsloth-openvino-" in name]


def test_openvino_export_signatures():
    save_sig = inspect.signature(unsloth_save_pretrained_openvino)
    for name in ("save_directory", "tokenizer", "quantization_type", "push_to_hub", "token"):
        assert name in save_sig.parameters
    assert "private" in save_sig.parameters
    hub_sig = inspect.signature(unsloth_push_to_hub_openvino)
    for name in ("repo_id", "tokenizer", "quantization_type", "token", "private"):
        assert name in hub_sig.parameters


@pytest.mark.parametrize("vision", [False, True])
def test_openvino_methods_attached_by_patch_saving_functions(vision):
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.config = type("cfg", (), {})()

        def push_to_hub(self, repo_id, **kwargs):
            """Push to hub docstring."""

    patched = patch_saving_functions(DummyModel(), vision = vision)
    assert callable(patched.save_pretrained_openvino)
    assert callable(patched.push_to_hub_openvino)


def test_export_runs_in_a_separate_process_after_the_merge(export, tmp_path):
    result = export(tokenizer = None)
    assert result == str(tmp_path / "ov_out")
    (merge,) = export.seen["merges"]
    assert merge["save_method"] == "merged_16bit" and merge["push_to_hub"] is False
    (cmd,) = export.seen["cmds"]
    # Not in this process: optimum would trace transformers classes Unsloth has patched.
    assert cmd[:5] == [sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino"]
    assert cmd[-1] == os.path.abspath(tmp_path / "ov_out")
    assert export.seen["staging"][0][1], "the export must read the finished merge"


@pytest.mark.parametrize("quantization_type", [None, "fp16", "F16", "none"])
def test_unquantized_export_pins_fp16(export, quantization_type):
    export(quantization_type = quantization_type)
    cmd = export.seen["cmds"][0]
    # Without a weight format optimum-intel int8-compresses any model over 1B parameters.
    assert cmd[cmd.index("--weight-format") + 1] == "fp16"
    assert "--sym" not in cmd


@pytest.mark.parametrize(
    "quantization_type, expected",
    [
        ("int8", ["--weight-format", "int8", "--sym"]),
        ("8bit", ["--weight-format", "int8", "--sym"]),
        ("int4", ["--weight-format", "int4", "--sym", "--group-size", "128"]),
        ("4", ["--weight-format", "int4", "--sym", "--group-size", "128"]),
    ],
)
def test_quantization_presets(export, quantization_type, expected):
    export(quantization_type = quantization_type)
    cmd = export.seen["cmds"][0]
    start = cmd.index("--weight-format")
    assert cmd[start : start + len(expected)] == expected


def test_keyword_options_override_the_preset(export):
    export(quantization_type = "int4", sym = False, group_size = 64, ratio = 0.8)
    cmd = export.seen["cmds"][0]
    assert "--sym" not in cmd
    assert cmd[cmd.index("--group-size") + 1] == "64"
    assert cmd[cmd.index("--ratio") + 1] == "0.8"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"quantization_type": "nf5"}, "Unknown OpenVINO quantization_type"),
        ({"quantization_type": "int4", "group_sise": 64}, "Invalid OpenVINO export option"),
        ({"weight_format": "int3"}, "Invalid OpenVINO export option"),
        ({"output": "elsewhere"}, "set by save_pretrained_openvino"),
    ],
)
def test_bad_options_fail_before_the_merge(export, tmp_path, kwargs, error):
    with pytest.raises(ValueError, match = error):
        export(**kwargs)
    assert export.seen["merges"] == [] and export.seen["cmds"] == []
    assert _leftover_staging(tmp_path) == []


def test_missing_optimum_intel_fails_before_the_merge(export, monkeypatch):
    monkeypatch.setattr(save_mod, "_openvino_cli_parser", _REAL_CLI_PARSER)
    monkeypatch.setitem(sys.modules, "optimum.commands.export.openvino", None)
    with pytest.raises(ImportError, match = "requires `optimum-intel` and `openvino`"):
        export()
    assert export.seen["merges"] == []


def test_staging_sits_beside_the_destination_and_is_removed(export, tmp_path):
    export()
    staging, _ = export.seen["staging"][0]
    assert os.path.dirname(os.path.dirname(staging)) == str(tmp_path)
    assert os.path.basename(os.path.dirname(staging)).startswith(".unsloth-openvino-")
    assert _leftover_staging(tmp_path) == []
    assert os.path.isfile(tmp_path / "ov_out" / "openvino_model.xml")


def test_staging_is_removed_when_the_export_fails(export, monkeypatch, tmp_path):
    def failing_check_call(cmd, env = None):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(save_mod.subprocess, "check_call", failing_check_call)
    with pytest.raises(RuntimeError, match = "optimum-cli exit 1"):
        export()
    assert _leftover_staging(tmp_path) == []


def test_export_that_writes_no_model_is_an_error(export, monkeypatch, tmp_path):
    monkeypatch.setattr(save_mod.subprocess, "check_call", lambda cmd, env = None: 0)
    with pytest.raises(RuntimeError, match = "wrote no model"):
        export()


def test_child_process_gets_no_hub_token(export, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_parent_secret")
    export(token = "hf_explicit_secret")
    env = export.seen["envs"][0]
    assert "hf_parent_secret" not in env.values()
    assert "hf_explicit_secret" not in env.values()
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_task_follows_the_model_type(export):
    export()
    cmd = export.seen["cmds"][-1]
    # optimum-cli cannot infer a task from the local staging directory.
    assert cmd[cmd.index("--task") + 1] == "text-generation-with-past"

    class _FakeVLM:
        config = type(
            "cfg",
            (),
            {"architectures": ["Qwen2VLForConditionalGeneration"], "vision_config": {}},
        )()

    export(model = _FakeVLM())
    cmd = export.seen["cmds"][-1]
    assert cmd[cmd.index("--task") + 1] == "image-text-to-text"


def test_remote_code_is_trusted_only_for_a_remote_code_model(export, monkeypatch):
    export()
    assert "--trust-remote-code" not in export.seen["cmds"][-1]
    remote = _FakeModel()
    monkeypatch.setattr(save_mod, "_loaded_via_remote_code", lambda obj: obj is remote)
    export(model = remote)
    assert "--trust-remote-code" in export.seen["cmds"][-1]


def test_non_main_process_does_nothing(export):
    assert export(is_main_process = False) is None
    assert export.seen["merges"] == [] and export.seen["cmds"] == []


def test_push_forwards_private_and_hub_arguments(export, monkeypatch):
    calls = []

    class RecordingHfApi:
        def __init__(self, token = None):
            calls.append(("init", token))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs, len(export.seen["merges"])))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs, sorted(os.listdir(kwargs["folder_path"]))))

    monkeypatch.setattr(save_mod, "HfApi", RecordingHfApi)
    result = unsloth_push_to_hub_openvino(
        _FakeModel(),
        "me/secret-model",
        token = "hf_x",
        private = True,
        commit_message = "mine",
        create_pr = True,
        revision = "dev",
    )
    assert result == "me/secret-model"
    assert calls[0] == ("init", "hf_x")
    _, create, merges_before_create = calls[1]
    assert create["repo_id"] == "me/secret-model" and create["private"] is True
    assert merges_before_create == 0, "Hub access is checked before the merge"
    upload = calls[2][1]
    assert upload["repo_id"] == "me/secret-model"
    assert (upload["commit_message"], upload["create_pr"], upload["revision"]) == (
        "mine",
        True,
        "dev",
    )
    assert "openvino_model.xml" in calls[2][2]
    assert len(export.seen["merges"]) == 1
    assert not os.path.exists(upload["folder_path"])


def _tiny_llama_and_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE(unk_token = "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["the capital of france is paris"] * 8,
        trainers.BpeTrainer(
            vocab_size = 300,
            special_tokens = ["<unk>", "<s>", "</s>"],
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, bos_token = "<s>", eos_token = "</s>", unk_token = "<unk>"
    )
    config = LlamaConfig(
        vocab_size = len(tokenizer),
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 128,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(config), tokenizer


def test_real_export_while_the_forward_is_patched(monkeypatch, tmp_path):
    """A real optimum-cli export of a tiny random Llama, with LlamaModel.forward broken in this
    process the way Unsloth's LlamaModel_fast_forward breaks it for a model Unsloth did not load.
    An in-process export traces that forward and fails; the child process never sees it."""
    pytest.importorskip("optimum.intel")
    pytest.importorskip("openvino_tokenizers")
    import transformers

    model, tokenizer = _tiny_llama_and_tokenizer()

    original_forward = transformers.LlamaModel.forward

    @functools.wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        self.max_seq_length  # Unsloth's forward reads this; only models Unsloth loaded carry it
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(transformers.LlamaModel, "forward", patched_forward)
    out = tmp_path / "ov_out"
    save_mod._unsloth_save_openvino(model, str(out), tokenizer = tokenizer, token = False)

    files = set(os.listdir(out))
    assert {"openvino_model.xml", "openvino_model.bin", "openvino_tokenizer.xml"} <= files
    assert "openvino_detokenizer.xml" in files
    assert _leftover_staging(tmp_path) == []

    import openvino as ov

    weight_types = {
        str(op.get_output_element_type(0))
        for op in ov.Core().read_model(str(out / "openvino_model.xml")).get_ops()
        if op.get_type_name() == "Constant" and len(op.get_output_shape(0)) == 2
    }
    # 16bit, not the int8 optimum-intel picks when no weight format is given.
    assert weight_types & {"<Type: 'bfloat16'>", "<Type: 'float16'>"}, weight_types
    assert not any("int8" in t or "int4" in t for t in weight_types), weight_types

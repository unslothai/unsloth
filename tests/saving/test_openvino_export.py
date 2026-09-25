from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import unsloth.save as save_mod

_REAL_PARSER, _REAL_BOUNDS = save_mod._openvino_cli_parser, save_mod._openvino_transformers_mismatch
_OUT = ("openvino_model.xml", "openvino_tokenizer.xml")
_CASES = {"nf5": {"quantization_type": "nf5"}, "typo": {"group_sise": 64}, "out": {"output": "x"}}
_BOUNDS = {'[null, "0.1"]': "transformers <= 0.1", '["99.0", null]': "transformers >= 99.0"}
_llama = lambda: SimpleNamespace(config = SimpleNamespace(model_type = "llama"))


def _parser():
    parser = argparse.ArgumentParser()
    for flag in ("output", "--model", "--weight-format", "--group-size", "--library", "--task"):
        parser.add_argument(flag)
    for flag in ("--sym", "--trust-remote-code"):
        parser.add_argument(flag, action = "store_true")
    return parser


def _write(cmd, env, *names):
    os.makedirs(cmd[-1], exist_ok = True)
    for name in names or _OUT:
        open(os.path.join(cmd[-1], name), "w").close()


@pytest.fixture
def run(monkeypatch, tmp_path):
    seen = SimpleNamespace(merges = [], cmds = [], envs = [])
    monkeypatch.setattr(save_mod, "_openvino_cli_parser", _parser)
    monkeypatch.setattr(save_mod, "_openvino_transformers_mismatch", lambda *args: None)
    monkeypatch.setattr(save_mod, "unsloth_generic_save", lambda **kw: seen.merges.append(kw))
    call = lambda cmd, env: (seen.cmds.append(cmd), seen.envs.append(env), _write(cmd, env))
    monkeypatch.setattr(save_mod.subprocess, "check_call", call)

    def export(*model, **kwargs):
        out, kwargs["token"] = str(tmp_path / kwargs.pop("out", "out")), kwargs.get("token", False)
        return save_mod._unsloth_save_openvino(*(model or [_llama()]), out, **kwargs)

    export.seen, export.left = seen, lambda: [p for p in os.listdir(tmp_path) if p[0] == "."]
    return export


@pytest.mark.parametrize("quantization_type", [None, "F16", "int8", "4bit"])
def test_exports_the_merge_in_a_child_process(run, tmp_path, quantization_type):
    # In-process export traces classes Unsloth patched; no weight format int8s models over 1B.
    assert run(quantization_type = quantization_type) == str(tmp_path / "out")
    (cmd,), (merge,) = run.seen.cmds, run.seen.merges
    fmt = {None: "fp16", "F16": "fp16", "int8": "int8", "4bit": "int4"}[quantization_type]
    assert cmd[:5] == [sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino"]
    assert cmd[5:9] == ["--model", merge["save_directory"], "--weight-format", fmt]
    assert ("--sym" in cmd, "--group-size" in cmd) == (fmt != "fp16", fmt == "int4")
    assert "--trust-remote-code" not in cmd  # only for models loaded through remote code
    assert cmd[-3:] == ["--task", "text-generation-with-past", str(tmp_path / "out")]
    assert os.path.dirname(os.path.dirname(cmd[6])) == str(tmp_path) and not run.left()


def test_options_vlm_task_remote_code_token_and_missing_tokenizer(run, monkeypatch):
    vlm = SimpleNamespace(config = SimpleNamespace(model_type = "qwen2_vl", vision_config = {}))
    monkeypatch.setattr(save_mod, "_loaded_via_remote_code", lambda obj: obj is vlm)
    monkeypatch.setattr(save_mod.logger, "warning_once", (warnings := []).append)
    monkeypatch.setattr(save_mod, "_openvino_transformers_mismatch", _REAL_BOUNDS)
    monkeypatch.setattr(save_mod, "_OPENVINO_BOUNDS_PROBE", "raise SystemExit(3)")  # defers
    monkeypatch.setenv("HF_TOKEN", "hf_parent")
    run(vlm, quantization_type = "int4", sym = False, group_size = 64, token = "hf_explicit")
    (cmd,), (env,) = run.seen.cmds, run.seen.envs
    assert "--sym" not in cmd and cmd[cmd.index("--group-size") + 1] == "64"
    assert cmd[cmd.index("--task") + 1] == "image-text-to-text" and "--trust-remote-code" in cmd
    assert not {"hf_parent", "hf_explicit"} & {*env.values()}  # the child reads a local checkpoint
    assert run(is_main_process = False) is None and len(run.seen.merges) == 1 and not warnings
    monkeypatch.setattr(save_mod.subprocess, "check_call", lambda c, env: _write(c, env, _OUT[0]))
    run(out = "no_tokenizer")
    assert len(warnings) == 1 and "openvino_tokenizer.xml" in warnings[0]


@pytest.mark.parametrize("case", [*_CASES, *_BOUNDS, "no optimum", "child fails", "no model"])
def test_failures_come_before_the_merge_or_clean_up(run, monkeypatch, case):
    if case in _BOUNDS:  # optimum-intel's transformers bounds for this architecture
        monkeypatch.setattr(save_mod, "_openvino_transformers_mismatch", _REAL_BOUNDS)
        monkeypatch.setattr(save_mod, "_OPENVINO_BOUNDS_PROBE", f"print({case!r})")
    if case == "no optimum":
        monkeypatch.setattr(save_mod, "_openvino_cli_parser", _REAL_PARSER)
        monkeypatch.setitem(sys.modules, "optimum.commands.export.openvino", None)
    fail = lambda cmd, env: subprocess.run([sys.executable, "-c", "exit(1)"], check = True)
    child = {"child fails": fail, "no model": lambda c, env: _write(c, env, "x")}.get(case)
    monkeypatch.setattr(save_mod.subprocess, "check_call", child or save_mod.subprocess.check_call)
    with pytest.raises((ValueError, RuntimeError, ImportError), match = _BOUNDS.get(case, "Unsloth")):
        run(**_CASES.get(case, {}))
    assert len(run.seen.merges) == bool(child) and not run.left()


def test_push_checks_the_repo_first_and_forwards_hub_arguments(run, monkeypatch):
    calls = []
    create = lambda **kw: calls.append((kw, len(run.seen.merges)))
    upload = lambda **kw: calls.append((kw, os.listdir(kw["folder_path"])))
    api = SimpleNamespace(create_repo = create, upload_folder = upload)
    monkeypatch.setattr(save_mod, "HfApi", lambda token: api)
    hub = dict(private = True, commit_message = "mine", create_pr = True, revision = "dev")
    assert save_mod.unsloth_push_to_hub_openvino(_llama(), "me/m", token = False, **hub) == "me/m"
    (created, merges_before), (uploaded, files) = calls
    assert created["private"] and merges_before == 0 and "openvino_model.xml" in files
    assert all(uploaded[k] == hub[k] for k in hub if k != "private")
    assert not os.path.exists(uploaded["folder_path"])


def test_real_export_with_the_forward_patched_in_process(monkeypatch, tmp_path):
    pytest.importorskip("optimum.intel")
    import transformers

    tiny = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(tiny)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tiny)
    except OSError:
        pytest.skip(f"{tiny} is not reachable")
    # Unsloth's forward reads max_seq_length, which only models it loaded carry.
    forward = transformers.LlamaModel.forward
    patched = lambda self, *a, **k: (self.max_seq_length, forward(self, *a, **k))[1]
    monkeypatch.setattr(transformers.LlamaModel, "forward", functools.wraps(forward)(patched))
    save_mod._unsloth_save_openvino(model, str(tmp_path / "o"), tokenizer = tokenizer, token = False)
    assert {"openvino_tokenizer.xml", "openvino_detokenizer.xml"} <= {*os.listdir(tmp_path / "o")}

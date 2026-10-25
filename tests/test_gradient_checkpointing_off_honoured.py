# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""use_gradient_checkpointing = False must survive TRL's SFTConfig(gradient_checkpointing = True) default.

The end to end test trains a tiny Qwen3 LoRA on CPU through the real Unsloth-patched SFTTrainer and counts
torch checkpoint calls and decoder-layer forwards during train(), so it proves what ran, not only a flag.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO / "unsloth" / "models" / "_gc_config_default.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_gc_config_default_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NOTED.clear()
    return module


def _config_class():
    class Config:
        def __init__(
            self,
            output_dir = None,
            gradient_checkpointing = True,
            **kwargs,
        ):
            self.output_dir = output_dir
            self.gradient_checkpointing = gradient_checkpointing

    return Config


class _Model:
    def __init__(self, recorded):
        self._unsloth_gradient_checkpointing = recorded


def test_explicit_flag_kwarg_positional_and_default():
    module = _load_module()
    Config = _config_class()
    assert module.track_explicit_gradient_checkpointing(Config)
    assert not module.track_explicit_gradient_checkpointing(Config)  # idempotent
    assert Config().__dict__[module.EXPLICIT_FLAG] is False
    assert Config(output_dir = "x").__dict__[module.EXPLICIT_FLAG] is False
    assert Config(gradient_checkpointing = True).__dict__[module.EXPLICIT_FLAG] is True
    assert Config("x", True).__dict__[module.EXPLICIT_FLAG] is True
    assert Config("x").__dict__[module.EXPLICIT_FLAG] is False


def test_honour_rules(capsys):
    module = _load_module()
    Config = _config_class()
    module.track_explicit_gradient_checkpointing(Config)

    default = Config()
    assert module.honour_gradient_checkpointing_off(default, _Model(False))
    assert default.gradient_checkpointing is False
    assert "stays off" in capsys.readouterr().out

    explicit = Config(gradient_checkpointing = True)
    assert not module.honour_gradient_checkpointing_off(explicit, _Model(False))
    assert explicit.gradient_checkpointing is True

    for recorded in ("unsloth", True, None):
        args = Config()
        assert not module.honour_gradient_checkpointing_off(args, _Model(recorded))
        assert args.gradient_checkpointing is True
    args = Config()
    assert not module.honour_gradient_checkpointing_off(args, object())
    assert args.gradient_checkpointing is True

    # A config class Unsloth did not build: origin unknown, so it is kept.
    class Foreign:
        gradient_checkpointing = True

    foreign = Foreign()
    assert not module.honour_gradient_checkpointing_off(foreign, _Model(False))
    assert foreign.gradient_checkpointing is True


def test_assignment_after_construction_is_explicit():
    import dataclasses

    module = _load_module()

    @dataclasses.dataclass
    class Config:
        output_dir: str = None
        gradient_checkpointing: bool = True

        def __post_init__(self):
            self.gradient_checkpointing = (
                self.gradient_checkpointing
            )  # assignments inside __init__ do not count

    assert module.track_explicit_gradient_checkpointing(Config)
    assert Config().__dict__[module.EXPLICIT_FLAG] is False

    assigned = Config()
    assigned.gradient_checkpointing = True
    assert assigned.__dict__[module.EXPLICIT_FLAG] is True
    assert not module.honour_gradient_checkpointing_off(assigned, _Model(False))
    assert assigned.gradient_checkpointing is True

    other = Config()
    other.output_dir = "y"
    assert module.honour_gradient_checkpointing_off(other, _Model(False))
    assert other.gradient_checkpointing is False


_E2E = textwrap.dedent(
    r"""
    import json, os, sys, types
    import unsloth  # noqa: F401
    # UNSLOTH_ALLOW_CPU=1 skips the TRL trainer rewrite; run it so SFTTrainer is Unsloth's, as on a GPU host.
    os.environ["UNSLOTH_ALLOW_CPU"] = "0"
    from unsloth.models.rl import PatchFastRL
    PatchFastRL()
    os.environ["UNSLOTH_ALLOW_CPU"] = "1"
    import torch, transformers, trl
    import torch.utils.checkpoint as tuc
    import transformers.modeling_utils as hf_modeling_utils
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    calls = {"ckpt": 0, "fwd": 0}
    original_checkpoint = tuc.checkpoint

    def counting_checkpoint(*args, **kwargs):
        calls["ckpt"] += 1
        return original_checkpoint(*args, **kwargs)

    tuc.checkpoint = counting_checkpoint
    if getattr(hf_modeling_utils, "checkpoint", None) is original_checkpoint:
        hf_modeling_utils.checkpoint = counting_checkpoint

    def run(recorded, config_kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.float32)
        model = get_peft_model(model, LoraConfig(r = 8, target_modules = ["q_proj", "v_proj"], task_type = "CAUSAL_LM"))
        # What FastLanguageModel.get_peft_model(use_gradient_checkpointing = ...) records.
        model._unsloth_gradient_checkpointing = recorded
        model.enable_input_require_grads()
        layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")]
        for layer in layers:
            layer.register_forward_pre_hook(lambda m, i: calls.__setitem__("fwd", calls["fwd"] + 1))
        dataset = Dataset.from_dict({"text": [f"hello world this is sentence number {i}" for i in range(8)]})
        assign_gc = config_kwargs.pop("_assign_gc", None)
        kwargs = dict(output_dir = os.environ["E2E_OUT"], per_device_train_batch_size = 2, max_steps = 2,
                      report_to = "none", use_cpu = True, save_strategy = "no", bf16 = False, fp16 = False,
                      dataset_text_field = "text", **config_kwargs)
        try:
            args = SFTConfig(max_length = 64, **kwargs)
        except TypeError:
            args = SFTConfig(max_seq_length = 64, **kwargs)
        if assign_gc is not None:
            args.gradient_checkpointing = assign_gc
        key = "processing_class" if "processing_class" in SFTTrainer.__init__.__code__.co_varnames else "tokenizer"
        trainer = SFTTrainer(model = model, args = args, train_dataset = dataset, **{key: tokenizer})
        assert type(trainer).__name__.startswith("Unsloth"), type(trainer)
        # TRL >= 1.x computes the SFT loss with a Triton fused LM head (no CPU driver); GC is decided elsewhere.
        trainer.compute_loss = types.MethodType(transformers.Trainer.compute_loss, trainer)
        calls.update(ckpt = 0, fwd = 0)
        trainer.train()
        return {"args_gc": bool(trainer.args.gradient_checkpointing), "ckpt": calls["ckpt"],
                "fwd": calls["fwd"], "layers": len(layers)}

    results = {
        "off_default": run(False, {}),
        "off_explicit_on": run(False, {"gradient_checkpointing": True}),
        "off_assigned_on": run(False, {"_assign_gc": True}),
        "unsloth_default": run("unsloth", {}),
        "on_default": run(True, {}),
    }
    print("E2E_RESULT " + json.dumps(results))
    """
)


@pytest.fixture(scope = "module")
def e2e_results(tmp_path_factory):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    tmp = tmp_path_factory.mktemp("gc_off_e2e")
    env = dict(os.environ)
    env.update(
        UNSLOTH_ALLOW_CPU = "1",
        CUDA_VISIBLE_DEVICES = "",
        UNSLOTH_COMPILE_LOCATION = str(tmp / "compiled"),
        E2E_OUT = str(tmp / "out"),
        PYTHONPATH = os.pathsep.join(p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p),
    )
    script = tmp / "e2e.py"
    script.write_text(_E2E, encoding = "utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)],
        env = env,
        cwd = str(tmp),
        capture_output = True,
        text = True,
        timeout = 900,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("E2E_RESULT ")]
    assert proc.returncode == 0 and lines, (proc.stdout[-3000:], proc.stderr[-6000:])
    return json.loads(lines[-1].split(" ", 1)[1]), proc.stdout


def test_gc_off_model_trains_without_checkpointing(e2e_results):
    results, stdout = e2e_results
    off = results["off_default"]
    assert off["args_gc"] is False
    assert off["ckpt"] == 0, off
    assert "so gradient checkpointing stays off" in stdout


def test_explicit_config_true_still_checkpoints(e2e_results):
    results, _ = e2e_results
    on = results["off_explicit_on"]
    off = results["off_default"]
    assert on["args_gc"] is True
    assert on["ckpt"] > 0
    # Every checkpointed layer forward is recomputed in the backward.
    assert on["fwd"] == 2 * off["fwd"], (on, off)


def test_config_true_assigned_after_construction_still_checkpoints(e2e_results):
    results, _ = e2e_results
    assigned = results["off_assigned_on"]
    assert assigned["args_gc"] is True
    assert assigned["ckpt"] > 0


@pytest.mark.parametrize("case", ["unsloth_default", "on_default"])
def test_gc_on_models_unchanged(e2e_results, case):
    results, _ = e2e_results
    assert results[case]["args_gc"] is True
    assert results[case]["ckpt"] > 0

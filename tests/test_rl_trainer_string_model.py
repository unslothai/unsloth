# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

_E2E = textwrap.dedent(
    r"""
    import json, os, types
    import unsloth  # noqa: F401
    os.environ["UNSLOTH_ALLOW_CPU"] = "0"
    from unsloth.models.rl import PatchFastRL
    PatchFastRL()
    os.environ["UNSLOTH_ALLOW_CPU"] = "1"
    import torch, transformers, trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hub_id = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    local_dir = os.path.join(os.environ["E2E_OUT"], "local-checkpoint")
    AutoModelForCausalLM.from_pretrained(hub_id).save_pretrained(local_dir)
    AutoTokenizer.from_pretrained(hub_id).save_pretrained(local_dir)

    prefs = {"prompt": ["Hi", "Yo"] * 4, "chosen": [" there", " man"] * 4, "rejected": [" no", " bye"] * 4}
    def reward(completions, **kwargs):
        return [float(len(c)) for c in completions]

    def trainers():
        common = dict(per_device_train_batch_size = 2, max_steps = 1, report_to = "none", use_cpu = True,
                      save_strategy = "no", bf16 = False, fp16 = False)
        yield "SFT", trl.SFTTrainer, trl.SFTConfig(max_length = 32, **common), {
            "train_dataset": Dataset.from_dict({"text": [f"hello world {i}" for i in range(8)]})}
        yield "GRPO", trl.GRPOTrainer, trl.GRPOConfig(num_generations = 2, max_completion_length = 4, **common), {
            "train_dataset": Dataset.from_dict({"prompt": ["Count:"] * 8}), "reward_funcs": reward}
        yield "DPO", trl.DPOTrainer, trl.DPOConfig(**common), {"train_dataset": Dataset.from_dict(prefs)}
        yield "KTO", trl.KTOTrainer, trl.KTOConfig(**common), {"train_dataset": Dataset.from_dict(
            {"prompt": prefs["prompt"], "completion": prefs["chosen"], "label": [True, False] * 4})}
        yield "Reward", trl.RewardTrainer, trl.RewardConfig(**common), {
            "train_dataset": Dataset.from_dict({"chosen": prefs["chosen"], "rejected": prefs["rejected"]})}

    results = {}
    for name, trainer_cls, args, kwargs in trainers():
        args.output_dir = os.path.join(os.environ["E2E_OUT"], name)
        for form, model_id in (("hub", hub_id), ("local", local_dir)):
            key = f"{name}/{form}"
            try:
                trainer = trainer_cls(model = model_id, args = args, **kwargs)
                result = {"unsloth": type(trainer).__name__.startswith("Unsloth"),
                          "loaded": isinstance(trainer.model, torch.nn.Module)}
                if name == "SFT":
                    trainer.compute_loss = types.MethodType(transformers.Trainer.compute_loss, trainer)
                    result["steps"] = trainer.train().global_step
                results[key] = result
            except Exception as e:
                results[key] = {"error": repr(e)[:300]}
    from peft import LoraConfig
    for name, trainer_cls, args, kwargs in trainers():
        try:
            trainer_cls(model = hub_id, args = args, peft_config = LoraConfig(r = 8, target_modules = ["q_proj"]), **kwargs)
            results[f"{name}/peft"] = {"error": None}
        except Exception as e:
            results[f"{name}/peft"] = {"error": repr(e)[:300]}
    print("E2E_RESULT " + json.dumps(results))
    """
)

_TRAINERS = ("SFT", "GRPO", "DPO", "KTO", "Reward")
_CASES = [f"{name}/{form}" for name in _TRAINERS for form in ("hub", "local")]


@pytest.fixture(scope = "module")
def e2e_results(tmp_path_factory):
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    pytest.importorskip("peft")
    tmp = tmp_path_factory.mktemp("rl_string_model")
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
        timeout = 1200,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("E2E_RESULT ")]
    assert proc.returncode == 0 and lines, (proc.stdout[-3000:], proc.stderr[-6000:])
    return json.loads(lines[-1].split(" ", 1)[1])


@pytest.mark.parametrize("case", _CASES)
def test_trainer_loads_a_model_given_by_name_or_path(e2e_results, case):
    result = e2e_results[case]
    assert "error" not in result, result
    assert result["unsloth"] and result["loaded"], result


@pytest.mark.parametrize("form", ["hub", "local"])
def test_sft_trains_a_model_given_by_name_or_path(e2e_results, form):
    assert e2e_results[f"SFT/{form}"].get("steps") == 1, e2e_results[f"SFT/{form}"]


@pytest.mark.parametrize("name", _TRAINERS)
def test_peft_config_with_a_model_name_is_refused_clearly(e2e_results, name):
    error = e2e_results[f"{name}/peft"]["error"] or ""
    assert error.startswith("ValueError(") and "FastLanguageModel.get_peft_model" in error, error

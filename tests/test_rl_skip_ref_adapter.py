# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 0.27+ RLOO (always) and GRPO (beta != 0) copy a PEFT model's adapter into a frozen "ref" adapter. Unsloth uses
the base model as the reference instead, and with fast_inference the vLLM LoRA sync would also send the ref
weights, which vLLM rejects. The generated trainers turn that branch off; DPO keeps its ref adapter."""

from __future__ import annotations

import importlib
import inspect

import pytest
import unsloth  # noqa: F401

from unsloth.models import rl

_BODY = """
        default_config = model.peft_config["default"]
        if default_config.target_parameters:
            log.append("warn")
        else:
            model.add_adapter("ref", default_config)
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_param.data.copy_(param.data)
"""
_FORMS = {
    # TRL 0.27 - 1.3
    "grpo_old": "    if is_peft_available() and is_peft_model(model) and args.beta != 0.0:" + _BODY,
    "rloo_old": "    if is_peft_available() and is_peft_model(model):" + _BODY,
    # TRL 1.4+
    "grpo_new": "    if peft_config is not None:\n        log.append('wrap')\n"
    "    elif is_peft_model(model) and args.beta != 0.0:" + _BODY,
    "rloo_new": "    if peft_config is not None:\n        log.append('wrap')\n"
    "    elif is_peft_model(model):" + _BODY + "    else:\n        log.append('else')\n",
}


class _Model:
    def __init__(self):
        self.peft_config = {"default": type("C", (), {"target_parameters": None})()}
        self.added = []

    def add_adapter(self, name, config):
        self.added.append(name)

    def named_parameters(self):
        return []


def _run(source, peft_config = None):
    code = (
        "def init(model, args, peft_config, log, ref_param = None):\n" + source + "    return log\n"
    )
    namespace = {"is_peft_available": lambda: True, "is_peft_model": lambda m: True}
    exec(code, namespace)
    model, log = _Model(), []
    namespace["init"](model, type("A", (), {"beta": 0.04})(), peft_config, log)
    return model.added, log


@pytest.mark.parametrize("form", sorted(_FORMS))
def test_ref_adapter_branch_is_off(form):
    source = _FORMS[form]
    assert _run(source)[0] == ["ref"]  # TRL's own code adds it
    new = rl._skip_rl_ref_adapter(source)
    assert "Unsloth: no frozen ref adapter" in new
    added, log = _run(new)
    assert added == [] and "warn" not in log
    # The rest of the chain keeps its meaning: the PEFT model still takes this branch, not a following else.
    assert "else" not in log
    if "peft_config is not None" in source:
        assert _run(new, peft_config = object())[1] == ["wrap"]


def test_dpo_and_commented_blocks_are_left_alone():
    dpo = "    elif is_peft_model(model) and ref_model is None:" + _BODY
    assert rl._skip_rl_ref_adapter(dpo) == dpo
    commented = "\n".join(
        "        # " + line.strip() for line in _FORMS["grpo_old"].splitlines() if line.strip()
    )
    assert rl._skip_rl_ref_adapter(commented) == commented


@pytest.mark.parametrize("trainer", ["rloo", "grpo"])
def test_generated_trainer_skips_the_ref_adapter(trainer):
    module = importlib.import_module(f"trl.trainer.{trainer}_trainer")
    if 'add_adapter("ref"' not in inspect.getsource(module):
        pytest.skip("this TRL does not add a ref adapter")
    cls = getattr(module, {"rloo": "RLOOTrainer", "grpo": "GRPOTrainer"}[trainer])
    assert cls.__name__.startswith("Unsloth"), f"{cls} was not patched"
    source = inspect.getsource(cls.__mro__[1].__init__)
    live = [
        line
        for line in source.splitlines()
        if 'add_adapter("ref"' in line and not line.lstrip().startswith("#")
    ]
    if live:
        assert "Unsloth: no frozen ref adapter" in source

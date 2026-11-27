# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_VLLM_STANDBY: RLOO and OnlineDPO must put a standby (enable_sleep_mode) engine to sleep for the
training phase, as GRPO does. TRL only sleeps when args.vllm_enable_sleep_mode is set, which Unsloth set for
GRPO alone, so the engine kept its 0.9 of VRAM through every optimizer step. OnlineDPO on TRL < 0.29 has no
sleep code at all, so Unsloth wraps its colocate generate with wake_up / sleep."""

import importlib.util
import os
import sys
import textwrap
import types

import pytest

# TRL rloo_trainer.RLOOTrainer.__init__ / experimental OnlineDPOTrainer.__init__, colocate vLLM part only.
_INIT_SOURCE = """
class {name}:
    def __init__(self, model, args):
        self.args = args
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        if self.use_vllm:
            if self.vllm_mode == "colocate":
                self.llm = LLM(model = model.name_or_path, enable_sleep_mode = self.args.vllm_enable_sleep_mode)
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level = {level})
            else:
                raise ValueError("bad vllm_mode")
            self._last_loaded_step = -1
        else:
            self.llm = None
"""


class _Engine:
    """Unsloth's shared fast_inference engine, built with or without sleep mode."""

    shared_weights = True

    def __init__(self, sleep_mode):
        self.calls = []
        config = types.SimpleNamespace(enable_sleep_mode = sleep_mode)
        self.llm_engine = types.SimpleNamespace(
            vllm_config = types.SimpleNamespace(model_config = config)
        )

    def sleep(self, level = 1):
        self.calls.append(("sleep", level))

    def wake_up(self, tags = None):
        self.calls.append("wake_up")

    def generate(self, *args, **kwargs):
        self.calls.append("generate")
        return ["out"]


def _patched_class(tmp_path, trainer_file, name, level):
    from unsloth.models.rl import patch_functions

    path = tmp_path / f"{trainer_file}_source.py"
    path.write_text(textwrap.dedent(_INIT_SOURCE.format(name = name, level = level)))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    source = patch_functions(getattr(module, name), trainer_file, name, [], [])
    namespace = {"os": os, "LLM": lambda **kwargs: kwargs}
    exec(source, namespace)
    return namespace[f"_Unsloth{name}"]


def _args():
    return types.SimpleNamespace(use_vllm = True, vllm_mode = "colocate", vllm_enable_sleep_mode = False)


_TRAINERS = [("rloo_trainer", "RLOOTrainer", 1), ("online_dpo_trainer", "OnlineDPOTrainer", 2)]


@pytest.mark.parametrize("trainer_file, name, level", _TRAINERS, ids = ["rloo", "online_dpo"])
def test_standby_engine_sleeps_at_init(tmp_path, monkeypatch, trainer_file, name, level):
    monkeypatch.delenv("UNSLOTH_VLLM_STANDBY", raising = False)
    engine = _Engine(sleep_mode = True)
    model = types.SimpleNamespace(name_or_path = "tiny", vllm_engine = engine)
    args = _args()
    trainer = _patched_class(tmp_path, trainer_file, name, level)(model, args)
    assert trainer.llm is engine
    assert args.vllm_enable_sleep_mode is True
    assert engine.calls == [("sleep", level)]


@pytest.mark.parametrize("trainer_file, name, level", _TRAINERS, ids = ["rloo", "online_dpo"])
def test_engine_without_sleep_mode_is_left_awake(tmp_path, monkeypatch, trainer_file, name, level):
    # The vision standby gate can build the engine without sleep mode even when the env var is set.
    monkeypatch.setenv("UNSLOTH_VLLM_STANDBY", "1")
    engine = _Engine(sleep_mode = False)
    model = types.SimpleNamespace(name_or_path = "tiny", vllm_engine = engine)
    args = _args()
    _patched_class(tmp_path, trainer_file, name, level)(model, args)
    assert args.vllm_enable_sleep_mode is False
    assert engine.calls == []


# trl 0.24 online_dpo_trainer.OnlineDPOTrainer._generate_vllm_colocate: no sleep / wake of its own.
_LEGACY_COLOCATE = textwrap.dedent(
    """
    def _generate_vllm_colocate(self, prompts, images = None):
        vllm_inputs = prompts
        outputs = self.llm.generate(vllm_inputs, self.generation_config, use_tqdm=False)
        return outputs
    """
)


@pytest.mark.parametrize(
    "sleep_mode, expected",
    [
        (True, ["wake_up", "generate", ("sleep", 1)]),
        (False, ["generate"]),
    ],
    ids = ["standby", "no_sleep_mode"],
)
def test_legacy_online_dpo_generate_wakes_and_sleeps(monkeypatch, sleep_mode, expected):
    from unsloth.models.rl_replacements import online_dpo_trainer__generate_vllm_colocate

    monkeypatch.delenv("VLLM_SLEEP_MODE", raising = False)
    source = online_dpo_trainer__generate_vllm_colocate("_generate_vllm_colocate", _LEGACY_COLOCATE)
    scope = {"os": os}
    exec(source, scope)
    engine = _Engine(sleep_mode)
    trainer = types.SimpleNamespace(llm = engine, generation_config = None)
    assert scope["_generate_vllm_colocate"](trainer, ["p"]) == ["out"]
    assert engine.calls == expected


# trl >= 0.29 experimental OnlineDPOTrainer._generate_vllm_colocate: sleeps / wakes by tag itself.
_TAGGED_COLOCATE = textwrap.dedent(
    """
    def _generate_vllm_colocate(self, prompts, images = None):
        if self.args.vllm_enable_sleep_mode:
            self.llm.wake_up(tags=["weights"])
        if self.args.vllm_enable_sleep_mode:
            self.llm.wake_up(tags=["kv_cache"])
        outputs = self.llm.generate(prompts, self.generation_config, use_tqdm=False)
        if self.args.vllm_enable_sleep_mode:
            self.llm.sleep(level=2)
        return outputs
    """
)


@pytest.mark.parametrize(
    "shared, expected",
    [
        # Unsloth's allocator keeps the shared weights, so a weights-only wake leaves vLLM's runtime / cudagraph pools
        # unmapped (vLLM >= 0.34): wake everything.
        (True, ["wake_up", "wake_up", "generate", ("sleep", 2)]),
        (False, [("wake_up", ["weights"]), ("wake_up", ["kv_cache"]), "generate", ("sleep", 2)]),
    ],
    ids = ["shared_engine", "trl_engine"],
)
def test_tagged_online_dpo_generate_wakes_shared_engine_untagged(shared, expected):
    from unsloth.models.rl_replacements import online_dpo_trainer__generate_vllm_colocate

    patched = online_dpo_trainer__generate_vllm_colocate(
        "_generate_vllm_colocate", _TAGGED_COLOCATE
    )
    assert "_unsloth_sleep_level" not in patched
    assert "self._last_loaded_step = -1" in patched
    scope = {}
    exec(patched, scope)

    class Engine(_Engine):
        shared_weights = shared

        def wake_up(self, tags = None):
            self.calls.append("wake_up" if tags is None else ("wake_up", tags))

    engine = Engine(True)
    trainer = types.SimpleNamespace(
        llm = engine, generation_config = None, args = types.SimpleNamespace(vllm_enable_sleep_mode = True)
    )
    assert scope["_generate_vllm_colocate"](trainer, ["p"]) == ["out"]
    assert engine.calls == expected
    assert trainer._last_loaded_step == -1

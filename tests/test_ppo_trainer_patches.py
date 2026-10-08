# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL PPOTrainer under Unsloth: rollout generation and the rollout sampling distribution.

TRL's PPOTrainer generates inside ``unwrap_model_for_generation`` on a ``PolicyAndValueWrapper``,
which has no ``generate()`` and, before TRL 0.26, no ``gradient_checkpointing_disable()``. Unsloth's
wrapper of that context manager used to crash on both. Separately, transformers replaces every
field of TRL's rollout GenerationConfig left at its global default (``top_p = 1.0``) with the
model's own default (Qwen3 Instruct ships ``top_p = 0.8``), so PPO's KL and ratio would read
logprobs from a truncated distribution. Finally Unsloth's training forward drops the attention
mask (it assumes right padding), while PPO left-pads every query, so the policy, reference and
value forwards attended to the pads; PPO training opts those models into keeping the mask.

The helpers are lifted with ``ast`` from ``unsloth/models/rl.py`` so the test stays CPU-only.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
from contextlib import contextmanager
from pathlib import Path

import pytest


RL_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
NAMES = (
    "_generation_target",
    "_generate_accepts_use_model_defaults",
    "_caller_sampling_only",
    "_hide_unsupported_gradient_checkpointing",
    "_ppo_padding_mask_modules",
    "_wrap_ppo_train",
)


def _load():
    tree = ast.parse(RL_PATH.read_text(encoding = "utf-8"), filename = str(RL_PATH))
    wanted = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES
    ]
    assert len(wanted) == len(NAMES), [n.name for n in wanted]
    namespace = {
        "copy": copy,
        "functools": functools,
        "inspect": inspect,
        "contextmanager": contextmanager,
    }
    exec(compile(ast.Module(body = wanted, type_ignores = []), str(RL_PATH), "exec"), namespace)
    return namespace


NS = _load()


class _Policy:
    def generate(self, *args, **kwargs):
        return "policy"


class _PolicyAndValueWrapper:
    """TRL 0.22 PolicyAndValueWrapper surface: no generate, no gradient_checkpointing_disable."""

    def __init__(self):
        self.policy = _Policy()
        self.is_gradient_checkpointing = True


def test_ppo_wrapper_generates_through_policy():
    wrapper = _PolicyAndValueWrapper()
    assert NS["_generation_target"](wrapper) is wrapper.policy


def test_plain_model_generates_itself():
    model = _Policy()
    assert NS["_generation_target"](model) is model


def test_gradient_checkpointing_hidden_from_trl_then_restored():
    wrapper = _PolicyAndValueWrapper()
    with NS["_hide_unsupported_gradient_checkpointing"](wrapper):
        # TRL's unwrap reads this and would call the missing gradient_checkpointing_disable().
        assert wrapper.is_gradient_checkpointing is False
    assert wrapper.is_gradient_checkpointing is True


def test_gradient_checkpointing_restored_on_error():
    wrapper = _PolicyAndValueWrapper()
    with pytest.raises(RuntimeError):
        with NS["_hide_unsupported_gradient_checkpointing"](wrapper):
            raise RuntimeError("generation failed")
    assert wrapper.is_gradient_checkpointing is True


def test_models_with_gradient_checkpointing_api_untouched():
    class _Model:
        is_gradient_checkpointing = True

        def gradient_checkpointing_disable(self):
            pass

    model = _Model()
    with NS["_hide_unsupported_gradient_checkpointing"](model):
        assert model.is_gradient_checkpointing is True


class _Generator:
    def __init__(self, generation_config):
        self.generation_config = generation_config


def test_ppo_rollouts_keep_caller_sampling():
    transformers = pytest.importorskip("transformers")
    model_config = transformers.GenerationConfig(
        top_p = 0.8, top_k = 20, temperature = 0.7, min_p = 0.05, eos_token_id = 5, pad_token_id = 7
    )
    trl_config = transformers.GenerationConfig(
        max_new_tokens = 8, temperature = 0.7, top_k = 0, top_p = 1.0, do_sample = True
    )
    kwargs = {"input_ids": "ids", "generation_config": trl_config}
    out = NS["_caller_sampling_only"](_Generator(model_config), kwargs)
    if not NS["_generate_accepts_use_model_defaults"]():
        # transformers 5 only fills fields left unset, and TRL sets its sampling fields explicitly.
        assert out is kwargs
        return
    assert out["use_model_defaults"] is False and out["input_ids"] == "ids"
    passed = out["generation_config"]
    assert (passed.top_p, passed.top_k, passed.min_p) == (1.0, 0, None)
    # Stopping still follows the model (TRL writes its stop token there).
    assert (passed.eos_token_id, passed.pad_token_id) == (5, 7)
    assert trl_config.eos_token_id is None and model_config.top_p == 0.8


def test_ppo_rollouts_respect_explicit_use_model_defaults():
    transformers = pytest.importorskip("transformers")
    kwargs = {"generation_config": transformers.GenerationConfig(), "use_model_defaults": True}
    assert (
        NS["_caller_sampling_only"](_Generator(transformers.GenerationConfig()), kwargs) is kwargs
    )
    assert NS["_caller_sampling_only"](_Generator(None), {"max_new_tokens": 4}) == {
        "max_new_tokens": 4
    }


def test_ppo_train_wrapped_once_and_mask_cleared_on_error():
    base = _Module(base = True)

    class _Trainer:
        policy_model = _Module(base)

        def train(self):
            assert base._unsloth_keep_padding_mask is True
            raise RuntimeError("oom")

    NS["_wrap_ppo_train"](_Trainer)
    first = _Trainer.train
    NS["_wrap_ppo_train"](_Trainer)
    assert _Trainer.train is first
    with pytest.raises(RuntimeError):
        _Trainer().train()
    assert base._unsloth_keep_padding_mask is False


def test_rollout_logits_freed_after_scoring():
    ppo = pytest.importorskip("trl.trainer.ppo_trainer")
    if not hasattr(ppo, "PPOTrainer"):
        pytest.skip("TRL without trl.trainer.ppo_trainer.PPOTrainer")
    import inspect

    path = RL_PATH.with_name("rl_replacements.py")
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "ppo_trainer_free_rollout_logits"
    )
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(path), "exec"), namespace)
    edit = namespace["ppo_trainer_free_rollout_logits"]

    source = inspect.getsource(ppo.PPOTrainer.train)
    patched = edit("train", source)
    assert "unwrapped_model, logitss)" in patched
    # Nothing past the rollout reads them.
    tail = patched.split("unwrapped_model, logitss)", 1)[1]
    assert "logitss" not in tail
    assert edit("generate_completions", source) == source


class _Module:
    def __init__(
        self,
        *children,
        base = False,
    ):
        self.children = children
        if base:
            self.embed_tokens = object()

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()


def test_ppo_training_keeps_padding_masks_then_restores():
    transformers = pytest.importorskip("transformers")
    policy_base, value_base, other_base = _Module(base = True), _Module(base = True), _Module(base = True)
    seen = {}

    class _Trainer:
        def __init__(self):
            self.policy_model = _Module(_Module(policy_base))
            self.policy_model.generation_config = transformers.GenerationConfig()
            self.value_model = _Module(value_base)
            self.ref_model = None
            self.reward_model = _Module(other_base)

        def train(self):
            seen["policy"] = policy_base._unsloth_keep_padding_mask
            seen["value"] = value_base._unsloth_keep_padding_mask
            seen["reward"] = getattr(other_base, "_unsloth_keep_padding_mask", None)

    NS["_wrap_ppo_train"](_Trainer)
    _Trainer().train()
    # The reward model runs in eval mode, which already keeps the mask.
    assert seen == {"policy": True, "value": True, "reward": None}
    assert policy_base._unsloth_keep_padding_mask is False
    assert value_base._unsloth_keep_padding_mask is False


def test_llama_training_forward_mask_is_opt_in():
    # Every other trainer keeps today's behaviour: training mode drops the mask unless the flag is set.
    source = (RL_PATH.parent / "llama.py").read_text(encoding = "utf-8")
    assert (
        'elif self.training and not getattr(self, "_unsloth_keep_padding_mask", False):\n'
        "        attention_mask = None\n"
    ) in source

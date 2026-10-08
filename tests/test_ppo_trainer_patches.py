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
import functools
from contextlib import contextmanager
from pathlib import Path

import pytest


RL_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
NAMES = (
    "_generation_target",
    "_hide_unsupported_gradient_checkpointing",
    "_ppo_padding_mask_modules",
    "_wrap_ppo_train",
)


def _load():
    tree = ast.parse(RL_PATH.read_text(encoding = "utf-8"), filename = str(RL_PATH))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in NAMES)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_PPO_ROLLOUT_FILTER_KEYS" for t in node.targets
            )
        )
    ]
    assert len(wanted) == len(NAMES) + 1, [getattr(n, "name", "assign") for n in wanted]
    namespace = {"functools": functools, "contextmanager": contextmanager}
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


def _trainer_class(seen):
    class _Trainer:
        def __init__(self, generation_config):
            self.policy_model = type("P", (), {"generation_config": generation_config})()

        def train(self):
            gc = self.policy_model.generation_config
            seen.update(top_p = gc.top_p, top_k = gc.top_k, temperature = gc.temperature, min_p = gc.min_p)
            return "trained"

    return _Trainer


def test_ppo_rollouts_sample_full_distribution_then_restore():
    transformers = pytest.importorskip("transformers")
    config = transformers.GenerationConfig(top_p = 0.8, top_k = 20, temperature = 0.7, min_p = 0.05)
    seen = {}
    trainer_cls = _trainer_class(seen)
    NS["_wrap_ppo_train"](trainer_cls)
    assert trainer_cls(config).train() == "trained"
    # Filters transformers would copy over TRL's defaults are reset; TRL's explicit values stay.
    assert seen["top_p"] == 1.0 and seen["min_p"] is None
    assert seen["top_k"] == 20 and seen["temperature"] == 0.7
    assert (config.top_p, config.min_p) == (0.8, 0.05)


def test_ppo_rollouts_restore_on_error_and_wrap_once():
    transformers = pytest.importorskip("transformers")
    config = transformers.GenerationConfig(top_p = 0.8)

    class _Trainer:
        policy_model = type("P", (), {"generation_config": config})()

        def train(self):
            raise RuntimeError("oom")

    NS["_wrap_ppo_train"](_Trainer)
    first = _Trainer.train
    NS["_wrap_ppo_train"](_Trainer)
    assert _Trainer.train is first
    with pytest.raises(RuntimeError):
        _Trainer().train()
    assert config.top_p == 0.8


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

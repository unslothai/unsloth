# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""PPOTrainer with an Unsloth policy (#884).

TRL < 1.0 PolicyAndValueWrapper copies is_gradient_checkpointing from the policy without the toggles
unwrap_model_for_generation calls, and has no generate() for Unsloth's unwrap to wrap. The PPO value
and reward models are plain transformers models built after Unsloth patched the rotary class, so on
transformers v5 their inv_freq buffer is uninitialized memory that extend_rope_embedding read.
The rl.py helpers are lifted out with ``ast`` so those checks run without ``import unsloth``.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

RL_PY = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _lift(
    name,
    namespace,
    parent = None,
):
    tree = ast.parse(RL_PY.read_text(encoding = "utf-8"))
    scope = tree
    if parent is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == parent)
    node = next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body = [node], type_ignores = []), str(RL_PY), "exec"), namespace)
    return namespace[name]


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_gradient_checkpointing = True
        self.calls = []

    def gradient_checkpointing_disable(self):
        self.calls.append("disable")
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.calls.append("enable")
        self.is_gradient_checkpointing = True

    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            return torch.ones(1, 3, dtype = torch.long)


def _old_trl_wrapper_class():
    # TRL 0.18 - 0.29 PolicyAndValueWrapper, minus the critic backbone.
    class PolicyAndValueWrapper(torch.nn.Module):
        def __init__(self, policy, value_model):
            super().__init__()
            self.policy = policy
            self.value_model = value_model
            self.is_gradient_checkpointing = policy.is_gradient_checkpointing

    return PolicyAndValueWrapper


@contextmanager
def _trl_unwrap(
    model,
    accelerator = None,
    **kwargs,
):
    # trl.models.utils.unwrap_model_for_generation without DeepSpeed.
    is_gradient_checkpointing = model.is_gradient_checkpointing
    if is_gradient_checkpointing:
        model.gradient_checkpointing_disable()
    yield model
    if is_gradient_checkpointing:
        model.gradient_checkpointing_enable()


def test_old_wrapper_gets_gradient_checkpointing_toggles():
    patch = _lift("_patch_ppo_policy_value_wrapper", {"sys": sys})
    module = ModuleType("fake_ppo_trainer")
    module.PolicyAndValueWrapper = _old_trl_wrapper_class()
    wrapper = module.PolicyAndValueWrapper(_Policy(), torch.nn.Linear(1, 1))
    with pytest.raises(AttributeError):
        with _trl_unwrap(wrapper):
            pass

    patch(module)
    patch(module)
    with _trl_unwrap(wrapper):
        assert wrapper.is_gradient_checkpointing is False
    assert wrapper.policy.calls == ["disable", "enable"]
    assert wrapper.is_gradient_checkpointing is True


def test_wrapper_found_through_the_trainer_module_and_upstream_toggles_kept():
    patch = _lift("_patch_ppo_policy_value_wrapper", {"sys": sys})
    real = ModuleType("fake_trl_experimental_ppo_trainer")
    real.PolicyAndValueWrapper = _old_trl_wrapper_class()
    real.PPOTrainer = type("PPOTrainer", (), {"__module__": real.__name__})
    sys.modules[real.__name__] = real
    try:
        shim = ModuleType("fake_trl_trainer_ppo_trainer")
        shim.PPOTrainer = real.PPOTrainer
        patch(shim)
        assert hasattr(real.PolicyAndValueWrapper, "gradient_checkpointing_disable")
    finally:
        del sys.modules[real.__name__]

    upstream = ModuleType("fake_trl_1x")
    upstream.PolicyAndValueWrapper = _old_trl_wrapper_class()
    own = lambda self: None
    upstream.PolicyAndValueWrapper.gradient_checkpointing_disable = own
    patch(upstream)
    assert upstream.PolicyAndValueWrapper.gradient_checkpointing_disable is own


def _unsloth_unwrap(calls):
    fast = SimpleNamespace(
        for_inference = lambda model: calls.append("for_inference"),
        for_training = lambda model, use_gradient_checkpointing: calls.append(
            ("for_training", use_gradient_checkpointing)
        ),
    )
    namespace = {
        "torch": torch,
        "contextmanager": contextmanager,
        "unwrap_model_for_generation": _trl_unwrap,
        "FastLanguageModel": fast,
        "copy": copy,
        "inspect": inspect,
    }
    _lift("_generate_accepts_use_model_defaults", namespace)
    _lift("_caller_sampling_only", namespace)
    return _lift("unsloth_unwrap_model_for_generation", namespace, parent = "PatchRL")


def test_unwrap_generates_through_the_ppo_policy():
    patch = _lift("_patch_ppo_policy_value_wrapper", {"sys": sys})
    module = ModuleType("fake_ppo_trainer")
    module.PolicyAndValueWrapper = _old_trl_wrapper_class()
    patch(module)
    policy = _Policy()
    policy.gradient_checkpointing = "unsloth"
    wrapper = module.PolicyAndValueWrapper(policy, torch.nn.Linear(1, 1))

    calls = []
    with _unsloth_unwrap(calls)(wrapper) as unwrapped:
        assert unwrapped is wrapper
        out = unwrapped.policy.generate()
    assert not out.is_inference()
    assert "generate" not in vars(wrapper)
    assert calls == ["for_inference", ("for_training", "unsloth")]


def test_unwrap_still_wraps_a_model_with_generate():
    policy = _Policy()
    calls = []
    with _unsloth_unwrap(calls)(policy) as unwrapped:
        out = unwrapped.generate()
    assert not out.is_inference()


def _has_real_gpu():
    try:
        torch.zeros(1).to("cuda")
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_real_gpu(), reason = "LlamaRotaryEmbedding builds per-device caches in __init__"
)
@pytest.mark.parametrize("garbage", [float("nan"), 1e30])
def test_rope_extension_ignores_a_refilled_inv_freq(garbage):
    from transformers import LlamaConfig
    from unsloth.models.llama import LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size = 256,
        num_attention_heads = 4,
        max_position_embeddings = 65536,
        rope_theta = 500000.0,
    )
    reference = LlamaRotaryEmbedding(config = config)
    rope = LlamaRotaryEmbedding(config = config)
    # What transformers v5 meta loading leaves in a non-persistent buffer it does not know how to init.
    rope.inv_freq.fill_(garbage)
    x = torch.zeros(1, device = "cuda", dtype = torch.float32)
    reference.extend_rope_embedding(x, config.max_position_embeddings)
    rope.extend_rope_embedding(x, config.max_position_embeddings)
    cos, sin = rope.get_cached(device_index = x.device.index)
    ref_cos, ref_sin = reference.get_cached(device_index = x.device.index)
    assert cos.shape[0] >= config.max_position_embeddings
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
    torch.testing.assert_close(cos, ref_cos, rtol = 0, atol = 0)
    torch.testing.assert_close(sin, ref_sin, rtol = 0, atol = 0)
    torch.testing.assert_close(rope.inv_freq, reference.inv_freq, rtol = 0, atol = 0)


def _rollout_helpers():
    namespace = {"copy": copy, "functools": functools, "inspect": inspect}
    for name in (
        "_generate_accepts_use_model_defaults",
        "_caller_sampling_only",
        "_ppo_padding_mask_modules",
        "_wrap_ppo_train",
    ):
        _lift(name, namespace)
    return namespace


NS = _rollout_helpers()


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
    path = RL_PY.with_name("rl_replacements.py")
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
    # Every other trainer keeps today's behaviour: training mode drops the mask unless the flag is
    # set or a label-less batch is left padded (test_llama_left_padding_training_mask.py).
    source = (RL_PY.parent / "llama.py").read_text(encoding = "utf-8")
    assert (
        "    elif (\n"
        "        self.training\n"
        '        and not getattr(self, "_unsloth_keep_padding_mask", False)\n'
        '        and not (getattr(self, "_has_no_labels", False) and _is_left_padded(attention_mask))\n'
        "    ):\n"
        "        attention_mask = None\n"
    ) in source

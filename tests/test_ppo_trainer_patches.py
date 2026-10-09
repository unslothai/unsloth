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
    }
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

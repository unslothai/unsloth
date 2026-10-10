# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unsloth trainers default DDP find_unused_parameters to False for a plain LoRA text model.

transformers picks True for every non-PreTrainedModel (all PEFT models). The helper is loaded by
path so the test stays CPU-only and does not import ``unsloth``.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "unsloth" / "models" / "_ddp_defaults.py"


def _helper():
    assert HELPER.exists(), "unsloth/models/_ddp_defaults.py is missing"
    spec = importlib.util.spec_from_file_location("_unsloth_ddp_defaults", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.unsloth_default_ddp_find_unused_parameters


def _llama(**extra):
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        **extra,
    )
    return transformers.LlamaForCausalLM(config)


def _lora(model, **kwargs):
    kwargs.setdefault("target_modules", ["q_proj", "v_proj", "gate_proj"])
    return peft.get_peft_model(model, peft.LoraConfig(r = 4, lora_alpha = 8, **kwargs))


def _args(**kwargs):
    return transformers.TrainingArguments(output_dir = "unused", report_to = "none", **kwargs)


@pytest.fixture
def ddp_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("UNSLOTH_DDP_FIND_UNUSED_PARAMETERS_AUTO", raising = False)


def test_plain_lora_defaults_to_false(ddp_env):
    args = _args()
    assert args.ddp_find_unused_parameters is None
    assert _helper()(args, _lora(_llama())) is False
    assert args.ddp_find_unused_parameters is False


def test_user_value_is_kept(ddp_env):
    args = _args(ddp_find_unused_parameters = True)
    assert _helper()(args, _lora(_llama())) is True


def test_not_distributed_is_untouched(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args()
    assert _helper()(args, _lora(_llama())) is None


def test_env_opt_out(ddp_env, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DDP_FIND_UNUSED_PARAMETERS_AUTO", "0")
    args = _args()
    assert _helper()(args, _lora(_llama())) is None


@pytest.mark.parametrize(
    "make",
    [
        # MoE: unrouted experts get no gradient.
        lambda: _lora(_llama(num_local_experts = 4)),
        # modules_to_save: a full trainable copy Unsloth's fused paths may bypass.
        lambda: _lora(_llama(), modules_to_save = ["lm_head"]),
        # LoRA on lm_head / embeddings.
        lambda: _lora(_llama(), target_modules = ["q_proj", "lm_head"]),
        lambda: _lora(_llama(), target_modules = ["q_proj", "embed_tokens"]),
        # Full fine-tuning goes through transformers' own rule.
        lambda: _llama(),
    ],
)
def test_skippable_models_keep_transformers_default(ddp_env, make):
    args = _args()
    assert _helper()(args, make()) is None


def test_non_lora_trainable_param_keeps_default(ddp_env):
    model = _lora(_llama())
    model.base_model.model.model.norm.weight.requires_grad_(True)
    assert _helper()(_args(), model) is None


def test_reused_args_redecides(ddp_env):
    args = _args()
    helper = _helper()
    assert helper(args, _lora(_llama())) is False
    # Same args, now a MoE model: the earlier automatic False must not stick.
    assert helper(args, _lora(_llama(num_local_experts = 4))) is None


def test_generated_trainer_calls_helper():
    source = (ROOT / "unsloth" / "models" / "rl.py").read_text()
    call = "unsloth_default_ddp_find_unused_parameters(args, model)"
    assert call in source
    # Before super().__init__: transformers builds the DDP kwargs inside Trainer.__init__ (5.x).
    template_init = source.index("class Unsloth{RLTrainer_name}(_Unsloth{RLTrainer_name}):")
    super_call = source.index("super().__init__({RLTrainer_call_args}", template_init)
    assert template_init < source.index(call, template_init) < super_call

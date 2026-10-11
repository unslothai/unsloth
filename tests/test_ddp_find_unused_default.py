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
    # TRL's configs (SFTConfig, GRPOConfig, ...) turn gradient checkpointing on by default.
    kwargs.setdefault("gradient_checkpointing", True)
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


def test_no_gradient_checkpointing_keeps_default(ddp_env):
    # Without checkpointing transformers' True works and tolerates an adapter a step never runs.
    args = _args(gradient_checkpointing = False)
    assert _helper()(args, _lora(_llama())) is None
    assert args.ddp_find_unused_parameters is None


def test_checkpointing_armed_on_model_only_defaults_to_false(ddp_env):
    # Unsloth arms checkpointing on the model (for_training) even when the args leave it off.
    model = _llama()
    model.gradient_checkpointing_enable()
    assert _helper()(_args(gradient_checkpointing = False), _lora(model)) is False


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


@pytest.mark.parametrize(
    "extra",
    [
        {"num_experts": 4},
        {"n_routed_experts": 4},
        {"moe_num_experts": 4},
        {"num_experts_per_tok": 2},
        {"layerdrop": 0.1},
        {"encoder_layerdrop": 0.1},
        {"decoder_layerdrop": 0.1},
        {"add_cross_attention": True},
    ],
)
def test_moe_and_layerdrop_config_keys_keep_default(ddp_env, extra):
    model = _lora(_llama())
    for key, value in extra.items():
        setattr(model.get_base_model().config, key, value)
    assert _helper()(_args(), model) is None


def test_vision_sub_config_keeps_default(ddp_env):
    model = _lora(_llama())
    # A VLM loaded through a text class still carries its tower config.
    model.get_base_model().config.vision_config = transformers.CLIPVisionConfig()
    assert _helper()(_args(), model) is None


def test_vision_config_under_text_config_keeps_default(ddp_env):
    model = _lora(_llama())
    outer = transformers.PretrainedConfig()
    outer.text_config = model.get_base_model().config
    outer.text_config.audio_config = transformers.PretrainedConfig()
    model.get_base_model().config = outer
    assert _helper()(_args(), model) is None


def test_two_adapters_keep_default(ddp_env):
    model = _lora(_llama())
    model.add_adapter("second", peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"]))
    assert _helper()(_args(), model) is None


def test_non_lora_peft_keeps_default(ddp_env):
    model = peft.get_peft_model(
        _llama(),
        peft.IA3Config(target_modules = ["q_proj", "down_proj"], feedforward_modules = ["down_proj"]),
    )
    assert _helper()(_args(), model) is None


@pytest.mark.parametrize("attr", ["target_parameters", "trainable_token_indices"])
def test_extra_trainable_targets_keep_default(ddp_env, attr):
    model = _lora(_llama())
    cfg = next(iter(model.peft_config.values()))
    object.__setattr__(
        cfg, attr, ["mlp.experts.gate_up_proj"] if attr == "target_parameters" else [1, 2]
    )
    assert _helper()(_args(), model) is None


def test_transformers_native_adapter_keeps_default(ddp_env):
    # PreTrainedModel.add_adapter keeps the model a PreTrainedModel: transformers' own rule decides.
    model = _llama()
    model.add_adapter(peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"]))
    assert isinstance(model.peft_config, dict) and len(model.peft_config) == 1
    assert _helper()(_args(), model) is None


def test_non_lora_peft_type_keeps_default(ddp_env):
    # Isolates the peft_type rule: LoRA-named tensors, but the adapter says it is something else.
    model = _lora(_llama())
    cfg = next(iter(model.peft_config.values()))
    object.__setattr__(cfg, "peft_type", peft.PeftType.ADALORA)
    assert _helper()(_args(), model) is None


def test_modules_to_save_config_keeps_default(ddp_env):
    # Isolates the modules_to_save rule from the tensor-name rules.
    model = _lora(_llama())
    cfg = next(iter(model.peft_config.values()))
    object.__setattr__(cfg, "modules_to_save", ["score"])
    assert _helper()(_args(), model) is None


def test_lora_on_linear_embedding_projection_keeps_default(ddp_env):
    # An embedding-side Linear (e.g. GPT-NeoX's `embed_out` head on transformers 4.x) gets lora_A / lora_B
    # tensors, so only the name rule keeps it out.
    model = _llama()
    model.model.embed_proj = torch.nn.Linear(16, 16, bias = False)
    model = _lora(model, target_modules = ["q_proj", "embed_proj"])
    assert any("embed_proj.lora_A" in n for n, _ in model.named_parameters())
    assert _helper()(_args(), model) is None


def test_lora_on_an_output_head_alias_keeps_default(ddp_env):
    # remote-code heads named output / output_layer: found through get_output_embeddings, not the name.
    model = _llama()
    model.output_layer = torch.nn.Linear(16, 64, bias = False)
    model.get_output_embeddings = lambda: model.output_layer
    model = _lora(model, target_modules = ["q_proj", "output_layer"])
    assert _helper()(_args(), model) is None


def test_dora_defaults_to_false(ddp_env):
    # DoRA magnitude vectors run on every forward, like lora_A / lora_B.
    assert _helper()(_args(), _lora(_llama(), use_dora = True)) is False


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


def test_reused_args_keep_a_value_set_after_the_auto_default(ddp_env):
    args = _args()
    helper = _helper()
    assert helper(args, _lora(_llama())) is False
    args.ddp_find_unused_parameters = True
    assert helper(args, _lora(_llama())) is True


def test_generated_trainer_calls_helper():
    source = (ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8")
    call = "unsloth_default_ddp_find_unused_parameters(args, model)"
    assert call in source
    # Before super().__init__: transformers builds the DDP kwargs inside Trainer.__init__ (5.x).
    template_init = source.index("class Unsloth{RLTrainer_name}(_Unsloth{RLTrainer_name}):")
    super_call = source.index("super().__init__({RLTrainer_call_args}", template_init)
    assert template_init < source.index(call, template_init) < super_call

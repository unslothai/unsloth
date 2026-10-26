# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unsloth trainers default DDP find_unused_parameters to False for a plain LoRA text model.

transformers picks True for every non-PreTrainedModel (all PEFT models). The helper is loaded by
path so the test stays CPU-only and does not import ``unsloth``.
"""

import importlib.util
import os
from pathlib import Path

import types

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
        {"cross_attention_layers": [3, 8]},  # MllamaTextConfig: layers skipped without images
        {"speech_encoder_layerdrop": 0.1},
        {"is_encoder_decoder": True},
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


def test_layerdrop_in_a_composite_encoder_config_keeps_default(ddp_env):
    # SpeechEncoderDecoder keeps Wav2Vec2's config (layerdrop 0.1) under config.encoder.
    model = _lora(_llama())
    model.get_base_model().config.encoder = types.SimpleNamespace(layerdrop = 0.1)
    assert _helper()(_args(), model) is None


def test_non_causal_lm_keeps_default(ddp_env):
    # SeamlessM4T-style models carry generation-only submodules (t2u_model) a training forward never runs.
    model = _llama()
    model.__class__ = type("TinyForSpeechToSpeech", (type(model),), {})
    assert _helper()(_args(), _lora(model)) is None


def test_lora_on_an_output_head_alias_keeps_default(ddp_env):
    # remote-code heads named output / output_layer: found through get_output_embeddings, not the name.
    model = _llama()
    model.output_layer = torch.nn.Linear(16, 64, bias = False)
    model.get_output_embeddings = lambda: model.output_layer
    model = _lora(model, target_modules = ["q_proj", "output_layer"])
    assert _helper()(_args(), model) is None


def test_dora_defaults_to_false(ddp_env):
    assert _helper()(_args(), _lora(_llama(), use_dora = True)) is False


def test_non_lora_trainable_param_keeps_default(ddp_env):
    model = _lora(_llama())
    model.base_model.model.model.norm.weight.requires_grad_(True)
    assert _helper()(_args(), model) is None


def test_reused_args_redecides(ddp_env):
    args = _args()
    helper = _helper()
    assert helper(args, _lora(_llama())) is False
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


def _llava(**text_extra):
    text = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        **text_extra,
    )
    vision = transformers.CLIPVisionConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        image_size = 8,
        patch_size = 4,
    )
    config = transformers.LlavaConfig(text_config = text, vision_config = vision, image_token_index = 63)
    model = transformers.LlavaForConditionalGeneration(config)
    # FastBaseModel.post_patch_model under DDP: VLMs checkpoint non-reentrantly.
    model._unsloth_vlm_gc_non_reentrant = True
    return model


def test_full_finetune_vlm_searches_unused_tower(ddp_env):
    # transformers' False would leave the tower without gradients on a text-only batch.
    args = _args()
    assert _helper()(args, _llava()) is True
    assert args.ddp_find_unused_parameters is True


def test_frozen_tower_keeps_transformers_default(ddp_env):
    model = _llava()
    for name, param in model.named_parameters():
        if "vision" in name or "projector" in name:
            param.requires_grad_(False)
    assert _helper()(_args(), model) is None


@pytest.mark.parametrize("case", ["no_flag", "reentrant_kwargs", "gemma4", "no_checkpointing"])
def test_vlm_tower_rule_needs_non_reentrant_checkpointing(ddp_env, case):
    model = _llava()
    kwargs = {}
    if case == "no_flag":
        model._unsloth_vlm_gc_non_reentrant = False
    elif case == "reentrant_kwargs":
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
    elif case == "gemma4":
        model.config.model_type = "gemma4"
    else:
        kwargs["gradient_checkpointing"] = False
    assert _helper()(_args(**kwargs), model) is None


def test_unsloth_config_reentrant_pin_is_overridden_by_the_ddp_wrapper(ddp_env, monkeypatch):
    # Unsloth's generated configs pin use_reentrant = True; under DDP vision.py swaps in a checkpoint
    # wrapper (marked _unsloth_original) that forces use_reentrant = False, so the pin is not what runs.
    import torch.utils.checkpoint as torch_checkpoint

    original = torch_checkpoint.checkpoint
    args = _args(gradient_checkpointing_kwargs = {"use_reentrant": True})
    assert _helper()(args, _llava()) is None

    def forced(function, *a, **k):
        k["use_reentrant"] = False
        return original(function, *a, **k)

    forced._unsloth_original = original
    monkeypatch.setattr(torch_checkpoint, "checkpoint", forced)
    args = _args(gradient_checkpointing_kwargs = {"use_reentrant": True})
    assert _helper()(args, _llava()) is True


def test_lora_vlm_tower_is_true_like_transformers(ddp_env):
    # transformers already picks True for a PeftModel; the helper agrees.
    model = _lora(_llava(), target_modules = ["q_proj", "v_proj"])
    assert _helper()(_args(), model) is True


def test_reused_args_redecide_after_an_auto_true(ddp_env):
    args = _args()
    helper = _helper()
    assert helper(args, _llava()) is True
    assert helper(args, _lora(_llama())) is False
    assert helper(args, _lora(_llama(num_local_experts = 4))) is None


def test_full_finetune_vlm_text_only_batches_train_under_ddp(tmp_path):
    # Two gloo ranks through the real Trainer: before the fix the second step raised
    # "Expected to have finished reduction in the prior iteration".
    import subprocess
    import sys

    worker = Path(__file__).with_name("_ddp_vlm_text_only_worker.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", TMPDIR = str(tmp_path))
    env.pop("UNSLOTH_DDP_FIND_UNUSED_PARAMETERS_AUTO", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(_free_port()),
            str(worker),
            str(tmp_path),
        ],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-4000:]
    assert out.count("RANK_OK find_unused=True") == 2, out[-4000:]


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

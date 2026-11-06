# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""DeepSpeed ZeRO-3 (#225): the load must not pass a device_map transformers refuses."""

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.models import loader_utils


@pytest.fixture
def zero3(monkeypatch):
    def set_enabled(enabled):
        monkeypatch.setattr(
            transformers.integrations, "is_deepspeed_zero3_enabled", lambda: enabled
        )

    return set_enabled


def test_zero3_drops_the_placement_unsloth_picked(zero3):
    zero3(True)
    for device_map in (loader_utils.DEFAULT_DEVICE_MAP, "sequential", "auto", "cuda"):
        assert loader_utils.deepspeed_zero3_device_map(device_map) is None, device_map


@pytest.mark.parametrize(
    "device_map", [{"model.layers.0": 0, "lm_head": 1}, {"": "cuda:1"}, "cuda:1", None]
)
def test_zero3_keeps_a_map_the_caller_chose(zero3, device_map):
    """transformers rejects these under ZeRO-3; silently dropping one would move the run to another card."""
    zero3(True)
    assert loader_utils.deepspeed_zero3_device_map(device_map) is device_map


def test_without_zero3_the_device_map_is_untouched(zero3):
    zero3(False)
    for device_map in (loader_utils.DEFAULT_DEVICE_MAP, "sequential", {"": "cuda:1"}, None):
        assert loader_utils.deepspeed_zero3_device_map(device_map) is device_map


_DS_ENV = (
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_DEEPSPEED_ZERO_STAGE",
    "ACCELERATE_DEEPSPEED_CONFIG_FILE",
)


@pytest.fixture
def no_ds_env(monkeypatch):
    for key in _DS_ENV:
        monkeypatch.delenv(key, raising = False)
    import transformers.integrations.deepspeed as hf_ds

    monkeypatch.setattr(hf_ds, "_hf_deepspeed_config_weak_ref", None, raising = False)
    try:
        from accelerate.state import AcceleratorState
        monkeypatch.setitem(AcceleratorState._shared_state, "deepspeed_plugins", None)
    except Exception:
        pass
    return monkeypatch


def test_zero_stage_unset_without_deepspeed(no_ds_env):
    assert loader_utils.deepspeed_zero_stage() is None
    assert not loader_utils.deepspeed_zero3_requested()


@pytest.mark.parametrize("stage", ["2", "3"])
def test_zero_stage_from_accelerate_env(no_ds_env, stage):
    no_ds_env.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    no_ds_env.setenv("ACCELERATE_DEEPSPEED_ZERO_STAGE", stage)
    assert loader_utils.deepspeed_zero_stage() == int(stage)
    assert loader_utils.deepspeed_zero3_requested() == (stage == "3")


def test_zero_stage_from_accelerate_config_file(no_ds_env, tmp_path):
    import json

    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"zero_optimization": {"stage": 3}}))
    no_ds_env.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    no_ds_env.setenv("ACCELERATE_DEEPSPEED_CONFIG_FILE", str(path))
    assert loader_utils.deepspeed_zero3_requested()
    no_ds_env.delenv("ACCELERATE_USE_DEEPSPEED")
    assert loader_utils.deepspeed_zero_stage() is None


def test_zero_stage_from_training_arguments_config(no_ds_env):
    import weakref
    import transformers.integrations.deepspeed as hf_ds

    class _Cfg:
        config = {"zero_optimization": {"stage": 3}}

    cfg = _Cfg()
    no_ds_env.setattr(hf_ds, "_hf_deepspeed_config_weak_ref", weakref.ref(cfg))
    assert loader_utils.deepspeed_zero3_requested()


def _tiny_qwen3_moe():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        moe_intermediate_size = 8,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 8,
        num_experts = 4,
        num_experts_per_tok = 2,
    )
    with torch.device("meta"):
        return Qwen3MoeForCausalLM(config)


@pytest.fixture
def fake_set_z3_leaf(monkeypatch):
    import sys
    import types

    calls = []

    def set_z3_leaf_modules(
        model,
        classes,
        raise_if_not_found = True,
    ):
        calls.append(list(classes))
        out = []
        for module in model.modules():
            if isinstance(module, tuple(classes)):
                module._z3_leaf = True
                out.append(module)
        return out

    utils = types.ModuleType("deepspeed.utils")
    utils.set_z3_leaf_modules = set_z3_leaf_modules
    ds = types.ModuleType("deepspeed")
    ds.utils = utils
    monkeypatch.setitem(sys.modules, "deepspeed", ds)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils)
    return calls


def test_moe_blocks_become_zero3_leaves(no_ds_env, fake_set_z3_leaf):
    no_ds_env.setattr(loader_utils, "deepspeed_zero3_requested", lambda: True)
    model = _tiny_qwen3_moe()
    names = loader_utils.set_deepspeed_moe_leaf_modules(model)
    assert names == ("Qwen3MoeSparseMoeBlock",), names
    blocks = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeSparseMoeBlock"]
    assert len(blocks) == 2 and all(getattr(m, "_z3_leaf", False) for m in blocks)
    assert sum(bool(getattr(m, "_z3_leaf", False)) for m in model.modules()) == 2


def test_moe_leaf_marking_is_zero3_only(no_ds_env, fake_set_z3_leaf):
    no_ds_env.setattr(loader_utils, "deepspeed_zero3_requested", lambda: False)
    model = _tiny_qwen3_moe()
    assert loader_utils.set_deepspeed_moe_leaf_modules(model) == ()
    assert fake_set_z3_leaf == []
    assert not any(getattr(m, "_z3_leaf", False) for m in model.modules())


def test_dense_model_has_no_moe_leaf(no_ds_env, fake_set_z3_leaf):
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    no_ds_env.setattr(loader_utils, "deepspeed_zero3_requested", lambda: True)
    with torch.device("meta"):
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size = 64,
                hidden_size = 16,
                intermediate_size = 32,
                num_hidden_layers = 1,
                num_attention_heads = 2,
                num_key_value_heads = 1,
            )
        )
    assert loader_utils.set_deepspeed_moe_leaf_modules(model) == ()
    assert fake_set_z3_leaf == []


def _filled_args(
    fp16,
    bf16,
    ds_fp16,
    ds_bf16,
    shared = True,
):
    import types

    config = {
        "fp16": {"enabled": ds_fp16},
        "bf16": {"enabled": ds_bf16},
        "zero_optimization": {"stage": 3},
    }
    plugin = types.SimpleNamespace(deepspeed_config = config)
    hf = types.SimpleNamespace(config = config if shared else dict(config))
    return types.SimpleNamespace(
        fp16 = fp16,
        bf16 = bf16,
        fp16_full_eval = False,
        bf16_full_eval = False,
        deepspeed_plugin = plugin,
        hf_deepspeed_config = hf,
    ), config


def test_forced_float32_disables_the_filled_deepspeed_fp16():
    # T4 + Gemma3 / Qwen3-MoE: fp16 = True was filled in, then Unsloth switched to float32.
    args, config = _filled_args(fp16 = False, bf16 = False, ds_fp16 = True, ds_bf16 = False)
    assert loader_utils.sync_deepspeed_mixed_precision(args) == ["fp16"]
    assert config["fp16"]["enabled"] is False and config["bf16"]["enabled"] is False
    assert loader_utils.sync_deepspeed_mixed_precision(args) == []


def test_resync_clears_queued_precision_mismatches_and_dtype():
    import inspect
    import types

    import torch

    from transformers.integrations import deepspeed as hf_ds

    assert 'self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")' in (
        inspect.getsource(hf_ds)
    )
    args, config = _filled_args(fp16 = False, bf16 = False, ds_fp16 = True, ds_bf16 = False)
    args.hf_deepspeed_config = types.SimpleNamespace(
        config = config,
        mismatches = [
            "- ds fp16.enabled=True vs hf fp16|fp16_full_eval=False",
            "- ds train_micro_batch_size_per_gpu=4 vs hf per_device_train_batch_size=2",
        ],
        _dtype = torch.float16,
    )
    assert loader_utils.sync_deepspeed_mixed_precision(args) == ["fp16"]
    assert args.hf_deepspeed_config.mismatches == [
        "- ds train_micro_batch_size_per_gpu=4 vs hf per_device_train_batch_size=2"
    ]
    assert args.hf_deepspeed_config._dtype is torch.float32


def test_resync_dtype_follows_the_flags_when_sections_are_absent():
    import types

    import torch

    config = {"zero_optimization": {"stage": 3}}
    args = types.SimpleNamespace(
        fp16 = False,
        bf16 = True,
        fp16_full_eval = False,
        bf16_full_eval = False,
        deepspeed_plugin = types.SimpleNamespace(deepspeed_config = config),
        hf_deepspeed_config = types.SimpleNamespace(
            config = config, mismatches = [], _dtype = torch.float32
        ),
    )
    assert loader_utils.sync_deepspeed_mixed_precision(args) == []
    assert args.hf_deepspeed_config._dtype is torch.bfloat16


def test_auto_and_absent_sections_are_left_to_transformers():
    import types

    config = {"fp16": {"enabled": "auto"}}
    args = types.SimpleNamespace(
        fp16 = False,
        bf16 = True,
        deepspeed_plugin = types.SimpleNamespace(deepspeed_config = config),
        hf_deepspeed_config = None,
    )
    assert loader_utils.sync_deepspeed_mixed_precision(args) == []
    assert config == {"fp16": {"enabled": "auto"}}


def test_no_deepspeed_config_is_a_no_op():
    import types
    args = types.SimpleNamespace(fp16 = True, bf16 = False)
    assert loader_utils.sync_deepspeed_mixed_precision(args) == []


def test_rl_trainer_source_resyncs_deepspeed_after_the_precision_switch():
    """The generated Unsloth*Trainer __init__ records the requested precision before its switches."""
    import inspect
    from unsloth.models import rl

    source = inspect.getsource(rl)
    first = source.index("_unsloth_precision_requested = tuple(")
    switch = source.index("Switching to float32 training since model cannot work with float16")
    sync = source.index("sync_deepspeed_mixed_precision(args)")
    assert first < switch < sync


@pytest.mark.parametrize(
    "distributed, fsdp2, stage, reentrant",
    [
        (False, False, None, True),  # one GPU
        (True, False, None, False),  # DDP: non-reentrant (#3713)
        (True, True, None, True),  # FSDP2
        (True, False, 3, True),  # DeepSpeed ZeRO-3: non-reentrant recompute fails its checks
        (True, False, 2, True),  # DeepSpeed ZeRO-2: no DDP reducer either
    ],
)
def test_vlm_gradient_checkpointing_reentrant(monkeypatch, distributed, fsdp2, stage, reentrant):
    monkeypatch.setattr(loader_utils, "is_distributed", lambda: distributed)
    monkeypatch.setattr(loader_utils, "fsdp2_requested", lambda: fsdp2)
    monkeypatch.setattr(loader_utils, "deepspeed_zero_stage", lambda: stage)
    assert loader_utils.vlm_gradient_checkpointing_reentrant() is reentrant


def test_fused_lora_layers_become_zero3_leaves_only_under_zero3(monkeypatch):
    torch = pytest.importorskip("torch")
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    monkeypatch.setattr(loader_utils, "deepspeed_zero3_requested", lambda: False)
    assert loader_utils.set_deepspeed_fused_lora_leaf_modules(layers) == 0
    assert not any(getattr(m, "_z3_leaf", False) for m in layers)
    monkeypatch.setattr(loader_utils, "deepspeed_zero3_requested", lambda: True)
    assert loader_utils.set_deepspeed_fused_lora_leaf_modules(layers) == 2
    assert all(m._z3_leaf for m in layers)
    try:
        from deepspeed.utils import z3_leaf_module
    except Exception:
        return
    # The flag DeepSpeed's hook installer reads (deepspeed/utils/z3_leaf_module.py).
    assert all(z3_leaf_module(m) for m in layers)


def test_only_layers_with_installed_fused_kernels_are_leaves():
    import types

    torch = pytest.importorskip("torch")
    from unsloth.models import llama

    def fused_mlp(self, x):
        return x

    def layer(
        qkv = None,
        o = None,
        mlp = False,
    ):
        m = torch.nn.Module()
        m.self_attn = torch.nn.Module()
        m.self_attn.apply_qkv = qkv or (lambda *a: None)
        m.self_attn.apply_o = o or (lambda *a: None)
        m.mlp = torch.nn.Module()
        if mlp:
            m.mlp.forward = types.MethodType(fused_mlp, m.mlp)
        return m

    # Qwen2 with a q_proj-only adapter: nothing fused, nothing gathered as a whole block.
    plain = layer()
    qkv = layer(qkv = llama.apply_lora_qkv)
    o = layer(o = llama.apply_lora_o)
    mlp = layer(mlp = True)
    assert llama._fused_lora_layers([plain, qkv, o, mlp], fused_mlp) == [qkv, o, mlp]


def test_trainer_hook_redoes_zero3_setup_for_a_model_loaded_before_deepspeed(
    no_ds_env, fake_set_z3_leaf, capsys
):
    """Loaded and PEFT-patched before TrainingArguments(deepspeed=...): the trainer hook marks the leaves."""
    import weakref
    import transformers.integrations.deepspeed as hf_ds

    model = _tiny_qwen3_moe()
    layers = list(model.model.layers)
    assert loader_utils.set_deepspeed_moe_leaf_modules(model) == ()
    assert loader_utils.set_deepspeed_fused_lora_leaf_modules(layers[:1]) == 0
    model._unsloth_vlm_gc_non_reentrant = True
    assert not any(getattr(m, "_z3_leaf", False) for m in model.modules())

    class _Cfg:
        config = {"zero_optimization": {"stage": 3}}

    cfg = _Cfg()
    no_ds_env.setattr(hf_ds, "_hf_deepspeed_config_weak_ref", weakref.ref(cfg))
    for _ in range(2):
        loader_utils.apply_deepspeed_zero3_model_setup(model)
        blocks = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeSparseMoeBlock"]
        assert len(blocks) == 2 and all(m._z3_leaf for m in blocks)
        assert layers[0]._z3_leaf and not getattr(layers[1], "_z3_leaf", False)
        assert sum(bool(getattr(m, "_z3_leaf", False)) for m in model.modules()) == 3
    assert capsys.readouterr().out.count("non-reentrant gradient checkpointing") == 1


def test_trainer_hook_is_a_no_op_without_zero3(no_ds_env, fake_set_z3_leaf):
    model = _tiny_qwen3_moe()
    loader_utils.set_deepspeed_fused_lora_leaf_modules(list(model.model.layers))
    loader_utils.apply_deepspeed_zero3_model_setup(model)
    loader_utils.apply_deepspeed_zero3_model_setup("not-a-model")
    assert fake_set_z3_leaf == []
    assert not any(getattr(m, "_z3_leaf", False) for m in model.modules())


def test_rl_trainer_source_redoes_zero3_setup():
    import inspect
    from unsloth.models import rl

    source = inspect.getsource(rl)
    assert source.index("sync_deepspeed_mixed_precision(args)") < source.index(
        "apply_deepspeed_zero3_model_setup(model)"
    )

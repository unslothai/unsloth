# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which multi-GPU route takes a `gguf_file` load (dense or `gguf_packed = True`), and what a refused one says.

* pipeline parallel splits layers by their logical weight elements, so packed GGUF blocks (uint8, about half a
  byte per element for Q4_K) cost what the dense layer costs and the split matches the dequantized model;
* tensor parallel refuses packed GGUF Linears by name (not as MXFP4 or a weightless GPTQ layer), and
  UNSLOTH_TP_SIZE refuses a `gguf_packed = True` load before loading anything;
* every sharded loader (native FSDP2, pipeline, tensor parallel) names the gguf_file it cannot read and says
  what to do, and the size-triggered (auto) sharded load always loads a GGUF whole;
* UNSLOTH_PARALLEL=auto never picks a route that refuses the load: a gguf_file is not sized from safetensors
  (FSDP2), and packed weights on a torch whose FSDP2 cannot keep them replicated take DDP.

CPU only; the gloo parity of the routes that run is tests/test_gguf_packed_parallel_gloo.py."""

import os
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_parallel_auto import _auto_env, clean  # noqa: E402, F401  (clean: a fixture)


def _gp():
    from unsloth.models import gguf_packed
    return gguf_packed


def _packed_linear(
    dense,
    type_name = "Q4_K",
    seed = 0,
):
    from test_gguf_packed_lora import _blocks

    gp = _gp()
    from unsloth.kernels import gguf_reference as ref

    blocks = _blocks(type_name, dense.out_features, dense.in_features, seed)
    return gp.UnslothGGUFLinear.from_packed(
        blocks,
        ref.GGML_TYPE_IDS[type_name],
        dense.in_features,
        dense.out_features,
        compute_dtype = torch.float32,
    )


def _qwen3(layers = 4):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size = 512,
            hidden_size = 256,
            intermediate_size = 512,
            num_hidden_layers = layers,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 128,
            tie_word_embeddings = False,
        )
    )


def _pack_layers(model, layers):
    for i in layers:
        layer = model.model.layers[i]
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if type(child) is nn.Linear:
                    setattr(parent, name, _packed_linear(child, "Q4_K" if i % 2 else "Q8_0", i))
    return model


# ------------------------------------------------------------------------------------------ pipeline split


def test_pp_layer_cost_counts_packed_gguf_weights_by_logical_elements():
    from unsloth.models import pp_native

    dense = _qwen3()
    packed = _pack_layers(_qwen3(), range(4))
    for i in range(4):
        want = pp_native._layer_cost(dense.model.layers[i])
        got = pp_native._layer_cost(packed.model.layers[i])
        assert got == want, (i, got, want)
    for stages in (2, 3, 4):
        assert pp_native.derive_pp_split(packed, stages) == pp_native.derive_pp_split(dense, stages)
    # Half the layers packed (a partly converted model): the split must not move work onto the packed half.
    mixed = _pack_layers(_qwen3(), (0, 1))
    assert pp_native.derive_pp_split(mixed, 2) == pp_native.derive_pp_split(dense, 2)


def test_pp_layer_cost_counts_packed_gguf_expert_stacks_by_logical_elements():
    from unsloth.models import pp_native

    Expert = _gp().gguf_expert_param_class()
    if Expert is None:
        pytest.skip("unsloth_zoo without the packed GGUF expert hook")
    from test_gguf_packed_lora import _blocks
    from unsloth.kernels import gguf_reference as ref

    E, N, K = 4, 64, 256
    blocks = torch.stack([_blocks("Q4_K", N, K, seed) for seed in range(E)])
    stack = Expert(blocks, ggml_type = ref.GGML_TYPE_IDS["Q4_K"], packed_cols = K)
    holder = nn.Module()
    holder.gate_up_proj = stack
    assert stack.numel() < E * N * K
    assert pp_native._layer_cost(holder) == E * N * K


# ------------------------------------------------------------------------------------------ tensor parallel


def test_tp_refuses_packed_gguf_linears_by_name():
    from unsloth.models import tp_native

    module = _packed_linear(nn.Linear(256, 64, bias = False))
    weight, base = tp_native._weight(module)
    with pytest.raises(ValueError) as info:
        tp_native._refuse_base(
            "model.layers.0.self_attn.q_proj", module, weight, base, "colwise", 2
        )
    message = str(info.value)
    assert "packed GGUF weights (gguf_packed = True" in message, message
    assert "without gguf_packed" in message and "UNSLOTH_FSDP2=1" in message, message
    assert "MXFP4" not in message and "GPTQ" not in message, message


@pytest.fixture
def tp_layout(monkeypatch):
    from unsloth.models import tp_env

    for key in ("UNSLOTH_PARALLEL", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "UNSLOTH_EP_SIZE", "UNSLOTH_FSDP2",
                "UNSLOTH_GGUF_PACKED"):  # fmt: skip
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "2")
    monkeypatch.setenv("WORLD_SIZE", "2")
    tp_env._reset_env_layout()
    yield monkeypatch, tp_env
    tp_env._reset_env_layout()


@pytest.mark.parametrize("via_env", [False, True])
def test_tp_env_refuses_a_gguf_packed_load_before_loading(tp_layout, via_env):
    monkeypatch, tp_env = tp_layout
    calls = []

    def from_pretrained(
        model_name = None,
        gguf_file = None,
        **kwargs,
    ):
        calls.append(model_name)

    kwargs = {"gguf_file": "model-Q4_K_M.gguf"}
    if via_env:
        monkeypatch.setenv("UNSLOTH_GGUF_PACKED", "1")
    else:
        kwargs["gguf_packed"] = True
    if via_env:
        # The loader keeps a GGUF packed only for the keyword; the variable alone loads it dense, which works
        # (past the refusal; this test has no process group for the tensor parallel setup that follows).
        try:
            tp_env.tensor_parallel_env_load(from_pretrained)("org/model-GGUF", **kwargs)
        except Exception as error:
            assert "gguf_packed" not in str(error), error
        return
    with pytest.raises(NotImplementedError) as info:
        tp_env.tensor_parallel_env_load(from_pretrained)("org/model-GGUF", **kwargs)
    message = str(info.value)
    assert message.startswith(
        "Unsloth: UNSLOTH_TP_SIZE=2 (tensor parallel) does not support"
    ), message
    assert "gguf_packed = True" in message and "without gguf_packed" in message, message
    assert "UNSLOTH_FSDP2=1" in message and not calls, message


# ------------------------------------------------------------------------------------------ sharded loading


def _fake_mesh():
    return types.SimpleNamespace(ndim = 1, size = lambda: 2, mesh = torch.arange(2))


def _gguf_kwargs(packed):
    kwargs = {"gguf_file": "tiny-Q4_K_M.gguf"}
    if packed:
        kwargs["quantization_config"] = _gp().gguf_packed_quantization_config()
    return kwargs


def _sharded_refusal(state, tmp_path, packed):
    from transformers import LlamaForCausalLM
    from unsloth.models import tp_native

    def original(*args, **kwargs):
        raise AssertionError("a refused sharded load must not load")

    state = dict({"mesh": _fake_mesh(), "plan": None, "active": False}, **state)
    with pytest.raises(ValueError) as info:
        tp_native._from_pretrained_sharded(
            original, LlamaForCausalLM, str(tmp_path), (), _gguf_kwargs(packed), state
        )
    return str(info.value)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("route", ["fsdp2", "pp", "tp"])
def test_sharded_loading_names_the_gguf_file_and_what_to_do(tmp_path, packed, route):
    if packed and _gp().gguf_packed_supported() is not None:
        pytest.skip("gguf_packed needs a newer transformers")
    from unsloth.models import fsdp2_load, pp_load

    if route == "fsdp2":
        state, label, hint = (
            {"label": "native FSDP2", "gguf_hint": getattr(fsdp2_load, "_GGUF_HINT", None)},
            "native FSDP2",
            "UNSLOTH_FSDP2_SHARDED_LOAD",
        )
    elif route == "pp":
        # The hint pipeline_sharded_loading passes, read off its own state.
        import inspect
        assert "gguf_hint" in inspect.getsource(
            pp_load._pipeline_loading
        ), "pipeline_sharded_loading passes no gguf_hint"
        state, label, hint = (
            {"label": "pipeline parallel", "gguf_hint": "Drop pipeline_sharded_loading"},
            "pipeline parallel",
            "Drop pipeline_sharded_loading",
        )
    else:
        state, label, hint = {}, "tensor parallel", "UNSLOTH_TP_SHARDED_LOAD=0"
    message = _sharded_refusal(state, tmp_path, packed)
    assert message.startswith(f"Unsloth: {label} sharded loading"), message
    assert "gguf_file" in message and "tiny-Q4_K_M.gguf" in message and hint in message, message
    what = "a gguf_file with gguf_packed = True (" if packed else "a gguf_file ("
    assert f"cannot load {what}tiny-Q4_K_M.gguf)" in message, message
    assert (
        "(got quantization_config)" not in message and "load_in_4bit = False" not in message
    ), message


def test_fsdp2_sharded_loading_context_refuses_a_gguf_file(tmp_path):
    """Through the real context (the swapped from_pretrained), not only the shared function."""
    from transformers import LlamaForCausalLM
    from unsloth.models import fsdp2_load

    with fsdp2_load.fsdp2_sharded_loading(mesh = _fake_mesh()):
        with pytest.raises(ValueError) as info:
            LlamaForCausalLM.from_pretrained(str(tmp_path), gguf_file = "tiny-Q4_K_M.gguf")
    message = str(info.value)
    assert message.startswith("Unsloth: native FSDP2 sharded loading"), message
    assert "gguf_file" in message and "UNSLOTH_FSDP2_SHARDED_LOAD" in message, message


@pytest.mark.parametrize("packed", [False, True])
def test_size_triggered_sharded_load_keeps_a_gguf_whole(tmp_path, monkeypatch, packed):
    """Native FSDP2 / UNSLOTH_PP_SIZE shard a load only when it does not fit the GPU, sized from the repo's
    safetensors; a GGUF repo that also holds a large safetensors checkpoint must still load the GGUF whole."""
    if packed and _gp().gguf_packed_supported() is not None:
        pytest.skip("gguf_packed needs a newer transformers")
    from transformers import LlamaForCausalLM
    from unsloth.models import tp_native

    monkeypatch.setattr(tp_native, "_whole_load_exceeds_device", lambda *a, **k: True)
    monkeypatch.setattr(tp_native, "_load_device", lambda *a, **k: torch.device("cuda", 0))
    calls = []

    def original(cls, name, *args, **kwargs):
        calls.append(kwargs.get("gguf_file"))
        return "whole"

    state = {
        "mesh": _fake_mesh(),
        "plan": None,
        "active": False,
        "auto": True,
        "label": "native FSDP2",
    }
    got = tp_native._from_pretrained_sharded(
        original, LlamaForCausalLM, str(tmp_path), (), _gguf_kwargs(packed), state
    )
    assert got == "whole" and calls == ["tiny-Q4_K_M.gguf"]


# ------------------------------------------------------------------------------------- UNSLOTH_PARALLEL=auto


def _loader(
    model_name = "x",
    load_in_4bit = True,
    gguf_file = None,
    gguf_packed = None,
    **kwargs,
):
    return model_name


def test_auto_does_not_size_a_gguf_file_from_safetensors(clean):
    """A small safetensors checkpoint beside the GGUF would size as fitting DDP; the GGUF load is not sized."""
    mp, _, auto = clean
    _auto_env(mp, auto, budget_mib = 4096)
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: [("model.a.weight", "BF16", (64, 64))])
    mp.setattr(auto, "_fsdp2_keeps_packed", lambda: True, raising = False)
    for kwargs, packed in (
        ({"gguf_file": "m-Q4_K_M.gguf"}, False),
        ({"gguf_file": "m.gguf", "gguf_packed": True}, True),
    ):
        auto._STATE.update(choice = None, reason = None)
        assert auto.resolve_for_load(_loader, ("org/model-GGUF",), kwargs) == "fsdp2", auto._STATE[
            "reason"
        ]
        reason = auto._STATE["reason"]
        assert "gguf_file load is not sized" in reason, reason
        assert ("keeps the gguf_packed = True blocks replicated" in reason) == packed, reason
        assert "rank 1:" not in reason, reason  # the same reason on every rank is given once


def test_auto_takes_ddp_for_gguf_packed_when_fsdp2_would_refuse_it(clean):
    """torch without fully_shard(ignored_params): native FSDP2 refuses packed GGUF blocks, DDP runs them."""
    mp, _, auto = clean
    _auto_env(mp, auto, budget_mib = 1)  # sized, nothing would fit: FSDP2, the refused route
    mp.setattr(
        auto, "_checkpoint_tensors", lambda *a, **k: [("model.a.weight", "BF16", (4096, 4096))]
    )
    mp.setattr(auto, "_fsdp2_keeps_packed", lambda: False, raising = False)
    kwargs = {"gguf_file": "m-Q4_K_M.gguf", "gguf_packed": True}
    assert auto.resolve_for_load(_loader, ("org/model-GGUF",), kwargs) == "ddp", auto._STATE[
        "reason"
    ]
    assert "gguf_packed = True blocks" in auto._STATE["reason"], auto._STATE["reason"]
    assert os.environ["UNSLOTH_FSDP2"] == "0"
    # The dense GGUF has no packed weights: FSDP2 shards it on any torch.
    auto._STATE.update(choice = None, reason = None)
    mp.delenv("UNSLOTH_FSDP2")
    mp.delenv("UNSLOTH_PARALLEL_AUTO_RESOLVED")
    assert auto.resolve_for_load(_loader, ("org/model-GGUF",), {"gguf_file": "m.gguf"}) == "fsdp2"


def test_auto_trainer_takes_ddp_for_packed_weights_fsdp2_cannot_keep(clean):
    mp, _, auto = clean
    _auto_env(mp, auto, budget_mib = 1)
    model = nn.Module()
    model.weight_packed = nn.Parameter(
        torch.zeros(4096, 4096, dtype = torch.uint8), requires_grad = False
    )
    model.lora = nn.Parameter(torch.zeros(16, 4096))
    mp.setattr(auto, "_fsdp2_keeps_packed", lambda: False, raising = False)
    assert auto.resolve_for_trainer(model) == "ddp", auto._STATE["reason"]
    assert "packed (integer-coded) weights" in auto._STATE["reason"], auto._STATE["reason"]
    # A torch that keeps them replicated sizes the model as before (here: too big, so FSDP2).
    auto._STATE.update(choice = None, reason = None)
    mp.delenv("UNSLOTH_FSDP2")
    mp.delenv("UNSLOTH_PARALLEL_AUTO_RESOLVED")
    mp.setattr(auto, "_fsdp2_keeps_packed", lambda: True, raising = False)
    assert auto.resolve_for_trainer(model) == "fsdp2", auto._STATE["reason"]

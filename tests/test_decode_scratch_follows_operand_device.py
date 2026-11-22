# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decode scratch buffers in Gemma, Gemma-2, Cohere and Falcon-H1 must be allocated on the device of
the tensor they are combined with, not on accelerator index 0.

A model placed on `cuda:1` (a DDP rank, or `device_map = {"": 1}`) otherwise builds its layernorm
weight copy and its paged KV cache on `cuda:0` and the first in-place op mixing the two raises a
cross-device error. The operands here live on the meta device, so the check runs on any host: every
`torch.empty` issued from the model module during the decode prologue is recorded and must be on
the operand's device.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

META = torch.device("meta")


class _Stop(Exception):
    pass


def _record_empty(monkeypatch, module_name):
    devices = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if sys._getframe(1).f_globals.get("__name__") == module_name:
            devices.append(torch.device(kwargs.get("device", "cpu")))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    return devices


def _run(fn, *args, **kwargs):
    # The fakes only carry what the allocation prologue reads; whatever runs after it may fail.
    try:
        fn(*args, **kwargs)
    except _Stop:
        pass
    except Exception:
        pass


def _norm(n):
    return types.SimpleNamespace(
        weight = torch.zeros(n, device = META),
        variance_epsilon = 1e-6,
    )


def _stop(*args, **kwargs):
    raise _Stop


@pytest.mark.parametrize(
    "module_name, function_name, norm_function",
    [
        (
            "unsloth.models.gemma",
            "GemmaDecoderLayer_fast_forward",
            "fast_rms_layernorm_inference_gemma",
        ),
        (
            "unsloth.models.gemma2",
            "Gemma2DecoderLayer_fast_forward",
            "fast_rms_layernorm_inference_gemma",
        ),
        ("unsloth.models.cohere", "CohereDecoderLayer_fast_forward", "fast_layernorm_inference"),
    ],
)
def test_decoder_layer_norm_weight_follows_hidden_states(
    monkeypatch, module_name, function_name, norm_function
):
    module = pytest.importorskip(module_name)
    captured = []

    def norm(
        layer,
        X,
        out_weight = None,
        *args,
        **kwargs,
    ):
        captured.append(out_weight.device)
        raise _Stop

    monkeypatch.setattr(module, norm_function, norm)
    devices = _record_empty(monkeypatch, module_name)
    layer = types.SimpleNamespace(input_layernorm = _norm(16), _flag_for_generation = True)
    hidden_states = torch.zeros(1, 1, 16, dtype = torch.float16, device = META)

    _run(getattr(module, function_name), layer, hidden_states, use_cache = True)

    assert devices, "the generation branch allocated nothing"
    assert devices == [META] * len(devices), f"scratch allocated on {devices}"
    assert captured == [META]


def test_cohere_attention_prefill_buffers_follow_hidden_states(monkeypatch):
    module = pytest.importorskip("unsloth.models.cohere")
    devices = _record_empty(monkeypatch, module.__name__)
    n_heads, n_kv_heads, head_dim, bsz, seq_len = 4, 2, 8, 1, 5
    attention = types.SimpleNamespace(
        config = types.SimpleNamespace(
            num_attention_heads = n_heads,
            num_key_value_heads = n_kv_heads,
            hidden_size = n_heads * head_dim,
        ),
        num_key_value_groups = n_heads // n_kv_heads,
        head_dim = head_dim,
        use_qk_norm = True,
        q_norm = _norm((n_heads, head_dim)),
        k_norm = _norm((n_kv_heads, head_dim)),
    )
    K1 = torch.zeros(bsz, n_kv_heads, seq_len, head_dim, dtype = torch.float16, device = META)
    hidden_states = torch.zeros(bsz, 1, n_heads * head_dim, dtype = torch.float16, device = META)

    _run(
        module.CohereAttention_fast_forward_inference,
        attention,
        hidden_states,
        (K1, K1.clone()),
        None,
        do_prefill = True,
    )

    # paged cache, Q/A, K/V, RoPE Q, attention scores, two QK-norm weights.
    assert len(devices) >= 7, devices
    assert devices == [META] * len(devices), f"scratch allocated on {devices}"


def test_falcon_h1_decode_buffers_follow_embeddings(monkeypatch):
    module = pytest.importorskip("unsloth.models.falcon_h1")
    devices = _record_empty(monkeypatch, module.__name__)
    hidden, mlp = 16, 32
    config = types.SimpleNamespace(
        hidden_size = hidden,
        intermediate_size = mlp,
        mlp_multipliers = (1.0, 1.0),
        embedding_multiplier = 1.0,
        torch_dtype = torch.float16,
        dtype = torch.float16,
    )

    def embed_tokens(input_ids):
        return torch.zeros(*input_ids.shape, hidden, dtype = torch.float16, device = META)

    class _NoCache:
        def __getitem__(self, index):
            raise _Stop

    model = types.SimpleNamespace(
        config = config,
        max_seq_length = 64,
        model = types.SimpleNamespace(embed_tokens = embed_tokens),
    )
    decode = module._FalconH1_fast_forward_inference()

    _run(decode, model, torch.zeros(1, 1, dtype = torch.long), _NoCache(), None)

    assert len(devices) == 4, devices
    assert devices == [META] * 4, f"scratch allocated on {devices}"


def test_gemma2_static_masks_are_built_on_the_embeddings_device():
    """The static sliding-window and global masks were built on the bare device type, which is
    the current device (cuda:0) for a model placed on cuda:1. Read from source: building them
    needs a whole model forward."""
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    devices = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_causal_4d"
        ):
            devices += [ast.unparse(k.value) for k in node.keywords if k.arg == "device"]
    assert len(devices) == 2, devices
    assert devices == ["inputs_embeds.device"] * 2, devices

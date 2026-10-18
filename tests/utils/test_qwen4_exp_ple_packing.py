# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Packed Qwen4Exp (Qwen3.8-Flash-Next) PLE keeps documents apart, on a tiny random model built
from the real transformers classes. GDN gets exact per-document reference kernels so PLE is the
only mixer that could leak."""

import importlib.util
import inspect
import sys

import pytest
import torch

qm = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

import unsloth.utils.packing as packing_module
from unsloth.utils.packing import patch_hybrid_linear_attention_varlen

EOS = 7
LENGTHS = [11, 1, 17, 9]


def _segments(
    T,
    seq_idx = None,
    cu_seqlens = None,
):
    if cu_seqlens is not None:
        bounds = cu_seqlens.tolist()
    elif seq_idx is not None:
        ids = seq_idx.reshape(-1)
        bounds = [0] + (torch.nonzero(ids[1:] != ids[:-1]).flatten() + 1).tolist() + [T]
    else:
        bounds = [0, T]
    return list(zip(bounds[:-1], bounds[1:]))


def _hub(torch_function, implementation):
    # Shape of transformers' kernel-hub wrapper, which the packing gate inspects.
    applicable_params = tuple(inspect.signature(implementation).parameters)

    def wrapped(*args, **kwargs):
        if implementation is None:
            return torch_function(*args, **kwargs)
        return implementation(*args, **{k: v for k, v in kwargs.items() if k in applicable_params})

    return wrapped


def _torch_fallback(name):
    # unsloth_zoo may have replaced the global on GPU: read a pristine copy of the modeling file.
    closure = packing_module._hub_closure(getattr(qm, name))
    if "torch_function" not in closure:
        spec = importlib.util.spec_from_file_location(
            "transformers.models.qwen4_exp._pristine", qm.__file__
        )
        pristine = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pristine  # auto_docstring inspects the module
        try:
            spec.loader.exec_module(pristine)
        finally:
            sys.modules.pop(spec.name, None)
        fn = getattr(pristine, name)
        # With the `kernels` package installed the hub decorator nests one more wrapper.
        return packing_module._hub_closure(fn).get("torch_function") or inspect.unwrap(fn)
    return closure["torch_function"]


@pytest.fixture
def reference_gdn_kernels(monkeypatch):
    conv_t = _torch_fallback("causal_conv1d_fn")
    gdn_t = _torch_fallback("torch_chunk_gated_delta_rule")

    def conv_ref(
        hidden_states,
        weight,
        bias = None,
        activation = None,
        seq_idx = None,
    ):
        T = hidden_states.shape[-1]
        parts = [
            conv_t(hidden_states[..., s:e], weight, bias, activation)
            for s, e in _segments(T, seq_idx)
        ]
        return torch.cat(parts, -1)

    def gdn_ref(
        query,
        key,
        value,
        g,
        beta,
        chunk_size = 64,
        initial_state = None,
        output_final_state = False,
        use_qk_l2norm_in_kernel = False,
        cu_seqlens = None,
    ):
        outs = [
            gdn_t(
                query[:, s:e],
                key[:, s:e],
                value[:, s:e],
                g[:, s:e],
                beta[:, s:e],
                chunk_size = chunk_size,
                use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel,
            )[0]
            for s, e in _segments(query.shape[1], cu_seqlens = cu_seqlens)
        ]
        return torch.cat(outs, 1), None

    monkeypatch.setattr(qm, "causal_conv1d_fn", _hub(conv_t, conv_ref))
    monkeypatch.setattr(qm, "torch_chunk_gated_delta_rule", _hub(gdn_t, gdn_ref))
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")


def _tiny_model():
    torch.manual_seed(0)
    config = Qwen4ExpTextConfig(
        vocab_size = 512,
        hidden_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        linear_conv_kernel_dim = 4,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        moe_intermediate_size = 32,
        shared_expert_intermediate_size = 32,
        num_experts = 4,
        num_experts_per_tok = 2,
        full_attention_interval = 4,
        hc_count = 4,
        hc_lowrank = 8,
        ple_layer_ids = [2],
        ple_embed_dim = 48,
        ple_conv_kernel_size = 4,
        ngram_size = 3,
        heads_per_ngram = 2,
        ngram_vocab_size_base = 1000,
        split_ngram_parts = 4,
        indexer_n_heads = 2,
        indexer_kv_heads = 1,
        indexer_head_dim = 16,
        indexer_budget = 2048,
        indexer_compress_ratio = 4,
        output_gate_type = "sigmoid",
        eos_token_id = EOS,
        bos_token_id = EOS,
        rope_parameters = {
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )
    config._attn_implementation = "sdpa"
    model = qm.Qwen4ExpForCausalLM(config).eval()
    # HF init zeroes the PLE conv; a trained checkpoint does not.
    (ple,) = [m for m in model.modules() if isinstance(m, qm.Qwen4ExpTextPLELayer)]
    with torch.no_grad():
        ple.conv1d.weight.copy_(
            0.5 * torch.randn(ple.conv1d.weight.shape, generator = torch.Generator().manual_seed(3))
        )
    return model, ple


@pytest.mark.parametrize("eos", [True, False])
def test_packed_ple_matches_separate_documents(reference_gdn_kernels, eos):
    model, ple = _tiny_model()
    assert patch_hybrid_linear_attention_varlen(model) is True

    g = torch.Generator().manual_seed(1)
    docs = [torch.randint(8, 512, (n,), generator = g) for n in LENGTHS]
    if eos:
        for d in docs:
            d[-1] = EOS
    ids = torch.cat(docs)[None]
    embed = model.get_input_embeddings()

    def run(input_ids, **kwargs):
        x = embed(input_ids).detach().requires_grad_(True)
        n = input_ids.shape[1]
        position_ids = kwargs.pop("position_ids", torch.arange(n)[None])
        out = model(
            inputs_embeds = x,
            ple_input_ids = input_ids,
            position_ids = position_ids,
            use_cache = False,
            **kwargs,
        )
        return x, out.logits

    x, packed = run(
        ids,
        position_ids = torch.cat([torch.arange(n) for n in LENGTHS])[None],
        packed_seq_lengths = torch.tensor(LENGTHS, dtype = torch.int32),
    )
    alone = torch.cat([run(d[None])[1] for d in docs], dim = 1)
    torch.testing.assert_close(packed, alone, rtol = 1e-5, atol = 1e-5)

    # A loss on the third document sends no gradient into any other document.
    start = sum(LENGTHS[:2])
    packed[0, start : start + LENGTHS[2]].sum().backward()
    grad = x.grad[0].abs().sum(-1)
    assert grad[start : start + LENGTHS[2]].sum() > 0
    assert grad[:start].sum() == 0 and grad[start + LENGTHS[2] :].sum() == 0

    kinds = {kind for _, kind in packing_module._iter_stateful_mixers(model)}
    assert kinds == {"gated_delta", "ple"}
    assert ple._unsloth_varlen_conv_hit and ple._unsloth_varlen_embedding_hit

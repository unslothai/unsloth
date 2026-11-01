# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Whole-step graph attention reads every row of its span, rows past the KV length masked. A masked
row still multiplies its V by 0 (NaN for inf / NaN), so a reused graph KV buffer must not carry an
earlier generation's rows into a later one. CPU: the real buffer helpers and attention, by path."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

MODELS = Path(__file__).resolve().parents[2] / "unsloth" / "models"


def _decode_graphs():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_decode_graphs_stale_tail", MODELS / "_decode_graphs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENABLED = True  # size buffers as the graph path does (replay itself is not run here)
    module.FULL = True
    return module


def _attention_functions():
    names = {"_decode_attention_core", "_decode_attention_span", "_gqa_decode_scores",
             "_gqa_decode_values"}  # fmt: skip
    tree = ast.parse((MODELS / "llama.py").read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = dict(
        torch = torch,
        torch_matmul = torch.matmul,
        torch_nn_functional_softmax = F.softmax,
        scaled_dot_product_attention = F.scaled_dot_product_attention,
        SDPA_HAS_GQA = True,
    )
    exec(compile(ast.Module(nodes, []), str(MODELS / "llama.py"), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_reused_buffer_does_not_leak_an_earlier_generation(dtype, bsz, bad):
    G, fns = _decode_graphs(), _attention_functions()
    attn = SimpleNamespace(
        config = SimpleNamespace(
            model_type = "llama", num_attention_heads = 2, num_key_value_heads = 1,
            sliding_window = None,
        ),
        num_key_value_groups = 2,
        head_dim = 2,
    )  # fmt: skip
    args = (bsz, 2, 1, 2, 4, dtype, torch.device("cpu"), 256)
    # An earlier, longer generation left a nonfinite K and V row (e.g. an fp16 overflow).
    k = torch.zeros(bsz, 1, 8, 2, dtype = dtype)
    v = torch.ones_like(k)
    k[:, :, 5, :] = bad
    v[:, :, 4, :] = bad
    G.alloc_decode_buffers(attn, k, v, *args)
    first = attn.paged_attention.data_ptr()
    # A new, short and finite generation reuses the same buffer.
    G.alloc_decode_buffers(attn, k[:, :, :2].nan_to_num(0.0), v[:, :, :2], *args)
    assert attn.paged_attention.data_ptr() == first

    kv_len, span = 2, 256
    keep = torch.zeros(bsz, 1, 1, span, dtype = torch.bool)
    keep[..., :kv_len] = True
    static = SimpleNamespace(span = span, keep = keep, drop = ~keep)
    q = torch.ones(bsz, 2, 1, 2, dtype = dtype)
    segmented = fns["_decode_attention_core"](attn, q.clone(), kv_len, None, bsz, True)[0].clone()
    whole_step = fns["_decode_attention_span"](attn, q.clone(), static, bsz, True).clone()
    assert torch.isfinite(segmented).all()
    assert torch.isfinite(whole_step).all()
    torch.testing.assert_close(whole_step, segmented)
    assert torch.isfinite(attn.paged_attention).all()


def test_segmented_only_keeps_buffer_untouched_past_the_prompt():
    # FULL off: only the segmented capture reads the cache, over the true length; no tail memset.
    G = _decode_graphs()
    G.FULL = False
    attn = SimpleNamespace(
        config = SimpleNamespace(model_type = "llama", num_attention_heads = 2,
                                 num_key_value_heads = 1, sliding_window = None),
        num_key_value_groups = 2,
        head_dim = 2,
    )  # fmt: skip
    args = (1, 2, 1, 2, 4, torch.float32, torch.device("cpu"), 256)
    G.alloc_decode_buffers(attn, torch.full((1, 1, 8, 2), 3.0), torch.full((1, 1, 8, 2), 3.0), *args)
    G.alloc_decode_buffers(attn, torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), *args)
    assert (attn.paged_attention_K[2:8] == 3.0).all()

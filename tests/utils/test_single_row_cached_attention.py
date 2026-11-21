# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import ast
import builtins
import importlib.util
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import pytest
import torch
from torch.nn import functional as F


def _linear(projection, inputs, out):
    return out.copy_(F.linear(inputs, projection.weight))


def _rms_norm(layer, inputs):
    return F.rms_norm(inputs, (inputs.shape[-1],), layer.weight, layer.variance_epsilon)


class _IdentityRope:
    def extend_rope_embedding(self, *args):
        pass

    def get_cached(self, seq_len, device):
        return torch.ones(seq_len, 2), torch.zeros(seq_len, 2)


def _decode_graphs_module(source):
    """The real decode buffer helpers, by file path: importing unsloth needs an accelerator. Its
    top level is torch and the standard library only. Graph replay is CUDA-only, so it is off."""
    spec = importlib.util.spec_from_file_location(
        "_unsloth_decode_graphs_cpu", source.with_name("_decode_graphs.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ENABLED = False
    return module


def _captured_attention_only(*args, **kwargs):
    raise AssertionError("a captured decode step ran on CPU; the harness keeps graph replay off")


@pytest.fixture(params = ["llama", "qwen3"])
def decoder(request):
    name = f"{request.param.title()}Attention_fast_forward_inference"
    source = Path(__file__).resolve().parents[2] / "unsloth" / "models" / f"{request.param}.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": Tuple,
        "KV_CACHE_INCREMENT": 256,
        "math_sqrt": sqrt,
        "fast_linear_forward": _linear,
        "fast_rms_layernorm_inference": _rms_norm,
        "torch_matmul": torch.matmul,
        "torch_nn_functional_softmax": F.softmax,
        "scaled_dot_product_attention": F.scaled_dot_product_attention,
        "SDPA_HAS_GQA": True,
        # The fused decode kernels are CUDA-only; on CPU they decline and the unfused path runs.
        "decode_rope_inplace": lambda *args, **kwargs: False,
        "decode_qk_norm_rope": lambda *args, **kwargs: None,
        # The cache and scratch buffers come from the real helpers; only the captured path is out.
        "_decode_graphs": _decode_graphs_module(source),
        "_decode_attention_core": _captured_attention_only,
        "_decode_attention_span": _captured_attention_only,
    }
    # The decode paths share llama.py's grouped KV helpers (qwen3 imports them from there).
    llama_source = source.with_name("llama.py")
    helpers = [
        node
        for node in ast.parse(llama_source.read_text(encoding = "utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_gqa_decode_scores", "_gqa_decode_values")
    ]
    exec(compile(ast.Module(body = helpers, type_ignores = []), str(llama_source), "exec"), namespace)
    bound = {n.arg for n in ast.walk(function) if isinstance(n, ast.arg)}
    bound |= {
        n.id
        for n in ast.walk(function)
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }
    free = {
        n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    missing = sorted(free - bound - set(namespace) - set(dir(builtins)))
    assert not missing, f"{name} now reads {missing}; give the harness a CPU stand-in for each"
    exec(compile(ast.Module(body = [function], type_ignores = []), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "mask_kind",
    [
        "padding",
        "boolean",
        "additive",
        "none",
    ],
)
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sliding_window", [None, 2])
def test_single_row_cached_attention_masks(decoder, mask_kind, groups, batch_size, sliding_window):
    attention = SimpleNamespace(
        config = SimpleNamespace(
            num_attention_heads = groups,
            num_key_value_heads = 1,
            hidden_size = 2,
            sliding_window = sliding_window,
        ),
        num_key_value_groups = groups,
        head_dim = 2,
        rotary_emb = _IdentityRope(),
        q_proj = SimpleNamespace(weight = torch.zeros(2 * groups, 2)),
        k_proj = SimpleNamespace(weight = torch.zeros(2, 2)),
        v_proj = SimpleNamespace(weight = torch.eye(2)),
        o_proj = SimpleNamespace(weight = torch.eye(2).repeat(1, groups) / groups),
        q_norm = SimpleNamespace(weight = torch.ones(2), variance_epsilon = 1e-6),
        k_norm = SimpleNamespace(weight = torch.ones(2), variance_epsilon = 1e-6),
    )
    keys = torch.zeros(batch_size, 1, 2, 2)
    values = torch.tensor([[[[100.0, 100.0], [2.0, 2.0]]]]).expand(batch_size, -1, -1, -1)
    keep_mask = torch.tensor([[[[False, True, True]]]]).expand(batch_size, -1, -1, -1)
    masks = {
        "padding": keep_mask[:, 0, 0, :].long(),
        "boolean": keep_mask,
        "additive": torch.tensor([[[[float("-inf"), 0.0, 0.0]]]]).expand(batch_size, -1, -1, -1),
        "none": None,
    }

    output, _ = decoder(
        attention,
        torch.zeros(batch_size, 1, 2),
        (keys, values),
        torch.ones(batch_size, 1, dtype = torch.long),
        do_prefill = True,
        attention_mask = masks[mask_kind],
    )

    # The window of 2 drops the padded slot, so every mask agrees there.
    expected = 34.0 if mask_kind == "none" and sliding_window is None else 1.0
    torch.testing.assert_close(output, torch.full((batch_size, 1, 2), expected))

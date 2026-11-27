# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Mode restores that only wrote `.training` skipped train() overrides that sync state: Int4PackedLinear
keeps quant_state.training (which int4_matmul reads to allow its fused inference kernels), and a
PreTrainedModel with use_kernels re-kernelizes. CPU, real Int4PackedLinear shells (no forward)."""

import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


def _int4_linear(n = 8, k = 16):
    from unsloth.models.compressed_tensors_int4 import Int4PackedLinear, make_int4_packed_linear

    scheme = types.SimpleNamespace(weights = types.SimpleNamespace(num_bits = 4, group_size = k))
    shapes = {"weight_packed": ((n, k // 8), torch.int32), "weight_scale": ((n, 1), torch.bfloat16)}
    module = make_int4_packed_linear(
        nn.Linear(k, n, bias = False), scheme, shapes, torch.bfloat16, device = "cpu"
    )
    assert isinstance(module, Int4PackedLinear)
    return module


def _in_sync(module):
    return module.quant_state.training == module.training


class _LoraLike(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        self.merged_adapters = []

    def get_base_layer(self):
        return self.base_layer

    def merge(self):
        pass


@pytest.mark.parametrize("training", [True, False])
def test_vllm_out_of_place_merge_keeps_int4_quant_state_in_sync(training):
    from unsloth.models.rl_replacements import _unsloth_vllm_peft_merge_out_of_place

    base = _int4_linear()
    layer = _LoraLike(base).train(training)
    assert _in_sync(base)
    ptr = base.weight_packed.data_ptr()
    _unsloth_vllm_peft_merge_out_of_place(base, [layer])
    assert base.training is training and layer.training is training
    assert base.weight_packed.data_ptr() == ptr  # the live bytes are restored
    assert _in_sync(
        base
    ), f"quant_state.training {base.quant_state.training}, module {base.training}"


class _Card:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def run_usage_snippet(self, *args, **kwargs):
        self.calls += 1
        self.model.eval()  # what sentence-transformers' encode() does


@pytest.mark.parametrize("training", [True, False])
def test_fsdp2_st_usage_snippet_keeps_int4_quant_state_in_sync(training):
    from unsloth.models.sentence_transformer import fsdp2_st_usage_snippet

    model = nn.Sequential(nn.Linear(16, 16), _int4_linear(8, 16))
    model.model_card_data = _Card(model)
    model.train(training)
    int4 = model[1]
    assert _in_sync(int4)  # builds the quant_state, as the first forward does
    with fsdp2_st_usage_snippet(model):
        pass
    assert model.model_card_data.calls == 1
    assert model.training is training and int4.training is training
    assert _in_sync(
        int4
    ), f"quant_state.training {int4.quant_state.training}, module {int4.training}"

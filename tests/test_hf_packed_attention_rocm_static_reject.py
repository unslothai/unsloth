# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ROCm, varlen refused inside the compiled graph: a packed layer the eager varlen wrapper rejects anyway
(sinks, logit softcapping, head_dim not a multiple of 8, non-causal, attention dropout) keeps the dense path
inside the block. Leaving the block for it only added a graph break before the same dense attention."""

import os
import types

import pytest
import torch

import unsloth.utils.hf_packed_attention as hpa

# Each checkout tests its own tree, not an installed unsloth.
assert os.path.realpath(hpa.__file__).startswith(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep
), hpa.__file__


@pytest.fixture
def rocm_refusing(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.13", raising = False)
    monkeypatch.setattr(hpa, "_ORIG_SDPA", [lambda *args, **kwargs: None])
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(hpa, "_torch_varlen", lambda device: (lambda *args, **kwargs: args[0]))
    monkeypatch.setattr(hpa, "_hip_window_ok", lambda device, window: True)
    monkeypatch.setattr(hpa, "_sdpa_policy_requested", lambda *args: False)
    monkeypatch.setattr(hpa, "_varlen_in_graph", lambda *args: False)
    monkeypatch.setattr(hpa, "_BLOCK_PREP", {})


def _prepare(
    head_dim = 16,
    config_extra = None,
    **attention_extra,
):
    config = types.SimpleNamespace(
        _attn_implementation = "sdpa",
        num_attention_heads = 4,
        num_key_value_heads = 4,
        **(config_extra or {}),
    )
    attention = types.SimpleNamespace(
        config = config, head_dim = head_dim, num_key_value_groups = 1, **attention_extra
    )
    mask = torch.ones(1, 1, 6, 6, dtype = torch.bool).tril()
    mask[0, 0, 3:, :3] = False
    lengths = torch.tensor([3, 3], dtype = torch.int32)
    hidden = torch.randn(1, 6, 8, dtype = torch.bfloat16)
    layer = types.SimpleNamespace(self_attn = attention)
    return hpa.block_varlen_kwargs(
        layer, hidden, {"packed_seq_lengths": lengths, "attention_mask": mask}
    )


@pytest.mark.parametrize(
    "head_dim, config_extra, attention_extra",
    [
        (16, None, {"sinks": torch.zeros(4)}),
        (16, {"attn_logit_softcapping": 50.0}, {}),
        (20, None, {}),
        (16, None, {"is_causal": False}),
        (16, None, {"training": True, "attention_dropout": 0.1}),
    ],
    ids = ["sinks", "softcap", "head_dim_20", "non_causal", "dropout"],
)
def test_layers_eager_varlen_rejects_stay_in_the_block(
    rocm_refusing, head_dim, config_extra, attention_extra
):
    assert _prepare(head_dim, config_extra, **attention_extra) is None


def test_eligible_layers_still_leave_the_block(rocm_refusing):
    assert _prepare() == hpa._LEAVE_BLOCK
    assert _prepare(training = True, attention_dropout = 0.0) == hpa._LEAVE_BLOCK
    assert _prepare(training = False, attention_dropout = 0.1) == hpa._LEAVE_BLOCK

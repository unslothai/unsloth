# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fitter arm places a hybrid's cache on the rows that hold it, and a layout whose KV head
count varies per row is not split across devices on an even spread."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference import offload_layout as OL  # noqa: E402
from core.inference.offload_planner import (  # noqa: E402
    HostProfile,
    PlanOptions,
    plan_placement,
    plan_to_args,
)
from test_offload_planner_seam import _inputs, _Stub  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB


class _F:
    def __init__(self, v):
        self.v = v

    def contents(self):
        return self.v


class _T:
    def __init__(self, name, n):
        self.name = name
        self.n_bytes = n


class _Reader:
    def __init__(self, fields, tensors):
        self.fields = {k: _F(v) for k, v in fields.items()}
        self.tensors = tuple(tensors)


# ------------------------------------------------ finding 1: a hybrid's cache in the fitter arm

N_HYBRID = 64
CTX = 8192
# 4 heads x (256 + 256) x 2 bytes on each of 16 attention rows: 512 MiB at 8192.
ATTN_BYTES = 512 * MIB
# 48 recurrent rows x (3 x (6144 + 2 x 16 x 128) + 128 x 6144) x 4 bytes: 149.625 MiB.
STATE_BYTES = 48 * (3 * (6144 + 2 * 16 * 128) + 128 * 6144) * 4


def _qwen35_reader(**extra):
    """A dense Qwen3.5-shaped hybrid: three recurrent rows, then one attention row (llama.cpp's
    default interval of 4, models/qwen35.cpp), 0.20 GiB of FFN and 0.045 GiB of the rest a row."""
    fields = {
        "general.architecture": "qwen35",
        "qwen35.block_count": N_HYBRID,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.embedding_length": 4096,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.context_length": 262144,
        "qwen35.ssm.inner_size": 6144,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.conv_kernel": 4,
        **extra,
    }
    tensors = [
        _T("token_embd.weight", GIB),
        _T("output.weight", GIB),
        _T("output_norm.weight", int(0.01 * GIB)),
    ]
    for i in range(N_HYBRID):
        tensors += [
            _T(f"blk.{i}.ffn_up.weight", int(0.20 * GIB)),
            _T(f"blk.{i}.attn_qkv.weight", int(0.045 * GIB)),
        ]
    return _Reader(fields, tensors)


def _kv_at(ctx, slots):
    return ATTN_BYTES * ctx // CTX * slots + STATE_BYTES * slots


def _gate(
    layout,
    card_mib,
    weights = (),
):
    return plan_placement(
        layout,
        [card_mib * MIB],
        64 * GIB,
        CTX,
        kv_layer_weights = list(weights),
        opts = PlanOptions(
            require_cost_win = True,
            host = HostProfile(threads = 6),
            kv_bytes_at = _kv_at,
        ),
    )


_EXACT_ROWS = [0, 0, 0, 1] * (N_HYBRID // 4)


def test_the_layout_names_a_hybrids_attention_rows():
    layout = OL._layout_from_reader(_qwen35_reader())
    assert layout.complete
    assert layout.n_attention_layers == 16
    assert layout.kv_bytes(CTX) == ATTN_BYTES
    assert layout.recurrent_bytes == STATE_BYTES
    assert layout.kv_heads_by_layer == tuple(4 * w for w in _EXACT_ROWS)
    assert _openelm_layout().kv_heads_by_layer == tuple(_OPENELM_KV)


def test_the_fitter_arm_does_not_move_cache_off_a_hybrids_recurrent_rows():
    """The reviewer's cell: 18528 MiB, six threads. Spread evenly, the fitter's first three moved
    rows carried 24 MiB of attention cache they do not hold, priced at the host cache rate, and the
    gate accepted a 0.80 GiB spill that the real row map declines."""
    layout = OL._layout_from_reader(_qwen35_reader())
    seam = _gate(layout, 18528)
    exact = _gate(layout, 18528, _EXACT_ROWS)
    assert exact.declined_by_gate and not exact.spills_anything, exact.reason
    assert seam.declined_by_gate and not seam.spills_anything, seam.reason
    assert seam.predicted_fit_request_ms == pytest.approx(exact.predicted_fit_request_ms)


@pytest.mark.parametrize("card_mib", [16384, 17408, 18000, 18528, 19000, 19456])
def test_the_seams_empty_vector_now_prices_what_the_row_map_prices(card_mib):
    layout = OL._layout_from_reader(_qwen35_reader())
    seam, exact = _gate(layout, card_mib), _gate(layout, card_mib, _EXACT_ROWS)
    assert (seam.spills_anything, seam.declined_by_gate) == (
        exact.spills_anything,
        exact.declined_by_gate,
    )
    assert (seam.predicted_fit_request_ms or 0) == pytest.approx(
        exact.predicted_fit_request_ms or 0
    )


@pytest.mark.parametrize("card_mib", [16384, 17408, 18528, 19000])
def test_a_dense_layout_is_priced_as_before(card_mib):
    """Control: equal heads on every row leave no map, so the even spread is exact."""
    reader = _qwen35_reader()
    for key in [k for k in reader.fields if ".ssm." in k]:
        del reader.fields[key]
    reader.fields["general.architecture"] = _F("qwen3")
    reader.fields = {k.replace("qwen35.", "qwen3."): v for k, v in reader.fields.items()}
    layout = OL._layout_from_reader(reader)
    assert layout.complete and not getattr(layout, "kv_heads_by_layer", ())
    even = [1] * N_HYBRID
    seam, explicit = _gate(layout, card_mib), _gate(layout, card_mib, even)
    assert (seam.spills_anything, seam.declined_by_gate, seam.predicted_fit_request_ms) == (
        explicit.spills_anything,
        explicit.declined_by_gate,
        explicit.predicted_fit_request_ms,
    )


def test_the_row_map_follows_the_files_own_markers():
    assert OL.layer_kv_heads("qwen35", 8, n_kv_head = 2) == (0, 0, 0, 2, 0, 0, 0, 2)
    assert OL.layer_kv_heads("qwen35", 6, n_kv_head = 2, full_attention_interval = 3) == (
        0,
        0,
        2,
        0,
        0,
        2,
    )
    mask = [True, False, True, True, False, True]
    assert OL.layer_kv_heads("qwen35", 6, n_kv_head = 2, recurrent_layers = mask) == (
        0,
        2,
        0,
        0,
        2,
        0,
    )
    # Zeros in the head list beat the interval, as in hybrid_layer_split.
    assert OL.layer_kv_heads("qwen35", 4, n_kv_head = [0, 2, 2, 0]) == (0, 2, 2, 0)
    assert OL.layer_kv_heads("llama", 4, n_kv_head = 8) == ()
    assert OL.layer_kv_heads("openelm", 4, n_kv_head = [3, 3, 6, 6]) == (3, 3, 6, 6)


# ------------------------------------------------ finding 2: unequal KV heads across devices

# apple/OpenELM-3B-Instruct config.json: 36 rows, model_dim 3072, head_dim 128.
_OPENELM_KV = [3] * 4 + [4] * 14 + [5] * 12 + [6] * 6
_OPENELM_Q = [12] * 4 + [16] * 14 + [20] * 12 + [24] * 6
_OPENELM_FFN = [int((m * 3072 + 255) // 256 * 256) for m in [0.5 + 0.1 * i for i in range(36)]]


def _openelm_layout(kv_heads = _OPENELM_KV):
    n, d, hd = 36, 3072, 128
    fields = {
        "general.architecture": "openelm",
        "openelm.block_count": n,
        "openelm.attention.head_count": list(_OPENELM_Q),
        "openelm.attention.head_count_kv": list(kv_heads),
        "openelm.embedding_length": d,
        "openelm.attention.key_length": hd,
        "openelm.attention.value_length": hd,
        "openelm.context_length": 2048,
        "openelm.feed_forward_length": list(_OPENELM_FFN),
    }
    tensors = [_T("token_embd.weight", 32000 * d * 2), _T("output_norm.weight", d * 4)]
    for i in range(n):
        tensors += [
            _T(f"blk.{i}.attn_qkv.weight", (_OPENELM_Q[i] + 2 * kv_heads[i]) * hd * d * 2),
            _T(f"blk.{i}.attn_output.weight", _OPENELM_Q[i] * hd * d * 2),
            _T(f"blk.{i}.ffn_up.weight", 2 * _OPENELM_FFN[i] * d * 2),
            _T(f"blk.{i}.ffn_down.weight", _OPENELM_FFN[i] * d * 2),
            _T(f"blk.{i}.attn_norm.weight", d * 4),
        ]
    return OL._layout_from_reader(_Reader(fields, tensors))


def _spill(
    layout,
    gpus,
    ctx = 32768,
    weights = None,
):
    class _S(_Stub):
        def _tensor_spill_layout(
            self,
            model_path,
            *,
            all_shards = False,
        ):
            return layout

    inputs = _inputs(
        model_size = layout.tensor_bytes,
        kv = layout.kv_bytes_per_token_f16 * ctx,
        gpus = gpus,
        n_ctx = ctx,
        n_threads = 6,
    )
    if weights is not None:
        inputs["kv_layer_weights"] = list(weights)
    return _S(avail_mib = 64 * 1024)._planned_tensor_spill(
        inputs, extra_args = None, env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    )


def _approved(plan):
    return plan is not None and plan.changed and not plan.declined_by_gate


@pytest.mark.parametrize("per_gpu_mib", [6100, 6200, 6300])
def test_an_unequal_head_layout_is_not_split_on_an_even_cache_spread(per_gpu_mib):
    """OpenELM-3B at 32768 on two cards: the even spread booked device 1 about 200 MiB short of
    the rows 18-35 it owns (5 and 6 heads against an average of 4.6) and pinned --fit off over it.
    The real per-layer vector declines the same cell."""
    layout = _openelm_layout()
    assert layout.complete
    gpus = [(0, per_gpu_mib), (1, per_gpu_mib)]
    assert not _approved(_spill(layout, gpus, weights = _OPENELM_KV))
    plan = _spill(layout, gpus)
    assert not _approved(plan), plan_to_args(plan)


@pytest.mark.parametrize("per_gpu_mib", [6100, 7000])
def test_equal_heads_on_two_cards_still_split(per_gpu_mib):
    """Control: the same shape with four heads on every row has nothing to place unevenly."""
    layout = _openelm_layout([4] * 36)
    assert layout.complete and not getattr(layout, "kv_heads_by_layer", ())
    plan = _spill(layout, [(0, per_gpu_mib), (1, per_gpu_mib)])
    assert _approved(plan) and "--tensor-split" in plan_to_args(plan), plan and plan.reason


def test_unequal_heads_on_one_card_still_plan():
    """Control: one device has no split to get wrong, so the total alone decides."""
    plan = _spill(_openelm_layout(), [(0, 12000)])
    assert _approved(plan), plan and plan.reason


def test_unequal_heads_with_a_supplied_vector_still_split():
    """A caller vector says where the wide rows are, so a roomy pair still splits."""
    plan = _spill(_openelm_layout(), [(0, 7000), (1, 7000)], weights = _OPENELM_KV)
    assert _approved(plan) and "--tensor-split" in plan_to_args(plan), plan and plan.reason

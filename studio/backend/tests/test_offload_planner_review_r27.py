# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fitter arm puts a Nemotron-H model's Mamba state on its Mamba rows only.

nemotron_h has three kinds of row: attention (KV heads, no FFN), Mamba (no KV heads, no FFN) and
MLP-only (no KV heads, an FFN). llama.cpp marks a row recurrent only when both are zero
(models/nemotron-h.cpp), so the state lives on the Mamba rows. The cache-free rows the fitter arm
used also took in the MLP rows, so a moved prefix carried the wrong share of the state and the
fitter's prefix landed one row off either way.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference import offload_layout as OL  # noqa: E402
from core.inference import offload_planner as OP  # noqa: E402
from core.inference.offload_planner import HostProfile, PlanOptions, plan_placement  # noqa: E402
from test_offload_planner_seam import _inputs, _Stub  # noqa: E402


@pytest.fixture(autouse = True)
def _six_physical_cores(monkeypatch):
    """These cells ask for -t 6; a CI runner with fewer physical cores would decline them on
    the thread check, not on the placement under test."""
    from core.inference import llama_cpp as _lc
    monkeypatch.setattr(
        _lc, "_spilled_decode_threads", lambda requested = None, extra_args = None: requested or 6
    )


MIB = 1024 * 1024
GIB = 1024 * MIB
Q4 = 0.5625

NANO9B = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
MAMBA_ROWS = tuple(i for i, c in enumerate(NANO9B) if c == "M")


class _F:
    def __init__(self, v):
        self.v = v

    def contents(self):
        return self.v


class _T:
    def __init__(self, name, n):
        self.name = name
        self.n_bytes = int(n)


class _Reader:
    def __init__(self, fields, tensors):
        self.fields = {k: _F(v) for k, v in fields.items()}
        self.tensors = tuple(tensors)


def _nemotron_h_layout(
    pattern = NANO9B,
    d = 4480,
    ff = 15680,
    inner = 10240,
    kvh = 8,
    qh = 40,
):
    arch, hd, state, groups = "nemotron_h", 128, 128, 8
    fields = {
        "general.architecture": arch,
        f"{arch}.block_count": len(pattern),
        f"{arch}.attention.head_count_kv": [kvh if c == "*" else 0 for c in pattern],
        f"{arch}.attention.head_count": qh,
        f"{arch}.embedding_length": d,
        f"{arch}.attention.key_length": hd,
        f"{arch}.attention.value_length": hd,
        f"{arch}.feed_forward_length": [ff if c == "-" else 0 for c in pattern],
        f"{arch}.context_length": 131072,
        f"{arch}.ssm.inner_size": inner,
        f"{arch}.ssm.state_size": state,
        f"{arch}.ssm.group_count": groups,
        f"{arch}.ssm.conv_kernel": 4,
    }
    tensors = [_T("token_embd.weight", 131072 * d * Q4), _T("output.weight", 131072 * d * Q4)]
    for i, c in enumerate(pattern):
        if c == "M":
            tensors += [
                _T(f"blk.{i}.ssm_in.weight", d * (2 * inner + 2 * groups * state + 128) * Q4),
                _T(f"blk.{i}.ssm_out.weight", inner * d * Q4),
            ]
        elif c == "*":
            tensors += [
                _T(f"blk.{i}.attn_q.weight", d * qh * hd * Q4),
                _T(f"blk.{i}.attn_k.weight", d * kvh * hd * Q4),
                _T(f"blk.{i}.attn_v.weight", d * kvh * hd * Q4),
                _T(f"blk.{i}.attn_output.weight", d * qh * hd * Q4),
            ]
        else:
            tensors += [
                _T(f"blk.{i}.ffn_up.weight", d * ff * Q4),
                _T(f"blk.{i}.ffn_down.weight", d * ff * Q4),
            ]
        tensors.append(_T(f"blk.{i}.attn_norm.weight", d * 4))
    return OL._layout_from_reader(_Reader(fields, tensors))


def _qwen35_layout(n = 64):
    fields = {
        "general.architecture": "qwen35",
        "qwen35.block_count": n,
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
    }
    tensors = [_T("token_embd.weight", GIB), _T("output.weight", GIB)]
    for i in range(n):
        tensors += [
            _T(f"blk.{i}.ffn_up.weight", 0.20 * GIB),
            _T(f"blk.{i}.attn_qkv.weight", 0.045 * GIB),
        ]
    return OL._layout_from_reader(_Reader(fields, tensors))


def _opts(layout, slots = 1):
    attn, state = layout.kv_bytes_per_token_f16, layout.recurrent_bytes
    return PlanOptions(
        require_cost_win = True,
        host = HostProfile(threads = 6),
        kv_bytes_at = lambda c, s: attn * c * s + state * s,
        n_parallel = slots,
    )


def _approved(plan):
    return plan is not None and plan.changed and not plan.declined_by_gate


def test_the_layout_names_nemotron_h_mamba_rows():
    layout = _nemotron_h_layout()
    assert layout.complete
    assert sum(1 for h in layout.kv_heads_by_layer if h == 0) == 52
    assert layout.recurrent_rows == MAMBA_ROWS
    assert len(MAMBA_ROWS) == 27


def test_both_planner_sites_read_the_mamba_rows():
    """_fit_fallback_placement and _per_device_usage share the one row set."""
    layout = _nemotron_h_layout()
    assert OP._recurrent_rows(layout, list(layout.kv_heads_by_layer)) == set(MAMBA_ROWS)


@pytest.mark.parametrize("card_mib", [4400, 4800, 5214, 5534, 6000])
def test_the_fitter_arm_carries_the_mamba_share_of_its_prefix(card_mib):
    """Spread over the 52 cache-free rows, a 'M-M-' prefix carried 4/52 of the state where it
    holds 2/27, so the fitter's prefix was sized off a share it does not move."""
    layout = _nemotron_h_layout()
    ctx = 8192
    opts = _opts(layout)
    floor = opts.kv_bytes_at(ctx, 1)
    budget = OP._usable_vram([card_mib * MIB], opts, ctx)
    arm = OP._fit_fallback_placement(
        layout, opts, budget, ctx, quantised = False, kv_bytes_floor = floor, kv_on_host = False
    )
    assert arm is not None and arm.host_groups
    moved_layers = next(g.bytes_total for g in arm.host_groups if g.name == "layers")
    running, moved = 0, 0
    for block in layout.blocks:
        if running >= moved_layers:
            break
        running += block.resident_bytes
        moved += 1
    assert running == moved_layers
    mamba_moved = sum(1 for b in layout.blocks[:moved] if b.index in MAMBA_ROWS)
    state = next(g.bytes_total for g in arm.host_groups if g.name.startswith("recurrent"))
    assert state == layout.recurrent_bytes * mamba_moved // len(MAMBA_ROWS)


def test_a_spill_that_the_mamba_rows_make_slower_than_the_fitter_is_declined():
    """Nano-9B-v2 at 8192 on 5214 MiB: the cache-free spread under-credited the prefix by 1.4 MiB
    of state, so the fitter arm moved one Mamba row more than llama.cpp needs, scored 12502 ms
    and the gate approved an 11093 ms spill. On the real rows the fitter scores 11606 ms."""
    layout = _nemotron_h_layout()
    plan = plan_placement(layout, [5214 * MIB], 512 * GIB, 8192, opts = _opts(layout))
    assert plan.declined_by_gate and not plan.spills_anything, plan.reason


def test_the_seam_approves_a_spill_the_fitter_needs_one_more_mamba_row_for():
    """Through the seam at 4400 MiB the spread over-credited 'M-M-' by 0.4 MiB, scored the
    fitter one row short at 3560 ms and declined a 3566 ms spill. The fitter really moves 'M-M-M'
    and pays 4494 ms."""
    layout = _nemotron_h_layout()

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
        kv = layout.kv_bytes_per_token_f16 * 8192,
        gpus = [(0, 4400)],
        n_ctx = 8192,
        n_threads = 6,
    )
    plan = _S(avail_mib = 64 * 1024)._planned_tensor_spill(
        inputs, extra_args = None, env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    )
    assert _approved(plan), plan and plan.reason


@pytest.mark.parametrize("card_mib", [16384, 17408, 18528, 19456])
def test_a_hybrid_whose_cache_free_rows_are_its_recurrent_rows_is_priced_as_before(card_mib):
    """Control (Qwen3.5 / Qwen3-Next / Flash-Next shape): every cache-free row is recurrent, so
    the file's row set changes nothing."""
    layout = _qwen35_layout()
    rows = getattr(layout, "recurrent_rows", None)
    assert rows is None or rows == tuple(
        i for i, h in enumerate(layout.kv_heads_by_layer) if h == 0
    )
    unmapped = dataclasses.replace(layout, recurrent_rows = ()) if rows is not None else layout
    a = plan_placement(layout, [card_mib * MIB], 64 * GIB, 8192, opts = _opts(layout))
    b = plan_placement(unmapped, [card_mib * MIB], 64 * GIB, 8192, opts = _opts(unmapped))
    assert (a.spills_anything, a.declined_by_gate, a.predicted_fit_request_ms) == (
        b.spills_anything,
        b.declined_by_gate,
        b.predicted_fit_request_ms,
    )


def test_a_layout_without_a_row_map_has_no_recurrent_rows():
    """Control: a dense file and a hybrid the reader cannot map leave the fallback in charge."""
    from test_offload_planner_review_r26 import _qwen35_reader

    reader = _qwen35_reader()
    for key in [k for k in reader.fields if ".ssm." in k]:
        del reader.fields[key]
    reader.fields["general.architecture"] = _F("qwen3")
    reader.fields = {k.replace("qwen35.", "qwen3."): v for k, v in reader.fields.items()}
    assert getattr(OL._layout_from_reader(reader), "recurrent_rows", ()) == ()

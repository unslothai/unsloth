# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where llama.cpp puts each weight, as the memory estimate prices it (#19795).

The rules mirror llama.cpp: the first matching tensor override wins (llama_model_loader), layers
are offloaded from the back with the input layer on the CPU (llama_model::load_tensors), and
common_params_parse reads inherited env before argv."""

import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.llama_cpp import _main_tensor_overrides  # noqa: E402
from core.inference.offload_layout import (  # noqa: E402
    FFN_DENSE_OVERRIDE,
    FFN_EXPS_OVERRIDE,
    block_ffn_overrides,
    place_weights,
)

EXPS = 1000
ATTN = 10
EMBD = 300


def _moe(
    n_blocks = 4,
    tied = False,
    scales = False,
):
    table = {"token_embd.weight": EMBD, "output_norm.weight": 1}
    if not tied:
        table["output.weight"] = EMBD
    for b in range(n_blocks):
        table[f"blk.{b}.attn_q.weight"] = ATTN
        table[f"blk.{b}.ffn_gate_up_exps.weight"] = 2 * EXPS
        table[f"blk.{b}.ffn_down_exps.weight"] = EXPS
        if scales:
            table[f"blk.{b}.ffn_gate_up_exps.scale"] = 1
            table[f"blk.{b}.ffn_down_exps.scale"] = 1
    return table


def test_the_llama_cpp_regexes_are_copied_verbatim():
    assert FFN_EXPS_OVERRIDE == "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
    assert FFN_DENSE_OVERRIDE == "\\.ffn_(up|down|gate)\\."
    assert block_ffn_overrides(2, FFN_EXPS_OVERRIDE) == [
        "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps",
        "blk\\.1\\.ffn_(up|down|gate|gate_up)_(ch|)exps",
    ]


def test_every_layer_offloaded_keeps_only_the_input_layer_on_the_host():
    p = place_weights(_moe(), None)
    assert p.host_bytes == EMBD
    assert p.device_bytes == sum(_moe().values()) - EMBD
    assert p.host_staging_bytes == 0
    assert p.resolved


def test_cpu_experts_leave_the_gpu_and_one_layer_of_them_is_staged():
    p = place_weights(_moe(), None, [("blk\\.[0-1]\\.ffn_.*_exps", True)])
    assert p.host_bytes == EMBD + 2 * 3 * EXPS
    assert p.device_bytes == sum(_moe().values()) - EMBD - 2 * 3 * EXPS
    assert p.host_staging_bytes == 3 * EXPS


def test_a_head_held_on_the_host_is_reported_apart_from_the_blocks():
    p = place_weights(_moe(), None, [("output\\.weight", True)])
    assert (p.host_head_bytes, p.host_staging_bytes) == (EMBD, 0)
    tied = place_weights(_moe(tied = True), None, [("token_embd\\.weight", True)])
    assert tied.host_head_bytes == EMBD
    assert place_weights(_moe(), 0, [("output\\.weight", True)]).host_head_bytes == 0


def test_n_cpu_moe_places_what_the_equivalent_override_places():
    count = place_weights(
        _moe(), None, [(p, True) for p in block_ffn_overrides(2, FFN_EXPS_OVERRIDE)]
    )
    pattern = place_weights(_moe(), None, [("blk\\.[0-1]\\.ffn_.*_exps", True)])
    assert count == pattern


def test_layers_are_offloaded_from_the_back_with_the_output_layer_first():
    blocks = 4
    per_block = ATTN + 3 * EXPS
    p = place_weights(_moe(blocks), 1)
    assert p.device_bytes == EMBD + 1
    p = place_weights(_moe(blocks), 2)
    assert p.device_bytes == EMBD + 1 + per_block
    assert p.host_staging_bytes == per_block
    assert place_weights(_moe(blocks), blocks + 1) == place_weights(_moe(blocks), None)
    assert place_weights(_moe(blocks), 0).device_bytes == 0


def test_the_first_matching_override_wins():
    device_first = place_weights(_moe(), None, [("exps", False), ("blk\\.0\\.", True)])
    assert device_first.host_bytes == EMBD + ATTN
    host_first = place_weights(_moe(), None, [("blk\\.0\\.", True), ("exps", False)])
    assert host_first.host_bytes == EMBD + ATTN + 3 * EXPS


def test_an_override_reaches_a_layer_already_on_the_cpu():
    # ngl 2: blocks 0-2 on the CPU; the override moves block 0's attention onto a device.
    base = place_weights(_moe(), 2)
    moved = place_weights(_moe(), 2, [("blk\\.0\\.attn_", False)])
    assert moved.device_bytes == base.device_bytes + ATTN
    assert place_weights(_moe(), 2, [("blk\\.0\\.", True)]) == base


def test_a_tied_head_loads_token_embd_again_on_the_output_device():
    tied = _moe(tied = True)
    p = place_weights(tied, None)
    assert p.host_bytes == EMBD
    assert p.device_bytes == sum(tied.values())
    assert p.duplicate_bytes == EMBD
    shared = place_weights(tied, 0)
    assert (shared.device_bytes, shared.host_bytes, shared.duplicate_bytes) == (
        0,
        sum(tied.values()),
        0,
    )


def test_a_tied_head_moved_with_token_embd_shares_its_allocation():
    tied = _moe(tied = True)
    p = place_weights(tied, None, [("token_embd\\.weight", False)])
    assert p.host_bytes == 0
    assert p.device_bytes == sum(tied.values())
    assert p.duplicate_bytes == 0


def test_an_encoder_has_no_head_to_tie():
    encoder = _moe(tied = True)
    p = place_weights(encoder, None, architecture = "bert")
    assert p.device_bytes == sum(encoder.values()) - EMBD
    assert p.duplicate_bytes == 0


def test_ranks_follow_the_blocks_present_not_their_indices():
    table = {"blk.0.attn_q.weight": 1, "blk.7.attn_q.weight": 10, "output.weight": 100}
    assert place_weights(table, 2).device_bytes == 110
    assert place_weights(table, 1).device_bytes == 100


def test_lookup_tables_are_left_to_their_own_pricing():
    table = {**_moe(), "per_layer_token_embd.weight": 10**9}
    assert place_weights(table, None) == place_weights(_moe(), None)


def test_posix_classes_read_as_std_regex_reads_them():
    assert place_weights(_moe(), 0, [("blk\\.[[:alpha:]]\\.attn_", False)]).device_bytes == 0
    p = place_weights(_moe(), None, [("blk\\.[[:digit:]]\\.ffn_gate_up", True)])
    assert p.resolved
    assert p.host_bytes == EMBD + 4 * 2 * EXPS
    moved = place_weights(_moe(), 0, [("blk\\.[[:digit:]]\\.attn_", False)])
    assert moved.device_bytes == 4 * ATTN


@pytest.mark.parametrize("cls", ["alnum", "d", "digit", "graph", "print", "w", "xdigit"])
def test_every_class_std_regex_knows_is_read(cls):
    p = place_weights(_moe(), 0, [(f"blk\\.[[:{cls}:]]\\.attn_", False)])
    assert p.resolved
    assert p.device_bytes == 4 * ATTN


@pytest.mark.parametrize("on_host", [True, False], ids = ["cpu", "device"])
def test_an_unreadable_pattern_leaves_the_placement_unresolved_either_way(on_host):
    p = place_weights(_moe(), None, [("blk(", on_host)])
    assert not p.resolved
    assert p.host_bytes == EMBD


def test_the_moe_cache_can_hold_the_host_experts_of_gpu_layers_only():
    p = place_weights(_moe(), 3, [("blk\\.[0-2]\\.ffn_.*_exps", True)])
    assert p.host_expert_bytes == 3 * EXPS
    partly = place_weights(_moe(), None, [("blk\\.0\\.ffn_gate_up_exps", True)])
    assert partly.host_expert_bytes == 0


def test_expert_scales_left_on_the_device_do_not_stop_the_cache():
    weights_only = place_weights(
        _moe(scales = True), None, [("blk\\.0\\.ffn_.*_exps\\.weight", True)]
    )
    assert weights_only.host_expert_bytes == 3 * EXPS


def test_no_block_tensor_means_no_placement():
    assert place_weights({}, None) is None
    assert place_weights({"token_embd.weight": 5}, None) is None


class TestOverrides:
    def test_env_comes_first_in_option_order_then_the_studio_count_then_argv(self):
        env = {
            "LLAMA_ARG_N_CPU_FFN": "1",
            "LLAMA_ARG_N_CPU_MOE": "1",
            "LLAMA_ARG_CPU_MOE": "1",
            "LLAMA_ARG_OVERRIDE_TENSOR": "env=CUDA0",
        }
        got = _main_tensor_overrides(["-ot", "argv=CPU"], env, n_cpu_moe = 1)
        assert got == [
            ("env", False),
            (FFN_EXPS_OVERRIDE, True),
            ("blk\\.0" + FFN_EXPS_OVERRIDE, True),
            ("blk\\.0" + FFN_DENSE_OVERRIDE, True),
            ("blk\\.0" + FFN_EXPS_OVERRIDE, True),
            ("argv", True),
        ]

    def test_argv_spellings_and_value_lists(self):
        got = _main_tensor_overrides(
            ["--override-tensor=a=CPU,b=CUDA0", "--override-tensor", "c=CUDA_Host", "-cmoe"], {}
        )
        assert got == [("a", True), ("b", False), ("c", True), (FFN_EXPS_OVERRIDE, True)]

    def test_an_empty_pattern_is_kept_and_matches_every_tensor(self):
        got = _main_tensor_overrides(["-ot", "=CPU"], {})
        assert got == [("", True)]
        p = place_weights(_moe(), None, got)
        assert (p.device_bytes, p.host_bytes) == (0, sum(_moe().values()))

    def test_a_count_names_no_more_blocks_than_llama_cpp_allows(self):
        assert len(_main_tensor_overrides(["-ncmoe", "1000000000"], {})) == 512

    def test_counts_and_flags_that_place_nothing(self):
        assert _main_tensor_overrides(["-ncmoe", "0", "--n-cpu-ffn=0", "--top-k", "40"], {}) == []
        assert _main_tensor_overrides(["-otd", "exps=CPU", "-cmoed", "-ncmoed", "3"], {}) == []
        assert _main_tensor_overrides(["--n-cpu-moe=2"], {}) == [
            (p, True) for p in block_ffn_overrides(2, FFN_EXPS_OVERRIDE)
        ]

    def test_os_environ_is_the_default(self, monkeypatch):
        monkeypatch.setenv("LLAMA_ARG_OVERRIDE_TENSOR", "from_env=CPU")
        assert _main_tensor_overrides([]) == [("from_env", True)]
        monkeypatch.delenv("LLAMA_ARG_OVERRIDE_TENSOR")
        assert "LLAMA_ARG_OVERRIDE_TENSOR" not in os.environ
        assert _main_tensor_overrides([]) == []

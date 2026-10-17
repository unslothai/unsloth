# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A plan the planner declines launches the flag-off argv: the spill micro-batch raise and the
MoE cache admission read the same resident expert share either way. And the share those two gates
read keeps the KV basis their bands were measured on, while placement charges the full cache."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_planner import Plan  # noqa: E402
from test_llama_cpp_placement import (  # noqa: E402
    _GIB,
    _cache_launch,
    _discrete_linux_host,  # noqa: F401
    _has_moe_cache,
    _launch,
    _moe_backend,
    _moe_cache_host,  # noqa: F401
    _ubatch_values,
)
from test_offload_planner_seam import _qwen4exp_backend  # noqa: E402

_MASKED = ("--port", "--slot-save-path", "--api-key-file", "-m")


def _masked(cmd):
    out, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        out.append(tok)
        if tok in _MASKED:
            out.append("<masked>")
            skip = True
    return out


def _compute_buffer_per_ubatch(monkeypatch, bytes_per_token):
    """A compute buffer that grows with the micro-batch, so the raise costs resident experts."""
    monkeypatch.setattr(
        LlamaCppBackend,
        "_estimate_compute_buffer_bytes",
        lambda self, n_ubatch = None, n_parallel = 1, per_device_tensor = False, **_kw: (
            int(n_ubatch or 512) * bytes_per_token
        ),
    )


def _declined(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend,
        "_planned_tensor_spill",
        lambda self, inputs, **_kw: Plan(reason = "declined for the test"),
    )


@pytest.mark.parametrize(
    "free_mib, ubatch_mib_per_token, cache, ubatch",
    [
        # ~53% of 15 GiB of experts resident: inside the cache band, the raise is free.
        (20_000, 0, True, ["2048"]),
        # 1 MiB per micro-batch token: the raise would push >10% of the resident experts off.
        (10_000, 1, False, []),
    ],
    ids = ["cache_in_band", "raise_withheld"],
)
def test_a_declined_plan_launches_the_flag_off_argv(
    tmp_path, _moe_cache_host, monkeypatch, free_mib, ubatch_mib_per_token, cache, ubatch
):
    """The resident-share estimate abstained whenever UNSLOTH_SMART_OFFLOAD was set, though both
    of its readers run before the planner and a declined plan launches --fit on like flag off.
    The unknown share dropped --moe-cache-mib auto (qwen_16 on an L4: decode -22..-25%) and kept
    a --ubatch-size 2048 flag off withholds (gemma_12 / qwen_8 on a T4: decode -2..-8%)."""
    if ubatch_mib_per_token:
        _compute_buffer_per_ubatch(monkeypatch, ubatch_mib_per_token * 1024 * 1024)
    _declined(monkeypatch)
    cmds = {}
    for flag in ("off", "on"):
        if flag == "on":
            monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
        else:
            monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
        arm = tmp_path / flag
        arm.mkdir()
        _backend_, cmds[flag] = _cache_launch(arm, memory = [(0, free_mib, 32_000)], ram_gib = 256)
    for flag, cmd in cmds.items():
        assert cmd[cmd.index("--fit") + 1] == "on", (flag, cmd)
        assert _has_moe_cache(cmd) is cache, (flag, cmd)
        assert _ubatch_values(cmd) == ubatch, (flag, cmd)
    assert _masked(cmds["on"]) == _masked(cmds["off"])


_FLASH_NEXT_RATIOS = [0, 0, 0, 4] * 12
_FLASH_NEXT_ATTRS = (
    "_architecture",
    "_n_layers",
    "_n_kv_heads",
    "_n_heads",
    "_embedding_length",
    "_kv_key_length",
    "_kv_value_length",
    "_kv_key_length_swa",
    "_kv_value_length_swa",
    "_sliding_window",
    "_sliding_window_pattern",
    "_n_kv_heads_by_layer",
    "_recurrent_layers",
    "_feed_forward_length_by_layer",
    "_shared_kv_layers",
    "_nextn_predict_layers",
    "_kv_lora_rank",
    "_key_length_mla",
    "_value_length_mla",
    "_kda_head_dim",
    "_ssm_inner_size",
    "_ssm_state_size",
    "_ssm_group_count",
    "_ssm_conv_kernel",
    "_full_attention_interval",
    "_indexer_key_length",
    "_indexer_compress_ratios",
    "_hyper_connection_count",
    "_ple_layers",
    "_ple_ngram_size",
    "_ple_conv_kernel",
)


def _flash_next_launch(tmp_path, free_mib, *, indexer):
    """Qwen3.8-Flash-Next's KV shape on a spilling MoE fixture at -c 131072. ``indexer=False``
    is the same model priced without the lightning-indexer cache and the PLE conv rows."""
    tmp_path.mkdir(parents = True, exist_ok = True)
    backend, gguf = _moe_backend(tmp_path, size_gib = 20, memory = [(0, free_mib, 98_000)])
    shape = _qwen4exp_backend(_indexer_compress_ratios = list(_FLASH_NEXT_RATIOS))
    for attr in _FLASH_NEXT_ATTRS:
        setattr(backend, attr, getattr(shape, attr, None))
    backend._n_experts = 512
    backend._context_length = 262144
    if not indexer:
        backend._indexer_key_length = None
        backend._ple_layers = None
    backend._can_estimate_kv = lambda: True
    backend._available_system_memory_mib = lambda: 256 * 1024
    backend._gguf_tensor_scan = lambda _path: ("qwen4exp", {}, 15 * _GIB)
    cmd = _launch(backend, gguf, n_ctx = 131072)["cmd"]
    kv = backend._estimate_kv_cache_bytes(131072, "f16", n_parallel = 1)
    return cmd, kv


@pytest.mark.parametrize(
    "free_mib, ubatch",
    [
        # The 10% loss edge: main's share keeps the raise; with the indexer cache in the
        # share the same launch dropped it. q38s_18's side of the edge.
        (16_000, ["2048"]),
        # The unpriced edge: the indexer cache pushed the share to unknown, which keeps the
        # raise, where main prices a share the raise would cut by more than 10%.
        (10_000, []),
    ],
    ids = ["loss_edge", "unknown_edge"],
)
def test_flash_next_keeps_the_micro_batch_raise_its_share_band_was_measured_on(
    tmp_path, _moe_cache_host, monkeypatch, free_mib, ubatch
):
    """Reading Flash-Next's full header prices the indexer cache llama.cpp allocates (the second
    KV buffer, 768 MiB at -c 131072 on b11541), which is right for placement. Fed into the
    resident share it moved the spill raise's edges by that much free VRAM: at 18 GB free flag off
    dropped main's --ubatch-size 2048 and lost 27-42% of the prompt speed (Gate B1). The two gates
    read the share on the basis they were measured on, so the micro-batch is the one the model
    priced without the indexer cache gets, while the KV estimate keeps the cache."""
    monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "probe_build_number", classmethod(lambda cls, b = None: 11541)
    )
    _compute_buffer_per_ubatch(monkeypatch, 128 * 1024)
    full_cmd, full_kv = _flash_next_launch(tmp_path / "full", free_mib, indexer = True)
    main_cmd, main_kv = _flash_next_launch(tmp_path / "main", free_mib, indexer = False)
    # 12 attention rows x 131072 cells x 2 pooled 128-wide f16 keys, plus the PLE rows.
    assert full_kv - main_kv >= 768 * 1024 * 1024, (full_kv, main_kv)
    assert full_cmd[full_cmd.index("--fit") + 1] == "on", full_cmd
    assert _ubatch_values(main_cmd) == ubatch, main_cmd
    assert _ubatch_values(full_cmd) == ubatch, full_cmd


def test_the_share_basis_drops_only_the_indexer_and_ple_terms():
    """Every model the share bands were measured on answers 0; Flash-Next answers exactly the
    indexer cache plus the PLE conv rows _estimate_kv_cache_bytes adds for it."""
    from test_offload_planner_seam import _qwen3next_backend

    plain = _qwen3next_backend()
    assert plain._kv_bytes_outside_moe_share_bands(131072, "f16") == 0
    full = _qwen4exp_backend(_indexer_compress_ratios = list(_FLASH_NEXT_RATIOS))
    bare = _qwen4exp_backend(_indexer_compress_ratios = list(_FLASH_NEXT_RATIOS))
    bare._indexer_key_length = None
    bare._ple_layers = None
    for n_ctx, n_parallel in ((8192, 1), (131072, 1), (65536, 4)):
        gap = full._estimate_kv_cache_bytes(
            n_ctx, "f16", n_parallel = n_parallel
        ) - bare._estimate_kv_cache_bytes(n_ctx, "f16", n_parallel = n_parallel)
        assert gap > 0
        assert full._kv_bytes_outside_moe_share_bands(n_ctx, "f16", n_parallel = n_parallel) == gap

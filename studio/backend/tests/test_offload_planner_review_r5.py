# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The indexer cache each llama.cpp build allocates, what a revoked plan puts back (the checkpoint
cap and the shared cache), and a device-changing retry's lazy mode."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_planner import Plan  # noqa: E402

MIB = 1024**2


# ------------------------------------------------ findings 1 / 7: the indexer cache per build


@pytest.mark.parametrize(
    "build, ratios, widths",
    [
        (None, None, (128, 256)),  # unknown: the conservative pre-#28330 charge
        (10888, [0, 4], (128, 256)),
        (10889, None, (128, 0)),  # #28330: K only
        (10889, [0, 4], (128, 0)),  # pooled rows not yet widened
        (11330, None, (128, 0)),
        (11330, [0, 0, 0], (128, 0)),
        (11330, [0, 4, 0, 4], (256, 0)),  # #29751: raw key | pooled key
        (11330, 4, (256, 0)),
    ],
)
def test_the_indexer_widths_follow_the_build(build, ratios, widths):
    from core.inference.offload_layout import indexer_cache_widths
    assert indexer_cache_widths(128, 256, ratios, build) == widths


@pytest.mark.parametrize(
    "build, ratios, k_rows, v_len",
    [(None, None, 1, 256), (10889, None, 1, 0), (11541, [0, 4] * 24, 2, 0), (11541, None, 1, 0)],
)
def test_both_readers_price_the_indexer_this_build_allocates(
    monkeypatch, build, ratios, k_rows, v_len
):
    """b10889 (ggml-org/llama.cpp#28330) dropped the indexer's V cache; b11330 (#29751) widened a
    pooled row to two keys. Charging V on a current build cut a fitting 64K request."""
    from test_offload_planner import _StubReader, _StubTensor, _layout_from_reader
    from test_offload_planner_seam import _QWEN4EXP_FIELDS, _qwen4exp_backend

    monkeypatch.setattr(
        LlamaCppBackend, "probe_build_number", classmethod(lambda cls, b = None: build)
    )
    n_ctx = 65536
    b = _qwen4exp_backend(_indexer_compress_ratios = ratios)
    with_idx = b._estimate_kv_cache_bytes(
        n_ctx, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
    )
    plain = _qwen4exp_backend(_architecture = "qwen3next")._estimate_kv_cache_bytes(
        n_ctx, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
    )
    assert with_idx - plain == 12 * n_ctx * (128 * k_rows + v_len) * 2

    fields = dict(_QWEN4EXP_FIELDS)
    if ratios is not None:
        fields["qwen4exp.attention.compress_ratios"] = ratios
    layout = _layout_from_reader(
        _StubReader(fields, [_StubTensor(f"blk.{i}.attn_q.weight", MIB) for i in range(48)]),
        build = build,
    )
    assert layout.kv_bytes(n_ctx) + layout.recurrent_bytes == with_idx


def test_the_reader_keeps_the_compress_ratios():
    from test_kv_cache_estimation import _backend_from_gguf
    b = _backend_from_gguf(
        "qwen4exp", {"block_count": 4, "attention.compress_ratios": [0, 4, 0, 4]}
    )
    assert b._indexer_compress_ratios == [0, 4, 0, 4]


# ------------------------------------------------ findings 2-5: what a revoked plan puts back


def test_a_revoked_slot_cut_redecides_the_checkpoint_cap_and_the_shared_cache(
    tmp_path, monkeypatch
):
    """The plan cut 4 slots to 1, so the per-slot checkpoint cap was decided for one slot and the
    integrity flags dropped --kv-unified. The revocation puts the slots and --kv-unified back; the
    cap and the published cache mode have to follow, as the single-sequence retry's do."""
    from test_offload_planner_launch import _launch_crash_then_ok

    monkeypatch.setattr(
        LlamaCppBackend,
        "_bounded_ctx_checkpoints",
        lambda self, n_parallel, *a, **k: 32 // max(1, int(n_parallel)),
    )
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(
        tmp_path, monkeypatch, plan, caps = {"ctx_checkpoints_flag": "--ctx-checkpoints"}
    )
    assert len(cmds) == 2, cmds
    first, retry = cmds
    assert first[first.index("--parallel") + 1] == "1"
    assert first[first.index("--ctx-checkpoints") + 1] == "32"
    assert retry[retry.index("--parallel") + 1] == "4"
    assert retry.count("--ctx-checkpoints") == 1, retry
    assert retry[retry.index("--ctx-checkpoints") + 1] == "8"
    assert "--kv-unified" in retry
    assert backend._kv_cache_unified is True


# ------------------------------------------------ finding 6: a device-changing retry


def test_a_device_changing_retry_hands_the_lazy_mode_back_to_auto(tmp_path, monkeypatch):
    """ "on" was decided for the crashed card; the respawn on an integrated sibling is one llama.cpp's
    auto turns off, so Studio's choice has to leave with the device it was made for."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent
    from test_llama_cpp_placement import _backend

    monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
    card = 24 * 1024
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, card, card), (1, card, card)])
    backend._studio_lazy_mode = lambda **_kw: "on"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._select_gpus_split_aware = lambda *a, **k: ([0], False)
    backend._arch_crash_retry_gpu_ids = lambda crashed, enumerated: [1]
    cmds = []
    masks = []
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        cmds.append(list(cmd))
        env = kwargs.get("env") or {}
        masks.append((env.get("CUDA_VISIBLE_DEVICES"), env.get("HIP_VISIBLE_DEVICES")))
        # Every launch on the crashed card fails; the retry is pinned to device 1.
        crashed = "1" not in masks[-1]
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": 1 if crashed else None,
                "poll": lambda self: 1 if crashed else None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        if "1" not in masks[-1]:
            backend._stdout_lines = ["ROCm error: device kernel image is invalid"]
            return False
        return True

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", speculative_type = "off")
        )
    assert "1" in masks[-1] and "1" not in masks[0], masks
    assert cmds[0][cmds[0].index("-lzm") + 1] == "on", "the control must emit Studio's mode"
    assert "-lzm" not in cmds[-1], cmds[-1]

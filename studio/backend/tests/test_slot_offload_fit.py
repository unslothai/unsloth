# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the offload-avoidance serving-slot reduction (`_slots_that_fit_on_gpu`).

When a pinned context does not fit at the requested `--parallel` slot count, Unsloth would
flip to `--fit on` and llama-server offloads layers to host RAM, collapsing decode ~3x
(oobabooga #6718). Instead the loader retries the on-GPU fit at fewer slots and keeps the
largest count that stays fully on GPU (`-ngl -1`). These tests drive the real helper with
synthetic VRAM maps; the KV term is mocked so totals are controlled and the reduction logic
is asserted directly (no GPU, network, or subprocess).
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from core.inference.llama_cpp import LlamaCppBackend

MIB = 1024 * 1024
CTX = 90624
FRAC = LlamaCppBackend._GPU_PIN_VRAM_FRACTION  # 0.97; usable = free - 0.03*total


def _backend(
    vocab = 248320,
    embd = 5120,
    kv_fixed_mib = 0,
    kv_calls = None,
):
    """Backend with the dims the compute buffer reads; KV mocked to a fixed size so the
    only slot-dependent term is the compute buffer (485 MiB/slot f32 output x 1.15)."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._vocab_size = vocab
    b._embedding_length = embd
    b._key_length_mla = None

    def estimate(
        ctx,
        t = None,
        **kwargs,
    ):
        if kv_calls is not None:
            kv_calls.append(kwargs)
        return kv_fixed_mib * MIB

    b._estimate_kv_cache_bytes = estimate
    b._can_estimate_kv = lambda: True
    return b


def _run(
    b,
    n_parallel,
    base_mib,
    gpus,
    total_by_idx,
    overhead_mib = 0,
    swa_full = False,
    split_extra_mib = 0,
):
    # Split step passed only when set, so the other cases keep exercising the default.
    extra = {"split_extra_bytes": int(split_extra_mib * MIB)} if split_extra_mib else {}
    return b._slots_that_fit_on_gpu(
        n_parallel,
        CTX,
        gpus,
        total_by_idx,
        int(base_mib * MIB),
        "q8_0",
        FRAC,
        int(overhead_mib * MIB),
        1,
        n_ubatch = 512,
        swa_full = swa_full,
        **extra,
    )


class TestSlotsThatFitOnGpu:
    """Compute-buffer per slot (vocab 248320, embd 5120): cb(1)=46, cb(2)=604, cb(3)=1162,
    cb(4)=1719 MiB. Single 24 GB card usable = 24576 - 0.03*24576 = 23839 MiB."""

    def test_reduces_to_largest_fitting_slot(self):
        # base+KV = 22500: par4 (24219) over 23839, par3 (23662) fits -> 3 slots on GPU.
        gi, use_fit, slots = _run(_backend(), 4, 22500, [(0, 24576)], {0: 24576})
        assert use_fit is False and gi == [0] and slots == 3

    def test_floor_when_only_one_slot_fits(self):
        # base 23400: par2 (24004) over, par1 (23446) fits -> drop all the way to 1.
        gi, use_fit, slots = _run(_backend(), 4, 23400, [(0, 24576)], {0: 24576})
        assert use_fit is False and gi == [0] and slots == 1

    def test_none_fit_stays_offload(self):
        # Even a single slot (24046) exceeds usable -> genuine offload, unchanged.
        gi, use_fit, slots = _run(_backend(), 4, 24000, [(0, 24576)], {0: 24576})
        assert use_fit is True and gi is None and slots == 4

    def test_roomy_would_keep_all_but_helper_only_reduces(self):
        # On a roomy card par4 fits, so load_model never calls this helper; if called it
        # still only searches < n_parallel and never raises the count above the request.
        gi, use_fit, slots = _run(_backend(), 4, 5000, [(0, 183000)], {0: 183000})
        assert use_fit is False and slots == 3 and slots < 4

    def test_single_slot_request_is_noop(self):
        # n_parallel == 1: nothing to reduce (range empty) -> report offload unchanged.
        gi, use_fit, slots = _run(_backend(), 1, 22500, [(0, 24576)], {0: 24576})
        assert use_fit is True and gi is None and slots == 1

    def test_multi_gpu_reduces_across_devices(self):
        # Needs 2 GPUs: usable/GPU = 23839, cumulative 47677. base+KV 46200: par4 (47919)
        # over, par3 (47362) fits across both -> 3 slots spanning [0, 1].
        gi, use_fit, slots = _run(
            _backend(), 4, 46200, [(0, 24576), (1, 24576)], {0: 24576, 1: 24576}
        )
        assert use_fit is False and gi == [0, 1] and slots == 3

    def test_kv_counted_per_candidate(self):
        # A non-zero (slot-independent) KV shifts the threshold: with 3000 MiB KV and
        # base 19500 (= 22500 total at par-independent terms) the same par3 fit holds.
        gi, use_fit, slots = _run(_backend(kv_fixed_mib = 3000), 4, 19500, [(0, 24576)], {0: 24576})
        assert use_fit is False and slots == 3

    def test_split_rate_is_rechecked_on_multi_gpu_candidates(self):
        # The base footprint carries the context-compute buffer at the single-device
        # rate, so a candidate that lands on 2 GPUs owes one enlarged copy per card.
        # Charging it drops the count further (3 -> 1) rather than pinning an OOM.
        gpus, totals = [(0, 24576), (1, 24576)], {0: 24576, 1: 24576}
        assert _run(_backend(), 4, 46200, gpus, totals, split_extra_mib = 500) == ([0, 1], False, 1)
        # And when no count clears it, offload (the pre-existing failure mode).
        assert _run(_backend(), 4, 46200, gpus, totals, split_extra_mib = 1000) == (None, True, 4)

    def test_split_step_does_not_touch_a_single_gpu_candidate(self):
        assert _run(_backend(), 4, 22500, [(0, 24576)], {0: 24576}, split_extra_mib = 500) == (
            _run(_backend(), 4, 22500, [(0, 24576)], {0: 24576})
        )

    def test_swa_full_is_used_for_every_candidate(self):
        calls = []
        _run(
            _backend(kv_calls = calls),
            4,
            22500,
            [(0, 24576)],
            {0: 24576},
            swa_full = True,
        )
        assert calls
        assert all(call["swa_full"] is True for call in calls)

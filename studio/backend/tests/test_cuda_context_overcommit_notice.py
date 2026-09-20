# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A hand-set context a discrete GPU cannot hold must say so (issues #11349, #11336).

The numbers here are not invented. A Qwen3-VL-8B-Instruct-UD-Q4_K_XL launched at
``-c 65536`` was measured against a live llama-server on an RTX PRO 6000 Blackwell:
it asks for ONE 9216.00 MiB KV allocation, and with 12 GiB free the server dies with

    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0:
        cudaMalloc failed: out of memory
    llama_init_from_model: failed to initialize the context:
        failed to allocate buffer for kv cache

On Windows that same allocation does NOT fail: since driver 536.40 the WDDM sysmem
fallback policy serves it from host RAM, the server starts, answers /health, and decodes
5-10x slower with nothing reported anywhere. Every recovery path in the spawn loop keys
off the child crashing, so on Windows none of them can fire. The advisory under test is
what remains.

``test_kv_cache_estimate_matches_measured_allocation`` is the load-bearing one: it pins
the estimator to a figure read off real hardware, so a refactor that drifts the KV maths
fails here rather than in a user's silent 5x regression.

No GPU, network or model files. Cross-platform.
"""

import logging
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Sibling test modules are imported by name here, the same way
# test_chat_message_identity.py and test_auto_offload_ctx_invariants.py do it.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Same stubbing contract as test_kv_cache_estimation.py: these are process-wide
# setdefaults, so they must stay compatible with that module's versions.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: logging.getLogger(name)
_loggers_stub.__path__ = [str(Path(_BACKEND_DIR) / "loggers")]
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: logging.getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# Reuse the GGUF builder rather than duplicating a second one that can drift.
from test_kv_cache_estimation import _backend_from_gguf  # noqa: E402

GIB = 1024**3
MIB = 1024**2

# Qwen3-VL-8B-Instruct, read out of the real GGUF's metadata.
QWEN3VL_8B = {
    "block_count": 36,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "embedding_length": 4096,
    "context_length": 262144,
}

# Measured on the reporter's configuration, and on ours.
REPORTED_CTX = 65536
MEASURED_KV_MIB = 9216.0  # the single allocation llama-server asks for
WEIGHTS_BYTES = int(4.80 * GIB)  # UD-Q4_K_XL on disk
MMPROJ_BYTES = int(1.40 * GIB)  # mmproj-F16, loaded because -ngl 99 puts it on GPU
CARD_16GIB_MIB = 16.0 * 1024


@pytest.fixture(scope = "module")
def backend():
    return _backend_from_gguf("qwen3vl", QWEN3VL_8B)


class TestKVEstimateAgainstHardware:
    """The estimator must reproduce what llama-server actually allocated."""

    def test_kv_cache_estimate_matches_measured_allocation(self, backend):
        kv = backend._estimate_kv_cache_bytes(REPORTED_CTX, None)
        assert kv / MIB == pytest.approx(
            MEASURED_KV_MIB, abs = 0.5
        ), f"expected the measured {MEASURED_KV_MIB} MiB allocation, got {kv / MIB:.1f} MiB"
        # Stated the other way, because 9.00 GiB exactly is the fact worth pinning.
        assert kv / GIB == pytest.approx(9.0, abs = 0.01)

    def test_kv_scales_linearly_with_context(self, backend):
        """Halving the context halves the cache: the term that dominates the budget."""
        full = backend._estimate_kv_cache_bytes(REPORTED_CTX, None)
        half = backend._estimate_kv_cache_bytes(REPORTED_CTX // 2, None)
        assert half / GIB == pytest.approx(full / GIB / 2, rel = 0.01)

    def test_q8_0_cache_is_substantially_smaller(self, backend):
        """q8_0 is the lever the planner never pulls; it must be worth suggesting."""
        f16 = backend._estimate_kv_cache_bytes(REPORTED_CTX, None)
        q8 = backend._estimate_kv_cache_bytes(REPORTED_CTX, "q8_0")
        assert q8 < f16
        # ~8.5 bits/elem including the scale, so a little over half, not exactly half.
        assert q8 / f16 == pytest.approx(0.53, abs = 0.03)
        assert (f16 - q8) / GIB > 4.0, "q8_0 must free more than 4 GiB to be worth offering"


class TestSixteenGibBudget:
    """The reported configuration must be recognised as not fitting a 16 GiB card."""

    def test_reported_config_exceeds_a_16gib_card(self, backend):
        kv = backend._estimate_kv_cache_bytes(REPORTED_CTX, None)
        footprint = (
            WEIGHTS_BYTES
            + MMPROJ_BYTES
            + int(MMPROJ_BYTES * (backend._MMPROJ_VRAM_SAFETY - 1.0))
            + backend._CUDA_CONTEXT_RESERVE_BYTES
            + kv
        )
        footprint_mib = footprint / MIB
        # Even the whole card, with no reserve at all, is not enough.
        assert (
            footprint_mib > CARD_16GIB_MIB
        ), f"footprint {footprint_mib:.0f} MiB should exceed a {CARD_16GIB_MIB:.0f} MiB card"
        # But only just: ~74 MiB, under half a percent. That near-miss is the whole
        # character of this bug. It is why the reporter could inspect memory and
        # conclude there was no overflow, and why on Windows the driver can absorb it
        # into host RAM without anything looking wrong. A margin test that demanded a
        # large overshoot here would be asserting a fiction; the overshoot is tiny.
        assert 0 < footprint_mib - CARD_16GIB_MIB < 200

    def test_the_budget_rejects_it_by_a_clear_margin(self, backend):
        """Against the 0.97 budget rather than the raw card, the verdict is unambiguous.

        16458 MiB against a 15892 MiB budget is 566 MiB over, so the planner already
        declines to pin this and hands llama.cpp the fitter. The failure therefore is not
        a mispriced KV cache; it is that on Windows an overshoot the planner rejected can
        still be satisfied by the driver if anything downstream asks for it anyway.
        """
        kv = backend._estimate_kv_cache_bytes(REPORTED_CTX, None)
        footprint_mib = (
            WEIGHTS_BYTES
            + MMPROJ_BYTES
            + int(MMPROJ_BYTES * (backend._MMPROJ_VRAM_SAFETY - 1.0))
            + backend._CUDA_CONTEXT_RESERVE_BYTES
            + kv
        ) / MIB
        budget_mib = CARD_16GIB_MIB * 0.97
        assert footprint_mib > budget_mib
        assert footprint_mib - budget_mib > 500

    def test_q8_0_brings_the_same_config_under_the_card(self, backend):
        """The suggestion the notice makes has to be true."""
        kv_q8 = backend._estimate_kv_cache_bytes(REPORTED_CTX, "q8_0")
        footprint = (
            WEIGHTS_BYTES
            + MMPROJ_BYTES
            + int(MMPROJ_BYTES * (backend._MMPROJ_VRAM_SAFETY - 1.0))
            + backend._CUDA_CONTEXT_RESERVE_BYTES
            + kv_q8
        )
        assert footprint / MIB < CARD_16GIB_MIB

    def test_text_only_at_16gib_fits_which_is_why_it_was_hard_to_see(self, backend):
        """Without the mmproj the same context fits, matching the Linux run that passed.

        On the G4 with ~16 GiB free and no mmproj the server started and decoded at full
        speed. That near-miss is the reason the reporter could rule out a memory problem
        by eye: the text-only budget really does fit, and the vision tower is what tips it.
        """
        kv = backend._estimate_kv_cache_bytes(REPORTED_CTX, None)
        footprint = WEIGHTS_BYTES + backend._CUDA_CONTEXT_RESERVE_BYTES + kv
        assert footprint / MIB < CARD_16GIB_MIB


class TestOvercommitNotice:
    """The advisory itself: when it speaks, when it stays quiet, and what it says."""

    def test_silent_when_the_context_fits(self):
        assert LlamaCppBackend._cuda_context_overcommit_notice(8192, 65536, None) is None

    def test_silent_without_a_measured_ceiling(self):
        """No ceiling means no fit maths ran; _unmeasured_context_notice owns that case."""
        assert LlamaCppBackend._cuda_context_overcommit_notice(65536, 0, None) is None
        assert LlamaCppBackend._cuda_context_overcommit_notice(0, 65536, None) is None

    def test_silent_at_exactly_the_ceiling(self):
        assert LlamaCppBackend._cuda_context_overcommit_notice(32768, 32768, None) is None

    def test_reports_both_numbers(self):
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None)
        assert msg is not None
        assert "65,536" in msg and "32,768" in msg

    def test_windows_names_the_driver_behaviour_and_the_setting(self):
        """On Windows the whole point is that nothing else will tell the user."""
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None, windows = True)
        assert "system memory" in msg
        assert "Sysmem Fallback Policy" in msg
        assert "Prefer No Sysmem Fallback" in msg
        # It must not claim an error will be reported, because none is.
        assert "does not report this as an error" in msg

    def test_windows_remedy_is_attributed_to_nvidia(self):
        """There is no vendor signal in this helper's scope, so the NVIDIA-only fix must
        be labelled. An AMD or Intel owner sent to the NVIDIA Control Panel is being given
        a wrong instruction, not just an unhelpful one."""
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None, windows = True)
        idx = msg.index("NVIDIA Control Panel")
        assert (
            "On NVIDIA GPUs" in msg[:idx]
        ), "the NVIDIA Control Panel instruction must be qualified before it is given"

    def test_non_windows_describes_cpu_offload_instead(self):
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None, windows = False)
        assert "moved to the CPU" in msg
        assert "Sysmem" not in msg, "the Windows-only remedy must not leak to other platforms"

    def test_q8_hint_only_when_it_actually_fixes_it(self):
        """A suggestion that would not help is worse than no suggestion."""
        helped = LlamaCppBackend._cuda_context_overcommit_notice(
            65536, 32768, None, quantised_ctx_fits = True
        )
        assert "q8_0" in helped

        not_helped = LlamaCppBackend._cuda_context_overcommit_notice(
            65536, 32768, None, quantised_ctx_fits = False
        )
        assert "q8_0" not in not_helped

    def test_no_q8_hint_when_the_cache_is_already_quantised(self):
        msg = LlamaCppBackend._cuda_context_overcommit_notice(
            65536, 32768, "q8_0", quantised_ctx_fits = True
        )
        assert "q8_0" not in msg

    @pytest.mark.parametrize("windows", [True, False])
    def test_never_raises_and_always_returns_text(self, windows):
        msg = LlamaCppBackend._cuda_context_overcommit_notice(
            65536, 4096, None, windows = windows, quantised_ctx_fits = True
        )
        assert isinstance(msg, str) and msg.strip()

    def test_is_an_advisory_not_a_refusal(self):
        """Contrast with Metal, which refuses. A discrete GPU has somewhere to spill to,
        so blocking the load would break configurations that work today."""
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None)
        for word in ("cannot load", "refused", "aborted", "will not start"):
            assert word not in msg.lower()

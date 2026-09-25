# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A hand-set context a discrete GPU cannot hold must say so (issues #11349, #11336)."""

import logging
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: logging.getLogger(name)
_loggers_stub.__path__ = [str(Path(_BACKEND_DIR) / "loggers")]
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: logging.getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

from test_kv_cache_estimation import _backend_from_gguf  # noqa: E402

GIB = 1024**3
MIB = 1024**2

QWEN3VL_8B = {
    "block_count": 36,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "embedding_length": 4096,
    "context_length": 262144,
}

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
        assert (
            footprint_mib > CARD_16GIB_MIB
        ), f"footprint {footprint_mib:.0f} MiB should exceed a {CARD_16GIB_MIB:.0f} MiB card"
        assert 0 < footprint_mib - CARD_16GIB_MIB < 200

    def test_the_budget_rejects_it_by_a_clear_margin(self, backend):
        """Against the 0.97 budget rather than the raw card, the verdict is unambiguous."""
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
        """Without the mmproj the same context fits, matching the Linux run that passed."""
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

    def test_describes_the_planned_cpu_offload(self):
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None)
        assert "moved to the CPU" in msg
        assert "Sysmem" not in msg

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

    def test_never_raises_and_always_returns_text(self):
        msg = LlamaCppBackend._cuda_context_overcommit_notice(
            65536, 4096, None, quantised_ctx_fits = True
        )
        assert isinstance(msg, str) and msg.strip()

    def test_is_an_advisory_not_a_refusal(self):
        """Unlike Metal, a discrete GPU spills, so this warns rather than refuses."""
        msg = LlamaCppBackend._cuda_context_overcommit_notice(65536, 32768, None)
        for word in ("cannot load", "refused", "aborted", "will not start"):
            assert word not in msg.lower()


def _launch_explicit_ctx(tmp_path, monkeypatch, model_gb, n_ctx, **load_kwargs):
    from test_llama_cpp_placement import _backend as _placement_backend, _launch

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    backend, gguf = _placement_backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    backend._get_gguf_size_bytes = lambda _path: int(model_gb * GIB)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * 64 * 1024
    backend._estimate_compute_buffer_bytes = lambda **k: 1
    cmd = _launch(backend, gguf, n_ctx = n_ctx, **load_kwargs)["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    return backend.last_load_warning or ""


def test_the_notice_names_a_context_that_fits(tmp_path, monkeypatch):
    warning = _launch_explicit_ctx(tmp_path, monkeypatch, model_gb = 18, n_ctx = 131072)
    assert "does not fit in this GPU's memory" in warning


def test_no_context_is_offered_when_the_weights_alone_overflow(tmp_path, monkeypatch):
    """40 GB of weights on a 24 GB card: no context fits, so none may be offered."""
    warning = _launch_explicit_ctx(tmp_path, monkeypatch, model_gb = 40, n_ctx = 32768)
    assert "The largest that fits is" not in warning


def test_a_windows_cuda_build_is_not_told_the_driver_spills(tmp_path, monkeypatch):
    """--fit on and spill plans place host tensors on purpose; WDDM advice cannot undo that."""
    monkeypatch.setattr(LlamaCppBackend, "_sysmem_fallback_risk", staticmethod(lambda b = None: True))
    warning = _launch_explicit_ctx(tmp_path, monkeypatch, model_gb = 18, n_ctx = 131072)
    assert "moved to the CPU" in warning
    assert "NVIDIA Control Panel" not in warning


@pytest.mark.parametrize(
    "q8_scratch_gib, extra_args, offered",
    [(0, [], True), (8, [], False), (0, ["--flash-attn", "off"], False)],
)
def test_the_q8_hint_prices_q8_compute_scratch(
    tmp_path, monkeypatch, q8_scratch_gib, extra_args, offered
):
    """q8_0 adds dequant scratch; the hint must hold with it, not with the f16 figure."""
    from test_llama_cpp_placement import _backend as _placement_backend, _launch

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    backend, gguf = _placement_backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    backend._get_gguf_size_bytes = lambda _path: 14 * GIB
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = (
        lambda ctx, cache = None, *a, **k: int(ctx)
        * (32 if str(cache).startswith("q8") else 64)
        * 1024
    )
    backend._estimate_compute_buffer_bytes = lambda **k: 1
    backend._compute_buffer_ctx_bytes = lambda ctx, ub, cache_type = None, **k: (
        q8_scratch_gib * GIB if str(cache_type).startswith("q8") else 0
    )
    _launch(backend, gguf, n_ctx = 196608, extra_args = extra_args)
    warning = backend.last_load_warning or ""
    assert "does not fit in this GPU's memory" in warning
    assert ("q8_0" in warning) is offered


def test_no_notice_when_the_kv_cache_stays_on_the_host(tmp_path, monkeypatch):
    """-nkvo keeps the cache off the GPU, so the priced KV overflow does not happen."""
    warning = _launch_explicit_ctx(
        tmp_path, monkeypatch, model_gb = 18, n_ctx = 131072, extra_args = ["-nkvo"]
    )
    assert "does not fit in this GPU's memory" not in warning


@pytest.mark.parametrize(
    "extra_args", [["--device", "none"], ["-ot", "exps=CPU"], ["--gpu-layers", "10"]]
)
def test_no_notice_for_a_deliberate_cpu_placement(tmp_path, monkeypatch, extra_args):
    warning = _launch_explicit_ctx(
        tmp_path, monkeypatch, model_gb = 18, n_ctx = 131072, extra_args = extra_args
    )
    assert "does not fit in this GPU's memory" not in warning

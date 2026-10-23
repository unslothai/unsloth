# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Context refit after slot reduction, checked on synthetic plans and slot costs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from test_llama_cpp_placement import (  # noqa: E402
    _backend,
    _install_slot_scaled_compute,
    _launch,
)

MIB = 1024 * 1024
NATIVE_CTX = 262144
CARD_MIB = 12 * 1024  # usable 11,919 MiB at the 0.97 pin fraction

# Weights chosen for their reduction band at the current _FIT_MIN_CTX; re-measure if the floor moves.
_KEEPS_THREE_MIB = 9_200
_KEEPS_TWO_MIB = 9_800
_ASK_STILL_MATTERS_MIB = 9_400
MIXED_CARDS = (CARD_MIB, 1_280)

# Qwen3.8-27B-GGUF metadata.
HYBRID = {
    "_architecture": "qwen35",
    "_vocab_size": 248320,
    "_nextn_predict_layers": 1,
    "_n_layers": 65,
    "_n_kv_heads": 4,
    "_n_heads": 24,
    "_embedding_length": 5120,
    "_kv_key_length": 256,
    "_kv_value_length": 256,
    "_key_length_mla": None,
    "_context_length": NATIVE_CTX,
    "_full_attention_interval": 4,
    "_ssm_inner_size": 6144,
    "_ssm_state_size": 128,
    "_ssm_group_count": 16,
    "_ssm_conv_kernel": 4,
}
DENSE = {
    "_architecture": "qwen3",
    "_vocab_size": 248320,
    "_n_layers": 64,
    "_n_kv_heads": 8,
    "_n_heads": 32,
    "_embedding_length": 5120,
    "_kv_key_length": 128,
    "_kv_value_length": 128,
    "_key_length_mla": None,
    "_context_length": NATIVE_CTX,
}

# Gemma 3 12B-shaped: five sliding-window layers in six.
SLIDING_WINDOW = {
    "_architecture": "gemma3",
    "_vocab_size": 262144,
    "_n_layers": 48,
    "_n_kv_heads": 8,
    "_n_heads": 16,
    "_embedding_length": 3840,
    "_feed_forward_length": 15360,
    "_kv_key_length": 256,
    "_kv_value_length": 256,
    "_key_length_mla": None,
    "_context_length": 131072,
    "_sliding_window": 1024,
    "_sliding_window_pattern": [i % 6 != 5 for i in range(48)],
}


def _plan(
    tmp_path,
    *,
    weights_mib,
    n_parallel,
    spec,
    vram_mib = CARD_MIB,
    n_ctx = 0,
    metadata = HYBRID,
    gpus = 1,
    slot_scaled_compute = True,
):
    """Return the generated placement plan. ``vram_mib`` may be a per-card sequence.
    ``slot_scaled_compute`` False prices the compute buffer with the real estimator."""
    cards = list(vram_mib) if isinstance(vram_mib, (tuple, list)) else [vram_mib] * gpus
    memory = [(i, mib, mib) for i, mib in enumerate(cards)]
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)

    def read(_path):
        for key, value in metadata.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: weights_mib * MIB
    del backend._can_estimate_kv
    if slot_scaled_compute:
        _install_slot_scaled_compute(backend)
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
    }
    launched = _launch(backend, gguf, speculative_type = spec, n_ctx = n_ctx, n_parallel = n_parallel)
    cmd = launched["cmd"]

    def flag(name, default = None):
        return cmd[cmd.index(name) + 1] if name in cmd else default

    return {
        "ctx": int(flag("-c", 0)),
        "slots": int(flag("--parallel", 1)),
        "fit": flag("--fit", "off"),
        "spec": flag("--spec-type", "-"),
        "ceiling": backend._max_context_length,
        "warning": backend.last_load_warning or "",
        "devices": (launched["env"] or {}).get("CUDA_VISIBLE_DEVICES"),
    }


class TestTheLaunchedCountOwnsTheContext:
    """Speculation is off so only slot reduction affects the context."""

    @pytest.mark.parametrize(
        "weights_mib,asked,slots,ctx",
        [
            (_KEEPS_THREE_MIB, 4, 3, 12_288),
            (_KEEPS_TWO_MIB, 4, 2, 14_080),
            (_KEEPS_TWO_MIB, 8, 2, 14_080),
        ],
    )
    def test_auto_context_follows_the_reduction(self, tmp_path, weights_mib, asked, slots, ctx):
        got = _plan(tmp_path, weights_mib = weights_mib, n_parallel = asked, spec = "off")
        assert (got["slots"], got["fit"]) == (slots, "off")
        assert got["ctx"] == got["ceiling"] == ctx

    @pytest.mark.parametrize(
        "weights_mib,asked,final", [(_KEEPS_THREE_MIB, 4, 3), (_KEEPS_TWO_MIB, 4, 2)]
    )
    def test_reducing_to_a_count_matches_starting_at_it(self, tmp_path, weights_mib, asked, final):
        """A reduced plan matches one started at its final slot count."""
        reduced = _plan(tmp_path, weights_mib = weights_mib, n_parallel = asked, spec = "off")
        direct = _plan(tmp_path, weights_mib = weights_mib, n_parallel = final, spec = "off")
        assert reduced["slots"] == direct["slots"] == final
        assert (reduced["ctx"], reduced["ceiling"]) == (direct["ctx"], direct["ceiling"])

    def test_the_published_ceiling_is_not_the_fit_floor_anchor(self, tmp_path):
        """The published ceiling must follow the final slot count, not equal the _FIT_MIN_CTX anchor."""
        got = _plan(tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off")
        assert got["ceiling"] == 14_080
        assert got["ceiling"] != llama_cpp._FIT_MIN_CTX

    def test_a_layer_split_across_two_cards_re_fits_the_same_way(self, tmp_path):
        """The same invariant holds for a two-card layer split."""
        reduced = _plan(
            tmp_path, weights_mib = 13_800, n_parallel = 4, spec = "off", vram_mib = 8_704, gpus = 2
        )
        direct = _plan(
            tmp_path, weights_mib = 13_800, n_parallel = 2, spec = "off", vram_mib = 8_704, gpus = 2
        )
        assert (reduced["slots"], reduced["fit"]) == (2, "off")
        assert reduced["ctx"] == reduced["ceiling"] == 10_752
        assert (reduced["ctx"], reduced["ceiling"]) == (direct["ctx"], direct["ceiling"])


class TestTheRealEstimatorStillReduces:
    """Real pricing reduces slots for hybrid state costs and keeps slots on dense models."""

    @pytest.mark.parametrize("weights_mib,slots,ctx", [(10_400, 3, 9_472), (10_500, 2, 10_240)])
    def test_a_hybrid_load_reduces_and_re_fits(self, tmp_path, weights_mib, slots, ctx):
        got = _plan(
            tmp_path, weights_mib = weights_mib, n_parallel = 4, spec = "off", slot_scaled_compute = False
        )
        assert (got["slots"], got["fit"], got["ctx"]) == (slots, "off", ctx)
        direct = _plan(
            tmp_path,
            weights_mib = weights_mib,
            n_parallel = slots,
            spec = "off",
            slot_scaled_compute = False,
        )
        assert (direct["slots"], direct["ctx"]) == (slots, ctx)

    def test_a_sliding_window_split_reprices_its_masks_per_candidate(self, tmp_path):
        """A layer split's sliding-window masks scale with slots, so each candidate
        prices its own count rather than the one asked for."""
        plans = [
            _plan(
                tmp_path,
                weights_mib = 21_200,
                n_parallel = asked,
                spec = "off",
                metadata = SLIDING_WINDOW,
                gpus = 2,
                slot_scaled_compute = False,
            )
            for asked in (16, 1)
        ]
        assert [(p["slots"], p["fit"], p["ctx"]) for p in plans] == [(1, "off", 8_448)] * 2

    def test_a_dense_load_keeps_its_slots(self, tmp_path):
        plans = [
            _plan(
                tmp_path,
                weights_mib = 8_200,
                n_parallel = asked,
                spec = "off",
                metadata = DENSE,
                slot_scaled_compute = False,
            )
            for asked in (1, 4)
        ]
        assert [(p["slots"], p["fit"]) for p in plans] == [(1, "off"), (4, "off")]
        assert plans[0]["ctx"] == plans[1]["ctx"]


class TestTheRefitStaysOnTheCardsTheReductionChose:
    """A 12 GiB primary next to a 1.25 GiB card: the reduced plan fits on the primary
    alone, and only the small card can hold a longer context."""

    def test_the_refit_does_not_pull_in_another_card(self, tmp_path):
        got = _plan(
            tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off", vram_mib = MIXED_CARDS
        )
        assert (got["slots"], got["fit"], got["devices"]) == (2, "off", "0")
        assert got["ctx"] == 14_080

    def test_it_matches_a_request_started_at_the_final_count(self, tmp_path):
        reduced = _plan(
            tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off", vram_mib = MIXED_CARDS
        )
        direct = _plan(
            tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 2, spec = "off", vram_mib = MIXED_CARDS
        )
        assert reduced["devices"] == direct["devices"] == "0"
        assert (reduced["ctx"], reduced["ceiling"]) == (direct["ctx"], direct["ceiling"])

    def test_the_ceiling_still_counts_the_card_the_launch_left_out(self, tmp_path):
        """The ceiling keeps measuring across both cards, not just the launched one."""
        mixed = _plan(
            tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off", vram_mib = MIXED_CARDS
        )
        alone = _plan(tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off")
        assert mixed["ctx"] == alone["ctx"] == 14_080
        assert (mixed["ceiling"], alone["ceiling"]) == (14_592, 14_080)


class TestAutoSpeculationStillDecidesBeforeTheReduction:
    """Pin the existing behavior where Auto admits MTP before slot reduction."""

    def test_a_direct_one_slot_request_drops_the_drafter_and_keeps_context(self, tmp_path):
        got = _plan(tmp_path, weights_mib = 10_200, n_parallel = 1, spec = "auto")
        assert (got["slots"], got["ctx"], got["spec"]) == (1, 18_944, "ngram-mod")

    @pytest.mark.parametrize("asked", [4, 8])
    def test_a_reduced_request_still_carries_the_drafter_it_admitted(self, tmp_path, asked):
        got = _plan(tmp_path, weights_mib = 10_200, n_parallel = asked, spec = "auto")
        assert (got["slots"], got["spec"]) == (1, "draft-mtp")
        assert got["ctx"] == 9_984

    def test_the_gap_narrows_and_never_widens(self, tmp_path):
        """The refit narrows the pre-existing gap."""
        direct = _plan(tmp_path, weights_mib = 10_200, n_parallel = 1, spec = "auto")
        reduced = _plan(tmp_path, weights_mib = 10_200, n_parallel = 4, spec = "auto")
        assert reduced["ctx"] <= direct["ctx"]
        assert direct["ctx"] - reduced["ctx"] < direct["ctx"] - 4096


class TestWhatMustNotMove:
    def test_an_explicit_context_is_still_honored_verbatim(self, tmp_path):
        """An explicit context remains unchanged after slot reduction."""
        got = _plan(tmp_path, weights_mib = 8_400, n_parallel = 4, spec = "off", n_ctx = 32768)
        assert (got["ctx"], got["slots"], got["fit"]) == (32768, 2, "off")
        assert got["ceiling"] == 36_096
        assert "does not fit in this GPU" not in got["warning"]

    def test_an_explicit_context_that_forces_offload_is_unchanged(self, tmp_path):
        """An explicit context that requires offload is unchanged."""
        got = _plan(tmp_path, weights_mib = 10_200, n_parallel = 4, spec = "off", n_ctx = 32768)
        assert (got["ctx"], got["slots"], got["fit"]) == (32768, 4, "on")
        # The ceiling is the offload fallback, tracking _AUTO_OFFLOAD_CTX.
        assert got["ceiling"] == min(llama_cpp._AUTO_OFFLOAD_CTX, NATIVE_CTX)
        assert got["ceiling"] < got["ctx"], "the sheet must still warn above the safe zone"

    def test_a_count_that_needs_no_reduction_is_untouched(self, tmp_path):
        """A plan that needs no slot reduction is unchanged."""
        got = _plan(tmp_path, weights_mib = 6_000, n_parallel = 4, spec = "off")
        assert (got["slots"], got["fit"], got["ctx"]) == (4, "off", 51_456)

    def test_weights_that_fit_nowhere_still_offload(self, tmp_path):
        """Oversized weights fall back to offload at the Auto offload context, read rather than
        hardcoded."""
        got = _plan(tmp_path, weights_mib = 11_400, n_parallel = 4, spec = "off")
        offload_ctx = min(llama_cpp._AUTO_OFFLOAD_CTX, NATIVE_CTX)
        assert (got["fit"], got["ctx"], got["ceiling"]) == ("on", offload_ctx, offload_ctx)
        assert got["slots"] == 4, "the reduction fired; this is no longer the offload case"
        assert got["ctx"] < NATIVE_CTX

    @pytest.mark.parametrize(
        "weights_mib,slots,ctx",
        [
            (8_200, 3, 8_704),
            (9_000, 1, 9_984),
        ],
    )
    def test_a_dense_target_gains_only_its_compute_buffer(self, tmp_path, weights_mib, slots, ctx):
        """Dense models also recover slot-scaled compute-buffer capacity."""
        got = _plan(tmp_path, weights_mib = weights_mib, n_parallel = 4, spec = "off", metadata = DENSE)
        assert (got["slots"], got["ctx"]) == (slots, ctx)


# Multiples of the fit floor: _AUTO_OFFLOAD_CTX never sits below _FIT_MIN_CTX.
_OFFLOAD_SWEEP = [llama_cpp._FIT_MIN_CTX * step for step in (1, 2, 4, 8)]


class TestTheReductionIsPricedAtTheFitFloor:
    """Price the slot search at _FIT_MIN_CTX, not _AUTO_OFFLOAD_CTX: a higher probe fails all-or-nothing."""

    # Rows a reduction rescues from offload (--fit on there means a ~3x decode collapse).
    RESCUED = [
        (10_200, HYBRID),
        (10_400, HYBRID),
        (10_600, HYBRID),
        (9_000, DENSE),
        (9_200, DENSE),
        (9_400, DENSE),
    ]

    @pytest.mark.parametrize("offload_ctx", _OFFLOAD_SWEEP)
    def test_moving_the_offload_fallback_does_not_move_the_placement(
        self, tmp_path, monkeypatch, offload_ctx
    ):
        """Across a sweep of the offload constant, slots, residency, context and devices must not move."""
        monkeypatch.setattr(llama_cpp, "_AUTO_OFFLOAD_CTX", offload_ctx)
        got = _plan(tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off")
        assert (got["slots"], got["fit"], got["ctx"], got["devices"]) == (2, "off", 14_080, "0")

    @pytest.mark.parametrize("weights_mib,metadata", RESCUED)
    @pytest.mark.parametrize("offload_ctx", _OFFLOAD_SWEEP[:2])
    def test_a_load_the_reduction_can_rescue_never_offloads(
        self, tmp_path, monkeypatch, weights_mib, metadata, offload_ctx
    ):
        monkeypatch.setattr(llama_cpp, "_AUTO_OFFLOAD_CTX", offload_ctx)
        got = _plan(tmp_path, weights_mib = weights_mib, n_parallel = 4, spec = "off", metadata = metadata)
        assert got["fit"] == "off", "a placeable load was handed to --fit offload"
        assert got["slots"] < 4

    def test_weights_past_the_band_still_offload(self, tmp_path):
        """Weights past the band must still offload, so the RESCUED rows cannot pass for the wrong
        reason."""
        got = _plan(tmp_path, weights_mib = 11_400, n_parallel = 4, spec = "off")
        assert (got["fit"], got["slots"]) == ("on", 4)

    def test_asking_for_more_slots_never_returns_fewer(self, tmp_path):
        """Monotone in the ask. At head, asking for 2 got 2 and asking for 3 got 1."""
        finals = [
            _plan(tmp_path, weights_mib = _ASK_STILL_MATTERS_MIB, n_parallel = n, spec = "off")["slots"]
            for n in (1, 2, 3, 4, 6, 8)
        ]
        assert finals == sorted(finals), finals
        assert len(set(finals)) > 1, finals


class TestTheSearchPredicate:
    def test_include_requested_answers_for_the_count_itself(self):
        """The helper can test the requested slot count itself."""
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._vocab_size = 248320
        backend._embedding_length = 5120
        backend._key_length_mla = None
        backend._estimate_kv_cache_bytes = lambda *_a, **_k: 0
        backend._can_estimate_kv = lambda: True

        def fit(include_requested):
            return backend._slots_that_fit_on_gpu(
                2,
                4096,
                [(0, 24576)],
                {0: 24576},
                22_500 * MIB,
                "q8_0",
                LlamaCppBackend._GPU_PIN_VRAM_FRACTION,
                0,
                1,
                n_ubatch = 512,
                include_requested = include_requested,
            )

        assert fit(True) == ([0], False, 2)
        assert fit(False) == ([0], False, 1)


class TestTheLoggedReserveFollowsTheLaunch:
    def test_identical_launches_log_the_same_reserve(self, tmp_path, monkeypatch):
        """Equivalent final plans log the same final-slot MTP reserve."""
        from core.inference.llama_cpp import logger as planner_logger

        lines: list[str] = []

        class _Recorder:
            def __getattr__(self, _level):
                def emit(msg, *args, **_kwargs):
                    lines.append(str(msg) % args if args else str(msg))

                return emit

        monkeypatch.setattr("core.inference.llama_cpp.logger", _Recorder())
        assert planner_logger is not None

        # Asks that converge on one plan at this floor; 2 fits outright at _FIT_MIN_CTX 8192.
        seen = []
        for asked in (4, 6, 8):
            lines.clear()
            got = _plan(tmp_path, weights_mib = 7_600, n_parallel = asked, spec = "auto", vram_mib = 9_728)
            reserves = set(re.findall(r"MTP reserve: ([\d.]+) GB", "\n".join(lines)))
            assert reserves, "the reserve was never logged"
            seen.append((got["slots"], got["ctx"], got["spec"], reserves))

        assert seen[0][:3] == (1, 11_520, "draft-mtp")
        assert len({row[:3] for row in seen}) == 1, "the three launches differ"
        assert len({frozenset(row[3]) for row in seen}) == 1, seen
        assert seen[0][3] == {"0.34"}


NO_NATIVE_CTX = {**DENSE, "_context_length": None}


class TestAGgufWithNoNativeContext:
    """Cover slot reduction when native-context metadata is absent."""

    @pytest.mark.parametrize(
        "vram_mib,weights_mib,asked",
        [
            (12 * 1024, 9_000, 8),
            (16 * 1024, 12_000, 8),
        ],
    )
    def test_the_reduction_still_pins_without_a_native_context(
        self, tmp_path, vram_mib, weights_mib, asked
    ):
        """Without native-context metadata the reduction pins at _FIT_MIN_CTX; the exact slot count
        is free."""
        got = _plan(
            tmp_path,
            weights_mib = weights_mib,
            n_parallel = asked,
            spec = "off",
            vram_mib = vram_mib,
            metadata = NO_NATIVE_CTX,
        )
        assert (got["fit"], got["ctx"]) == ("off", llama_cpp._FIT_MIN_CTX)
        assert (
            1 <= got["slots"] < asked
        ), f"the reduction did not fire: asked for {asked}, kept {got['slots']}"

    def test_no_planner_exception_is_swallowed(self, tmp_path, monkeypatch):
        """The broad placement handler must not hide a planner failure."""
        warnings: list[str] = []

        class _Recorder:
            def __getattr__(self, _level):
                def emit(msg, *args, **_kwargs):
                    warnings.append(str(msg) % args if args else str(msg))

                return emit

        monkeypatch.setattr("core.inference.llama_cpp.logger", _Recorder())
        _plan(
            tmp_path,
            weights_mib = 9_000,
            n_parallel = 8,
            spec = "off",
            vram_mib = 12 * 1024,
            metadata = NO_NATIVE_CTX,
        )
        assert not [w for w in warnings if "GPU selection failed" in w], warnings

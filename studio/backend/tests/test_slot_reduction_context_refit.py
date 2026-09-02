# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for context fitting after serving-slot reduction.

The synthetic plans use Qwen3.8-27B metadata and capture the generated command.
Each case names its speculative mode because unknown values resolve to Auto.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from test_llama_cpp_placement import _backend, _launch  # noqa: E402

MIB = 1024 * 1024
NATIVE_CTX = 262144
CARD_MIB = 12 * 1024  # usable 11,919 MiB at the 0.97 pin fraction

# The weights these cases are built on, named for the slot count the reduction
# leaves them at on CARD_MIB when four are asked for. The reduction only has
# something to do inside a band, and the band moves with _FIT_MIN_CTX, since the
# search prices every candidate slot count at the floor: raising it 4096 -> 8192
# moved the band down about 1,400 MiB and left the old weights either fitting
# whole at the floor under `--fit on` or collapsing to one slot for every ask.
# These were re-measured against the floor rather than nudged until green -- each
# keeps the role its test needs (a three-slot survivor, a two-slot survivor, and
# an ask that still discriminates), which is what the assertions below are about.
_KEEPS_THREE_MIB = 9_200
_KEEPS_TWO_MIB = 9_800
_ASK_STILL_MATTERS_MIB = 9_400
MIXED_CARDS = (CARD_MIB, 1_280)  # a primary plus a card too small to plan onto

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
):
    """Return the generated placement plan. ``vram_mib`` may be a per-card sequence."""
    cards = list(vram_mib) if isinstance(vram_mib, (tuple, list)) else [vram_mib] * gpus
    memory = [(i, mib, mib) for i, mib in enumerate(cards)]
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)

    def read(_path):
        for key, value in metadata.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: weights_mib * MIB
    del backend._can_estimate_kv  # the real one, now that the dims are set
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        # Keep the requested slot count available to the planner.
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
        "devices": (launched["env"] or {}).get("CUDA_VISIBLE_DEVICES"),
    }


class TestTheLaunchedCountOwnsTheContext:
    """Speculation is off so only slot reduction affects the context."""

    @pytest.mark.parametrize(
        "weights_mib,asked,slots,ctx",
        [
            (_KEEPS_THREE_MIB, 4, 3, 12_288),
            (_KEEPS_TWO_MIB, 4, 2, 13_824),
            (_KEEPS_TWO_MIB, 8, 2, 13_824),  # the same answer from a larger ask
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
        """The published ceiling follows the final slot count.

        Named for the floor rather than 4096 because the anchor it must not be is
        whatever _FIT_MIN_CTX currently is: the reduction prices its search there,
        so a ceiling that came back equal to the floor would mean the search result
        was published instead of the context the final slot count actually affords.
        """
        got = _plan(tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off")
        assert got["ceiling"] == 13_824
        assert got["ceiling"] != llama_cpp._FIT_MIN_CTX

    def test_a_layer_split_across_two_cards_re_fits_the_same_way(self, tmp_path):
        """The same invariant holds for a two-card layer split."""
        reduced = _plan(
            tmp_path, weights_mib = 14_000, n_parallel = 4, spec = "off", vram_mib = 8_704, gpus = 2
        )
        direct = _plan(
            tmp_path, weights_mib = 14_000, n_parallel = 2, spec = "off", vram_mib = 8_704, gpus = 2
        )
        assert (reduced["slots"], reduced["fit"]) == (2, "off")
        assert reduced["ctx"] == reduced["ceiling"] == 8_448
        assert (reduced["ctx"], reduced["ceiling"]) == (direct["ctx"], direct["ceiling"])


class TestTheRefitStaysOnTheCardsTheReductionChose:
    """A 12 GiB primary next to a 1.25 GiB card: the reduced plan fits on the primary
    alone, and only the small card can hold a longer context."""

    def test_the_refit_does_not_pull_in_another_card(self, tmp_path):
        got = _plan(
            tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off", vram_mib = MIXED_CARDS
        )
        assert (got["slots"], got["fit"], got["devices"]) == (2, "off", "0")
        assert got["ctx"] == 13_824

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
        assert mixed["ctx"] == alone["ctx"] == 13_824
        assert (mixed["ceiling"], alone["ceiling"]) == (14_848, 13_824)


class TestAutoSpeculationStillDecidesBeforeTheReduction:
    """Pin the existing behavior where Auto admits MTP before slot reduction."""

    def test_a_direct_one_slot_request_drops_the_drafter_and_keeps_context(self, tmp_path):
        got = _plan(tmp_path, weights_mib = 10_200, n_parallel = 1, spec = "auto")
        assert (got["slots"], got["ctx"], got["spec"]) == (1, 18_688, "ngram-mod")

    @pytest.mark.parametrize("asked", [4, 8])
    def test_a_reduced_request_still_carries_the_drafter_it_admitted(self, tmp_path, asked):
        got = _plan(tmp_path, weights_mib = 10_200, n_parallel = asked, spec = "auto")
        assert (got["slots"], got["spec"]) == (1, "draft-mtp")
        # The retained MTP reserve keeps this below the direct one-slot context.
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
        # The measured ceiling still follows the final slot count.
        assert got["ceiling"] == 35_840

    def test_an_explicit_context_that_forces_offload_is_unchanged(self, tmp_path):
        """An explicit context that requires offload is unchanged."""
        got = _plan(tmp_path, weights_mib = 10_200, n_parallel = 4, spec = "off", n_ctx = 32768)
        # The launched context stays a literal: honouring the request verbatim IS the
        # contract, so a derived value here would assert nothing.
        assert (got["ctx"], got["slots"], got["fit"]) == (32768, 4, "on")
        # The ceiling is the published safe zone, which no reduction reached, so it is
        # the offload fallback. Tracks the constant; #9492 moved it 4096 -> 8192.
        assert got["ceiling"] == min(llama_cpp._AUTO_OFFLOAD_CTX, NATIVE_CTX)
        assert got["ceiling"] < got["ctx"], "the sheet must still warn above the safe zone"

    def test_a_count_that_needs_no_reduction_is_untouched(self, tmp_path):
        """A plan that needs no slot reduction is unchanged."""
        got = _plan(tmp_path, weights_mib = 6_000, n_parallel = 4, spec = "off")
        assert (got["slots"], got["fit"], got["ctx"]) == (4, "off", 51_200)

    def test_weights_that_fit_nowhere_still_offload(self, tmp_path):
        """Oversized weights still fall back to offload, at the Auto offload context.

        The one case in this file that reads the constant rather than the reduction:
        no slot count places these weights, so the block below never fires and the
        context is the fallback itself. Tracked rather than hardcoded, since #9492
        moved it from 4096 to 8192 and a literal here is a re-edit every time.
        """
        got = _plan(tmp_path, weights_mib = 11_400, n_parallel = 4, spec = "off")
        offload_ctx = min(llama_cpp._AUTO_OFFLOAD_CTX, NATIVE_CTX)
        assert (got["fit"], got["ctx"], got["ceiling"]) == ("on", offload_ctx, offload_ctx)
        # The case is still the one this name claims: nothing was rescued, and the
        # native length really was cut down to the fallback.
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


# Multiples of the fit floor rather than literals. _AUTO_OFFLOAD_CTX is documented
# to sit at or above _FIT_MIN_CTX, so sweeping it below the floor would drive the
# planner through a state the product never reaches and call whatever came back a
# regression. Identical to [4096, 8192, 16384, 32768] while the floor is 4096, and
# still the intended sweep after it moves.
_OFFLOAD_SWEEP = [llama_cpp._FIT_MIN_CTX * step for step in (1, 2, 4, 8)]


class TestTheReductionIsPricedAtTheFitFloor:
    """The search that picks the slot count must not be priced at _AUTO_OFFLOAD_CTX.

    That constant is what Auto settles for once offload is unavoidable. This block
    exists to overturn that verdict by re-asking at fewer slots, so a higher probe
    context can only make the search fail -- and its failure is all-or-nothing: no
    reduction, `--fit on`, and the slot count kept at the ask it could not afford.

    #9492 raised the constant 4096 -> 8192 and, through this one shared read, moved
    placement. The arithmetic, on the 12 GiB card these tests use (budget
    12288 x 0.97 = 11,919.36 MiB) at the 10,200 MiB of weights this block was first
    written against: the candidates are 707 MiB apart (557.75 MiB of per-slot
    compute buffer plus 149.625 MiB of per-slot Mamba state), while probing 256
    tokens higher costs a uniform 64 KiB/token. The 2-slot candidate was 11,685.0
    MiB at a 4096 probe and 11,947.0 MiB at 8192, so a display constant moved it
    across the budget by 27.6 MiB. The weights above are lighter now only because
    _FIT_MIN_CTX itself moved to 8192 and carried the reducible band down with it;
    the margins being this thin is the reason the sweep below exists.
    """

    # A band that a reduction rescues from offload. Every one of these launched
    # `--fit off` before #9492 and `--fit on` after it, which is the ~3x decode
    # collapse (#6718) the reduction was written to avoid.
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
        """Sweeping the constant leaves slots, residency, context and cards alone.

        Pinned against the sweep rather than against one value, so this cannot be
        satisfied by re-baselining the constant into the expectation.
        """
        monkeypatch.setattr(llama_cpp, "_AUTO_OFFLOAD_CTX", offload_ctx)
        got = _plan(tmp_path, weights_mib = _KEEPS_TWO_MIB, n_parallel = 4, spec = "off")
        assert (got["slots"], got["fit"], got["ctx"], got["devices"]) == (2, "off", 13_824, "0")

    @pytest.mark.parametrize("weights_mib,metadata", RESCUED)
    @pytest.mark.parametrize("offload_ctx", _OFFLOAD_SWEEP[:2])
    def test_a_load_the_reduction_can_rescue_never_offloads(
        self, tmp_path, monkeypatch, weights_mib, metadata, offload_ctx
    ):
        monkeypatch.setattr(llama_cpp, "_AUTO_OFFLOAD_CTX", offload_ctx)
        got = _plan(tmp_path, weights_mib = weights_mib, n_parallel = 4, spec = "off", metadata = metadata)
        assert got["fit"] == "off", "a placeable load was handed to --fit offload"
        # It was rescued BY the reduction rather than fitting outright, or the row
        # would prove nothing about this block.
        assert got["slots"] < 4

    def test_weights_past_the_band_still_offload(self, tmp_path):
        """The counterweight: the rows above are not green because everything is.

        Without this, RESCUED could drift into sizes that place at the full ask and
        the assertions there would hold for the wrong reason.
        """
        got = _plan(tmp_path, weights_mib = 11_400, n_parallel = 4, spec = "off")
        assert (got["fit"], got["slots"]) == ("on", 4)

    def test_asking_for_more_slots_never_returns_fewer(self, tmp_path):
        """Monotone in the ask. At head, asking for 2 got 2 and asking for 3 got 1."""
        finals = [
            _plan(tmp_path, weights_mib = _ASK_STILL_MATTERS_MIB, n_parallel = n, spec = "off")["slots"]
            for n in (1, 2, 3, 4, 6, 8)
        ]
        assert finals == sorted(finals), finals
        # Not a row of identical numbers, which would sort trivially.
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
        assert planner_logger is not None  # the real one is restored on teardown

        # Asks that converge on one plan at this floor. 2 is not among them any
        # more: at _FIT_MIN_CTX 8192 two slots fit on this card outright, so the
        # ask survives untouched and Auto settles on ngram-mod with no MTP reserve
        # to log. That is a different launch, not a differing reserve, so pinning
        # it here would test the fixture rather than the claim.
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


# KV-estimable metadata without a native context length.
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
        """Without native-context metadata the search has no length to expand to,
        so the reduction pins at the fit floor itself.

        The floor is read, not spelled 4096, for the reason
        test_weights_that_fit_nowhere_still_offload gives about _AUTO_OFFLOAD_CTX:
        a literal here is a re-edit every time the constant moves, and the re-edit
        is indistinguishable from noticing that placement changed. The exact
        surviving slot count is left unpinned for the same reason -- it is
        arithmetic against the floor, not the claim in this test's name. What must
        hold is that a reduction happened at all and that it stayed resident.
        """
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

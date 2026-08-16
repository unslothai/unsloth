# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The projector CPU pin (#8967) and the embedded-MTP stand-down (#8875) meeting.

Both mutate the same placement decision and each shipped with tests that exercise
it alone. They run in one region and in this order:

  1. the automatic projector pin                    (llama_cpp.py ~14624-14760)
  2. the drafter-VRAM drop probe                    (~14771-15010)
  3. ``_placement_verdict_partial``                 (~15612)
  4. the stand-down that reads it                   (~16190)

So the pin is decided FIRST and everything after it reads a budget the pin has
already changed. That is the right order -- but it means the pin has to price the
drafter's reserve itself, because the fit that must deliver the residency does
(``_soft_overhead`` + ``_pin_fraction`` + ``_mtp_bytes``). Before the fix it did
not, and the omission was worth ~1.5 GB in both directions on the model here:

  * it declined to pin loads where pinning would have kept every layer AND the
    drafter, and the drafter was dropped instead (``drafter_no_vram``);
  * it pinned loads that then went out ``--fit on`` anyway, so the ~3.6x per-image
    encode was paid for a residency the launch never received -- and where the
    stand-down also fired, MTP was lost as well. Both stand-downs, no upside.

Everything is simulated. The compute-buffer estimator is pinned to 100 MiB for
the reason ``test_mmproj_pin_platform_matrix`` pins it: the real one answers in
gigabytes against a stub GGUF, which swamps every other term so nothing fits
anywhere and a policy test passes whatever the policy does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuses the placement harness (and its module stubs). Import before anything
# from core.inference.
from test_llama_cpp_placement import _backend, _launch  # noqa: E402,F401

from core.inference import llama_cpp  # noqa: E402

_GB = 1024**3
_MIB = 1024 * 1024
_PIN = "--no-mmproj-offload"
# The automatic pin's own log line (llama_cpp.py ~14755). Distinguishes "never
# pinned" from "pinned, then refunded", which the argv alone cannot.
_PIN_TAKEN = "Auto: running the vision projector on the CPU"

# One 24 GB-class card. Only `free` varies across the cells below, so the model,
# the projector and the drafter are the same question asked at every tier.
_TOTAL_MIB = 24_576
_MODEL_GB = 8.0
_MMPROJ_GB = 1.5
_DRAFTER_GB = 1.5
_KV_GB = 1.0


def _hybrid_vision_backend(
    tmp_path: Path,
    *,
    free_mib: int,
    vision: bool = True,
    drafter: bool = True,
    can_estimate_kv: bool = True,
    total_mib: int = _TOTAL_MIB,
):
    """A vision + embedded-MTP hybrid-Mamba model on one discrete card.

    ``_select_gpus`` is deliberately NOT stubbed: the point of these cells is
    what the real fit decides after the pin has moved the numbers.
    """
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, free_mib, total_mib)])
    backend._get_gguf_size_bytes = lambda _path: int(_MODEL_GB * _GB)
    # See the module docstring: the real estimator swamps every other term.
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * _MIB
    backend._can_estimate_kv = lambda: can_estimate_kv
    backend._estimate_kv_cache_bytes = lambda *a, **k: int(_KV_GB * _GB)
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._mtp_draft_kv_bytes = lambda *a, **k: 0
    backend._estimate_mtp_overhead_bytes = lambda *a, **k: int(_DRAFTER_GB * _GB)

    caps = {"supports_no_mmproj_offload": True, "supports_kv_unified": True}
    if drafter:
        caps.update(
            {
                "mtp_token": "draft-mtp",
                "supports_ngram_mod": True,
                "spec_draft_n_max_flag": "--spec-draft-n-max",
            }
        )

    def read_metadata(_path):
        backend._n_layers = 65
        backend._n_kv_heads = 4
        backend._n_heads = 24
        backend._embedding_length = 5120
        backend._kv_key_length = 256
        backend._kv_value_length = 256
        if drafter:
            # The #8875 shape: an embedded MTP head on a hybrid-Mamba trunk.
            backend._nextn_predict_layers = 1
            backend._full_attention_interval = 4
            backend._ssm_inner_size = 6144
            backend._ssm_state_size = 128
            backend._ssm_group_count = 16
            backend._ssm_conv_kernel = 4

    backend._read_gguf_metadata = read_metadata
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)

    if vision:
        mmproj = tmp_path / "model-mmproj.gguf"
        mmproj.write_bytes(b"\x00" * 16)
        backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
        backend._mmproj_vram_bytes = lambda _path: int(_MMPROJ_GB * _GB)
        backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _outcome(
    tmp_path: Path,
    *,
    free_mib: int,
    speculative_type: str = "auto",
    **kwargs,
):
    backend, gguf = _hybrid_vision_backend(tmp_path, free_mib = free_mib, **kwargs)
    cmd = _launch(
        backend,
        gguf,
        is_vision = kwargs.get("vision", True),
        speculative_type = speculative_type,
        n_ctx = 4096,
    )["cmd"]

    def _val(flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    return {
        "pin": _PIN in cmd,
        "fit": _val("--fit"),
        "spec": _val("--spec-type"),
        "reason": backend.spec_fallback_reason,
        "vision_on_cpu": backend.vision_on_cpu,
        "cmd": cmd,
    }


# The three tiers the interaction turns on, for this model on this card. Usable
# budget is `free - max((1-frac)*total, floor)`, i.e. free - 737 MiB at 0.97.
#
#   11_000: nothing fits with the projector on the GPU; the model alone does.
#   13_000: model + projector fit; model + projector + drafter do not; model +
#           drafter do. This is the tier the pin was blind to.
#   20_000: everything fits with room to spare.
_TIER_PIN_BUYS_RESIDENCY_ONLY = 11_000
_TIER_PIN_BUYS_THE_DRAFTER = 13_000
_TIER_ROOMY = 20_000


# --------------------------------------------------------------------------
# Hypothesis 2 / 3: the pin flips partial to full, so the drafter survives
# --------------------------------------------------------------------------


def test_the_pin_prices_the_drafter_so_it_can_save_it(tmp_path):
    """Projector + drafter do not fit; drafter alone does. Pin, and keep MTP.

    The pin's own doctrine -- a bounded per-image cost against a per-token one --
    applies to the drafter as much as to a spilled layer. Priced without the
    drafter's reserve the pin saw a model that fit, declined, and the drop probe
    then took the drafter away instead.
    """
    result = _outcome(tmp_path, free_mib = _TIER_PIN_BUYS_THE_DRAFTER)

    assert result["pin"] is True
    assert result["cmd"].count(_PIN) == 1
    # Pinned, not disabled: the projector still loads, on the CPU.
    assert "--mmproj" in result["cmd"]
    assert result["vision_on_cpu"] is True
    # The whole point: full residency, and the drafter kept.
    assert result["fit"] == "off"
    assert result["spec"] == "draft-mtp"
    assert result["reason"] is None


def test_the_stand_down_reads_the_verdict_the_pin_left(tmp_path):
    """#8875's verdict is computed after the pin, so a pin that buys the fit
    keeps MTP. Guards the ordering: a verdict sampled before the pin would be
    the stale kind #8875 was fixing in the first place."""
    result = _outcome(tmp_path, free_mib = _TIER_PIN_BUYS_THE_DRAFTER)

    assert result["pin"] is True
    assert result["spec"] != "none"
    assert result["reason"] != "mtp_partial_offload"


# --------------------------------------------------------------------------
# Hypothesis 4: never both stand-downs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "free_mib",
    [9_000, 10_000, _TIER_PIN_BUYS_RESIDENCY_ONLY, 12_000, _TIER_PIN_BUYS_THE_DRAFTER, 15_000],
)
def test_the_projector_is_never_pinned_for_a_residency_it_does_not_get(tmp_path, free_mib):
    """The pin is a bet on the placement. `--fit on` means the bet lost.

    Emitting both is the worst of the two: the per-image encode is paid for and
    llama-server is still free to spill layers. Where the stand-down fires on the
    same load, MTP goes too -- both costs, neither benefit.
    """
    for spec in ("auto", "mtp"):
        for kv in (True, False):
            result = _outcome(
                tmp_path,
                free_mib = free_mib,
                speculative_type = spec,
                can_estimate_kv = kv,
            )
            if result["pin"]:
                assert result["fit"] == "off", (
                    f"pinned the projector but still launched --fit on "
                    f"(free={free_mib}, spec={spec}, can_estimate_kv={kv})"
                )


def test_a_forced_drafter_is_not_pinned_against_a_reserve_it_cannot_drop(tmp_path, capfd):
    """Forced mtp: the drop probe is gated off, so the reserve survives to the
    fit whatever the pin decides. Pinning cannot buy residency here, so it must
    not be paid for.

    Asserted on the decision, not only on the argv. "Never taken" and "taken,
    then handed back by the post-placement check" produce the same command line,
    and the first is what the pin's own rank is responsible for -- the refund is
    the backstop, not the answer.
    """
    result = _outcome(tmp_path, free_mib = _TIER_PIN_BUYS_RESIDENCY_ONLY, speculative_type = "mtp")
    _out = capfd.readouterr()

    assert result["fit"] == "on"
    assert result["pin"] is False
    assert result["vision_on_cpu"] is False
    assert _PIN_TAKEN not in (_out.out + _out.err)
    # Forced, so it survives a partial placement by design (#8875).
    assert result["spec"] == "draft-mtp"


def test_an_unsizable_kv_does_not_pin_and_stand_down_at_once(tmp_path, capfd):
    """No KV estimate -> no drop probe, so the drafter's reserve reaches the fit.

    This is the cell that produced both stand-downs at once: the pin fired on a
    drafter-blind estimate, the fit failed on the reserve, and #8875 then read
    that (correct) partial verdict and disabled MTP.
    """
    result = _outcome(tmp_path, free_mib = _TIER_PIN_BUYS_RESIDENCY_ONLY, can_estimate_kv = False)
    _out = capfd.readouterr()

    assert _PIN_TAKEN not in (_out.out + _out.err)
    assert not (
        result["pin"] and result["spec"] == "none"
    ), "paid the per-image cost AND lost the drafter"
    assert result["pin"] is False
    assert result["spec"] == "none"
    assert result["reason"] == "mtp_partial_offload"


def test_a_placement_the_pooled_probe_did_not_predict_hands_the_pin_back(tmp_path):
    """``_mmproj_fits`` is a pooled prediction; the loop is not.

    The loop additionally enforces a per-device reserve (``_every_gpu_holds_reserve``),
    starts at ``_layer_min_gpus`` rather than one GPU, and can exhaust its candidates
    outright -- none of which the pooled probe models, so the two can disagree no
    matter how the probe is priced. The placement returning ``--fit on`` is
    simulated directly here (the same way #8875's own cells simulate a partial
    placement) because the point is the verification, not the arithmetic: whatever
    made the loop refuse, the pin has to be handed back rather than paid for.
    """
    backend, gguf = _hybrid_vision_backend(tmp_path, free_mib = _TIER_PIN_BUYS_THE_DRAFTER)
    # Sanity: this tier pins when the loop agrees (see the test above), so the
    # pin really is on the table before the placement refuses.
    backend._select_gpus = lambda *a, **k: (None, True)
    backend._select_gpus_split_aware = lambda *a, **k: (None, True)

    cmd = _launch(backend, gguf, is_vision = True, speculative_type = "auto", n_ctx = 4096)["cmd"]

    assert cmd[cmd.index("--fit") + 1] == "on"
    assert _PIN not in cmd
    assert backend.vision_on_cpu is False
    # Vision is never dropped, only placed.
    assert "--mmproj" in cmd
    assert backend.is_vision is True
    # And #8875 still stands MTP down on that genuinely partial verdict.
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


# --------------------------------------------------------------------------
# The pin's original job, unchanged
# --------------------------------------------------------------------------


def test_the_pin_still_fires_where_only_residency_is_on_offer(tmp_path):
    """Too tight for the drafter either way, but the projector still tips the
    layers over. Pin, take full residency, and let the probe drop the drafter --
    the behaviour the pin shipped with, which the drafter pricing must not undo.
    """
    result = _outcome(tmp_path, free_mib = _TIER_PIN_BUYS_RESIDENCY_ONLY)

    assert result["pin"] is True
    assert result["fit"] == "off"
    assert result["spec"] == "ngram-mod"
    assert result["reason"] == "drafter_no_vram"


def test_a_roomy_card_pins_nothing_and_keeps_everything(tmp_path):
    result = _outcome(tmp_path, free_mib = _TIER_ROOMY)

    assert result["pin"] is False
    assert result["fit"] == "off"
    assert result["spec"] == "draft-mtp"


# --------------------------------------------------------------------------
# Hypothesis 6: no cross-contamination in the single-feature cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "free_mib, pin, fit",
    [
        (10_000, False, "on"),
        (_TIER_PIN_BUYS_RESIDENCY_ONLY, True, "off"),
        (12_500, True, "off"),
        (_TIER_PIN_BUYS_THE_DRAFTER, False, "off"),
        (_TIER_ROOMY, False, "off"),
    ],
)
def test_a_vision_model_with_no_drafter_is_untouched(tmp_path, free_mib, pin, fit):
    """No drafter, no reserve, so the drafter pricing must be a no-op: the pin
    fires exactly where #8967's own tests say it does."""
    result = _outcome(tmp_path, free_mib = free_mib, drafter = False)

    assert result["pin"] is pin
    assert result["fit"] == fit
    assert result["spec"] is None


@pytest.mark.parametrize(
    "free_mib, fit, spec, reason",
    [
        (10_000, "on", "none", "mtp_partial_offload"),
        (_TIER_PIN_BUYS_RESIDENCY_ONLY, "off", "ngram-mod", "drafter_no_vram"),
        (_TIER_PIN_BUYS_THE_DRAFTER, "off", "draft-mtp", None),
        (_TIER_ROOMY, "off", "draft-mtp", None),
    ],
)
def test_a_drafter_with_no_vision_is_untouched(tmp_path, free_mib, fit, spec, reason):
    """No projector, so nothing for the pin to move and #8875 answers alone."""
    result = _outcome(tmp_path, free_mib = free_mib, vision = False)

    assert result["pin"] is False
    assert result["fit"] == fit
    assert result["spec"] == spec
    assert result["reason"] == reason


# --------------------------------------------------------------------------
# Monotonicity, and #8589's tunable budget
# --------------------------------------------------------------------------

# Ordered worst to best. Two features that each mutate the placement can only be
# composed safely if a bigger card never buys a worse launch.
_LADDER = [
    9_000,
    10_000,
    10_500,
    11_000,
    11_500,
    12_000,
    12_500,
    13_000,
    13_500,
    14_000,
    15_000,
    16_000,
    _TIER_ROOMY,
]


def _rank(result) -> int:
    if result["fit"] != "off":
        return 0
    return 2 if result["spec"] == "draft-mtp" else 1


@pytest.mark.parametrize("fraction", [None, 0.85, 0.92, 1.0])
def test_more_free_vram_never_buys_a_worse_launch(monkeypatch, tmp_path, fraction):
    """Across the whole ladder, and at a non-default budget fraction (#8589).

    A drafter-blind pin made this non-monotone: at one tier the pin fired and
    bought full residency, and at the NEXT tier up it declined, the fit failed on
    the reserve it had not priced, and the launch came out partial with MTP off.
    """
    if fraction is not None:
        monkeypatch.setattr(llama_cpp, "_active_vram_fraction", lambda: fraction)

    seen = []
    for free_mib in _LADDER:
        result = _outcome(tmp_path, free_mib = free_mib)
        seen.append((free_mib, _rank(result), result["pin"], result["fit"], result["spec"]))
        # The pin's premise, at every fraction: it only ever buys full residency.
        if result["pin"]:
            assert result["fit"] == "off", f"pin without residency at {seen[-1]}"

    ranks = [r for _f, r, _p, _fit, _s in seen]
    assert ranks == sorted(ranks), f"non-monotone: {seen}"
    # The ladder has to actually cross the tiers, or monotonicity is vacuous.
    assert len(set(ranks)) >= 2, f"ladder never changed outcome at fraction={fraction}: {seen}"

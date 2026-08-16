# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The projector pin must rank GPUs at the fraction it is about to price them at.

``_gpu_usable(g, frac)`` says so in its own comment -- "callers pass the ACTIVE
fraction so the ranking matches the budget the fit then tests, else mixed totals
mis-order" -- and the MTP drop probe obeys it by carrying two ``_probe_orders``.
The projector pin did not: it built its candidate subsets from one ranking at the
DEFAULT ``_vram_frac`` and then priced them at ``_mm_mtp_frac``, five points
lower whenever an unsized drafter is GPU-resident.

Only heterogeneous pools can see it. With ``total > 0`` on every card (which
``_discrete_vram`` guarantees inside the pin block) the budget is
``free - (1 - f) * total``, so the gap between two cards is
``dF - (1 - f) * dT`` and shifting ``f`` by 0.05 reorders them whenever
``0.03 * dT < dF < 0.08 * dT`` -- a window 0.05 * dT wide, ~2.8 GB for an
80 GB + 24 GB pair. It is empty only when ``dT == 0``, which is why the existing
multi-GPU pools (uniform totals) never caught it.

The consequence is not absorbed downstream. ``_mm_any`` is an ``any`` over
prefixes, so the wrong order only costs a fit when the good singleton is missed
AND the whole pool is too expensive -- but that is the common case here, because
a second device costs ``_pipeline_overhead_bytes`` (1024 MiB). The pin then
declines, the load ships ``--fit on`` with layers on the CPU, and nothing
re-opens it: the only pin re-check (``_mmproj_auto_pinned and use_fit``) can
hand a pin BACK, never take one out. It misfires the other way too, pinning
loads that go out ``--fit off`` and pay the projector's ~3.6x image encode for a
residency they already had.

Everything is simulated at the probe boundary, as the rest of the mmproj suites
do; this host has one GPU. ``_estimate_compute_buffer_bytes`` is pinned to
100 MiB because the real estimator returns several GB against the stub GGUF and
swamps every other term, which makes a pin test unable to fail.
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

_MIB = 1024 * 1024
_PIN = "--no-mmproj-offload"

# GPU0 an 80 GB card with other work on it, GPU1 an idle 24 GB card. The first
# leads at the default fraction and the second leads once the drafter's five
# points come off, which is the whole point of the pool.
_SKEWED_POOL = [(0, 7_000, 81_920), (1, 5_000, 24_576)]
# Same free memory, same totals: no inversion is possible, so the fix must be a
# no-op here.
_UNIFORM_POOL = [(0, 7_000, 24_576), (1, 5_000, 24_576)]

_PROJECTOR_MIB = 900
# Inside the band where only the drafter-aware ranking finds a placement:
# need(1 GPU) fits on GPU1's 0.92 budget, need(2 GPUs) does not fit anywhere
# once the 1024 MiB pipeline overhead is added.
_MODEL_MIB_NEEDS_THE_SECOND_RANKING = 2_000
# Below the band: GPU1 alone holds the model AND the projector, so pinning buys
# nothing.
_MODEL_MIB_FITS_WITH_THE_PROJECTOR = 1_000


def _usable(free_mib, total_mib, frac):
    return llama_cpp._vram_usable_mib(free_mib, total_mib, frac)


def _rank(pool, frac):
    return [g[0] for g in sorted(pool, key = lambda g: _usable(g[1], g[2], frac), reverse = True)]


def _vision_backend(
    tmp_path,
    *,
    model_mib,
    pool,
    drafter = True,
):
    """A vision GGUF on a multi-GPU pool, with an UNSIZED GPU drafter.

    Unsized is the load-bearing half: ``_estimate_mtp_overhead_bytes`` returning
    None leaves ``mtp_overhead_fn`` None, which is the condition under which the
    pin falls back to the flat ``_MTP_VRAM_RESERVE_FRAC`` off the fraction rather
    than a byte reserve. A suite that sizes the draft KV never reaches the
    fractional path and so cannot see this at all.
    """
    backend, gguf = _backend(tmp_path, vulkan = False, memory = list(pool))
    backend._get_gguf_size_bytes = lambda _path: model_mib * _MIB
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * _MIB
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._mtp_draft_kv_bytes = lambda *a, **k: None
    backend._estimate_mtp_overhead_bytes = lambda *a, **k: None
    # No KV estimate, so _mtp_drop_probe_applies is False and _mm_rank collapses
    # to {0, 2}: the assertions then read the ranking and nothing else.
    backend._can_estimate_kv = lambda: False

    caps = {"supports_no_mmproj_offload": True}
    if drafter:
        caps.update({"mtp_token": "draft-mtp", "supports_ngram_mod": True})
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)

    def _read_metadata(_path):
        if drafter:
            backend._nextn_predict_layers = 1

    backend._read_gguf_metadata = _read_metadata

    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: _PROJECTOR_MIB * _MIB
    backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _run(
    tmp_path,
    *,
    model_mib,
    pool = _SKEWED_POOL,
    drafter = True,
):
    backend, gguf = _vision_backend(tmp_path, model_mib = model_mib, pool = pool, drafter = drafter)
    cmd = _launch(backend, gguf, is_vision = True, speculative_type = "mtp" if drafter else "off")[
        "cmd"
    ]
    return {
        "cmd": cmd,
        "pin": _PIN in cmd,
        "fit": cmd[cmd.index("--fit") + 1] if "--fit" in cmd else None,
        "vision_on_cpu": backend.vision_on_cpu,
    }


# --------------------------------------------------------------------------


def test_the_two_fractions_really_do_rank_this_pool_differently():
    """The premise, asserted rather than assumed.

    If this ever stops holding -- a changed reserve, a changed floor -- the two
    launch tests below would still pass while testing nothing, so the pool is
    pinned here instead of in a comment.
    """
    default = llama_cpp._CTX_FIT_VRAM_FRACTION
    drafter = default - llama_cpp._MTP_VRAM_RESERVE_FRAC

    assert _rank(_SKEWED_POOL, default) == [0, 1]
    assert _rank(_SKEWED_POOL, drafter) == [1, 0]
    # The uniform control cannot invert at any fraction: equal totals cancel.
    assert _rank(_UNIFORM_POOL, default) == _rank(_UNIFORM_POOL, drafter) == [0, 1]


@pytest.mark.parametrize(
    "label, pool",
    [
        # Ordinary consumer and workstation mixes, not just the 80 GB case the
        # launch tests use. The window is 0.05 * (difference in total), so it is
        # wider the more the cards differ, and 24 + 8 is already 819 MiB of free
        # memory wide.
        ("24+8", [(0, 5_000, 24_576), (1, 4_000, 8_192)]),
        ("48+8", [(0, 6_000, 49_152), (1, 4_200, 8_192)]),
        # Three cards, and the order reverses completely.
        ("24+16+8", [(0, 5_000, 24_576), (1, 4_400, 16_384), (2, 4_000, 8_192)]),
    ],
)
def test_other_ordinary_pools_invert_too(label, pool):
    """The 80 GB pool is not a special case constructed to break this."""
    default = llama_cpp._CTX_FIT_VRAM_FRACTION
    assert _rank(pool, default) != _rank(pool, default - llama_cpp._MTP_VRAM_RESERVE_FRAC), label


def test_the_pin_finds_the_card_only_the_drafter_ranking_leads_with(tmp_path):
    """The regression.

    GPU1 alone holds every layer once the projector moves to the CPU, but it is
    second at the default fraction, so a single ranking offers ``[GPU0]`` and
    then ``[GPU0, GPU1]`` -- neither of which fits, the pair having to pay
    1024 MiB of pipeline overhead. Ranked at the fraction the placement loop
    itself sorts under, ``[GPU1]`` is on the table and the pin is taken.

    ``--fit off`` is the proof that the pin bought a real placement: the planner
    placed the model and lowered the flag. ``--fit on`` is the starting value and
    means no placement was made at all.
    """
    result = _run(tmp_path, model_mib = _MODEL_MIB_NEEDS_THE_SECOND_RANKING)

    assert result["pin"] is True
    assert result["fit"] == "off"
    assert result["vision_on_cpu"] is True
    # Pinned, not disabled: the projector still loads, on the CPU.
    assert "--mmproj" in result["cmd"]


def test_no_pin_where_the_projector_already_fits_on_the_ranked_card(tmp_path):
    """The other direction, on the same pool.

    A smaller model fits WITH the projector on GPU1, so pinning buys nothing and
    only costs image-encode speed. Ranking at the wrong fraction hid that too --
    it priced GPU0's 446 MiB and concluded nothing fit either way.
    """
    result = _run(tmp_path, model_mib = _MODEL_MIB_FITS_WITH_THE_PROJECTOR)

    assert result["pin"] is False
    assert result["fit"] == "off"
    assert result["vision_on_cpu"] is False


@pytest.mark.parametrize(
    "model_mib", [_MODEL_MIB_FITS_WITH_THE_PROJECTOR, _MODEL_MIB_NEEDS_THE_SECOND_RANKING]
)
def test_a_uniform_pool_is_untouched(tmp_path, model_mib):
    """Control: equal totals, so both fractions rank the same and the second
    ranking is never even built. Pins the fix as a no-op on the pools every other
    suite uses."""
    result = _run(tmp_path, model_mib = model_mib, pool = _UNIFORM_POOL)

    assert result["pin"] is False
    assert result["fit"] == "off"
    assert result["vision_on_cpu"] is False


@pytest.mark.parametrize(
    "model_mib", [_MODEL_MIB_FITS_WITH_THE_PROJECTOR, _MODEL_MIB_NEEDS_THE_SECOND_RANKING]
)
def test_no_drafter_means_one_fraction_and_no_change(tmp_path, model_mib):
    """Control: with no GPU drafter there is no reserve, ``_mm_mtp_frac`` IS
    ``_vram_frac``, and the skewed pool behaves exactly as it did before."""
    result = _run(tmp_path, model_mib = model_mib, drafter = False)

    assert result["pin"] is False
    assert result["fit"] == "off"
    assert result["vision_on_cpu"] is False

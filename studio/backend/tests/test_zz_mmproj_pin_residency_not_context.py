# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the automatic projector pin is allowed to buy: residency, never context.

The pin (``--no-mmproj-offload``, #8967) moves the vision encoder to host RAM --
a measured ~3.6x per image -- and its whole justification is that the projector's
VRAM was what pushed model LAYERS onto the CPU: a bounded per-image cost traded
for a per-token one.

``_mm_rank`` decides that by asking ``_mmproj_fits`` whether each side fits at
``effective_ctx``, which in Auto is the model's NATIVE context length. But Auto
never spills layers to make a native context fit -- the placement loop shrinks
the CONTEXT instead (``llama_cpp.py`` ~15286-15343) and only reaches ``--fit on``
once even its 4096 fallback will not place. So "does not fit at 131k" is a
context answer being read as a residency one, and the two come apart in both
directions:

  * Above the band, both sides place with every layer resident and only the
    context differs, so the pin buys a few thousand tokens and pays 3.6x on
    every image for them -- while logging "so every model layer fits in VRAM",
    which was already true.
  * Below the band, the load genuinely goes out ``--fit on`` with layers on the
    CPU and the pin -- the one case it exists for -- does not fire, because the
    projector-free side does not reach the native length either.

Both cells here are stated as an A/B against the same load with the pin
unavailable, so neither can pass vacuously: the baseline assertion says what the
launch does when Studio cannot pin, and the pinned assertion says the pin only
changes it where a layer was actually at stake.

The KV estimate is context-PROPORTIONAL here. The existing pin tests stub it to
a flat 1 GiB, which makes every context cost the same and hides this entire
question; 128 KiB/token is what this 65-layer, 4-KV-head shape really costs.
The compute-buffer estimator is pinned to 100 MiB for the reason
``test_mmproj_pin_platform_matrix`` pins it: the real one answers in gigabytes
against a stub GGUF and nothing fits anywhere, so a policy test passes whatever
the policy does.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

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

_TOTAL_MIB = 24_576
_MODEL_GB = 8.0
_MMPROJ_GB = 1.5
_NATIVE_CTX = 32_768
# 65 layers x 4 KV heads x (256 + 256) bytes x 2 (K and V, f16) is ~266 KiB per
# token; 128 KiB/token is the conservative half of that.
_KV_PER_TOKEN = 128 * 1024


def _vision_backend(tmp_path: Path, *, free_mib: int):
    """An 8 GB vision GGUF with a 1.5 GB projector on one discrete card."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, free_mib, _TOTAL_MIB)])
    backend._get_gguf_size_bytes = lambda _path: int(_MODEL_GB * _GB)
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * _MIB
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * _KV_PER_TOKEN
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._mtp_draft_kv_bytes = lambda *a, **k: 0

    def read_metadata(_path):
        backend._n_layers = 65
        backend._n_kv_heads = 4
        backend._n_heads = 24
        backend._embedding_length = 5120
        backend._kv_key_length = 256
        backend._kv_value_length = 256
        backend._context_length = _NATIVE_CTX

    backend._read_gguf_metadata = read_metadata
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
        "supports_kv_unified": True,
    }
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: int(_MMPROJ_GB * _GB)
    backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _outcome(tmp_path: Path, *, free_mib: int, pin_available: bool):
    """Launch once. ``pin_available=False`` is the same load on a build that
    cannot emit the flag -- i.e. what the placement does with no pin at all."""
    backend, gguf = _vision_backend(tmp_path, free_mib = free_mib)
    with patch.object(
        llama_cpp, "_paravirtual_mmproj_pinnable", lambda _caps: pin_available
    ):
        cmd = _launch(
            backend, gguf, is_vision = True, speculative_type = "off", n_ctx = 0
        )["cmd"]
    return {
        "pin": _PIN in cmd,
        "fit": cmd[cmd.index("--fit") + 1],
        "ctx": backend._effective_context_length,
        "vision_on_cpu": backend.vision_on_cpu,
    }


# Free VRAM (MiB) on the one card. Usable budget is free - 737 (3% of a 24 GB
# total), and the model alone needs ~8.6 GB before any KV.
#
#   11_000: the projector is what spills layers -- without it the load places,
#           with it the load goes out `--fit on`. The pin's whole reason.
#   15_500: both sides place with every layer resident; only the context
#           differs. Nothing is at stake but a few thousand tokens.
_FREE_PROJECTOR_SPILLS_LAYERS = 11_000
_FREE_BOTH_SIDES_ARE_RESIDENT = 15_500


def test_the_pin_does_not_move_the_projector_to_buy_context(tmp_path):
    """Both sides keep every layer on the GPU; the pin must not fire.

    The baseline assertion is the point: with no pin the load is ALREADY
    ``--fit off`` at a large context, so there is no layer on the CPU for the
    projector to be blamed for. Moving it can only buy tokens, and the pin's own
    doctrine prices a bounded per-image cost against a PER-TOKEN one -- context
    is neither, and 3.6x on every image is not what a few thousand tokens are
    worth.
    """
    baseline = _outcome(
        tmp_path, free_mib = _FREE_BOTH_SIDES_ARE_RESIDENT, pin_available = False
    )
    # Not vacuous: with the projector charged, every layer is resident already.
    assert baseline["fit"] == "off"
    assert baseline["ctx"] > 4096

    pinned = _outcome(
        tmp_path, free_mib = _FREE_BOTH_SIDES_ARE_RESIDENT, pin_available = True
    )
    assert pinned["pin"] is False, (
        "the projector was moved to the CPU on a load that was already fully "
        f"GPU-resident: context {baseline['ctx']} -> {pinned['ctx']} bought with "
        "a ~3.6x image encode"
    )
    assert pinned["vision_on_cpu"] is False
    # And nothing else changed: the placement was never the projector's fault.
    assert pinned["fit"] == "off"


def test_the_pin_fires_where_the_projector_is_what_spills_layers(tmp_path):
    """The projector's bytes are what force ``--fit on``; the pin must fire.

    This is the case #8967 was written for and the one the native-length
    comparison walks straight past: neither side reaches 32768, so both rank 0,
    the ranks tie and the pin declines -- on the one load where every layer
    would have stayed on the GPU without the projector.
    """
    baseline = _outcome(
        tmp_path, free_mib = _FREE_PROJECTOR_SPILLS_LAYERS, pin_available = False
    )
    # Not vacuous: charged for the projector, llama-server is free to spill.
    assert baseline["fit"] == "on"

    pinned = _outcome(
        tmp_path, free_mib = _FREE_PROJECTOR_SPILLS_LAYERS, pin_available = True
    )
    assert pinned["pin"] is True
    assert pinned["vision_on_cpu"] is True
    # The trade delivered: every layer resident, which is what was bought.
    assert pinned["fit"] == "off"


@pytest.mark.parametrize("free_mib", list(range(9_000, 18_001, 500)))
def test_the_pin_never_fires_where_the_load_already_places(tmp_path, free_mib):
    """Swept form of the first cell, so the boundary cannot drift into the band.

    Wherever the load places every layer WITH the projector charged, the pin has
    nothing to buy and must stay out of the argv.
    """
    baseline = _outcome(tmp_path, free_mib = free_mib, pin_available = False)
    pinned = _outcome(tmp_path, free_mib = free_mib, pin_available = True)
    if baseline["fit"] == "off":
        assert pinned["pin"] is False, free_mib
    # And the converse direction of the same rule: a load the projector spills
    # is one the pin has to rescue whenever pinning places it.
    elif pinned["pin"]:
        assert pinned["fit"] == "off", free_mib

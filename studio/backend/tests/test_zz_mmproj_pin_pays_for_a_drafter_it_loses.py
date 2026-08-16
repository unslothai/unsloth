# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The projector pin taken to save a drafter the drop probe then takes anyway.

``_mm_rank`` grades each side of the pin 2 (placed WITH the drafter) / 1 (placed
only after the drop probe removes it) / 0 (not placed), and pins when moving the
projector ranks higher. Rank 2 vs rank 1 is a bet on a decision the pin does not
make: the drafter-VRAM drop probe runs immediately after and prices the reserve
at the context the TARGET ALONE would reach (``_fit_context_to_vram`` per subset,
``llama_cpp.py`` ~14884), while ``_mmproj_fits`` prices it at ``effective_ctx``.

When those two disagree the pin is taken to save a drafter, the probe drops the
drafter regardless, and nothing hands the pin back: the ``use_fit`` backstop
below only fires when the placement failed, and here both sides place. The load
pays a measured ~3.6x on every image and gets nothing the unpinned load did not
already have except context -- which is in neither half of the preference order
the ranking encodes.

The invariant asserted here is the ranking's own: a pin must improve residency,
or keep a drafter the unpinned load loses. Buying context is not on the list.

Everything is simulated, the compute-buffer estimator is pinned to 100 MiB (see
``test_mmproj_pin_platform_matrix``), and the KV estimate is context-proportional
because a flat one makes every context cost the same and hides the disagreement
entirely.
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

_MODEL_GB = 8.0
_MMPROJ_GB = 1.5
_DRAFTER_GB = 1.5
_NATIVE_CTX = 32_768
_KV_PER_TOKEN = 64 * 1024
_CC_PER_TOKEN = 8 * 1024


def _backend_with_drafter(tmp_path: Path, *, memory):
    """A vision + embedded-MTP hybrid-Mamba target on a layer-split pool."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)
    backend._get_gguf_size_bytes = lambda _path: int(_MODEL_GB * _GB)
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * _MIB
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * _KV_PER_TOKEN
    backend._compute_buffer_ctx_bytes = lambda ctx, *a, **k: int(ctx) * _CC_PER_TOKEN
    backend._mtp_draft_kv_bytes = lambda *a, **k: 0
    backend._estimate_mtp_overhead_bytes = lambda *a, **k: int(_DRAFTER_GB * _GB)

    def read_metadata(_path):
        backend._n_layers = 65
        backend._n_kv_heads = 4
        backend._n_heads = 24
        backend._embedding_length = 5120
        backend._kv_key_length = 256
        backend._kv_value_length = 256
        backend._context_length = _NATIVE_CTX
        # The #8875 shape: an embedded MTP head on a hybrid-Mamba trunk.
        backend._nextn_predict_layers = 1
        backend._full_attention_interval = 4
        backend._ssm_inner_size = 6144
        backend._ssm_state_size = 128
        backend._ssm_group_count = 16
        backend._ssm_conv_kernel = 4

    backend._read_gguf_metadata = read_metadata
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
        "supports_kv_unified": True,
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: int(_MMPROJ_GB * _GB)
    backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _outcome(tmp_path: Path, *, memory, pin_available: bool):
    backend, gguf = _backend_with_drafter(tmp_path, memory = memory)
    with patch.object(
        llama_cpp, "_paravirtual_mmproj_pinnable", lambda _caps: pin_available
    ):
        cmd = _launch(
            backend, gguf, is_vision = True, speculative_type = "auto", n_ctx = 0
        )["cmd"]
    return {
        "pin": _PIN in cmd,
        "fit": cmd[cmd.index("--fit") + 1],
        "spec": cmd[cmd.index("--spec-type") + 1] if "--spec-type" in cmd else None,
        "drafter_kept": backend.spec_fallback_reason != "drafter_no_vram",
        "ctx": backend._effective_context_length,
    }


# One roomy card plus one small one, so the pool is a layer split and the
# per-device compute buffer is real. Each pair is a cell where the pin fires.
_POOLS = [
    [(0, 13_000, 24_576), (1, 2_000, 8_192)],
    [(0, 12_000, 24_576), (1, 3_000, 8_192)],
    [(0, 13_000, 24_576), (1, 3_000, 8_192)],
    [(0, 12_000, 24_576), (1, 4_500, 8_192)],
    [(0, 14_000, 24_576), (1, 2_000, 8_192)],
    [(0, 11_000, 24_576), (1, 4_500, 8_192)],
]


@pytest.mark.parametrize("memory", _POOLS, ids = lambda m: f"{m[0][1]}+{m[1][1]}")
def test_a_pin_must_buy_residency_or_the_drafter(tmp_path, memory):
    """The ranking's own preference order, asserted on the launched command.

    2 beats 1 beats 0 because a per-token loss -- layers on the CPU, then the
    drafter -- outweighs the projector's bounded per-image cost. So a pin that
    changes neither is a cost with no entry in that order, whatever it does to
    the context.
    """
    unpinned = _outcome(tmp_path, memory = memory, pin_available = False)
    pinned = _outcome(tmp_path, memory = memory, pin_available = True)

    if not pinned["pin"]:
        # Declined or handed back. Then it must have left nothing behind either:
        # the launch is the one a build that cannot pin at all would have made.
        assert pinned == unpinned, (memory, unpinned, pinned)
        return

    bought_residency = unpinned["fit"] == "on" and pinned["fit"] == "off"
    bought_the_drafter = pinned["drafter_kept"] and not unpinned["drafter_kept"]
    assert bought_residency or bought_the_drafter, (
        "the projector was moved to host RAM (~3.6x per image) and the launch is "
        f"no better for it: unpinned {unpinned}, pinned {pinned}"
    )


# The two pools where the pin genuinely saves the drafter, named so the test
# above cannot go quiet by never pinning anywhere.
_POOLS_THAT_MUST_STILL_PIN = [
    [(0, 14_000, 24_576), (1, 2_000, 8_192)],
    [(0, 11_000, 24_576), (1, 4_500, 8_192)],
]


@pytest.mark.parametrize(
    "memory", _POOLS_THAT_MUST_STILL_PIN, ids = lambda m: f"{m[0][1]}+{m[1][1]}"
)
def test_the_pin_still_fires_where_it_does_save_the_drafter(tmp_path, memory):
    """The other half of the bargain, so the fix cannot be "never pin"."""
    unpinned = _outcome(tmp_path, memory = memory, pin_available = False)
    assert unpinned["drafter_kept"] is False

    pinned = _outcome(tmp_path, memory = memory, pin_available = True)
    assert pinned["pin"] is True
    assert pinned["drafter_kept"] is True
    assert pinned["spec"] == "draft-mtp"

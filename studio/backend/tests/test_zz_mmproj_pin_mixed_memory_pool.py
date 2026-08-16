# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_discrete_vram`` over a MIXED pool: one card with its own memory, one without.

``_discrete_vram`` (``llama_cpp.py`` ~14556) is ``all(total > 0)`` over the
devices in play, and the existing coverage exercises the two uniform ends of that
quantifier -- every device shared (``test_zz_hand_pin_charge_by_memory_kind``'s
single-APU pool) and every device discrete (everything else). The quantifier
itself, the case where the two kinds are enumerated together, was untested, and
``all`` versus ``any`` is exactly the kind of choice that survives a review
because both readings look defensible in isolation.

``all`` is the right one: a layer split puts weights on EVERY device in the
subset, so a pool that contains a shared-memory device is a pool where moving the
projector to host RAM can be moving it inside the same pool that is holding the
weights. It also protects ``_apu_ram_shortfall_message``, which prices system RAM
and must keep weighing a projector that is still in system RAM whichever side of
the pin it sits on.

The two pools below differ in ONE number -- the second device's reported total,
0 for a shared pool and a real figure for a discrete one -- and the shared
reading is given the LARGER usable budget of the two (``free * frac`` beats
``free - (1 - frac) * total`` at these sizes), so a pin that disappears on the
mixed pool cannot be explained by it having less room. Only the memory-kind
verdict can explain it.

Simulated throughout; the compute-buffer estimator is pinned to 100 MiB for the
reason ``test_mmproj_pin_platform_matrix`` pins it.
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

_MIB = 1024 * 1024
_PIN = "--no-mmproj-offload"

_MODEL_BYTES = 4_500 * _MIB
_PROJECTOR_BYTES = 900 * _MIB

# One card the model alone nearly fills, so the projector's 900 MiB is what tips
# it over, plus one small companion that cannot rescue the placement on its own.
# The companion is the only thing that changes between the two pools.
_MAIN = (0, 6_000, 8_000)
_COMPANION_DISCRETE = (1, 500, 8_000)   # usable 500 - 240 = 260 MiB
_COMPANION_SHARED = (1, 500, 0)         # usable 500 * 0.97 = 485 MiB -- MORE

_ALL_DISCRETE = [_MAIN, _COMPANION_DISCRETE]
_MIXED = [_MAIN, _COMPANION_SHARED]


def _pin_backend(tmp_path: Path, *, memory):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)
    backend._get_gguf_size_bytes = lambda _path: _MODEL_BYTES
    # The real estimator answers in gigabytes against a stub GGUF, which swamps
    # every other term so nothing fits and the assertions stop discriminating.
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * _MIB
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: _PROJECTOR_BYTES
    backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _launch_on(tmp_path: Path, *, memory, extra_args = None):
    backend, gguf = _pin_backend(tmp_path, memory = memory)
    cmd = _launch(backend, gguf, is_vision = True, extra_args = extra_args)["cmd"]
    return backend, [str(a) for a in cmd]


def test_the_all_discrete_pool_is_the_control_and_does_pin(tmp_path):
    """Non-vacuity: on this pool, sized this way, the pin genuinely fires.

    Without this the mixed-pool assertion below would pass on a pool where
    nothing would have pinned anyway, which is the shape of "clean report, no
    coverage".
    """
    backend, cmd = _launch_on(tmp_path, memory = _ALL_DISCRETE)

    assert cmd.count(_PIN) == 1
    assert backend.vision_on_cpu is True


def test_one_shared_device_takes_the_whole_pool_out_of_the_refund(tmp_path):
    """Same pool, same sizes, one device that reports no total of its own."""
    backend, cmd = _launch_on(tmp_path, memory = _MIXED)

    assert _PIN not in cmd, (
        "the projector's VRAM was handed back over a pool containing a "
        "shared-memory device, where a layer split puts weights on that device too"
    )
    # Pinned or not, vision is untouched: the projector still loads.
    assert "--mmproj" in cmd
    assert backend.vision_on_cpu is False


@pytest.mark.parametrize("memory, refunded", [(_ALL_DISCRETE, True), (_MIXED, False)])
def test_the_hand_pin_is_charged_by_the_same_quantifier(tmp_path, memory, refunded):
    """The user's own ``--no-mmproj-offload`` reads the same predicate.

    A hand pin over a discrete pool gets its bytes back and the fit places the
    model; over a mixed pool the bytes stay charged, so the placement is the
    conservative one -- the projector is in system RAM either way there, and the
    APU RAM refusal downstream has to keep weighing it.
    """
    backend, cmd = _launch_on(tmp_path, memory = memory, extra_args = [_PIN])

    # The user's token is the only one in the argv either way: Studio never adds
    # a second answer on top of a placement the user named.
    assert cmd.count(_PIN) == 1
    assert backend.vision_on_cpu is True
    # Whether the refund happened is visible in the placement: refunded, the
    # weights fit and the planner pins a device; charged, it falls back to --fit.
    assert (cmd[cmd.index("--fit") + 1] == "off") is refunded

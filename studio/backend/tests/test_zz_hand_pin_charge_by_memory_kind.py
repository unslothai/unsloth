# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CPU-pinned projector's bytes come back only where something else can use them.

Two validation branches changed this term in opposite directions and each was
right about its own machine, so the pair has to be pinned together or whichever
fix lands second silently undoes the first.

  * On a Mac the fit budget is ``_apple_metal_memory_budget_bytes`` -- UNIFIED
    memory. A projector pinned to the CPU is still sitting in it, so subtracting
    its bytes hands out a context the machine cannot back. That is true whoever
    pinned it: the paravirtual guard, or the user's own ``--no-mmproj-offload``.
    ``test_zz_paravirtual_projector_charge`` covers the guard; the extras are
    covered here, because that is the exact request the discrete-GPU fix wants
    to zero and must not.

  * On a discrete card the bytes are real VRAM the layers can have. Auto already
    hands them back when it pins; a user who pins by hand must get the same
    placement, not a worse one for having said so out loud.

``gpus`` is what separates the two, which is why the hand-pin is honoured next
to the automatic pin (after the GPU probe) rather than at the charge site (before
it), where the token is readable but the answer is not.

Every number below is asserted against a figure captured by executing the same
builder, and the projector's bytes are shown to be inside the totals being
compared, so no assertion here can be satisfied by two zeroes agreeing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _launch  # noqa: E402,F401
from test_mmproj_pin_platform_matrix import (  # noqa: E402
    _PIN,
    _TIGHT_MEMORY,
    _TIGHT_MODEL_BYTES,
    _accel_backend,
    _apply_os,
)
from test_zz_paravirtual_projector_charge import (  # noqa: E402
    MAIN_CTX,
    MAIN_MODEL_SIZE_FIT,
    PROJECTOR_BYTES,
    _launch_metal,
)

import core.inference.llama_cpp as llama_cpp  # noqa: E402

MIB = 1024 * 1024

_UNPIN = "--mmproj-offload"
# The projector's file plus the _MMPROJ_VRAM_SAFETY surcharge: what the planner
# stops charging when the encoder genuinely leaves the card. Built the way the
# two _soft_overhead sites build it -- the file, then a separately truncated
# surcharge -- rather than as one 1.4x multiply, which lands a byte higher
# because 1.4 - 1.0 is not 0.4 in binary floating point.
_SAFETY = llama_cpp.LlamaCppBackend._MMPROJ_VRAM_SAFETY
_CHARGE_BYTES = PROJECTOR_BYTES + int(PROJECTOR_BYTES * (_SAFETY - 1.0))


# --------------------------------------------------------------------------
# Unified memory: the pin frees nothing, so nothing is given back
# --------------------------------------------------------------------------


@pytest.mark.parametrize("paravirtual", [False, True])
def test_a_hand_pinned_projector_is_still_charged_against_unified_memory(
    tmp_path, monkeypatch, paravirtual
):
    """The regression the discrete-GPU fix must not reintroduce.

    ``--no-mmproj-offload`` in the advanced arguments is the one placement token
    a Mac user can type, and reading it at the charge site -- where ``gpus`` has
    not been probed yet -- looks like the obvious place to honour it. It is not:
    that site cannot tell unified memory from a discrete card, and on Metal the
    encoder's bytes are still in the budget the context is sized against.

    Both Macs, and both reasons for the pin, must plan the load exactly as
    origin/main did.
    """
    got = _launch_metal(
        tmp_path,
        paravirtual = paravirtual,
        monkeypatch = monkeypatch,
        extra_args = [_PIN],
    )

    assert got["model_size_fit"] == MAIN_MODEL_SIZE_FIT
    assert got["ctx"] == MAIN_CTX


def test_the_hand_pin_on_a_mac_weighs_what_the_untouched_load_weighs(tmp_path, monkeypatch):
    """Stated as a difference rather than a constant, so the pair still holds if
    the model geometry above is ever retuned. The projector is shown to be inside
    both totals, so this is not two zeroes agreeing."""
    plain = _launch_metal(tmp_path / "plain", paravirtual = True, monkeypatch = monkeypatch)
    hand = _launch_metal(
        tmp_path / "hand", paravirtual = True, monkeypatch = monkeypatch, extra_args = [_PIN]
    )

    assert hand["model_size_fit"] == plain["model_size_fit"]
    assert hand["ctx"] == plain["ctx"]
    # The bytes at stake are really in there.
    assert plain["model_size_fit"] > _CHARGE_BYTES


def test_a_bigger_hand_pinned_projector_does_not_diverge_on_a_mac(tmp_path, monkeypatch):
    """Not a rounding error. Unfixed, zeroing a 3 GiB projector against a 10 GiB
    unified budget moved the emitted context by nearly 4x -- all of it memory the
    encoder is still holding."""
    plain = _launch_metal(
        tmp_path / "plain",
        paravirtual = True,
        monkeypatch = monkeypatch,
        mmproj_bytes = 3 * 1024**3,
    )
    hand = _launch_metal(
        tmp_path / "hand",
        paravirtual = True,
        monkeypatch = monkeypatch,
        mmproj_bytes = 3 * 1024**3,
        extra_args = [_PIN],
    )

    assert plain["ctx"] == 4096
    assert hand["ctx"] == 4096


def test_the_mac_still_reports_the_projector_as_cpu_resident(tmp_path, monkeypatch):
    """Charging the bytes is a budget decision, not a claim about where the
    encoder runs. The UI note must still say CPU, or the fix above would have
    bought its correctness by lying to the user instead."""
    got = _launch_metal(tmp_path, paravirtual = False, monkeypatch = monkeypatch, extra_args = [_PIN])

    assert got["backend"].vision_on_cpu is True
    # The user's own token, once, and Studio adds no second one.
    assert got["cmd"].count(_PIN) == 1


# --------------------------------------------------------------------------
# Discrete VRAM: the pin frees real bytes, so they are given back
# --------------------------------------------------------------------------


def _discrete_charge(
    monkeypatch,
    tmp_path,
    *,
    extra_args,
    caps = None,
    accel = "nvidia",
):
    """Launch on a card the projector is what tips over, and capture the weight
    the planner priced the placement at.

    ``_can_estimate_kv`` is False in this harness, so the fit takes the
    file-size-only arm and hands ``model_size_fit`` straight to ``_select_gpus``
    -- one scalar, no KV curve to reason about. ``_accel_backend`` pins the
    compute-buffer estimate to 100 MiB; against the real estimator's multi-GB
    answer for a stub GGUF every term below is noise and the comparison would
    hold whatever the policy did.
    """
    _apply_os(monkeypatch, "linux")
    tmp_path.mkdir(parents = True, exist_ok = True)
    backend, gguf = _accel_backend(
        monkeypatch,
        tmp_path,
        accel,
        memory = _TIGHT_MEMORY,
        model_bytes = _TIGHT_MODEL_BYTES,
    )
    if caps is not None:
        backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    record: dict = {}
    _real_select = backend._select_gpus

    def _capture(model_size, gpus, *args, **kwargs):
        record.setdefault("model_size_fit", model_size)
        return _real_select(model_size, gpus, *args, **kwargs)

    backend._select_gpus = _capture
    record["cmd"] = [
        str(a) for a in _launch(backend, gguf, is_vision = True, extra_args = extra_args)["cmd"]
    ]
    record["backend"] = backend
    return record


def test_a_hand_pinned_projector_gives_its_vram_back_on_a_discrete_card(monkeypatch, tmp_path):
    """The other direction, at the byte level.

    Auto already stops charging the projector when it decides to pin it. A user
    who types the same token gets the same physical placement, so they must get
    the same price -- and the difference between the two answers has to be
    exactly the projector's file plus its surcharge, or something else moved.
    """
    auto = _discrete_charge(monkeypatch, tmp_path / "auto", extra_args = None)
    hand = _discrete_charge(monkeypatch, tmp_path / "hand", extra_args = [_PIN])
    # The control: the same argv with the projector left on the card. Without it
    # the equality above could be satisfied by a planner that charges nothing.
    on_gpu = _discrete_charge(monkeypatch, tmp_path / "on_gpu", extra_args = [_UNPIN])

    assert hand["model_size_fit"] == auto["model_size_fit"]
    assert on_gpu["model_size_fit"] - hand["model_size_fit"] == _CHARGE_BYTES
    # The surcharge is really part of what came back, not just the file.
    assert _CHARGE_BYTES > PROJECTOR_BYTES


def test_the_explicit_request_places_no_worse_than_the_automatic_one(monkeypatch, tmp_path):
    """The user-visible shape of the same bug.

    Charging bytes the card is not holding made the model stop fitting, so
    Studio shipped ``--fit on`` with no layer plan and let llama-server spill --
    while the identical argv reached automatically got every layer resident.
    Saying out loud what Auto would have done anyway must never be the slower
    request.
    """
    auto = _discrete_charge(monkeypatch, tmp_path / "auto", extra_args = None)
    hand = _discrete_charge(monkeypatch, tmp_path / "hand", extra_args = [_PIN])

    # The premise: Auto pins this model on this card and places it fully.
    assert auto["cmd"].count(_PIN) == 1
    assert auto["cmd"][auto["cmd"].index("--fit") + 1] == "off"

    assert hand["cmd"][hand["cmd"].index("--fit") + 1] == "off"
    assert hand["cmd"][hand["cmd"].index("-ngl") + 1] == "-1"
    # Still exactly one placement token, and it is the user's own.
    assert hand["cmd"].count(_PIN) == 1
    assert hand["backend"].vision_on_cpu is True


# --------------------------------------------------------------------------
# A device is enumerated, but its memory is the host's
# --------------------------------------------------------------------------

# How the two probes report a shared pool: free is system RAM (already less the
# iGPU host reserve) and the TOTAL is a deliberate 0 -- see the `0 if shared` in
# _get_gpu_memory and the `0 if is_igpu` in _vulkan_auto_gpu_memory. The same 0
# also stands for a total the probe could not read at all.
_APU_MEMORY = [(0, 20_000, 0)]
_APU_MODEL_BYTES = 4_500 * MIB
# Enough for the weights and change, but not for the projector's charge on top:
# this is the band where the refusal below flips.
_APU_AVAIL_MIB = (_APU_MODEL_BYTES // MIB) + 2048 + 200


def _apu_charge(monkeypatch, tmp_path, *, extra_args, ram_guard):
    """Launch against an enumerated device whose 'VRAM' is system RAM."""
    _apply_os(monkeypatch, "linux")
    tmp_path.mkdir(parents = True, exist_ok = True)
    backend, gguf = _accel_backend(
        monkeypatch,
        tmp_path,
        "nvidia",
        memory = list(_APU_MEMORY),
        model_bytes = _APU_MODEL_BYTES,
    )
    if ram_guard:
        # The refusal that stops an APU load being OOM-killed mid-read. The shared
        # placement harness stubs all three of these out, so put the real message
        # back rather than assert against a stub that always answers "fine".
        backend._amd_apu_wants_unified_memory = lambda *a, **k: True
        backend._available_system_memory_mib = lambda: _APU_AVAIL_MIB
        backend._apu_ram_shortfall_message = (
            lambda *a, **k: llama_cpp.LlamaCppBackend._apu_ram_shortfall_message(*a, **k)
        )
    record: dict = {}
    _real_select = backend._select_gpus

    def _capture(model_size, gpus, *args, **kwargs):
        record.setdefault("model_size_fit", model_size)
        return _real_select(model_size, gpus, *args, **kwargs)

    backend._select_gpus = _capture
    record["cmd"] = [
        str(a) for a in _launch(backend, gguf, is_vision = True, extra_args = extra_args)["cmd"]
    ]
    return record


def test_a_shared_pool_is_not_discrete_memory_however_many_gpus_it_enumerates(
    monkeypatch, tmp_path
):
    """A non-empty ``gpus`` is not the same claim as "the memory is discrete".

    A unified-memory APU and an integrated Vulkan GPU both enumerate a device and
    both report system RAM as its free VRAM. Pinning the encoder to the CPU moves
    it within one pool, so its bytes must stay charged for exactly the reason the
    Mac's do -- and ``total == 0`` is the marker both probes already set for it.
    """
    plain = _apu_charge(monkeypatch, tmp_path / "plain", extra_args = None, ram_guard = False)
    hand = _apu_charge(monkeypatch, tmp_path / "hand", extra_args = [_PIN], ram_guard = False)

    assert hand["model_size_fit"] == plain["model_size_fit"]
    # The projector really is inside the figure that did not move.
    assert plain["model_size_fit"] > _APU_MODEL_BYTES + PROJECTOR_BYTES


def test_the_apu_ram_refusal_survives_the_users_own_pin(monkeypatch, tmp_path):
    """The sharp end of the same bug, and the reason the byte assertion above is
    not academic.

    ``_apu_ram_shortfall_message`` refuses a load whose weights will not fit in
    system RAM, because the alternative is the OS killing it mid-read. It prices
    ``model_size``. Handing the projector's bytes back on a shared pool took this
    load under the threshold, so typing a flag that frees nothing on this machine
    converted a deliberate refusal into the launch it exists to prevent.
    """
    for label, extras in (("plain", None), ("hand", [_PIN])):
        with pytest.raises(RuntimeError, match = "unified-memory"):
            _apu_charge(monkeypatch, tmp_path / label, extra_args = extras, ram_guard = True)


def test_a_build_without_the_flag_keeps_charging_the_hand_pinned_projector(monkeypatch, tmp_path):
    """A conclusive probe that does not list ``--no-mmproj-offload`` means a
    build too old to honour it, so the encoder stays on the card whatever the
    extras say. Handing the bytes back there would plan against VRAM that is
    still occupied -- the discrete-card version of the Mac bug above."""
    hand = _discrete_charge(
        monkeypatch,
        tmp_path / "hand",
        extra_args = [_PIN],
        caps = {"supports_metrics": False},
    )
    on_gpu = _discrete_charge(
        monkeypatch,
        tmp_path / "on_gpu",
        extra_args = [_UNPIN],
        caps = {"supports_metrics": False},
    )

    assert hand["model_size_fit"] == on_gpu["model_size_fit"]

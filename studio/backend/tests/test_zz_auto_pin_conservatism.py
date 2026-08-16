# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The automatic projector pin is a trade, and it must not be paid on credit.

Auto pins the vision encoder to the CPU when the projector's VRAM is the only
thing keeping model layers off the GPU. The trade is explicit and one-directional:
image encoding gets ~3.6x slower, and what buys that is every layer resident. So
the pin is only defensible where the residency is real and the arithmetic behind
it is real. Three arms fail one of those tests, and all three made the placement
LESS conservative than the same load gets without a projector at all:

  * **No KV estimate.** ``_estimate_kv_cache_bytes`` returns 0 outright when the
    GGUF metadata will not support an estimate, so the probe that decided to pin
    priced the native context's KV at zero. Its refund then reached the
    file-size-only placement, which is the arm whose whole job is to be careful:
    it clamps Auto context to 4096 and leaves ``--fit on`` as a valve. Measured on
    the PR head, the pin turned that into ``-c 32768 -ngl -1 --fit off``.

  * **Tensor parallelism.** ``_mmproj_fits`` prices a layer split. TP replicates a
    per-device buffer with a geometry these numbers do not model, and the MTP-drop
    probe twenty lines below already carries ``and not tensor_parallel`` for
    exactly that reason.

  * **The fit's own except arm.** It restores ``--fit on`` with no layer plan,
    which is llama-server free to spill layers again. The residency the pin was
    bought for is gone; the slower image encode is not.

Every cell is compared against a control in which the pin SHOULD fire, so none of
these can be satisfied by a policy that simply never pins. ``_estimate_compute_
buffer_bytes`` is pinned to 100 MiB throughout: left real it answers several GB
against a stub GGUF and swamps every term being compared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch  # noqa: E402,F401
from test_mmproj_pin_platform_matrix import _PIN, _apply_os  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402

MIB = 1024 * 1024

_MODEL_BYTES = 4_500 * MIB
_PROJECTOR_BYTES = 900 * MIB
# One card the model nearly fills: the projector's 900 MiB is what tips it over,
# so this is a load the pin is supposed to act on.
_TIGHT = [(0, 6_000, 8_000)]
# Two cards, each too small alone but pooled enough for the weights, so tensor
# parallelism survives its own <2-GPU and pooled-budget downgrades.
_TP_PAIR = [(0, 3_400, 4_000), (1, 3_400, 4_000)]
_NATIVE_CTX = 32768
# Small enough that the projector, not the KV cache, is what decides whether the
# load fits: the pin declines outright on a model too large either way, so an
# oversized KV would make the control below silently vacuous.
_KV_PER_TOKEN = 8 * 1024


def _pin_backend(
    monkeypatch,
    tmp_path: Path,
    *,
    memory = _TIGHT,
    can_estimate_kv = False,
    fit_raises = False,
    paravirtual = False,
):
    _apply_os(monkeypatch, "linux")
    tmp_path.mkdir(parents = True, exist_ok = True)
    backend, gguf = _backend(tmp_path, vulkan = False, memory = list(memory))
    backend._get_gguf_size_bytes = lambda _path: _MODEL_BYTES
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * MIB
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: _PROJECTOR_BYTES
    backend._mmproj_matches_model_family = lambda *a, **k: True
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: False)
    )
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: paravirtual)
    # A big native length, so "clamped to 4096" and "left at native" are far apart.
    backend._context_length = _NATIVE_CTX
    if can_estimate_kv:
        backend._can_estimate_kv = lambda: True
        backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx) * _KV_PER_TOKEN
        backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    if fit_raises:
        def _boom(*a, **k):
            raise RuntimeError("simulated GPU-selection failure")

        backend._select_gpus = _boom
        backend._select_gpus_split_aware = _boom
    return backend, gguf


def _plan(backend, gguf, **load_kwargs):
    cmd = [str(a) for a in _launch(
        backend, gguf, is_vision = True, n_ctx = 0, **load_kwargs
    )["cmd"]]
    return {
        "cmd": cmd,
        "pins": cmd.count(_PIN),
        "fit": cmd[cmd.index("--fit") + 1] if "--fit" in cmd else None,
        "ngl": cmd[cmd.index("-ngl") + 1] if "-ngl" in cmd else None,
        "ctx": int(cmd[cmd.index("-c") + 1]) if "-c" in cmd else None,
        "backend": backend,
    }


# --------------------------------------------------------------------------
# B5: the pin must not lift the no-KV-estimate clamp
# --------------------------------------------------------------------------


def test_the_pin_does_not_lift_the_no_kv_estimate_context_clamp(monkeypatch, tmp_path):
    """Auto context stays at 4096 when there is no estimate behind it.

    This is the arm where ``_kv_bytes`` answers 0 for every context, so the probe
    that decided to pin priced 32768 tokens of cache at nothing. Measured on the
    PR head, its refund promoted the deliberately conservative ``-c 4096`` to
    ``-c 32768``: a maximal context chosen on the strength of an estimate known to
    be absent, which is the OOM the clamp exists to avoid.
    """
    backend, gguf = _pin_backend(monkeypatch, tmp_path, can_estimate_kv = False)

    got = _plan(backend, gguf)

    assert got["ctx"] == 4096
    # The premise: this really is the no-KV arm, and the pin really did fire, so
    # the clamp is being held against a live refund rather than a load that never
    # had one.
    assert backend._can_estimate_kv() is False
    assert got["pins"] == 1


def test_the_pin_still_buys_residency_without_a_kv_estimate(monkeypatch, tmp_path):
    """The other half of the same decision, deliberately NOT withdrawn.

    Whether the weights fit is arithmetic over exact file sizes and needs no KV
    estimate, so the pin's residency is honoured: the trade is still paid for and
    still received. Only the context question -- the one that genuinely depends on
    the missing number -- is held back. Asserted so a future 'simplification' that
    withdraws the whole refund, or gates the pin on ``_can_estimate_kv``, cannot
    land quietly: either would silently disable the pin in every cell of the
    platform matrix, which runs entirely on this arm.
    """
    backend, gguf = _pin_backend(monkeypatch, tmp_path, can_estimate_kv = False)

    got = _plan(backend, gguf)

    assert got["pins"] == 1
    assert got["fit"] == "off"
    assert got["ngl"] == "-1"
    assert got["backend"].vision_on_cpu is True


def test_a_real_kv_estimate_lets_the_pin_hand_out_context_too(monkeypatch, tmp_path):
    """The control, and the reason the clamp test is not just asserting "always
    4096". Given an estimate the context question can actually be answered, so the
    same load on the same card gets a context sized against it rather than the
    fallback."""
    backend, gguf = _pin_backend(monkeypatch, tmp_path, can_estimate_kv = True)

    got = _plan(backend, gguf)

    assert got["pins"] == 1
    assert got["fit"] == "off"
    assert got["ngl"] == "-1"
    assert got["ctx"] > 4096


# --------------------------------------------------------------------------
# B4: tensor parallelism is the wrong geometry for _mmproj_fits
# --------------------------------------------------------------------------


def test_the_pin_stands_clear_of_tensor_parallelism(monkeypatch, tmp_path):
    """``_mmproj_fits`` prices a layer split over ranked subsets. Under TP the
    per-device buffer has a geometry none of those numbers model, so its verdict
    is not about the placement that runs -- and it would also reach the pooled TP
    weight-budget check and talk it out of a downgrade it should make."""
    backend, gguf = _pin_backend(monkeypatch, tmp_path, memory = _TP_PAIR)

    got = _plan(backend, gguf, tensor_parallel = True)

    assert got["pins"] == 0
    # And TP itself still engages: the gate declines to pin, it does not
    # quietly disable the mode.
    assert "--split-mode" in got["cmd"]
    assert got["cmd"][got["cmd"].index("--split-mode") + 1] == "tensor"


def test_the_same_pool_still_pins_when_tensor_parallelism_is_off(monkeypatch, tmp_path):
    """The control: identical cards, identical model, layer split. The gate has
    to be reading `tensor_parallel`, not the pool shape."""
    backend, gguf = _pin_backend(monkeypatch, tmp_path, memory = _TP_PAIR)

    got = _plan(backend, gguf)

    assert got["pins"] == 1


# --------------------------------------------------------------------------
# B6: a failed fit gives back the residency the pin was bought for
# --------------------------------------------------------------------------


def test_the_automatic_pin_stands_down_when_the_fit_fails(monkeypatch, tmp_path):
    """``--fit on`` with no layer plan is llama-server free to spill layers again.
    Keeping the pin there charges every image ~3.6x for a residency the load does
    not have."""
    backend, gguf = _pin_backend(monkeypatch, tmp_path, fit_raises = True)

    got = _plan(backend, gguf)

    # The premise: this is the except arm.
    assert got["fit"] == "on"
    assert got["ngl"] is None
    assert got["pins"] == 0
    assert got["backend"].vision_on_cpu is False


def test_the_paravirtual_pin_survives_a_failed_fit(monkeypatch, tmp_path):
    """Only the automatic reason is contingent on the fit. A virtualised Metal
    device returns corrupt vision output whatever the placement turns out to be,
    so that pin holds -- and the stand-down above must not have taken it with it."""
    backend, gguf = _pin_backend(monkeypatch, tmp_path, memory = [], paravirtual = True)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
    }

    def _boom(_path):
        raise RuntimeError("simulated size failure")

    # Raise inside the fit try, after the paravirtual pin was already seeded.
    backend._get_gguf_size_bytes = _boom

    got = _plan(backend, gguf)

    assert got["pins"] == 1
    assert got["backend"].vision_on_cpu is True


def test_a_hand_pinned_projector_survives_a_failed_fit(monkeypatch, tmp_path):
    """The user's own flag is an instruction, not a bet on the placement. It is
    in the extras, so it reaches the argv whatever the fit did, and the UI must
    still report the encoder as CPU-resident."""
    backend, gguf = _pin_backend(monkeypatch, tmp_path, fit_raises = True)

    got = _plan(backend, gguf, extra_args = [_PIN])

    assert got["pins"] == 1
    assert got["backend"].vision_on_cpu is True

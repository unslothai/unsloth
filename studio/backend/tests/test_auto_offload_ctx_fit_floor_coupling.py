# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The coupling that makes the Auto host-offload context safe for placement.

The failure mode guarded: ``_AUTO_OFFLOAD_CTX`` is read inside the placement
decision. The measured-KV subset loop falls through to it, lowers the context to
that value, and then RE-CHECKS whether any GPU subset can hold the model at the
lower context. That re-check can award residency, so the constant is not merely
cosmetic -- it is an input to which devices a load pins.

What makes it cosmetic in practice is a coupling that is nowhere written down:

    _AUTO_OFFLOAD_CTX >= the fit helpers' minimum context

The subset loop above the fallback already tried every subset at a context the fit
helpers floored at that minimum. Footprint is monotone in context, so a subset that
failed at the floor fails at anything at or above it, and the re-check can only ever
award below the floor. While the fallback was the literal 4096 the two were the same
number by accident of spelling. This change gave the fallback a name and left the
floor as a bare default argument on two helpers, so nothing connects them any more.
Lower the fallback (or raise the floor) and the re-check re-enters the live region
where 4096 and 8192 place a model differently.

Two tests, deliberately paired: one pins the invariant, one proves the invariant is
load-bearing rather than vacuous by breaking it and measuring what comes back.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_auto_offload_matrix_for_floor", _TESTS_DIR / "test_auto_offload_ctx_platform_matrix.py"
)
_matrix = importlib.util.module_from_spec(_spec)
import sys as _sys  # noqa: E402

# dataclasses resolves annotations through sys.modules, so the module has to be
# registered before it executes.
_sys.modules["_auto_offload_matrix_for_floor"] = _matrix
_spec.loader.exec_module(_matrix)

from core.inference import llama_cpp as _llama_cpp  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    _AUTO_OFFLOAD_CTX,
    _FIT_MIN_CTX,
    LlamaCppBackend,
)

Accelerator = _matrix.Accelerator
MIB = _matrix.MIB
PLATFORMS = _matrix.PLATFORMS
LINUX = next(p for p in PLATFORMS if p[0] == "linux")

# One 24 GB card with 20000 MiB free is 19280 MiB of pooled budget after the
# occupancy reserve. 17000 MiB of weights leaves 2280 MiB, which holds the KV of
# 4560 tokens at the 0.5 MiB/token rate the matrix harness prices: over the floor,
# under 8192. That is exactly the window in which the fallback value decides
# placement, and it is empty only because the floor closes it.
CARD = Accelerator("nvidia-single", False, ((0, 20_000, 24_000),))
MODEL_MIB = 17_000


def _fit_helper_floors() -> dict:
    """The minimum context each fit helper applies when the caller names none.

    Read from the signatures rather than hardcoded, because the auto loop's calls
    pass no ``min_ctx`` at all: these defaults, not ``_FIT_MIN_CTX``, are what
    actually floors the search on the path the fallback sits on.
    """
    return {
        name: inspect.signature(getattr(LlamaCppBackend, name)).parameters["min_ctx"].default
        for name in ("_fit_context_to_vram", "_cap_ctx_to_per_device_reserve")
    }


def _award_at(
    tmp_path,
    monkeypatch,
    offload_ctx: int,
    *,
    model_mib: int = MODEL_MIB,
):
    """Run the fallback with the constant set to ``offload_ctx``; report the award."""
    monkeypatch.setattr(_llama_cpp, "_AUTO_OFFLOAD_CTX", offload_ctx)
    backend, gguf = _matrix.cell_backend(
        _matrix._subdir(tmp_path, f"ctx-{offload_ctx}-{model_mib}"),
        monkeypatch,
        LINUX,
        CARD,
        model_fraction = 1.0,
    )
    backend._get_gguf_size_bytes = lambda _path: model_mib * MIB
    result, hits = _matrix._traced(lambda: _matrix._launch(backend, gguf, n_ctx = 0))
    assert _matrix.SITE_A in hits, "the cell no longer reaches the fallback"
    return {
        "awarded": _matrix.SITE_A_AWARD in hits,
        "fit": _matrix._flag(result["cmd"], "--fit"),
        "devices": _matrix._selected_devices(result["cmd"], result["env"]),
    }


def test_the_offload_context_never_sits_below_the_fit_search_floor():
    """The invariant itself. Nothing else in the tree states it.

    Both halves matter. ``_FIT_MIN_CTX`` is the named floor and the one a reader
    would check; the bare defaults on the two helpers are the ones the auto loop
    actually gets, because neither of its calls passes ``min_ctx``. Breaking either
    relation puts the re-check back in the region where the fallback moves devices.
    """
    floors = _fit_helper_floors()

    assert _AUTO_OFFLOAD_CTX >= _FIT_MIN_CTX, (
        f"_AUTO_OFFLOAD_CTX ({_AUTO_OFFLOAD_CTX}) dropped below _FIT_MIN_CTX "
        f"({_FIT_MIN_CTX}); the Site A re-check can now award GPU residency and the "
        "fallback is a placement decision, not a context default"
    )
    for name, floor in floors.items():
        assert _AUTO_OFFLOAD_CTX >= floor, (
            f"_AUTO_OFFLOAD_CTX ({_AUTO_OFFLOAD_CTX}) dropped below the default "
            f"min_ctx of {name} ({floor})"
        )
    # The helpers' defaults are what _FIT_MIN_CTX documents. If they ever diverge,
    # the comment on _FIT_MIN_CTX is wrong and so is the reasoning above.
    assert set(floors.values()) == {_FIT_MIN_CTX}, floors


@pytest.mark.parametrize(
    "offload_ctx,expect_award",
    [
        (256, True),
        (512, True),
        (1024, True),
        (2048, True),
        (3072, True),
        (_FIT_MIN_CTX, False),
        (6144, False),
        (_AUTO_OFFLOAD_CTX, False),
    ],
)
def test_the_floor_is_the_only_thing_keeping_the_fallback_out_of_placement(
    tmp_path, monkeypatch, offload_ctx, expect_award
):
    """The invariant is load-bearing, measured rather than argued.

    Same host, same model, same everything but the constant. Below the fit floor the
    re-check hands the model a device and turns ``--fit`` off, which is a placement
    change of exactly the kind the change is claimed not to make; at the floor and
    above it never does. This is why the test above is not a tautology.
    """
    outcome = _award_at(tmp_path, monkeypatch, offload_ctx)

    assert outcome["awarded"] is expect_award
    if expect_award:
        # Residency awarded: Unsloth owns placement and llama.cpp's fitter is off.
        assert outcome["fit"] == "off"
        assert outcome["devices"] == (0,)
    else:
        assert outcome["fit"] == "on"
        assert outcome["devices"] is None


@pytest.mark.parametrize(
    "free_mib,total_mib",
    [(20_000, 24_000), (12_000, 16_000), (9_000, 0)],
    ids = ["24g-card", "16g-card", "shared-pool"],
)
def test_no_model_size_awards_residency_at_or_above_the_floor(
    tmp_path, monkeypatch, free_mib, total_mib
):
    """The same claim swept over model sizes, on three cards including a shared pool
    reporting ``total_mib == 0``.

    Counted rather than asserted cell by cell, because "awards below the floor" is
    true only in the band where the weights leave room for a small KV and not a
    large one, and which fractions land in that band depends on the card. What has
    to hold everywhere is the pair: zero awards at the floor and above it, and a
    non-zero number below it on every card, so no card is silent merely because
    nothing there could ever be awarded.
    """
    card = Accelerator(f"card-{free_mib}", False, ((0, free_mib, total_mib),))
    fractions = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    below_floor = (256, 1024, 2048)
    awards = {"below": 0, "at-or-above": 0}
    reached = 0

    for fraction in fractions:
        model_mib = int(fraction * free_mib)
        for offload_ctx in (*below_floor, _FIT_MIN_CTX, _AUTO_OFFLOAD_CTX):
            monkeypatch.setattr(_llama_cpp, "_AUTO_OFFLOAD_CTX", offload_ctx)
            backend, gguf = _matrix.cell_backend(
                _matrix._subdir(tmp_path, f"{free_mib}-{model_mib}-{offload_ctx}"),
                monkeypatch,
                LINUX,
                card,
                model_fraction = 1.0,
            )
            backend._get_gguf_size_bytes = lambda _path: model_mib * MIB
            _result, hits = _matrix._traced(lambda: _matrix._launch(backend, gguf, n_ctx = 0))
            if _matrix.SITE_A not in hits:
                continue
            reached += 1
            if _matrix.SITE_A_AWARD in hits:
                awards["below" if offload_ctx < _FIT_MIN_CTX else "at-or-above"] += 1

    assert reached, "no cell on this card reached the fallback"
    assert awards["at-or-above"] == 0
    assert awards["below"] > 0


def test_the_projector_residency_floor_is_the_fit_floor_and_not_the_offload_context():
    """``_MMPROJ_FIT_FLOOR_CTX`` happened to equal the old fallback; it does not
    follow the new one, and must not.

    The projector probe decides whether a vision encoder stays on the GPU by pricing
    the load at that floor. The number it wants is the LOWEST context at which
    placement can still award GPU residency, and that is the fit floor: the subset
    loop's fit bottoms out there and returns it whenever it fits. The offload
    fallback is what the loop emits after it has already surrendered and handed
    placement to ``--fit``, which is past the point the probe is asking about.

    Pinned here because the two constants were the same literal before this change,
    so a reader could reasonably assume they still move together.

    The value check is on the fit floor only. Raising ``_FIT_MIN_CTX`` to 8192 put it
    on the same number as ``_AUTO_OFFLOAD_CTX`` again, so ``!=`` on the values no
    longer separates the two concepts: it would fail on a correct tree and pass on a
    wrong one the moment the offload fallback moved. Assert instead that the projector
    floor is DERIVED from the fit floor and never from the offload fallback, which is
    the mistake this test exists to catch and holds whatever the two numbers are.
    """
    assert LlamaCppBackend._MMPROJ_FIT_FLOOR_CTX == _FIT_MIN_CTX
    source = inspect.getsource(LlamaCppBackend)
    assignment = re.search(r"^\s*_MMPROJ_FIT_FLOOR_CTX\s*=\s*(.+)$", source, re.MULTILINE)
    assert assignment, "_MMPROJ_FIT_FLOOR_CTX is no longer a class attribute of the backend"
    assert "_AUTO_OFFLOAD_CTX" not in assignment.group(1), (
        "_MMPROJ_FIT_FLOOR_CTX is being set from _AUTO_OFFLOAD_CTX. It must track the FIT "
        "floor: the projector probe asks for the lowest context at which placement can "
        f"still award GPU residency, which the offload fallback is past. Got {assignment.group(1)!r}."
    )

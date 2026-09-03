# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two invariants that make ``_AUTO_OFFLOAD_CTX`` safe to move.

Raising the Auto offload context from 4096 to 8192 changes no GPU placement, but
not because the value is unimportant. It is safe because of a coupling that was
previously implicit in the two numbers being the same literal:

1. ``_AUTO_OFFLOAD_CTX >= _FIT_MIN_CTX``. The Auto offload branch re-checks
   whether some subset holds the model at the reduced context. That re-check can
   only ever award residency BELOW ``_FIT_MIN_CTX``, because both
   ``_fit_context_to_vram`` and ``_cap_ctx_to_per_device_reserve`` floor there,
   so a subset winnable at or above the floor was already taken by the fit loop
   that runs first. Once the constant drops under the floor the re-check re-enters
   the live region and the Auto context starts deciding which GPUs hold the model,
   which is a different and much larger change than picking a chat length.

2. The published UI ceiling tracks the same constant. ``max_context_length`` is
   the threshold the chat settings sheet warns above. If it is anchored below the
   context Auto actually selects, every Auto load in this branch exceeds its own
   published ceiling and warns about itself, telling the user to lower the context
   or leave it on Auto when Auto is what produced the value.

Neither invariant is expressible as a type, and both are one edited literal away
from silently breaking, so they are pinned here. No GPU, subprocess or GGUF I/O.
Cross-platform: Linux, macOS, Windows, WSL.
"""

from __future__ import annotations

import inspect
import re
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

from core.inference.llama_cpp import (  # noqa: E402
    _AUTO_OFFLOAD_CTX,
    _FIT_MIN_CTX,
    LlamaCppBackend,
)

# Reuse the two existing mirrors rather than growing a third. Both stub the same
# way this file does, so importing them costs no extra setup. Sibling imports need
# the tests dir on the path: pytest inserts rootdir, not this package.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_context_fit import _drive  # noqa: E402
from test_llama_cpp_max_context_threshold import (  # noqa: E402
    _compute_max_available_ctx,
)


def test_auto_offload_context_is_not_below_the_fit_floor():
    """Invariant 1. Below the floor, the offload re-check starts awarding GPU
    residency again and the constant stops being a display choice."""
    assert _AUTO_OFFLOAD_CTX >= _FIT_MIN_CTX


def test_the_fit_helpers_still_floor_where_the_invariant_assumes():
    """Invariant 1 holds against a floor that is NOT ``_FIT_MIN_CTX``.

    Neither auto call site passes ``min_ctx``, so what actually bounds the search
    is the bare default on each helper; ``_FIT_MIN_CTX`` is only handed in
    explicitly on the Apple arm. The two agree today, which is what lets the
    constant above stand in for the floor, and this is where that agreement is
    pinned. A default lowered here moves the dead region without touching either
    constant, and nothing else in the tree would notice.
    """
    for func in (
        LlamaCppBackend._fit_context_to_vram,
        LlamaCppBackend._cap_ctx_to_per_device_reserve,
    ):
        params = inspect.signature(func).parameters
        min_ctx = params.get("min_ctx")
        assert min_ctx is not None, (
            f"{func.__qualname__} no longer takes min_ctx; the Auto offload "
            "re-check's dead region is defined by that floor"
        )
        assert min_ctx.default == _FIT_MIN_CTX, (
            f"{func.__qualname__} defaults min_ctx to {min_ctx.default}, not "
            f"_FIT_MIN_CTX ({_FIT_MIN_CTX}). The Auto offload re-check awards GPU "
            "residency below the floor, so the two must not drift apart"
        )


def test_the_published_ui_ceiling_tracks_the_auto_offload_context():
    """Invariant 2, asserted on the source because the value is produced deep
    inside ``load_model`` and the failure is a stale literal, not a bad number.
    """
    source = inspect.getsource(LlamaCppBackend.load_model)
    anchor = re.search(
        r"max_available_ctx\s*=\s*min\(\s*([A-Za-z_0-9]+)\s*,\s*native_ctx_for_cap",
        source,
    )
    assert anchor is not None, "the no-fit UI safe-zone anchor moved or was renamed"
    assert anchor.group(1) == "_AUTO_OFFLOAD_CTX", (
        "the UI safe zone is anchored at a literal again; it must follow the "
        "Auto offload context or every Auto load in this branch warns about itself"
    )


@pytest.mark.parametrize(
    "native, model_gib, gpus",
    [
        # MiniMax-like: weights alone dwarf a single large card.
        (196608, 131, [(0, 97_000)]),
        # Nothing fits even pooled across four cards.
        (131072, 400, [(0, 80_000), (1, 80_000), (2, 80_000), (3, 80_000)]),
        # Mixed sizes, so the ranked-subset walk runs before giving up.
        (131072, 200, [(0, 48_000), (1, 24_000), (2, 8_000)]),
        # Native below the fallback: both sides must land on native, not 8192.
        (2048, 200, [(0, 80_000)]),
    ],
)
def test_auto_never_publishes_a_ceiling_below_the_context_it_runs(native, model_gib, gpus):
    """The behavioural half of invariant 2, driven through both real mirrors.

    ``_max_context_length`` is what the status route serves as
    ``max_context_length``, and the chat sheet warns when the running context
    exceeds it. On an offloading model the running context IS the Auto fallback,
    so a ceiling computed from a different constant makes the load warn about
    itself. Drive the ceiling probe and the context decision from the same inputs
    and require that they agree.
    """
    published = _compute_max_available_ctx(native_ctx = native, model_gib = model_gib, gpus = gpus)
    plan = _drive(n_ctx = 0, model_gib = model_gib, gpus = gpus, native_ctx = native)
    running = plan["c_arg"]

    assert running > 0
    assert running <= published, (
        f"Auto runs at {running} but publishes a ceiling of {published}, "
        "so the chat sheet warns on a context Auto chose itself"
    )

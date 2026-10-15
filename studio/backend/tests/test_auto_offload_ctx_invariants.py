# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two invariants: _AUTO_OFFLOAD_CTX stays >= _FIT_MIN_CTX, and the published ceiling tracks it."""

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
    """The fit helpers' bare min_ctx defaults must match the floor, since auto call sites pass none."""
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
        (196608, 131, [(0, 97_000)]),
        (131072, 400, [(0, 80_000), (1, 80_000), (2, 80_000), (3, 80_000)]),
        (131072, 200, [(0, 48_000), (1, 24_000), (2, 8_000)]),
        (2048, 200, [(0, 80_000)]),
    ],
)
def test_auto_never_publishes_a_ceiling_below_the_context_it_runs(native, model_gib, gpus):
    """Auto's published ceiling must match the context it runs, or Auto loads warn about themselves."""
    published = _compute_max_available_ctx(native_ctx = native, model_gib = model_gib, gpus = gpus)
    plan = _drive(n_ctx = 0, model_gib = model_gib, gpus = gpus, native_ctx = native)
    running = plan["c_arg"]

    assert running > 0
    assert running <= published, (
        f"Auto runs at {running} but publishes a ceiling of {published}, "
        "so the chat sheet warns on a context Auto chose itself"
    )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_AUTO_OFFLOAD_CTX must stay at or above the fit helpers' min_ctx defaults, or placement changes."""

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

# dataclasses resolves annotations via sys.modules: register before exec.
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

# 20000 MiB free gives 19280 MiB after the reserve; 17000 MiB of weights leaves 2280 (4560 tokens).
CARD = Accelerator("nvidia-single", False, ((0, 20_000, 24_000),))
MODEL_MIB = 17_000


def _fit_helper_floors() -> dict:
    """Fit helpers' min_ctx defaults, read from signatures since the auto loop never passes one."""
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
    """Offload context stays at or above both _FIT_MIN_CTX and the helpers' bare min_ctx defaults."""
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
    """Below the fit floor the re-check awards residency and turns --fit off; at or above it never does."""
    outcome = _award_at(tmp_path, monkeypatch, offload_ctx)

    assert outcome["awarded"] is expect_award
    if expect_award:
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
    """Sweeps model sizes on three cards: no award at or above the floor, and some below it on each card."""
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
    """_MMPROJ_FIT_FLOOR_CTX must derive from the fit floor, never the offload fallback."""
    assert LlamaCppBackend._MMPROJ_FIT_FLOOR_CTX == _FIT_MIN_CTX
    source = inspect.getsource(LlamaCppBackend)
    assignment = re.search(r"^\s*_MMPROJ_FIT_FLOOR_CTX\s*=\s*(.+)$", source, re.MULTILINE)
    assert assignment, "_MMPROJ_FIT_FLOOR_CTX is no longer a class attribute of the backend"
    assert "_AUTO_OFFLOAD_CTX" not in assignment.group(1), (
        "_MMPROJ_FIT_FLOOR_CTX is being set from _AUTO_OFFLOAD_CTX. It must track the FIT "
        "floor: the projector probe asks for the lowest context at which placement can "
        f"still award GPU residency, which the offload fallback is past. Got {assignment.group(1)!r}."
    )

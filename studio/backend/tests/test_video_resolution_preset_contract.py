# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract: every video resolution the UI offers is a real family preset, and vice versa.

The Video tab's Resolution select is populated from ``status.defaults.resolution_presets``,
which is ``VideoFamily.resolution_presets`` copied straight through
``video.py::status()`` -> ``VideoGenerationDefaults``. Before a model is loaded it falls back
to ``FALLBACK_RESOLUTION_PRESETS`` in ``video-page.tsx``. Two things have to hold:

  * the offline fallback names only sizes some family actually declares, so the first paint
    cannot offer a shape no checkpoint was trained at;
  * every family declares at least one preset, because ``video.py``'s generate path indexes
    ``fam.resolution_presets[0]`` unguarded when width/height are omitted -- an empty tuple
    there is an IndexError on the very first generate of that family.

Pure-module: no torch, no network, no browser. The frontend half reads the TSX source, the
same way the other cross-language contract checks in this suite do.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.inference.video_families import (
    detect_video_family,
    snap_video_size,
    supported_video_family_names,
)
from models.inference import VideoGenerationDefaults

_BACKEND = Path(__file__).resolve().parent.parent
_VIDEO_PAGE = _BACKEND.parent / "frontend" / "src" / "features" / "video" / "video-page.tsx"

_FAMILY_NAMES = supported_video_family_names()


def _family(name: str):
    fam = detect_video_family("", override = name)
    assert fam is not None, f"{name} is listed but does not resolve"
    return fam


def _all_presets() -> set[tuple[int, int]]:
    """Every (width, height) any family offers -- the domain the UI may draw from."""
    return {tuple(p) for name in _FAMILY_NAMES for p in _family(name).resolution_presets}


# ── the family registry ───────────────────────────────────────────────────────


def test_the_registry_is_not_empty():
    # Guards the parametrised cases below from silently covering nothing.
    assert len(_FAMILY_NAMES) >= 5


@pytest.mark.parametrize("name", _FAMILY_NAMES)
def test_every_family_declares_at_least_one_resolution_preset(name):
    """``video.py`` resolves an omitted size with ``width or fam.resolution_presets[0][0]``.
    A family with an empty tuple would IndexError there instead of generating."""
    fam = _family(name)
    assert len(fam.resolution_presets) >= 1, (
        f"{name} declares no resolution presets; generate() indexes resolution_presets[0] "
        "unguarded when the request omits width/height, so this is an IndexError, and the "
        "UI's Resolution select would render empty"
    )
    # The unguarded index itself, exactly as the generate path performs it.
    assert fam.resolution_presets[0][0] > 0 and fam.resolution_presets[0][1] > 0


@pytest.mark.parametrize("name", _FAMILY_NAMES)
def test_every_preset_survives_the_family_snap_unchanged(name):
    """A preset is what the UI sends verbatim, so it must already sit on the family's
    grid -- otherwise the clip comes back a different size than the one selected."""
    fam = _family(name)
    for width, height in fam.resolution_presets:
        assert isinstance(width, int) and isinstance(height, int)
        assert snap_video_size(fam, width, height) == (width, height), (
            f"{name} preset {width}x{height} is not a multiple of "
            f"resolution_multiple={fam.resolution_multiple}, so the pipeline floors it and the "
            "recorded size disagrees with the rendered clip"
        )


@pytest.mark.parametrize("name", _FAMILY_NAMES)
def test_presets_are_unique_and_land_in_the_status_payload(name):
    """Through the real ``VideoBackend.status()``, not a hand-built VideoGenerationDefaults.

    Constructing the model here would assert only that Pydantic round-trips a list, and would
    stay green if status() ever hardcoded a preset list or stopped emitting the key at all --
    which is the loaded UI offering shapes the checkpoint was never trained at.
    """
    import core.inference.video as video_module

    fam = _family(name)
    presets = [tuple(p) for p in fam.resolution_presets]
    assert len(set(presets)) == len(presets), f"{name} repeats a preset: {presets}"

    backend = video_module.VideoBackend()
    backend._state = video_module._VideoLoadState(
        pipe = object(),
        family = fam,
        repo_id = f"unsloth/{name}",
        base_repo = fam.base_repo,
        device = "cpu",
        dtype = "bfloat16",
        kind = "pipeline",
    )
    defaults_payload = backend.status()["defaults"]
    assert defaults_payload is not None, "status() stopped emitting defaults"

    # ...and through the response model the route declares, which is what reaches the browser.
    defaults = VideoGenerationDefaults(**defaults_payload)
    assert (
        [tuple(p) for p in defaults.resolution_presets] == presets
    ), f"{name}: status() serves {defaults.resolution_presets} but the family declares {presets}"
    assert defaults.frame_step == fam.frame_step
    assert defaults.resolution_multiple == fam.resolution_multiple


# ── the frontend fallback ─────────────────────────────────────────────────────


def _fallback_presets() -> list[tuple[int, int]]:
    """``FALLBACK_RESOLUTION_PRESETS`` from video-page.tsx."""
    src = _VIDEO_PAGE.read_text(encoding = "utf-8")
    start = src.index("const FALLBACK_RESOLUTION_PRESETS")
    body = src[start : src.index("];", start)]
    pairs = [(int(w), int(h)) for w, h in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", body)]
    assert pairs, f"failed to parse FALLBACK_RESOLUTION_PRESETS out of {body!r}"
    return pairs


def test_the_frontend_fallback_offers_only_real_family_presets():
    fallback = _fallback_presets()
    real = _all_presets()
    unknown = [p for p in fallback if p not in real]
    assert not unknown, (
        f"video-page.tsx offers {unknown} before a model is loaded, but no VideoFamily "
        "declares those sizes; a user who picks one gets a shape no checkpoint was trained at"
    )


def test_the_frontend_fallback_is_a_usable_default_for_the_family_it_mirrors():
    """The fallback exists to populate the select on first paint, so it must be non-empty
    and its first entry must be some family's DEFAULT (presets[0]) -- the size the loader
    plans against -- not an arbitrary member of the union."""
    fallback = _fallback_presets()
    assert len(fallback) >= 1
    # An empty tuple is its own test above; skip it here so that failure is not double-reported.
    firsts = {
        tuple(fam.resolution_presets[0])
        for fam in (_family(name) for name in _FAMILY_NAMES)
        if fam.resolution_presets
    }
    assert (
        fallback[0] in firsts
    ), f"the fallback leads with {fallback[0]}, which is not the default preset of any family"


def test_the_resolution_select_renders_the_backend_presets_not_a_hardcoded_list():
    """The select must map over the memo fed by ``status.defaults.resolution_presets``;
    if it were rebound to the fallback the backend's per-family list would never show."""
    src = _VIDEO_PAGE.read_text(encoding = "utf-8")
    memo = src[src.index("const resolutionPresets = useMemo<") :]
    memo = memo[: memo.index("\n  }, [")]
    assert "status?.defaults?.resolution_presets" in memo
    assert "FALLBACK_RESOLUTION_PRESETS" in memo
    # Only the empty/absent case falls back.
    assert "presets && presets.length > 0" in memo
    assert "{resolutionPresets.map(([w, h], i) => (" in src
    # And the generate call sends the SELECTED preset, so the offered pair is the one rendered.
    assert "const preset = resolutionPresets[resolutionIdx] ?? resolutionPresets[0];" in src

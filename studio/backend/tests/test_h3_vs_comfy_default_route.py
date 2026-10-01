# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 routing and canvas gaps found against ComfyUI's H3 int8 template.

* ``/api/system.diffusers_offload_tiers`` publishes the Diffusers tiers this backend can run, so the
  picker stops sending 24-48 GB cards to the 2.5-3.2x slower GGUF (sd-cli) row. Off with
  ``UNSLOTH_H3_DIFFUSERS_WIDE_TIERS=0``.
* 864x480 (ComfyUI's template default, ResolutionSelector 16:9 at 0.4 MP, multiple 32) used to be
  refused with 422. Off with ``UNSLOTH_VIDEO_H3_480P=0``.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MAIN = (_BACKEND / "main.py").read_text(encoding = "utf-8")


def _src(name: str) -> str:
    node = next(
        n for n in ast.walk(ast.parse(_MAIN)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(_MAIN, node)


def test_the_tiers_are_published_on_the_system_route():
    assert '"diffusers_offload_tiers": _diffusers_offload_tiers()' in _src("get_system_info")


def test_the_route_reader_is_torch_free_and_keyed_by_lowercased_repo(monkeypatch):
    for name in ("torch", "diffusers", "torchao"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delenv("UNSLOTH_H3_DIFFUSERS_WIDE_TIERS", raising = False)
    namespace: dict = {}
    exec(_src("_diffusers_offload_tiers"), namespace)  # noqa: S102 -- the real body
    tiers = namespace["_diffusers_offload_tiers"]()
    assert list(tiers) == ["minimaxai/minimax-h3"]
    for tier in tiers["minimaxai/minimax-h3"]:
        assert set(tier) == {"gpu_gb", "system_ram_gb", "requires_quantised_streaming"}
        assert tier["requires_quantised_streaming"] is True
    # Strictly wider than the catalog's static 30 GiB / 80 GiB streamed tier somewhere.
    assert any(
        t["gpu_gb"] < 30.0 or t["system_ram_gb"] < 80.0 for t in tiers["minimaxai/minimax-h3"]
    )


@pytest.mark.parametrize("flag", ["0", "false", "off", " NO "])
def test_the_kill_switch_withdraws_every_extra_tier(monkeypatch, flag):
    monkeypatch.setenv("UNSLOTH_H3_DIFFUSERS_WIDE_TIERS", flag)
    from core.inference.video_minimax_h3 import h3_diffusers_fit_tiers

    assert h3_diffusers_fit_tiers() == []
    namespace: dict = {}
    exec(_src("_diffusers_offload_tiers"), namespace)  # noqa: S102
    assert namespace["_diffusers_offload_tiers"]() == {}


def _reload_families(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("UNSLOTH_VIDEO_H3_480P", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_VIDEO_H3_480P", flag)
    import core.inference.video_families as vf

    return importlib.reload(vf)


@pytest.fixture
def families(monkeypatch):
    yield lambda flag = None: _reload_families(monkeypatch, flag)
    monkeypatch.delenv("UNSLOTH_VIDEO_H3_480P", raising = False)
    import core.inference.video_families as vf

    importlib.reload(vf)


def test_comfy_template_default_864x480_is_accepted(families):
    vf = families()
    fam = vf.detect_video_family("MiniMaxAI/MiniMax-H3")
    for width, height in ((864, 480), (480, 864)):
        vf.validate_video_request_shape(fam, width, height, 124)
        assert width % 32 == 0 and height % 32 == 0
    assert fam.resolution_presets[0] == (1344, 768), "16:9 1344x768 stays the default"


def test_864x480_kill_switch_restores_the_422(families):
    vf = families("0")
    fam = vf.detect_video_family("MiniMaxAI/MiniMax-H3")
    assert (864, 480) not in fam.resolution_presets
    with pytest.raises(vf.VideoShapeError):
        vf.validate_video_request_shape(fam, 864, 480, 124)


def test_off_canvas_sizes_are_still_refused(families):
    vf = families()
    fam = vf.detect_video_family("MiniMaxAI/MiniMax-H3")
    for width, height in ((848, 480), (864, 486), (512, 512)):
        with pytest.raises(vf.VideoShapeError):
            vf.validate_video_request_shape(fam, width, height, 124)


def test_the_estimators_size_864x480_below_960x544(families):
    families()
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_vram_gb

    assert estimate_h3_diffusers_vram_gb(864, 480, 124) < estimate_h3_diffusers_vram_gb(960, 544, 124)

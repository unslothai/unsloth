# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Publish Diffusers offload tiers so 24-48 GB cards avoid the slower GGUF row; 864x480 must not 422."""

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
    assert estimate_h3_diffusers_vram_gb(864, 480, 124) < estimate_h3_diffusers_vram_gb(
        960, 544, 124
    )


def _streamed_set_tier_gib() -> float:
    import math
    import sys
    from pathlib import Path

    backend = str(Path(__file__).resolve().parents[1])
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from core.inference.video_minimax_h3 import H3_DIFFUSERS_HOST_RAM_STREAMED_SET_GB

    return float(math.ceil(H3_DIFFUSERS_HOST_RAM_STREAMED_SET_GB * 1e9 / 2**30))


_STREAMED_SET_TIER_GIB = _streamed_set_tier_gib()


@pytest.mark.parametrize(
    "te_stream, arena, expected",
    [
        (
            None,
            None,
            [
                {
                    "gpu_gb": 14.0,
                    "system_ram_gb": _STREAMED_SET_TIER_GIB,
                    "requires_quantised_streaming": True,
                }
            ],
        ),
        (
            "0",
            None,
            [{"gpu_gb": 30.0, "system_ram_gb": 61.0, "requires_quantised_streaming": True}],
        ),
        (
            None,
            "0",
            [{"gpu_gb": 14.0, "system_ram_gb": 80.0, "requires_quantised_streaming": True}],
        ),
        ("0", "0", []),
    ],
)
def test_each_widening_follows_the_kill_switch_of_the_behaviour_it_relies_on(
    monkeypatch, te_stream, arena, expected
):
    # Each wide tier depends on its streaming behaviour; disabling one must withdraw only its widening.
    monkeypatch.delenv("UNSLOTH_H3_DIFFUSERS_WIDE_TIERS", raising = False)
    for name, value in (
        ("UNSLOTH_H3_TE_STREAM", te_stream),
        ("UNSLOTH_DIFFUSION_PIN_ARENA", arena),
    ):
        if value is None:
            monkeypatch.delenv(name, raising = False)
        else:
            monkeypatch.setenv(name, value)
    from core.inference.video_minimax_h3 import h3_diffusers_fit_tiers

    assert h3_diffusers_fit_tiers() == expected


def test_the_ram_tier_is_the_single_count_host_floor_in_gib():
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_host_ram_gb

    floor_gb = estimate_h3_diffusers_host_ram_gb(0.0, text_encoder_gb = 27.2, transformer_gb = 20.3)
    assert floor_gb == pytest.approx(64.5)
    assert 60.0 < floor_gb * 1e9 / 2**30 <= 61.0


def test_a_streamed_int8_set_is_priced_at_the_measured_floor_not_the_component_sum():
    from core.inference.video_minimax_h3 import (
        H3_DIFFUSERS_HOST_RAM_STREAMED_RSS_GB,
        H3_DIFFUSERS_HOST_RAM_STREAMED_SET_GB,
        estimate_h3_diffusers_host_ram_gb,
    )

    summed = estimate_h3_diffusers_host_ram_gb(30.0, text_encoder_gb = 27.2, transformer_gb = 20.3)
    streamed = estimate_h3_diffusers_host_ram_gb(
        30.0, text_encoder_gb = 27.2, transformer_gb = 20.3, text_encoder_streamed = True
    )
    assert summed == pytest.approx(64.5)
    assert streamed == H3_DIFFUSERS_HOST_RAM_STREAMED_SET_GB < summed
    # A 58 GiB cgroup (62.3 GB) holds the measured set: main renders it there.
    assert streamed < 58 * 2**30 / 1e9
    assert estimate_h3_diffusers_host_ram_gb(
        30.0, text_encoder_gb = 66.7, transformer_gb = 20.3, text_encoder_streamed = True
    ) == pytest.approx(66.7 + 20.3 + 11.1 + 5.9)
    assert H3_DIFFUSERS_HOST_RAM_STREAMED_RSS_GB == 70.0
    assert estimate_h3_diffusers_host_ram_gb(
        30.0,
        text_encoder_gb = 27.2,
        transformer_gb = 20.3,
        transformer_streamed = True,
        text_encoder_streamed = True,
    ) == pytest.approx(84.8)


def test_the_guard_prices_a_streamed_conditioner_at_the_streamed_set_floor(monkeypatch):
    import core.inference.video_minimax_h3 as h3

    floor = h3.H3_DIFFUSERS_HOST_RAM_STREAMED_SET_GB
    kw = dict(text_encoder_gb = 27.2, transformer_gb = 20.3)
    monkeypatch.setattr(h3, "h3_host_capacity_bytes", lambda: int((floor + 0.6) * 1e9))
    assert h3.h3_host_ram_shortfall(30.0, text_encoder_streamed = True, **kw) is None
    monkeypatch.setattr(h3, "h3_host_capacity_bytes", lambda: int((floor - 1.0) * 1e9))
    message = h3.h3_host_ram_shortfall(30.0, text_encoder_streamed = True, **kw)
    assert message is not None and f"{floor:.0f} GB" in message


def test_the_vram_tier_admits_only_cards_that_render_the_default_request(monkeypatch):
    """The picker's VRAM tier is the generate guard's floor for the page's default request (first preset, default
    length), so the row it selects is never refused on the first render. A 12 GB card keeps GGUF as its default."""
    monkeypatch.delenv("UNSLOTH_H3_DIFFUSERS_WIDE_TIERS", raising = False)
    monkeypatch.delenv("UNSLOTH_H3_TE_STREAM", raising = False)
    from core.inference.video_families import detect_video_family
    from core.inference.video_minimax_h3 import (
        estimate_h3_diffusers_vram_gb,
        h3_diffusers_fit_tiers,
    )

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    width, height = fam.resolution_presets[0]
    floor_gib = (
        estimate_h3_diffusers_vram_gb(
            width,
            height,
            fam.default_num_frames,
            transformer_streamed = True,
            text_encoder_streamed = True,
        )
        * 1e9
        / 2**30
    )
    (tier,) = h3_diffusers_fit_tiers()
    assert floor_gib <= tier["gpu_gb"] < floor_gib + 0.5
    assert tier["gpu_gb"] > 12.0
    assert tier["gpu_gb"] <= 15.99

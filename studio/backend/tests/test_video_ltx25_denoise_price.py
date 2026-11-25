# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5's request guard prices the denoise phase from G4 measurements once the connector, prompt-row and VAE
encoder levers are all engaged: 768x512x49 (and the two 6k-token shapes) are admitted on a 6 GiB card, 1280x704x121
(OOM there) is still refused, and nothing the base overhead admitted is refused."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video as V  # noqa: E402
from core.inference import video_ltx2 as L  # noqa: E402
from core.inference import video_ltx25 as v25  # noqa: E402

# Colab G4, 6 GiB free after the ballast, all three levers: what the request guard read
FLOOR_6GIB, DECODE_FLOOR_6GIB, AVAILABLE_6GIB = 3_329, 1_124, 5_356
ENCODER_MIB = 612
# (width, height, frames) -> rendered at 6 GiB (guards bypassed by a probe)
CELLS = {
    (768, 512, 49): True,
    (768, 512, 121): True,
    (1280, 704, 49): True,
    (1280, 704, 121): False,  # OOM at the denoise
}


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for env in (
        V.VIDEO_LTX25_DENOISE_PRICE_ENV,
        L.LTX2_PARK_VAE_ENCODER_ENV,
        v25.RELEASE_PROMPT_ROWS_ENV,
        v25.COMPACT_CONDITIONING_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(v25, "two_stage_enabled", lambda: True)
    monkeypatch.setattr(V.sys if hasattr(V, "sys") else __import__("sys"), "platform", "linux")
    monkeypatch.setattr(torch.version, "hip", None, raising = False)


class AutoencoderKLLTX2Video(torch.nn.Module):
    def __init__(self, encoder_mib = ENCODER_MIB):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.register_buffer("w", torch.empty(encoder_mib << 19, dtype = torch.bfloat16))


def _state(**overrides):
    connectors = types.SimpleNamespace(**{L._LTX2_LEAN_CONNECTOR_ATTR: True})
    pipe = types.SimpleNamespace(
        connectors = connectors,
        vae = AutoencoderKLLTX2Video(),
        vae_spatial_compression_ratio = 32,
        vae_temporal_compression_ratio = 8,
    )
    state = types.SimpleNamespace(
        pipe = pipe, offload_policy = "streaming", transformer_quant = "int8", device = "cuda"
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _price(
    state,
    shape,
    family = "ltx-2.5",
    conditioned = False,
    cfg = False,
    fps = None,
):
    w, h, f = shape
    return V._ltx25_denoise_price_mib(
        state,
        family = family,
        width = w,
        height = h,
        frames = f,
        conditioned = conditioned,
        cfg = cfg,
        fps = fps,
    )


def _guard(shape, denoise_mib):
    w, h, f = shape
    return V.video_offload_shortfall_message(
        family = "ltx-2.5",
        floor_mib = FLOOR_6GIB,
        available_mib = AVAILABLE_6GIB,
        placement = "'streaming' offload",
        width = w,
        height = h,
        frames = f,
        decode_floor_mib = DECODE_FLOOR_6GIB,
        denoise_mib = denoise_mib,
    )


@pytest.mark.parametrize("shape", list(CELLS))
def test_6gib_admits_exactly_the_shapes_that_rendered_there(shape):
    state = _state()
    assert (_guard(shape, _price(state, shape)) is None) == CELLS[shape]
    # the base overhead refuses every one of them
    assert _guard(shape, None) is not None


def test_price_covers_the_measured_peaks_plus_what_grew_outside_torch():
    state = _state()
    measured = {
        (768, 512, 49): 536,
        (768, 512, 121): 900,
        (1280, 704, 49): 899,
        (1280, 704, 121): 1_725,
    }
    for shape, over_floor in measured.items():
        assert _price(state, shape) >= over_floor + 564
    # the connector phase (424 MiB over the floor, any shape) is covered even for a tiny clip
    assert _price(state, (256, 256, 9)) >= 424 + 564


def test_price_never_exceeds_what_the_floor_with_the_encoder_and_the_base_overhead_asked():
    state = _state()
    cap = V.DEFAULT_BASE_OVERHEAD_MIB + ENCODER_MIB
    assert _price(state, (1920, 1088, 241)) == cap
    for shape in CELLS:
        assert _price(state, shape) <= cap


@pytest.mark.parametrize(
    "why",
    [
        "conditioned",
        "family",
        "group_tier",
        "bf16_dit",
        "encoder_on_card",
        "no_lean_connectors",
        "price_off",
        "release_off",
        "compact_off",
        "parking_off",
        "one_stage",
    ],
)
def test_unmeasured_configurations_keep_the_base_overhead(why, monkeypatch):
    state = _state()
    kwargs = {}
    if why == "conditioned":
        kwargs["conditioned"] = True
    elif why == "family":
        kwargs["family"] = "ltx-2.3"
    elif why == "group_tier":
        state.offload_policy = "group"
    elif why == "bf16_dit":
        state.transformer_quant = None
    elif why == "encoder_on_card":
        state.pipe.vae.encoder.w = torch.empty(0, device = "meta")
    elif why == "no_lean_connectors":
        state.pipe.connectors = types.SimpleNamespace()
    elif why == "price_off":
        monkeypatch.setenv(V.VIDEO_LTX25_DENOISE_PRICE_ENV, "0")
    elif why == "release_off":
        monkeypatch.setenv(v25.RELEASE_PROMPT_ROWS_ENV, "0")
    elif why == "compact_off":
        monkeypatch.setenv(v25.COMPACT_CONDITIONING_ENV, "0")
    elif why == "parking_off":
        monkeypatch.setenv(L.LTX2_PARK_VAE_ENCODER_ENV, "0")
    elif why == "one_stage":
        monkeypatch.setattr(v25, "two_stage_enabled", lambda: False)
    assert _price(state, (768, 512, 49), **kwargs) is None


def test_without_a_price_the_guard_is_unchanged():
    for shape in CELLS:
        w, h, f = shape
        common = dict(
            family = "ltx-2.5",
            floor_mib = FLOOR_6GIB,
            available_mib = 7_404,
            placement = "'streaming' offload",
            width = w,
            height = h,
            frames = f,
            decode_floor_mib = DECODE_FLOOR_6GIB,
        )
        assert V.video_offload_shortfall_message(**common) == V.video_offload_shortfall_message(
            **common, denoise_mib = None
        )


@pytest.mark.parametrize(
    "dev, cfg, priced",
    [(False, False, True), (False, True, False), (True, True, False), (True, False, False)],
    ids = ["distilled_cfg1", "distilled_cfg3", "dev_cfg3", "dev_cfg1"],
)
def test_only_the_measured_distilled_two_stage_render_without_cfg_is_priced(dev, cfg, priced):
    """CFG batches the negative rows beside the positive ones and the dev checkpoint renders in one stage outside the
    connector reuse: neither releases the prompt rows, so both keep the base overhead (refused at 6 GiB, as before)."""
    state = _state()
    state.pipe._unsloth_ltx25_dev = dev
    shape = (768, 512, 49)
    price = _price(state, shape, cfg = cfg)
    assert (price is not None) == priced
    assert (_guard(shape, price) is None) == priced


def test_the_measured_peak_key_names_the_engaged_levers(monkeypatch):
    state = _state()
    state.pipe._unsloth_ltx25_encoder = object()
    assert V._memory_levers_key(state) == {
        "memory_levers": "lean_connectors,released_prompt_rows,parked_vae_encoder"
    }
    monkeypatch.setenv(v25.RELEASE_PROMPT_ROWS_ENV, "0")
    assert V._memory_levers_key(state) == {"memory_levers": "lean_connectors,parked_vae_encoder"}
    monkeypatch.setenv(L.LTX2_PARK_VAE_ENCODER_ENV, "0")
    state.pipe.connectors = types.SimpleNamespace()
    # nothing engaged: no entry, so the key (and its peaks) is the one a load without the levers always had
    assert V._memory_levers_key(state) == {}


@pytest.fixture
def no_release_binding(monkeypatch):
    monkeypatch.setattr(v25, "_RELEASE_SUPPORTED", [])
    monkeypatch.delattr(torch._C, "_storage_Use_Count")
    yield
    v25._RELEASE_SUPPORTED.clear()


def test_without_the_storage_binding_nothing_is_released_and_nothing_is_discounted(
    no_release_binding,
):
    """torch 2.4 / 2.6 lack ``_storage_Use_Count``: the rows stay allocated, so the measured price must not apply."""
    assert not v25.prompt_rows_release_supported()
    rows = torch.zeros(4, 8)
    assert (
        v25._release_rows(rows, rows) == 0 and rows.untyped_storage().nbytes() == rows.numel() * 4
    )
    shape = (768, 512, 49)
    assert _price(_state(), shape) is None
    assert _guard(shape, _price(_state(), shape)) is not None
    state = _state()
    state.pipe._unsloth_ltx25_encoder = object()
    assert "released_prompt_rows" not in V._memory_levers_key(state).get("memory_levers", "")


def test_a_release_that_fell_short_stops_the_discount_for_the_load():
    state = _state()
    assert _price(state, (768, 512, 49)) is not None
    setattr(state.pipe.connectors, v25.RELEASE_SHORT_ATTR, True)
    assert _price(state, (768, 512, 49)) is None


@pytest.mark.parametrize("where", ["windows", "rocm", "rocm_sdk_wheel", "xpu"])
def test_the_allowance_outside_the_allocator_applies_only_where_it_was_measured(where, monkeypatch):
    state = _state()
    if where == "windows":
        monkeypatch.setattr(__import__("sys"), "platform", "win32")
    elif where == "rocm":
        monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    elif where == "rocm_sdk_wheel":
        # AMD SDK / Radeon wheels leave version.hip unset; the tag is in __version__.
        monkeypatch.setattr(torch, "__version__", "2.11.0+rocm7.14.0")
    else:
        state.device = "xpu"
    assert _price(state, (768, 512, 49)) is None


@pytest.mark.parametrize("why", ["one_stage", "dev"])
def test_the_peak_key_names_released_rows_only_where_they_are_released(why, monkeypatch):
    state = _state()
    state.pipe._unsloth_ltx25_encoder = object()
    assert "released_prompt_rows" in V._memory_levers_key(state)["memory_levers"]
    if why == "one_stage":
        monkeypatch.setattr(v25, "two_stage_enabled", lambda: False)
    else:
        state.pipe._unsloth_ltx25_dev = True
    assert "released_prompt_rows" not in V._memory_levers_key(state)["memory_levers"]


@pytest.mark.parametrize(
    "fps, priced", [(None, True), (24, True), (30, True), (12, False), (1, False)]
)
def test_a_frame_rate_below_the_measured_one_keeps_the_base_overhead(fps, priced):
    """LTX-2.5's audio latent length is frames / fps: a 121-frame clip at 1 fps carries 24x the audio tokens of the
    measured 24 fps render, so only the measured rate and above get the measured price."""
    shape = (768, 512, 49)
    assert (_price(_state(), shape, fps = fps) is not None) == priced

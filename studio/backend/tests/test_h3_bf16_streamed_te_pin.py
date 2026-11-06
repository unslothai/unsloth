# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Released bf16 MiniMax-H3 denoiser placement: a card that holds the denoiser beside a streamed conditioner
(but not beside a rotating bf16 one) keeps the denoiser resident and streams the conditioner leaf by leaf, instead
of paging both through the CPU-offload rotation every clip. Cards that cannot hold it keep the rotation."""

import inspect

import pytest


def _family():
    from core.inference.video_families import detect_video_family
    return detect_video_family("MiniMaxAI/MiniMax-H3")


class _Denoiser:
    def parameters(self):
        import torch
        return iter([torch.empty(8, dtype = torch.bfloat16)])

    def buffers(self):
        return iter(())


def _sizes(
    monkeypatch,
    *,
    te_scheme,
    te_streamed,
    denoiser_gb = 66.3,
):
    import torch

    from core.inference import diffusion_prequant
    from core.inference import video as vid

    monkeypatch.setattr(
        diffusion_prequant, "tensor_payload_bytes", lambda t: int(denoiser_gb * 1e9)
    )
    return vid._h3_dense_denoiser_resident_bytes(
        _family(),
        denoiser = _Denoiser(),
        te_scheme = te_scheme,
        dtype = torch.bfloat16,
        rotating = True,
        te_streamed = te_streamed,
    )


def _plan(
    monkeypatch,
    free_gb,
    te_scheme = None,
):
    from core.inference import video as vid

    rotating = _sizes(monkeypatch, te_scheme = te_scheme, te_streamed = False)
    streamed = _sizes(monkeypatch, te_scheme = te_scheme, te_streamed = True)
    return vid._h3_dense_pin_plan(rotating, streamed, int(free_gb * 1e9))


def test_streamed_conditioner_is_sized_at_its_encode_footprint(monkeypatch):
    from core.inference.video_minimax_h3_te import H3_TE_STREAMED_GB

    fam = _family()
    rotating = _sizes(monkeypatch, te_scheme = None, te_streamed = False)
    streamed = _sizes(monkeypatch, te_scheme = None, te_streamed = True)
    assert rotating[0] == streamed[0]
    vae_gb = fam.bf16_components_gb[2]
    # The rotating figure carries the 66.8 GB bf16 conditioner; streamed, only the larger of its footprint and the VAE.
    expected = int(fam.bf16_components_gb[1] * 1e9) - int(max(H3_TE_STREAMED_GB, vae_gb) * 1e9)
    assert abs((rotating[1] - streamed[1]) - expected) < 1000
    # Default stays the rotating sizing every other caller relies on.
    import torch

    from core.inference import video as vid

    default = vid._h3_dense_denoiser_resident_bytes(
        fam, denoiser = _Denoiser(), te_scheme = None, dtype = torch.bfloat16, rotating = True
    )
    assert default == rotating


@pytest.mark.parametrize(
    "free_gb, te_scheme, expected",
    [
        # RTX PRO 6000 96 GB (101.4 GB free measured): the bf16 conditioner cannot rotate beside a pinned
        # denoiser (142 GB), but a streamed one can (86 GB).
        (101.4, None, "pin_stream_te"),
        (101.4, "int8", "pin_stream_te"),
        # B200 / H200 class: fits beside the rotating conditioner as before, no stream needed.
        (178.0, None, "pin"),
        (178.0, "int8", "pin"),
        # 80 GB and smaller cards: stays in the rotation.
        (79.0, None, None),
        (47.0, None, None),
        (12.0, "int8", None),
    ],
)
def test_bf16_denoiser_placement_by_card(monkeypatch, free_gb, te_scheme, expected):
    assert _plan(monkeypatch, free_gb, te_scheme) == expected


def test_unreadable_or_unsized_keeps_the_rotation():
    from core.inference import video as vid

    assert vid._h3_dense_pin_plan(None, None, 100 * 10**9) is None
    assert vid._h3_dense_pin_plan((1, 1), (1, 1), None) is None
    # The stream kill switch passes no streamed sizes: only the rotating tier is asked.
    assert vid._h3_dense_pin_plan((60 * 10**9, 60 * 10**9), None, 100 * 10**9) is None


def test_load_streams_the_conditioner_before_pinning_and_falls_back_when_refused():
    """The streamed tier must unhook the conditioner from the rotation before the pin, and a refused stream must
    leave the denoiser in the rotation (its sizing assumed the stream)."""
    from core.inference import video as vid

    src = inspect.getsource(vid.VideoBackend)
    start = src.index("pin_plan = _h3_dense_pin_plan(")
    end = src.index("video.h3_denoiser_placement: %.1f GB free is under", start)
    block = src[start:end]
    stream_at = block.index("stream_h3_text_encoder(")
    pin_at = block.index("pin_prequantized_module(")
    assert stream_at < pin_at
    assert "pin_plan = None" in block[stream_at:pin_at]
    assert "h3_te_stream_enabled()" in src[src.rindex("streamed_sizes = (", 0, start) : start]


def test_streamed_conditioner_tier_keeps_the_eager_tier_of_the_rotation_it_replaces():
    """Placement alone is bit-identical to the rotation; the regional compile is not. The new tier holds the compile
    off, through the same downgrade the rotation takes, so a load's output does not change."""
    from core.inference import video as vid

    src = inspect.getsource(vid.VideoBackend)
    start = src.index("pin_plan = _h3_dense_pin_plan(")
    held = src.index("dense_compile_held = bool(denoiser_pinned)", start)
    assert held < src.index("video.h3_denoiser_placement: %.1f GB free is under", start)
    gate = src.index("if effective_speed in (SPEED_DEFAULT, SPEED_MAX) and dense_compile_held:")
    assert src.index("effective_speed = SPEED_EAGER", gate) < src.index(
        "snapshot_backend_flags()", gate
    )

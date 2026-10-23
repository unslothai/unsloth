# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tile verdict streams the still-resident offload groups again and re-checks before tiling the VAE.

Seen on a 15 GB T4 (Qwen-Image-2.1 fp16, edit at 1024x1024): with 5374 MiB of the text encoder resident, the request
guard streamed only the MiB the request was priced at, the verdict then said "tile", and the edit decoded in 40-latent
tiles (PSNR 52.6 dB, max abs 27/255 against the same edit with the encoder streamed, which decoded untiled). Every text
encoder and transformer output was bit-identical between the two; the first differing module was the decoder input.
CPU-only: the guard, the snapshot and the group release are stubbed.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion as diffusion
from core.inference.diffusion_memory import (
    ACTIVATION_RUN,
    ACTIVATION_TILE,
    ImageActivationVerdict,
)


class _Card:
    """Free MiB a guard re-check sees, raised by a release."""

    def __init__(self, free_mib: int, resident_mib: int) -> None:
        self.free = free_mib
        self.resident = resident_mib
        self.releases: list = []
        self.restored = 0


@pytest.fixture
def card(monkeypatch):
    c = _Card(free_mib = 6000, resident_mib = 4283)

    def release(
        pipe,
        need,
        logger = None,
        reason = "",
        device = None,
        **kw,
    ):
        if c.resident <= 0:
            return None
        c.releases.append((need, reason))
        freed = min(need, c.resident)
        c.resident -= freed
        c.free += freed

        def restore():
            c.restored += 1
            c.resident += freed
            c.free -= freed

        return restore

    def verdict(**kw):
        # untiled needs 9000 MiB, tiled 5000: what the edit guard priced on the T4
        free = kw["device_memory"]
        if free >= 9000:
            return ImageActivationVerdict(ACTIVATION_RUN)
        if free >= 5000:
            return ImageActivationVerdict(ACTIVATION_TILE)
        raise ValueError("refused")

    monkeypatch.setattr(diffusion, "resident_group_mib", lambda pipe: c.resident)
    monkeypatch.setattr(diffusion, "release_resident_groups", release)
    monkeypatch.setattr(diffusion, "reclaimable_snapshot_device_memory", lambda target: c.free)
    monkeypatch.setattr(diffusion, "raise_on_image_activation_shortfall", verdict)
    monkeypatch.delenv(diffusion.RELEASE_BEFORE_TILING_ENV, raising = False)
    return c


TARGET = types.SimpleNamespace(torch_device = "cuda:0")


def test_tile_verdict_streams_resident_groups_and_keeps_the_decode_untiled(card):
    tile = ImageActivationVerdict(ACTIVATION_TILE)
    verdict, restore = diffusion._untiled_after_release(
        object(), tile, TARGET, {"device_memory": 6000}, None
    )
    assert verdict.action == ACTIVATION_RUN  # the untiled decode now fits: no tiling
    assert card.releases == [(4283, "an untiled VAE decode")]
    assert restore is not None and card.resident == 0
    restore()
    assert card.resident == 4283 and card.restored == 1  # pinned back after the render


def test_still_tiled_after_the_release_restores_and_keeps_the_tile_verdict(card):
    card.resident = 1000  # 6000 + 1000 < 9000: the untiled decode still does not fit
    tile = ImageActivationVerdict(ACTIVATION_TILE)
    verdict, restore = diffusion._untiled_after_release(
        object(), tile, TARGET, {"device_memory": 6000}, None
    )
    assert verdict is tile and restore is None
    assert card.resident == 1000 and card.restored == 1  # nothing left streamed for a tiled render


def test_nothing_resident_or_not_a_tile_verdict_changes_nothing(card):
    run = ImageActivationVerdict(ACTIVATION_RUN)
    assert diffusion._untiled_after_release(object(), run, TARGET, {}, None) == (run, None)
    card.resident = 0
    tile = ImageActivationVerdict(ACTIVATION_TILE)
    assert diffusion._untiled_after_release(object(), tile, TARGET, {}, None) == (tile, None)
    overridden = ImageActivationVerdict(ACTIVATION_TILE, overridden = True)
    assert diffusion._untiled_after_release(object(), overridden, TARGET, {}, None) == (
        overridden,
        None,
    )
    assert card.releases == []


def test_kill_switch_tiles_as_before(card, monkeypatch):
    monkeypatch.setenv(diffusion.RELEASE_BEFORE_TILING_ENV, "0")
    tile = ImageActivationVerdict(ACTIVATION_TILE)
    assert diffusion._untiled_after_release(object(), tile, TARGET, {}, None) == (tile, None)
    assert card.releases == []


def test_a_failing_recheck_restores_and_keeps_the_tile_verdict(card, monkeypatch):
    def boom(**kw):
        raise RuntimeError("probe broke")

    monkeypatch.setattr(diffusion, "raise_on_image_activation_shortfall", boom)
    tile = ImageActivationVerdict(ACTIVATION_TILE)
    verdict, restore = diffusion._untiled_after_release(object(), tile, TARGET, {}, None)
    assert verdict is tile and restore is None and card.restored == 1


def test_chained_restore_runs_the_later_release_first_then_the_request_one():
    order: list = []
    chained = diffusion._chain_restores(
        lambda: order.append("untiled"), lambda: order.append("request")
    )
    chained()
    assert order == ["untiled", "request"]
    only = lambda: None  # noqa: E731
    assert diffusion._chain_restores(None, only) is only
    assert diffusion._chain_restores(only, None) is only
    assert diffusion._chain_restores(None, None) is None


def test_generate_calls_the_release_before_tiling():
    import inspect

    src = inspect.getsource(diffusion.DiffusionBackend.generate)
    assert src.index("_untiled_after_release(") < src.index("restore_vae = _tile_for_verdict(")
    assert "_chain_restores(restore_untiled, restore_resident)" in src


def _priced(free: int) -> ImageActivationVerdict:
    # untiled: 6952 MiB of activations + the 2048 MiB fixed overhead = the 9000 MiB the stub guard wants
    return ImageActivationVerdict(
        ACTIVATION_TILE, needed_mib = 6952, tiled_needed_mib = 2952, budget_mib = free
    )


def test_a_priced_verdict_streams_only_the_shortfall(card):
    verdict, restore = diffusion._untiled_after_release(
        object(), _priced(6000), TARGET, {"device_memory": 6000}, None
    )
    assert verdict.action == ACTIVATION_RUN
    # short 3000 MiB, plus a quarter for the budget's reserve and 256 MiB: 4006 of the 4283 resident
    assert card.releases == [(4006, "an untiled VAE decode")]
    assert card.resident == 277
    restore()
    assert card.resident == 4283


def test_a_short_first_release_streams_the_rest_once(card, monkeypatch):
    real = diffusion.release_resident_groups
    monkeypatch.setattr(
        diffusion,
        "release_resident_groups",
        lambda pipe, need, *a, **k: real(pipe, need if card.releases else need // 4, *a, **k),
    )
    verdict, restore = diffusion._untiled_after_release(
        object(), _priced(6000), TARGET, {"device_memory": 6000}, None
    )
    assert verdict.action == ACTIVATION_RUN
    assert len(card.releases) == 2 and card.resident == 0
    restore()
    assert card.resident == 4283 and card.restored == 2


def test_a_shortfall_past_every_resident_group_streams_nothing(card):
    card.resident = 2000  # short 3000 MiB: even all of it leaves the decode tiled
    tile = _priced(6000)
    assert diffusion._untiled_after_release(object(), tile, TARGET, {}, None) == (tile, None)
    assert card.releases == []


def test_a_vae_on_another_card_streams_nothing(card, monkeypatch):
    monkeypatch.setattr(diffusion, "_vae_on_other_card", lambda pipe, target: "cuda:1")
    tile = ImageActivationVerdict(ACTIVATION_TILE)
    assert diffusion._untiled_after_release(object(), tile, TARGET, {}, None) == (tile, None)
    assert card.releases == []


def test_a_broken_release_keeps_the_tile_verdict_and_pins_back(card, monkeypatch):
    calls = {"n": 0}
    real = diffusion.release_resident_groups

    def flaky(pipe, need, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("copy stream broke")
        return real(pipe, need // 4, *a, **k)

    monkeypatch.setattr(diffusion, "release_resident_groups", flaky)
    tile = _priced(6000)
    verdict, restore = diffusion._untiled_after_release(object(), tile, TARGET, {}, None)
    assert verdict is tile and restore is None
    assert card.resident == 4283 and card.restored == 1


def test_a_cancel_mid_release_pins_back_and_propagates(card, monkeypatch):
    def cancel(**kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(diffusion, "raise_on_image_activation_shortfall", cancel)
    with pytest.raises(KeyboardInterrupt):
        diffusion._untiled_after_release(
            object(), ImageActivationVerdict(ACTIVATION_TILE), TARGET, {}, None
        )
    assert card.resident == 4283 and card.restored == 1

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the fit-driven ``--load-mode`` pick.

Pins the predicate behind it: a load that fits in VRAM, or in VRAM plus host RAM,
takes ``none`` and llama.cpp's async pinned-buffer loader; anything larger, or
anything that cannot be priced, keeps ``auto`` and its mapping.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import LlamaCppBackend

GIB = 1024**3


class _Stub:
    """Just enough backend for the unbound predicate: it reads host RAM and nothing else."""

    def __init__(self, avail_mib):
        self._avail_mib = avail_mib

    def _available_system_memory_mib(self):
        return self._avail_mib


def _fits(
    footprint,
    gpus,
    *,
    avail_mib = 64 * 1024,
    **kwargs,
):
    return LlamaCppBackend._fits_without_paging(_Stub(avail_mib), footprint, gpus, **kwargs)


def test_fits_in_vram_alone():
    # 8 GiB model, 24 GiB card: VRAM settles it, host RAM is never consulted.
    assert _fits(8 * GIB, [(0, 24 * 1024)], avail_mib = None) is True


def test_fits_across_pooled_vram():
    assert _fits(20 * GIB, [(0, 11 * 1024), (1, 11 * 1024)]) is True


def test_spill_fits_in_host_ram():
    # 20 GiB against an 8 GiB card: 12 GiB spills, and 64 GiB of RAM holds it.
    assert _fits(20 * GIB, [(0, 8 * 1024)]) is True


def test_spill_exceeds_host_ram():
    # Same spill, 8 GiB of RAM, of which 2 GiB is headroom: it does not fit.
    assert _fits(20 * GIB, [(0, 8 * 1024)], avail_mib = 8 * 1024) is False


def test_headroom_is_kept_free():
    # 10 GiB spill against exactly 10 GiB of RAM fails on the 2 GiB headroom alone.
    assert _fits(10 * GIB, [], avail_mib = 10 * 1024) is False
    assert _fits(10 * GIB, [], avail_mib = 12 * 1024) is True


def test_unreadable_host_ram_abstains():
    # Nothing to price the spill against -> None, so the caller keeps llama.cpp's auto.
    assert _fits(20 * GIB, [(0, 8 * 1024)], avail_mib = None) is None


@pytest.mark.parametrize("footprint", [0, None, -1])
def test_unsized_footprint_abstains(footprint):
    assert _fits(footprint, [(0, 24 * 1024)]) is None


def test_shared_igpu_vram_is_not_added_to_host_ram():
    # The iGPU's 32 GiB IS host RAM, so it must not count on both sides: priced
    # once, 16 GiB of RAM (14 after headroom) cannot hold a 40 GiB load.
    assert (
        _fits(
            40 * GIB,
            [(0, 32 * 1024)],
            shared_gpu_ids = [0],
            avail_mib = 16 * 1024,
        )
        is False
    )


def test_unpinned_cards_hold_nothing():
    # Two 16 GiB cards, but the launch pins one: the 8 GiB spill needs host RAM.
    gpus = [(0, 16 * 1024), (1, 16 * 1024)]
    assert _fits(24 * GIB, gpus, gpu_indices = [0], avail_mib = 4 * 1024) is False
    assert _fits(24 * GIB, gpus, avail_mib = 4 * 1024) is True


def test_negative_free_vram_is_floored():
    # A probe that reports a card as over-subscribed must not credit negative VRAM.
    assert _fits(4 * GIB, [(0, -8 * 1024)], avail_mib = 4 * 1024) is False

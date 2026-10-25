# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A ``transformer_2 = None`` slot (Wan2.1, Wan2.2-TI2V-5B) must not make the stream pad None. CPU only."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

torch = pytest.importorskip("torch")


class _Group:
    def __init__(self, module, nbytes):
        self.modules = [module]
        self.parameters = []
        self.buffers = []
        self._nbytes = nbytes


@pytest.fixture
def streamed(monkeypatch):
    dit = torch.nn.Linear(64, 64)
    groups = {id(dit): [_Group(dit, 64 * 64 * 4)]}
    real = dm._offload_groups  # raises on None, as for any non-module

    monkeypatch.setattr(
        dm,
        "_offload_groups",
        lambda module: list(groups[id(module)]) if id(module) in groups else real(module),
    )
    return dit


def test_pad_skips_an_empty_denoiser_slot(streamed):
    pipe = types.SimpleNamespace(components = {"transformer": streamed, "transformer_2": None})
    assert dm.stream_window_pad_mib(pipe) == 0
    assert (
        dm.stream_window_pad_mib(types.SimpleNamespace(components = {"transformer": streamed})) == 0
    )


def test_streamed_group_window_skips_an_empty_denoiser_slot(streamed):
    pipe = types.SimpleNamespace(components = {"transformer": streamed, "transformer_2": None})
    assert dm._streamed_group_window_mib(pipe) > 0


def test_an_unhooked_off_card_denoiser_still_reads_unknown(streamed):
    # unhooked off-card weights stay unknown
    other = torch.nn.Linear(8, 8)
    pipe = types.SimpleNamespace(components = {"transformer": streamed, "transformer_2": other})
    assert dm.stream_window_pad_mib(pipe) is None

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
hooks = pytest.importorskip("diffusers.hooks")

import core.inference.diffusion_memory as dm


class _Blocks(torch.nn.Module):
    def __init__(self, n: int = 4) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(n)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _streamed(n: int = 4) -> torch.nn.Module:
    module = _Blocks(n)
    hooks.apply_group_offloading(
        module,
        onload_device = torch.device("cpu"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    return module


def _pin(module: torch.nn.Module, count: int) -> None:
    for group in dm._offload_groups(module)[:count]:
        group._unsloth_resident = True


def test_an_unhooked_transformer_under_group_offload_is_resident():
    pipe = types.SimpleNamespace(transformer = _Blocks())
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) == dm.DENOISER_RESIDENT


def test_every_group_streamed_is_streamed():
    pipe = types.SimpleNamespace(transformer = _streamed())
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) == dm.DENOISER_STREAMED


def test_some_groups_pinned_is_partial():
    module = _streamed()
    _pin(module, 2)
    pipe = types.SimpleNamespace(transformer = module)
    assert dm.denoiser_residency(pipe) == (2, 4)
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) == dm.DENOISER_PARTIAL


def test_every_group_pinned_is_resident():
    module = _streamed()
    _pin(module, 4)
    pipe = types.SimpleNamespace(transformer = module)
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) == dm.DENOISER_RESIDENT


def test_a_second_streamed_denoiser_counts():
    first = _streamed()
    _pin(first, 4)
    pipe = types.SimpleNamespace(transformer = first, transformer_2 = _streamed())
    assert dm.denoiser_residency(pipe) == (4, 8)
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) == dm.DENOISER_PARTIAL


def test_an_unhooked_resident_denoiser_beside_a_streamed_one_is_partial():
    pipe = types.SimpleNamespace(transformer = _streamed(), unconditional_transformer = _Blocks())
    assert dm.denoiser_residency(pipe) == (1, 5)
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) == dm.DENOISER_PARTIAL


@pytest.mark.parametrize(
    "policy", [dm.OFFLOAD_NONE, dm.OFFLOAD_MODEL, dm.OFFLOAD_STREAMING, dm.OFFLOAD_SEQUENTIAL, None]
)
def test_other_policies_report_nothing(policy):
    pipe = types.SimpleNamespace(transformer = _streamed())
    assert dm.denoiser_offload(pipe, policy) is None


def test_an_unreadable_denoiser_reports_nothing_instead_of_raising():
    pipe = types.SimpleNamespace(transformer = object())
    assert dm.denoiser_offload(pipe, dm.OFFLOAD_GROUP) is None


def test_the_gallery_record_keeps_the_denoiser_placement():
    from models.inference import DiffusionStatusResponse, GalleryImage
    assert "denoiser_offload" in GalleryImage.model_fields
    assert "denoiser_offload" in DiffusionStatusResponse.model_fields

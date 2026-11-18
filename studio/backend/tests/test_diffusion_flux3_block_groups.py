# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3's per-mode block stacks (ModuleDicts of ModuleLists) become their own block-level offload groups instead of
riding in the top-level group, which a streamed load onloads whole for every forward. CPU only."""

from __future__ import annotations

import pytest

import core.inference.diffusion_memory as dm


def _flux3_like():
    import torch
    class Flux3Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = torch.nn.Linear(4, 8)
            self.mode_blocks = torch.nn.ModuleDict(
                {
                    name: torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
                    for name in ("image", "image_cond")
                }
            )
            self.modulations = torch.nn.ModuleDict({"image": torch.nn.Linear(8, 8)})
            self.txt_mode_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])

        def forward(self, x):
            x = self.x_embedder(x)
            for block in self.txt_mode_blocks:
                x = block(x)
            for block in self.mode_blocks["image"]:
                x = block(x)
            return x

    return Flux3Transformer2DModel()


def test_flux3_mode_block_dicts_become_block_groups(monkeypatch):
    import torch

    monkeypatch.delenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, raising = False)
    model = _flux3_like()
    assert dm._dict_block_modules(model) == ["mode_blocks"]

    def apply(
        module,
        *,
        offload_type = "block_level",
        block_modules = None,
        **kw,
    ):
        return None

    def old_apply(
        module,
        *,
        offload_type = "block_level",
        **kw,
    ):
        return None

    assert dm._with_dict_block_modules(apply, model, {"offload_type": "block_level"})[
        "block_modules"
    ] == ["mode_blocks"]
    # leaf-level, an older diffusers without the parameter, and other classes keep the stock grouping
    assert "block_modules" not in dm._with_dict_block_modules(
        apply, model, {"offload_type": "leaf_level"}
    )
    assert "block_modules" not in dm._with_dict_block_modules(
        old_apply, model, {"offload_type": "block_level"}
    )
    assert (
        dm._dict_block_modules(
            torch.nn.ModuleDict({"a": torch.nn.ModuleList([torch.nn.Linear(2, 2)])})
        )
        == []
    )
    monkeypatch.setenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, "0")
    assert dm._dict_block_modules(model) == []


def test_flux3_mode_blocks_get_their_own_diffusers_groups(monkeypatch):
    import inspect

    import torch

    apply_group_offloading = pytest.importorskip("diffusers.hooks").apply_group_offloading

    # an older diffusers without block_modules keeps the stock grouping
    supported = "block_modules" in inspect.signature(apply_group_offloading).parameters
    monkeypatch.delenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, raising = False)
    counts = {}
    for env in ("1", "0"):
        monkeypatch.setenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, env)
        model = _flux3_like()
        x = torch.randn(2, 4)
        want = model(x)
        kwargs = dict(
            onload_device = torch.device("cpu"),
            offload_device = torch.device("cpu"),
            offload_type = "block_level",
            num_blocks_per_group = 1,
        )
        dm._apply_group_offloading_packed(apply_group_offloading, model, kwargs)
        groups = dm._offload_groups(model)
        counts[env] = len(groups)
        mode_groups = [
            g for g in groups if any(m is b for b in model.mode_blocks["image"] for m in g.modules)
        ]
        assert (len(mode_groups) == 3) is (env == "1" and supported)
        torch.testing.assert_close(model(x), want)
    # 6 mode blocks become groups of their own; the stock grouping folds them into the top-level group
    assert counts["1"] == counts["0"] + (6 if supported else 0)


def _mib_model(monkeypatch):
    """A FLUX.3-like model whose parameter sizes are read from fake storage sizes (MiB per tensor)."""
    model = _flux3_like()
    sizes = {}
    for name, p in model.named_parameters():
        sizes[id(p)] = 64 if ".mode_blocks." in "." + name else 4
    monkeypatch.setattr(dm, "_storage_nbytes", lambda t: [sizes.get(id(t), 0) * 1024 * 1024])
    return model


@pytest.mark.parametrize("supported", [True, False])
def test_top_level_group_leaves_the_mode_blocks_out(monkeypatch, supported):
    monkeypatch.delenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, raising = False)
    monkeypatch.setattr(dm, "_diffusers_takes_block_modules", lambda: supported)
    model = _mib_model(monkeypatch)
    pipe = type("P", (), {"components": {"transformer": model}})()
    top = dm._top_level_group_mib(pipe)
    # x_embedder + modulations (16 MiB) stay top level; the 6 mode blocks move out only if diffusers can split them
    assert top == (16 if supported else 16 + 6 * 128)
    monkeypatch.setenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, "0")
    assert dm._top_level_group_mib(pipe) == 16 + 6 * 128


def test_stream_window_counts_the_mode_blocks(monkeypatch):
    monkeypatch.delenv(dm.GROUP_OFFLOAD_DICT_BLOCKS_ENV, raising = False)
    model = _mib_model(monkeypatch)
    pipe = type("P", (), {"components": {"transformer": model}})()
    monkeypatch.setattr(dm, "_diffusers_takes_block_modules", lambda: True)
    with_dict = dm._stream_window_mib(pipe)
    monkeypatch.setattr(dm, "_diffusers_takes_block_modules", lambda: False)
    without = dm._stream_window_mib(pipe)
    # the largest block is now a mode block (128 MiB) instead of a txt block (8 MiB)
    assert with_dict > without > 0

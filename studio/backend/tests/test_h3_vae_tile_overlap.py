# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video passes --vae-tile-overlap 0.25 whenever Studio tiles its VAE (the reference's 64 px overlap)."""

from __future__ import annotations

import ast
import inspect

import pytest

from core.inference import video
from core.inference.sd_cpp_args import (
    H3_VAE_TILE_OVERLAP,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    offload_flags,
    with_h3_vae_tile_overlap,
)
from core.inference.video_minimax_h3 import h3_native_memory_fit, h3_native_render_flags


def _h3_policy_flags(policy: str) -> tuple[str, ...]:
    # The H3 native loader's call.
    return with_h3_vae_tile_overlap(
        offload_flags(policy, vae_tiling = False, diffusion_fa = True, vae_on_cpu = False)
    )


def test_overlap_follows_vae_tiling():
    assert H3_VAE_TILE_OVERLAP == "0.25"
    flags = with_h3_vae_tile_overlap(["--offload-to-cpu", "--vae-tiling"])
    assert flags == ("--offload-to-cpu", "--vae-tiling", "--vae-tile-overlap", "0.25")


def test_no_tiling_no_overlap():
    assert with_h3_vae_tile_overlap(["--offload-to-cpu", "--diffusion-fa"]) == (
        "--offload-to-cpu",
        "--diffusion-fa",
    )
    assert with_h3_vae_tile_overlap(()) == ()


def test_an_explicit_overlap_is_kept():
    flags = ("--vae-tiling", "--vae-tile-overlap", "0.5")
    assert with_h3_vae_tile_overlap(flags) == flags
    # Idempotent.
    once = with_h3_vae_tile_overlap(["--vae-tiling"])
    assert with_h3_vae_tile_overlap(once) == once


@pytest.mark.parametrize(
    "policy, tiled",
    [(OFFLOAD_NONE, False), (OFFLOAD_GROUP, False), (OFFLOAD_MODEL, True)],
)
def test_memory_mode_policies(policy, tiled):
    flags = _h3_policy_flags(policy)
    assert ("--vae-tiling" in flags) is tiled
    if tiled:
        i = flags.index("--vae-tiling")
        assert flags[i : i + 3] == ("--vae-tiling", "--vae-tile-overlap", "0.25")
    else:
        assert "--vae-tile-overlap" not in flags


def test_low_vram_overlap_survives_the_render_and_fit_steps():
    flags = _h3_policy_flags(OFFLOAD_MODEL)
    render, resident = h3_native_render_flags(
        flags, memory_mode = "low_vram", free_bytes = 1 << 40, need_bytes = 1
    )
    assert not resident and tuple(render) == flags
    gib = 1 << 30
    fit = h3_native_memory_fit(
        flags,
        accelerator = "cuda",
        memory_mode = "low_vram",
        dit_bytes = 10 * gib,
        te_bytes = 10 * gib,
        vae_bytes = gib,
        pinned_free_bytes = None,
        host_ram_bytes = 200 * gib,
    )
    assert fit.refusal is None
    i = fit.flags.index("--vae-tiling")
    assert fit.flags[i : i + 3] == ("--vae-tiling", "--vae-tile-overlap", "0.25")


def test_the_h3_native_loader_wraps_its_offload_flags():
    """The loader's native_offload is built through with_h3_vae_tile_overlap (no other H3 path adds --vae-tiling)."""
    tree = ast.parse(inspect.getsource(video))
    wrapped = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "native_offload" for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            func = node.value.func
            wrapped.append(getattr(func, "id", getattr(func, "attr", None)))
    assert wrapped == ["with_h3_vae_tile_overlap"]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The offload path keeps the VAE on the CUDA card when its free VRAM holds the tiled decode, else the CPU VAE."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import sd_cpp_args as args_mod
from core.inference import sd_cpp_backend as backend_mod
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
)
from core.inference.sd_cpp_args import (
    GPU_VAE_DECODE_MIB_PER_MPX,
    GPU_VAE_ENV,
    device_backend_flags,
    gpu_vae_tile_size,
    offload_flags,
)
from core.inference.sd_cpp_multigpu import CARD_RESERVE_MIB

VAE_MIB = 320


def _need(size: int, vae: int = VAE_MIB) -> int:
    return int(GPU_VAE_DECODE_MIB_PER_MPX * (size / 1024) ** 2) + 2 * vae


@pytest.mark.parametrize(
    "free, want",
    [
        (None, None),
        (-1000, None),
        (0, None),
        (_need(256) - 1, None),
        (_need(256), 256),
        (_need(512) - 1, 256),
        (_need(512), 512),
        (_need(768), 768),
        (_need(1024) - 1, 768),
        (_need(1024), 1024),
        (10**6, 1024),
    ],
)
def test_tile_follows_free_vram(free, want):
    assert gpu_vae_tile_size(free, VAE_MIB, env = {}) == want


def test_vae_weights_count_against_the_room():
    free = _need(1024, vae = 0)
    assert gpu_vae_tile_size(free, 0, env = {}) == 1024
    assert gpu_vae_tile_size(free, 1, env = {}) == 768
    # An unreadable VAE size (embedded VAE, 0) does not refuse the card.
    assert gpu_vae_tile_size(free, None, env = {}) == 1024


@pytest.mark.parametrize("value", ["0", "off", "false", "no", " Off "])
def test_kill_switch_keeps_the_cpu_vae(value):
    assert gpu_vae_tile_size(10**6, VAE_MIB, env = {GPU_VAE_ENV: value}) is None


@pytest.mark.parametrize("policy", [OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL])
def test_gpu_vae_flags(policy):
    flags = offload_flags(policy, vae_on_cpu = False, vae_tile_size = 512)
    assert "--vae-on-cpu" not in flags
    assert "--offload-to-cpu" in flags and "--clip-on-cpu" in flags
    assert flags.count("--vae-tiling") == 1
    assert flags[-2:] == ["--vae-tile-size", "512"]
    # The device pin then keeps the VAE on the card and the text encoder on the CPU.
    assert device_backend_flags("CUDA0", flags) == ["--backend", "diffusion=CUDA0,te=cpu,vae=CUDA0"]


def test_video_opt_out_is_unchanged():
    """video.py drops --vae-on-cpu without a tile size: still the bare --vae-tiling."""
    flags = offload_flags(OFFLOAD_MODEL, vae_tiling = False, diffusion_fa = True, vae_on_cpu = False)
    assert (
        "--vae-on-cpu" not in flags and "--vae-tile-size" not in flags and "--vae-tiling" in flags
    )


def _files(tmp_path, vae_mib: int = VAE_MIB) -> SimpleNamespace:
    path = tmp_path / "vae.safetensors"
    with open(path, "wb") as f:
        f.truncate(vae_mib << 20)
    return SimpleNamespace(diffusion_model = None, vae = str(path))


@pytest.fixture
def card(monkeypatch):
    """Pixel-unit build, no VRAM Budget, RAM tile stub; returns a setter for the card's (free, total) reading."""
    from core.inference import diffusion_memory

    monkeypatch.delenv(GPU_VAE_ENV, raising = False)
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.setattr(diffusion_memory, "_user_budget_reserve_mib", lambda base: 0)
    monkeypatch.setattr(backend_mod, "_cpu_vae_tile_size", lambda files, binary: 768)
    seen: list = []

    def set_reading(free, total = 16384):
        def read(klass, description, ordinal):
            seen.append((klass, ordinal))
            return free, total

        monkeypatch.setattr(backend_mod, "_unmeasured_card_free_mib", read)
        return seen

    return set_reading


def _flags(
    tmp_path,
    policy = OFFLOAD_MODEL,
    klass = "cuda",
    ordinal = None,
):
    return backend_mod._offload_vae_flags(
        policy, _files(tmp_path), "/bin/sd-server", klass, ordinal
    )


@pytest.mark.parametrize("policy", [OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL])
def test_card_with_room_keeps_the_vae_on_the_card(tmp_path, card, policy):
    seen = card(CARD_RESERVE_MIB + _need(1024))
    flags = _flags(tmp_path, policy, ordinal = 1)
    assert "--vae-on-cpu" not in flags
    assert flags[-2:] == ["--vae-tile-size", "1024"]
    assert seen == [("cuda", 1)]


def test_card_reserve_shrinks_the_tile(tmp_path, card):
    card(CARD_RESERVE_MIB + _need(1024) - 1)
    assert _flags(tmp_path)[-2:] == ["--vae-tile-size", "768"]
    card(CARD_RESERVE_MIB + _need(256))
    assert _flags(tmp_path)[-2:] == ["--vae-tile-size", "256"]


def test_full_card_keeps_the_cpu_vae_with_its_ram_tile(tmp_path, card):
    card(CARD_RESERVE_MIB + _need(256) - 1)
    flags = _flags(tmp_path)
    assert "--vae-on-cpu" in flags
    assert flags[-2:] == ["--vae-tile-size", "768"]
    assert flags == offload_flags(OFFLOAD_MODEL, vae_tile_size = 768)


def test_unread_card_keeps_the_cpu_vae(tmp_path, card):
    card(None, None)
    assert "--vae-on-cpu" in _flags(tmp_path)


def test_vram_budget_reserve_counts(tmp_path, card, monkeypatch):
    from core.inference import diffusion_memory

    card(CARD_RESERVE_MIB + _need(1024))
    monkeypatch.setattr(
        diffusion_memory, "_user_budget_reserve_mib", lambda base: CARD_RESERVE_MIB + 4096
    )
    assert _flags(tmp_path)[-2:] == ["--vae-tile-size", "512"]


@pytest.mark.parametrize("klass", ["vulkan", "metal", "rocm", None])
def test_other_accelerators_keep_the_cpu_vae(tmp_path, card, klass):
    seen = card(10**6)
    assert "--vae-on-cpu" in _flags(tmp_path, klass = klass)
    assert seen == []


def test_latent_unit_build_keeps_the_cpu_vae(tmp_path, card, monkeypatch):
    card(10**6)
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: False)
    assert "--vae-on-cpu" in _flags(tmp_path)


def test_kill_switch_through_the_backend(tmp_path, card, monkeypatch):
    card(10**6)
    monkeypatch.setenv(GPU_VAE_ENV, "0")
    assert "--vae-on-cpu" in _flags(tmp_path)


def test_card_read_failure_keeps_the_cpu_vae(tmp_path, card, monkeypatch):
    card(10**6)

    def boom(*a):
        raise OSError("nvidia-smi gone")

    monkeypatch.setattr(backend_mod, "_unmeasured_card_free_mib", boom)
    assert "--vae-on-cpu" in _flags(tmp_path)


@pytest.mark.parametrize("policy", [OFFLOAD_NONE, OFFLOAD_GROUP])
def test_policies_without_a_cpu_vae_are_untouched(tmp_path, card, policy):
    seen = card(10**6)
    assert _flags(tmp_path, policy) == offload_flags(policy)
    assert seen == []

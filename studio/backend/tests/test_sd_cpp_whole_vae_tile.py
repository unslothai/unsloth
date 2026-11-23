# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The offload path's VAE tile covers the family's largest output (one tile = the untiled decode) when the budget holds
that measured whole decode; otherwise, and for unmeasured families, the per-tile ladder is unchanged."""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from core.inference import sd_cpp_args as args_mod
from core.inference import sd_cpp_backend as backend_mod
from core.inference.diffusion_memory import OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL
from core.inference.sd_cpp_args import (
    CPU_VAE_TILE_ENV,
    CPU_WHOLE_VAE_MARGIN,
    GPU_TE_ENV,
    GPU_VAE_ENV,
    GPU_WHOLE_VAE_MARGIN,
    WHOLE_VAE_DECODE_MIB,
    cpu_vae_tile_size,
    gpu_vae_tile_size,
    offload_flags,
    whole_vae_tiles,
)
from core.inference.sd_cpp_multigpu import CARD_RESERVE_MIB

VAE_MIB = 320
MEASURED = (
    "flux.2-klein",
    "flux.2-dev",
    "z-image",
    "flux.1",
    "flux.1-kontext",
    "chroma",
    "qwen-image-2.1",
)
# Unmeasured VAEs (Wan-style Qwen-Image, SDXL, FLUX.3, ...) and MiniMax-H3's own tiler.
UNMEASURED = (
    "minimax-h3-image",
    "qwen-image",
    "qwen-image-edit",
    "sdxl",
    "flux.3-klein",
    "krea-2",
    None,
    "",
)
L4_FREE_MIB = 22563  # nvidia-smi memory.free of an idle Colab L4


def _price(family: str, side: int, margin: float) -> int:
    return int(dict(WHOLE_VAE_DECODE_MIB[family])[side] * margin)


def _gpu_need(
    family: str,
    side: int,
    vae: int = VAE_MIB,
) -> int:
    return _price(family, side, GPU_WHOLE_VAE_MARGIN) + 2 * vae


def _cpu_need(family: str, side: int) -> int:
    return _price(family, side, CPU_WHOLE_VAE_MARGIN)


@pytest.mark.parametrize("family", MEASURED)
def test_measured_families_try_the_whole_image_tiles(family):
    assert [side for side, _ in whole_vae_tiles(family)] == [2048, 1536]
    assert whole_vae_tiles(family.upper()) == whole_vae_tiles(family)


@pytest.mark.parametrize("family", UNMEASURED)
def test_other_families_keep_the_ladder(family):
    assert whole_vae_tiles(family) == ()


def test_prices_are_the_measured_buffers():
    """ggml VAE compute buffer of one whole decode on the pinned build (L4): FLUX.2 / FLUX.1 / Qwen-Image 2.1."""
    assert dict(whole_vae_tiles("flux.2-klein")) == {2048: 17040, 1536: 6336}
    assert dict(whole_vae_tiles("z-image")) == {2048: 17028, 1536: 6336}
    assert dict(whole_vae_tiles("qwen-image-2.1")) == {2048: 15552, 1536: 8748}
    # Every whole tile costs more than the ladder's own 1024 px tile, so a budget that only held 1024 still gets it.
    for family in MEASURED:
        for side, _ in whole_vae_tiles(family):
            assert _price(family, side, GPU_WHOLE_VAE_MARGIN) > args_mod.GPU_VAE_DECODE_MIB_PER_MPX
            assert _price(family, side, CPU_WHOLE_VAE_MARGIN) > args_mod.CPU_VAE_DECODE_MIB_PER_MPX


@pytest.mark.parametrize(
    "family, vae, free, want",
    [
        # Idle L4 (22563 free less the card reserve): klein / Z-Image / Qwen-Image 2.1 decode a 2048 px image whole.
        ("flux.2-klein", 320, L4_FREE_MIB - CARD_RESERVE_MIB, 2048),
        ("z-image", 319, L4_FREE_MIB - CARD_RESERVE_MIB, 2048),
        ("qwen-image-2.1", 644, L4_FREE_MIB - CARD_RESERVE_MIB, 2048),
        ("flux.2-klein", VAE_MIB, _gpu_need("flux.2-klein", 2048), 2048),
        ("flux.2-klein", VAE_MIB, _gpu_need("flux.2-klein", 2048) - 1, 1536),
        ("flux.2-klein", VAE_MIB, _gpu_need("flux.2-klein", 1536), 1536),
        ("flux.2-klein", VAE_MIB, _gpu_need("flux.2-klein", 1536) - 1, 1024),
        ("qwen-image-2.1", VAE_MIB, _gpu_need("qwen-image-2.1", 1536) - 1, 1024),
    ],
)
def test_gpu_whole_tile_follows_free_vram(family, vae, free, want):
    assert gpu_vae_tile_size(free, vae, env = {}, whole_tiles = whole_vae_tiles(family)) == want


@pytest.mark.parametrize(
    "family, spare, want",
    [
        ("z-image", 10**6, 2048),
        ("z-image", _cpu_need("z-image", 2048), 2048),
        ("z-image", _cpu_need("z-image", 2048) - 1, 1536),
        ("z-image", _cpu_need("z-image", 1536) - 1, 1024),
        ("qwen-image-2.1", _cpu_need("qwen-image-2.1", 1536), 1536),
    ],
)
def test_cpu_whole_tile_follows_spare_ram(family, spare, want):
    assert cpu_vae_tile_size(spare, env = {}, whole_tiles = whole_vae_tiles(family)) == want


@pytest.mark.parametrize("family", MEASURED)
def test_tight_budgets_are_unchanged(family):
    """Below the 1536 px whole decode both choosers give exactly what they gave without whole tiles."""
    tiles = whole_vae_tiles(family)
    for free in [None, -1, 0, 100, 1000, 2000, 4000, 6000, _gpu_need(family, 1536) - 1]:
        assert gpu_vae_tile_size(free, VAE_MIB, env = {}, whole_tiles = tiles) == gpu_vae_tile_size(
            free, VAE_MIB, env = {}
        )
    for spare in [None, -1, 0, 1000, 3000, 5000, 8000, _cpu_need(family, 1536) - 1]:
        assert cpu_vae_tile_size(spare, env = {}, whole_tiles = tiles) == cpu_vae_tile_size(
            spare, env = {}
        )


def test_env_switches_still_win():
    tiles = whole_vae_tiles("flux.2-klein")
    assert cpu_vae_tile_size(10**6, env = {CPU_VAE_TILE_ENV: "off"}, whole_tiles = tiles) is None
    assert cpu_vae_tile_size(10**6, env = {CPU_VAE_TILE_ENV: "640"}, whole_tiles = tiles) == 640
    assert gpu_vae_tile_size(10**6, VAE_MIB, env = {GPU_VAE_ENV: "0"}, whole_tiles = tiles) is None


def _files(tmp_path, vae_mib: int = VAE_MIB) -> SimpleNamespace:
    path = tmp_path / "vae.safetensors"
    with open(path, "wb") as f:
        f.truncate(vae_mib << 20)
    return SimpleNamespace(diffusion_model = None, vae = str(path))


@pytest.fixture
def card(monkeypatch):
    from core.inference import diffusion_memory

    monkeypatch.delenv(GPU_VAE_ENV, raising = False)
    monkeypatch.delenv(GPU_TE_ENV, raising = False)
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.setattr(diffusion_memory, "_user_budget_reserve_mib", lambda base: 0)

    def set_reading(free, total = 23034):
        monkeypatch.setattr(backend_mod, "_unmeasured_card_free_mib", lambda *a: (free, total))

    return set_reading


def _tile(flags):
    return flags[flags.index("--vae-tile-size") + 1] if "--vae-tile-size" in flags else None


@pytest.mark.parametrize("policy", [OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL])
def test_idle_l4_decodes_klein_whole_on_the_card(tmp_path, card, policy):
    card(L4_FREE_MIB)
    flags = backend_mod._offload_vae_flags(
        policy, _files(tmp_path), "/bin/sd-server", "cuda", 0, family = "flux.2-klein"
    )
    assert "--vae-on-cpu" not in flags and _tile(flags) == "2048"
    # Same card, unmeasured family or no family: the 1024 px tile it had.
    for family in ("minimax-h3-image", "qwen-image", None):
        flags = backend_mod._offload_vae_flags(
            policy, _files(tmp_path), "/bin/sd-server", "cuda", 0, family = family
        )
        assert _tile(flags) == "1024"


def test_cpu_vae_and_cpu_restart_take_the_family_tile(tmp_path, card, monkeypatch):
    """A card too small for any GPU tile keeps the CPU VAE, sized for the family; a restart onto the CPU backend
    re-sizes the RAM tile for the family too."""
    calls = []

    def ram_tile(
        files,
        binary,
        family = None,
    ):
        calls.append(family)
        return 2048 if family else 1024

    monkeypatch.setattr(backend_mod, "_cpu_vae_tile_size", ram_tile)
    card(CARD_RESERVE_MIB + 100)
    flags = backend_mod._offload_vae_flags(
        OFFLOAD_MODEL, _files(tmp_path), "/bin/sd-server", "cuda", 0, family = "z-image"
    )
    assert "--vae-on-cpu" in flags and _tile(flags) == "2048"
    gpu = offload_flags(OFFLOAD_MODEL, vae_on_cpu = False, vae_tile_size = 2048)
    restarted = backend_mod._cpu_backend_offload(gpu, _files(tmp_path), "/bin/sd-server", "z-image")
    assert "--vae-on-cpu" in restarted and _tile(restarted) == "2048"
    assert calls == ["z-image", "z-image"]


class _HoldingServer:
    running: list = []

    def __init__(self, binary):
        self.binary = binary

    def is_alive(self):
        return self in _HoldingServer.running

    def start(self, files, **kwargs):
        _HoldingServer.running.append(self)

    def stop(self):
        if self in _HoldingServer.running:
            _HoldingServer.running.remove(self)


def _sized(tmp_path, name, mib):
    p = tmp_path / name
    with open(p, "wb") as f:
        f.truncate(mib << 20)
    return str(p)


@pytest.mark.parametrize("family_name, want", [("z-image", "2048"), ("sdxl", "1024")])
def test_a_load_passes_its_family_to_the_tile_choice(tmp_path, monkeypatch, family_name, want):
    """Through _run_load: a roomy host gives Z-Image the whole-image CPU tile, SDXL (unmeasured) the 1024 px one."""
    from core.inference import diffusion_small_host
    from core.inference.diffusion_families import detect_family

    paths = {
        "diffusion_model": _sized(tmp_path, "d.gguf", 2000),
        "vae": _sized(tmp_path, "vae.sft", 100),
        "llm": _sized(tmp_path, "llm.gguf", 3000),
    }
    total = 65536
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (total, 60000))
    monkeypatch.setattr(diffusion_small_host, "host_commit_mib", lambda: None)
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.delenv(CPU_VAE_TILE_ENV, raising = False)
    monkeypatch.setattr(backend_mod, "_installed_accelerator_of", lambda binary: "vulkan")
    monkeypatch.setattr(backend_mod, "_unmeasured_card_free_mib", lambda *a: (None, None))
    monkeypatch.setattr(backend_mod, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(backend_mod, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(backend_mod, "SdCppServer", _HoldingServer)
    monkeypatch.setattr(
        backend_mod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    _HoldingServer.running = []
    fam = detect_family(family_name)
    assert fam is not None and fam.name == family_name
    b = backend_mod.SdCppDiffusionBackend()
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(b, "_fetch_assets", lambda *a, **k: dict(paths))
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/x-GGUF",
        gguf_filename = "d.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )
    flags = list(b._state.offload_flags)
    assert "--vae-on-cpu" in flags
    assert _tile(flags) == want

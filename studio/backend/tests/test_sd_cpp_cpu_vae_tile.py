# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The offload path's CPU VAE tile size: RAM-sized, kill switch, pixel-unit builds only, argv shape."""

from __future__ import annotations

import os
import stat
import sys
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
    CPU_VAE_DECODE_MIB_PER_MPX,
    CPU_VAE_TILE_ENV,
    SdCppGenParams,
    SdCppModelFiles,
    build_sd_cpp_command,
    build_sd_cpp_server_command,
    cpu_vae_tile_size,
    offload_flags,
    sd_cpp_vae_tile_size_in_pixels,
)

PIXEL_HELP = (
    "  --vae-tile-size                          tile size for vae encode and decode in image pixels, format [W]x[H] or [S]\n"
    "                                           (default: 256x256; requires --vae-tiling)\n"
)
PIXEL_HELP_WRAPPED = "  --vae-tile-size   tile size for vae encode and decode in image\n      pixels, format [W]x[H]\n"
LATENT_HELP = "  --vae-tile-size                    tile size for vae tiling, format [X]x[Y] (default: 32x32)\n"


def _need(size: int) -> int:
    return int(CPU_VAE_DECODE_MIB_PER_MPX * (size / 1024) ** 2)


@pytest.mark.parametrize(
    "spare, want",
    [
        (None, None),
        (-5000, None),
        (0, None),
        (_need(512) - 1, None),
        (_need(512), 512),
        (_need(768) - 1, 512),
        (_need(768), 768),
        (_need(1024) - 1, 768),
        (_need(1024), 1024),
        (10**7, 1024),
    ],
)
def test_tile_size_follows_spare_ram(spare, want):
    assert cpu_vae_tile_size(spare, env = {}) == want


@pytest.mark.parametrize("value", ["0", "off", "false", "no", " OFF "])
def test_kill_switch_keeps_256(value):
    assert cpu_vae_tile_size(10**7, env = {CPU_VAE_TILE_ENV: value}) is None


def test_env_pins_a_size_even_without_a_ram_reading():
    assert cpu_vae_tile_size(None, env = {CPU_VAE_TILE_ENV: "640"}) == 640
    assert cpu_vae_tile_size(10**7, env = {CPU_VAE_TILE_ENV: "12"}) == 1024
    assert cpu_vae_tile_size(10**7, env = {CPU_VAE_TILE_ENV: "big"}) == 1024


def test_offload_flags_carry_the_size_only_for_the_cpu_vae():
    for policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL):
        flags = offload_flags(policy, vae_tile_size = 1024)
        assert flags[-2:] == ["--vae-tile-size", "1024"]
        assert flags.count("--vae-tiling") == 1 and "--vae-on-cpu" in flags
        assert offload_flags(policy, vae_tile_size = None) == flags[:-2]
        assert "--vae-tile-size" not in offload_flags(policy, vae_on_cpu = False, vae_tile_size = 1024)
    for policy in (OFFLOAD_NONE, OFFLOAD_GROUP):
        assert "--vae-tile-size" not in offload_flags(policy, vae_tile_size = 1024)


def _files() -> SdCppModelFiles:
    return SdCppModelFiles(
        diffusion_model = "/m/dit.gguf", vae = "/m/vae.safetensors", llm = "/m/te.gguf"
    )


def test_cli_and_server_argv_keep_the_pair_and_user_args_win(tmp_path):
    offload = offload_flags(OFFLOAD_MODEL, vae_tile_size = 768)
    cli = build_sd_cpp_command(
        "/bin/sd-cli",
        _files(),
        SdCppGenParams(prompt = "a fox", width = 1024, height = 1024),
        output_path = str(tmp_path / "o.png"),
        offload = offload,
        extra_args = ["--vae-tile-size", "256"],
    )
    i = cli.index("--vae-tile-size")
    assert cli[i + 1] == "768"
    # sd.cpp is last-wins and user args go last.
    assert cli[len(cli) - 1 - cli[::-1].index("--vae-tile-size") + 1] == "256"
    server = build_sd_cpp_server_command(
        "/bin/sd-server",
        _files(),
        host = "127.0.0.1",
        port = 1,
        offload = offload,
        native_speed = "default",
        accelerator = "cuda",
    )
    j = server.index("--vae-tile-size")
    assert server[j + 1] == "768" and "--vae-tiling" in server


def _fake_binary(tmp_path, help_text: str) -> str:
    path = tmp_path / "sd-cli"
    path.write_text("#!/bin/sh\ncat <<'EOF'\n" + help_text + "EOF\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.mark.skipif(sys.platform == "win32", reason = "shell script stands in for sd-cli")
@pytest.mark.parametrize(
    "text, want",
    [(PIXEL_HELP, True), (PIXEL_HELP_WRAPPED, True), (LATENT_HELP, False), ("", False)],
)
def test_pixel_unit_detection_from_help(tmp_path, text, want):
    assert sd_cpp_vae_tile_size_in_pixels(_fake_binary(tmp_path, text)) is want


def test_pixel_unit_detection_fails_closed(tmp_path):
    assert sd_cpp_vae_tile_size_in_pixels(None) is False
    assert sd_cpp_vae_tile_size_in_pixels(str(tmp_path / "missing")) is False


def _sized(tmp_path, name: str, mib: int) -> str:
    path = tmp_path / name
    with open(path, "wb") as f:
        f.truncate(mib << 20)
    return str(path)


def test_backend_subtracts_offloaded_weights_and_reserve(tmp_path, monkeypatch):
    from core.inference import diffusion_small_host

    files = SimpleNamespace(
        diffusion_model = _sized(tmp_path, "dit.gguf", 2000),
        vae = _sized(tmp_path, "vae.st", 100),
        llm = _sized(tmp_path, "te.gguf", 3000),
    )
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.delenv(CPU_VAE_TILE_ENV, raising = False)
    monkeypatch.setattr(diffusion_small_host, "host_commit_mib", lambda: None)
    reserve = diffusion_small_host.host_reserve_mib(32768)
    weights = 5100

    def with_available(avail):
        monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (32768, avail))
        return backend_mod._cpu_vae_tile_size(files, "/bin/sd-cli")

    assert with_available(weights + reserve + _need(1024)) == 1024
    assert with_available(weights + reserve + _need(1024) - 1) == 768
    assert with_available(weights + reserve + _need(512)) == 512
    assert with_available(weights + reserve) is None
    assert with_available(None) is None
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: False)
    assert with_available(10**7) is None


def test_windows_commit_headroom_caps_the_tile(tmp_path, monkeypatch):
    """Windows: commit headroom below physical available RAM sizes the tile."""
    from core.inference import diffusion_small_host

    files = SimpleNamespace(
        diffusion_model = _sized(tmp_path, "dit.gguf", 2000),
        vae = _sized(tmp_path, "vae.st", 100),
        llm = _sized(tmp_path, "te.gguf", 3000),
    )
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.delenv(CPU_VAE_TILE_ENV, raising = False)
    reserve = diffusion_small_host.host_reserve_mib(32768)
    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", lambda: (32768, 30000))

    def with_commit(commit):
        monkeypatch.setattr(diffusion_small_host, "host_commit_mib", lambda: commit)
        return backend_mod._cpu_vae_tile_size(files, "/bin/sd-cli")

    assert with_commit(None) == 1024
    assert with_commit(10**6) == 1024
    assert with_commit(5100 + reserve + _need(512)) == 512
    assert with_commit(0) is None


def test_backend_unreadable_host_keeps_256(tmp_path, monkeypatch):
    from core.inference import diffusion_small_host

    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.delenv(CPU_VAE_TILE_ENV, raising = False)

    def boom():
        raise OSError("no /proc")

    monkeypatch.setattr(diffusion_small_host, "host_ram_mib", boom)
    assert backend_mod._cpu_vae_tile_size(SimpleNamespace(), "/bin/sd-cli") is None


def test_real_pinned_help_is_pixel_units():
    binary = os.environ.get("SD_CLI_PATH")
    if not binary or not os.path.exists(binary):
        pytest.skip(reason = "no sd-cli binary on this machine (SD_CLI_PATH unset or missing)")
    assert sd_cpp_vae_tile_size_in_pixels(binary) is True


class _HoldingServer:
    running: list = []

    def __init__(self, binary):
        self.binary = binary
        self.started = None

    def is_alive(self):
        return self in _HoldingServer.running

    def start(self, files, **kwargs):
        self.started = kwargs
        _HoldingServer.running.append(self)

    def stop(self):
        if self in _HoldingServer.running:
            _HoldingServer.running.remove(self)


def test_a_model_switch_sizes_the_tile_after_the_outgoing_server_frees_its_weights(
    tmp_path, monkeypatch
):
    """Placement reruns after the outgoing server frees its RAM, so a reload gets the same tile."""
    import types

    from core.inference import diffusion_small_host
    from core.inference.diffusion_families import detect_family

    paths = {
        "diffusion_model": _sized(tmp_path, "z.gguf", 2000),
        "vae": _sized(tmp_path, "vae.sft", 100),
        "llm": _sized(tmp_path, "llm.gguf", 3000),
    }
    weights = 5100
    reserve = diffusion_small_host.host_reserve_mib(32768)
    fresh = weights + reserve + _need(1024)
    monkeypatch.setattr(
        diffusion_small_host,
        "host_ram_mib",
        lambda: (32768, fresh - weights * len(_HoldingServer.running)),
    )
    monkeypatch.setattr(diffusion_small_host, "host_commit_mib", lambda: None)
    monkeypatch.setattr(args_mod, "sd_cpp_vae_tile_size_in_pixels", lambda b: True)
    monkeypatch.delenv(CPU_VAE_TILE_ENV, raising = False)
    # Vulkan build with CPU torch: unmeasured card, auto offloads.
    monkeypatch.setattr(backend_mod, "_installed_accelerator_of", lambda binary: "vulkan")
    monkeypatch.setattr(backend_mod, "find_sd_server_binary", lambda: "/x/sd-server")
    monkeypatch.setattr(backend_mod, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(backend_mod, "SdCppServer", _HoldingServer)
    monkeypatch.setattr(
        backend_mod, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    _HoldingServer.running = []
    fam = detect_family("z-image")
    b = backend_mod.SdCppDiffusionBackend()
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(b, "_fetch_assets", lambda *a, **k: dict(paths))
    tiles = []
    for token in (1, 2):
        b._load_token = token
        b._run_load(
            repo_id = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z.gguf",
            base = fam.base_repo,
            fam = fam,
            hf_token = None,
            _load_token = token,
        )
        flags = list(b._state.offload_flags)
        assert "--vae-on-cpu" in flags
        tiles.append(
            flags[flags.index("--vae-tile-size") + 1] if "--vae-tile-size" in flags else None
        )
        assert len(_HoldingServer.running) == 1
    assert tiles == ["1024", "1024"]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-start fused-VAE Triton prebuild in a child process (``diffusion_vae_prebuild``)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.torch_version import TorchVersion  # noqa: E402


from core.inference import diffusion_vae_prebuild as prebuild  # noqa: E402

# A CUDA build's version tag: Studio reads ROCm from the "+rocm" tag too (AMD SDK / Radeon wheels leave
# torch.version.hip unset), so a test that fakes NVIDIA clears both.
_CUDA_TORCH_VERSION = TorchVersion(str(torch.__version__).split("+")[0] + "+cu130")

BACKEND = Path(__file__).resolve().parents[1]


class _Param:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def is_floating_point(self):
        return True


class _Decoder:
    def __init__(self, param):
        self._p = param

    def parameters(self):
        yield self._p


def _vae(
    config,
    *,
    fused = 5,
    device = types.SimpleNamespace(type = "cuda", index = 1),
):
    vae = types.SimpleNamespace(config = config, decoder = _Decoder(_Param(torch.bfloat16, device)))
    vae._unsloth_vae_fused_installed = fused
    return vae


def test_plan_describes_the_class_config_dtype_and_a_small_latent():
    job = prebuild.plan(
        types.SimpleNamespace(vae = _vae({"latent_channels": 16, "block_out_channels": (128, 256)}))
    )
    assert job["name"] == "SimpleNamespace" and job["dtype"] == "bfloat16" and job["device"] == 1
    assert job["shape"] == [1, 16, prebuild._LATENT_SIDE, prebuild._LATENT_SIDE]
    assert job["config"]["block_out_channels"] == [128, 256]  # JSON-safe for the spawn
    video = prebuild.plan(types.SimpleNamespace(vae = _vae({"z_dim": 16})))
    assert video["shape"] == [1, 16, 1, prebuild._LATENT_SIDE, prebuild._LATENT_SIDE]


def test_nothing_to_prebuild_without_fused_passes_on_a_cuda_vae():
    assert prebuild.plan(types.SimpleNamespace(vae = None)) is None
    assert prebuild.plan(types.SimpleNamespace(vae = _vae({"latent_channels": 4}, fused = 0))) is None
    cpu = types.SimpleNamespace(type = "cpu", index = None)
    assert (
        prebuild.plan(types.SimpleNamespace(vae = _vae({"latent_channels": 4}, device = cpu))) is None
    )
    assert prebuild.plan(types.SimpleNamespace(vae = _vae({"something_else": 1}))) is None


def test_restart_and_kill_switch_never_spawn(monkeypatch):
    spawned = []
    monkeypatch.setattr(prebuild, "_spawn", lambda job, logger: spawned.append(job) or True)
    monkeypatch.setattr(prebuild, "plan", lambda pipe: {"device": 0})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", _CUDA_TORCH_VERSION, raising = False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * 1024**3, 96 * 1024**3))
    pipe = types.SimpleNamespace()
    assert prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = True)) is False
    assert prebuild.maybe_kick(pipe, None) is False
    monkeypatch.setenv("UNSLOTH_DIFFUSION_VAE_PREBUILD", "0")
    assert prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = False)) is False
    monkeypatch.delenv("UNSLOTH_DIFFUSION_VAE_PREBUILD")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * 1024**3, 24 * 1024**3))
    assert (
        prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = False)) is False
    )  # too little VRAM for a 2nd context
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * 1024**3, 96 * 1024**3))
    assert prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = False)) is True
    assert spawned == [{"device": 0, "owner": "image"}]


def test_a_video_vae_plans_a_5d_latent():
    job = prebuild.plan(
        types.SimpleNamespace(
            vae = _vae({"latent_channels": 128, "temporal_compression_ratio": 8, "patch_size": 4})
        )
    )
    assert job["shape"] == [1, 128, *prebuild._VIDEO_LATENT]


def _video_gates(
    monkeypatch,
    tmp_path,
    name = "AutoencoderKLLTX2Video",
):
    spawned = []
    monkeypatch.setattr(prebuild, "_spawn", lambda job, logger: spawned.append(job) or True)
    monkeypatch.setattr(
        prebuild,
        "plan",
        lambda pipe: {
            "name": name,
            "config": {"a": 1},
            "dtype": "bfloat16",
            "shape": [1],
            "device": 0,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", _CUDA_TORCH_VERSION, raising = False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * 1024**3, 96 * 1024**3))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    return spawned


def test_video_kick_spawns_once_per_triton_cache(monkeypatch, tmp_path):
    spawned = _video_gates(monkeypatch, tmp_path)
    pipe = types.SimpleNamespace()
    assert prebuild.maybe_kick_video(pipe) is True
    marker = spawned[0]["marker"]
    assert marker.startswith(str(tmp_path)) and not os.path.exists(marker)
    # the child writes the marker after its decode: the next load (or restart) on this cache spawns nothing
    os.makedirs(os.path.dirname(marker))
    Path(marker).write_text("x")
    assert prebuild.maybe_kick_video(pipe) is False
    assert len(spawned) == 1
    # another card arch keys another build
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert prebuild.maybe_kick_video(pipe) is True


def test_video_kick_gates(monkeypatch, tmp_path):
    spawned = _video_gates(monkeypatch, tmp_path)
    pipe = types.SimpleNamespace()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_VAE_PREBUILD", "0")
    assert prebuild.maybe_kick_video(pipe) is False
    monkeypatch.delenv("UNSLOTH_DIFFUSION_VAE_PREBUILD")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * 1024**3, 24 * 1024**3))
    assert prebuild.maybe_kick_video(pipe) is False  # too little VRAM for a 2nd context
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * 1024**3, 96 * 1024**3))
    monkeypatch.setattr(prebuild, "plan", lambda pipe: None)
    assert prebuild.maybe_kick_video(pipe) is False
    assert spawned == []
    other = _video_gates(
        monkeypatch, tmp_path, name = "AutoencoderKLWan"
    )  # only the measured video VAE
    assert prebuild.maybe_kick_video(pipe) is False and other == []


def test_video_kick_never_raises(monkeypatch, tmp_path):
    _video_gates(monkeypatch, tmp_path)

    def boom(job, logger):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(prebuild, "_spawn", boom)
    assert prebuild.maybe_kick_video(types.SimpleNamespace()) is False


def test_a_quit_during_the_load_never_starts_the_child(monkeypatch):
    import multiprocessing as mp

    from utils import process_lifetime

    started = []
    monkeypatch.setattr(mp, "get_context", lambda method: started.append(method))
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: True)
    assert prebuild._spawn({"name": "AutoencoderKL"}, None) is False
    assert started == []


def test_unload_kills_a_child_that_is_still_building():
    class _Proc:
        pid = None
        alive = True

        def is_alive(self):
            return self.alive

        def kill(self):
            self.alive = False

        def join(self, timeout = None):
            pass

    proc = _Proc()
    prebuild._LIVE[proc] = ("image", 0)
    assert prebuild.cancel_all("image") == 1
    assert proc.alive is False and proc not in prebuild._LIVE
    assert prebuild.cancel_all() == 0


class _LiveProc:
    pid = None

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def kill(self):
        self.alive = False

    def join(self, timeout = None):
        pass


def test_an_unload_cancels_only_its_own_kind_of_child():
    image, video = _LiveProc(), _LiveProc()
    prebuild._LIVE[image] = ("image", 0)
    prebuild._LIVE[video] = ("video", 0)
    try:
        assert prebuild.cancel_all("video") == 1
        assert video.alive is False and image.alive is True and image in prebuild._LIVE
        assert prebuild.cancel_all("image") == 1
        assert image.alive is False and not prebuild._LIVE
    finally:
        prebuild._LIVE.clear()


@pytest.mark.parametrize("free_gib, stopped", [(6.5, True), (12.0, False)])
def test_a_render_on_a_tight_card_stops_the_building_video_child(monkeypatch, free_gib, stopped):
    import core.inference.diffusion_memory as mem

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (int(free_gib * 2**30), 24 * 2**30))
    monkeypatch.setattr(mem, "user_budget_reserve_bytes", lambda d, t: 0)
    video, image = _LiveProc(), _LiveProc()
    prebuild._LIVE[video] = ("video", 0)
    prebuild._LIVE[image] = ("image", 0)
    try:
        assert prebuild.yield_to_render("video") == int(stopped)
        assert video.alive is (not stopped)
        assert image.alive is True  # another owner's child is never touched
    finally:
        prebuild._LIVE.clear()


def test_the_video_marker_follows_the_triton_cache_key_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 9))
    job = {
        "name": "AutoencoderKLLTX2Video",
        "config": {"a": 1},
        "dtype": "bfloat16",
        "shape": [1, 128, 2, 8, 8],
        "device": 0,
    }
    monkeypatch.delenv("TRITON_DISABLE_LINE_INFO", raising = False)
    a = prebuild._marker_path(job)
    monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", "1")
    b = prebuild._marker_path(job)
    assert a != b and a.startswith(str(tmp_path)) and b.startswith(str(tmp_path))


def test_legacy_triton_cache_dir_honours_triton_home(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_triton(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("no triton")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("TRITON_CACHE_DIR", raising = False)
    monkeypatch.setenv("TRITON_HOME", str(tmp_path))
    monkeypatch.setattr(builtins, "__import__", no_triton)
    assert prebuild._triton_cache_dir() == os.path.join(str(tmp_path), ".triton", "cache")


_SCRIPT = textwrap.dedent(
    r"""
    import os, sys, json
    sys.path.insert(0, os.environ["BACKEND"])
    import torch
    from diffusers import AutoencoderKL
    from core.inference import diffusion_vae_fused, diffusion_vae_prebuild as prebuild

    cfg = dict(block_out_channels = [32, 64], down_block_types = ["DownEncoderBlock2D"] * 2,
               up_block_types = ["UpDecoderBlock2D"] * 2, latent_channels = 4, norm_num_groups = 32,
               layers_per_block = 1)
    vae = AutoencoderKL.from_config(cfg).cuda().to(torch.bfloat16).eval()
    if not diffusion_vae_fused.install(vae, None):
        print("RESULT", json.dumps({"skip": "fused VAE does not install here"}))
        raise SystemExit
    cache = os.environ["TRITON_CACHE_DIR"]
    count = lambda: sum(len(f) for _, _, f in os.walk(cache))
    mode = sys.argv[1]
    if mode == "child":
        prebuild._child_entry(prebuild.plan(type("P", (), {"vae": vae})()))
        print("RESULT", json.dumps({"files": count()}))
    else:
        before = count()
        with torch.inference_mode():
            vae.decode(torch.zeros(1, 4, 128, 128, device = "cuda", dtype = torch.bfloat16))
        torch.cuda.synchronize()
        print("RESULT", json.dumps({"before": before, "after": count()}))
    """
)


def _run(tmp_path: Path, mode: str) -> dict:
    import json

    env = dict(
        os.environ,
        BACKEND = str(BACKEND),
        TRITON_CACHE_DIR = str(tmp_path / "triton"),
        PYTHONPATH = str(BACKEND),
    )
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT, mode],
        cwd = str(BACKEND),
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(line[-1][len("RESULT ") :])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "the child JIT-compiles real Triton kernels, which needs a CUDA GPU",
)
def test_child_fills_the_triton_cache_the_full_size_decode_then_reads(tmp_path):
    child = _run(tmp_path, "child")
    if "skip" in child:
        pytest.skip(child["skip"])
    assert child["files"] > 0
    parent = _run(tmp_path, "parent")
    assert parent["after"] == parent["before"], parent


def test_video_kick_runs_under_the_teardown_lock_after_the_token_check():
    """An unload between the load's lock release and the kick would run cancel_all() before the child exists."""
    import ast

    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        ctx = [ast.unparse(item.context_expr) for item in node.items]
        if ctx != ["self._lock"]:
            continue
        body = ast.unparse(node)
        if "maybe_kick_video" in body:
            hits.append(body)
    assert len(hits) == 1, "maybe_kick_video must be called inside one `with self._lock:` block"
    body = hits[0]
    assert body.index("_load_token != self._load_token") < body.index("maybe_kick_video")
    assert src.count("maybe_kick_video(") == 1

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

from core.inference import diffusion_vae_prebuild as prebuild  # noqa: E402

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
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * 1024**3, 96 * 1024**3))
    pipe = types.SimpleNamespace()
    assert (
        prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = True)) is False
    )  # restart: cache already warm
    assert prebuild.maybe_kick(pipe, None) is False  # no compile bundle at all
    monkeypatch.setenv("UNSLOTH_DIFFUSION_VAE_PREBUILD", "0")
    assert prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = False)) is False
    monkeypatch.delenv("UNSLOTH_DIFFUSION_VAE_PREBUILD")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * 1024**3, 24 * 1024**3))
    assert (
        prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = False)) is False
    )  # too little VRAM for a 2nd context
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (64 * 1024**3, 96 * 1024**3))
    assert prebuild.maybe_kick(pipe, types.SimpleNamespace(hit = False)) is True
    assert spawned == [{"device": 0}]


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
    prebuild._LIVE.add(proc)
    assert prebuild.cancel_all() == 1
    assert proc.alive is False and proc not in prebuild._LIVE
    assert prebuild.cancel_all() == 0


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
        pytest.skip(
            child["skip"]
        )  # e.g. Triton absent / fused passes refuse this GPU: nothing to prebuild
    assert child["files"] > 0
    parent = _run(tmp_path, "parent")
    # A full-resolution decode in a fresh process compiles nothing new: every kernel came from the child's 64x64 decode.
    assert parent["after"] == parent["before"], parent

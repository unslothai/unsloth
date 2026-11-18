# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Family sd.cpp --scheduler: the diffusers / baseline sigma grid on both transports, with its kill switch."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.sd_cpp_args import (
    FAMILY_SCHEDULER_ENV,
    SdCppModelFiles,
    build_sd_cpp_command,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

CASES = [
    ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", "simple", None),
    ("unsloth/MOSS-Image-SFT-GGUF", "MOSS-Image-SFT-Q8_0.gguf", "simple", None),
    ("vantagewithai/Krea-2-Turbo-GGUF", "krea2_turbo-Q4_K_M.gguf", "simple", None),
    ("unsloth/FLUX.1-schnell-GGUF", "flux1-schnell-Q2_K.gguf", "simple", 0.0),
    ("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", "simple", None),
    ("unsloth/Qwen-Image-GGUF", "qwen-image-Q4_K_M.gguf", "simple", 3.0),
    # Qwen-Image-2.1 keeps sd.cpp's resolution-dependent official schedule.
    ("unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q4_K_M.gguf", None, None),
]
FILES = SdCppModelFiles(diffusion_model = "/m/dit.gguf", vae = "/m/vae.safetensors", llm = "/m/te.gguf")


def _state(
    repo,
    gguf,
    *,
    mode = "oneshot",
    server = None,
    grid = None,
):
    fam = detect_family_for_pick(repo, gguf, None)
    return bk._SdState(
        repo_id = repo,
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = FILES,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
        scheduler = fam.sd_cpp_scheduler,
        sample_sigmas = grid,
        mode = mode,
        server = server,
        gguf_filename = gguf,
    )


def _oneshot_argv(repo, gguf, **kw):
    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state(repo, gguf, **kw)
    b.generate(prompt = "p", steps = 8, guidance = 1.0, seed = 3)
    files, params, out, extra = engine.calls[-1]
    return build_sd_cpp_command(
        "sd-cli", files, params, output_path = out, extra_args = list(extra.get("extra_args") or [])
    )


def _server_sample_params(repo, gguf):
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(repo, gguf, mode = "server", server = server)
    b.generate(prompt = "p", steps = 8, guidance = 1.0, seed = 3)
    return server.payloads[-1]["sample_params"]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


@pytest.mark.parametrize("repo,gguf,scheduler,shift", CASES)
def test_oneshot_argv_carries_the_family_scheduler(repo, gguf, scheduler, shift):
    argv = _oneshot_argv(repo, gguf)
    assert _flag(argv, "--scheduler") == scheduler
    got = _flag(argv, "--flow-shift")
    assert (float(got) if got is not None else None) == shift


@pytest.mark.parametrize("repo,gguf,scheduler,shift", CASES)
def test_server_body_carries_the_family_scheduler(repo, gguf, scheduler, shift):
    sp = _server_sample_params(repo, gguf)
    assert sp.get("scheduler") == scheduler
    assert sp.get("flow_shift") == shift


@pytest.mark.parametrize("repo,gguf,scheduler,shift", CASES)
def test_kill_switch_restores_sd_cpp_defaults(monkeypatch, repo, gguf, scheduler, shift):
    monkeypatch.setenv(FAMILY_SCHEDULER_ENV, "0")
    argv = _oneshot_argv(repo, gguf)
    assert "--scheduler" not in argv
    fam = detect_family_for_pick(repo, gguf, None)
    got = _flag(argv, "--flow-shift")
    assert (float(got) if got is not None else None) == fam.sd_cpp_flow_shift
    assert "scheduler" not in _server_sample_params(repo, gguf)


@pytest.mark.parametrize(
    "repo,scheduler",
    [
        ("black-forest-labs/FLUX.1-Kontext-dev", "simple"),
        ("Qwen/Qwen-Image-Edit-2511", "simple"),
        ("Qwen/Qwen-Image-Layered", "simple"),
        ("black-forest-labs/FLUX.2-klein-4B", None),
        ("black-forest-labs/FLUX.2-dev", None),
        ("black-forest-labs/flux-3-image-klein", None),
    ],
)
def test_edit_and_untouched_families(repo, scheduler):
    assert detect_family_for_pick(repo, None, None).sd_cpp_scheduler == scheduler


def test_a_shipped_grid_sends_no_scheduler():
    argv = _oneshot_argv("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", grid = (1.0, 0.5))
    assert "--scheduler" not in argv and "--sigmas" in argv


def test_a_dev_gguf_never_takes_the_schnell_shift_from_its_base():
    state = _state("someone/FLUX.1-dev-gguf", "model-Q4_K_M.gguf")
    assert state.base_repo.lower().endswith("schnell")
    assert bk._request_flow_shift(state, 1024, 1024) is None

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) MOSS-Image negative-branch skip per checkpoint, on both transports: SFT sends
moss_uncond_skip=false, Base the sigma-percent window 0 .. 0.5, the kill switch and every other family nothing."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.sd_cpp_args import (
    MOSS_REFERENCE_SKIP_ENV,
    SdCppModelFiles,
    build_img_gen_request,
    moss_is_base_checkpoint,
    moss_uncond_skip_sample_args,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

SFT_ARGS = "moss_uncond_skip=false"
BASE_ARGS = "moss_uncond_skip_start=0,moss_uncond_skip_end=0.5"
FILES = SdCppModelFiles(
    diffusion_model = "/m/model.gguf",
    vae = "/m/vae/diffusion_pytorch_model.safetensors",
    llm = "/m/Qwen3.5-9B-Q8_0.gguf",
    tokenizer = "/m/tokenizer/tokenizer.json",
)


@pytest.fixture(autouse = True)
def _no_kill_switch(monkeypatch):
    monkeypatch.delenv(MOSS_REFERENCE_SKIP_ENV, raising = False)


def _state(
    repo,
    gguf,
    *,
    mode = "oneshot",
    server = None,
    base_repo = None,
):
    fam = detect_family_for_pick(repo, gguf, None)
    return bk._SdState(
        repo_id = repo,
        base_repo = base_repo or fam.base_repo,
        family = fam,
        device = "cpu",
        files = FILES,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
        mode = mode,
        server = server,
        gguf_filename = gguf,
    )


def _server_sample_params(repo, gguf, **kw):
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(repo, gguf, mode = "server", server = server, **kw)
    b.generate(prompt = "p", steps = 25, guidance = 4.0, seed = 3)
    return server.payloads[-1].get("sample_params") or {}


def _oneshot_extra_args(repo, gguf, **kw):
    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state(repo, gguf, **kw)
    b.generate(prompt = "p", steps = 25, guidance = 4.0, seed = 3)
    return list(engine.calls[-1][3].get("extra_args") or [])


_SFT = ("unsloth/MOSS-Image-SFT-GGUF", "MOSS-Image-SFT-Q4_K_M.gguf")
_BASE = ("unsloth/MOSS-Image-Base-GGUF", "MOSS-Image-Base-Q4_K_M.gguf")


@pytest.mark.parametrize(
    "identities, base",
    [
        (("unsloth/MOSS-Image-Base-GGUF",), True),
        (("OpenMOSS-Team/MOSS-Image-Base",), True),
        (("me/my-moss", "moss_image_base-q8_0.gguf"), True),
        (("unsloth/MOSS-Image-SFT-GGUF", "MOSS-Image-SFT-Q4_K_M.gguf"), False),
        ((None, ""), False),
    ],
)
def test_base_is_spelled_by_any_identity(identities, base):
    assert moss_is_base_checkpoint(*identities) is base


def test_policy_strings_and_kill_switch(monkeypatch):
    assert moss_uncond_skip_sample_args(*_SFT) == SFT_ARGS
    assert moss_uncond_skip_sample_args(*_BASE) == BASE_ARGS
    for off in ("0", "off", "false", "No"):
        monkeypatch.setenv(MOSS_REFERENCE_SKIP_ENV, off)
        assert moss_uncond_skip_sample_args(*_SFT) is None
        assert moss_uncond_skip_sample_args(*_BASE) is None


def test_request_builder_only_adds_the_key_when_given():
    assert "extra_sample_args" not in build_img_gen_request(prompt = "p", steps = 25).get(
        "sample_params", {}
    )
    req = build_img_gen_request(prompt = "p", steps = 25, extra_sample_args = BASE_ARGS)
    assert req["sample_params"]["extra_sample_args"] == BASE_ARGS


@pytest.mark.parametrize("pick, expected", [(_SFT, SFT_ARGS), (_BASE, BASE_ARGS)])
def test_server_body_carries_the_checkpoint_window(pick, expected):
    assert _server_sample_params(*pick)["extra_sample_args"] == expected


@pytest.mark.parametrize("pick, expected", [(_SFT, SFT_ARGS), (_BASE, BASE_ARGS)])
def test_oneshot_argv_carries_the_checkpoint_window(pick, expected):
    extra = _oneshot_extra_args(*pick)
    i = extra.index("--extra-sample-args")
    assert extra[i + 1] == expected
    assert extra.count("--extra-sample-args") == 1


def test_a_base_gguf_resolved_to_the_sft_base_repo_still_windows():
    # The Base pick is recognised from its repo / filename even when base_repo resolved to the family's SFT base.
    assert (
        _server_sample_params(*_BASE, base_repo = "OpenMOSS-Team/MOSS-Image-SFT")["extra_sample_args"]
        == BASE_ARGS
    )


def test_kill_switch_sends_nothing_on_both_transports(monkeypatch):
    monkeypatch.setenv(MOSS_REFERENCE_SKIP_ENV, "0")
    for pick in (_SFT, _BASE):
        assert "extra_sample_args" not in _server_sample_params(*pick)
        assert "--extra-sample-args" not in _oneshot_extra_args(*pick)


def test_other_families_send_nothing():
    pick = ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf")
    assert "extra_sample_args" not in _server_sample_params(*pick)
    assert "--extra-sample-args" not in _oneshot_extra_args(*pick)

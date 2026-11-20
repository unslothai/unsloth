# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A generate's prompt upsampler (FLUX.3 Klein, MiniMax-H3-Image) only starts when the whole GGUF fits on the GPU
beside the loaded image model. With less free memory the old start split the layers onto the CPU, and one caption
took minutes (H3-Image's thinking Qwen3.5-9B hit the 600 s request timeout on an 8-16 GB card) for a ~7 s render:
the render now runs the prompt as typed with status "unavailable". A preview, an opt-in env and a card that fits keep
the old start. No GPU: memory, the binary and Popen are faked."""

from __future__ import annotations

import pytest

from core.inference import flux3_prompt_upsampler as pup

GIB = 1024**3


class _Popened(Exception):
    pass


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    """One CUDA card owned by the image model; ``state["free"]`` MiB free. Popen records argv and raises."""
    import torch

    from core.inference import llama_cpp, stt_mtmd_sidecar, stt_sidecar
    from utils import process_lifetime

    model = tmp_path / "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    with open(model, "wb") as fh:
        fh.truncate(6 * GIB)  # sparse: only its size is read
    state = {"free": 4096, "argv": []}
    monkeypatch.delenv(pup.ENV_ALLOW_PARTIAL_OFFLOAD, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False)
    )
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    monkeypatch.setattr(
        pup, "_accelerator_memory_mib", lambda o = None: (state["free"], 48 * 1024, "cuda")
    )
    monkeypatch.setattr(pup, "_gpu_unowned", lambda: False)
    monkeypatch.setattr(pup, "_image_model_owns_gpu", lambda: True)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    monkeypatch.setattr(stt_mtmd_sidecar, "_llama_server_child_env", lambda b: {})
    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: False)
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: False)
    monkeypatch.setattr(process_lifetime, "spawn_on_lifetime_thread", lambda fn: fn())

    def _popen(
        cmd,
        env = None,
        **_kwargs,
    ):
        state["argv"].append(cmd)
        raise _Popened

    monkeypatch.setattr(pup.subprocess, "Popen", _popen)
    return state, str(model)


def _ngl(argv):
    return int(argv[argv.index("-ngl") + 1])


def _spawn_for_generate(model):
    token = pup._GENERATE_NEEDS_FULL_OFFLOAD.set(True)
    try:
        pup.PromptUpsamplerSidecar()._spawn_server("/x/llama-server", model, None)
    finally:
        pup._GENERATE_NEEDS_FULL_OFFLOAD.reset(token)


def test_a_generate_does_not_start_a_split_server(gpu):
    state, model = gpu
    with pytest.raises(pup.UpsamplerUnavailableError, match = "does not fit"):
        _spawn_for_generate(model)
    assert state["argv"] == []


def test_a_generate_starts_a_server_that_fits(gpu):
    state, model = gpu
    state["free"] = 16 * 1024
    with pytest.raises(_Popened):
        _spawn_for_generate(model)
    assert _ngl(state["argv"][-1]) == 99


def test_a_preview_and_the_opt_in_keep_the_split_start(gpu, monkeypatch):
    state, model = gpu
    with pytest.raises(_Popened):
        pup.PromptUpsamplerSidecar()._spawn_server("/x/llama-server", model, None)
    assert 0 < _ngl(state["argv"][-1]) < 99 or _ngl(state["argv"][-1]) == 0
    monkeypatch.setenv(pup.ENV_ALLOW_PARTIAL_OFFLOAD, "1")
    with pytest.raises(_Popened):
        _spawn_for_generate(model)
    assert len(state["argv"]) == 2


def test_a_cpu_only_host_keeps_its_cpu_start(gpu, monkeypatch):
    state, model = gpu
    monkeypatch.setattr(pup, "_accelerator_memory_mib", lambda o = None: (None, None, "cpu"))
    with pytest.raises(_Popened):
        _spawn_for_generate(model)
    assert _ngl(state["argv"][-1]) == 0


@pytest.mark.parametrize("family", ["minimax-h3-image", "flux.3-klein"])
def test_generate_runs_the_prompt_as_typed_when_the_upsampler_cannot_fit(gpu, monkeypatch, family):
    from core.inference import h3_image_prompt_enhancer as h3pe

    state, model = gpu
    monkeypatch.setattr(pup, "find_llama_server_binary", lambda: "/x/llama-server")
    monkeypatch.setattr(pup, "model_path", lambda: model)
    monkeypatch.setattr(h3pe, "model_path", lambda: model)
    monkeypatch.setattr(pup.PromptUpsamplerSidecar, "_wait_for_render_slot", lambda self: None)
    sidecar = pup.PromptUpsamplerSidecar()
    monkeypatch.setattr(pup, "get_sidecar", lambda: sidecar)
    assert pup.supports_family(family) and pup.resolve_enhance(family, None)
    prompt = f"a ginger cat on old books ({family})"
    outcome = pup.enhance_for_generation([prompt], family_name = family, enhance_prompt = None)
    assert outcome.status == pup.STATUS_UNAVAILABLE
    assert outcome.encoded(prompt) == prompt
    # Never started: no llama-server holding memory beside the render, no minutes of CPU decode.
    assert state["argv"] == []
    assert not sidecar.loaded
    # The flag is scoped to the generate: a preview afterwards still starts split.
    with pytest.raises(_Popened):
        sidecar._spawn_server("/x/llama-server", model, None)


def test_a_running_split_server_is_not_used_for_a_render(gpu, monkeypatch):
    _state, model = gpu

    class _Alive:
        def poll(self):
            return None

    sidecar = pup.PromptUpsamplerSidecar()
    sidecar._process = _Alive()
    sidecar._key = ("/x/llama-server", model, None)
    sidecar._gpu_layers = 8
    sidecar._split_on_accelerator = True
    posted = []
    monkeypatch.setattr(pup, "_post_chat", lambda *a, **k: posted.append(1) or "caption")
    token = pup._GENERATE_NEEDS_FULL_OFFLOAD.set(True)
    try:
        with pytest.raises(pup.UpsamplerUnavailableError):
            sidecar.generate({}, binary = "/x/llama-server", model = model, mmproj = None)
    finally:
        pup._GENERATE_NEEDS_FULL_OFFLOAD.reset(token)
    assert posted == []
    # A preview still uses it.
    assert sidecar.generate({}, binary = "/x/llama-server", model = model, mmproj = None) == "caption"
    sidecar._cancel_idle_locked()
    sidecar._process = None

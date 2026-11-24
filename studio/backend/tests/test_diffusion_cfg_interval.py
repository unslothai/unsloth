# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fast CFG (limited-interval CFG) on FLUX.3 Klein base: the guider the pipeline installs, its state reset between
calls, and the route / request surface."""

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("diffusers")

from core.inference.diffusion_flux3 import _Flux3KleinPipelineMixin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_routes import client  # noqa: E402,F401  (fixture)


class _MP:
    def __init__(self, guider):
        self.guider = guider

    def update_components(self, guider):
        self.guider = guider


def _guider(scale = 3.5, stop = 1.0):
    from diffusers import ClassifierFreeGuidance
    return ClassifierFreeGuidance(guidance_scale = scale, stop = stop)


def _set(mp, scale, stop):
    _Flux3KleinPipelineMixin._set_guidance(types.SimpleNamespace(), mp, scale, stop)


def test_stop_installs_an_interval_guider():
    mp = _MP(_guider())
    _set(mp, 3.5, 0.8)
    assert mp.guider.config.stop == 0.8
    assert mp.guider.config.guidance_scale == 3.5
    mp.guider.set_state(step = 39, num_inference_steps = 50, timestep = None)
    assert mp.guider.num_conditions == 2
    mp.guider.set_state(step = 40, num_inference_steps = 50, timestep = None)
    assert mp.guider.num_conditions == 1


def test_default_keeps_full_cfg_and_reverts_an_interval():
    mp = _MP(_guider(stop = 0.8))
    _set(mp, 3.5, None)
    assert mp.guider.config.stop == 1.0
    mp.guider.set_state(step = 49, num_inference_steps = 50, timestep = None)
    assert mp.guider.num_conditions == 2


def test_unchanged_guider_is_kept():
    g = _guider(stop = 0.8)
    mp = _MP(g)
    _set(mp, 3.5, 0.8)
    assert mp.guider is g


def test_state_is_reset_before_each_call():
    g = _guider(stop = 0.8)
    g.set_state(step = 49, num_inference_steps = 50, timestep = None)
    assert g.num_conditions == 1
    mp = _MP(g)
    _set(mp, 3.5, 0.8)
    assert mp.guider is g
    assert mp.guider.num_conditions == 2


def test_call_refuses_an_out_of_range_stop():
    pipe = types.SimpleNamespace(modular_pipeline = lambda: _MP(_guider()), is_distilled = False)
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match = "guidance_stop"):
            _Flux3KleinPipelineMixin.__call__(
                pipe, prompt = "a sloth", guidance_scale = 3.5, guidance_stop = bad
            )


def test_route_passes_or_refuses_cfg_interval_end():
    from routes.inference import _cfg_interval_kwargs

    def diffusers_generate(*, prompt, cfg_interval_end = None):
        return None

    def native_generate(*, prompt):
        return None

    unset = types.SimpleNamespace(cfg_interval_end = None)
    req = types.SimpleNamespace(cfg_interval_end = 0.8)
    assert _cfg_interval_kwargs(native_generate, unset) == {}
    assert _cfg_interval_kwargs(diffusers_generate, unset) == {}
    assert _cfg_interval_kwargs(diffusers_generate, req) == {"cfg_interval_end": 0.8}
    with pytest.raises(ValueError, match = "Fast CFG"):
        _cfg_interval_kwargs(native_generate, req)


def test_request_model_bounds():
    from pydantic import ValidationError

    from models.inference import DiffusionGenerateRequest

    assert DiffusionGenerateRequest(prompt = "a").cfg_interval_end is None
    assert DiffusionGenerateRequest(prompt = "a", cfg_interval_end = 0.8).cfg_interval_end == 0.8
    for bad in (0.0, 1.01, -1):
        with pytest.raises(ValidationError):
            DiffusionGenerateRequest(prompt = "a", cfg_interval_end = bad)


def test_no_scale_drops_a_stale_interval():
    mp = _MP(_guider(scale = 4.0, stop = 0.8))
    _set(mp, None, None)
    assert mp.guider.config.stop == 1.0
    assert mp.guider.config.guidance_scale == 4.0


class _GuidedModular:
    """Runs the guider over every step like the FLUX.3 blocks and records the conditions the text step sees."""

    def __init__(self, guider):
        self.guider = guider
        self.scheduler = types.SimpleNamespace(step = lambda *a, **k: None)
        self.transformer = None
        self.text_conditions = []

    def update_components(self, guider):
        self.guider = guider

    def __call__(
        self,
        output = None,
        **kwargs,
    ):
        self.text_conditions.append(self.guider.num_conditions)
        steps = kwargs["num_inference_steps"]
        for i in range(steps):
            self.guider.set_state(step = i, num_inference_steps = steps, timestep = None)
        return [f"img:{kwargs['prompt']}"]


def test_every_image_of_a_batch_gets_its_negative_embeddings():
    from core.inference.diffusion_flux3 import FLUX3_BASE_BLOCKS

    class _Base(_Flux3KleinPipelineMixin):
        def __init__(self):
            self.config = types.SimpleNamespace(blocks_class_name = FLUX3_BASE_BLOCKS)
            self.mp = _GuidedModular(_guider())

        def modular_pipeline(self):
            return self.mp

        def progress_bar(self, total = None):
            return types.SimpleNamespace(close = lambda: None)

    pipe = _Base()
    out = pipe(
        prompt = ["a", "b", "c"],
        generator = ["g1", "g2", "g3"],
        num_inference_steps = 10,
        guidance_scale = 3.5,
        guidance_stop = 0.8,
        negative_prompt = "blurry",
    )
    assert out.images == ["img:a", "img:b", "img:c"]
    assert pipe.mp.guider.config.stop == 0.8
    assert pipe.mp.text_conditions == [2, 2, 2]
    pipe(prompt = "d", num_inference_steps = 10, guidance_scale = 3.5)
    assert pipe.mp.guider.config.stop == 1.0
    assert pipe.mp.text_conditions[-1] == 2


def test_infotext_names_fast_cfg_only_when_it_applied():
    from core.inference.image_gallery import _params_text

    base = {
        "prompt": "p",
        "steps": 50,
        "guidance": 3.5,
        "seed": 1,
        "width": 64,
        "height": 64,
        "model": "m",
    }
    assert "Fast CFG" not in _params_text(base)
    assert _params_text({**base, "cfg_interval_end": None}) == _params_text(base)
    assert _params_text({**base, "cfg_interval_end": 0.8}).endswith(", Fast CFG: 0.8")


def _tiny_flux3_pipe():
    import torch
    from diffusers import ClassifierFreeGuidance
    from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel

    torch.manual_seed(0)

    class _Pipe(_Flux3KleinPipelineMixin):
        def __init__(self):
            pass

    pipe = _Pipe()
    pipe.transformer = Flux3Transformer2DModel(
        in_channels = 8,
        cond_in_channels = 8,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 128,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        timestep_channels = 16,
    ).eval()
    pipe.mp = _MP(ClassifierFreeGuidance(guidance_scale = 3.5))
    pipe.mp.transformer = pipe.transformer
    return pipe


def _render(
    pipe,
    stop,
    steps = 5,
):
    import torch
    from diffusers.modular_pipelines.flux3.denoise import Flux3LoopDenoiser
    from diffusers.modular_pipelines.modular_pipeline import BlockState

    from core.inference.diffusion_cfg_batch import end_cfg_batch_render

    pipe._set_guidance(pipe.mp, 3.5, stop)
    rng = torch.Generator().manual_seed(42)
    ids = torch.zeros(8, 4)
    state = BlockState(
        latents = torch.randn(1, 8, 8, generator = rng),
        num_inference_steps = steps,
        denoiser_input_fields = {
            "prompt_embeds": torch.randn(1, 8, 24, generator = rng),
            "negative_prompt_embeds": torch.randn(1, 8, 24, generator = rng),
            "txt_ids": ids,
            "negative_txt_ids": ids,
            "img_ids": torch.zeros(8, 4),
        },
    )
    try:
        with torch.no_grad():
            for i in range(steps):
                t = torch.tensor(1000.0 - i * (1000.0 / steps))
                _, state = Flux3LoopDenoiser()(pipe.mp, state, i, t)
                state.latents = state.latents - 0.1 * state.noise_pred
    finally:
        end_cfg_batch_render(pipe)


def test_short_fast_cfg_renders_keep_offloaded_cfg_batching():
    pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    from core.inference import diffusion_cfg_batch

    pipe = _tiny_flux3_pipe()
    hooks = {"group_offloading": object()}
    pipe.transformer.single_transformer_blocks[0]._diffusers_hook = types.SimpleNamespace(
        hooks = hooks, get_hook = hooks.get
    )
    assert diffusion_cfg_batch.install_cfg_batch(pipe, offload_policy = "group") is None
    router = diffusion_cfg_batch.router_for(pipe)
    assert router.offloaded
    _render(pipe, None)
    assert router.describe()["hits"] > 0
    for _ in range(3):
        _render(pipe, 0.2)
    before = router.describe()
    assert before["shape_off"] is None and before["misses"] == 0
    _render(pipe, None)
    after = router.describe()
    assert after["shape_off"] is None
    assert after["hits"] - before["hits"] == 4


def test_external_connection_refuses_cfg_interval_end(client, monkeypatch):
    import routes.inference as inference_routes

    looked_up = []
    monkeypatch.setattr(
        inference_routes.providers_db, "get_provider", lambda pid: looked_up.append(pid)
    )
    response = client.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "p",
            "provider_id": "openai-1",
            "model": "gpt-image-1",
            "cfg_interval_end": 0.8,
        },
    )
    assert response.status_code == 400, response.text
    assert (
        response.json()["detail"]
        == "An external image connection only supports text to image and image edits."
    )
    assert looked_up == []

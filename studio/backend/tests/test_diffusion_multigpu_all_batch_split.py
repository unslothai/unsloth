# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch split for the remaining DiT classes and Cosmos 3 CFG pairing on tiny random diffusers models (CPU)."""

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_multigpu as mg  # noqa: E402

# literal, not ALL_ENV: on a tree without the switch the tests fail on behaviour, not on a name
ALL_ENV = "UNSLOTH_DIFFUSION_MULTIGPU_ALL"
B = 2
MASK = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def _need(name):
    cls = getattr(diffusers, name, None)
    if cls is None:
        pytest.skip(f"diffusers has no {name}")
    return cls


def _router(model):
    built = []

    def factory():
        r = mg.build_replica(model, torch.device("cpu"))
        built.append(r)
        return r

    router = mg.CfgParallelRouter(model, primary = "cpu", secondary = "cpu", replica_factory = factory)
    model.forward = router
    return router, built


def _ltx2():
    m = _need("LTX2VideoTransformer3DModel")(
        in_channels = 8,
        out_channels = 8,
        num_attention_heads = 2,
        attention_head_dim = 8,
        cross_attention_dim = 16,
        audio_in_channels = 8,
        audio_out_channels = 8,
        audio_num_attention_heads = 2,
        audio_attention_head_dim = 8,
        audio_cross_attention_dim = 16,
        num_layers = 2,
        caption_channels = 16,
    )
    frames, height, width, audio_len = 2, 2, 2, 3
    g = torch.Generator().manual_seed(1)
    video_coords = m.rope.prepare_video_coords(
        1, frames, height, width, torch.device("cpu"), fps = 24.0
    )
    audio_coords = m.audio_rope.prepare_audio_coords(1, audio_len, torch.device("cpu"))
    # LTX2Pipeline batches uncond + cond into one call
    state = {
        "video": torch.randn(1, frames * height * width, 8, generator = g),
        "audio": torch.randn(1, audio_len, 8, generator = g),
    }
    text = torch.randn(B, 5, 16, generator = g)
    audio_text = torch.randn(B, 5, 16, generator = g)
    mask = torch.tensor(MASK)

    def step(model, i):
        t = torch.tensor(1000.0 * (1 - i / 4)).expand(B)
        video, audio = model(
            hidden_states = torch.cat([state["video"]] * 2),
            audio_hidden_states = torch.cat([state["audio"]] * 2),
            encoder_hidden_states = text,
            audio_encoder_hidden_states = audio_text,
            timestep = t,
            sigma = t,
            audio_timestep = t,
            encoder_attention_mask = mask,
            audio_encoder_attention_mask = mask,
            num_frames = frames,
            height = height,
            width = width,
            fps = 24.0,
            audio_num_frames = audio_len,
            video_coords = video_coords.repeat(2, 1, 1, 1),
            audio_coords = audio_coords.repeat(2, 1, 1, 1),
            isolate_modalities = False,
            spatio_temporal_guidance_blocks = None,
            perturbation_mask = None,
            use_cross_timestep = False,
            attention_kwargs = None,
            return_dict = False,
        )
        uv, cv = video.chunk(2)
        ua, ca = audio.chunk(2)
        state["video"] = state["video"] - 0.1 * (uv + 3.0 * (cv - uv))
        state["audio"] = state["audio"] - 0.1 * (ua + 3.0 * (ca - ua))
        return torch.cat([state["video"].flatten(), state["audio"].flatten()])

    return m, step


def _chroma():
    m = _need("ChromaTransformer2DModel")(
        in_channels = 8,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 16,
        axes_dims_rope = (4, 6, 6),
        approximator_num_channels = 8,
        approximator_hidden_dim = 16,
        approximator_layers = 1,
    )
    g = torch.Generator().manual_seed(2)
    state = {"x": torch.randn(B, 16, 8, generator = g)}
    kw = dict(
        encoder_hidden_states = torch.randn(B, 5, 16, generator = g),
        txt_ids = torch.zeros(5, 3),
        img_ids = torch.randn(16, 3, generator = g),
        attention_mask = torch.tensor([[1] * 3 + [0] * 2 + [1] * 16, [1] * 21]).bool(),
        joint_attention_kwargs = None,
        return_dict = False,
    )
    return m, _distilled_step(state, kw, lambda i: torch.full((B,), 1 - i / 4))


def _qwen():
    m = _need("QwenImageTransformer2DModel")(
        patch_size = 2,
        in_channels = 16,
        out_channels = 4,
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 16,
        axes_dims_rope = (4, 6, 6),
    )
    g = torch.Generator().manual_seed(3)
    state = {"x": torch.randn(B, 16, 16, generator = g)}
    kw = dict(
        guidance = None,
        encoder_hidden_states_mask = torch.tensor(MASK),
        encoder_hidden_states = torch.randn(B, 5, 16, generator = g),
        img_shapes = [[(1, 4, 4)]] * B,
        attention_kwargs = None,
        return_dict = False,
    )
    return m, _distilled_step(state, kw, lambda i: torch.full((B,), 1 - i / 4))


def _hunyuan():
    m = _need("HunyuanImageTransformer2DModel")(
        in_channels = 8,
        out_channels = 8,
        num_attention_heads = 2,
        attention_head_dim = 8,
        num_layers = 1,
        num_single_layers = 1,
        num_refiner_layers = 1,
        text_embed_dim = 16,
        rope_axes_dim = (4, 4),
        guidance_embeds = True,
    )
    g = torch.Generator().manual_seed(4)
    state = {"x": torch.randn(B, 8, 4, 4, generator = g)}
    kw = dict(
        timestep_r = None,
        guidance = torch.full((B,), 3500.0),
        encoder_hidden_states = torch.randn(B, 5, 16, generator = g),
        encoder_attention_mask = torch.tensor(MASK),
        attention_kwargs = None,
        return_dict = False,
    )
    return m, _distilled_step(state, kw, lambda i: torch.full((B,), 1000.0 * (1 - i / 4)))


def _krea2():
    m = _need("Krea2Transformer2DModel")(
        in_channels = 8,
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        intermediate_size = 16,
        timestep_embed_dim = 8,
        text_hidden_dim = 16,
        num_text_layers = 2,
        text_num_attention_heads = 2,
        text_num_key_value_heads = 2,
        text_intermediate_size = 16,
        num_layerwise_text_blocks = 1,
        num_refiner_text_blocks = 1,
        axes_dims_rope = (4, 6, 6),
    )
    g = torch.Generator().manual_seed(5)
    image_ids = torch.zeros(4, 4, 3)
    image_ids[..., 1] = torch.arange(4)[:, None]
    image_ids[..., 2] = torch.arange(4)[None, :]
    position_ids = torch.cat(
        [torch.zeros(5, 3), image_ids.reshape(16, 3)]
    )  # Krea2Pipeline.prepare_position_ids
    state = {"x": torch.randn(B, 16, 8, generator = g)}
    kw = dict(
        encoder_hidden_states = torch.randn(B, 5, 2, 16, generator = g),
        position_ids = position_ids,
        encoder_attention_mask = torch.tensor(MASK).bool(),
        attention_kwargs = None,
        return_dict = False,
    )
    return m, _distilled_step(state, kw, lambda i: torch.full((B,), 1 - i / 4))


def _distilled_step(state, kw, timestep):
    def step(model, i):
        out = model(hidden_states = state["x"], timestep = timestep(i), **kw)[0]
        state["x"] = state["x"] - 0.1 * out
        return state["x"].flatten()

    return step


CASES = {
    "LTX2VideoTransformer3DModel": _ltx2,
    "ChromaTransformer2DModel": _chroma,
    "QwenImageTransformer2DModel": _qwen,
    "HunyuanImageTransformer2DModel": _hunyuan,
    "Krea2Transformer2DModel": _krea2,
}
STEPS = 4


def _render(build, routed):
    torch.manual_seed(0)
    model, step = build()
    model.eval()
    router = built = None
    if routed:
        router, built = _router(model)
    with torch.no_grad():
        out = [step(model, i) for i in range(STEPS)][-1]
    return out, model, router, built


@pytest.mark.parametrize("name", sorted(CASES))
def test_new_class_batch_splits_and_matches_one_card(name, monkeypatch):
    monkeypatch.delenv(ALL_ENV, raising = False)
    ref, _m, _r, _b = _render(CASES[name], routed = False)
    out, model, router, built = _render(CASES[name], routed = True)
    assert type(model).__name__ == name
    assert router.stats["batch_splits"] == STEPS - 1, router.stats
    assert len(built) == 1
    torch.testing.assert_close(out, ref, rtol = 1e-4, atol = 1e-5)


@pytest.mark.parametrize("name", sorted(CASES))
def test_kill_switch_keeps_the_new_classes_whole(name, monkeypatch):
    monkeypatch.setenv(ALL_ENV, "0")
    ref, _m, _r, _b = _render(CASES[name], routed = False)
    out, _model, router, built = _render(CASES[name], routed = True)
    assert router.stats["batch_splits"] == 0 and not built
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    monkeypatch.delenv(ALL_ENV)
    _out, _model, router, built = _render(CASES[name], routed = True)
    assert router.stats["batch_splits"] == STEPS - 1 and len(built) == 1


def _identity_class(name):
    return type(name, (torch.nn.Module,), {"forward": lambda self, **k: (k["hidden_states"] * 2,)})


def _distilled_calls(model, steps = 3):
    x = torch.ones(2, 3)
    for i in range(steps):
        x = x - 0.1 * model(hidden_states = x, timestep = torch.ones(2) * i, return_dict = False)[0]
    return x


def test_existing_classes_are_unchanged_by_the_kill_switch(monkeypatch):
    for value in ("0", "1"):
        monkeypatch.setenv(ALL_ENV, value)
        for name in (
            "FluxTransformer2DModel",
            "Flux2Transformer2DModel",
            "Lumina2Transformer2DModel",
        ):
            model = _identity_class(name)()
            router, built = _router(model)
            _distilled_calls(model)
            assert router.stats["batch_splits"] == 2 and len(built) == 1, (name, value)
        # never added: HiDream batches its llama states on dim 1
        model = _identity_class("HiDreamImageTransformer2DModel")()
        router, built = _router(model)
        _distilled_calls(model)
        assert router.stats["batch_splits"] == 0 and not built


def test_attention_kwargs_keep_a_new_class_whole(monkeypatch):
    monkeypatch.delenv(ALL_ENV, raising = False)
    torch.manual_seed(0)
    model = type(
        "QwenImageTransformer2DModel",
        (torch.nn.Module,),
        {"forward": lambda self, **k: (k["hidden_states"],)},
    )()
    router, built = _router(model)
    x = torch.ones(2, 3)
    for i in range(3):
        model(hidden_states = x, timestep = torch.ones(2) * i, attention_kwargs = {"scale": 0.5})
        x = x + 1
    assert router.stats["batch_splits"] == 0 and not built
    assert router.stats["batch_split_skip"] == "attention kwargs"


class Cosmos3OmniTransformer(torch.nn.Module):
    """Shaped like diffusers' Cosmos 3 call: packed per-branch ids, the noisy tokens as a one-element list."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(8, 8)
        self.e = torch.nn.Embedding(32, 8)

    def forward(
        self,
        input_ids,
        vision_tokens,
        vision_timesteps,
        return_dict = True,
    ):
        ctx = self.e(input_ids).mean(0)
        preds = [self.w(vision_tokens[0]) * vision_timesteps[:, None] + ctx]
        return preds, None, None


def _cosmos3_render(model, steps = 5):
    g = torch.Generator().manual_seed(7)
    latents = torch.randn(6, 8, generator = g)
    cond_ids = torch.randint(0, 32, (9,), generator = g)
    uncond_ids = torch.randint(
        0, 32, (4,), generator = g
    )  # the uncond pack is shorter, as in the pipeline
    for i in range(steps):
        vision = latents.to(
            dtype = torch.float32
        )  # same object: already fp32 (pipeline: latents.to(dtype))
        ts = torch.full((6,), 1.0 - i / steps)
        cond = model(
            input_ids = cond_ids, vision_tokens = [vision], vision_timesteps = ts, return_dict = False
        )[0][0]
        uncond = model(
            input_ids = uncond_ids, vision_tokens = [vision], vision_timesteps = ts, return_dict = False
        )[0][0]
        latents = latents - 0.1 * (uncond + 6.0 * (cond - uncond))
    return latents


def test_cosmos3_vision_tokens_pair_on_the_replica_and_match_one_card(monkeypatch):
    monkeypatch.delenv(ALL_ENV, raising = False)
    torch.manual_seed(0)
    ref = _cosmos3_render(Cosmos3OmniTransformer())
    torch.manual_seed(0)
    model = Cosmos3OmniTransformer()
    router, built = _router(model)
    with torch.no_grad():
        out = _cosmos3_render(model)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    assert len(built) == 1 and router.stats["replica_calls"] >= 4, router.stats
    monkeypatch.setenv(ALL_ENV, "0")
    torch.manual_seed(0)
    model = Cosmos3OmniTransformer()
    router, built = _router(model)
    with torch.no_grad():
        out = _cosmos3_render(model)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    assert not built and router.stats["replica_calls"] == 0


def test_one_card_selection_arms_nothing_and_two_cards_split(monkeypatch):
    monkeypatch.delenv(ALL_ENV, raising = False)
    torch.manual_seed(0)
    model, _step = _chroma()
    forward = model.forward
    pipe = types.SimpleNamespace(transformer = model)
    assert mg.install_cfg_parallel(pipe, offload_policy = "none") == "one card selected"
    assert model.forward == forward and "forward" not in model.__dict__
    # the same class through the router (what a two-card selection installs) splits
    _out, _model, router, built = _render(_chroma, routed = True)
    assert router.stats["batch_splits"] == STEPS - 1 and len(built) == 1

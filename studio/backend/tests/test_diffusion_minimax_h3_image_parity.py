# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image against the reference implementation's process, on CPU with the tiny random release: reference
sizing (resize to the 32 grid, never crop), every random draw on CPU generators seeded from the request seed (initial
noise, the seed-42 reference posterior, the per-reference augmentation noise in patch-row layout), the checkpoint's
rope frequencies, the uncapped prompt and the empty-prompt pad token, and two exact speedups (the text refiner run once
per render, the output head on the generated rows only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope = "module")
def h3i():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    try:
        from diffusers import DiffusionPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a broken diffusers / hub pair
        pytest.skip(f"diffusers pipelines unavailable: {exc}")
    from core.inference import diffusion_minimax_h3_image

    return diffusion_minimax_h3_image


@pytest.fixture(scope = "module")
def release(h3i, tmp_path_factory):
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_parity_release")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    return root, ref, te


@pytest.fixture(scope = "module")
def pipe(h3i, release):
    root, _ref, te = release
    return h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)


def _picture(w, h, seed):
    import numpy as np
    from PIL import Image
    return Image.fromarray((np.random.RandomState(seed).rand(h, w, 3) * 255).astype("uint8"))


def _reference_patchify(latent):
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw] with patch (1, 2, 2), as the reference implementation packs rows."""
    b, c, t, h, w = latent.shape
    x = latent.reshape(b, c, t, 1, h // 2, 2, w // 2, 2)
    import torch

    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * (h // 2) * (w // 2), c * 4)


def _first_forward(pipe, request):
    """Every transformer call's kwargs, read by a pre-hook (the forward itself stays in place)."""
    seen = []

    def spy(_module, _args, kwargs):
        seen.append({key: (v.clone() if hasattr(v, "clone") else v) for key, v in kwargs.items()})

    handle = pipe.transformer.register_forward_pre_hook(spy, with_kwargs = True)
    request.addfinalizer(handle.remove)
    return seen


def test_references_are_resized_to_the_grid_with_lanczos_never_cropped(h3i):
    from PIL import Image

    pic = _picture(1050, 1030, 0)
    vae_refs, vision_refs = h3i.MiniMaxH3ImagePipeline.prepare_references([pic])
    assert vae_refs[0].size == h3i.size_short_edge(1050, 1030, 1024, 2048) == (1056, 1024)
    assert vision_refs[0].size == h3i.size_short_edge(1050, 1030, 1024, 1024) == (1056, 1024)
    import numpy as np

    want = np.asarray(pic.resize((1056, 1024), resample = Image.LANCZOS))
    assert np.array_equal(np.asarray(vae_refs[0]), want)
    assert np.array_equal(np.asarray(vision_refs[0]), want)
    # already on the grid and in range: the very same pixels
    on_grid = _picture(1536, 1024, 1)
    vae_refs, vision_refs = h3i.MiniMaxH3ImagePipeline.prepare_references([on_grid])
    assert np.array_equal(np.asarray(vae_refs[0]), np.asarray(on_grid))
    assert vision_refs[0].size == (1536, 1024)
    # a big picture: VAE short edge clamped to 2048, encoder copy to 1024
    vae_refs, vision_refs = h3i.MiniMaxH3ImagePipeline.prepare_references([_picture(3001, 2201, 2)])
    assert vae_refs[0].size == (2784, 2048) and vision_refs[0].size == (1408, 1024)


def test_the_initial_noise_is_the_seeded_cpu_stream(h3i, pipe, monkeypatch, request):
    import torch

    seen = _first_forward(pipe, request)
    seed = 1234
    gen = torch.Generator().manual_seed(seed)
    torch.randn(
        5, generator = gen
    )  # a generator already drawn from: the seed decides, not the stream position
    pipe(
        prompt = "a cat",
        height = 256,
        width = 256,
        generator = gen,
        min_generation_area = 1024,
        output_type = "latent",
    )
    with torch.random.fork_rng(devices = []):
        g = torch.manual_seed(seed)
        noise = torch.randn([1, 24, 1, 16, 16], dtype = torch.float32, generator = g, device = "cpu")
    rows = seen[0]["hidden_states"][0]
    assert torch.equal(rows, _reference_patchify(noise))
    assert pipe._unsloth_h3_last_stats["seed"] == seed


def test_images_sharing_one_generator_take_successive_batch_noise(h3i, pipe, monkeypatch, request):
    """num_images_per_prompt > 1 with one generator: image i starts from slice i of one seeded batch draw, so the
    images differ and each is reproducible."""
    import torch

    seen = _first_forward(pipe, request)
    seed = 77
    pipe(
        prompt = "a cat",
        height = 256,
        width = 256,
        num_images_per_prompt = 2,
        generator = torch.Generator().manual_seed(seed),
        min_generation_area = 1024,
        output_type = "latent",
    )
    with torch.random.fork_rng(devices = []):
        g = torch.manual_seed(seed)
        batch = torch.randn([2, 24, 1, 16, 16], dtype = torch.float32, generator = g, device = "cpu")
    assert len(seen) % 2 == 0 and seen
    first, second = seen[0]["hidden_states"][0], seen[len(seen) // 2]["hidden_states"][0]
    assert torch.equal(first, _reference_patchify(batch[0:1]))
    assert torch.equal(second, _reference_patchify(batch[1:2]))


def test_reference_rows_use_the_seed42_posterior_and_per_reference_restarted_noise(
    h3i, pipe, monkeypatch, request
):
    import numpy as np
    import torch

    if pipe.processor is None:
        pytest.skip("no Qwen3-VL processor in this transformers")
    monkeypatch.setenv(h3i.REF_CACHE_ENV, "0")
    seen = _first_forward(pipe, request)
    seed = 77
    pics = [_picture(70, 50, 3), _picture(60, 60, 4)]
    pipe(
        prompt = "<Picture 1> and <Picture 2>",
        image = pics,
        height = 256,
        width = 256,
        generator = torch.Generator().manual_seed(seed),
        min_generation_area = 1024,
        output_type = "latent",
    )
    vae_refs, _ = h3i.MiniMaxH3ImagePipeline.prepare_references(pics)
    vae = pipe.vae
    mean = torch.tensor(h3i.H3_IMAGE_PIXEL_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(h3i.H3_IMAGE_PIXEL_STD).view(1, -1, 1, 1)
    lmean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    lstd = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    expected = []
    for ref in vae_refs:
        pixels = torch.from_numpy(np.array(ref)).permute(2, 0, 1)[None].float().div(255.0)
        with torch.no_grad():
            moments = vae.encode(
                ((pixels - mean) / std).to(h3i._module_dtype(vae)), return_dict = False
            )[0]
        m, logvar = moments.parameters.float()[:, :, None].chunk(2, dim = 1)
        noise = torch.randn(m.shape, generator = torch.Generator().manual_seed(42))
        z = (torch.addcmul(m, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise) - lmean) / lstd
        r = _reference_patchify(z.to(torch.float32))
        aug_noise = torch.randn(
            r.shape, generator = torch.Generator("cpu").manual_seed(seed), dtype = torch.float32
        )
        expected.append(0.999 * r + (1.0 - 0.999) * aug_noise)
    expected = torch.cat(expected)
    rows = seen[0]["hidden_states"][0]
    assert torch.equal(rows[: expected.shape[0]], expected)
    # a different request seed changes the augmentation noise, never the posterior sample
    seen.clear()
    pipe(
        prompt = "<Picture 1> and <Picture 2>",
        image = pics,
        height = 256,
        width = 256,
        generator = torch.Generator().manual_seed(seed + 1),
        min_generation_area = 1024,
        output_type = "latent",
    )
    other = seen[0]["hidden_states"][0][: expected.shape[0]]
    assert not torch.equal(other, expected)
    assert torch.allclose(other, expected, atol = 0.02)


def test_the_cached_reference_latent_is_bitwise_the_uncached_one(h3i, release, monkeypatch):
    import torch

    root, _ref, te = release
    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    if p.processor is None:
        pytest.skip("no Qwen3-VL processor in this transformers")
    calls = []
    real = p.vae.encode
    monkeypatch.setattr(p.vae, "encode", lambda *a, **k: calls.append(1) or real(*a, **k))
    pic = _picture(90, 70, 5)
    kw = dict(height = 256, width = 256, min_generation_area = 1024, output_type = "latent", image = [pic])
    a = p(prompt = "<Picture 1> at night", generator = torch.Generator().manual_seed(3), **kw).images[0]
    b = p(prompt = "<Picture 1> at noon", generator = torch.Generator().manual_seed(4), **kw).images[0]
    assert len(calls) == 1
    monkeypatch.setenv(h3i.REF_CACHE_ENV, "0")
    c = p(prompt = "<Picture 1> at noon", generator = torch.Generator().manual_seed(4), **kw).images[0]
    assert len(calls) == 2 and torch.equal(b, c) and not torch.equal(a, b)
    p.release()


def test_the_checkpoint_inv_freq_is_checked_and_wins_when_it_differs(h3i, release):
    import torch

    root, _ref, _te = release
    model, stats = h3i.load_h3_image_transformer(
        root / "", dtype = torch.float32, scheme = None, device = "cpu", keep_on_device = True
    )
    assert stats["rope_inv_freq"] == "match"
    computed = model.rope.inv_freq.clone()
    assert h3i.apply_checkpoint_inv_freq(model, computed.clone()) == "match"
    other = computed * 1.5
    assert h3i.apply_checkpoint_inv_freq(model, other) == "checkpoint"
    assert torch.equal(model.rope.inv_freq, other)
    assert "inv_freq" in dict(model.rope.named_buffers())
    with pytest.raises(ValueError, match = "frequencies"):
        h3i.apply_checkpoint_inv_freq(model, torch.ones(computed.numel() + 1))


def test_the_prompt_is_not_truncated_and_an_empty_one_is_the_pad_token(h3i, pipe, monkeypatch):
    import torch

    seen = {}
    real = h3i.qwen3vl_hidden_state

    class _Stop(Exception):
        pass

    def spy(encoder, processor, token_ids, vision_inputs, **kw):
        seen["ids"] = list(token_ids)
        if len(token_ids) > 4096:
            raise _Stop
        return real(encoder, processor, token_ids, vision_inputs, **kw)

    monkeypatch.setattr(h3i, "qwen3vl_hidden_state", spy)
    with pytest.raises(_Stop):
        pipe(prompt = " ".join(["cat"] * 5000), height = 64, width = 64, min_generation_area = 1024)
    assert len(seen["ids"]) > 5000
    out = pipe(
        prompt = "",
        height = 64,
        width = 64,
        min_generation_area = 1024,
        generator = torch.Generator().manual_seed(0),
    )
    assert out.images[0].size == (64, 64)
    assert seen["ids"] == [pipe.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    if pipe.processor is None:
        return
    pipe(
        prompt = "",
        image = [_picture(40, 40, 6)],
        height = 64,
        width = 64,
        min_generation_area = 1024,
        generator = torch.Generator().manual_seed(0),
    )
    end = pipe.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    assert seen["ids"][-1] == end


def test_the_encoder_copy_reaches_the_vision_tower_unresampled(h3i, pipe, monkeypatch):
    if pipe.processor is None:
        pytest.skip("no Qwen3-VL processor in this transformers")
    seen = {}
    real = h3i.qwen3vl_hidden_state

    def spy(encoder, processor, token_ids, vision_inputs, **kw):
        seen["grid"] = vision_inputs["image_grid_thw"].tolist()
        return real(encoder, processor, token_ids, vision_inputs, **kw)

    monkeypatch.setattr(h3i, "qwen3vl_hidden_state", spy)
    import torch

    pipe(
        prompt = "<Picture 1>",
        image = [_picture(50, 70, 7)],
        height = 64,
        width = 64,
        min_generation_area = 1024,
        generator = torch.Generator().manual_seed(0),
    )
    w, h = h3i.size_short_edge(50, 70, 1024, 1024)
    assert seen["grid"] == [[1, h // 16, w // 16]]


def test_the_refiner_runs_once_per_render_and_matches_every_step_bitwise(h3i, release, monkeypatch):
    import torch

    root, _ref, te = release
    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    assert h3i.h3_image_forward_active(p.transformer)
    calls = []
    hook = p.transformer.token_refiner.register_forward_hook(lambda *a: calls.append(1))
    image = [_picture(60, 40, 8)] if p.processor is not None else None
    prompt = "<Picture 1> a cat" if image else "a cat"
    kw = dict(height = 256, width = 256, min_generation_area = 1024, output_type = "latent", image = image)
    cached = p(prompt = prompt, generator = torch.Generator().manual_seed(9), **kw).images[0]
    assert len(calls) == 1 and p._unsloth_h3_last_stats["refiner_cached"]
    monkeypatch.setenv(h3i.REFINER_CACHE_ENV, "0")
    stock = p(prompt = prompt, generator = torch.Generator().manual_seed(9), **kw).images[0]
    assert len(calls) == 1 + 6 and not p._unsloth_h3_last_stats["refiner_cached"]
    assert torch.equal(cached, stock)
    # a wrapper around the forward (a graph or offload hook) gets the stock call: still the same latents
    monkeypatch.delenv(h3i.REFINER_CACHE_ENV)
    real = p.transformer.forward
    monkeypatch.setattr(p.transformer, "forward", lambda *a, **k: real(*a, **k))
    assert not h3i.h3_image_forward_active(p.transformer)
    wrapped = p(prompt = prompt, generator = torch.Generator().manual_seed(9), **kw).images[0]
    assert not p._unsloth_h3_last_stats["refiner_cached"] and torch.equal(wrapped, stock)
    hook.remove()
    p.release()


def test_the_head_on_the_generated_rows_equals_the_head_over_every_image_row(
    h3i, pipe, monkeypatch, request
):
    import torch

    seen = _first_forward(pipe, request)
    image = [_picture(60, 40, 9)] if pipe.processor is not None else None
    prompt = "<Picture 1> a cat" if image else "a cat"
    render = dict(
        prompt = prompt,
        image = image,
        height = 256,
        width = 256,
        min_generation_area = 1024,
        output_type = "latent",
    )
    # off by default: the stock head over every image row
    default = pipe(generator = torch.Generator().manual_seed(4), **render).images[0]
    assert "output_indices" not in seen[0]
    seen.clear()
    monkeypatch.setenv(h3i.TARGET_HEAD_ENV, "1")
    opted = pipe(generator = torch.Generator().manual_seed(4), **render).images[0]
    assert torch.equal(opted, default)  # exact on CPU; on a GPU it can differ at rounding level
    kw = dict(seen[0])
    n_target = 16 * 16 // 4
    target = kw.pop("output_indices")
    kw.pop("text_embeds", None)
    assert target.numel() == n_target and int(target[-1]) == kw["position_ids"].shape[0] - 1
    with torch.no_grad():
        stock = type(pipe.transformer).forward(pipe.transformer, **kw)[0]
        head = pipe.transformer(**kw, output_indices = target)[0]
        both = pipe.transformer(
            **kw,
            output_indices = target,
            text_embeds = h3i.refined_text(pipe.transformer, kw["encoder_hidden_states"]),
        )[0]
    assert stock.shape[1] == kw["image_indices"].numel() and head.shape[1] == n_target
    assert torch.equal(stock[:, -n_target:], head)
    assert torch.equal(head, both)

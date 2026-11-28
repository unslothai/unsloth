# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image against the reference implementation's process, on CPU with the tiny random release: reference
sizing (resize to the 32 grid, never crop), every random draw on CPU generators seeded from the request seed (initial
noise, the seed-42 reference posterior, the per-reference augmentation noise in patch-row layout), the checkpoint's
rope frequencies, the uncapped prompt and the empty-prompt pad token, one exact speedup (the text refiner run once per
render), and the reference defaults: the "simple" scheduler's sigmas and the model's float32 timesteps, AdaLN rows
interpolated on the curve knot grid, the PDD head bank (any step count from 1 to 6), the Euler step on the denoised
estimate with the model's input and output in its compute dtype, native size, the edit workflow's 4 MP Picture 1,
the vision tower's own resize above its pixel bound, and the head on the generated rows only. The reference formulas
below are transcribed from the reference implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse = True)
def _picture_one_at_its_own_size(monkeypatch):
    """Picture 1 keeps its own size here (the reference edit workflow's 4 MP rescale would make every CPU edit render
    a 4 MP reference); the tests of that rescale turn it back on."""
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")


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
    # the reference casts the model input to its compute dtype
    assert torch.equal(rows, _reference_patchify(noise).to(torch.bfloat16).float())
    assert pipe._unsloth_h3_last_stats["seed"] == seed


def test_images_sharing_one_generator_take_successive_batch_noise(h3i, pipe, monkeypatch, request):
    """Images sharing one generator take successive slices of one seeded noise draw."""
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
    assert seen
    rows = seen[0]["hidden_states"]
    if rows.shape[0] == 2:  # one batched forward over both images
        first, second = rows[0], rows[1]
    else:
        assert len(seen) % 2 == 0
        first, second = rows[0], seen[len(seen) // 2]["hidden_states"][0]
    # the model input is in its compute dtype, as the reference casts it
    assert torch.equal(first, _reference_patchify(batch[0:1]).to(torch.bfloat16).float())
    assert torch.equal(second, _reference_patchify(batch[1:2]).to(torch.bfloat16).float())


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
        # the reference VAE input: 2x - 1, then (x + 1) * 0.5 normalized (float32 off CUDA)
        x = pixels * 2.0 - 1.0
        with torch.no_grad():
            moments = vae.encode(x.add(1.0).mul_(0.5).sub_(mean).div_(std), return_dict = False)[0]
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


def test_the_head_runs_on_the_generated_rows_by_default_and_equals_the_head_over_every_image_row(
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
    # on by default, as the reference's final layer: the head on the generated rows only
    default = pipe(generator = torch.Generator().manual_seed(4), **render).images[0]
    kw = dict(seen[0])
    seen.clear()
    monkeypatch.setenv(h3i.TARGET_HEAD_ENV, "0")
    stock_rows = pipe(generator = torch.Generator().manual_seed(4), **render).images[0]
    assert "output_indices" not in seen[0]
    assert torch.equal(
        stock_rows, default
    )  # exact on CPU; on a GPU it can differ at rounding level
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


# The reference sampler, transcribed: model sampling linear in the grid index over 1000 timesteps, the "simple"
# scheduler, the model's timestep = sigma * 1000 (float32) read back in double, and the PDD head bank.
def _reference_sigmas(grid, steps):
    import torch

    g = torch.tensor(list(grid), dtype = torch.float32).flip(0)
    n = g.shape[0] - 1
    ts = (torch.arange(1, 1001, 1) / 1000) * 1000
    pos = (ts / 1000).clamp(0.0, 1.0) * n
    i = pos.floor().long().clamp(max = n - 1)
    sigmas = torch.lerp(g[i], g[i + 1], pos - i)
    ss = len(sigmas) / steps
    return torch.FloatTensor([float(sigmas[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


def _reference_model_sigma(sigma):
    import torch
    ts = (sigma * torch.ones([1]) * 1000).float()
    return float(ts.flatten()[0]) / 1000.0


def _reference_pdd_head(bank_w, bank_b, grid, sigma, sample_sigmas, h):
    import torch

    n = len(grid) - 1
    i = int((sample_sigmas - sigma).abs().argmin())
    sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
    start, stop = (
        min(range(n + 1), key = lambda k: abs(grid[k] - float(s))) for s in (sigma, sigma_next)
    )
    start = min(start, n - 1)
    stop = max(stop, start + 1)
    dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
    w = torch.tensor(dt, dtype = h.dtype) / sum(dt)
    rows = bank_w.reshape(len(grid) - 1, -1, bank_w.shape[1])
    brows = bank_b.reshape(len(grid) - 1, -1)
    first = max(start, 1)
    return torch.nn.functional.linear(
        h,
        rows[0] + torch.einsum("n,noi->oi", w[first - start :], rows[first:stop]),
        brows[0] + torch.einsum("n,no->o", w[first - start :], brows[first:stop]),
    )


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 6])
def test_sigmas_and_timesteps_are_the_reference_simple_scheduler(h3i, steps):
    import torch

    grid = list(h3i.H3_IMAGE_SIGMAS)
    want = _reference_sigmas(grid, steps)
    s = h3i.MiniMaxH3ImageScheduler(sample_sigmas = grid)
    s.set_timesteps(steps)
    assert torch.equal(s.sigmas, want) and torch.equal(h3i.reference_sigmas(steps), want)
    model = [_reference_model_sigma(x) for x in want[:-1]]
    assert s.model_sigmas == model
    assert torch.equal(s.timesteps, torch.tensor([1.0 - m for m in model], dtype = torch.float32))
    # every timestep and reference row level has a row in the AdaLN table
    table = h3i.table_timesteps()
    assert all(float(t) in table for t in s.timesteps)
    assert float(torch.tensor(0.999, dtype = torch.float32)) in table
    if steps == 6:
        # up to 9e-4 off the distilled grid, as in the reference
        off = (want - torch.tensor(grid)).abs().max()
        assert 5e-4 < float(off) < 1e-3


def test_the_euler_step_is_the_reference_denoised_form(h3i):
    import torch

    s = h3i.MiniMaxH3ImageScheduler(sample_sigmas = list(h3i.H3_IMAGE_SIGMAS))
    s.set_timesteps(6)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 96, generator = g)
    ref = x.clone()
    for i in range(6):
        v = torch.randn(40, 96, generator = g).to(torch.bfloat16).float()
        x = s.step(v, s.timesteps[i], x)[0]
        sigma, sigma_next = s.sigmas[i], s.sigmas[i + 1]
        out = -v  # the reference model returns minus the head
        denoised = ref - out * sigma
        d = (ref - denoised) / sigma
        ref = ref + d * (sigma_next - sigma)
        assert torch.equal(x, ref)


@pytest.mark.parametrize("steps", [1, 3, 4, 6])
def test_the_head_bank_is_the_reference_dt_weighted_mean(h3i, steps):
    import torch

    g = torch.Generator().manual_seed(steps)
    fused_w = torch.randn(6, 96, 40, generator = g)
    fused_b = torch.randn(6, 96, generator = g)
    module = torch.nn.Module()
    module.pdd_proj_out = torch.nn.ModuleList([torch.nn.Linear(40, 96) for _ in range(6)])
    with torch.no_grad():
        for k in range(6):
            module.pdd_proj_out[k].weight.copy_(fused_w[k])
            module.pdd_proj_out[k].bias.copy_(fused_b[k])
    assert h3i.install_pdd_head_bank(module)
    # the reference bank: head 0, then offsets
    bank_w = torch.cat([fused_w[0]] + [fused_w[k] - fused_w[0] for k in range(1, 6)])
    bank_b = torch.cat([fused_b[0]] + [fused_b[k] - fused_b[0] for k in range(1, 6)])
    bank = getattr(module, h3i.PDD_BANK_ATTR)
    assert torch.equal(bank[0].reshape(-1, 40), bank_w) and torch.equal(bank[1].reshape(-1), bank_b)
    s = h3i.MiniMaxH3ImageScheduler(sample_sigmas = list(h3i.H3_IMAGE_SIGMAS))
    s.set_timesteps(steps)
    h = torch.randn(7, 40, generator = g)
    for sigma in s.model_sigmas:
        want = _reference_pdd_head(bank_w, bank_b, s.head_grid, sigma, s.sigmas, h)
        w, b = h3i.pdd_head_weights(bank, sigma, s.sigmas, s.head_grid)
        assert torch.equal(torch.nn.functional.linear(h, w, b), want)
        start, stop = h3i.pdd_head_span(sigma, s.sigmas, s.head_grid)
        if stop - start == 1:
            # a one-interval step runs pdd_proj_out[start], which now is the reference's head
            assert torch.equal(module.pdd_proj_out[start](h), want)


def test_step_counts_one_to_six_render_and_blend_heads(h3i, pipe, request):
    import torch

    seen = _first_forward(pipe, request)
    kw = dict(prompt = "a cat", height = 64, width = 64, output_type = "latent")
    for steps in (3, 4, 1):
        seen.clear()
        pipe(num_inference_steps = steps, generator = torch.Generator().manual_seed(1), **kw)
        stats = pipe._unsloth_h3_last_stats
        assert len(seen) == steps and stats["steps"] == steps
        # 3 and 4 steps span several grid intervals in some steps: those take the blended head
        assert stats["pdd_blended_steps"] >= 1 and any("head" in call for call in seen)
    with pytest.raises(ValueError, match = "1 to 6 steps"):
        pipe(num_inference_steps = 7, **kw)
    with pytest.raises(ValueError, match = "1 to 6 steps"):
        pipe(num_inference_steps = 0, **kw)


def test_dense_adaln_rows_interpolate_on_the_knot_grid(h3i, release):
    import copy

    import torch

    _root, ref, _te = release
    dense = ref["transformer"]
    table = copy.deepcopy(dense)
    h3i.convert_dense_transformer_to_table(table)
    ts = h3i.table_timesteps()
    assert list(table._unsloth_adaln_table) == ts
    knots = h3i.H3_IMAGE_ADALN_KNOTS
    t = torch.tensor(ts, dtype = torch.float32)
    pos = t.clamp(0.0, 1.0) * (knots - 1)
    i0 = pos.floor().long().clamp(max = knots - 2)
    frac = (pos - i0).unsqueeze(1)
    with torch.no_grad():
        lo = dense.time_embedder(dense.time_proj(i0.float() / (knots - 1)).float())
        hi = dense.time_embedder(dense.time_proj((i0 + 1).float() / (knots - 1)).float())
        lin = dense.transformer_blocks[0].adaln_proj.linear
        project = lambda e: torch.nn.functional.linear(
            torch.nn.functional.silu(e.float()), lin.weight.float(), lin.bias.float()
        )
        want = torch.lerp(project(lo), project(hi), frac).to(lin.weight.dtype)
        onehot = torch.eye(len(ts), dtype = torch.bfloat16)
        got = table.transformer_blocks[0].adaln_proj.linear(onehot)
        # the knot rows of a timestep on a knot are exact, an off-knot one blends two knot rows
        assert torch.allclose(got.float(), want.float(), atol = 1e-2, rtol = 1e-2)
        out_lin = table.norm_out.linear
        assert out_lin.weight.dtype == torch.float32  # the final modulation stays float32
    with pytest.raises(ValueError, match = "AdaLN table"):
        table.time_embedder(torch.tensor([0.5]))


def test_a_request_renders_natively_without_the_canvas_floor(h3i, pipe, request):
    seen = _first_forward(pipe, request)
    import torch

    pipe(
        prompt = "a cat",
        height = 96,
        width = 160,
        output_type = "latent",
        generator = torch.Generator().manual_seed(0),
    )
    assert pipe._unsloth_h3_last_stats["canvas"] == (160, 96)
    assert seen[0]["hidden_states"].shape[1] == (96 // 32) * (160 // 32)
    with pytest.raises(ValueError, match = "multiples of 32"):
        pipe(prompt = "a cat", height = 100, width = 160)
    # the floor stays available on request
    pipe(prompt = "a cat", height = 64, width = 64, min_generation_area = 128 * 128, output_type = "latent")
    assert pipe._unsloth_h3_last_stats["canvas"] == (128, 128)


def _reference_source(pic):
    """The reference edit workflow on Picture 1: LoadImage, scale total pixels (4 MP, area), scale to multiple 32
    (area, cover, centre crop)."""
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.array(pic).astype(np.float32) / 255.0)[None]
    total = int(4 * 1024 * 1024)
    s = math.sqrt(total / (x.shape[2] * x.shape[1]))
    w, h = round(x.shape[2] * s), round(x.shape[1] * s)
    x = F.interpolate(x.movedim(-1, 1), size = (h, w), mode = "area").movedim(1, -1)
    tw, th = (w // 32) * 32, (h // 32) * 32
    if (tw, th) != (w, h):
        s_w, s_h = tw / w, th / h
        if s_w >= s_h:
            sw, sh = tw, max(th, int(math.ceil(h * s_w)))
        else:
            sw, sh = max(tw, int(math.ceil(w * s_h))), th
        x = F.interpolate(x.movedim(-1, 1), size = (sh, sw), mode = "area").movedim(1, -1)
        x0, y0 = (sw - tw) // 2, (sh - th) // 2
        x = x[:, y0 : y0 + th, x0 : x0 + tw, :]
    return x[0].numpy()


def test_picture_one_follows_the_reference_edit_workflow(h3i, monkeypatch):
    import numpy as np

    for w, h in ((1475, 981), (600, 400), (427, 640), (1203, 1203), (1001, 749)):
        pic = _picture(w, h, w + h)
        want = _reference_source(pic)
        vae_refs, vision_refs = h3i.MiniMaxH3ImagePipeline.prepare_references(
            [pic, _picture(70, 50, 1)], source_resize = True
        )
        got = vae_refs[0]
        # kept float (no 8-bit rounding) when it is already at its VAE size
        assert isinstance(got, np.ndarray) and np.array_equal(got, want), (w, h)
        assert (
            h3i.picture_size(got) == h3i.reference_edit_size(w, h) == (want.shape[1], want.shape[0])
        )
        # the encoder copy is the LANCZOS of the 8-bit truncated picture; other pictures are untouched
        assert vision_refs[0].size == h3i.size_short_edge(want.shape[1], want.shape[0], 1024, 1024)
        assert vae_refs[1].size == h3i.size_short_edge(70, 50, 1024, 2048)
    # on by default; "0" keeps the conditioning node's rule only
    monkeypatch.delenv(h3i.SOURCE_RESIZE_ENV)
    vae_refs, _ = h3i.MiniMaxH3ImagePipeline.prepare_references([_picture(1475, 981, 3)])
    assert h3i.picture_size(vae_refs[0]) == (2496, 1664)
    monkeypatch.setenv(h3i.SOURCE_RESIZE_ENV, "0")
    vae_refs, _ = h3i.MiniMaxH3ImagePipeline.prepare_references([_picture(1475, 981, 3)])
    assert vae_refs[0].size == h3i.size_short_edge(1475, 981, 1024, 2048)


def _reference_vision(pic):
    """The reference Qwen3-VL preprocessing: float picture, bilinear to the bounded size, normalize, patchify."""
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F

    img = torch.from_numpy(np.array(pic).astype(np.float32) / 255.0)[None].permute(0, 3, 1, 2)[0]
    height, width = img.shape[1], img.shape[2]
    factor, lo, hi = 32, 3136, 12845056
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > hi:
        beta = math.sqrt((height * width) / hi)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < lo:
        beta = math.sqrt(lo / (height * width))
        h_bar, w_bar = (
            math.ceil(height * beta / factor) * factor,
            math.ceil(width * beta / factor) * factor,
        )
    img = F.interpolate(
        img.unsqueeze(0), size = (h_bar, w_bar), mode = "bilinear", align_corners = False
    ).squeeze(0)
    norm = img.clone()
    for c in range(3):
        norm[c] = (img[c] - 0.5) / 0.5
    x = norm.unsqueeze(0).repeat(2, 1, 1, 1)
    gt, gh, gw = 1, h_bar // 16, w_bar // 16
    x = x.reshape(gt, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gt * gh * gw, 3 * 2 * 16 * 16), [[gt, gh, gw]]


@pytest.mark.parametrize("size", [(1536, 1024), (14336, 1024), (4096, 3328)])
def test_vision_pixels_follow_the_reference_including_above_the_pixel_bound(h3i, size):
    import torch

    pic = _picture(*size, 5)
    want, grid = _reference_vision(pic)
    got, got_grid = h3i.vision_pixel_values([pic], 16, 2, 2)
    assert torch.equal(got, want) and got_grid.tolist() == grid
    if size[0] * size[1] > h3i.H3_IMAGE_VISION_MAX_PIXELS:
        assert grid[0][1] * grid[0][2] * 256 <= h3i.H3_IMAGE_VISION_MAX_PIXELS


def test_an_edit_with_no_size_renders_at_picture_one_at_4mp(h3i, pipe, monkeypatch):
    seen = {}

    def spy(self, prompt, vae_refs, vision_refs, height, width, *a, **k):
        seen["size"] = (width, height)
        return None

    monkeypatch.setattr(type(pipe), "_render_one", spy)
    monkeypatch.delenv(h3i.SOURCE_RESIZE_ENV)
    pipe(prompt = "<Picture 1>", image = [_picture(300, 200, 2)])
    assert seen["size"] == h3i.reference_edit_size(300, 200) == (2496, 1664)
    pipe(prompt = "a cat")
    assert seen["size"] == (2048, 2048)


def test_prepared_pictures_are_cached_per_source_picture(h3i):
    import hashlib

    import numpy as np

    pic = _picture(600, 400, 11)
    first = h3i.MiniMaxH3ImagePipeline.prepare_references([pic], source_resize = True)
    again = h3i.MiniMaxH3ImagePipeline.prepare_references([pic.copy()], source_resize = True)
    # the same content reuses the prepared copies: no rescale, no resize, no rehash of the float picture
    assert again[0][0] is first[0][0] and again[1][0] is first[1][0]
    vae = first[0][0]
    assert vae.digest == hashlib.blake2b(np.asarray(vae).tobytes(), digest_size = 16).hexdigest()
    # the rule is part of the key
    plain = h3i.MiniMaxH3ImagePipeline.prepare_references([pic], source_resize = False)
    assert plain[0][0].size == h3i.size_short_edge(600, 400, 1024, 2048)


def test_the_vae_encodes_in_the_reference_working_dtype(h3i):
    import torch

    # float16 on CUDA, float32 on CPU or with the opt-out; normalized after the cast, in that dtype
    assert h3i.reference_encode_dtype("cuda") == torch.float16
    assert h3i.reference_encode_dtype("cpu") == torch.float32
    pic = torch.rand(1, 64, 96, 3, generator = torch.Generator().manual_seed(5))
    mean = torch.tensor(h3i.H3_IMAGE_PIXEL_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(h3i.H3_IMAGE_PIXEL_STD).view(1, -1, 1, 1)
    for dt in (torch.float16, torch.float32):
        x = (pic * 2.0 - 1.0).to(dt).permute(0, 3, 1, 2)
        ref = x.add(1.0).mul_(0.5).sub_(mean.to(x)).div_(std.to(x))
        mine = h3i.reference_vae_pixels(pic, dt, mean, std)
        assert mine.dtype == dt and torch.equal(mine, ref)


def test_the_vae_encode_opt_out_and_overflow_fall_back_to_float32(h3i, monkeypatch):
    import types

    import torch

    monkeypatch.setenv(h3i.VAE_ENCODE_HALF_ENV, "0")
    assert h3i.reference_encode_dtype("cuda") == torch.float32
    monkeypatch.delenv(h3i.VAE_ENCODE_HALF_ENV)
    assert (
        h3i.reference_encode_dtype("cuda", types.SimpleNamespace(_unsloth_h3_encode_fp32 = True))
        == torch.float32
    )
    vae = types.SimpleNamespace(
        encoder = torch.nn.Linear(2, 2),
        quant_conv = torch.nn.Linear(2, 2),
        decoder = torch.nn.Linear(2, 2),
    )
    h3i.set_encoder_dtype(vae, torch.float16)
    assert vae.encoder.weight.dtype == vae.quant_conv.weight.dtype == torch.float16
    assert vae.decoder.weight.dtype == torch.float32

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 PDD checkpoints: detection, the without-heads refusal, and the stacked head bank against a direct port of
ComfyUI's ``FinalLayer`` PDD math and a float64 oracle, per step (the last one included), through the real loader,
under torch.compile, a CUDA graph replaying changing sigmas, and diffusers group offload.

The model is a tiny random ``MiniMaxH3Transformer3DModel`` (no checkpoint, no network); the CUDA cases skip on a host
without a GPU."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from core.inference import video_minimax_h3_pdd as pdd  # noqa: E402

HEADS = 32
VIDEO_DIM = 96
AUDIO_DIM = 32
SHIFT_V, SHIFT_A = 12.0, 3.0


# ── a direct port of ComfyUI's MiniMax-H3 PDD head math (comfy/ldm/minimax_h3/model.py) ──


def comfy_time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def comfy_span(sigma, sample_sigmas, n, shift_v):
    i = int((sample_sigmas - sigma).abs().argmin())
    sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
    start, stop = (
        round(float(1.0 - comfy_time_shift_sigma(s, shift_v, 1.0)) * n) for s in (sigma, sigma_next)
    )
    start = min(start, n - 1)
    stop = max(stop, start + 1)
    return start, stop


def comfy_pdd_head(weight, bias, h, n, start, stop, flow_shift):
    grid = torch.linspace(1.0, 0.0, n + 1, dtype = torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = (dt / dt.sum()).to(h)
    rows = weight.to(h).reshape(n, -1, weight.shape[1])
    brows = bias.to(h).reshape(n, -1)
    first = max(start, 1)
    return torch.nn.functional.linear(
        h,
        rows[0] + torch.einsum("n,noi->oi", w[first - start :], rows[first:stop]),
        brows[0] + torch.einsum("n,no->o", w[first - start :], brows[first:stop]),
    )


def oracle_head(weight, bias, h, n, start, stop, flow_shift):
    """The same blend entirely in float64 (weights, activations and the matmul)."""
    grid = torch.linspace(1.0, 0.0, n + 1, dtype = torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = dt / dt.sum()
    rows = weight.double().reshape(n, -1, weight.shape[1])
    brows = bias.double().reshape(n, -1)
    W, B = rows[0].clone(), brows[0].clone()
    for j, k in enumerate(range(start, stop)):
        if k >= 1:
            W += w[j] * rows[k]
            B += w[j] * brows[k]
    return h.double() @ W.T + B


def comfy_simple_sigmas(steps, shift = SHIFT_V):
    """ComfyUI's ``simple`` scheduler over ModelSamplingDiscreteFlow (1000 timesteps, multiplier 1000)."""
    t = torch.arange(1, 1001, dtype = torch.float32) / 1000.0
    table = shift * t / (1 + (shift - 1) * t)
    ss = len(table) / steps
    return torch.tensor([float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


def h3_sigmas(steps, shift = SHIFT_V):
    """MiniMaxH3Scheduler.set_timesteps(steps + 1): the grid Studio drives a PDD bank with."""
    base = torch.linspace(1.0, 0.0, pdd.h3_pdd_num_inference_steps(steps), dtype = torch.float32)
    return torch.unique_consecutive(shift * base / (1 + (shift - 1) * base))


def _bank(
    in_features = 24,
    out_features = VIDEO_DIM,
    heads = HEADS,
    seed = 0,
):
    torch.manual_seed(seed)
    bank = pdd.head_bank_class()(in_features, out_features, heads)
    with torch.no_grad():
        bank.weight.normal_(0, 0.05)
        bank.bias.normal_(0, 0.05)
    return bank


def _state(
    sigma,
    sigma_next,
    select = SHIFT_V,
    stream = SHIFT_V,
    device = "cpu",
):
    return torch.tensor(
        [float(sigma), float(sigma_next), select, stream], dtype = torch.float64, device = device
    )


# ── detection and refusal ──


def _header(
    video_rows,
    audio_rows,
    metadata = None,
):
    header = {
        "video_patch_proj.weight": {"dtype": "F32", "shape": [5376, VIDEO_DIM]},
        "audio_patch_proj.weight": {"dtype": "F32", "shape": [5376, AUDIO_DIM]},
        pdd.VIDEO_HEAD_KEY: {"dtype": "F32", "shape": [video_rows, 5376]},
        pdd.AUDIO_HEAD_KEY: {"dtype": "F32", "shape": [audio_rows, 5376]},
    }
    if metadata is not None:
        header["__metadata__"] = metadata
    return header


WITH = "minimax_h3_ref2va_pruned_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors"
WITHOUT = "minimax_h3_ref2va_pdd_acc_8step_baked_without_heads_int8_convrot.safetensors"
# The without-heads file's former name: nothing in it says "without heads".
WITHOUT_OLD = "minimax_h3_ref2va_pdd_acc_8step_baked_int8_convrot.safetensors"


def test_a_with_heads_bank_is_detected_from_its_shapes_and_metadata():
    meta = {"config": json.dumps({"transformer": {"pdd_steps": 32}}), "pdd_acc_baked": "true"}
    assert pdd.h3_pdd_heads_from_header(_header(3072, 1024, meta), filename = WITH) == 32
    # shapes alone are enough
    assert pdd.h3_pdd_heads_from_header(_header(3072, 1024), filename = "opaque.safetensors") == 32
    assert (
        pdd.h3_pdd_heads_from_header(
            _header(VIDEO_DIM, AUDIO_DIM), filename = "minimax_h3_fl2va_int8_convrot.safetensors"
        )
        == 1
    )


@pytest.mark.parametrize(
    "header,name",
    [
        (_header(VIDEO_DIM, AUDIO_DIM, {"pdd_acc_baked": "true"}), WITHOUT_OLD),
        (_header(VIDEO_DIM, AUDIO_DIM, {"pdd_acc_baked": "true"}), "opaque.safetensors"),
        (_header(VIDEO_DIM, AUDIO_DIM), WITHOUT),
        (
            _header(
                VIDEO_DIM, AUDIO_DIM, {"config": json.dumps({"transformer": {"pdd_steps": 32}})}
            ),
            "x.safetensors",
        ),
    ],
)
def test_a_pdd_trunk_without_its_head_bank_is_refused_at_load(header, name):
    with pytest.raises(ValueError, match = "no head bank"):
        pdd.h3_pdd_heads_from_header(header, filename = name)


def test_an_inconsistent_bank_is_refused():
    with pytest.raises(ValueError, match = "not one stacked bank"):
        pdd.h3_pdd_heads_from_header(_header(3072, 512), filename = WITH)
    with pytest.raises(ValueError, match = "declares 16"):
        pdd.h3_pdd_heads_from_header(
            _header(3072, 1024, {"config": json.dumps({"transformer": {"pdd_steps": 16}})}),
            filename = WITH,
        )


@pytest.mark.parametrize(
    "name,match",
    [
        (WITHOUT, "no head bank"),
        ("minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors", "acceleration LoRA"),
        ("minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors", "acceleration LoRA"),
    ],
)
def test_the_family_refuses_pdd_files_it_cannot_run_by_name(name, match):
    from core.inference.video import VideoBackend
    with pytest.raises(ValueError, match = match):
        VideoBackend().validate_load_request(
            "aptech0081/MiniMax-H3-PDD-ComfyUI",
            gguf_filename = name,
            model_kind = "single_file",
            family_override = "minimax-h3",
        )


def test_the_with_heads_file_passes_the_name_refusals():
    from core.inference.video import VideoBackend
    try:
        VideoBackend().validate_load_request(
            "aptech0081/MiniMax-H3-PDD-ComfyUI",
            gguf_filename = WITH,
            model_kind = "single_file",
            family_override = "minimax-h3",
        )
    except (
        ValueError,
        ImportError,
    ) as exc:  # later probes may refuse on this host; never a PDD refusal
        assert "PDD" not in str(exc) and "single .safetensors checkpoint" not in str(exc)


def test_pdd_files_default_to_comfyuis_8_step_recipe():
    from core.inference.video_families import default_video_generation_params, detect_video_family

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert default_video_generation_params(
        None, WITH, "aptech0081/MiniMax-H3-PDD-ComfyUI", fallback = (30, 1.0), family = fam.name
    ) == (8, 1.0)
    assert default_video_generation_params(
        None,
        "minimax_h3_fl2va_int8_convrot.safetensors",
        "MiniMaxAI/MiniMax-H3",
        fallback = (30, 1.0),
        family = fam.name,
    ) == (30, 1.0)
    assert (fam.default_flow_shift, fam.default_audio_flow_shift) == (SHIFT_V, SHIFT_A)


def test_the_default_grid_is_comfyuis_simple_schedule():
    for steps in (4, 8):
        ours, comfy = h3_sigmas(steps), comfy_simple_sigmas(steps)
        assert ours.shape == comfy.shape == (steps + 1,)
        assert torch.allclose(ours, comfy, atol = 1e-6, rtol = 0)
    sched = pytest.importorskip("diffusers").MiniMaxH3Scheduler(shift = SHIFT_V)
    sched.set_timesteps(pdd.h3_pdd_num_inference_steps(8))
    assert torch.equal(sched.sigmas.cpu(), h3_sigmas(8))
    assert len(sched.timesteps) == 8  # 8 model evaluations, as in ComfyUI


# ── head selection and the blend, per step ──


@pytest.mark.parametrize("steps", [8, 4, 6, 20])
def test_head_selection_matches_comfyui_at_every_step(steps):
    sigmas = h3_sigmas(steps)
    for i in range(steps):
        sigma = sigmas[i]
        start, stop = comfy_span(sigma, sigmas, HEADS, SHIFT_V)
        state = _state(sigma, sigmas[i + 1])
        w = pdd.pdd_head_weights(state, HEADS)
        picked = [k for k in range(HEADS) if w[k] != 0]
        assert picked == [k for k in range(max(start, 1), stop)], (steps, i)
        if steps == 8:
            assert (start, stop) == (4 * i, 4 * i + 4)
    # the last step lands on sigma_next = 0 and spans to the end of the bank
    last = comfy_span(sigmas[-2], sigmas, HEADS, SHIFT_V)
    assert float(sigmas[-1]) == 0.0 and last[1] == HEADS


@pytest.mark.parametrize("steps", [8, 4])
def test_the_blend_matches_comfyui_and_the_float64_oracle_per_step(steps):
    sigmas = h3_sigmas(steps)
    for out_dim, stream_shift, seed in ((VIDEO_DIM, SHIFT_V, 1), (AUDIO_DIM, SHIFT_A, 2)):
        bank = _bank(out_features = out_dim, seed = seed)
        h = torch.randn(7, 24, generator = torch.Generator().manual_seed(seed))
        for i in range(steps):
            bank.pdd_state = _state(sigmas[i], sigmas[i + 1], stream = stream_shift)
            start, stop = comfy_span(sigmas[i], sigmas, HEADS, SHIFT_V)
            with torch.no_grad():
                ours = bank(h)
                comfy = comfy_pdd_head(bank.weight, bank.bias, h, HEADS, start, stop, stream_shift)
                oracle = oracle_head(bank.weight, bank.bias, h, HEADS, start, stop, stream_shift)
            assert (ours.double() - oracle).abs().max() < 1e-6
            assert (comfy.double() - oracle).abs().max() < 1e-6
            assert (ours - comfy).abs().max() < 1e-6


def test_sigma_pair_comes_off_the_device_schedule_like_comfyui():
    sigmas = h3_sigmas(8)
    timestep = (1.0 - sigmas[:-1]).float()
    for i in range(8):
        # three conditioning video rows pinned near clean, then the generated rows at this step
        uniq = torch.unique(torch.tensor([float(timestep[i]), 0.999, 1.0]))
        rows_t = torch.tensor([0.999] * 3 + [float(timestep[i])] * 5 + [1.0] * 2)
        indices = torch.searchsorted(uniq, rows_t)
        pair = pdd.h3_pdd_sigma_pair(uniq, indices, torch.arange(8), sigmas)
        assert pair.dtype == torch.float64
        assert abs(float(pair[0]) - float(sigmas[i])) < 1e-6
        assert float(pair[1]) == float(sigmas[i + 1])
    assert float(pdd.h3_pdd_sigma_pair(uniq, indices, torch.arange(8), sigmas)[1]) == 0.0


# ── the real denoiser: tiny random MiniMaxH3Transformer3DModel ──

TINY = dict(
    num_attention_heads = 2,
    attention_head_dim = 16,
    hidden_size = 32,
    num_layers = 2,
    num_refiner_layers = 1,
    ffn_dim = 64,
    in_channels = 24,
    audio_in_channels = 32,
    text_dim = 16,
    freq_dim = 32,
    time_embed_hidden_dim = 32,
    time_embed_dim = 16,
    rope_freq_dim = 2,
)


def _diffusers():
    return pytest.importorskip("diffusers")


def _tiny(
    heads = HEADS,
    seed = 0,
    device = "cpu",
):
    diffusers = _diffusers()
    torch.manual_seed(seed)
    model = diffusers.MiniMaxH3Transformer3DModel(**TINY)
    if heads > 1:
        dense = {
            "proj_out": model.proj_out,
            "audio_proj_out": model.audio_proj_out,
        }
        assert pdd.install_h3_pdd_heads(model, heads)
        with torch.no_grad():
            for name, lin in dense.items():
                bank = getattr(model, name)
                bank.weight.copy_(torch.randn_like(bank.weight) * 0.05)
                bank.bias.copy_(torch.randn_like(bank.bias) * 0.05)
                bank.weight[: lin.out_features].copy_(lin.weight)
                bank.bias[: lin.out_features].copy_(lin.bias)
    return model.to(device).eval()


class _Sched:
    def __init__(self, sigmas):
        self.sigmas = sigmas


def _inputs(
    device,
    t_video,
    n_text = 4,
    n_cond = 2,
    n_video = 6,
    n_audio = 3,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    s = n_text + n_cond + n_video + n_audio
    text = torch.arange(n_text)
    video = torch.arange(n_text, n_text + n_cond + n_video)
    audio = torch.arange(n_text + n_cond + n_video, s)
    tags = torch.zeros(s, dtype = torch.long)
    tags[text], tags[audio] = 1, 2
    row_t = torch.full((s,), float(t_video))
    row_t[video[:n_cond]] = max(float(t_video), 0.999)
    row_t[audio] = 1.0 - pdd_shift_sigma(1.0 - float(t_video))
    uniq, inv = torch.unique(row_t, sorted = True, return_inverse = True)
    pos = torch.stack(
        [torch.arange(s), torch.zeros(s, dtype = torch.long), torch.arange(s) % 3], 1
    ).float()
    return dict(
        hidden_states = torch.randn(1, n_cond + n_video, 96, generator = g).to(device),
        audio_hidden_states = torch.randn(1, n_audio, 32, generator = g).to(device),
        encoder_hidden_states = torch.randn(1, n_text, 16, generator = g).to(device),
        timestep = uniq.to(device),
        timestep_indices = inv.to(device),
        token_tags = tags.to(device),
        position_ids = pos.to(device),
        video_indices = video.to(device),
        audio_indices = audio.to(device),
        text_indices = text.to(device),
        return_dict = False,
    )


def pdd_shift_sigma(sigma_v):
    return float(comfy_time_shift_sigma(torch.tensor(sigma_v), SHIFT_V, SHIFT_A))


def _run_and_capture(model, kw):
    seen = {}

    def grab(name):
        def hook(_m, args, out):
            seen[name] = (args[0].detach().clone(), out.detach().clone())

        return hook

    handles = [
        model.proj_out.register_forward_hook(grab("video")),
        model.audio_proj_out.register_forward_hook(grab("audio")),
    ]
    try:
        with torch.no_grad():
            out = model(**kw)
    finally:
        for handle in handles:
            handle.remove()
    return out, seen


def _check_step(model, seen, sigmas, i):
    start, stop = comfy_span(sigmas[i], sigmas, HEADS, SHIFT_V)
    worst = 0.0
    for name, bank, shift in (
        ("video", model.proj_out, SHIFT_V),
        ("audio", model.audio_proj_out, SHIFT_A),
    ):
        h, out = seen[name]
        w, b = bank.weight.detach().cpu(), bank.bias.detach().cpu()
        oracle = oracle_head(w, b, h.cpu(), HEADS, start, stop, shift)
        comfy = comfy_pdd_head(w, b, h.cpu().float(), HEADS, start, stop, shift)
        worst = max(worst, float((out.cpu().double() - oracle).abs().max()))
        assert (out.cpu().double() - oracle).abs().max() < 1e-5, (name, i)
        assert (out.cpu() - comfy).abs().max() < 1e-5, (name, i)
    return worst


def _bound(model, steps = 8):
    sigmas = h3_sigmas(steps)
    pdd.bind_h3_pdd_schedule(model, _Sched(sigmas), SHIFT_V, SHIFT_A)
    return sigmas


def test_the_denoiser_blends_the_heads_comfyui_picks_at_every_step():
    model = _tiny()
    sigmas = _bound(model)
    for i in range(8):
        kw = _inputs("cpu", 1.0 - float(sigmas[i]), seed = i)
        _out, seen = _run_and_capture(model, kw)
        state = model.proj_out.pdd_state
        assert abs(float(state[0]) - float(sigmas[i])) < 1e-6 and float(state[1]) == float(
            sigmas[i + 1]
        )
        _check_step(model, seen, sigmas, i)


def test_an_unbound_pdd_denoiser_refuses_to_run():
    model = _tiny()
    with pytest.raises(ValueError, match = "sigma schedule"):
        with torch.no_grad():
            model(**_inputs("cpu", 0.5))


def test_a_plain_denoiser_is_untouched():
    model = _tiny(heads = 1)
    assert isinstance(model.proj_out, torch.nn.Linear) and pdd.h3_pdd_heads(model) == 0
    assert not pdd.bind_h3_pdd_schedule(model, _Sched(h3_sigmas(8)), SHIFT_V, SHIFT_A)


class _Pipe:
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.scheduler = scheduler


def test_the_render_binding_finds_the_partition_and_sizes_the_grid():
    model = _tiny()
    pipe = _Pipe(model, _Sched(h3_sigmas(8)))
    assert pdd.pipe_h3_pdd_heads(pipe) == HEADS
    assert pdd.bind_h3_pdd_render(pipe, SHIFT_V, SHIFT_A) == HEADS
    assert model.audio_proj_out.pdd_state[2:].tolist() == [SHIFT_V, SHIFT_A]
    assert pdd.h3_pdd_num_inference_steps(8) == 9
    assert pdd.bind_h3_pdd_render(_Pipe(_tiny(heads = 1), _Sched(None)), SHIFT_V, SHIFT_A) == 0


# ── the real loader: a ComfyUI-layout file built from the tiny model ──


def _comfy_key(name):
    from core.inference.video_minimax_h3_comfy import _RENAMES

    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks.") :]
    elif name.startswith("token_refiner.refiner_blocks."):
        name = "token_refiner.blocks." + name[len("token_refiner.refiner_blocks.") :]
    for old, new in reversed(_RENAMES):
        if new in name and (not new.startswith(("proj_", "audio_proj")) or name.startswith(new)):
            name = name.replace(new, old, 1)
            break
    return name


def _write_comfy_file(
    tmp_path,
    model,
    name,
    metadata = None,
    int8_layer = True,
):
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = {}
    for key in list(sd):
        if ".attn.to_q.weight" in key:
            stem = key[: -len("to_q.weight")]
            fused = torch.cat([sd.pop(stem + p + ".weight") for p in ("to_q", "to_k", "to_v")])
            out[_comfy_key(stem).replace("transformer_blocks.", "blocks.") + "qkv_proj.weight"] = (
                fused
            )
    for key, value in sd.items():
        if key.endswith(".ff.net.0.proj.weight"):
            value_half, gate = value.chunk(2)
            out[_comfy_key(key.replace(".ff.net.0.proj.weight", ".mlp.fc1.weight"))] = torch.cat(
                [gate, value_half]
            )
            continue
        out[_comfy_key(key)] = value
    if int8_layer:
        key = "blocks.0.attn.out_proj.weight"
        w = out[key].float()
        scale = w.abs().amax().clamp_min(1e-8) / 127.0
        out[key] = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        out[key.replace(".weight", ".weight_scale")] = scale.reshape(())
        out["blocks.0.attn.out_proj.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype = torch.uint8
        )
    path = tmp_path / name
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in out.items()}, str(path), metadata = metadata
    )
    cfg = tmp_path / "base" / "transformer"
    cfg.mkdir(parents = True, exist_ok = True)
    (cfg / "config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Transformer3DModel", **TINY})
    )
    return str(path), str(tmp_path / "base")


def _load(path, base):
    from core.inference import diffusion_comfy_quant as cq
    from core.inference.video_minimax_h3_comfy import load_h3_comfy_transformer
    return load_h3_comfy_transformer(
        _diffusers().MiniMaxH3Transformer3DModel,
        path,
        cq.refuse_comfy_quant(path),
        config = base,
        subfolder = "transformer",
        dtype = torch.float32,
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        int8_backend = None,
        fp8_backend = None,
    )


def test_a_with_heads_file_loads_through_the_comfy_loader_with_its_bank(tmp_path):
    src = _tiny()
    meta = {"config": json.dumps({"transformer": {"pdd_steps": HEADS}}), "pdd_acc_baked": "true"}
    path, base = _write_comfy_file(tmp_path, src, WITH, metadata = meta)
    model = _load(path, base).eval()
    assert pdd.h3_pdd_heads(model) == HEADS
    assert tuple(model.proj_out.weight.shape) == (HEADS * VIDEO_DIM, TINY["hidden_size"])
    assert model.proj_out.weight.dtype == torch.float32
    assert torch.equal(model.proj_out.weight, src.proj_out.weight)
    assert torch.equal(model.audio_proj_out.bias, src.audio_proj_out.bias)
    sigmas = _bound(model)
    for i in (0, 7):
        _out, seen = _run_and_capture(model, _inputs("cpu", 1.0 - float(sigmas[i])))
        _check_step(model, seen, sigmas, i)


def test_a_without_heads_bake_is_refused_by_the_comfy_loader(tmp_path):
    src = _tiny(heads = 1)
    for name, meta in (
        (WITHOUT_OLD, {"pdd_acc_baked": "true"}),
        (WITHOUT, None),
    ):
        path, base = _write_comfy_file(tmp_path, src, name, metadata = meta)
        with pytest.raises(ValueError, match = "no head bank"):
            _load(path, base)
    plain, base = _write_comfy_file(tmp_path, src, "minimax_h3_fl2va_int8_convrot.safetensors")
    assert isinstance(_load(plain, base).proj_out, torch.nn.Linear)


# ── torch.compile and CUDA graphs ──


def test_the_bank_compiles_with_no_graph_break_and_no_recompile_across_steps():
    import torch._dynamo as dynamo

    dynamo.reset()
    bank = _bank()
    h = torch.randn(5, 24)
    sigmas = h3_sigmas(8)
    explain = dynamo.explain(bank)(h)
    assert explain.graph_break_count == 0 and explain.graph_count == 1
    dynamo.reset()
    counts = dynamo.utils.counters
    counts.clear()
    compiled = torch.compile(bank, fullgraph = True, backend = "eager")
    with torch.no_grad():
        for i in range(8):
            bank.set_state(_state(sigmas[i], sigmas[i + 1])[:2])
            ref = comfy_pdd_head(
                bank.weight,
                bank.bias,
                h,
                HEADS,
                *comfy_span(sigmas[i], sigmas, HEADS, SHIFT_V),
                SHIFT_V,
            )
            assert (compiled(h) - ref).abs().max() < 1e-6, i
    assert counts["stats"]["unique_graphs"] == 1, dict(counts["stats"])
    assert not counts["recompiles"] if "recompiles" in counts else True


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def test_the_bank_compiles_with_inductor_on_cuda_and_tracks_every_step():
    _cuda()
    import torch._dynamo as dynamo

    dynamo.reset()
    dynamo.utils.counters.clear()
    bank = _bank().cuda()
    h = torch.randn(5, 24, device = "cuda")
    sigmas = h3_sigmas(8)
    compiled = torch.compile(bank, fullgraph = True)
    with torch.no_grad():
        for i in range(8):
            bank.set_state(_state(sigmas[i], sigmas[i + 1], device = "cuda")[:2])
            ref = comfy_pdd_head(
                bank.weight.cpu(),
                bank.bias.cpu(),
                h.cpu(),
                HEADS,
                *comfy_span(sigmas[i], sigmas, HEADS, SHIFT_V),
                SHIFT_V,
            )
            assert (compiled(h).cpu() - ref).abs().max() < 1e-5, i
    assert dynamo.utils.counters["stats"]["unique_graphs"] == 1


def test_a_cuda_graph_of_the_denoiser_replays_with_changing_sigmas(monkeypatch):
    _cuda()
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setenv("UNSLOTH_DIFFUSION_CUDA_GRAPHS", "1")
    model = _tiny(device = "cuda")
    sigmas = _bound(model)
    eager = []
    with torch.no_grad():
        for i in range(8):
            eager.append(model(**_inputs("cuda", 1.0 - float(sigmas[i]), seed = 0)))
    handle = cg.GraphedForward(model).install().enable()
    try:
        replayed = []
        with torch.inference_mode():
            for _round in range(2):
                for i in range(8):
                    out = model(**_inputs("cuda", 1.0 - float(sigmas[i]), seed = 0))
                    replayed.append([o.clone() for o in out])
        stats = dict(handle.stats)
    finally:
        handle.uninstall()
    # one recording per distinct input shape (step 0's audio t equals its video t, so its timestep table is shorter);
    # every other call is a replay with that step's sigmas
    shapes = {tuple(_inputs("cuda", 1.0 - float(sigmas[i]))["timestep"].shape) for i in range(8)}
    assert stats["captures"] == len(shapes), stats
    assert stats["replays"] >= 16 - len(shapes) * 2, stats
    assert stats["fallbacks"] == 0 and stats["eager_calls"] == 0, stats
    for k, out in enumerate(replayed):
        ref = eager[k % 8]
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]), k
    # the steps really differ: a graph that baked the first step's heads would repeat it
    assert not torch.equal(eager[0][0], eager[7][0])


def test_the_denoiser_under_group_offload_matches_resident():
    _cuda()
    resident = _tiny(device = "cuda")
    sigmas = _bound(resident)
    ref = []
    with torch.no_grad():
        for i in range(8):
            ref.append(resident(**_inputs("cuda", 1.0 - float(sigmas[i]))))
    for kind in ("leaf_level", "block_level"):
        model = _tiny()
        _bound(model)
        # block_level as Studio's streamed H3 denoiser applies it (stream, record_stream, non_blocking)
        kwargs = (
            {"num_blocks_per_group": 1, "record_stream": True, "non_blocking": True}
            if kind == "block_level"
            else {}
        )
        model.enable_group_offload(
            torch.device("cuda"),
            offload_device = torch.device("cpu"),
            offload_type = kind,
            use_stream = True,
            **kwargs,
        )
        with torch.no_grad():
            for i in range(8):
                out = model(**_inputs("cuda", 1.0 - float(sigmas[i])))
                assert (out[0] - ref[i][0]).abs().max() < 1e-5, (kind, i)
                assert (out[1] - ref[i][1]).abs().max() < 1e-5, (kind, i)


def test_the_whole_pdd_denoiser_forward_compiles_fullgraph_without_recompiles():
    """The pre-hook runs eager in ``Module.__call__`` (Studio compiles the forward / its repeated blocks, never the
    call), so the compiled forward reads the step's sigmas as a tensor: one graph for every step."""
    import torch._dynamo as dynamo

    model = _tiny()
    eager = _tiny()
    sigmas = _bound(model)
    _bound(eager)
    kw = _inputs("cpu", 1.0 - float(sigmas[3]))
    pdd._pre_hook(model, (), kw)
    dynamo.reset()
    explain = dynamo.explain(model.forward)(**kw)
    assert explain.graph_break_count == 0 and explain.graph_count == 1
    dynamo.reset()
    dynamo.utils.counters.clear()
    model.__dict__["forward"] = torch.compile(model.forward, fullgraph = True, backend = "eager")
    with torch.no_grad():
        # steps 1..7 share one input shape (step 0's timestep table is shorter, a different shape by design)
        for i in range(1, 8):
            kw = _inputs("cpu", 1.0 - float(sigmas[i]), seed = i)
            out, ref = model(**kw), eager(**kw)
            assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]), i
            assert float(model.proj_out.pdd_state[1]) == float(sigmas[i + 1]), i
    assert dynamo.utils.counters["stats"]["unique_graphs"] == 1, dict(
        dynamo.utils.counters["stats"]
    )


def test_the_denoiser_under_model_cpu_offload_matches_resident():
    _cuda()
    accelerate = pytest.importorskip("accelerate")
    resident = _tiny(device = "cuda")
    sigmas = _bound(resident)
    model = _tiny()
    _bound(model)
    model, hook = accelerate.cpu_offload_with_hook(model, execution_device = torch.device("cuda"))
    with torch.no_grad():
        for i in range(8):
            kw = _inputs("cuda", 1.0 - float(sigmas[i]))
            out, ref = model(**kw), resident(**kw)
            assert (out[0] - ref[0]).abs().max() < 1e-5 and (out[1] - ref[1]).abs().max() < 1e-5, i
            hook.offload()
            assert model.proj_out.weight.device.type == "cpu"


def test_a_bound_render_reads_the_schedule_the_pipeline_sets_afterwards():
    """Studio binds the pipeline's scheduler before the call; the modular loop's set_timesteps runs inside it."""
    diffusers = _diffusers()
    model = _tiny()
    scheduler = diffusers.MiniMaxH3Scheduler(shift = 1.0)
    pipe = _Pipe(model, scheduler)
    scheduler.set_shift(SHIFT_V)
    assert pdd.bind_h3_pdd_render(pipe, SHIFT_V, SHIFT_A) == HEADS
    scheduler.set_timesteps(pdd.h3_pdd_num_inference_steps(pdd.H3_PDD_DEFAULT_STEPS))
    spans = []
    latents = torch.randn(8, 96)
    for i, t in enumerate(scheduler.timesteps):
        kw = _inputs("cpu", float(t), seed = i)
        out, seen = _run_and_capture(model, kw)
        _check_step(model, seen, scheduler.sigmas, i)
        spans.append(comfy_span(scheduler.sigmas[i], scheduler.sigmas, HEADS, SHIFT_V))
        latents = scheduler.step(out[0][0].float(), t, latents, return_dict = False)[0]
    assert spans == [(4 * i, 4 * i + 4) for i in range(8)]


def test_a_second_render_can_rebind_a_bank_its_first_render_moved_in_inference_mode():
    """The pipeline runs under inference_mode, so the first render's device move made pdd_state an inference tensor
    and the second render's set_shifts (outside it) raised. A pdd_state left as an inference tensor is replaced too."""
    bank = _bank()
    with torch.inference_mode():
        bank.pdd_state = bank.pdd_state.clone()
    assert bank.pdd_state.is_inference()
    bank.set_shifts(SHIFT_V, SHIFT_A)
    assert not bank.pdd_state.is_inference()
    assert bank.pdd_state.tolist()[2:] == [SHIFT_V, SHIFT_A]
    with torch.inference_mode():
        bank.set_state(_state(0.7, 0.6)[:2])
    assert not bank.pdd_state.is_inference()
    assert bank.pdd_state.tolist()[:2] == pytest.approx([0.7, 0.6])
    if torch.cuda.is_available():
        bank = _bank()
        with torch.inference_mode():
            bank.set_state(_state(0.9, 0.8, device = "cuda")[:2])
        assert not bank.pdd_state.is_inference() and bank.pdd_state.is_cuda
        bank.set_shifts(SHIFT_V, SHIFT_A)


def test_pdd_defaults_apply_only_to_minimax_h3():
    """A Wan or Hunyuan checkpoint under a path naming "pdd" keeps its family defaults."""
    from core.inference.video_families import default_video_generation_params

    h3 = "/models/minimax_h3_ref2va_pruned_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors"
    assert default_video_generation_params(h3, fallback = (30, 5.0), family = "minimax-h3") == (8, 1.0)
    wan = "/models/pdd/wan2.2-ti2v-5b.safetensors"
    assert default_video_generation_params(wan, fallback = (50, 5.0), family = "wan2.2") == (20, 5.0)
    other = "/models/pdd/hunyuan.safetensors"
    assert default_video_generation_params(
        other, fallback = (50, 6.0), family = "hunyuanvideo-1.5"
    ) == (50, 6.0)
    assert default_video_generation_params(other, fallback = (50, 6.0)) == (50, 6.0)
    # A stock H3 checkpoint in a folder named pdd keeps the stock recipe: only the file name decides.
    stock = "/models/pdd/minimax_h3_fl2va_int8_convrot.safetensors"
    assert default_video_generation_params(stock, fallback = (30, 1.0), family = "minimax-h3") == (
        30,
        1.0,
    )


@pytest.mark.parametrize("has_fp64", [False, True])
def test_device_has_fp64_reads_the_xpu_capability(monkeypatch, has_fp64):
    import types

    monkeypatch.setattr(pdd, "_FP64_BY_DEVICE", {})
    monkeypatch.setattr(
        torch.xpu,
        "get_device_properties",
        lambda index = None: types.SimpleNamespace(has_fp64 = has_fp64),
        raising = False,
    )
    assert pdd.device_has_fp64(torch.device("xpu", 0)) is has_fp64
    assert pdd.device_has_fp64("cpu") and pdd.device_has_fp64(torch.device("cuda", 0))
    assert not pdd.device_has_fp64(torch.device("mps"))


def test_an_unreadable_xpu_counts_as_without_fp64(monkeypatch):
    def boom(index = None):
        raise RuntimeError("no XPU")

    monkeypatch.setattr(pdd, "_FP64_BY_DEVICE", {})
    monkeypatch.setattr(torch.xpu, "get_device_properties", boom, raising = False)
    assert not pdd.device_has_fp64(torch.device("xpu", 0))


def test_the_denoiser_never_puts_float64_on_a_device_without_it(monkeypatch):
    """CUDA spoofed as an fp64-less XPU (float64 on it raises like Arc A-series): the blend still matches every step."""
    _cuda()
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class _NoDeviceFp64(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten((args, kwargs, out))[0]:
                if isinstance(t, torch.Tensor) and t.dtype == torch.float64 and t.is_cuda:
                    raise RuntimeError(
                        f"Required aspect fp64 is not supported on the device ({func})"
                    )
            return out

    monkeypatch.setattr(
        pdd, "device_has_fp64", lambda d: getattr(d, "type", d) != "cuda", raising = False
    )
    model = _tiny(device = "cuda")
    sigmas = _bound(model)
    for i in range(8):
        kw = _inputs("cuda", 1.0 - float(sigmas[i]), seed = i)
        with _NoDeviceFp64():
            _out, seen = _run_and_capture(model, kw)
        state = model.proj_out.pdd_state
        assert state.device.type == "cpu" and state.dtype == torch.float64
        assert abs(float(state[0]) - float(sigmas[i])) < 1e-6
        assert float(state[1]) == float(sigmas[i + 1])
        assert seen["video"][1].is_cuda
        _check_step(model, seen, sigmas, i)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which diffusers-dispatch families the unset INT8 attention switch races, on tiny real DiTs of each family:
FLUX.1, FLUX.2 and HunyuanVideo-1.5 (through Studio's padded-text trim) route their unmasked calls; Qwen-Image
(accuracy) stays on request only; HunyuanImage-2.1 and Krea-2 always attend with a text padding mask. CPU: the card is faked
and the kernel replaced by a recorder (test_diffusion_int8_attention_dispatch.py)."""

from __future__ import annotations

import types

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_int8_attention as I  # noqa: E402
from core.inference import diffusion_int8_attention_dispatch as D  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    monkeypatch.delenv(D.DISPATCH_ENV, raising = False)
    stats = dict(I.STATS)
    D.ROUTE_STATS.clear()
    D._DECISIONS.clear()
    monkeypatch.setattr(D, "MIN_TOKENS", 0)  # the toy DiTs run a few dozen tokens
    yield
    I.STATS.clear()
    I.STATS.update(stats)
    D.ROUTE_STATS.clear()


@pytest.fixture
def card(monkeypatch):
    """CPU tensors count as the route's card; the op records its calls and returns the native result."""
    calls = []
    monkeypatch.setattr(D, "_on_card", lambda x, route: True)

    def impl(q, k, v, *mode):
        calls.append((tuple(q.shape), mode))
        return I.native_attention(q, k, v)

    monkeypatch.setattr(I, "int8_attention_impl", impl)
    I.int8_attention_op()
    return calls


def _fake_cuda(
    monkeypatch,
    found,
    impl = "cuda",
):
    """A CUDA card whose accuracy check / race returns ``found``; the extra race points lose."""
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: impl)
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: False)
    seen = []

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        seen.append({"heads": heads, "head_dim": head_dim, "name": incumbent_name, "timed": timed})
        return {"impl": impl, **found}

    monkeypatch.setattr(I, "verdict", verdict)
    return seen


_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
_BF = torch.bfloat16
# measured: kernel per-step error within the family's INT8 weights' and faster on the card (the race decides per card)
RACED = ("HunyuanVideo15Transformer3DModel",)
SPEED_UNSET = ()
# routed on request only: INT8 LPIPS above the kernel-swap floor (see AUTO_DITS)
REQUEST_ONLY = ("FluxTransformer2DModel", "Flux2Transformer2DModel")


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _r(*shape, seed = 1):
    return torch.randn(*shape, generator = torch.Generator().manual_seed(seed)).to(_BF)


def _t(value):
    return torch.tensor([value]).to(_BF)


def _flux1():
    torch.manual_seed(0)
    dit = diffusers.FluxTransformer2DModel(
        in_channels = 4,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        pooled_projection_dim = 16,
        axes_dims_rope = (4, 6, 6),
    )
    kw = dict(
        hidden_states = _r(1, 16, 4),
        encoder_hidden_states = _r(1, 8, 32),
        pooled_projections = _r(1, 16),
        timestep = _t(0.5),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(8, 3),
        return_dict = False,
    )
    return dit, kw


def _flux2():
    torch.manual_seed(0)
    dit = diffusers.Flux2Transformer2DModel(
        in_channels = 4,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        timestep_guidance_channels = 32,
        axes_dims_rope = (4, 4, 4, 4),
        guidance_embeds = False,
    )
    kw = dict(
        hidden_states = _r(1, 16, 4),
        encoder_hidden_states = _r(1, 8, 32),
        timestep = _t(0.5),
        img_ids = torch.zeros(1, 16, 4),
        txt_ids = torch.zeros(1, 8, 4),
        return_dict = False,
    )
    return dit, kw


def _qwen():
    torch.manual_seed(0)
    dit = diffusers.QwenImageTransformer2DModel(
        patch_size = 2,
        in_channels = 16,
        out_channels = 4,
        num_layers = 1,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        axes_dims_rope = (4, 6, 6),
    )
    # the pipeline drops an all-ones prompt mask (pipeline_qwenimage.py: prompt_embeds_mask.all() -> None)
    kw = dict(
        hidden_states = _r(1, 16, 16),
        encoder_hidden_states = _r(1, 8, 32),
        timestep = _t(0.5),
        img_shapes = [(1, 4, 4)],
        return_dict = False,
    )
    return dit, kw


def _hunyuan15():
    torch.manual_seed(0)
    dit = diffusers.HunyuanVideo15Transformer3DModel(
        in_channels = 4,
        out_channels = 4,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 1,
        num_refiner_layers = 1,
        text_embed_dim = 32,
        text_embed_2_dim = 16,
        image_embed_dim = 16,
        rope_axes_dim = (4, 6, 6),
        target_size = 64,
        task_type = "t2v",
    )
    kw = dict(
        hidden_states = _r(1, 4, 2, 4, 4),
        timestep = _t(500.0),
        encoder_hidden_states = _r(1, 8, 32),
        encoder_attention_mask = torch.ones(1, 8, dtype = torch.long),
        encoder_hidden_states_2 = _r(1, 4, 16),
        encoder_attention_mask_2 = torch.ones(1, 4, dtype = torch.long),
        image_embeds = _r(1, 4, 16),
        return_dict = False,
    )
    return dit, kw


def _trimmed_hunyuan15():
    """HunyuanVideo-1.5 as Studio loads it: the padded-text trim (null-mask processors) installed."""
    from core.inference import diffusion_attention as A

    dit, kw = _hunyuan15()
    dit = dit.to(_BF)
    family = types.SimpleNamespace(transformer_class = "HunyuanVideo15Transformer3DModel")
    assert A.install_hunyuan_attention_trim(_Pipe(dit), family)
    return dit, kw


def _hunyuan_image():
    torch.manual_seed(0)
    dit = diffusers.HunyuanImageTransformer2DModel(
        in_channels = 4,
        out_channels = 4,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 1,
        num_single_layers = 1,
        num_refiner_layers = 1,
        text_embed_dim = 32,
        rope_axes_dim = (8, 8),
    )
    kw = dict(
        hidden_states = _r(1, 4, 4, 4),
        timestep = _t(500.0),
        encoder_hidden_states = _r(1, 8, 32),
        encoder_attention_mask = torch.ones(1, 8, dtype = torch.long),
        return_dict = False,
    )
    return dit, kw


def _krea2():
    torch.manual_seed(0)
    dit = diffusers.Krea2Transformer2DModel(
        in_channels = 16,
        num_layers = 1,
        attention_head_dim = 16,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        intermediate_size = 32,
        timestep_embed_dim = 32,
        text_hidden_dim = 32,
        num_text_layers = 1,
        text_num_attention_heads = 2,
        text_num_key_value_heads = 2,
        text_intermediate_size = 32,
        num_layerwise_text_blocks = 1,
        num_refiner_text_blocks = 1,
        axes_dims_rope = (4, 6, 6),
    )
    kw = dict(
        hidden_states = _r(1, 16, 16),
        encoder_hidden_states = _r(1, 8, 1, 32),
        timestep = _t(0.5),
        position_ids = torch.zeros(24, 3, dtype = torch.long),
        encoder_attention_mask = torch.ones(1, 8, dtype = torch.bool),
    )
    return dit, kw


_BUILDERS = {
    "FluxTransformer2DModel": _flux1,
    "Flux2Transformer2DModel": _flux2,
    "QwenImageTransformer2DModel": _qwen,
    "HunyuanVideo15Transformer3DModel": _trimmed_hunyuan15,
}


def _build(name):
    dit, kw = _BUILDERS[name]()
    dit = dit.to(_BF)
    dit.set_attention_backend("native")
    return dit, kw


def _forward(dit, kw):
    D.ROUTE_STATS.clear()
    with torch.no_grad():
        dit(**kw)
    return dict(D.ROUTE_STATS)


def test_the_measured_families_are_listed():
    assert set(RACED) <= D.AUTO_DITS
    assert set(SPEED_UNSET) <= D.SPEED_UNSET_DITS
    assert set(RACED) <= D.AUTO_CUDA_ONLY and set(RACED) <= I.SINK_DITS
    for name in REQUEST_ONLY:
        assert (
            name not in D.AUTO_DITS
            and name not in D.SPEED_UNSET_DITS
            and name not in D.AUTO_CUDA_ONLY
        )
        assert name in I.SINK_DITS
    assert (
        "HunyuanVideo15Transformer3DModel" not in D.SPEED_UNSET_DITS
    )  # video: speed profile only, as Wan
    for name in (
        "QwenImageTransformer2DModel",
        "HunyuanImageTransformer2DModel",
    ):
        assert name not in D.AUTO_DITS and name not in I.SINK_DITS
    # Krea 2 races on the plain path (its block calls are mask-free over the valid key prefix), no sink path
    assert (
        "Krea2Transformer2DModel" in D.AUTO_DITS and "Krea2Transformer2DModel" in D.AUTO_CUDA_ONLY
    )
    assert "Krea2Transformer2DModel" not in I.SINK_DITS


@pytest.mark.parametrize("name", RACED)
def test_unset_switch_races_and_routes_the_family(card, monkeypatch, name):
    """Switch unset: the card check and race run on every routable module, and each unmasked call reaches the
    kernel on the attention-sink path (raced against torch)."""
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit, kw = _build(name)
    routed = D.install(_Pipe(dit), _TARGET)
    assert routed == len(D.routable_modules(dit)) > 0
    assert len(seen) == routed and all(s["timed"] for s in seen)
    routes = [m.processor._attention_backend for m in D.routable_modules(dit)]
    assert all(isinstance(r, D.Int8Route) and r.sinks == I.MODE_RACE for r in routes)
    assert _forward(dit, kw) == {"int8": routed}
    assert len(card) == routed


@pytest.mark.parametrize("name", RACED)
def test_a_lost_race_keeps_the_incumbent(card, monkeypatch, name):
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": None})
    dit, kw = _build(name)
    assert D.install(_Pipe(dit), _TARGET) == 0
    assert all(
        not isinstance(m.processor._attention_backend, D.Int8Route) for m in D.routable_modules(dit)
    )
    assert _forward(dit, kw) == {} and not card


@pytest.mark.parametrize("name", RACED)
def test_speed_unset_rule(card, monkeypatch, name):
    """Speed unset: image families race; HunyuanVideo-1.5 waits for a speed profile, like Wan."""
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit, _ = _build(name)
    routed = D.install(_Pipe(dit), _TARGET, speed_unset_only = True)
    assert (routed > 0) == (name in SPEED_UNSET) and bool(seen) == (name in SPEED_UNSET)


@pytest.mark.parametrize("name", RACED)
def test_speed_off_and_kill_switches_route_nothing(card, monkeypatch, name):
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit, kw = _build(name)
    pipe = _Pipe(dit)
    assert D.install(pipe, _TARGET, speed_active = False) == 0
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert D.install(pipe, _TARGET) == 0
    monkeypatch.delenv(I.INT8_ATTENTION_ENV)
    monkeypatch.setenv(D.DISPATCH_ENV, "0")
    assert D.install(pipe, _TARGET) == 0
    assert not seen and _forward(dit, kw) == {} and not card


@pytest.mark.parametrize("name", RACED)
def test_a_triton_stand_in_keeps_the_incumbent_unless_requested(card, monkeypatch, name):
    """Their accuracy was measured on the CUDA kernel: unset, a Triton stand-in keeps the incumbent."""
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0}, impl = "triton")
    dit, _ = _build(name)
    assert D.install(_Pipe(dit), _TARGET) == 0
    assert D.install(_Pipe(dit), _TARGET, requested = I.BACKEND) > 0


@pytest.mark.parametrize("name", REQUEST_ONLY)
def test_flux1_flux2_route_on_request_only(card, monkeypatch, name):
    """FLUX.1 / FLUX.2: unset races nothing; a request routes every module on the sink path."""
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit, kw = _build(name)
    assert D.install(_Pipe(dit), _TARGET) == 0
    assert D.install(_Pipe(dit), _TARGET, speed_unset_only = True) == 0
    assert not seen and _forward(dit, kw) == {} and not card
    routed = D.install(_Pipe(dit), _TARGET, requested = I.BACKEND)
    assert routed == len(D.routable_modules(dit)) > 0
    routes = [m.processor._attention_backend for m in D.routable_modules(dit)]
    assert all(isinstance(r, D.Int8Route) and r.sinks == I.MODE_KERNELS for r in routes)
    assert _forward(dit, kw) == {"int8": routed}


def test_qwen_image_routes_on_request_only(card, monkeypatch):
    """Qwen-Image (v1 / Edit / 2512): unset races nothing; a request routes its unmasked joint calls (main's kernels,
    no sink path), while a padded prompt (a batched CFG pair of different lengths) passes the mask and stays native."""
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit, kw = _build("QwenImageTransformer2DModel")
    assert D.install(_Pipe(dit), _TARGET) == 0 and not seen
    assert D.install(_Pipe(dit), _TARGET, requested = I.BACKEND) == 1
    assert dit.transformer_blocks[0].attn.processor._attention_backend.sinks == I.MODE_MAIN
    mask = torch.ones(1, 8, dtype = torch.long)
    mask[:, 6:] = 0
    assert _forward(dit, {**kw, "encoder_hidden_states_mask": mask}) == {"mask": 1}
    assert _forward(dit, kw) == {"int8": 1} and card == [((1, 24, 2, 16), ())]


def test_hunyuan15_trimmed_processor_reads_the_dispatch_at_call_time(card, monkeypatch):
    """The null-mask processor must call dispatch, or the stock one would pass the text mask."""
    _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    from core.inference import diffusion_attention as A

    assert D._calls_dispatch(A._null_mask_processor_cls()())
    stock, kw = _hunyuan15()
    stock = stock.to(_BF)
    stock.set_attention_backend("native")
    assert D.install(_Pipe(stock), _TARGET) == 1
    assert _forward(stock, kw) == {"mask": 1}
    dit, kw = _build("HunyuanVideo15Transformer3DModel")
    assert D.install(_Pipe(dit), _TARGET) == 1
    padded = dict(kw)
    padded["encoder_attention_mask"] = torch.tensor([[1] * 6 + [0] * 2])
    assert _forward(dit, padded) == {"int8": 1}
    D.remove(_Pipe(dit))
    assert _forward(dit, kw) == {}


@pytest.mark.parametrize(
    "build, routable, outcome",
    [(_hunyuan_image, 1, {"mask": 1}), (_krea2, 0, {})],
    ids = ["HunyuanImage-2.1", "Krea-2"],
)
def test_masked_families_stay_as_they_were(card, monkeypatch, build, routable, outcome):
    """HunyuanImage-2.1 pads every call's mask (image tokens + text mask) and Krea-2 always passes its text padding
    mask (its attention has no ``heads``, so no route at all): unset races nothing, a request reaches no call."""
    seen = _fake_cuda(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit, kw = build()
    dit = dit.to(_BF)
    dit.set_attention_backend("native")
    assert D.install(_Pipe(dit), _TARGET) == 0 and not seen
    assert D.install(_Pipe(dit), _TARGET, requested = I.BACKEND) == routable
    assert _forward(dit, kw) == outcome and not card


# ----------------------------------------------------------------------------------------------- CUDA (real kernel)


def _card_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16


needs_kernel = pytest.mark.skipif(
    not torch.cuda.is_available() or I.implementation(torch.cuda.current_device(), 128) is None,
    reason = "needs a CUDA card with an INT8 attention kernel",
)


def _cuda_case(name, dtype):
    """(DiT, inputs) at head dim 128 with 1024 image tokens and 64 text tokens: every joint call is over the floor."""
    g = torch.Generator().manual_seed(3)

    def r(*shape):
        return torch.randn(*shape, generator = g).to("cuda", dtype)

    t = torch.tensor([0.5], device = "cuda", dtype = dtype)
    ids = torch.stack(
        torch.meshgrid(torch.zeros(1), torch.arange(32.0), torch.arange(32.0), indexing = "ij"), -1
    ).reshape(-1, 3)
    torch.manual_seed(0)
    if name == "FluxTransformer2DModel":
        dit = diffusers.FluxTransformer2DModel(
            in_channels = 16,
            num_layers = 1,
            num_single_layers = 1,
            attention_head_dim = 128,
            num_attention_heads = 4,
            joint_attention_dim = 64,
            pooled_projection_dim = 32,
            axes_dims_rope = (16, 56, 56),
        )
        kw = dict(
            hidden_states = r(1, 1024, 16),
            encoder_hidden_states = r(1, 64, 64),
            pooled_projections = r(1, 32),
            timestep = t,
            img_ids = ids.cuda(),
            txt_ids = torch.zeros(64, 3, device = "cuda"),
        )
    elif name == "Flux2Transformer2DModel":
        dit = diffusers.Flux2Transformer2DModel(
            in_channels = 16,
            num_layers = 1,
            num_single_layers = 1,
            attention_head_dim = 128,
            num_attention_heads = 4,
            joint_attention_dim = 64,
            timestep_guidance_channels = 64,
            axes_dims_rope = (32, 32, 32, 32),
            guidance_embeds = False,
        )
        ids4 = torch.cat([ids, torch.zeros(1024, 1)], 1)[None].cuda()
        kw = dict(
            hidden_states = r(1, 1024, 16),
            encoder_hidden_states = r(1, 64, 64),
            timestep = t,
            img_ids = ids4,
            txt_ids = torch.zeros(1, 64, 4, device = "cuda"),
        )
    elif name == "QwenImageTransformer2DModel":
        dit = diffusers.QwenImageTransformer2DModel(
            patch_size = 2,
            in_channels = 16,
            out_channels = 4,
            num_layers = 1,
            attention_head_dim = 128,
            num_attention_heads = 4,
            joint_attention_dim = 64,
            axes_dims_rope = (16, 56, 56),
        )
        kw = dict(
            hidden_states = r(1, 1024, 16),
            encoder_hidden_states = r(1, 64, 64),
            timestep = t,
            img_shapes = [(1, 32, 32)],
        )
    else:
        from core.inference import diffusion_attention as A

        dit = diffusers.HunyuanVideo15Transformer3DModel(
            in_channels = 4,
            out_channels = 4,
            num_attention_heads = 4,
            attention_head_dim = 128,
            num_layers = 1,
            num_refiner_layers = 1,
            text_embed_dim = 64,
            text_embed_2_dim = 32,
            image_embed_dim = 32,
            rope_axes_dim = (16, 56, 56),
            target_size = 64,
            task_type = "t2v",
        )
        family = types.SimpleNamespace(transformer_class = name)
        assert A.install_hunyuan_attention_trim(_Pipe(dit), family)
        kw = dict(
            hidden_states = r(1, 4, 1, 32, 32),
            timestep = t * 1000,
            encoder_hidden_states = r(1, 64, 64),
            encoder_attention_mask = torch.ones(1, 64, dtype = torch.long, device = "cuda"),
            encoder_hidden_states_2 = r(1, 8, 32),
            encoder_attention_mask_2 = torch.ones(1, 8, dtype = torch.long, device = "cuda"),
            image_embeds = r(1, 4, 32),
        )
    dit = dit.to("cuda", dtype)
    dit.set_attention_backend("native")
    return dit, {**kw, "return_dict": False}


@pytest.mark.gpu
@needs_kernel
@pytest.mark.parametrize("name", RACED + REQUEST_ONLY + ("QwenImageTransformer2DModel",))
def test_cuda_family_routes_and_matches(monkeypatch, name):
    """Named requests run each family's real kernel and stay close to native attention."""
    import torch.nn.functional as F

    monkeypatch.setattr(D, "MIN_TOKENS", 1024)
    dtype = _card_dtype()
    dit, kw = _cuda_case(name, dtype)
    target = types.SimpleNamespace(device = "cuda", dtype = dtype)
    with torch.no_grad():
        ref = dit(**kw)[0].float()
        routed = D.install(_Pipe(dit), target, requested = I.BACKEND)
        assert routed == len(D.routable_modules(dit)) > 0
        before = I.STATS.get("int8", 0)
        D.ROUTE_STATS.clear()
        got = dit(**kw)[0].float()
        assert (
            I.STATS.get("int8", 0) - before >= 1 and D.ROUTE_STATS.get("int8", 0) >= 1
        ), D.ROUTE_STATS
        D.remove(_Pipe(dit))
        again = dit(**kw)[0].float()
    assert torch.equal(again, ref)
    rel = float((got - ref).norm() / ref.norm())
    cos = float(F.cosine_similarity(got.flatten(), ref.flatten(), dim = 0))
    assert torch.isfinite(got).all() and rel < 0.05 and cos > 0.998, (rel, cos)

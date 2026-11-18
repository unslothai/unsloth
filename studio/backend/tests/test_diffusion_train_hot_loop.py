# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion training hot loop: no per-micro-step host syncs, one decode per cached image, batched EMA, conditioning
placement and the streamed int8 load. Every change here must leave the numbers bit-identical."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.training import diffusion_dit_trainer as dit  # noqa: E402
from core.training import diffusion_train_extras as extras  # noqa: E402


class _ScalarReads(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts tensor -> Python scalar reads (each one a host sync on an accelerator)."""

    def __init__(self):
        super().__init__()
        self.reads = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func is torch.ops.aten._local_scalar_dense.default:
            self.reads += 1
        return func(*args, **(kwargs or {}))


def test_loss_accumulation_matches_the_python_float_sum_without_scalar_reads():
    g = torch.Generator().manual_seed(0)
    losses = [torch.rand((), generator = g) * 3 for _ in range(8)]
    expected = 0.0
    for loss in losses:
        expected += float(loss) / 8
    acc = None
    with _ScalarReads() as mode:
        for loss in losses:
            acc = extras.accumulate_loss(acc, loss, 8)
    assert mode.reads == 0
    with _ScalarReads() as mode:
        got = extras.loss_value(acc)
    assert mode.reads == 1
    assert got == expected  # bitwise: fp64 device math is the Python double math
    assert extras.loss_value(None) == 0.0


def test_loss_accumulation_keeps_the_differentiable_loss_untouched():
    w = torch.ones(3, requires_grad = True)
    loss = (w * 2).sum()
    acc = extras.accumulate_loss(None, loss, 2)
    assert not acc.requires_grad
    loss.backward()
    assert torch.equal(w.grad, torch.full((3,), 2.0))


def test_loss_accumulation_keeps_the_python_float_on_a_device_without_fp64(monkeypatch):
    from core.inference import video_minimax_h3_pdd as pdd

    class _ArcLoss:
        device = torch.device("xpu")

        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def __float__(self):
            return self.value

        def double(self):
            raise AssertionError("Required aspect fp64 is not supported on the device")

    monkeypatch.setitem(pdd._FP64_BY_DEVICE, ("xpu", None), False)
    acc = None
    for v in (0.25, 1.5, 3.0):
        acc = extras.accumulate_loss(acc, _ArcLoss(v), 4)
    assert isinstance(acc, float) and extras.loss_value(acc) == 0.25 / 4 + 1.5 / 4 + 3.0 / 4


def _ema_reference(shadows, params, decay):
    for s, p in zip(shadows, params):
        s.mul_(decay).add_(p, alpha = 1.0 - decay)


def test_lora_ema_batched_update_is_bitwise_the_per_tensor_update():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.Linear(8, 4))
    model[0].bias.requires_grad_(False)
    ema = extras.LoRAEMA(model, decay = 0.97)
    ref = {n: t.clone() for n, t in ema._shadow.items()}
    for step in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        decay = ema.effective_decay()
        names = [n for n, p in model.named_parameters() if n in ref]
        _ema_reference(
            [ref[n] for n in names],
            [dict(model.named_parameters())[n].detach() for n in names],
            decay,
        )
        ema.update(model)
        for n in ref:
            assert torch.equal(ema._shadow[n], ref[n]), (step, n)
    assert ema.updates == 5
    assert "0.bias" not in ema._shadow


def test_lora_ema_update_walks_the_parameters_once():
    model = torch.nn.Linear(4, 4)
    ema = extras.LoRAEMA(model, decay = 0.9)
    walks = []
    original = model.named_parameters

    def _counting(*a, **k):
        walks.append(1)
        return original(*a, **k)

    model.named_parameters = _counting
    for _ in range(4):
        ema.update(model)
    assert len(walks) == 1
    # A different model object rebuilds the lists.
    other = torch.nn.Linear(4, 4)
    ema.update(other)
    assert torch.equal(ema._shadow["weight"], ema._shadow["weight"])


def test_latent_cache_decodes_each_image_once_with_identical_variants(monkeypatch, tmp_path):
    from PIL import Image
    import numpy as np

    path = tmp_path / "a.png"
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (90, 70, 3), dtype = np.uint8)).save(
        path
    )
    opens = []
    real_open = dit._open_resized

    def _counting_open(p, res):
        opens.append(p)
        return real_open(p, res)

    monkeypatch.setattr(dit, "_open_resized", _counting_open)
    plan = [[(0.0, 0.0, False), (0.9, 0.9, True), (0.5, 0.1, False), (0.2, 0.7, True)]]
    got = []
    spec = types.SimpleNamespace(
        encode_latent_stats = lambda vae, px: (got.append(px.clone()) or px[:, :1], None)
    )
    cfg = types.SimpleNamespace(
        resolution = 48, center_crop = False, cache_variants = 4, random_flip = True, seed = 0
    )
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", "1")
    cache = dit._build_latent_cache(
        spec,
        None,
        [str(path)],
        cfg,
        "cpu",
        torch.float32,
        None,
        lambda: False,
        plan = plan,
    )
    assert len(cache[0]) == 4
    assert len(opens) == 1
    for px, (u_left, u_top, flip) in zip(got, plan[0]):
        expected = dit._load_pixel_tensor_planned(str(path), 48, False, u_left, u_top, flip)
        assert torch.equal(px[0], expected)


def _old_qwen_collate(
    entries,
    weight_dtype,
    pad_to = None,
):
    import torch.nn.functional as F

    seqs = [e[0].shape[1] for e in entries]
    target = max(pad_to or 0, max(seqs))
    pes, masks = [], []
    for pe, mask in entries:
        s = pe.shape[1]
        if mask is None:
            mask = torch.ones((1, s), dtype = torch.int64)
        if s < target:
            pe = F.pad(pe, (0, 0, 0, target - s))
            mask = F.pad(mask, (0, target - s))
        pes.append(pe)
        masks.append(mask)
    pe_b = torch.cat(pes).to(dtype = weight_dtype)
    mask_b = torch.cat(masks)
    if len(entries) == 1 and entries[0][1] is None and target == seqs[0]:
        mask_b = None
    return pe_b, mask_b


@pytest.mark.parametrize("pad_to", [None, 9])
@pytest.mark.parametrize("n", [1, 3])
def test_qwen_collate_pads_on_device_identically(n, pad_to):
    g = torch.Generator().manual_seed(n)
    entries = []
    for i in range(n):
        s = 4 + i
        mask = None if i % 2 == 0 else torch.ones((1, s), dtype = torch.int64)
        entries.append((torch.randn(1, s, 6, generator = g), mask))
    pe, mask = dit._qwen_collate(entries, "cpu", torch.bfloat16, pad_to = pad_to)
    ref_pe, ref_mask = _old_qwen_collate(entries, torch.bfloat16, pad_to)
    assert pe.dtype == torch.bfloat16 and torch.equal(pe, ref_pe)
    assert (mask is None) == (ref_mask is None)
    if mask is not None:
        assert torch.equal(mask, ref_mask)


def test_fixed_length_collates_match_a_host_cat():
    a = (torch.randn(1, 5, 4), torch.randn(1, 4), torch.zeros(5, 3))
    b = (torch.randn(1, 5, 4), torch.randn(1, 4), torch.zeros(5, 3))
    pe, pooled, ids = dit._flux_collate([a, b], "cpu", torch.bfloat16)
    assert torch.equal(pe, torch.cat([a[0], b[0]]).to(torch.bfloat16))
    assert torch.equal(pooled, torch.cat([a[1], b[1]]).to(torch.bfloat16))
    assert ids.dtype == torch.float32
    v = [(torch.randn(1, 3, 2), torch.randn(1, 3, 2), torch.ones(1, 3)) for _ in range(2)]
    video, audio, mask = dit._ltx2_collate(v, "cpu", torch.bfloat16)
    assert torch.equal(video, torch.cat([e[0] for e in v]).to(torch.bfloat16))
    assert torch.equal(mask, torch.cat([e[2] for e in v]))


def test_caption_embeds_pin_only_on_cuda_and_under_budget(monkeypatch):
    embeds = {"a": (torch.zeros(1, 2), torch.zeros(1))}
    assert dit._pin_caption_embeds(embeds, "cpu") is embeds
    pinned = []
    monkeypatch.setattr("core.inference.host_pinning.host_pinning_allowed", lambda *a: True)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: pinned.append(1) or self)
    out = dit._pin_caption_embeds(embeds, "cuda")
    assert len(pinned) == 2 and set(out) == {"a"} and isinstance(out["a"], tuple)
    monkeypatch.setattr(dit, "_CAPTION_PIN_BUDGET_BYTES", 1)
    pinned.clear()
    assert dit._pin_caption_embeds(embeds, "cuda") is embeds and not pinned


def test_sampled_timesteps_and_sigmas_are_unchanged(monkeypatch):
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(shift = 3.0)
    torch.manual_seed(5)
    ts, idx = dit._sample_timesteps(sched, 4, "cpu")
    torch.manual_seed(5)
    _ts, _idx = dit._sample_timesteps(sched, 4, "cpu")
    assert torch.equal(idx, _idx)
    assert torch.equal(ts, sched.timesteps[idx])
    sig = dit._gather_sigmas(sched.sigmas, idx, "cpu", torch.bfloat16, 4)
    assert sig.shape == (4, 1, 1, 1)
    assert torch.equal(sig.flatten(), sched.sigmas[idx].to(torch.bfloat16))


def test_conditioner_load_keeps_the_vae_on_the_host():
    moves = []

    class _Vae:
        def to(self, *args, **kwargs):
            moves.append((args, kwargs))
            return self

    class _PipeCls:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return types.SimpleNamespace(vae = _Vae())

    cfg = types.SimpleNamespace(base_model = "x", hf_token = None)
    pipe, vae = dit._load_pipe_without_transformer(_PipeCls, cfg, "cuda")
    assert moves == [((), {"dtype": torch.float32})]


def test_int8_base_streams_onto_the_device(monkeypatch):
    seen = {}

    def _fake_quantize(
        model,
        config,
        filter_fn = None,
        device = None,
    ):
        seen["device"] = device

    fake = types.ModuleType("torchao.quantization")
    fake.Int8WeightOnlyConfig = lambda: object()
    fake.quantize_ = _fake_quantize
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake)
    model = torch.nn.Linear(8, 8)
    dit._int8_quantize_base(model, None, device = "cpu")
    assert seen["device"] == "cpu"
    # The legacy call (no device) stays a pure in-place quantize.
    seen.clear()
    dit._int8_quantize_base(model, None)
    assert seen["device"] is None

    loads = []

    class _Model:
        def to(self, device):
            loads.append(device)
            return self

    class _Cls:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Model()

    cfg = types.SimpleNamespace(base_model = "x", hf_token = None, resolved_family = None)
    dit._load_dit_transformer(_Cls, cfg, "cuda", "int8")
    assert loads == []
    dit._load_dit_transformer(_Cls, cfg, "cuda", "bf16")
    assert loads == ["cuda"]


def _tiny_qwen3_vl(num_layers):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers without Qwen3-VL")
    config = transformers.Qwen3VLConfig(
        text_config = dict(
            vocab_size = 64,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = num_layers,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config = dict(
            depth = 1,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 4,
            out_hidden_size = 32,
            deepstack_visual_indexes = [0],
        ),
    )
    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration(config).eval()


@pytest.mark.parametrize("layer", [1, 3, 5])
def test_h3_text_encoder_early_exit_is_bitwise_and_skips_the_tail(layer):
    from core.training.diffusion_h3_trainer import _early_exit_hidden_state

    model = _tiny_qwen3_vl(6)
    ids = torch.randint(0, 64, (1, 9), generator = torch.Generator().manual_seed(1))
    kwargs = dict(input_ids = ids, attention_mask = torch.ones_like(ids), use_cache = False)
    layers = model.model.language_model.layers
    ran = []
    hooks = [
        m.register_forward_pre_hook(lambda _m, _a, i = i: ran.append(i)) for i, m in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            full = model.model(**kwargs, output_hidden_states = True).hidden_states[layer]
            ran.clear()
            fast = _early_exit_hidden_state(model, layer, kwargs)
            assert ran == list(range(layer))
            assert torch.equal(fast, full)
            # Repeat: a raised early exit must not leave the encoder patched.
            assert torch.equal(_early_exit_hidden_state(model, layer, kwargs), full)
            assert len(model.model(**kwargs, output_hidden_states = True).hidden_states) == 7
    finally:
        for h in hooks:
            h.remove()
    # Out of range (would need the normed last state): the caller runs the full encoder instead.
    assert _early_exit_hidden_state(model, 6, kwargs) is None


def test_h3_vaes_reach_the_device_only_after_the_conditioner_is_freed(monkeypatch):
    from core.training import diffusion_h3_trainer as h3

    order = []

    class _Placed:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            order.append((self.name, device))
            return self

    class _Pipe:
        def __init__(self):
            self.text_encoder = _Placed("text_encoder")
            self.vae = _Placed("vae")
            self.audio_vae = _Placed("audio_vae")

        def load_components(self, **_k):
            pass

    class _Modular:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Pipe()

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(ModularPipeline = _Modular))
    monkeypatch.setattr(h3, "_assert_component_grid", lambda pipe: None)
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: "/hub")
    h3._load_conditioners(types.SimpleNamespace(base_model = "m", hf_token = None), "cuda")
    assert order == [("text_encoder", "cuda")]

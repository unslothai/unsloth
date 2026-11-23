# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 flex attention (unsloth/models/inkling2_flex.py) against transformers' eager attention.

CPU only: flex has no CPU backward kernel, so the tests run its reference math (the HOP's dense path the CUDA kernels
are checked against) with the device check lifted. The kernels themselves are covered by the GPU runs in the PR."""

import importlib.util
import pathlib
import re
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
flex_mod = pytest.importorskip("torch.nn.attention.flex_attention")

_ROOT = pathlib.Path(__file__).resolve().parents[1]


_MODULE = []


@pytest.fixture
def fx(monkeypatch):
    # One module instance: the classes it wraps keep pointing at its counters.
    if not _MODULE:
        path = _ROOT / "unsloth" / "models" / "inkling2_flex.py"
        spec = importlib.util.spec_from_file_location("_unsloth_inkling2_flex", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE.append(mod)
    mod = _MODULE[0]
    # Registered while a test runs: code dynamo compiles from it (flex, the whole-model trace) re-imports the module
    # by name for its guards.
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    if not mod.available():
        pytest.skip("flex_attention with captured-buffer gradients needs torch >= 2.7")
    monkeypatch.setattr(flex_mod, "_validate_device", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_ALLOW_CPU", True)
    monkeypatch.delenv(mod.ENV, raising = False)
    return mod


def _reference(q, k, v, rel, sink, scale, window, extent, keep):
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    L = q.shape[2]
    i = torch.arange(L)
    d = i[:, None] - i[None, :]
    s = q @ k.transpose(-1, -2) * scale
    bias = (
        rel.transpose(1, 2)
        .gather(-1, d.clamp(0, extent - 1).expand(*s.shape))
        .masked_fill((d < 0) | (d >= extent), 0)
    )
    visible = (d >= 0) & ((d < window) if window else True) & keep[:, None, None, :]
    s = (s + bias).masked_fill(~visible, torch.finfo(s.dtype).min)
    p = torch.softmax(torch.cat([s, sink.transpose(1, 2)[..., None]], -1), -1)[..., :-1]
    return (p @ v).transpose(1, 2)


@pytest.mark.parametrize("window", [None, 16])
def test_kernel_math_matches_eager_formula(fx, window):
    torch.manual_seed(0)
    B, H, KV, L, D, E = (
        2,
        4,
        2,
        150,
        8,
        24,
    )  # L > 128: two sparse blocks; E < L: bias stops at the extent
    q = torch.randn(B, H, L, D, dtype = torch.float64, requires_grad = True)
    k = torch.randn(B, KV, L, D, dtype = torch.float64, requires_grad = True)
    v = torch.randn(B, KV, L, D, dtype = torch.float64, requires_grad = True)
    rel = torch.randn(B, L, H, E, dtype = torch.float64, requires_grad = True)
    sink = torch.randn(B, L, H, dtype = torch.float64, requires_grad = True)
    keep = torch.ones(B, L, dtype = torch.bool)
    keep[1, :7] = False  # left padding: rows 0..6 of batch 1 see nothing

    def mask_mod(b, h, qi, ki):
        ok = (ki <= qi) & keep[b, ki]
        return ok & (ki > qi - window) if window else ok

    block_mask = flex_mod.create_block_mask(mask_mod, B, None, L, L, device = "cpu", _compile = False)
    out = fx.inkling2_flex_attention(q, k, v, block_mask, rel, sink, D**-0.5, E)
    ref = _reference(q, k, v, rel, sink, D**-0.5, window, E, keep)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol = 1e-10, atol = 1e-10)
    grad = torch.randn_like(ref)
    got = torch.autograd.grad((out * grad).sum(), (q, k, v, rel, sink))
    want = torch.autograd.grad((ref * grad).sum(), (q, k, v, rel, sink))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol = 1e-9, atol = 1e-10)


def _tiny(loop):
    from transformers import AutoModelForImageTextToText
    from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

    text = dict(
        vocab_size = 256,
        unpadded_vocab_size = 250,
        hidden_size = 32,
        num_hidden_layers = 12,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        swa_num_attention_heads = 4,
        swa_num_key_value_heads = 2,
        swa_head_dim = 8,
        d_rel = 4,
        rel_extent = 12,
        sliding_window_size = 6,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        moe_intermediate_size = 16,
        attention_dropout = 0.0,
    )
    vision = dict(
        model_type = "inkling2_vision",
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        projection_hidden_dim = 32,
        text_hidden_size = 32,
    )
    if loop:
        text.update(
            loop_layer_index_start = 6, loop_layer_index_end = 12, loop_num_passes = 2, loop_gate_rank = 4
        )
        config = Inkling2LoopConfig(text_config = text, vision_config = vision)
    else:
        config = Inkling2Config(text_config = text, vision_config = vision)
    torch.manual_seed(0)
    # float64 and the reference expert loop (grouped_mm has no float64 kernel): both paths are exact up to rounding.
    model = AutoModelForImageTextToText.from_config(
        config, dtype = torch.float64, experts_implementation = "eager"
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.is_floating_point():
                noise = torch.randn(p.shape, generator = generator, dtype = torch.float64)
                p.copy_(
                    noise * (0.5 if p.ndim == 1 else 0.2)
                    + (1.0 if p.ndim == 1 and "norm" in name else 0.0)
                )
    return model


def _inputs(mode, L):
    generator = torch.Generator().manual_seed(L)
    if mode == "packed":
        ids = torch.randint(3, 250, (1, L), generator = generator)
        cut = L // 3
        position_ids = torch.cat([torch.arange(cut), torch.arange(L - cut)])[None]
        return dict(input_ids = ids, position_ids = position_ids), torch.ones(1, L, dtype = torch.bool)
    ids = torch.randint(3, 250, (2, L), generator = generator)
    mask = torch.ones_like(ids)
    if mode == "padded":
        mask[0, :3] = 0
        mask[1, -4:] = 0
    return dict(input_ids = ids, attention_mask = mask), mask.bool()


def _rel(a, b):
    # In float64: some gradients are float32 (e.g. a loop gate that gets exactly zero on this input), where the
    # 1e-300 floor rounds to 0 and 0 / 0 would put a NaN into the sorted error list.
    a, b = a.double(), b.double()
    diff = float((a - b).norm())
    return 0.0 if diff == 0 else diff / max(float(b.norm()), 1e-300)


def _assert_matches(flex_logits, eager_logits, flex_grads, eager_grads):
    # The fp32-strict islands (router, convolutions, residual queries, message gates) round at 1e-7 even in a float64
    # model, so this is the agreement bound, not exactness. A wrong bias / sink / mask term is off by O(1).
    assert _rel(flex_logits, eager_logits) < 1e-5
    # The random init can overflow the fp32 message residual of the loop: skip tensors eager itself gets non-finite.
    finite = {n for n in eager_grads if torch.isfinite(eager_grads[n]).all()}
    assert all(torch.isfinite(flex_grads[n]).all() for n in finite)
    errors = sorted((_rel(flex_grads[n], eager_grads[n]), n) for n in finite)
    assert errors[-1][0] < 1e-4, errors[-3:]
    assert errors[len(errors) // 2][0] < 1e-5, errors[len(errors) // 2]
    # The tensors the flex path feeds directly (the fp32 convolutions inside k_proj / v_proj round like the islands).
    direct = re.compile(
        r"self_attn\.(q_proj\.linear|k_proj\.linear|v_proj\.linear|o_proj|sink_proj|relative_position_bias)"
    )
    attention = [e for e in errors if direct.search(e[1])]
    assert attention and max(attention)[0] < 2e-5, max(attention)


def _forward_backward(model, inputs, valid):
    model.zero_grad(set_to_none = True)
    logits = model(**inputs, use_cache = False).logits
    loss = (logits.logsumexp(-1) * valid).sum() / valid.sum()
    loss.backward()
    grads = {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None and "vision" not in n
    }
    return logits.detach(), grads


@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("mode,L", [("plain", 5), ("plain", 30), ("padded", 30), ("packed", 30)])
def test_model_matches_eager(fx, loop, mode, L):
    model = _tiny(loop)
    model.train()
    assert fx.enable_inkling2_flex(model, verbose = False), fx._FAILED
    text_config = model.config.text_config
    inputs, valid = _inputs(mode, L)

    text_config._attn_implementation = "eager"
    fx.reset_stats()
    eager_logits, eager_grads = _forward_backward(model, inputs, valid)
    assert fx.stats()["flex"] == 0

    text_config._attn_implementation = fx.ATTN_KEY
    fx.reset_stats()
    flex_logits, flex_grads = _forward_backward(model, inputs, valid)
    stats = fx.stats()
    assert (
        stats["flex"] == text_config.num_hidden_layers + (2 - 1) * 6 * loop and stats["eager"] == 0
    ), stats

    assert set(flex_grads) == set(eager_grads)
    _assert_matches(flex_logits[valid], eager_logits[valid], flex_grads, eager_grads)
    assert any("relative_position_bias.proj" in n for n in eager_grads)
    assert any("sink_proj" in n for n in eager_grads)


@pytest.mark.parametrize("reentrant", [False, True])
def test_gradient_checkpointing_matches_eager(fx, reentrant):
    model = _tiny(True)
    model.train()
    assert fx.enable_inkling2_flex(model, verbose = False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": reentrant})
    model.enable_input_require_grads()
    inputs, valid = _inputs("padded", 30)
    text_config = model.config.text_config
    text_config._attn_implementation = "eager"
    eager_logits, eager_grads = _forward_backward(model, inputs, valid)
    text_config._attn_implementation = fx.ATTN_KEY
    fx.reset_stats()
    flex_logits, flex_grads = _forward_backward(model, inputs, valid)
    if not reentrant:
        assert fx.stats()["recomputed"] > 0
    _assert_matches(flex_logits[valid], eager_logits[valid], flex_grads, eager_grads)


def test_accelerate_hooked_layers_take_flex(fx):
    # A model split over GPUs runs each layer through accelerate's hook, which calls the forward bound at load.
    hooks = pytest.importorskip("accelerate.hooks")
    model = _tiny(True)
    model.train()
    for module in model.modules():
        if type(module).__name__.endswith(("SlidingAttention", "FullAttention", "TextModel")):
            hooks.add_hook_to_module(module, hooks.ModelHook())
    assert fx.enable_inkling2_flex(model, verbose = False)
    inputs, valid = _inputs("plain", 30)
    model.config.text_config._attn_implementation = fx.ATTN_KEY
    fx.reset_stats()
    _forward_backward(model, inputs, valid)
    stats = fx.stats()
    assert stats["flex"] == 18 and stats["eager"] == 0 and stats["distance_skipped"] == 2, stats


def test_decode_keeps_eager_and_generation_matches(fx):
    model = _tiny(True)
    model.eval()
    assert fx.enable_inkling2_flex(model, verbose = False)
    ids = torch.randint(3, 250, (1, 20), generator = torch.Generator().manual_seed(1))
    out = {}
    for impl in ("eager", fx.ATTN_KEY):
        model.config.text_config._attn_implementation = impl
        fx.reset_stats()
        with torch.no_grad():
            out[impl] = (
                model.generate(input_ids = ids, max_new_tokens = 6, do_sample = False),
                fx.stats(),
            )
    assert torch.equal(out["eager"][0], out[fx.ATTN_KEY][0])
    # Prefill on flex, every one-token decode step on eager.
    assert out[fx.ATTN_KEY][1]["flex"] > 0 and out[fx.ATTN_KEY][1]["eager"] > 0


def test_kill_switch_and_fingerprint(fx, monkeypatch):
    model = _tiny(True)
    monkeypatch.setenv(fx.ENV, "0")
    assert not fx.enable_inkling2_flex(model, verbose = False)
    assert model.config.text_config._attn_implementation != fx.ATTN_KEY
    monkeypatch.delenv(fx.ENV)
    assert fx._patch_classes(model) is None, fx._FAILED


def test_changed_modeling_code_keeps_eager(fx, monkeypatch):
    import transformers.models.inkling2.modeling_inkling2 as modeling

    model = _tiny(False)
    original = modeling.Inkling2RelativePositionBias.forward

    def forward(self, hidden_states, relative_distance, position_ids):
        return original(self, hidden_states, relative_distance, position_ids) * 2

    monkeypatch.setattr(modeling.Inkling2RelativePositionBias, "forward", forward)
    assert not fx.enable_inkling2_flex(model, verbose = False)
    assert model.config.text_config._attn_implementation != fx.ATTN_KEY


@pytest.mark.parametrize(
    "capability,mode,min_length,expected",
    [
        ((7, 5), "auto", None, False),  # T4: flex measured 1.7x slower, eager stays
        ((8, 9), "auto", None, True),  # L4: flex measured 15% faster
        ((9, 0), None, None, True),  # unset = auto
        ((7, 5), "1", None, True),
        ((8, 0), "0", None, False),
        ((7, 5), "auto", "512", True),
        ((8, 0), "auto", "4096", False),
    ],
)
def test_auto_policy(fx, monkeypatch, capability, mode, min_length, expected):
    monkeypatch.setattr(fx, "_capability", lambda device: capability)
    # NVIDIA capabilities: on a ROCm build torch.version.hip is set and auto keeps eager (test_auto_policy_keeps_rocm_eager).
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    if mode is None:
        monkeypatch.delenv(fx.ENV, raising = False)
    else:
        monkeypatch.setenv(fx.ENV, mode)
    if min_length is None:
        monkeypatch.delenv(fx.MIN_LEN_ENV, raising = False)
    else:
        monkeypatch.setenv(fx.MIN_LEN_ENV, min_length)
    assert fx._wants_flex(1024, 1024, 0, 0, "cuda") is expected
    # Decode steps and cached continuations never take flex.
    assert fx._wants_flex(1, 1025, 1024, 0, "cuda") is False


@pytest.mark.parametrize("mode,expected", [("auto", False), (None, False), ("1", True)])
def test_auto_policy_keeps_rocm_eager(fx, monkeypatch, mode, expected):
    # ROCm reports gfx numbers as capability (gfx1151 = (11, 5)); the dynamic-shape compiled flex is unmeasured there and
    # gave NaN LoRA gradients on gfx1151 after a recompile (unsloth-zoo#2663), so only an explicit "1" takes it.
    monkeypatch.setattr(fx, "_capability", lambda device: (11, 5))
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    monkeypatch.delenv(fx.MIN_LEN_ENV, raising = False)
    if mode is None:
        monkeypatch.delenv(fx.ENV, raising = False)
    else:
        monkeypatch.setenv(fx.ENV, mode)
    assert fx._wants_flex(1024, 1024, 0, 0, "cuda") is expected


@pytest.mark.parametrize("loop", [True, False])
def test_output_attentions_keeps_eager_maps(fx, loop):
    # flex never forms the attention probabilities: a forward that collects them keeps eager attention, so every
    # language layer still returns its map (transformers drops a layer's None from `attentions`).
    model = _tiny(loop)
    model.eval()
    assert fx.enable_inkling2_flex(model, verbose = False)
    inputs, _ = _inputs("padded", 9)
    text_config = model.config.text_config
    out = {}
    for impl in ("eager", fx.ATTN_KEY):
        text_config._attn_implementation = impl
        fx.reset_stats()
        with torch.no_grad():
            out[impl] = (model(**inputs, use_cache = False, output_attentions = True), fx.stats())
    eager, (flex, stats) = out["eager"][0], out[fx.ATTN_KEY]
    assert len(eager.attentions) == text_config.num_hidden_layers + (2 - 1) * 6 * loop
    assert len(flex.attentions) == len(eager.attentions), stats
    assert stats["flex"] == 0 and stats["block_masks"] == 0, stats
    for a, b in zip(eager.attentions, flex.attentions):
        assert torch.equal(a, b)
    assert torch.equal(eager.logits, flex.logits)
    # Without the request the same model takes flex.
    fx.reset_stats()
    with torch.no_grad():
        model(**inputs, use_cache = False)
    assert fx.stats()["flex"] > 0


def test_whole_model_compile_traces_eager_attention(fx):
    # torch.compile(model, fullgraph = True) over the whole model: stock eager Inkling 2 traces in one graph, so the
    # flex wrappers (ContextVar state, the eager-only mask builder) step aside inside that trace.
    model = _tiny(False)
    model.eval()
    if not hasattr(model, "set_experts_implementation"):
        pytest.skip("no set_experts_implementation")
    # The tiny model's eager expert loop (.nonzero) cannot be captured in one graph; the grouped one (no float64) can.
    model.set_experts_implementation("grouped_mm")
    model.float()
    assert fx.enable_inkling2_flex(model, verbose = False)
    ids = torch.randint(3, 250, (1, 9), generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        model.config.text_config._attn_implementation = "eager"
        want = model(input_ids = ids, use_cache = False).logits
        model.config.text_config._attn_implementation = fx.ATTN_KEY
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda x: model(input_ids = x, use_cache = False).logits, fullgraph = True
        )
        got = compiled(ids)
    torch._dynamo.reset()
    assert fx._TRACING_TEXT_FORWARD == [False]
    # Inductor fusion rounding in float32; a wrong mask or bias term is off by O(1).
    torch.testing.assert_close(got, want, rtol = 1e-4, atol = 1e-4)


def test_auto_policy_mixed_cards_keep_eager(fx, monkeypatch):
    # The mask takes flex from the embedding device: embeddings on an L4 and layers on a T4 would run flex on the T4.
    monkeypatch.setattr(fx, "_ALLOW_CPU", False)
    monkeypatch.delenv(fx.MIN_LEN_ENV, raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    capability = {0: (8, 9), 1: (7, 5)}
    monkeypatch.setattr(fx, "_capability", lambda device: capability[device.index])
    model = _tiny(False)
    devices = {torch.device("cuda", 0), torch.device("cuda", 1)}
    monkeypatch.setattr(fx, "_cuda_devices", lambda m: devices)
    assert not fx.enable_inkling2_flex(model, verbose = False)
    assert model.config.text_config._attn_implementation != fx.ATTN_KEY
    monkeypatch.setenv(fx.ENV, "1")  # forcing still takes it
    assert fx.enable_inkling2_flex(model, verbose = False)
    fx.disable_inkling2_flex(model)
    monkeypatch.delenv(fx.ENV)
    capability[1] = (8, 0)
    assert fx.enable_inkling2_flex(model, verbose = False)
    fx.disable_inkling2_flex(model)


def test_rocm_compiles_flex_with_static_shapes(fx, monkeypatch):
    # A forced ROCm run: dynamic-shape compiled flex gave NaN LoRA gradients on gfx1151 after a recompile
    # (unsloth-zoo#2663), dynamic = False was clean.
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or fn)
    for hip, expected in (("6.4.0", False), (None, None)):
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(fx, "_COMPILED_FLEX", None)
        fx._flex_function(torch.device("cuda", 0))
        assert calls[-1]["dynamic"] is expected

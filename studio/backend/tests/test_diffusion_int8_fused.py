# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_int8_fused.py``. The kernel / model tests need CUDA, Triton and torchao's Int8Tensor."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_fused as fused

torch = pytest.importorskip("torch")


def _cuda_int8_ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig  # noqa: F401
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return fused._device_ok(torch.cuda.current_device())


needs_cuda = pytest.mark.skipif(
    not _cuda_int8_ready(), reason = "needs CUDA (not ROCm), Triton and torchao Int8Tensor"
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fused.INT8_FUSED_ENV, raising = False)
    yield


@pytest.mark.parametrize(
    "version, ok",
    [
        ("3.7.1", True),
        ("3.2.0", True),
        ("3.1.0", False),
        ("2.3.1", False),
        ("garbage", False),
        ("4.0", True),
    ],
)
def test_triton_version_gate(version, ok):
    assert fused._triton_version_ok(version) is ok


@pytest.mark.parametrize(
    "value, off", [("0", True), ("off", True), ("false", True), ("1", False), ("", False)]
)
def test_kill_switch(monkeypatch, value, off):
    monkeypatch.setenv(fused.INT8_FUSED_ENV, value)
    assert fused.int8_fused_disabled() is off


def test_install_noop_when_disabled(monkeypatch):
    monkeypatch.setenv(fused.INT8_FUSED_ENV, "0")
    mod = torch.nn.Sequential(torch.nn.Linear(8, 8))
    assert fused.install(mod) == 0
    assert "forward" not in mod[0].__dict__


def test_plain_tensor_is_not_eligible():
    assert fused._plain_int8_weight(torch.zeros(8, 8)) is False
    assert fused._plain_int8_weight(None) is False


def test_install_noop_on_cpu_module():
    FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward

    ff = FeedForward(64, 64, activation_fn = "gelu-approximate")
    assert fused.install(ff) == 0
    assert "forward" not in ff.__dict__


def _rand_inputs(
    m,
    n,
    *,
    ws_dtype = torch.float32,
    bias = True,
    seed = 0,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    c = torch.randint(-(2**20), 2**20, (m, n), generator = g, dtype = torch.int32).cuda()
    xs = (torch.rand(m, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16).float().cuda()
    ws = (torch.rand(n, generator = g) * 1e-4 + 1e-6).to(ws_dtype).cuda()
    b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).cuda() if bias else None
    return c, xs, ws, b


@needs_cuda
@pytest.mark.parametrize(
    "m, n, ws_dtype, bias",
    [
        (4096, 12288, torch.float32, True),  # Qwen-Image / FLUX.1 MLP
        (4101, 3000, torch.float32, True),  # odd rows, N not a multiple of the chunk
        (512, 12288, torch.bfloat16, False),  # bf16 weight scale, no bias
        (17, 8, torch.float32, True),  # smallest eligible
    ],
)
def test_kernel_bit_exact_vs_eager_reference(m, n, ws_dtype, bias):
    c, xs, ws, b = _rand_inputs(m, n, ws_dtype = ws_dtype, bias = bias)
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert torch.equal(s, s_ref)
    assert torch.equal(q, q_ref)


@needs_cuda
def test_kernel_small_activations_take_the_exact_amax_path():
    # Every pre-activation negative: max|gelu| comes from the negative lobe, not gelu(max y).
    c, xs, ws, b = _rand_inputs(256, 1024, bias = False)
    c = -c.abs() - 1
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


@needs_cuda
@pytest.mark.parametrize("transposed", [False, True])
def test_kernel_prefix_segment(transposed):
    bsz, seq, heads, hd, n = 2, 300, 4, 64, 1024
    c, xs, ws, b = _rand_inputs(bsz * seq, n)
    if transposed:  # SDPA output layout [B, H, S, D] seen as [B, S, H, D]
        prefix = (
            (torch.randn(bsz, heads, seq, hd, device = "cuda") * 3).to(torch.bfloat16).transpose(1, 2)
        )
    else:
        prefix = (torch.randn(bsz, seq, heads, hd, device = "cuda") * 3).to(torch.bfloat16)
    q, s = fused._launch(c, xs, ws, b, prefix)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, prefix)
    assert q.shape == (bsz * seq, heads * hd + n)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


def _quantized_ff(
    dim = 256,
    inner = 1024,
    seed = 0,
):
    FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(seed)
    ff = (
        FeedForward(dim, dim, mult = inner // dim, activation_fn = "gelu-approximate")
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    for p in ff.parameters():
        p.data.normal_(0, 0.05)
    quantize_(ff, Int8DynamicActivationInt8WeightConfig())
    return ff


@needs_cuda
def test_feedforward_bit_identical_to_stock_eager():
    ff = _quantized_ff()
    x = torch.randn(2, 300, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = ff(x)
        assert fused.install(ff) == 1
        out = ff(x)
    assert torch.equal(out, ref)


@needs_cuda
def test_feedforward_small_m_keeps_stock_path():
    ff = _quantized_ff()
    x = torch.randn(1, 8, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = ff(x)
        fused.install(ff)
        out = ff(x)
    assert torch.equal(out, ref)


@needs_cuda
def test_feedforward_compiles_without_graph_break():
    ff = _quantized_ff()
    x = torch.randn(1, 512, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        eager = ff(x)
        torch._dynamo.reset()
        stock_compiled = torch.compile(ff, fullgraph = True)(x)
        fused.install(ff)
        torch._dynamo.reset()
        out = torch.compile(ff, fullgraph = True)(x)
    _assert_within_compile_floor(out, stock_compiled, eager)


@needs_cuda
def test_install_is_idempotent_and_holds_no_global_reference():
    import gc
    import weakref

    ff = _quantized_ff()
    assert fused.install(ff) == 1
    assert fused.install(ff) == 1
    ref = weakref.ref(ff)
    del ff
    gc.collect()
    assert ref() is None


@needs_cuda
def test_uninstall_restores_stock_forward():
    ff = _quantized_ff()
    fused.install(ff)
    assert "forward" in ff.__dict__
    fused.uninstall(ff)
    assert "forward" not in ff.__dict__
    assert not fused.is_installed(ff)


@needs_cuda
def test_flux_single_block_bit_identical_to_stock_eager():
    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(0)
    blk = (
        tf.FluxSingleTransformerBlock(dim = 256, num_attention_heads = 4, attention_head_dim = 64)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    quantize_(
        blk,
        Int8DynamicActivationInt8WeightConfig(),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and "norm" not in fqn,
    )
    hid = torch.randn(1, 200, 256, device = "cuda", dtype = torch.bfloat16)
    enc = torch.randn(1, 40, 256, device = "cuda", dtype = torch.bfloat16)
    temb = torch.randn(1, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = blk(hid, enc, temb)
        assert fused.install(blk) == 1
        out = blk(hid, enc, temb)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])


@needs_cuda
@pytest.mark.parametrize(
    "m, n, gate_col, value_col", [(4352, 10240, 0, 10240), (300, 1024, 1024, 0), (17, 64, 0, 64)]
)
def test_swiglu_kernel_bit_exact_vs_eager_reference(m, n, gate_col, value_col):
    c, xs, ws, b = _rand_inputs(m, 2 * n, bias = False)
    q, s = fused._launch_swiglu(c, xs, ws, None, gate_col, value_col, n)
    q_ref, s_ref = fused.reference_dq_swiglu_quant(c, xs, ws, None, gate_col, value_col, n)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


def _quantize(module):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    for p in module.parameters():
        p.data.normal_(0, 0.05)
    quantize_(module, Int8DynamicActivationInt8WeightConfig())
    return module


@needs_cuda
@pytest.mark.parametrize("kind", ["diffusers_swiglu", "zimage", "flux2", "qwenimage21"])
def test_swiglu_mlps_bit_identical_to_stock_eager(kind, monkeypatch):
    # Kernel exactness on every SwiGLU layout, including the ones the quality gate keeps on the stock path.
    monkeypatch.setattr(fused, "_SWIGLU_ALL_LAYOUTS", True)
    torch.manual_seed(0)
    if kind == "diffusers_swiglu":
        FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward
        ff = FeedForward(256, inner_dim = 512, activation_fn = "swiglu", bias = False)
    elif kind == "zimage":
        zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")
        ff = zmod.FeedForward(256, 512)
    elif kind == "flux2":
        f2 = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
        ff = f2.Flux2FeedForward(256, inner_dim = 512)
    else:
        q21 = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
        ff = q21.QwenImage21SwiGLUFeedForward(256, 512)
    ff = _quantize(ff.cuda().to(torch.bfloat16).eval())
    x = torch.randn(2, 150, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = ff(x)
        torch._dynamo.reset()
        stock_compiled = torch.compile(ff, fullgraph = True)(x)
        assert fused.install(ff) == 1
        out = ff(x)
        torch._dynamo.reset()
        compiled = torch.compile(ff, fullgraph = True)(x)
    assert torch.equal(out, ref)
    _assert_within_compile_floor(compiled, stock_compiled, ref)


def _assert_within_compile_floor(compiled, stock_compiled, eager):
    # Inductor's own act-quant codegen is not eager-exact, so the bar is the stock compile's distance from eager.
    floor = (stock_compiled.float() - eager.float()).abs()
    ours = (compiled.float() - eager.float()).abs()
    assert ours.max().item() <= 1.5 * floor.max().item() + 1e-6
    assert ours.mean().item() <= 1.1 * floor.mean().item() + 1e-6


@needs_cuda
def test_cpu_placed_model_is_swapped_at_the_first_forward():
    # Studio runs the speed optims BEFORE placement: install() on CPU weights defers to the first call.
    ff = _quantized_ff().cpu()
    assert fused.install(ff) == 1
    assert not fused.is_installed(ff)
    ff = ff.cuda()
    x = torch.randn(1, 64, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ff(x)
    assert fused.is_installed(ff)
    assert (
        "_unsloth_first_call_hooks" in ff.__dict__ and not ff.__dict__["_unsloth_first_call_hooks"]
    )


def test_offload_skips_install():
    assert fused.install(torch.nn.Linear(8, 8), offload_active = True) == 0


@needs_cuda
@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
def test_convrot_linears_keep_the_stock_forward(kind, monkeypatch):
    # MiniMax-H3's hosted int8 checkpoint swaps its MLP Linears onto ConvRotLinear, which rotates the input by a block
    # Hadamard before the GEMM. The fused forward calls _int_mm on the weight directly and would skip the rotation.
    from core.inference.diffusion_convrot import _install_rotation

    if kind == "gelu":
        ff = _quantized_ff()
        lins = (ff.net[0].proj, ff.net[2])
    else:
        FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward

        torch.manual_seed(0)
        ff = _quantize(
            FeedForward(256, inner_dim = 512, activation_fn = "swiglu", bias = False)
            .cuda()
            .to(torch.bfloat16)
            .eval()
        )
        lins = (ff.net[0].proj, ff.net[2])
    monkeypatch.setattr(
        fused, "_SWIGLU_ALL_LAYOUTS", True
    )  # the layout gate alone would already refuse diffusers SwiGLU
    for lin in lins:
        _install_rotation(lin, 16)
    assert fused.install(ff) == 0
    assert not fused.is_installed(ff)


def _swiglu_module(kind):
    if kind == "diffusers_swiglu":
        FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward
        return FeedForward(256, inner_dim = 512, activation_fn = "swiglu", bias = False)
    if kind == "zimage":
        return pytest.importorskip("diffusers.models.transformers.transformer_z_image").FeedForward(
            256, 512
        )
    if kind == "flux2":
        return pytest.importorskip(
            "diffusers.models.transformers.transformer_flux2"
        ).Flux2FeedForward(256, inner_dim = 512)
    return pytest.importorskip(
        "diffusers.models.transformers.transformer_qwenimage21"
    ).QwenImage21SwiGLUFeedForward(256, 512)


@pytest.mark.parametrize("kind", ["diffusers_swiglu", "zimage", "flux2", "qwenimage21"])
def test_swiglu_quality_gate_allows_zimage_only(kind):
    # FLUX.2 and Qwen-Image-2.1 drifted further from bf16 with the fused SwiGLU than the stock compiled path does
    # (see _SWIGLU_ALL_LAYOUTS). Device-free: the gate is a layout check, the candidate count drives the deferred swap.
    ff = _swiglu_module(kind)
    assert fused._swiglu_candidate(ff) is (kind == "zimage")
    assert fused._swiglu_layout_allowed(ff) is (kind == "zimage")


@needs_cuda
@pytest.mark.parametrize("kind", ["zimage", "flux2", "qwenimage21"])
def test_swiglu_install_count_follows_the_quality_gate(kind):
    torch.manual_seed(0)
    ff = _quantize(_swiglu_module(kind).cuda().to(torch.bfloat16).eval())
    assert fused.install(ff) == (1 if kind == "zimage" else 0)
    assert fused.is_installed(ff) is (kind == "zimage")


def test_ineligible_resident_model_never_probes_the_kernels(monkeypatch):
    # A bf16 / fp16 DiT already on the GPU has nothing to fuse: the Triton validation launch must not run.
    probed = []
    monkeypatch.setattr(fused, "resident_cuda_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(fused, "_device_ok", lambda index: probed.append(index) or True)
    mod = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
    assert fused.install(mod) == 0
    assert probed == []
    assert "forward" not in mod[0].__dict__


def test_uninstall_cancels_the_deferred_install(monkeypatch):
    mod = torch.nn.Sequential(torch.nn.Linear(8, 8))
    target = mod[0]
    monkeypatch.setattr(fused, "_ff_eligible", lambda m: m is target)
    finalized = []
    monkeypatch.setattr(fused, "_finalize", lambda t, logger = None: finalized.append(t) or 1)
    assert fused.install(mod) == 1
    assert "int8_fused" in mod.__dict__["_unsloth_first_call_hooks"]
    fused.uninstall(mod)
    mod(torch.randn(2, 8))
    assert finalized == []
    assert "int8_fused" not in mod.__dict__.get("_unsloth_first_call_hooks", {})
    assert not mod._forward_pre_hooks


def _tiny_flux():
    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    torch.manual_seed(0)
    return tf.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 4,
        num_layers = 1,
        num_single_layers = 3,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        pooled_projection_dim = 32,
        axes_dims_rope = (4, 6, 6),
    ).eval()


def _flux_inputs():
    g = torch.Generator().manual_seed(1)
    return dict(
        hidden_states = torch.randn(1, 16, 4, generator = g),
        encoder_hidden_states = torch.randn(1, 8, 32, generator = g),
        pooled_projections = torch.randn(1, 32, generator = g),
        timestep = torch.tensor([1.0]),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(8, 3),
        return_dict = False,
    )


def _fake_cuda_install(monkeypatch, model):
    """Run the real _finalize swap on a CPU model: the fused forwards fall back to the class forward off CUDA, so the
    outputs stay exact and only the forward plumbing is under test. Returns the list the swapped forward appends to."""
    calls = []
    real = fused._flux_single_forward

    def spy(self, *args, **kwargs):
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(fused, "_flux_single_forward", spy)
    monkeypatch.setattr(fused, "resident_cuda_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(fused, "_device_ok", lambda index: True)
    monkeypatch.setattr(fused, "_op", lambda: None)
    monkeypatch.setattr(fused, "_ff_eligible", lambda m: False)
    monkeypatch.setattr(fused, "_prepare_swiglu", lambda m: False)
    monkeypatch.setattr(fused, "_has_eligible", lambda t: True)
    monkeypatch.setattr(
        fused, "_flux_single_eligible", lambda m: type(m).__name__ == "FluxSingleTransformerBlock"
    )
    assert fused.install(model) == len(model.single_transformer_blocks)
    return calls


def _fbcache(model):
    hooks = pytest.importorskip("diffusers.hooks")
    model.enable_cache(hooks.FirstBlockCacheConfig(threshold = 1e9))


def _two_steps(model):
    outs = []
    with torch.no_grad(), model.cache_context("cond"):
        for _ in range(2):
            outs.append(model(**_flux_inputs())[0])
    return outs


def test_fused_flux_single_keeps_fbcache_hooks_installed_before(monkeypatch):
    # Studio engages the step cache before the speed layer: the swap must go under the FBCache block hooks, else
    # the tail hook never records its residuals and the first reuse step reads None.
    import copy

    model = _tiny_flux()
    ref_model = copy.deepcopy(model)
    _fbcache(ref_model)
    ref = _two_steps(ref_model)

    _fbcache(model)
    wrappers = [b.__dict__.get("forward") for b in model.single_transformer_blocks]
    calls = _fake_cuda_install(monkeypatch, model)
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert [b.__dict__.get("forward") for b in model.single_transformer_blocks] == wrappers
    # Step 1 computes every single block through the fused forward; step 2 reuses the cached tail and skips them.
    assert len(calls) == len(model.single_transformer_blocks)

    # Turning the cache off splices the hook's inner forward back: the fused forward must survive it.
    model.disable_cache()
    for b in model.single_transformer_blocks:
        assert b.forward.__func__ is fused._flux_single_forward
    fused.uninstall(model)
    for b in model.single_transformer_blocks:
        assert not fused.is_installed(b)
        assert b.forward.__func__ is type(b).forward


def test_fused_flux_single_then_fbcache_and_uninstall_keeps_hooks(monkeypatch):
    import copy

    model = _tiny_flux()
    ref_model = copy.deepcopy(model)
    _fbcache(ref_model)
    ref = _two_steps(ref_model)

    calls = _fake_cuda_install(monkeypatch, model)
    _fbcache(model)
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert len(calls) == len(model.single_transformer_blocks)
    # Uninstall under a live cache restores the stock inner forward and leaves the hook wrappers in place.
    wrappers = [b.__dict__.get("forward") for b in model.single_transformer_blocks]
    fused.uninstall(model)
    assert [b.__dict__.get("forward") for b in model.single_transformer_blocks] == wrappers
    calls.clear()
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]) and not calls


def test_fused_flux_single_rearms_a_compiled_cache_inner(monkeypatch):
    # Deferred install: the speed layer already armed the FBCache hooks' inner forward with a compiled wrapper of the
    # stock forward before the first-forward swap; the swap must re-arm it on the fused forward, not bypass it.
    import copy

    from core.inference import diffusion_cache

    model = _tiny_flux()
    ref_model = copy.deepcopy(model)
    _fbcache(ref_model)
    ref = _two_steps(ref_model)

    compiled = []

    def fake_compile(fn, **kwargs):
        def wrapper(*args, **kw):
            return fn(*args, **kw)

        wrapper.inner = fn
        compiled.append(fn)
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    _fbcache(model)
    for b in model.single_transformer_blocks:
        b._compiled_call_impl = b._call_impl  # stands in for compile_repeated_blocks
    armed = diffusion_cache._compile_hooked_block_inners(model)
    assert armed == len(model.single_transformer_blocks)
    calls = _fake_cuda_install(monkeypatch, model)
    for b in model.single_transformer_blocks:
        inner = b._diffusers_hook.hooks["fbc_block_hook"].fn_ref.original_forward
        assert inner.inner.__func__ is fused._flux_single_forward
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert len(calls) == len(model.single_transformer_blocks)

    fused.uninstall(model)
    for b in model.single_transformer_blocks:
        inner = b._diffusers_hook.hooks["fbc_block_hook"].fn_ref.original_forward
        assert inner.__func__ is type(b).forward


@needs_cuda
def test_int8_flux_under_fbcache_renders_through_the_fused_kernel():
    # End to end on the real kernel: FBCache engaged first (Studio's order), then the swap; the reuse step must not
    # fail and both steps must match the same int8 model without the swap.
    import copy

    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    hooks = pytest.importorskip("diffusers.hooks")
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(0)
    model = (
        tf.FluxTransformer2DModel(
            patch_size = 1,
            in_channels = 16,
            num_layers = 1,
            num_single_layers = 2,
            attention_head_dim = 64,
            num_attention_heads = 4,
            joint_attention_dim = 64,
            pooled_projection_dim = 64,
            axes_dims_rope = (16, 24, 24),
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    quantize_(
        model,
        Int8DynamicActivationInt8WeightConfig(),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear)
        and "single_transformer_blocks" in fqn
        and "norm" not in fqn,
    )
    ref_model = copy.deepcopy(model)
    g = torch.Generator(device = "cuda").manual_seed(1)
    kwargs = dict(
        hidden_states = torch.randn(1, 256, 16, device = "cuda", dtype = torch.bfloat16, generator = g),
        encoder_hidden_states = torch.randn(
            1, 64, 64, device = "cuda", dtype = torch.bfloat16, generator = g
        ),
        pooled_projections = torch.randn(1, 64, device = "cuda", dtype = torch.bfloat16, generator = g),
        timestep = torch.tensor([1.0], device = "cuda"),
        img_ids = torch.zeros(256, 3, device = "cuda"),
        txt_ids = torch.zeros(64, 3, device = "cuda"),
        return_dict = False,
    )

    def run(m):
        outs = []
        with torch.no_grad(), m.cache_context("cond"):
            for _ in range(2):
                outs.append(m(**kwargs)[0])
        return outs

    ref_model.enable_cache(hooks.FirstBlockCacheConfig(threshold = 1e9))
    ref = run(ref_model)
    model.enable_cache(hooks.FirstBlockCacheConfig(threshold = 1e9))
    assert fused.install(model) == 2
    out = run(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert all(fused.is_installed(b) for b in model.single_transformer_blocks)

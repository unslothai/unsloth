# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused FLUX.3 RoPE: gating, kill switch, clean restore, bit-identical to stock (CUDA tests)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.inference import diffusion_flux3_rope as fr

torch = pytest.importorskip("torch")
from torch.torch_version import TorchVersion  # noqa: E402

fmod = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
from diffusers.models.embeddings import apply_rotary_emb as stock_rope  # noqa: E402

# Studio also reads ROCm from the "+rocm" tag, so faking NVIDIA clears both tag and version.hip.
_CUDA_TORCH_VERSION = TorchVersion(str(torch.__version__).split("+")[0] + "+cu130")

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or bool(getattr(torch.version, "hip", None))
    or fr._kernel() is None,
    reason = "needs CUDA (not ROCm) and Triton",
)


class _Flux3Like:
    def __init__(self, module = fr._MODULE):
        self.transformer = type("Flux3Transformer2DModel", (), {"__module__": module})()


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(fr.FLUX3_FUSED_ROPE_ENV, raising = False)
    fr.uninstall()
    yield
    fr.uninstall()
    assert fmod.apply_rotary_emb is stock_rope


@pytest.fixture
def fake_kernel(monkeypatch):
    monkeypatch.setattr(fr, "_kernel", lambda: (lambda x, cos, sin: None))
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", _CUDA_TORCH_VERSION, raising = False)


def test_stock_symbol_is_the_one_flux3_calls():
    assert fmod.apply_rotary_emb is stock_rope
    src = Path(fmod.__file__).read_text(encoding = "utf-8")
    # the processor rotates the fp32 norm outputs, then casts: the patch keeps that contract
    assert "apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)" in src
    assert "attn.norm_q(query.float())" in src


def test_flux3_load_installs(fake_kernel):
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    assert fr.is_installed()
    assert fmod.apply_rotary_emb is fr._fused_apply_rotary_emb


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_non_cuda_devices_keep_stock(fake_kernel, device):
    assert not fr.install_for_pipe(_Flux3Like(), device)
    assert fmod.apply_rotary_emb is stock_rope


def test_rocm_keeps_stock(fake_kernel, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert not fr.install_for_pipe(_Flux3Like(), "cuda")


def test_other_families_keep_stock(fake_kernel):
    pipe = _Flux3Like(module = "diffusers.models.transformers.transformer_flux2")
    assert not fr.install_for_pipe(pipe, "cuda")
    assert fmod.apply_rotary_emb is stock_rope


def test_other_family_load_removes_a_previous_install(fake_kernel):
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    assert not fr.install_for_pipe(_Flux3Like(module = "x.y"), "cuda")
    assert fmod.apply_rotary_emb is stock_rope


def test_flux2_patch_is_untouched(fake_kernel):
    from core.inference import diffusion_flux2_rope

    f2 = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    before = f2.apply_rotary_emb
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    assert f2.apply_rotary_emb is before and not diffusion_flux2_rope.is_installed()


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch(fake_kernel, monkeypatch, value):
    monkeypatch.setenv(fr.FLUX3_FUSED_ROPE_ENV, value)
    assert not fr.install_for_pipe(_Flux3Like(), "cuda")
    assert fmod.apply_rotary_emb is stock_rope


def test_kill_switch_after_install_restores_on_next_load(fake_kernel, monkeypatch):
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    monkeypatch.setenv(fr.FLUX3_FUSED_ROPE_ENV, "0")
    assert not fr.install_for_pipe(_Flux3Like(), "cuda")
    assert fmod.apply_rotary_emb is stock_rope


def test_no_triton_keeps_stock(monkeypatch):
    monkeypatch.setattr(fr, "_kernel", lambda: None)
    assert not fr.install_for_pipe(_Flux3Like(), "cuda")


def test_install_is_idempotent_and_uninstall_restores(fake_kernel):
    assert fr.install("cuda") and fr.install("cuda")
    fr.uninstall()
    fr.uninstall()
    assert fmod.apply_rotary_emb is stock_rope


def _freqs(
    S,
    D,
    device = "cpu",
    dtype = torch.float32,
    generator = None,
):
    pos = torch.randn(S, D // 2, dtype = torch.float64, device = device, generator = generator) * 50
    return (
        pos.cos().repeat_interleave(2, -1).to(dtype),
        pos.sin().repeat_interleave(2, -1).to(dtype),
    )


def test_cpu_tensors_fall_through_to_stock(fake_kernel):
    assert fr.install("cuda")
    x = torch.randn(1, 6, 2, 8, dtype = torch.float32)
    freqs = _freqs(6, 8)
    assert torch.equal(
        fmod.apply_rotary_emb(x, freqs, sequence_dim = 1), stock_rope(x, freqs, sequence_dim = 1)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_dim": 2},
        {"sequence_dim": 1, "use_real_unbind_dim": -2},
    ],
)
def test_other_call_forms_are_not_eligible(kwargs):
    x = torch.randn(1, 6, 2, 8)
    args = {"use_real": True, "use_real_unbind_dim": -1, "sequence_dim": 1}
    args.update(kwargs)
    assert not fr._eligible(
        x, _freqs(6, 8), args["use_real"], args["use_real_unbind_dim"], args["sequence_dim"]
    )


def test_teardown_and_load_wiring():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    teardown = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_uninstall_fused_dit_patches"
    )
    # unload restores the fused-DiT group, which carries this spec
    assert "uninstall_group(GROUP_FUSED_DIT)" in ast.unparse(teardown)
    from core.inference.diffusion_autoopt import patches as ap

    spec = next(s for s in ap.specs(group = ap.GROUP_FUSED_DIT) if s.name == "flux3_rope")
    assert spec.stage == ap.STAGE_LOAD and spec.backends == ("image",) and not spec.speed_only
    assert spec.env == fr.FLUX3_FUSED_ROPE_ENV
    # the load runs it on every tier, speed off included
    import types as _types

    seen = []
    orig_install, orig_uninstall = fr.install_for_pipe, fr.uninstall
    try:
        fr.install_for_pipe = lambda *a: seen.append(("install", a)) or False
        fr.uninstall = lambda: seen.append(("uninstall",))
        pipe = _types.SimpleNamespace(transformer = None)
        ap.install_stage(
            ap.STAGE_LOAD,
            ap.PatchContext(pipe = pipe, dtype = "bf16", device = "cuda", logger = "log", speed_mode = "off"),
        )
        from core.inference import diffusion

        diffusion._uninstall_fused_dit_patches()
    finally:
        fr.install_for_pipe, fr.uninstall = orig_install, orig_uninstall
    assert ("install", (pipe, "cuda", "log")) in seen and ("uninstall",) in seen


def test_call_after_uninstall_reaches_stock(fake_kernel):
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    patched = fmod.apply_rotary_emb
    fr.uninstall()
    x = torch.randn(1, 4, 2, 8)
    freqs = (torch.ones(4, 8), torch.zeros(4, 8))
    assert torch.equal(patched(x, freqs, sequence_dim = 1), x)


def test_a_failed_launch_disables_the_fused_path_for_the_load(monkeypatch):
    calls = []

    def failing_launch(x, cos, sin):
        calls.append(1)
        raise RuntimeError("PTX JIT compilation failed")

    monkeypatch.setattr(fr, "_kernel", lambda: failing_launch)
    monkeypatch.setattr(fr, "_eligible", lambda *a, **k: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", _CUDA_TORCH_VERSION, raising = False)
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    x = torch.randn(1, 6, 2, 8)
    freqs = _freqs(6, 8)
    want = stock_rope(x, freqs, sequence_dim = 1)
    for _ in range(4):
        assert torch.equal(fmod.apply_rotary_emb(x, freqs, sequence_dim = 1), want)
    assert len(calls) == 1
    # the next load re-arms even without a teardown
    assert fr.install_for_pipe(_Flux3Like(), "cuda")
    fmod.apply_rotary_emb(x, freqs, sequence_dim = 1)
    assert len(calls) == 2


@needs_cuda
@pytest.mark.parametrize(
    "shape,x_dtype,freq_dtype",
    [
        (
            (1, 4224, 24, 128),
            "float32",
            "float32",
        ),  # 1024px: 128 text + 4096 image tokens, fp32 norm output
        ((2, 1000, 24, 128), "float32", "float32"),  # batched CFG
        ((1, 333, 5, 64), "float32", "bfloat16"),
        ((1, 257, 24, 128), "bfloat16", "float32"),
        ((1, 257, 24, 128), "float16", "float32"),
    ],
)
def test_bit_identical_to_stock(shape, x_dtype, freq_dtype):
    B, S, H, D = shape
    g = torch.Generator("cuda").manual_seed(0)
    x = (torch.randn(*shape, device = "cuda", generator = g) * 4).to(getattr(torch, x_dtype))
    freqs = _freqs(S, D, "cuda", getattr(torch, freq_dtype), g)
    ref = stock_rope(x, freqs, sequence_dim = 1)
    assert fr.install("cuda")
    n = fr.call_count()
    out = fmod.apply_rotary_emb(x, freqs, sequence_dim = 1)
    assert fr.call_count() == n + 1
    assert out.dtype == ref.dtype and out.stride() == ref.stride()
    assert torch.equal(out, ref)


def _tiny_flux3_case():
    torch.manual_seed(0)
    model = fmod.Flux3Transformer2DModel(
        in_channels = 16,
        cond_in_channels = 16,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 32,
        num_attention_heads = 4,
        joint_attention_dim = 32,
        timestep_channels = 32,
        axes_dims_rope = (8, 8, 8, 8),
    )
    model = model.to("cuda", torch.bfloat16).eval()
    g = torch.Generator().manual_seed(1)
    img, txt, ref = 256, 17, 96
    kw = {
        "hidden_states": torch.randn(2, img, 16, generator = g),
        "encoder_hidden_states": torch.randn(2, txt, 32, generator = g),
        "timestep": torch.tensor([0.7, 0.7]),
        "img_ids": torch.randint(0, 16, (img, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 16, (txt, 4), generator = g).float(),
        "image_cond_hidden_states": torch.randn(2, ref, 16, generator = g),
        "image_cond_ids": torch.randint(0, 16, (ref, 4), generator = g).float(),
        "return_dict": False,
    }
    kw = {
        k: (
            v.to("cuda", torch.bfloat16)
            if torch.is_tensor(v) and k.endswith("states")
            else v.to("cuda")
        )
        if torch.is_tensor(v)
        else v
        for k, v in kw.items()
    }
    return model, kw


@needs_cuda
def test_whole_denoiser_forward_bit_identical():
    """Every FLUX.3 attention (mode blocks with a reference image, single blocks, CFG batch of 2): same output bits."""
    model, kw = _tiny_flux3_case()
    with torch.no_grad():
        ref = model(**kw)[0]
        assert fr.install("cuda")
        n = fr.call_count()
        out = model(**kw)[0]
    assert fr.call_count() > n
    assert torch.equal(out, ref)


@needs_cuda
def test_attention_processor_bit_identical():
    """A bf16 FLUX.3 attention (bf16 norm weights on the fp32 Q / K, as a real load has): same output bits."""
    attn = (
        fmod.Flux3Attention(query_dim = 3072, heads = 24, dim_head = 128)
        .to("cuda", torch.bfloat16)
        .eval()
    )
    with torch.no_grad():
        attn.norm_q.weight.normal_(1, 0.2)
        attn.norm_k.weight.normal_(1, 0.2)
    g = torch.Generator("cuda").manual_seed(1)
    hs = torch.randn(1, 1152, 3072, device = "cuda", dtype = torch.bfloat16, generator = g)
    freqs = _freqs(1152, 128, "cuda", torch.float32, g)
    with torch.no_grad():
        ref = attn(hidden_states = hs, image_rotary_emb = freqs)
        assert fr.install("cuda")
        n = fr.call_count()
        out = attn(hidden_states = hs, image_rotary_emb = freqs)
    assert fr.call_count() == n + 2
    assert torch.equal(out, ref)


@needs_cuda
def test_compiled_attention_has_no_graph_break():
    """Studio compiles FLUX.3 blocks from the 3rd render: the patch must trace as stock, never split the block."""
    attn = fmod.Flux3Attention(query_dim = 64, heads = 2, dim_head = 32).to("cuda", torch.bfloat16).eval()
    hs = torch.randn(1, 40, 64, device = "cuda", dtype = torch.bfloat16)
    freqs = _freqs(40, 32, "cuda")
    torch._dynamo.reset()
    with torch.no_grad():
        ref = torch.compile(
            lambda t: attn(hidden_states = t, image_rotary_emb = freqs), fullgraph = True
        )(hs)
        torch._dynamo.reset()
        assert fr.install("cuda")
        n = fr.call_count()
        out = torch.compile(
            lambda t: attn(hidden_states = t, image_rotary_emb = freqs), fullgraph = True
        )(hs)
    assert fr.call_count() == n
    assert torch.equal(out, ref)


@needs_cuda
def test_grad_calls_use_stock():
    assert fr.install("cuda")
    x = torch.randn(1, 16, 2, 8, device = "cuda", requires_grad = True)
    freqs = _freqs(16, 8, "cuda")
    n = fr.call_count()
    fmod.apply_rotary_emb(x, freqs, sequence_dim = 1).sum().backward()
    assert x.grad is not None and fr.call_count() == n

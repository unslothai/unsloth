# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Query-chunked VAE attention where no fused SDPA takes the head dim; the score tile is bounded."""

import types

import pytest

torch = pytest.importorskip("torch")
from torch.torch_version import TorchVersion  # noqa: E402

F = torch.nn.functional
attention = pytest.importorskip("torch.nn.attention")
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from core.inference import diffusion_vae_attn_chunked as C  # noqa: E402

# A CUDA build's version tag: Studio reads ROCm from the "+rocm" tag too (AMD SDK / Radeon wheels leave
# torch.version.hip unset), so a test that fakes NVIDIA clears both.
_CUDA_TORCH_VERSION = TorchVersion(str(torch.__version__).split("+")[0] + "+cu130")


def _math():
    return attention.sdpa_kernel(attention.SDPBackend.MATH)


class _LargestTensor(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.max_bytes = 0

    def __torch_dispatch__(
        self,
        func,
        types_,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.max_bytes = max(self.max_bytes, t.numel() * t.element_size())
        return out


@pytest.fixture
def rocm(monkeypatch):
    """Pretend to be ROCm with a math-only SDPA: the fused-kernel query says no for every call."""
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising = False)
    monkeypatch.setattr(C, "fused_sdpa_available", lambda q, k, v: False)
    monkeypatch.delenv(C.VAE_ATTN_CHUNK_ENV, raising = False)
    return types.SimpleNamespace(backend = "rocm", device = "cuda")


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("shape", [(1, 1, 1000, 64), (2, 300, 32), (1, 2, 517, 16)])
@pytest.mark.parametrize("budget_rows", [1, 64, 333, 10_000])
def test_chunked_equals_full_softmax(dtype, tol, shape, budget_rows):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(*shape, generator = g).to(dtype) for _ in range(3))
    rows = 1
    for s in shape[:-2]:
        rows *= s
    budget = budget_rows * rows * shape[-2] * 4
    with _math():
        ref = F.scaled_dot_product_attention(q, k, v)
        out = C.chunked_sdpa(F.scaled_dot_product_attention, q, k, v, budget)
        ref_s = F.scaled_dot_product_attention(q, k, v, scale = 0.3)
        out_s = C.chunked_sdpa(F.scaled_dot_product_attention, q, k, v, budget, scale = 0.3)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol = tol, rtol = tol)
    torch.testing.assert_close(out_s, ref_s, atol = tol, rtol = tol)


def test_mode_bounds_the_score_tile(rocm):
    q = torch.randn(1, 1, 2048, 64)
    full = 2048 * 2048 * 4
    budget = full // 16
    with _math():
        probe = _LargestTensor()
        with probe:
            ref = F.scaled_dot_product_attention(q, q, q)
        assert probe.max_bytes >= full  # the math kernel materialises every score
        mode = C.chunked_attention_mode(budget)
        probe = _LargestTensor()
        with mode, probe:
            out = F.scaled_dot_product_attention(q, q, q)
    assert mode.chunked == 1
    assert probe.max_bytes <= budget
    torch.testing.assert_close(out, ref, atol = 1e-5, rtol = 1e-5)


def test_mode_chunks_keyword_calls(rocm):
    q = torch.randn(1, 1, 512, 32)
    with _math():
        ref = F.scaled_dot_product_attention(q, q, q, scale = 0.2)
        mode = C.chunked_attention_mode(512 * 512 * 4 // 8)
        with mode:
            # the call shape of diffusers' native dispatch_attention_fn
            out = F.scaled_dot_product_attention(
                query = q,
                key = q,
                value = q,
                attn_mask = None,
                dropout_p = 0.0,
                is_causal = False,
                scale = 0.2,
                enable_gqa = False,
            )
            mixed = F.scaled_dot_product_attention(q, key = q, value = q, scale = 0.2)
    assert mode.chunked == 2
    torch.testing.assert_close(out, ref, atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(mixed, ref, atol = 1e-5, rtol = 1e-5)


def test_mode_leaves_fused_small_masked_and_causal_calls_alone(monkeypatch):
    calls = []
    real = C.chunked_sdpa
    monkeypatch.setattr(C, "chunked_sdpa", lambda *a, **k: calls.append(1) or real(*a, **k))
    q = torch.randn(1, 1, 256, 16)
    big = 1024  # bytes: every call here is over budget
    mask = torch.ones(256, 256, dtype = torch.bool)
    with _math():
        ref = F.scaled_dot_product_attention(q, q, q)
        # fused kernel available (CPU, and NVIDIA at head dim 512): the stock call, bit-identical
        with C.chunked_attention_mode(big) as mode:
            assert torch.equal(F.scaled_dot_product_attention(q, q, q), ref)
        assert mode.chunked == 0
        monkeypatch.setattr(C, "fused_sdpa_available", lambda q, k, v: False)
        with C.chunked_attention_mode(big) as mode:
            F.scaled_dot_product_attention(q, q, q, attn_mask = mask)
            F.scaled_dot_product_attention(q, q, q, mask)
            F.scaled_dot_product_attention(q, q, q, is_causal = True)
            F.scaled_dot_product_attention(q, q, q, dropout_p = 0.1)
        with C.chunked_attention_mode(256 * 256 * 4) as small:
            assert torch.equal(F.scaled_dot_product_attention(q, q, q), ref)
    assert mode.chunked == 0 and small.chunked == 0 and calls == []


def test_fused_query_is_stock_off_cuda():
    q = torch.randn(1, 1, 8, 512)
    assert C.fused_sdpa_available(q, q, q) is True


@pytest.mark.parametrize("head_dim, fused", [(512, False), (384, False), (128, True)])
def test_fused_query_on_mps_asks_the_prefill_kernel(monkeypatch, head_dim, fused):
    # No MPS fused kernel takes the VAE's head dim 384 / 512, whatever the torch build.
    monkeypatch.setattr(torch, "__version__", "2.14.1")
    monkeypatch.delenv("PYTORCH_MPS_DISABLE_PREFILL_ATTENTION", raising = False)
    q = types.SimpleNamespace(
        device = types.SimpleNamespace(type = "mps"),
        dtype = torch.bfloat16,
        shape = (1, 1, 16384, head_dim),
        requires_grad = False,
    )
    assert C.fused_sdpa_available(q, q, q) is fused


def _tiny_vae(name):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    if name == "AutoencoderKL":
        return diffusers.AutoencoderKL(
            in_channels = 3,
            out_channels = 3,
            down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels = (32, 64),
            layers_per_block = 1,
            latent_channels = 4,
            norm_num_groups = 8,
        ).eval()
    if name == "AutoencoderKLWan":
        return diffusers.AutoencoderKLWan(
            base_dim = 16, z_dim = 4, dim_mult = [1, 2], num_res_blocks = 1, temperal_downsample = [False]
        ).eval()
    if name == "AutoencoderKLQwenImage":
        return diffusers.AutoencoderKLQwenImage(
            base_dim = 16, z_dim = 4, dim_mult = [1, 2], num_res_blocks = 1, temperal_downsample = [False]
        ).eval()
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["AutoencoderKL", "AutoencoderKLWan", "AutoencoderKLQwenImage"])
def test_vae_decode_peak_bounded_and_equal(rocm, monkeypatch, name):
    vae = _tiny_vae(name)
    if name == "AutoencoderKL":
        z = torch.randn(1, 4, 48, 48)
    else:
        z = torch.randn(1, 4, 1, 48, 48)
    tokens = 48 * 48
    full = tokens * tokens * 4
    with torch.no_grad(), _math():
        probe = _LargestTensor()
        with probe:
            ref = vae.decode(z).sample
        assert probe.max_bytes >= full
        monkeypatch.setenv(
            C.VAE_ATTN_CHUNK_MB_ENV, "1"
        )  # 1 MiB of scores per chunk, well under the 21 MiB matrix
        assert C.install(vae, rocm) >= 1
        probe = _LargestTensor()
        with probe:
            out = vae.decode(z).sample
    assert probe.max_bytes < full // 4
    torch.testing.assert_close(out, ref, atol = 1e-4, rtol = 1e-4)


@pytest.mark.parametrize(
    "backend, env, budget_mb", [("rocm", None, 1024), ("mps", None, 256), ("mps", "64", 64)]
)
def test_install_budget_follows_the_device(rocm, monkeypatch, backend, env, budget_mb):
    # MPSGraph holds several score-sized buffers, so a 1 GiB FLUX.1 tile is chunked there but not on ROCm.
    if env is not None:
        monkeypatch.setenv(C.VAE_ATTN_CHUNK_MB_ENV, env)
    else:
        monkeypatch.delenv(C.VAE_ATTN_CHUNK_MB_ENV, raising = False)
    monkeypatch.setattr(torch.version, "hip", None if backend == "mps" else "7.2.0", raising = False)
    vae = _tiny_vae("AutoencoderKL")
    assert C.install(vae, types.SimpleNamespace(backend = backend, device = backend)) == 2
    budgets = []
    real = C.chunked_attention_mode
    monkeypatch.setattr(
        C, "chunked_attention_mode", lambda budget = None: budgets.append(budget) or real(budget)
    )
    with torch.no_grad():
        vae.decode(torch.randn(1, 4, 8, 8))
    assert budgets == [budget_mb * 2**20]


def test_install_gates(monkeypatch, rocm):
    vae = _tiny_vae("AutoencoderKL")
    assert C.install(vae, types.SimpleNamespace(backend = "cuda", device = "cuda")) == 0
    monkeypatch.setenv(C.VAE_ATTN_CHUNK_ENV, "0")
    assert C.install(vae, rocm) == 0
    monkeypatch.delenv(C.VAE_ATTN_CHUNK_ENV)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", _CUDA_TORCH_VERSION, raising = False)
    assert C.install(vae, rocm) == 0
    assert not any(getattr(m, C._PATCHED, False) for m in vae.modules())
    assert C.install(None, rocm) == 0


def test_install_patches_encoder_and_decoder_once(rocm):
    vae = _tiny_vae("AutoencoderKL")
    assert C.install(vae, rocm) == 2
    assert C.install(vae, rocm) == 0
    # a processor reset (fuse_qkv_projections) keeps the module-level patch
    for m in vae.modules():
        if type(m).__name__ == "Attention":
            m.set_processor(type(m.processor)())
            assert m.forward._unsloth_vae_attn_chunked


def test_kill_switch_at_call_time(rocm, monkeypatch):
    vae = _tiny_vae("AutoencoderKL")
    monkeypatch.setenv(C.VAE_ATTN_CHUNK_MB_ENV, "1")
    C.install(vae, rocm)
    monkeypatch.setenv(C.VAE_ATTN_CHUNK_ENV, "off")
    z = torch.randn(1, 4, 48, 48)
    with torch.no_grad(), _math():
        probe = _LargestTensor()
        with probe:
            vae.decode(z)
    assert probe.max_bytes >= 48 * 48 * 48 * 48 * 4


def test_speed_layer_installs_on_rocm_and_mps_even_when_off(rocm, monkeypatch):
    from core.inference import diffusion_speed as S

    vae = _tiny_vae("AutoencoderKL")
    pipe = types.SimpleNamespace(vae = vae)
    S.apply_speed_optims(
        pipe,
        types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = torch.bfloat16),
        is_gguf = False,
        family = types.SimpleNamespace(),
        speed_mode = S.SPEED_OFF,
    )
    assert sum(getattr(m, C._PATCHED, False) for m in vae.modules()) == 2
    mac = _tiny_vae("AutoencoderKL")
    S.apply_speed_optims(
        types.SimpleNamespace(vae = mac),
        types.SimpleNamespace(backend = "mps", device = "mps", dtype = torch.bfloat16),
        is_gguf = False,
        family = types.SimpleNamespace(),
        speed_mode = S.SPEED_OFF,
    )
    assert sum(getattr(m, C._PATCHED, False) for m in mac.modules()) == 2
    nvidia = _tiny_vae("AutoencoderKL")
    S.apply_speed_optims(
        types.SimpleNamespace(vae = nvidia),
        types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = torch.bfloat16),
        is_gguf = False,
        family = types.SimpleNamespace(),
        speed_mode = S.SPEED_OFF,
    )
    assert not any(getattr(m, C._PATCHED, False) for m in nvidia.modules())

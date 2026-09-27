# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_hv15_vae as hv  # noqa: E402


def _stock(
    n_frame,
    n_hw,
    dtype,
    device,
    batch_size = None,
):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype = dtype, device = device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class HunyuanVideo15AttnBlock(torch.nn.Module):
    prepare_causal_attention_mask = staticmethod(_stock)

    def forward(self, n_frame, n_hw):
        return self.prepare_causal_attention_mask(n_frame, n_hw, torch.float32, "cpu", batch_size = 1)


class AutoencoderKLHunyuanVideo15(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mid = HunyuanVideo15AttnBlock()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n_frame,n_hw", [(1, 4), (3, 5), (7, 16), (2, 1)])
def test_vectorised_mask_equals_the_stock_loop(dtype, n_frame, n_hw):
    for batch in (None, 2):
        want = _stock(n_frame, n_hw, dtype, "cpu", batch)
        got = hv.causal_attention_mask(n_frame, n_hw, dtype, "cpu", batch)
        assert got.dtype == want.dtype and got.shape == want.shape and got.stride() == want.stride()
        assert torch.equal(got, want)


def test_install_patches_instances_only():
    vae = AutoencoderKLHunyuanVideo15()
    assert hv.install_vectorised_causal_mask(types.SimpleNamespace(vae = vae)) == 1
    assert vae.mid.prepare_causal_attention_mask is hv.causal_attention_mask
    assert torch.equal(vae.mid(3, 4), _stock(3, 4, torch.float32, "cpu", 1))
    # the class, and so every other pipe, keeps diffusers' own mask
    assert HunyuanVideo15AttnBlock.__dict__["prepare_causal_attention_mask"].__func__ is _stock
    assert HunyuanVideo15AttnBlock().prepare_causal_attention_mask is _stock


def test_other_vaes_and_missing_vae_are_left_alone():
    class AutoencoderKLWan(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mid = HunyuanVideo15AttnBlock()

    wan = AutoencoderKLWan()
    assert hv.install_vectorised_causal_mask(types.SimpleNamespace(vae = wan)) == 0
    assert "prepare_causal_attention_mask" not in wan.mid.__dict__
    assert hv.install_vectorised_causal_mask(types.SimpleNamespace()) == 0


def test_a_rewritten_upstream_mask_is_not_replaced():
    class HunyuanVideo15AttnBlock(torch.nn.Module):  # noqa: N801 - mirrors the diffusers name
        @staticmethod
        def prepare_causal_attention_mask(
            n_frame,
            n_hw,
            dtype,
            device,
            batch_size = None,
        ):
            return torch.zeros(1)

    class AutoencoderKLHunyuanVideo15(torch.nn.Module):  # noqa: N801
        def __init__(self):
            super().__init__()
            self.mid = HunyuanVideo15AttnBlock()

    vae = AutoencoderKLHunyuanVideo15()
    assert hv.install_vectorised_causal_mask(types.SimpleNamespace(vae = vae)) == 0


def test_real_diffusers_block_is_recognised_when_installed():
    mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15")
    cls = getattr(mod, "HunyuanVideo15AttnBlock", None)
    if cls is None:
        pytest.skip("diffusers without HunyuanVideo-1.5")
    if not hv._stock_loop(cls):
        pytest.skip("this diffusers already rewrote the mask")
    for n_frame, n_hw in ((2, 3), (5, 8)):
        assert torch.equal(
            hv.causal_attention_mask(n_frame, n_hw, torch.bfloat16, "cpu", 1),
            cls.prepare_causal_attention_mask(n_frame, n_hw, torch.bfloat16, "cpu", batch_size = 1),
        )


def test_mask_allocates_nothing_quadratic_besides_the_output():
    """The stock loop allocates only the seq_len x seq_len mask; so must this (no seq_len x seq_len bool predicate)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class _Allocs(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.storages = []

        def __torch_dispatch__(self, func, types, args = (), kwargs = None):
            out = func(*args, **(kwargs or {}))
            for t in tree_leaves(out):
                if isinstance(t, torch.Tensor):
                    self.storages.append((t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()))
            return out

    n_frame, n_hw = 6, 32
    seq_len = n_frame * n_hw
    rec = _Allocs()
    with rec:
        mask = hv.causal_attention_mask(n_frame, n_hw, torch.bfloat16, "cpu", 2)
    own = mask.untyped_storage().data_ptr()
    assert [nb for ptr, nb in rec.storages if ptr != own and nb >= seq_len * seq_len] == []
    assert torch.equal(mask, _stock(n_frame, n_hw, torch.bfloat16, "cpu", 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_mask_peak_cuda_memory_is_the_output_alone():
    n_frame, n_hw = 6, 256
    seq_len = n_frame * n_hw
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    mask = hv.causal_attention_mask(n_frame, n_hw, torch.float32, "cuda", 1)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base - mask.untyped_storage().nbytes()
    # A seq_len x seq_len bool would be 2.36 MB here; the frame-level predicate is n_frame ** 2 bytes.
    assert extra < seq_len * seq_len // 4

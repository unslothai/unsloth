# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the image-conditioned VAE encode dtype guard.

Image Transform on a bf16 denoiser whose text encoder settled on fp32 died with
``RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same``:
``ZImageImg2ImgPipeline.prepare_latents`` casts the preprocessed upload to
``prompt_embeds[0].dtype`` and feeds it straight to ``vae.encode``, so neither the
denoiser dtype nor the VAE's own dtype gets a say. ``_make_vae_encode_dtype_safe``
closes that gap at the encoder boundary. No torch autograd, no GPU, no diffusers.
"""

from __future__ import annotations

import types

from core.inference.diffusion import DiffusionBackend


class _RecordingVae:
    """A VAE stand-in that records the dtype its ``encode`` was actually handed."""

    def __init__(self, dtype):
        self._dtype = dtype
        self.seen_dtype = None
        self.seen_args = None
        self.encode = self._encode

    def parameters(self):
        yield types.SimpleNamespace(dtype = self._dtype)

    def _encode(self, x, *args, **kwargs):
        self.seen_dtype = getattr(x, "dtype", None)
        self.seen_args = (args, kwargs)
        return "latents"


def test_encode_input_is_cast_to_the_vae_dtype():
    import torch
    # The reported crash: fp32 image tensor (text-encoder dtype) into a bf16 VAE.
    vae = _RecordingVae(dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(vae = vae)
    DiffusionBackend._make_vae_encode_dtype_safe(pipe)
    out = vae.encode(torch.zeros(1, 3, 64, 64, dtype = torch.float32))
    assert vae.seen_dtype == torch.bfloat16
    # The wrapper is transparent: the encoder's return value passes straight through.
    assert out == "latents"


def test_encode_input_is_cast_the_other_direction_too():
    import torch
    # The mirror mismatch (bf16 image into an fp32 / force_upcast VAE) is the same bug.
    vae = _RecordingVae(dtype = torch.float32)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(torch.zeros(1, 3, 64, 64, dtype = torch.bfloat16))
    assert vae.seen_dtype == torch.float32


def test_matching_dtype_is_left_alone_and_extra_args_pass_through():
    import torch
    vae = _RecordingVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    x = torch.zeros(1, 3, 64, 64, dtype = torch.bfloat16)
    vae.encode(x, return_dict = False)
    assert vae.seen_dtype == torch.bfloat16
    assert vae.seen_args == ((), {"return_dict": False})


def test_non_tensor_input_passes_through_untouched():
    import torch
    # Some callers hand encode a list of tensors or a distribution; never guess, just forward.
    vae = _RecordingVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(["not", "a", "tensor"])
    assert vae.seen_dtype is None


def test_wrapping_is_idempotent():
    import torch
    # generate() runs this on every image-conditioned call against a warm pipe; stacking
    # wrappers per generation would be an unbounded call-depth leak.
    vae = _RecordingVae(dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(vae = vae)
    DiffusionBackend._make_vae_encode_dtype_safe(pipe)
    first = vae.encode
    DiffusionBackend._make_vae_encode_dtype_safe(pipe)
    assert vae.encode is first


def test_dtype_is_read_at_call_time_not_wrap_time():
    import torch
    # _align_vae_dtype may re-cast the VAE after the wrap (and a txt2img decode re-upcasts
    # it), so a dtype snapshotted at wrap time would go stale on the very next call.
    vae = _RecordingVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae._dtype = torch.float32
    vae.encode(torch.zeros(1, 3, 64, 64, dtype = torch.bfloat16))
    assert vae.seen_dtype == torch.float32


def test_missing_vae_and_broken_probe_are_no_ops():
    import torch
    # Best-effort: a pipe without a VAE, and a VAE whose parameters() raises, must never
    # turn a working generation into a 500.
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace())
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = None))

    class _BrokenVae(_RecordingVae):
        def parameters(self):
            raise RuntimeError("meta device")

    vae = _BrokenVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    # The probe fails inside the call, so the tensor is forwarded exactly as it arrived.
    vae.encode(torch.zeros(1, 3, 8, 8, dtype = torch.float32))
    assert vae.seen_dtype == torch.float32

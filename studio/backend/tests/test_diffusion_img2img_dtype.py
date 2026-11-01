# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VAE encode casts image uploads to the VAE's dtype, not the prompt dtype, or bf16 fails."""

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

    vae = _RecordingVae(dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(vae = vae)
    DiffusionBackend._make_vae_encode_dtype_safe(pipe)
    out = vae.encode(torch.zeros(1, 3, 64, 64, dtype = torch.float32))
    assert vae.seen_dtype == torch.bfloat16
    assert out == "latents"


def test_encode_input_is_cast_the_other_direction_too():
    import torch

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

    vae = _RecordingVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(["not", "a", "tensor"])
    assert vae.seen_dtype is None


def test_wrapping_is_idempotent():
    import torch

    # generate() calls this on every image-conditioned call, so wrapping must not stack.
    vae = _RecordingVae(dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(vae = vae)
    DiffusionBackend._make_vae_encode_dtype_safe(pipe)
    first = vae.encode
    DiffusionBackend._make_vae_encode_dtype_safe(pipe)
    assert vae.encode is first


def test_dtype_is_read_at_call_time_not_wrap_time():
    import torch

    # The VAE is re-cast later, so its dtype must be read at call time, not wrap time.
    vae = _RecordingVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae._dtype = torch.float32
    vae.encode(torch.zeros(1, 3, 64, 64, dtype = torch.bfloat16))
    assert vae.seen_dtype == torch.float32


def test_missing_vae_and_broken_probe_are_no_ops():
    import torch

    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace())
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = None))

    class _BrokenVae(_RecordingVae):
        def parameters(self):
            raise RuntimeError("meta device")

    vae = _BrokenVae(dtype = torch.bfloat16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(torch.zeros(1, 3, 8, 8, dtype = torch.float32))
    assert vae.seen_dtype == torch.float32


def _encoder(conv_dtype, other_dtypes = ()):
    weight = types.SimpleNamespace(dtype = conv_dtype)

    class _Encoder:
        conv_in = types.SimpleNamespace(weight = weight)

        def parameters(self):
            yield weight
            for dtype in other_dtypes:
                yield types.SimpleNamespace(dtype = dtype)

    return _Encoder()


def test_encode_follows_conv_in_when_the_vae_probe_fails():
    # #16375: under offload the VAE parameter walk can fail; the layer that eats the image still says fp32.
    import torch

    class _BrokenVae(_RecordingVae):
        def parameters(self):
            raise RuntimeError("meta device")

    vae = _BrokenVae(dtype = torch.float32)
    vae.encoder = _encoder(torch.float32)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(torch.zeros(1, 3, 8, 8, dtype = torch.float16))
    assert vae.seen_dtype == torch.float32


def test_encode_follows_conv_in_over_an_earlier_mismatched_vae_weight():
    import torch

    vae = _RecordingVae(dtype = torch.float16)  # first VAE weight is half, the encoder runs fp32
    vae.encoder = _encoder(torch.float32)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(torch.zeros(1, 3, 8, 8, dtype = torch.float16))
    assert vae.seen_dtype == torch.float32


def test_a_half_conv_in_keeps_half_input_beside_fp32_norm_weights():
    import torch

    vae = _RecordingVae(dtype = torch.float16)
    vae.encoder = _encoder(torch.float16, other_dtypes = (torch.float32,))
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(torch.zeros(1, 3, 8, 8, dtype = torch.float16))
    assert vae.seen_dtype == torch.float16


def test_an_unreadable_dtype_leaves_the_input_alone():
    import torch

    class _OpaqueVae(_RecordingVae):
        def parameters(self):
            raise RuntimeError("meta device")

    vae = _OpaqueVae(dtype = torch.float16)
    DiffusionBackend._make_vae_encode_dtype_safe(types.SimpleNamespace(vae = vae))
    vae.encode(torch.zeros(1, 3, 8, 8, dtype = torch.float16))
    assert vae.seen_dtype == torch.float16

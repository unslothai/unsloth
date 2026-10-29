# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A compiled DiT VAE decode must not recompile when the resolution changes (real Dynamo, eager backend)."""

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_speed as ds  # noqa: E402


def test_compiled_vae_decode_serves_a_second_resolution_without_a_recompile(monkeypatch):
    from torch._dynamo.utils import counters

    real_compile = torch.compile
    seen = []

    def eager_backend_compile(fn, **kwargs):
        seen.append(dict(kwargs))
        kwargs.pop("mode", None)
        return real_compile(fn, backend = "eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_backend_compile)
    monkeypatch.setattr(ds, "_install_inductor_backports", lambda *a, **k: False)
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKL(
        block_out_channels = (32,), norm_num_groups = 32, latent_channels = 4, layers_per_block = 1
    ).eval()
    pipe = types.SimpleNamespace(vae = vae, transformer = types.SimpleNamespace())
    torch._dynamo.reset()
    assert ds._compile_vae_decode(pipe, None, max_autotune = True) is True
    assert seen and seen[0]["dynamic"] is True
    graphs = []
    with torch.no_grad():
        for side in (16, 24, 20):
            before = int(counters["stats"]["unique_graphs"])
            out = vae.decode(torch.randn(1, 4, side, side)).sample
            assert out.shape[-1] == side
            graphs.append(int(counters["stats"]["unique_graphs"]) - before)
    torch._dynamo.reset()
    assert graphs[0] >= 1
    assert graphs[1:] == [0, 0], graphs

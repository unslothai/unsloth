# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan 2.2 / Qwen-Image-2.1 VAE residual blocks skip the redundant input clone (diffusers #15278): bitwise the stock
encode and decode, one ``aten.clone`` fewer per block call, installed by both loaders, replaced by the fused CUDA up
block."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_vae_noclone as noclone

BACKEND = Path(__file__).resolve().parents[1]


def _tiny_wan22():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLWan(
        base_dim = 16,
        decoder_base_dim = 16,
        z_dim = 4,
        dim_mult = [1, 2, 2],
        num_res_blocks = 1,
        temperal_downsample = [False, True],
        is_residual = True,
        in_channels = 12,
        out_channels = 12,
        patch_size = 2,
        scale_factor_spatial = 8,
        scale_factor_temporal = 2,
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    ).eval()


def _tiny_qwen21():
    cls = getattr(diffusers, "AutoencoderKLQwenImage21", None)
    if cls is None:
        pytest.skip("this diffusers has no AutoencoderKLQwenImage21")
    import inspect

    params = inspect.signature(cls.__init__).parameters
    kwargs = dict(base_dim = 16, z_dim = 4, dim_mult = [1, 2, 2], num_res_blocks = 1)
    for name, value in (
        ("decoder_base_dim", 16),
        ("temperal_downsample", [False, True]),
        ("latents_mean", [0.0] * 4),
        ("latents_std", [1.0] * 4),
    ):
        if name in params:
            kwargs[name] = value
    torch.manual_seed(0)
    return cls(**kwargs).eval()


VAES = {"wan22": _tiny_wan22, "qwen21": _tiny_qwen21}


class _CountClones(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.clones = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func is torch.ops.aten.clone.default:
            self.clones += 1
        return func(*args, **(kwargs or {}))


def _block_calls(vae):
    calls = []
    for module in vae.modules():
        name = type(module).__name__
        if name.endswith("ResidualDownBlock") or name.endswith("ResidualUpBlock"):
            module.register_forward_hook(lambda m, a, o: calls.append(type(m).__name__))
    return calls


def _run(vae, x):
    counter = _CountClones()
    calls = _block_calls(vae)
    with torch.no_grad(), counter:
        latent = vae.encode(x).latent_dist.mode()
        out = vae.decode(latent).sample
    return latent, out, counter.clones, calls


def _input(vae):
    patch = vae.config.get("patch_size") or 1
    channels = vae.config.in_channels // (patch * patch)
    torch.manual_seed(1)
    # Wan: 5 frames, so the causal encode runs a first chunk and then a 4-frame chunk. Qwen-Image-2.1 is image-only.
    frames = 1 if "QwenImage21" in type(vae).__name__ else 5
    return torch.randn(1, channels, frames, 32, 32)


@pytest.mark.parametrize("name", list(VAES))
def test_bitwise_stock_and_one_clone_fewer_per_block(name):
    stock = VAES[name]()
    patched = copy.deepcopy(stock)
    n = noclone.install(patched)
    downs = sum(type(m).__name__.endswith("ResidualDownBlock") for m in patched.modules())
    ups = sum(type(m).__name__.endswith("ResidualUpBlock") for m in patched.modules())
    assert downs and ups and n == downs + ups
    assert noclone.install(patched) == 0  # idempotent

    x = _input(stock)
    s_latent, s_out, s_clones, s_calls = _run(stock, x)
    p_latent, p_out, p_clones, p_calls = _run(patched, x)
    assert torch.equal(s_latent, p_latent)
    assert torch.equal(s_out, p_out)
    assert s_calls == p_calls and len(p_calls) > 0
    assert s_clones - p_clones == len(p_calls)


def test_unknown_forward_source_is_left_alone(monkeypatch):
    vae = _tiny_wan22()
    monkeypatch.setattr(noclone, "_KNOWN_DOWN", frozenset())
    monkeypatch.setattr(noclone, "_KNOWN_UP", frozenset())
    monkeypatch.setattr(noclone, "_VERIFIED", {})
    assert noclone.install(vae) == 0
    assert not any("forward" in m.__dict__ for m in vae.modules())


def test_kill_switch_and_foreign_forward(monkeypatch):
    vae = _tiny_wan22()
    up = next(m for m in vae.modules() if type(m).__name__.endswith("ResidualUpBlock"))
    foreign = up.forward
    up.forward = foreign  # e.g. the fused CUDA up block
    monkeypatch.setenv(noclone.NO_CLONE_ENV, "0")
    assert noclone.install(vae) == 0
    monkeypatch.delenv(noclone.NO_CLONE_ENV)
    assert noclone.install(vae) > 0
    assert up.forward is foreign and not getattr(up, noclone.MARK, False)


def test_fused_cuda_decoder_replaces_the_patched_up_block():
    """``install_wan_vae`` skips instance forwards it did not set, except this module's up blocks."""
    source = (BACKEND / "core" / "inference" / "diffusion_vae_fused.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "install_wan_vae"
    )
    body = ast.get_source_segment(source, fn)
    assert f'not getattr(module, "{noclone.MARK}", False)' in body


@pytest.mark.parametrize("loader", ["diffusion.py", "video.py"])
def test_both_loaders_install_it(loader):
    source = (BACKEND / "core" / "inference" / loader).read_text(encoding = "utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "install_vae_noclone"
    ]
    assert len(calls) == 1

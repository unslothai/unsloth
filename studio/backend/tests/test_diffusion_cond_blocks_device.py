# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 compact modulation and offload cache on a real CUDA / ROCm device, against a float64 CPU oracle."""

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_cond_blocks as cb  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA or ROCm device")


def _cls():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "Flux3Transformer2DModel", None)
    if cls is None or (
        cb._forward_sha(cls) not in cb._FLUX3_FORWARD_SHA and cls not in cb._PATCHED
    ):
        pytest.skip("no matching diffusers Flux3Transformer2DModel")
    return cls


def _tiny(cls, flow_map):
    torch.manual_seed(0)
    return cls(
        in_channels = 16,
        cond_in_channels = 16,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 32,
        num_attention_heads = 4,
        joint_attention_dim = 64,
        timestep_channels = 64,
        axes_dims_rope = (8, 8, 8, 8),
        flow_map = flow_map,
    ).eval()


def _inputs(refs, flow_map, device, dtype):
    g = torch.Generator().manual_seed(1)
    img, txt = 256, 24
    kw = {
        "hidden_states": torch.randn(1, img, 16, generator = g),
        "encoder_hidden_states": torch.randn(1, txt, 64, generator = g),
        "timestep": torch.tensor([0.7]),
        "img_ids": torch.randint(0, 16, (img, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 16, (txt, 4), generator = g).float(),
        "return_dict": False,
    }
    if refs:
        kw["image_cond_hidden_states"] = torch.randn(1, 192, 16, generator = g)
        kw["image_cond_ids"] = torch.randint(0, 16, (192, 4), generator = g).float()
    if flow_map:
        kw["target_timestep"] = torch.tensor([0.4])
    out = {}
    for k, v in kw.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if k.endswith("states"):
                v = v.to(dtype)
        out[k] = v
    return out


@pytest.mark.parametrize(
    "flow_map,refs", [(False, False), (False, True), (True, False), (True, True)]
)
def test_prepared_forward_tracks_the_float64_oracle(flow_map, refs):
    cls = _cls()
    ref = _tiny(cls, flow_map).double()
    with torch.no_grad():
        want = ref(**_inputs(refs, flow_map, "cpu", torch.float64))[0]
    dev = torch.device("cuda")
    stock = _tiny(cls, flow_map).to(dev, torch.bfloat16)
    prepared = _tiny(cls, flow_map).to(dev, torch.bfloat16)
    kw = _inputs(refs, flow_map, dev, torch.bfloat16)
    with torch.no_grad():
        base = stock(**kw)[0].double().cpu()
        cb.prepare(prepared)
        cb.begin_render(prepared)
        try:
            got = prepared(**kw)[0].double().cpu()
        finally:
            cb.end_render(prepared)
    err_stock = (base - want).abs().max().item()
    err_got = (got - want).abs().max().item()
    assert torch.isfinite(got).all()
    assert err_got <= 1.5 * err_stock + 1e-3, (err_got, err_stock)


def test_group_offloaded_base_edit_matches_stock():
    cls = _cls()
    from diffusers.hooks import apply_group_offloading

    dev = torch.device("cuda")
    stock = _tiny(cls, False).to(dev, torch.bfloat16)
    model = _tiny(cls, False).to(torch.bfloat16)
    apply_group_offloading(
        model,
        onload_device = dev,
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    cond = _inputs(True, False, dev, torch.bfloat16)
    uncond = dict(cond, encoder_hidden_states = torch.randn_like(cond["encoder_hidden_states"]))
    with torch.no_grad():
        cb.prepare(stock)
        cb.prepare(model)
        want, got = [], []
        for net, out in ((stock, want), (model, got)):
            cb.begin_render(net)
            try:
                for i in range(3):
                    for kw in (cond, uncond):
                        out.append(
                            net(**dict(kw, timestep = torch.tensor([0.9 - 0.2 * i], device = dev)))[0]
                        )
            finally:
                cb.end_render(net)
    for a, b in zip(want, got):
        assert torch.allclose(a.float(), b.float(), rtol = 0, atol = 1e-2), (a - b).abs().max()
    stats = cb.status(model) or {}
    if getattr(torch.version, "hip", None) is None and hasattr(cb, "offload_cache_enabled"):
        assert (stats.get("misses"), stats.get("hits"), stats.get("in_forward")) == (2, 4, 2)
    else:
        assert not stats.get("hits") and not stats.get("misses")

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CFG replica re-fuses MOSS-Image's and Z-Image's Q/K/V on its own card instead of keeping the source's fused
Linear (a second copy on the source card, and for MOSS a GEMM across cards)."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_multigpu as mg  # noqa: E402
from core.inference.diffusion_moss_image import FUSED_QKV_ATTR, _Block  # noqa: E402
from core.inference.diffusion_zimage_fused import _QKV_ATTR as Z_ATTR  # noqa: E402
from core.inference.diffusion_zimage_fused import _fuse_linears, _share_storage  # noqa: E402


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(256, 512, 2)])
        attn = nn.Module()
        attn.to_q, attn.to_k, attn.to_v = (nn.Linear(256, 256, bias = False) for _ in range(3))
        self.attn = attn


def _fused_source():
    src = _Root().to(torch.bfloat16)
    block = src.blocks[0]
    parts = [block.proj[n] for n in ("q", "k", "v")]
    fused = _fuse_linears(parts)
    assert fused is not None and _share_storage(fused, parts)
    block.__dict__[FUSED_QKV_ATTR] = (fused, *parts, None)
    zparts = [src.attn.to_q, src.attn.to_k, src.attn.to_v]
    zfused = _fuse_linears(zparts)
    assert zfused is not None and _share_storage(zfused, zparts)
    src.attn.__dict__[Z_ATTR] = (zfused, *zparts)
    return src


def _storage(t):
    return t.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "owner, attr, names",
    [("moss", FUSED_QKV_ATTR, ("q", "k", "v")), ("zimage", Z_ATTR, ("to_q", "to_k", "to_v"))],
)
def test_the_replica_fuses_its_own_qkv(owner, attr, names):
    src = _fused_source()
    rep = mg._copy_module(src, torch.device("cpu"))
    s_mod, r_mod = (src.blocks[0], rep.blocks[0]) if owner == "moss" else (src.attn, rep.attn)
    part = (lambda m, n: m.proj[n]) if owner == "moss" else getattr
    s_rec, r_rec = s_mod.__dict__[attr], r_mod.__dict__.get(attr)
    assert r_rec is not None, "the replica runs one Q/K/V GEMM like its source"
    assert r_rec[0] is not s_rec[0] and _storage(r_rec[0].weight) != _storage(s_rec[0].weight)
    for i, name in enumerate(names, start = 1):
        assert r_rec[i] is part(r_mod, name), "the record names the replica's own projections"
        assert _storage(part(r_mod, name).weight) == _storage(
            r_rec[0].weight
        ), "parts view the replica's buffer"
    y = torch.randn(3, 256, dtype = torch.bfloat16)
    assert torch.equal(r_rec[0](y), s_rec[0](y))


def test_no_source_record_survives_a_copy_to_another_device():
    src = _fused_source()
    rep = mg._copy_module(src, torch.device("meta"))
    for mod in (rep.blocks[0], rep.attn):
        for attr in (FUSED_QKV_ATTR, Z_ATTR):
            rec = mod.__dict__.get(attr)
            assert (
                rec is None or rec[0].weight.device.type == "meta"
            ), "never the source card's fused Linear"

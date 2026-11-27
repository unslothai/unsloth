# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video LoRAs in the community layout (fused attn.qkv_proj, gated mlp.fc1) bake into the diffusers
denoiser. The row map is checked against real rows of the released weights (tests/fixtures/h3_community_lora_rows.json:
the first 64 columns of block 0 / 30 rows from the community-layout bf16 file and from the diffusers shards)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from core.inference import video_lora  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "h3_community_lora_rows.json"
H3 = "minimax-h3"


def _rows():
    return json.loads(FIXTURE.read_text(encoding = "utf-8"))


def _index_lora(
    tmp_path,
    shapes,
    prefix = "diffusion_model.",
    suffix = ".lora_{}.weight",
    alpha = False,
):
    """Rank-1 LoRA whose B column holds each row's own index, so a mapped B says where every target row came from."""
    st = {}
    for module, rows in shapes.items():
        st[prefix + module + suffix.format("A")] = torch.ones(1, 8)
        st[prefix + module + suffix.format("B")] = torch.arange(rows, dtype = torch.float32).reshape(
            rows, 1
        )
        if alpha:
            st[prefix + module + ".alpha"] = torch.tensor(1.0)
    path = tmp_path / "h3_community.safetensors"
    save_file(st, str(path))
    return str(path)


def _sources(deltas, target):
    ((_a, b, _s),) = deltas[target]
    return [int(v) for v in b[:, 0].tolist()]


@pytest.mark.parametrize(
    "prefix, suffix, alpha",
    [
        ("diffusion_model.", ".lora_{}.weight", False),  # trainer / ai-toolkit exports
        ("diffusion_model.", ".lora_{}.weight", True),  # generic LoRA with .alpha keys
        ("", ".lora_{}.default.weight", False),  # PEFT-suffixed export of the original module names
    ],
)
def test_community_rows_map_bit_equal_to_the_released_weights(tmp_path, prefix, suffix, alpha):
    fx = _rows()
    shapes = {}
    for blk in (0, 30):
        shapes[f"blocks.{blk}.attn.qkv_proj"] = fx["shapes"][f"blocks.{blk}.attn.qkv_proj.weight"][
            0
        ]
        shapes[f"blocks.{blk}.mlp.fc1"] = fx["shapes"][f"blocks.{blk}.mlp.fc1.weight"][0]
    deltas = video_lora.read_deltas(
        [(_index_lora(tmp_path, shapes, prefix, suffix, alpha), 1.0)], family = H3
    )
    checked = 0
    for dkey, bits in fx["diffusers"].items():
        name, row = dkey.rsplit("@", 1)
        blk = name.split(".")[1]
        src_key = (
            f"blocks.{blk}.mlp.fc1.weight"
            if ".ff.net.0.proj." in name
            else f"blocks.{blk}.attn.qkv_proj.weight"
        )
        src = _sources(deltas, name[: -len(".weight")])[int(row)]
        assert fx["community"][f"{src_key}@{src}"] == bits, (dkey, src)
        checked += 1
    assert checked == 20


def test_identity_row_map_does_not_match_the_released_weights():
    # The trap the test above guards: fc1 and ff.net.0.proj share a shape, so an unswapped (or unsplit) map loads silently.
    fx = _rows()
    for blk in (0, 30):
        for name, src_name, rows in (
            ("attn.to_k", "attn.qkv_proj", (0, 1)),
            ("attn.to_v", "attn.qkv_proj", (0, 1)),
            ("ff.net.0.proj", "mlp.fc1", (0, 1, 14336, 14337)),
        ):
            for r in rows:
                got = fx["diffusers"][f"transformer_blocks.{blk}.{name}.weight@{r}"]
                assert got != fx["community"][f"blocks.{blk}.{src_name}.weight@{r}"], (blk, name, r)


def test_community_module_names(tmp_path):
    shapes = {
        "blocks.3.attn.qkv_proj": 12,
        "blocks.3.attn.out_proj": 4,
        "blocks.3.mlp.fc1": 8,
        "blocks.3.mlp.fc2": 4,
        "blocks.3.adaln_proj.linear": 6,
        "token_refiner.blocks.1.attn.qkv_proj": 12,
        "token_refiner.blocks.1.mlp.fc1": 8,
    }
    deltas = video_lora.read_deltas([(_index_lora(tmp_path, shapes), 1.0)], family = H3)
    assert sorted(deltas) == sorted(
        [
            "transformer_blocks.3.attn.to_q",
            "transformer_blocks.3.attn.to_k",
            "transformer_blocks.3.attn.to_v",
            "transformer_blocks.3.attn.to_out.0",
            "transformer_blocks.3.ff.net.0.proj",
            "transformer_blocks.3.ff.net.2",
            "transformer_blocks.3.adaln_proj.linear",
            "token_refiner.refiner_blocks.1.attn.to_q",
            "token_refiner.refiner_blocks.1.attn.to_k",
            "token_refiner.refiner_blocks.1.attn.to_v",
            "token_refiner.refiner_blocks.1.ff.net.0.proj",
        ]
    )
    assert _sources(deltas, "transformer_blocks.3.attn.to_k") == [4, 5, 6, 7]
    assert _sources(deltas, "transformer_blocks.3.ff.net.0.proj") == [4, 5, 6, 7, 0, 1, 2, 3]
    assert _sources(deltas, "transformer_blocks.3.ff.net.2") == [0, 1, 2, 3]


def test_other_families_keep_the_names(tmp_path):
    path = _index_lora(tmp_path, {"blocks.3.attn.qkv_proj": 12})
    assert list(video_lora.read_deltas([(path, 1.0)])) == ["blocks.3.attn.qkv_proj"]


def test_peft_layout_files_are_unchanged(tmp_path):
    torch.manual_seed(0)
    st = {}
    for m in ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.ff.net.0.proj"):
        st[f"transformer.{m}.lora_A.weight"] = torch.randn(2, 8)
        st[f"transformer.{m}.lora_B.weight"] = torch.randn(6, 2)
    path = tmp_path / "peft.safetensors"
    save_file(st, str(path))
    plain = video_lora.read_deltas([(str(path), 0.5)])
    h3 = video_lora.read_deltas([(str(path), 0.5)], family = H3)
    assert list(plain) == list(h3)
    for m in plain:
        ((a0, b0, s0),) = plain[m]
        ((a1, b1, s1),) = h3[m]
        assert torch.equal(a0, a1) and torch.equal(b0, b1) and s0 == s1


def _h3_like():
    torch.manual_seed(1)
    net = torch.nn.Module()
    shapes = {
        "transformer_blocks.0.attn.to_q": (4, 8),
        "transformer_blocks.0.attn.to_k": (4, 8),
        "transformer_blocks.0.attn.to_v": (4, 8),
        "transformer_blocks.0.attn.to_out.0": (8, 4),
        "transformer_blocks.0.ff.net.0.proj": (12, 8),
        "transformer_blocks.0.ff.net.2": (8, 6),
    }
    for fqn, (out, inp) in shapes.items():
        parent = net
        parts = fqn.split(".")
        for p in parts[:-1]:
            if not hasattr(parent, p):
                parent.add_module(p, torch.nn.Module())
            parent = getattr(parent, p)
        lin = torch.nn.Linear(inp, out, bias = False)
        lin.weight.data = torch.randn(out, inp).to(torch.bfloat16)
        parent.add_module(parts[-1], lin)
    return net


def test_community_bake_equals_the_converted_peft_bake(tmp_path):
    torch.manual_seed(2)
    a = {
        m: torch.randn(2, i)
        for m, i in (("attn.qkv_proj", 8), ("attn.out_proj", 4), ("mlp.fc1", 8), ("mlp.fc2", 6))
    }
    b = {
        m: torch.randn(o, 2)
        for m, o in (("attn.qkv_proj", 12), ("attn.out_proj", 8), ("mlp.fc1", 12), ("mlp.fc2", 8))
    }
    community = {}
    for m in a:
        community[f"diffusion_model.blocks.0.{m}.lora_A.weight"] = a[m]
        community[f"diffusion_model.blocks.0.{m}.lora_B.weight"] = b[m]
    cpath = tmp_path / "community.safetensors"
    save_file(community, str(cpath))
    # the same adapter hand-converted to diffusers names: q / k / v thirds, fc1 halves swapped
    q, k, v = b["attn.qkv_proj"].split(4)
    gate, value = b["mlp.fc1"].split(6)
    peft = {}
    for name, aa, bb in (
        ("attn.to_q", a["attn.qkv_proj"], q),
        ("attn.to_k", a["attn.qkv_proj"], k),
        ("attn.to_v", a["attn.qkv_proj"], v),
        ("attn.to_out.0", a["attn.out_proj"], b["attn.out_proj"]),
        ("ff.net.0.proj", a["mlp.fc1"], torch.cat([value, gate])),
        ("ff.net.2", a["mlp.fc2"], b["mlp.fc2"]),
    ):
        peft[f"transformer.transformer_blocks.0.{name}.lora_A.weight"] = aa.clone()
        peft[f"transformer.transformer_blocks.0.{name}.lora_B.weight"] = bb.clone()
    ppath = tmp_path / "peft.safetensors"
    save_file(peft, str(ppath))

    got, want = _h3_like(), _h3_like()
    assert video_lora.bake_into_module_(got, [(str(cpath), 0.8)], family = H3) == 6
    assert video_lora.bake_into_module_(want, [(str(ppath), 0.8)]) == 6
    for (n, p), (_n, w) in zip(got.named_parameters(), want.named_parameters()):
        assert torch.equal(p, w), n
    with pytest.raises(ValueError, match = "does not have"):
        video_lora.bake_into_module_(_h3_like(), [(str(cpath), 0.8)])


def test_key_map_rows():
    assert video_lora.h3_community_key_map("blocks.2.attn.qkv_proj", 9) == [
        ("transformer_blocks.2.attn.to_q", [(0, 3)]),
        ("transformer_blocks.2.attn.to_k", [(3, 3)]),
        ("transformer_blocks.2.attn.to_v", [(6, 3)]),
    ]
    assert video_lora.h3_community_key_map("blocks.2.mlp.fc1", 8) == [
        ("transformer_blocks.2.ff.net.0.proj", [(4, 4), (0, 4)])
    ]
    with pytest.raises(ValueError):
        video_lora.h3_community_key_map("blocks.2.attn.qkv_proj", 10)

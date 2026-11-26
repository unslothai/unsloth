# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""musubi-tuner Wan LoRAs bake into diffusers Wan exactly like Wan's dotted names, incl. I2V k_img / v_img."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from core.inference import video_lora

RANK = 4
_T2V = {
    "self_attn_q": ("self_attn.q", "attn1.to_q"),
    "self_attn_k": ("self_attn.k", "attn1.to_k"),
    "self_attn_v": ("self_attn.v", "attn1.to_v"),
    "self_attn_o": ("self_attn.o", "attn1.to_out.0"),
    "cross_attn_q": ("cross_attn.q", "attn2.to_q"),
    "cross_attn_k": ("cross_attn.k", "attn2.to_k"),
    "cross_attn_v": ("cross_attn.v", "attn2.to_v"),
    "cross_attn_o": ("cross_attn.o", "attn2.to_out.0"),
    "ffn_0": ("ffn.0", "ffn.net.0.proj"),
    "ffn_2": ("ffn.2", "ffn.net.2"),
}
_I2V = {
    **_T2V,
    "cross_attn_k_img": ("cross_attn.k_img", "attn2.add_k_proj"),
    "cross_attn_v_img": ("cross_attn.v_img", "attn2.add_v_proj"),
}
_NUM_LAYERS = 2


def _tiny_wan(i2v: bool) -> torch.nn.Module:
    WanTransformer3DModel = pytest.importorskip("diffusers").WanTransformer3DModel

    torch.manual_seed(0)
    extra = {"image_dim": 16, "added_kv_proj_dim": 16} if i2v else {}
    return WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 8,
        in_channels = 4,
        out_channels = 4,
        text_dim = 16,
        freq_dim = 16,
        ffn_dim = 32,
        num_layers = _NUM_LAYERS,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        rope_max_seq_len = 32,
        **extra,
    ).to(torch.bfloat16)


def _musubi_state(module, modules, alpha):
    gen = torch.Generator().manual_seed(1)
    params = dict(module.named_parameters())
    state = {}
    for n in range(_NUM_LAYERS):
        for stem, (_dotted, leaf) in modules.items():
            out_f, in_f = params[f"blocks.{n}.{leaf}.weight"].shape
            key = f"lora_unet_blocks_{n}_{stem}"
            state[f"{key}.lora_down.weight"] = (torch.randn(RANK, in_f, generator = gen) * 0.1).to(
                torch.bfloat16
            )
            state[f"{key}.lora_up.weight"] = (torch.randn(out_f, RANK, generator = gen) * 0.1).to(
                torch.bfloat16
            )
            if alpha is not None:
                state[f"{key}.alpha"] = torch.tensor(float(alpha))
    return state


def _dotted_state(musubi, modules):
    out = {}
    for key, tensor in musubi.items():
        stem, suffix = key.split(".", 1)
        block, name = stem[len("lora_unet_blocks_") :].split("_", 1)
        out[f"diffusion_model.blocks.{block}.{modules[name][0]}.{suffix}"] = tensor
    return out


def _save(tmp_path, name, state):
    path = tmp_path / f"{name}.safetensors"
    save_file(state, str(path))
    return str(path)


@pytest.mark.parametrize("i2v", [False, True], ids = ["t2v", "i2v"])
@pytest.mark.parametrize("alpha", [None, RANK // 2, 16.0])
def test_musubi_wan_maps_like_the_dotted_wan_layout(tmp_path, i2v, alpha):
    modules = _I2V if i2v else _T2V
    module = _tiny_wan(i2v)
    musubi = _musubi_state(module, modules, alpha)
    m_path = _save(tmp_path, "musubi", musubi)
    d_path = _save(tmp_path, "dotted", _dotted_state(musubi, modules))
    assert video_lora._unbakeable_feature(m_path) is None

    got = video_lora.read_deltas([(m_path, 1.0)])
    want = video_lora.read_deltas([(d_path, 1.0)])
    expected = {f"blocks.{n}.{leaf}" for n in range(_NUM_LAYERS) for (_d, leaf) in modules.values()}
    assert set(got) == set(want) == expected
    linears = {n for n, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    assert expected <= linears
    for name in expected:
        ((a, b, scale),) = got[name]
        ((a2, b2, scale2),) = want[name]
        assert torch.equal(a, a2) and torch.equal(b, b2)
        assert scale == scale2 == (1.0 if alpha is None else float(alpha) / RANK)


@pytest.mark.parametrize("i2v", [False, True], ids = ["t2v", "i2v"])
def test_musubi_wan_bakes_every_module_into_a_diffusers_wan(tmp_path, i2v):
    modules = _I2V if i2v else _T2V
    module = _tiny_wan(i2v)
    base = {k: v.clone() for k, v in module.state_dict().items()}
    musubi = _musubi_state(module, modules, alpha = RANK // 2)
    m_path = _save(tmp_path, "musubi", musubi)
    merged = video_lora.bake_into_module_(module, [(m_path, 1.0)], label = "Wan transformer")
    assert merged == _NUM_LAYERS * len(modules) > 0
    baked = module.state_dict()

    dotted = _tiny_wan(i2v)
    d_path = _save(tmp_path, "dotted", _dotted_state(musubi, modules))
    assert video_lora.bake_into_module_(dotted, [(d_path, 1.0)]) == merged
    for key, value in dotted.state_dict().items():
        assert torch.equal(baked[key], value), key

    changed = {k[: -len(".weight")] for k in base if not torch.equal(base[k], baked[k])}
    assert changed == {
        f"blocks.{n}.{leaf}" for n in range(_NUM_LAYERS) for (_d, leaf) in modules.values()
    }


def test_musubi_wan_delta_matches_the_diffusers_musubi_conversion(tmp_path):
    """Diffusers' own musubi Wan converter folds alpha / rank into A and B; the baked delta is the same."""
    conv = pytest.importorskip("diffusers.loaders.lora_conversion_utils")
    if not hasattr(conv, "_convert_musubi_wan_lora_to_diffusers"):
        pytest.skip("diffusers has no musubi Wan converter")
    module = _tiny_wan(True)
    musubi = _musubi_state(module, _I2V, alpha = 3.0)
    ref = conv._convert_musubi_wan_lora_to_diffusers(
        {k: v.float().clone() for k, v in musubi.items()}
    )
    got = video_lora.read_deltas([(_save(tmp_path, "musubi", musubi), 1.0)])
    assert {
        k[len("transformer.") : -len(".lora_A.weight")] for k in ref if k.endswith(".lora_A.weight")
    } == set(got)
    for name, ((a, b, scale),) in got.items():
        want = ref[f"transformer.{name}.lora_B.weight"] @ ref[f"transformer.{name}.lora_A.weight"]
        torch.testing.assert_close(scale * (b @ a), want, rtol = 1e-5, atol = 1e-6)


def test_a_musubi_key_for_a_layer_wan_lacks_is_still_refused(tmp_path):
    """T2V refuses I2V k_img keys and unknown flattened modules stay unmapped: nothing is silently skipped."""
    t2v = _tiny_wan(False)
    i2v_state = _musubi_state(_tiny_wan(True), _I2V, alpha = None)
    with pytest.raises(ValueError, match = "different model"):
        video_lora.bake_into_module_(t2v, [(_save(tmp_path, "i2v", i2v_state), 1.0)])
    odd = {
        "lora_unet_blocks_0_self_attn_norm_q.lora_down.weight": torch.zeros(RANK, 16),
        "lora_unet_blocks_0_self_attn_norm_q.lora_up.weight": torch.zeros(16, RANK),
    }
    assert set(video_lora.read_deltas([(_save(tmp_path, "odd", odd), 1.0)])) == {
        "lora_unet_blocks_0_self_attn_norm_q"
    }
    with pytest.raises(ValueError, match = "different model"):
        video_lora.bake_into_module_(t2v, [(_save(tmp_path, "odd2", odd), 1.0)])


def test_musubi_full_weight_keys_are_refused_before_the_bake(tmp_path):
    state = {
        "lora_unet_blocks_0_self_attn_q.lora_down.weight": torch.zeros(RANK, 16),
        "lora_unet_blocks_0_self_attn_q.lora_up.weight": torch.zeros(16, RANK),
        "lora_unet_blocks_0_norm3.diff_b": torch.zeros(16),
    }
    assert (
        video_lora._unbakeable_feature(_save(tmp_path, "diff", state))
        == "full-weight (modules_to_save or diff)"
    )

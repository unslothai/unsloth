# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video LoRAs as an unmerged side branch: base weights untouched (bf16 or hosted int8 ConvRot codes), the
forward is base(x) + (x A^T) B^T with the files' scales folded into B, stacked adapters sum, nothing half-attached."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from safetensors.torch import save_file  # noqa: E402

from core.inference import video_lora  # noqa: E402

H3 = "minimax-h3"
D = 32


def _net(quantized: bool = False):
    torch.manual_seed(0)
    net = torch.nn.Module()
    net.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    blk = net.transformer_blocks[0]
    blk.attn = torch.nn.Module()
    for name, (out, inp) in {"to_q": (48, D), "to_k": (48, D), "to_v": (48, D)}.items():
        lin = torch.nn.Linear(inp, out, bias = False)
        w = torch.randn(out, inp) * 0.05
        if quantized:
            from core.inference import prequant_legacy_int8 as li8
            from core.inference.diffusion_convrot import (
                _install_rotation,
                quantize_convrot_int8_weight,
            )

            codes, scale = quantize_convrot_int8_weight(w, 16)
            lin.weight = torch.nn.Parameter(
                li8._to_int8_tensor(codes, scale, torch.bfloat16, li8._int8_tensor_api()),
                requires_grad = False,
            )
            _install_rotation(lin, 16)
        else:
            lin.weight = torch.nn.Parameter(w.to(torch.bfloat16), requires_grad = False)
        setattr(blk.attn, name, lin)
    return net


def _community(
    tmp_path,
    name = "c",
    seed = 1,
    alpha = None,
):
    torch.manual_seed(seed)
    st = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(4, D) * 0.1,
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(144, 4) * 0.1,
    }
    if alpha is not None:
        st["diffusion_model.blocks.0.attn.qkv_proj.alpha"] = torch.tensor(float(alpha))
    path = tmp_path / f"{name}.safetensors"
    save_file(st, str(path))
    return str(path), st


@pytest.mark.parametrize("quantized", [False, True])
def test_side_branch_forward_is_base_plus_low_rank(tmp_path, quantized):
    path, st = _community(tmp_path, alpha = 8.0)
    net = _net(quantized)
    x = torch.randn(5, D, dtype = torch.bfloat16)
    base = {
        n: getattr(net.transformer_blocks[0].attn, n)(x).float() for n in ("to_q", "to_k", "to_v")
    }
    codes = net.transformer_blocks[0].attn.to_k.weight.qdata.clone() if quantized else None
    assert video_lora.attach_side_branch_(net, [(path, 0.5)], family = H3) == 3
    a = st["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
    b = st["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]
    for i, n in enumerate(("to_q", "to_k", "to_v")):
        layer = getattr(net.transformer_blocks[0].attn, n)
        want = base[n] + (x.float() @ a.t()) @ b[48 * i : 48 * (i + 1)].t() * (8.0 / 4) * 0.5
        got = layer(x).float()
        assert torch.allclose(got, want, atol = 2e-2, rtol = 2e-2), n
        assert layer.lora_A[video_lora.SIDE_BRANCH_ADAPTER].weight.dtype == torch.bfloat16
    if quantized:
        # the hosted codes are never rewritten
        assert torch.equal(codes, net.transformer_blocks[0].attn.to_k.get_base_layer().weight.qdata)
    assert getattr(net, video_lora.LORAS_BAKED_ATTR) is True


def test_stacked_adapters_sum(tmp_path):
    p1, s1 = _community(tmp_path, "one", 1)
    p2, s2 = _community(tmp_path, "two", 2)
    net = _net()
    x = torch.randn(3, D, dtype = torch.bfloat16)
    base = net.transformer_blocks[0].attn.to_v(x).float()
    assert video_lora.attach_side_branch_(net, [(p1, 1.0), (p2, -0.5)], family = H3) == 3
    layer = net.transformer_blocks[0].attn.to_v
    assert layer.lora_A[video_lora.SIDE_BRANCH_ADAPTER].weight.shape == (8, D)
    want = base.clone()
    for st, w in ((s1, 1.0), (s2, -0.5)):
        a = st["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"]
        b = st["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"][96:]
        want += (x.float() @ a.t()) @ b.t() * w
    assert torch.allclose(layer(x).float(), want, atol = 2e-2, rtol = 2e-2)


def test_side_branch_keeps_the_bf16_weight(tmp_path):
    path, _ = _community(tmp_path)
    net = _net()
    before = net.transformer_blocks[0].attn.to_q.weight.detach().clone()
    video_lora.attach_side_branch_(net, [(path, 1.0)], family = H3)
    assert torch.equal(before, net.transformer_blocks[0].attn.to_q.get_base_layer().weight)


def test_side_branch_refuses_before_attaching(tmp_path):
    path, _ = _community(tmp_path)
    net = _net()
    del net.transformer_blocks[0].attn.to_v
    with pytest.raises(ValueError, match = "does not have"):
        video_lora.attach_side_branch_(net, [(path, 1.0)], family = H3)
    assert type(net.transformer_blocks[0].attn.to_q) is torch.nn.Linear


def test_switch_is_h3_only_and_on_by_default(monkeypatch):
    class H3Fam:
        name, label, modular_workflow = "minimax-h3", "MiniMax-H3", "fl2va"

    class Wan:
        name = "wan2.2"

    loras = [("x", 1.0)]
    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    assert video_lora.side_branch_enabled(H3Fam)
    assert not video_lora.side_branch_enabled(Wan)
    # the hosted codes are never rewritten, so an unset precision keeps the hosted pick and int8 is no longer refused
    assert video_lora.dense_transformer_quant(H3Fam, None, loras) is None
    assert video_lora.load_refusal(H3Fam, "pipeline", "int8", loras) is None
    monkeypatch.setenv(video_lora.SIDE_BRANCH_ENV, "0")
    assert "bfloat16" in video_lora.load_refusal(H3Fam, "pipeline", "fp8", loras)
    assert not video_lora.side_branch_enabled(H3Fam)
    assert video_lora.dense_transformer_quant(H3Fam, None, loras) == "none"
    assert video_lora.load_refusal(H3Fam, "pipeline", "int8", loras) is not None


def test_side_branch_reaches_through_the_small_m_pad_wrapper(tmp_path):
    from core.inference.diffusion_quant_pad import PadToMinM

    path, _ = _community(tmp_path)
    net = _net(True)
    attn = net.transformer_blocks[0].attn
    attn.to_k = PadToMinM(attn.to_k)
    assert video_lora.attach_side_branch_(net, [(path, 1.0)], family = H3) == 3
    assert type(attn.to_k) is PadToMinM and hasattr(attn.to_k.inner, "lora_A")


def test_hosted_fp8_is_refused_before_the_load(monkeypatch):
    # On Ampere the hosted fp8 denoiser becomes stored-fp8 native layers (not nn.Linear), which the side branch
    # cannot wrap, so the pick is refused up front instead of failing on every projection after the load.
    class H3Fam:
        name, label, modular_workflow = "minimax-h3", "MiniMax-H3", "fl2va"

    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    loras = [("x", 1.0)]
    why = video_lora.load_refusal(H3Fam, "pipeline", "fp8", loras)
    assert why is not None and "'fp8'" in why and "int8" in why
    assert video_lora.load_refusal(H3Fam, "pipeline", "int8", loras) is None
    assert video_lora.load_refusal(H3Fam, "pipeline", "none", loras) is None
    assert video_lora.load_refusal(H3Fam, "pipeline", None, loras) is None


def test_side_branch_refuses_a_non_linear_layer(tmp_path):
    from core.inference.diffusion_native_quant import NATIVE_FP8, native_linear_class

    path, _ = _community(tmp_path)
    net = _net()
    attn = net.transformer_blocks[0].attn
    stand_in = torch.nn.Linear(D, 48, bias = False, device = "meta", dtype = torch.bfloat16)
    codes = torch.zeros(48, D).to(torch.float8_e4m3fn)
    attn.to_k = native_linear_class()(stand_in, NATIVE_FP8, codes = codes, scale = torch.ones(48, 1))
    with pytest.raises(ValueError, match = "not a Linear"):
        video_lora.attach_side_branch_(net, [(path, 1.0)], family = H3)
    assert type(attn.to_q) is torch.nn.Linear


@pytest.mark.parametrize(
    "key",
    [
        "diffusion_model.blocks.0.adaln_proj.linear",
        "transformer.transformer_blocks.0.adaln_proj.linear",
        "transformer.norm_out.linear",
        "transformer.time_embedder.linear_1",
    ],
)
def test_modulation_lora_pins_the_released_denoiser(tmp_path, monkeypatch, key):
    # The hosted int8 H3 denoiser is the pruned-modulation (curve) form: adaln_proj / norm_out take 8 inputs, not
    # 2688, and the time embedder is replaced, so an adapter on those layers cannot ride on it.
    class H3Fam:
        name, label, modular_workflow = "minimax-h3", "MiniMax-H3", "fl2va"

    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    path = tmp_path / "mod.safetensors"
    save_file(
        {f"{key}.lora_A.weight": torch.zeros(4, D), f"{key}.lora_B.weight": torch.zeros(8, 4)},
        str(path),
    )
    resolved = (("mod", str(path), 1.0, None),)
    assert video_lora.side_branch_transformer_quant(H3Fam, None, resolved) == "none"
    assert video_lora.side_branch_transformer_quant(H3Fam, "auto", resolved) == "none"
    assert video_lora.side_branch_transformer_quant(H3Fam, "none", resolved) == "none"
    with pytest.raises(ValueError, match = "pruned modulation"):
        video_lora.side_branch_transformer_quant(H3Fam, "int8", resolved)
    monkeypatch.setenv(video_lora.SIDE_BRANCH_ENV, "0")
    assert video_lora.side_branch_transformer_quant(H3Fam, None, resolved) is None


def test_attention_lora_keeps_the_hosted_int8_pick(tmp_path, monkeypatch):
    class H3Fam:
        name, label, modular_workflow = "minimax-h3", "MiniMax-H3", "fl2va"

    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    path, _ = _community(tmp_path)
    resolved = (("c", str(path), 1.0, None),)
    assert video_lora.side_branch_transformer_quant(H3Fam, None, resolved) is None
    assert video_lora.side_branch_transformer_quant(H3Fam, "int8", resolved) == "int8"


def test_load_repins_after_resolving_the_loras():
    import inspect

    from core.inference import video

    src = inspect.getsource(video.VideoBackend)
    resolve_at = src.index("baked_loras = video_lora.resolve(")
    pin_at = src.index("video_lora.side_branch_transformer_quant(", resolve_at)
    assert pin_at < src.index("is_h3_native(fam, kind)", resolve_at)


class _H3PlanFam:
    name, label, modular_workflow = "minimax-h3", "MiniMax-H3", "fl2va"


def _modulation_file(tmp_path, key = "diffusion_model.blocks.0.adaln_proj.linear"):
    path = tmp_path / "mod.safetensors"
    save_file(
        {f"{key}.lora_A.weight": torch.zeros(4, D), f"{key}.lora_B.weight": torch.zeros(8, 4)},
        str(path),
    )
    return str(path)


def _resolve_to(monkeypatch, path):
    seen = {}

    def fake_resolve(specs, **kw):
        seen.update(kw)
        if path is None:
            raise FileNotFoundError("not cached")
        return (("x", path, 1.0, None),)

    monkeypatch.setattr(video_lora, "resolve", fake_resolve)
    return seen


def test_plan_pins_the_released_denoiser_for_a_modulation_lora(tmp_path, monkeypatch):
    # Planned before the denoiser shards: an adaln_proj LoRA with precision unset must stage the released bf16
    # shards, not the hosted int8 checkpoint the assembly would then abandon.
    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    seen = _resolve_to(monkeypatch, _modulation_file(tmp_path))
    loras = [("x", 1.0)]
    assert video_lora.planned_side_branch_quant(_H3PlanFam, None, loras, hf_token = None) == "none"
    assert seen["local_files_only"] is True
    with pytest.raises(ValueError, match = "pruned modulation"):
        video_lora.planned_side_branch_quant(_H3PlanFam, "int8", loras, hf_token = None)


def test_plan_keeps_the_int8_pick_for_an_attention_lora(tmp_path, monkeypatch):
    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    path, _ = _community(tmp_path)
    _resolve_to(monkeypatch, path)
    loras = [("x", 1.0)]
    assert video_lora.planned_side_branch_quant(_H3PlanFam, None, loras, hf_token = None) is None
    assert video_lora.planned_side_branch_quant(_H3PlanFam, "int8", loras, hf_token = None) == "int8"


def test_plan_with_an_uncached_lora_stages_the_released_denoiser(monkeypatch):
    # Targets unknown before the adapter is cached: the conservative plan includes the dense shards.
    monkeypatch.delenv(video_lora.SIDE_BRANCH_ENV, raising = False)
    _resolve_to(monkeypatch, None)
    loras = [("x", 1.0)]
    assert video_lora.planned_side_branch_quant(_H3PlanFam, None, loras, hf_token = None) == "none"
    assert video_lora.planned_side_branch_quant(_H3PlanFam, "int8", loras, hf_token = None) == "int8"
    assert video_lora.planned_side_branch_quant(_H3PlanFam, "fp8", loras, hf_token = None) == "fp8"


def test_download_planners_decide_before_planning_the_denoiser():
    import inspect

    from core.inference import video
    for fn, marker in (
        (video.VideoBackend.download_plan, "_h3_planned_auto_denoiser_scheme("),
        (video.VideoBackend._run_load, "_denoiser_prequant_verified("),
    ):
        src = inspect.getsource(fn)
        pin_at = src.index("video_lora.planned_side_branch_quant(")
        assert pin_at < src.index("is_h3_native(fam, kind)")
        assert pin_at < src.index(marker)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan2.2-A14B step-distill LoRA pairs: the high-noise file bakes into ``transformer``, the low-noise one into
``transformer_2``, Wan's original module names (``self_attn.q``, ``ffn.0``) map onto diffusers', and a baked
step-distilled adapter sets the generation defaults (4 steps at CFG 1). Image Lightning adapters carry the same
preset. CPU only, tiny modules, no network."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from core.inference import diffusion_lora, video_lora
from core.inference.video_families import detect_video_family

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_video_lora_load import _WAN_LEAVES, _WanLike  # noqa: E402

RANK = 4
PAIR = "lightx2v/Wan2.2-Distill-Loras"
LIGHTNING = "lightx2v/Wan2.2-Lightning"
# Wan's original names for the diffusers leaves, as lightx2v and kohya write them.
_NATIVE = {
    "attn1.to_q": "self_attn.q",
    "attn1.to_k": "self_attn.k",
    "attn1.to_v": "self_attn.v",
    "attn1.to_out.0": "self_attn.o",
    "attn2.to_q": "cross_attn.q",
    "attn2.to_k": "cross_attn.k",
    "attn2.to_v": "cross_attn.v",
    "attn2.to_out.0": "cross_attn.o",
    "ffn.net.0.proj": "ffn.0",
    "ffn.net.2": "ffn.2",
}


def _write_native(
    path: Path,
    module: torch.nn.Module,
    *,
    seed: int,
    alpha = None,
):
    """A lightx2v-layout adapter: ``diffusion_model.blocks.N.<wan name>.lora_down/up.weight`` (+ ``.alpha``)."""
    gen = torch.Generator().manual_seed(seed)
    params = dict(module.named_parameters())
    state, factors = {}, {}
    for n in range(len(module.blocks)):
        for leaf in _WAN_LEAVES:
            out_f, in_f = params[f"blocks.{n}.{leaf}.weight"].shape
            a = (torch.randn(RANK, in_f, generator = gen) * 0.1).to(torch.bfloat16)
            b = (torch.randn(out_f, RANK, generator = gen) * 0.1).to(torch.bfloat16)
            key = f"diffusion_model.blocks.{n}.{_NATIVE[leaf]}"
            state[f"{key}.lora_down.weight"] = a
            state[f"{key}.lora_up.weight"] = b
            if alpha is not None:
                state[f"{key}.alpha"] = torch.tensor(alpha, dtype = torch.int64)
            factors[f"blocks.{n}.{leaf}"] = (a.float(), b.float())
    save_file(state, str(path))
    return factors


def _delta(module, base, name):
    return module.state_dict()[f"{name}.weight"].float() - base[f"{name}.weight"].float()


def _assert_baked(module, base, factors, scale):
    for name, (a, b) in factors.items():
        want = (base[f"{name}.weight"].float() + scale * (b @ a)).to(torch.bfloat16)
        assert torch.allclose(
            module.state_dict()[f"{name}.weight"].float(), want.float(), atol = 1e-2
        ), name


@pytest.mark.parametrize("alpha", [None, RANK // 2])
def test_wan_native_names_bake_into_the_diffusers_modules(tmp_path, alpha):
    torch.manual_seed(0)
    module = _WanLike().to(torch.bfloat16)
    base = {k: v.clone() for k, v in module.state_dict().items()}
    path = tmp_path / "native.safetensors"
    factors = _write_native(path, module, seed = 1, alpha = alpha)
    merged = video_lora.bake_into_module_(module, [(str(path), 1.0)])
    assert merged == 2 * len(_WAN_LEAVES)
    # No alpha: alpha = rank (scale 1), as diffusers reads such a file; an ``.alpha`` key is alpha / rank.
    _assert_baked(module, base, factors, scale = 1.0 if alpha is None else alpha / RANK)


def test_a_pair_bakes_high_noise_into_transformer_and_low_noise_into_transformer_2(tmp_path):
    from core.inference.video import VideoBackend

    torch.manual_seed(0)
    high, low = _WanLike().to(torch.bfloat16), _WanLike().to(torch.bfloat16)
    base_high = {k: v.clone() for k, v in high.state_dict().items()}
    base_low = {k: v.clone() for k, v in low.state_dict().items()}
    f_high = _write_native(tmp_path / "high.safetensors", high, seed = 1)
    f_low = _write_native(tmp_path / "low.safetensors", low, seed = 2)
    pipe = types.SimpleNamespace(transformer = high, transformer_2 = low)
    fam = detect_video_family("", override = "wan2.2-t2v-a14b")
    assert fam is not None and fam.is_moe
    merged = VideoBackend._bake_video_loras(
        pipe,
        (
            (
                "pair",
                str(tmp_path / "high.safetensors"),
                1.0,
                str(tmp_path / "low.safetensors"),
            ),
        ),
        [{"id": PAIR, "weight": 1.0}],
        fam = fam,
        target = types.SimpleNamespace(device = "cpu"),
    )
    assert merged == 2 * 2 * len(_WAN_LEAVES)
    _assert_baked(high, base_high, f_high, 1.0)
    _assert_baked(low, base_low, f_low, 1.0)
    # Each expert got its own file, not the other's.
    name = "blocks.0.attn1.to_q"
    a, b = f_low[name]
    assert not torch.allclose(_delta(high, base_high, name), b @ a, atol = 1e-2)
    # Status records one adapter (the pair), and its recipe becomes the defaults.
    assert video_lora.baked_lora_names(pipe) == ["pair"]
    assert video_lora.baked_lora_specs(pipe) == [{"id": PAIR, "weight": 1.0}]
    assert video_lora.recommended_params(pipe) == (4, 1.0)


def test_a_missing_expert_fails_before_either_is_baked(tmp_path):
    from core.inference.video import VideoBackend

    high = _WanLike().to(torch.bfloat16)
    base = {k: v.clone() for k, v in high.state_dict().items()}
    _write_native(tmp_path / "h.safetensors", high, seed = 1)
    _write_native(tmp_path / "l.safetensors", high, seed = 2)
    pipe = types.SimpleNamespace(transformer = high, transformer_2 = None)
    fam = detect_video_family("", override = "wan2.2-t2v-a14b")
    with pytest.raises(ValueError, match = "low-noise denoiser did not load"):
        VideoBackend._bake_video_loras(
            pipe,
            (("pair", str(tmp_path / "h.safetensors"), 1.0, str(tmp_path / "l.safetensors")),),
            [{"id": PAIR, "weight": 1.0}],
            fam = fam,
        )
    assert all(torch.equal(v, base[k]) for k, v in high.state_dict().items())


def test_two_expert_models_take_catalog_pairs_only():
    moe = detect_video_family("", override = "wan2.2-t2v-a14b")
    wan5b = detect_video_family("", override = "wan2.2-ti2v-5b")
    # The picker is offered (no structural refusal) ...
    assert video_lora.load_refusal(moe, "pipeline", None) is None
    # ... a catalog pair loads, a single adapter does not.
    for pair in (PAIR, LIGHTNING):
        assert video_lora.load_refusal(moe, "pipeline", None, [(pair, 1.0)]) is None
    msg = video_lora.load_refusal(moe, "pipeline", None, [("someone/wan-style-lora", 1.0)])
    assert "needs a pair" in msg
    # Disabled (weight 0) picks are not loads.
    assert video_lora.load_refusal(moe, "pipeline", None, [("someone/x", 0.0)]) is None
    assert "GGUF" in video_lora.load_refusal(moe, "gguf", None, [(PAIR, 1.0)])
    assert video_lora.load_refusal(wan5b, "pipeline", None, [("someone/x", 1.0)]) is None


def test_resolve_fetches_both_files_of_a_pair_and_gates_by_shape(monkeypatch, tmp_path):
    module = _WanLike().to(torch.bfloat16)
    files = {}
    for name in ("high", "low"):
        files[name] = tmp_path / f"{name}.safetensors"
        _write_native(files[name], module, seed = 1)
    fetched = []

    def fake_download(repo_id, weight_name, token, **kwargs):
        fetched.append((repo_id, weight_name))
        return str(files["low" if "low_noise" in weight_name else "high"])

    monkeypatch.setattr(diffusion_lora, "hf_hub_download_with_xet_fallback", fake_download)
    monkeypatch.setattr(video_lora, "_fetch_hub_adapter_config", lambda *a, **k: None)
    out = video_lora.resolve(
        [(PAIR, 1.0)], family = "wan2.2-t2v-a14b", hf_token = None, expert_pair = True
    )
    assert out == (("Wan2_2-Distill-Loras", str(files["high"]), 1.0, str(files["low"])),)
    assert [w for _r, w in fetched] == [
        "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
    ]
    # A pair is family-gated to the two-expert model.
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "not the loaded"):
        video_lora.resolve([(PAIR, 1.0)], family = "wan2.2-ti2v-5b", hf_token = None)
    # A single Hub adapter resolved for a two-expert load is refused, not baked into one expert.
    with pytest.raises(diffusion_lora.LoraSelectionError, match = "needs a pair"):
        video_lora.resolve(
            [("someone/one:high_noise.safetensors", 1.0)],
            family = "wan2.2-t2v-a14b",
            hf_token = None,
            expert_pair = True,
        )


def test_distill_presets_are_listed_and_read_back():
    pairs = diffusion_lora.list_loras(family = "wan2.2-t2v-a14b")
    assert {PAIR, LIGHTNING} <= {e.id for e in pairs if e.weight_name_2}
    # Pairs never show on a single-expert family's picker.
    assert not [e for e in diffusion_lora.list_loras(family = "wan2.2-ti2v-5b") if e.weight_name_2]
    qwen = [e for e in diffusion_lora.list_loras(family = "qwen-image") if e.recommended_steps]
    assert {e.recommended_steps for e in qwen} == {4, 8}
    q8 = "lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
    assert diffusion_lora.recommended_params(["other", q8]) == (8, 1.0)
    assert diffusion_lora.recommended_params(["krea/Krea-2-LoRA-retroanime"]) is None
    assert video_lora.recommended_params(types.SimpleNamespace()) is None
    assert video_lora.recommended_params(None) is None


def test_the_lora_listing_route_carries_the_preset(monkeypatch):
    import asyncio

    from routes import models as models_route

    monkeypatch.setattr(models_route.account_access, "managed_account", lambda: False)
    body = asyncio.run(
        models_route.scan_diffusion_loras(family = "wan2.2-t2v-a14b", current_subject = "u")
    )
    row = next(r for r in body["loras"] if r["id"] == PAIR)
    assert row["recommended_steps"] == 4 and row["recommended_guidance"] == 1.0
    assert row["expert_pair"] is True


def _wan22_unipc():
    from diffusers import UniPCMultistepScheduler

    # Wan-AI/Wan2.2-T2V-A14B-Diffusers scheduler/scheduler_config.json
    return UniPCMultistepScheduler(
        beta_end = 0.02,
        beta_schedule = "linear",
        beta_start = 0.0001,
        flow_shift = 3.0,
        lower_order_final = True,
        num_train_timesteps = 1000,
        predict_x0 = True,
        prediction_type = "flow_prediction",
        solver_order = 2,
        solver_type = "bh2",
        use_flow_sigmas = True,
    )


@pytest.mark.parametrize("pair", [PAIR, LIGHTNING])
def test_a_baked_pair_samples_with_the_publishers_euler(tmp_path, pair):
    """lightx2v's recipes (LightX2V WanStepDistillScheduler, the Lightning V1.1 workflow) step first-order Euler on the
    shift-5 grid; the base Wan scheduler is 2nd-order UniPC with corrector. The bake sets the preset sampler, the
    family's static-shift rebuild keeps it, and its 4 steps equal the Euler update exactly."""
    from core.inference.diffusion_flow_shift import apply_comfy_flow_shift
    from core.inference.video import VideoBackend

    high, low = _WanLike().to(torch.bfloat16), _WanLike().to(torch.bfloat16)
    _write_native(tmp_path / "h.safetensors", high, seed = 1)
    _write_native(tmp_path / "l.safetensors", low, seed = 2)
    pipe = types.SimpleNamespace(transformer = high, transformer_2 = low, scheduler = _wan22_unipc())
    fam = detect_video_family("", override = "wan2.2-t2v-a14b")
    VideoBackend._bake_video_loras(
        pipe,
        (("pair", str(tmp_path / "h.safetensors"), 1.0, str(tmp_path / "l.safetensors")),),
        [{"id": pair, "weight": 1.0}],
        fam = fam,
    )
    assert pipe.scheduler.config.solver_order == 1
    # The load's later static shift rebuild (ComfyUI 5 for the A14B) keeps the sampler.
    apply_comfy_flow_shift(pipe, 5.0)
    sched = pipe.scheduler
    assert sched.config.solver_order == 1 and sched.config.flow_shift == 5.0
    sched.set_timesteps(4)
    sigmas = [float(s) for s in sched.sigmas]
    assert sigmas == pytest.approx([1.0, 0.9375, 0.8333, 0.625, 0.0], abs = 1e-3)
    assert [int(t) for t in sched.timesteps] == [999, 937, 833, 625]

    w = torch.randn(8, 8, dtype = torch.float64, generator = torch.Generator().manual_seed(0)) * 0.3

    def velocity(x, s):
        return torch.tanh(x @ w) + s

    x = torch.randn(2, 8, dtype = torch.float64, generator = torch.Generator().manual_seed(1))
    euler = x.clone()
    for i, t in enumerate(sched.timesteps):
        x = sched.step(velocity(x, sigmas[i]).float(), t, x.float()).prev_sample.double()
        euler = euler + (sigmas[i + 1] - sigmas[i]) * velocity(euler, sigmas[i])
    torch.testing.assert_close(x, euler, rtol = 0, atol = 1e-5)


def test_a_plain_bake_keeps_the_loaded_scheduler(tmp_path):
    from core.inference.video import VideoBackend

    module = _WanLike().to(torch.bfloat16)
    path = tmp_path / "w.safetensors"
    _write_native(path, module, seed = 1)
    pipe = types.SimpleNamespace(transformer = module, scheduler = _wan22_unipc())
    before = pipe.scheduler
    VideoBackend._bake_video_loras(
        pipe,
        (("w", str(path), 1.0, None),),
        [{"id": "someone/wan-lora", "weight": 1.0}],
        fam = detect_video_family("", override = "wan2.2-ti2v-5b"),
    )
    assert pipe.scheduler is before and before.config.solver_order == 2
    assert video_lora.recommended_scheduler(pipe) == {}


@pytest.mark.parametrize(
    "family, pairs_only",
    [("wan2.2-t2v-a14b", True), ("wan2.2-ti2v-5b", False), ("qwen-image", False), (None, False)],
)
def test_each_picker_lists_only_what_its_load_accepts(monkeypatch, tmp_path, family, pairs_only):
    """The two-expert picker shows only pairs (an untagged local adapter would be refused at load); every other
    picker, including an unfiltered one, never shows a pair."""
    import asyncio
    import json

    from routes import models as models_route

    root = tmp_path / "loras"
    root.mkdir()
    save_file(
        {"transformer.blocks.0.attn1.to_q.lora_A.weight": torch.zeros(4, 16)},
        str(root / "untagged.safetensors"),
    )
    (root / "untagged.json").write_text(json.dumps({"kind": "diffusion-lora"}), encoding = "utf-8")
    monkeypatch.setattr(diffusion_lora, "loras_dir", lambda: root)
    monkeypatch.setattr(models_route.account_access, "managed_account", lambda: False)
    body = asyncio.run(models_route.scan_diffusion_loras(family = family, current_subject = "u"))
    rows = body["loras"]
    assert rows
    assert all(r["expert_pair"] is pairs_only for r in rows), [
        (r["id"], r["expert_pair"]) for r in rows
    ]

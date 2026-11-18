# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA ModelOpt FP8 transformers (nvidia/Cosmos3-Nano-FP8, nvidia/Cosmos3-Edge-FP8): the header scan, the
refusals, and the loader over a tiny checkpoint written in the published key layout (random tensors, real configs
shrunk). The layout fixture is the real shard index of each repo at its pinned revision."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
Cosmos3OmniTransformer = getattr(diffusers, "Cosmos3OmniTransformer", None)
if Cosmos3OmniTransformer is None:  # pragma: no cover - diffusers < 0.41
    pytest.skip("diffusers has no Cosmos3OmniTransformer", allow_module_level = True)
from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion_modelopt_fp8 as mo  # noqa: E402

_LAYOUT = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "cosmos3_modelopt_fp8_layout.json").read_text()
)
NANO, EDGE = "nvidia/Cosmos3-Nano-FP8", "nvidia/Cosmos3-Edge-FP8"
_CALIB = (".weight_quantizer._amax", ".weight_quantizer._scale", ".input_quantizer._amax")


def _tiny_config(repo: str) -> dict:
    """The real transformer config shrunk (2 layers, 64 wide); every flag that shapes the key layout kept."""
    cfg = copy.deepcopy(_LAYOUT[repo]["transformer_config"])
    for k in [k for k in cfg if k.startswith("_")]:
        del cfg[k]
    cfg.update(
        hidden_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        intermediate_size = 128,
        vocab_size = 256,
        rope_axes_dim = [6, 5, 5],
    )
    if isinstance(cfg.get("rope_scaling"), dict):
        cfg["rope_scaling"] = {**cfg["rope_scaling"], "mrope_section": [6, 5, 5]}
    return cfg


def _pattern(key: str) -> str:
    return re.sub(r"^layers\.\d+\.", "layers.N.", key)


def _quantize(weight):
    amax = weight.abs().amax().float()
    scale = amax / mo.FP8_E4M3_MAX
    codes = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes, scale


def _write_checkpoint(
    root: Path,
    repo: str,
    *,
    shards: int = 2,
    mutate = None,
) -> tuple[Path, dict, dict]:
    """A diffusers pipeline folder whose transformer is a tiny ModelOpt FP8 checkpoint in ``repo``'s key layout.
    Returns (root, dequantized reference state, static input scales)."""
    torch.manual_seed(0)
    cfg = _tiny_config(repo)
    model = Cosmos3OmniTransformer.from_config(cfg)
    real = _LAYOUT[repo]["key_patterns"]
    quantized = {p[: -len(".weight_scale")] for p in real if p.endswith(".weight_scale")}
    state: dict = {}
    reference: dict = {}
    input_scales: dict = {}
    for key, value in model.state_dict().items():
        value = (value.float() + 0.02 * torch.randn_like(value.float())).to(torch.bfloat16)
        stem = key[: -len(".weight")]
        if key.endswith(".weight") and _pattern(stem) in quantized:
            codes, scale = _quantize(value)
            x_amax = torch.tensor(float(torch.rand(()) * 4 + 0.5))
            state[key] = codes
            state[stem + ".weight_scale"] = scale.reshape(()).clone()
            state[stem + ".input_scale"] = (x_amax / mo.FP8_E4M3_MAX).reshape(())
            state[stem + ".weight_quantizer._amax"] = (scale * mo.FP8_E4M3_MAX).reshape(())
            state[stem + ".weight_quantizer._scale"] = scale.reshape(()).clone()
            state[stem + ".input_quantizer._amax"] = x_amax.reshape(())
            reference[key] = codes.float() * scale
            input_scales[stem] = float(x_amax / mo.FP8_E4M3_MAX)
        else:
            state[key] = value
            reference[key] = value.float()
    if mutate is not None:
        mutate(state)
    folder = root / "transformer"
    folder.mkdir(parents = True)
    keys = sorted(state)
    per = -(-len(keys) // shards)
    weight_map = {}
    for i in range(shards):
        name = f"diffusion_pytorch_model-{i + 1:05d}-of-{shards:05d}.safetensors"
        part = {k: state[k].contiguous().clone() for k in keys[i * per : (i + 1) * per]}
        save_file(part, str(folder / name))
        weight_map.update({k: name for k in part})
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    config = {"_class_name": "Cosmos3OmniTransformer", **cfg}
    config["quantization_config"] = copy.deepcopy(_LAYOUT[repo]["quantization_config"])
    (folder / "config.json").write_text(json.dumps(config))
    (root / "model_index.json").write_text(json.dumps({"_class_name": "Cosmos3OmniPipeline"}))
    return root, reference, input_scales


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_the_tiny_checkpoint_has_the_published_key_layout(tmp_path, repo):
    root, _ref, _scales = _write_checkpoint(tmp_path, repo)
    index = json.loads(
        (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    layers = _LAYOUT[repo]["num_hidden_layers"]
    real = {
        p: c // layers if p.startswith("layers.N.") else c
        for p, c in _LAYOUT[repo]["key_patterns"].items()
    }
    tiny = Counter(_pattern(k) for k in index["weight_map"])
    tiny = {p: c // 2 if p.startswith("layers.N.") else c for p, c in tiny.items()}
    assert tiny == real


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_the_scan_finds_every_quantized_linear_and_none_the_config_excludes(tmp_path, repo):
    root, ref, _scales = _write_checkpoint(tmp_path, repo)
    scan = mo.scan_modelopt_fp8(root)
    assert scan is not None and scan.problems == []
    assert scan.quant_algo == "FP8" and scan.producer == "modelopt 0.44.0"
    fp8_weights = {
        k[: -len(".weight")]
        for k in ref
        if k.endswith(".weight") and k[: -len(".weight")] + ".weight" in ref
    }
    expected = {s for s in fp8_weights if s + ".weight" in ref and _is_quantized(repo, s)}
    assert set(scan.layers) == expected
    for stem in (
        "proj_in",
        "proj_out",
        "time_embedder.linear_1",
        "time_embedder.linear_2",
        "lm_head",
    ):
        assert stem not in scan.layers
    # both MoT paths are quantized
    assert any(".mlp_moe_gen." in s for s in scan.layers) and any(".mlp." in s for s in scan.layers)
    assert scan.resident_bytes(True) < scan.resident_bytes(False)


def _is_quantized(repo: str, stem: str) -> bool:
    return _pattern(stem) + ".weight_scale" in _LAYOUT[repo]["key_patterns"]


def test_a_plain_folder_is_not_a_modelopt_checkpoint(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"_class_name": "Cosmos3OmniTransformer"}))
    assert mo.scan_modelopt_fp8(tmp_path) is None
    assert mo.scan_modelopt_fp8(tmp_path / "missing") is None
    assert mo.scan_modelopt_fp8(None) is None


def test_the_quantization_config_reads_as_a_dict_or_a_repr_string(tmp_path):
    root, _ref, _scales = _write_checkpoint(tmp_path, NANO)
    path = root / "transformer" / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"] = repr(config["quantization_config"])
    path.write_text(json.dumps(config))
    assert mo.quantization_config(root)["quant_algo"] == "FP8"
    assert mo.scan_modelopt_fp8(root).problems == []


def _quantize_in_place(state: dict, stem: str) -> None:
    codes, scale = _quantize(state[stem + ".weight"])
    state[stem + ".weight"] = codes
    state[stem + ".weight_scale"] = scale.reshape(()).clone()
    state[stem + ".input_scale"] = torch.tensor(0.01)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda s: s.pop("layers.0.self_attn.to_q.input_scale"), "to_q.input_scale is missing"),
        (
            lambda s: s.__setitem__("layers.0.self_attn.to_q.weight_scale", torch.ones(4)),
            "only a per-tensor float32 scale",
        ),
        (
            lambda s: s.__setitem__(
                "layers.0.self_attn.to_q.k_bmm_quantizer._amax", torch.ones(())
            ),
            "does not implement",
        ),
        (
            lambda s: s.__setitem__(
                "layers.0.self_attn.to_q.weight",
                s["layers.0.self_attn.to_q.weight"].to(torch.bfloat16),
            ),
            "expected a 2-D F8_E4M3",
        ),
        (
            lambda s: _quantize_in_place(s, "proj_in"),
            "quantized but its quantization config excludes it",
        ),
    ],
)
def test_layouts_the_loader_cannot_run_are_refused(tmp_path, mutate, needle):
    root, _ref, _scales = _write_checkpoint(tmp_path, EDGE, mutate = mutate)
    scan = mo.scan_modelopt_fp8(root)
    assert scan is not None
    message = mo.modelopt_fp8_error(scan, "Cosmos3-Edge-FP8")
    assert message and needle in message, message
    with pytest.raises(ValueError, match = "ModelOpt FP8"):
        mo.load_modelopt_fp8_transformer(Cosmos3OmniTransformer, root, scan = scan)


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"quant_algo": "NVFP4"}, "ModelOpt quantization is NVFP4"),
        ({"quant_algo": "FP8_PB_WO"}, "FP8_PB_WO"),
        ({"weight_only": True}, "weight-only"),
        ({"block_quantize": 128}, "block_quantize"),
    ],
)
def test_other_modelopt_formats_are_refused_before_any_weight_is_read(tmp_path, change, needle):
    root, _ref, _scales = _write_checkpoint(tmp_path, NANO)
    path = root / "transformer" / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"].update(change)
    path.write_text(json.dumps(config))
    message = mo.modelopt_fp8_error(mo.scan_modelopt_fp8(root))
    assert message and needle in message


def test_a_scale_that_disagrees_with_its_calibrated_amax_is_refused(tmp_path):
    root, _ref, _scales = _write_checkpoint(
        tmp_path,
        EDGE,
        mutate = lambda s: s.__setitem__(
            "layers.1.mlp.up_proj.input_quantizer._amax", torch.tensor(1000.0)
        ),
    )
    with pytest.raises(ValueError, match = "does not match its calibrated amax"):
        mo.load_modelopt_fp8_transformer(Cosmos3OmniTransformer, root)


def _stock_bf16(tmp_path: Path, repo: str, reference: dict):
    """The bf16 sibling as a stock ``from_pretrained`` loads it: the dequantized weights, same config."""
    folder = tmp_path / "bf16" / "transformer"
    folder.mkdir(parents = True)
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in reference.items()},
        str(folder / "diffusion_pytorch_model.safetensors"),
    )
    (folder / "config.json").write_text(
        json.dumps({"_class_name": "Cosmos3OmniTransformer", **_tiny_config(repo)})
    )
    return Cosmos3OmniTransformer.from_pretrained(str(folder), torch_dtype = torch.bfloat16)


@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_dequant_rebuilds_the_bf16_model_a_stock_load_of_the_same_weights_gives(tmp_path, repo):
    root, ref, _scales = _write_checkpoint(tmp_path, repo)
    model = mo.load_modelopt_fp8_transformer(
        Cosmos3OmniTransformer, root, runtime = mo.RUNTIME_DEQUANT
    )
    stock = _stock_bf16(tmp_path, repo, ref)
    ours, theirs = model.state_dict(), stock.state_dict()
    assert set(ours) == set(theirs)
    for key in theirs:
        assert ours[key].dtype == theirs[key].dtype, key
        assert torch.equal(ours[key], theirs[key]), key
    # time_embedder stays fp32 as diffusers keeps it; every unquantized module is still a plain Linear
    assert model.time_embedder.linear_1.weight.dtype == torch.float32
    assert isinstance(model.proj_in, torch.nn.Linear) and isinstance(model.lm_head, torch.nn.Linear)
    for buf_name, buf in model.named_buffers():
        assert buf.dtype == dict(stock.named_buffers())[buf_name].dtype, buf_name
    assert getattr(model, "_unsloth_source_precision", None) == "fp8"
    assert getattr(model, mo.MODELOPT_ATTR)["dequantized"] == len(mo.scan_modelopt_fp8(root).layers)


def _w8a8_reference(x, w, s_x, act):
    """The exact emulation: the same e4m3 activation codes (static scale, or per-row dynamic) times the same weights."""
    xb = x.to(torch.bfloat16).float()
    if act == "dynamic":
        s_x = xb.abs().amax(dim = -1, keepdim = True).clamp(min = 1e-12) / 448
    return (xb / s_x).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s_x @ w.T


@pytest.mark.parametrize("runtime", ["static", "dynamic"])
@pytest.mark.parametrize("repo", [NANO, EDGE])
def test_the_w8a8_linears_match_the_dequantized_reference_within_fp8_tolerance(
    tmp_path, repo, runtime
):
    root, ref, scales = _write_checkpoint(tmp_path, repo)
    model = mo.load_modelopt_fp8_transformer(Cosmos3OmniTransformer, root, runtime = runtime)
    scan = mo.scan_modelopt_fp8(root)
    cls = mo.fp8_linear_class()
    policy = model._unsloth_modelopt_step_policy
    # the checkpoint's policy: reasoner path W8A16 always, generation path W8A8 between the first / last 3 steps
    assert (policy.first_steps, policy.last_steps, policy.reasoner_a16) == (3, 3, True)
    with model.cache_context("cond", step_index = 10, num_inference_steps = 20):
        assert policy.generation_a16 is False
    seen = {"generation": 0, "reasoner": 0}
    torch.manual_seed(1)
    for stem, (_rows, cols) in scan.layers.items():
        layer = model.get_submodule(stem)
        assert isinstance(layer, cls) and layer.act == runtime, stem
        assert layer.path == mo.linear_path(stem) in seen, stem
        w = ref[stem + ".weight"]
        assert torch.equal(layer.dequantized_weight(torch.float32), w), stem
        s_x = scales[stem]
        # activations inside the calibrated range (the static scale covers their amax)
        x = (torch.rand(3, 5, cols) * 2 - 1) * s_x * 400
        y = layer(x.to(torch.bfloat16)).float()
        dense = x.to(torch.bfloat16).float() @ w.T
        if layer.path == "reasoner":
            # W8A16: bf16 activations on the widened weight
            expected = torch.nn.functional.linear(
                x.to(torch.bfloat16), w.to(torch.bfloat16)
            ).float()
            assert torch.allclose(y, expected, rtol = 1e-2, atol = 1e-3), stem
        else:
            exact = _w8a8_reference(x, w, s_x, runtime)
            assert torch.allclose(y, exact, rtol = 2e-2, atol = 1e-2 * exact.abs().max().item()), stem
            # against unquantized activations: within e4m3's activation rounding (3 mantissa bits)
            rel = (y - dense).norm() / dense.norm()
            assert rel < 0.06, (stem, float(rel))
        seen[layer.path] += 1
    assert seen["generation"] and seen["reasoner"] and sum(seen.values()) == len(scan.layers)
    calls = mo.fp8_gemm_calls(model)
    assert (
        calls["gemm"] == 0 and calls["a16"] == seen["reasoner"]
    )  # CPU: emulated, never _scaled_mm
    assert getattr(model, "_unsloth_runtime_quant", None) == "fp8"
    # a module-wide cast (offload hooks, pipeline .to) never widens the codes or rounds the scales
    layer = model.get_submodule(next(iter(scan.layers)))
    before = layer.dequantized_weight(torch.float32)
    model.to(torch.float16)
    assert layer.weight_q.dtype == torch.uint8 and torch.equal(
        layer.dequantized_weight(torch.float32), before
    )


@pytest.mark.parametrize(
    "step, a16", [(0, True), (2, True), (3, False), (16, False), (17, True), (19, True)]
)
def test_the_step_policy_runs_the_first_and_last_three_steps_w8a16(tmp_path, step, a16):
    root, ref, scales = _write_checkpoint(tmp_path, EDGE)
    model = mo.load_modelopt_fp8_transformer(
        Cosmos3OmniTransformer, root, runtime = mo.RUNTIME_STATIC
    )
    stem = next(s for s in mo.scan_modelopt_fp8(root).layers if mo.linear_path(s) == "generation")
    layer = model.get_submodule(stem)
    x = ((torch.rand(4, layer.in_features) * 2 - 1) * scales[stem] * 300).to(torch.bfloat16)
    with model.cache_context("cond", step_index = step, num_inference_steps = 20):
        y = layer(x).float()
    w = ref[stem + ".weight"]
    if a16:
        expected = torch.nn.functional.linear(x, w.to(torch.bfloat16)).float()
        assert torch.allclose(y, expected, rtol = 1e-2, atol = 1e-3)
    else:
        assert torch.allclose(
            y, _w8a8_reference(x.float(), w, scales[stem], "static"), rtol = 2e-2, atol = 1e-2
        )
    assert layer.a16_calls == int(a16)


def test_without_a_step_policy_every_step_is_w8a8(tmp_path, monkeypatch):
    root, _ref, _scales = _write_checkpoint(tmp_path, EDGE)
    path = root / "transformer" / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"].pop("runtime")
    path.write_text(json.dumps(config))
    model = mo.load_modelopt_fp8_transformer(
        Cosmos3OmniTransformer, root, runtime = mo.RUNTIME_STATIC
    )
    assert not hasattr(model, "_unsloth_modelopt_step_policy")
    assert getattr(model, mo.MODELOPT_ATTR)["step_policy"] is None
    assert all(getattr(m, "policy", 0) is None for m in model.modules() if hasattr(m, "weight_q"))


def test_a_step_policy_this_loader_does_not_run_is_refused(tmp_path):
    root, _ref, _scales = _write_checkpoint(tmp_path, EDGE)
    path = root / "transformer" / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"]["runtime"]["diffusion_step_policy"]["type"] = "sigma_threshold"
    path.write_text(json.dumps(config))
    message = mo.modelopt_fp8_error(mo.scan_modelopt_fp8(root))
    assert message and "first_last_n" in message


def test_the_stored_runtime_keeps_fp8_codes_and_widens_to_the_reference(tmp_path):
    pytest.importorskip("core.inference.diffusion_native_quant")
    root, ref, _scales = _write_checkpoint(tmp_path, EDGE)
    model = mo.load_modelopt_fp8_transformer(
        Cosmos3OmniTransformer, root, runtime = mo.RUNTIME_STORED
    )
    scan = mo.scan_modelopt_fp8(root)
    for stem, (rows, cols) in scan.layers.items():
        layer = model.get_submodule(stem)
        assert layer.weight_q.dtype == torch.uint8 and tuple(layer.weight_q.shape) == (rows, cols)
        x = torch.randn(4, cols).to(torch.bfloat16)
        expected = torch.nn.functional.linear(x, ref[stem + ".weight"].to(torch.bfloat16))
        assert torch.allclose(layer(x).float(), expected.float(), rtol = 1e-2, atol = 1e-2), stem
    assert getattr(model, "_unsloth_fp8_stored") == len(scan.layers)
    assert getattr(model, mo.MODELOPT_ATTR)["runtime"] == mo.RUNTIME_STORED


def test_runtime_choice_follows_the_card(monkeypatch):
    def gemms(static, dynamic):
        return lambda target = None, device = None, act = None: {"static": static, "dynamic": dynamic}[
            act or "static"
        ]

    monkeypatch.setattr(mo, "fp8_gemm_runs", gemms(True, True))
    assert mo.modelopt_runtime(None) == mo.RUNTIME_STATIC
    assert mo.modelopt_runtime(None, keep = False) == mo.RUNTIME_DEQUANT
    monkeypatch.setenv(mo.ACT_SCALE_ENV, "dynamic")
    assert mo.modelopt_runtime(None) == mo.RUNTIME_DYNAMIC
    # no rowwise GEMM on this card: the static one still runs
    monkeypatch.setattr(mo, "fp8_gemm_runs", gemms(True, False))
    assert mo.modelopt_runtime(None) == mo.RUNTIME_STATIC
    monkeypatch.delenv(mo.ACT_SCALE_ENV)
    # no fp8 GEMM (A100 / T4): weight-only, stored where Studio's fp8 rules keep codes, else dequantized at load
    monkeypatch.setattr(mo, "fp8_gemm_runs", gemms(False, False))
    assert mo.modelopt_runtime(None, fp8_storage = True) == mo.RUNTIME_STORED
    assert mo.modelopt_runtime(None, fp8_backend = "native") == mo.RUNTIME_STORED
    assert mo.modelopt_runtime(None) == mo.RUNTIME_DEQUANT


def test_the_gemm_switch_turns_every_fp8_gemm_off(monkeypatch):
    monkeypatch.setenv(mo.GEMM_ENV, "0")
    assert mo.fp8_gemm_runs(None) is False and mo.fp8_gemm_runs(None, act = "dynamic") is False

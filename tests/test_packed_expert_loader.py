# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Per-expert compressed-tensors NVFP4 / MXFP8 checkpoints load into packed fused expert stacks (transformers 5):
the exact checkpoint bytes, no missing / unexpected keys, a 16-bit kill switch, and a plain-transformers baseline
that shows the defect (NVFP4 experts scaled by their global scale, MXFP8 experts failing to convert)."""

import json
import os

import pytest
import torch

import unsloth  # noqa: F401

transformers = pytest.importorskip("transformers")
pytest.importorskip("compressed_tensors")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("fused expert stacks are transformers 5", allow_module_level = True)
pe = pytest.importorskip("unsloth_zoo.packed_experts")
if not hasattr(pe, "Nvfp4ExpertParam"):
    pytest.skip("needs unsloth_zoo with packed expert stacks", allow_module_level = True)

from unsloth.models import packed_expert_loader as pel

E, H, I = 4, 64, 32
PRESET = {"nvfp4": ("NVFP4A16", "nvfp4-pack-quantized"), "mxfp8": ("MXFP8A16", "mxfp8-quantized")}


def _config():
    return transformers.Qwen3MoeConfig(
        vocab_size = 96,
        hidden_size = H,
        intermediate_size = 128,
        moe_intermediate_size = I,
        num_experts = E,
        num_experts_per_tok = 2,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
        tie_word_embeddings = False,
    )


def _quantize(w, fmt):
    q = (pe.quantize_experts_nvfp4 if fmt == "nvfp4" else pe.quantize_experts_mxfp8)(
        w[None].float()
    )
    if fmt == "nvfp4":
        return {
            "weight_packed": q.data[0],
            "weight_scale": q.packed_scales[0].view(torch.float8_e4m3fn),
            "weight_global_scale": q.packed_global[0].reshape(1).clone(),
        }
    return {"weight": q.data[0].view(torch.float8_e4m3fn), "weight_scale": q.packed_scales[0]}


def _write_checkpoint(
    path,
    fmt,
    seed = 0,
):
    from compressed_tensors.quantization import QuantizationConfig, preset_name_to_scheme
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    model = transformers.Qwen3MoeForCausalLM(_config()).to(torch.bfloat16)
    state = {}
    want = {}
    for key, value in model.state_dict().items():
        if ".mlp.experts.gate_up_proj" in key or ".mlp.experts.down_proj" in key:
            continue
        state[key] = value.contiguous()
    for layer in range(2):
        experts = model.model.layers[layer].mlp.experts
        gate_up, down = experts.gate_up_proj.data, experts.down_proj.data
        for e in range(E):
            for proj, w in (
                ("gate_proj", gate_up[e, :I]),
                ("up_proj", gate_up[e, I:]),
                ("down_proj", down[e]),
            ):
                packed = _quantize(w, fmt)
                for suffix, tensor in packed.items():
                    key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.{suffix}"
                    state[key] = tensor.contiguous()
                    want[key] = tensor
    save_file(state, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"})
    config = model.config.to_dict()
    preset, ct_format = PRESET[fmt]
    scheme = preset_name_to_scheme(preset, [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"])
    quant = QuantizationConfig(
        config_groups = {"group_0": scheme},
        format = ct_format,
        quantization_status = "compressed",
        ignore = ["lm_head"],
    )
    q = json.loads(json.dumps(quant.model_dump(exclude_none = False), default = str))
    q["quant_method"] = "compressed-tensors"
    config["quantization_config"] = q
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    return want


def _load(path):
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        path,
        dtype = torch.bfloat16,
        output_loading_info = True,
    )
    return model, {k: v for k, v in info.items() if v}


@pytest.fixture(params = ["nvfp4", "mxfp8"])
def checkpoint(request, tmp_path):
    return request.param, str(tmp_path), _write_checkpoint(str(tmp_path), request.param)


def _expected_stack(want, layer, fmt):
    code_key = "weight_packed" if fmt == "nvfp4" else "weight"
    out = {}
    for stack, projs in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
        get = lambda e, p, s: want[f"model.layers.{layer}.mlp.experts.{e}.{p}.{s}"]  # noqa: E731
        codes = torch.stack(
            [torch.cat([get(e, p, code_key).view(torch.uint8) for p in projs]) for e in range(E)]
        )
        scales = torch.stack(
            [
                torch.cat([get(e, p, "weight_scale").view(torch.uint8) for p in projs])
                for e in range(E)
            ]
        )
        glob = None
        if fmt == "nvfp4":
            glob = torch.stack(
                [torch.cat([get(e, p, "weight_global_scale") for p in projs]) for e in range(E)]
            )
        out[stack] = (codes, scales, glob)
    return out


def test_packed_experts_load_the_checkpoint_bytes(checkpoint, monkeypatch):
    fmt, path, want = checkpoint
    monkeypatch.delenv(pel.PACKED_EXPERTS_ENV, raising = False)
    assert pel.install_packed_expert_loading()
    model, info = _load(path)
    assert info == {} or not any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")
    ), info
    for layer in range(2):
        experts = model.model.layers[layer].mlp.experts
        cls = pe.Nvfp4ExpertParam if fmt == "nvfp4" else pe.Mxfp8ExpertParam
        for stack, (codes, scales, glob) in _expected_stack(want, layer, fmt).items():
            param = experts._parameters[stack]
            assert type(param) is cls
            assert torch.equal(param.data, codes) and torch.equal(param.packed_scales, scales)
            if glob is not None:
                assert param.packed_global.dtype == torch.float32 and torch.equal(
                    param.packed_global, glob
                )
            assert param.packed_dtype == torch.bfloat16
        assert not any(k.startswith(pel._PLACEHOLDER) for k in experts._parameters)
        assert "gate_up_proj" not in experts.__dict__
    assert tuple(model.model.layers[0].mlp.experts.gate_up_proj._original_shape) == (E, 2 * I, H)


def test_kill_switch_installs_the_same_values_in_16_bit(checkpoint, monkeypatch):
    fmt, path, want = checkpoint
    assert pel.install_packed_expert_loading()
    monkeypatch.setenv(pel.PACKED_EXPERTS_ENV, "0")
    dense_model, _ = _load(path)
    monkeypatch.delenv(pel.PACKED_EXPERTS_ENV)
    packed_model, _ = _load(path)
    for stack in ("gate_up_proj", "down_proj"):
        dense = dense_model.model.layers[1].mlp.experts._parameters[stack]
        packed = packed_model.model.layers[1].mlp.experts._parameters[stack]
        assert type(dense) is torch.nn.Parameter and dense.dtype == torch.bfloat16
        assert torch.equal(dense, packed.dequantize(torch.bfloat16))


def test_a_partly_packed_checkpoint_is_left_to_transformers(tmp_path, monkeypatch):
    want = _write_checkpoint(str(tmp_path), "nvfp4")
    from safetensors.torch import load_file, save_file

    state = load_file(os.path.join(tmp_path, "model.safetensors"))
    for key in [k for k in state if ".layers.1.mlp.experts.3." in k]:
        del state[key]
    save_file(state, os.path.join(tmp_path, "model.safetensors"), metadata = {"format": "pt"})
    assert pel.install_packed_expert_loading()

    model = transformers.Qwen3MoeForCausalLM(_config())
    assert pel._plan(model, [os.path.join(tmp_path, "model.safetensors")]) is None
    assert want


def test_prefix_matching_and_format_detection():
    prefixes = ["model.layers.0.mlp", "model.layers.10.mlp", "mtp.layers.0.mlp"]
    assert pel._match_prefix("model.layers.0.mlp.experts", prefixes) == "model.layers.0.mlp"
    assert (
        pel._match_prefix("model.language_model.layers.10.mlp.experts", prefixes)
        == "model.layers.10.mlp"
    )
    # Ambiguous tails decline rather than guess.
    assert pel._match_prefix("model.language_model.layers.0.mlp.experts", prefixes) is None
    assert pel._match_prefix("model.layers.0.mlp.shared_expert", prefixes) is None
    u8, f8, f32 = torch.uint8, torch.float8_e4m3fn, torch.float32
    ct = {
        (0, "gate_proj", "weight_packed"): ("f", (8, 4), u8, "k"),
        (0, "gate_proj", "weight_scale"): ("f", (8, 1), f8, "k"),
        (0, "gate_proj", "weight_global_scale"): ("f", (1,), f32, "k"),
    }
    assert pel._format_of(ct) == "nvfp4"
    mo = {
        (0, "gate_proj", "weight"): ("f", (8, 4), u8, "k"),
        (0, "gate_proj", "weight_scale"): ("f", (8, 1), f8, "k"),
        (0, "gate_proj", "weight_scale_2"): ("f", (), f32, "k"),
    }
    assert pel._format_of(mo) == "nvfp4_modelopt"
    mx = {
        (0, "gate_proj", "weight"): ("f", (8, 64), f8, "k"),
        (0, "gate_proj", "weight_scale"): ("f", (8, 2), u8, "k"),
    }
    assert pel._format_of(mx) == "mxfp8"
    assert (
        pel._format_of({**ct, (1, "gate_proj", "weight"): ("f", (8, 8), torch.bfloat16, "k")})
        is None
    )


def test_transformers_alone_shows_the_defect(tmp_path):
    """Base arm: without the loader, 5.x scales NVFP4 experts by their global scale and fails on MXFP8 experts."""
    import subprocess
    import sys

    want = _write_checkpoint(str(tmp_path), "nvfp4")
    code = (
        "import torch, transformers, sys\n"
        f"m = transformers.AutoModelForCausalLM.from_pretrained({str(tmp_path)!r}, dtype = torch.bfloat16)\n"
        "print('MAX', float(m.model.layers[0].mlp.experts.down_proj.float().abs().max()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        timeout = 600,
        env = dict(os.environ),
    )
    if out.returncode != 0:
        pytest.skip(f"plain load failed outright: {out.stderr[-300:]}")
    plain_max = float(out.stdout.split("MAX")[-1])
    true_max = max(
        float(
            pe.nvfp4_experts_dequantize_torch(
                want[f"model.layers.0.mlp.experts.{e}.down_proj.weight_packed"][None],
                want[f"model.layers.0.mlp.experts.{e}.down_proj.weight_scale"][None].view(
                    torch.uint8
                ),
                want[f"model.layers.0.mlp.experts.{e}.down_proj.weight_global_scale"].reshape(1, 1),
                torch.float32,
            )
            .abs()
            .max()
        )
        for e in range(E)
    )
    if plain_max <= 2 * true_max:
        pytest.skip(
            "this transformers loads NVFP4 experts with their global scale (upstream fixed)"
        )
    assert plain_max > 100 * true_max


def test_offloaded_device_maps_keep_transformers_path(tmp_path):
    _write_checkpoint(str(tmp_path), "nvfp4")
    model = transformers.Qwen3MoeForCausalLM(_config())

    class _Q:
        quantization_config = None

    pel._before_loading(
        _Q(), model, [os.path.join(tmp_path, "model.safetensors")], {"model": "cpu", "lm_head": 0}
    )
    assert pel._PLAN_ATTR not in model.__dict__
    assert "gate_up_proj" in model.model.layers[0].mlp.experts._parameters


def test_a_declined_converter_filter_restores_stacks_and_ignore_lists(tmp_path):
    _write_checkpoint(str(tmp_path), "nvfp4")
    model = transformers.Qwen3MoeForCausalLM(_config())
    before = getattr(model, "_keys_to_ignore_on_load_unexpected", None)
    before = set(before) if isinstance(before, set) else before

    class _Q:
        quantization_config = None

    q = _Q()
    pel._before_loading(q, model, [os.path.join(tmp_path, "model.safetensors")])
    assert pel._PLAN_ATTR in model.__dict__
    assert pel._filter_conversions(q, []) == []  # no merge converter to drop: decline
    after = getattr(model, "_keys_to_ignore_on_load_unexpected", None)
    assert (set(after) if isinstance(after, set) else after) == before
    experts = model.model.layers[0].mlp.experts
    assert "gate_up_proj" in experts._parameters and not any(
        k.startswith(pel._PLACEHOLDER) for k in experts._parameters
    )


def test_malformed_scales_are_refused(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    _write_checkpoint(str(tmp_path), "nvfp4")
    f = os.path.join(tmp_path, "model.safetensors")
    state = load_file(f)
    key = "model.layers.0.mlp.experts.1.up_proj.weight_scale"
    state[key] = state[key][:, :1].contiguous()  # (rows, 1): would broadcast
    save_file(state, f, metadata = {"format": "pt"})
    monkeypatch.delenv(pel.PACKED_EXPERTS_ENV, raising = False)
    assert pel.install_packed_expert_loading()
    with pytest.raises(Exception):
        _load(str(tmp_path))


@pytest.mark.parametrize("pread", ["1", "0"])
def test_packed_experts_read_with_pread_where_wanted(checkpoint, monkeypatch, pread):
    """UNSLOTH_SAFETENSORS_PREAD=1 (what a unified-memory GPU gets by default) opens every expert shard with pread,
    0 keeps mmap; the installed bytes are the checkpoint's either way."""
    import sys

    import safetensors
    from unsloth.models import _windows_safetensors as ws

    if not ws._supports_backend():
        pytest.skip("safetensors < 0.8 has no backend keyword")
    fmt, path, want = checkpoint
    monkeypatch.delenv(pel.PACKED_EXPERTS_ENV, raising = False)
    monkeypatch.setenv(ws.PREAD_ENV, pread)
    backends = []
    real = safetensors.safe_open

    def spy(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "get":  # the expert reader, not the header scan
            backends.append(kwargs.get("backend"))
        return real(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", spy)
    assert pel.install_packed_expert_loading()
    model, _ = _load(path)
    assert backends and set(backends) == ({"pread"} if pread == "1" else {None})
    for layer in range(2):
        experts = model.model.layers[layer].mlp.experts
        for stack, (codes, scales, glob) in _expected_stack(want, layer, fmt).items():
            param = experts._parameters[stack]
            assert torch.equal(param.data, codes) and torch.equal(param.packed_scales, scales)
            if glob is not None:
                assert torch.equal(param.packed_global, glob)

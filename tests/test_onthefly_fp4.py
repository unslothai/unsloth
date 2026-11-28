# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""load_in_4bit = "nvfp4" / "mxfp4" (W4A16 quantized at load) and load_in_8bit = "fp8" (= load_in_fp8)."""

import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(rel, name):
    # By path: `import unsloth` needs a GPU, these modules only need torch.
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oq = _load("unsloth/models/onthefly_quant.py", "_g_onthefly_quant")
fq = _load("unsloth/kernels/fp4_quantize.py", "_g_fp4_quantize")

_E2M1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))


def _decode(packed, scale, global_scale):
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    values = _E2M1[(codes & 7).long()] * torch.where((codes & 8).bool(), -1.0, 1.0)
    return values * (scale.float() / global_scale.float()).repeat_interleave(16, 1)


# ---- flags ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "l4, l8, fp8, want",
    [
        (True, False, False, (True, False, False, None)),
        (False, True, False, (False, True, False, None)),
        ("nvfp4", False, False, (False, False, False, "nvfp4")),
        ("MXFP4", False, False, (False, False, False, "mxfp4")),
        ("nf4", False, False, (True, False, False, None)),
        (False, "fp8", False, (False, False, True, None)),
        (False, "fp8_block", False, (False, False, "block", None)),
        (False, "FP8-Block", "block", (False, False, "block", None)),
    ],
)
def test_flags_resolve(l4, l8, fp8, want):
    assert oq.normalize_lowbit_flags(l4, l8, fp8) == want


@pytest.mark.parametrize(
    "l4, l8", [("int4", False), ("4bit", False), (False, "int8"), (False, "yes")]
)
def test_unknown_strings_raise_instead_of_enabling_bitsandbytes(l4, l8):
    with pytest.raises(ValueError):
        oq.normalize_lowbit_flags(l4, l8)


def test_fp4_refuses_full_finetuning_and_conflicting_fp8():
    with pytest.raises(ValueError, match = "full finetune"):
        oq.normalize_lowbit_flags("nvfp4", False, full_finetuning = True)
    with pytest.raises(ValueError, match = "conflicts"):
        oq.normalize_lowbit_flags(False, "fp8", "block")


def test_decorator_rewrites_only_string_flags():
    seen = []

    def from_pretrained(
        model_name = "m",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        full_finetuning = False,
        fast_inference = False,
        load_in_fp8 = False,
        *args,
        **kwargs,
    ):
        seen.append((load_in_4bit, load_in_8bit, load_in_fp8, oq._ACTIVE.get(), kwargs))
        return "ok"

    wrapped = oq.onthefly_lowbit_load(from_pretrained)
    assert wrapped("m", load_in_4bit = True, extra = 1) == "ok"
    assert seen[-1] == (True, False, False, None, {"extra": 1})
    wrapped("m", load_in_8bit = "fp8")  # load_in_4bit left at its default True
    assert seen[-1][:4] == (False, False, True, None)
    wrapped("m", 2048, None, True, "fp8_block")  # an explicit positional load_in_4bit is kept
    assert seen[-1][:3] == (True, False, "block")
    if oq._install_hooks():
        wrapped("m", load_in_4bit = "mxfp4")
        assert seen[-1][:3] == (False, False, False) and seen[-1][3]["fmt"] == "mxfp4"
    with pytest.raises(NotImplementedError):
        wrapped("m", load_in_4bit = "nvfp4", fast_inference = True)


@pytest.mark.parametrize(
    "extra",
    [
        dict(load_in_8bit = True),
        dict(load_in_8bit = "fp8"),
        dict(load_in_16bit = True),
        dict(load_in_fp8 = True),
        dict(load_in_fp8 = "block"),
        dict(float8_training = "fp8_lora"),
    ],
)
def test_fp4_string_refuses_a_second_precision(extra):
    # The FP4 request reaches from_pretrained as load_in_4bit = False, so its 4bit / 8bit / 16bit conflict
    # check cannot see it: the decorator refuses the combination itself.
    calls = []

    def from_pretrained(
        model_name = "m",
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        load_in_fp8 = False,
        float8_training = None,
        **kwargs,
    ):
        calls.append(1)

    wrapped = oq.onthefly_lowbit_load(from_pretrained)
    with pytest.raises(ValueError, match = "cannot be combined"):
        wrapped("m", load_in_4bit = "nvfp4", **extra)
    assert not calls

    # FastLanguageModel / FastModel take float8_training through **kwargs.
    def loose(
        model_name = "m",
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        load_in_fp8 = False,
        **kwargs,
    ):
        calls.append(1)

    loose = oq.onthefly_lowbit_load(loose)
    with pytest.raises(ValueError, match = "cannot be combined"):
        loose("m", load_in_4bit = "mxfp4", **extra)
    assert not calls
    # The same flags with a bool load_in_4bit pass through for from_pretrained's own checks.
    wrapped("m", load_in_4bit = False, **{k: v for k, v in extra.items() if not isinstance(v, str)})


# ---- quantizers ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["max6", "4over6"])
def test_nvfp4_matches_unsloth_zoo_expert_quantizer(method):
    pe = pytest.importorskip("unsloth_zoo.packed_experts")
    if not hasattr(pe, "quantize_experts_nvfp4"):
        pytest.skip("unsloth_zoo without the NVFP4 expert quantizer")
    torch.manual_seed(0)
    w = (torch.randn(96, 256) * 0.02).to(torch.bfloat16)
    w[3, :16] = 0
    packed, scale, g = fq.nvfp4_quantize_weight(w, method)
    ref = pe.quantize_experts_nvfp4(w[None], method = method)
    scales, glob = ref._companions()
    assert torch.equal(packed, ref.data[0])
    assert torch.equal(scale.view(torch.uint8), scales[0].view(torch.uint8))
    assert torch.equal(g, glob[0].reshape(1))


def test_nvfp4_round_trip_error_is_nvfp4_sized():
    torch.manual_seed(0)
    w = torch.randn(128, 512) * 0.02
    packed, scale, g = fq.nvfp4_quantize_weight(w, "max6")
    assert packed.dtype == torch.uint8 and packed.shape == (128, 256)
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (128, 32)
    rel = ((_decode(packed, scale, g) - w).norm() / w.norm()).item()
    assert 0.02 < rel < 0.15, rel


def _mx_reference(w):
    # OCP MXFP4 round to nearest: shared exponent floor(log2(amax)) - 2 per 32, E2M1 saturating at 6.
    N, K = w.shape
    blocks = w.float().view(N, K // 32, 32)
    amax = blocks.abs().amax(-1)
    e = torch.floor(torch.log2(amax.clamp(min = 2.0**-126))) - 2
    e = torch.where(amax > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
    y = (blocks / torch.exp2(e)[..., None]).clamp(-6, 6)
    mid = (_E2M1[1:] + _E2M1[:-1]) / 2
    idx = torch.bucketize(y.abs(), mid)
    tie = (idx < 7) & (y.abs() == mid[idx.clamp(max = 6)]) & (idx % 2 == 1)
    q = _E2M1[torch.where(tie, idx + 1, idx)] * torch.sign(y)
    return (q * torch.exp2(e)[..., None]).view(N, K)


@pytest.mark.parametrize("spread", ["narrow", "wide"])
def test_mxfp4_decodes_exactly_in_the_nvfp4_layout(spread):
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    w[5, :32] = 0
    if spread == "wide":
        w[7, 32:64] *= 2.0**-30  # 30 octaves below the rest: past e4m3's power-of-two range
    packed, scale, g = fq.mxfp4_quantize_weight(w)
    assert scale.dtype == (torch.float8_e4m3fn if spread == "narrow" else torch.bfloat16)
    assert scale.shape == (64, 16)
    assert torch.equal(_decode(packed, scale, g), _mx_reference(w))


def test_skip_keys_match_whole_dotted_components():
    skip = ["lm_head", "mlp.gate", "visual", "attention.q_proj"]
    assert oq._name_skipped("model.layers.0.mlp.gate", skip)
    assert not oq._name_skipped("model.layers.0.mlp.gate_proj", skip)
    assert oq._name_skipped("model.visual.blocks.0.attn.qkv", skip)
    assert oq._name_skipped("encoder.layers.1.attention.q_proj", skip)
    assert not oq._name_skipped("model.layers.0.self_attention.q_proj", skip)


def test_quantizers_refuse_unaligned_widths():
    with pytest.raises(ValueError):
        fq.nvfp4_quantize_weight(torch.randn(4, 24))
    with pytest.raises(ValueError):
        fq.mxfp4_quantize_weight(torch.randn(4, 48))


# ---- the load hook (CPU, plain transformers) ----------------------------------------------------------------


class _Routed(torch.nn.Linear):
    pass


@pytest.fixture
def cpu_hook(monkeypatch):
    if not oq._install_hooks():
        pytest.skip("transformers without _load_pretrained_model")
    monkeypatch.setattr(oq, "_QUANT_DEVICES", ("cuda", "xpu", "cpu"))
    monkeypatch.setattr(
        oq, "_routed_linear_parts", lambda: (_Routed, lambda: None, lambda self, x: x)
    )
    monkeypatch.setattr(oq, "_skip_names", lambda: ["lm_head", "mlp.gate", "router"])
    monkeypatch.setattr(oq, "_architecture_skips", lambda config: [])
    monkeypatch.setattr(
        oq,
        "_dense_weight",
        lambda m: _decode(m.weight_packed, m.weight_scale, m.weight_global_scale).to(
            torch.bfloat16
        ),
    )


@pytest.fixture(scope = "module")
def tiny_dense(tmp_path_factory):
    import _onthefly_fp4_case as case
    return case.build(str(tmp_path_factory.mktemp("dense")))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_linears_are_quantized_while_the_checkpoint_loads(cpu_hook, tiny_dense, fmt):
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
    oq.onthefly_fp4_stats(reset = True)
    seen, quantize = [], oq._quantize_linear_

    def recording(module, weight):
        # transformers 5 must load on the main thread, or its workers stage the 16-bit model on the GPU first.
        seen.append(os.environ.get("HF_DEACTIVATE_ASYNC_LOAD"))
        return quantize(module, weight)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(oq, "_quantize_linear_", recording)
        before = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
        with oq.onthefly_fp4_loading(fmt):
            model = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
        assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == before
    assert seen and set(seen) == {"1"}
    stats = oq.onthefly_fp4_stats()
    # 2 layers x (q, k, v, o, gate, up, down); all assigned through the loader, none filled in afterwards.
    assert stats["linear"] == 14 and stats["late"] == 0 and stats["skipped"] == 0
    assert type(model.lm_head) is torch.nn.Linear and model.lm_head.weight.dtype == torch.bfloat16
    refs = dict(ref.named_modules())
    for name, module in model.named_modules():
        if type(module) is _Routed:
            assert "weight" not in module._parameters  # no 16-bit copy kept
            assert module.weight_packed.dtype == torch.uint8
            W = _decode(module.weight_packed, module.weight_scale, module.weight_global_scale)
            R = refs[name].weight.float()
            assert ((W - R).norm() / R.norm()).item() < 0.15
            if fmt == "mxfp4":
                assert torch.equal(W, _mx_reference(R))


def test_full_save_writes_the_dequantized_weights_and_reloads(cpu_hook, tiny_dense, tmp_path):
    from transformers import AutoModelForCausalLM

    with oq.onthefly_fp4_loading("nvfp4"):
        model = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
    keys = set(
        model.state_dict()
    )  # outside save_pretrained the packed layout is what state_dict holds
    assert any(k.endswith("weight_packed") for k in keys)
    model.save_pretrained(str(tmp_path))
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype = torch.bfloat16)
    mods = dict(model.named_modules())
    for name, module in reloaded.named_modules():
        if type(mods.get(name)) is _Routed:
            m = mods[name]
            want = _decode(m.weight_packed, m.weight_scale, m.weight_global_scale).to(
                torch.bfloat16
            )
            assert torch.equal(module.weight, want), name
    assert not any(k.endswith("weight_packed") for k in reloaded.state_dict())
    # A caller-supplied state dict (unsloth/save.py's generic save passes one) is converted too.
    other = tmp_path / "given"
    model.save_pretrained(str(other), state_dict = model.state_dict())
    again = AutoModelForCausalLM.from_pretrained(str(other), dtype = torch.bfloat16)
    assert all(
        torch.equal(a, b)
        for a, b in zip(again.state_dict().values(), reloaded.state_dict().values())
    )


@pytest.fixture(scope = "module")
def tiny_moe(tmp_path_factory):
    import _onthefly_fp4_case as case
    return case.build(str(tmp_path_factory.mktemp("moe")), moe = True)


def test_moe_expert_stacks_quantize_at_load_and_save_dense(cpu_hook, tiny_moe, tmp_path):
    pe = pytest.importorskip("unsloth_zoo.packed_experts")
    if not hasattr(pe, "quantize_moe_expert_stacks"):
        pytest.skip("unsloth_zoo without the expert quantizer")
    from transformers import AutoModelForCausalLM

    oq.onthefly_fp4_stats(reset = True)
    with oq.onthefly_fp4_loading("nvfp4"):
        model = AutoModelForCausalLM.from_pretrained(tiny_moe, dtype = torch.bfloat16)
    stacks = {
        f"{n}.{s}": p
        for n, m in model.named_modules()
        for s, p in m._parameters.items()
        if oq._is_packed_stack(p)
    }
    if not stacks:  # transformers 4.57: per-expert Linears, quantized as Linears
        assert oq.onthefly_fp4_stats()["linear"] == 32
        return
    assert oq.onthefly_fp4_stats()["experts"] == len(stacks) == 4
    model.save_pretrained(str(tmp_path))
    reloaded = dict(
        AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype = torch.bfloat16).named_parameters()
    )
    for key, p in stacks.items():
        assert torch.equal(reloaded[key], p.dequantize().to(torch.bfloat16)), key


def test_dequantize_turns_a_layer_into_its_dense_linear(cpu_hook, tiny_dense):
    # PEFT merges and SVD-style inits call dequantize_(); compressed-tensors would skip these scheme-less layers.
    from transformers import AutoModelForCausalLM

    with oq.onthefly_fp4_loading("nvfp4"):
        model = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
    module = model.model.layers[0].mlp.down_proj
    want = _decode(module.weight_packed, module.weight_scale, module.weight_global_scale).to(
        torch.bfloat16
    )
    x = torch.randn(3, module.in_features, dtype = torch.bfloat16)
    module.dequantize_()
    assert type(module) is torch.nn.Linear and "weight_packed" not in module._parameters
    assert torch.equal(module.weight, want) and not module.weight.requires_grad
    torch.testing.assert_close(module(x), x @ want.t())


def test_kill_switch_keeps_16_bit(cpu_hook, tiny_dense, monkeypatch):
    from transformers import AutoModelForCausalLM

    monkeypatch.setenv(oq.ONTHEFLY_ENV, "0")
    oq.onthefly_fp4_stats(reset = True)
    with oq.onthefly_fp4_loading("nvfp4"):
        model = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
    assert oq.onthefly_fp4_stats()["linear"] == 0
    assert not any(type(m) is _Routed for m in model.modules())


def test_loads_outside_the_block_are_untouched(cpu_hook, tiny_dense):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
    assert not any(type(m) in (_Routed, oq._PendingFP4Linear) for m in model.modules())


def test_save_writes_the_supplied_state_dict_not_the_live_weights(cpu_hook, tiny_dense, tmp_path):
    from transformers import AutoModelForCausalLM

    with oq.onthefly_fp4_loading("nvfp4"):
        model = AutoModelForCausalLM.from_pretrained(tiny_dense, dtype = torch.bfloat16)
    key = "model.layers.0.self_attn.q_proj"
    live = model.get_submodule(key)
    assert type(live) is _Routed
    live_dense = _decode(live.weight_packed, live.weight_scale, live.weight_global_scale)
    assert live_dense.abs().sum() > 0
    # A snapshot whose packed codes all decode to zero (code 0), the rest of the model unchanged.
    supplied = {k: v.clone() for k, v in model.state_dict().items()}
    supplied[key + ".weight_packed"].zero_()
    model.save_pretrained(str(tmp_path), state_dict = supplied)
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype = torch.bfloat16)
    saved = reloaded.get_submodule(key).weight
    assert torch.count_nonzero(saved).item() == 0, "save_pretrained wrote the live weight"
    # Every other packed layer is the decode of its supplied (unchanged) tensors.
    other = "model.layers.1.mlp.down_proj"
    m = model.get_submodule(other)
    want = _decode(m.weight_packed, m.weight_scale, m.weight_global_scale).to(torch.bfloat16)
    assert torch.equal(reloaded.get_submodule(other).weight, want)
    # The live model is untouched by the save.
    assert torch.equal(
        _decode(live.weight_packed, live.weight_scale, live.weight_global_scale), live_dense
    )


def _drop_key(src, dst, key):
    import shutil

    from safetensors.torch import load_file, save_file

    shutil.copytree(src, dst)
    path = os.path.join(dst, "model.safetensors")
    tensors = load_file(path)
    del tensors[key]
    save_file(tensors, path, metadata = {"format": "pt"})
    return dst


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_missing_weights_are_initialized_before_they_are_quantized(
    cpu_hook, tiny_dense, tmp_path, monkeypatch, fmt
):
    import transformers.modeling_utils as mu
    from transformers import AutoModelForCausalLM

    if mu.PreTrainedModel.__dict__.get("_finalize_model_loading") is None:
        pytest.skip("transformers 4.x materializes missing weights on the CPU after the load")
    key = "model.layers.0.self_attn.q_proj"
    path = _drop_key(tiny_dense, str(tmp_path / "missing"), key + ".weight")
    # Uninitialized storage as a sentinel: whatever the loader allocates for the missing weight reads 17.
    empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda t, *a, **k: empty_like(t, *a, **k).fill_(17.0)
        if t.is_floating_point()
        else empty_like(t, *a, **k),
    )
    oq.onthefly_fp4_stats(reset = True)
    with oq.onthefly_fp4_loading(fmt):
        model = AutoModelForCausalLM.from_pretrained(path, dtype = torch.bfloat16)
    module = model.get_submodule(key)
    assert type(module) is _Routed and module.weight_packed.dtype == torch.uint8
    W = _decode(module.weight_packed, module.weight_scale, module.weight_global_scale)
    std = model.config.initializer_range
    assert W.abs().max().item() < 1.0, "the missing weight was quantized before its initialization"
    assert 0.5 * std < W.float().std().item() < 2.0 * std
    stats = oq.onthefly_fp4_stats()
    assert stats["linear"] == 14 and stats["skipped"] == 0


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("method", ["max6", "4over6"])
def test_tiny_magnitude_weights_keep_finite_scales(fmt, method):
    # amax 2^-118: 6 * 448 / amax (NVFP4) and 2^(8 - e_max) (MXFP4) overflow float32 to inf, which decoded
    # all-zero blocks to NaN (0 * inf) and every other block to 0.
    w = torch.zeros(4, 64, dtype = torch.bfloat16)
    w[0, 0] = 2.0**-118
    w[1, 37] = -(2.0**-119)
    packed, scale, global_scale = fq.fp4_quantize_weight(w, fmt, method)
    assert torch.isfinite(global_scale).all()
    W = _decode(packed, scale, global_scale)
    assert torch.isfinite(W).all()
    assert torch.count_nonzero(W[2:]).item() == 0
    R = w.float()
    if fmt == "mxfp4":
        # Both values are E2M1 multiples of an E8M0 scale in range: OCP MXFP4 holds them exactly.
        assert torch.equal(W, R)
    else:
        # (W - R).norm() underflows fp32 at this magnitude: compare against the largest value instead.
        assert ((W - R).abs().max() / R.abs().max()).item() < 0.15


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_ordinary_weights_quantize_as_before(fmt):
    # In-range tensors never reach the overflow guards: ordinary FP4 error, finite scales.
    torch.manual_seed(0)
    w = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)
    w[3, :32] *= 2.0**-20  # a wide-exponent block (MXFP4 keeps e4m3 scales up to a 17-octave span)
    packed, scale, global_scale = fq.fp4_quantize_weight(w, fmt, "4over6")
    W = _decode(packed, scale, global_scale)
    assert torch.isfinite(global_scale).all() and torch.isfinite(W).all()
    assert ((W - w.float()).norm() / w.float().norm()).item() < 0.15


# ---- end to end on a GPU (fresh process per case) -----------------------------------------------------------


def _case(
    path,
    api,
    mode,
    tmp_path,
    merge = False,
):
    env = dict(
        os.environ,
        PYTHONPATH = ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
        UNSLOTH_COMPILE_LOCATION = str(tmp_path / f"compiled_{api}_{mode}"),
        UNSLOTH_IS_PRESENT = "1",
    )
    cmd = [sys.executable, os.path.join(ROOT, "tests", "_onthefly_fp4_case.py"), "--case", path, api,
           mode, str(tmp_path)] + (["--merge"] if merge else [])  # fmt: skip
    run = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 1200)
    lines = [l for l in run.stdout.splitlines() if l.startswith("ONTHEFLY_CASE ")]
    assert run.returncode == 0 and lines, run.stderr[-3000:]
    return json.loads(lines[-1][len("ONTHEFLY_CASE ") :])


@pytest.mark.skipif(not has_real_cuda(), reason = "W4A16 kernels need a GPU")
@pytest.mark.parametrize("api", ["FastLanguageModel", "FastModel"])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_lora_trains_on_an_onthefly_fp4_base(tiny_dense, tmp_path, api, fmt):
    r = _case(tiny_dense, api, fmt, tmp_path, merge = fmt == "nvfp4")
    assert r["unsloth"].startswith(ROOT)
    assert r["stats"]["linear"] == 14 and r["stats"]["late"] == 0
    assert r["census"].get("_UnslothNVFP4Linear") == 14
    assert r["lm_head"] == "Linear:torch.bfloat16" and r["embed"] == "torch.bfloat16"
    assert r["dequant_calls"] + r["fused_fwd"] > 0  # the packed weights fed the GEMMs
    assert r["packed_unchanged"] and r["packed_layers"] == 14
    assert r["lora_grads_nonzero"] == r["lora_params"] and r["lora_changed"] == r["lora_params"]
    assert all(torch.isfinite(torch.tensor(r["losses"]))) and r["losses"][-1] < r["losses"][0]
    assert "adapter_model.safetensors" in r["saved"]
    if fmt == "nvfp4":
        # merged_16bit folds the adapter into the 16-bit checkpoint weights (as for NF4), not the FP4 ones.
        assert r["merged_vs_16bit_base_rel"] < 2e-2 < r["merged_rel"], r
        assert r["base_save_reload_rel"] < 1e-2, r["base_save_reload_rel"]
        assert r["peft_merge_packed_left"] == 0 and r["peft_merge_rel"] < 2e-2, r


@pytest.mark.skipif(not has_real_cuda(), reason = "W4A16 kernels need a GPU")
def test_bool_load_in_4bit_is_still_bitsandbytes(tiny_dense, tmp_path):
    r = _case(tiny_dense, "FastLanguageModel", "nf4", tmp_path)
    assert r["stats"]["linear"] == 0 and r["dequant_calls"] == 0
    assert r["census"].get("Linear4bit", 0) == 14

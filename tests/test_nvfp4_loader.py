# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
# NVFP4 compressed-tensors checkpoints (unsloth/Qwen3.8-27B-NVFP4 layout) stay packed and train W4A16 with LoRA.
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressed_tensors")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "NVFP4 routing needs CUDA weights"
)

sys.path.insert(0, os.path.dirname(__file__))
import _nvfp4_fixtures as fx  # noqa: E402


class _Checkpoints(dict):
    # Built on first use, so a transformers without qwen3_5 (4.57.x) still runs the qwen3 cases.
    def __init__(self, tmp_path_factory):
        super().__init__()
        self.tmp_path_factory = tmp_path_factory

    def __missing__(self, arch):
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        if arch not in CONFIG_MAPPING_NAMES:
            pytest.skip(f"transformers has no {arch}")
        d = self.tmp_path_factory.mktemp(arch)
        self[arch] = (str(d), fx.build(str(d), arch))
        return self[arch]


@pytest.fixture(scope = "module")
def ckpt(tmp_path_factory):
    return _Checkpoints(tmp_path_factory)


def _load_raw(path, arch):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    cls = AutoModelForCausalLM if arch == "qwen3" else AutoModelForImageTextToText
    return cls.from_pretrained(path, device_map = "cuda", dtype = torch.bfloat16)


def _module(model, name):
    return dict(model.named_modules())[name]


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_5"])
def test_fixture_decompresses_to_the_packed_weights(ckpt, arch):
    path, kinds = ckpt[arch]
    from safetensors import safe_open

    f = safe_open(os.path.join(path, "model.safetensors"), "pt")
    name = next(n for n, k in kinds.items() if k == "nvfp4")
    packed, scale, gs = (
        f.get_tensor(f"{name}.{s}")
        for s in ("weight_packed", "weight_scale", "weight_global_scale")
    )
    from unsloth.kernels.nvfp4 import nvfp4_dequantize

    ref = fx.dequant_nvfp4_reference(packed, scale, gs)
    assert torch.equal(nvfp4_dequantize(packed.cuda(), scale.cuda(), gs.cuda()).cpu(), ref)
    model = _load_raw(path, arch)
    module = _module(model, name)
    assert (
        str(module.quantization_status.value) == "compressed"
        and module.weight_packed.dtype == torch.uint8
    )
    # The broad MLP target overlaps the last layer's FP8 override; the override wins, as in the real checkpoint.
    last = next(n for n, k in kinds.items() if k == "fp8" and ".mlp." in n)
    assert _module(model, last).quantization_scheme.weights.num_bits == 8


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_5"])
def test_nvfp4_layers_stay_packed_and_run_w4a16(ckpt, arch, monkeypatch):
    from safetensors import safe_open
    from unsloth.kernels.nvfp4 import nvfp4_dequantize
    from unsloth.models import loader_utils

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "0")
    path, kinds = ckpt[arch]
    model = _load_raw(path, arch)
    loader_utils._prepare_compressed_tensors_model(model)
    nvfp4 = [n for n, k in kinds.items() if k == "nvfp4"]
    assert model._unsloth_compressed_tensors_nvfp4 == len(nvfp4)
    assert getattr(model, "ct_decompress_hook", None) is None and not model._forward_pre_hooks
    keys = set(safe_open(os.path.join(path, "model.safetensors"), "pt").keys())
    # Same keys as the checkpoint, except that decompressed layers are dense and drop their scales
    # (a tied lm_head may add its alias).
    dense = [n for n, k in kinds.items() if k == "fp8"] + ["lm_head"]
    keys -= {n + ".weight_scale" for n in dense}
    assert set(model.state_dict()) - {"lm_head.weight"} == keys - {"lm_head.weight"}
    assert not any(
        n.endswith(".weight") and n[: -len(".weight")] in nvfp4 for n, _ in model.named_parameters()
    )
    for name, kind in kinds.items():
        module = _module(model, name)
        if kind == "fp8":
            assert module.weight.dtype == torch.bfloat16  # decompressed per module
    with torch.no_grad():
        model(input_ids = torch.randint(0, 1000, (2, 8), device = "cuda"))
    for name in nvfp4:
        module = _module(model, name)
        assert module.weight_packed.dtype == torch.uint8 and module.weight.dtype == torch.uint8
        assert module.weight.quant_state.shape == (module.out_features, module.in_features)
        x = torch.randn(3, module.in_features, device = "cuda", dtype = torch.bfloat16)
        W = nvfp4_dequantize(
            module.weight_packed, module.weight_scale, module.weight_global_scale, torch.bfloat16
        )
        with torch.no_grad():
            # A16: exactly the dense matmul, no activation fake-quantization.
            assert torch.equal(module(x), torch.nn.functional.linear(x, W))


def test_dtype_casts_keep_the_nvfp4_storage_dtypes(ckpt, monkeypatch):
    # model.to(fp16) must not cast the scales: an fp16 global scale overflows past 65504 and zeroes the layer.
    from unsloth.models import loader_utils

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "0")
    path, kinds = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    module = _module(model, next(n for n, k in kinds.items() if k == "nvfp4"))
    dtypes = {n: p.dtype for n, p in module.named_parameters()}
    x = torch.randn(3, module.in_features, device = "cuda", dtype = torch.float16)
    with torch.no_grad():
        module.weight_global_scale.fill_(1e5)
        want = module(x)
        model.to(torch.float16)
        assert {n: p.dtype for n, p in module.named_parameters()} == dtypes
        assert torch.equal(module(x), want) and want.abs().max() > 0
        # Fused LoRA paths dequantize through the quant state, which must follow the cast.
        assert module.weight.quant_state.dtype == torch.float16


def _fp8_route_available():
    from unsloth.models import loader_utils
    return loader_utils._zoo_peft_forward_keeps_fp8_inputs()


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_peft_merge_dequantizes_the_routed_base(ckpt, kind, monkeypatch):
    # merge_and_unload / merge_adapter / merged_4bit add the dense delta into base_layer.weight.
    from peft import LoraConfig, get_peft_model
    from unsloth.models import loader_utils

    if kind == "fp8" and not _fp8_route_available():
        pytest.skip("installed unsloth_zoo predates the FP8 kernel route")
    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "1" if kind == "fp8" else "0")
    path, kinds = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    name = next(
        n
        for n, k in kinds.items()
        if k == kind and ("layers.0." if kind == "nvfp4" else "layers.1.") in n
    )
    model = get_peft_model(model, LoraConfig(r = 4, target_modules = [name.split(".")[-1]]))
    lora = _module(model, "base_model.model." + name)
    torch.nn.init.normal_(lora.lora_B["default"].weight, std = 0.02)
    want = loader_utils._routed_dense_weight(lora.get_base_layer())
    want += lora.get_delta_weight("default")
    merged = model.merge_and_unload()
    layer = _module(merged, name)
    assert type(layer) is torch.nn.Linear and layer.weight.dtype == torch.bfloat16
    assert torch.equal(layer.weight, want)


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_dora_reads_the_dense_routed_weight(ckpt, kind, monkeypatch):
    from peft import LoraConfig, get_peft_model
    from unsloth.models import loader_utils

    if kind == "fp8" and not _fp8_route_available():
        pytest.skip("installed unsloth_zoo predates the FP8 kernel route")
    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "1" if kind == "fp8" else "0")
    path, kinds = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    name = next(
        n
        for n, k in kinds.items()
        if k == kind and ("layers.0." if kind == "nvfp4" else "layers.1.") in n
    )
    model = get_peft_model(
        model, LoraConfig(r = 4, target_modules = [name.split(".")[-1]], use_dora = True)
    )
    lora = _module(model, "base_model.model." + name)
    W = loader_utils._routed_dense_weight(lora.get_base_layer()).float()
    magnitude = lora.lora_magnitude_vector["default"].weight
    delta = lora.get_delta_weight("default").float()
    assert torch.allclose(magnitude.float(), (W + delta).norm(dim = 1), rtol = 1e-2)
    out = model(input_ids = torch.randint(0, 1000, (1, 8), device = "cuda")).logits
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("merge", [False, True])
def test_saved_checkpoint_reloads_with_plain_transformers(ckpt, merge, tmp_path, monkeypatch):
    # save_pretrained (after merge_and_unload or not): per-module decompressed layers such as lm_head and the
    # merged ones are dense, so they must join the ignore list, or the reload reads them as FP8 / packed.
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from unsloth.models import loader_utils

    if not _fp8_route_available():
        pytest.skip("installed unsloth_zoo predates the FP8 kernel route")
    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "1")
    path, kinds = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    if merge:
        model = get_peft_model(model, LoraConfig(r = 4, target_modules = ["gate_proj", "q_proj"]))
        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std = 0.02)
        model = model.merge_and_unload()
    ids = torch.randint(0, 1000, (1, 16), device = "cuda")
    with torch.no_grad():
        want = model(input_ids = ids).logits.float()
    model.save_pretrained(str(tmp_path / "out"))
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(tmp_path / "out"), device_map = "cuda", dtype = torch.bfloat16
    )
    with torch.no_grad():
        got = reloaded(input_ids = ids).logits.float()
    # Plain compressed-tensors fake-quantizes activations (W4A4 / W8A8), so only close, not equal;
    # a layer read in the wrong format gives logits near zero.
    assert (got - want).abs().max() < 0.1 * want.abs().max()


@pytest.mark.parametrize("init", ["pissa_niter_2", "olora", "loftq"])
def test_svd_style_lora_inits_see_the_dense_base(ckpt, init, monkeypatch):
    # PiSSA / OLoRA rewrite base_layer.weight from its SVD / QR, so the packed base goes dense first.
    from peft import LoraConfig, get_peft_model
    from unsloth.models import loader_utils

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "0")
    path, kinds = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    name = next(n for n, k in kinds.items() if k == "nvfp4")
    extra = {}
    if init == "loftq":
        pytest.importorskip("scipy")
        from peft import LoftQConfig
        extra["loftq_config"] = LoftQConfig(loftq_bits = 4, loftq_iter = 1)
    model = get_peft_model(
        model,
        LoraConfig(r = 4, target_modules = [name.split(".")[-1]], init_lora_weights = init, **extra),
    )
    base = _module(model, "base_model.model." + name).get_base_layer()
    assert type(base) is torch.nn.Linear and base.weight.dtype == torch.bfloat16
    out = model(input_ids = torch.randint(0, 1000, (1, 8), device = "cuda")).logits
    assert torch.isfinite(out).all()


def test_decompressed_layers_join_a_missing_ignore_list(ckpt, monkeypatch):
    from unsloth.models import loader_utils

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "0")
    path, _ = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    model.config.quantization_config.quantization_config.ignore = None
    loader_utils._prepare_compressed_tensors_model(model)
    assert "lm_head" in model.config.quantization_config.quantization_config.ignore


def test_nvfp4_lm_head_is_decompressed(tmp_path, monkeypatch):
    # Decode and the fused CE loss read lm_head.weight directly, so it cannot stay packed.
    from unsloth.models import loader_utils

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "0")
    kinds = fx.build(str(tmp_path), "qwen3", lm_head_kind = "nvfp4")
    assert kinds["lm_head"] == "nvfp4"
    model = _load_raw(str(tmp_path), "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    head = model.get_output_embeddings()
    assert head.weight.dtype == torch.bfloat16
    assert tuple(head.weight.shape) == (model.config.vocab_size, model.config.hidden_size)
    assert model._unsloth_compressed_tensors_nvfp4 == len(kinds) - 1 - sum(
        k == "fp8" for k in kinds.values()
    )


def test_opt_out_keeps_the_full_decompress(ckpt, monkeypatch):
    from unsloth.models import loader_utils

    path, kinds = ckpt["qwen3"]
    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_NVFP4_KERNELS", "0")
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    name = next(n for n, k in kinds.items() if k == "nvfp4")
    assert _module(model, name).weight.dtype == torch.bfloat16
    assert not getattr(model, "_unsloth_compressed_tensors_nvfp4", 0)


def test_full_finetuning_decompresses_everything(ckpt):
    from unsloth.models import loader_utils

    path, kinds = ckpt["qwen3"]
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model, full_finetuning = True)
    assert all(_module(model, n).weight.dtype == torch.bfloat16 for n in kinds)


def test_fp8_group_routes_with_the_fp8_kernels_by_default(ckpt, monkeypatch):
    from unsloth.models import loader_utils

    path, kinds = ckpt["qwen3"]
    monkeypatch.delenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", raising = False)
    monkeypatch.setattr(loader_utils, "_zoo_peft_forward_keeps_fp8_inputs", lambda: True)
    model = _load_raw(path, "qwen3")
    loader_utils._prepare_compressed_tensors_model(model)
    lm_head = model.get_output_embeddings()
    for name, kind in kinds.items():
        module = _module(model, name)
        if module is lm_head:
            # The fused CE loss reads lm_head.weight directly, so it is decompressed, not routed.
            assert module.weight.dtype == torch.bfloat16
            assert not getattr(module, "_unsloth_compressed_tensors_fp8", False)
        elif kind == "fp8":
            assert (
                module.weight.dtype == torch.float8_e4m3fn
                and module._unsloth_compressed_tensors_fp8
            )
        else:
            assert module._unsloth_compressed_tensors_nvfp4


@pytest.mark.parametrize(
    "arch, api, fp8_kernels",
    [
        ("qwen3", "FastLanguageModel", False),
        ("qwen3", "FastModel", False),
        ("qwen3_5", "FastModel", False),
        ("qwen3_5", "FastLanguageModel", False),
        # The qwen3 fixture has an untied FP8 lm_head, which the FP8 route once handed to the fused CE loss.
        ("qwen3", "FastModel", True),
    ],
)
def test_lora_trains_on_the_packed_base_and_reloads(ckpt, arch, api, fp8_kernels, tmp_path):
    _check_lora_case(ckpt, arch, api, fp8_kernels, tmp_path)


@pytest.mark.parametrize("api", ["FastLanguageModel", "FastModel"])
def test_lora_trains_in_float16(ckpt, api, tmp_path):
    # T4 / V100 load in fp16: the NVFP4 quant state must dequantize to fp16, and the bf16-only FP8 kernels stay off.
    _check_lora_case(ckpt, "qwen3", api, None, tmp_path, dtype = "float16")


def _check_lora_case(
    ckpt,
    arch,
    api,
    fp8_kernels,
    tmp_path,
    dtype = "bfloat16",
):
    import json
    import subprocess

    if fp8_kernels:
        from unsloth.models import loader_utils
        if not loader_utils._zoo_peft_forward_keeps_fp8_inputs():
            pytest.skip("installed unsloth_zoo predates the FP8 kernel route")
    path, _ = ckpt[arch]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # A fresh process per case: patches and the compiled module cache from one loader leak into the next.
    env = dict(
        os.environ,
        PYTHONPATH = root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        UNSLOTH_COMPILE_LOCATION = str(tmp_path / "compiled"),
        UNSLOTH_IS_PRESENT = "1",
        NVFP4_CASE_DTYPE = dtype,
    )
    if fp8_kernels is not None:
        env["UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS"] = "1" if fp8_kernels else "0"
    else:
        env.pop("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", None)
    run = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "_nvfp4_fixtures.py"),
            "--lora-case",
            path,
            arch,
            api,
            str(tmp_path),
        ],
        env = env,
        capture_output = True,
        text = True,
        timeout = 1200,
    )
    lines = [l for l in run.stdout.splitlines() if l.startswith("LORA_CASE ")]
    assert run.returncode == 0 and lines, run.stderr[-3000:]
    r = json.loads(lines[-1][len("LORA_CASE ") :])
    assert all(torch.isfinite(torch.tensor(r["losses"]))) and r["losses"][-1] < r["losses"][0]
    # The packed base is never cast into, and activations reach it in a float dtype (W4A16, not uint8).
    assert r["input_dtypes"] and set(r["input_dtypes"]) <= {
        "torch.bfloat16",
        "torch.float16",
        "torch.float32",
    }
    assert r["packed_unchanged"] and r["packed_dtype"] == "torch.uint8" and r["lora_changed"]
    assert "adapter_model.safetensors" in r["saved"]
    assert r["adapters_equal"]
    # FastLanguageModel trains through fused LoRA kernels; a plain PEFT reload differs by ~0.04 on bf16 main too.
    if api == "FastModel":
        assert r["reload_max_abs"] <= 1e-2


def test_float8_lora_adapters_are_upcast_on_every_peft():
    # PEFT < 0.19 gives an FP8 base float8 LoRA weights, on get_peft_model and on PeftModel.from_pretrained;
    # the FP8 route beside NVFP4 relies on Unsloth's upcast.
    import unsloth  # noqa: F401
    from peft import LoraConfig, get_peft_model

    base = torch.nn.Sequential(torch.nn.Linear(32, 32, bias = False)).cuda()
    base[0].weight.data = base[0].weight.data.to(torch.float8_e4m3fn)
    model = get_peft_model(base, LoraConfig(r = 4, target_modules = ["0"]))
    dtypes = {n: p.dtype for n, p in model.named_parameters()}
    assert all(d == torch.float32 for n, d in dtypes.items() if "lora_" in n), dtypes
    assert model.base_model.model[0].base_layer.weight.dtype == torch.float8_e4m3fn

    import tempfile
    from peft import PeftModel

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        fresh = torch.nn.Sequential(torch.nn.Linear(32, 32, bias = False)).cuda()
        fresh[0].weight.data = fresh[0].weight.data.to(torch.float8_e4m3fn)
        fresh = PeftModel.from_pretrained(fresh, d, torch_device = "cuda")
    assert all(p.dtype == torch.float32 for n, p in fresh.named_parameters() if "lora_" in n)

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


@pytest.fixture(scope = "module")
def ckpt(tmp_path_factory):
    out = {}
    for arch in ("qwen3", "qwen3_5"):
        d = tmp_path_factory.mktemp(arch)
        out[arch] = (str(d), fx.build(str(d), arch))
    return out


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
    # Same keys as the checkpoint (a tied lm_head may add its alias).
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
        UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS = "1" if fp8_kernels else "0",
    )
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pipeline memory plan prices weights at the dtype the load casts them to.

A bf16 repo loaded in fp32 (a Mac without bf16, an fp16-incompatible family on an fp16 card)
doubles the same way.

SDXL (and Lumina-2's text encoder, Z-Image-Turbo's transformer) publish fp32 shards that halve once
loaded in bf16 / fp16, and a cache can also hold variant twins or single-file checkpoints that
``from_pretrained`` never opens. Sizing the plan from those bytes sent SDXL-Turbo to whole-model
offload on 24 GB cards."""

import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion as dmod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_memory import (
    OFFLOAD_NONE,
    DeviceMemory,
    unified_memory_shortfall_message,
)

MIB = 1024 * 1024
REPO = "org/sdxl-like"
REV = "a" * 40


def _safetensors(path, tensors):
    """A sparse safetensors file: a real header for ``{name: (dtype, numel)}`` and a zero payload."""
    width = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I64": 8}
    header, offset = {}, 0
    for name, (dtype, numel) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [numel],
            "data_offsets": [offset, offset + numel * width[dtype]],
        }
        offset += numel * width[dtype]
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as fh:
        fh.write(len(raw).to_bytes(8, "little"))
        fh.write(raw)
        fh.truncate(8 + len(raw) + offset)
    return path


def _f32(mib):
    return {"w": ("F32", mib * MIB // 4)}


def _bf16(mib):
    return {"w": ("BF16", mib * MIB // 2)}


def _snapshot(
    tmp_path,
    monkeypatch,
    files,
    repo = REPO,
):
    """``files`` = {relative path: tensors} under the live cache root's refs/main snapshot."""
    from huggingface_hub import constants as hf_constants

    live = tmp_path / "hub"
    unused = tmp_path / "import-time-hub"
    unused.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(dmod, "hub_cache_dir", lambda: str(live))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(unused))
    repo_dir = live / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / REV
    for rel, tensors in files.items():
        if isinstance(tensors, dict):
            _safetensors(snapshot / rel, tensors)
        else:
            (snapshot / rel).parent.mkdir(parents = True, exist_ok = True)
            (snapshot / rel).write_text(tensors)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(REV)
    return snapshot


def _card(
    monkeypatch,
    free = 9000,
    total = 9216,
    kind = "discrete_vram",
):
    """budget = free - 2048 reserve; resident margin 0.85 x budget. Runtime headroom fixed at 100."""
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", kind, free, total),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 100)


def _target(dtype):
    return types.SimpleNamespace(
        device = "cuda", backend = "cuda", supports_model_cpu_offload = True, dtype = dtype
    )


FAM = types.SimpleNamespace(name = "sdxl", base_repo = "unrelated/base")

SDXL_LIKE = {
    "model_index.json": "{}",
    "unet/diffusion_pytorch_model.safetensors": _f32(3600),
    "text_encoder/model.safetensors": _f32(300),
    "vae/diffusion_pytorch_model.safetensors": _f32(100),
}


def _plan(
    dtype,
    kind = "pipeline",
    single_file = None,
    repo = REPO,
):
    return DiffusionBackend()._plan_memory(
        _target(dtype), single_file, repo, FAM, None, False, kind = kind, repo_id = repo
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_an_fp32_pipeline_is_planned_at_the_half_precision_it_loads_in(
    tmp_path, monkeypatch, dtype
):
    _snapshot(tmp_path, monkeypatch, SDXL_LIKE)
    _card(monkeypatch)
    plan = _plan(getattr(torch, dtype))
    # 4000 MiB of fp32 shards hold 2000 MiB once cast: 2000 + 100 + 2048 fits the 5909 MiB margin.
    assert plan.estimates["model_dense_mib"] == 2000
    assert plan.estimates["companion_dense_mib"] == 2000
    assert plan.estimates["text_encoder_dense_mib"] == 150
    assert plan.offload_policy == OFFLOAD_NONE


def test_an_fp32_load_keeps_the_stored_size(tmp_path, monkeypatch):
    _snapshot(tmp_path, monkeypatch, SDXL_LIKE)
    _card(monkeypatch)
    plan = _plan(torch.float32)
    assert plan.estimates["model_dense_mib"] == 4000
    assert plan.offload_policy != OFFLOAD_NONE


def test_a_target_without_a_dtype_sizes_from_cached_bytes(tmp_path, monkeypatch):
    _snapshot(tmp_path, monkeypatch, SDXL_LIKE)
    _card(monkeypatch)
    plan = _plan(None)
    assert plan.estimates["model_dense_mib"] == 4000


def test_variant_twins_bin_twins_and_single_files_are_not_counted(tmp_path, monkeypatch):
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            "model_index.json": "{}",
            "unet/diffusion_pytorch_model.safetensors": _bf16(1000),
            "unet/diffusion_pytorch_model.fp16.safetensors": _bf16(1000),
            "text_encoder/model.safetensors": _bf16(300),
            "text_encoder/pytorch_model.bin": _bf16(300),
            # A single-file checkpoint of the same model, cached by a single-file pick of this repo.
            "sd_xl_like_fp16.safetensors": _bf16(2500),
        },
    )
    _card(monkeypatch)
    plan = _plan(torch.bfloat16)
    assert plan.estimates["model_dense_mib"] == 1300
    assert plan.estimates["companion_dense_mib"] == 1300
    assert plan.estimates["text_encoder_dense_mib"] == 300
    assert plan.offload_policy == OFFLOAD_NONE


def test_a_folder_holding_only_a_bin_keeps_it(tmp_path, monkeypatch):
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            "unet/diffusion_pytorch_model.safetensors": _bf16(1000),
            "text_encoder/pytorch_model.bin": _bf16(300),
        },
    )
    _card(monkeypatch)
    assert _plan(torch.bfloat16).estimates["model_dense_mib"] == 1300


def test_a_mixed_precision_or_quantised_file_keeps_its_stored_size(tmp_path, monkeypatch):
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            # bf16 weights with fp32 norms kept on purpose.
            "transformer/diffusion_pytorch_model.safetensors": {
                "w": ("BF16", 900 * MIB // 2),
                "norm": ("F32", 100 * MIB // 4),
            },
            # A 4-bit checkpoint: packed bytes plus fp32 quant state.
            "text_encoder/model.safetensors": {
                "w": ("U8", 200 * MIB),
                "absmax": ("F32", 20 * MIB // 4),
            },
        },
    )
    _card(monkeypatch)
    assert _plan(torch.bfloat16).estimates["model_dense_mib"] == 1220


def test_a_component_class_pinning_fp32_modules_keeps_its_full_size(tmp_path, monkeypatch):
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            "transformer/diffusion_pytorch_model.safetensors": _f32(2000),
            "text_encoder/config.json": json.dumps({"architectures": ["T5EncoderModel"]}),
            "text_encoder/model.safetensors": _f32(400),
        },
    )
    _card(monkeypatch)
    pinned = types.SimpleNamespace(T5EncoderModel = type("T5", (), {"_keep_in_fp32_modules": ["wo"]}))
    monkeypatch.setitem(sys.modules, "transformers", pinned)
    assert _plan(torch.float16).estimates["model_dense_mib"] == 1000 + 400
    free = types.SimpleNamespace(T5EncoderModel = type("T5", (), {"_keep_in_fp32_modules": None}))
    monkeypatch.setitem(sys.modules, "transformers", free)
    assert _plan(torch.float16).estimates["model_dense_mib"] == 1000 + 200


def test_gguf_companions_are_priced_at_the_load_dtype(tmp_path, monkeypatch):
    # Lumina-2 publishes its Gemma encoder in fp32: a GGUF pick loads it in bf16 beside the GGUF.
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": _f32(2000),
            "vae/diffusion_pytorch_model.safetensors": _f32(200),
            "transformer/diffusion_pytorch_model.safetensors": _f32(4000),
        },
    )
    gguf = tmp_path / "t-Q8_0.gguf"
    with open(gguf, "wb") as fh:
        fh.truncate(1000 * MIB)
    _card(monkeypatch)
    plan = _plan(torch.bfloat16, kind = "gguf", single_file = str(gguf))
    assert plan.estimates["companion_dense_mib"] == 1100
    assert plan.estimates["text_encoder_dense_mib"] == 1000
    assert plan.estimates["model_dense_mib"] == int(1000 * 1.05) + 1100


def test_a_unified_pool_no_longer_refuses_an_fp32_repo_that_fits_in_bf16(tmp_path, monkeypatch):
    _snapshot(tmp_path, monkeypatch, SDXL_LIKE)
    # budget 7000 - 2048 = 4952: 4000 + 2048 is refused, 2000 + 2048 fits.
    _card(monkeypatch, free = 7000, total = 7000, kind = "unified_memory")
    fp32 = _plan(torch.float32)
    bf16 = _plan(torch.bfloat16)
    assert unified_memory_shortfall_message(fp32, family = "sdxl") is not None
    assert unified_memory_shortfall_message(bf16, family = "sdxl") is None


@pytest.mark.parametrize(
    "tensors, itemsize, expected",
    [
        ({"a": ("F32", 10), "b": ("I64", 2)}, 2, 10 * 2 + 16),
        ({"a": ("F64", 10)}, 4, 40),
        ({"a": ("F32", 10)}, 4, 40),
        ({"a": ("BF16", 10), "b": ("F32", 5)}, 4, 60),
        ({"a": ("I64", 10)}, 2, None),
        ({"a": ("F32", 10), "b": ("F16", 10)}, 2, None),
        ({"a": ("U8", 10), "b": ("F32", 1)}, 4, None),
    ],
)
def test_safetensors_cast_bytes(tmp_path, tensors, itemsize, expected):
    path = _safetensors(tmp_path / "x.safetensors", tensors)
    assert DiffusionBackend._safetensors_cast_bytes(path, itemsize) == expected


def test_an_unreadable_header_keeps_the_stored_size(tmp_path):
    path = tmp_path / "x.safetensors"
    with open(path, "wb") as fh:
        fh.truncate(4096)
    assert DiffusionBackend._safetensors_cast_bytes(path, 2) is None


def test_an_oversized_header_length_is_not_read(tmp_path):
    path = tmp_path / "x.safetensors"
    with open(path, "wb") as fh:
        fh.write((2**62).to_bytes(8, "little"))
    assert DiffusionBackend._safetensors_cast_bytes(path, 2) is None


def test_a_half_precision_repo_loaded_in_fp32_is_priced_at_fp32(tmp_path, monkeypatch):
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            "transformer/diffusion_pytorch_model.safetensors": _bf16(1800),
            "text_encoder/model.safetensors": _bf16(200),
        },
    )
    _card(monkeypatch)
    assert _plan(torch.bfloat16).estimates["model_dense_mib"] == 2000
    fp32 = _plan(torch.float32)
    assert fp32.estimates["model_dense_mib"] == 4000
    assert fp32.estimates["text_encoder_dense_mib"] == 400
    assert fp32.offload_policy != OFFLOAD_NONE


def test_a_pinned_component_is_widened_on_an_fp32_load(tmp_path, monkeypatch):
    _snapshot(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/config.json": json.dumps({"architectures": ["T5EncoderModel"]}),
            "text_encoder/model.safetensors": _bf16(400),
        },
    )
    _card(monkeypatch)
    pinned = types.SimpleNamespace(T5EncoderModel = type("T5", (), {"_keep_in_fp32_modules": ["wo"]}))
    monkeypatch.setitem(sys.modules, "transformers", pinned)
    assert _plan(torch.float32).estimates["model_dense_mib"] == 800


def test_a_bin_is_kept_beside_safetensors_shards_the_loader_cannot_select(tmp_path, monkeypatch):
    snapshot = _snapshot(
        tmp_path,
        monkeypatch,
        {
            # Numbered shards with no index: the loader falls back to the .bin.
            "text_encoder/model-00001-of-00002.safetensors": _bf16(100),
            "text_encoder/pytorch_model.bin": _bf16(300),
            # Indexed shards: selectable, so the .bin twin is never opened.
            "text_encoder_2/model-00001-of-00002.safetensors": _bf16(100),
            "text_encoder_2/model-00002-of-00002.safetensors": _bf16(100),
            "text_encoder_2/model.safetensors.index.json": "{}",
            "text_encoder_2/pytorch_model.bin": _bf16(200),
        },
    )
    assert snapshot.is_dir()
    _card(monkeypatch)
    assert _plan(torch.bfloat16).estimates["model_dense_mib"] == 100 + 300 + 200

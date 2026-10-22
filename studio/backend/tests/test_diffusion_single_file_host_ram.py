# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A single-file DiT widened on load beyond free host RAM is refused before widening (it used to OOM
the app); files already at the compute dtype stay memory-mapped and are never refused."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

from core.inference import diffusion_comfy_quant as cq
from core.inference import diffusion_single_file_converters as conv
from core.inference import diffusion_small_host as small_host

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_single_file_krea2_hunyuanimage import (  # noqa: E402
    KREA2_CFG,
    _class,
    _tiny,
    krea2_original,
)

safetensors_torch = pytest.importorskip("safetensors.torch")

_LINEAR = re.compile(r"^blocks\.\d+\.(attn\.(wq|wk|wv|wo|gate)|mlp\.(gate|up|down))\.weight$")
_KWARGS = {"torch_dtype": torch.bfloat16, "config": "krea/Krea-2-Turbo", "subfolder": "transformer"}


def _write(tmp_path, kind):
    model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
    tensors = {}
    for key, value in krea2_original(model.state_dict()).items():
        if kind != "bf16" and _LINEAR.match(key):
            if kind == "fp8_scaled":
                scale = value.abs().max().clamp_min(1e-8) / 448.0
                tensors[key] = (value / scale).to(torch.float8_e4m3fn)
                tensors[key[: -len(".weight")] + ".scale_weight"] = scale.reshape(1).float()
            else:
                tensors[key] = value.to(torch.float8_e4m3fn)
        else:
            tensors[key] = value.to(torch.bfloat16).contiguous()
    if kind == "fp8_scaled":
        tensors["scaled_fp8"] = torch.zeros(0, dtype = torch.float8_e4m3fn)
    path = tmp_path / f"krea2_{kind}.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    widened = sum(
        v.numel() * 2 for k, v in tensors.items() if v.dtype == torch.float8_e4m3fn and v.numel()
    )
    return path, widened


@pytest.fixture
def krea2(monkeypatch):
    cls = _class("Krea2Transformer2DModel")
    monkeypatch.setattr(cls, "load_config", classmethod(lambda c, *a, **k: dict(KREA2_CFG)))
    # The tiny model widens a few KB; price every byte so the real check runs on it.
    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 0, raising = False)
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV, raising = False)
    return cls


def _host(
    monkeypatch,
    available_mib,
    total_mib = 32768,
):
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (total_mib, available_mib))


def _no_read(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the weights were read before the host RAM check")

    monkeypatch.setattr(safetensors_torch, "load_file", boom)


def test_fp8_file_widened_past_free_host_ram_is_refused_before_reading(
    tmp_path, monkeypatch, krea2
):
    path, _ = _write(tmp_path, "fp8")
    _host(monkeypatch, available_mib = 2048)  # under the 3 GB reserve alone
    _no_read(monkeypatch)
    with pytest.raises(RuntimeError, match = r"Not enough system RAM to load krea2_fp8"):
        conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS))


def test_fp8_scaled_comfy_file_without_fp8_runtime_is_refused(tmp_path, monkeypatch, krea2):
    path, _ = _write(tmp_path, "fp8_scaled")
    scan = cq.refuse_comfy_quant(str(path))
    assert scan is not None and scan.legacy_scaled_fp8
    _host(monkeypatch, available_mib = 2048)
    with pytest.raises(RuntimeError, match = "expanding its quantized weights"):
        cq.load_comfy_quant_transformer(
            krea2, str(path), scan, dict(_KWARGS), int8_backend = None, fp8_backend = None
        )


def test_same_files_load_when_the_host_has_room(tmp_path, monkeypatch, krea2):
    _host(monkeypatch, available_mib = 64 * 1024)
    path, _ = _write(tmp_path, "fp8")
    model = conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS))
    assert model.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    path, _ = _write(tmp_path, "fp8_scaled")
    scan = cq.refuse_comfy_quant(str(path))
    model = cq.load_comfy_quant_transformer(
        krea2, str(path), scan, dict(_KWARGS), int8_backend = None, fp8_backend = None
    )
    assert model.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16


def test_bf16_file_stays_mapped_and_is_never_refused(tmp_path, monkeypatch, krea2):
    """A bf16 file at a bf16 compute dtype writes no new host memory (only the fp32-kept norms)."""
    path, _ = _write(tmp_path, "bf16")
    need = conv.converted_host_bytes(
        str(path), torch.bfloat16, keep_fp32 = krea2._keep_in_fp32_modules
    )
    sizes = path.stat().st_size
    assert need < sizes // 10
    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 2)  # its few KB stay under any real floor
    _host(monkeypatch, available_mib = 0)
    conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS))


def test_widened_bytes_count_the_fp8_weights_at_bf16(tmp_path):
    path, widened = _write(tmp_path, "fp8")
    need = conv.converted_host_bytes(str(path), torch.bfloat16)
    assert need == widened
    # fp32 compute (CPU) widens the bf16 tensors too
    assert conv.converted_host_bytes(str(path), torch.float32) > 2 * widened


def test_kill_switch_and_unreadable_ram_never_refuse(tmp_path, monkeypatch, krea2):
    path, _ = _write(tmp_path, "fp8")
    _host(monkeypatch, available_mib = 1)
    monkeypatch.setenv(small_host.HOST_RAM_CHECK_ENV, "0")
    conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS))
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV)
    _host(monkeypatch, available_mib = None)
    conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS))


def _spy_need(monkeypatch):
    """The header estimate each load checks up front, recorded."""
    seen = []
    real = conv.HostRamBudget.check

    def spy(self, need_bytes):
        seen.append(need_bytes)
        return real(self, need_bytes)

    monkeypatch.setattr(conv.HostRamBudget, "check", spy)
    return seen


def test_fp16_pipeline_counts_the_deferred_dequantization(tmp_path, monkeypatch, krea2):
    """fp16 with ``_keep_in_fp32_modules`` defers the dtype choice until after key conversion, but
    with no fp8 runtime (always the case off bf16) every fp8 layer is still expanded to fp16."""
    path, widened = _write(tmp_path, "fp8_scaled")
    scan = cq.refuse_comfy_quant(str(path))
    seen = _spy_need(monkeypatch)
    _host(monkeypatch, available_mib = 64 * 1024)
    model = cq.load_comfy_quant_transformer(
        krea2,
        str(path),
        scan,
        {**_KWARGS, "torch_dtype": torch.float16},
        int8_backend = None,
        fp8_backend = None,
    )
    assert model.transformer_blocks[0].attn.to_q.weight.dtype == torch.float16
    # Everything else in the file is bf16, widened to fp16 (or fp32 for the kept norms) as well.
    keep = krea2._keep_in_fp32_modules
    rest = sum(
        v.numel() * (4 if any(m in k.split(".") for m in keep) else 2)
        for k, v in safetensors_torch.load_file(str(path)).items()
        if v.dtype == torch.bfloat16
    )
    assert seen and seen[0] >= widened + rest


def test_fnuz_float8_weights_are_priced_as_widened(tmp_path):
    """ROCm FNUZ float8 tensors are widened by ``value.to(want)`` like any other float8."""
    model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
    tensors = {}
    for key, value in krea2_original(model.state_dict()).items():
        if _LINEAR.match(key):
            tensors[key] = value.to(torch.float8_e4m3fnuz)
        else:
            tensors[key] = value.to(torch.bfloat16).contiguous()
    path = tmp_path / "krea2_fnuz.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    widened = sum(v.numel() * 2 for v in tensors.values() if v.dtype == torch.float8_e4m3fnuz)
    assert conv.converted_host_bytes(str(path), torch.bfloat16) == widened


@pytest.mark.parametrize("kind", ["fp8", "fp8_scaled"])
def test_kill_switch_charges_nothing(tmp_path, monkeypatch, krea2, kind):
    """``UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0`` installs no budget, so no helper charges one."""
    path, _ = _write(tmp_path, kind)
    charged = []
    monkeypatch.setattr(conv.HostRamBudget, "charge", lambda self, n: charged.append(n))
    monkeypatch.setenv(small_host.HOST_RAM_CHECK_ENV, "0")
    _host(monkeypatch, available_mib = 1)
    if kind == "fp8":
        conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS))
    else:
        cq.load_comfy_quant_transformer(
            krea2,
            str(path),
            cq.refuse_comfy_quant(str(path)),
            dict(_KWARGS),
            int8_backend = None,
            fp8_backend = None,
        )
    assert charged == []


@pytest.mark.parametrize("kind", ["fp8", "fp8_scaled"])
def test_a_file_that_fits_only_stored_loads_with_fp8_storage_and_is_refused_widened(
    tmp_path, monkeypatch, krea2, kind
):
    """fp8 storage keeps the codes memory-mapped, so the up-front estimate prices them at their stored bytes (plus
    the fp32 row scales): free RAM between the stored and the widened need loads with storage, and is refused when
    UNSLOTH_DIFFUSION_FP8_STORAGE=0 makes the load widen."""
    path, _ = _write(tmp_path, kind)
    keep = krea2._keep_in_fp32_modules
    # Price in bytes, no reserve: the tiny model's needs are a few KB apart.
    monkeypatch.setattr(conv, "_MIB", 1)
    monkeypatch.setattr(small_host, "host_reserve_mib", lambda total: 0)
    if kind == "fp8":
        widened = conv.converted_host_bytes(str(path), torch.bfloat16, keep_fp32 = keep)
        stored = conv.converted_host_bytes(
            str(path), torch.bfloat16, keep_fp32 = keep, dequantized = conv.fp8_stored_estimate(keep)
        ) + conv._fp8_stored_scale_bytes(str(path), keep)
    else:
        scan = cq.refuse_comfy_quant(str(path))
        widened = conv.converted_host_bytes(
            str(path), torch.bfloat16, dequantized = {n + ".weight": 1 for n in scan.layers}.get
        )
        stored = conv.converted_host_bytes(
            str(path), torch.bfloat16, dequantized = {n + ".weight": 0 for n in scan.layers}.get
        )
    assert 0 < stored < widened
    _host(monkeypatch, available_mib = (stored + widened) // 2)

    def load(**kw):
        if kind == "fp8":
            return conv.load_original_layout_transformer(krea2, str(path), dict(_KWARGS), **kw)
        return cq.load_comfy_quant_transformer(
            krea2,
            str(path),
            cq.refuse_comfy_quant(str(path)),
            dict(_KWARGS),
            int8_backend = None,
            fp8_backend = None,
            **kw,
        )

    model = load(fp8_storage = True)
    assert model._unsloth_fp8_stored > 0
    assert model.transformer_blocks[0].attn.to_q.weight_q.dtype == torch.uint8
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "0")
    with pytest.raises(RuntimeError, match = "Not enough system RAM"):
        load(fp8_storage = cq.comfy_fp8_storage(None))

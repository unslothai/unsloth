# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host load route (``diffusion_small_host.py``): host-RAM pre-check, decision table, the memory-mapped
conversions, and the plan the route pins.

Sizes are the ones a Kaggle T4 (fp16 only, 15 GB, 31 GB host RAM) sees: FLUX.1-schnell stores a 22.7 GB bf16
transformer and an 8.9 GB bf16 T5; Qwen-Image a 38.9 GB transformer and a 15.8 GB Qwen2.5-VL. CPU only.
"""

from __future__ import annotations

import json
import struct
import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_small_host as sh

FLUX1 = {
    "transformer": sh.StoredComponent("transformer", 22700, "bfloat16"),
    "text_encoder_2": sh.StoredComponent("text_encoder_2", 9083, "bfloat16"),
    "text_encoder": sh.StoredComponent("text_encoder", 235, "float32"),
    "vae": sh.StoredComponent("vae", 160, "float32"),
}
QWEN = {
    "transformer": sh.StoredComponent("transformer", 38900, "bfloat16"),
    "text_encoder": sh.StoredComponent("text_encoder", 15820, "bfloat16"),
    "vae": sh.StoredComponent("vae", 242, "bfloat16"),
}
KAGGLE_TOTAL = 31 * 1024
KAGGLE_AVAILABLE = 29900


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.SMALL_HOST_ENV, sh.HOST_RAM_CHECK_ENV):
        monkeypatch.delenv(name, raising = False)


def _decide(
    comps,
    dtype = torch.float16,
    total = KAGGLE_TOTAL,
    available = KAGGLE_AVAILABLE,
    **kw,
):
    return sh.decide_small_host(
        comps,
        dtype,
        device = kw.pop("device", "cuda"),
        host_total_mib = total,
        host_available_mib = available,
        **kw,
    )


def test_flux1_on_a_31gb_fp16_host_takes_the_route():
    d = _decide(FLUX1)
    assert d.engaged and d.refuse is None
    # only the large bf16 components load memory-mapped; the fp32 CLIP and VAE convert as before
    assert d.storage_dtypes == {"transformer": "bfloat16", "text_encoder_2": "bfloat16"}
    assert d.dense_host_mib > KAGGLE_AVAILABLE
    assert d.route_host_mib == 22700 // 2


def test_unet_and_large_vae_are_budgeted_at_their_converted_size():
    comps = {
        "unet": sh.StoredComponent("unet", 4900, "bfloat16"),
        "vae": sh.StoredComponent("vae", 600, "bfloat16"),
        "text_encoder_2": sh.StoredComponent("text_encoder_2", 1300, "bfloat16"),
    }
    d = _decide(comps, total = 8 * 1024, available = 6000)
    assert d.engaged
    # only a DiT is stored as int8; the UNet and VAE convert dense on the host, the encoder stays memory-mapped
    assert d.route_host_mib == 4900 + 600
    assert d.refuse is not None
    assert (
        d.route_host_mib
        == _decide(comps, dtype = torch.float32, total = 8 * 1024, available = 6000).route_host_mib // 2
    )


def test_qwen_image_fp32_promoted_takes_the_route():
    d = _decide(QWEN, dtype = torch.float32)
    assert d.engaged and d.refuse is None
    assert d.dense_host_mib > 2 * (38900 + 15820)


def test_large_host_and_bf16_cards_are_untouched():
    assert not _decide(FLUX1, total = 230 * 1024, available = 220 * 1024).engaged
    # bf16 compute loads a bf16 repo memory-mapped already: never routed, whatever the host
    assert not _decide(FLUX1, dtype = torch.bfloat16).engaged
    assert not _decide(FLUX1, device = "mps").engaged
    assert not _decide({"transformer": sh.StoredComponent("transformer", 9000, "float16")}).engaged


def test_kill_switch_and_force(monkeypatch):
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "0")
    assert not _decide(FLUX1).engaged
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "1")
    assert _decide(FLUX1, total = 230 * 1024, available = 220 * 1024).engaged


def test_refuses_before_loading_when_even_the_route_cannot_fit(monkeypatch):
    d = _decide(QWEN, dtype = torch.float32, total = 16 * 1024, available = 14000)
    assert d.engaged and d.refuse and "Not enough system RAM" in d.refuse
    monkeypatch.setenv(sh.HOST_RAM_CHECK_ENV, "0")
    assert _decide(QWEN, dtype = torch.float32, total = 16 * 1024, available = 14000).refuse is None


def test_lora_keeps_the_dense_load():
    d = _decide(FLUX1, lora_active = True)
    assert not d.engaged and d.refuse is None and "LoRA" in d.reason


def test_torch_dtype_map_only_names_routed_components():
    d = _decide(FLUX1)
    m = sh.torch_dtype_map(d, torch.float16)
    assert m == {
        "default": torch.float16,
        "transformer": torch.bfloat16,
        "text_encoder_2": torch.bfloat16,
    }
    assert sh.torch_dtype_map(_decide(FLUX1, available = 10**6), torch.float16) is torch.float16


def _write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, (dtype, nbytes) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [nbytes // 2],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\0" * offset)


def test_stored_components_read_headers_only(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "scheduler").mkdir()
    _write_safetensors(tmp_path / "transformer" / "a.safetensors", {"w": ("BF16", 3 << 20)})
    _write_safetensors(tmp_path / "transformer" / "b.safetensors", {"x": ("F32", 1 << 20)})
    _write_safetensors(tmp_path / "vae" / "v.safetensors", {"w": ("F32", 2 << 20)})
    comps = sh.stored_components(tmp_path)
    assert comps["transformer"] == sh.StoredComponent("transformer", 4, "bfloat16")
    assert comps["vae"].dtype == "float32"
    assert "scheduler" not in comps
    assert sh.resolve_snapshot_dir(tmp_path) == tmp_path


def test_t5_kept_fp32_layers_are_budgeted_and_variants_ignored(tmp_path):
    te = tmp_path / "text_encoder_2"
    te.mkdir()
    (te / "config.json").write_text(json.dumps({"architectures": ["T5EncoderModel"]}))
    _write_safetensors(
        te / "model.safetensors",
        {
            "encoder.block.0.layer.1.DenseReluDense.wo.weight": ("BF16", 2 << 20),
            "encoder.block.0.layer.1.DenseReluDense.wi.weight": ("BF16", 6 << 20),
        },
    )
    _write_safetensors(te / "model.fp16.safetensors", {"w": ("F16", 8 << 20)})
    comps = sh.stored_components(tmp_path)
    assert comps["text_encoder_2"] == sh.StoredComponent("text_encoder_2", 8, "bfloat16", 2)
    tiny = {
        "transformer": sh.StoredComponent("transformer", 4000, "bfloat16"),
        "text_encoder_2": sh.StoredComponent("text_encoder_2", 9084, "bfloat16", 1920),
    }
    # the streamed encoder's ``wo`` converts to fp32 on the host: 1920 MiB stored -> 3840 MiB loaded
    assert _decide(tiny).route_host_mib == 2000 + 3840


def test_int8_weight_storage_matches_the_dense_layer():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096), torch.nn.LayerNorm(4096), torch.nn.Linear(4096, 16)
    ).to(torch.bfloat16)
    ref = [m.weight.detach().float().clone() for m in (model[0], model[2])]
    stats = sh.quantize_int8_weight_(model, compute_dtype = torch.float32, work_device = "cpu")
    assert stats["linears"] == 1 and stats["dense_linears"] == 1
    big = model[0]
    assert isinstance(big, sh.int8_linear_class()) and big.qweight.dtype == torch.int8
    assert model[1].weight.dtype == torch.float32 and model[2].weight.dtype == torch.float32
    x = torch.randn(8, 2048)
    out = big(x)
    want = torch.nn.functional.linear(x, ref[0], model[0].bias.float())
    rel = (out - want).norm() / want.norm()
    assert rel < 0.01, rel


def test_streamed_encoder_keeps_linear_storage_and_reports_compute_dtype():
    pytest.importorskip("diffusers")
    from diffusers.hooks import HookRegistry

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wi = torch.nn.Linear(64, 128, bias = False)
            self.wo = torch.nn.Linear(128, 64, bias = False)

        def forward(self, x):
            # T5's pattern: the parent reads wo.weight.dtype BEFORE wo runs
            h = self.wi(x)
            return self.wo(h.to(self.wo.weight.dtype))

    class Enc(torch.nn.Module):
        _keep_in_fp32_modules = ["wo"]

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 64)
            self.block = Block()
            self.norm = torch.nn.LayerNorm(64)
            # computed at init, never stored: fp32 in the dense fp16 load too (RoPE inv_freq)
            self.register_buffer("inv_freq", torch.rand(8), persistent = False)

        @property
        def dtype(self):
            return next(p.dtype for p in self.parameters() if p.is_floating_point())

        def forward(self, ids):
            return self.norm(self.block(self.embed(ids)))

    torch.manual_seed(0)
    enc = Enc()
    for p in enc.parameters():
        p.data = p.data.to(
            torch.bfloat16
        )  # the stored-dtype load: parameters bf16, init-time buffers fp32
    dense = Enc()
    dense.load_state_dict({k: v.float() for k, v in enc.state_dict().items()})
    dense.inv_freq = enc.inv_freq.clone()
    sh.prepare_streamed_encoder_(enc, torch.float16)
    assert enc.inv_freq.dtype == torch.float32
    enc_fp16 = enc
    enc = Enc()
    for p in enc.parameters():
        p.data = p.data.to(torch.bfloat16)
    dense.load_state_dict({k: v.float() for k, v in enc.state_dict().items()})
    dense.inv_freq = enc.inv_freq.clone()
    del enc_fp16
    sh.prepare_streamed_encoder_(enc, torch.float32)
    # the first float parameter's owner converts so .dtype reports the compute dtype
    assert enc.dtype == torch.float32
    assert enc.block.wi.weight.dtype == torch.bfloat16  # memory-mapped storage stays
    assert (
        HookRegistry.check_if_exists_or_initialize(enc.block.wi).get_hook("layerwise_casting")
        is not None
    )
    assert enc.block.wo.weight.dtype == torch.float32  # kept-fp32 converts now
    ids = torch.randint(0, 100, (2, 7))
    with torch.no_grad():
        assert torch.equal(enc(ids), dense(ids))  # bf16 -> fp32 is exact


def _group_plan(**kw):
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 14647, 15095)
    base = dict(
        requested_mode = dm.MEMORY_MODE_AUTO,
        offload_policy = dm.OFFLOAD_MODEL,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = memory,
        estimates = {
            "safe_device_budget_mib": 12599,
            "runtime_headroom_mib": 3072,
            "base_overhead_mib": 2048,
        },
    )
    base.update(kw)
    return dm.MemoryPlan(**base)


def test_route_plan_streams_encoders_and_keeps_what_fits(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        dm,
        "_loaded_component_mib",
        lambda pipe: {
            "transformer": (11353, "dit"),
            "text_encoder_2": (9346, "text_encoder"),
            "vae": (160, "other"),
        },
    )
    pipe = types.SimpleNamespace(_unsloth_small_host = {"components": {}})
    log = types.SimpleNamespace(info = lambda *a, **k: None)
    plan = DiffusionBackend._small_host_plan(None, pipe, _group_plan(), None, log)
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_text_encoders
    assert plan.stream_transformer and plan.resident_transformer_mib == 12599 - 3072 - 2048 - 160
    # a plan that already keeps everything resident, or a pipe the route did not touch, is left alone
    none = _group_plan(offload_policy = dm.OFFLOAD_NONE)
    assert DiffusionBackend._small_host_plan(None, pipe, none, None, log) is none
    other = _group_plan()
    assert DiffusionBackend._small_host_plan(None, object(), other, None, log) is other


def test_lora_refused_and_status_reports_the_int8_route():
    from core.inference.diffusion import _small_host_int8

    pipe = types.SimpleNamespace(
        _unsloth_small_host = {"components": {"transformer": "int8 weights (1 MiB)"}}
    )
    assert _small_host_int8(pipe)
    assert not _small_host_int8(
        types.SimpleNamespace(_unsloth_small_host = {"components": {"transformer": "cast on device"}})
    )
    assert not _small_host_int8(object())

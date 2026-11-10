# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#17137: on Windows an image load onto an Intel Arc iGPU reads same-dtype shards through ``pread`` too, so a mapped
shard is not charged to commit while its tensors are copied into the GPU's share of host RAM. CPU only."""

from __future__ import annotations

import sys
import types

import pytest
import torch

import core.inference.safetensors_pread as sp
from core.inference import host_pinning

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as modeling_utils  # noqa: E402

requires_backend = pytest.mark.skipif(
    not sp._supports_backend(modeling_utils.safe_open),
    reason = "installed safetensors has no backend keyword",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    yield
    assert sp._depth == 0 and sp._original is None and not sp._active_rules()


def _fake_xpu(
    monkeypatch,
    device_id,
    available = True,
):
    props = types.SimpleNamespace(device_id = device_id, name = "Intel(R) Arc(TM) GPU")
    xpu = types.SimpleNamespace(
        is_available = lambda: available, get_device_properties = lambda *a: props
    )
    monkeypatch.setattr(torch, "xpu", xpu, raising = False)


@pytest.mark.parametrize(
    "device_id, available, expected",
    [
        (0x64A0, True, True),  # Lunar Lake Arc 130V / 140V (the report)
        (0x7D55, True, True),  # Meteor Lake Arc
        (0xB080, True, True),  # Panther Lake Arc
        (0xE20B, True, False),  # Arc B580 (discrete)
        (0x56A0, True, False),  # Arc A770 (discrete)
        (0x0BD5, True, False),  # Data Center GPU Max
        (0x64A0, False, False),
        (None, True, False),  # torch without device_id
    ],
)
def test_xpu_is_integrated(monkeypatch, device_id, available, expected):
    _fake_xpu(monkeypatch, device_id, available)
    assert host_pinning.xpu_is_integrated() is expected


def test_xpu_is_integrated_never_raises(monkeypatch):
    def boom(*a):
        raise RuntimeError("no level zero")

    monkeypatch.setattr(
        torch,
        "xpu",
        types.SimpleNamespace(is_available = lambda: True, get_device_properties = boom),
        raising = False,
    )
    assert host_pinning.xpu_is_integrated() is False


def _record_backends(monkeypatch):
    seen = []
    real = modeling_utils.safe_open

    def spy(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return real(*args, **kwargs)

    spy.__text_signature__ = real.__text_signature__
    monkeypatch.setattr(modeling_utils, "safe_open", spy)
    return seen


@requires_backend
@pytest.mark.parametrize(
    "stored, keep, every, expected",
    [
        (torch.bfloat16, (), False, None),  # unchanged off an iGPU: a same-dtype shard stays mapped
        (torch.bfloat16, (), True, "pread"),
        (torch.float16, (), True, "pread"),
        (torch.float32, (), True, "pread"),
        (torch.bfloat16, ("text_encoder",), True, None),  # a kept component still stays mapped
    ],
)
def test_every_shard_backend_choice(monkeypatch, tmp_path, stored, keep, every, expected):
    from safetensors.torch import save_file

    comp = tmp_path / "text_encoder"
    comp.mkdir()
    path = comp / "model.safetensors"
    save_file({"w": torch.zeros(64, 64, dtype = stored)}, str(path))
    monkeypatch.setattr(sys, "platform", "win32")
    seen = _record_backends(monkeypatch)
    with sp.pread_safetensors(convert_to = stored, keep = keep, every_shard = every):
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu") as handle:
            handle.get_slice("w")[...]
    assert seen == [expected]


@pytest.mark.parametrize(
    "platform, igpu, every",
    [("win32", True, True), ("win32", False, False), ("linux", True, False)],
)
def test_image_load_scope_engages_on_windows_igpu_only(monkeypatch, platform, igpu, every):
    from core.inference import diffusion

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(host_pinning, "xpu_is_integrated", lambda: igpu)
    got = {}

    def fake_scope(
        convert_to = None,
        keep = (),
        every_shard = False,
    ):
        got.update(convert_to = convert_to, keep = keep, every_shard = every_shard)
        return "scope"

    monkeypatch.setattr(sp, "pread_safetensors", fake_scope)
    assert diffusion._converted_shards_pread(torch.bfloat16, None) == "scope"
    assert got == {"convert_to": torch.bfloat16, "keep": (), "every_shard": every}


_COMMIT_ARM = r"""
import ctypes, sys, torch
sys.path.insert(0, sys.argv[1])
import core.inference.safetensors_pread as sp
from transformers import LlamaForCausalLM

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
        "PeakPagefileUsage")]

k = ctypes.WinDLL("kernel32", use_last_error = True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]

def counters():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    assert k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb), ctypes.get_last_error()
    return c

import transformers.modeling_utils as mu

base = counters().PagefileUsage
with sp.pread_safetensors(convert_to = torch.bfloat16, every_shard = sys.argv[3] == "pread"):
    scoped = mu.safe_open

    # Studio pins transformers 5.5, which passes no backend; 5.17+ asks for pread on Windows itself
    def as_studio_pin(*args, **kwargs):
        kwargs.pop("backend", None)
        return scoped(*args, **kwargs)

    mu.safe_open = as_studio_pin
    model = LlamaForCausalLM.from_pretrained(sys.argv[2], dtype = torch.bfloat16)
# what .to("xpu") does on an iGPU: each weight gets a fresh allocation in the same RAM, one at a time
with torch.no_grad():
    for p in model.parameters():
        p.data = p.data.clone()
print("PEAK_MIB", (counters().PeakPagefileUsage - base) >> 20)
"""


@pytest.mark.skipif(sys.platform != "win32", reason = "process commit is a Windows counter")
@requires_backend
def test_windows_igpu_load_commits_about_one_copy(tmp_path):
    """bf16 load at bf16 then moved onto memory the host shares: a mapped shard stays charged until its last tensor
    has moved (stored + copy); pread frees each tensor as it moves."""
    import subprocess
    import warnings
    from pathlib import Path

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 32000,
        hidden_size = 2048,
        intermediate_size = 5632,
        num_hidden_layers = 12,
        num_attention_heads = 16,
        num_key_value_heads = 16,
    )
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model = model.to_empty(device = "cpu").to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    # one shard, like Qwen-Image-2.1's 16.3 GB text_encoder/model.safetensors
    model.save_pretrained(tmp_path, max_shard_size = "4GB")
    stored_mib = sum(f.stat().st_size for f in tmp_path.glob("*.safetensors")) >> 20
    del model

    backend = str(Path(sp.__file__).resolve().parents[2])
    peaks = {}
    for arm in ("mmap", "pread"):
        out = subprocess.run(
            [sys.executable, "-c", _COMMIT_ARM, backend, str(tmp_path), arm],
            capture_output = True,
            text = True,
            timeout = 900,
        )
        assert out.returncode == 0, out.stderr[-3000:]
        peaks[arm] = int(
            next(l for l in out.stdout.splitlines() if l.startswith("PEAK_MIB")).split()[1]
        )
    warnings.warn(
        f"windows commit for a {stored_mib} MiB bf16 load moved onto shared memory: {peaks}"
    )
    assert peaks["mmap"] - peaks["pread"] >= stored_mib * 0.7, peaks

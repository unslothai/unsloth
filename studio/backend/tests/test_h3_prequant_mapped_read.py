# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""H3's host-placed prequant pickle is mapped on Linux; ``UNSLOTH_DIFFUSION_PREQUANT_MMAP=0`` restores the full read."""

import ast
import inspect
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]


def test_host_destination_maps_only_when_a_host_copy_follows_on_linux(monkeypatch):
    from core.inference import diffusion_prequant as pq

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_MMAP", raising = False)
    monkeypatch.setattr("sys.platform", "linux")
    assert not pq.prequant_mmap_enabled("cpu")
    assert pq.prequant_mmap_enabled("cpu", host_copy_follows = True)
    assert pq.prequant_mmap_enabled("cuda", host_copy_follows = False)
    monkeypatch.setattr("sys.platform", "win32")
    assert not pq.prequant_mmap_enabled("cpu", host_copy_follows = True)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PREQUANT_MMAP", "0")
    assert not pq.prequant_mmap_enabled("cpu", host_copy_follows = True)


def test_mapped_host_read_matches_the_full_read(tmp_path, monkeypatch):
    from core.inference import diffusion_prequant as pq

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_MMAP", raising = False)
    monkeypatch.setattr("sys.platform", "linux")
    ckpt = {
        "format": pq.PREQUANT_FORMAT,
        "state_dict": {"w": torch.arange(64, dtype = torch.int8).reshape(8, 8), "s": torch.rand(8)},
        "metadata": {"scheme": "int8"},
    }
    path = tmp_path / "x.pt"
    torch.save(ckpt, path)
    seen = []
    real = pq._load_prequant_checkpoint

    def spy(p, **kwargs):
        seen.append(kwargs.get("mmap"))
        return real(p, **kwargs)

    monkeypatch.setattr(pq, "_load_prequant_checkpoint", spy)
    mapped = pq._read_prequant_for(str(path), "cpu", host_copy_follows = True)
    full = pq._read_prequant_for(str(path), "cpu")
    assert seen == [True, None]
    for key, value in ckpt["state_dict"].items():
        assert torch.equal(mapped["state_dict"][key], value)
        assert torch.equal(full["state_dict"][key], value)


def test_loader_forwards_the_hint():
    from core.inference import diffusion_prequant as pq

    params = inspect.signature(pq.load_prequantized_transformer).parameters
    assert params["host_copy_follows"].default is False
    src = inspect.getsource(pq.load_prequantized_transformer)
    assert "host_copy_follows = host_copy_follows" in src


def _h3_prequant_call() -> ast.Call:
    tree = ast.parse((BACKEND / "core/inference/video.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_h3_modular_pipeline":
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "load_prequantized_transformer"
                ):
                    return call
    raise AssertionError("no load_prequantized_transformer call in _load_h3_modular_pipeline")


def test_h3_load_passes_the_hint_from_its_render_device():
    call = _h3_prequant_call()
    kw = {k.arg: k.value for k in call.keywords}
    assert ast.unparse(kw["device"]) == "'cpu'"
    assert ast.unparse(kw["host_copy_follows"]) == "_h3_host_copy_follows(device)"


def test_h3_host_copy_follows_only_on_a_discrete_gpu(monkeypatch):
    from core.inference import host_pinning
    from core.inference import video as V

    monkeypatch.setattr(host_pinning, "device_is_integrated", lambda device = None: False)
    assert V._h3_host_copy_follows("cuda")
    assert V._h3_host_copy_follows("cuda:1")
    assert not V._h3_host_copy_follows("cpu")
    assert not V._h3_host_copy_follows(None)
    monkeypatch.setattr(host_pinning, "device_is_integrated", lambda device = None: True)
    assert not V._h3_host_copy_follows("cuda")

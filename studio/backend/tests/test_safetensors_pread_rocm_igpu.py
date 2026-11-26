# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux ROCm integrated GPUs (unified memory) read every safetensors shard through ``pread``, transformers' and
diffusers' readers alike; CUDA, discrete ROCm and Windows keep their behaviour. CPU only."""

from __future__ import annotations

import pytest
import torch

import core.inference.safetensors_pread as sp

transformers = pytest.importorskip("transformers")
import safetensors.torch as st_torch  # noqa: E402
import transformers.modeling_utils as modeling_utils  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

requires_backend = pytest.mark.skipif(
    not sp._supports_backend(modeling_utils.safe_open),
    reason = "installed safetensors has no backend keyword",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    original_load_file = st_torch.load_file
    yield
    assert sp._depth == 0 and sp._original is None and not sp._active_rules()
    assert st_torch.load_file is original_load_file


def _platform(monkeypatch, platform, rocm, integrated):
    monkeypatch.setattr(sp.sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", "7.13" if rocm else None, raising = False)
    if not rocm:
        monkeypatch.setattr(torch, "__version__", "2.11.0+cu128", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import core.inference.host_pinning as hp

    monkeypatch.setattr(hp, "device_is_integrated", lambda device = None: integrated)


@pytest.mark.parametrize(
    "platform, rocm, integrated, expected",
    [
        ("linux", True, True, True),  # Strix Halo and other APUs
        ("linux", True, False, False),  # discrete Radeon / Instinct
        ("linux", False, True, False),  # CUDA (Jetson / GB10 keep their own route)
        ("linux", False, False, False),  # CUDA discrete: untouched
        ("darwin", True, True, False),
    ],
)
def test_rocm_integrated_and_default(monkeypatch, platform, rocm, integrated, expected):
    _platform(monkeypatch, platform, rocm, integrated)
    assert sp.rocm_integrated() is expected
    assert sp.pread_wanted() is expected


def test_kill_switch_wins_on_rocm_integrated(monkeypatch):
    _platform(monkeypatch, "linux", True, True)
    monkeypatch.setenv(sp.PREAD_ENV, "0")
    assert not sp.pread_wanted()
    with sp.pread_safetensors(every_shard = True) as on:
        assert not on
    assert not getattr(st_torch.load_file, "_unsloth_pread", False)


def test_rocm_integrated_never_raises(monkeypatch):
    monkeypatch.setattr(sp.sys, "platform", "linux")

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(torch.version, "hip", "7.13", raising = False)
    assert sp.rocm_integrated() is False


@requires_backend
def test_diffusers_reader_goes_through_pread_in_scope(monkeypatch, tmp_path):
    _platform(monkeypatch, "linux", True, True)
    shard = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"
    shard.parent.mkdir()
    tensors = {
        "a": torch.randn(4, 3, dtype = torch.bfloat16),
        "b": torch.arange(6, dtype = torch.int64),
    }
    save_file(tensors, str(shard))
    seen = []
    from safetensors import safe_open as real_safe_open

    import functools

    @functools.wraps(real_safe_open)
    def spy(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr("safetensors.safe_open", spy)
    plain = st_torch.load_file(str(shard))
    with sp.pread_safetensors(every_shard = True) as on:
        assert on
        assert getattr(st_torch.load_file, "_unsloth_pread", False)
        got = st_torch.load_file(str(shard))
    assert seen == ["pread"]
    assert set(got) == set(plain) and all(torch.equal(got[k], plain[k]) for k in plain)


@requires_backend
def test_cuda_discrete_keeps_the_diffusers_reader(monkeypatch, tmp_path):
    _platform(monkeypatch, "linux", False, False)
    monkeypatch.setenv(
        sp.PREAD_ENV, "1"
    )  # even forced on, diffusers' reader is only routed on a ROCm APU
    with sp.pread_safetensors(every_shard = True) as on:
        assert on
        assert not getattr(st_torch.load_file, "_unsloth_pread", False)


@requires_backend
def test_reader_outside_a_scope_is_the_original(monkeypatch, tmp_path):
    _platform(monkeypatch, "linux", True, True)
    shard = tmp_path / "model.safetensors"
    save_file({"x": torch.ones(2)}, str(shard))
    with sp.pread_safetensors(every_shard = True):
        wrapped = st_torch.load_file
    # another thread (no rules) calling the wrapper while a scope is open elsewhere gets the original path
    out = sp._with_pread_load_file(st_torch.load_file, None)(str(shard))
    assert torch.equal(out["x"], torch.ones(2))
    assert wrapped is not st_torch.load_file


def _fadvise_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sp.os,
        "posix_fadvise",
        lambda fd, off, ln, advice: calls.append((off, ln, advice)),
        raising = False,
    )
    monkeypatch.setattr(sp.os, "POSIX_FADV_DONTNEED", 4, raising = False)
    return calls


def _spans_of(path):
    import json
    import struct

    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    return {
        k: (8 + size + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
        for k, v in header.items()
        if k != "__metadata__"
    }


@requires_backend
def test_diffusers_reader_drops_each_tensors_pages(monkeypatch, tmp_path):
    _platform(monkeypatch, "linux", True, True)
    calls = _fadvise_spy(monkeypatch)
    shard = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"
    shard.parent.mkdir()
    tensors = {
        "a": torch.randn(64, 3, dtype = torch.bfloat16),
        "b": torch.arange(6, dtype = torch.int64),
    }
    save_file(tensors, str(shard), metadata = {"format": "pt"})
    with sp.pread_safetensors(every_shard = True):
        got = st_torch.load_file(str(shard))
    assert all(torch.equal(got[k], tensors[k]) for k in tensors)
    assert sorted(calls) == sorted((off, ln, 4) for off, ln in _spans_of(shard).values())


@requires_backend
@pytest.mark.parametrize("env", ["0", "off"])
def test_fadvise_kill_switch_keeps_pages(monkeypatch, tmp_path, env):
    _platform(monkeypatch, "linux", True, True)
    monkeypatch.setenv(sp.FADVISE_ENV, env)
    calls = _fadvise_spy(monkeypatch)
    shard = tmp_path / "model.safetensors"
    save_file({"x": torch.ones(8)}, str(shard))
    with sp.pread_safetensors(every_shard = True):
        got = st_torch.load_file(str(shard))
    assert torch.equal(got["x"], torch.ones(8)) and calls == []


@pytest.mark.parametrize(
    "platform, rocm, integrated",
    [("linux", False, False), ("linux", True, False), ("win32", True, True)],
)
def test_fadvise_and_te_backend_only_on_rocm_integrated(monkeypatch, platform, rocm, integrated):
    _platform(monkeypatch, platform, rocm, integrated)
    assert not sp.fadvise_wanted()
    assert sp.igpu_pread_backend() is None


def test_page_dropper_never_raises_on_a_bad_header(monkeypatch, tmp_path):
    _platform(monkeypatch, "linux", True, True)
    calls = _fadvise_spy(monkeypatch)
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\x01")
    with sp._page_dropper(str(bad)) as drop:
        drop("x")
    assert calls == []


@requires_backend
def test_precast_text_encoder_reads_through_pread_on_rocm_integrated(monkeypatch, tmp_path):
    import json

    import core.inference.prequant_safetensors as ps

    _platform(monkeypatch, "linux", True, True)
    calls = _fadvise_spy(monkeypatch)
    assert sp.igpu_pread_backend() == "pread"
    path = tmp_path / "te.safetensors"
    tensors = {"w": torch.randn(4, 4), "lm_head.weight": torch.randn(2, 4)}
    meta = {
        ps.UNSLOTH_FORMAT_KEY: "int8",
        ps.UNSLOTH_METADATA_KEY: "{}",
        "tensor_names": json.dumps(sorted(tensors)),
        **{k: json.dumps({"_type": "Tensor"}) for k in tensors},
    }
    save_file(tensors, str(path), metadata = meta)
    seen = []
    from safetensors import safe_open as real_safe_open

    import functools

    @functools.wraps(real_safe_open)
    def spy(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr("safetensors.safe_open", spy)
    plain = ps.load_plain_prequant_safetensors(str(path), skip_names = ["lm_head.weight"])
    routed = ps.load_plain_prequant_safetensors(
        str(path), skip_names = ["lm_head.weight"], backend = sp.igpu_pread_backend()
    )
    assert seen == [None, "pread"]
    assert set(routed["state_dict"]) == {"w"} and torch.equal(
        routed["state_dict"]["w"], plain["state_dict"]["w"]
    )
    assert [c[:2] for c in calls] == [_spans_of(path)["w"]]

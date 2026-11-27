# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reader decision table: integrated NVIDIA always pread (#19905), a ROCm integrated GPU pread only under host-memory
pressure, discrete GPUs mmap, UNSLOTH_SAFETENSORS_PREAD=auto|pread|mmap wins. Checksums unchanged. CPU only."""

from __future__ import annotations

import functools
import os
import sys

import pytest
from torch.torch_version import TorchVersion
import torch

import core.inference.safetensors_pread as sp
from core.inference import host_pinning

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as modeling_utils  # noqa: E402

mlu = pytest.importorskip("diffusers.models.model_loading_utils")
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion  # noqa: E402

GiB = sp.GiB
MMAP = {None, "mmap"}
PLENTY = (100 * GiB, 128 * GiB)  # Strix Halo 128 GB, idle
TIGHT = (20 * GiB, 128 * GiB)  # 20 GiB free < 32 GiB headroom before the load

requires_backend = pytest.mark.skipif(
    not sp._supports_backend(modeling_utils.safe_open),
    reason = "installed safetensors has no backend keyword",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    monkeypatch.delenv(sp.FADVISE_ENV, raising = False)
    host_pinning.reset_cache()
    yield
    host_pinning.reset_cache()
    assert sp._depth == 0 and sp._original is None and not sp._active_rules()
    assert not sp._diffusers_original


def _gpu(
    monkeypatch,
    vendor,
    integrated,
    memory = PLENTY,
    platform = "linux",
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", "7.13" if vendor == "rocm" else None, raising = False)
    # is_rocm_torch also reads the build tag, which a real ROCm wheel carries in __version__
    monkeypatch.setattr(
        torch,
        "__version__",
        TorchVersion("2.11.0+rocm7.13.0" if vendor == "rocm" else "2.11.0+cu130"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(host_pinning, "device_integrated_status", lambda device = None: integrated)
    monkeypatch.setattr(sp, "host_memory", lambda: memory)


def _spy(monkeypatch):
    seen = []
    real_tf = modeling_utils.safe_open

    @functools.wraps(real_tf)
    def tf_spy(*args, **kwargs):
        seen.append(("transformers", kwargs.get("backend")))
        return real_tf(*args, **kwargs)

    real_st = safetensors.safe_open

    @functools.wraps(real_st)
    def st_spy(*args, **kwargs):
        seen.append(("diffusers", kwargs.get("backend")))
        return real_st(*args, **kwargs)

    monkeypatch.setattr(modeling_utils, "safe_open", tf_spy)
    monkeypatch.setattr(safetensors, "safe_open", st_spy)
    monkeypatch.setattr(safetensors.torch, "safe_open", st_spy)
    return seen


def _expected(name):
    base = torch.arange(96 * 80, dtype = torch.float32).reshape(96, 80)
    return (base * (2 if name == "b" else 1)).to(torch.bfloat16)


def _snapshot(tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    out = {}
    for comp in ("text_encoder", "transformer"):
        d = tmp_path / comp
        d.mkdir()
        path = d / "model.safetensors"
        save_file({"a": _expected("a"), "b": _expected("b")}, str(path))
        out[comp] = path
    return out


def _load(shards):
    with modeling_utils.safe_open(
        str(shards["text_encoder"]), framework = "pt", device = "cpu"
    ) as handle:
        te = {k: handle.get_tensor(k) for k in handle.keys()}
    return te, mlu.load_state_dict(str(shards["transformer"]))


def _run(monkeypatch, tmp_path):
    shards = _snapshot(tmp_path)
    seen = _spy(monkeypatch)
    with diffusion._converted_shards_pread(torch.bfloat16, None, "cuda"):
        te, dit = _load(shards)
    for got in (te, dit):
        assert set(got) == {"a", "b"}
        assert all(torch.equal(got[k], _expected(k)) for k in got)  # checksums unchanged
    return {backend for _, backend in seen}


@pytest.mark.parametrize(
    "vendor, integrated, memory, env, reader",
    [
        (
            "cuda",
            True,
            PLENTY,
            None,
            "pread",
        ),  # integrated NVIDIA (#19905): unchanged, memory never read
        ("cuda", True, TIGHT, None, "pread"),
        ("cuda", True, PLENTY, "0", "mmap"),
        ("rocm", True, PLENTY, None, "mmap"),  # ROCm APU, memory free: speed unchanged
        ("rocm", True, TIGHT, None, "pread"),  # ROCm APU under pressure
        ("rocm", True, None, None, "mmap"),  # memory unreadable: keep mmap
        ("rocm", True, PLENTY, "pread", "pread"),  # override forces
        ("rocm", True, PLENTY, "1", "pread"),
        ("rocm", True, TIGHT, "mmap", "mmap"),  # override vetoes
        ("rocm", True, TIGHT, "0", "mmap"),
        ("rocm", True, TIGHT, "auto", "pread"),
        ("rocm", False, TIGHT, None, "mmap"),  # discrete Radeon / Instinct
        ("cuda", False, TIGHT, None, "mmap"),  # discrete NVIDIA
        ("rocm", None, TIGHT, None, "mmap"),  # unclassifiable
    ],
)
@requires_backend
def test_reader_decision_table(monkeypatch, tmp_path, vendor, integrated, memory, env, reader):
    _gpu(monkeypatch, vendor, integrated, memory)
    if env is not None:
        monkeypatch.setenv(sp.PREAD_ENV, env)
    backends = _run(monkeypatch, tmp_path)
    if reader == "pread":
        assert backends == {"pread"}, backends
    else:
        assert backends <= MMAP, backends


@requires_backend
def test_rocm_pressure_logs_reason_and_drops_pages(monkeypatch, tmp_path):
    _gpu(monkeypatch, "rocm", True, TIGHT)
    before = sp.fadvise_calls
    sp.last_reader_decision = None
    assert _run(monkeypatch, tmp_path) == {"pread"}
    decision = sp.last_reader_decision
    assert decision["reader"] == "pread" and decision["mode"] == "auto"
    assert "< headroom 32.0 GiB" in decision["reason"]
    # Windows has no posix_fadvise: pread still runs there, the page drop is skipped by design.
    expected_drops = 2 if hasattr(os, "posix_fadvise") else 0  # the diffusers shard's two tensors
    assert sp.fadvise_calls - before == expected_drops


@requires_backend
def test_rocm_plenty_logs_mmap_without_drops(monkeypatch, tmp_path):
    _gpu(monkeypatch, "rocm", True, PLENTY)
    before = sp.fadvise_calls
    sp.last_reader_decision = None
    _run(monkeypatch, tmp_path)
    assert sp.last_reader_decision["reader"] == "mmap"
    assert ">= headroom" in sp.last_reader_decision["reason"]
    assert sp.fadvise_calls == before


@requires_backend
def test_fadvise_kill_switch(monkeypatch, tmp_path):
    _gpu(monkeypatch, "rocm", True, TIGHT)
    monkeypatch.setenv(sp.FADVISE_ENV, "0")
    before = sp.fadvise_calls
    assert _run(monkeypatch, tmp_path) == {"pread"}
    assert sp.fadvise_calls == before


@requires_backend
def test_integrated_nvidia_never_drops_pages(monkeypatch, tmp_path):
    _gpu(monkeypatch, "cuda", True, TIGHT)
    before = sp.fadvise_calls
    assert _run(monkeypatch, tmp_path) == {"pread"}
    assert sp.fadvise_calls == before


@pytest.mark.parametrize(
    "available, total, load, pread",
    [
        (100, 128, 24.3, False),  # 75.7 left >= 32 (25% of 128)
        (56, 128, 24.3, True),  # 31.7 left < 32
        (57, 128, 24.3, False),
        (40, 64, 24.3, True),  # 15.7 < 16 (floor beats 25% of 64)
        (41, 64, 24.3, False),
        (30, 32, 4, False),  # small load on a small box: 26 >= 16
    ],
)
def test_pressure_rule(available, total, load, pread):
    use, reason = sp.pressure_decision(int(load * GiB), (int(available * GiB), int(total * GiB)))
    assert use is pread, reason


def test_pressure_unreadable_memory_keeps_mmap(monkeypatch):
    monkeypatch.setattr(sp, "host_memory", lambda: None)
    assert sp.pressure_decision(GiB) == (False, "host memory unreadable")


@pytest.mark.parametrize(
    "value, mode",
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("pread", "pread"),
        ("1", "pread"),
        ("on", "pread"),
        ("mmap", "mmap"),
        ("0", "mmap"),
        ("off", "mmap"),
        ("bogus", "auto"),
    ],
)
def test_reader_mode(monkeypatch, value, mode):
    if value is not None:
        monkeypatch.setenv(sp.PREAD_ENV, value)
    assert sp.reader_mode() == mode


def test_checkpoint_bytes_counts_the_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    shards = _snapshot(snap)
    total = sum(p.stat().st_size for p in shards.values())
    assert sp.checkpoint_bytes(shards["transformer"]) == total
    lone = tmp_path / "plain" / "lone"
    lone.mkdir(parents = True)
    save_file({"a": _expected("a")}, str(lone / "m.safetensors"))
    assert sp.checkpoint_bytes(lone / "m.safetensors") == (lone / "m.safetensors").stat().st_size


def test_host_memory_reads_meminfo():
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux /proc/meminfo")
    available, total = sp.host_memory()
    assert 0 < available <= total


@pytest.mark.parametrize(
    "hip, status, expected",
    [("7.13", True, True), ("7.13", False, False), ("7.13", None, False), (None, True, False)],
)
def test_rocm_is_integrated(monkeypatch, hip, status, expected):
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(
        torch, "__version__", TorchVersion("2.11.0+rocm7.13.0" if hip else "2.11.0+cu130")
    )
    monkeypatch.setattr(host_pinning, "device_integrated_status", lambda device = None: status)
    assert host_pinning.rocm_is_integrated("cuda") is expected
    if hip:
        assert host_pinning.cuda_is_integrated("cuda") is False  # NVIDIA rule never sees a ROCm APU


def _fp32_shard(tmp_path, name = "text_encoder"):
    d = tmp_path / name
    d.mkdir()
    path = d / "model.safetensors"
    save_file({"a": _expected("a").float(), "b": _expected("b").float()}, str(path))
    return path


def _open_through_transformers(seen, path):
    with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu") as handle:
        got = {k: handle.get_tensor(k) for k in handle.keys()}
    assert all(torch.equal(got[k], _expected(k).float()) for k in got)
    return {backend for _, backend in seen}


@requires_backend
def test_rocm_ample_memory_converting_fp32_shard_stays_mapped(monkeypatch, tmp_path):
    # the gate's mmap must not fall through to the conversion predicates
    _gpu(monkeypatch, "rocm", True, PLENTY)
    sp.last_reader_decision = None
    path = _fp32_shard(tmp_path)
    seen = _spy(monkeypatch)
    with diffusion._converted_shards_pread(torch.bfloat16, None, "cuda"):
        backends = _open_through_transformers(seen, path)
    assert backends <= MMAP, backends
    assert sp.last_reader_decision["reader"] == "mmap"


@requires_backend
def test_rocm_ample_memory_fp32_video_shard_stays_mapped(monkeypatch, tmp_path):
    _gpu(monkeypatch, "rocm", True, PLENTY)
    sp.last_reader_decision = None
    path = _fp32_shard(tmp_path, "transformer")
    seen = _spy(monkeypatch)
    with sp.pread_safetensors(integrated_cuda = False, integrated_rocm = True):
        backends = _open_through_transformers(seen, path)
    assert backends <= MMAP, backends
    assert sp.last_reader_decision["reader"] == "mmap"


@requires_backend
@pytest.mark.parametrize("video", [False, True])
def test_rocm_pressure_converting_and_video_fp32_shards_read_pread(monkeypatch, tmp_path, video):
    _gpu(monkeypatch, "rocm", True, TIGHT)
    sp.last_reader_decision = None
    path = _fp32_shard(tmp_path)
    seen = _spy(monkeypatch)
    scope = (
        sp.pread_safetensors(integrated_rocm = True)
        if video
        else diffusion._converted_shards_pread(torch.bfloat16, None, "cuda")
    )
    with scope:
        backends = _open_through_transformers(seen, path)
    assert backends == {"pread"} and sp.last_reader_decision["reader"] == "pread"


@requires_backend
@pytest.mark.parametrize("video", [False, True])
def test_windows_discrete_conversion_rules_unchanged(monkeypatch, tmp_path, video):
    _gpu(monkeypatch, "cuda", False, PLENTY, platform = "win32")
    path = _fp32_shard(tmp_path)
    seen = _spy(monkeypatch)
    scope = (
        sp.pread_safetensors()
        if video
        else diffusion._converted_shards_pread(torch.bfloat16, None, "cuda")
    )
    with scope:
        backends = _open_through_transformers(seen, path)
    assert backends == {"pread"}


@requires_backend
def test_integrated_nvidia_converting_fp32_shard_still_pread(monkeypatch, tmp_path):
    _gpu(monkeypatch, "cuda", True, PLENTY)
    path = _fp32_shard(tmp_path)
    seen = _spy(monkeypatch)
    with diffusion._converted_shards_pread(torch.bfloat16, None, "cuda"):
        backends = _open_through_transformers(seen, path)
    assert backends == {"pread"}


@requires_backend
def test_windows_rocm_ample_memory_keeps_the_commit_rule(monkeypatch, tmp_path):
    # Windows converting loads read through pread for commit (#14513), as on main; the reason says so
    _gpu(monkeypatch, "rocm", True, PLENTY, platform = "win32")
    sp.last_reader_decision = None
    path = _fp32_shard(tmp_path)
    seen = _spy(monkeypatch)
    with diffusion._converted_shards_pread(torch.bfloat16, None, "cuda"):
        backends = _open_through_transformers(seen, path)
    assert backends == {"pread"}
    assert "Windows converting loads still read through pread" in sp.last_reader_decision["reason"]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integrated NVIDIA targets read every safetensors shard (transformers and diffusers) through ``pread``; CPU only."""

from __future__ import annotations

import ast
import functools
import sys
import types
from pathlib import Path

import pytest
import torch

import core.inference.safetensors_pread as sp
from core.inference import host_pinning

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as modeling_utils  # noqa: E402

# imported before any test stubs the device: diffusers reads properties the stub lacks at import
mlu = pytest.importorskip("diffusers.models.model_loading_utils")
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
MMAP = {None, "mmap"}

requires_backend = pytest.mark.skipif(
    not sp._supports_backend(modeling_utils.safe_open),
    reason = "installed safetensors has no backend keyword",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    host_pinning.reset_cache()
    yield
    host_pinning.reset_cache()
    assert sp._depth == 0 and sp._original is None and not sp._active_rules()
    assert not getattr(sp, "_diffusers_original", {})


def _stub_cuda(
    monkeypatch,
    integrated = (True,),
    rocm = False,
):
    def props(index = None):
        i = 0 if index is None else int(getattr(index, "index", index) or 0)
        flag = 1 if integrated[i] else 0
        return types.SimpleNamespace(
            is_integrated = flag, integrated = flag, name = "Fake GPU", gcnArchName = ""
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(integrated))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(torch.version, "hip", "6.4.0" if rocm else None)


@pytest.mark.parametrize(
    "integrated, rocm, device, expected",
    [
        ((True,), False, "cuda", True),
        ((True,), False, "cuda:0", True),
        ((True,), False, None, True),  # the current device
        ((False,), False, "cuda", False),  # discrete
        ((True,), True, "cuda", False),  # ROCm APU: classified apart
        ((True,), False, "cpu", False),
        ((True,), False, "mps", False),
        ((False, True), False, "cuda:1", True),  # the load's own card, not the current one
        ((False, True), False, "cuda:0", False),
    ],
)
def test_cuda_is_integrated(monkeypatch, integrated, rocm, device, expected):
    _stub_cuda(monkeypatch, integrated, rocm)
    assert host_pinning.cuda_is_integrated(device) is expected


def test_cuda_is_integrated_never_raises(monkeypatch):
    def boom(*a):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    monkeypatch.setattr(torch.version, "hip", None)
    assert host_pinning.cuda_is_integrated("cuda") is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert host_pinning.cuda_is_integrated("cuda") is False


@pytest.mark.parametrize(
    "platform, env, integrated_cuda, wanted",
    [
        ("linux", None, True, True),
        ("win32", None, True, True),
        ("linux", None, False, False),  # unchanged off an integrated NVIDIA GPU
        ("win32", None, False, True),
        ("linux", "0", True, False),  # explicit off wins
        ("win32", "0", True, False),
    ],
)
def test_pread_wanted_integrated_cuda(monkeypatch, platform, env, integrated_cuda, wanted):
    monkeypatch.setattr(sys, "platform", platform)
    if env is not None:
        monkeypatch.setenv(sp.PREAD_ENV, env)
    assert sp.pread_wanted(integrated_cuda) is wanted


def _spy_backends(monkeypatch):
    """(library, backend) per shard open."""
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


def _shards(tmp_path, dtype = torch.bfloat16):
    out = {}
    for comp in ("text_encoder", "transformer"):
        d = tmp_path / comp
        d.mkdir()
        path = d / "model.safetensors"
        save_file(
            {"w": torch.arange(64 * 64, dtype = torch.float32).reshape(64, 64).to(dtype)}, str(path)
        )
        out[comp] = path
    return out


def _load_both(shards):
    with modeling_utils.safe_open(
        str(shards["text_encoder"]), framework = "pt", device = "cpu"
    ) as handle:
        te = handle.get_slice("w")[...]
    dit = mlu.load_state_dict(str(shards["transformer"]))
    return te, dit["w"]


def _image_scope(
    device,
    dtype = torch.bfloat16,
    small_host = None,
):
    return diffusion._converted_shards_pread(dtype, small_host, device)


@requires_backend
@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_integrated_cuda_reads_every_shard_through_pread(monkeypatch, tmp_path, platform):
    shards = _shards(tmp_path)
    monkeypatch.setattr(sys, "platform", platform)
    _stub_cuda(monkeypatch, (True,))
    seen = _spy_backends(monkeypatch)
    with _image_scope("cuda") as engaged:
        assert engaged is True
        te, dit = _load_both(shards)
    assert seen == [("transformers", "pread"), ("diffusers", "pread")]
    expected = torch.arange(64 * 64, dtype = torch.float32).reshape(64, 64).to(torch.bfloat16)
    assert torch.equal(te, expected) and torch.equal(dit, expected)
    assert mlu.load_state_dict.__module__ == "diffusers.models.model_loading_utils"
    assert not hasattr(mlu.load_state_dict, "__wrapped__")


@requires_backend
@pytest.mark.parametrize(
    "platform, integrated, rocm, env",
    [
        ("linux", (False,), False, None),  # discrete NVIDIA
        ("win32", (False,), False, None),
        ("linux", (True,), True, None),  # ROCm APU: left to its pending timing rerun
        ("linux", (True,), False, "0"),  # explicit off wins
        ("win32", (True,), False, "0"),
    ],
)
def test_other_targets_keep_their_backend(monkeypatch, tmp_path, platform, integrated, rocm, env):
    shards = _shards(tmp_path)
    monkeypatch.setattr(sys, "platform", platform)
    if env is not None:
        monkeypatch.setenv(sp.PREAD_ENV, env)
    _stub_cuda(monkeypatch, integrated, rocm)
    seen = _spy_backends(monkeypatch)
    with _image_scope("cuda"):
        _load_both(shards)
    # a same-dtype bf16 shard stays mapped on Windows too, and diffusers' own read is untouched
    assert [lib for lib, _ in seen] == ["transformers", "diffusers"]
    assert {backend for _, backend in seen} <= MMAP


@requires_backend
@pytest.mark.parametrize("device", ["cpu", "mps", None])
def test_non_cuda_targets_unchanged(monkeypatch, tmp_path, device):
    shards = _shards(tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_cuda(monkeypatch, (True,))
    seen = _spy_backends(monkeypatch)
    with _image_scope(device) as engaged:
        assert engaged is False
        _load_both(shards)
    assert {backend for _, backend in seen} <= MMAP


@requires_backend
def test_small_host_kept_component_stays_mapped(monkeypatch, tmp_path):
    shards = _shards(tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_cuda(monkeypatch, (True,))
    seen = _spy_backends(monkeypatch)
    small_host = types.SimpleNamespace(engaged = True, storage_dtypes = {"transformer": "bfloat16"})
    with _image_scope("cuda", small_host = small_host):
        _load_both(shards)
    assert seen[0] == ("transformers", "pread")
    assert seen[1][0] == "diffusers" and seen[1][1] in MMAP and len(seen) == 2


@requires_backend
def test_diffusers_shard_untouched_on_windows_discrete_converting_load(monkeypatch, tmp_path):
    # the Windows converting rule stays transformers-only: a diffusers shard is never rerouted off an integrated GPU
    shards = _shards(tmp_path, dtype = torch.float32)
    monkeypatch.setattr(sys, "platform", "win32")
    _stub_cuda(monkeypatch, (False,))
    seen = _spy_backends(monkeypatch)
    with _image_scope("cuda", dtype = torch.bfloat16):
        _load_both(shards)
    assert seen[0] == ("transformers", "pread")
    assert seen[1][0] == "diffusers" and seen[1][1] in MMAP and len(seen) == 2


def test_safetensors_without_backend_stays_mapped(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_cuda(monkeypatch, (True,))

    def old_safe_open(
        filename,
        framework = None,
        device = "cpu",
    ):
        raise AssertionError("not reached")

    monkeypatch.setattr(modeling_utils, "safe_open", old_safe_open)
    before = mlu.load_state_dict
    with _image_scope("cuda") as engaged:
        assert engaged is False
        assert modeling_utils.safe_open is old_safe_open
        assert mlu.load_state_dict is before


def test_diffusers_without_backend_stays_mapped(monkeypatch):
    # transformers' safe_open takes the keyword, the safetensors diffusers reaches does not: diffusers is not patched
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_cuda(monkeypatch, (True,))

    def old_safe_open(
        filename,
        framework = None,
        device = "cpu",
    ):
        raise AssertionError("not reached")

    monkeypatch.setattr(safetensors, "safe_open", old_safe_open)
    before = mlu.load_state_dict
    with _image_scope("cuda"):
        assert mlu.load_state_dict is before


def test_every_media_scope_passes_its_target():
    sites = {}
    for rel in (
        "core/inference/diffusion.py",
        "core/inference/video.py",
        "core/inference/diffusion_hidream.py",
    ):
        tree = ast.parse((BACKEND / rel).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) in (
                "pread_safetensors",
                "_converted_shards_pread",
            ):
                if isinstance(node.func, ast.Name) and node.func.id == "_converted_shards_pread":
                    ok = len(node.args) == 3
                else:
                    ok = any(k.arg == "integrated_cuda" for k in node.keywords)
                sites.setdefault(rel, []).append(ok)
    assert set(sites) == {
        "core/inference/diffusion.py",
        "core/inference/video.py",
        "core/inference/diffusion_hidream.py",
    }
    assert all(all(v) for v in sites.values()), sites


@requires_backend
@pytest.mark.parametrize(
    "call, rerouted",
    [
        ({}, True),
        ({"dduf_entries": None, "disable_mmap": False}, True),  # diffusers 0.40's own call
        ({"disable_mmap": True}, False),  # diffusers' read-whole-file mode
        ({"dduf_entries": {"x": 1}}, False),  # a DDUF archive entry
        ({"map_location": "meta"}, False),
    ],
)
def test_diffusers_wrapper_reroutes_only_the_plain_shard_read(
    monkeypatch, tmp_path, call, rerouted
):
    # diffusers 0.40 orders the parameters (file, dduf_entries, disable_mmap, map_location), 0.42 drops dduf_entries
    path = _shards(tmp_path)["transformer"]
    stock = []

    def load_state_dict_040(
        checkpoint_file,
        dduf_entries = None,
        disable_mmap = False,
        map_location = "cpu",
    ):
        stock.append(checkpoint_file)
        return {"stock": True}

    seen = _spy_backends(monkeypatch)
    wrapped = sp._with_pread_state_dict(load_state_dict_040)
    monkeypatch.setattr(sp._local, "rules", ((None, frozenset(), True, True),), raising = False)
    got = wrapped(str(path), **call)
    monkeypatch.setattr(sp._local, "rules", (), raising = False)
    if rerouted:
        assert seen == [("diffusers", "pread")] and not stock and set(got) == {"w"}
    else:
        assert seen == [] and stock == [str(path)] and got == {"stock": True}

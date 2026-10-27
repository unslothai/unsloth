# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A native cache write in flight must not keep a module's old weight storage alive once the residency code moves
the weights (group offload release swaps ``param.data`` back to host), and must still write the exact weights."""

from __future__ import annotations

import os
import sys
import threading
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
safetensors_torch = pytest.importorskip("safetensors.torch")
pytest.importorskip("accelerate")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "_shared"))

from real_accelerator import has_real_cuda  # noqa: E402

from core.inference import diffusion_native_cache as NC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402

FAMILY = "wan2.1-t2v-1.3b"
CONFIG = dict(
    patch_size = (1, 2, 2),
    num_attention_heads = 2,
    attention_head_dim = 256,
    in_channels = 4,
    out_channels = 4,
    text_dim = 64,
    freq_dim = 32,
    ffn_dim = 1024,
    num_layers = 2,
    cross_attn_norm = True,
    qk_norm = "rms_norm_across_heads",
    eps = 1e-6,
    rope_max_seq_len = 64,
)


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    return root


@pytest.fixture
def base_dir(tmp_path):
    root = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    torch.manual_seed(3)
    diffusers.WanTransformer3DModel(**CONFIG).save_pretrained(
        str(root / "transformer"), safe_serialization = True
    )
    return str(root)


def _native(base_dir):
    model = diffusers.WanTransformer3DModel.from_pretrained(
        base_dir, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    view = types.SimpleNamespace(transformer = model)
    assert (
        tq._quantize_native(
            view,
            "int8",
            family = FAMILY,
            min_features = tq.DEFAULT_MIN_LINEAR_FEATURES,
            act_int8 = True,
            logger = None,
        )
        == "int8"
    )
    return model


def _live_tensors(model):
    out = [p for _n, p in model.named_parameters()]
    out += [b for _n, b in model.named_buffers() if b is not None]
    return out


def _paused_save(monkeypatch, model, base_dir, during):
    """Run ``NC.save`` on a thread, paused inside the file write; ``during(tensors)`` runs while it is paused."""
    entered, resume = threading.Event(), threading.Event()
    seen: dict = {}
    original = C.write_streaming_safetensors

    def hold(path, tensors, *a, **kw):
        seen["tensors"] = tensors
        entered.set()
        assert resume.wait(60)
        return original(path, tensors, *a, **kw)

    monkeypatch.setattr(C, "write_streaming_safetensors", hold)
    fam = detect_video_family("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", override = FAMILY)
    spec = NC.native_spec("int8", FAMILY, act_int8 = True)
    result: dict = {}
    thread = threading.Thread(
        target = lambda: result.setdefault(
            "path", NC.save(model, fam, spec, base_repo = base_dir, component = "transformer")
        )
    )
    thread.start()
    assert entered.wait(60), "the cache writer never reached the file write"
    try:
        during(seen.pop("tensors"))
    finally:
        resume.set()
        thread.join(60)
    assert not thread.is_alive()
    return result.get("path")


def _entry_tensors(path):
    path = str(path)
    if os.path.isdir(path):
        path = os.path.join(path, C.FILE_NAME)
    return safetensors_torch.load_file(path)


def _as_if_on_the_accelerator(monkeypatch):
    # CPU stands in for resident accelerator storage here.
    monkeypatch.setattr(NC, "_moves_with_offload", lambda tensor: True)


def test_writer_holds_no_old_storage_after_the_weights_move(monkeypatch, base_dir, cache_root):
    _as_if_on_the_accelerator(monkeypatch)
    model = _native(base_dir)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    old_ptrs: set = set()
    old_storage: list = []

    def offload(tensors):
        # what release_resident_groups does to a resident group: param.data swapped to another copy
        for t in _live_tensors(model):
            old_storage.append(t.data)
            old_ptrs.add(t.data_ptr())
            t.data = t.data.clone()
        held = [v for v in tensors.values() if isinstance(v, torch.Tensor)]
        stale = [v for v in held if v.data_ptr() in old_ptrs]
        assert (
            not stale
        ), f"{len(stale)} tensors in the in-flight write still alias the moved storage"

    path = _paused_save(monkeypatch, model, base_dir, offload)
    assert path is not None
    got = _entry_tensors(path)
    assert set(got) == set(expected)
    for k, v in expected.items():
        assert got[k].dtype == v.dtype and torch.equal(got[k], v), k


def test_a_weight_that_changes_shape_mid_write_leaves_no_entry(monkeypatch, base_dir, cache_root):
    _as_if_on_the_accelerator(monkeypatch)
    model = _native(base_dir)

    def reshape(_tensors):
        p = next(p for _n, p in model.named_parameters() if p.dim() == 2)
        p.data = p.data[:1].clone()

    assert _paused_save(monkeypatch, model, base_dir, reshape) is None
    leftovers = [f for f in cache_root.rglob("*") if f.is_file() and C.FILE_NAME in f.name]
    assert leftovers == [], leftovers


def test_a_host_weight_is_written_as_captured_even_if_it_is_onloaded_mid_write(
    monkeypatch, base_dir, cache_root
):
    """A host-resident weight keeps its captured copy, never an in-flight onload swapped in mid-write."""
    model = _native(base_dir)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def onload(_tensors):
        for t in _live_tensors(model):
            if t.is_floating_point():
                t.data = torch.full_like(t.data, 7)

    path = _paused_save(monkeypatch, model, base_dir, onload)
    assert path is not None
    got = _entry_tensors(path)
    for k, v in expected.items():
        assert torch.equal(got[k], v), k


def test_an_onload_right_after_the_sync_does_not_reach_the_written_bytes(
    monkeypatch, base_dir, cache_root
):
    _as_if_on_the_accelerator(monkeypatch)
    model = _native(base_dir)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    by_ptr = {t.data_ptr(): t for t in _live_tensors(model) if t.is_floating_point()}

    def sync_then_onload(value):
        # a stream onload swapping in a buffer whose copy is still in flight, just after the writer's sync
        owner = by_ptr.pop(value.data_ptr(), None)
        if owner is not None:
            owner.data = torch.full_like(owner.data, 7)

    monkeypatch.setattr(NC, "_wait_for_transfers", sync_then_onload)
    fam = detect_video_family("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", override = FAMILY)
    spec = NC.native_spec("int8", FAMILY, act_int8 = True)
    path = NC.save(model, fam, spec, base_repo = base_dir, component = "transformer")
    assert path is not None
    got = _entry_tensors(path)
    for k, v in expected.items():
        assert torch.equal(got[k], v), k


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
def test_cuda_storage_is_reclaimed_while_the_write_is_in_flight(monkeypatch, base_dir):
    baseline = torch.cuda.memory_allocated()
    model = _native(base_dir).to("cuda")
    expected = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.cuda.synchronize()
    seen: dict = {}

    def offload(_tensors):
        before = torch.cuda.memory_allocated()
        for t in _live_tensors(model):
            t.data = t.data.cpu()
        _tensors = None  # noqa: F841 - only the writer's own reference may remain
        torch.cuda.synchronize()
        seen["before"], seen["after"] = before, torch.cuda.memory_allocated()

    path = _paused_save(monkeypatch, model, base_dir, offload)
    assert seen["before"] > baseline
    held = seen["after"] - baseline
    assert held <= 0, f"{held} CUDA bytes still held by the in-flight write"
    got = _entry_tensors(path)
    for k, v in expected.items():
        assert torch.equal(got[k], v), k


def test_a_hook_built_view_of_a_weight_keeps_its_captured_tensor(monkeypatch, base_dir, cache_root):
    _as_if_on_the_accelerator(monkeypatch)
    model = _native(base_dir)
    key = next(
        k for k, v in model.state_dict().items() if v.dim() == 2 and v.shape[0] == v.shape[1] > 1
    )

    def hook(module, state, prefix, local_metadata):
        state[key] = state[key].t()  # same storage, same shape, different view
        return state

    model._register_state_dict_hook(hook)
    want = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fam = detect_video_family("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", override = FAMILY)
    spec = NC.native_spec("int8", FAMILY, act_int8 = True)
    path = NC.save(model, fam, spec, base_repo = base_dir, component = "transformer")
    assert path is not None
    got = _entry_tensors(path)
    assert torch.equal(got[key], want[key])

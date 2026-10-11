# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The LTX-2.3 direct checkpoint read: same bytes, shapes and dtypes as the safetensors loader, in chunks."""

import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from core.inference import video_ltx2  # noqa: E402


def _write_checkpoint(path):
    gen = torch.Generator().manual_seed(0)
    tensors = {
        "model.diffusion_model.a.weight": torch.randn(37, 41, generator = gen).to(torch.bfloat16),
        "model.diffusion_model.scale_shift_table": torch.randn(9, 13, generator = gen),
        "vae.decoder.conv.weight": torch.randn(3, 5, 7, generator = gen).to(torch.float16),
        "model.diffusion_model.empty": torch.empty(0, 4),
        "model.diffusion_model.ids": torch.arange(11, dtype = torch.int64),
        "model.diffusion_model.mask": torch.tensor([True, False, True]),
        "model.diffusion_model.scalar": torch.tensor(3.5),
    }
    safetensors_torch.save_file(tensors, str(path))
    return tensors


def _assert_same(got, want):
    assert set(got) == set(want)
    for name, tensor in want.items():
        assert got[name].dtype == tensor.dtype, name
        assert tuple(got[name].shape) == tuple(tensor.shape), name
        assert torch.equal(got[name].cpu(), tensor), name


@pytest.mark.parametrize("chunk_bytes", [7, 64, 1 << 20])
def test_read_matches_the_safetensors_loader_on_the_host(tmp_path, chunk_bytes):
    path = tmp_path / "ltx.safetensors"
    want = _write_checkpoint(path)
    got = video_ltx2.read_safetensors_to_device(path, "cpu", chunk_bytes = chunk_bytes)
    _assert_same(got, want)


def test_read_keeps_only_the_requested_tensors(tmp_path):
    path = tmp_path / "ltx.safetensors"
    want = _write_checkpoint(path)
    got = video_ltx2.read_safetensors_to_device(
        path, "cpu", keep = lambda key: video_ltx2._checkpoint_group(key)[0] != "dit", chunk_bytes = 16
    )
    _assert_same(got, {"vae.decoder.conv.weight": want["vae.decoder.conv.weight"]})


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("buffers, threads", [(1, 1), (2, 3), (16, 8)])
def test_read_matches_the_safetensors_loader_on_cuda(tmp_path, buffers, threads):
    # Tiny chunks and a small ring make every slot get reused many times while uploads are in flight.
    path = tmp_path / "ltx.safetensors"
    want = _write_checkpoint(path)
    got = video_ltx2.read_safetensors_to_device(
        path, "cuda", chunk_bytes = 32, buffers = buffers, threads = threads
    )
    assert all(t.device.type == "cuda" for t in got.values())
    _assert_same(got, want)


def test_unmapped_dtype_returns_none(tmp_path, monkeypatch):
    path = tmp_path / "ltx.safetensors"
    _write_checkpoint(path)
    monkeypatch.delitem(video_ltx2._SAFETENSORS_DTYPES, "BF16")
    assert video_ltx2.read_safetensors_to_device(path, "cpu") is None


def test_full_checkpoint_reader_declines_gguf_and_falls_back_on_errors(tmp_path, monkeypatch):
    assert video_ltx2.load_checkpoint_to_device(tmp_path / "ltx.gguf", "cpu") is None
    path = tmp_path / "ltx.safetensors"
    _write_checkpoint(path)

    def _boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(video_ltx2, "read_safetensors_to_device", _boom)
    assert video_ltx2.load_checkpoint_to_device(path, "cpu") is None
    assert video_ltx2._load_checkpoint_without_dit_to_device(path, "cpu") is None


def test_direct_load_device_is_cuda_only_and_has_a_kill_switch(monkeypatch):
    monkeypatch.delenv(video_ltx2.DIRECT_LOAD_ENV, raising = False)
    assert video_ltx2.direct_load_device("cpu") is None
    assert video_ltx2.direct_load_device("mps") is None
    assert video_ltx2.direct_load_device(None) is None
    if torch.cuda.is_available():
        assert video_ltx2.direct_load_device("cuda") == torch.device(
            "cuda", torch.cuda.current_device()
        )
        assert video_ltx2.direct_load_device("cuda:0") == torch.device("cuda", 0)
    for off in ("0", "off", "false", "no"):
        monkeypatch.setenv(video_ltx2.DIRECT_LOAD_ENV, off)
        assert video_ltx2.direct_load_device("cuda:0") is None


class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))


class _Pipe:
    def __init__(self):
        self.text_encoder = _Module()
        self.transformer = _Module()
        self.moved = []

    @property
    def components(self):
        return {"text_encoder": self.text_encoder, "transformer": self.transformer}

    def to(self, device):
        self.moved.append(str(device))
        return self


def _plan(policy, **extra):
    return types.SimpleNamespace(offload_policy = policy, reasons = (), **extra)


def test_direct_loaded_modules_go_back_to_the_host_only_when_the_plan_moved_on(monkeypatch):
    from core.inference import video as video_mod

    monkeypatch.setattr(video_mod, "_video_plan_label", lambda plan: plan.offload_policy)
    # Still resident: nothing moves.
    pipe = _Pipe()
    video_mod._return_direct_loaded_modules(pipe, _plan("none"))
    assert pipe.moved == []
    pipe = _Pipe()
    video_mod._return_direct_loaded_modules(
        pipe, _plan("group", stream_transformer = False, stream_text_encoders = False)
    )
    assert pipe.moved == []
    # The DiT is offloaded now: the whole pipeline returns to the host, where offload starts from.
    pipe = _Pipe()
    video_mod._return_direct_loaded_modules(pipe, _plan("group", stream_transformer = True))
    assert pipe.moved == ["cpu"]
    pipe = _Pipe()
    video_mod._return_direct_loaded_modules(pipe, _plan("model"))
    assert pipe.moved == ["cpu"]
    # The DiT stays but the encoders stream: only an encoder already on a device goes back (here: on the host).
    pipe = _Pipe()
    calls = []
    pipe.text_encoder.to = lambda device: calls.append(device)
    video_mod._return_direct_loaded_modules(
        pipe, _plan("group", stream_transformer = False, stream_text_encoders = True)
    )
    assert pipe.moved == [] and calls == []

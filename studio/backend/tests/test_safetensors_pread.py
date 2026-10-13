# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``safetensors_pread.py``: on Windows a media load reads transformers' safetensors shards with the ``pread`` backend
instead of torch's copy-on-write file mapping (#9977), and every other platform keeps the byte-identical mmap path.
CPU only."""

from __future__ import annotations

import inspect
import sys
import threading

import pytest
import torch

import core.inference.safetensors_pread as sp

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as modeling_utils  # noqa: E402

# transformers < 5.6 passes no backend (None), later releases pass "mmap" off Windows.
MMAP = {None, "mmap"}

requires_backend = pytest.mark.skipif(
    not sp._supports_backend(modeling_utils.safe_open),
    reason = "installed safetensors has no backend keyword",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    yield
    assert sp._depth == 0 and sp._original is None


def _record_backends(monkeypatch):
    """Spy under the patch: the backend keyword each transformers safe_open call carried (None = default mmap)."""
    seen = []
    real = modeling_utils.safe_open

    def spy(*args, **kwargs):
        seen.append(kwargs.get("backend"))
        return real(*args, **kwargs)

    spy.__text_signature__ = real.__text_signature__
    monkeypatch.setattr(modeling_utils, "safe_open", spy)
    return seen


@pytest.mark.parametrize(
    "platform, env, wanted",
    [
        ("win32", None, True),
        ("linux", None, False),
        ("darwin", None, False),
        ("win32", "0", False),
        ("linux", "1", True),
    ],
)
def test_pread_wanted(monkeypatch, platform, env, wanted):
    monkeypatch.setattr(sys, "platform", platform)
    if env is not None:
        monkeypatch.setenv(sp.PREAD_ENV, env)
    assert sp.pread_wanted() is wanted


def test_off_windows_nothing_is_patched(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    before = modeling_utils.safe_open
    with sp.pread_safetensors() as engaged:
        assert engaged is False
        assert modeling_utils.safe_open is before


def test_old_safetensors_without_backend_is_left_alone(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def old_safe_open(
        filename,
        framework,
        device = "cpu",
    ):
        raise AssertionError("not called")

    monkeypatch.setattr(modeling_utils, "safe_open", old_safe_open)
    with sp.pread_safetensors() as engaged:
        assert engaged is False
        assert modeling_utils.safe_open is old_safe_open


def _save_llama(path, dtype):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(dtype).save_pretrained(path, max_shard_size = "20KB")
    assert len(list(path.glob("*.safetensors"))) > 1
    return LlamaForCausalLM


@requires_backend
def test_windows_fp32_load_reads_with_pread_and_matches_mmap(monkeypatch, tmp_path):
    """The reported path: an fp32 checkpoint loaded at bf16 through transformers' from_pretrained."""
    model_cls = _save_llama(tmp_path, torch.float32)
    seen = _record_backends(monkeypatch)
    dense = model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) <= MMAP

    monkeypatch.setenv(sp.PREAD_ENV, "1")
    seen.clear()
    spy = modeling_utils.safe_open
    with sp.pread_safetensors() as engaged:
        assert engaged is True
        assert modeling_utils.safe_open.__wrapped__ is spy
        loaded = model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) == {"pread"}
    assert modeling_utils.safe_open is spy

    expected = dense.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, expected[name]), name


@requires_backend
def test_windows_bf16_shards_keep_mmap(monkeypatch, tmp_path):
    """A bf16 shard's mapping is the loaded weight, so it stays on the default backend."""
    model_cls = _save_llama(tmp_path, torch.bfloat16)
    monkeypatch.setenv(sp.PREAD_ENV, "1")
    seen = _record_backends(monkeypatch)
    with sp.pread_safetensors() as engaged:
        assert engaged is True
        model_cls.from_pretrained(tmp_path, dtype = torch.bfloat16)
    assert seen and set(seen) <= MMAP


@requires_backend
@pytest.mark.parametrize(
    "dtype, passed, expected",
    [
        (torch.float32, None, "pread"),
        (torch.float32, "mmap", "pread"),
        (torch.float32, "pread", "pread"),
        (torch.bfloat16, None, None),
        (torch.bfloat16, "mmap", "mmap"),
    ],
)
def test_backend_choice(monkeypatch, tmp_path, dtype, passed, expected):
    from safetensors.torch import save_file

    path = tmp_path / "shard.safetensors"
    save_file({"w": torch.zeros(64, 64, dtype = dtype)}, str(path))
    monkeypatch.setattr(sys, "platform", "win32")
    seen = _record_backends(monkeypatch)
    kwargs = {} if passed is None else {"backend": passed}
    with sp.pread_safetensors():
        with modeling_utils.safe_open(str(path), framework = "pt", device = "cpu", **kwargs) as handle:
            assert torch.equal(handle.get_slice("w")[...], torch.zeros(64, 64, dtype = dtype))
    assert seen == [expected]


@requires_backend
def test_nested_and_concurrent_loads_restore_once(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    original = modeling_utils.safe_open
    inside = threading.Event()
    release = threading.Event()

    def other_load():
        with sp.pread_safetensors() as engaged:
            assert engaged
            inside.set()
            release.wait(5)

    worker = threading.Thread(target = other_load)
    with sp.pread_safetensors() as outer:
        assert outer
        patched = modeling_utils.safe_open
        worker.start()
        assert inside.wait(5)
        with sp.pread_safetensors() as inner:
            assert inner and modeling_utils.safe_open is patched
        assert modeling_utils.safe_open is patched
    assert modeling_utils.safe_open is patched
    release.set()
    worker.join(5)
    assert modeling_utils.safe_open is original


def test_video_load_runs_inside_the_scope():
    from core.inference.video import VideoBackend

    source = inspect.getsource(VideoBackend._run_load)
    scope = source.index("with pread_safetensors():")
    assert scope < source.index("self.load_pipeline(**kwargs)", scope)

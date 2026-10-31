# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deferred prequant cache write: same bytes as the sync write, nothing published early, fallbacks, kill switch."""

from __future__ import annotations

import gc
import hashlib
import inspect
import sys
import threading
import time
import types
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("safetensors")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_diffusion_prequant_cache as T  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402

scheme = T.scheme


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    monkeypatch.delenv(C.ENV_MODE, raising = False)
    monkeypatch.delenv(C.ENV_MAX_GB, raising = False)
    monkeypatch.delenv(C.ENV_ASYNC, raising = False)
    C._HEADER_MEMO.clear()
    getattr(C, "_UNREACHABLE", {}).clear()
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (1 << 40, 1 << 39))
    yield root
    assert C.cancel_background_writes(timeout = 30.0)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _leftovers(root: Path) -> list:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.name != C._LOCK_NAME and not p.name.endswith(".writing")
    ]


def _two_builds(tmp_path, scheme_name):
    base, model = T._write_base(tmp_path)
    other = T.Tiny().to(torch.bfloat16)
    other.load_state_dict(model.state_dict())
    return base, T._quantize(model, scheme_name), T._quantize(other, scheme_name)


def _sync_reference(tmp_path, monkeypatch, base, model) -> str:
    monkeypatch.setenv(C.ENV_DIR, str(tmp_path / "ref_cache"))
    path = C.save_runtime_quant(model, T._fam(base), base_repo = str(base))
    assert isinstance(path, str)
    return path


def test_deferred_write_publishes_nothing_until_it_runs_then_matches_the_sync_bytes(
    env, tmp_path, scheme, monkeypatch
):
    base, model, twin = _two_builds(tmp_path, scheme)
    ref = _sync_reference(tmp_path, monkeypatch, base, twin)
    monkeypatch.setenv(C.ENV_DIR, str(env))
    fam = T._fam(base)
    pending = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)
    assert isinstance(pending, C.PendingWrite)
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    assert _leftovers(env) == []
    assert C._writer_active(env, pending.key)
    assert pending.start()
    assert not pending.start()  # once
    assert pending.wait(60.0) and pending.done()
    assert pending.result and Path(pending.result).is_file()
    assert _sha(pending.result) == _sha(ref)
    assert C.lookup(fam, scheme, base_repo = str(base)) == pending.result
    assert not C._writer_active(env, pending.key)
    assert pending._job is None and pending._stack is None


def test_the_snapshot_is_independent_of_the_live_module(env, tmp_path, scheme, monkeypatch):
    """Unload / reload / placement replace or drop the module's tensors after the hand-off: the write is unchanged."""
    base, model, twin = _two_builds(tmp_path, scheme)
    ref = _sync_reference(tmp_path, monkeypatch, base, twin)
    monkeypatch.setenv(C.ENV_DIR, str(env))
    fam = T._fam(base)
    pending = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)
    assert isinstance(pending, C.PendingWrite)
    with torch.no_grad():
        for _name, p in model.named_parameters():
            if type(p) in (torch.nn.Parameter, torch.Tensor):
                p.data = torch.full_like(p.data, 7)
    model.pad_token = torch.nn.Parameter(torch.zeros(1, 512, dtype = torch.bfloat16))
    del model
    gc.collect()
    assert pending.start() and pending.wait(60.0)
    assert _sha(pending.result) == _sha(ref)


def test_the_snapshot_holds_host_tensors_only_and_frees_each_once_written(
    env, tmp_path, scheme, monkeypatch
):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    pending = C.save_runtime_quant(model, T._fam(base), base_repo = str(base), defer = True)
    job = pending._job
    assert job.release
    assert not any(isinstance(t, C._FlatRef) for t in job.flat.values())
    assert all(t.device.type == "cpu" for t in job.flat.values())
    assert all(type(t) is torch.Tensor or type(t) is torch.nn.Parameter for t in job.flat.values())
    refs = [weakref.ref(t) for t in job.flat.values()]
    del model
    gc.collect()
    seen_during = []
    real = C.write_streaming_safetensors

    def spy(
        path,
        tensors,
        metadata = None,
        cancel_check = None,
        hasher = None,
        release = False,
    ):
        def check():
            seen_during.append(len(tensors))
            if cancel_check is not None:
                cancel_check()

        return real(path, tensors, metadata, cancel_check = check, hasher = hasher, release = release)

    monkeypatch.setattr(C, "write_streaming_safetensors", spy)
    job = None
    assert pending.start() and pending.wait(60.0)
    assert pending.result
    assert seen_during == sorted(seen_during, reverse = True) and seen_during[0] > seen_during[-1]
    gc.collect()
    assert all(r() is None for r in refs)


def test_release_writes_the_same_file_as_a_plain_write(tmp_path):
    torch.manual_seed(0)
    tensors = {
        "a": torch.randn(4, 8, dtype = torch.bfloat16),
        "b": torch.randint(-128, 127, (16, 3), dtype = torch.int8),
        "c": torch.randn(5, dtype = torch.float32),
        "e": torch.empty(0, dtype = torch.float32),
    }
    keep = dict(tensors)
    C.write_streaming_safetensors(tmp_path / "x", dict(tensors), {"m": "1"})
    drained = dict(tensors)
    C.write_streaming_safetensors(tmp_path / "y", drained, {"m": "1"}, release = True)
    assert drained == {}
    assert _sha(tmp_path / "x") == _sha(tmp_path / "y")
    assert keep.keys() == tensors.keys()


def test_cancel_mid_write_unlinks_the_temp_and_publishes_nothing(
    env, tmp_path, scheme, monkeypatch
):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    fam = T._fam(base)
    pending = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)
    entered = threading.Event()
    gate = threading.Event()
    real = C.write_streaming_safetensors

    def slow(
        path,
        tensors,
        metadata = None,
        cancel_check = None,
        hasher = None,
        release = False,
    ):
        def check():
            entered.set()
            gate.wait(10.0)
            cancel_check()

        return real(path, tensors, metadata, cancel_check = check, hasher = hasher, release = release)

    monkeypatch.setattr(C, "write_streaming_safetensors", slow)
    assert pending.start()
    assert entered.wait(10.0)
    assert list(env.rglob(f"*{C._TEMP_SUFFIX}"))  # the write really began
    pending.cancel()
    gate.set()
    assert C.cancel_background_writes(timeout = 30.0)
    assert pending.result is None
    assert _leftovers(env) == []
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    monkeypatch.setattr(C, "write_streaming_safetensors", real)
    assert isinstance(C.save_runtime_quant(model, fam, base_repo = str(base)), str)


def test_a_failed_load_discards_the_prepared_write(env, tmp_path, scheme):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    fam = T._fam(base)
    pending = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)
    pending.discard()
    assert pending.done() and not pending.start()
    assert _leftovers(env) == [] and not C._writer_active(env, pending.key)
    assert C.lookup(fam, scheme, base_repo = str(base)) is None


def test_a_writer_thread_that_cannot_start_releases_the_snapshot(
    env, tmp_path, scheme, monkeypatch
):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    fam = T._fam(base)
    pending = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)

    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(C.threading.Thread, "start", refuse)
    assert pending.start() is False
    assert pending.done() and pending not in C._BACKGROUND
    assert _leftovers(env) == [] and not C._writer_active(env, pending.key)
    assert C.lookup(fam, scheme, base_repo = str(base)) is None


def test_fallbacks_write_synchronously(env, tmp_path, scheme, monkeypatch):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    fam = T._fam(base)
    monkeypatch.setenv(C.ENV_ASYNC, "0")
    out = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)
    assert isinstance(out, str) and Path(out).is_file()
    monkeypatch.delenv(C.ENV_ASYNC)
    job = C._WriteJob(
        "k", env / "k", {"w": torch.zeros(1 << 20, dtype = torch.int8)}, {}, {}, 1 << 20, 0.0, None
    )
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (8 << 30, 1 << 30))
    assert "host RAM short" in C._defer_blocker(job)
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (None, None))
    assert "unknown" in C._defer_blocker(job)
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (1 << 40, 1 << 39))
    assert C._defer_blocker(job) is None
    on_meta = C._WriteJob(
        "k", env / "k", {"w": torch.empty(4, device = "meta")}, {}, {}, 4, 0.0, None
    )
    assert "not the host" in C._defer_blocker(on_meta)


def test_one_background_write_at_a_time(env, tmp_path, scheme, monkeypatch):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    first = C.save_runtime_quant(model, T._fam(base), base_repo = str(base), defer = True)
    assert isinstance(first, C.PendingWrite)
    job = C._WriteJob("k", env / "k", {"w": torch.zeros(4)}, {}, {}, 16, 0.0, None)
    assert "another background" in C._defer_blocker(job)
    first.discard()
    assert C._defer_blocker(job) is None


def test_disk_short_still_skips_before_any_hand_off(env, tmp_path, scheme, monkeypatch):
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    monkeypatch.setattr(
        C.shutil, "disk_usage", lambda p: types.SimpleNamespace(total = 1, used = 1, free = 0)
    )
    monkeypatch.setattr(C, "evict", lambda **kw: None)
    assert C.save_runtime_quant(model, T._fam(base), base_repo = str(base), defer = True) is None
    assert not C._BACKGROUND


def test_an_unload_lets_a_running_write_finish(env, tmp_path, scheme, monkeypatch):
    """An unload waits for the write (no lost entry, no host RAM shared with the next load) rather than cancelling."""
    base, model = T._write_base(tmp_path)
    T._quantize(model, scheme)
    fam = T._fam(base)
    pending = C.save_runtime_quant(model, fam, base_repo = str(base), defer = True)
    entered = threading.Event()
    real = C.write_streaming_safetensors

    def slow(
        path,
        tensors,
        metadata = None,
        cancel_check = None,
        hasher = None,
        release = False,
    ):
        def check():
            entered.set()
            time.sleep(0.05)
            cancel_check()

        return real(path, tensors, metadata, cancel_check = check, hasher = hasher, release = release)

    monkeypatch.setattr(C, "write_streaming_safetensors", slow)
    del model
    assert pending.start() and entered.wait(10.0)
    assert C.finish_background_writes(timeout = 60.0)
    assert pending.result and C.lookup(fam, scheme, base_repo = str(base)) == pending.result
    assert not C._BACKGROUND


def test_the_pipeline_load_defers_and_unload_finishes_the_write():
    from core.inference import diffusion as D

    src = inspect.getsource(D.DiffusionBackend.load_pipeline)
    call = src.index("pending_cache_write = self._save_pipeline_quant_to_cache(")
    assert "defer = True" in src[call : call + 700]
    fin = src.index("pending_cache_write.start()")
    # Started only from the finally, after publication, and discarded when the load did not commit.
    assert src.index("state_committed = True") < fin
    assert "pending_cache_write.discard()" in src[fin : fin + 300]
    assert "defer = defer" in inspect.getsource(D.DiffusionBackend._save_pipeline_quant_to_cache)
    unload = inspect.getsource(D.DiffusionBackend._unload_locked)
    assert unload.index("finish_background_writes(") < unload.index("state = self._state")


def _mapped(path) -> bool:
    return any(r[5] == str(path) for r in C._file_regions())


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason = "needs /proc/self/maps")
def test_a_view_of_a_mapped_shard_is_re_read_not_held(env, tmp_path):
    """Mmapped-shard views become file slices: no mapping pinned, same bytes written."""
    src = tmp_path / "shard.bin"
    torch.manual_seed(0)
    payload = torch.randn(1 << 18, dtype = torch.bfloat16)
    src.write_bytes(b"\0" * 4096 + payload.view(torch.uint8).numpy().tobytes())
    mapped = torch.from_file(
        str(src), shared = False, size = 4096 + payload.numel() * 2, dtype = torch.uint8
    )
    view = mapped[4096:].view(torch.bfloat16).reshape(512, 512)
    own = torch.randint(-128, 127, (256, 64), dtype = torch.int8)
    assert _mapped(src)
    ref = tmp_path / "ref.safetensors"
    C.write_streaming_safetensors(ref, {"a": view, "q": own}, {"m": "1"})

    job = C._WriteJob("k", env / "k", {"a": view, "q": own}, {"m": "1"}, {}, 0, 0.0, None)
    job.resolve()
    assert isinstance(job.flat["a"], C._FileSlice) and job.flat["q"] is own
    assert job.held_bytes == own.numel() and job.pinned_map_bytes == 0
    assert job.file_bytes == view.numel() * 2
    del mapped, view
    gc.collect()
    assert not _mapped(src)  # the snapshot does not keep the shard mapped
    out = tmp_path / "out.safetensors"
    C.write_streaming_safetensors(out, job.flat, job.header, release = True)
    job.close()
    assert _sha(out) == _sha(ref)


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason = "needs /proc/self/maps")
def test_a_shard_changed_under_a_file_slice_fails_the_write(env, tmp_path):
    src = tmp_path / "shard.bin"
    src.write_bytes(bytes(range(256)) * 64)
    mapped = torch.from_file(str(src), shared = False, size = 256 * 64, dtype = torch.uint8)
    job = C._WriteJob("k", env / "k", {"a": mapped[256:]}, {}, {}, 0, 0.0, None)
    job.resolve()
    assert isinstance(job.flat["a"], C._FileSlice)
    with open(src, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(OSError):
        C.write_streaming_safetensors(tmp_path / "o", job.flat, {})
    job.close()


def test_the_gate_counts_what_the_snapshot_keeps_alive(env, monkeypatch):
    job = C._WriteJob(
        "k", env / "k", {"w": torch.zeros(1 << 20, dtype = torch.int8)}, {}, {}, 1 << 30, 0.0, None
    )
    # Total artifact 1 GiB, but the snapshot holds 1 MiB: 4.5 GiB free clears max(4 GiB, 10%) + 1 MiB.
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (16 << 30, (4 << 30) + (512 << 20)))
    assert C._defer_blocker(job) is None and job.held_bytes == 1 << 20
    job2 = C._WriteJob(
        "k", env / "k", {"w": torch.zeros(1 << 20, dtype = torch.int8)}, {}, {}, 1 << 20, 0.0, None
    )
    job2.resolve()
    job2.pinned_map_bytes = 1 << 30  # a view that could not be sliced pins its whole mapping
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (16 << 30, (4 << 30) + (512 << 20)))
    assert "keeps 1.0 GiB alive" in C._defer_blocker(job2)


def test_a_cgroup_limit_narrows_the_host_reading(monkeypatch):
    monkeypatch.setattr(C, "_cgroup_memory_bytes", lambda: (20 << 30, 3 << 30))
    total, avail = C._host_memory_bytes()
    assert total <= 20 << 30 and avail <= 3 << 30

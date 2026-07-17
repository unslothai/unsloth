# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from types import SimpleNamespace

import core.inference.llama_cpp as llama_cpp
from core.inference.llama_cpp import LlamaCppBackend


def _resume_backend(tmp_path, n_slots = 1):
    backend = LlamaCppBackend()
    backend._healthy = True
    # No-op lifecycle methods so the atexit cleanup can kill the fake quietly.
    backend._process = SimpleNamespace(
        poll = lambda: None,
        terminate = lambda: None,
        wait = lambda *a, **k: 0,
        kill = lambda: None,
        pid = 0,
    )
    backend._port = 8081
    backend._slot_save_dir = str(tmp_path)
    backend._slot_save_binary = ("/bin/llama-server", 1)
    (tmp_path / "model.gguf").write_bytes(b"gguf")
    backend._gguf_path = str(tmp_path / "model.gguf")
    backend._effective_parallel_slots = n_slots
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0
    return backend


def _fake_disk(monkeypatch, free = 1 << 40):
    monkeypatch.setattr(llama_cpp.shutil, "disk_usage", lambda _p: SimpleNamespace(free = free))


class _Resp:
    def __init__(
        self,
        status_code = 200,
        body = None,
    ):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


def test_save_returns_none_when_slot_save_disabled(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    backend._slot_save_dir = None
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_skipped_when_prompt_cache_disabled(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    backend._prompt_cache_disabled = True
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_skipped_when_insufficient_free_disk(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    backend._estimate_kv_cache_bytes = lambda *a, **k: 1 << 40
    _fake_disk(monkeypatch, free = 1 << 20)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_collects_manifest_across_slots(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 2)
    _fake_disk(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["params"], kwargs["json"]))
        return _Resp(200, {"n_saved": 40, "n_written": 100})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    manifest = backend.save_slots_for_resume()
    assert manifest is not None
    assert manifest["dir"] == str(tmp_path)
    assert manifest["binary"] == ("/bin/llama-server", 1)
    assert manifest["gguf"] == str(tmp_path / "model.gguf")
    st = os.stat(manifest["gguf"])
    assert manifest["gguf_stat"] == ((st.st_size, st.st_mtime_ns),)
    assert manifest["launch"] == backend._slot_launch_fingerprint()
    assert [e["id"] for e in manifest["slots"]] == [0, 1]
    assert all(e["n_saved"] == 40 for e in manifest["slots"])
    assert [c[1] for c in calls] == [{"action": "save"}] * 2
    assert "/slots/0" in calls[0][0] and "/slots/1" in calls[1][0]


def test_save_unlinks_empty_slot_and_returns_none(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    _fake_disk(monkeypatch)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"")
        return _Resp(200, {"n_saved": 0, "n_written": 0})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert list(tmp_path.glob("resume-*.bin")) == []  # empty-slot file removed


def test_save_cap_breach_discards_all_files(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 2)
    _fake_disk(monkeypatch)
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 150)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"x" * 100)
        return _Resp(200, {"n_saved": 40, "n_written": 100})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None  # 200 bytes > 150 cap
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_transport_error_aborts_remaining_slots(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 3)
    _fake_disk(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        raise OSError("connection refused")

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert len(calls) == 1  # no retries against a dead server


def test_save_transport_error_unlinks_partial_file(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    _fake_disk(monkeypatch)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"partial")
        raise OSError("timed out")

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_gguf_file_identity_covers_split_shards(tmp_path):
    backend = _resume_backend(tmp_path)
    first = tmp_path / "m-00001-of-00002.gguf"
    second = tmp_path / "m-00002-of-00002.gguf"
    first.write_bytes(b"a")
    second.write_bytes(b"bb")

    before = backend._gguf_file_identity(str(first))
    st1, st2 = os.stat(first), os.stat(second)
    assert before == ((st1.st_size, st1.st_mtime_ns), (st2.st_size, st2.st_mtime_ns))

    second.write_bytes(b"rewritten")  # sibling changes, primary untouched
    after = backend._gguf_file_identity(str(first))
    assert after is not None and after != before
    assert after[0] == before[0]  # primary shard unchanged

    second.unlink()
    assert backend._gguf_file_identity(str(first)) is None  # missing shard


def test_save_skipped_when_user_disabled_prompt_cache(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    backend._extra_args = ["--no-cache-prompt"]
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_stops_writing_once_cap_exceeded(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 3)
    _fake_disk(monkeypatch)
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 150)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"x" * 100)
        return _Resp(200, {"n_saved": 1, "n_written": 100})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert len(calls) == 2  # cap blown after slot 1; slot 2 never attempted
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_aborts_between_slots_when_no_longer_idle(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 3)
    _fake_disk(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return _Resp(200, {"n_saved": 5, "n_written": 10})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    aborts = iter([False, True, True])
    manifest = backend.save_slots_for_resume(should_abort = lambda: next(aborts))
    assert len(calls) == 1  # slots 1 and 2 skipped
    assert manifest is not None
    assert [e["id"] for e in manifest["slots"]] == [0]


def test_save_non_200_slot_is_skipped_but_others_kept(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 2)
    _fake_disk(monkeypatch)

    def fake_post(url, **kwargs):
        if "/slots/0" in url:
            return _Resp(500)
        return _Resp(200, {"n_saved": 5, "n_written": 10})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    manifest = backend.save_slots_for_resume()
    assert manifest is not None
    assert [e["id"] for e in manifest["slots"]] == [1]


def test_restore_posts_each_slot_and_tolerates_failures(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["params"], kwargs["json"]))
        return _Resp(500 if "/slots/0" in url else 200, {"n_restored": 5})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    backend.restore_slots_for_resume(
        {
            "slots": [
                {"id": 0, "filename": "resume-a-slot0.bin", "n_saved": 5},
                {"id": 1, "filename": "resume-a-slot1.bin", "n_saved": 5},
            ]
        }
    )
    assert [c[1] for c in calls] == [{"action": "restore"}] * 2
    assert calls[0][2] == {"filename": "resume-a-slot0.bin"}


def test_restore_transport_error_stops_early(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        raise OSError("connection refused")

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    backend.restore_slots_for_resume(
        {"slots": [{"id": 0, "filename": "a.bin"}, {"id": 1, "filename": "b.bin"}]}
    )
    assert len(calls) == 1

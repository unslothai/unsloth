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


def test_fingerprint_tracks_lora_sidecar_rewrite(tmp_path):
    backend = _resume_backend(tmp_path)
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"v1")
    backend._extra_args = ["--lora", str(adapter)]

    before = backend._slot_launch_fingerprint()
    adapter.write_bytes(b"v2-different")  # re-exported adapter, same path
    assert backend._slot_launch_fingerprint() != before

    backend._extra_args = [f"--lora={adapter}"]
    assert backend._sidecar_weight_files() == [str(adapter)]
    backend._extra_args = ["--lora-scaled", str(adapter), "0.5"]
    assert backend._sidecar_weight_files() == [str(adapter)]
    backend._extra_args = ["--control-vector", str(adapter), "--threads", "4"]
    assert backend._sidecar_weight_files() == [str(adapter)]


def test_sidecar_files_parse_csv_and_colon_scale(tmp_path):
    backend = _resume_backend(tmp_path)
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"

    backend._extra_args = ["--lora", f"{a},{b}"]
    files = backend._sidecar_weight_files()
    assert str(a) in files and str(b) in files

    backend._extra_args = ["--lora-scaled", f"{a}:0.5"]
    assert str(a) in backend._sidecar_weight_files()

    backend._extra_args = ["--control-vector-scaled", f"{a}:1.0,{b}:2.0"]
    files = backend._sidecar_weight_files()
    assert str(a) in files and str(b) in files

    # Windows drive letter must not be mistaken for a scale separator.
    backend._extra_args = ["--lora-scaled", "C:\\adapters\\a.gguf:0.75"]
    assert "C:\\adapters\\a.gguf" in backend._sidecar_weight_files()
    backend._extra_args = ["--lora", "C:\\adapters\\a.gguf"]
    assert backend._sidecar_weight_files() == ["C:\\adapters\\a.gguf"]


def test_fingerprint_tracks_colon_scaled_adapter_rewrite(tmp_path):
    backend = _resume_backend(tmp_path)
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"v1")
    backend._extra_args = ["--lora-scaled", f"{adapter}:0.5"]

    before = backend._slot_launch_fingerprint()
    adapter.write_bytes(b"v2-different")  # re-exported adapter, same path
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_effective_context_length(tmp_path):
    backend = _resume_backend(tmp_path)
    backend._effective_context_length = 8192

    before = backend._slot_launch_fingerprint()
    backend._effective_context_length = 4096  # auto-fit landed smaller on reload
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_effective_projector_state(tmp_path):
    backend = _resume_backend(tmp_path)
    backend._load_mmproj = True

    before = backend._slot_launch_fingerprint()
    backend._load_mmproj = False
    assert backend._slot_launch_fingerprint() != before


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


def test_save_skipped_when_env_disables_prompt_cache(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    monkeypatch.setenv("LLAMA_ARG_CACHE_PROMPT", "0")
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None
    monkeypatch.delenv("LLAMA_ARG_CACHE_PROMPT")
    monkeypatch.setenv("LLAMA_ARG_NO_CACHE_PROMPT", "1")  # legacy negative form
    assert backend.save_slots_for_resume() is None


def test_explicit_cache_prompt_flag_overrides_env(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    monkeypatch.setenv("LLAMA_ARG_CACHE_PROMPT", "0")
    backend._extra_args = ["--cache-prompt"]  # CLI wins over env in llama.cpp
    _fake_disk(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: _Resp(200, {"n_saved": 1, "n_written": 1}),
        raising = False,
    )
    assert backend.save_slots_for_resume() is not None


def test_user_cache_prompt_overrides_studio_no_cache_flag(monkeypatch, tmp_path):
    # User extras follow Studio's flags, so an explicit --cache-prompt wins.
    backend = _resume_backend(tmp_path)
    backend._prompt_cache_disabled = True
    backend._extra_args = ["--cache-prompt"]
    _fake_disk(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: _Resp(200, {"n_saved": 1, "n_written": 1}),
        raising = False,
    )
    assert backend.save_slots_for_resume() is not None
    # Last flag wins when both appear in extras.
    backend._extra_args = ["--cache-prompt", "--no-cache-prompt"]
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


def test_save_deletes_orphan_on_malformed_response(monkeypatch, tmp_path):
    # A 200 that writes a file but returns a non-numeric counter must be cleaned
    # up like any other save failure, not left orphaned holding chat KV.
    backend = _resume_backend(tmp_path)
    _fake_disk(monkeypatch)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"chat-kv")
        return _Resp(200, {"n_saved": "not-an-int"})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_deletes_orphan_on_non_dict_response(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    _fake_disk(monkeypatch)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"chat-kv")
        return _Resp(200, ["unexpected", "list"])

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_cap_uses_actual_file_size_not_reported_bytes(monkeypatch, tmp_path):
    # A binary under-reporting n_written must not slip past the disk cap: the
    # cap is enforced against the bytes actually on disk.
    backend = _resume_backend(tmp_path)
    _fake_disk(monkeypatch)
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 150)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"x" * 200)
        return _Resp(200, {"n_saved": 5, "n_written": 1})  # under-reported

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None  # 200 real bytes > 150 cap
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_skipped_when_estimate_exceeds_cap(monkeypatch, tmp_path):
    # An estimate over the cap skips before writing any slot at all.
    backend = _resume_backend(tmp_path)
    backend._estimate_kv_cache_bytes = lambda *a, **k: 1 << 40
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 1 << 20)
    _fake_disk(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_skipped_when_model_file_changed_since_load(monkeypatch, tmp_path):
    # The GGUF/sidecars were swapped on disk after the server loaded them, so the
    # live KV belongs to the old weights: refuse to persist it (no POST at all).
    backend = _resume_backend(tmp_path)
    backend._slot_loaded_identity = ((("stale", 0),), ())  # != current identity
    _fake_disk(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_proceeds_when_load_identity_matches(monkeypatch, tmp_path):
    # Matching load-time snapshot: the save runs normally.
    backend = _resume_backend(tmp_path)
    backend._slot_loaded_identity = (
        backend._gguf_file_identity(backend._gguf_path),
        backend._slot_launch_fingerprint(),
    )
    _fake_disk(monkeypatch)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"kv")
        return _Resp(200, {"n_saved": 5, "n_written": 2})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    manifest = backend.save_slots_for_resume()
    assert manifest is not None
    assert [e["id"] for e in manifest["slots"]] == [0]


def test_save_skipped_when_estimate_unavailable_and_low_disk(monkeypatch, tmp_path):
    # A 0 estimate means metadata was insufficient, not a zero-byte cache: the save
    # must demand room for the whole cap, not just 1 GiB, on a low-disk host.
    backend = _resume_backend(tmp_path)
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0  # metadata unavailable
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 8 << 30)  # 8 GiB cap
    _fake_disk(monkeypatch, free = 2 << 30)  # 2 GiB free < 8 + 1 GiB required
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None

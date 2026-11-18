# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from types import SimpleNamespace

import core.inference.llama_cpp as llama_cpp
from core.inference.llama_cpp import LlamaCppBackend


def _resume_backend(tmp_path, n_slots = 1):
    backend = LlamaCppBackend()
    backend._healthy = True
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
    _fake_disk(monkeypatch)
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
    _fake_disk(monkeypatch)
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
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_cap_breach_discards_all_files(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 2)
    _fake_disk(monkeypatch)
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 150)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"x" * 100)
        return _Resp(200, {"n_saved": 40, "n_written": 100})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
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
    assert len(calls) == 1


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
    adapter.write_bytes(b"v2-different")
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
    adapter.write_bytes(b"v2-different")
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_effective_context_length(tmp_path):
    backend = _resume_backend(tmp_path)
    backend._effective_context_length = 8192

    before = backend._slot_launch_fingerprint()
    backend._effective_context_length = 4096
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_swa_full_mode(tmp_path):
    backend = _resume_backend(tmp_path)
    before = backend._slot_launch_fingerprint()
    backend._swa_full = True
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_unified_cache_mode(tmp_path):
    backend = _resume_backend(tmp_path)
    before = backend._slot_launch_fingerprint()
    backend._kv_cache_unified = True
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_flash_attention_mode(tmp_path):
    backend = _resume_backend(tmp_path)
    before = backend._slot_launch_fingerprint()
    backend._flash_attn_enabled = False
    assert backend._slot_launch_fingerprint() != before


def test_fingerprint_tracks_effective_cache_types(tmp_path):
    backend = _resume_backend(tmp_path)
    before = backend._slot_launch_fingerprint()
    backend._effective_cache_types = ("f32", "f16")
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

    second.write_bytes(b"rewritten")
    after = backend._gguf_file_identity(str(first))
    assert after is not None and after != before
    assert after[0] == before[0]

    second.unlink()
    assert backend._gguf_file_identity(str(first)) is None


def test_save_skipped_when_user_disabled_prompt_cache(monkeypatch, tmp_path):
    _fake_disk(monkeypatch)
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
    _fake_disk(monkeypatch)
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
    monkeypatch.setenv("LLAMA_ARG_NO_CACHE_PROMPT", "1")
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
    # User extras follow Unsloth's flags, so an explicit --cache-prompt wins.
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
    assert len(calls) == 2
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
    assert len(calls) == 1
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
    # The disk cap is enforced on actual bytes on disk, not the reported n_written.
    backend = _resume_backend(tmp_path)
    _fake_disk(monkeypatch)
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 150)

    def fake_post(url, **kwargs):
        (tmp_path / kwargs["json"]["filename"]).write_bytes(b"x" * 200)
        return _Resp(200, {"n_saved": 5, "n_written": 1})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    assert backend.save_slots_for_resume() is None
    assert list(tmp_path.glob("resume-*.bin")) == []


def test_save_skipped_when_estimate_exceeds_cap(monkeypatch, tmp_path):
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


def test_save_estimate_uses_total_context_and_active_cache_settings(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path, n_slots = 4)
    backend._effective_context_length = 8192
    backend._kv_cache_context_total = 32768
    backend._sliding_window = 4096
    backend._swa_full = True
    backend._flash_attn_enabled = False
    backend._effective_cache_types = ("f32", "f16")
    calls = []

    def estimate(ctx, cache_type, **kwargs):
        calls.append((ctx, cache_type, kwargs))
        return 0

    backend._estimate_kv_cache_bytes = estimate
    _fake_disk(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: _Resp(200, {"n_saved": 1, "n_written": 1}),
        raising = False,
    )

    assert backend.save_slots_for_resume() is not None
    assert calls == [
        (
            32768,
            "f32",
            {
                "n_parallel": 4,
                "swa_full": True,
                "kv_unified": False,
                "n_ubatch": 512,
                "flash_attn": False,
            },
        )
    ]


def test_compact_swa_slot_save_is_skipped(monkeypatch, tmp_path):
    _fake_disk(monkeypatch)
    backend = _resume_backend(tmp_path)
    backend._sliding_window = 4096
    backend._kv_key_length = 256
    backend._kv_value_length = 256
    backend._swa_full = False
    backend._estimate_kv_cache_bytes = lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_window_without_kv_dims_still_saves(monkeypatch, tmp_path):
    # phi3 has a window but no key/value length and runs non-SWA in llama.cpp.
    _fake_disk(monkeypatch)
    backend = _resume_backend(tmp_path)
    backend._sliding_window = 262144
    backend._kv_key_length = None
    backend._kv_value_length = None
    backend._swa_full = False
    posted = []
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: posted.append(a)
        or SimpleNamespace(status_code = 200, json = lambda: {"filename": "slot.bin"}),
        raising = False,
    )
    backend.save_slots_for_resume()
    assert posted


def test_save_skipped_when_model_file_changed_since_load(monkeypatch, tmp_path):
    backend = _resume_backend(tmp_path)
    backend._slot_loaded_identity = ((("stale", 0),), ())
    _fake_disk(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def test_save_proceeds_when_load_identity_matches(monkeypatch, tmp_path):
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
    # A 0 estimate means unknown, so the save must require room for the whole cap.
    backend = _resume_backend(tmp_path)
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0
    monkeypatch.setattr(llama_cpp, "_SLOT_SAVE_MAX_BYTES", 8 << 30)
    _fake_disk(monkeypatch, free = 2 << 30)
    monkeypatch.setattr(
        llama_cpp.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError),
        raising = False,
    )
    assert backend.save_slots_for_resume() is None


def _ctx_backend(
    tmp_path,
    *,
    n_ctx,
    n_slots = 1,
    unified = False,
):
    backend = _resume_backend(tmp_path, n_slots = n_slots)
    backend._slot_long_rope = False
    backend._requested_n_ctx = n_ctx
    backend._effective_context_length = n_ctx if unified else n_ctx // n_slots
    backend._kv_cache_unified = unified
    return backend


def _manifest_from(backend, *, n_saved):
    return {
        "dir": backend._slot_save_dir,
        "binary": backend._slot_save_binary,
        "gguf": backend._gguf_path,
        "gguf_stat": backend._gguf_file_identity(backend._gguf_path),
        "launch": backend._slot_launch_fingerprint(),
        "long_rope": backend._slot_long_rope,
        "slots": [
            {"id": slot, "filename": f"resume-x-slot{slot}.bin", "n_saved": n_saved}
            for slot in range(backend.effective_parallel_slots)
        ],
    }


def test_slot_capacity_is_the_context_split_across_slots_unless_unified(tmp_path):
    assert _ctx_backend(tmp_path, n_ctx = 8192, n_slots = 4)._slot_context_capacity() == 2048
    assert (
        _ctx_backend(tmp_path, n_ctx = 8192, n_slots = 4, unified = True)._slot_context_capacity() == 8192
    )
    backend = _ctx_backend(tmp_path, n_ctx = 8192, n_slots = 4)
    backend._effective_context_length = None
    assert backend._slot_context_capacity() == 2048
    backend._requested_n_ctx = None
    assert backend._slot_context_capacity() is None


def test_an_identical_launch_fits(tmp_path):
    backend = _ctx_backend(tmp_path, n_ctx = 8192)
    assert backend.slot_resume_fits(_manifest_from(backend, n_saved = 100)) is True


def test_a_launch_that_only_raised_the_context_fits_when_the_slots_hold_the_tokens(tmp_path):
    saved = _manifest_from(_ctx_backend(tmp_path, n_ctx = 8192), n_saved = 8000)
    assert _ctx_backend(tmp_path, n_ctx = 32768).slot_resume_fits(saved) is True
    assert _ctx_backend(tmp_path, n_ctx = 16384, n_slots = 4).slot_resume_fits(saved) is False
    split = _manifest_from(_ctx_backend(tmp_path, n_ctx = 16384, n_slots = 4), n_saved = 4000)
    assert _ctx_backend(tmp_path, n_ctx = 32768, n_slots = 4).slot_resume_fits(split) is True
    assert _ctx_backend(tmp_path, n_ctx = 4096).slot_resume_fits(saved) is False
    assert (
        _ctx_backend(tmp_path, n_ctx = 8192, n_slots = 4, unified = True).slot_resume_fits(saved) is False
    ), "unified is a fingerprint field of its own; only the context may differ"


def test_reload_uses_engine_sequence_length_even_when_props_caps_the_window(tmp_path):
    def launch(requested, allocated):
        backend = _ctx_backend(tmp_path, n_ctx = requested)
        backend._stdout_lines = [
            "0.00.013.211 I print_info: n_ctx_orig_yarn       = 2048",
            f"0.00.013.722 I llama_context: n_ctx_seq             = {allocated}",
        ]
        backend._query_server_n_ctx = lambda: 2048
        backend._reconcile_effective_ctx_with_server(requested)
        return backend

    small, large = launch(2048, 2048), launch(4096, 4096)
    assert small._slot_context_capacity() == large._slot_context_capacity() == 2048
    saved = _manifest_from(small, n_saved = 100)
    assert not large.slot_resume_fits(saved)
    assert not launch(2048, 4096).slot_resume_fits(saved)
    assert launch(8192, 8192).slot_resume_fits(_manifest_from(large, n_saved = 100))


def test_context_change_requires_unambiguous_engine_values_on_both_launches(tmp_path):
    threshold = "print_info: n_ctx_orig_yarn = 2048"
    sequence = "llama_context: n_ctx_seq = 4096"
    for lines in (
        [],
        [threshold],
        [sequence],
        [threshold, sequence, "llama_context: n_ctx_seq = 1024"],
    ):
        assert llama_cpp._slot_rope_regime(lines) is None

    source = _ctx_backend(tmp_path, n_ctx = 1024)
    target = _ctx_backend(tmp_path, n_ctx = 2048)
    legacy = _manifest_from(source, n_saved = 100)
    del legacy["long_rope"]
    assert source.slot_resume_fits(legacy)
    assert not target.slot_resume_fits(legacy)
    target._slot_long_rope = None
    assert not target.slot_resume_fits(_manifest_from(source, n_saved = 100))


def test_a_context_change_with_metadata_or_slot_size_overrides_refuses(tmp_path):
    for extra in (
        ["--override-kv", "phi3.rope.scaling.original_context_length=int:8192"],
        ["--kv-unified-per-slot=4096"],
        ["--override_kv", "x=int:1"],
        ["--kv_unified_per_slot", "4096"],
    ):
        small, large = _ctx_backend(tmp_path, n_ctx = 8192), _ctx_backend(tmp_path, n_ctx = 16384)
        small._extra_args = large._extra_args = extra
        assert large.slot_resume_fits(_manifest_from(small, n_saved = 100)) is False


def test_a_context_change_under_the_per_slot_env_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_ARG_KV_UNIFIED_PER_SLOT", "2048")
    saved = _manifest_from(_ctx_backend(tmp_path, n_ctx = 8192), n_saved = 100)
    assert _ctx_backend(tmp_path, n_ctx = 16384).slot_resume_fits(saved) is False


def test_the_context_is_the_only_field_allowed_to_differ(tmp_path):
    saved = _manifest_from(_ctx_backend(tmp_path, n_ctx = 8192), n_saved = 100)
    backend = _ctx_backend(tmp_path, n_ctx = 32768)
    backend._effective_cache_types = ("q8_0", "q8_0")
    assert backend.slot_resume_fits(saved) is False
    backend = _ctx_backend(tmp_path, n_ctx = 32768)
    backend._extra_args = ["--rope-freq-scale", "0.5"]
    assert backend.slot_resume_fits(saved) is False
    backend = _ctx_backend(tmp_path, n_ctx = 32768)
    backend._flash_attn_enabled = False
    assert backend.slot_resume_fits(saved) is False


def test_a_malformed_or_empty_manifest_never_fits(tmp_path):
    backend = _ctx_backend(tmp_path, n_ctx = 8192)
    good = _manifest_from(backend, n_saved = 100)
    assert backend.slot_resume_fits({}) is False
    assert backend.slot_resume_fits({**good, "launch": good["launch"][:-1]}) is False
    bigger = _ctx_backend(tmp_path, n_ctx = 32768)
    assert bigger.slot_resume_fits({**good, "slots": []}) is False
    assert bigger.slot_resume_fits({**good, "slots": [{"id": 0, "n_saved": 0}]}) is False
    assert bigger.slot_resume_fits({**good, "slots": [{"id": 0, "n_saved": "many"}]}) is False


def test_the_fingerprint_still_tracks_the_context_for_the_strict_comparison(tmp_path):
    launch = _ctx_backend(tmp_path, n_ctx = 8192)._slot_launch_fingerprint()
    assert launch != _ctx_backend(tmp_path, n_ctx = 32768)._slot_launch_fingerprint()
    assert launch == tuple(launch)


def test_keep_warm_restores_a_stash_onto_a_launch_with_a_larger_context(monkeypatch, tmp_path):
    from core.inference import llama_keepwarm as kw

    # _resume_backend rewrites model.gguf, so build both before stashing.
    unloaded, reloaded = _ctx_backend(tmp_path, n_ctx = 8192), _ctx_backend(tmp_path, n_ctx = 32768)
    saved = _manifest_from(unloaded, n_saved = 8000)
    for entry in saved["slots"]:
        (tmp_path / entry["filename"]).write_bytes(b"kv")
    monkeypatch.setattr(kw, "_kv_resume", None)
    monkeypatch.setattr(kw, "_unload_kv", [])
    monkeypatch.setattr(kw, "_loaded_by", None, raising = False)
    kw.stash_unload_kv(saved)

    posts = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs["params"], kwargs["json"]))
        return _Resp(200, {"n_restored": 8000})

    monkeypatch.setattr(llama_cpp.httpx, "post", fake_post, raising = False)
    kw.note_model_loaded(reloaded)
    assert [p[1] for p in posts] == [{"action": "restore"}]
    assert posts[0][2] == {"filename": "resume-x-slot0.bin"}


def test_keep_warm_still_refuses_a_stash_whose_tokens_no_longer_fit(monkeypatch, tmp_path):
    from core.inference import llama_keepwarm as kw

    unloaded, reloaded = _ctx_backend(tmp_path, n_ctx = 8192), _ctx_backend(tmp_path, n_ctx = 4096)
    saved = _manifest_from(unloaded, n_saved = 8000)
    for entry in saved["slots"]:
        (tmp_path / entry["filename"]).write_bytes(b"kv")
    monkeypatch.setattr(kw, "_kv_resume", None)
    monkeypatch.setattr(kw, "_unload_kv", [])
    monkeypatch.setattr(kw, "_loaded_by", None, raising = False)
    kw.stash_unload_kv(saved)

    posts = []
    monkeypatch.setattr(
        llama_cpp.httpx, "post", lambda url, **kw_: posts.append(url) or _Resp(200), raising = False
    )
    kw.note_model_loaded(reloaded)
    assert posts == []
    assert not (tmp_path / "resume-x-slot0.bin").exists(), "a stash that cannot restore is deleted"


def test_every_slot_save_test_fakes_the_disk():
    # save_slots_for_resume skips on low free disk, so a test that leaves it to the host passes or fails
    # with the runner's disk: a save test goes red there, and a skip test can pass for the wrong reason.
    import ast
    import inspect
    import sys

    module = sys.modules[__name__]
    tree = ast.parse(inspect.getsource(module))
    unfaked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            calls = {
                n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
            }
            if "save_slots_for_resume" in calls and "_fake_disk" not in calls:
                unfaked.append(node.name)
    assert not unfaked, f"call _fake_disk(monkeypatch) first: {unfaked}"

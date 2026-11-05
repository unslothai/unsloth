# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On-demand HNSW: a scope that needs the pair while it is missing keeps the exact / vec0 answers, asks for one
background fetch, and moves to HNSW once the fetch lands; failures back off and never break search."""

import importlib.abc
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

np = pytest.importorskip("numpy")

from core.rag import hnsw_runtime, store, vec_ann, vec_cache

DIM = 16


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    monkeypatch.setenv("RAG_VEC_ANN_MIN_ROWS", "50")
    monkeypatch.setenv("RAG_VEC_CACHE", "1")
    monkeypatch.delenv("RAG_VEC_ANN_AUTO_INSTALL", raising = False)
    vec_cache.clear()
    hnsw_runtime.reset()
    yield
    vec_cache.drain()
    _settle()
    vec_cache.clear()
    hnsw_runtime.reset()


def _undo(monkeypatch):
    """monkeypatch.undo() also reverts _fresh's env (tests/conftest.py sets RAG_VEC_ANN_AUTO_INSTALL=0)."""
    monkeypatch.undo()
    monkeypatch.setenv("RAG_VEC_ANN_MIN_ROWS", "50")
    monkeypatch.setenv("RAG_VEC_CACHE", "1")
    monkeypatch.delenv("RAG_VEC_ANN_AUTO_INSTALL", raising = False)


def _settle(timeout = 10.0):
    end = time.monotonic() + timeout
    while hnsw_runtime._inflight and time.monotonic() < end:
        time.sleep(0.01)
    assert not hnsw_runtime._inflight


def _chunk(i):
    from core.rag.chunking import Chunk
    return Chunk(
        text = f"t{i}",
        token_count = 1,
        page_number = None,
        source_page_index = 0,
        chunk_index = i,
        page_char_start = 0,
        page_char_end = 2,
    )


def _add(conn, doc, vectors):
    store.create_document(conn, scope = "kb", filename = "f", sha256 = doc, document_id = doc)
    store.add_chunks(conn, "kb", doc, [_chunk(i) for i in range(len(vectors))], vectors)


def _vectors(n, seed = 0):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def _dense(
    conn,
    q,
    k = 5,
):
    seen = []
    conn.set_trace_callback(seen.append)
    try:
        hits = store.search_dense(conn, "kb", q, k)
    finally:
        conn.set_trace_callback(None)
    return hits, any("MATCH" in s.upper() for s in seen)


@pytest.fixture
def no_pair(monkeypatch):
    """usearch missing; hnsw_runtime.request recorded instead of fetching."""
    calls = []
    monkeypatch.setattr(vec_ann, "usearch", lambda: None)
    monkeypatch.setattr(hnsw_runtime, "request", lambda cb: calls.append(cb) or True)
    return calls


def test_a_missing_pair_keeps_exact_answers_and_asks_for_the_fetch(rag_conn, no_pair):
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _dense(rag_conn, rows[3])
    assert vec_cache.drain(60) and len(no_pair) == 1
    hits, scanned = _dense(rag_conn, rows[3])
    assert not scanned and hits[0][0] == "d:3" and vec_cache._entries
    # Searches of a waiting scope ask again; hnsw_runtime dedupes and backs off, nothing is rebuilt.
    assert len(no_pair) == 2 and not vec_cache._ann and vec_cache.drain(60)


def test_the_fetch_landing_moves_a_waiting_scope_to_hnsw_without_a_write(
    rag_conn, no_pair, monkeypatch
):
    monkeypatch.setenv("RAG_VEC_CACHE_MB", "0.001")
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _, scanned = _dense(rag_conn, rows[5])
    assert scanned and vec_cache.drain(60)
    _, scanned = _dense(rag_conn, rows[5])
    assert scanned and not vec_cache._ann and no_pair
    pytest.importorskip("usearch")
    # The fetch landed: the real import works again (undo also restores the default cache budget).
    _undo(monkeypatch)
    monkeypatch.setenv("RAG_VEC_ANN_MIN_ROWS", "50")
    assert vec_ann.usearch() is not None
    no_pair[0]()
    assert vec_cache.drain(60) and vec_cache._ann
    hits, scanned = _dense(rag_conn, rows[5])
    assert not scanned and hits[0][0] == "d:5"


def test_an_hnsw_build_that_fails_after_the_fetch_is_not_retried(rag_conn, no_pair, monkeypatch):
    pytest.importorskip("usearch")
    monkeypatch.setenv("RAG_VEC_CACHE_MB", "0.001")
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _dense(rag_conn, rows[0])
    assert vec_cache.drain(60) and len(no_pair) == 1
    _undo(monkeypatch)
    monkeypatch.setenv("RAG_VEC_ANN_MIN_ROWS", "50")
    monkeypatch.setenv("RAG_VEC_CACHE_MB", "0.001")
    attempts = []

    def broken(*a):
        attempts.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(vec_cache, "_build_ann", broken)
    no_pair[0]()
    assert vec_cache.drain(60) and attempts == [1]
    for _ in range(3):
        _, scanned = _dense(rag_conn, rows[0])
        assert scanned
    assert vec_cache.drain(60) and attempts == [1]


def test_a_waiting_scope_retries_a_transient_fetch_failure_without_rebuilding(
    rag_conn, monkeypatch
):
    monkeypatch.setenv("RAG_VEC_CACHE_MB", "0.001")
    monkeypatch.setattr(vec_ann, "usearch", lambda: None)
    runs = []
    monkeypatch.setattr(
        hnsw_runtime, "_run_installer", lambda: runs.append(1) or {"status": "failed"}
    )
    builds = []
    real_build = vec_cache._build
    monkeypatch.setattr(vec_cache, "_build", lambda key: (builds.append(key), real_build(key))[1])
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _dense(rag_conn, rows[2])
    assert vec_cache.drain(60)
    _settle()
    _dense(rag_conn, rows[2])
    _settle()
    assert runs == [1] and len(builds) == 1
    monkeypatch.setattr(hnsw_runtime, "_retry_after", 0.0)
    _, scanned = _dense(rag_conn, rows[2])
    _settle()
    assert scanned and runs == [1, 1] and len(builds) == 1


def test_an_exact_cached_scope_retries_a_failed_fetch_then_moves_to_hnsw(rag_conn, monkeypatch):
    pytest.importorskip("usearch")
    real = vec_ann.usearch
    state = {"avail": False}
    monkeypatch.setattr(vec_ann, "usearch", lambda: real() if state["avail"] else None)
    runs = []

    def installer():
        runs.append(1)
        if len(runs) == 1:
            return {"status": "failed"}
        state["avail"] = True
        return {"status": "ready"}

    monkeypatch.setattr(hnsw_runtime, "_run_installer", installer)
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _dense(rag_conn, rows[6])
    assert vec_cache.drain(60)
    _settle()
    _, scanned = _dense(rag_conn, rows[6])
    _settle()
    assert not scanned and vec_cache._entries and runs == [1]
    monkeypatch.setattr(hnsw_runtime, "_retry_after", 0.0)
    _dense(rag_conn, rows[6])
    _settle()
    assert runs == [1, 1] and vec_cache.drain(60)
    assert vec_cache._ann and not vec_cache._entries
    hits, scanned = _dense(rag_conn, rows[6])
    assert not scanned and hits[0][0] == "d:6"


def test_a_pair_landing_mid_build_still_promotes_the_scope_after(rag_conn, monkeypatch):
    pytest.importorskip("usearch")
    real = vec_ann.usearch
    state = {"avail": False}
    monkeypatch.setattr(vec_ann, "usearch", lambda: real() if state["avail"] else None)

    def lands_now(cb):
        state["avail"] = True
        cb()
        return True

    monkeypatch.setattr(hnsw_runtime, "request", lands_now)
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _dense(rag_conn, rows[8])
    assert vec_cache.drain(60)
    assert vec_cache._ann and not vec_cache._entries
    hits, scanned = _dense(rag_conn, rows[8])
    assert not scanned and hits[0][0] == "d:8"


def _run(monkeypatch, result):
    seen = []
    monkeypatch.setattr(hnsw_runtime, "_run_installer", lambda: seen.append(1) or result)
    return seen


@pytest.mark.parametrize("status", ["failed", "busy"])
def test_a_transient_failure_backs_off(monkeypatch, status):
    seen = _run(monkeypatch, {"status": status})
    called = []
    assert hnsw_runtime.request(lambda: called.append(1))
    _settle()
    assert not hnsw_runtime.request(lambda: called.append(1))
    assert seen == [1] and not called
    monkeypatch.setattr(hnsw_runtime, "_retry_after", 0.0)
    assert hnsw_runtime.request(lambda: None)
    _settle()
    assert seen == [1, 1]


@pytest.mark.parametrize("status", ["disabled", "offline", "unsupported"])
def test_a_settled_refusal_is_never_requested_again(monkeypatch, status):
    seen = _run(monkeypatch, {"status": status})
    assert hnsw_runtime.request(lambda: None)
    _settle()
    monkeypatch.setattr(hnsw_runtime, "_retry_after", 0.0)
    assert not hnsw_runtime.request(lambda: None)
    assert seen == [1]


def test_the_auto_install_switch_never_spawns(monkeypatch):
    monkeypatch.setenv("RAG_VEC_ANN_AUTO_INSTALL", "0")
    seen = _run(monkeypatch, {"status": "ready"})
    assert not hnsw_runtime.request(lambda: None)
    assert not seen and hnsw_runtime._final == "disabled"


def test_ready_calls_back_once(monkeypatch):
    monkeypatch.setattr(vec_ann, "usearch", lambda: object())
    _run(monkeypatch, {"status": "ready"})
    called = []
    assert hnsw_runtime.request(lambda: called.append(1))
    _settle()
    assert called == [1] and not hnsw_runtime.request(lambda: called.append(2))


def test_ready_that_still_does_not_import_counts_as_failed(monkeypatch):
    monkeypatch.setattr(vec_ann, "usearch", lambda: None)
    _run(monkeypatch, {"status": "ready"})
    called = []
    hnsw_runtime.request(lambda: called.append(1))
    _settle()
    assert not called and hnsw_runtime._final is None and hnsw_runtime._retry_after > 0


def _fake_installer(tmp_path, monkeypatch, body):
    script = tmp_path / "install_hnsw_prebuilt.py"
    script.write_text(textwrap.dedent(body), encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_HNSW_INSTALLER", str(script))


def test_the_installer_runs_as_a_subprocess_with_a_filtered_env(tmp_path, monkeypatch):
    _fake_installer(
        tmp_path,
        monkeypatch,
        """
        import json, os, sys
        print("noise from uv")
        print(json.dumps({"status": "busy", "argv": sys.argv[1:], "token": os.environ.get("HF_TOKEN"),
                          "mirror": os.environ.get("UNSLOTH_HNSW_MIRROR")}))
    """,
    )
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("UNSLOTH_HNSW_MIRROR", "https://mirror.example/")
    result = hnsw_runtime._run_installer()
    assert result["status"] == "busy" and result["argv"] == ["--json"]
    assert result["token"] is None and result["mirror"] == "https://mirror.example/"


@pytest.mark.parametrize("stdout", ["", "garbage\n", '{"status": 1}\n', "{not json\n", "[1, 2]\n"])
def test_garbage_installer_output_is_a_failure_not_an_exception(tmp_path, monkeypatch, stdout):
    _fake_installer(tmp_path, monkeypatch, f"import sys; sys.stdout.write({stdout!r})")
    assert hnsw_runtime._run_installer()["status"] == "failed"


def test_a_hung_installer_times_out_as_a_failure(monkeypatch, tmp_path):
    _fake_installer(tmp_path, monkeypatch, "print('x')")

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("installer", 1)

    monkeypatch.setattr(hnsw_runtime.subprocess, "run", hang)
    assert hnsw_runtime._run_installer()["status"] == "failed"


def test_a_missing_installer_is_unsupported(monkeypatch):
    monkeypatch.setattr(hnsw_runtime, "_installer", lambda: None)
    assert hnsw_runtime._run_installer()["status"] == "unsupported"


def test_request_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no threads")

    monkeypatch.setattr(hnsw_runtime.threading, "Thread", boom)
    assert hnsw_runtime.request(lambda: None) is False and not hnsw_runtime._inflight
    _undo(monkeypatch)
    monkeypatch.setattr(
        hnsw_runtime, "_run_installer", lambda: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert hnsw_runtime.request(lambda: None)
    _settle()
    assert hnsw_runtime._retry_after > 0


class _Tripwire(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.hit = []

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if name.split(".")[0] == "usearch":
            self.hit.append(name)
        return None


@pytest.fixture
def tripwire(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    for name in [m for m in sys.modules if m.split(".")[0] == "usearch"]:
        monkeypatch.delitem(sys.modules, name)
    wire = _Tripwire()
    monkeypatch.setattr(sys, "meta_path", [wire, *sys.meta_path])
    return wire


def _marker(tmp_path, **over):
    marker = {
        "pair": vec_ann.PAIR,
        "fingerprint": vec_ann.hnsw_fingerprint(),
        "result": "failed",
        "reason": "probe",
    }
    marker.update(over)
    (tmp_path / vec_ann.MARKER).write_text(json.dumps(marker), encoding = "utf-8")


@pytest.mark.parametrize("result", ["failed", "unverified"])
def test_a_failed_or_unverified_probe_marker_skips_the_import(tripwire, tmp_path, result):
    _marker(tmp_path, result = result)
    assert vec_ann.usearch() is None and not tripwire.hit


@pytest.mark.parametrize(
    "over",
    [{"result": "ok"}, {"fingerprint": "someone-else"}, {"pair": "usearch==1.0 numkong==1.0"}],
)
def test_other_markers_still_import(tripwire, tmp_path, over):
    _marker(tmp_path, **over)
    vec_ann.usearch()
    assert tripwire.hit


def test_a_corrupt_marker_still_imports(tripwire, tmp_path):
    (tmp_path / vec_ann.MARKER).write_text("{", encoding = "utf-8")
    vec_ann.usearch()
    assert tripwire.hit


def test_a_live_install_lock_skips_the_import_until_it_goes(tripwire, tmp_path):
    lock = tmp_path / vec_ann.LOCK
    lock.write_text("123", encoding = "utf-8")
    assert vec_ann.usearch() is None and not tripwire.hit
    old = time.time() - vec_ann.LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    vec_ann.usearch()
    assert tripwire.hit


def test_an_exact_cached_waiting_scope_moves_to_hnsw_when_the_pair_lands(
    rag_conn, no_pair, monkeypatch
):
    pytest.importorskip("usearch")
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    _dense(rag_conn, rows[4])
    assert vec_cache.drain(60) and vec_cache._entries and no_pair
    _undo(monkeypatch)
    monkeypatch.setenv("RAG_VEC_ANN_MIN_ROWS", "50")
    no_pair[0]()
    assert vec_cache.drain(60) and vec_cache._ann and not vec_cache._entries
    hits, scanned = _dense(rag_conn, rows[4])
    assert not scanned and hits[0][0] == "d:4"


def test_the_installer_is_found_without_importing_the_studio_update_stack(monkeypatch):
    monkeypatch.delenv("UNSLOTH_HNSW_INSTALLER", raising = False)
    monkeypatch.setitem(sys.modules, "utils.prebuilt.update_flow", None)
    script = hnsw_runtime._installer()
    assert script is not None and script.name == "install_hnsw_prebuilt.py" and script.is_file()


def test_a_pair_that_lands_during_the_scopes_own_build_still_switches_it(rag_conn, monkeypatch):
    pytest.importorskip("usearch")
    monkeypatch.setenv("RAG_VEC_CACHE_MB", "0.001")
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    real = vec_ann.usearch
    state = {"avail": False, "cb": []}
    monkeypatch.setattr(vec_ann, "usearch", lambda: real() if state["avail"] else None)

    def request(cb):
        if state["avail"]:
            return False
        state["cb"].append(cb)
        return True

    monkeypatch.setattr(hnsw_runtime, "request", request)
    orig_enabled = vec_cache.ann_enabled

    def enabled():
        r = orig_enabled()
        if not r and not state["avail"]:
            state["avail"] = True
            vec_cache._on_ann_available()
        return r

    monkeypatch.setattr(vec_cache, "ann_enabled", enabled)
    _dense(rag_conn, rows[1])
    assert vec_cache.drain(60)
    for _ in range(3):
        _, scanned = _dense(rag_conn, rows[1])
        assert vec_cache.drain(60)
    assert vec_cache._ann, "scope stuck on vec0 after the pair became importable"


def test_a_scope_just_over_budget_is_not_rebuilt_on_every_query(rag_conn, monkeypatch):
    rows = _vectors(300)
    _add(rag_conn, "d", rows)
    monkeypatch.setattr(vec_ann, "usearch", lambda: None)
    monkeypatch.setattr(hnsw_runtime, "request", lambda cb: False)
    est = 300 * (16 * 4 + 8 + vec_cache._ID_OVERHEAD)
    monkeypatch.setenv("RAG_VEC_CACHE_MB", str((est + 600) / 2**20))
    builds = []
    orig = vec_cache._build
    monkeypatch.setattr(vec_cache, "_build", lambda key: (builds.append(key), orig(key))[1])
    for _ in range(5):
        _dense(rag_conn, rows[1])
        assert vec_cache.drain(60)
    assert len(builds) <= 1, f"{len(builds)} full rebuilds for 5 queries"

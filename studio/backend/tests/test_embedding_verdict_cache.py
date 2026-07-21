# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration tests for the offline embedding verdict cache: an ONLINE clean load records a
verdict, and a later OFFLINE load of the SAME content is allowed instead of fail-closed. Every
mismatch (moved commit, changed / added pickle, expired record) must keep blocking.

Snapshot resolution is stubbed (``_active_snapshot_dir`` / ``_active_commit``) so no HF cache or
network is touched; UNSLOTH_STUDIO_HOME isolates the verdict store per test.
"""

import json
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

import core.rag.config as config
import core.rag.embeddings as embeddings
import utils.models.model_config as mc
import utils.security.embedding_scan_verdicts as verdicts
import utils.security.file_security as fs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", raising = False)
    verdicts.clear()
    return tmp_path


def _snap(tmp_path, files: dict):
    """Build a snapshot dir with *files* (name -> bytes/str; names may be nested like ``a/b.bin``)."""
    d = tmp_path / "cache" / "snap"
    d.mkdir(parents = True, exist_ok = True)
    for name, body in files.items():
        p = d / name
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_bytes(body if isinstance(body, bytes) else body.encode())
    return d


def _blocked(monkeypatch, snap, commit, load_subdirs = ()):
    monkeypatch.setattr(mc, "_active_snapshot_dir", lambda name: snap)
    monkeypatch.setattr(mc, "_active_commit", lambda name: commit)
    return fs.evaluate_file_security(
        "org/model", None, load_subdirs = load_subdirs, local_only_load = True
    ).blocked


def _sha(path):
    return verdicts.sha256_file(path)


# ── Offline verify matrix ────────────────────────────────────────────

def test_matching_record_allows_offline_pickle(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")})
    assert _blocked(monkeypatch, snap, "c1") is False


def test_no_record_blocks_offline_pickle(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    assert _blocked(monkeypatch, snap, "c1") is True


def test_wrong_commit_blocks(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")})
    assert _blocked(monkeypatch, snap, "c2") is True


def test_tampered_pickle_blocks(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")})
    (snap / "pytorch_model.bin").write_bytes(b"TAMPERED-SAME-COMMIT")
    assert _blocked(monkeypatch, snap, "c1") is True


def test_partial_record_blocks(home, tmp_path, monkeypatch):
    # Two module dirs each ship a pickle; recording only one leaves the set mismatched -> block.
    snap = _snap(
        tmp_path,
        {
            "modules.json": json.dumps(
                [{"path": "0_A", "type": "..."}, {"path": "0_B", "type": "..."}]
            ),
            "0_A/pytorch_model.bin": b"aaa",
            "0_B/pytorch_model.bin": b"bbb",
        },
    )
    verdicts.record_clean("org/model", "c1", {"0_A/pytorch_model.bin": _sha(snap / "0_A/pytorch_model.bin")})
    assert _blocked(monkeypatch, snap, "c1") is True


def test_distinct_basenames_in_different_roots_are_keyed_apart(home, tmp_path, monkeypatch):
    # Same basename in two module dirs must be recorded/verified as two distinct entries; if the
    # enumerator collapsed them by basename, one record could vouch for both. Recording BOTH with
    # their real (distinct) hashes allows; the map has two keys.
    snap = _snap(
        tmp_path,
        {
            "modules.json": json.dumps(
                [{"path": "0_A", "type": "..."}, {"path": "0_B", "type": "..."}]
            ),
            "0_A/pytorch_model.bin": b"aaa",
            "0_B/pytorch_model.bin": b"bbb",
        },
    )
    verdicts.record_clean(
        "org/model",
        "c1",
        {
            "0_A/pytorch_model.bin": _sha(snap / "0_A/pytorch_model.bin"),
            "0_B/pytorch_model.bin": _sha(snap / "0_B/pytorch_model.bin"),
        },
    )
    assert _blocked(monkeypatch, snap, "c1") is False


def test_expired_record_blocks(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")})
    path = verdicts._store_path()
    data = json.loads(path.read_text())
    data["records"]["org/model"]["recorded_at"] = (
        datetime.now(timezone.utc) - timedelta(days = 31)
    ).isoformat()
    path.write_text(json.dumps(data))
    assert _blocked(monkeypatch, snap, "c1") is True


def test_router_child_pickle_recorded_allows_else_blocks(home, tmp_path, monkeypatch):
    # A Router child pickle (declared only in router_config.json) is a load root the offline gate
    # scopes. Recording it allows; without a record it blocks.
    snap = _snap(
        tmp_path,
        {
            "modules.json": json.dumps([{"path": "", "type": "sentence_transformers.models.Router.Router"}]),
            "router_config.json": json.dumps({"types": {"query_0_WordEmbeddings": "..."}}),
            "query_0_WordEmbeddings/pytorch_model.bin": b"child-weights",
        },
    )
    child = snap / "query_0_WordEmbeddings" / "pytorch_model.bin"
    assert _blocked(monkeypatch, snap, "c1") is True
    verdicts.record_clean("org/model", "c1", {"query_0_WordEmbeddings/pytorch_model.bin": _sha(child)})
    assert _blocked(monkeypatch, snap, "c1") is False


def test_disabled_cache_never_allows(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")})
    monkeypatch.setenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", "1")
    assert _blocked(monkeypatch, snap, "c1") is True


# ── record_embedding_verdict (the post-load recorder) ────────────────

def _stub_snapshot(monkeypatch, snap, commit):
    monkeypatch.setattr(mc, "_active_snapshot_dir", lambda name: snap)
    monkeypatch.setattr(mc, "_active_commit", lambda name: commit)


def test_record_embedding_verdict_records_and_allows(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"w"})
    _stub_snapshot(monkeypatch, snap, "c1")
    fs.record_embedding_verdict("org/model", "c1", load_subdirs = ())
    assert verdicts.lookup("org/model", "c1") == {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    assert fs.evaluate_file_security("org/model", None, local_only_load = True).blocked is False


def test_record_skips_when_commit_moved(home, tmp_path, monkeypatch):
    # The active cached commit differs from the scanned commit (branch advanced): record nothing,
    # so an unscanned commit is never blessed.
    snap = _snap(tmp_path, {"pytorch_model.bin": b"w"})
    _stub_snapshot(monkeypatch, snap, "c2")
    fs.record_embedding_verdict("org/model", "c1", load_subdirs = ())
    assert verdicts.lookup("org/model", "c1") is None
    assert verdicts.lookup("org/model", "c2") is None


def test_record_skips_pickle_free_cache(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"model.safetensors": b"\0"})
    _stub_snapshot(monkeypatch, snap, "c1")
    fs.record_embedding_verdict("org/model", "c1", load_subdirs = ())
    assert verdicts.lookup("org/model", "c1") is None


# ── _get() wiring: record only after a clean ONLINE load ─────────────

def _drive_get(monkeypatch, decision, *, offline):
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = lambda name, **k: object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr("utils.models.resolve_st_cached_repo_id_case", lambda r: r)
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: offline)
    monkeypatch.setattr(embeddings, "_guard_model_security", lambda name, lo: decision)
    rec = {}
    monkeypatch.setattr(
        embeddings, "_record_embedding_verdict_safe", lambda n, c: rec.update(name = n, commit = c)
    )
    monkeypatch.setattr(embeddings, "_model", None, raising = False)
    monkeypatch.setattr(embeddings, "_name", None, raising = False)
    embeddings._get()
    return rec


def test_get_records_after_clean_online_load(home, monkeypatch):
    dec = fs.FileSecurityDecision("org/model", False, commit = "c1", scanned_clean = True)
    assert _drive_get(monkeypatch, dec, offline = False) == {"name": "org/model", "commit": "c1"}


def test_get_does_not_record_when_scan_not_definitive(home, monkeypatch):
    # A fail-open ("scan unavailable") online load has scanned_clean False -> record nothing.
    dec = fs.FileSecurityDecision("org/model", False, commit = None, scanned_clean = False)
    assert _drive_get(monkeypatch, dec, offline = False) == {}


def test_get_does_not_record_offline_load(home, monkeypatch):
    # Offline: no authoritative scan happened, so nothing is recorded regardless.
    dec = fs.FileSecurityDecision("org/model", False, commit = "c1", scanned_clean = True)
    assert _drive_get(monkeypatch, dec, offline = True) == {}


def test_get_does_not_record_on_gate_error(home, monkeypatch):
    # _guard_model_security returns None on a gate error -> record nothing.
    assert _drive_get(monkeypatch, None, offline = False) == {}

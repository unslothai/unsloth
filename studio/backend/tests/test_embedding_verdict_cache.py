# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration tests for the offline embedding verdict cache: an ONLINE clean load records a
verdict, and a later OFFLINE load of the SAME content is allowed instead of fail-closed. Every
mismatch (moved commit, changed / added pickle, expired record, unreadable/uninspectable cache)
must keep blocking.

Snapshot resolution is stubbed (``_active_snapshot_dir`` / ``_st_cache_repo_dir``) over a real
``models--org--model/snapshots/<commit>/`` layout so the commit is taken from the snapshot dir
name, as in production; no HF cache or network is touched, and UNSLOTH_STUDIO_HOME isolates the
verdict store per test.
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

_REPO = "models--org--model"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", raising = False)
    verdicts.clear()
    return tmp_path


def _repo_dir(tmp_path):
    return tmp_path / "cache" / _REPO


def _snap(
    tmp_path,
    files: dict,
    commit = "c1",
):
    """Build ``models--org--model/snapshots/<commit>/`` with *files* (name -> bytes/str; names may
    be nested like ``a/b.bin``). Returns the snapshot dir, whose ``.name`` is the commit."""
    d = _repo_dir(tmp_path) / "snapshots" / commit
    d.mkdir(parents = True, exist_ok = True)
    for name, body in files.items():
        p = d / name
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_bytes(body if isinstance(body, bytes) else body.encode())
    return d


def _blocked(
    monkeypatch,
    snap,
    load_subdirs = (),
):
    # The offline verify derives the commit from snap.name; only _active_snapshot_dir is consulted.
    monkeypatch.setattr(mc, "_active_snapshot_dir", lambda name: snap)
    return fs.evaluate_file_security(
        "org/model", None, load_subdirs = load_subdirs, local_only_load = True
    ).blocked


def _sha(path):
    return verdicts.sha256_file(path)


# ── Offline verify matrix ────────────────────────────────────────────


def test_matching_record_allows_offline_pickle(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean(
        "org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    )
    assert _blocked(monkeypatch, snap) is False


def test_no_record_blocks_offline_pickle(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    assert _blocked(monkeypatch, snap) is True


def test_wrong_commit_blocks(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"}, commit = "c1")
    # Record under a DIFFERENT commit than the cached snapshot's -> lookup by snap.name misses.
    verdicts.record_clean(
        "org/model", "c2", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    )
    assert _blocked(monkeypatch, snap) is True


def test_tampered_pickle_blocks(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean(
        "org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    )
    (snap / "pytorch_model.bin").write_bytes(b"TAMPERED-SAME-COMMIT")
    assert _blocked(monkeypatch, snap) is True


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
    verdicts.record_clean(
        "org/model", "c1", {"0_A/pytorch_model.bin": _sha(snap / "0_A/pytorch_model.bin")}
    )
    assert _blocked(monkeypatch, snap) is True


def test_distinct_basenames_in_different_roots_are_keyed_apart(home, tmp_path, monkeypatch):
    # Same basename in two module dirs must be recorded/verified as two distinct entries.
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
    assert _blocked(monkeypatch, snap) is False


def test_case_variant_pickles_are_both_hashed(home, tmp_path, monkeypatch):
    # On a case-sensitive FS pytorch_model.bin and PYTORCH_MODEL.BIN are distinct load-root files;
    # both must be hashed. Recording only the exact-case one leaves the collision unmatched -> block;
    # recording BOTH (their real distinct hashes) allows. Proves no last-wins collapse.
    if (tmp_path / "A").exists():  # skip on a case-insensitive filesystem
        pytest.skip("case-insensitive filesystem")
    (tmp_path / "casecheck").write_text("x")
    if (tmp_path / "CASECHECK").exists():
        pytest.skip("case-insensitive filesystem")
    snap = _snap(tmp_path, {"pytorch_model.bin": b"real", "PYTORCH_MODEL.BIN": b"decoy"})
    real = _sha(snap / "pytorch_model.bin")
    decoy = _sha(snap / "PYTORCH_MODEL.BIN")
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": real})
    assert _blocked(monkeypatch, snap) is True  # decoy unrecorded -> mismatch
    verdicts.record_clean(
        "org/model", "c1", {"pytorch_model.bin": real, "PYTORCH_MODEL.BIN": decoy}
    )
    assert _blocked(monkeypatch, snap) is False


def test_expired_record_blocks(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean(
        "org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    )
    path = verdicts._store_path()
    data = json.loads(path.read_text())
    data["records"]["org/model"]["recorded_at"] = (
        datetime.now(timezone.utc) - timedelta(days = 31)
    ).isoformat()
    path.write_text(json.dumps(data))
    assert _blocked(monkeypatch, snap) is True


def test_router_child_pickle_recorded_allows_else_blocks(home, tmp_path, monkeypatch):
    snap = _snap(
        tmp_path,
        {
            "modules.json": json.dumps(
                [{"path": "", "type": "sentence_transformers.models.Router.Router"}]
            ),
            "router_config.json": json.dumps({"types": {"query_0_WordEmbeddings": "..."}}),
            "query_0_WordEmbeddings/pytorch_model.bin": b"child-weights",
        },
    )
    child = snap / "query_0_WordEmbeddings" / "pytorch_model.bin"
    assert _blocked(monkeypatch, snap) is True
    verdicts.record_clean(
        "org/model", "c1", {"query_0_WordEmbeddings/pytorch_model.bin": _sha(child)}
    )
    assert _blocked(monkeypatch, snap) is False


def test_disabled_cache_never_allows(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean(
        "org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    )
    monkeypatch.setenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", "1")
    assert _blocked(monkeypatch, snap) is True


def test_unreadable_recorded_pickle_blocks(home, tmp_path, monkeypatch):
    # A recorded pickle that cannot be re-hashed (sha256_file -> None) must block, not allow.
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})
    verdicts.record_clean(
        "org/model", "c1", {"pytorch_model.bin": _sha(snap / "pytorch_model.bin")}
    )
    monkeypatch.setattr(verdicts, "sha256_file", lambda p: None)
    assert _blocked(monkeypatch, snap) is True


def test_uninspectable_cache_blocks(home, tmp_path, monkeypatch):
    # If the cache tree cannot be enumerated (rglob OSError), fail CLOSED, not "pickle-free allow".
    snap = _snap(tmp_path, {"pytorch_model.bin": b"weights"})

    def _boom(*a, **k):
        raise OSError("EIO")

    monkeypatch.setattr(fs, "_cached_pickle_weight_paths", _boom)
    assert _blocked(monkeypatch, snap) is True


def test_sharded_safetensors_index_credited_only_at_root(home, tmp_path, monkeypatch):
    # A complete model.safetensors.index.json covers a sibling pickle ONLY at a from_pretrained
    # root. A non-Transformer ST module (Dense/WordEmbeddings) loads via load_torch_weights, which
    # ignores the index and reads pytorch_model.bin -- so a sharded index there must NOT credit the
    # pickle, or it deserializes unblocked offline.
    index = json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})

    # Module dir (not root): index does NOT credit the pickle -> blocked.
    mod = _snap(
        tmp_path,
        {
            "modules.json": json.dumps([{"path": "0_Dense", "type": "..."}]),
            "0_Dense/pytorch_model.bin": b"pickle",
            "0_Dense/model.safetensors.index.json": index,
            "0_Dense/model-00001-of-00001.safetensors": b"\0",
        },
        commit = "cmod",
    )
    assert _blocked(monkeypatch, mod) is True

    # Snapshot root (from_pretrained): the same index DOES credit the pickle -> allowed.
    root = _snap(
        tmp_path,
        {
            "pytorch_model.bin": b"pickle",
            "model.safetensors.index.json": index,
            "model-00001-of-00001.safetensors": b"\0",
        },
        commit = "croot",
    )
    assert _blocked(monkeypatch, root) is False


def test_unresolvable_snapshot_blocks(home, monkeypatch):
    # A snapshot that ERRORS on resolution (not a clean None) fails closed.
    def _boom(name):
        raise OSError("refs/main unreadable")

    monkeypatch.setattr(mc, "_active_snapshot_dir", _boom)
    assert fs.evaluate_file_security("org/model", None, local_only_load = True).blocked is True


# ── record_embedding_verdict (the post-load recorder) ────────────────


def _stub_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "_st_cache_repo_dir", lambda name: _repo_dir(tmp_path))


def test_record_embedding_verdict_records_and_allows(home, tmp_path, monkeypatch):
    snap = _snap(tmp_path, {"pytorch_model.bin": b"w"}, commit = "c1")
    _stub_repo(monkeypatch, tmp_path)
    fs.record_embedding_verdict("org/model", "c1", load_subdirs = ())
    assert verdicts.lookup("org/model", "c1") == {
        "pytorch_model.bin": _sha(snap / "pytorch_model.bin")
    }
    assert _blocked(monkeypatch, snap) is False


def test_record_skips_when_scanned_commit_not_cached(home, tmp_path, monkeypatch):
    # Only snapshots/c1 exists; recording the scanned commit c2 (branch moved) finds no snapshot
    # for c2 and records nothing, so an unscanned commit is never blessed.
    _snap(tmp_path, {"pytorch_model.bin": b"w"}, commit = "c1")
    _stub_repo(monkeypatch, tmp_path)
    fs.record_embedding_verdict("org/model", "c2", load_subdirs = ())
    assert verdicts.lookup("org/model", "c1") is None
    assert verdicts.lookup("org/model", "c2") is None


def test_record_skips_pickle_free_cache(home, tmp_path, monkeypatch):
    _snap(tmp_path, {"model.safetensors": b"\0"}, commit = "c1")
    _stub_repo(monkeypatch, tmp_path)
    fs.record_embedding_verdict("org/model", "c1", load_subdirs = ())
    assert verdicts.lookup("org/model", "c1") is None


# ── forget() on an authoritative unsafe verdict (end-to-end) ─────────


def test_online_unsafe_forgets_recorded_verdict(home, monkeypatch):
    # An online scan that flags a load-path pickle must delete any recorded clean verdict.
    verdicts.record_clean("org/model", "c1", {"pytorch_model.bin": "a" * 64})
    monkeypatch.setattr(
        fs,
        "_fetch_security_status",
        lambda name, token: (
            {
                "scansDone": True,
                "filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}],
            },
            "c1",
        ),
    )
    decision = fs.evaluate_file_security("org/model", None, local_only_load = False)
    assert decision.blocked is True
    assert verdicts.lookup("org/model", "c1") is None


# ── recordable-clean predicate (only a COMPLETED, entirely-benign scan) ──


def _scanned_clean(monkeypatch, status):
    monkeypatch.setattr(fs, "_fetch_security_status", lambda name, token: (status, "c1"))
    return fs.evaluate_file_security("org/model", None, local_only_load = False).scanned_clean


def test_recordable_requires_completed_scan(home, monkeypatch):
    assert _scanned_clean(monkeypatch, {"scansDone": True, "filesWithIssues": []}) is True
    # scansDone must be the boolean True, not a truthy string.
    assert _scanned_clean(monkeypatch, {"scansDone": "false", "filesWithIssues": []}) is False
    assert _scanned_clean(monkeypatch, {"filesWithIssues": []}) is False


def test_recordable_rejects_malformed_or_flagged_manifest(home, monkeypatch):
    assert _scanned_clean(monkeypatch, {"scansDone": True}) is False  # filesWithIssues not a list
    assert _scanned_clean(monkeypatch, {"scansDone": True, "filesWithIssues": "bad"}) is False
    assert _scanned_clean(monkeypatch, {"scansDone": True, "filesWithIssues": [None]}) is False
    # A pending/error flagged file (even outside the load path) disqualifies a durable clean record.
    assert (
        _scanned_clean(
            monkeypatch,
            {"scansDone": True, "filesWithIssues": [{"path": "archive/x.bin", "level": "error"}]},
        )
        is False
    )


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
    dec = fs.FileSecurityDecision("org/model", False, commit = None, scanned_clean = False)
    assert _drive_get(monkeypatch, dec, offline = False) == {}


def test_get_does_not_record_offline_load(home, monkeypatch):
    dec = fs.FileSecurityDecision("org/model", False, commit = "c1", scanned_clean = True)
    assert _drive_get(monkeypatch, dec, offline = True) == {}


def test_get_does_not_record_on_gate_error(home, monkeypatch):
    assert _drive_get(monkeypatch, None, offline = False) == {}


def _drive_get_capturing_st(monkeypatch, st_factory, *, offline):
    """Drive ``_get`` with a caller-supplied SentenceTransformer factory (to probe the exact
    kwargs the constructor receives). Returns nothing; the factory records what it needs."""
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = st_factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    monkeypatch.setattr("utils.models.resolve_st_cached_repo_id_case", lambda r: r)
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: offline)
    monkeypatch.setattr(
        embeddings,
        "_guard_model_security",
        lambda name, lo: fs.FileSecurityDecision("org/model", False),
    )
    monkeypatch.setattr(embeddings, "_record_embedding_verdict_safe", lambda n, c: None)
    monkeypatch.setattr(embeddings, "_model", None, raising = False)
    monkeypatch.setattr(embeddings, "_name", None, raising = False)
    embeddings._get()


def test_get_online_omits_local_files_only_for_old_st(home, monkeypatch):
    # pyproject pins no minimum sentence-transformers, and older releases lack the
    # local_files_only constructor arg. An ONLINE warm must not forward it, or such an install
    # raises TypeError on every embedder load. The strict-signature factory (no **kwargs) raises
    # exactly that if the kwarg is passed.
    seen = {}

    def _old_st(name, *, device, model_kwargs):
        seen["called"] = True
        return object()

    _drive_get_capturing_st(monkeypatch, _old_st, offline = False)
    assert seen.get("called") is True


def test_get_offline_still_passes_local_files_only(home, monkeypatch):
    # Offline (the new capability) still forwards local_files_only=True, so a modern install
    # loads purely from cache.
    seen = {}

    def _st(name, **k):
        seen.update(k)
        return object()

    _drive_get_capturing_st(monkeypatch, _st, offline = True)
    assert seen.get("local_files_only") is True


# ── guard fails CLOSED offline on a gate error ───────────────────────


def test_guard_offline_gate_error_fails_closed(home, monkeypatch):
    def _boom(*a, **k):
        raise OSError("gate blew up")

    monkeypatch.setattr("utils.security.evaluate_file_security", _boom)
    # Offline: a gate error must refuse the load (fail closed), not return None and proceed.
    with pytest.raises(embeddings.UnsafeEmbeddingModelError):
        embeddings._guard_model_security("org/model", True)
    # Online: a gate error stays fail-open (best-effort Hub scan) -> returns None.
    assert embeddings._guard_model_security("org/model", False) is None

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the persistent embedding "clean verdict" store.

The store is a fail-safe optimization: a good record lets an offline pickle model load, and ANY
problem (missing / wrong commit / expired / corrupt / disabled) must resolve to "no record" so the
offline gate keeps blocking. These tests pin that contract with an isolated UNSLOTH_STUDIO_HOME.
"""

import json

import pytest


@pytest.fixture
def verdicts(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", raising = False)
    import utils.security.embedding_scan_verdicts as v

    v.clear()
    return v


_SHA = "a" * 64
_SHA2 = "b" * 64


def test_record_then_lookup_round_trip(verdicts):
    verdicts.record_clean("BAAI/bge-m3", "commit1", {"pytorch_model.bin": _SHA})
    assert verdicts.lookup("BAAI/bge-m3", "commit1") == {"pytorch_model.bin": _SHA}


def test_lookup_is_case_insensitive_on_repo_id(verdicts):
    # Hub repo ids are case-insensitive; a record made under one casing is found under another.
    verdicts.record_clean("BAAI/bge-m3", "commit1", {"pytorch_model.bin": _SHA})
    assert verdicts.lookup("baai/bge-m3", "commit1") == {"pytorch_model.bin": _SHA}


def test_wrong_commit_misses(verdicts):
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    assert verdicts.lookup("acme/model", "commit2") is None


def test_lookup_without_commit_is_none(verdicts):
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    assert verdicts.lookup("acme/model", None) is None


def test_record_without_commit_is_noop(verdicts):
    verdicts.record_clean("acme/model", None, {"pytorch_model.bin": _SHA})
    assert verdicts.lookup("acme/model", "commit1") is None


def test_record_with_empty_map_is_noop(verdicts):
    verdicts.record_clean("acme/model", "commit1", {})
    assert verdicts.lookup("acme/model", "commit1") is None


def test_malformed_hash_is_rejected(verdicts):
    # A non-64-hex digest can never be persisted, so it can never seed a spurious match.
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": "not-a-hash"})
    assert verdicts.lookup("acme/model", "commit1") is None


def test_ttl_expiry_returns_none(verdicts):
    from datetime import datetime, timedelta, timezone

    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    # Backdate the record past the 30-day TTL by editing the store on disk.
    path = verdicts._store_path()
    data = json.loads(path.read_text())
    old = (datetime.now(timezone.utc) - timedelta(days = 31)).isoformat()
    data["records"]["acme/model"]["recorded_at"] = old
    path.write_text(json.dumps(data))
    assert verdicts.lookup("acme/model", "commit1") is None


def test_unparseable_timestamp_returns_none(verdicts):
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    path = verdicts._store_path()
    data = json.loads(path.read_text())
    data["records"]["acme/model"]["recorded_at"] = "not-a-timestamp"
    path.write_text(json.dumps(data))
    assert verdicts.lookup("acme/model", "commit1") is None


def test_forget_drops_record(verdicts):
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    verdicts.forget("acme/model")
    assert verdicts.lookup("acme/model", "commit1") is None


def test_corrupt_store_fails_safe(verdicts):
    path = verdicts._store_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("{ this is not json")
    assert verdicts.lookup("acme/model", "commit1") is None


def test_wrong_schema_version_fails_safe(verdicts):
    path = verdicts._store_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(
            {
                "version": 999,
                "records": {
                    "acme/model": {
                        "commit": "commit1",
                        "recorded_at": "2999-01-01T00:00:00+00:00",
                        "pickles": {"pytorch_model.bin": _SHA},
                    }
                },
            }
        )
    )
    assert verdicts.lookup("acme/model", "commit1") is None


def test_non_dict_records_fails_safe(verdicts):
    path = verdicts._store_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps({"version": 1, "records": []}))
    assert verdicts.lookup("acme/model", "commit1") is None


def test_cache_disabled_noops_record_and_lookup(verdicts, monkeypatch):
    monkeypatch.setenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", "1")
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    assert verdicts.lookup("acme/model", "commit1") is None
    # And a record made while enabled is not honored once disabled.
    monkeypatch.delenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE")
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    monkeypatch.setenv("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", "true")
    assert verdicts.lookup("acme/model", "commit1") is None


def test_latest_record_per_repo_wins(verdicts):
    verdicts.record_clean("acme/model", "commit1", {"pytorch_model.bin": _SHA})
    verdicts.record_clean("acme/model", "commit2", {"pytorch_model.bin": _SHA2})
    assert verdicts.lookup("acme/model", "commit1") is None
    assert verdicts.lookup("acme/model", "commit2") == {"pytorch_model.bin": _SHA2}


def test_sha256_file(tmp_path, verdicts):
    import hashlib

    f = tmp_path / "w.bin"
    f.write_bytes(b"hello world")
    assert verdicts.sha256_file(f) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_missing_is_none(tmp_path, verdicts):
    assert verdicts.sha256_file(tmp_path / "nope.bin") is None


def test_record_persists_across_reload(verdicts):
    # A fresh _load() (simulating a restart) still sees the record: the write is durable.
    verdicts.record_clean("acme/model", "commit1", {"a/pytorch_model.bin": _SHA})
    assert verdicts._load()["records"]["acme/model"]["pickles"] == {"a/pytorch_model.bin": _SHA}

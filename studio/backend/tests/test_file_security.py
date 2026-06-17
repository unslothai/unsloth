# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the malware / unsafe-file gate (utils.security.file_security).

The gate queries Hugging Face's own security scan
(``model_info(securityStatus=True).security_repo_status``) -- metadata only, it
never downloads the flagged files. Only the Hub call is stubbed (no network).
Policy: hard block on unsafe/suspicious/malicious; fail open when the scan is
unavailable; skip local paths and GGUF; no first-party exemption.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils.security import evaluate_file_security


def _patch_status(status):
    """Patch huggingface_hub.model_info to return one fixed security_repo_status."""
    def _mi(*_args, **_kwargs):
        return SimpleNamespace(security_repo_status = status)

    return patch("huggingface_hub.model_info", side_effect = _mi)


def _patch_raises(exc = RuntimeError("offline")):
    return patch("huggingface_hub.model_info", side_effect = exc)


@pytest.mark.parametrize("level", ["unsafe", "suspicious", "malicious"])
def test_blocks_each_blocking_level(level):
    status = {"scansDone": True, "filesWithIssues": [{"path": "pytorch_model.bin", "level": level}]}
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "pytorch_model.bin", "level": level}]
    assert d.response_payload()["security_blocked"] is True


def test_ignores_safe_only():
    status = {"scansDone": True, "filesWithIssues": [{"path": "model.safetensors", "level": "safe"}]}
    with _patch_status(status):
        d = evaluate_file_security("good/repo")
    assert d.blocked is False
    assert d.unsafe_files == []


def test_blocks_unsafe_even_when_scans_not_done():
    # scansDone is frequently False even for clean repos; a file ALREADY flagged
    # unsafe must still block (do not gate blocking on scansDone).
    status = {"scansDone": False, "filesWithIssues": [{"path": "x.pkl", "level": "unsafe"}]}
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True


def test_fail_open_when_scan_unavailable():
    # model_info returns no security_repo_status -> unknown -> allow.
    with _patch_status(None):
        d = evaluate_file_security("unknown/repo")
    assert d.blocked is False


def test_fail_open_on_exception_offline():
    with _patch_raises():
        d = evaluate_file_security("offline/repo")
    assert d.blocked is False


def test_fail_open_scans_done_no_issues():
    with _patch_status({"scansDone": True, "filesWithIssues": []}):
        d = evaluate_file_security("clean/repo")
    assert d.blocked is False


def test_skips_local_path():
    # A local path has no Hub scan; must not even call model_info.
    with patch("huggingface_hub.model_info", side_effect = AssertionError("should not be called")):
        d = evaluate_file_security("/tmp/some/local/model")
    assert d.blocked is False
    assert "local" in d.reason


def test_skips_gguf():
    with patch("huggingface_hub.model_info", side_effect = AssertionError("should not be called")):
        d = evaluate_file_security("unsloth/Model-GGUF/model.gguf")
    assert d.blocked is False


def test_no_first_party_exemption():
    # A poisoned pickle in a first-party repo still blocks (compromised-repo defense).
    status = {"scansDone": True, "filesWithIssues": [{"path": "pytorch_model.bin", "level": "unsafe"}]}
    with _patch_status(status):
        d = evaluate_file_security("unsloth/some-model")
    assert d.blocked is True


def test_malformed_entries_are_ignored():
    status = {"scansDone": True, "filesWithIssues": ["not-a-dict", {"path": "ok.pkl", "level": "unsafe"}]}
    with _patch_status(status):
        d = evaluate_file_security("evil/repo")
    assert d.blocked is True
    assert d.unsafe_files == [{"path": "ok.pkl", "level": "unsafe"}]


def test_response_payload_shape():
    status = {"scansDone": True, "filesWithIssues": [{"path": "a.pkl", "level": "malicious"}]}
    with _patch_status(status):
        payload = evaluate_file_security("evil/repo").response_payload()
    assert set(payload) == {"unsafe_files", "security_blocked", "reason"}
    assert payload["security_blocked"] is True
    assert payload["unsafe_files"] == [{"path": "a.pkl", "level": "malicious"}]

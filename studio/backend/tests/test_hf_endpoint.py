# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for HF_ENDPOINT / HF_DATASETS_SERVER env var handling in utils.hf_endpoint.

Both getters read the environment on every call (no import-time freezing), so
the tests just monkeypatch the environment and call them directly.
"""

from __future__ import annotations

import logging

import pytest

from utils.hf_endpoint import get_hf_datasets_server, get_hf_endpoint

OFFICIAL_HF = "https://huggingface.co"
OFFICIAL_DS = "https://datasets-server.huggingface.co"


@pytest.fixture(autouse = True)
def _isolate_env(monkeypatch):
    """Start every test from both vars unset, and reset the once-only warn flag."""
    monkeypatch.delenv("HF_ENDPOINT", raising = False)
    monkeypatch.delenv("HF_DATASETS_SERVER", raising = False)
    import utils.hf_endpoint as _mod

    monkeypatch.setattr(_mod, "_ds_mirror_warned", False)
    monkeypatch.setattr(_mod, "_rejected_warned", set())
    yield


class TestGetHfEndpoint:
    def test_default_when_unset(self):
        assert get_hf_endpoint() == OFFICIAL_HF

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_falls_back_to_default(self, monkeypatch, blank):
        monkeypatch.setenv("HF_ENDPOINT", blank)
        assert get_hf_endpoint() == OFFICIAL_HF

    @pytest.mark.parametrize(
        "mirror",
        [
            "https://hf-mirror.com",
            "https://hf-mirror.com/",  # trailing slash stripped
            "hf-mirror.com",  # scheme-less gets https://
            "hf-mirror.com/",  # scheme-less + trailing slash
        ],
    )
    def test_mirror_forms_normalised(self, monkeypatch, mirror):
        monkeypatch.setenv("HF_ENDPOINT", mirror)
        assert get_hf_endpoint() == "https://hf-mirror.com"

    def test_reads_env_per_call(self, monkeypatch):
        """The getter must pick up env changes without a re-import."""
        assert get_hf_endpoint() == OFFICIAL_HF
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        assert get_hf_endpoint() == "https://hf-mirror.com"


class TestGetHfDatasetsServer:
    def test_default_when_unset(self):
        assert get_hf_datasets_server() == OFFICIAL_DS

    def test_explicit_override(self, monkeypatch):
        monkeypatch.setenv("HF_DATASETS_SERVER", "https://ds.example.com")
        assert get_hf_datasets_server() == "https://ds.example.com"

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_blank_falls_back_to_default(self, monkeypatch, raw):
        monkeypatch.setenv("HF_DATASETS_SERVER", raw)
        assert get_hf_datasets_server() == OFFICIAL_DS

    @pytest.mark.parametrize(
        "raw",
        [
            "https://ds.example.com/",
            "ds.example.com",  # scheme-less gets https://
            "ds.example.com/",  # scheme-less + trailing slash
        ],
    )
    def test_forms_normalised(self, monkeypatch, raw):
        monkeypatch.setenv("HF_DATASETS_SERVER", raw)
        assert get_hf_datasets_server() == "https://ds.example.com"

    def test_mirror_hub_does_not_redirect_datasets_server(self, monkeypatch, caplog):
        """A mirrored HF_ENDPOINT alone must not point datasets-server at the mirror."""
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            assert get_hf_datasets_server() == OFFICIAL_DS
        assert any("HF_DATASETS_SERVER" in r.message for r in caplog.records)

    def test_mirror_warning_emitted_once(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            get_hf_datasets_server()
            get_hf_datasets_server()
        warnings = [r for r in caplog.records if "HF_DATASETS_SERVER" in r.message]
        assert len(warnings) == 1

    def test_no_warning_on_default_endpoint(self, caplog):
        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            get_hf_datasets_server()
        assert not [r for r in caplog.records if "HF_DATASETS_SERVER" in r.message]

    def test_no_warning_when_datasets_server_explicit(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_DATASETS_SERVER", "https://ds.example.com")
        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            get_hf_datasets_server()
        assert not [r for r in caplog.records if "HF_DATASETS_SERVER" in r.message]


# These values are interpolated into request URLs *and* into the CSP connect-src
# directive built in main.py. A CSP source list is whitespace-separated and
# semicolon-delimited, so anything carrying those characters would widen the
# policy rather than name one origin.
HOSTILE_ENDPOINTS = [
    "https://hf-mirror.com; script-src *",
    "https://hf-mirror.com *",
    "https://hf-mirror.com\nscript-src *",
    "https://hf-mirror.com\r\nscript-src *",
    "https://hf-mirror.com\tfoo",
    "https://hf-mirror.com,https://evil.com",
    "https://hf-mirror.com'",
    'https://hf-mirror.com"',
    # An embedded NUL is not listed: os.environ rejects it before we ever see it.
]

MALFORMED_ENDPOINTS = [
    "javascript:alert(1)",
    "file:///etc/passwd",
    "data:text/html,x",
    "ftp://hf-mirror.com",
    "https://",
    "https://user:pass@hf-mirror.com",
    "https://hf-mirror.com?x=1",
    "https://hf-mirror.com#frag",
    "https://hf-mirror.com:",
    "https://hf-mirror.com:not-a-port",
    # "*" would reach the CSP connect-src as "https://*", which allows every
    # https origin -- the opposite of what the policy exists for.
    "*",
    "https://*",
    "https://*.evil.com",
]


class TestRejectsUnusableEndpoints:
    @pytest.mark.parametrize("raw", HOSTILE_ENDPOINTS + MALFORMED_ENDPOINTS)
    def test_hub_endpoint_falls_back_to_official(self, monkeypatch, raw):
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == OFFICIAL_HF

    @pytest.mark.parametrize("raw", HOSTILE_ENDPOINTS + MALFORMED_ENDPOINTS)
    def test_datasets_server_falls_back_to_official(self, monkeypatch, raw):
        monkeypatch.setenv("HF_DATASETS_SERVER", raw)
        assert get_hf_datasets_server() == OFFICIAL_DS

    def test_rejection_is_logged(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com; script-src *")
        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            get_hf_endpoint()
        assert any("HF_ENDPOINT" in r.getMessage() for r in caplog.records)

    def test_rejection_logged_once_not_per_call(self, monkeypatch, caplog):
        """_build_csp runs per response, so a per-call warning would flood the log."""
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com; script-src *")
        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            for _ in range(5):
                get_hf_endpoint()
        assert len([r for r in caplog.records if "HF_ENDPOINT" in r.getMessage()]) == 1

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("https://hub.internal:8443", "https://hub.internal:8443"),
            ("http://localhost:8080", "http://localhost:8080"),
            ("https://hub.internal/hf", "https://hub.internal/hf"),
            ("https://hub.internal/hf/", "https://hub.internal/hf"),
            ("hub.internal:8443", "https://hub.internal:8443"),
        ],
    )
    def test_legitimate_forms_survive(self, monkeypatch, raw, expected):
        """Ports, http for a LAN mirror, and a path prefix are all valid mirrors."""
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == expected

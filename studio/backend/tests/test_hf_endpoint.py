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

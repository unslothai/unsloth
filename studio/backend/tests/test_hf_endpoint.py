# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for HF_ENDPOINT / HF_DATASETS_SERVER env var handling in utils.hf_endpoint.

The module reads both env vars at *import time* into module-level
variables, so every test pops the module from ``sys.modules`` and
re-imports it under a fresh, monkeypatched environment.
"""

from __future__ import annotations

import logging
import sys

import pytest

OFFICIAL_HF = "https://huggingface.co"
OFFICIAL_DS = "https://datasets-server.huggingface.co"


@pytest.fixture(autouse = True)
def _clean_hf_endpoint_module():
    """Drop any cached ``utils.hf_endpoint`` before and after each test.

    Without this, a mirror value set by one test leaks into the next
    because the module is cached in ``sys.modules`` with its env-derived
    constants already frozen.
    """
    sys.modules.pop("utils.hf_endpoint", None)
    yield
    sys.modules.pop("utils.hf_endpoint", None)


def _reload(
    monkeypatch,
    *,
    endpoint: str | None = None,
    datasets_server: str | None = None,
):
    """(Re-)import ``utils.hf_endpoint`` under a controlled environment.

    ``None`` means *unset*. Empty/whitespace strings are passed through
    so the fallback logic can be exercised.
    """
    sys.modules.pop("utils.hf_endpoint", None)

    if endpoint is None:
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
    else:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)

    if datasets_server is None:
        monkeypatch.delenv("HF_DATASETS_SERVER", raising = False)
    else:
        monkeypatch.setenv("HF_DATASETS_SERVER", datasets_server)

    import utils.hf_endpoint as hf_endpoint

    return hf_endpoint


# ---------------------------------------------------------------------------
# get_hf_endpoint() — HF_ENDPOINT resolution
# ---------------------------------------------------------------------------


class TestGetHfEndpoint:
    """HF_ENDPOINT resolution into get_hf_endpoint()."""

    def test_unset_uses_default(self, monkeypatch):
        mod = _reload(monkeypatch)
        assert mod.get_hf_endpoint() == OFFICIAL_HF

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        mod = _reload(monkeypatch, endpoint = "")
        assert mod.get_hf_endpoint() == OFFICIAL_HF

    def test_whitespace_only_falls_back_to_default(self, monkeypatch):
        mod = _reload(monkeypatch, endpoint = "   ")
        assert mod.get_hf_endpoint() == OFFICIAL_HF

    def test_mirror_value_is_used(self, monkeypatch):
        mod = _reload(monkeypatch, endpoint = "https://hf-mirror.com")
        assert mod.get_hf_endpoint() == "https://hf-mirror.com"

    def test_trailing_slash_is_stripped(self, monkeypatch):
        mod = _reload(monkeypatch, endpoint = "https://hf-mirror.com/")
        assert mod.get_hf_endpoint() == "https://hf-mirror.com"


# ---------------------------------------------------------------------------
# get_hf_datasets_server() — HF_DATASETS_SERVER resolution
# ---------------------------------------------------------------------------


class TestGetHfDatasetsServer:
    """HF_DATASETS_SERVER resolution into get_hf_datasets_server().

    The critical contract covered here: when ``HF_ENDPOINT`` is a mirror
    but ``HF_DATASETS_SERVER`` is unset, the function must fall back to
    the official datasets-server — NOT silently reuse the Hub mirror
    URL. Most Hub mirrors do not proxy ``/splits`` and the prior
    fallback caused every dataset-splits lookup to 404.
    """

    def test_both_unset_returns_official(self, monkeypatch):
        mod = _reload(monkeypatch)
        assert mod.get_hf_datasets_server() == OFFICIAL_DS

    def test_explicit_value_wins(self, monkeypatch):
        mod = _reload(
            monkeypatch,
            datasets_server = "https://my-datasets.internal",
        )
        assert mod.get_hf_datasets_server() == "https://my-datasets.internal"

    def test_empty_string_falls_back_to_official(self, monkeypatch):
        mod = _reload(monkeypatch, datasets_server = "")
        assert mod.get_hf_datasets_server() == OFFICIAL_DS

    def test_whitespace_only_falls_back_to_official(self, monkeypatch):
        mod = _reload(monkeypatch, datasets_server = "   ")
        assert mod.get_hf_datasets_server() == OFFICIAL_DS

    def test_trailing_slash_is_stripped(self, monkeypatch):
        mod = _reload(
            monkeypatch,
            datasets_server = "https://my-datasets.internal/",
        )
        assert mod.get_hf_datasets_server() == "https://my-datasets.internal"

    def test_mirror_without_datasets_server_returns_official(self, monkeypatch):
        """Regression guard: mirrored HF_ENDPOINT must NOT leak into datasets-server."""
        mod = _reload(monkeypatch, endpoint = "https://hf-mirror.com")
        assert mod.get_hf_datasets_server() == OFFICIAL_DS
        # And the hub endpoint did switch to the mirror — proves the
        # two settings are decoupled.
        assert mod.get_hf_endpoint() == "https://hf-mirror.com"

    def test_mirror_with_explicit_datasets_server(self, monkeypatch):
        mod = _reload(
            monkeypatch,
            endpoint = "https://hf-mirror.com",
            datasets_server = "https://my-datasets.internal",
        )
        assert mod.get_hf_endpoint() == "https://hf-mirror.com"
        assert mod.get_hf_datasets_server() == "https://my-datasets.internal"


# ---------------------------------------------------------------------------
# Startup WARNING — emitted when a mirror is configured without an
# explicit HF_DATASETS_SERVER so operators notice the safe fallback.
# ---------------------------------------------------------------------------


class TestStartupWarning:
    """Module-import-time warning about mirror + missing HF_DATASETS_SERVER."""

    def _import_with_caplog(
        self,
        monkeypatch,
        caplog,
        *,
        endpoint = None,
        datasets_server = None,
    ):
        sys.modules.pop("utils.hf_endpoint", None)
        if endpoint is None:
            monkeypatch.delenv("HF_ENDPOINT", raising = False)
        else:
            monkeypatch.setenv("HF_ENDPOINT", endpoint)
        if datasets_server is None:
            monkeypatch.delenv("HF_DATASETS_SERVER", raising = False)
        else:
            monkeypatch.setenv("HF_DATASETS_SERVER", datasets_server)

        with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
            import utils.hf_endpoint  # noqa: F401

        return [
            r
            for r in caplog.records
            if r.name == "utils.hf_endpoint" and r.levelno == logging.WARNING
        ]

    def test_warning_fired_for_mirror_without_explicit_datasets_server(
        self,
        monkeypatch,
        caplog,
    ):
        records = self._import_with_caplog(
            monkeypatch,
            caplog,
            endpoint = "https://hf-mirror.com",
        )
        assert (
            records
        ), "expected a WARNING on import when mirror is set without HF_DATASETS_SERVER"
        joined = " ".join(r.getMessage() for r in records)
        assert "HF_DATASETS_SERVER" in joined
        assert "hf-mirror.com" in joined

    def test_no_warning_for_default_endpoint(self, monkeypatch, caplog):
        records = self._import_with_caplog(monkeypatch, caplog)
        assert records == []

    def test_no_warning_when_datasets_server_set_explicitly(
        self,
        monkeypatch,
        caplog,
    ):
        records = self._import_with_caplog(
            monkeypatch,
            caplog,
            endpoint = "https://hf-mirror.com",
            datasets_server = "https://my-datasets.internal",
        )
        assert records == []

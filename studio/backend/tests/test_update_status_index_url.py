# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The update check must resolve versions from the configured package index.

Issue #13737: the installer hard-coded pypi.org instead of respecting
``PIP_INDEX_URL`` / ``UV_INDEX_URL`` (and their extra-index twins). These
tests pin which index URL the latest-version check consults with those
variables set and unset, without any network access.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.update_status as update_status  # noqa: E402

_INDEX_ENV_VARS = (
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)

MIRROR = "https://mirror.corp.example/simple"
EXTRA = "https://extra.corp.example/simple"


@pytest.fixture(autouse = True)
def _no_index_env(monkeypatch):
    for name in _INDEX_ENV_VARS:
        monkeypatch.delenv(name, raising = False)
    update_status.reset_update_status_cache()
    yield
    update_status.reset_update_status_cache()


def test_unset_env_falls_back_to_pypi():
    assert update_status.configured_package_index_urls() == ["https://pypi.org/simple"]
    assert update_status.package_index_url() == "https://pypi.org/simple"
    assert update_status._index_metadata_url(update_status.package_index_url()) == (
        "https://pypi.org/pypi/unsloth/json"
    )


def test_pip_index_url_is_honored(monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", MIRROR)
    assert update_status.package_index_url() == MIRROR
    assert update_status._index_metadata_url(MIRROR) == MIRROR + "/unsloth/"


def test_pip_extra_index_url_is_consulted_in_addition(monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", MIRROR)
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", EXTRA)
    assert update_status.configured_package_index_urls() == [MIRROR, EXTRA]


def test_uv_index_url_is_honored(monkeypatch):
    monkeypatch.setenv("UV_INDEX_URL", MIRROR)
    assert update_status.package_index_url() == MIRROR


def test_uv_default_index_is_honored(monkeypatch):
    monkeypatch.setenv("UV_DEFAULT_INDEX", MIRROR)
    assert update_status.package_index_url() == MIRROR


def test_uv_extra_index_url_is_consulted_in_addition(monkeypatch):
    monkeypatch.setenv("UV_INDEX_URL", MIRROR)
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", EXTRA)
    assert update_status.configured_package_index_urls() == [MIRROR, EXTRA]


def test_extra_index_alone_adds_to_the_pypi_primary(monkeypatch):
    # An extra index adds to the primary; it never replaces it, so with no
    # primary set PyPI stays first and the extra is consulted in addition.
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", EXTRA)
    assert update_status.configured_package_index_urls() == ["https://pypi.org/simple", EXTRA]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _stub_urlopen(monkeypatch, payloads):
    seen = []

    def fake_urlopen(request, timeout = None):
        seen.append(request.full_url)
        return _Response(json.dumps(payloads[request.full_url]).encode("utf-8"))

    monkeypatch.setattr(update_status.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_fetch_queries_pypi_json_api_by_default(monkeypatch):
    seen = _stub_urlopen(
        monkeypatch,
        {"https://pypi.org/pypi/unsloth/json": {"info": {"version": "1.2.3"}}},
    )
    result = update_status._fetch_latest_pypi_version()
    assert seen == ["https://pypi.org/pypi/unsloth/json"]
    assert result.latest_version == "1.2.3"


def test_fetch_queries_configured_uv_index_not_pypi(monkeypatch):
    monkeypatch.setenv("UV_INDEX_URL", MIRROR)
    seen = _stub_urlopen(
        monkeypatch,
        {MIRROR + "/unsloth/": {"versions": ["1.0.0", "1.2.3", "1.3.0rc1"]}},
    )
    result = update_status._fetch_latest_pypi_version()
    assert seen == [MIRROR + "/unsloth/"]
    # PEP 691 versions list: newest stable wins over a newer prerelease.
    assert result.latest_version == "1.2.3"


def test_fetch_queries_configured_pip_index_not_pypi(monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", MIRROR)
    seen = _stub_urlopen(
        monkeypatch,
        {MIRROR + "/unsloth/": {"info": {"version": "2.0.0"}}},
    )
    result = update_status._fetch_latest_pypi_version()
    assert seen == [MIRROR + "/unsloth/"]
    assert result.latest_version == "2.0.0"

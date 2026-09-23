# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for HF_ENDPOINT / HF_DATASETS_SERVER env var handling in utils.hf_endpoint.

Both getters read the environment on every call (no import-time freezing), so
the tests just monkeypatch the environment and call them directly.
"""

from __future__ import annotations

import logging
import os

import pytest

from utils import hf_endpoint
from utils.hf_endpoint import (
    client_reachable_endpoint,
    get_hf_datasets_server,
    get_hf_endpoint,
)

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
    monkeypatch.setattr(_mod, "_unreachable_warned", set())
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


# These values reach the CSP connect-src built in main.py, and a source list is
# whitespace-separated and semicolon-delimited.
HOSTILE_ENDPOINTS = [
    "https://hf-mirror.com; script-src *",
    "https://hf-mirror.com *",
    "https://hf-mirror.com\nscript-src *",
    "https://hf-mirror.com\r\nscript-src *",
    "https://hf-mirror.com\tfoo",
    "https://hf-mirror.com,https://evil.com",
    "https://hf-mirror.com'",
    'https://hf-mirror.com"',
    # No NUL case: os.environ rejects it before we see it.
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
    # An IPv6 literal written without its brackets, and the other authorities that
    # make SplitResult.port raise. The rejection has to come from this module, not
    # from an exception out of main.py's startup call to normalize_hf_endpoint_env.
    "https://::1",
    "https://a:b:c",
    "https://hf-mirror.com:99999",
    "https://[::1",
    # "https://*" as a CSP source allows every https origin.
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


class TestMalformedUrlDoesNotCrash:
    """`_build_csp` runs on every response, so a raise here is a 500 for every request."""

    @pytest.mark.parametrize(
        "raw", ["https://[", "https://[::1", "https://[bad]", "http://[", "https://a[b"]
    )
    def test_bracketed_host_falls_back_instead_of_raising(self, monkeypatch, raw):
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == OFFICIAL_HF
        monkeypatch.delenv("HF_ENDPOINT")
        monkeypatch.setenv("HF_DATASETS_SERVER", raw)
        assert get_hf_datasets_server() == OFFICIAL_DS


class TestPlainHttpIsLoopbackOnly:
    """The frontend attaches the Hub token to these requests, so off-box HTTP
    would put a bearer token on the wire in cleartext."""

    @pytest.mark.parametrize(
        "raw",
        ["http://127.0.0.1:9700", "http://localhost:8080", "http://127.1.2.3", "http://[::1]:9000"],
    )
    def test_loopback_http_is_kept(self, monkeypatch, raw):
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == raw

    @pytest.mark.parametrize(
        "raw", ["http://192.168.1.10:8080", "http://hf-mirror.com", "http://10.0.0.5:8080"]
    )
    def test_off_box_http_is_refused(self, monkeypatch, raw):
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == OFFICIAL_HF

    @pytest.mark.parametrize("raw", ["https://192.168.1.10:8080", "https://hf-mirror.com"])
    def test_https_to_the_same_hosts_is_fine(self, monkeypatch, raw):
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == raw


class TestAssetSources:
    def test_no_asset_sources_without_a_mirror(self):
        from utils.hf_endpoint import csp_asset_sources
        assert csp_asset_sources() == ()

    def test_an_https_mirror_needs_none(self, monkeypatch):
        """img-src/media-src already carry a bare https:."""
        from utils.hf_endpoint import csp_asset_sources

        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        assert csp_asset_sources() == ()

    def test_a_loopback_http_mirror_needs_one(self, monkeypatch):
        from utils.hf_endpoint import csp_asset_sources
        monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")
        assert csp_asset_sources() == ("http://127.0.0.1:9700",)


def test_an_uppercase_scheme_is_accepted_and_folded(monkeypatch):
    """RFC 3986 3.1: HTTPS://mirror is a real mirror, and the frontend keys its
    cache on the folded form."""
    monkeypatch.setenv("HF_ENDPOINT", "HTTPS://hf-mirror.com")
    assert get_hf_endpoint() == "https://hf-mirror.com"
    monkeypatch.setenv("HF_ENDPOINT", "HTTP://127.0.0.1:9700")
    assert get_hf_endpoint() == "http://127.0.0.1:9700"
    monkeypatch.setenv("HF_ENDPOINT", "HTTP://hf-mirror.com")
    assert get_hf_endpoint() == "https://huggingface.co"


def test_a_loopback_endpoint_is_not_handed_to_a_remote_browser(monkeypatch):
    """A remote browser handed http://127.0.0.1:9700 opens its OWN localhost, so
    the publish link would be dead where before the feature it worked."""
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")
    assert client_reachable_endpoint("127.0.0.1") == "http://127.0.0.1:9700"
    assert client_reachable_endpoint("::1") == "http://127.0.0.1:9700"
    assert client_reachable_endpoint("192.168.1.50") == "https://huggingface.co"
    assert client_reachable_endpoint(None) == "https://huggingface.co"
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    assert client_reachable_endpoint("192.168.1.50") == "https://hf-mirror.com"


def test_the_reachable_endpoint_follows_the_tunnel_aware_client_ip(monkeypatch):
    """Every caller pairs this with client_ip(): through the managed tunnel the
    socket peer is the local cloudflared process, not the visitor."""
    from types import SimpleNamespace

    from utils.client_ip import client_ip

    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")

    def request(peer: str, headers: dict | None = None):
        return SimpleNamespace(client = SimpleNamespace(host = peer), headers = headers or {})

    local = request("127.0.0.1")
    assert client_reachable_endpoint(client_ip(local)) == "http://127.0.0.1:9700"

    tunneled = request("127.0.0.1", {"cf-connecting-ip": "203.0.113.7"})
    assert client_reachable_endpoint(client_ip(tunneled)) == "https://huggingface.co"

    lan = request("192.168.1.50")
    assert client_reachable_endpoint(client_ip(lan)) == "https://huggingface.co"


def test_a_unicode_host_cannot_reach_the_csp_header(monkeypatch):
    """Starlette encodes header values as latin-1, so a Unicode host in connect-src
    turns EVERY response into a 500. All three sides take the punycode form."""
    monkeypatch.setenv("HF_ENDPOINT", "https://例子.测试")
    assert get_hf_endpoint() == "https://huggingface.co"
    monkeypatch.setenv("HF_ENDPOINT", "https://xn--fsqu00a.xn--0zwm56d")
    assert get_hf_endpoint() == "https://xn--fsqu00a.xn--0zwm56d"


def test_an_ipv6_loopback_mirror_is_compressed_the_way_the_browser_sends_it(monkeypatch):
    """A host-source is matched as a string (CSP3 6.7.2.5) and the browser sends
    http://[::1]:9700 whichever spelling was configured."""
    for raw in ("http://[0:0:0:0:0:0:0:1]:9700", "http://[::1]:9700"):
        monkeypatch.setenv("HF_ENDPOINT", raw)
        assert get_hf_endpoint() == "http://[::1]:9700", raw
    monkeypatch.setenv("HF_ENDPOINT", "http://[0:0:0:0:0:0:0:1]")
    assert get_hf_endpoint() == "http://[::1]"
    monkeypatch.setenv("HF_ENDPOINT", "http://[2001:db8::1]:9700")
    assert get_hf_endpoint() == "https://huggingface.co"


def test_the_environment_is_normalised_for_huggingface_hub(monkeypatch):
    """huggingface_hub reads HF_ENDPOINT itself, at import, unvalidated: a
    scheme-less value breaks every HfApi call and a rejected one would still be
    handed the user's token. Rewriting the variable first settles both."""
    monkeypatch.setenv("HF_ENDPOINT", "hf-mirror.com")
    hf_endpoint.normalize_hf_endpoint_env()
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    hf_endpoint.normalize_hf_endpoint_env()
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    # Removed, so the library falls back rather than using a host we refused.
    for rejected in (
        "http://192.168.1.10:8080",
        "https://hf-mirror.com; script-src *",
        "https://例子.测试",
        "*",
    ):
        monkeypatch.setenv("HF_ENDPOINT", rejected)
        hf_endpoint.normalize_hf_endpoint_env()
        assert "HF_ENDPOINT" not in os.environ, rejected

    monkeypatch.delenv("HF_ENDPOINT", raising = False)
    hf_endpoint.normalize_hf_endpoint_env()
    assert "HF_ENDPOINT" not in os.environ


def test_a_blank_endpoint_is_cleared_rather_than_left_for_the_library(monkeypatch):
    """huggingface_hub's os.getenv("HF_ENDPOINT", default) falls back only when the key
    is ABSENT, and its rstrip("/") does not touch whitespace, so "   " reached the
    library verbatim while Studio served the default. datasets/config.py is the same."""
    for blank in ("", "   ", "\t", "\n", " \t\n "):
        monkeypatch.setenv("HF_ENDPOINT", blank)
        hf_endpoint.normalize_hf_endpoint_env()
        assert "HF_ENDPOINT" not in os.environ, repr(blank)

    # Idempotent, and it does not invent the variable when it was never set.
    hf_endpoint.normalize_hf_endpoint_env()
    assert "HF_ENDPOINT" not in os.environ

    # Whitespace AROUND a real endpoint is trimmed, not treated as blank.
    monkeypatch.setenv("HF_ENDPOINT", "  https://hf-mirror.com  ")
    hf_endpoint.normalize_hf_endpoint_env()
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_a_private_endpoint_reaches_only_a_client_on_a_local_network(monkeypatch):
    """A private address means the VISITOR's network when the visitor is
    elsewhere, so it is handed out only to a client that is itself local."""
    monkeypatch.setenv("HF_ENDPOINT", "https://10.0.0.5:8443")
    assert client_reachable_endpoint("192.168.1.50") == "https://10.0.0.5:8443"
    assert client_reachable_endpoint("127.0.0.1") == "https://10.0.0.5:8443"
    assert client_reachable_endpoint("8.8.8.8") == "https://huggingface.co"
    assert client_reachable_endpoint(None) == "https://huggingface.co"
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")
    assert client_reachable_endpoint("192.168.1.50") == "https://huggingface.co"
    assert client_reachable_endpoint("127.0.0.1") == "http://127.0.0.1:9700"
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.internal")
    assert client_reachable_endpoint("8.8.8.8") == "https://hub.internal"


def test_the_per_client_fallback_is_logged_once_per_endpoint(monkeypatch, caplog):
    """Both /api/health and the publish link call this on every request."""
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")
    with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
        for _ in range(3):
            assert client_reachable_endpoint("8.8.8.8") == OFFICIAL_HF
    assert len([r for r in caplog.records if "not reachable" in r.getMessage()]) == 1

    # A different endpoint is a different configuration, so it warns on its own.
    caplog.clear()
    monkeypatch.setenv("HF_ENDPOINT", "https://10.0.0.5:8443")
    with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
        assert client_reachable_endpoint("8.8.8.8") == OFFICIAL_HF
    assert len([r for r in caplog.records if "not reachable" in r.getMessage()]) == 1

    # A client that CAN reach it is not a fallback and must not warn.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger = "utils.hf_endpoint"):
        assert client_reachable_endpoint("192.168.1.50") == "https://10.0.0.5:8443"
    assert not [r for r in caplog.records if "not reachable" in r.getMessage()]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract tests for the server-side Hugging Face discovery route."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from fastapi import HTTPException

from hub.routes import discovery


class _Params:
    """Stand-in for Starlette QueryParams (a multi-dict)."""

    def __init__(self, items):
        self._items = list(items)

    def multi_items(self):
        return list(self._items)


class _Request:
    def __init__(
        self,
        items,
        base_url = "http://studio.local:1234/",
    ):
        self.query_params = _Params(items)
        self.base_url = base_url


def _call(items, hf_token = None):
    return asyncio.run(
        discovery.discovery_search(
            "models",
            _Request(items),
            hf_token = hf_token,
            current_subject = "tester",
        )
    )


class TestQueryAllowlist:
    def test_known_parameters_pass_through(self):
        pairs = discovery.build_discovery_query(
            _Params([("search", "gemma"), ("sort", "downloads"), ("direction", "-1")])
        )
        assert ("search", "gemma") in pairs
        assert ("sort", "downloads") in pairs

    @pytest.mark.parametrize(
        "items",
        [
            [("endpoint", "http://169.254.169.254/latest/meta-data")],
            [("url", "http://127.0.0.1:8888/api/auth")],
            [("hubUrl", "https://evil.example")],
            [("next", "https://evil.example/api/models")],
        ],
    )
    def test_unknown_parameters_are_rejected(self, items):
        with pytest.raises(discovery.DiscoveryQueryError):
            discovery.build_discovery_query(_Params(items))

    def test_duplicate_scalars_are_rejected(self):
        with pytest.raises(discovery.DiscoveryQueryError):
            discovery.build_discovery_query(_Params([("search", "a"), ("search", "b")]))

    def test_repeated_filters_are_bounded(self):
        items = [("filter", f"tag-{i}") for i in range(discovery._MAX_REPEATED_VALUES + 1)]
        with pytest.raises(discovery.DiscoveryQueryError):
            discovery.build_discovery_query(_Params(items))

    @pytest.mark.parametrize(
        "items",
        [
            [("sort", "; DROP")],
            [("direction", "2")],
            [("limit", "0")],
            [("limit", "99999")],
            [("limit", "abc")],
            [("search", "x" * 500)],
        ],
    )
    def test_invalid_scalar_values_are_rejected(self, items):
        with pytest.raises(discovery.DiscoveryQueryError):
            discovery.build_discovery_query(_Params(items))


class TestDestinationIsServerControlled:
    def test_base_comes_from_hf_endpoint(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
        url = discovery.build_upstream_url("models", [("search", "gemma")])
        assert url.startswith("https://hf-mirror.example/api/models?")

    def test_blank_endpoint_falls_back_to_the_official_hub(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "   ")
        url = discovery.build_upstream_url("models", [])
        assert url == "https://huggingface.co/api/models"

    def test_no_request_parameter_can_redirect_the_destination(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        # Every field a caller controls is either rejected outright or lands in
        # the query string; none of them may change the host.
        pairs = discovery.build_discovery_query(
            _Params([("search", "https://evil.example"), ("author", "..")])
        )
        url = discovery.build_upstream_url("models", pairs)
        assert url.startswith("https://huggingface.co/api/models?")
        assert "evil.example/api" not in url


class TestUpstreamFailureMapping:
    def test_upstream_401_is_not_surfaced_as_401(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (401, b"", ""))
        with pytest.raises(HTTPException) as excinfo:
            _call([("search", "gemma")])
        assert excinfo.value.status_code != 401, (
            "an HF 401 echoed as our own 401 makes authFetch clear the Studio "
            "session and log the user out"
        )
        assert excinfo.value.status_code == discovery._UPSTREAM_AUTH_STATUS

    def test_upstream_403_is_not_surfaced_as_403(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (403, b"", ""))
        with pytest.raises(HTTPException) as excinfo:
            _call([("search", "gemma")])
        assert excinfo.value.status_code == discovery._UPSTREAM_AUTH_STATUS

    def test_redirects_are_refused_not_followed(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (302, b"", ""))
        with pytest.raises(HTTPException) as excinfo:
            _call([("search", "gemma")])
        assert excinfo.value.status_code == 502
        assert "redirect" in str(excinfo.value.detail).lower()

    def test_upstream_5xx_becomes_502(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (503, b"", ""))
        with pytest.raises(HTTPException) as excinfo:
            _call([("search", "gemma")])
        assert excinfo.value.status_code == 502

    def test_malformed_json_becomes_502(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (200, b"not json", ""))
        with pytest.raises(HTTPException) as excinfo:
            _call([("search", "gemma")])
        assert excinfo.value.status_code == 502

    def test_bad_query_is_400(self):
        with pytest.raises(HTTPException) as excinfo:
            _call([("endpoint", "http://127.0.0.1")])
        assert excinfo.value.status_code == 400


class TestTokenHandling:
    def test_token_is_never_echoed_in_an_error(self, monkeypatch):
        secret = "hf_averysecrettokenvalue"

        def _boom(url, token):
            raise RuntimeError(f"connection failed using {secret}")

        monkeypatch.setattr(discovery, "_fetch_upstream", _boom)
        with pytest.raises(HTTPException) as excinfo:
            _call([("search", "gemma")], hf_token = secret)
        assert secret not in str(excinfo.value.detail)

    def test_token_is_never_echoed_in_a_success(self, monkeypatch):
        secret = "hf_averysecrettokenvalue"
        monkeypatch.setattr(
            discovery, "_fetch_upstream", lambda url, token: (200, b'[{"id":"a"}]', "")
        )
        response = _call([("search", "gemma")], hf_token = secret)
        assert secret not in response.body.decode("utf-8")
        assert secret not in repr(dict(response.headers))
        assert json.loads(response.body) == [{"id": "a"}]


class TestRouteWiring:
    def test_route_requires_authentication(self):
        # The dependency is what makes the route non-public; a missing default
        # here means anyone reaching the port can drive it.
        import inspect

        from auth.authentication import get_current_subject

        sig = inspect.signature(discovery.discovery_search)
        dep = sig.parameters["current_subject"].default
        assert getattr(dep, "dependency", None) is get_current_subject

    def test_hf_token_uses_the_shared_header_dependency(self):
        import inspect

        from hub.dependencies import get_hf_token

        sig = inspect.signature(discovery.discovery_search)
        dep = sig.parameters["hf_token"].default
        assert getattr(dep, "dependency", None) is get_hf_token


class TestPaginationLink:
    def test_next_link_is_rewritten_onto_this_route(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        header = '<https://huggingface.co/api/models?search=gemma&limit=100>; rel="next"'
        rewritten = discovery.rewrite_next_link("models", discovery.parse_next_link(header))
        assert rewritten is not None
        assert rewritten.startswith("/api/hub/discovery/models?")
        assert "huggingface.co" not in rewritten

    def test_next_link_off_the_configured_endpoint_is_dropped(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        header = '<https://evil.example/api/models?search=x>; rel="next"'
        assert (
            discovery.rewrite_next_link("models", discovery.parse_next_link(header)) is None
        ), "an off-endpoint next link must never be followed or handed onward"

    def test_next_link_with_a_rejected_param_is_dropped(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        header = '<https://huggingface.co/api/models?url=http://127.0.0.1>; rel="next"'
        assert discovery.rewrite_next_link("models", discovery.parse_next_link(header)) is None

    def test_response_carries_the_rewritten_link(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        monkeypatch.setattr(
            discovery,
            "_fetch_upstream",
            lambda url, token: (
                200,
                b'[{"id":"a"}]',
                '<https://huggingface.co/api/models?search=gemma>; rel="next"',
            ),
        )
        response = _call([("search", "gemma")])
        link = response.headers.get("link") or response.headers.get("Link")
        assert link is not None and "/api/hub/discovery/models" in link

    def test_next_link_is_absolute_so_the_hub_sdk_can_parse_it(self, monkeypatch):
        # @huggingface/hub's parseLinkHeader matches /<(https?:[/][/][^>]+)>;\s+rel="..."/
        # only, so a relative target yields no next URL and pagination stops
        # after the first upstream page.
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        monkeypatch.setattr(
            discovery,
            "_fetch_upstream",
            lambda url, token: (
                200,
                b"[]",
                '<https://huggingface.co/api/models?search=gemma&cursor=abc123>; rel="next"',
            ),
        )
        response = _call([("search", "gemma")])
        link = response.headers.get("link") or response.headers.get("Link")
        assert link is not None
        sdk_regex = re.compile(r'<(https?://[^>]+)>;\s+rel="([^"]+)"')
        match = sdk_regex.search(link)
        assert match is not None, f"the Hub SDK cannot parse {link!r}"
        assert match.group(2) == "next"
        target = match.group(1)
        assert target.startswith("http://studio.local:1234/api/hub/discovery/models?"), target
        assert "cursor=abc123" in target
        assert "huggingface.co" not in target

    def test_no_upstream_link_means_no_header(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (200, b"[]", ""))
        response = _call([("search", "gemma")])
        assert (response.headers.get("link") or response.headers.get("Link")) is None

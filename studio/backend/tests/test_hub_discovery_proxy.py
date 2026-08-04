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
        # Caller input is rejected or lands in the query string, never in the host.
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
        # Without this dependency, anyone reaching the port can drive the route.
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
        # @huggingface/hub's parseLinkHeader only matches an <http(s)://...> target,
        # so a relative one stops pagination after the first upstream page.
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

    def test_relative_next_link_from_a_mirror_is_accepted(self, monkeypatch):
        # RFC 8288 permits a relative target; a mirror emitting one must still paginate.
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
        rewritten = discovery.rewrite_next_link(
            "models",
            discovery.parse_next_link('</api/models?cursor=abc123>; rel="next"'),
            "http://127.0.0.1:8888/",
        )
        assert rewritten is not None, "a relative mirror link must resolve, not be dropped"
        assert "cursor=abc123" in rewritten

    def test_relative_link_still_cannot_escape_the_endpoint(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
        assert (
            discovery.rewrite_next_link(
                "models",
                discovery.parse_next_link('<https://evil.example/api/models>; rel="next"'),
                "http://127.0.0.1:8888/",
            )
            is None
        )


class TestInfoRoute:
    def _info(
        self,
        repo = "unsloth/gemma-3-4b-it",
        revision = "HEAD",
        items = (),
    ):
        return asyncio.run(
            discovery.discovery_info(
                "models",
                _Request(list(items)),
                repo = repo,
                revision = revision,
                hf_token = None,
                current_subject = "tester",
            )
        )

    def test_the_repo_path_is_preserved_on_the_mirror(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
        seen = {}

        def _capture(url, token):
            seen["url"] = url
            return 200, b'{"id": "x"}', ""

        monkeypatch.setattr(discovery, "_fetch_upstream", _capture)
        self._info()
        assert seen["url"].startswith("https://hf-mirror.example/api/models/"), seen
        assert "unsloth/gemma-3-4b-it/revision/HEAD" in seen["url"], seen

    @pytest.mark.parametrize("repo", ["../../etc/passwd", "a b", "", "x" * 300])
    def test_invalid_repos_are_rejected(self, repo):
        with pytest.raises(HTTPException) as excinfo:
            self._info(repo = repo)
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("revision", ["a b", "rev;rm -rf", "..\\x"])
    def test_invalid_revisions_are_rejected(self, revision):
        with pytest.raises(HTTPException) as excinfo:
            self._info(revision = revision)
        assert excinfo.value.status_code == 400

    def test_listing_filters_are_not_accepted_here(self):
        with pytest.raises(HTTPException) as excinfo:
            self._info(items = [("search", "gemma")])
        assert excinfo.value.status_code == 400


class TestStreamDeadline:
    def test_a_slow_drip_is_cut_off(self, monkeypatch):
        # requests' timeout bounds inactivity, not total time, so without a deadline
        # this drip would hold the worker thread forever.
        monkeypatch.setattr(discovery, "_REQUEST_TIMEOUT_SECONDS", 0.05)

        class _Resp:
            status_code = 200
            headers = {}

            def iter_content(self, chunk_size = 0):
                import time as _t
                while True:
                    _t.sleep(0.02)
                    yield b"x"

            def close(self):
                pass

        class _Session:
            def get(self, url, **kw):
                return _Resp()

        monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
        with pytest.raises(HTTPException) as excinfo:
            discovery._fetch_upstream("https://huggingface.co/api/models", None)
        assert excinfo.value.status_code == 504

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://HF-Mirror.example",
            "https://hf-mirror.example:443",
            "https://HF-MIRROR.EXAMPLE:443",
        ],
    )
    def test_equivalent_authorities_still_paginate(self, monkeypatch, endpoint):
        # A textual netloc check would drop the mirror's canonical link and stop
        # pagination, since case and a default port are equivalent.
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        rewritten = discovery.rewrite_next_link(
            "models",
            discovery.parse_next_link(
                '<https://hf-mirror.example/api/models?cursor=abc123>; rel="next"'
            ),
            "http://127.0.0.1:8888/",
        )
        assert rewritten is not None, endpoint
        assert "cursor=abc123" in rewritten

    @pytest.mark.parametrize(
        "target",
        [
            "https://huggingface.co:notaport/api/models",
            "https://huggingface.co:99999/api/models",
            "https://[oops/api/models",
        ],
    )
    def test_a_malformed_link_is_dropped_not_raised(self, monkeypatch, target):
        # The next-page header is optional; a bad one must not 500 a good first page.
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        assert (
            discovery.rewrite_next_link(
                "models",
                discovery.parse_next_link(f'<{target}>; rel="next"'),
                "http://127.0.0.1:8888/",
            )
            is None
        )

    def test_a_malformed_link_still_returns_the_page(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        monkeypatch.setattr(
            discovery,
            "_fetch_upstream",
            lambda url, token: (
                200,
                b'[{"id":"a"}]',
                '<https://huggingface.co:notaport/api/models>; rel="next"',
            ),
        )
        response = _call([("search", "gemma")])
        assert json.loads(response.body) == [{"id": "a"}]
        assert (response.headers.get("link") or response.headers.get("Link")) is None


class TestHealthHubEndpoint:
    def test_userinfo_never_reaches_the_client(self, monkeypatch):
        import main

        monkeypatch.setenv("HF_ENDPOINT", "https://user:password@mirror.example")
        value = main._hub_endpoint()
        assert "password" not in value and "user" not in value, value
        assert value == "https://mirror.example"

    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("https://HuggingFace.CO", "https://huggingface.co"),
            ("https://mirror.example/hf", "https://mirror.example"),
            ("", "https://huggingface.co"),
            ("   ", "https://huggingface.co"),
        ],
    )
    def test_only_a_normalised_origin_is_reported(self, monkeypatch, endpoint, expected):
        import main
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        assert main._hub_endpoint() == expected

    def test_the_socket_timeout_is_bounded_by_the_window(self, monkeypatch):
        # The timeout is per read, so the whole budget could be spent twice.
        seen = {}

        class _Resp:
            status_code = 200
            headers = {}

            def iter_content(self, chunk_size = 0):
                yield b"[]"

            def close(self):
                pass

        class _Session:
            def get(self, url, **kw):
                seen["timeout"] = kw.get("timeout")
                return _Resp()

        monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
        discovery._fetch_upstream("https://huggingface.co/api/models", None)
        assert seen["timeout"] <= discovery._REQUEST_TIMEOUT_SECONDS / 2

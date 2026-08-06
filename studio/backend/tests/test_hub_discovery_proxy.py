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


class _Err:
    """Either form of failure, so a test asserts on the outcome not the mechanism."""

    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


def _error(fn, *args, **kwargs):
    """Call a route and return its failure, whether raised or returned stamped."""
    try:
        response = fn(*args, **kwargs)
    except HTTPException as e:
        return _Err(e.status_code, e.detail)
    body = json.loads(response.body)
    return _Err(response.status_code, body.get("detail"))


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
        err = _error(_call, [("search", "gemma")])
        assert err.status_code != 401, (
            "an HF 401 echoed as our own 401 makes authFetch clear the Studio "
            "session and log the user out"
        )
        assert err.status_code == discovery._UPSTREAM_AUTH_STATUS

    def test_upstream_403_is_not_surfaced_as_403(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (403, b"", ""))
        err = _error(_call, [("search", "gemma")])
        assert err.status_code == discovery._UPSTREAM_AUTH_STATUS

    def test_redirects_are_refused_not_followed(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (302, b"", ""))
        err = _error(_call, [("search", "gemma")])
        assert err.status_code == 502
        assert "redirect" in str(err.detail).lower()

    def test_upstream_5xx_becomes_502(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (503, b"", ""))
        err = _error(_call, [("search", "gemma")])
        assert err.status_code == 502

    def test_malformed_json_becomes_502(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (200, b"not json", ""))
        err = _error(_call, [("search", "gemma")])
        assert err.status_code == 502

    def test_bad_query_is_400(self):
        err = _error(_call, [("endpoint", "http://127.0.0.1")])
        assert err.status_code == 400


class TestTokenHandling:
    def test_token_is_never_echoed_in_an_error(self, monkeypatch):
        secret = "hf_averysecrettokenvalue"

        def _boom(url, token):
            raise RuntimeError(f"connection failed using {secret}")

        monkeypatch.setattr(discovery, "_fetch_upstream", _boom)
        err = _error(_call, [("search", "gemma")], hf_token = secret)
        assert secret not in str(err.detail)

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
        err = _error(self._info, repo = repo)
        assert err.status_code == 400

    @pytest.mark.parametrize("revision", ["a b", "rev;rm -rf", "..\\x"])
    def test_invalid_revisions_are_rejected(self, revision):
        err = _error(self._info, revision = revision)
        assert err.status_code == 400

    def test_listing_filters_are_not_accepted_here(self):
        err = _error(self._info, items = [("search", "gemma")])
        assert err.status_code == 400


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
            # The path is reported now: the proxy routes fetch from the whole
            # endpoint, and the frontend resolves a card's relative assets
            # against this, so a subpath mirror needs its prefix.
            ("https://mirror.example/hf", "https://mirror.example/hf"),
            ("https://mirror.example/hf/", "https://mirror.example/hf"),
            ("https://mirror.example/hf?token=x", "https://mirror.example/hf"),
            # urlsplit().hostname drops the brackets, and the frontend parses
            # this with new URL(), which throws on a bare IPv6 literal and then
            # resolves a mirror card's assets against the public Hub.
            ("http://[fd00::1]:8080/hf", "http://[fd00::1]:8080/hf"),
            ("https://[::1]", "https://[::1]"),
            ("", "https://huggingface.co"),
            ("   ", "https://huggingface.co"),
        ],
    )
    def test_only_a_normalised_endpoint_is_reported(self, monkeypatch, endpoint, expected):
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

    def test_the_request_itself_refuses_to_follow_a_redirect(self, monkeypatch):
        """A hop could walk onto an internal address, so it is never followed.

        Every other redirect test stubs _fetch_upstream, so without this nothing
        pins the flag on the call that actually leaves the process.
        """
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
                seen["allow_redirects"] = kw.get("allow_redirects")
                return _Resp()

        monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
        discovery._fetch_upstream("https://huggingface.co/api/models", None)
        assert seen["allow_redirects"] is False


class TestTheDeadlineIsWallClock:
    """requests bounds each socket operation, not the whole call.

    A stalled DNS lookup or OS proxy discovery is not covered by that, and it is
    the environment this route exists for, so the deadline is applied around the
    threaded call instead.
    """

    def test_a_stalled_lookup_does_not_outlive_the_window(self, monkeypatch):
        import time as _time

        def _hang(url, token):
            _time.sleep(5)
            return 200, b"[]", ""

        monkeypatch.setattr(discovery, "_fetch_upstream", _hang)
        monkeypatch.setattr(discovery, "_REQUEST_TIMEOUT_SECONDS", 0.25)

        async def _timed():
            started = _time.monotonic()
            response = await discovery.discovery_search(
                "models",
                _Request([("search", "gemma")]),
                hf_token = None,
                current_subject = "tester",
            )
            return _time.monotonic() - started, response

        # Not asyncio.run: its shutdown drains the executor, so it would measure
        # the orphaned thread finishing rather than what the route made the
        # caller wait for. A real server's loop outlives the request.
        loop = asyncio.new_event_loop()
        try:
            elapsed, response = loop.run_until_complete(_timed())
        finally:
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

        assert elapsed < 3, f"the route waited {elapsed:.1f}s past its deadline"
        assert response.status_code == 504
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER) == "1"

    def test_the_timeout_is_named_rather_than_reported_as_unknown(self):
        err = discovery._upstream_error(asyncio.TimeoutError(), None)
        assert err.status_code == 504


class TestErrorDetailScrubbing:
    def test_the_search_query_is_not_echoed_in_a_transport_error(self, monkeypatch):
        # requests names the failing URL in its message, and for this proxy that
        # URL carries the user's search terms.
        def _boom(url, token):
            # The real requests/urllib3 shape, which names the target twice.
            raise RuntimeError(
                "HTTPSConnectionPool(host='hf-mirror.acme.internal', port=443): Max "
                "retries exceeded with url: /api/models?search=acme-internal&limit=100"
            )

        monkeypatch.setattr(discovery, "_fetch_upstream", _boom)
        err = _error(_call, [("search", "acme-internal")])
        detail = str(err.detail)
        assert "acme-internal" not in detail
        assert "/api/models?" not in detail
        assert "hf-mirror" not in detail

    def test_the_egress_proxy_host_is_not_echoed(self, monkeypatch):
        # A proxied failure names the internal proxy, which we report nowhere else.
        def _boom(url, token):
            raise RuntimeError(
                "ProxyError('Unable to connect to proxy', NameResolutionError("
                "\"HTTPSConnection(host='squid.corp.internal', port=3128): "
                'Failed to resolve"))'
            )

        monkeypatch.setattr(discovery, "_fetch_upstream", _boom)
        err = _error(_call, [("search", "gemma")])
        assert "squid.corp.internal" not in str(err.detail)

    def test_scrubbing_keeps_the_diagnosis_readable(self):
        cleaned = discovery._scrub_detail(
            "HTTPSConnectionPool(host='h.example', port=443): Max retries exceeded "
            "with url: /api/models?search=x",
            None,
        )
        assert (
            "Max retries exceeded" in cleaned
        ), "stripping the URL must not strip the reason the request failed"


class TestResponseSizeCap:
    def test_the_cap_clears_a_real_listing(self):
        # Measured against the live Hub with the shape the feed actually sends
        # (limit=500, the SDK keys plus ALL_FIELDS): 7.7 MB unfiltered and
        # 9.9 MB for filter=text-generation. The previous 8 MiB cap 502'd the
        # latter, so this pins the headroom rather than leaving it to taste.
        measured_worst_case = 9_879_823
        assert discovery._MAX_RESPONSE_BYTES > measured_worst_case

    def test_the_cap_still_bounds_memory(self):
        assert discovery._MAX_RESPONSE_BYTES <= 32 * 1024 * 1024


class TestReadmeRoute:
    """The repo card, for a browser that cannot fetch it or a mirror it must not."""

    def _call(
        self,
        repo = "Org/Model",
        branch = "main",
        resource = "models",
    ):
        return asyncio.run(
            discovery.discovery_readme(
                resource,
                repo = repo,
                branch = branch,
                hf_token = None,
                current_subject = "tester",
            )
        )

    def test_the_host_comes_from_the_endpoint_not_the_caller(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
        seen = {}

        def _fetch(url, token):
            seen["url"] = url
            return 200, b"# card", ""

        monkeypatch.setattr(discovery, "_fetch_upstream", _fetch)
        self._call()
        assert seen["url"].startswith("https://hf-mirror.example/Org/Model/raw/main/")

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_a_redirect_is_not_served_as_a_blank_card(self, monkeypatch, status):
        """_fetch_upstream returns redirects with an empty body, and they are
        under 400, so without an explicit branch they shipped as a 200."""
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (status, b"", ""))
        response = self._call()
        assert response.status_code == 502
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER) == "1"

    def test_a_connection_failure_is_stamped_not_a_500(self, monkeypatch):
        """The search and info routes map these; without it this one 500s raw."""
        monkeypatch.delenv("HF_ENDPOINT", raising = False)

        def _fetch(url, token):
            raise OSError("HTTPSConnectionPool(host='huggingface.co', port=443)")

        monkeypatch.setattr(discovery, "_fetch_upstream", _fetch)
        response = self._call()
        assert response.status_code == 502
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER) == "1"
        # And the host is not handed back: on a mirror it is the operator's.
        assert "huggingface.co" not in json.loads(response.body)["detail"]

    def test_a_rejected_repo_is_stamped_too(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        response = self._call(repo = "../etc/passwd")
        assert response.status_code == 400
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER) == "1"

    def test_datasets_take_the_datasets_prefix(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        seen = {}
        monkeypatch.setattr(
            discovery,
            "_fetch_upstream",
            lambda url, token: (seen.update(url = url), (200, b"# card", ""))[1],
        )
        self._call(resource = "datasets")
        assert "/datasets/Org/Model/raw/" in seen["url"]

    @pytest.mark.parametrize("branch", ["../../etc/passwd", "main;rm", "HEAD"])
    def test_only_main_and_master_are_accepted(self, branch):
        err = _error(self._call, branch = branch)
        assert err.status_code == 400

    def test_a_bad_repo_is_rejected(self):
        err = _error(self._call, repo = "../../secrets")
        assert err.status_code == 400

    def test_upstream_auth_is_not_surfaced_as_our_own(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (401, b"", ""))
        err = _error(self._call)
        assert err.status_code == discovery._UPSTREAM_AUTH_STATUS

    def test_a_missing_card_is_a_plain_404(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (404, b"", ""))
        err = _error(self._call)
        assert err.status_code == 404

    def test_the_card_is_returned_with_the_private_headers(self, monkeypatch):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (200, b"# card", ""))
        response = self._call()
        assert response.body == b"# card"
        assert response.headers.get("cache-control") == "no-store"
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER.lower()) == "1"


class TestEveryResponseCarriesTheMarker:
    """The frontend reads a 404 without the marker as "this backend has no route".

    An upstream 404 is passed through with its own status, so if error responses
    were unstamped a real Hub 404 would look identical to the SPA catch-all and
    the fallback would be disabled for the session.
    """

    def _listing(self, monkeypatch, status):
        monkeypatch.setattr(discovery, "_fetch_upstream", lambda url, token: (status, b"{}", ""))
        return asyncio.run(
            discovery.discovery_search(
                "models",
                _Request([("search", "gemma")]),
                hf_token = None,
                current_subject = "tester",
            )
        )

    @pytest.mark.parametrize("status", [404, 429, 500, 401])
    def test_an_upstream_failure_is_still_stamped(self, monkeypatch, status):
        response = self._listing(monkeypatch, status)
        assert response.status_code >= 400
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER.lower()) == "1"

    def test_a_rejected_query_is_stamped_too(self):
        response = asyncio.run(
            discovery.discovery_search(
                "models",
                _Request([("endpoint", "http://127.0.0.1")]),
                hf_token = None,
                current_subject = "tester",
            )
        )
        assert response.status_code == 400
        assert response.headers.get(discovery.HUB_PROXY_MARKER_HEADER.lower()) == "1"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The forced-anonymous sentinel is a credential of its own, not the absence of one.

``hf_token_arg`` returns three values where everything downstream expects ``Optional[str]``,
so ``False`` breaks four ways: a truthiness test reaches for the ambient token anyway, an
identity test hits ``.encode()`` on a bool, a shared cache fingerprint crosses the boundary,
and a child env inherits what it was never granted. ``is`` throughout: ``False == 0 == ""``.
"""

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import diffusion_compat
from hub.dependencies import get_request_hf_token
from hub.utils import hf_tokens
from picker import service as picker_service
from hub.utils.hf_tokens import (
    ANONYMOUS_CACHE_IDENTITY,
    apply_token_to_child_env,
    cache_reads_authorized,
    hf_token_arg,
    is_anonymous,
    normalize_token,
    reset_repo_access_cache,
)
from hub.utils.inventory_scan import token_fingerprint
from routes import models as models_routes
from utils.models.model_config import _token_fingerprint as capability_fingerprint
from utils.transformers_version import _token_cache_key


def _models_client(via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(models_routes.router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


@pytest.mark.parametrize(
    "hf_token, allow_ambient, expected",
    [
        (None, False, False),
        (None, True, None),
        ("  request-token  ", False, "request-token"),
        ("  request-token  ", True, "request-token"),
        ("", False, False),
        ("   ", True, None),
    ],
)
def test_the_resolver_returns_one_of_exactly_three_values(hf_token, allow_ambient, expected):
    resolved = hf_token_arg(hf_token, allow_ambient_token = allow_ambient)
    assert resolved is expected if expected in (None, False) else resolved == expected


def test_only_the_sentinel_reads_as_anonymous():
    assert is_anonymous(False) is True
    # The three values a "simplification" to `not hf_token` would wrongly sweep in.
    for absent in (None, "", 0):
        assert is_anonymous(absent) is False


def test_cache_reads_require_a_credential_that_reaches_the_repo(monkeypatch):
    """A token-shaped string is not authorization to read the host cache."""
    reset_repo_access_cache()
    probes = {"n": 0}

    def _probe(_repo_id, _token, _repo_type):
        probes["n"] += 1
        return False

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", _probe)
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)

    assert cache_reads_authorized(False, repo_id = "org/private") is False
    assert cache_reads_authorized(None, repo_id = "org/private") is True
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is False
    assert probes["n"] == 1, "ambient and anonymous must not pay a Hub round trip"
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is False
    assert probes["n"] == 1, "the negative answer was not memoized"


def test_a_ui_session_that_saved_a_token_still_reads_its_own_cache(monkeypatch):
    """The UI attaches the operator's saved token on most Hub routes, so an ordinary
    single-user session is an EXPLICIT-token caller on exactly the routes this gates.

    Measured before this: with a token saved in Settings and the Hub unreachable, that
    session lost the GGUF variant list, the default chat template and the dataset format
    check on repos already in its own cache. The same session with no token saved kept
    all three. Sending your own credential must not buy you less than sending none.
    """
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: True)
    probes = {"n": 0}
    monkeypatch.setattr(
        hf_tokens, "_probe_repo_access", lambda *_a, **_k: probes.__setitem__("n", probes["n"] + 1)
    )

    ui = hf_token_arg("  hf_saved  ", allow_ambient_token = True)
    api_key = hf_token_arg("  hf_saved  ", allow_ambient_token = False)

    assert ui == "hf_saved" and api_key == "hf_saved", "same value, different caller class"
    assert cache_reads_authorized(ui, repo_id = "org/private") is True
    assert cache_reads_authorized(api_key, repo_id = "org/private") is False
    assert probes["n"] == 0, "the UI session must not pay a round trip for its own token"


def test_the_ambient_marker_is_a_string_everywhere_it_is_used_as_a_value(monkeypatch):
    """It rides along the existing Optional[str | False] type, so every consumer that wants
    the token VALUE sees a plain token: comparisons, hashing, encoding, the child env.

    Cache IDENTITY is the deliberate exception, and this test used to assert the opposite.
    A UI session and an API key holding the same value have different cache authorization,
    so a fingerprint that collapsed them let either read back the other's verdict. Value
    transparent, entitlement not: see the caller-class identity test below.
    """
    ui = hf_token_arg("hf_saved", allow_ambient_token = True)

    assert isinstance(ui, str)
    assert ui == "hf_saved"
    assert hash(ui) == hash("hf_saved")
    assert ui.encode() == b"hf_saved"
    env: dict = {}
    apply_token_to_child_env(env, ui)
    assert env["HF_TOKEN"] == "hf_saved"
    assert type(env["HF_TOKEN"]) is str or isinstance(env["HF_TOKEN"], str)
    # The Hub still receives the real credential, marker or not.
    assert capability_fingerprint(ui).endswith(capability_fingerprint("hf_saved"))


def test_trimming_does_not_demote_a_ui_session_to_an_api_key():
    """str.strip returns a plain str, which would put the operator back behind the probe."""
    ui = hf_token_arg("  hf_saved  ", allow_ambient_token = True)

    assert isinstance(normalize_token(ui), hf_tokens.AmbientAuthorizedToken)
    assert normalize_token(ui) == "hf_saved"
    assert not isinstance(
        normalize_token(hf_token_arg("hf_saved", allow_ambient_token = False)),
        hf_tokens.AmbientAuthorizedToken,
    )
    assert normalize_token(False) is False
    assert normalize_token(None) is None


def test_a_verified_token_may_read_the_host_cache(monkeypatch):
    reset_repo_access_cache()
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: True)
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)

    assert cache_reads_authorized("hf_real", repo_id = "org/private") is True


def test_offline_explicit_token_is_denied_without_a_prior_probe(monkeypatch):
    """Fail closed: no Hub round trip means no cache for an unverified credential."""
    reset_repo_access_cache()
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: True)

    assert cache_reads_authorized("hf_real", repo_id = "org/private") is False


def test_offline_explicit_token_may_use_a_recent_online_probe(monkeypatch):
    reset_repo_access_cache()
    probes = {"n": 0}

    def _probe(_repo_id, _token, _repo_type):
        probes["n"] += 1
        return True

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", _probe)
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)

    assert cache_reads_authorized("hf_real", repo_id = "org/private") is True
    assert probes["n"] == 1

    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: True)
    assert cache_reads_authorized("hf_real", repo_id = "org/private") is True
    assert probes["n"] == 1, "offline must reuse the memo, not re-probe"


def _gated_hub_error(message = "gated"):
    """GatedRepoError without hub's response-bearing constructor (0.x vs 1.x)."""
    from huggingface_hub.errors import GatedRepoError

    class _Gated(GatedRepoError):
        def __init__(self, text):
            Exception.__init__(self, text)

    return _Gated(message)


class _FakeHubSession:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    def get(
        self,
        url,
        *,
        headers = None,
        timeout = None,
        **kwargs,
    ):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout, **kwargs})
        return self._handler(url, headers = headers, timeout = timeout, **kwargs)


def _patch_auth_check_get(monkeypatch, handler):
    session = _FakeHubSession(handler)
    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: session)
    return session


def _ok_auth_check_response():
    return SimpleNamespace(status_code = 200, raise_for_status = lambda: None)


def test_the_access_probe_hits_auth_check_not_repo_info():
    """repo_info succeeds on gated public metadata with an invalid token."""
    import inspect
    from hub.utils import hf_tokens

    source = inspect.getsource(hf_tokens._probe_repo_access)
    assert "auth-check" in source
    assert "get_session()" in source
    assert "timeout" in source
    assert "repo_info(" not in source
    assert "Thread(" not in source
    assert "_REPO_ACCESS_PROBE_TIMEOUT_S" in source


def test_the_hand_rolled_url_still_matches_the_one_auth_check_builds(monkeypatch):
    """The probe copies auth_check because auth_check takes no timeout. The copy has to
    track it.

    /auth-check is not a documented REST contract, it is an implementation detail of the
    client, and it has already moved once: hub 1.5.0 added a write= parameter that appends
    a /write segment. Ask the INSTALLED auth_check what URL it builds and compare, so an
    upstream change fails here instead of silently probing a path that 404s and denying
    every explicit-token caller their cache.
    """
    import huggingface_hub

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.delenv("HF_ENDPOINT", raising = False)

    upstream = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())
    # hf_api binds get_session at import, so patching huggingface_hub.utils misses it.
    monkeypatch.setattr("huggingface_hub.hf_api.get_session", lambda: upstream, raising = False)
    try:
        huggingface_hub.auth_check("org/repo", repo_type = "model", token = "hf_dummy")
    except Exception:
        pass
    assert upstream.calls, "could not observe the URL auth_check builds"
    upstream_url = upstream.calls[0]["url"]

    ours = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())
    cache_reads_authorized("hf_dummy", repo_id = "org/repo")

    assert ours.calls, "the probe did not reach the session"
    assert ours.calls[0]["url"] == upstream_url


def test_a_redirect_is_not_an_authorization(monkeypatch):
    """hf_raise_for_status passes 3xx through, and a client factory installed without
    follow_redirects would hand a bare 307 back. Legacy repo aliases do redirect."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    _patch_auth_check_get(
        monkeypatch,
        lambda *_a, **_k: SimpleNamespace(status_code = 307, raise_for_status = lambda: None),
    )

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False


def test_a_hanging_auth_check_probe_times_out(monkeypatch):
    """A stalled /auth-check must fail closed without leaving probe workers behind."""
    import requests

    reset_repo_access_cache()
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)
    monkeypatch.setattr("hub.utils.hf_tokens._REPO_ACCESS_PROBE_TIMEOUT_S", 0.2)

    def _hang(
        url,
        *,
        headers = None,
        timeout = None,
        **_k,
    ):
        assert timeout == 0.2
        raise requests.exceptions.Timeout("auth-check timed out")

    session = _patch_auth_check_get(monkeypatch, _hang)
    started = time.monotonic()
    assert cache_reads_authorized("hf_real", repo_id = "org/private") is False
    assert time.monotonic() - started < 5
    assert "/auth-check" in session.calls[0]["url"]
    assert not [t for t in threading.enumerate() if t.name == "hf-repo-auth-check"]


def test_twenty_distinct_cold_keys_leave_no_probe_workers_after_timeout(monkeypatch):
    """Each cold key must time out on the HTTP call, not in a detached probe thread."""
    import requests

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_REPO_ACCESS_PROBE_TIMEOUT_S", 0.05)

    def _hang(
        url,
        *,
        headers = None,
        timeout = None,
        **_k,
    ):
        raise requests.exceptions.Timeout("auth-check timed out")

    _patch_auth_check_get(monkeypatch, _hang)

    with ThreadPoolExecutor(max_workers = 20) as pool:
        answers = [
            f.result(timeout = 5)
            for f in [
                pool.submit(cache_reads_authorized, f"tok{i}", repo_id = f"org/repo{i}")
                for i in range(20)
            ]
        ]

    assert answers == [False] * 20
    assert not [t for t in threading.enumerate() if t.name == "hf-repo-auth-check"]


@pytest.mark.parametrize(
    "exc_name",
    ["TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"],
)
def test_every_httpx_timeout_reads_as_a_timeout_not_a_denial(monkeypatch, exc_name):
    """httpx puts its exceptions in ``httpx`` itself, so a ``"httpx."`` prefix matched
    none of them and only the two whose bare names are listed were classified. On hub
    1.x the session IS httpx, and a pool timeout is what a burst of concurrent probes
    produces, so the miss sent the common case to the full TTL."""
    httpx = pytest.importorskip("httpx")
    exc_cls = getattr(httpx, exc_name, None)
    if exc_cls is None:
        pytest.skip(f"httpx has no {exc_name}")

    assert hf_tokens._is_probe_timeout(exc_cls("stalled")) is True


def test_a_denial_is_not_mistaken_for_a_timeout(monkeypatch):
    """The other half: a real refusal must keep the full TTL, or a revoked token gets
    re-probed every five seconds forever."""
    for exc in (_gated_hub_error(), OSError("refused"), ValueError("bad token")):
        assert hf_tokens._is_probe_timeout(exc) is False


def test_a_repo_id_cannot_truncate_the_probe_url(monkeypatch):
    """The probe builds the URL itself now. A raw "?" ends the path, so the request would
    land on /api/models/{id}, which answers 200 with public metadata for a gated repo and
    an invalid token: the repo_info weakness this probe exists to avoid."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    cache_reads_authorized("hf_dummy", repo_id = "org/gated?ignored=")
    url = session.calls[0]["url"]
    assert url.endswith("/auth-check")
    assert "?" not in url
    assert "/api/models/org/gated%3Fignored%3D/auth-check" in url


def test_an_ordinary_repo_id_is_not_mangled_by_quoting(monkeypatch):
    """Valid repo ids are [A-Za-z0-9._-] and "/", so quoting must be invisible."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = "unsloth/Llama-3.2-1B") is True
    assert session.calls[0]["url"].endswith("/api/models/unsloth/Llama-3.2-1B/auth-check")


@pytest.mark.parametrize("repo_id", ["org/../x", "../x", "org/./x", "..", "."])
def test_a_dot_segment_repo_id_is_refused_before_the_wire(monkeypatch, repo_id):
    """Quoting keeps "/" so "org/repo" stays two segments, which keeps ".." intact too, and
    both clients then apply RFC 3986 dot-segment removal: "org/../x" is requested as
    "/api/models/x". The memo would be keyed on what the caller named while the wire proof
    was about a different repo."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = repo_id) is False
    assert session.calls == []


def test_a_scheme_less_mirror_endpoint_is_normalized(monkeypatch):
    """HfApi().endpoint hands back HF_ENDPOINT verbatim, so a scheme-less mirror built a
    URL both clients reject outright and every explicit-token cache read on that machine
    was denied without a request ever leaving the process. hf_endpoint_url is where the
    backend already normalizes it."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setenv("HF_ENDPOINT", "hf-mirror.example")
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is True
    assert session.calls[0]["url"] == ("https://hf-mirror.example/api/models/org/repo/auth-check")


def test_a_mirror_endpoint_keeps_its_own_scheme_and_loses_a_trailing_slash(monkeypatch):
    """The normalizer must not rewrite an endpoint that is already well formed, and must
    not leave "//api/" behind."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:8080/")
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is True
    assert session.calls[0]["url"] == "http://127.0.0.1:8080/api/models/org/repo/auth-check"


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: __import__("requests").exceptions.ConnectionError("refused"),
        lambda: __import__("requests").exceptions.ProxyError("dead proxy"),
        lambda: __import__("httpx").ConnectError("refused"),
        lambda: ConnectionRefusedError("refused"),
    ],
)
def test_a_hub_that_could_not_be_asked_is_not_a_denial(exc_factory):
    """A refusal, a DNS failure and a dead proxy are as much "could not ask" as a stall is.
    The asymmetry was measurable: a proxy that hung denied a valid token briefly, a proxy
    that refused denied it for a full minute."""
    assert hf_tokens._is_probe_timeout(exc_factory()) is True


def test_the_unreachable_ttl_outlives_a_probe(monkeypatch):
    """At 5s against a stalled Hub the memo expired before the next request could reach it:
    a probe takes the full timeout and the memo then lived less than that, so callers
    re-dialled and paid the stall again. The constant only does its job above the timeout."""
    assert hf_tokens._REPO_ACCESS_UNREACHABLE_TTL_S > hf_tokens._REPO_ACCESS_PROBE_TIMEOUT_S
    assert hf_tokens._REPO_ACCESS_UNREACHABLE_TTL_S < hf_tokens._REPO_ACCESS_TTL_S


def test_an_unreachable_hub_memo_actually_spans_the_next_request(monkeypatch):
    """The behaviour the constant above exists for, driven rather than asserted."""
    import requests

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_REPO_ACCESS_PROBE_TIMEOUT_S", 0.2)
    probes = {"n": 0}

    def _refuse(*_a, **_k):
        probes["n"] += 1
        raise requests.exceptions.ConnectionError("refused")

    _patch_auth_check_get(monkeypatch, _refuse)
    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False
    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False
    assert probes["n"] == 1, "the unreachable answer did not survive to the next request"


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_a_gated_repo_denies_cache_reads_for_an_invalid_token(monkeypatch, repo_type):
    """auth_check 401s; serving the host cache would bypass the gate."""
    reset_repo_access_cache()
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)
    seen = {}

    def _auth_check(
        url,
        *,
        headers = None,
        timeout = None,
        **_k,
    ):
        seen["args"] = (url, headers, timeout)
        raise _gated_hub_error()

    _patch_auth_check_get(monkeypatch, _auth_check)

    assert cache_reads_authorized("hf_invalid", repo_id = "org/gated", repo_type = repo_type) is False
    assert seen["args"][0].endswith(f"/api/{repo_type}s/org/gated/auth-check")


def test_repo_info_success_does_not_authorize_a_gated_cache_read(monkeypatch):
    """The previous probe: repo_info returns, auth_check raises, cache must stay closed."""
    reset_repo_access_cache()
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)

    def _auth_check(*_a, **_k):
        raise _gated_hub_error()

    _patch_auth_check_get(monkeypatch, _auth_check)

    assert cache_reads_authorized("hf_invalid", repo_id = "org/gated") is False


def test_a_public_repo_still_authorizes_cache_reads(monkeypatch):
    reset_repo_access_cache()
    monkeypatch.setattr("hub.utils.hf_tokens._hub_offline", lambda: False)
    _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = "org/public") is True


def test_a_timed_out_probe_is_not_memoized_for_the_full_ttl(monkeypatch):
    """A timeout says nothing about the credential, so it must not deny it for a minute.

    The sibling above pins that the hanging probe returns. This pins what it leaves
    behind: one stalled connection denying a valid token for the whole TTL is the same
    outage twice.
    """
    import requests

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_REPO_ACCESS_PROBE_TIMEOUT_S", 0.2)

    def _hang(*_a, **_k):
        raise requests.exceptions.Timeout("auth-check timed out")

    _patch_auth_check_get(monkeypatch, _hang)
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is False
    (expiry, allowed) = next(iter(hf_tokens._repo_access_cache.values()))
    assert allowed is False
    assert expiry - time.monotonic() <= hf_tokens._REPO_ACCESS_UNREACHABLE_TTL_S


def test_one_probe_per_key_no_matter_how_many_callers(monkeypatch):
    """The probe runs outside the shared lock, so nothing but this stops a herd.

    Measured before the in-flight lock: 32 concurrent callers on one cold key produced 32
    probes, and 16 concurrent callers against a stalled Hub opened 16 TCP connections.
    """
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    probes = []
    start = threading.Barrier(16)

    def _slow_probe(*_a, **_k):
        probes.append(1)
        time.sleep(0.2)
        return True

    monkeypatch.setattr(hf_tokens, "_probe_repo_access", _slow_probe)

    def _call():
        start.wait(10)
        return cache_reads_authorized("hf_dummy", repo_id = "org/private")

    with ThreadPoolExecutor(max_workers = 16) as pool:
        answers = [f.result(timeout = 30) for f in [pool.submit(_call) for _ in range(16)]]

    assert answers == [True] * 16
    assert len(probes) == 1


def test_a_slow_probe_still_memoizes_something_that_is_not_already_expired(monkeypatch):
    """The expiry was ``clock_before_probe + TTL``, so a probe slower than the TTL stored
    an entry that had already expired and every later request re-probed forever."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_REPO_ACCESS_TTL_S", 0.3)
    probes = []

    def _slow_probe(*_a, **_k):
        probes.append(1)
        time.sleep(0.5)
        return True

    monkeypatch.setattr(hf_tokens, "_probe_repo_access", _slow_probe)
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is True
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is True
    assert len(probes) == 1


@pytest.mark.parametrize("value", [True, 0, 1, 1.5, b"hf_bytes", ["hf"], {"t": 1}, object()])
def test_a_token_that_is_not_a_string_is_denied_rather_than_read_as_ambient(value):
    """``not isinstance(hf_token, str)`` returned the ambient answer, so every non-string
    took the operator's cache. Nothing reaches here today -- every HTTP boundary is a
    pydantic ``Optional[str]``, which rejects rather than coerces -- and this keeps the
    default at deny so one untyped ``payload.get("hf_token")`` cannot become a bypass."""
    assert cache_reads_authorized(value, repo_id = "org/private") is False


@pytest.mark.parametrize(
    "path",
    ["/home/u/models/foo", "C:\\models\\foo", "\\\\srv\\share\\foo", "./rel/model", "~/m/foo"],
)
def test_a_local_path_is_not_probed_against_the_hub(monkeypatch, path):
    """auth_check interpolates its argument into the URL without validating it, so a local
    path went to huggingface.co with the caller's bearer token attached, for a round trip
    whose only answer is no."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    probes = []
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *a, **k: probes.append(a) or True)

    assert cache_reads_authorized("hf_dummy", repo_id = path) is False
    assert probes == []


def test_the_local_config_probe_stays_local_for_an_explicit_token(monkeypatch, tmp_path):
    """prefer_local_cache on a local FOLDER must not lose local_files_only just because the
    caller sent a token the Hub has never been asked about.

    The sibling above pins the same argument for a repo id. A local path is not the Hub
    cache, which is why _model_config_inspection_target returns before the gate for one.
    """
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    local_dir = tmp_path / "my-model"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}", encoding = "utf-8")
    seen = {}

    def _is_vision(
        target,
        hf_token = None,
        local_files_only = False,
        **kwargs,
    ):
        seen["local_files_only"] = local_files_only
        return False

    monkeypatch.setattr(models_routes, "is_vision_model", _is_vision)
    monkeypatch.setattr(models_routes, "is_embedding_model", lambda *_a, **_k: False)
    monkeypatch.setattr(models_routes, "load_model_defaults", lambda *_a, **_k: {}, raising = False)

    import asyncio

    try:
        asyncio.run(
            models_routes.get_model_config(
                str(local_dir),
                hf_token = None,
                prefer_local_cache = True,
                local_path = None,
                header_hf_token = "hf_dummy",
                allow_ambient_token = False,
                current_subject = "tester",
            )
        )
    except Exception:
        pass

    assert seen.get("local_files_only") is True


def test_the_unreachable_hub_does_not_hand_back_the_template_the_walk_refused(monkeypatch):
    """picker's fallback is hf_hub_download, which serves the cached copy when the Hub is
    unreachable without consulting the credential. Its gate was ``hf_env_offline()``, and
    "hub unreachable" is not "env offline" -- an explicit ``HF_HUB_OFFLINE=0`` keeps the
    route from forcing offline at all, so an unverified token still got the template."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: False)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter(())
    )
    monkeypatch.setattr(picker_service, "get_cache_path", lambda _n: Path("/cached/repo"))
    downloads: list = []

    class _Api:
        def __init__(self, *_a, **_k):
            pass

        def get_paths_info(self, *_a, **_k):
            raise ConnectionError("hub unreachable")

    def _download(*a, **k):
        downloads.append(a)
        raise AssertionError("the cache fallback must not run for an unverified token")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert picker_service.read_default_chat_template("org/private", "hf_dummy") is None
    assert downloads == []


def test_an_uncached_public_template_is_still_fetched_when_paths_info_blips(monkeypatch, tmp_path):
    """Fail-closed on the inconclusive lookup must not cost the ordinary public fetch.

    With nothing cached for the repo the fallback has nothing to hand back, so the branch
    stays open. Otherwise the anonymous sentinel would lose a public repo's default
    template every time the paths-info API returned a 429.
    """
    reset_repo_access_cache()
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: False)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter(())
    )
    monkeypatch.setattr(picker_service, "get_cache_path", lambda _n: None)
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ public }}", encoding = "utf-8")

    class _Api:
        def __init__(self, *_a, **_k):
            pass

        def get_paths_info(self, *_a, **_k):
            raise ConnectionError("paths-info blipped")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *_a, **_k: str(template))

    assert picker_service.read_default_chat_template("org/public", False) == "{{ public }}"


def test_the_unreachable_hub_still_serves_the_operator_that_template(monkeypatch, tmp_path):
    """The same inconclusive lookup must keep working for the ambient UI session."""
    reset_repo_access_cache()
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: False)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter(())
    )
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ operator }}", encoding = "utf-8")

    class _Api:
        def __init__(self, *_a, **_k):
            pass

        def get_paths_info(self, *_a, **_k):
            raise ConnectionError("hub unreachable")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *_a, **_k: str(template))

    assert picker_service.read_default_chat_template("org/private", None) == "{{ operator }}"


def test_normalizing_a_token_does_not_launder_the_sentinel():
    # `(hf_token or "").strip() or None` turns "stay anonymous" into "use the backend's".
    assert normalize_token(False) is False
    assert normalize_token(None) is None
    assert normalize_token("  tok  ") == "tok"
    assert normalize_token("") is None


@pytest.mark.parametrize(
    "fingerprint, absent",
    [
        (token_fingerprint, ""),
        (capability_fingerprint, None),
        (diffusion_compat._token_fingerprint, ""),
    ],
    ids = ["inventory_scan", "model_capability", "diffusion_compat"],
)
def test_every_fingerprint_separates_anonymous_from_ambient(fingerprint, absent):
    anonymous = fingerprint(False)
    ambient = fingerprint(None)

    assert ambient == absent
    assert anonymous == ANONYMOUS_CACHE_IDENTITY
    assert anonymous != ambient, (
        "an anonymous caller sharing the ambient cache slot reads back metadata "
        "fetched with the operator's credential"
    )
    assert fingerprint("hf_realtoken") not in (anonymous, ambient)


def test_the_anonymous_identity_cannot_collide_with_a_token_digest():
    assert not all(character in "0123456789abcdef" for character in ANONYMOUS_CACHE_IDENTITY)


def test_a_fingerprint_never_carries_the_token():
    secret = "hf_supersecretvalue123"
    for fingerprint in (
        token_fingerprint,
        capability_fingerprint,
        diffusion_compat._token_fingerprint,
    ):
        assert secret not in str(fingerprint(secret))


def test_config_metadata_cache_keys_separate_anonymous_from_ambient():
    # These carry a private repo's config.json / tokenizer_config.json.
    assert _token_cache_key("org/private", False) != _token_cache_key("org/private", None)
    assert _token_cache_key("org/private", False) == ("org/private", ANONYMOUS_CACHE_IDENTITY)


def test_capability_fingerprint_does_not_raise_on_the_sentinel():
    # `if token is None` let False through to `.encode()`: /check-vision and /config 500d.
    assert capability_fingerprint(False) == ANONYMOUS_CACHE_IDENTITY
    assert hashlib.sha256(b"x").hexdigest() != capability_fingerprint(False)


def _child_env(hf_token):
    env = {
        "HF_TOKEN": "ambient-operator-token",
        "HUGGING_FACE_HUB_TOKEN": "ambient-legacy-alias",
        "HUGGINGFACEHUB_API_TOKEN": "ambient-other-alias",
        "PATH": "/usr/bin",
    }
    apply_token_to_child_env(env, hf_token)
    return env


def test_an_anonymous_probe_child_cannot_inherit_the_ambient_token():
    env = _child_env(False)

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        assert key not in env, f"{key} survived into an anonymous child"
    # get_token() still answers from a cached login file otherwise.
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["PATH"] == "/usr/bin"


def test_a_ui_session_probe_child_keeps_the_ambient_token():
    # Scrubbing here would break gated repos on installs whose token is in the env.
    env = _child_env(None)

    assert env["HF_TOKEN"] == "ambient-operator-token"
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in env


def test_an_explicit_token_replaces_the_ambient_one_in_the_child():
    env = _child_env("hf_caller_token")

    assert env["HF_TOKEN"] == "hf_caller_token"
    # An inherited "1" would 401 a gated repo the caller does have access to.
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"


@pytest.mark.parametrize("via_api_key", [True, False])
def test_capability_routes_answer_both_callers_without_a_server_error(monkeypatch, via_api_key):
    """The blocker this file was written for: /check-vision 500ed for an API-key caller."""
    seen = {}

    def _fake_is_vision_model(
        model_name,
        hf_token = None,
        **_kwargs,
    ):
        seen["hf_token"] = hf_token
        return False

    monkeypatch.setattr(models_routes, "is_vision_model", _fake_is_vision_model)
    response = _models_client(via_api_key).get(
        "/api/models/check-vision/org/some-model",
        headers = {"Authorization": "Bearer token"},
    )

    assert response.status_code == 200, response.text
    assert seen["hf_token"] is (False if via_api_key else None)


@pytest.mark.parametrize(
    "header, query, expected",
    [
        # An `or` chain ending on the query value would turn False into None here.
        (False, None, False),
        (None, None, None),
        (False, "", False),
        ("header-token", "query-token", "header-token"),
        ("header-token", None, "header-token"),
        (None, "query-token", "query-token"),
        (False, "query-token", "query-token"),
        ("  header-token  ", None, "header-token"),
    ],
)
def test_gguf_variants_token_precedence_survives_the_conversion(header, query, expected):
    resolved = models_routes._resolve_hub_token(header, query)

    if expected in (None, False):
        assert resolved is expected
    else:
        assert resolved == expected


def test_seed_inspection_derives_its_policy_from_the_caller(monkeypatch):
    """A UI session on an ambient-token install keeps gated seed inspection."""
    from routes.data_recipe import seed as seed_routes

    seen = {}

    def _fake_list(*, dataset_name, token):
        seen["token"] = token
        return []

    monkeypatch.setattr(seed_routes, "_list_hf_data_files", _fake_list)

    for via_api_key in (True, False):
        app = FastAPI()
        app.include_router(seed_routes.router, prefix = "/api/data-recipe")
        app.dependency_overrides[get_current_subject] = lambda: "alice"
        app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
        client = TestClient(app, raise_server_exceptions = False)

        client.post(
            "/api/data-recipe/seed/inspect",
            json = {"dataset_name": "org/private-seed"},
            headers = {"Authorization": "Bearer token"},
        )
        assert seen["token"] is (
            False if via_api_key else None
        ), "hardcoding allow_ambient_token=False takes the fallback away from the UI too"


def test_an_explicit_seed_token_wins_for_either_caller(monkeypatch):
    from routes.data_recipe import seed as seed_routes

    seen = {}
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: True)
    reset_repo_access_cache()
    monkeypatch.setattr(
        seed_routes,
        "_list_hf_data_files",
        lambda *, dataset_name, token: seen.update(token = token) or [],
    )
    monkeypatch.setattr(seed_routes, "load_dataset", lambda **_k: iter([]), raising = False)

    for via_api_key in (True, False):
        app = FastAPI()
        app.include_router(seed_routes.router, prefix = "/api/data-recipe")
        app.dependency_overrides[get_current_subject] = lambda: "alice"
        app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
        TestClient(app, raise_server_exceptions = False).post(
            "/api/data-recipe/seed/inspect",
            json = {"dataset_name": "org/private-seed", "hf_token": "  caller-token  "},
            headers = {"Authorization": "Bearer token"},
        )
        assert seen["token"] == "caller-token"


@pytest.mark.parametrize(
    "hf_token, allow_ambient, expected",
    [
        (None, False, False),
        (None, True, None),
        ("request-token", False, "request-token"),
        (" request-token ", True, "request-token"),
    ],
)
def test_the_request_dependency_keeps_the_caller_boundary(hf_token, allow_ambient, expected):
    resolved = get_request_hf_token(hf_token = hf_token, allow_ambient_token = allow_ambient)

    assert resolved == expected
    if expected in (None, False):
        assert resolved is expected


def test_the_media_load_models_still_reject_the_sentinel():
    """Why /v1/images/generations, /v1/videos and /video/generate keep the old dependency.

    All three reach maybe_auto_switch_media_model, whose _start_load builds a
    DiffusionLoadRequest / VideoLoadRequest. Both declare ``hf_token: Optional[str]``, so the
    sentinel is a ValidationError that kills the switch. Threading HfTokenArg through the load
    path means auditing ~30 `request.hf_token` consumers in routes/inference.py, which is a
    change of its own; until then those routes stay on get_hf_token and this test says so.
    """
    import pydantic
    from models.inference import DiffusionLoadRequest, VideoLoadRequest

    for model in (DiffusionLoadRequest, VideoLoadRequest):
        assert model.model_fields["hf_token"].annotation == Optional[str]
        with pytest.raises(pydantic.ValidationError):
            model(model_path = "org/repo", hf_token = False)


@pytest.mark.parametrize(
    "route_line, expected",
    [("audio/stt/download", True), ("audio/stt/validate", True)],
)
def test_the_stt_routes_keep_the_caller_boundary(route_line, expected):
    """The STT pair is converted: it downloads a whole repo and has no pydantic sink."""
    source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    marker = source.index(f'"/{route_line}"')
    signature = source[marker : marker + 400]
    assert ("Depends(get_request_hf_token)" in signature) is expected


def test_an_explicit_token_evicts_the_ambient_aliases_too():
    """Granting HF_TOKEN alone leaves the operator's credential in a legacy alias."""
    env = _child_env("hf_caller_token")

    assert env["HF_TOKEN"] == "hf_caller_token"
    for alias in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        assert alias not in env, f"{alias} still carried the operator credential"


def test_the_audio_tokenizer_fallback_does_not_reach_for_the_ambient_token(monkeypatch):
    """`hf_token or os.environ.get("HF_TOKEN")` reads past the sentinel to the operator's."""
    import utils.models.model_config as mc

    monkeypatch.setenv("HF_TOKEN", "ambient-operator-token")
    seen = {}

    class _Resp:
        status_code = 404
        text = ""

        def json(self):
            return {}

    def _get(
        url,
        headers = None,
        timeout = None,
        **_kw,
    ):
        seen.setdefault("headers", headers)
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "get", _get)
    mc._detect_audio_from_tokenizer("org/private", hf_token = False, revision = None)

    assert "Authorization" not in (
        seen.get("headers") or {}
    ), "an anonymous caller's tokenizer probe carried the operator's bearer"


def test_the_stt_sidecars_pass_the_sentinel_through_unchanged():
    """`hf_token or None` before spawn_download makes the child-env scrub unreachable."""
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent / "core" / "inference"
    for name in ("stt_sidecar.py", "stt_ggml_sidecar.py", "stt_mtmd_sidecar.py"):
        source = (root / name).read_text()
        assert (
            "hf_token or None" not in source
        ), f"{name} launders the sentinel into ambient access before the worker"


def test_an_anonymous_caller_does_not_get_the_unauthenticated_preview_cache(monkeypatch):
    """The disk fast path returns real rows without asking the Hub anything."""
    from hub.services.datasets import formatting

    called = {"cache": 0, "processed": 0}
    monkeypatch.setattr(
        formatting,
        "_load_cached_hf_preview_slice",
        lambda *_a, **_k: called.__setitem__("cache", called["cache"] + 1) or "ROWS",
    )
    # The processed path is the second disk read: it loads through
    # DownloadConfig(local_files_only=True) and drops the sentinel, so it never authorizes.
    monkeypatch.setattr(
        formatting,
        "_load_processed_hf_preview_slice",
        lambda *_a, **_k: (
            called.__setitem__("processed", called["processed"] + 1) or "PROCESSED_ROWS"
        ),
    )
    request = SimpleNamespace(
        dataset_name = "org/private", local_path = None, subset = None, train_split = "train"
    )

    assert formatting._load_any_cached_hf_preview_slice(request, 5, None) == "ROWS"
    assert called["cache"] == 1

    assert formatting._load_any_cached_hf_preview_slice(request, 5, False) is None
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    assert formatting._load_any_cached_hf_preview_slice(request, 5, "hf_dummy") is None
    assert called["cache"] == 1, "an unverified token reached the unauthenticated cache"
    assert called["processed"] == 0, "the anonymous caller reached the processed cache"


def test_an_anonymous_caller_does_not_read_a_cached_chat_template(monkeypatch):
    """The snapshot walk returns a private repo's raw template with no Hub call."""
    from picker import service as picker_service

    walked = {"n": 0}

    def _snapshots(*_a, **_k):
        walked["n"] += 1
        return [Path("/nonexistent-snapshot")]

    monkeypatch.setattr(picker_service, "iter_snapshots_preferring_whole", _snapshots)
    monkeypatch.setattr(picker_service, "_chat_template_from_dir", lambda *_a, **_k: "TEMPLATE")

    assert picker_service.read_default_chat_template("org/private", None) == "TEMPLATE"
    assert walked["n"] == 1

    picker_service.read_default_chat_template("org/private", False)
    assert walked["n"] == 1, "the anonymous caller walked the cached snapshots"
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    picker_service.read_default_chat_template("org/private", "hf_dummy")
    assert walked["n"] == 1, "an unverified token walked the cached snapshots"


def test_the_config_inspection_target_still_uses_the_cache_for_the_ambient_caller():
    """A caller allowed the ambient credential keeps the prefer_local_cache fast path.

    With no snapshot on disk the resolver raises its own 404; reaching that proves the
    cache branch ran rather than short-circuiting back to the bare repo id.
    """
    import fastapi
    with pytest.raises(fastapi.HTTPException):
        models_routes._model_config_inspection_target("org/private", True, None, None)


def test_the_config_inspection_target_skips_the_cache_for_an_unverified_token(monkeypatch):
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    target = models_routes._model_config_inspection_target("org/private", True, None, "hf_dummy")
    assert target == "org/private", "an unverified token was pointed at the cache"


def test_the_config_inspection_target_uses_the_cache_for_a_verified_token(monkeypatch):
    import fastapi

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: True)
    reset_repo_access_cache()
    with pytest.raises(fastapi.HTTPException):
        models_routes._model_config_inspection_target("org/private", True, None, "hf_real")


def test_the_config_inspection_target_skips_the_cache_for_anonymous():
    """The snapshot read returns private metadata without consulting the token."""
    target = models_routes._model_config_inspection_target("org/private", True, None, False)

    assert target == "org/private", "the anonymous caller was pointed at the cache"


def test_resolving_the_hub_token_never_returns_an_unresolved_dependency():
    """Callers that invoke the route function directly leave a ``Depends`` in the slot.

    The backend's own tests do exactly that, so returning ``header_token`` unchanged put
    a ``Depends`` object into the cache fingerprint and 500ed the whole variants route.
    """
    from fastapi import Depends

    unresolved = Depends(lambda: None)

    resolved = models_routes._resolve_hub_token(unresolved, None)

    assert resolved is None, "an unresolved dependency reached the Hub call"


def test_resolving_the_hub_token_keeps_the_anonymous_sentinel():
    """Rebuilding the sentinel must not quietly restore ambient access."""
    assert models_routes._resolve_hub_token(False, None) is False
    assert models_routes._resolve_hub_token(False, "  ") is False
    assert models_routes._resolve_hub_token(None, None) is None
    assert models_routes._resolve_hub_token(False, "hf_query") == "hf_query"
    assert models_routes._resolve_hub_token("hf_header", "hf_query") == "hf_header"


def test_an_anonymous_caller_gets_no_template_from_the_offline_fallback(monkeypatch):
    """Offline, hf_hub_download answers from disk and never checks the credential.

    The chat-template route forces offline whenever the Hub looks unreachable, so
    without this the Hub fallback hands back the template the cache walk just refused.
    """
    from picker import service as picker_service

    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: True)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)

    def _exploded(*args, **kwargs):
        raise AssertionError("the anonymous caller reached the hub fallback")

    monkeypatch.setattr(picker_service, "iter_snapshots_preferring_whole", _exploded)

    assert picker_service.read_default_chat_template("org/private", False) is None


def test_an_anonymous_config_read_does_not_strip_the_process_credential(monkeypatch):
    """The sentinel goes to the hub as `token=False`, not via without_hf_auth().

    That context deletes HF_TOKEN and moves the login token files process-wide, so a
    concurrent download in another worker thread would lose the operator's credential.
    """
    import utils.models.model_config as model_config_module

    monkeypatch.setenv("HF_TOKEN", "ambient-operator-token")
    seen = {}

    class _Config:
        pass

    def _from_pretrained(model_name, **kwargs):
        seen["token"] = kwargs.get("token", "<absent>")
        seen["ambient"] = os.environ.get("HF_TOKEN")
        return _Config()

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_from_pretrained))
    monkeypatch.setattr(model_config_module, "active_hf_hub_cache", lambda: None, raising = False)

    model_config_module.load_model_config("org/private", token = False)

    assert seen["token"] is False, "the sentinel was not passed through to the hub"
    assert (
        seen["ambient"] == "ambient-operator-token"
    ), "the anonymous probe removed a credential another thread was still using"


@pytest.mark.parametrize(
    # Explicit-token leg is an API key; as a UI session it asserted the regression.
    "hf_token, expected_local_only",
    [(None, True), ("hf_tok", False), (False, False)],
)
def test_the_config_probes_do_not_go_local_only_for_an_anonymous_caller(
    monkeypatch, hf_token, expected_local_only
):
    """local_files_only resolves config.json out of the cache without any authorization.

    Sending the anonymous caller back to the bare repo id only helps if the probe then
    goes over the wire, where `token=False` is refused for a private repo.
    """
    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    seen = {}

    def _is_vision(
        target,
        hf_token = None,
        local_files_only = False,
        **kwargs,
    ):
        seen["local_files_only"] = local_files_only
        return False

    monkeypatch.setattr(models_routes, "is_vision_model", _is_vision)
    monkeypatch.setattr(models_routes, "is_embedding_model", lambda *_a, **_k: False)
    monkeypatch.setattr(models_routes, "load_model_defaults", lambda *_a, **_k: {}, raising = False)
    monkeypatch.setattr(models_routes, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        models_routes,
        "_model_config_inspection_target",
        lambda *_a, **_k: "org/private",
    )

    import asyncio

    try:
        asyncio.run(
            models_routes.get_model_config(
                "org/private",
                hf_token = None,
                prefer_local_cache = True,
                local_path = None,
                header_hf_token = hf_token if isinstance(hf_token, str) else None,
                allow_ambient_token = hf_token is None,
                current_subject = "tester",
            )
        )
    except Exception:
        # The handler continues past the probe into machinery this test does not stand up;
        # the probe argument is what is being pinned.
        pass

    assert seen.get("local_files_only") is expected_local_only


def test_offline_embedding_detection_does_not_read_the_cache_anonymously(monkeypatch):
    """The marker read answers for a private repo without ever authorizing."""
    import utils.models.model_config as model_config_module

    monkeypatch.setattr(model_config_module, "is_local_path", lambda _n: False)
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: True, raising = False)

    def _marker(_name):
        raise AssertionError("the anonymous caller read the embedding cache marker")

    monkeypatch.setattr(model_config_module, "_embedding_marker_in_hf_cache", _marker)

    assert model_config_module.is_embedding_model("org/private", hf_token = False) is False


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_gguf_variants_serve_the_hf_cache_only_to_an_authorized_caller(monkeypatch, hf_token):
    """prefer_local_cache answers off disk with the credential never consulted.

    The listing carries variant filenames, sizes and the vision flag, so a caller denied
    the ambient token could name a private repo the UI had cached and read it back.
    """
    import asyncio

    from hub.services.models import gguf_variants

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    reads = {"snapshot": 0, "state": 0}

    def _snapshot(*_a, **_k):
        reads["snapshot"] += 1
        return None

    def _state(*_a, **_k):
        reads["state"] += 1
        return None

    monkeypatch.setattr(gguf_variants, "select_gguf_cache_snapshot", _snapshot)
    monkeypatch.setattr(gguf_variants, "_quants_from_state", _state)

    try:
        asyncio.run(
            gguf_variants.get_gguf_variants_answer(
                "org/private",
                prefer_local_cache = True,
                offline = True,
                local_path = None,
                hf_token = hf_token,
            )
        )
    except Exception:
        # Offline with nothing cached is a 404 either way; the reads are the point.
        pass

    if is_anonymous(hf_token) or hf_token == "hf_tok":
        assert reads == {
            "snapshot": 0,
            "state": 0,
        }, "an unverified caller was served from the hub cache"
    else:
        assert reads["snapshot"] > 0, "the authorized caller lost its cache fast path"


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_offline_capability_probes_do_not_read_the_cache_anonymously(monkeypatch, hf_token):
    """Offline, is_vision_model derives local_files_only from the environment.

    So passing local_files_only=False does not put the anonymous caller back on the wire:
    the probe reads the cached config.json off disk and never authorizes.
    """
    import utils.models.model_config as model_config_module

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    monkeypatch.setattr(model_config_module, "_env_offline", lambda: True)
    reached = {"vision": 0, "audio": 0}

    def _vision(*_a, **_k):
        reached["vision"] += 1
        return True

    def _audio(*_a, **_k):
        reached["audio"] += 1
        return "stt", True

    monkeypatch.setattr(model_config_module, "_is_vision_model_uncached", _vision)
    monkeypatch.setattr(model_config_module, "_detect_audio_type_uncached", _audio, raising = False)
    # A fresh probe every time, so a warm entry cannot stand in for the guard.
    monkeypatch.setattr(model_config_module, "_vision_detection_cache", {})
    monkeypatch.setattr(model_config_module, "_audio_detection_cache", {})
    monkeypatch.setattr(model_config_module, "_audio_offline_miss_cache", {})

    is_vision = model_config_module.is_vision_model(
        "org/private", hf_token = hf_token, local_files_only = False
    )

    if is_anonymous(hf_token) or hf_token == "hf_tok":
        assert is_vision is False
        assert reached["vision"] == 0, "an unverified caller probed the offline cache"
    else:
        assert reached["vision"] == 1, "the authorized caller lost its offline probe"


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_the_config_json_fallbacks_do_not_reach_the_cache_anonymously(monkeypatch, hf_token):
    """Keying the memo apart is not enough when the value came off disk to begin with."""
    import utils.transformers_version as tv

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    monkeypatch.setattr(tv, "_env_offline", lambda: True)
    monkeypatch.setattr(tv, "_safe_is_file", lambda _p: False)
    monkeypatch.setattr(tv, "_safe_is_dir", lambda _p: False)
    monkeypatch.setattr(tv, "_config_json_cache", {})
    reads = {"n": 0}

    def _from_cache(_name):
        reads["n"] += 1
        return {"max_position_embeddings": 4096}

    monkeypatch.setattr(tv, "_config_json_from_hf_cache", _from_cache)

    cfg = tv._load_config_json("org/private", hf_token = hf_token)

    if is_anonymous(hf_token) or hf_token == "hf_tok":
        assert cfg is None
        assert reads["n"] == 0, "an unverified caller read the offline config cache"
    else:
        assert cfg == {"max_position_embeddings": 4096}


def test_a_cache_only_gguf_listing_is_refused_for_an_anonymous_caller(monkeypatch):
    """siblings is None means the lister already answered from its own cache.

    Declining to build a second cached response is not enough: falling through would
    serialize the first one.
    """
    import asyncio

    import fastapi

    from hub.services.models import gguf_variants

    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants",
        lambda *_a, **_k: ([SimpleNamespace(filename = "m-Q4.gguf")], False, None),
    )
    monkeypatch.setattr(gguf_variants, "select_gguf_cache_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(gguf_variants, "_quants_from_state", lambda *_a, **_k: None)

    with pytest.raises(fastapi.HTTPException) as excinfo:
        asyncio.run(gguf_variants.get_gguf_variants_answer("org/private", hf_token = False))

    assert excinfo.value.status_code == 404


def test_the_scan_route_derives_its_caller_rather_than_trusting_an_absent_body_token():
    """An absent body token must not read as ambient-authorized."""
    import inspect

    signature = inspect.signature(models_routes.scan_model_remote_code)

    assert (
        "allow_ambient_token" in signature.parameters
    ), "the scan route cannot tell an api key from a ui session"
    source = inspect.getsource(models_routes.scan_model_remote_code)
    assert (
        "hf_token_arg(hf_token" in source
    ), "the body token reaches the cache-backed scan target unresolved"


def test_an_anonymous_seed_preview_is_refused_while_offline(monkeypatch):
    """Offline, `datasets` satisfies a streaming load from its own cache.

    The sentinel never reaches an authorization check there, so a previously cached
    private dataset would come back as rows.
    """
    import asyncio

    import fastapi

    from routes.data_recipe import seed as seed_routes

    monkeypatch.setattr(seed_routes, "hf_env_offline", lambda: True)
    # The dataset IS cached here, which is the premise the denial rests on and what these
    # docstrings already describe. An uncached one has nothing to serve, so it is allowed
    # through to the loader and covered by its own test.
    monkeypatch.setattr(seed_routes, "dataset_cache_can_answer", lambda *_a, **_k: True)

    def _never(*_a, **_k):
        raise AssertionError("the anonymous caller reached the dataset load")

    monkeypatch.setattr(seed_routes, "_list_hf_data_files", _never)

    payload = SimpleNamespace(
        dataset_name = "org/private",
        split = None,
        subset = None,
        hf_token = None,
        preview_size = 5,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        asyncio.run(seed_routes.inspect_seed_dataset(payload, allow_ambient_token = False))

    assert excinfo.value.status_code == 404


def test_an_unverified_seed_preview_is_refused(monkeypatch):
    """`datasets` will satisfy the load from cache without asking the Hub."""
    import asyncio

    import fastapi

    from routes.data_recipe import seed as seed_routes

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    monkeypatch.setattr(seed_routes, "hf_env_offline", lambda: False)
    # Cached, which is the premise the denial rests on: the gate exists to stop a read off
    # the operator's disk. An uncached dataset has its own test.
    monkeypatch.setattr(seed_routes, "dataset_cache_can_answer", lambda *_a, **_k: True)

    def _never(*_a, **_k):
        raise AssertionError("an unverified token reached the dataset load")

    monkeypatch.setattr(seed_routes, "_list_hf_data_files", _never)

    payload = SimpleNamespace(
        dataset_name = "org/private",
        split = None,
        subset = None,
        hf_token = "hf_dummy",
        preview_size = 5,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        asyncio.run(seed_routes.inspect_seed_dataset(payload, allow_ambient_token = False))

    assert excinfo.value.status_code == 404


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_the_lora_resolver_does_not_launder_the_sentinel(monkeypatch, hf_token):
    """`hf_token if hf_token else None` turned the sentinel back into ambient access."""
    import utils.models.model_config as model_config_module

    seen = {}

    def _absent(
        _identifier,
        _filename,
        token = None,
    ):
        seen["token"] = token
        return True

    monkeypatch.setattr("utils.hf_probe.hf_file_definitely_absent", _absent, raising = False)

    model_config_module.get_base_model_from_lora_identifier(
        "org/private-adapter", hf_token = hf_token
    )

    if is_anonymous(hf_token):
        assert seen["token"] is False, "the scan pipeline restored the ambient token"
    else:
        assert seen["token"] == (hf_token or None)


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_the_audio_tokenizer_probe_does_not_read_the_cache_anonymously(monkeypatch, hf_token):
    """The cache root is walked before any network branch, online as well as offline."""
    import utils.models.model_config as model_config_module

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    reads = {"n": 0}

    def _cache_path(_name):
        reads["n"] += 1
        return None

    monkeypatch.setattr(model_config_module, "get_cache_path", _cache_path)
    monkeypatch.setattr(model_config_module, "is_local_path", lambda _n: False)

    try:
        model_config_module._detect_audio_from_tokenizer("org/private", hf_token = hf_token)
    except Exception:
        pass

    if is_anonymous(hf_token) or hf_token == "hf_tok":
        assert reads["n"] == 0, "an unverified caller walked the hub cache"
    else:
        assert reads["n"] > 0, "the authorized caller lost its cache fast path"


def test_the_embedding_transient_fallback_is_denied_to_an_anonymous_caller(monkeypatch):
    """The anonymous 404 for a private repo lands in the same except branch."""
    import utils.models.model_config as model_config_module

    monkeypatch.setattr(model_config_module, "is_local_path", lambda _n: False)
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: False, raising = False)
    monkeypatch.setattr(model_config_module, "_embedding_detection_cache", {})

    def _boom(*_a, **_k):
        raise RuntimeError("404")

    monkeypatch.setattr(model_config_module, "model_info", _boom, raising = False)

    def _marker(_name):
        raise AssertionError("the anonymous caller read the embedding cache marker")

    monkeypatch.setattr(model_config_module, "_embedding_marker_in_hf_cache", _marker)

    assert model_config_module.is_embedding_model("org/private", hf_token = False) is False


def test_a_public_model_keeps_its_size_when_the_cache_is_bypassed():
    """The anonymous short-circuit returns the bare repo id, which is not a path.

    Sizing it as one returns None, so public models lost model_size_bytes entirely.
    """
    import inspect

    source = inspect.getsource(models_routes.get_model_config)

    assert (
        "inspection_target != model_name" in source
    ), "snapshot sizing is still chosen from the flag rather than the target"


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_the_offline_autoconfig_read_is_denied_to_an_anonymous_caller(monkeypatch, hf_token):
    """token=False disables authentication but not the local cache.

    The repo is cached here, which is the premise the denial rests on: the gate exists to
    stop a caller reading config.json off the operator's disk. An uncached repo has nothing
    to read and is covered separately, because refusing it protects nothing and breaks a
    mirror without /auth-check.
    """
    import transformers

    import utils.models.model_config as model_config_module

    monkeypatch.setattr("hub.utils.hf_tokens._probe_repo_access", lambda *_a, **_k: False)
    reset_repo_access_cache()
    monkeypatch.setattr(model_config_module, "_env_offline", lambda: True)
    monkeypatch.setattr(model_config_module, "active_hf_hub_cache", lambda: None, raising = False)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    reached = {"n": 0}

    def _from_pretrained(_name, **_kwargs):
        reached["n"] += 1
        return object()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_from_pretrained))

    if is_anonymous(hf_token) or hf_token == "hf_tok":
        with pytest.raises(OSError):
            model_config_module.load_model_config("org/private", token = hf_token)
        assert reached["n"] == 0, "an unverified caller read the offline config cache"
    else:
        model_config_module.load_model_config("org/private", token = hf_token)
        assert reached["n"] == 1


def test_the_prefer_local_scan_branch_carries_the_anonymous_guard():
    """Only the exact-snapshot branch was gated; the sibling resolved the cache anyway."""
    import inspect

    source = inspect.getsource(models_routes.scan_model_remote_code)
    marker = source.index("prefer_local_cache is True")
    branch = source[marker : marker + 280]

    assert (
        "cache_reads_authorized(hf_token, repo_id = model_name)" in branch
    ), "the prefer-local scan branch still resolves a cached snapshot for any caller"


@pytest.mark.parametrize(
    "hf_token, offline, denied",
    [
        (False, True, True),
        (False, False, False),
        (None, True, False),
        ("hf_tok", True, False),
    ],
)
def test_the_offline_anonymous_rule_is_stated_once(hf_token, offline, denied, monkeypatch):
    """One precondition, not one guard per reader.

    Six separate readers were fixed in turn -- the snapshot walk, the config probes, the
    embedding marker, the GGUF listing, the preview slices, AutoConfig -- and each fix
    only moved the boundary to the next one. This pins the shared rule itself.
    """
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: offline)

    assert utils_module.anonymous_and_offline(hf_token) is denied


def test_every_offline_reachable_route_refuses_before_it_reads(monkeypatch):
    """The three routes that reach disk offline all consult the shared rule."""
    import inspect

    from hub.services.datasets import formatting

    for owner, name in (
        (models_routes.get_model_config, "/config"),
        (models_routes.scan_model_remote_code, "scan-remote-code"),
        (formatting.check_format_response, "check-format"),
    ):
        source = inspect.getsource(owner)
        assert (
            "anonymous_and_offline" in source
        ), f"{name} can still be answered from disk for a denied caller"


def test_an_unreachable_hub_is_a_404_not_a_500():
    """check-format has no refusal of its own: a denied caller returns None from the disk
    route, falls through to a Hub that cannot answer, and the pair below came back. They
    were not mapped, so the catch-all turned "I could not reach the Hub" into a 500.
    seed/inspect raises its own 404; this is the same answer for the same condition."""
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
    from hub.utils.hf_errors import hf_error_status

    assert hf_error_status(LocalEntryNotFoundError("no cache, no hub")) == 404
    assert hf_error_status(OfflineModeIsEnabled("offline")) == 404
    assert hf_error_status(RuntimeError("boom")) is None


# --- The four defects found reviewing this branch's own gates -------------------------


def test_a_ui_sessions_marker_survives_the_route_level_token_normalizer():
    """``routes.models._normalize_hf_token`` trimmed with ``str.strip()``, which returns a
    plain ``str``. That silently demoted a UI session to an API key between the dependency
    and the gate, so an ordinary session lost its own cache offline: the exact regression
    ``AmbientAuthorizedToken`` exists to prevent, reintroduced one call later."""
    ui = hf_token_arg("  hf_saved  ", allow_ambient_token = True)
    resolved = models_routes._normalize_hf_token(ui)

    assert resolved == "hf_saved"
    assert isinstance(resolved, hf_tokens.AmbientAuthorizedToken), "marker lost in normalization"
    # An API key must not acquire the marker on the way through.
    api_key = hf_token_arg("  hf_saved  ", allow_ambient_token = False)
    assert not isinstance(
        models_routes._normalize_hf_token(api_key), hf_tokens.AmbientAuthorizedToken
    )
    assert models_routes._normalize_hf_token("   ") is None
    assert models_routes._normalize_hf_token(False) is None


def test_the_format_check_authorizes_before_the_streaming_tiers(monkeypatch):
    """Measured against the installed ``datasets``: offline, ``load_dataset`` answers BOTH
    ``streaming=True`` tiers out of its own prepared cache, logging "using the latest cached
    version", with the token never consulted. Both tiers run on the default
    ``prefer_local_cache=false``, ahead of the guarded cache reader, so the gate has to
    stand in front of them the way the seed-inspect route already does."""
    from hub.services.datasets import formatting
    from hub.schemas.datasets import CheckFormatRequest

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)
    # Cached, which is the premise the denial rests on: the gate exists to stop a read off
    # the operator's disk. An uncached dataset has its own test.
    monkeypatch.setattr(formatting, "dataset_cache_can_answer", lambda *_a, **_k: True)

    loads = {"n": 0}

    def _explode(*_a, **_k):
        loads["n"] += 1
        raise AssertionError("load_dataset must not run for an unauthorized caller")

    monkeypatch.setattr("datasets.load_dataset", _explode)

    request = CheckFormatRequest(dataset_name = "acme/private-secrets")
    api_key = hf_token_arg("hf_cannot_read_this", allow_ambient_token = False)

    with pytest.raises(HTTPException) as excinfo:
        formatting.check_format_response(request, api_key)
    assert excinfo.value.status_code == 404
    assert loads["n"] == 0


def test_the_format_check_still_serves_an_ordinary_ui_session(monkeypatch):
    """The gate above must not become the UI regression again: a session holding its own
    saved token is entitled to ambient, so it reaches the loader with no probe."""
    from hub.services.datasets import formatting
    from hub.schemas.datasets import CheckFormatRequest

    reset_repo_access_cache()
    probes = {"n": 0}
    monkeypatch.setattr(
        hf_tokens, "_probe_repo_access", lambda *_a, **_k: probes.__setitem__("n", probes["n"] + 1)
    )
    reached = {"n": 0}

    def _reached(*_a, **_k):
        reached["n"] += 1
        raise RuntimeError("stop here: the gate let us through, which is all this asserts")

    monkeypatch.setattr("datasets.load_dataset", _reached)

    request = CheckFormatRequest(dataset_name = "acme/private-secrets")
    ui = hf_token_arg("hf_saved", allow_ambient_token = True)

    with pytest.raises(HTTPException):
        formatting.check_format_response(request, ui)
    assert reached["n"] > 0, "an ordinary UI session was denied its own dataset"
    assert probes["n"] == 0


def test_an_explicit_tokens_offline_config_read_is_not_memoized_forever(monkeypatch):
    """``cache_reads_authorized`` expires in 60 s precisely so a revoked token stops reading.
    ``_config_json_cache`` has no TTL, so memoizing a value that came off the operator's disk
    outlived the access it was granted under: the first read authorized, every later one was
    served from the memo without the gate running again."""
    from utils import transformers_version as tv

    reset_repo_access_cache()
    tv._config_json_cache.clear()
    monkeypatch.setattr(tv, "_env_offline", lambda: True)
    monkeypatch.setattr(
        tv, "_config_json_from_hf_cache", lambda *_a, **_k: {"model_type": "secret"}
    )

    authorized = {"v": True}
    monkeypatch.setattr(tv, "cache_reads_authorized", lambda *_a, **_k: authorized["v"])

    token = hf_token_arg("hf_explicit", allow_ambient_token = False)
    assert tv._load_config_json("acme/private", token) == {"model_type": "secret"}

    # The credential is revoked; the next read must re-derive, not replay the memo.
    authorized["v"] = False
    assert tv._load_config_json("acme/private", token) is None


def test_an_ambient_offline_config_read_keeps_its_memo(monkeypatch):
    """The fix above must not turn every ambient read into a fresh disk walk."""
    from utils import transformers_version as tv

    reset_repo_access_cache()
    tv._config_json_cache.clear()
    monkeypatch.setattr(tv, "_env_offline", lambda: True)
    reads = {"n": 0}

    def _from_cache(*_a, **_k):
        reads["n"] += 1
        return {"model_type": "public"}

    monkeypatch.setattr(tv, "_config_json_from_hf_cache", _from_cache)

    assert tv._load_config_json("acme/public", None) == {"model_type": "public"}
    assert tv._load_config_json("acme/public", None) == {"model_type": "public"}
    assert reads["n"] == 1, "the ambient memo stopped working"


def test_the_vision_config_read_refuses_an_unauthorized_cache_fallback(monkeypatch, tmp_path):
    """Measured against the installed huggingface_hub: with a planted cache entry and a dead
    endpoint, ``hf_hub_download(local_files_only=False)`` returns the operator's CACHED
    config.json for a token that cannot read the repo, because it falls back to disk whenever
    the Hub is unreachable and never consults the credential to do it."""
    from utils.models import model_config as mc

    reset_repo_access_cache()
    # A real cached config, so an unguarded read returns the leak rather than an error.
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text('{"vision_config": {}, "model_type": "secret"}')
    monkeypatch.setattr(mc, "_config_json_already_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(mc, "cache_reads_authorized", lambda *_a, **_k: False)

    # Counted, not raised: the reader wraps everything in `except Exception`, so an
    # AssertionError thrown in here would be swallowed and the test would pass unguarded.
    calls = {"n": 0}

    def _download(*_a, **_k):
        calls["n"] += 1
        return str(tmp_config)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    api_key = hf_token_arg("hf_cannot_read_this", allow_ambient_token = False)
    assert mc._raw_config_has_vision_config("acme/private-vlm", hf_token = api_key) is None
    assert calls["n"] == 0, "hf_hub_download ran for an unauthorized cached repo"


def test_the_vision_config_read_keeps_the_wire_for_a_repo_not_on_disk(monkeypatch):
    """The guard belongs at the leak, not at the API boundary. A repo that is not cached has
    nothing to leak, so it must reach the Hub, which enforces its own access control. Gating
    the whole call instead would deny a legitimate token its answer on any Hub hiccup, and it
    broke exactly that: TestVisionCacheTokenHandling asserts a gated model still classifies."""
    from utils.models import model_config as mc

    reset_repo_access_cache()
    monkeypatch.setattr(mc, "_config_json_already_cached", lambda *_a, **_k: False)

    probes = {"n": 0}
    monkeypatch.setattr(
        mc,
        "cache_reads_authorized",
        lambda *_a, **_k: probes.__setitem__("n", probes["n"] + 1) or False,
    )
    downloads = {"n": 0}

    def _download(*_a, **_k):
        downloads["n"] += 1
        raise RuntimeError("reached the wire, which is all this asserts")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    monkeypatch.setattr(mc, "hf_file_definitely_absent", lambda *_a, **_k: False, raising = False)

    api_key = hf_token_arg("hf_explicit", allow_ambient_token = False)
    mc._raw_config_has_vision_config("acme/not-cached", hf_token = api_key)

    assert downloads["n"] == 1, "an uncached repo lost its wire read"
    assert probes["n"] == 0, "an uncached repo has nothing to leak, so it must not pay a probe"


def test_a_failed_cache_check_does_not_open_the_path_it_guards(monkeypatch):
    """If the cache lookup itself raises, the guard must assume a hit and authorize."""
    from utils.models import model_config as mc

    def _boom(*_a, **_k):
        raise OSError("cache unreadable")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _boom)
    assert mc._config_json_already_cached("acme/private-vlm") is True


def test_the_legacy_body_token_keeps_its_caller_class():
    """The deprecated /api/datasets/check-format still accepts the token in the BODY. That
    value arrived as a plain str and skipped the marker get_request_hf_token attaches, so the
    same UI session, sending the same token, lost its own cached dataset offline purely for
    using the legacy route. The header and the body are one credential, classified alike."""
    from routes import datasets as datasets_routes

    app = FastAPI()
    app.include_router(datasets_routes.router, prefix = "/api/datasets")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False  # a UI session

    seen = {}

    def _capture(
        request,
        hf_token = None,
        **_k,
    ):
        seen["token"] = hf_token
        raise HTTPException(status_code = 418, detail = "captured")

    original = datasets_routes.formatting.check_format_response
    datasets_routes.formatting.check_format_response = _capture
    try:
        with TestClient(app) as client:
            client.post(
                "/api/datasets/check-format",
                json = {"dataset_name": "acme/private-secrets", "hf_token": "hf_saved"},
            )
    finally:
        datasets_routes.formatting.check_format_response = original

    assert isinstance(
        seen.get("token"), hf_tokens.AmbientAuthorizedToken
    ), "a UI session's body token was demoted to an API key"


def test_the_same_token_from_two_caller_classes_takes_two_cache_identities():
    """The marker is a str subclass with the same encoded value, so every fingerprint that
    hashes the token alone collided. Any cache keyed on one could hand a caller the other's
    verdict, and the GGUF in-flight coalescer merged their scans into a single computation
    whose authorization was settled by whichever request arrived first."""
    from hub.utils import inventory_scan
    from utils.models import model_config as mc
    from core.inference import diffusion_compat as dc

    ui = hf_token_arg("hf_saved", allow_ambient_token = True)
    api = hf_token_arg("hf_saved", allow_ambient_token = False)

    for fingerprint in (
        inventory_scan.token_fingerprint,
        mc._token_fingerprint,
        dc._token_fingerprint,
    ):
        ui_id, api_id = fingerprint(ui), fingerprint(api)
        assert ui_id != api_id, f"{fingerprint.__module__} still collides across caller classes"
        # The qualifier must not smuggle the credential into a dict key.
        assert "hf_saved" not in str(ui_id) and "hf_saved" not in str(api_id)

    # The sentinel and the ambient caller keep the identities they already had.
    assert inventory_scan.token_fingerprint(False) == ANONYMOUS_CACHE_IDENTITY
    assert inventory_scan.token_fingerprint(None) == ""


def test_the_gguf_inflight_key_does_not_coalesce_two_caller_classes():
    """The concrete consequence: same repo, same token value, different entitlement."""
    from hub.utils import inventory_scan

    def _key(token):
        return ("acme/private", False, True, "", inventory_scan.token_fingerprint(token), "cache-a")

    ui = hf_token_arg("hf_saved", allow_ambient_token = True)
    api = hf_token_arg("hf_saved", allow_ambient_token = False)
    assert _key(ui) != _key(api), "one scan would serve both entitlements"


def test_the_legacy_query_token_is_classified_like_the_header(monkeypatch):
    """normalize_token can carry a marker through but cannot create one, and ?hf_token= never
    had it: the value arrives as a bare string, not from the dependency. Measured before this,
    for one UI session sending one token two ways with the Hub unreachable:

        query   type=str                    authorized=False
        header  type=AmbientAuthorizedToken authorized=True
    """
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)

    ui_header_absent = hf_token_arg(None, allow_ambient_token = True)
    resolved = models_routes._resolve_hub_token(ui_header_absent, "hf_saved")
    assert isinstance(resolved, hf_tokens.AmbientAuthorizedToken)
    assert cache_reads_authorized(resolved, repo_id = "acme/private") is True

    # An API key sending only the query parameter stays an API key: its header is the sentinel.
    api_no_header = hf_token_arg(None, allow_ambient_token = False)
    api_resolved = models_routes._resolve_hub_token(api_no_header, "hf_saved")
    assert not isinstance(api_resolved, hf_tokens.AmbientAuthorizedToken)
    assert cache_reads_authorized(api_resolved, repo_id = "acme/private") is False

    # The header still wins over a stale query value, and the no-token cases are unchanged.
    header = hf_token_arg("hf_header", allow_ambient_token = True)
    assert models_routes._resolve_hub_token(header, "hf_query") == "hf_header"
    assert (
        models_routes._resolve_hub_token(hf_token_arg(None, allow_ambient_token = False), None)
        is False
    )
    assert (
        models_routes._resolve_hub_token(hf_token_arg(None, allow_ambient_token = True), None) is None
    )


def test_the_embedding_memo_does_not_cross_caller_classes(monkeypatch):
    """The memo is read at the top of is_embedding_model, above every authorization check, and
    was keyed on the raw token. The marker hashes and compares equal to a plain API token of
    the same value, so a UI-computed classification came straight back to an unverified API
    caller. Keyed on the fingerprint now, which carries the caller class."""
    from utils.models import model_config as mc

    ui = hf_token_arg("hf_saved", allow_ambient_token = True)
    api = hf_token_arg("hf_saved", allow_ambient_token = False)

    monkeypatch.setattr(mc, "_embedding_detection_cache", {})
    monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
    monkeypatch.setattr(mc, "_embedding_marker_in_hf_cache", lambda *_a, **_k: True)
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: False)

    def _no_hub(*_a, **_k):
        raise RuntimeError("hub unreachable, so the cached marker is the fallback")

    monkeypatch.setattr("huggingface_hub.model_info", _no_hub)
    # Only the UI session is entitled to the operator's cache here.
    monkeypatch.setattr(
        mc,
        "cache_reads_authorized",
        lambda token, **_k: isinstance(token, hf_tokens.AmbientAuthorizedToken),
    )

    # The UI session classifies from its own cached marker and memoizes that.
    assert mc.is_embedding_model("acme/private-emb", ui) is True
    # The API caller must be refused, not handed the entry the UI just wrote.
    assert mc.is_embedding_model("acme/private-emb", api) is False


def test_the_embedding_settings_routes_classify_their_payload_token():
    """Both the PUT and its resolve twin took the token as a bare trimmed str, so a UI session
    saving an embedding model was treated as an API key and refused its own cached modules.json,
    returning the forceable 409. The resolve endpoint says it must refuse exactly what the PUT
    refuses, so fixing only one would have made them disagree."""
    import inspect
    from routes import settings as settings_routes

    for endpoint in (
        settings_routes.update_embedding_model,
        settings_routes.resolve_embedding_model,
    ):
        params = inspect.signature(endpoint).parameters
        assert "allow_ambient_token" in params, f"{endpoint.__name__} cannot tell its callers apart"

    source = inspect.getsource(settings_routes.update_embedding_model)
    assert "hf_token_arg(" in source, "the payload token is not classified"
    assert '(payload.hf_token or "").strip()' not in source, "still trimming into a plain str"


def test_an_uncached_repo_is_not_refused_by_the_autoconfig_gate(monkeypatch):
    """The gate refused any explicit token whose probe failed, including for a repo with
    nothing on disk. There it protects nothing: AutoConfig's own authenticated request is
    what the Hub checks. A mirror that serves /resolve but not the undocumented /auth-check,
    or one transient probe failure, then broke capability detection for a usable model."""
    from utils.models import model_config as mc

    monkeypatch.setattr(mc, "is_local_path", lambda *_a, **_k: False)
    monkeypatch.setattr(mc, "cache_reads_authorized", lambda *_a, **_k: False)
    monkeypatch.setattr(mc, "active_hf_hub_cache", lambda: None, raising = False)

    calls = {"n": 0}

    def _from_pretrained(*_a, **_k):
        calls["n"] += 1
        return SimpleNamespace(model_type = "llama")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", _from_pretrained)

    # Nothing cached: the read must reach the Hub, which enforces its own access control.
    monkeypatch.setattr(mc, "_config_json_already_cached", lambda *_a, **_k: False)
    mc.load_model_config("acme/not-cached", token = "hf_explicit")
    assert calls["n"] == 1, "an uncached repo was refused for nothing"

    # Cached: the operator's disk is reachable, so authorization is required again.
    monkeypatch.setattr(mc, "_config_json_already_cached", lambda *_a, **_k: True)
    with pytest.raises(OSError):
        mc.load_model_config("acme/cached-private", token = "hf_explicit")
    assert calls["n"] == 1, "the cached repo still reached AutoConfig"


def test_a_request_that_asked_for_offline_does_not_probe(monkeypatch):
    """cache_reads_authorized only saw the process-level env, so a request carrying its own
    offline=true still put the caller's token and repo id on the wire and could stall for the
    full probe timeout, to reach a branch that was never going to use the network."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    probes = {"n": 0}

    def _probe(*_a, **_k):
        probes["n"] += 1
        return True

    monkeypatch.setattr(hf_tokens, "_probe_repo_access", _probe)

    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private", offline = True) is False
    assert probes["n"] == 0, "an offline request still went to the network"

    # Online, the same call still probes: the flag is the caller's, not a new default.
    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private") is True
    assert probes["n"] == 1


def test_an_offline_request_still_honours_a_memoized_authorization(monkeypatch):
    """Fail-closed must not throw away a decision already paid for: the memo is consulted
    before the offline short-circuit, so a token verified moments ago keeps its answer."""
    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    probes = {"n": 0}
    monkeypatch.setattr(
        hf_tokens,
        "_probe_repo_access",
        lambda *_a, **_k: probes.__setitem__("n", probes["n"] + 1) or True,
    )

    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private") is True
    assert probes["n"] == 1
    # Now the caller asks for cache-only service: the memoized yes still stands.
    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private", offline = True) is True
    assert probes["n"] == 1


def test_an_uncached_dataset_is_not_denied_for_an_unavailable_probe(monkeypatch):
    """Same rule as the config reader: authorize only where a cached read could be served.
    An uncached dataset has nothing to leak, so denying it just costs a legitimate caller its
    preview whenever the probe is unavailable rather than negative, which is what an
    HF_ENDPOINT mirror without the undocumented /auth-check route looks like."""
    from hub.services.datasets import formatting
    from hub.schemas.datasets import CheckFormatRequest

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)
    monkeypatch.setattr(formatting, "dataset_cache_can_answer", lambda *_a, **_k: False)

    reached = {"n": 0}

    def _reached(*_a, **_k):
        reached["n"] += 1
        raise RuntimeError("reached the loader, which is all this asserts")

    monkeypatch.setattr("datasets.load_dataset", _reached)

    api_key = hf_token_arg("hf_explicit", allow_ambient_token = False)
    with pytest.raises(Exception):
        formatting.check_format_response(
            CheckFormatRequest(dataset_name = "acme/not-cached"), api_key
        )
    assert reached["n"] > 0, "an uncached dataset was refused for nothing"

    # Cached: the operator's disk can answer, so authorization is required again.
    monkeypatch.setattr(formatting, "dataset_cache_can_answer", lambda *_a, **_k: True)
    reached["n"] = 0
    with pytest.raises(HTTPException) as excinfo:
        formatting.check_format_response(
            CheckFormatRequest(dataset_name = "acme/cached-private"), api_key
        )
    assert excinfo.value.status_code == 404
    assert reached["n"] == 0


def test_the_dataset_cache_predicate_counts_both_caches(monkeypatch):
    """`datasets` answers a streaming load from its own PREPARED cache, which is where the
    measured leak was; the hub snapshot backs the file-level readers. Either can answer."""
    from hub.utils import dataset_cache as dc

    monkeypatch.setattr(dc, "latest_processed_dataset_cache_path", lambda *_a, **_k: None)
    monkeypatch.setattr(dc, "latest_cached_dataset_snapshot", lambda *_a, **_k: None)
    assert dc.dataset_cache_can_answer("acme/ds") is False

    monkeypatch.setattr(dc, "latest_processed_dataset_cache_path", lambda *_a, **_k: Path("/x"))
    assert dc.dataset_cache_can_answer("acme/ds") is True

    monkeypatch.setattr(dc, "latest_processed_dataset_cache_path", lambda *_a, **_k: None)
    monkeypatch.setattr(dc, "latest_cached_dataset_snapshot", lambda *_a, **_k: Path("/y"))
    assert dc.dataset_cache_can_answer("acme/ds") is True

    def _boom(*_a, **_k):
        raise OSError("cache unreadable")

    monkeypatch.setattr(dc, "latest_processed_dataset_cache_path", _boom)
    assert dc.dataset_cache_can_answer("acme/ds") is True, "a failed check must not open the gate"


def test_the_remote_code_scan_is_gated_before_any_scanner_runs():
    """The authorization check only guarded the prefer_local snapshot optimization, so a
    denied caller left scan_target as the repo id and the scan ran anyway. Downstream
    _load_remote_code_configs uses hf_hub_download, which resolves a cached private repo's
    configs without consulting the credential and can answer a definitive has_remote_code."""
    import inspect

    source = inspect.getsource(models_routes.scan_model_remote_code)
    gate = source.find("_repo_in_any_hf_cache(model_name)")
    optimization = source.find("prefer_local_cache is True")

    assert gate != -1, "the scan is not gated on the repo being cached"
    assert 0 < gate < optimization, "the gate must precede the prefer_local optimization"


def test_the_embedding_planner_gates_every_disk_backed_branch():
    """_resolve_embedding_model_plan called _cached_st_source and _cached_embedding_gguf with
    no notion of the caller, so the resolve endpoint reported cached private-repo state to an
    unauthorized caller, and the PUT accepted the same model through hf_cache_snapshot_is_loadable."""
    import inspect
    from routes import settings as settings_routes

    plan_src = inspect.getsource(settings_routes._resolve_embedding_model_plan)
    assert "cache_reads_authorized(token, repo_id = resolved)" in plan_src
    for helper in ("_cached_st_source(resolved)", "_cached_embedding_gguf("):
        idx = plan_src.find(helper)
        assert idx != -1, f"{helper} vanished; this test no longer guards anything"
    assert plan_src.count("if cache_ok else None") == 3, "a disk-backed branch is still ungated"

    put_src = inspect.getsource(settings_routes.update_embedding_model)
    loadable = put_src.find("hf_cache_snapshot_is_loadable(verify_target)")
    authorized = put_src.find("cache_reads_authorized(hf_token, repo_id = verify_target)")
    assert authorized != -1 and authorized < loadable, "the offline fallback is still ungated"

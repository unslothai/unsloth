# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The forced-anonymous sentinel is a credential of its own, not the absence of one.

Planting a cached repo is deliberate: a denial gate only fires where the disk could answer,
so "cached" is the premise every refusal test rests on. The uncached direction has its own
tests, asserting the request reaches the Hub instead.

``hf_token_arg`` returns three values where everything downstream expects ``Optional[str]``,
so ``False`` breaks four ways: truthiness reaches for the ambient token, identity hits
``.encode()`` on a bool, a shared fingerprint crosses the boundary, and a child env inherits
what it was never granted. ``is`` throughout, since ``False == 0 == ""``.
"""

import asyncio
import hashlib
import inspect
import os
import threading
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

import fastapi
import pytest
import requests
import transformers
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import utils.models.model_config as model_config_module
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import diffusion_compat
from hub.dependencies import get_request_hf_token
from hub.schemas.datasets import CheckFormatRequest
from hub.services.datasets import formatting
from hub.services.models import gguf_variants
from hub.utils import dataset_cache, hf_tokens
from hub.utils.gguf import GgufVariantInfo
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
from routes import settings as settings_routes
from routes.data_recipe import seed as seed_routes
from utils import transformers_version
from utils.models.model_config import _token_fingerprint as capability_fingerprint
from utils.transformers_version import _token_cache_key


@pytest.fixture(autouse = True)
def _isolate_repo_access_cache():
    """``_repo_access_cache`` is a process global, so one test's memoized verdict would
    otherwise decide the next one's. Every test in this file used to reset it by hand."""
    reset_repo_access_cache()
    yield
    reset_repo_access_cache()


def _hub_reachable(monkeypatch, *, offline: bool = False) -> None:
    """Pin the offline question, which is the other half of nearly every probe setup."""
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: offline)


def _counting_probe(
    monkeypatch,
    verdict = True,
    *,
    offline: Optional[bool] = None,
    delay: float = 0.0,
) -> dict:
    """Stub the access probe and return a live counter, so a test can assert both the
    verdict it produced and how many round trips it cost.

    ``offline`` pins the other half of nearly every probe setup. Left at ``None`` the real
    ``_hub_offline`` stands, which is what the tests that never stubbed it expect.
    """
    if offline is not None:
        _hub_reachable(monkeypatch, offline = offline)
    calls = {"n": 0}

    def _probe(*_a, **_k):
        calls["n"] += 1
        if delay:
            time.sleep(delay)
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict

    monkeypatch.setattr(hf_tokens, "_probe_repo_access", _probe)
    return calls


def _router_client(
    router,
    prefix: str,
    *,
    via_api_key: bool,
    subject: str = "alice",
) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


def _models_client(via_api_key: bool) -> TestClient:
    return _router_client(models_routes.router, "/api/models", via_api_key = via_api_key)


def _st_resolver(monkeypatch, **overrides) -> None:
    """The sentence-transformers preamble of ``_resolve_embedding_model_plan``.

    Named overrides replace a default rather than adding to it, so a test still says which
    stub it cares about and the rest stays out of the way.
    """
    stubs = {
        "_llama_backend_active": lambda _m: False,
        "_local_sentence_transformer_is_present": lambda _m: False,
        "_st_weight_source": lambda *_a, **_k: None,
    }
    for name, value in {**stubs, **overrides}.items():
        monkeypatch.setattr(settings_routes, name, value)


def _gguf_resolver(monkeypatch, **overrides) -> None:
    """The llama-server preamble: six stubs is what it takes to reach the candidate lookup."""
    stubs = {
        "_llama_backend_active": lambda _m: True,
        "_llama_runtime_available": lambda: True,
        "_resolves_as_local_gguf": lambda _m: False,
        "_local_gguf_backend_error": lambda _m: None,
        "_embedding_gguf_candidates": lambda m: [m],
        "_cached_embedding_gguf": lambda *_a, **_k: None,
    }
    for name, value in {**stubs, **overrides}.items():
        monkeypatch.setattr(settings_routes, name, value)


def _picker_env(
    monkeypatch,
    *,
    offline: bool = False,
    no_snapshots: bool = False,
) -> None:
    """The template walk reads its own offline flag and canonicalizes the repo id first.

    ``no_snapshots`` empties the snapshot walk, which is how a test says the fallback is
    the only route left to the template.
    """
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: offline)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    if no_snapshots:
        monkeypatch.setattr(
            picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter(())
        )


def _scan_refused(model_name: str, hf_token = "hf_dummy") -> int:
    """Drive the remote-code scan as an api key and return the status it refused with."""
    with pytest.raises(fastapi.HTTPException) as excinfo:
        asyncio.run(
            models_routes.scan_model_remote_code(
                model_name = model_name,
                hf_token = hf_token,
                allow_ambient_token = False,
                current_subject = "alice",
            )
        )
    return excinfo.value.status_code


def _one_cached_quant(
    monkeypatch,
    tmp_path,
    *,
    complete: bool = True,
):
    """A repo publishing one Q4_K_M, present in a single snapshot.

    The premise the GGUF listing tests rest on: a denial only means something where the disk
    could have answered. ``complete`` drops the bytes for the partial-state case.
    """
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    if complete:
        (snapshot / "Model-Q4_K_M.gguf").write_bytes(b"x" * 256)
    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (
            [GgufVariantInfo(filename = "Model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 256)],
            False,
            [],
        ),
    )
    monkeypatch.setattr(gguf_variants, "iter_hf_cache_snapshots", lambda *_a, **_k: [snapshot])
    return snapshot


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
    probes = _counting_probe(monkeypatch, False, offline = False)

    assert cache_reads_authorized(False, repo_id = "org/private") is False
    assert cache_reads_authorized(None, repo_id = "org/private") is True
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is False
    assert probes["n"] == 1, "ambient and anonymous must not pay a Hub round trip"
    assert cache_reads_authorized("hf_dummy", repo_id = "org/private") is False
    assert probes["n"] == 1, "the negative answer was not memoized"


def test_a_ui_session_that_saved_a_token_still_reads_its_own_cache(monkeypatch):
    """The UI attaches the operator's saved token on most Hub routes, so an ordinary
    single-user session is an EXPLICIT-token caller on exactly the routes this gates.

    Measured: with a token saved and the Hub unreachable, that session lost the GGUF variant
    list, the chat template and the dataset format check on repos already in its own cache.
    With no token saved it kept all three.
    """
    probes = _counting_probe(monkeypatch, None, offline = True)

    ui = hf_token_arg("  hf_saved  ", allow_ambient_token = True)
    api_key = hf_token_arg("  hf_saved  ", allow_ambient_token = False)

    assert ui == "hf_saved" and api_key == "hf_saved", "same value, different caller class"
    assert cache_reads_authorized(ui, repo_id = "org/private") is True
    assert cache_reads_authorized(api_key, repo_id = "org/private") is False
    assert probes["n"] == 0, "the UI session must not pay a round trip for its own token"


def test_the_ambient_marker_is_a_string_everywhere_it_is_used_as_a_value(monkeypatch):
    """Value transparent, entitlement not. Every consumer wanting the token VALUE sees a
    plain token; cache IDENTITY is the deliberate exception, pinned by the caller-class
    test below (this test once asserted the opposite, which was the bug)."""
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


@pytest.mark.parametrize(
    "cached, authorized, refused",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_the_shared_gate_refuses_only_where_a_cache_could_answer(
    monkeypatch, cached, authorized, refused
):
    _counting_probe(monkeypatch, authorized, offline = False)
    assert (
        hf_tokens.cached_read_refused(
            "hf_explicit", repo_id = "acme/private", is_cached = lambda: cached
        )
        is refused
    )


def test_the_shared_gate_asks_the_cache_before_the_hub(monkeypatch):
    """``is_cached`` is local and the probe is a round trip, so nothing on disk must mean no
    probe at all."""
    probes = _counting_probe(monkeypatch, False, offline = False)

    assert (
        hf_tokens.cached_read_refused(
            "hf_explicit", repo_id = "acme/private", is_cached = lambda: False
        )
        is False
    )
    assert probes["n"] == 0, "an uncached repo paid a Hub round trip"


def test_a_verified_token_may_read_the_host_cache(monkeypatch):
    _counting_probe(monkeypatch, True, offline = False)

    assert cache_reads_authorized("hf_real", repo_id = "org/private") is True


def test_offline_explicit_token_is_denied_without_a_prior_probe(monkeypatch):
    """Fail closed: no Hub round trip means no cache for an unverified credential."""
    _hub_reachable(monkeypatch, offline = True)

    assert cache_reads_authorized("hf_real", repo_id = "org/private") is False


def test_offline_explicit_token_may_use_a_recent_online_probe(monkeypatch):
    probes = _counting_probe(monkeypatch, True, offline = False)

    assert cache_reads_authorized("hf_real", repo_id = "org/private") is True
    assert probes["n"] == 1

    _hub_reachable(monkeypatch, offline = True)
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


def test_a_redirect_is_not_an_authorization(monkeypatch):
    """hf_raise_for_status passes 3xx through, and a client factory installed without
    follow_redirects would hand a bare 307 back. Legacy repo aliases do redirect."""
    _hub_reachable(monkeypatch)
    _patch_auth_check_get(
        monkeypatch,
        lambda *_a, **_k: SimpleNamespace(status_code = 307, raise_for_status = lambda: None),
    )

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False


def test_a_hanging_auth_check_probe_times_out(monkeypatch):
    """A stalled /auth-check must fail closed without leaving probe workers behind."""
    _hub_reachable(monkeypatch)
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


@pytest.mark.parametrize(
    "exc_factory, is_timeout",
    [
        # A "httpx." prefix matched none of these: httpx's exceptions live in ``httpx``.
        (lambda: __import__("httpx").TimeoutException("stalled"), True),
        (lambda: __import__("httpx").ConnectTimeout("stalled"), True),
        (lambda: __import__("httpx").PoolTimeout("stalled"), True),
        # A refusal, a DNS failure and a dead proxy are "could not ask" as much as a stall.
        (lambda: __import__("requests").exceptions.ConnectionError("refused"), True),
        (lambda: __import__("requests").exceptions.ProxyError("dead proxy"), True),
        (lambda: __import__("httpx").ConnectError("refused"), True),
        (lambda: ConnectionRefusedError("refused"), True),
        # The other half: a real refusal keeps the full TTL, or a revoked token re-probes.
        (lambda: _gated_hub_error(), False),
        (lambda: OSError("refused"), False),
        (lambda: ValueError("bad token"), False),
    ],
)
def test_could_not_ask_is_told_apart_from_told_no(exc_factory, is_timeout):
    """The unreachable TTL is short and the denial TTL is long, so misclassifying either way
    is a real outage: a stall that denies for a minute, or a revocation re-probed forever."""
    pytest.importorskip("httpx")
    assert hf_tokens._is_probe_timeout(exc_factory()) is is_timeout


@pytest.mark.parametrize(
    "endpoint, repo_id, expected_url",
    [
        # A raw "?" ends the path at /api/models/{id}, the repo_info weakness this avoids.
        (None, "org/gated?ignored=", "/api/models/org/gated%3Fignored%3D/auth-check"),
        (None, "unsloth/Llama-3.2-1B", "/api/models/unsloth/Llama-3.2-1B/auth-check"),
        # HfApi().endpoint returns HF_ENDPOINT verbatim, so a scheme-less mirror built a URL
        # both clients reject, denying every explicit-token cache read on that machine.
        (
            "hf-mirror.example",
            "org/repo",
            "https://hf-mirror.example/api/models/org/repo/auth-check",
        ),
        (
            "https://hf-mirror.example/",
            "org/repo",
            "https://hf-mirror.example/api/models/org/repo/auth-check",
        ),
    ],
)
def test_the_probe_url_is_built_safely(monkeypatch, endpoint, repo_id, expected_url):
    """The probe hand-builds the auth-check URL, so id quoting and endpoint normalization are
    its own responsibility."""
    _hub_reachable(monkeypatch)
    if endpoint is not None:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = repo_id) is True
    url = session.calls[0]["url"]
    assert "?" not in url
    assert url.endswith(expected_url)


@pytest.mark.parametrize("repo_id", ["org/../x", "../x", "org/./x", "..", "."])
def test_a_dot_segment_repo_id_is_refused_before_the_wire(monkeypatch, repo_id):
    """Quoting keeps "/" so "org/repo" stays two segments, which keeps ".." intact too, and
    both clients then apply RFC 3986 dot-segment removal: "org/../x" is requested as
    "/api/models/x". The memo would be keyed on what the caller named while the wire proof
    was about a different repo."""
    _hub_reachable(monkeypatch)
    session = _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert cache_reads_authorized("hf_dummy", repo_id = repo_id) is False
    assert session.calls == []


def test_an_unreachable_probe_takes_the_short_ttl_and_spans_the_next_request(monkeypatch):
    """A timeout says nothing about the credential, so it must not deny it for the full minute,
    and the short TTL must still outlive the probe or every caller re-pays the stall."""
    _hub_reachable(monkeypatch)
    monkeypatch.setattr(hf_tokens, "_REPO_ACCESS_PROBE_TIMEOUT_S", 0.2)
    probes = {"n": 0}

    def _refuse(*_a, **_k):
        probes["n"] += 1
        raise requests.exceptions.ConnectionError("refused")

    _patch_auth_check_get(monkeypatch, _refuse)

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False
    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False
    assert probes["n"] == 1, "the unreachable answer did not survive to the next request"

    # Held under the SHORT ttl, not the denial one, and the constants must stay ordered so
    # the memo cannot expire before the next request reaches it.
    (expiry, allowed) = next(iter(hf_tokens._repo_access_cache.values()))
    assert allowed is False
    assert expiry - time.monotonic() <= hf_tokens._REPO_ACCESS_UNREACHABLE_TTL_S
    assert (
        hf_tokens._REPO_ACCESS_PROBE_TIMEOUT_S
        < hf_tokens._REPO_ACCESS_UNREACHABLE_TTL_S
        < hf_tokens._REPO_ACCESS_TTL_S
    )


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_a_gated_repo_denies_cache_reads_for_an_invalid_token(monkeypatch, repo_type):
    """auth_check 401s; serving the host cache would bypass the gate. This is also why the
    probe is auth_check and not repo_info: repo_info succeeds on a gated repo's public
    metadata for an invalid token, so its success authorizes nothing."""
    _hub_reachable(monkeypatch)
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
    _hub_reachable(monkeypatch)
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
    _counting_probe(monkeypatch, False)
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


@pytest.mark.parametrize(
    "hf_token, probe, cached, may_download, metadata_ok",
    [
        # picker's fallback is hf_hub_download, which serves the cached copy when the Hub is
        # unreachable without consulting the credential. Its gate was hf_env_offline(), and
        # "hub unreachable" is not "env offline", so an unverified token still got it.
        ("hf_dummy", False, True, False, False),
        # Ambient is entitled to the operator's cache, so it keeps the fallback.
        (None, None, True, True, False),
        # Nothing cached is nothing to withhold: refusing here would break the ordinary
        # first-run download, which is the common case rather than the edge case.
        ("hf_dummy", False, False, True, False),
        # A gated repo can publish its file metadata publicly, so a 200 from get_paths_info
        # is not permission to download: hf_hub_download returns the cached pointer for any
        # failed head call, a 403 as much as an unreachable Hub, before it re-raises.
        ("hf_dummy", False, True, False, True),
        # The sentinel cannot authorize itself, which is not the same as being unentitled:
        # a public repo is one it was always allowed to read, and refusing there costs an
        # ordinary API-key caller the prompt formatting for a public model.
        (False, True, True, True, True),
        (False, False, True, False, True),
    ],
    ids = [
        "unverified-denied",
        "ambient-served",
        "uncached-still-fetched",
        "public-metadata-is-not-authorization",
        "anonymous-keeps-a-public-template",
        "anonymous-loses-a-private-one",
    ],
)
def test_the_chat_template_fallback_follows_the_caller(
    monkeypatch, hf_token, probe, cached, may_download, metadata_ok
):
    if probe is not None:
        _counting_probe(monkeypatch, probe, offline = False)
    _picker_env(monkeypatch, no_snapshots = True)
    # The gate asks about THIS template file, not the repo directory: a snapshot holding
    # only weights can serve no template, so refusing it would cost an authorized caller
    # one the Hub would have given it.
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda **_k: "/cached/repo/chat_template.jinja" if cached else None,
    )
    downloads: list = []

    class _Api:
        def __init__(self, *_a, **_k):
            pass

        def get_paths_info(self, _repo, paths, *_a, **_k):
            if not metadata_ok:
                raise ConnectionError("hub unreachable")
            return [SimpleNamespace(path = paths[0], size = 1024)]

    def _download(*a, **k):
        downloads.append(a)
        raise FileNotFoundError("stop here; reaching the fallback is what this asserts")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert picker_service.read_default_chat_template("org/private", hf_token) is None
    assert bool(downloads) is may_download


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
    seen = {}

    def _fake_list(*, dataset_name, token):
        seen["token"] = token
        return []

    monkeypatch.setattr(seed_routes, "_list_hf_data_files", _fake_list)

    for via_api_key in (True, False):
        _router_client(seed_routes.router, "/api/data-recipe", via_api_key = via_api_key).post(
            "/api/data-recipe/seed/inspect",
            json = {"dataset_name": "org/private-seed"},
            headers = {"Authorization": "Bearer token"},
        )
        assert seen["token"] is (
            False if via_api_key else None
        ), "hardcoding allow_ambient_token=False takes the fallback away from the UI too"


def test_an_explicit_seed_token_wins_for_either_caller(monkeypatch):
    seen = {}
    _counting_probe(monkeypatch, True)
    monkeypatch.setattr(
        seed_routes,
        "_list_hf_data_files",
        lambda *, dataset_name, token: seen.update(token = token) or [],
    )
    monkeypatch.setattr(seed_routes, "load_dataset", lambda **_k: iter([]), raising = False)

    for via_api_key in (True, False):
        _router_client(seed_routes.router, "/api/data-recipe", via_api_key = via_api_key).post(
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

    monkeypatch.setattr(requests, "get", _get)
    model_config_module._detect_audio_from_tokenizer("org/private", hf_token = False, revision = None)

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
    probes = _counting_probe(monkeypatch, False)
    assert formatting._load_any_cached_hf_preview_slice(request, 5, "hf_dummy") is None
    assert called["processed"] == 0, "the raw slice answered; the processed cache is extra reach"

    # The disk read runs first now, and the gate stands between it and the caller: reading our
    # own disk is not the leak, returning it is. That ordering is also what keeps a cache-only
    # request off the wire, since a miss never reaches the probe at all.
    called["cache"] = 0
    monkeypatch.setattr(formatting, "_load_cached_hf_preview_slice", lambda *_a, **_k: None)
    monkeypatch.setattr(formatting, "_load_processed_hf_preview_slice", lambda *_a, **_k: None)
    # The verdict above is memoized for 60s, which would hide the round trip this asserts.
    hf_tokens.reset_repo_access_cache()
    probes["n"] = 0
    assert formatting._load_any_cached_hf_preview_slice(request, 5, "hf_dummy") is None
    assert probes["n"] == 0, "a cache miss still put the caller's token on the wire"


def test_an_anonymous_caller_does_not_read_a_cached_chat_template(monkeypatch):
    """The snapshot walk returns a private repo's raw template with no Hub call.

    The walk itself runs for everyone: it reads our own disk, which is not the leak, and
    ordering it first is what keeps an uncached repo off the wire. Handing the template
    back is what is gated.
    """
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: [Path("/snap")]
    )
    monkeypatch.setattr(picker_service, "_chat_template_from_dir", lambda *_a, **_k: "TEMPLATE")

    assert picker_service.read_default_chat_template("org/private", None) == "TEMPLATE"

    assert (
        picker_service.read_default_chat_template("org/private", False) != "TEMPLATE"
    ), "the anonymous caller was handed the cached template"
    probes = _counting_probe(monkeypatch, False)
    assert (
        picker_service.read_default_chat_template("org/private", "hf_dummy") != "TEMPLATE"
    ), "an unverified token was handed the cached template"

    # Nothing cached: there is no disk-backed answer to protect, so no probe is warranted.
    monkeypatch.setattr(picker_service, "_chat_template_from_dir", lambda *_a, **_k: None)
    hf_tokens.reset_repo_access_cache()
    probes["n"] = 0
    picker_service.read_default_chat_template("org/uncached", "hf_dummy")
    assert probes["n"] == 0, "an uncached repo still cost the caller a probe"


def test_the_config_inspection_target_still_uses_the_cache_for_the_ambient_caller():
    """A caller allowed the ambient credential keeps the prefer_local_cache fast path.

    With no snapshot on disk the resolver raises its own 404; reaching that proves the
    cache branch ran rather than short-circuiting back to the bare repo id.
    """
    with pytest.raises(fastapi.HTTPException):
        models_routes._model_config_inspection_target("org/private", True, None, None)


def test_the_config_inspection_target_skips_the_cache_for_an_unverified_token(monkeypatch):
    _counting_probe(monkeypatch, False)
    reset_repo_access_cache()
    target = models_routes._model_config_inspection_target("org/private", True, None, "hf_dummy")
    assert target == "org/private", "an unverified token was pointed at the cache"


def test_the_config_inspection_target_uses_the_cache_for_a_verified_token(monkeypatch):
    _counting_probe(monkeypatch, True)
    reset_repo_access_cache()
    with pytest.raises(fastapi.HTTPException):
        models_routes._model_config_inspection_target("org/private", True, None, "hf_real")


def test_the_config_inspection_target_skips_the_cache_for_anonymous():
    """The snapshot read returns private metadata without consulting the token."""
    target = models_routes._model_config_inspection_target("org/private", True, None, False)

    assert target == "org/private", "the anonymous caller was pointed at the cache"


def test_resolving_the_hub_token_keeps_the_anonymous_sentinel():
    """Rebuilding the sentinel must not quietly restore ambient access."""
    assert models_routes._resolve_hub_token(False, None) is False
    assert models_routes._resolve_hub_token(False, "  ") is False
    assert models_routes._resolve_hub_token(None, None) is None
    assert models_routes._resolve_hub_token(False, "hf_query") == "hf_query"
    assert models_routes._resolve_hub_token("hf_header", "hf_query") == "hf_header"


def test_resolving_the_hub_token_never_returns_an_unresolved_dependency():
    """Callers that invoke the route function directly leave a ``Depends`` in the slot.

    The backend's own tests do exactly that, so returning ``header_token`` unchanged put
    a ``Depends`` object into the cache fingerprint and 500ed the whole variants route.
    """
    from fastapi import Depends

    unresolved = Depends(lambda: None)

    resolved = models_routes._resolve_hub_token(unresolved, None)

    assert resolved is None, "an unresolved dependency reached the Hub call"


def test_an_anonymous_caller_gets_no_template_from_the_offline_fallback(monkeypatch):
    """Offline, hf_hub_download answers from disk and never checks the credential.

    The chat-template route forces offline whenever the Hub looks unreachable, so
    without this the Hub fallback hands back the template the cache walk just refused.
    """
    _picker_env(monkeypatch, offline = True)

    def _exploded(*args, **kwargs):
        raise AssertionError("the anonymous caller reached the hub fallback")

    monkeypatch.setattr(picker_service, "iter_snapshots_preferring_whole", _exploded)

    assert picker_service.read_default_chat_template("org/private", False) is None


def test_an_anonymous_config_read_does_not_strip_the_process_credential(monkeypatch):
    """The sentinel goes to the hub as `token=False`, not via without_hf_auth().

    That context deletes HF_TOKEN and moves the login token files process-wide, so a
    concurrent download in another worker thread would lose the operator's credential.
    """
    monkeypatch.setenv("HF_TOKEN", "ambient-operator-token")
    seen = {}

    class _Config:
        pass

    def _from_pretrained(model_name, **kwargs):
        seen["token"] = kwargs.get("token", "<absent>")
        seen["ambient"] = os.environ.get("HF_TOKEN")
        return _Config()

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
    _counting_probe(monkeypatch, False)
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
    _counting_probe(monkeypatch, False)
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
    _counting_probe(monkeypatch, False)
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
    _counting_probe(monkeypatch, False)
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: True)
    monkeypatch.setattr(transformers_version, "_safe_is_file", lambda _p: False)
    monkeypatch.setattr(transformers_version, "_safe_is_dir", lambda _p: False)
    monkeypatch.setattr(transformers_version, "_config_json_cache", {})
    reads = {"n": 0}

    def _from_cache(_name):
        reads["n"] += 1
        return {"max_position_embeddings": 4096}

    monkeypatch.setattr(transformers_version, "_config_json_from_hf_cache", _from_cache)

    cfg = transformers_version._load_config_json("org/private", hf_token = hf_token)

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
    signature = inspect.signature(models_routes.scan_model_remote_code)

    assert (
        "allow_ambient_token" in signature.parameters
    ), "the scan route cannot tell an api key from a ui session"
    source = inspect.getsource(models_routes.scan_model_remote_code)
    assert (
        "hf_token_arg(hf_token" in source
    ), "the body token reaches the cache-backed scan target unresolved"


@pytest.mark.parametrize(
    "reason, hf_token, offline",
    [
        # Offline, `datasets` serves a streaming load from its own cache without reaching an
        # authorization check, so a cached private dataset would come back as rows.
        ("the anonymous caller, offline", None, True),
        # Online, the same load is satisfied from cache without asking the Hub.
        ("an unverified token, online", "hf_dummy", False),
    ],
)
def test_a_seed_preview_off_the_cache_is_refused(monkeypatch, reason, hf_token, offline):
    _counting_probe(monkeypatch, False)
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: offline)
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda *_a, **_k: True)

    def _never(*_a, **_k):
        raise AssertionError(f"{reason} reached the dataset load")

    monkeypatch.setattr(seed_routes, "_list_hf_data_files", _never)

    payload = SimpleNamespace(
        dataset_name = "org/private",
        split = None,
        subset = None,
        hf_token = hf_token,
        preview_size = 5,
    )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        asyncio.run(seed_routes.inspect_seed_dataset(payload, allow_ambient_token = False))

    assert excinfo.value.status_code == 404


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_the_lora_resolver_does_not_launder_the_sentinel(monkeypatch, hf_token):
    """`hf_token if hf_token else None` turned the sentinel back into ambient access."""
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
    _counting_probe(monkeypatch, False)
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

    _counting_probe(monkeypatch, False)
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
    # The rest of the condition this marker opens, bounded by indentation: it ends at the
    # closing "):", which dedents. A fixed character window drifted with every edit to the
    # surrounding function, and widened the claim silently rather than failing.
    lines = inspect.getsource(models_routes.scan_model_remote_code).splitlines()
    start = next(i for i, line in enumerate(lines) if "prefer_local_cache is True" in line)
    depth = len(lines[start]) - len(lines[start].lstrip())
    condition = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < depth:
            break
        condition.append(line)
    branch = "\n".join(condition)

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
    """The three routes that reach disk offline all consult a shared rule rather than
    open-coding one, which is how the per-site version drifted six times before."""

    shared_rules = ("anonymous_and_offline", "refuse_unauthorized_dataset_preview")
    for owner, name in (
        (models_routes.get_model_config, "/config"),
        (models_routes.scan_model_remote_code, "scan-remote-code"),
        (formatting.check_format_response, "check-format"),
    ):
        source = inspect.getsource(owner)
        assert any(
            rule in source for rule in shared_rules
        ), f"{name} can still be answered from disk for a denied caller"


def test_an_unreachable_hub_is_a_503_not_a_404_and_not_a_500():
    """check-format has no refusal of its own: a denied caller returns None from the disk
    route, falls through to a Hub that cannot answer, and the pair below came back. They
    were not mapped, so the catch-all turned "I could not reach the Hub" into a 500.

    404 is the wrong repair. "I could not ask" is not "it is not there", and
    ``openai_auto_download._admit_and_start`` calls ``_mark_not_servable`` for every 404,
    which suppresses the repo for ten minutes. ``force_hf_offline`` flips
    ``huggingface_hub.constants`` process-wide for the length of one unreachable download,
    so a concurrent /v1 request really does see ``OfflineModeIsEnabled`` from a plain
    ``model_info``. 503 is retryable, which is also what routes/training's preflight
    already retries on."""
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
    from hub.utils.hf_errors import hf_error_status

    assert hf_error_status(LocalEntryNotFoundError("no cache, no hub")) == 503
    assert hf_error_status(OfflineModeIsEnabled("offline")) == 503
    assert hf_error_status(RuntimeError("boom")) is None


def test_a_forced_offline_window_does_not_quarantine_a_repo_for_ten_minutes():
    """The regression the mapping above caused: ``_admit_and_start`` reads the status, and
    404 both reports "not found on Hugging Face" and pins the repo in ``_not_servable``."""
    from unittest import mock

    import huggingface_hub
    from huggingface_hub.errors import OfflineModeIsEnabled

    from core.inference import openai_auto_download as auto

    auto._not_servable.clear()

    class _Api:
        def __init__(self, *a, **k):
            pass

        def model_info(self, *a, **k):
            raise OfflineModeIsEnabled("offline mode is enabled")

    with mock.patch.object(huggingface_hub, "HfApi", _Api):
        refusal = asyncio.run(
            auto._admit_and_start(
                "org/repo", "Q4_K_M", "org/repo:Q4_K_M", None, auto._Active(repo_id = "org/repo")
            )
        )

    assert refusal is not None and refusal.status == 503, "a forced-offline window is retryable"
    assert not auto._is_not_servable("org/repo", None), "an unreachable Hub quarantined the repo"


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


@pytest.mark.parametrize(
    "via_api_key, gated",
    [(True, True), (False, False)],
    ids = ["api-key-denied", "ui-session-served"],
)
def test_the_format_check_gates_the_streaming_tiers_by_caller(monkeypatch, via_api_key, gated):
    """Measured against the installed ``datasets``: offline, ``load_dataset`` answers BOTH
    ``streaming=True`` tiers out of its own prepared cache, logging "using the latest cached
    version", with the token never consulted. Both tiers run on the default
    ``prefer_local_cache=false``, ahead of the guarded cache reader, so the gate stands in
    front of them. The UI leg is the regression guard: a session holding its own saved token
    is entitled to ambient and must reach the loader with no probe."""
    probes = _counting_probe(monkeypatch, False)
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda *_a, **_k: True)

    loads = {"n": 0}

    def _loader(*_a, **_k):
        loads["n"] += 1
        raise RuntimeError("reached the loader, which is all this leg asserts")

    monkeypatch.setattr("datasets.load_dataset", _loader)

    token = hf_token_arg("hf_saved", allow_ambient_token = not via_api_key)
    request = CheckFormatRequest(dataset_name = "acme/private-secrets")

    with pytest.raises(Exception) as excinfo:
        formatting.check_format_response(request, token)

    if gated:
        assert isinstance(excinfo.value, HTTPException) and excinfo.value.status_code == 404
        assert loads["n"] == 0, "load_dataset ran for an unauthorized caller"
    else:
        assert loads["n"] > 0, "an ordinary UI session was denied its own dataset"
        assert probes["n"] == 0, "the UI session paid a round trip for its own token"


@pytest.mark.parametrize(
    "explicit, memoized",
    [(True, False), (False, True)],
    ids = ["explicit-token-re-derives", "ambient-keeps-memo"],
)
def test_the_offline_config_memo_follows_the_caller(monkeypatch, explicit, memoized):
    """``cache_reads_authorized`` expires in 60 s so a revoked token stops reading, but
    ``_config_json_cache`` has no TTL: memoizing a value that came off the operator's disk
    outlived the access it was granted under. Ambient still memoizes, or every read becomes a
    fresh disk walk."""
    transformers_version._config_json_cache.clear()
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: True)
    reads = {"n": 0}

    def _from_cache(*_a, **_k):
        reads["n"] += 1
        return {"model_type": "secret"}

    monkeypatch.setattr(transformers_version, "_config_json_from_hf_cache", _from_cache)
    authorized = {"v": True}
    monkeypatch.setattr(
        transformers_version, "cache_reads_authorized", lambda *_a, **_k: authorized["v"]
    )

    token = hf_token_arg("hf_explicit", allow_ambient_token = False) if explicit else None
    assert transformers_version._load_config_json("acme/private", token) == {"model_type": "secret"}

    if memoized:
        assert transformers_version._load_config_json("acme/private", token) == {
            "model_type": "secret"
        }
        assert reads["n"] == 1, "the ambient memo stopped working"
    else:
        # Revoke: an explicit token must re-derive rather than replay the memo.
        authorized["v"] = False
        assert transformers_version._load_config_json("acme/private", token) is None


def test_the_vision_config_read_refuses_an_unauthorized_cache_fallback(monkeypatch, tmp_path):
    """Measured against the installed huggingface_hub: with a planted cache entry and a dead
    endpoint, ``hf_hub_download(local_files_only=False)`` returns the operator's CACHED
    config.json for a token that cannot read the repo, because it falls back to disk whenever
    the Hub is unreachable and never consults the credential to do it."""
    # A real cached config, so an unguarded read returns the leak rather than an error.
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text('{"vision_config": {}, "model_type": "secret"}')
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    monkeypatch.setattr(model_config_module, "cache_reads_authorized", lambda *_a, **_k: False)

    # Counted, not raised: the reader wraps everything in `except Exception`, so an
    # AssertionError thrown in here would be swallowed and the test would pass unguarded.
    calls = {"n": 0}

    def _download(*_a, **_k):
        calls["n"] += 1
        return str(tmp_config)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    api_key = hf_token_arg("hf_cannot_read_this", allow_ambient_token = False)
    assert (
        model_config_module._raw_config_has_vision_config("acme/private-vlm", hf_token = api_key)
        is None
    )
    assert calls["n"] == 0, "hf_hub_download ran for an unauthorized cached repo"


def test_a_failed_cache_check_does_not_open_the_path_it_guards(monkeypatch):
    """If the cache lookup itself raises, the guard must assume a hit and authorize."""

    def _boom(*_a, **_k):
        raise OSError("cache unreadable")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _boom)
    assert model_config_module._config_json_already_cached("acme/private-vlm") is True


def test_the_remote_code_scan_refuses_a_cached_repo_it_cannot_authorize(monkeypatch):
    """_load_remote_code_configs calls hf_hub_download without local_files_only, which
    serves a cached file on an unreachable Hub without consulting the credential, so
    has_remote_code is answered off the operator's disk; gating only the prefer_local path
    left the scan running anyway. Uncached is the shared gate's own test."""
    _counting_probe(monkeypatch, False, offline = False)
    monkeypatch.setattr(models_routes, "_repo_in_any_hf_cache", lambda *_a, **_k: True)
    # config.json cached, which is what makes the scan answerable off disk.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_k: "/cache/config.json")

    async def _call():
        return await models_routes.scan_model_remote_code(
            model_name = "acme/private-vlm",
            hf_token = "hf_cannot_read_this",
            allow_ambient_token = False,
            current_subject = "alice",
        )

    with pytest.raises(fastapi.HTTPException) as excinfo:
        asyncio.run(_call())
    assert excinfo.value.status_code == 404


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
    from core.inference import diffusion_compat as dc

    ui = hf_token_arg("hf_saved", allow_ambient_token = True)
    api = hf_token_arg("hf_saved", allow_ambient_token = False)

    for fingerprint in (
        inventory_scan.token_fingerprint,
        model_config_module._token_fingerprint,
        dc._token_fingerprint,
    ):
        ui_id, api_id = fingerprint(ui), fingerprint(api)
        assert ui_id != api_id, f"{fingerprint.__module__} still collides across caller classes"
        # The qualifier must not smuggle the credential into a dict key.
        assert "hf_saved" not in str(ui_id) and "hf_saved" not in str(api_id)

    # The sentinel and the ambient caller keep the identities they already had.
    assert inventory_scan.token_fingerprint(False) == ANONYMOUS_CACHE_IDENTITY
    assert inventory_scan.token_fingerprint(None) == ""


def test_the_legacy_query_token_is_classified_like_the_header(monkeypatch):
    """normalize_token can carry a marker through but cannot create one, and ?hf_token= never
    had it: the value arrives as a bare string, not from the dependency. Measured before this,
    for one UI session sending one token two ways with the Hub unreachable:

        query   type=str                    authorized=False
        header  type=AmbientAuthorizedToken authorized=True
    """
    _counting_probe(monkeypatch, False)

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
    ui = hf_token_arg("hf_saved", allow_ambient_token = True)
    api = hf_token_arg("hf_saved", allow_ambient_token = False)

    monkeypatch.setattr(model_config_module, "_embedding_detection_cache", {})
    monkeypatch.setattr(model_config_module, "is_local_path", lambda *_a, **_k: False)
    monkeypatch.setattr(
        model_config_module, "_embedding_marker_in_hf_cache", lambda *_a, **_k: True
    )
    monkeypatch.setattr("utils.utils.hf_env_offline", lambda: False)

    def _no_hub(*_a, **_k):
        raise RuntimeError("hub unreachable, so the cached marker is the fallback")

    monkeypatch.setattr("huggingface_hub.model_info", _no_hub)
    # Only the UI session is entitled to the operator's cache here.
    monkeypatch.setattr(
        model_config_module,
        "cache_reads_authorized",
        lambda token, **_k: isinstance(token, hf_tokens.AmbientAuthorizedToken),
    )

    # The UI session classifies from its own cached marker and memoizes that.
    assert model_config_module.is_embedding_model("acme/private-emb", ui) is True
    # The API caller must be refused, not handed the entry the UI just wrote.
    assert model_config_module.is_embedding_model("acme/private-emb", api) is False


def test_an_offline_request_is_fail_closed_but_keeps_a_paid_for_answer(monkeypatch):
    """``cache_reads_authorized`` saw only the process-level env, so a request carrying its own
    offline=true still put the caller's token and repo id on the wire and could stall for the
    full probe timeout, to reach a branch that was never going to use the network. Fail closed
    must not discard an authorization already paid for, so the memo is consulted first."""
    probes = _counting_probe(monkeypatch, True, offline = False)

    # Cold and offline: refuse without asking.
    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private", offline = True) is False
    assert probes["n"] == 0, "an offline request still went to the network"

    # Online, the same call probes: the flag is the caller's, not a new default.
    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private") is True
    assert probes["n"] == 1

    # Now memoized, an offline call keeps the yes rather than downgrading it.
    assert cache_reads_authorized("hf_explicit", repo_id = "acme/private", offline = True) is True
    assert probes["n"] == 1


def test_an_uncached_dataset_is_not_denied_for_an_unavailable_probe(monkeypatch):
    """Same rule as the config reader: authorize only where a cached read could be served.
    An uncached dataset has nothing to leak, so denying it just costs a legitimate caller its
    preview whenever the probe is unavailable rather than negative, which is what an
    HF_ENDPOINT mirror without the undocumented /auth-check route looks like."""
    _counting_probe(monkeypatch, False)
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda *_a, **_k: False)

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
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda *_a, **_k: True)
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


def test_a_redirected_probe_does_not_authorize_the_repo_it_left(monkeypatch):
    """get_session builds httpx.Client(follow_redirects=True), so the 3xx check never sees
    the hop: a redirect to a login page returns a 200 that approved somewhere else."""
    _hub_reachable(monkeypatch)
    _patch_auth_check_get(
        monkeypatch,
        lambda *_a, **_k: SimpleNamespace(
            status_code = 200,
            raise_for_status = lambda: None,
            url = "https://huggingface.co/login",
        ),
    )

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False


def test_a_cache_only_caller_is_never_put_on_the_wire(monkeypatch):
    """`local_files_only` and `prefer_local_cache` are promises, not hints. Both gates read
    only the process env, so on a host with no offline variables they probed anyway: one Hub
    round trip per cached repo in the /loras scan, for an answer neither branch would use."""
    probes = _counting_probe(monkeypatch, True, offline = False)

    assert (
        model_config_module._offline_cache_read_refused("hf_dummy", "acme/m", "acme/m", True)
        is True
    )

    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda *_a, **_k: True)
    with pytest.raises(HTTPException) as excinfo:
        formatting.check_format_response(
            CheckFormatRequest(dataset_name = "acme/ds", prefer_local_cache = True), "hf_dummy"
        )
    assert excinfo.value.status_code == 404
    assert probes["n"] == 0


@pytest.mark.parametrize("public, refused", [(True, False), (False, True)])
def test_an_anonymous_caller_keeps_a_public_cached_dataset_and_loses_a_private_one(
    monkeypatch, public, refused
):
    """Refused only under a declared offline env, which missed the case that matters: an
    unreachable Hub is not a Hub declared absent, and `datasets` takes its prepared cache
    either way. It cannot authorize itself, so ask whether the repo is public at all."""
    # Answers only for the dataset endpoint: asking /api/models/<dataset id>/auth-check
    # gets a 404 that reads as "private", which would refuse a public cached preview.
    monkeypatch.setattr(
        hf_tokens,
        "_probe_repo_access",
        lambda _repo, _token, repo_type: public and repo_type == "dataset",
    )
    _hub_reachable(monkeypatch)
    monkeypatch.setattr(dataset_cache, "dataset_cache_can_answer", lambda *_a, **_k: True)

    def _preview():
        dataset_cache.refuse_unauthorized_dataset_preview(False, "acme/ds")

    if refused:
        with pytest.raises(HTTPException) as excinfo:
            _preview()
        assert excinfo.value.status_code == 404
    else:
        _preview()


def test_the_public_probe_does_not_borrow_the_operators_login(monkeypatch):
    """`build_hf_headers(token=None)` falls back to the ambient login, which would call
    every private repo the operator can reach public. False is the value meaning none."""
    _hub_reachable(monkeypatch)
    seen: dict = {}

    def _headers(*, token = "unset", **_k):
        seen["token"] = token
        return {}

    monkeypatch.setattr("huggingface_hub.utils.build_hf_headers", _headers)
    _patch_auth_check_get(monkeypatch, lambda *_a, **_k: _ok_auth_check_response())

    assert hf_tokens.public_cache_read_authorized(repo_id = "acme/ds") is True
    assert seen["token"] is False


def test_the_public_verdict_takes_its_own_memo_key(monkeypatch):
    """A public repo answers 200 for every token, so sharing one key would let any string
    read back the anonymous verdict as its own authorization."""
    verdicts = iter([True, False])
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: next(verdicts))
    _hub_reachable(monkeypatch)

    assert hf_tokens.public_cache_read_authorized(repo_id = "acme/ds") is True
    assert cache_reads_authorized("hf_dummy", repo_id = "acme/ds") is False


def test_a_cached_alias_repo_is_authorized_in_its_own_right(monkeypatch):
    """One decision from the base covered lookups answering with a DIFFERENT repo: an alias
    or a derived `-GGUF` conversion. /auth-check returns 200 for any string on a public base,
    so that decision is nearly free and hands back a cached private conversion."""
    reachable = {"acme/base"}
    monkeypatch.setattr(
        hf_tokens,
        "_probe_repo_access",
        lambda repo_id, *_a, **_k: repo_id in reachable,
    )
    _hub_reachable(monkeypatch)
    _st_resolver(
        monkeypatch,
        _cached_st_source = lambda _m: (
            "sentence-transformers/private-conversion",
            Path("/cache/snap"),
        ),
    )

    plan = settings_routes._resolve_embedding_model_plan("acme/base", "hf_dummy")

    assert plan.cached is False, "the cached alias was never authorized"
    assert plan.error, "with the alias withheld there is no artifact to offer"


def test_the_gguf_listing_withholds_local_readiness_from_a_denied_caller(monkeypatch, tmp_path):
    """A gated repo can serve file metadata publicly, so the lister succeeding is not
    authorization. The cache-only and exception paths refuse that caller; the success path
    walked the snapshots anyway and reported `downloaded`, which is the same fact."""
    _one_cached_quant(monkeypatch, tmp_path)
    _hub_reachable(monkeypatch)

    def _answer(token):
        return asyncio.run(gguf_variants.get_gguf_variants_answer("acme/gated", hf_token = token))

    _counting_probe(monkeypatch, True)
    assert _answer("hf_dummy").response.variants[0].downloaded is True
    reset_repo_access_cache()
    _counting_probe(monkeypatch, False)
    answer = _answer("hf_dummy")
    assert answer.response.variants[0].downloaded is False
    assert answer.cache_authorized is False


def test_the_context_length_lookup_honours_the_listings_refusal(monkeypatch):
    """With no directory named the route falls back to the bare repo id, and reading that
    walks every local cache. The one local fact the service cannot suppress, so the answer
    carries the verdict out to it."""
    from models.models import GgufVariantsResponse as ServiceResponse

    reads: list = []
    monkeypatch.setattr(
        models_routes,
        "_read_native_context_length",
        lambda model, *, is_local: reads.append(model) or 8192,
    )

    async def _denied(repo_id, **_k):
        return gguf_variants.VariantsAnswer(
            ServiceResponse(repo_id = repo_id, variants = [], has_vision = False),
            None,
            False,
        )

    monkeypatch.setattr(gguf_variants, "get_gguf_variants_answer", _denied)
    monkeypatch.setattr(gguf_variants, "pinned_snapshot_for_request", lambda *_a, **_k: None)

    result = asyncio.run(
        models_routes.get_gguf_variants(
            repo_id = "acme/gated", hf_token = "hf_dummy", current_subject = "alice"
        )
    )

    assert result.context_length is None
    assert reads == [], "the cache walk ran for a caller the listing had just refused"


def test_every_scan_target_is_authorized_not_only_the_one_named(monkeypatch):
    """The scan expands to the adapter's base, native-audio dependencies and auto_map repos
    and downloads each with the same token, falling back to their cached configs and Python
    files. Refused, not dropped: an unscanned base would under-report has_remote_code."""
    reachable = {"acme/adapter"}
    monkeypatch.setattr(
        hf_tokens,
        "_probe_repo_access",
        lambda repo_id, *_a, **_k: repo_id in reachable,
    )
    _hub_reachable(monkeypatch)
    monkeypatch.setattr(models_routes, "_repo_in_any_hf_cache", lambda *_a, **_k: True)
    # config.json cached, which is what makes the scan answerable off disk.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_k: "/cache/config.json")
    monkeypatch.setattr(
        "core.inference.native_audio.native_audio_security_targets",
        lambda target, **_k: [target, "acme/private-base"],
    )

    assert _scan_refused("acme/adapter") == 404


def test_the_embedding_resolver_gates_the_base_repo_too(monkeypatch):
    """The per-candidate check answers for an alias or a derived conversion, and defers to
    the base decision when the cache answered under the requested id itself. That leg is the
    one the base decision still owns, so it needs its own row."""
    _counting_probe(monkeypatch, False, offline = False)
    _st_resolver(monkeypatch, _cached_st_source = lambda m: (m, Path("/cache/snap")))

    plan = settings_routes._resolve_embedding_model_plan("acme/private", "hf_dummy")

    assert plan.cached is False, "the operator's cached copy was reported to a denied caller"


def test_the_embedding_resolver_does_not_probe_before_a_cache_lookup(monkeypatch):
    """Every cache lookup in the resolver is authorized against the repo it actually matched,
    so gating the lookups on the requested id as well bought nothing and cost a /auth-check on
    every resolve, including the local and sentence-transformers paths that never read the GGUF
    cache. The probe is bounded at 10s against a 20s resolver deadline, so it was half the
    budget spent to reach a miss."""
    probes = _counting_probe(monkeypatch, True, offline = False)
    _st_resolver(monkeypatch, _local_sentence_transformer_is_present = lambda _m: True)

    plan = settings_routes._resolve_embedding_model_plan("acme/local-st", "hf_dummy")

    assert plan.cached is True
    assert probes["n"] == 0, "a local sentence-transformers hit still probed the Hub"


def test_the_scan_is_refused_before_it_expands_its_targets(monkeypatch):
    """The per-target check inside the loop cannot stand in for this one. Target expansion
    resolves the adapter's base and its native-audio dependencies, which reads the cache and
    talks to the Hub with the caller's token, so the primary has to be refused ahead of it."""
    _counting_probe(monkeypatch, False, offline = False)
    monkeypatch.setattr(models_routes, "_repo_in_any_hf_cache", lambda *_a, **_k: True)
    # config.json cached, which is what makes the scan answerable off disk.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_k: "/cache/config.json")

    def _must_not_run(*_a, **_k):
        raise AssertionError("target expansion ran for a caller that was already refused")

    monkeypatch.setattr("core.inference.native_audio.native_audio_security_targets", _must_not_run)

    assert _scan_refused("acme/private") == 404


def test_local_only_audio_detection_refuses_without_asking(monkeypatch):
    """The twin of the vision gate, and its own call site. /loras drives this one with
    local_files_only=True over every adapter it finds, so an unentitled caller must be told
    no from the disk state alone, without a probe and without a cached answer."""
    probes = _counting_probe(monkeypatch, True, offline = False)
    monkeypatch.setattr(model_config_module, "_env_offline", lambda: False)

    audio_type, definitive = model_config_module.detect_audio_type_checked(
        "acme/private-audio", hf_token = "hf_dummy", local_files_only = True
    )

    assert (audio_type, definitive) == (None, False)
    assert probes["n"] == 0


@pytest.mark.parametrize("public", [True, False])
def test_an_anonymous_caller_keeps_a_cached_public_gguf(monkeypatch, tmp_path, public):
    """The same rule the chat template needed, at the site that still asked the raw
    question. The sentinel can never authorize itself, so a downloaded public quant was
    reported as absent and a cache-only listing fell through to the network."""
    _one_cached_quant(monkeypatch, tmp_path)
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: public)
    _hub_reachable(monkeypatch)

    answer = asyncio.run(gguf_variants.get_gguf_variants_answer("acme/repo", hf_token = False))

    assert answer.response.variants[0].downloaded is public
    assert answer.cache_authorized is public


def test_the_safetensors_fallback_authorizes_the_repo_it_found(monkeypatch):
    """The GGUF branch was gated and this one was not: after the remote probes come back
    empty, the plan reads the operator's snapshot and reports the repo it is filed under as
    cached, which is how a denied caller discovers and then force-saves private weights."""
    _counting_probe(monkeypatch, False, offline = False)
    _gguf_resolver(monkeypatch)
    monkeypatch.setattr(settings_routes, "_remote_embedding_gguf_plan", lambda *_a, **_k: None)
    monkeypatch.setattr(settings_routes, "_search_hub_for_gguf", lambda *_a, **_k: None)
    monkeypatch.setattr(settings_routes, "_sentence_transformers_fallback_allowed", lambda _m: True)
    monkeypatch.setattr(
        settings_routes, "_safetensors_plan", lambda *_a, **_k: ("acme/private-st", [])
    )
    reported: list = []
    monkeypatch.setattr(
        settings_routes,
        "_cached_snapshot_has_st_weights",
        lambda repo: reported.append(repo) or True,
    )

    plan = settings_routes._resolve_embedding_model_plan("acme/private", "hf_dummy")

    assert plan.download_repo is None, "a denied caller was pointed at the private weights"
    assert plan.error, "with the snapshot withheld there is no artifact to offer"
    assert reported == [], "the cache was consulted for a repo nothing had authorized"


def test_the_config_memo_is_rechecked_before_it_is_served(monkeypatch):
    """The memo is untimed and was read before the authorization check, so a token revoked
    after one successful online fetch kept its answer for the life of the process, outliving
    the TTL that exists to end exactly that."""
    _hub_reachable(monkeypatch)
    key = transformers_version._token_cache_key("acme/private", "hf_dummy")
    monkeypatch.setitem(transformers_version._config_json_cache, key, {"model_type": "llama"})

    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: True)
    assert transformers_version._load_config_json("acme/private", "hf_dummy") == {
        "model_type": "llama"
    }

    reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: True)
    assert (
        transformers_version._load_config_json("acme/private", "hf_dummy") is None
    ), "revoked token kept the memo"


def test_the_inner_preview_gate_does_not_veto_the_outer_one(monkeypatch):
    """Two gates on the same request, and only the outer one was public-aware, so the
    anonymous sentinel cleared the guard and was then refused by the reader behind it:
    a prefer-local request answered local-cache-miss for a public dataset on disk."""
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: True)
    _hub_reachable(monkeypatch)
    monkeypatch.setattr(formatting, "_load_cached_hf_preview_slice", lambda *_a, **_k: ("ROWS", 3))

    served = formatting._load_any_cached_hf_preview_slice(
        SimpleNamespace(dataset_name = "acme/public"), 5, False
    )

    assert served == ("ROWS", 3), "a public cached preview was withheld from the sentinel"


def test_the_gguf_partial_state_is_withheld_with_the_rest(monkeypatch, tmp_path):
    """The snapshot walk was gated and the partial-download accounting beside it was not,
    so a caller who can list a gated repo's public metadata still learned the operator had
    an interrupted download: `partial`, its transport, and the bytes left to fetch."""
    reads: list = []
    _one_cached_quant(monkeypatch, tmp_path, complete = False)
    monkeypatch.setattr(
        gguf_variants.download_registry,
        "incomplete_blob_hashes",
        lambda *_a, **_k: reads.append("registry") or {"deadbeef"},
    )
    monkeypatch.setattr(
        gguf_variants,
        "_local_main_gguf_blobs_by_quant",
        lambda *_a, **_k: reads.append("blobs") or {},
    )
    monkeypatch.setattr(
        gguf_variants.hf_cache_scan,
        "is_variant_partial",
        lambda *_a, **_k: reads.append("manifest") or True,
    )
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: False)
    _hub_reachable(monkeypatch)

    answer = asyncio.run(gguf_variants.get_gguf_variants_answer("acme/gated", hf_token = "hf_dummy"))
    variant = answer.response.variants[0]

    assert variant.partial is False
    assert reads == [], "the operator's download state was read for a refused caller"


def test_a_repo_directory_is_not_a_cached_template(monkeypatch):
    """The predicate asks what the read could actually be served, which is one file. A repo
    cached for its weights alone holds no template, so refusing there protects nothing and
    costs an authorized caller the copy the Hub still has, whenever the probe is merely
    unavailable: a mirror without /auth-check, one transient failure."""
    _counting_probe(monkeypatch, False, offline = False)
    _picker_env(monkeypatch, no_snapshots = True)
    # A snapshot is here, the template file is not.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_k: None)
    downloads: list = []

    class _Api:
        def __init__(self, *_a, **_k):
            pass

        def get_paths_info(self, _repo, paths, *_a, **_k):
            return [SimpleNamespace(path = paths[0], size = 1024)]

    def _download(*a, **k):
        downloads.append(a)
        raise FileNotFoundError("reaching the download is what this asserts")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert picker_service.read_default_chat_template("org/repo", "hf_dummy") is None
    assert downloads, "a template the repo does not hold cached was refused anyway"


def test_the_transformers_cache_key_separates_the_caller_classes():
    """The qualifier reached three fingerprints and missed this one. The tokenizer and
    config-tier caches keyed here return before any authorization check, so a UI session
    populating one handed its verdict to an API caller with the same token text."""
    ui = hf_token_arg("hf_saved", allow_ambient_token = True)
    api = hf_token_arg("hf_saved", allow_ambient_token = False)

    assert _token_cache_key("acme/m", ui) != _token_cache_key("acme/m", api)
    assert "hf_saved" not in str(_token_cache_key("acme/m", ui))
    # The values that already had their own slots keep them.
    assert _token_cache_key("acme/m", False) == ("acme/m", ANONYMOUS_CACHE_IDENTITY)
    assert _token_cache_key("acme/m", None) == ("acme/m", None)


def test_a_resolved_gguf_plan_does_not_report_an_unauthorized_cache(monkeypatch):
    """A gated repo can publish its filenames, so a plan coming back is not permission to
    report the operator's copy of it, and the repo the plan names need not be the one the
    caller asked about."""
    _counting_probe(monkeypatch, False, offline = False)
    _gguf_resolver(monkeypatch)
    monkeypatch.setattr(
        settings_routes,
        "_remote_embedding_gguf_plan",
        lambda *_a, **_k: ("acme/private-GGUF", ["model-Q4_K_M.gguf"]),
    )
    reported: list = []
    monkeypatch.setattr(
        settings_routes,
        "_cached_embedding_gguf_files",
        lambda repo, files: reported.append(repo) or True,
    )
    monkeypatch.setattr(settings_routes, "_hf_files_size", lambda *_a, **_k: 1)

    plan = settings_routes._resolve_embedding_model_plan("acme/private", "hf_dummy")

    assert plan.cached is not True, "the operator's cached GGUF was reported to a denied caller"
    assert reported == [], "the cache was consulted for a repo nothing had authorized"


def test_a_weights_only_cache_does_not_block_the_scan(monkeypatch):
    """config.json, not the repo directory: has_remote_code comes from its auto_map and the
    Python files are reached through it, so a snapshot holding only weights can answer
    nothing. Refusing it cost a valid token the scan a mirror would have served, in exactly
    the case the probe is unavailable rather than negative."""
    _counting_probe(monkeypatch, False, offline = False)
    monkeypatch.setattr(models_routes, "_repo_in_any_hf_cache", lambda *_a, **_k: True)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_k: None)

    reached: list = []
    monkeypatch.setattr(
        "core.inference.native_audio.native_audio_security_targets",
        lambda target, **_k: reached.append(target) or [target],
    )

    try:
        asyncio.run(
            models_routes.scan_model_remote_code(
                model_name = "acme/weights-only",
                hf_token = "hf_dummy",
                allow_ambient_token = False,
                current_subject = "alice",
            )
        )
    except Exception:
        # Whatever the scan does next needs the network; getting there is the assertion.
        pass

    assert reached == ["acme/weights-only"], "a weights-only cache was treated as an answer"


def test_an_external_auto_map_repo_is_authorized_before_it_is_scanned(monkeypatch):
    """auto_map repos are read out of the primary's config, so they are discovered after the
    target loop has authorized what it knew. The preflight downloads and scans each one with
    the same token, and its cached Python files reach the response as source snippets."""
    reachable = {"acme/adapter"}
    monkeypatch.setattr(
        hf_tokens, "_probe_repo_access", lambda repo_id, *_a, **_k: repo_id in reachable
    )
    _hub_reachable(monkeypatch)
    monkeypatch.setattr(models_routes, "_repo_in_any_hf_cache", lambda *_a, **_k: True)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_k: "/cache/config.json")
    monkeypatch.setattr(
        "core.inference.native_audio.native_audio_security_targets",
        lambda target, **_k: [target],
    )
    monkeypatch.setattr(
        "utils.security.remote_code_scan.external_auto_map_repos",
        lambda *_a, **_k: ["acme/private-code"],
    )

    assert _scan_refused("acme/adapter") == 404


def test_the_cached_alias_is_looked_up_before_the_literal_name_is_judged(monkeypatch):
    """A slashless alias caches under sentence-transformers/, so gating the lookup on the
    literal name's verdict skipped it entirely when that name has no repo to authorize, and
    a public alias already fully cached came back as a download to run."""
    reachable = {"sentence-transformers/all-MiniLM-L6-v2"}
    monkeypatch.setattr(
        hf_tokens, "_probe_repo_access", lambda repo_id, *_a, **_k: repo_id in reachable
    )
    _hub_reachable(monkeypatch)
    _st_resolver(
        monkeypatch,
        _cached_st_source = lambda _m: (
            "sentence-transformers/all-MiniLM-L6-v2",
            Path("/cache/snap"),
        ),
    )

    plan = settings_routes._resolve_embedding_model_plan("all-MiniLM-L6-v2", "hf_dummy")

    assert plan.cached is True, "a cached, authorized alias was offered as a download"
    assert plan.download_repo == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize(
    "endpoint, final_url, authorized",
    [
        # httpx lower-cases the host and drops an explicit default port while HF_ENDPOINT
        # keeps its spelling, so comparing the strings denied these mirrors outright.
        (
            "https://HF-MIRROR.example",
            "https://hf-mirror.example/api/models/org/repo/auth-check",
            True,
        ),
        (
            "https://mirror.example:443",
            "https://mirror.example/api/models/org/repo/auth-check",
            True,
        ),
        ("https://mirror.example", "https://mirror.example/api/models/org/repo/auth-check/", True),
        # Still not the target we asked about.
        ("https://mirror.example", "https://mirror.example/login", False),
        ("https://mirror.example", "https://mirror.example/api/models/org/other/auth-check", False),
        # Same path, but a query is where a login hand-off hides.
        (
            "https://mirror.example",
            "https://mirror.example/api/models/org/repo/auth-check?next=login",
            False,
        ),
    ],
    ids = [
        "uppercase-host",
        "default-port",
        "trailing-slash",
        "login-page",
        "other-repo",
        "query-added",
    ],
)
def test_the_probe_compares_targets_not_spellings(monkeypatch, endpoint, final_url, authorized):
    """A 200 only counts when it came from the repo that was asked about, and the client
    is free to rewrite the parts of a URL that carry no meaning."""
    _hub_reachable(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    _patch_auth_check_get(
        monkeypatch,
        lambda *_a, **_k: SimpleNamespace(
            status_code = 200, raise_for_status = lambda: None, url = final_url
        ),
    )

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is authorized


def test_a_recorded_redirect_is_refused_whatever_the_final_url(monkeypatch):
    """The direct signal, for the clients that keep it: any hop at all means the answer
    came from somewhere other than the endpoint the memo is keyed on."""
    _hub_reachable(monkeypatch)
    _patch_auth_check_get(
        monkeypatch,
        lambda *_a, **_k: SimpleNamespace(
            status_code = 200,
            raise_for_status = lambda: None,
            history = [SimpleNamespace(status_code = 302)],
        ),
    )

    assert cache_reads_authorized("hf_dummy", repo_id = "org/repo") is False


@pytest.mark.parametrize(
    "cached_name", ["config.json", "tokenizer_config.json", "video_preprocessor_config.json"]
)
def test_the_scan_predicate_covers_every_config_the_scanner_reads(monkeypatch, cached_name):
    """auto_map is declared in any of REMOTE_CODE_CONFIG_FILES, so asking about config.json
    alone let four of the five open the gate. The lookup also has to name the cache the
    scanner's own downloads pass, or it asks the library default about a read that happens
    in the operator's chosen root."""
    seen: list = []
    _counting_probe(monkeypatch, False, offline = False)
    monkeypatch.setattr(models_routes, "_repo_in_any_hf_cache", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache", lambda: Path("/studio/cache")
    )

    def _lookup(
        *,
        repo_id,
        filename,
        cache_dir = None,
        **_k,
    ):
        seen.append((filename, str(cache_dir)))
        return "/studio/cache/hit" if filename == cached_name else None

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _lookup)

    assert _scan_refused("acme/private") == 404
    assert all(root == "/studio/cache" for _n, root in seen), "asked the wrong cache root"


def test_the_template_predicate_asks_the_active_cache(monkeypatch):
    """The download names active_hf_hub_cache(), so a predicate that omits it reports a miss
    for a template living only in the operator's chosen root and opens the gate on it."""
    roots: list = []
    _counting_probe(monkeypatch, False, offline = False)
    _picker_env(monkeypatch, no_snapshots = True)
    monkeypatch.setattr(picker_service, "active_hf_hub_cache", lambda: Path("/studio/cache"))

    def _lookup(
        *,
        repo_id,
        filename,
        cache_dir = None,
        **_k,
    ):
        roots.append(str(cache_dir))
        return "/studio/cache/template.jinja"

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _lookup)
    downloads: list = []

    class _Api:
        def __init__(self, *_a, **_k):
            pass

        def get_paths_info(self, _repo, paths, *_a, **_k):
            return [SimpleNamespace(path = paths[0], size = 1024)]

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: downloads.append(a) or "/x"
    )

    assert picker_service.read_default_chat_template("org/private", "hf_dummy") is None
    assert roots and all(r == "/studio/cache" for r in roots), "asked the wrong cache root"
    assert not downloads, "a template cached in the active root was served to a denied caller"


def test_a_tokenless_api_caller_keeps_a_public_cached_embedder(monkeypatch):
    """The sentinel cannot authorize itself, so the raw check refused it whatever the repo
    was: an online resolve offered a multi-gigabyte download of something already on disk."""
    monkeypatch.setattr(hf_tokens, "_probe_repo_access", lambda *_a, **_k: True)
    _hub_reachable(monkeypatch)
    _st_resolver(monkeypatch, _cached_st_source = lambda m: (m, Path("/cache/snap")))

    plan = settings_routes._resolve_embedding_model_plan("acme/public-embed", False)

    assert plan.cached is True, "a public cached embedder was offered as a download"


def test_a_local_only_config_read_stays_off_the_wire(monkeypatch):
    """local_files_only is a contract. The guard tested the caller's flag and did not
    forward it, so the read it protects dialled /auth-check anyway."""
    probes = _counting_probe(monkeypatch, True, offline = False)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)

    with pytest.raises(OSError):
        model_config_module.load_model_config(
            "acme/private", token = "hf_dummy", local_files_only = True
        )

    assert probes["n"] == 0

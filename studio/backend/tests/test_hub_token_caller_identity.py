# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A caller forced anonymous is a credential of its own, not the absence of one.

``hf_token_arg`` resolves a Hub token to three values, not two: an explicit token, ``None``
for a UI session that may fall back to the backend's ambient ``HF_TOKEN``, and ``False`` for
an API key that may not. Everything downstream predates the third value and was written for
``Optional[str]``, so the ways it goes wrong are specific:

* a truthiness test (``if token:``) treats ``False`` as "no token" and reaches for the
  ambient credential anyway, which is the boundary inverted rather than enforced;
* an identity test (``if token is None:``) sends ``False`` to ``.encode()`` and 500s;
* a cache fingerprint that folds both into one identity lets a UI session's private-repo
  metadata be read back by an API key, and lets an API key's anonymous 403 blank the UI;
* a child process seeded from ``os.environ`` inherits the ambient token unless it is
  scrubbed, so declining to *set* one is not the same as denying one.

These pin all four. ``is`` comparisons throughout: ``False == 0 == ""`` under ``==``.
"""

import hashlib


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import diffusion_compat
from hub.dependencies import get_request_hf_token
from hub.utils.hf_tokens import (
    ANONYMOUS_CACHE_IDENTITY,
    apply_token_to_child_env,
    hf_token_arg,
    is_anonymous,
    normalize_token,
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


# --- the sentinel itself -------------------------------------------------------------


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


def test_normalizing_a_token_does_not_launder_the_sentinel():
    # `(hf_token or "").strip() or None` predates the sentinel and turns "stay anonymous"
    # into "use the backend's credential", which is the one answer that must never happen.
    assert normalize_token(False) is False
    assert normalize_token(None) is None
    assert normalize_token("  tok  ") == "tok"
    assert normalize_token("") is None


# --- cache identity ------------------------------------------------------------------


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
    # And neither may collide with a real token's digest.
    assert fingerprint("hf_realtoken") not in (anonymous, ambient)


def test_the_anonymous_identity_cannot_collide_with_a_token_digest():
    # Non-hex by construction, so no token can ever hash to it.
    assert not all(character in "0123456789abcdef" for character in ANONYMOUS_CACHE_IDENTITY)


def test_a_fingerprint_never_carries_the_token():
    secret = "hf_supersecretvalue123"
    for fingerprint in (token_fingerprint, capability_fingerprint, diffusion_compat._token_fingerprint):
        assert secret not in str(fingerprint(secret))


def test_config_metadata_cache_keys_separate_anonymous_from_ambient():
    # These carry a private repo's config.json / tokenizer_config.json.
    assert _token_cache_key("org/private", False) != _token_cache_key("org/private", None)
    assert _token_cache_key("org/private", False) == ("org/private", ANONYMOUS_CACHE_IDENTITY)


def test_capability_fingerprint_does_not_raise_on_the_sentinel():
    # The regression that made /check-vision and /config 500 for every API-key caller:
    # `if token is None` let False through to `.encode()`.
    assert capability_fingerprint(False) == ANONYMOUS_CACHE_IDENTITY
    assert hashlib.sha256(b"x").hexdigest() != capability_fingerprint(False)


# --- child process credentials -------------------------------------------------------


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
    # The half of the boundary that is not a security property but a regression risk:
    # scrubbing here would break gated repos on installs whose token is in the env.
    env = _child_env(None)

    assert env["HF_TOKEN"] == "ambient-operator-token"
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in env


def test_an_explicit_token_replaces_the_ambient_one_in_the_child():
    env = _child_env("hf_caller_token")

    assert env["HF_TOKEN"] == "hf_caller_token"
    # An inherited "1" would 401 a gated repo the caller does have access to.
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"


# --- routes --------------------------------------------------------------------------


@pytest.mark.parametrize("via_api_key", [True, False])
def test_capability_routes_answer_both_callers_without_a_server_error(monkeypatch, via_api_key):
    """The blocker this file was written for: /check-vision 500ed for an API-key caller."""
    seen = {}

    def _fake_is_vision_model(model_name, hf_token = None, **_kwargs):
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
        # Neither explicit token: the caller's sentinel must survive the `or` chain. An
        # `or` that ended on the query value would turn False into None here.
        (False, None, False),
        (None, None, None),
        (False, "", False),
        # Precedence is header-over-query, as it was before the route was converted.
        ("header-token", "query-token", "header-token"),
        ("header-token", None, "header-token"),
        # The legacy query parameter still works as a fallback.
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
        assert seen["token"] is (False if via_api_key else None), (
            "hardcoding allow_ambient_token=False takes the fallback away from the UI too"
        )


def test_an_explicit_seed_token_wins_for_either_caller(monkeypatch):
    from routes.data_recipe import seed as seed_routes

    seen = {}
    monkeypatch.setattr(
        seed_routes,
        "_list_hf_data_files",
        lambda *, dataset_name, token: seen.update(token = token) or [],
    )

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


# --- the dependency ------------------------------------------------------------------


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

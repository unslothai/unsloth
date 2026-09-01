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
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

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


def _models_client(via_api_key: bool, subject: str = "unsloth") -> TestClient:
    app = FastAPI()
    app.include_router(models_routes.router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: subject
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


# The subject axis as well as the caller axis: the ambient token is the
# installation owner's own credential, so a managed account does not reach it
# either, and the sentinel is what says so.
@pytest.mark.parametrize(
    "via_api_key, subject",
    [(True, "unsloth"), (False, "unsloth"), (True, "alice"), (False, "alice")],
)
def test_capability_routes_answer_both_callers_without_a_server_error(
    monkeypatch, via_api_key, subject
):
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
    response = _models_client(via_api_key, subject).get(
        "/api/models/check-vision/org/some-model",
        headers = {"Authorization": "Bearer token"},
    )

    assert response.status_code == 200, response.text
    owner_ui_session = not via_api_key and subject == "unsloth"
    assert seen["hf_token"] is (None if owner_ui_session else False)


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


@pytest.mark.parametrize("subject", ["unsloth", "alice"])
def test_seed_inspection_derives_its_policy_from_the_caller(monkeypatch, subject):
    """The OWNER's UI session on an ambient-token install keeps gated seed inspection.

    A managed account does not: the process token is not that account's to spend,
    which is the same rule the hub download routes apply.
    """
    from routes.data_recipe import seed as seed_routes

    seen = {}

    def _fake_list(*, dataset_name, token):
        seen["token"] = token
        return []

    monkeypatch.setattr(seed_routes, "_list_hf_data_files", _fake_list)

    for via_api_key in (True, False):
        app = FastAPI()
        app.include_router(seed_routes.router, prefix = "/api/data-recipe")
        app.dependency_overrides[get_current_subject] = lambda: subject
        app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
        client = TestClient(app, raise_server_exceptions = False)

        client.post(
            "/api/data-recipe/seed/inspect",
            json = {"dataset_name": "org/private-seed"},
            headers = {"Authorization": "Bearer token"},
        )
        owner_ui_session = not via_api_key and subject == "unsloth"
        assert seen["token"] is (
            None if owner_ui_session else False
        ), "hardcoding allow_ambient_token=False takes the fallback away from the owner's UI too"


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
    assert called["cache"] == 1, "the anonymous caller reached the unauthenticated cache"
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


@pytest.mark.parametrize("hf_token", [None, "hf_tok"])
def test_the_config_inspection_target_still_uses_the_cache_for_the_ambient_caller(hf_token):
    """A caller allowed the ambient credential keeps the prefer_local_cache fast path.

    With no snapshot on disk the resolver raises its own 404; reaching that proves the
    cache branch ran rather than short-circuiting back to the bare repo id.
    """
    import fastapi
    with pytest.raises(fastapi.HTTPException):
        models_routes._model_config_inspection_target("org/private", True, None, hf_token)


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
    "hf_token, expected_local_only",
    [(None, True), ("hf_tok", True), (False, False)],
)
def test_the_config_probes_do_not_go_local_only_for_an_anonymous_caller(
    monkeypatch, hf_token, expected_local_only
):
    """local_files_only resolves config.json out of the cache without any authorization.

    Sending the anonymous caller back to the bare repo id only helps if the probe then
    goes over the wire, where `token=False` is refused for a private repo.
    """
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
                allow_ambient_token = hf_token is not False,
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

    if is_anonymous(hf_token):
        assert reads == {
            "snapshot": 0,
            "state": 0,
        }, "the anonymous caller was served from the hub cache"
    else:
        assert reads["snapshot"] > 0, "the authorized caller lost its cache fast path"


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_offline_capability_probes_do_not_read_the_cache_anonymously(monkeypatch, hf_token):
    """Offline, is_vision_model derives local_files_only from the environment.

    So passing local_files_only=False does not put the anonymous caller back on the wire:
    the probe reads the cached config.json off disk and never authorizes.
    """
    import utils.models.model_config as model_config_module

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

    if is_anonymous(hf_token):
        assert is_vision is False
        assert reached["vision"] == 0, "the anonymous caller probed the offline cache"
    else:
        assert reached["vision"] == 1, "the authorized caller lost its offline probe"


@pytest.mark.parametrize("hf_token", [None, "hf_tok", False])
def test_the_config_json_fallbacks_do_not_reach_the_cache_anonymously(monkeypatch, hf_token):
    """Keying the memo apart is not enough when the value came off disk to begin with."""
    import utils.transformers_version as tv

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

    if is_anonymous(hf_token):
        assert cfg is None
        assert reads["n"] == 0, "the anonymous caller read the offline config cache"
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

    if is_anonymous(hf_token):
        assert reads["n"] == 0, "the anonymous caller walked the hub cache"
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
    """token=False disables authentication but not the local cache."""
    import transformers

    import utils.models.model_config as model_config_module

    monkeypatch.setattr(model_config_module, "_env_offline", lambda: True)
    monkeypatch.setattr(model_config_module, "active_hf_hub_cache", lambda: None, raising = False)
    reached = {"n": 0}

    def _from_pretrained(_name, **_kwargs):
        reached["n"] += 1
        return object()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(_from_pretrained))

    if is_anonymous(hf_token):
        with pytest.raises(OSError):
            model_config_module.load_model_config("org/private", token = hf_token)
        assert reached["n"] == 0, "the anonymous caller read the offline config cache"
    else:
        model_config_module.load_model_config("org/private", token = hf_token)
        assert reached["n"] == 1


def test_the_prefer_local_scan_branch_carries_the_anonymous_guard():
    """Only the exact-snapshot branch was gated; the sibling resolved the cache anyway."""
    import inspect

    source = inspect.getsource(models_routes.scan_model_remote_code)
    marker = source.index("elif prefer_local_cache is True")
    branch = source[marker : marker + 200]

    assert (
        "not is_anonymous(hf_token)" in branch
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in auto-download of a GGUF a /v1 request names but this server lacks.

No network: huggingface_hub, the consent probe and the Hub download service are
all mocked. The invariant these guard is that with the setting off nothing here
runs at all, and with it on a name that isn't shaped like a repo still falls
through to the resident model.
"""

import asyncio

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from core.inference import openai_auto_download as auto_dl
from utils import openai_auto_switch_settings as settings


class _Sibling:
    def __init__(
        self,
        rfilename,
        size = 0,
    ):
        self.rfilename = rfilename
        self.size = size


class _Info:
    def __init__(
        self,
        siblings,
        sha = "abc123",
        gated = False,
        private = False,
    ):
        self.siblings = siblings
        self.sha = sha
        self.gated = gated
        self.private = private


def _gguf_repo_info():
    gb = 1024**3
    return _Info(
        [
            _Sibling("model-UD-Q4_K_XL.gguf", 4 * gb),
            _Sibling("model-UD-Q5_K_XL.gguf", 5 * gb),
            _Sibling("model-Q8_0-00001-of-00002.gguf", 4 * gb),
            _Sibling("model-Q8_0-00002-of-00002.gguf", 4 * gb),
            _Sibling("mmproj-F16.gguf", 1 * gb),
            _Sibling("mtp-model.gguf", 1 * gb),
            _Sibling("README.md", 1024),
        ]
    )


@pytest.fixture(autouse = True)
def _clean_slot():
    auto_dl.reset_for_tests()
    yield
    auto_dl.reset_for_tests()


@pytest.fixture
def hub(monkeypatch):
    """Wire the whole remote surface to fakes and record what was dispatched."""
    import huggingface_hub
    from hub.services.models import downloads

    state = {"info": _gguf_repo_info(), "raise": None, "auto_map": False, "started": []}

    class _FakeApi:
        def __init__(self, token = None):
            state["token"] = token

        def model_info(self, repo_id, **kwargs):
            if state["raise"] is not None:
                raise state["raise"]
            return state["info"]

    async def _start(body, hf_token = None):
        state["started"].append((body.repo_id, body.gguf_variant, hf_token))
        return {"job_key": "k", "state": "running"}

    async def _no_watch(active, hf_token):
        return None

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(downloads, "download_model_response", _start)
    monkeypatch.setattr(auto_dl, "_watch", _no_watch)
    monkeypatch.setattr(auto_dl, "_enough_disk", lambda need: (True, 10 * 1024**4))
    monkeypatch.setattr(
        "utils.security.consent._config_has_auto_map",
        lambda repo, token = None: state["auto_map"],
    )
    return state


def _run(model, hf_token = None):
    return asyncio.run(auto_dl.maybe_auto_download(model, hf_token = hf_token))


# --- pure helpers ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("org/repo:UD-Q4_K_XL", ("org/repo", "UD-Q4_K_XL")),
        ("org/repo", ("org/repo", None)),
        ("gpt-4", ("gpt-4", None)),
        # A colon followed by a path segment is not a quant.
        ("C:/models/x.gguf", ("C:/models/x.gguf", None)),
        ("org/repo:", ("org/repo:", None)),
    ],
)
def test_split_model_ref(raw, expected):
    assert auto_dl.split_model_ref(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "gpt-4",  # no namespace: a foreign id, must keep falling through
        "gpt-4o-mini",
        "../../etc/passwd",
        "https://evil.example/x",
        "/abs/path/model.gguf",
        "org/repo/extra",
        "org/re..po",
        "org/repo\nX-Injected: 1",
        "",
    ],
)
def test_not_downloadable(raw):
    assert auto_dl.is_downloadable_ref(raw) is False


@pytest.mark.parametrize(
    "raw", ["unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-31B-it-GGUF:UD-Q5_K_XL"]
)
def test_downloadable(raw):
    assert auto_dl.is_downloadable_ref(raw) is True


def test_gguf_variants_skips_companions():
    variants = auto_dl._gguf_variants(_gguf_repo_info().siblings)
    assert set(variants) == {"UD-Q4_K_XL", "UD-Q5_K_XL", "Q8_0"}
    # Shards of one quant sum together.
    assert variants["Q8_0"] == 8 * 1024**3


def test_match_variant_is_case_insensitive_and_exact():
    variants = {"UD-Q4_K_XL": 1, "Q8_0": 2}
    assert auto_dl._match_variant("ud-q4_k_xl", variants) == "UD-Q4_K_XL"
    assert auto_dl._match_variant("Q5_K_M", variants) is None
    # A bare id picks a real local label, never invents one.
    assert auto_dl._match_variant(None, variants) in variants


# --- admission ---------------------------------------------------------------


def test_foreign_id_never_probes(hub):
    assert _run("gpt-4") is None
    assert hub["started"] == []


def test_starts_download_and_asks_for_a_retry(hub):
    refusal = _run("unsloth/x-GGUF:UD-Q5_K_XL")
    assert refusal.status == 503
    assert refusal.code == "model_downloading"
    assert refusal.retry_after and refusal.retry_after > 0
    assert "unsloth/x-GGUF:UD-Q5_K_XL" in refusal.message
    assert hub["started"] == [("unsloth/x-GGUF", "UD-Q5_K_XL", None)]


def test_bare_id_freezes_the_same_quant_a_manual_load_would_pick(hub):
    from utils.models.model_config import _extract_quant_label, _pick_best_gguf

    refusal = _run("unsloth/x-GGUF")
    assert refusal.status == 503
    repo, variant, _token = hub["started"][0]
    expected = _extract_quant_label(
        _pick_best_gguf([s.rfilename for s in _gguf_repo_info().siblings])
    )
    assert (repo, variant) == ("unsloth/x-GGUF", expected)
    assert variant == "UD-Q4_K_XL"


def test_missing_quant_lists_the_real_ones(hub):
    refusal = _run("unsloth/x-GGUF:Q2_K")
    assert refusal.status == 404 and refusal.code == "model_not_found"
    assert "UD-Q4_K_XL" in refusal.message and "Q8_0" in refusal.message
    assert hub["started"] == []


def test_missing_repo_is_404_without_confirming_existence(hub):
    from huggingface_hub.utils import RepositoryNotFoundError

    hub["raise"] = RepositoryNotFoundError("nope")
    refusal = _run("unsloth/not-real")
    assert refusal.status == 404 and refusal.code == "model_not_found"
    assert "not accessible" in refusal.message
    assert hub["started"] == []


def test_gated_repo_is_403(hub):
    from huggingface_hub.utils import GatedRepoError

    hub["raise"] = GatedRepoError("gated")
    refusal = _run("meta-llama/Llama-2-7b-hf")
    assert refusal.status == 403 and refusal.code == "model_access_denied"
    assert hub["started"] == []


def test_hub_unreachable_is_retryable(hub):
    hub["raise"] = OSError("network down")
    refusal = _run("unsloth/x-GGUF")
    assert refusal.status == 503 and refusal.code == "model_lookup_failed"
    assert refusal.retry_after
    assert hub["started"] == []


def test_non_gguf_repo_is_refused(hub):
    hub["info"] = _Info([_Sibling("model.safetensors", 100), _Sibling("config.json", 10)])
    refusal = _run("unsloth/plain-transformers")
    assert refusal.status == 400 and refusal.code == "model_not_supported"
    assert hub["started"] == []


def test_remote_code_repo_is_refused(hub):
    hub["auto_map"] = True
    refusal = _run("someone/custom-arch-GGUF")
    assert refusal.status == 403 and refusal.code == "remote_code_consent_required"
    assert "Unsloth Studio" in refusal.message
    assert hub["started"] == []


def test_unreadable_config_fails_closed(hub):
    # _config_has_auto_map returns None when it cannot tell; refuse rather than
    # assume the repo is safe.
    hub["auto_map"] = None
    refusal = _run("someone/unknown-GGUF")
    assert refusal.status == 403 and refusal.code == "remote_code_consent_required"
    assert hub["started"] == []


def test_insufficient_disk_never_downgrades_the_quant(hub, monkeypatch):
    monkeypatch.setattr(auto_dl, "_enough_disk", lambda need: (False, 1024**3))
    refusal = _run("unsloth/x-GGUF:UD-Q5_K_XL")
    assert refusal.status == 507 and refusal.code == "insufficient_disk_space"
    assert hub["started"] == []


def test_second_model_waits_for_the_first(hub):
    assert _run("unsloth/first-GGUF").code == "model_downloading"
    refusal = _run("unsloth/second-GGUF")
    assert refusal.status == 503 and refusal.code == "model_download_busy"
    assert "unsloth/first-GGUF" in refusal.message
    # Only the first was dispatched.
    assert len(hub["started"]) == 1


def test_repeat_request_reports_progress_without_reprobing(hub, monkeypatch):
    assert _run("unsloth/x-GGUF:UD-Q5_K_XL").code == "model_downloading"

    async def _running(repo, variant):
        return "running", None

    async def _pct(repo, variant, expected, token):
        return 42.0

    monkeypatch.setattr(auto_dl, "_job_state", _running)
    monkeypatch.setattr(auto_dl, "_progress_percent", _pct)
    refusal = _run("unsloth/x-GGUF:UD-Q5_K_XL")
    assert refusal.code == "model_downloading" and "42%" in refusal.message
    assert len(hub["started"]) == 1


def test_progress_is_scaled_to_a_percentage(monkeypatch):
    # The hub service reports a 0-1 fraction; a raw 0.492 would render as "0%".
    from hub.services.models import downloads

    async def _fraction(
        repo_id,
        variant = "",
        expected_bytes = 0,
        hf_token = None,
    ):
        return {"progress": 0.492}

    monkeypatch.setattr(downloads, "get_gguf_download_progress_response", _fraction)
    percent = asyncio.run(auto_dl._progress_percent("org/repo", "Q4_K_M", 0, None))
    assert percent == pytest.approx(49.2)


def test_failed_job_surfaces_once_then_frees_the_slot(hub, monkeypatch):
    assert _run("unsloth/x-GGUF").code == "model_downloading"

    async def _errored(repo, variant):
        return "error", "disk exploded"

    monkeypatch.setattr(auto_dl, "_job_state", _errored)
    refusal = _run("unsloth/x-GGUF")
    assert refusal.status == 502 and "disk exploded" in refusal.message
    # Slot released, so a different model can now start.
    assert _run("unsloth/other-GGUF").code == "model_downloading"


def test_hf_token_is_passed_to_the_worker(hub):
    _run("unsloth/x-GGUF", hf_token = "hf_secret")
    assert hub["started"][0][2] == "hf_secret"


# --- route wiring ------------------------------------------------------------


class _Url:
    def __init__(self, path):
        self.path = path


class _Req:
    def __init__(
        self,
        path = "/v1/chat/completions",
        headers = None,
    ):
        self.url = _Url(path)
        self.headers = headers or {}


def _hook(model, request, enabled):
    import utils.openai_auto_switch_settings as s

    original = s.get_openai_auto_download_enabled
    s.get_openai_auto_download_enabled = lambda: enabled
    try:
        return asyncio.run(inference_route._maybe_auto_download_model(model, request))
    finally:
        s.get_openai_auto_download_enabled = original


def test_setting_off_does_nothing_at_all(hub):
    # The compatibility invariant: no probe, no dispatch, no raise.
    assert _hook("unsloth/x-GGUF:UD-Q5_K_XL", _Req(), enabled = False) is None
    assert hub["started"] == []


def test_hook_raises_the_openai_envelope_with_retry_after(hub):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _hook("unsloth/x-GGUF:UD-Q5_K_XL", _Req(), enabled = True)
    exc = excinfo.value
    assert exc.status_code == 503
    assert exc.headers and exc.headers["Retry-After"]
    assert exc.detail["error"]["code"] == "model_downloading"
    assert exc.detail["error"]["param"] == "model"
    assert exc.detail["error"]["type"] == "api_error"


def test_hook_uses_the_anthropic_envelope_on_messages(hub):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _hook("unsloth/x-GGUF", _Req(path = "/v1/messages"), enabled = True)
    detail = excinfo.value.detail
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "api_error"


def test_hook_swallows_unexpected_failures(hub, monkeypatch):
    # A broken download path must not turn a servable request into a 500.
    async def _boom(model, hf_token = None):
        raise RuntimeError("boom")

    monkeypatch.setattr(auto_dl, "maybe_auto_download", _boom)
    assert _hook("unsloth/x-GGUF", _Req(), enabled = True) is None


def test_hook_prefers_the_hub_header_token(hub):
    from fastapi import HTTPException
    from hub.dependencies import HUB_HF_TOKEN_HEADER

    with pytest.raises(HTTPException):
        _hook(
            "unsloth/x-GGUF",
            _Req(headers = {HUB_HF_TOKEN_HEADER: "hf_from_header"}),
            enabled = True,
        )
    assert hub["started"][0][2] == "hf_from_header"


# --- never answer as a different model ----------------------------------------


class _Loaded:
    """Minimal stand-in for the GGUF backend with one model resident."""

    def __init__(
        self,
        identifier,
        variant = None,
        advertised = None,
    ):
        self.is_loaded = True
        self.model_identifier = identifier
        self.hf_variant = variant
        self._openai_advertised_id = advertised


def _reject(
    model,
    loaded,
    monkeypatch,
    *,
    downloaded = False,
    auto_switch = False,
):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda name: ("/p", None, name) if downloaded else None,
    )
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled",
        lambda: auto_switch,
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    return asyncio.run(inference_route._reject_unservable_model(model, _Req()))


async def _fake_unavailable_message(model):
    return f"The model '{model}' is not downloaded on this server."


def test_wrong_quant_is_not_answered_by_the_loaded_one(monkeypatch):
    # The reported bug: asking for UD-Q6_K_XL while UD-Q4_K_XL is resident used to
    # return 200 from the wrong weights.
    loaded = _Loaded("unsloth/gemma-4-E2B-it-GGUF", "UD-Q4_K_XL")
    with pytest.raises(HTTPException) as excinfo:
        _reject("unsloth/gemma-4-E2B-it-GGUF:UD-Q6_K_XL", loaded, monkeypatch)
    assert excinfo.value.status_code == 404


def test_bare_repo_id_is_satisfied_by_any_loaded_quant(monkeypatch):
    # No quant named means "this model", so the resident quant answers it.
    loaded = _Loaded("unsloth/gemma-4-E2B-it-GGUF", "UD-Q4_K_XL")
    assert _reject("unsloth/gemma-4-E2B-it-GGUF", loaded, monkeypatch) is None


def test_matching_quant_is_served(monkeypatch):
    loaded = _Loaded("unsloth/gemma-4-E2B-it-GGUF", "UD-Q4_K_XL")
    assert _reject("unsloth/gemma-4-E2B-it-GGUF:ud-q4_k_xl", loaded, monkeypatch) is None


def test_advertised_alias_counts_as_serving(monkeypatch):
    # Loaded by path, requested by the repo id auto-switch advertised for it.
    loaded = _Loaded("/cache/snap/abc", "UD-Q4_K_XL", "unsloth/gemma-4-E2B-it-GGUF")
    assert _reject("unsloth/gemma-4-E2B-it-GGUF", loaded, monkeypatch) is None


@pytest.mark.parametrize("foreign", ["gpt-4", "gpt-4o-mini", "claude-3-5-sonnet", "default"])
def test_foreign_ids_still_fall_through(monkeypatch, foreign):
    # Drop-in compatibility: an id with no namespace is a label, not a reference.
    loaded = _Loaded("unsloth/gemma-4-E2B-it-GGUF", "UD-Q4_K_XL")
    assert _reject(foreign, loaded, monkeypatch) is None


def test_downloaded_but_auto_switch_off_says_so(monkeypatch):
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    with pytest.raises(HTTPException) as excinfo:
        _reject("unsloth/B-GGUF", loaded, monkeypatch, downloaded = True)
    assert "Switch model by request" in str(excinfo.value.detail)


def test_downloaded_with_auto_switch_on_falls_through(monkeypatch):
    # On disk and switching allowed means the swap failed; the resident model is
    # the sane fallback rather than a hard error.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    assert _reject("unsloth/B-GGUF", loaded, monkeypatch, downloaded = True, auto_switch = True) is None


def test_nothing_loaded_leaves_the_existing_error_alone(monkeypatch):
    # With no model resident the handler's own no-model-loaded error is already
    # correct, so this check must not preempt it.
    idle = type("B", (), {"is_loaded": False, "model_identifier": None, "hf_variant": None})()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: idle)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    assert asyncio.run(inference_route._reject_unservable_model("unsloth/B-GGUF", _Req())) is None


def test_reload_only_sentinel_is_ignored(monkeypatch):
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    assert _reject(inference_route._RELOAD_ONLY_MODEL, loaded, monkeypatch) is None


def test_diagnosis_failure_never_breaks_a_servable_request(monkeypatch):
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")

    def _boom(_name):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr("core.inference.local_model_resolver.resolve_local_gguf", _boom)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    assert asyncio.run(inference_route._reject_unservable_model("unsloth/B-GGUF", _Req())) is None


def test_anthropic_surface_gets_its_own_envelope(monkeypatch):
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr("core.inference.local_model_resolver.resolve_local_gguf", lambda name: None)
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            inference_route._reject_unservable_model("unsloth/B-GGUF", _Req(path = "/v1/messages"))
        )
    assert excinfo.value.detail["type"] == "error"


# --- settings ----------------------------------------------------------------


def test_auto_download_defaults_off_and_is_gated_on_auto_switch(monkeypatch):
    store = {}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    assert settings.get_stored_openai_auto_download_enabled() is False
    assert settings.get_openai_auto_download_enabled() is False

    store[settings.OPENAI_AUTO_DOWNLOAD_SETTING_KEY] = True
    # Stored on, but auto-switch off: nothing would load the result, so it is off.
    assert settings.get_stored_openai_auto_download_enabled() is True
    assert settings.get_openai_auto_download_enabled() is False

    store[settings.OPENAI_AUTO_SWITCH_SETTING_KEY] = True
    assert settings.get_openai_auto_download_enabled() is True


def test_setter_round_trips_auto_download_in_one_transaction(monkeypatch):
    import storage.studio_db as db

    calls = []
    store = {}

    def _upsert(mapping):
        calls.append(dict(mapping))
        store.update(mapping)

    monkeypatch.setattr(db, "upsert_app_settings", _upsert)
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))

    result = settings.set_openai_auto_switch(True, 120, None, True)
    assert result == (True, 120, True, True)
    assert len(calls) == 1
    assert calls[0][settings.OPENAI_AUTO_DOWNLOAD_SETTING_KEY] is True


def test_setter_rejects_a_non_boolean_auto_download(monkeypatch):
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: None)
    with pytest.raises(ValueError, match = "true or false"):
        settings.set_openai_auto_switch(True, None, None, "garbage")


def test_settings_route_exposes_auto_download(monkeypatch):
    import routes.settings as settings_route

    monkeypatch.setattr(settings_route, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(settings_route, "get_stored_auto_unload_idle_seconds", lambda: 0)
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)
    monkeypatch.setattr(settings_route, "get_auto_unload_keep_kv", lambda: True)
    monkeypatch.setattr(settings_route, "get_stored_openai_auto_download_enabled", lambda: True)
    assert settings_route.get_openai_auto_switch("tester").auto_download_model is True


# --- the placeholder API key -------------------------------------------------


def test_placeholder_api_key_gets_a_specific_message():
    from auth.authentication import API_KEY_PLACEHOLDER, _invalid_api_key_detail

    detail = _invalid_api_key_detail(API_KEY_PLACEHOLDER)
    assert "placeholder" in detail
    assert "Settings > API" in detail


def test_every_other_bad_key_stays_indistinguishable():
    from auth.authentication import _invalid_api_key_detail

    generic = "Invalid or expired API key"
    assert _invalid_api_key_detail("sk-unsloth-revoked") == generic
    assert _invalid_api_key_detail("sk-unsloth-YOUR_KEY ") == generic
    assert _invalid_api_key_detail("sk-unsloth-your_key") == generic

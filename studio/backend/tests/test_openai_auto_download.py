# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in auto-download of a GGUF a /v1 request names but this server lacks.

No network: huggingface_hub, the consent probe and the Hub download service are
all mocked. The invariant these guard is that with the setting off nothing here
runs at all, and with it on a name that isn't shaped like a repo still falls
through to the resident model.
"""

import asyncio
import time

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from core.inference import openai_auto_download as auto_dl
from core.inference.local_model_resolver import warm_index_soon as _real_warm_index_soon
from utils import openai_auto_switch_settings as settings


class _Sibling:
    def __init__(
        self,
        rfilename,
        size = 0,
        blob_id = None,
    ):
        self.rfilename = rfilename
        self.size = size
        self.blob_id = blob_id


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
    from core.inference import local_model_resolver

    auto_dl.reset_for_tests()
    # The hook warms the index in the background; drop it so one test's scan is
    # never handed to the next one inside the TTL.
    local_model_resolver.invalidate_index()
    yield
    auto_dl.reset_for_tests()
    local_model_resolver.invalidate_index()


def _repo_not_found_error():
    from huggingface_hub.utils import RepositoryNotFoundError
    return RepositoryNotFoundError


def _gated_error():
    from huggingface_hub.utils import GatedRepoError
    return GatedRepoError


def _hub_error(error_type, status_code: int, message: str):
    """Build a Hub exception across huggingface_hub majors.

    huggingface_hub 1.x made ``response`` a required keyword-only argument, and
    the project floor is 0.34, so construct positionally and fall back.
    """
    try:
        return error_type(message)
    except TypeError:
        import httpx
        return error_type(
            message,
            response = httpx.Response(
                status_code,
                request = httpx.Request("GET", "https://huggingface.co/api/models/org/repo"),
            ),
        )


@pytest.fixture
def hub(monkeypatch):
    """Wire the whole remote surface to fakes and record what was dispatched."""
    import huggingface_hub
    from hub.services.models import downloads

    state = {
        "info": _gguf_repo_info(),
        "raise": None,
        "auto_map": False,
        "started": [],
        "watched": [],
        # What the hub service returns; accepted=False means no worker was launched.
        "dispatch_result": {"job_key": "k", "state": "running", "accepted": True},
        "on_probe": None,
        "probes": 0,
        "auth_denied": False,
        "allow_ambient": None,
    }

    class _FakeApi:
        def __init__(self, token = None):
            state["token"] = token

        def model_info(self, repo_id, **kwargs):
            state["probes"] += 1
            if state["on_probe"] is not None:
                state["on_probe"]()
            if state["raise"] is not None:
                raise state["raise"]
            return state["info"]

    async def _start(
        body,
        hf_token = None,
        *,
        allow_ambient_token = True,
    ):
        state["started"].append((body.repo_id, body.gguf_variant, hf_token))
        state["allow_ambient"] = allow_ambient_token
        return state["dispatch_result"]

    async def _no_watch(active, hf_token):
        state["watched"].append(active)
        return None

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(downloads, "download_model_response", _start)
    # Keep the real watcher reachable: one test drives its cleanup directly.
    state["real_watch"] = auto_dl._watch
    monkeypatch.setattr(auto_dl, "_watch", _no_watch)
    monkeypatch.setattr(auto_dl, "_enough_disk", lambda need: (True, 10 * 1024**4))
    monkeypatch.setattr(auto_dl, "_auth_denied", lambda repo, token: state["auth_denied"])
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
    # Companions are not quants of their own...
    assert set(variants) == {"UD-Q4_K_XL", "UD-Q5_K_XL", "Q8_0"}
    # ...but every quant fetches them, so they count, and shards sum on top.
    companions = 2 * 1024**3  # mmproj + MTP drafter
    assert variants["Q8_0"] == 8 * 1024**3 + companions
    assert variants["UD-Q4_K_XL"] == 4 * 1024**3 + companions


def test_looks_like_quant_separates_quants_from_foreign_tags():
    assert auto_dl.looks_like_quant("UD-Q6_K_XL")
    assert auto_dl.looks_like_quant("q4_k_m")
    assert auto_dl.looks_like_quant("F16")
    # Ollama-style tags are not quants and must not read as a GGUF reference.
    assert not auto_dl.looks_like_quant("latest")
    assert not auto_dl.looks_like_quant("8b")
    assert not auto_dl.looks_like_quant(None)


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
    hub["raise"] = _hub_error(_repo_not_found_error(), 404, "nope")
    # An explicit quant is a deliberate GGUF reference, so a miss is answered.
    refusal = _run("unsloth/not-real:UD-Q4_K_XL")
    assert refusal.status == 404 and refusal.code == "model_not_found"
    assert "not accessible" in refusal.message
    assert hub["started"] == []


def test_an_id_the_hub_does_not_know_falls_through(hub):
    hub["raise"] = _hub_error(_repo_not_found_error(), 404, "nope")
    # LiteLLM and OpenRouter name every provider "vendor/model", so a namespace is not
    # intent: an id the Hub never heard of stays a foreign label the resident model answers.
    for foreign in (
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "meta-llama/llama-3-70b-instruct",
    ):
        assert _run(foreign) is None
    assert hub["started"] == []


def test_a_foreign_id_is_probed_once_then_cached(hub):
    hub["raise"] = _hub_error(_repo_not_found_error(), 404, "nope")
    assert _run("anthropic/claude-3.5-sonnet") is None
    assert hub["probes"] == 1
    # Every later request would otherwise pay another Hub round trip.
    assert _run("anthropic/claude-3.5-sonnet") is None
    assert hub["probes"] == 1


def test_an_anonymous_404_does_not_silence_an_authorised_caller(hub):
    # The Hub 404s a private repo the caller cannot see, so caching that verdict globally
    # would let one anonymous request hide it from the token holder for the whole TTL.
    hub["raise"] = _hub_error(_repo_not_found_error(), 404, "nope")
    assert _run("myorg/private-GGUF") is None
    assert hub["probes"] == 1

    hub["raise"] = None
    refusal = _run("myorg/private-GGUF", hf_token = "hf_caller_own")
    assert hub["probes"] == 2
    assert refusal.code == "model_downloading"


def test_the_cache_is_per_token(hub):
    hub["raise"] = _hub_error(_repo_not_found_error(), 404, "nope")
    assert _run("myorg/private-GGUF", hf_token = "hf_a") is None
    assert _run("myorg/private-GGUF", hf_token = "hf_a") is None
    assert hub["probes"] == 1
    # A different credential gets its own verdict.
    assert _run("myorg/private-GGUF", hf_token = "hf_b") is None
    assert hub["probes"] == 2


def test_the_gated_message_names_the_header_that_actually_works(hub):
    # Auto-download never uses the server's token, so naming a Studio setting would loop
    # the caller on the same 403.
    hub["info"] = _Info(_gguf_repo_info().siblings, gated = "manual")
    hub["auth_denied"] = True
    refusal = _run("meta-llama/Llama-2-7b-hf")
    assert "X-Unsloth-HF-Token" in refusal.message


def test_gated_repo_is_403(hub):
    hub["raise"] = _hub_error(_gated_error(), 403, "gated")
    refusal = _run("meta-llama/Llama-2-7b-hf")
    assert refusal.status == 403 and refusal.code == "model_access_denied"
    assert hub["started"] == []


def test_a_gated_repo_that_still_returns_metadata_is_403(hub):
    # Metadata for a gated repo is not file access, so report the licence gate rather than
    # the custom-code refusal the unreadable config would otherwise produce.
    hub["info"] = _Info(_gguf_repo_info().siblings, gated = "manual")
    hub["auth_denied"] = True
    refusal = _run("meta-llama/Llama-2-7b-hf")
    assert refusal.status == 403 and refusal.code == "model_access_denied"
    assert "licence" in refusal.message
    assert hub["started"] == []


def test_a_gated_repo_this_token_may_read_still_downloads(hub):
    hub["info"] = _Info(_gguf_repo_info().siblings, gated = "manual")
    refusal = _run("meta-llama/Llama-2-7b-hf")
    assert refusal.code == "model_downloading"
    assert len(hub["started"]) == 1


def test_hub_unreachable_is_retryable(hub):
    hub["raise"] = OSError("network down")
    refusal = _run("unsloth/x-GGUF")
    assert refusal.status == 503 and refusal.code == "model_lookup_failed"
    assert refusal.retry_after
    assert hub["started"] == []


def test_non_gguf_repo_is_refused(hub):
    hub["info"] = _Info([_Sibling("model.safetensors", 100), _Sibling("config.json", 10)])
    refusal = _run("unsloth/plain-transformers:Q4_K_M")
    assert refusal.status == 400 and refusal.code == "model_not_supported"
    assert hub["started"] == []


def test_a_bare_non_gguf_id_falls_through(hub):
    # Without a quant this is indistinguishable from a foreign provider label.
    hub["info"] = _Info([_Sibling("model.safetensors", 100)])
    assert _run("unsloth/plain-transformers") is None
    assert hub["started"] == []


def test_remote_code_repo_is_refused(hub):
    hub["auto_map"] = True
    refusal = _run("someone/custom-arch-GGUF")
    assert refusal.status == 403 and refusal.code == "remote_code_consent_required"
    assert "Unsloth Studio" in refusal.message
    assert hub["started"] == []


def test_unreadable_config_fails_closed(hub):
    # _config_has_auto_map returns None when it cannot tell; never assume safe.
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


# --- the single-flight slot ---------------------------------------------------


def test_a_refused_dispatch_is_not_reported_as_downloading(hub):
    # The hub service can decline a claim without raising (accepted=False, no worker), so
    # the caller must hear "busy", not a download that was never started.
    hub["dispatch_result"] = {
        "job_key": "unsloth/x-gguf::ud-q5_k_xl",
        "state": "running",  # the blocking job's state, not ours
        "accepted": False,
        "generation": 3,
    }
    refusal = _run("unsloth/x-GGUF:UD-Q5_K_XL")
    assert refusal.status == 503 and refusal.code == "model_download_busy"
    # No watcher installed for a job that is not running.
    assert hub["watched"] == []
    # The slot is free, so an unrelated repo is still admitted.
    assert auto_dl._active is None
    hub["dispatch_result"] = {"job_key": "k", "state": "running", "accepted": True}
    assert _run("unsloth/other-GGUF").code == "model_downloading"


def test_an_adoptable_dispatch_still_tracks_the_existing_job(hub):
    # accepted=True with claimed=False means this repo+quant is already downloading
    # (started from the Hub UI); attach to it rather than refuse.
    hub["dispatch_result"] = {"job_key": "k", "state": "running", "accepted": True}
    assert _run("unsloth/x-GGUF:UD-Q5_K_XL").code == "model_downloading"
    assert len(hub["watched"]) == 1


def test_a_failed_status_probe_does_not_end_the_watch(hub, monkeypatch):
    # A probe that raised says nothing about the worker. Reading it as "idle" freed the
    # slot under a live download, admitting a second multi-GB fetch alongside the first.
    from hub.services.models import downloads

    async def _boom(repo_id, gguf_variant = ""):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(downloads, "get_download_status_response", _boom)
    state, error = asyncio.run(auto_dl._job_state("unsloth/x-GGUF", "UD-Q4_K_XL"))
    assert (state, error) == ("unknown", None)


def test_an_unknown_state_still_reports_the_download_to_a_retry(hub, monkeypatch):
    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_downloading"

    async def _unknown(repo, variant):
        return "unknown", None

    monkeypatch.setattr(auto_dl, "_job_state", _unknown)
    # Still downloading as far as anyone knows, so the slot stays taken.
    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_downloading"
    assert _run("unsloth/other-GGUF").code == "model_download_busy"


def test_a_stale_watcher_cannot_release_a_newer_download(hub, monkeypatch):
    # Variant A is downloading; its watcher holds the slot.
    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_downloading"
    watcher_a = hub["watched"][-1]

    # A fails, so an adopting request surfaces the error and frees the slot.
    real_job_state = auto_dl._job_state
    errored = {"on": True}

    async def _maybe_errored(repo, variant):
        if errored["on"]:
            return "error", "boom"
        return await real_job_state(repo, variant)

    monkeypatch.setattr(auto_dl, "_job_state", _maybe_errored)
    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_download_failed"
    errored["on"] = False

    # The retry starts variant B of the same repo, which now owns the slot.
    assert _run("unsloth/x-GGUF:UD-Q5_K_XL").code == "model_downloading"
    watcher_b = hub["watched"][-1]
    assert auto_dl._active is watcher_b

    # Only now does A's watcher clean up. Keyed on repo_id alone, that cleared B.
    errored["on"] = True
    monkeypatch.setattr(auto_dl, "_WATCH_POLL_S", 0.0)
    asyncio.run(hub["real_watch"](watcher_a, None))
    assert auto_dl._active is watcher_b
    assert _run("unsloth/other-GGUF").code == "model_download_busy"


def test_a_cancelled_admission_does_not_wedge_the_slot(hub):
    # CancelledError is a BaseException, so an `except Exception` cleanup would leave the
    # provisional slot installed and refuse every later request.
    def _cancel():
        raise asyncio.CancelledError()

    hub["on_probe"] = _cancel

    async def _cancelled_request():
        with pytest.raises(asyncio.CancelledError):
            await auto_dl.maybe_auto_download("unsloth/x-GGUF")

    asyncio.run(_cancelled_request())
    assert auto_dl._active is None
    hub["on_probe"] = None
    assert _run("unsloth/other-GGUF").code == "model_downloading"


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


class _CatalogInfo:
    """Minimal stand-in for a local model the /v1/models scan listed."""

    def __init__(self, model_id, path):
        self.model_id = model_id
        self.id = model_id
        self.path = path


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
        lambda name, **_kw: ("/p", None, name) if downloaded else None,
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
    # The reported bug: asking for UD-Q6_K_XL while UD-Q4_K_XL is resident returned 200.
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


def test_a_failed_switch_is_reported_not_answered_by_the_resident_model(monkeypatch):
    # On disk and switching allowed means the swap failed, so answering from the resident
    # model would be the wrong weights under the right name.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    with pytest.raises(HTTPException) as excinfo:
        _reject("unsloth/B-GGUF", loaded, monkeypatch, downloaded = True, auto_switch = True)
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail["error"]["code"] == "model_switch_failed"
    assert excinfo.value.headers["Retry-After"] == "5"


@pytest.mark.parametrize(
    "foreign",
    [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "meta-llama/llama-3-70b-instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ],
)
def test_a_provider_prefixed_label_still_reaches_the_resident_model(foreign, monkeypatch):
    # LiteLLM and OpenRouter address every provider as "vendor/model", so treating a
    # namespace as a concrete reference would 404 all of them.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    assert _reject(foreign, loaded, monkeypatch) is None


def test_an_explicit_quant_is_still_refused(monkeypatch):
    # A quant is the signal: no LiteLLM or OpenRouter id carries one.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    with pytest.raises(HTTPException) as excinfo:
        _reject("unsloth/B-GGUF:UD-Q6_K_XL", loaded, monkeypatch)
    assert excinfo.value.status_code == 404


def test_a_repo_that_is_here_is_refused_without_a_quant(monkeypatch):
    # The other half of the evidence test: a repo this server has is a reference to it.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    with pytest.raises(HTTPException) as excinfo:
        _reject("unsloth/B-GGUF", loaded, monkeypatch, downloaded = True)
    assert excinfo.value.status_code == 404


def test_a_diagnosis_failure_does_not_serve_the_wrong_model(monkeypatch):
    # The mismatch is established before the diagnosis runs, so only the wording is
    # uncertain; falling through here would answer as another model.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )

    def _boom(name, **_kw):
        raise OSError("cache scan unavailable")

    monkeypatch.setattr("core.inference.local_model_resolver.resolve_local_gguf", _boom)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("unsloth/B-GGUF:UD-Q6_K_XL", _Req()))
    assert excinfo.value.status_code == 404


def test_nothing_loaded_leaves_the_existing_error_alone(monkeypatch):
    # The handler's own no-model-loaded error is already correct; don't preempt it.
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

    def _boom(_name, **_kw):
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
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf", lambda name, **_kw: None
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            inference_route._reject_unservable_model(
                "unsloth/B-GGUF:UD-Q6_K_XL", _Req(path = "/v1/messages")
            )
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


def test_the_servers_own_hf_token_is_never_borrowed(monkeypatch):
    # The repo is named by whoever holds an API key, so fetching under the owner's Hub
    # identity would let that key pull the owner's private repos.
    import routes.settings as settings_route

    monkeypatch.setattr(settings_route, "_ambient_hf_token", lambda: "hf_owner_secret")
    assert inference_route._auto_download_hf_token(_Req()) is None
    caller = _Req(headers = {"X-Unsloth-HF-Token": "hf_caller_own"})
    assert inference_route._auto_download_hf_token(caller) == "hf_caller_own"


def test_a_quant_cannot_be_satisfied_by_a_non_gguf_backend(monkeypatch):
    # llama.cpp matches :QUANT against hf_variant; Transformers has no quant identity.
    idle = type("B", (), {"is_loaded": False, "model_identifier": None, "hf_variant": None})()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: idle)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": "org/model"})(),
    )
    assert inference_route._loaded_satisfies("org/model") is True
    assert inference_route._loaded_satisfies("org/model:Q4_K_M") is False
    # An Ollama-style tag is not a claim about the weights, so it still matches.
    assert inference_route._loaded_satisfies("org/model:latest") is True


def test_the_worker_is_never_given_the_servers_own_token(hub):
    # A falsy token would make the worker fall back to the server owner's HF_TOKEN.
    assert _run("unsloth/x-GGUF").code == "model_downloading"
    assert hub["started"][0][2] is None
    assert hub["allow_ambient"] is False


def test_the_metadata_probe_is_explicitly_anonymous(hub):
    # token=None means "use the cached login" to huggingface_hub; only False is anonymous.
    _run("unsloth/x-GGUF")
    assert hub["token"] is False
    auto_dl.reset_for_tests()
    _run("unsloth/y-GGUF", hf_token = "hf_caller_own")
    assert hub["token"] == "hf_caller_own"


def test_an_ollama_tag_still_matches_the_resident_gguf(monkeypatch):
    # looks_like_quant() calls these foreign, so they must not be checked against hf_variant.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    assert inference_route._loaded_satisfies("unsloth/A-GGUF:latest") is True
    assert inference_route._loaded_satisfies("unsloth/A-GGUF:8b") is True
    assert inference_route._loaded_satisfies("unsloth/A-GGUF:UD-Q4_K_XL") is True
    assert inference_route._loaded_satisfies("unsloth/A-GGUF:Q8_0") is False


def test_a_probing_adoption_never_releases_the_slot(hub, monkeypatch):
    # The whole-repo job key can hold a stale error that would free the probe's slot.
    hub["on_probe"] = lambda: _run_nested()
    seen = {}

    def _run_nested():
        async def _stale(repo, variant):
            seen["queried"] = True
            return "error", "an older failure"

        monkeypatch.setattr(auto_dl, "_job_state", _stale)
        seen["refusal"] = _run("unsloth/x-GGUF")

    assert _run("unsloth/x-GGUF").code == "model_downloading"
    assert seen["refusal"].code == "model_downloading"
    assert "queried" not in seen  # the stale job key was never consulted


def test_a_bpw_qualified_quant_is_a_quant_request():
    # _extract_quant_label emits these for repos shipping several files at one base quant.
    assert auto_dl.looks_like_quant("IQ4_XS-3.53bpw")
    assert auto_dl.looks_like_quant("UD-Q4_K_XL-4.19BPW")
    assert not auto_dl.looks_like_quant("3.53bpw")


def test_the_default_pick_survives_lowercase_quant_labels():
    # Preference tokens match case-sensitively, so a lower-case repo would take F16.
    lowered = {"f16": 20, "ud-q4_k_xl": 4, "q8_0": 9}
    assert auto_dl._match_variant(None, lowered) == "ud-q4_k_xl"
    assert auto_dl._match_variant(None, {"F16": 20, "UD-Q4_K_XL": 4}) == "UD-Q4_K_XL"


def test_a_slashless_local_model_is_still_a_concrete_reference(monkeypatch):
    # /v1/models advertises these without a namespace, so a namespace decides nothing.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: False
    )

    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda name, **_kw: ("/p", None, name) if name.startswith("standalone-Q4_K_M") else None,
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("standalone-Q4_K_M", _Req()))
    assert excinfo.value.status_code == 404

    # A slashless name that is not here stays a foreign label.
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf", lambda name, **_kw: None
    )
    assert asyncio.run(inference_route._reject_unservable_model("gpt-4", _Req())) is None
    assert asyncio.run(inference_route._reject_unservable_model("default", _Req())) is None


def test_a_cancelled_download_is_not_reported_as_failed(hub, monkeypatch):
    # fail_open rendered a deliberate cancel as "Model download failed".
    from core.inference import api_monitor as monitor_module

    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_downloading"
    active = hub["watched"][-1]

    async def _cancelled(repo, variant):
        return "cancelled", None

    monkeypatch.setattr(auto_dl, "_job_state", _cancelled)
    monkeypatch.setattr(auto_dl, "_WATCH_POLL_S", 0)
    asyncio.run(hub["real_watch"](active, None))
    [row] = [e for e in monitor_module.api_monitor.snapshot() if e["id"] == active.monitor_id]
    assert row["status"] == "cancelled"
    assert row.get("error") is None


def test_disk_admission_counts_only_what_is_left_to_fetch(hub, monkeypatch):
    # Charging again for bytes already on disk 507s a download that fits.
    seen = {}

    def _enough(need):
        seen["need"] = need
        return True, 10 * 1024**4

    gb = 1024**3
    hub["info"] = _Info(
        [
            _Sibling("model-UD-Q4_K_XL.gguf", 4 * gb, blob_id = "sha-main"),
            _Sibling("mmproj-F16.gguf", 1 * gb, blob_id = "sha-mmproj"),
            _Sibling("mtp-model.gguf", 1 * gb, blob_id = "sha-mtp"),
        ]
    )
    monkeypatch.setattr(auto_dl, "_enough_disk", _enough)
    monkeypatch.setattr(
        "hub.utils.download_registry.existing_blob_bytes",
        lambda repo_type, repo_id, hashes: 3 * gb,
    )
    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_downloading"
    # 4 GB quant + 2 GB companions, 3 GB of which is already cached.
    assert seen["need"] == 3 * gb


def test_a_resolver_alias_for_the_resident_model_is_not_refused(monkeypatch):
    # A manual load stores the on-disk path /v1/models aliases as publisher/model.
    loaded = _Loaded("/models/publisher/model/weights.gguf", None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda name, **_kw: ("/models/publisher/model/weights.gguf", None, "publisher/model"),
    )
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: False
    )
    assert asyncio.run(inference_route._reject_unservable_model("publisher/model", _Req())) is None


def test_the_request_path_never_triggers_a_model_index_rescan(monkeypatch):
    # The scan walks several model dirs and HF caches under a lock every other
    # caller queues behind, and takes seconds on a large install. With
    # auto-switch off this hook is the only thing between a request and that
    # scan, so it must answer from the last built index instead.
    from core.inference import local_model_resolver as resolver

    scans = []
    warmed = []
    monkeypatch.setattr(resolver, "_build_index", lambda: scans.append(1) or {})
    monkeypatch.setattr(resolver, "_scan", (1.0, {}))
    # Stub the warm: it is allowed to scan, just not on the thread serving the request.
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: False
    )
    for model in ("gpt-4", "anthropic/claude-3.5-sonnet", "unsloth/B-GGUF:UD-Q6_K_XL"):
        try:
            asyncio.run(inference_route._reject_unservable_model(model, _Req()))
        except HTTPException:
            pass
    assert scans == []
    assert warmed == [1, 1, 1]


def test_a_cold_index_still_refuses_an_explicit_quant_and_warms_in_the_background(monkeypatch):
    # Before the first scan there is no evidence of what is on disk, but an explicit
    # quant is evidence enough on its own: answering it from the resident model's own
    # quant would be the wrong weights under the right name. A bare name proves
    # nothing, so it still falls through. Either way the warm happens off this request.
    from core.inference import local_model_resolver as resolver

    warmed = []
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("unsloth/A-GGUF:Q8_0", _Req()))
    assert excinfo.value.status_code == 404
    assert asyncio.run(inference_route._reject_unservable_model("gpt-4", _Req())) is None
    assert warmed == [1, 1]


def test_warming_the_index_never_waits_on_the_scan_lock(monkeypatch):
    # _lock is held for the whole scan. If the request path contended for it, the
    # first cold request would park every later one on the event loop for the length
    # of the very scan it is meant to stay off.
    import threading
    import time as _time

    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    released = threading.Event()
    monkeypatch.setattr(resolver, "_build_index", lambda: (released.wait(5), {})[1])
    _real_warm_index_soon()
    try:
        started = _time.perf_counter()
        _real_warm_index_soon()
        resolver.resolve_local_gguf("unsloth/A-GGUF", allow_scan = False)
        elapsed = _time.perf_counter() - started
    finally:
        released.set()
        # Join before the monkeypatches unwind, or the scan finishes against the real
        # module state and publishes its stub result over it.
        for _ in range(500):
            if not resolver._warming:
                break
            _time.sleep(0.01)
    assert elapsed < 0.5, f"request path blocked on the warm scan for {elapsed:.2f}s"


def test_a_stale_index_is_refreshed_so_a_hub_download_becomes_visible(monkeypatch):
    # Only the API auto-download watcher calls invalidate_index. A model fetched in
    # the Hub UI has no such hook, so if the warm only ever ran once this path would
    # never see it and would keep answering from the resident model.
    from core.inference import local_model_resolver as resolver

    scans = []
    monkeypatch.setattr(resolver, "_build_index", lambda: scans.append(1) or {})
    monkeypatch.setattr(resolver, "_scan", (time.monotonic() - resolver._CACHE_TTL_S - 1, {}))
    monkeypatch.setattr(resolver, "_last_scan_s", 0.0)
    _real_warm_index_soon()
    for _ in range(500):
        if scans and not resolver._warming:
            break
        time.sleep(0.01)
    assert scans == [1]


def test_an_id_v1_models_advertised_is_refused_before_the_resolver_warms(monkeypatch):
    # Codex P1: /v1/models scans on its own schedule, so it can advertise an
    # unloaded local GGUF while the resolver index is still cold. A bare id carries
    # no quant to refuse on, so without the catalog as evidence that request would
    # be answered by the unrelated resident model.
    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(
        inference_route,
        "_CATALOG_CACHE",
        {"at": 1.0, "models": [_CatalogInfo("org/Other", "/srv/models/org--Other")]},
    )
    monkeypatch.setattr(inference_route, "_ADVERTISED_CACHE", {"at": None, "paths": {}})
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("org/Other", _Req()))
    assert excinfo.value.status_code == 404
    # An id the catalog never listed still proves nothing, so it falls through.
    assert asyncio.run(inference_route._reject_unservable_model("org/Unlisted", _Req())) is None


def test_an_advertised_alias_for_the_resident_weights_is_still_served(monkeypatch):
    # The flip side: the catalog can list the resident weights under an alias the
    # loaded entry does not answer to. That is not evidence of a different model.
    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(
        inference_route,
        "_CATALOG_CACHE",
        {"at": 2.0, "models": [_CatalogInfo("publisher/Qwen3", "/srv/models")]},
    )
    monkeypatch.setattr(inference_route, "_ADVERTISED_CACHE", {"at": None, "paths": {}})
    loaded = _Loaded("/srv/models/Qwen3-Q4.gguf", "Q4_K_M")
    loaded.gguf_path = "/srv/models/Qwen3-Q4.gguf"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    assert asyncio.run(inference_route._reject_unservable_model("publisher/Qwen3", _Req())) is None

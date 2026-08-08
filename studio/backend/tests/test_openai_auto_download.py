# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in auto-download of a GGUF a /v1 request names but this server lacks.

No network: huggingface_hub, the consent probe and the Hub download service are
all mocked. The invariant: with the setting off nothing here runs at all, and
with it on a name not shaped like a repo still falls through to the resident model.
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
    # The hook warms the index in the background; drop it so a scan never leaks between tests.
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

    huggingface_hub 1.x made ``response`` a required keyword-only argument and the
    project floor is 0.34, so construct positionally and fall back. The positional
    form carries no response, which hf_error_status reads, so attach one either way.
    """
    try:
        exc = error_type(message)
    except TypeError:
        import httpx
        exc = error_type(
            message,
            response = httpx.Response(
                status_code,
                request = httpx.Request("GET", "https://huggingface.co/api/models/org/repo"),
            ),
        )
    if getattr(getattr(exc, "response", None), "status_code", None) != status_code:
        from types import SimpleNamespace
        try:
            exc.response = SimpleNamespace(status_code = status_code)
        except AttributeError:
            pass
    return exc


def test_the_hub_error_helper_carries_a_status_on_both_majors():
    # CI runs huggingface_hub 1.x and this box 0.x, and each takes only one of the
    # constructor shapes. A helper that silently dropped the response would make an
    # error-mapping test pass here and fail there.
    from hub.utils.hf_errors import hf_error_status

    class _Legacy(Exception):
        """0.x: response is optional and unset when built positionally."""

    class _Modern(Exception):
        """1.x: response is required and keyword-only."""

        def __init__(self, message, *, response):
            super().__init__(message)
            self.response = response

    for error_type in (_Legacy, _Modern):
        assert hf_error_status(_hub_error(error_type, 401, "unauthorized")) == 401


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
        # An unrecognized GGUF below a subdirectory keys on its path, and that key is
        # what the catalog advertises, so pinning it has to parse.
        ("org/repo:build/llama-13b", ("org/repo", "build/llama-13b")),
        # Still a path, not a variant: no Hub repo precedes the colon.
        ("/home/me/models/x:build/llama-13b", ("/home/me/models/x:build/llama-13b", None)),
        ("D:/models/repo:build/llama-13b", ("D:/models/repo:build/llama-13b", None)),
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
    # "vendor/model" is how LiteLLM names providers, so an unknown id stays a foreign label.
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
    # The Hub 404s a private repo, so a global verdict would hide it from the token holder.
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
    # Auto-download never uses the server's token, so a Studio setting would loop the caller.
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
    # Metadata for a gated repo is not file access, so report the licence gate, not custom code.
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
    # The hub service can decline without raising (accepted=False), so the caller hears "busy".
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
    # accepted=True with claimed=False means it is already downloading (Hub UI); attach to it.
    hub["dispatch_result"] = {"job_key": "k", "state": "running", "accepted": True}
    assert _run("unsloth/x-GGUF:UD-Q5_K_XL").code == "model_downloading"
    assert len(hub["watched"]) == 1


def test_a_failed_status_probe_does_not_end_the_watch(hub, monkeypatch):
    # A probe that raised says nothing: reading it as "idle" freed the slot mid-download.
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


def test_a_hanging_code_probe_does_not_pin_the_slot(hub, monkeypatch):
    # hf_hub_download and auth_check take no timeout and run while the provisional slot
    # is held, so an unresponsive Hub stalled the request and reported every other model
    # busy. Unchecked is not cleared, so the bounded probe refuses instead of admitting.
    import threading

    entered, release = threading.Event(), threading.Event()

    def _hang(repo, token = None):
        entered.set()
        release.wait(30)
        return False

    monkeypatch.setattr("utils.security.consent._config_has_auto_map", _hang)
    monkeypatch.setattr(auto_dl, "_CODE_PROBE_TIMEOUT_S", 0.2)

    async def _timed():
        # Time the await, not asyncio.run: the probe thread cannot be cancelled, so
        # loop shutdown waits for it here in a way a long-lived server loop never does.
        started = time.monotonic()
        refusal = await auto_dl.maybe_auto_download("unsloth/x-GGUF:UD-Q4_K_XL")
        waited = time.monotonic() - started
        release.set()
        return refusal, waited

    refusal, waited = asyncio.run(_timed())
    assert entered.is_set()
    assert refusal.status == 403 and refusal.code == "remote_code_consent_required"
    assert waited < 5
    # The slot was handed back, so the next request is admitted rather than told busy.
    assert auto_dl._active is None


def test_a_hanging_auth_check_falls_through_to_the_download(hub, monkeypatch):
    # Inconclusive, not denied: the download's own auth is the real gate, so a slow
    # gated-repo check must not turn into a refusal.
    import threading

    hub["info"].gated = True
    release = threading.Event()

    def _hang(repo, token = None):
        release.wait(30)
        return True

    monkeypatch.setattr(auto_dl, "_auth_denied", _hang)
    monkeypatch.setattr(auto_dl, "_MODEL_INFO_TIMEOUT_S", 0.2)

    async def _timed():
        refusal = await auto_dl.maybe_auto_download("unsloth/x-GGUF:UD-Q4_K_XL")
        release.set()
        return refusal

    assert asyncio.run(_timed()).code == "model_downloading"


def test_a_companion_only_repo_is_not_held_at_busy(hub):
    # mmproj and MTP files are companions, not quants, so such a repo is non-servable
    # and falls through to the resident model. The busy probe accepted any .gguf, which
    # stranded that ordinary traffic behind an unrelated multi-hour download.
    assert _run("unsloth/x-GGUF:UD-Q4_K_XL").code == "model_downloading"
    gb = 1024**3
    hub["info"] = _Info([_Sibling("mmproj-F16.gguf", gb), _Sibling("mtp-model.gguf", gb)])
    assert _run("unsloth/companions-GGUF") is None
    # A repo that does hold a real quant is still a second download.
    hub["info"] = _gguf_repo_info()
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
    # CancelledError is a BaseException, so an `except Exception` cleanup would wedge the slot.
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


def _hook(
    model,
    request,
    enabled,
    current_subject = None,
):
    import utils.openai_auto_switch_settings as s

    original = s.get_openai_auto_download_enabled
    s.get_openai_auto_download_enabled = lambda: enabled
    try:
        return asyncio.run(
            inference_route._maybe_auto_download_model(
                model, request, current_subject = current_subject
            )
        )
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
    async def _boom(
        model,
        hf_token = None,
        **kwargs,
    ):
        raise RuntimeError("boom")

    monkeypatch.setattr(auto_dl, "maybe_auto_download", _boom)
    assert _hook("unsloth/x-GGUF", _Req(), enabled = True) is None


def _download_rows():
    from core.inference.api_monitor import api_monitor
    return [e for e in api_monitor.snapshot() if e["event"] == "download"]


def test_a_ui_session_download_is_not_marked_as_api_traffic(hub):
    """The monitor overlay auto-opens on via_api_key, which exists to separate
    "someone is serving other clients" from "someone is using Unsloth". Studio's
    own chat hits these same /v1 endpoints with a session JWT, so hardcoding the
    flag on the download row popped the panel open mid-chat."""
    from fastapi import HTTPException
    from core.inference.api_monitor import api_monitor

    api_monitor.clear()
    with pytest.raises(HTTPException):
        # No Authorization header: the UI's session-JWT path.
        _hook("unsloth/x-GGUF", _Req(), enabled = True, current_subject = "unsloth")
    rows = _download_rows()
    assert rows and all(row["via_api_key"] is False for row in rows)


def test_an_api_key_download_keeps_the_attribution_and_names_its_caller(hub):
    """The row is shared, so it needs the subject as well: without one the
    attribution is reported to every logged-in browser instead of the caller."""
    from fastapi import HTTPException
    from auth.authentication import API_KEY_PREFIX
    from core.inference.api_monitor import api_monitor

    api_monitor.clear()
    with pytest.raises(HTTPException):
        _hook(
            "unsloth/x-GGUF",
            _Req(headers = {"authorization": f"Bearer {API_KEY_PREFIX}abc123"}),
            enabled = True,
            current_subject = "unsloth",
        )
    rows = _download_rows()
    assert rows and all(row["via_api_key"] is True for row in rows)
    # Still shared: another subject sees the row, just not the attribution.
    others = [e for e in api_monitor.snapshot(subject = "someone-else") if e["event"] == "download"]
    assert len(others) == len(rows)
    assert all(row["via_api_key"] is False for row in others)


def test_an_api_key_caller_waiting_on_someone_elses_download_gets_a_row(hub):
    """A download started by Studio's own chat is attributed to the session, so an
    API-key client that asks for the same repo while it runs is refused before the
    handler's own api_monitor.start. Without a row of its own that call is invisible:
    the only row is the session's via_api_key=False download, so the overlay stays
    shut and the monitor presents API traffic as Studio's own."""
    from fastapi import HTTPException
    from auth.authentication import API_KEY_PREFIX
    from core.inference.api_monitor import api_monitor

    api_monitor.clear()
    with pytest.raises(HTTPException):
        # Studio's chat (session JWT) starts the download and takes the slot.
        _hook("unsloth/x-GGUF", _Req(), enabled = True, current_subject = "unsloth")
    seeded = {row["id"] for row in api_monitor.snapshot(subject = "unsloth")}

    with pytest.raises(HTTPException) as excinfo:
        # The adopted-download branch: same repo, an sk-unsloth key this time.
        _hook(
            "unsloth/x-GGUF",
            _Req(headers = {"authorization": f"Bearer {API_KEY_PREFIX}abc123"}),
            enabled = True,
            current_subject = "unsloth",
        )
    assert excinfo.value.status_code == 503

    fresh = [e for e in api_monitor.snapshot(subject = "unsloth") if e["id"] not in seeded]
    # New (so the overlay counts it as unseen traffic) and attributed to this caller.
    assert [e for e in fresh if e["via_api_key"]], "the refused API-key call left no row"
    row = next(e for e in fresh if e["via_api_key"])
    assert row["endpoint"] == "/v1/chat/completions"
    assert row["status"] == "error"
    # Shared rows aside, another subject must not inherit the attribution.
    others = [e for e in api_monitor.snapshot(subject = "someone-else") if e["id"] == row["id"]]
    assert others == []


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
    # On disk and switching allowed means the swap failed; the resident model is wrong weights.
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
    # A namespace is how LiteLLM addresses providers, so reading it as a reference 404s them.
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
    # The mismatch is already established, so falling through would answer as another model.
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
    assert result == (True, 120, True, True, False)
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
    # The repo is named by an API key holder, so the owner's Hub identity must not be used.
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
    # The scan takes seconds under a lock, so this hook must answer from the last built index.
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


def test_a_cold_index_is_scanned_rather_than_read_as_nothing_here(monkeypatch):
    # With no cached evidence yet, reading that as "not downloaded" answers a named
    # local model with the resident one. Pay the scan once, off the loop.
    from core.inference import local_model_resolver as resolver

    entry = resolver._LocalGgufEntry("org/other", "/srv/models/org--other", ("Q4_K_M",))
    scans = []

    def _build():
        scans.append(1)
        return {"org/other": entry}

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(resolver, "_build_index", _build)
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

    # The bug: a bare name that IS on disk used to fall through to the resident model.
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("org/other", _Req()))
    assert excinfo.value.status_code == 404
    assert scans == [1], "the cold index was not scanned"

    # Built now, so the request path reads the cache and never scans again.
    assert asyncio.run(inference_route._reject_unservable_model("gpt-4", _Req())) is None
    assert scans == [1]


def test_a_cold_scan_that_never_finishes_says_so_instead_of_guessing(monkeypatch):
    # The scan is bounded, but an unfinished one knows nothing about the name, and
    # falling through would put the resident model behind it: answer "not yet".
    import threading

    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(inference_route, "_COLD_INDEX_WAIT_S", 0.05)
    released = threading.Event()
    monkeypatch.setattr(resolver, "_build_index", lambda: (released.wait(5), {})[1])
    warmed = []
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    try:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(inference_route._reject_unservable_model("gpt-4", _Req()))
        assert excinfo.value.status_code == 503
        assert excinfo.value.headers.get("Retry-After")
        assert warmed == [1], "the scan was not left to finish in the background"
    finally:
        released.set()


def test_a_refusal_is_never_swallowed_by_the_cannot_verify_handler(monkeypatch):
    # The checks run inside a broad `except Exception` that turns a failure to decide
    # into a fallthrough. An HTTPException there is a decision, but was logged as a
    # failure and answered by the resident model.
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )

    def _boom(*_a, **_k):
        raise HTTPException(status_code = 418, detail = "decided")

    monkeypatch.setattr(inference_route, "_resolves_to_resident", _boom)
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda *_a, **_k: ("/srv/models/x", "Q4_K_M", "x"),
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("org/x", _Req()))
    assert excinfo.value.status_code == 418


def test_warming_the_index_never_waits_on_the_scan_lock(monkeypatch):
    # _lock is held for the whole scan, so contending for it would park every later request.
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
        # Join before the monkeypatches unwind, or the scan publishes its stub result over them.
        for _ in range(500):
            if not resolver._warming:
                break
            _time.sleep(0.01)
    assert elapsed < 0.5, f"request path blocked on the warm scan for {elapsed:.2f}s"


def test_a_stale_index_is_refreshed_so_a_hub_download_becomes_visible(monkeypatch):
    # Only the auto-download watcher calls invalidate_index, so a Hub UI download is seen
    # only if the warm can run again.
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
    # /v1/models can advertise an unloaded local GGUF while the resolver index is cold. A bare
    # id has no quant to refuse on, so without that evidence the resident model would answer.
    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    # Stub the walk: a real multi-root scan inside the cold-wait budget makes this
    # test time out into a 503 under load instead of asserting what it is here for.
    monkeypatch.setattr(resolver, "_build_index", lambda: {})
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
    # The flip side: the catalog can list the resident weights under an alias, which is not
    # evidence of a different model.
    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    # Stub the walk: a real multi-root scan inside the cold-wait budget makes this
    # test time out into a 503 under load instead of asserting what it is here for.
    monkeypatch.setattr(resolver, "_build_index", lambda: {})
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


def test_a_rejected_token_says_so_instead_of_asking_for_a_retry(hub):
    # Hugging Face 401s an expired X-Unsloth-HF-Token. Only 403/404 were handled, so it
    # fell through to a 503 telling the caller to retry something that cannot work.
    from huggingface_hub.utils import HfHubHTTPError

    hub["raise"] = _hub_error(HfHubHTTPError, 401, "unauthorized")
    refusal = _run("unsloth/x-GGUF:UD-Q5_K_XL", hf_token = "hf_expired")
    assert refusal.status == 401 and refusal.code == "model_access_denied"
    assert "token" in refusal.message.lower()
    assert hub["started"] == []


def test_an_image_request_does_not_download_a_text_only_model(hub):
    # The capability guard only ever sees an already-local target, so without this an
    # image request spends gigabytes on weights that then 400 on every retry.
    gb = 1024**3
    hub["info"] = _Info([_Sibling("model-UD-Q5_K_XL.gguf", 5 * gb)])
    refusal = asyncio.run(
        auto_dl.maybe_auto_download("unsloth/text-GGUF:UD-Q5_K_XL", require_vision = True)
    )
    assert refusal.status == 400 and refusal.code == "invalid_value"
    assert "mmproj" in refusal.message
    assert hub["started"] == []
    # The stock fixture repo ships mmproj-F16.gguf, so that one is allowed to start.
    hub["info"] = _gguf_repo_info()
    assert (
        asyncio.run(
            auto_dl.maybe_auto_download("unsloth/x-GGUF:UD-Q5_K_XL", require_vision = True)
        ).code
        == "model_downloading"
    )
    assert len(hub["started"]) == 1


def test_two_models_differing_only_in_case_are_not_the_same_weights(monkeypatch):
    # Lowercasing paths made /srv/models/Foo and /srv/models/foo compare equal, so
    # on a case-sensitive filesystem a request for one was answered by the other.
    import os

    loaded = _Loaded("/srv/models/Foo/model.gguf")
    loaded.gguf_path = "/srv/models/Foo/model.gguf"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    assert inference_route._resolves_to_resident("/srv/models/Foo") is True
    same = os.path.normcase("A") == os.path.normcase("a")
    assert inference_route._resolves_to_resident("/srv/models/foo") is same


def test_a_quant_request_is_not_satisfied_by_transformers_weights(monkeypatch):
    # A Transformers model active from a directory that also holds GGUF exports resolves
    # to that directory, so the path match let admission answer an explicit quant with
    # the safetensors weights. Only llama.cpp has a quant identity.
    from core.inference import local_model_resolver as resolver

    entry = resolver._LocalGgufEntry("alias", "/srv/models/tuned", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"alias": entry}))
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: type("L", (), {"is_loaded": False})()
    )
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": "/srv/models/tuned"})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("alias:Q4_K_M", _Req()))
    assert excinfo.value.status_code == 404
    # A bare name claims nothing about the weights, so the active model still answers.
    assert asyncio.run(inference_route._reject_unservable_model("alias", _Req())) is None


def test_a_timed_out_download_keeps_the_slot_while_it_is_still_running(monkeypatch):
    # The watch window only bounds progress reporting. Releasing on the clock while
    # the worker is alive would admit a second multi-GB download beside it.
    monkeypatch.setattr(auto_dl, "_MAX_WATCH_S", 0.0)
    monkeypatch.setattr(auto_dl, "_WATCH_POLL_S", 0.001)
    monkeypatch.setattr(auto_dl, "_TIMED_OUT_POLL_S", 0.001)
    active = auto_dl._Active(repo_id = "org/big-GGUF", variant = "Q4_K_M")

    async def _drive():
        finished = asyncio.Event()

        async def _state(repo, variant):
            return ("complete" if finished.is_set() else "running"), None

        monkeypatch.setattr(auto_dl, "_job_state", _state)
        auto_dl._active = active
        watcher = asyncio.create_task(auto_dl._watch(active, None))
        # Long past the deadline, and still running: the slot must not come back.
        await asyncio.sleep(0.05)
        held = auto_dl._active is active
        finished.set()
        await watcher
        return held, auto_dl._active

    held, after = asyncio.run(_drive())
    assert held, "the slot was released while the worker was still running"
    assert after is None, "the slot was not released once the job finished"


def test_a_timed_out_download_stops_holding_the_slot_once_unprobeable(monkeypatch):
    # The other direction: a probe that can no longer confirm the worker is alive
    # must not wedge auto-download for the life of the process.
    monkeypatch.setattr(auto_dl, "_MAX_WATCH_S", 0.0)
    monkeypatch.setattr(auto_dl, "_WATCH_POLL_S", 0.001)
    monkeypatch.setattr(auto_dl, "_TIMED_OUT_POLL_S", 0.001)
    active = auto_dl._Active(repo_id = "org/big-GGUF", variant = "Q4_K_M")

    async def _unknown(repo, variant):
        return "unknown", None

    monkeypatch.setattr(auto_dl, "_job_state", _unknown)

    async def _drive():
        auto_dl._active = active
        await auto_dl._watch(active, None)
        return auto_dl._active

    assert asyncio.run(_drive()) is None


def test_a_sibling_quant_in_the_same_directory_is_not_the_resident_one(monkeypatch):
    # Quants of one repo share a directory, so the path match alone cannot tell them
    # apart, and an explicit :Q8_0 was answered by a resident Q4_K_M.
    from core.inference import local_model_resolver as resolver

    entry = resolver._LocalGgufEntry("org/model", "/hf/org--model/snap", ("Q4_K_M", "Q8_0"))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/model": entry}))
    loaded = _Loaded("org/model", "Q4_K_M")
    loaded.gguf_path = "/hf/org--model/snap/model-Q4_K_M.gguf"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException):
        asyncio.run(inference_route._reject_unservable_model("org/model:Q8_0", _Req()))
    # The quant that is actually resident still answers.
    assert asyncio.run(inference_route._reject_unservable_model("org/model:Q4_K_M", _Req())) is None


def test_a_remote_tag_that_names_no_quant_picks_the_preferred_one(hub):
    # ":latest" and ":8b" name no quant, so remote admission must default-select like a
    # bare repo id (as the local resolver does) instead of 404ing on a non-quant.
    assert _run("unsloth/x-GGUF").code == "model_downloading"
    bare_repo, bare_variant, _ = hub["started"][0]
    for tag in (":latest", ":8b"):
        auto_dl.reset_for_tests()
        hub["started"].clear()
        assert _run(f"unsloth/x-GGUF{tag}").code == "model_downloading"
        assert hub["started"][0][0] == bare_repo
        assert hub["started"][0][1] == bare_variant, f"{tag} did not default-select"

    # A real quant the repo does not have is still a 404, never a substitution.
    auto_dl.reset_for_tests()
    hub["started"].clear()
    refusal = _run("unsloth/x-GGUF:Q2_K")
    assert refusal.status == 404 and "no quant" in refusal.message
    assert hub["started"] == []


def test_a_generic_gguf_advertises_the_label_the_worker_resolves(hub):
    # With no recognized quant token the extractors part ways: one takes the last
    # hyphenated segment, the plan and worker key the whole stem. Dispatching ours
    # made the worker exit with "No GGUF shards matching variant".
    from hub.utils.gguf import extract_quant_label as canonical
    from hub.utils.gguf_plan import build_gguf_variant_plans

    sibling = _Sibling("llama-7b.gguf", 4 * 1024**3)
    hub["info"] = _Info([sibling])
    assert _run("unsloth/generic-GGUF").code == "model_downloading"
    dispatched = hub["started"][0][1]
    assert dispatched == canonical("llama-7b.gguf")
    # The key the worker will look up has to contain it, which is the whole point.
    assert dispatched.lower() in build_gguf_variant_plans([sibling])


def test_windows_style_paths_still_match_their_own_directory(monkeypatch):
    # normcase rewrites "/" to a backslash on Windows, so normalizing before it left the
    # descendant checks comparing against a path with none, and a resident model read
    # as a different one.
    import ntpath

    monkeypatch.setattr(inference_route.os.path, "normcase", ntpath.normcase)
    # A manual load records the file, so only the descendant check can match the
    # directory the resolver returns; an equality match would prove nothing here.
    loaded = _Loaded("C:\\models\\repo\\model.gguf")
    loaded.gguf_path = "C:\\models\\repo\\model.gguf"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    assert inference_route._resolves_to_resident("C:\\models\\repo") is True
    assert inference_route._resolves_to_resident("C:\\Models\\Repo") is True
    assert inference_route._resolves_to_resident("C:\\models\\other") is False


def test_a_bare_request_for_a_just_downloaded_model_is_refused(monkeypatch):
    # End of the same chain: the note has to reach admission, or a bare request between
    # the download landing and the scan is served by the resident model.
    from core.inference import local_model_resolver as resolver

    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {}))
    monkeypatch.setattr(resolver, "_just_downloaded", {"org/fresh"})
    loaded = _Loaded("unsloth/A-GGUF", "UD-Q4_K_XL")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(inference_route, "_unavailable_model_message", _fake_unavailable_message)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("org/fresh", _Req()))
    assert excinfo.value.status_code == 404
    assert asyncio.run(inference_route._reject_unservable_model("org/never", _Req())) is None


def test_a_non_quant_tag_does_not_tear_down_a_serving_quant(monkeypatch):
    # _already_serving split on ":" rather than on whether the suffix names a quant, so
    # org/model:latest against a serving Q8_0 counted as a mismatch and swapped in the
    # preferred Q4_K_M, for a request either one satisfies.
    from core.inference import local_model_resolver as resolver

    entry = resolver._LocalGgufEntry("org/model", "/hf/org--model/snap", ("Q4_K_M", "Q8_0"))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/model": entry}))
    loaded = _Loaded("org/model", "Q8_0")
    loaded.gguf_path = "/hf/org--model/snap/model-Q8_0.gguf"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    loads: list = []

    async def _record_load(request, *a, **k):
        loads.append(getattr(request, "gguf_variant", None))

    monkeypatch.setattr(inference_route, "_load_model_impl", _record_load)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: True
    )
    for tag in ("org/model:latest", "org/model:8b", "org/model"):
        asyncio.run(inference_route._maybe_auto_switch_model(tag, _Req(), "tester"))
    assert loads == [], "a tag naming no quant swapped the serving model out"


def test_the_trust_probe_never_falls_back_to_the_server_identity(hub, monkeypatch):
    # huggingface_hub treats None as "use the cached login", so only an explicit False
    # is anonymous. This probe passed None, so a caller-named repo was read with the
    # server's identity.
    seen: list = []

    def _probe(model_name, hf_token = None):
        seen.append(hf_token)
        return False

    monkeypatch.setattr("utils.security.consent._config_has_auto_map", _probe)
    _run("unsloth/x-GGUF:UD-Q5_K_XL")
    assert seen == [False], f"trust probe ran with {seen!r}, not an explicit anonymous token"

    seen.clear()
    auto_dl.reset_for_tests()
    _run("unsloth/x-GGUF:UD-Q5_K_XL", hf_token = "hf_caller")
    assert seen == ["hf_caller"], "the caller's own token must still be used"


def test_a_foreign_label_is_not_told_to_wait_for_someone_elses_download(hub):
    # The busy refusal fired before the probe, so any namespaced label a drop-in client
    # sends (LiteLLM/OpenRouter style) was told to wait out an unrelated download.
    assert _run("unsloth/first-GGUF").code == "model_downloading"

    hub["info"] = _Info([_Sibling("README.md", 1024)])  # real repo, no GGUF
    assert _run("anthropic/claude-3.5-sonnet") is None, "a foreign label was refused as busy"

    # A label that really is another downloadable model still gets the busy refusal.
    hub["info"] = _gguf_repo_info()
    refusal = _run("unsloth/second-GGUF")
    assert refusal.status == 503 and refusal.code == "model_download_busy"


def test_a_failed_download_keeps_the_slot_until_someone_is_told(monkeypatch):
    # The watcher freed the slot on the error, but Retry-After is 30s and the poll 2s,
    # so the client came back to an empty slot and restarted the same failing download.
    monkeypatch.setattr(auto_dl, "_MAX_WATCH_S", 60.0)
    monkeypatch.setattr(auto_dl, "_WATCH_POLL_S", 0.001)

    async def _errored(repo, variant):
        return "error", "disk exploded"

    monkeypatch.setattr(auto_dl, "_job_state", _errored)
    active = auto_dl._Active(repo_id = "org/x-GGUF", variant = "Q4_K_M")
    auto_dl._active = active
    asyncio.run(auto_dl._watch(active, None))
    assert auto_dl._active is active, "the slot was freed before anyone was told"
    assert active.error == "disk exploded"


def test_the_retry_after_a_failure_is_told_instead_of_restarting_it(hub, monkeypatch):
    # End of the same chain: the held failure has to reach the caller.
    active = auto_dl._Active(
        repo_id = "unsloth/x-GGUF",
        variant = "UD-Q5_K_XL",
        error = "disk exploded",
        failed_at = 1.0,
    )
    auto_dl._active = active

    async def _idle(repo, variant):
        return "idle", None

    monkeypatch.setattr(auto_dl, "_job_state", _idle)
    refusal = _run("unsloth/x-GGUF:UD-Q5_K_XL")
    assert refusal.status == 502 and "disk exploded" in refusal.message
    assert hub["started"] == [], "the retry restarted the failing download"
    # Told once, so the slot is free again for a fresh attempt.
    assert auto_dl._active is None


def test_a_completed_download_does_not_restage_the_scan_it_just_warmed(monkeypatch):
    # finalize_worker_exit invalidates and warms. A second invalidation here marks
    # that fresh scan stale and pushes a synchronous rescan onto the client's retry.
    import inspect

    src = inspect.getsource(auto_dl._watch)
    complete_branch = src[src.index('if state == "complete"') :]
    assert "invalidate_index" not in complete_branch


def test_an_exact_generic_variant_beats_the_default_pick(hub):
    # Canonicalizing generic labels made them real worker keys, but the matcher read
    # anything non-quant-shaped as a tag, so repo:llama-13b default-selected llama-7b.
    gb = 1024**3
    hub["info"] = _Info([_Sibling("llama-7b.gguf", 4 * gb), _Sibling("llama-13b.gguf", 8 * gb)])
    assert _run("unsloth/generic-GGUF:llama-13b").code == "model_downloading"
    assert hub["started"][0][1] == "llama-13b"

    # A quant-shaped suffix that matches nothing is still a miss, never a swap.
    auto_dl.reset_for_tests()
    hub["started"].clear()
    hub["info"] = _gguf_repo_info()
    assert _run("unsloth/x-GGUF:Q2_K").status == 404
    assert hub["started"] == []

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Nano: the gated guardrail's access is checked before any model download."""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

from core.inference import video_cosmos3 as vc3

NANO, EDGE = "nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Edge"
GUARDRAIL = "nvidia/Cosmos-1.0-Guardrail"
OPT_OUT = "UNSLOTH_COSMOS3_SAFETY_CHECKER"


@pytest.fixture
def guardrail(monkeypatch):
    """Package installed, no opt-out, nothing cached, online; returns the metadata requests made."""
    module = types.ModuleType("cosmos_guardrail")
    module.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    monkeypatch.setitem(sys.modules, "cosmos_guardrail", module)
    monkeypatch.delenv(OPT_OUT, raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False, raising = False)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(vc3, "_probe_ok", {})
    calls = []
    state = {"raise": None}

    def get_hf_file_metadata(
        url,
        token = None,
        timeout = None,
        **kwargs,
    ):
        calls.append({"url": url, "token": token})
        if state["raise"] is not None:
            raise state["raise"]
        return types.SimpleNamespace(size = 10)

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", get_hf_file_metadata)
    return calls, state


def _http_error(cls_name, status):
    from huggingface_hub import errors

    cls = getattr(errors, cls_name)
    exc = cls.__new__(cls)
    Exception.__init__(exc, f"{status} Client Error: Access to model {GUARDRAIL} is restricted.")
    exc.response = types.SimpleNamespace(status_code = status, headers = {})
    return exc


def _assert_refusal(message):
    for part in (
        f"'{NANO}'",
        f"https://huggingface.co/{GUARDRAIL}",
        "accept its licence",
        "token in Settings",
        f"{OPT_OUT}=0",
    ):
        assert part in message, part


@pytest.mark.parametrize(
    "error",
    [
        ("GatedRepoError", 401),
        ("GatedRepoError", 403),
        ("HfHubHTTPError", 403),
        ("HfHubHTTPError", 401),
    ],
)
def test_no_guardrail_access_is_refused_with_the_steps(guardrail, error):
    calls, state = guardrail
    state["raise"] = _http_error(*error)
    message = vc3.guardrail_access_preflight(NANO, "hf_settings")
    _assert_refusal(message)
    assert calls == [
        {
            "url": f"https://huggingface.co/{GUARDRAIL}/resolve/main/blocklist/custom/branding",
            "token": "hf_settings",
        }
    ]


@pytest.mark.parametrize("error", [OSError("connection reset"), TimeoutError("read timed out")])
def test_a_network_failure_does_not_block_the_load(guardrail, error):
    calls, state = guardrail
    state["raise"] = error
    assert vc3.guardrail_access_preflight(NANO, None) is None
    assert len(calls) == 1 and calls[0]["token"] is None


def test_granted_access_is_asked_once_per_token(guardrail):
    calls, _ = guardrail
    assert vc3.guardrail_access_preflight(NANO, "hf_a") is None
    assert vc3.guardrail_access_preflight(NANO, " hf_a ") is None
    assert vc3.guardrail_access_preflight(NANO, "hf_b") is None
    assert [c["token"] for c in calls] == ["hf_a", "hf_b"]


def test_a_cached_guardrail_offline_mode_and_local_only_loads_send_nothing(guardrail, monkeypatch):
    calls, state = guardrail
    state["raise"] = _http_error("GatedRepoError", 401)
    assert vc3.guardrail_access_preflight(NANO, None, local_files_only = True) is None
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True, raising = False)
    assert vc3.guardrail_access_preflight(NANO, None) is None
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False, raising = False)
    seen = []

    def cached(repo_id, filename, **kwargs):
        seen.append((repo_id, filename))
        return f"/cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    assert vc3.guardrail_access_preflight(NANO, None) is None
    assert calls == [] and (GUARDRAIL, "blocklist/custom/branding") in seen


def test_only_a_load_that_runs_the_guardrail_is_checked(guardrail, monkeypatch):
    calls, state = guardrail
    state["raise"] = _http_error("GatedRepoError", 401)
    # Edge ships enable_safety_checker false.
    assert vc3.guardrail_access_preflight(EDGE, None) is None
    monkeypatch.setenv(OPT_OUT, "0")
    assert vc3.guardrail_access_preflight(NANO, None) is None
    monkeypatch.delenv(OPT_OUT)
    # Package absent: the install refusal (guardrail_refusal) answers instead.
    monkeypatch.setitem(sys.modules, "cosmos_guardrail", None)
    assert vc3.guardrail_access_preflight(NANO, None) is None
    assert calls == []


def test_begin_load_refuses_before_the_worker_downloads(guardrail, monkeypatch):
    """The UI's one-step download-and-load: the refusal comes before _run_load (which downloads) starts."""
    from core.inference.video import VideoBackend

    calls, state = guardrail
    state["raise"] = _http_error("GatedRepoError", 403)
    fam = types.SimpleNamespace(name = "cosmos3-nano")
    monkeypatch.setattr(VideoBackend, "validate_load_request", lambda self, *a, **k: fam)
    started = []
    monkeypatch.setattr(VideoBackend, "_run_load", lambda self, **kwargs: started.append(kwargs))
    monkeypatch.setattr(
        VideoBackend, "_predownload_base", lambda self, *a, **k: started.append("download")
    )
    import core.inference.diffusion_nvfp4_route as route

    monkeypatch.setattr(route, "plan_nvfp4_route", lambda **kwargs: None)
    backend = VideoBackend()
    with pytest.raises(ValueError) as refused:
        backend.begin_load(NANO, model_kind = "pipeline", hf_token = "hf_settings", gpu_ordinal = 0)
    _assert_refusal(str(refused.value))
    assert started == [] and backend.loading_repo_ids() == ()
    assert [c["token"] for c in calls] == ["hf_settings"]
    # The same check guards the synchronous path before anything is fetched.
    with pytest.raises(ValueError, match = "no access"):
        backend.load_pipeline(NANO, model_kind = "pipeline", hf_token = "hf_settings")
    assert started == []


def test_the_load_route_checks_before_evicting(guardrail):
    import inspect

    from routes import video as video_routes

    source = inspect.getsource(video_routes.load_video_model_gated)
    # The awaited call, not the import of the same name.
    check = source.index("asyncio.to_thread(\n            refuse_without_guardrail_access,")
    assert check < source.index("refuse_replacing_media, VIDEO") and check < source.index(
        "begin_media_load(VIDEO"
    )


def test_the_download_plan_route_checks_before_staging(guardrail):
    """The picker stages /video/download-plan before /video/load, so the plan route must refuse first."""
    import inspect

    from routes import video as video_routes

    source = inspect.getsource(video_routes.video_download_plan)
    check = source.index("refuse_without_guardrail_access,\n")
    assert check < source.index("backend.download_plan,")


def test_an_unknown_fine_tune_follows_its_own_index_else_its_family(guardrail, monkeypatch):
    calls, state = guardrail
    state["raise"] = _http_error("GatedRepoError", 403)
    fetched = []

    def remote(repo, token):
        fetched.append((repo, token))
        return indexes.get(repo)

    indexes = {"me/nano-ft-off": {"enable_safety_checker": False}, "me/nano-ft-on": {}}
    monkeypatch.setattr(vc3, "_remote_model_index", remote)
    # No index anywhere: an Edge fine-tune is not refused, a Nano one is.
    assert vc3.guardrail_access_preflight("me/edge-ft", "hf_x", default_enabled = False) is None
    assert calls == []
    assert "no access" in vc3.guardrail_access_preflight("me/nano-ft", "hf_x", default_enabled = True)
    # The checkpoint's own index decides when it can be read.
    assert vc3.guardrail_access_preflight("me/nano-ft-off", "hf_x", default_enabled = True) is None
    assert "no access" in vc3.guardrail_access_preflight(
        "me/nano-ft-on", "hf_x", default_enabled = False
    )
    assert ("me/edge-ft", "hf_x") in fetched
    # The base repos never need the remote index.
    assert not any(repo in (NANO, EDGE) for repo, _ in fetched)


def test_refuse_without_access_passes_the_family_default(guardrail, monkeypatch):
    from core.inference import video as video_mod

    seen = []
    monkeypatch.setattr(
        vc3,
        "guardrail_access_preflight",
        lambda repo, token, **kw: seen.append(kw["default_enabled"]),
    )
    for name in ("cosmos3-nano", "cosmos3-edge"):
        video_mod.refuse_without_guardrail_access(
            types.SimpleNamespace(name = name), "pipeline", "me/ft", None, False
        )
    assert seen == [True, False]

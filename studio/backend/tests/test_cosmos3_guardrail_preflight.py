# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Nano: the gated guardrail's access is checked before any model download, and the guardrail package can be
installed into Studio's own environment."""

from __future__ import annotations

import importlib.machinery
import subprocess
import sys
import types

import pytest

from core.inference import cosmos_guardrail_install as cgi
from core.inference import video_cosmos3 as vc3

NANO, EDGE = "nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Edge"
GUARDRAIL = "nvidia/Cosmos-1.0-Guardrail"
CHECKER_ENV = "UNSLOTH_COSMOS3_SAFETY_CHECKER"


@pytest.fixture
def guardrail(monkeypatch):
    """Safety checker turned on, package installed, nothing cached, online; returns the metadata requests made."""
    module = types.ModuleType("cosmos_guardrail")
    module.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    monkeypatch.setitem(sys.modules, "cosmos_guardrail", module)
    monkeypatch.setenv(CHECKER_ENV, "1")
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
        f"{CHECKER_ENV}=1",
        "off by default",
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

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert vc3.guardrail_access_preflight(NANO, None) is None
    monkeypatch.delenv("TRANSFORMERS_OFFLINE")
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
    # The checker is off by default, and 0 keeps it off: no request, whatever the checkpoint ships.
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv(CHECKER_ENV)
        else:
            monkeypatch.setenv(CHECKER_ENV, value)
        assert vc3.guardrail_access_preflight(NANO, None) is None
        assert vc3.guardrail_access_preflight(EDGE, None) is None
    monkeypatch.setenv(CHECKER_ENV, "1")
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


def test_turned_on_the_checker_covers_edge_too(guardrail):
    # The opt-in turns the checker on for every Cosmos3 checkpoint, Edge (which ships it off) included.
    calls, state = guardrail
    state["raise"] = _http_error("GatedRepoError", 403)
    assert "no access" in vc3.guardrail_access_preflight(EDGE, "hf_x")
    assert [c["token"] for c in calls] == ["hf_x"]


# ── install ──────────────────────────────────────────────────────────────────


def test_the_manual_command_targets_this_studio_environment(monkeypatch):
    monkeypatch.setattr(cgi.shutil, "which", lambda name: "/usr/local/bin/uv")
    assert cgi.manual_install_command("/usr/local/bin/uv") == (
        f"uv pip install --python {sys.executable} cosmos-guardrail"
    )
    assert cgi.manual_install_command("/home/me/.local/bin/uv") == (
        f"/home/me/.local/bin/uv pip install --python {sys.executable} cosmos-guardrail"
    )
    assert cgi.manual_install_command(None) == f"{sys.executable} -m pip install cosmos-guardrail"


def test_the_windows_command_runs_in_powershell(monkeypatch):
    # Studio installs on Windows through PowerShell, which runs a quoted executable path only via &.
    monkeypatch.setattr(cgi.os, "name", "nt")
    line = cgi._quote(
        [r"C:\Program Files\Unsloth\python.exe", "-m", "pip", "install", "cosmos-guardrail"]
    )
    assert line == '& "C:\\Program Files\\Unsloth\\python.exe" -m pip install cosmos-guardrail'
    assert cgi._quote(["uv", "pip", "install"]) == "uv pip install"


def test_the_refusal_names_the_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "cosmos_guardrail", None)
    monkeypatch.setenv(CHECKER_ENV, "1")
    monkeypatch.setattr(cgi, "_uv", lambda: None)
    message = vc3.guardrail_refusal(NANO)
    assert f"`{sys.executable} -m pip install cosmos-guardrail`" in message
    assert "Install guardrail on the Video page" in message and f"{CHECKER_ENV}=1" in message
    # Off by default: nothing to install, nothing refused.
    monkeypatch.delenv(CHECKER_ENV)
    assert vc3.guardrail_refusal(NANO) is None


def test_the_install_holds_every_installed_package(monkeypatch, tmp_path):
    import core.inference.diffusion_nvfp4_install as nv

    present = {"torch": "2.10.0+cu128", "transformers": "5.5.4"}
    after = dict(present, **{"cosmos-guardrail": "0.3.2", "retinaface-py": "0.0.2"})
    snapshots = [present, after]
    monkeypatch.setattr(nv, "installed_distributions", lambda: snapshots.pop(0))
    monkeypatch.setattr(nv, "_installer_config", lambda uv, run: {"refusal": None})
    monkeypatch.setattr(cgi, "_uv", lambda: "/usr/bin/uv")
    installed = {"now": False}
    monkeypatch.setattr(vc3, "guardrail_installed", lambda: installed["now"])
    monkeypatch.setattr(vc3, "refresh_guardrail_import", lambda: True)
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["constraints"] = open(cmd[cmd.index("-c") + 1]).read()
        installed["now"] = True
        return subprocess.CompletedProcess(cmd, 0, stdout = "Installed 2 packages")

    cgi._state.update(state = "running", error = None, added = [])
    result = cgi.run_install(run)
    assert seen["cmd"][:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
    assert seen["cmd"][-1] == "cosmos-guardrail"
    assert "torch==2.10.0+cu128\n" in seen["constraints"]
    assert "transformers==5.5.4\n" in seen["constraints"]
    assert result["state"] == "done" and result["installed"] is True
    assert result["added"] == ["cosmos-guardrail", "retinaface-py"]


def test_a_failed_install_reports_the_installer_output(monkeypatch):
    import core.inference.diffusion_nvfp4_install as nv

    monkeypatch.setattr(nv, "installed_distributions", lambda: {"transformers": "4.57.6"})
    monkeypatch.setattr(nv, "_installer_config", lambda uv, run: {"refusal": None})
    monkeypatch.setattr(cgi, "_uv", lambda: None)
    monkeypatch.setattr(vc3, "guardrail_installed", lambda: False)

    def run(cmd, **kwargs):
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"] and "--no-input" in cmd
        return subprocess.CompletedProcess(
            cmd, 1, stdout = "cosmos-guardrail depends on transformers>=5.0.0"
        )

    result = cgi.run_install(run)
    assert result["state"] == "failed" and "transformers>=5.0.0" in result["error"]


def test_the_install_routes_are_owner_only_and_ui_only(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from auth import policy
    from routes import video as video_routes

    async def owner():
        return None

    async def not_owner():
        raise HTTPException(status_code = 403, detail = "Only the installation owner can do this")

    def check(require, full, api_refused):
        monkeypatch.setattr(policy, "require_owner", require)
        monkeypatch.setattr(policy, "full_access_permitted", lambda: full)
        monkeypatch.setattr(policy, "api_full_access_refused", lambda: api_refused)
        try:
            asyncio.run(video_routes._require_guardrail_installer("subject"))
        except HTTPException as exc:
            return exc.status_code
        return 200

    assert check(owner, True, False) == 200
    assert check(not_owner, True, False) == 403
    assert check(owner, False, False) == 403
    assert check(owner, True, True) == 403
    paths = {
        (route.path, tuple(sorted(route.methods)))
        for route in video_routes.router.routes
        if "cosmos-guardrail" in getattr(route, "path", "")
    }
    assert paths == {
        ("/video/cosmos-guardrail", ("GET",)),
        ("/video/cosmos-guardrail/install", ("POST",)),
    }
    for route in video_routes.router.routes:
        if "cosmos-guardrail" in getattr(route, "path", ""):
            deps = [d.call for d in route.dependant.dependencies]
            assert video_routes._require_guardrail_installer in deps


def test_refresh_points_a_stale_diffusers_at_the_installed_guardrail(monkeypatch):
    real_cls = type("CosmosSafetyChecker", (), {"__module__": "cosmos_guardrail.cosmos_guardrail"})
    module = types.ModuleType("cosmos_guardrail")
    module.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    module.CosmosSafetyChecker = real_cls
    monkeypatch.setitem(sys.modules, "cosmos_guardrail", module)
    utils = types.ModuleType("diffusers.utils.import_utils")
    utils._cosmos_guardrail_available = False
    stub = type(
        "CosmosSafetyChecker",
        (),
        {"__module__": "diffusers.pipelines.cosmos.pipeline_cosmos3_omni"},
    )
    pipeline = types.ModuleType("diffusers.pipelines.cosmos.pipeline_cosmos3_omni")
    pipeline.CosmosSafetyChecker = stub
    for name in [n for n in sys.modules if n.startswith("diffusers.pipelines.cosmos.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "diffusers.utils.import_utils", utils)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.cosmos.pipeline_cosmos3_omni", pipeline)
    assert vc3.refresh_guardrail_import() is True
    assert utils._cosmos_guardrail_available is True
    assert pipeline.CosmosSafetyChecker is real_cls

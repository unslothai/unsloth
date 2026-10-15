# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Switch and update both run on the update job, so they never write to the same install at once."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_freshness as freshness  # noqa: E402
import utils.llama_cpp_update as upd  # noqa: E402
import utils.whisper_cpp_update as whisper_upd  # noqa: E402

MARKER = "UNSLOTH_PREBUILT_INFO.json"

# Captured before the autouse fixture stubs them out on the module.
_whisper_phase_plan = upd._whisper_phase_plan
_resolve_backends_for_host = upd._resolve_backends_for_host


class _FakeInstallerPopen:
    def __init__(
        self,
        cmd,
        *,
        on_start = None,
        returncode = 0,
        lines = None,
        **kwargs,
    ):
        if on_start is not None:
            on_start(list(cmd), kwargs)
        self.pid = 424242
        self.returncode = returncode
        self.stdout = iter(lines or ["installed\n"])

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        pass


def _patch_installer(
    monkeypatch,
    *,
    on_start = None,
    returncode = 0,
):
    monkeypatch.setattr(
        upd.subprocess,
        "Popen",
        lambda cmd, **kw: _FakeInstallerPopen(cmd, on_start = on_start, returncode = returncode, **kw),
    )


def _write_install(dir_: Path, **marker_fields) -> str:
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("#!/bin/sh\necho stub\n")
    marker = {
        "tag": "b9596",
        "release_tag": "b9596-mix-abc",
        "published_repo": "unslothai/llama.cpp",
        "installed_at_utc": "2020-01-01T00:00:00Z",
        "asset": "app-b9596-mix-abc-linux-x64-cuda12.tar.gz",
        "install_kind": "linux-cuda",
        "backend": "cuda",
        "backend_request": "auto",
    }
    marker.update(marker_fields)
    (dir_ / MARKER).write_text(json.dumps(marker))
    return str(binary)


def _write_vulkan_install(
    *args,
    asset = "app-b9596-mix-abc-linux-x64-vulkan.tar.gz",
    backend = "vulkan",
    backend_request = "auto",
    install_kind = "linux-vulkan",
    **kwargs,
):
    """_write_install for the stock linux vulkan build."""
    return _write_install(
        *args,
        asset = asset,
        backend = backend,
        backend_request = backend_request,
        install_kind = install_kind,
        **kwargs,
    )


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch, tmp_path):
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    upd._backends_memo.clear()
    monkeypatch.setattr(freshness, "_cache_dir", lambda: tmp_path / ".freshness_cache")
    for name in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN"):
        monkeypatch.delenv(name, raising = False)
    # nothing here may touch the network or the real whisper install
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    monkeypatch.setattr(upd, "_whisper_phase_plan", lambda *a, **k: {})
    monkeypatch.setattr(
        upd,
        "_resolve_backends_for_host",
        lambda install_dir, **kwargs: {
            "backends": [
                {
                    "backend": backend,
                    "available": True,
                    "resolved_backend": "cuda" if backend == "auto" else backend,
                    "asset": (
                        "app-b9596-mix-abc-linux-x64-cuda12.tar.gz"
                        if backend in ("auto", "cuda")
                        else f"app-b9596-mix-abc-linux-x64-{backend}.tar.gz"
                    ),
                }
                for backend in ("auto", "cpu", "cuda", "rocm", "vulkan")
            ]
        },
    )


def _install(monkeypatch, tmp_path, **marker_fields) -> Path:
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, **marker_fields)
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    return install_dir


def _await_job(state = ("success", "error")) -> dict:
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in state:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never reached {state}: {upd.get_update_status()['job']}")


def _drifted(monkeypatch, resolved_backend = "vulkan"):
    """Detection now resolves somewhere other than what the marker records."""
    monkeypatch.setattr(
        upd,
        "_resolve_backends_for_host",
        lambda install_dir, **kwargs: {
            "backends": [
                {
                    "backend": backend,
                    "available": True,
                    "resolved_backend": (resolved_backend if backend == "auto" else backend),
                    "asset": f"app-b9596-mix-abc-linux-x64-{backend}.tar.gz",
                }
                for backend in ("auto", "cpu", "cuda", "rocm", "vulkan")
            ]
        },
    )


def test_a_drifted_automatic_install_is_offered_as_an_update(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)
    status = upd.get_update_status()
    assert status["update_available"] is False
    assert status["backend_migration_available"] is True
    assert (status["from_backend"], status["to_backend"]) == ("rocm", "vulkan")


def test_a_deliberate_backend_choice_is_never_offered_a_migration(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "rocm")
    _drifted(monkeypatch)
    status = upd.get_update_status()
    assert status["backend_migration_available"] is False
    assert status["to_backend"] is None


def test_an_undrifted_automatic_install_is_offered_nothing(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "cuda", backend_request = "auto")
    status = upd.get_update_status()
    assert status["backend_migration_available"] is False


def test_an_environment_override_suppresses_the_migration_offer(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "rocm")
    _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)
    assert upd.get_update_status()["backend_migration_available"] is False


def test_applying_a_migration_re_applies_auto_and_stays_an_update(monkeypatch, tmp_path):
    # "auto", not "vulkan": naming it stores a choice nobody made and drops rocm_gfx
    install_dir = _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)
    seen: dict = {}

    def _on_start(cmd, kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env") or {}
        _write_vulkan_install(install_dir)

    _patch_installer(monkeypatch, on_start = _on_start)

    assert upd.start_update()["started"] is True
    job = _await_job()

    assert job["state"] == "success", job
    assert job["operation"] == "update"
    assert job["requested_backend"] == "auto"
    assert seen["cmd"][seen["cmd"].index("--llama-backend") + 1] == "auto"
    assert seen["cmd"][seen["cmd"].index("--published-release-tag") + 1] == "b9596-mix-abc"


def test_a_migration_that_lands_back_on_the_old_backend_says_so(monkeypatch, tmp_path):
    install_dir = _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)

    def _on_start(cmd, kwargs):
        _write_install(install_dir, backend = "rocm", backend_request = "auto")

    _patch_installer(monkeypatch, on_start = _on_start)

    assert upd.start_update()["started"] is True
    job = _await_job()

    assert job["state"] == "success", job
    assert "could not be moved to vulkan" in job["message"]
    assert "now running on rocm" not in job["message"]


def test_a_migration_that_lands_on_its_target_reports_the_new_backend(monkeypatch, tmp_path):
    install_dir = _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)
    _patch_installer(
        monkeypatch,
        on_start = lambda cmd, kwargs: _write_vulkan_install(install_dir),
    )

    assert upd.start_update()["started"] is True
    job = _await_job()

    assert job["state"] == "success", job
    assert "now running on vulkan" in job["message"]


def test_an_up_to_date_install_with_no_drift_still_refuses(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "cuda", backend_request = "auto")
    result = upd.start_update()
    assert result["started"] is False
    assert result["reason"] == "up_to_date"


def test_a_switch_names_the_backend_and_keeps_the_installed_release(monkeypatch, tmp_path):
    """A switch keeps the installed release, because the paired whisper.cpp borrows its ggml modules."""
    install_dir = _install(monkeypatch, tmp_path)
    seen: dict = {}

    def _on_start(cmd, kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env") or {}
        _write_vulkan_install(install_dir, backend_request = "vulkan")

    _patch_installer(monkeypatch, on_start = _on_start)

    assert upd.start_backend_switch("vulkan")["started"] is True
    job = _await_job()

    assert job["state"] == "success", job
    assert job["operation"] == "switch"
    assert job["requested_backend"] == "vulkan"
    assert seen["cmd"][seen["cmd"].index("--llama-backend") + 1] == "vulkan"
    assert seen["cmd"][seen["cmd"].index("--published-release-tag") + 1] == "b9596-mix-abc"
    assert seen["env"]["UNSLOTH_LLAMA_CPP_BACKEND"] == "vulkan"
    assert job["message"] == "llama.cpp is now running on vulkan."


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("UNSLOTH_LLAMA_CPP_BACKEND", "cpu", "cpu"),
        ("UNSLOTH_LLAMA_CPP_BACKEND", "auto", "auto"),
        ("UNSLOTH_FORCE_VULKAN", "1", "vulkan"),
    ],
)
def test_environment_override_refuses_a_switch(monkeypatch, tmp_path, name, value, expected):
    _install(monkeypatch, tmp_path)
    monkeypatch.setenv(name, value)

    action = upd.start_backend_switch("vulkan")

    assert action["started"] is False
    assert action["reason"] == "environment_override"
    assert expected in action["message"]


def test_a_backend_with_no_build_here_is_reported_by_name(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    _patch_installer(monkeypatch, returncode = upd._EXIT_BACKEND_UNAVAILABLE)

    assert upd.start_backend_switch("rocm")["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert "Could not install a rocm llama.cpp build on this machine" in job["error"]


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("UNSLOTH_LLAMA_CPP_BACKEND", "cpu", "cpu"),
        ("UNSLOTH_FORCE_VULKAN", "1", "vulkan"),
    ],
)
def test_update_failure_names_the_environment_pinned_backend(
    monkeypatch, tmp_path, name, value, expected
):
    install_dir = _install(monkeypatch, tmp_path)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        upd,
        "_plan_llama_phase",
        lambda backend_request = None: {
            "spec": {
                "install_dir": install_dir,
                "repo": "unslothai/llama.cpp",
                "asset": None,
                "script": tmp_path / "install_llama_prebuilt.py",
                "pin_release_tag": "b9597-mix-new",
                "from_tag": "b9596-mix-abc",
                "llama_backend": "auto",
                "rocm_gfx": None,
                "backend_request": None,
            }
        },
    )
    _patch_installer(monkeypatch, returncode = upd._EXIT_BACKEND_UNAVAILABLE)

    assert upd.start_update()["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert f"Could not install a {expected} llama.cpp build on this machine" in job["error"]


def test_an_automatic_update_does_not_blame_a_backend_nobody_requested(monkeypatch, tmp_path):
    """An automatic update has no backend_request, so its failure must not blame a requested backend."""
    install_dir = _install(monkeypatch, tmp_path)
    monkeypatch.setattr(
        upd,
        "_plan_llama_phase",
        lambda backend_request = None: {
            "spec": {
                "install_dir": install_dir,
                "repo": "unslothai/llama.cpp",
                "asset": None,
                "script": tmp_path / "install_llama_prebuilt.py",
                "pin_release_tag": "b9597-mix-new",
                "from_tag": "b9596-mix-abc",
                "llama_backend": "auto",
                "rocm_gfx": None,
                "backend_request": None,
            }
        },
    )
    _patch_installer(monkeypatch, returncode = upd._EXIT_BACKEND_UNAVAILABLE)

    assert upd.start_update()["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert "Could not install a llama.cpp build on this machine" in job["error"]
    assert "requested" not in job["error"]


def test_a_switch_rejects_a_cross_repository_result(monkeypatch, tmp_path):
    install_dir = _install(monkeypatch, tmp_path)

    def _on_start(cmd, kwargs):
        _write_vulkan_install(
            install_dir,
            release_tag = "b9596",
            published_repo = "ggml-org/llama.cpp",
            asset = "llama-b9596-bin-ubuntu-vulkan-arm64.tar.gz",
            backend_request = "vulkan",
        )

    _patch_installer(monkeypatch, on_start = _on_start)

    assert upd.start_backend_switch("vulkan")["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert "backend switch must preserve" in job["error"]


def test_switching_to_the_recorded_choice_is_refused(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        tmp_path,
        asset = "app-b9596-mix-abc-linux-x64-vulkan.tar.gz",
        backend = "vulkan",
        backend_request = "vulkan",
    )

    action = upd.start_backend_switch("vulkan")

    assert action["started"] is False
    assert action["reason"] == "already_selected"


def test_a_failed_whisper_repair_can_be_retried_from_the_same_selection(monkeypatch):
    """Retrying a failed whisper repair must start a job even though the backend is already selected."""
    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: True)
    monkeypatch.setattr(whisper_upd, "repair_pairing_plan", lambda: {"phase": {"repair": True}})

    plan = _whisper_phase_plan("cuda", llama_will_run = False, llama_skip_reason = "already_selected")

    assert plan["phase"]["repair"] is True


def test_an_already_selected_request_with_a_healthy_pairing_stays_refused(monkeypatch):
    """Only a stale pairing is an owed repair; a healthy re-selection must stay refused, not succeed."""
    planned = []
    monkeypatch.setattr(
        whisper_upd, "repair_pairing_plan", lambda: planned.append(1) or {"phase": {"repair": True}}
    )

    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: False)
    assert (
        _whisper_phase_plan("cuda", llama_will_run = False, llama_skip_reason = "already_selected")
        == {}
    )

    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: True)
    assert _whisper_phase_plan("cuda", llama_will_run = False, llama_skip_reason = "local_link") == {}
    assert planned == []


def test_re_selecting_a_backend_is_refused_even_when_its_asset_moved(monkeypatch, tmp_path):
    """Re-selecting the installed backend is refused even if its asset moved; updates own that."""
    _install(
        monkeypatch,
        tmp_path,
        asset = "app-b9596-mix-abc-linux-x64-cuda12-old.tar.gz",
        backend = "cuda",
        backend_request = "cuda",
    )
    monkeypatch.setattr(
        upd,
        "_resolve_backends_for_host",
        lambda install_dir, **kwargs: {
            "backends": [
                {
                    "backend": "cuda",
                    "available": True,
                    "resolved_backend": "cuda",
                    "asset": "app-b9596-mix-abc-linux-x64-cuda13.tar.gz",
                }
            ]
        },
    )

    action = upd.start_backend_switch("cuda")

    assert action["started"] is False
    assert action["reason"] == "already_selected"


def test_auto_reapplies_when_hardware_detection_changes(monkeypatch, tmp_path):
    install_dir = _install(monkeypatch, tmp_path, backend = "cpu", backend_request = "auto")
    _patch_installer(
        monkeypatch,
        on_start = lambda cmd, kwargs: _write_install(
            install_dir, backend = "cuda", backend_request = "auto"
        ),
    )

    assert upd.start_backend_switch("auto")["started"] is True
    assert _await_job()["state"] == "success"


def test_pinning_a_detected_install_to_its_own_backend_is_a_real_change(monkeypatch, tmp_path):
    """auto -> cuda on a CUDA box installs the same bundle, and still matters: it
    stops the next update from re-detecting the machine onto something else."""
    install_dir = _install(monkeypatch, tmp_path, backend = "cuda", backend_request = "auto")
    _patch_installer(
        monkeypatch,
        on_start = lambda cmd, kwargs: _write_install(
            install_dir, backend = "cuda", backend_request = "cuda"
        ),
    )

    assert upd.start_backend_switch("cuda")["started"] is True
    assert _await_job()["state"] == "success"


def test_unavailable_backend_is_refused_before_the_runtime_is_unloaded(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(
        upd,
        "_resolve_backends_for_host",
        lambda install_dir, **kwargs: {
            "backends": [{"backend": "vulkan", "available": False, "resolved_backend": None}]
        },
    )
    installer_started = False

    def _unexpected_installer(cmd, kwargs):
        nonlocal installer_started
        installer_started = True

    _patch_installer(monkeypatch, on_start = _unexpected_installer)

    action = upd.start_backend_switch("vulkan")

    assert action["started"] is False
    assert action["reason"] == "backend_unavailable"
    assert installer_started is False


def test_switch_preflight_uses_the_install_recorded_repository(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, published_repo = "owner/custom-llama")
    seen = {}

    def _resolve(install_dir, **kwargs):
        seen.update(kwargs)
        return {"backends": [{"backend": "cpu", "available": False, "resolved_backend": None}]}

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve)

    action = upd.start_backend_switch("cpu")

    assert action["started"] is False
    assert seen["published_repo"] == "owner/custom-llama"


def test_switch_fails_if_the_installer_does_not_record_the_requested_backend(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "cuda", backend_request = "auto")
    _patch_installer(monkeypatch)

    assert upd.start_backend_switch("cpu")["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert "requested cpu" in job["error"]


def test_an_unknown_backend_is_refused_without_starting_a_job(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)

    action = upd.start_backend_switch("sycl")

    assert action["started"] is False
    assert action["reason"] == "unknown_backend"
    assert action["job"]["state"] == "idle"


def test_a_source_build_has_no_backend_to_switch(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    (install_dir / "build" / "bin").mkdir(parents = True)
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install.py")

    action = upd.start_backend_switch("cpu")

    assert action["started"] is False
    assert action["reason"] == "not_prebuilt"


def test_a_switch_and_an_update_cannot_run_at_once(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    with upd._job_lock:
        upd._job.update(state = upd._JOB_RUNNING, message = "busy")
    try:
        switch = upd.start_backend_switch("cpu")
        update = upd.start_update()
        assert switch["reason"] == "already_running"
        assert update["reason"] == "already_running"
        assert switch["message"] == "Another llama.cpp install is already running."
        assert update["message"] == "Another llama.cpp install is already running."
    finally:
        upd._reset_job_for_tests()


def test_backend_resolution_is_part_of_the_serialized_operation(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    with upd._job_lock:
        upd._job.update(
            state = upd._JOB_SUCCESS,
            message = "old completed job",
            finished_at = "2020-01-01T00:00:00Z",
        )
    entered = threading.Event()
    release = threading.Event()

    def _resolve(*args, **kwargs):
        entered.set()
        assert release.wait(timeout = 5)
        return {"backends": [{"backend": "rocm", "available": False, "resolved_backend": None}]}

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve)
    first: dict = {}
    thread = threading.Thread(
        target = lambda: first.update(upd.start_backend_switch("rocm")), daemon = True
    )
    thread.start()
    assert entered.wait(timeout = 5)

    second = upd.start_update()
    assert second["reason"] == "already_running"
    assert second["job"]["state"] == "running"
    assert second["job"]["operation"] == "switch"
    assert second["job"]["requested_backend"] == "rocm"
    assert second["job"]["finished_at"] is None
    release.set()
    thread.join(timeout = 5)

    assert first["reason"] == "backend_unavailable"
    assert first["job"]["state"] == "error"
    assert not upd._operation_lock.locked()


def test_status_reports_the_install_and_the_options(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(
        upd,
        "_resolve_backends_for_host",
        lambda install_dir, **kwargs: {
            "backends": [
                {
                    "backend": "auto",
                    "available": True,
                    "resolved_backend": "cuda",
                    "release_tag": "b9596-mix-abc",
                    "asset": "app-b9596-mix-abc-linux-x64-cuda12.tar.gz",
                },
                {"backend": "rocm", "available": False, "reason": "unavailable"},
                {"backend": "sycl", "available": True},
            ]
        },
    )
    monkeypatch.setattr(
        # Bound into the update module at import, so the freshness module's copy
        # is not the one it calls.
        upd,
        "latest_release_assets",
        lambda repo, force_refresh = False: {"app-b9596-mix-abc-linux-x64-cuda12.tar.gz": 1234},
    )

    status = upd.get_backend_status()

    assert status["supported"] is True
    assert status["backend"] == "cuda"
    assert status["backend_request"] == "auto"
    assert status["selection_applied"] is True
    assert status["installed_tag"] == "b9596-mix-abc"
    by_backend = {option["backend"]: option for option in status["options"]}
    assert by_backend["auto"]["resolved_backend"] == "cuda"
    assert by_backend["auto"]["download_size_bytes"] == 1234
    assert by_backend["rocm"]["available"] is False
    assert "sycl" not in by_backend


def test_status_skips_the_resolver_when_update_checks_disabled(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")

    def _no_resolve(*args, **kwargs):
        raise AssertionError("resolved backends despite UNSLOTH_DISABLE_UPDATE_CHECK=1")

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _no_resolve)
    monkeypatch.setattr(upd, "latest_release_assets", _no_resolve)

    status = upd.get_backend_status(force_refresh = True)

    assert status["supported"] is False
    assert status["reason"] == "update_checks_disabled"
    assert status["backend"] == "cuda"
    assert status["options"] == []


def test_status_reports_when_auto_now_resolves_to_another_backend(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "cpu", backend_request = "auto")

    status = upd.get_backend_status()

    assert status["backend"] == "cpu"
    assert status["backend_request"] == "auto"
    assert status["selection_applied"] is False


def test_status_keeps_a_concrete_choice_applied_when_its_asset_moves(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        tmp_path,
        asset = "app-b9596-mix-abc-linux-x64-cuda12-old.tar.gz",
        backend = "cuda",
        backend_request = "cuda",
    )

    status = upd.get_backend_status()

    assert status["backend"] == "cuda"
    assert status["backend_request"] == "cuda"
    assert status["selection_applied"] is True


def test_a_switch_that_installs_nothing_plans_no_whisper_phase(monkeypatch):
    # whisper re-pairs only because llama's ggml is replaced; no llama phase, nothing to re-pair
    planned = []
    monkeypatch.setattr(
        whisper_upd, "repair_pairing_plan", lambda: planned.append(1) or {"phase": {"repair": True}}
    )

    assert _whisper_phase_plan("cuda", llama_will_run = False) == {}
    assert planned == []
    assert _whisper_phase_plan("cuda", llama_will_run = True)["phase"]["repair"] is True


def _slim_whisper(
    monkeypatch,
    tmp_path,
    install_kind = "slim",
) -> str:
    whisper_dir = tmp_path / f"whisper.cpp-{install_kind}"
    binary = _write_install(whisper_dir, install_kind = install_kind)
    (whisper_dir / MARKER).rename(whisper_dir / whisper_upd._INSTALL_MARKER_NAME)
    monkeypatch.setattr(whisper_upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        whisper_upd, "_installer_script", lambda: tmp_path / "install_whisper_prebuilt.py"
    )
    return binary


def test_only_a_slim_whisper_install_is_re_paired(monkeypatch, tmp_path):
    _slim_whisper(monkeypatch, tmp_path)
    assert whisper_upd.repair_pairing_plan()["phase"]["repair"] is True

    _slim_whisper(monkeypatch, tmp_path, install_kind = "fat")
    # a fat bundle ships its own ggml, so llama's backend is not its backend
    assert whisper_upd.repair_pairing_plan()["skip_reason"] == "self_contained"


def test_whisper_repair_installs_the_backend_llama_landed_on(monkeypatch, tmp_path):
    _slim_whisper(monkeypatch, tmp_path)
    monkeypatch.setattr(
        whisper_upd,
        "_installed_llama_bundle",
        lambda: ("rocm", "app-b9596-linux-x64-rocm-gfx1100.tar.gz"),
    )
    calls = []
    monkeypatch.setattr(
        whisper_upd,
        "_install_latest",
        lambda *args, **kwargs: calls.append(args) or {},
    )

    result = whisper_upd.run_repair_phase(
        {"install_dir": tmp_path, "repo": "unslothai/whisper.cpp", "script": tmp_path / "i.py"},
        lambda progress: None,
    )

    assert calls[0][2] == "app-b9596-linux-x64-rocm-gfx1100.tar.gz"
    assert calls[0][3] == "rocm"
    assert "rocm backend" in result["message"]


def test_whisper_repair_skips_a_backend_it_is_already_built_against(monkeypatch, tmp_path):
    # _slim_whisper records marker backend "cuda"
    _slim_whisper(monkeypatch, tmp_path)
    monkeypatch.setattr(whisper_upd, "_installed_llama_bundle", lambda: ("cuda", "asset.tar.gz"))
    monkeypatch.setattr(
        whisper_upd, "_install_latest", lambda *a, **k: pytest.fail("reinstalled needlessly")
    )

    assert whisper_upd.run_repair_phase({}, lambda progress: None) == {}


def test_whisper_repair_treats_only_incompatibility_as_unavailable(monkeypatch, tmp_path):
    _slim_whisper(monkeypatch, tmp_path)
    monkeypatch.setattr(whisper_upd, "_installed_llama_bundle", lambda: ("vulkan", "asset.tar.gz"))
    monkeypatch.setattr(
        whisper_upd,
        "_install_latest",
        lambda *a, **k: (_ for _ in ()).throw(whisper_upd._flow.InstallerExit(2, "incompatible")),
    )

    result = whisper_upd.run_repair_phase(
        {"install_dir": tmp_path, "repo": "r", "script": tmp_path / "i.py"}, lambda p: None
    )

    assert "no whisper.cpp build is published" in result["message"]


@pytest.mark.parametrize("returncode", [1, 3])
def test_whisper_repair_surfaces_retryable_installer_failures(monkeypatch, tmp_path, returncode):
    _slim_whisper(monkeypatch, tmp_path)
    monkeypatch.setattr(whisper_upd, "_installed_llama_bundle", lambda: ("vulkan", "asset.tar.gz"))
    monkeypatch.setattr(
        whisper_upd,
        "_install_latest",
        lambda *a, **k: (_ for _ in ()).throw(
            whisper_upd._flow.InstallerExit(returncode, "retryable")
        ),
    )

    with pytest.raises(whisper_upd._flow.InstallerExit) as raised:
        whisper_upd.run_repair_phase(
            {"install_dir": tmp_path, "repo": "r", "script": tmp_path / "i.py"}, lambda p: None
        )

    assert raised.value.returncode == returncode


def test_status_surfaces_an_environment_pin(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "vulkan")
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(upd, "_resolve_backends_for_host", lambda install_dir, **kwargs: {})

    status = upd.get_backend_status()

    assert status["env_backend"] == "vulkan"


def test_status_says_why_an_unmanaged_install_cannot_be_switched(monkeypatch, tmp_path):
    monkeypatch.setattr(upd, "_find_binary", lambda: None)

    status = upd.get_backend_status()

    assert status["supported"] is False
    assert status["reason"] == "not_installed"
    assert status["options"] == []


def test_status_degrades_when_the_options_cannot_be_resolved(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(upd, "_resolve_backends_for_host", lambda install_dir, **kwargs: None)

    status = upd.get_backend_status()

    assert status["reason"] == "unresolved"
    assert status["supported"] is False
    assert status["backend"] == "cuda"
    assert status["options"] == []


def test_backend_resolution_failures_are_not_cached(monkeypatch, tmp_path):
    memo = {}
    responses = [
        (1, ""),
        (
            0,
            json.dumps(
                {"backends": [{"backend": "auto", "available": True, "resolved_backend": "cuda"}]}
            ),
        ),
    ]

    def _run(cmd, **kwargs):
        returncode, stdout = responses.pop(0)
        return type("Result", (), {"returncode": returncode, "stdout": stdout})()

    monkeypatch.setattr(upd.subprocess, "run", _run)

    kwargs = {
        "force_refresh": False,
        "memo": memo,
        "installer_script": lambda: tmp_path / "install.py",
        "log_message": "test resolver failed",
        "mode": ("--resolve-backends", "latest"),
    }
    assert upd._flow.resolve_prebuilt_for_host(**kwargs) is None
    resolved = upd._flow.resolve_prebuilt_for_host(**kwargs)
    assert resolved["backends"][0]["available"] is True
    assert responses == []


def test_the_migration_resolver_replays_the_arch_the_marker_recorded(monkeypatch, tmp_path):
    """Replay the marker's arch, or a working ROCm install reads as drifted and offers a CPU migration."""
    install_dir = _install(
        monkeypatch,
        tmp_path,
        backend = "rocm",
        install_kind = "linux-rocm",
        asset = "app-b9596-mix-abc-linux-x64-rocm-gfx1151.tar.gz",
        rocm_gfx = "gfx1151",
    )
    marker = upd.read_install_marker(upd._find_binary())
    seen: list = []

    def _resolver(**kwargs):
        seen.append(kwargs.get("extra_env"))
        gfx = (kwargs.get("extra_env") or {}).get("UNSLOTH_ROCM_GFX_REMEMBERED")
        resolved = "rocm" if gfx else "cpu"
        return {"backends": [{"backend": "auto", "available": True, "resolved_backend": resolved}]}

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    upd._backends_memo.clear()

    assert upd._pending_backend_migration(upd._find_binary(), marker) is None
    assert seen == [{"UNSLOTH_ROCM_GFX_REMEMBERED": "gfx1151"}]
    assert install_dir.exists()


def test_a_marker_with_no_recorded_arch_passes_no_replay(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    marker = upd.read_install_marker(upd._find_binary())
    seen: list = []

    def _resolver(**kwargs):
        seen.append(kwargs.get("extra_env"))
        return {"backends": [{"backend": "auto", "available": True, "resolved_backend": "cpu"}]}

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    upd._backends_memo.clear()

    upd._pending_backend_migration(upd._find_binary(), marker)
    assert seen == [None]


def test_the_picker_describes_the_same_host_the_update_check_does(monkeypatch, tmp_path):
    """The picker's resolve must replay the recovered arch, or it reads an AMD host as CPU-only."""
    _install(
        monkeypatch,
        tmp_path,
        backend = "rocm",
        backend_request = "auto",
        install_kind = "windows-rocm",
        asset = "app-b9596-mix-abc-windows-x64-rocm-gfx1151.zip",
    )
    assert upd.read_install_marker(upd._find_binary()).get("rocm_gfx") is None

    def _resolver(**kwargs):
        gfx = (kwargs.get("extra_env") or {}).get("UNSLOTH_ROCM_GFX_REMEMBERED")
        return {
            "backends": [
                {
                    "backend": backend,
                    "available": bool(gfx) or backend in ("auto", "cpu", "vulkan"),
                    "resolved_backend": (
                        ("vulkan" if gfx else "cpu") if backend == "auto" else backend
                    ),
                }
                for backend in ("auto", "cpu", "rocm", "vulkan")
            ]
        }

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    monkeypatch.setattr(upd, "latest_release_assets", lambda repo, force_refresh = False: {})
    upd._backends_memo.clear()

    status = upd.get_backend_status()
    by_backend = {option["backend"]: option for option in status["options"]}
    assert by_backend["auto"]["resolved_backend"] == "vulkan"
    assert by_backend["rocm"]["available"] is True


def test_the_arch_is_recovered_from_the_installed_bundle_when_none_was_recorded(
    monkeypatch, tmp_path
):
    _install(
        monkeypatch,
        tmp_path,
        backend = "rocm",
        install_kind = "windows-rocm",
        asset = "app-b9596-mix-abc-windows-x64-rocm-gfx1151.zip",
    )
    marker = upd.read_install_marker(upd._find_binary())
    assert marker.get("rocm_gfx") is None
    seen: list = []

    def _resolver(**kwargs):
        seen.append(kwargs.get("extra_env"))
        gfx = (kwargs.get("extra_env") or {}).get("UNSLOTH_ROCM_GFX_REMEMBERED")
        return {
            "backends": [
                {
                    "backend": "auto",
                    "available": True,
                    "resolved_backend": "rocm" if gfx else "cpu",
                }
            ]
        }

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    upd._backends_memo.clear()

    assert upd._pending_backend_migration(upd._find_binary(), marker) is None
    assert seen == [{"UNSLOTH_ROCM_GFX_REMEMBERED": "gfx1151"}]


def test_a_gpu_install_is_never_offered_a_migration_onto_cpu(monkeypatch, tmp_path):
    # an empty probe and a host that lost its GPU look the same here; Settings still can
    _install(
        monkeypatch,
        tmp_path,
        backend = "rocm",
        install_kind = "linux-rocm",
        asset = "app-b9596-mix-abc-linux-x64-rocm-gfx1151.tar.gz",
        rocm_gfx = "gfx1151",
    )
    marker = upd.read_install_marker(upd._find_binary())

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(
        upd._flow,
        "resolve_prebuilt_for_host",
        lambda **_kw: {
            "backends": [{"backend": "auto", "available": True, "resolved_backend": "cpu"}]
        },
    )
    upd._backends_memo.clear()
    assert upd._pending_backend_migration(upd._find_binary(), marker) is None

    monkeypatch.setattr(
        upd._flow,
        "resolve_prebuilt_for_host",
        lambda **_kw: {
            "backends": [{"backend": "auto", "available": True, "resolved_backend": "vulkan"}]
        },
    )
    upd._backends_memo.clear()
    assert upd._pending_backend_migration(upd._find_binary(), marker) == "vulkan"


def test_applying_a_migration_replays_the_arch_the_offer_was_made_with(monkeypatch, tmp_path):
    """Apply must re-resolve with the offer's arch, or it reads as already applied and refuses silently."""
    install_dir = _install(
        monkeypatch,
        tmp_path,
        backend = "rocm",
        backend_request = "auto",
        install_kind = "linux-rocm",
        asset = "app-b9596-mix-abc-linux-x64-rocm-gfx1151.tar.gz",
        rocm_gfx = "gfx1151",
    )

    def _resolver(**kwargs):
        gfx = (kwargs.get("extra_env") or {}).get("UNSLOTH_ROCM_GFX_REMEMBERED")
        auto = "vulkan" if gfx else "rocm"
        return {
            "backends": [
                {
                    "backend": backend,
                    "available": True,
                    "resolved_backend": (auto if backend == "auto" else backend),
                    "asset": f"app-b9596-mix-abc-linux-x64-{backend}.tar.gz",
                }
                for backend in ("auto", "cpu", "rocm", "vulkan")
            ]
        }

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    upd._backends_memo.clear()

    assert upd.get_update_status()["backend_migration_available"] is True

    seen: dict = {}

    def _on_start(cmd, kwargs):
        seen["cmd"] = cmd
        _write_vulkan_install(install_dir)

    _patch_installer(monkeypatch, on_start = _on_start)

    result = upd.start_update()
    assert result["started"] is True, result
    job = _await_job()
    assert job["state"] == "success", job
    assert seen["cmd"][seen["cmd"].index("--llama-backend") + 1] == "auto"


def test_applying_a_migration_replays_an_arch_only_the_bundle_names(monkeypatch, tmp_path):
    """Apply must use the same recovered arch as the offer, including one only the bundle names."""
    install_dir = _install(
        monkeypatch,
        tmp_path,
        backend = "rocm",
        backend_request = "auto",
        install_kind = "linux-rocm",
        asset = "app-b9596-mix-abc-linux-x64-rocm-gfx1151.tar.gz",
    )
    assert upd.read_install_marker(upd._find_binary()).get("rocm_gfx") is None

    def _resolver(**kwargs):
        gfx = (kwargs.get("extra_env") or {}).get("UNSLOTH_ROCM_GFX_REMEMBERED")
        auto = "vulkan" if gfx else "rocm"
        return {
            "backends": [
                {
                    "backend": backend,
                    "available": True,
                    "resolved_backend": (auto if backend == "auto" else backend),
                    "asset": f"app-b9596-mix-abc-linux-x64-{backend}.tar.gz",
                }
                for backend in ("auto", "cpu", "rocm", "vulkan")
            ]
        }

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    upd._backends_memo.clear()

    assert upd.get_update_status()["backend_migration_available"] is True

    def _on_start(cmd, kwargs):
        _write_vulkan_install(install_dir)

    _patch_installer(monkeypatch, on_start = _on_start)

    result = upd.start_update()
    assert result["started"] is True, result
    job = _await_job()
    assert job["state"] == "success", job


def test_an_apply_that_no_longer_drifts_still_refuses(monkeypatch, tmp_path):
    _install(
        monkeypatch,
        tmp_path,
        backend = "vulkan",
        backend_request = "auto",
        install_kind = "linux-vulkan",
        asset = "app-b9596-mix-abc-linux-x64-vulkan.tar.gz",
        rocm_gfx = "gfx1151",
    )

    def _resolver(**kwargs):
        return {
            "backends": [
                {
                    "backend": backend,
                    "available": True,
                    "resolved_backend": ("vulkan" if backend == "auto" else backend),
                    "asset": f"app-b9596-mix-abc-linux-x64-{backend}.tar.gz",
                }
                for backend in ("auto", "cpu", "rocm", "vulkan")
            ]
        }

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _resolve_backends_for_host)
    monkeypatch.setattr(upd._flow, "resolve_prebuilt_for_host", _resolver)
    upd._backends_memo.clear()

    result = upd.start_update()
    assert result["started"] is False
    assert result["reason"] == "up_to_date"


def test_running_job_status_does_not_resolve_options_again(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)

    def _unexpected_resolver(*args, **kwargs):
        raise AssertionError("resolver raced the running installer")

    monkeypatch.setattr(upd, "_resolve_backends_for_host", _unexpected_resolver)
    with upd._job_lock:
        upd._job.update(state = upd._JOB_RUNNING, operation = "update")
    try:
        status = upd.get_backend_status()
    finally:
        upd._reset_job_for_tests()

    assert status["job"]["state"] == "running"
    assert status["options"] == []


def test_an_explicit_auto_in_the_environment_still_gets_the_migration(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "auto")
    _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)
    assert upd.get_update_status()["backend_migration_available"] is True


def test_an_explicit_recheck_re_resolves_the_backend(monkeypatch, tmp_path):
    # The resolver is memoized for 24h, so force_refresh has to reach it or "check again"
    # answers from the pre-change resolve for the rest of the TTL.
    _install(monkeypatch, tmp_path, backend = "rocm", backend_request = "auto")
    _drifted(monkeypatch)
    seen: list = []
    real = upd._pending_backend_migration

    def _spy(
        binary,
        marker,
        *,
        force_refresh = False,
    ):
        seen.append(force_refresh)
        return real(binary, marker, force_refresh = force_refresh)

    monkeypatch.setattr(upd, "_pending_backend_migration", _spy)
    upd.get_update_status(force_refresh = True)
    assert seen == [True], seen


def test_a_migration_keeps_a_pending_whisper_update(monkeypatch):
    chained = {"phase": {"kind": "whisper", "tag": "w2"}, "update_available": True}
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kw: chained)
    repair_calls = []
    monkeypatch.setattr(
        whisper_upd, "repair_pairing_plan", lambda: (repair_calls.append(True), {})[1]
    )
    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: True)

    plan = _whisper_phase_plan("auto", llama_will_run = True, migration = True)
    assert plan == chained, plan
    assert repair_calls == [], "a migration must not take the repair-only branch"


def test_a_plain_update_that_can_move_the_backend_still_re_pairs_whisper(monkeypatch, tmp_path):
    # an update passes no --llama-backend, so the "auto" default can move under it
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kw: {"skip_reason": "up_to_date"})
    repair = {"update_available": True, "phase": {"repair": True}}
    monkeypatch.setattr(whisper_upd, "repair_pairing_plan", lambda: repair)

    assert _whisper_phase_plan(None, llama_will_run = True) is repair
    assert _whisper_phase_plan(None, llama_will_run = False)["skip_reason"] == "up_to_date"


def test_a_recorded_backend_choice_does_not_schedule_a_re_pair(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend_request = "cuda")
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kw: {"skip_reason": "up_to_date"})
    monkeypatch.setattr(whisper_upd, "repair_pairing_plan", lambda: {"phase": {"repair": True}})
    assert _whisper_phase_plan(None, llama_will_run = True)["skip_reason"] == "up_to_date"


def test_a_self_contained_whisper_keeps_its_up_to_date_reason(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kw: {"skip_reason": "up_to_date"})
    monkeypatch.setattr(
        whisper_upd, "repair_pairing_plan", lambda: {"skip_reason": "self_contained", "phase": None}
    )
    assert _whisper_phase_plan(None, llama_will_run = True)["skip_reason"] == "up_to_date"


def test_a_deliberate_switch_still_takes_the_repair_branch(monkeypatch):
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kw: {"phase": {"kind": "whisper"}})
    marker = {"repaired": True}
    monkeypatch.setattr(whisper_upd, "repair_pairing_plan", lambda: marker)
    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: True)
    assert _whisper_phase_plan("vulkan", llama_will_run = True) is marker

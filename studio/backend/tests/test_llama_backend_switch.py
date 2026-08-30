# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Switching the llama.cpp backend from the app.

The picker in Settings > System reads utils.llama_cpp_update.get_backend_status
and applies with start_backend_switch. Both run on the update job, so a switch and
an update can never write to the same install at once.

The installer subprocess is stubbed; no download or GPU needed.
"""

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

# Captured before the autouse fixture stubs it out on the module.
_whisper_phase_plan = upd._whisper_phase_plan


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


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch, tmp_path):
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    upd._backends_memo.clear()
    monkeypatch.setattr(freshness, "_cache_dir", lambda: tmp_path / ".freshness_cache")
    for name in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN"):
        monkeypatch.delenv(name, raising = False)
    # Nothing in this suite may touch the network or the real whisper install.
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


# ── Applying ──


def test_a_switch_names_the_backend_and_keeps_the_installed_release(monkeypatch, tmp_path):
    """A switch changes the backend and nothing else.

    Pinned to the release already installed rather than the latest: bundling an
    update into it would also break the paired whisper.cpp install, whose slim
    bundle names the exact llama.cpp release it borrows ggml modules from.
    """
    install_dir = _install(monkeypatch, tmp_path)
    seen: dict = {}

    def _on_start(cmd, kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env") or {}
        _write_install(
            install_dir,
            asset = "app-b9596-mix-abc-linux-x64-vulkan.tar.gz",
            install_kind = "linux-vulkan",
            backend = "vulkan",
            backend_request = "vulkan",
        )

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
    # Hardware or published bundles can change after option resolution.
    _install(monkeypatch, tmp_path)
    _patch_installer(monkeypatch, returncode = upd._EXIT_BACKEND_UNAVAILABLE)

    assert upd.start_backend_switch("rocm")["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert "Could not install the rocm llama.cpp build on this machine" in job["error"]


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
    assert f"Could not install the {expected} llama.cpp build on this machine" in job["error"]


def test_a_switch_rejects_a_cross_repository_result(monkeypatch, tmp_path):
    install_dir = _install(monkeypatch, tmp_path)

    def _on_start(cmd, kwargs):
        _write_install(
            install_dir,
            release_tag = "b9596",
            published_repo = "ggml-org/llama.cpp",
            asset = "llama-b9596-bin-ubuntu-vulkan-arm64.tar.gz",
            install_kind = "linux-vulkan",
            backend = "vulkan",
            backend_request = "vulkan",
        )

    _patch_installer(monkeypatch, on_start = _on_start)

    assert upd.start_backend_switch("vulkan")["started"] is True
    job = _await_job()

    assert job["state"] == "error"
    assert "backend switch must preserve" in job["error"]


# ── Refusing ──


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
    """The one already-selected request that must still start a job.

    The llama phase runs first and records the new backend, so a retryable whisper
    failure (a dropped download, an install that was busy) ends with llama switched and
    dictation still hardlinked to the old runtime. Retrying is then "already selected",
    and refusing it leaves the reported failure unfixable except by switching llama.cpp
    away and back.
    """
    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: True)
    monkeypatch.setattr(whisper_upd, "repair_pairing_plan", lambda: {"phase": {"repair": True}})

    plan = _whisper_phase_plan("cuda", llama_will_run = False, llama_skip_reason = "already_selected")

    assert plan["phase"]["repair"] is True


def test_an_already_selected_request_with_a_healthy_pairing_stays_refused(monkeypatch):
    """Staleness is what separates an owed repair from an ordinary no-op.

    Without it every already-selected request would start a whisper job that finds
    nothing to do and reports success, replacing a clear refusal with a false one.
    """
    planned = []
    monkeypatch.setattr(
        whisper_upd, "repair_pairing_plan", lambda: planned.append(1) or {"phase": {"repair": True}}
    )

    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: False)
    assert (
        _whisper_phase_plan("cuda", llama_will_run = False, llama_skip_reason = "already_selected")
        == {}
    )

    # Every other refusal stays refused whatever the pairing looks like.
    monkeypatch.setattr(whisper_upd, "slim_pairing_is_stale", lambda: True)
    assert _whisper_phase_plan("cuda", llama_will_run = False, llama_skip_reason = "local_link") == {}
    assert planned == []


def test_re_selecting_a_backend_is_refused_even_when_its_asset_moved(monkeypatch, tmp_path):
    """The picker chooses a backend, not a bundle.

    A newer per-architecture asset for the backend already selected is an update,
    which the update flow offers on its own schedule. Making Apply mean "reinstall
    too" would hand the same install to two jobs with two different triggers.
    """
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
    # The shared job prevents concurrent installers.
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


# ── Status ──


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
                # Older pickers ignore backends they cannot label.
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


def test_status_reports_when_auto_now_resolves_to_another_backend(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, backend = "cpu", backend_request = "auto")

    status = upd.get_backend_status()

    assert status["backend"] == "cpu"
    assert status["backend_request"] == "auto"
    assert status["selection_applied"] is False


def test_status_keeps_a_concrete_choice_applied_when_its_asset_moves(monkeypatch, tmp_path):
    # The recorded name and the installed backend agree, which is all a concrete
    # choice claims. A newer asset for it is an update, not an unapplied selection.
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
    # Whisper only re-pairs because llama's ggml is being replaced. Without a llama
    # phase there is nothing to re-pair, so a refusal stays a refusal instead of
    # turning into a whisper-only job that reports a switch nobody performed.
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
    # A fat bundle ships its own ggml, so llama's backend is not its backend.
    assert whisper_upd.repair_pairing_plan()["skip_reason"] == "self_contained"


def test_whisper_repair_installs_the_backend_llama_landed_on(monkeypatch, tmp_path):
    _slim_whisper(monkeypatch, tmp_path)  # marker backend "cuda"
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

    # The llama asset carries the AMD arch the new bundle was built for.
    assert calls[0][2] == "app-b9596-linux-x64-rocm-gfx1100.tar.gz"
    assert calls[0][3] == "rocm"
    assert "rocm backend" in result["message"]


def test_whisper_repair_skips_a_backend_it_is_already_built_against(monkeypatch, tmp_path):
    # Detection can land back where it was; the release is preserved either way,
    # so the hardlinks still point at the same build.
    _slim_whisper(monkeypatch, tmp_path)  # marker backend "cuda"
    monkeypatch.setattr(whisper_upd, "_installed_llama_bundle", lambda: ("cuda", "asset.tar.gz"))
    monkeypatch.setattr(
        whisper_upd, "_install_latest", lambda *a, **k: pytest.fail("reinstalled needlessly")
    )

    assert whisper_upd.run_repair_phase({}, lambda progress: None) == {}


def test_whisper_repair_treats_only_incompatibility_as_unavailable(monkeypatch, tmp_path):
    _slim_whisper(monkeypatch, tmp_path)  # marker backend "cuda"
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
    _slim_whisper(monkeypatch, tmp_path)  # marker backend "cuda"
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
    # Surface environment overrides instead of accepting an ineffective choice.
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
    # Offline status keeps the installed backend without guessing alternatives.
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the combined llama+whisper update item.

llama.cpp is the single main update item; whisper.cpp piggybacks on it. These
pin the union status (update_available = llama behind OR whisper behind), the
chained apply (llama phase first, whisper phase only when behind), the failure
policy (llama failure aborts; whisper failure keeps the llama partial success),
the silent whisper skips, and the backward-compatible payload shape.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_freshness as freshness  # noqa: E402
import utils.llama_cpp_update as upd  # noqa: E402
import utils.whisper_cpp_freshness as wfresh  # noqa: E402
import utils.whisper_cpp_update as wupd  # noqa: E402

MARKER = "UNSLOTH_PREBUILT_INFO.json"
WHISPER_MARKER = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

# The top-level status and job fields that predate the whisper piggyback; the
# combined payload must stay an exact superset so current UI code keeps working.
LEGACY_STATUS_FIELDS = {
    "supported",
    "update_available",
    "stale",
    "installed_tag",
    "latest_tag",
    "published_repo",
    "installed_at_utc",
    "age_days",
    "source_build",
    "update_size_bytes",
    "job",
}
LEGACY_JOB_FIELDS = {
    "state",
    "message",
    "from_tag",
    "to_tag",
    "reload_required",
    "error",
    "progress",
    "started_at",
    "finished_at",
}


class _FakeInstallerPopen:
    """Stands in for the streamed llama installer process."""

    def __init__(
        self,
        cmd,
        *,
        returncode = 0,
        lines = None,
        on_start = None,
        **kwargs,
    ):
        if on_start is not None:
            on_start(list(cmd))
        self.returncode = returncode
        self.stdout = iter(lines or [])

    def wait(self):
        return self.returncode

    def kill(self):
        pass


def _patch_llama_installer(
    monkeypatch,
    *,
    returncode = 0,
    lines = None,
    on_start = None,
):
    # Only intercept the installer invocation: importing routes.inference inside
    # the worker can Popen unrelated host probes (ldconfig etc).
    def _popen(cmd, **kw):
        is_installer = any("install_llama_prebuilt" in str(part) for part in cmd)
        return _FakeInstallerPopen(
            cmd,
            returncode = returncode if is_installer else 0,
            lines = lines if is_installer else None,
            on_start = on_start if is_installer else None,
        )

    monkeypatch.setattr(upd.subprocess, "Popen", _popen)


def _write_llama_install(dir_: Path, tag: str) -> str:
    """Create a fake llama prebuilt install and return the llama-server path."""
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("stub")
    (dir_ / MARKER).write_text(
        json.dumps(
            {
                "tag": tag,
                "release_tag": tag,
                "published_repo": "unslothai/llama.cpp",
                "installed_at_utc": "2020-01-01T00:00:00Z",
            }
        )
    )
    return str(binary)


def _write_whisper_install(
    dir_: Path,
    tag: str,
    backend: str = "cpu",
) -> str:
    """Create a fake whisper prebuilt install and return the whisper-server path."""
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "whisper-server"
    binary.write_text("stub")
    (dir_ / WHISPER_MARKER).write_text(
        json.dumps(
            {
                "release_tag": tag,
                "upstream_tag": tag.split("-")[0],
                "published_repo": "unslothai/whisper.cpp",
                "backend": backend,
                "installed_at_utc": "2020-01-01T00:00:00Z",
            }
        )
    )
    return str(binary)


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch, tmp_path):
    freshness.reset_caches()
    wfresh.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    wupd._resolve_memo.clear()
    monkeypatch.setattr(freshness, "_cache_dir", lambda: tmp_path / ".llama_cache")
    monkeypatch.setattr(wfresh, "_cache_dir", lambda: tmp_path / ".whisper_cache")
    for var in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        "WHISPER_SERVER_PATH",
        "UNSLOTH_WHISPER_CPP_PATH",
    ):
        monkeypatch.delenv(var, raising = False)
    # Never hit the network in these tests.
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    monkeypatch.setattr(wfresh, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    yield
    freshness.reset_caches()
    wfresh.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    wupd._resolve_memo.clear()


def _setup_llama(
    monkeypatch,
    tmp_path,
    *,
    installed = "b9493",
    latest = "b9518",
):
    """Marker-managed llama install; behind when installed != latest."""
    install_dir = tmp_path / "llama.cpp"
    binary = _write_llama_install(install_dir, installed)
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: latest)
    return install_dir


def _setup_whisper(
    monkeypatch,
    tmp_path,
    *,
    installed = "v1.9.1-unsloth.1",
    latest = "v1.9.2-unsloth.1",
):
    """Marker-managed whisper install; behind when latest is newer."""
    install_dir = tmp_path / "whisper.cpp"
    binary = _write_whisper_install(install_dir, installed)
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(wupd, "_installer_script", lambda: tmp_path / "install_whisper_prebuilt.py")
    monkeypatch.setattr(wfresh, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: latest)
    return install_dir


def _patch_whisper_phase(
    monkeypatch,
    events,
    *,
    to_tag = "v1.9.2-unsloth.1",
    error = None,
):
    """Record whisper phase runs without touching a real installer."""

    def _run(phase, set_progress):
        events.append("whisper")
        if error is not None:
            raise RuntimeError(error)
        set_progress(0.5)
        return {
            "to_tag": to_tag,
            "reload_required": False,
            "message": f"Updated whisper.cpp to {to_tag}.",
        }

    monkeypatch.setattr(wupd, "run_chained_phase", _run)


def _wait_for_job():
    deadline = time.time() + 10
    while time.time() < deadline:
        with upd._job_lock:
            job = dict(upd._job)
        if job["state"] in ("success", "error"):
            return job
        time.sleep(0.05)
    with upd._job_lock:
        return dict(upd._job)


# --- status: the single item folds whisper in ---


def test_status_payload_is_exact_superset_of_legacy_fields(monkeypatch, tmp_path):
    _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path)
    st = upd.get_update_status(force_refresh = True)
    assert LEGACY_STATUS_FIELDS <= set(st)
    assert LEGACY_JOB_FIELDS <= set(st["job"])
    # The new fields ride alongside, never replacing the legacy ones.
    assert st["llama_update_available"] is True
    assert st["whisper"]["update_available"] is True
    assert st["whisper"]["latest_tag"] == "v1.9.2-unsloth.1"
    assert st["update_component"] == "llama"


def test_status_union_whisper_only_surfaces_update(monkeypatch, tmp_path):
    # llama current, whisper behind: the single item still shows an update.
    _setup_llama(monkeypatch, tmp_path, installed = "b9518", latest = "b9518")
    _setup_whisper(monkeypatch, tmp_path)
    st = upd.get_update_status(force_refresh = True)
    assert st["llama_update_available"] is False
    assert st["whisper"]["update_available"] is True
    assert st["update_available"] is True
    assert st["update_component"] == "whisper"
    assert st["installed_tag"] == "b9518"
    assert st["latest_tag"] == "b9518"
    assert st["whisper"]["installed_tag"] == "v1.9.1-unsloth.1"
    assert st["whisper"]["latest_tag"] == "v1.9.2-unsloth.1"


def test_status_whisper_current_does_not_flip_union(monkeypatch, tmp_path):
    _setup_llama(monkeypatch, tmp_path, installed = "b9518", latest = "b9518")
    _setup_whisper(monkeypatch, tmp_path, installed = "v1.9.2-unsloth.1", latest = "v1.9.2-unsloth.1")
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is False
    assert st["whisper"]["skip_reason"] == "up_to_date"
    assert st["update_component"] is None


def test_status_survives_whisper_probe_failure(monkeypatch, tmp_path):
    # The piggyback fails open: llama status still works without a whisper probe.
    _setup_llama(monkeypatch, tmp_path)

    def _boom(*, force_refresh = False):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(wupd, "chained_phase_plan", _boom)
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is True
    assert st["whisper"] is None


# --- whisper chained_phase_plan: silent skips ---


def test_whisper_plan_skips_local_link(monkeypatch, tmp_path):
    monkeypatch.setattr(wupd, "_find_binary", lambda: str(tmp_path / "whisper-server"))
    monkeypatch.setattr(wupd, "_active_install_is_local_link", lambda b: True)
    plan = wupd.chained_phase_plan()
    assert plan["update_available"] is False
    assert plan["skip_reason"] == "local_link"
    assert plan["phase"] is None


def test_whisper_plan_skips_source_build(monkeypatch, tmp_path):
    binary = tmp_path / "whisper.cpp" / "build" / "bin" / "whisper-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # no marker
    monkeypatch.setattr(wupd, "_find_binary", lambda: str(binary))
    plan = wupd.chained_phase_plan()
    assert plan["skip_reason"] == "source_build"
    assert plan["phase"] is None


def test_whisper_update_targets_canonical_root_when_inner_marker_exists(tmp_path):
    install_dir = tmp_path / "whisper.cpp"
    binary = install_dir / "build" / "bin" / "whisper-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    (install_dir / WHISPER_MARKER).write_text("{}")
    (binary.parent / WHISPER_MARKER).write_text("{}")
    assert wupd._install_dir_for(str(binary)) == install_dir


def test_whisper_plan_skips_when_not_installed(monkeypatch):
    monkeypatch.setattr(wupd, "_find_binary", lambda: None)
    plan = wupd.chained_phase_plan()
    assert plan["skip_reason"] == "not_installed"
    assert plan["phase"] is None


def test_whisper_plan_eligible_when_behind(monkeypatch, tmp_path):
    install_dir = _setup_whisper(monkeypatch, tmp_path)
    script = tmp_path / "install_whisper_prebuilt.py"
    script.write_text("stub")
    plan = wupd.chained_phase_plan(force_refresh = True)
    assert plan["update_available"] is True
    assert plan["skip_reason"] is None
    assert plan["phase"]["install_dir"] == install_dir
    assert plan["phase"]["repo"] == "unslothai/whisper.cpp"
    assert plan["phase"]["backend"] == "cpu"
    # Pin to the exact release the freshness check offered: unpinned, the
    # installer's download-host /releases/latest pointer can lag published_at
    # and reinstall an older build in a loop.
    assert plan["phase"]["pin_release_tag"] == "v1.9.2-unsloth.1"


def test_whisper_phase_pins_installer_to_checked_release(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        wupd._flow,
        "stream_installer",
        lambda cmd, env, **kw: calls.append(cmd),
    )
    monkeypatch.setattr(wupd, "reset_caches", lambda **kw: None)
    monkeypatch.setattr(wupd, "latest_published_release", lambda repo, **kw: "v9")
    install_dir = tmp_path / "whisper.cpp"
    binary = _write_whisper_install(install_dir, "v9")
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    wupd.run_chained_phase(
        {
            "install_dir": install_dir,
            "repo": "unslothai/whisper.cpp",
            "asset": None,
            "backend": "cpu",
            "script": tmp_path / "install_whisper_prebuilt.py",
            "pin_release_tag": "v9",
        },
        lambda f: None,
    )
    cmd = calls[0]
    assert "--published-release-tag" in cmd
    assert cmd[cmd.index("--published-release-tag") + 1] == "v9"


def test_whisper_phase_exit_2_is_a_failed_phase(monkeypatch, tmp_path):
    # No install occurred, so incompatibility must remain an actionable job
    # error instead of producing a false success toast and hiding the banner.
    def _raise_exit_2(cmd, env, **kw):
        raise wupd._flow.InstallerExit(2, "installer exited 2: incompatible release")

    monkeypatch.setattr(wupd._flow, "stream_installer", _raise_exit_2)
    install_dir = tmp_path / "whisper.cpp"
    binary = _write_whisper_install(install_dir, "v1")
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    with pytest.raises(wupd._flow.InstallerExit) as exc_info:
        wupd.run_chained_phase(
            {
                "install_dir": install_dir,
                "repo": "unslothai/whisper.cpp",
                "asset": None,
                "backend": "cpu",
                "script": tmp_path / "install_whisper_prebuilt.py",
                "pin_release_tag": None,
            },
            lambda f: None,
        )
    assert exc_info.value.returncode == 2


def test_llama_update_survives_unavailable_whisper_module(monkeypatch, tmp_path):
    import builtins

    llama_dir = _setup_llama(monkeypatch, tmp_path)
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kw: None)
    _patch_llama_installer(
        monkeypatch,
        on_start = lambda cmd: _write_llama_install(llama_dir, "b9518"),
    )
    real_import = builtins.__import__

    def guarded_import(
        name,
        globals = None,
        locals = None,
        fromlist = (),
        level = 0,
    ):
        if name == "utils" and "whisper_cpp_update" in fromlist:
            raise AssertionError("whisper module was re-imported after its failed probe")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    # A failed optional whisper probe must not be followed by an unconditional
    # import. The valid llama phase still starts and completes.
    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert job["phases"]["llama"]["state"] == "success"
    assert job["phases"]["whisper"]["state"] == "skipped"
    assert job["phases"]["whisper"]["reason"] == "unavailable"


def test_macos_status_uses_compatible_resolver_release(monkeypatch, tmp_path):
    _setup_whisper(
        monkeypatch,
        tmp_path,
        installed = "v1.9.1-unsloth.1",
        latest = "v1.9.2-unsloth.1",
    )
    monkeypatch.setattr(wupd.sys, "platform", "darwin")
    monkeypatch.setattr(
        wupd,
        "_resolve_prebuilt_for_host",
        lambda **kw: {
            "prebuilt_available": True,
            "release_tag": "v1.9.1-unsloth.1",
        },
    )

    status = wupd.get_update_status(force_refresh = True)
    assert status["latest_tag"] == "v1.9.1-unsloth.1"
    assert status["update_available"] is False
    assert status["stale"] is False


def test_whisper_phase_integrity_failure_is_not_swallowed(monkeypatch, tmp_path):
    def _raise_exit_1(cmd, env, **kw):
        raise wupd._flow.InstallerExit(1, "installer exited 1: checksum mismatch")

    monkeypatch.setattr(wupd._flow, "stream_installer", _raise_exit_1)
    install_dir = tmp_path / "whisper.cpp"
    binary = _write_whisper_install(install_dir, "v1")
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    with pytest.raises(wupd._flow.InstallerExit, match = "checksum mismatch"):
        wupd.run_chained_phase(
            {
                "install_dir": install_dir,
                "repo": "unslothai/whisper.cpp",
                "asset": None,
                "backend": "cpu",
                "script": tmp_path / "install_whisper_prebuilt.py",
                "pin_release_tag": None,
            },
            lambda f: None,
        )


# --- apply: the chained job ---


def test_apply_runs_llama_then_whisper(monkeypatch, tmp_path):
    llama_dir = _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path)
    (tmp_path / "install_whisper_prebuilt.py").write_text("stub")

    events = []
    _patch_llama_installer(
        monkeypatch,
        on_start = lambda cmd: (events.append("llama"), _write_llama_install(llama_dir, "b9518")),
    )
    _patch_whisper_phase(monkeypatch, events)

    res = upd.start_update()
    assert res["started"] is True, res
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert events == ["llama", "whisper"]  # llama phase strictly first
    assert job["phases"]["llama"]["state"] == "success"
    assert job["phases"]["llama"]["to_tag"] == "b9518"
    assert job["phases"]["whisper"]["state"] == "success"
    assert job["phases"]["whisper"]["to_tag"] == "v1.9.2-unsloth.1"
    # Legacy top-level fields keep their llama meaning.
    assert job["from_tag"] == "b9493"
    assert job["to_tag"] == "b9518"
    assert "Updated llama.cpp to b9518." in job["message"]
    assert "Updated whisper.cpp to v1.9.2-unsloth.1." in job["message"]
    assert job["progress"] == 1.0
    assert LEGACY_JOB_FIELDS <= set(job)


def test_apply_llama_only_when_whisper_current(monkeypatch, tmp_path):
    llama_dir = _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path, installed = "v1.9.2-unsloth.1", latest = "v1.9.2-unsloth.1")

    events = []
    _patch_llama_installer(
        monkeypatch,
        on_start = lambda cmd: (events.append("llama"), _write_llama_install(llama_dir, "b9518")),
    )
    _patch_whisper_phase(monkeypatch, events)

    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert events == ["llama"]
    assert job["phases"]["whisper"]["state"] == "skipped"
    assert job["phases"]["whisper"]["reason"] == "up_to_date"


def test_apply_whisper_only_noops_llama(monkeypatch, tmp_path):
    # llama current + whisper behind: the same single apply runs, with the llama
    # phase a cheap already-matches no-op and the whisper phase doing the work.
    _setup_llama(monkeypatch, tmp_path, installed = "b9518", latest = "b9518")
    _setup_whisper(monkeypatch, tmp_path)
    (tmp_path / "install_whisper_prebuilt.py").write_text("stub")

    events = []
    _patch_llama_installer(monkeypatch, on_start = lambda cmd: events.append("llama"))
    _patch_whisper_phase(monkeypatch, events)

    res = upd.start_update()
    assert res["started"] is True, res
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert events == ["whisper"]  # the llama installer never ran
    # The legacy job-level to_tag means "llama tag"; a whisper-only round
    # leaves it unset so the UI never reports a llama update that never ran.
    assert job["to_tag"] is None
    assert job["phases"]["llama"]["state"] == "skipped"
    assert job["phases"]["llama"]["reason"] == "up_to_date"
    assert job["phases"]["whisper"]["state"] == "success"
    assert "Updated whisper.cpp to v1.9.2-unsloth.1." in job["message"]


def test_whisper_reload_never_raises_job_reload_flag(monkeypatch, tmp_path):
    # A whisper-only update that had to unload a warm sidecar reports
    # reload_required on its phase, but the JOB flag stays down: the chat
    # frontend resyncs (and clears the local checkpoint) off the job flag,
    # which must mean "the llama server changed", not "the sidecar restarted".
    _setup_llama(monkeypatch, tmp_path, installed = "b9518", latest = "b9518")
    _setup_whisper(monkeypatch, tmp_path)
    (tmp_path / "install_whisper_prebuilt.py").write_text("stub")

    def _whisper_phase(phase, set_progress):
        return {
            "to_tag": "v1.9.2-unsloth.1",
            "reload_required": True,
            "message": "Updated whisper.cpp to v1.9.2-unsloth.1.",
        }

    monkeypatch.setattr(wupd, "run_chained_phase", _whisper_phase)
    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert job["phases"]["whisper"]["reload_required"] is True
    assert not job["reload_required"]


def test_apply_refuses_when_both_current(monkeypatch, tmp_path):
    _setup_llama(monkeypatch, tmp_path, installed = "b9518", latest = "b9518")
    _setup_whisper(monkeypatch, tmp_path, installed = "v1.9.2-unsloth.1", latest = "v1.9.2-unsloth.1")
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "up_to_date"


def test_apply_llama_failure_aborts_before_whisper(monkeypatch, tmp_path):
    _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path)
    (tmp_path / "install_whisper_prebuilt.py").write_text("stub")

    events = []
    _patch_llama_installer(monkeypatch, returncode = 2, lines = ["boom: disk full\n"])
    _patch_whisper_phase(monkeypatch, events)

    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "error", job
    assert "boom" in (job["error"] or "")
    assert events == []  # whisper never attempted
    assert job["phases"]["llama"]["state"] == "error"
    assert job["phases"]["whisper"]["state"] == "skipped"
    assert job["phases"]["whisper"]["reason"] == "aborted"
    assert job["message"] == "llama.cpp update failed."


def test_apply_whisper_failure_keeps_llama_partial_success(monkeypatch, tmp_path):
    llama_dir = _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path)
    (tmp_path / "install_whisper_prebuilt.py").write_text("stub")

    # An active model makes the llama phase report reload_required.
    import threading
    from types import ModuleType

    class _FakeBackend:
        def __init__(self):
            self._serial_load_lock = threading.Lock()
            self._llama_update_in_progress = False
            self.is_active = True

        def unload_model(self):
            self.is_active = False

    backend = _FakeBackend()
    routes_pkg = ModuleType("routes")
    routes_pkg.__path__ = []
    inference_mod = ModuleType("routes.inference")
    inference_mod.get_llama_cpp_backend = lambda: backend
    monkeypatch.setitem(sys.modules, "routes", routes_pkg)
    monkeypatch.setitem(sys.modules, "routes.inference", inference_mod)

    events = []
    _patch_llama_installer(
        monkeypatch,
        on_start = lambda cmd: (events.append("llama"), _write_llama_install(llama_dir, "b9518")),
    )
    _patch_whisper_phase(monkeypatch, events, error = "whisper installer exploded")

    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "error", job
    assert events == ["llama", "whisper"]
    # The message says both halves: llama landed, whisper did not.
    assert "Updated llama.cpp to b9518." in job["message"]
    assert "whisper.cpp update failed." in job["message"]
    assert "whisper installer exploded" in (job["error"] or "")
    # The llama phase's reload_required survives the whisper failure.
    assert job["reload_required"] is True
    assert job["phases"]["llama"]["state"] == "success"
    assert job["phases"]["whisper"]["state"] == "error"


def test_apply_skips_whisper_local_link_silently(monkeypatch, tmp_path):
    llama_dir = _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path)
    monkeypatch.setattr(wupd, "_active_install_is_local_link", lambda b: True)

    events = []
    _patch_llama_installer(
        monkeypatch,
        on_start = lambda cmd: (events.append("llama"), _write_llama_install(llama_dir, "b9518")),
    )
    _patch_whisper_phase(monkeypatch, events)

    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert events == ["llama"]
    assert job["phases"]["whisper"]["state"] == "skipped"
    assert job["phases"]["whisper"]["reason"] == "local_link"
    assert job["message"] == "Updated llama.cpp to b9518."


def test_chained_progress_windows(monkeypatch, tmp_path):
    # The llama phase fills roughly the first 0.7 slice and whisper the rest.
    llama_dir = _setup_llama(monkeypatch, tmp_path)
    _setup_whisper(monkeypatch, tmp_path)
    (tmp_path / "install_whisper_prebuilt.py").write_text("stub")

    seen = {}

    def _whisper_phase(phase, set_progress):
        with upd._job_lock:
            seen["at_whisper_start"] = upd._job["progress"]
        set_progress(0.5)
        with upd._job_lock:
            seen["mid_whisper"] = upd._job["progress"]
        return {"to_tag": "v1.9.2-unsloth.1", "reload_required": False, "message": "ok"}

    monkeypatch.setattr(wupd, "run_chained_phase", _whisper_phase)
    _patch_llama_installer(
        monkeypatch,
        lines = ["Downloading app.tar.gz: 100.0% (35.0 MiB/35.0 MiB) at 9.0 MiB/s\n"],
        on_start = lambda cmd: _write_llama_install(llama_dir, "b9518"),
    )

    assert upd.start_update()["started"] is True
    job = _wait_for_job()
    assert job["state"] == "success", job
    assert seen["at_whisper_start"] == pytest.approx(0.7)
    assert seen["mid_whisper"] == pytest.approx(0.7 + 0.5 * 0.3)
    assert job["progress"] == 1.0

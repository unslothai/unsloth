# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import threading
import time

import pytest

from auth import storage as auth_storage
from core import colab_launcher as cl
from core.inference import linked_instances
from storage import credential_secrets, linked_instances_db

_REAL_DETECT = cl.detect_runner
REMOTE_KEY = "sk-unsloth-" + "b" * 32
TUNNEL = "https://quiet-otter.trycloudflare.com"
READY_PROBE = "UNSLOTH_PROBE uv=yes\nUNSLOTH_PROBE cli=/root/.local/bin/colab\nUNSLOTH_PROBE auth=oauth2\nUNSLOTH_PROBE sessions_rc=0\n[colab] No active sessions found on server."


@pytest.fixture(autouse = True)
def isolated_databases(tmp_path, monkeypatch):
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    for module in (credential_secrets, linked_instances_db):
        monkeypatch.setattr(module, "studio_db_path", lambda: studio_db)
        monkeypatch.setattr(module, "ensure_dir", lambda p: p.mkdir(parents = True, exist_ok = True))
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    monkeypatch.setattr(linked_instances, "_catalog_cache", {})
    monkeypatch.setattr(cl, "detect_runner", lambda: cl.Runner("native"))
    monkeypatch.setattr(cl, "_job", None)
    yield studio_db


class FakeColab:
    """Stands in for bash + the colab CLI; answers by what the script asks for."""

    def __init__(
        self,
        *,
        probe = READY_PROBE,
        new = (0, "[colab] Session READY."),
        install = None,
        start = None,
    ):
        self.probe = probe
        self.new = new
        self.install = install if install is not None else (0, "Installing...\n" + cl.INSTALL_OK)
        self.start = (
            start
            if start is not None
            else (
                0,
                "booting\n" + cl.LINK_MARKER + json.dumps({"url": TUNNEL, "api_key": REMOTE_KEY}),
            )
        )
        self.calls: list[tuple[str, str]] = []
        self.stop_rc = (0, "[colab] Session terminated.")
        self.status_rc = (0, "[colab] Session 'x' not found.")
        self.on_install = None

    def __call__(
        self,
        runner,
        script,
        *,
        stdin = None,
        timeout,
        on_line = None,
        cancel = None,
    ):
        if script == cl.PROBE_SCRIPT:
            kind, (rc, out) = "probe", (0, self.probe)
        elif " new " in script:
            kind, (rc, out) = "new", self.new
        elif " stop " in script:
            kind, (rc, out) = "stop", self.stop_rc
        elif " status " in script:
            kind, (rc, out) = "status", self.status_rc
        elif stdin and cl.INSTALL_OK.split()[0] in stdin and "setup.sh" in stdin:
            kind, (rc, out) = "install", self.install
            if self.on_install:
                self.on_install(cancel)
        elif stdin and "UNSLOTH_LINK" in stdin:
            kind, (rc, out) = "start", self.start
        else:
            raise AssertionError(f"unexpected script: {script[:80]}")
        self.calls.append((kind, script))
        for line in out.splitlines():
            if on_line:
                on_line(line)
        return cl.ShellResult(rc, out)

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


def _job(name = "colab-l4", gpu = "L4") -> cl.LaunchJob:
    return cl.LaunchJob(id = "j", name = name, gpu = gpu, session = f"unsloth-{name}")


def test_parse_link_reads_the_marker_line():
    out = "noise\n" + cl.LINK_MARKER + json.dumps({"url": TUNNEL, "api_key": REMOTE_KEY}) + "\nmore"
    assert cl.parse_link(out) == {"url": TUNNEL, "api_key": REMOTE_KEY}


@pytest.mark.parametrize(
    "out, needle",
    [
        ("no marker here", "never reported"),
        (cl.LINK_MARKER + "{not json", "never reported"),
        (
            cl.LINK_MARKER + json.dumps({"error": "RuntimeError: tunnel " + REMOTE_KEY}),
            "did not start",
        ),
        (
            cl.LINK_MARKER + json.dumps({"url": "http://x.trycloudflare.com", "api_key": "k"}),
            "incomplete",
        ),
        (cl.LINK_MARKER + json.dumps({"url": TUNNEL}), "incomplete"),
    ],
)
def test_parse_link_failures(out, needle):
    with pytest.raises(cl.LaunchError) as err:
        cl.parse_link(out)
    assert needle in str(err.value)
    assert REMOTE_KEY not in str(err.value)


def test_status_not_found_exits_zero_but_is_not_alive():
    assert not cl.session_alive("[colab] Session 'unsloth-colab-l4' not found.", "unsloth-colab-l4")
    assert cl.session_alive(
        "[unsloth-colab-l4] https://x.colab.dev | Hardware: L4 | Shape: STANDARD | Variant: GPU",
        "unsloth-colab-l4",
    )


@pytest.mark.parametrize(
    "probe, state, setup_has",
    [
        (
            "UNSLOTH_PROBE cli=missing",
            "missing_cli",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        ),
        (
            "UNSLOTH_PROBE uv=yes\nUNSLOTH_PROBE cli=/c\nUNSLOTH_PROBE auth=none",
            "signed_out",
            "colab --auth oauth2 sessions",
        ),
        (
            "UNSLOTH_PROBE cli=/c\nUNSLOTH_PROBE auth=oauth2\nUNSLOTH_PROBE sessions_rc=1\nAborted.",
            "signed_out",
            "colab --auth oauth2 sessions",
        ),
        (
            "UNSLOTH_PROBE cli=/c\nUNSLOTH_PROBE kernel_client=bad",
            "kernel_client",
            "jupyter-kernel-client",
        ),
        ("", "unsupported", "wsl --install -d Ubuntu-24.04"),
        (READY_PROBE, "ready", None),
    ],
)
def test_capability_states(monkeypatch, probe, state, setup_has):
    monkeypatch.setattr(cl, "run_shell", FakeColab(probe = probe))
    cap = cl.capability(cl.Runner("wsl", "Ubuntu-24.04"))
    assert cap["state"] == state
    assert cap["ready"] is (state == "ready")
    assert cap["runner"] == "wsl" and cap["distro"] == "Ubuntu-24.04"
    if setup_has:
        assert any(setup_has in c for c in cap["setup"])
    else:
        assert cap["setup"] == [] and cap["auth"] == "oauth2"


def test_capability_with_uv_skips_the_uv_installer(monkeypatch):
    monkeypatch.setattr(
        cl, "run_shell", FakeColab(probe = "UNSLOTH_PROBE uv=yes\nUNSLOTH_PROBE cli=missing")
    )
    assert cl.capability()["setup"] == [
        "uv tool install google-colab-cli",
        "colab --auth oauth2 sessions",
    ]


def test_no_wsl_on_windows_is_unsupported(monkeypatch):
    monkeypatch.setattr(cl.sys, "platform", "win32")
    monkeypatch.setattr(cl.shutil, "which", lambda name: None)
    monkeypatch.setattr(cl, "detect_runner", _REAL_DETECT)
    assert cl.detect_runner() is None
    cap = cl.capability()
    assert cap["state"] == "unsupported" and cap["setup"][0].startswith("wsl --install")


def test_colab_flags_are_global_and_auth_is_passed():
    script = cl._colab("unsloth-a", "oauth2", "exec", "-s", "unsloth-a", "-")
    colab_part = script.split("&& ", 1)[1]
    assert colab_part.startswith(
        "colab --config ~/.config/unsloth-studio/colab/unsloth-a.json --auth oauth2 exec"
    )
    assert "--env" not in script


def test_launch_links_the_vm_and_never_exposes_the_key(monkeypatch):
    fake = FakeColab()
    stages = []
    real = fake.__call__

    def spy(runner, script, **kw):
        stages.append(job.stage)
        return real(runner, script, **kw)

    monkeypatch.setattr(cl, "run_shell", spy)
    job = _job()
    cl._run_job(job)

    assert job.state == "ready" and job.stage == "ready", job.error
    assert fake.kinds() == ["probe", "new", "install", "start"]
    assert stages == ["allocating", "allocating", "installing", "starting"]
    new_script = fake.calls[1][1]
    assert "new -s unsloth-colab-l4 --gpu L4" in new_script
    start_script = fake.calls[3][1]
    assert "rm -f ~/.config/colab-cli/history/unsloth-colab-l4.jsonl" in start_script

    instance = linked_instances_db.get_instance_by_name("colab-l4")
    assert instance["base_url"] == TUNNEL and job.instance_id == instance["id"]
    assert linked_instances_db.get_api_key(instance["id"]) == REMOTE_KEY
    record = linked_instances_db.get_colab_session("unsloth-colab-l4")
    assert (
        record["instance_id"] == instance["id"]
        and record["gpu"] == "L4"
        and record["auth"] == "oauth2"
    )

    public = json.dumps(job.public())
    assert REMOTE_KEY not in public and "sk-unsloth-" not in public
    assert not any(line.startswith("UNSLOTH_LINK") for line in job.log)
    assert "api_key" not in json.dumps(cl.list_sessions())


def test_a_key_in_ordinary_output_is_redacted(monkeypatch):
    fake = FakeColab(install = (0, f"echo {REMOTE_KEY}\n" + cl.INSTALL_OK))
    monkeypatch.setattr(cl, "run_shell", fake)
    job = _job()
    cl._run_job(job)
    assert REMOTE_KEY not in json.dumps(job.public())


def test_install_that_exits_zero_without_the_marker_fails_and_stops_the_vm(monkeypatch):
    fake = FakeColab(
        install = (0, "Traceback (most recent call last):\nCalledProcessError: git clone")
    )
    monkeypatch.setattr(cl, "run_shell", fake)
    job = _job()
    cl._run_job(job)
    assert job.state == "failed" and "Installing Unsloth Studio" in job.error
    assert "git clone" in job.error
    assert fake.kinds() == ["probe", "new", "install", "stop"]
    assert linked_instances_db.get_colab_session(job.session) is None
    assert linked_instances_db.list_instances() == []


def test_capacity_error_names_the_gpu_and_stops(monkeypatch):
    fake = FakeColab(new = (1, "HTTP 412 Precondition Failed: TooManyAssignments"))
    monkeypatch.setattr(cl, "run_shell", fake)
    job = _job(gpu = "H100", name = "colab-h100")
    cl._run_job(job)
    assert job.state == "failed" and "no H100" in job.error
    assert fake.kinds()[-1] == "stop"


def test_start_error_is_reported_and_stops(monkeypatch):
    fake = FakeColab(
        start = (
            0,
            cl.LINK_MARKER
            + json.dumps({"error": "RuntimeError: the Cloudflare tunnel did not produce a URL"}),
        )
    )
    monkeypatch.setattr(cl, "run_shell", fake)
    job = _job()
    cl._run_job(job)
    assert job.state == "failed" and "Cloudflare tunnel" in job.error
    assert fake.kinds()[-1] == "stop" and linked_instances_db.list_instances() == []


def test_not_signed_in_fails_before_allocating(monkeypatch):
    fake = FakeColab(probe = "UNSLOTH_PROBE cli=/c\nUNSLOTH_PROBE auth=none")
    monkeypatch.setattr(cl, "run_shell", fake)
    job = _job()
    cl._run_job(job)
    assert job.state == "failed" and "not signed in" in job.error
    assert job.setup == ["colab --auth oauth2 sessions"]
    assert fake.kinds() == ["probe"]
    assert linked_instances_db.list_colab_sessions() == []


def test_cancel_during_install_stops_the_vm(monkeypatch):
    fake = FakeColab()
    job = _job()
    fake.on_install = lambda cancel: job.cancel.set()
    monkeypatch.setattr(cl, "run_shell", fake)
    cl._run_job(job)
    assert job.state == "cancelled"
    assert fake.kinds() == ["probe", "new", "install", "stop"]
    assert linked_instances_db.list_colab_sessions() == []


def test_stop_removes_the_link_and_the_record(monkeypatch):
    fake = FakeColab()
    monkeypatch.setattr(cl, "run_shell", fake)
    cl._run_job(_job())
    cl.stop_session("unsloth-colab-l4")
    assert fake.kinds()[-1] == "stop"
    assert linked_instances_db.list_instances() == [] and cl.list_sessions() == []


def test_a_failed_stop_keeps_the_record_only_while_status_shows_the_vm(monkeypatch):
    fake = FakeColab()
    monkeypatch.setattr(cl, "run_shell", fake)
    cl._run_job(_job())

    fake.stop_rc = (1, "ConnectionError: network down")
    fake.status_rc = (
        0,
        "[unsloth-colab-l4] https://x.colab.dev | Hardware: L4 | Shape: STANDARD | Variant: GPU",
    )
    with pytest.raises(cl.LaunchError):
        cl.stop_session("unsloth-colab-l4")
    assert len(cl.list_sessions()) == 1 and len(linked_instances_db.list_instances()) == 1

    # `colab status` exits 0 for a missing session; the message, not the code, says it is gone.
    fake.status_rc = (0, "[colab] Session 'unsloth-colab-l4' not found.")
    cl.stop_session("unsloth-colab-l4")
    assert fake.kinds()[-2:] == ["stop", "status"]
    assert cl.list_sessions() == [] and linked_instances_db.list_instances() == []

    with pytest.raises(KeyError):
        cl.stop_session("unsloth-colab-l4")


def test_stop_that_reports_not_found_is_already_stopped(monkeypatch):
    fake = FakeColab()
    monkeypatch.setattr(cl, "run_shell", fake)
    cl._run_job(_job())
    fake.stop_rc = (1, "[colab] Session 'unsloth-colab-l4' not found.")
    cl.stop_session("unsloth-colab-l4")
    assert fake.kinds()[-1] == "stop" and cl.list_sessions() == []


def test_start_launch_validates_before_spawning(monkeypatch):
    fake = FakeColab()
    monkeypatch.setattr(cl, "run_shell", fake)
    with pytest.raises(ValueError):
        cl.start_launch("V100", "colab-x")
    with pytest.raises(ValueError):
        cl.start_launch("T4", "Bad/Name")
    linked_instances_db.create_instance("taken", "http://127.0.0.1:1", "k")
    with pytest.raises(ValueError):
        cl.start_launch("T4", "taken")
    assert fake.calls == []


def test_start_launch_runs_in_the_background_and_allows_one_at_a_time(monkeypatch):
    gate = threading.Event()
    fake = FakeColab()
    fake.on_install = lambda cancel: gate.wait(5)
    monkeypatch.setattr(cl, "run_shell", fake)
    job = cl.start_launch("t4", "colab-t4")
    assert job["gpu"] == "T4" and job["state"] == "running"
    with pytest.raises(ValueError):
        cl.start_launch("L4", "colab-other")
    gate.set()
    for _ in range(100):
        if cl.current_job()["state"] != "running":
            break
        time.sleep(0.05)
    assert cl.current_job()["state"] == "ready"


def test_vm_scripts_are_valid_python():
    compile(cl.INSTALL_SCRIPT.format(repo = cl.REPO_URL, branch = cl.REPO_BRANCH), "install", "exec")
    start = cl.START_SCRIPT.format(launcher = cl.LAUNCHER_SCRIPT)
    compile(start, "start", "exec")
    compile(cl.LAUNCHER_SCRIPT, "launcher", "exec")
    assert "--env" not in start

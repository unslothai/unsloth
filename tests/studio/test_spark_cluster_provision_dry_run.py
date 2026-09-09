# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A provisioning dry run must not change the peer.

`--rsync-path` is a command rsync runs ON THE PEER before transferring anything, so `--dry-run`
does not suppress it. Wrapping it as `mkdir -p "<parent>" && rsync` therefore created the
directories the dry run was only supposed to report. Every rsync argv the dry run builds is
captured here and checked, rather than asserting on a flag, because the defect was that a flag
did not mean what it looked like it meant.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _cluster():
    spec = importlib.util.spec_from_file_location(
        "spark_cluster_for_dry_run", REPO / "studio" / "spark_cluster.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Stands in for subprocess.run, answering `test -d` and recording everything."""

    def __init__(self, *, peer_dirs_exist: bool):
        self.calls: list[list[str]] = []
        self.peer_dirs_exist = peer_dirs_exist

    def __call__(self, cmd, *a, **k):
        self.calls.append(list(cmd))
        code = 0 if (cmd[0] != "ssh" or self.peer_dirs_exist) else 1
        return subprocess.CompletedProcess(cmd, code, stdout = "", stderr = "")

    def rsync_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "rsync"]

    def ssh_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "ssh"]


def _run(
    cluster,
    monkeypatch,
    tmp_path,
    *,
    dry_run: bool,
    peer_dirs_exist: bool = True,
):
    local = tmp_path / "studio"
    local.mkdir(parents = True, exist_ok = True)
    (local / "marker").write_text("x", encoding = "utf-8")

    monkeypatch.setattr(cluster.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cluster, "provision_paths", lambda: [(str(local), "studio")])
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(
        cluster,
        "peer_gpu_busy",
        lambda ip: {"busy": False, "reason": "", "processes": []},
    )
    monkeypatch.setattr(
        cluster,
        "fast_path_decision",
        lambda ip, no_fast = False: {
            "ok": False,
            "reason": "disabled for this test",
            "local_ip": None,
        },
    )
    recorder = _Recorder(peer_dirs_exist = peer_dirs_exist)
    monkeypatch.setattr(cluster.subprocess, "run", recorder)
    results = cluster.provision_peer("192.0.2.7", dry_run = dry_run, no_fast = True)
    return recorder, results


def test_a_dry_run_never_asks_the_peer_to_create_anything(monkeypatch, tmp_path) -> None:
    cluster = _cluster()
    recorder, _ = _run(cluster, monkeypatch, tmp_path, dry_run = True)

    assert recorder.rsync_calls(), "no rsync was attempted, so this proved nothing"
    for cmd in recorder.calls:
        joined = " ".join(cmd)
        assert "mkdir" not in joined, f"a dry run would run: {joined}"
    for cmd in recorder.rsync_calls():
        assert "--dry-run" in cmd, cmd


def test_a_real_run_still_creates_the_parent(monkeypatch, tmp_path) -> None:
    """No regression: rsync creates only the last component, so a brand-new peer needs this."""
    cluster = _cluster()
    recorder, _ = _run(cluster, monkeypatch, tmp_path, dry_run = False)

    rsyncs = recorder.rsync_calls()
    assert rsyncs, "no rsync was attempted"
    for cmd in rsyncs:
        assert "--dry-run" not in cmd, cmd
        path_arg = cmd[cmd.index("--rsync-path") + 1]
        assert path_arg.startswith("mkdir -p "), path_arg


def test_a_missing_peer_directory_is_reported_not_attempted(monkeypatch, tmp_path) -> None:
    """With nothing on the far side there is nothing to compare against, so the dry run says
    what the real run would create instead of failing on a destination that is not there."""
    cluster = _cluster()
    recorder, results = _run(cluster, monkeypatch, tmp_path, dry_run = True, peer_dirs_exist = False)

    assert not recorder.rsync_calls(), "a dry run tried to transfer into a missing directory"
    assert recorder.ssh_calls(), "the check that replaced the mkdir did not run"
    for cmd in recorder.ssh_calls():
        assert "test" in cmd and "-d" in cmd, cmd
    assert any("would create" in reason for _, reason in results["skipped"]), results["skipped"]
    assert not results["failed"], results["failed"]


# ── the GPU probe's fail-closed contract ────────────────────────────────────────
# A row that cannot be read is not an absent process. nvidia-smi reports `[N/A]` for
# used_memory in real situations, and skipping such a row and then declaring the peer idle
# inverts the whole point of the probe: provisioning would overwrite a venv the process behind
# that row is running out of.


def _probe(cluster, monkeypatch, stdout: str):
    monkeypatch.setattr(cluster.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, stdout = stdout, stderr = "")

    monkeypatch.setattr(cluster.subprocess, "run", fake_run)
    return cluster.peer_gpu_busy("192.0.2.7")


@pytest.mark.parametrize(
    "row",
    [
        "4242, [N/A]",  # the documented one: nvidia-smi cannot report the memory
        "4242, ",  # empty memory field
        "4242, not-a-number",
        "[N/A], 900 MiB",  # unreadable pid
    ],
)
def test_an_unreadable_process_row_leaves_the_peer_busy(monkeypatch, row: str) -> None:
    cluster = _cluster()
    out = _probe(cluster, monkeypatch, f"pid, used_memory\n{row}\nRC=0\n")
    assert out["busy"] is True, out
    assert out["known"] is False, out
    assert "could not be read" in out["reason"], out


def test_a_clean_idle_peer_is_still_idle(monkeypatch) -> None:
    """No regression: the whole reason provisioning is allowed to run at all."""
    cluster = _cluster()
    out = _probe(cluster, monkeypatch, "pid, used_memory\nRC=0\n")
    assert out["busy"] is False and out["known"] is True, out


def test_a_readable_busy_peer_is_still_busy(monkeypatch) -> None:
    cluster = _cluster()
    out = _probe(cluster, monkeypatch, "pid, used_memory\n4242, 40000 MiB\nRC=0\n")
    assert out["busy"] is True and out["known"] is True, out
    assert out["processes"] == [{"pid": 4242, "used_mib": 40000}], out


# ── the NCCL probe must not leave a rank running on the peer ────────────────────
# The peer rank is launched detached with setsid/nohup. Every way the local side can end after
# that left it in rendezvous or NCCL timeout handling, still holding the 1 GiB probe buffer, and
# the next provisioning run then correctly refuses the peer as busy over a probe nobody wants.


class _Probe:
    """Records commands and fails whichever stage the test asks it to."""

    def __init__(self, *, fail: str = ""):
        self.calls: list = []
        self.fail = fail

    def __call__(self, cmd, *a, **k):
        self.calls.append(cmd)
        text = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        if self.fail == "launch" and "setsid" in text:
            raise OSError("ssh failed")
        if self.fail == "local" and isinstance(cmd, str) and "node_rank=0" in text:
            raise subprocess.TimeoutExpired(cmd, 5)
        stdout = ""
        if isinstance(cmd, str) and "node_rank=0" in text and self.fail != "silent":
            stdout = "SPARK_NCCL_BUSBW 21.4\n"
        return subprocess.CompletedProcess(cmd, 0, stdout = stdout, stderr = "")

    def cleaned_up(self) -> bool:
        return any(
            "spark_nccl_probe.pid" in (c if isinstance(c, str) else " ".join(map(str, c)))
            and "kill" in (c if isinstance(c, str) else " ".join(map(str, c)))
            for c in self.calls
        )


def _measure(cluster, monkeypatch, tmp_path, probe):
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(cluster, "venv_activate", lambda: str(tmp_path / "activate"))
    monkeypatch.setattr(cluster.subprocess, "run", probe)
    monkeypatch.setattr(cluster.time, "sleep", lambda s: None, raising = False)
    return cluster.nccl_bandwidth("192.0.2.7", "192.0.2.1")


@pytest.mark.parametrize("fail", ["", "local", "silent"])
def test_the_peer_rank_is_stopped_however_the_measurement_ends(
    monkeypatch, tmp_path, fail: str
) -> None:
    """Success, a local timeout, and a run that produced no reading at all."""
    cluster = _cluster()
    probe = _Probe(fail = fail)
    _measure(cluster, monkeypatch, tmp_path, probe)
    assert probe.cleaned_up(), f"fail={fail!r}: the peer rank was left running"


def test_the_stop_uses_the_recorded_pid_not_a_name_match(monkeypatch) -> None:
    """A pattern kill on a shared machine can take out something else that matches."""
    cluster = _cluster()
    sent = []
    import subprocess as sp

    class _R:
        def __call__(self, cmd, *a, **k):
            sent.append(" ".join(str(c) for c in cmd))
            return sp.CompletedProcess(cmd, 0)

    # Through monkeypatch: `cluster.subprocess` IS the stdlib module, so a bare assignment
    # replaced `subprocess.run` for the rest of the session and every later test that shells
    # out silently got a stub that reports success and does nothing.
    monkeypatch.setattr(cluster.subprocess, "run", _R())
    assert cluster.stop_peer_nccl_probe("192.0.2.7", "someuser", []) is True
    joined = " ".join(sent)
    assert "spark_nccl_probe.pid" in joined and "kill -TERM" in joined, joined
    assert "pkill" not in joined and "killall" not in joined, joined


def test_the_launch_records_a_pid_to_stop() -> None:
    """Without it there is nothing to kill and the cleanup is decorative."""
    source = (REPO / "studio" / "spark_cluster.py").read_text(encoding = "utf-8")
    assert "echo $! > " in source and "spark_nccl_probe.pid" in source


# ── which host is the peer ──────────────────────────────────────────────────────
# Setup assigns NODE_BASE_OCTET + node_index, so node 0 is .12 and node 1 is .13. Always adding
# one is right only on node 0: from the second Spark it returned .14, which does not exist, and
# doctor, provisioning, serving and training launched from there all aimed at it.


@pytest.mark.parametrize(
    "local,expected",
    [
        ("192.168.200.12", "192.168.200.13"),  # node 0 looks up
        ("192.168.200.13", "192.168.200.12"),  # node 1 looks DOWN, the case that was wrong
        ("10.0.5.12", "10.0.5.13"),  # the subnet is not assumed
        ("10.0.5.13", "10.0.5.12"),
    ],
)
def test_the_peer_is_the_other_endpoint_whichever_end_this_is(local: str, expected: str) -> None:
    cluster = _cluster()
    assert cluster.peer_address_of(local) == expected


@pytest.mark.parametrize("bad", ["192.168.200.11", "192.168.200.x", "notanaddress", ""])
def test_an_address_that_is_not_a_rail_endpoint_has_no_peer(bad: str) -> None:
    """Below the base octet or unparseable is not a rail endpoint, so guessing would be worse
    than saying nothing: callers treat None as 'no peer configured'."""
    cluster = _cluster()
    assert cluster.peer_address_of(bad) is None


def test_peer_ip_for_uses_the_same_rule(monkeypatch) -> None:
    cluster = _cluster()
    rails = [{"ipv4": ["192.168.200.13"]}]
    assert cluster.peer_ip_for(rails) == "192.168.200.12"


def test_the_status_path_does_not_keep_its_own_copy_of_the_rule() -> None:
    """It had the identical off-by-one, so it must not re-derive the address itself."""
    source = (REPO / "studio" / "spark_cluster.py").read_text(encoding = "utf-8")
    assert "int(octets[1]) + 1" not in source, "the duplicated increment is still there"
    assert source.count("peer_address_of(addr)") >= 2, "the status path does not use the rule"


# ── plaintext bulk transfer belongs on a point-to-point cable only ──────────────
# The fast path's own SECURITY note says the unencrypted rsync daemon is acceptable ONLY
# because the rail is a cable with no other host on it. After `setup --nodes N --switched`
# every node shares these subnets, so the same-/24 test still passed while the fabric carried
# other hosts. `hosts allow` and a one-shot secret restrict access; neither is confidentiality.


def _fast(cluster, monkeypatch, config: dict):
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    monkeypatch.setattr(cluster.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cluster, "load_config", lambda: config)
    return cluster.fast_path_decision("192.168.200.13", env = {}, local_ip = "192.168.200.12")


def test_a_two_node_direct_rail_still_uses_the_fast_path(monkeypatch) -> None:
    """No regression: this is the configuration the fast path was measured on."""
    cluster = _cluster()
    decision = _fast(cluster, monkeypatch, {"n_nodes": 2, "switched": False})
    assert decision["ok"] is True and decision["reason"] == "direct rail", decision


@pytest.mark.parametrize(
    "config",
    [
        {"n_nodes": 3, "switched": True},
        {"n_nodes": 2, "switched": True},  # switched even at two nodes is a shared fabric
        {"n_nodes": 4, "switched": False},  # more than a pair cannot be point-to-point
    ],
)
def test_a_shared_fabric_falls_back_to_ssh(monkeypatch, config: dict) -> None:
    cluster = _cluster()
    decision = _fast(cluster, monkeypatch, config)
    assert decision["ok"] is False, decision
    assert "plaintext" in decision["reason"], decision


def test_an_unconfigured_cluster_is_not_blocked(monkeypatch) -> None:
    """An absent or unreadable config must not disable the fast path on a real pair; the
    switched plan is written by setup, so its absence means nobody asked for one."""
    cluster = _cluster()
    assert _fast(cluster, monkeypatch, {})["ok"] is True
    assert _fast(cluster, monkeypatch, {"n_nodes": "not a number"})["ok"] is True

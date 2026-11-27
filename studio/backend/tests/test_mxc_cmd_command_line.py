# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace

import pytest

from core.inference import mxc_policy, mxc_probe, mxc_runtime, os_sandbox


@pytest.fixture(autouse = True)
def _dacl_journal_outside_grants(monkeypatch, tmp_path):
    outside = Path(os.path.abspath(os.sep)) / "unsloth-test-dacl-journal-never-created"
    monkeypatch.setattr(mxc_runtime, "dacl_state_path", lambda: outside)
    monkeypatch.setattr(mxc_runtime, "dacl_state_dir", lambda: tmp_path)


@pytest.fixture
def system_cmd(monkeypatch, tmp_path):
    cmd = tmp_path / "Windows" / "System32" / "cmd.exe"
    cmd.parent.mkdir(parents = True)
    cmd.touch()
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    monkeypatch.setattr(mxc_policy, "_system_cmd", lambda: str(cmd))
    return str(cmd)


def _workdir(tmp_path):
    workdir = tmp_path / "session"
    workdir.mkdir()
    return workdir


def _cmd_plan(
    workdir,
    payload,
    argv0 = "cmd",
):
    return os_sandbox.ToolLaunchPlan(
        argv = (argv0, "/c", payload),
        workdir = str(workdir),
        env = {"PATH": ""},
        execution_kind = "terminal",
    )


@pytest.mark.parametrize(
    "argv, expected",
    [
        (("cmd", "/c", "dir"), True),
        (("CMD.EXE", "/C", "dir"), True),
        (("C:\\Windows\\System32\\cmd.exe", "/d", "/s", "/c", "dir"), True),
        (("cmd", "/s", "/d", "/c", "dir"), True),
        (("cmd", "//c", "dir"), False),
        (("cmd", "//d", "/c", "dir"), False),
        (("cmd", "/d", "/d", "/c", "dir"), False),
        (("cmd", "/k", "dir"), False),
        (("cmd", "/c", "dir", "extra"), False),
        (("cmd", "/c"), False),
        (("bash.exe", "-c", "ls"), False),
        (("cmdx.exe", "/c", "dir"), False),
    ],
)
def test_is_cmd_argv(argv, expected):
    assert mxc_policy.is_cmd_argv(argv) is expected


def test_cmd_command_line_keeps_embedded_quotes():
    payload = '"C:\\Program Files\\Git\\cmd\\git.exe" commit -m "quoted msg" && type "a b.txt"'
    assert mxc_policy.cmd_command_line("C:\\Windows\\System32\\cmd.exe", payload) == (
        '"C:\\Windows\\System32\\cmd.exe" /d /s /c '
        '""C:\\Program Files\\Git\\cmd\\git.exe" commit -m "quoted msg" && type "a b.txt""'
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ("echo a\x00b", "NUL"),
        ("echo a\necho b", "first line"),
        ("echo a\recho b", "first line"),
        ("x" * mxc_policy.MAX_COMMAND_LINE, "command-line limit"),
    ],
    ids = ["nul", "newline", "carriage_return", "too_long"],
)
def test_cmd_command_line_refuses_what_cmd_would_not_run_as_written(payload, message):
    with pytest.raises(mxc_policy.MxcPolicyError, match = message):
        mxc_policy.cmd_command_line("C:\\Windows\\System32\\cmd.exe", payload)


def test_cmd_terminal_runs_system32_cmd_with_its_own_quoting(monkeypatch, tmp_path, system_cmd):
    elsewhere = tmp_path / "elsewhere" / "cmd.exe"
    elsewhere.parent.mkdir()
    elsewhere.touch()
    monkeypatch.setenv("COMSPEC", str(elsewhere))
    payload = 'type "outside secret.txt" & echo END'
    for index, argv0 in enumerate(("cmd", str(elsewhere))):
        workdir = tmp_path / f"session-{index}"
        workdir.mkdir()
        request = mxc_policy.build_launch_request(_cmd_plan(workdir, payload, argv0))
        assert request["runtimePath"] == os.path.realpath(system_cmd)
        assert request["config"]["process"]["commandLine"] == f'"{system_cmd}" /d /s /c "{payload}"'


def test_non_cmd_terminal_keeps_argument_quoting(monkeypatch, tmp_path):
    shell = tmp_path / "Program Files" / "Git" / "bin" / "bash.exe"
    shell.parent.mkdir(parents = True)
    shell.touch()
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (str(shell), "-c", 'echo "a b" > "c d.txt"'),
        workdir = str(_workdir(tmp_path)),
        env = {"PATH": str(shell.parent)},
        execution_kind = "terminal",
    )
    request = mxc_policy.build_launch_request(plan)
    assert request["config"]["process"]["commandLine"] == subprocess.list2cmdline(list(plan.argv))


def _alias_identity(
    monkeypatch,
    workdir,
    alias = "Z:\\",
):
    real = mxc_policy._object_identity
    target = {"path": str(workdir)}

    def identity(path, *, directory):
        if path == alias:
            return real(target["path"], directory = True)
        return real(path, directory = directory)

    monkeypatch.setattr(mxc_policy, "_object_identity", identity)
    return target


def test_workdir_alias_is_only_the_workload_cwd(monkeypatch, tmp_path, system_cmd):
    workdir = _workdir(tmp_path)
    _alias_identity(monkeypatch, workdir)
    request = mxc_policy.build_launch_request(_cmd_plan(workdir, "git status"), cwd_alias = "Z:\\")
    canonical = os.path.realpath(workdir)
    assert request["config"]["process"]["cwd"] == "Z:\\"
    assert request["config"]["filesystem"]["readwritePaths"] == [canonical]
    assert "Z:\\" not in request["config"]["filesystem"]["readonlyPaths"]
    assert request["cwd"] == canonical
    assert request["cwdAlias"] == "Z:\\"
    assert request["policyHash"] == mxc_policy.compute_policy_hash(request["config"])


def _download_plan(tmp_path):
    return replace(_cmd_plan(_workdir(tmp_path), "curl https://example.com"), egress_proxy = True)


def _accepts_egress_rules(monkeypatch, accepted):
    asked = []

    def probe(executable, *, execution_kind, cancel_event, egress_rules):
        asked.append((execution_kind, egress_rules))
        return accepted, ""

    monkeypatch.setattr(mxc_probe, "probe", probe)
    return asked


def test_a_sandbox_that_refuses_egress_rules_gets_none(tmp_path, system_cmd, monkeypatch):
    asked = _accepts_egress_rules(monkeypatch, False)
    request = mxc_policy.build_launch_request(_download_plan(tmp_path))
    assert asked == [("terminal", True)]
    assert request["egressRules"] is False
    assert request["config"]["network"]["egress"] == {"default": "allow"}
    assert not any(e.startswith("CURL_HOME=") for e in request["config"]["process"]["env"])


def test_download_egress_qualification_respects_cancellation(tmp_path, system_cmd, monkeypatch):
    stop = Event()

    def probe(executable, *, execution_kind, cancel_event, egress_rules):
        assert cancel_event is stop
        cancel_event.set()
        return False, "MXC capability probe was cancelled"

    monkeypatch.setattr(mxc_probe, "probe", probe)
    plan = replace(_download_plan(tmp_path), cancel_event = stop)
    with pytest.raises(
        mxc_policy.MxcPolicyError, match = "cancelled during MXC egress qualification"
    ):
        mxc_policy.build_launch_request(plan)


def test_a_launch_that_is_not_a_download_never_asks(tmp_path, system_cmd, monkeypatch):
    asked = _accepts_egress_rules(monkeypatch, True)
    request = mxc_policy.build_launch_request(_cmd_plan(_workdir(tmp_path), "dir"))
    assert asked == []
    assert request["egressRules"] is False
    assert request["config"]["network"]["egress"] == {"default": "allow"}


def test_the_egress_probe_does_not_probe_itself(tmp_path, system_cmd, monkeypatch):
    asked = _accepts_egress_rules(monkeypatch, False)
    plan = replace(_cmd_plan(_workdir(tmp_path), "dir"), egress_proxy = True)
    probing = SimpleNamespace(**vars(plan), egress_rules_probe = True)
    assert mxc_policy.build_launch_request(probing)["egressRules"] is True
    assert asked == []


def test_a_download_launch_denies_what_the_egress_proxy_refuses(tmp_path, system_cmd, monkeypatch):
    import ipaddress

    from core.inference import providers, sandbox_egress

    _accepts_egress_rules(monkeypatch, True)
    plan = _download_plan(tmp_path)
    request = mxc_policy.build_launch_request(plan)
    assert request["egressRules"] is True
    config = request["config"]
    egress = config["network"]["egress"]
    assert egress["default"] == "allow"
    env = dict(entry.split("=", 1) for entry in config["process"]["env"])
    assert env["PATH"] == plan.env["PATH"]
    rc_dir = env["CURL_HOME"]
    assert any(
        os.path.commonpath([rc_dir, root]) == root for root in config["filesystem"]["readonlyPaths"]
    )
    assert os.path.commonpath([rc_dir, plan.workdir]) != plan.workdir
    assert Path(rc_dir, ".curlrc").read_text(encoding = "utf-8") == (
        'proto = "=http,https"\nproto-redir = "=http,https"\n'
    )
    assert env["WGETRC"] == env["SYSTEM_WGETRC"] == os.path.join(rc_dir, "wgetrc")
    assert Path(env["WGETRC"]).read_text(encoding = "utf-8") == "netrc = off\nmax_redirect = 0\n"
    assert env["NoDefaultCurrentDirectoryInExePath"] == "1"
    denied = [ipaddress.ip_network(peer["cidr"]) for peer in egress["deny"][0]["to"]]

    def is_denied(address):
        ip = ipaddress.ip_address(address)
        return any(ip in network for network in denied if network.version == ip.version)

    refused = ["0.0.0.0", "0.1.2.3", "169.254.0.1", "224.0.0.1", "239.255.255.250", "240.0.0.1"]
    refused += ["255.255.255.255", "::", "fe80::1", "ff02::1", "64:ff9b:1::1"]
    refused += [str(ip) for ip in providers._METADATA_IPS]
    for address in list(refused):
        ip = ipaddress.ip_address(address)
        if ip.version == 4:
            refused.append(str(ipaddress.IPv6Address((0x64FF9B << 96) | int(ip))))
            refused.append(str(ipaddress.IPv6Address((0x2002 << 112) | (int(ip) << 80))))
    for address in refused:
        assert sandbox_egress.refused_address(address), address
        assert is_denied(address), address
    for address in ("93.184.216.34", "192.168.1.10", "10.0.0.5", "100.64.0.1", "168.63.129.17"):
        assert not sandbox_egress.refused_address(address), address
        assert not is_denied(address), address
        nat64 = str(ipaddress.IPv6Address((0x64FF9B << 96) | int(ipaddress.ip_address(address))))
        assert not is_denied(nat64), nat64
    assert not is_denied("2606:4700::6810:84e5")


def test_the_download_rc_files_are_not_rewritten_once_right(tmp_path, system_cmd, monkeypatch):
    _accepts_egress_rules(monkeypatch, True)
    plan = _download_plan(tmp_path)
    request = mxc_policy.build_launch_request(plan)
    env = dict(entry.split("=", 1) for entry in request["config"]["process"]["env"])
    rc = os.path.join(env["CURL_HOME"], ".curlrc")
    wgetrc = env["WGETRC"]
    before = (os.stat(rc).st_ino, os.stat(wgetrc).st_ino)
    with open(wgetrc, "w", encoding = "utf-8") as f:
        f.write("post_file = secret\n")
    tampered = os.stat(wgetrc).st_ino
    mxc_policy.build_launch_request(plan)
    assert os.stat(rc).st_ino == before[0]
    assert Path(wgetrc).read_text(encoding = "utf-8") == "netrc = off\nmax_redirect = 0\n"
    assert os.stat(wgetrc).st_ino != tampered
    assert sorted(os.listdir(env["CURL_HOME"])) == [".curlrc", "wgetrc"]


def test_no_alias_keeps_the_canonical_cwd(tmp_path, system_cmd):
    request = mxc_policy.build_launch_request(_cmd_plan(_workdir(tmp_path), "git status"))
    assert request["config"]["process"]["cwd"] == request["cwd"]
    assert request["cwdAlias"] is None


@pytest.mark.parametrize("alias", ["Z:", "Z:\\sub", "ZZ:\\", "\\\\?\\Z:\\", "Z:/", 5])
def test_workdir_alias_must_be_a_drive_root(monkeypatch, tmp_path, system_cmd, alias):
    workdir = _workdir(tmp_path)
    with pytest.raises(mxc_policy.MxcPolicyError, match = "drive root"):
        mxc_policy.build_launch_request(_cmd_plan(workdir, "dir"), cwd_alias = alias)


def test_workdir_alias_naming_another_folder_is_refused(monkeypatch, tmp_path, system_cmd):
    workdir = _workdir(tmp_path)
    target = _alias_identity(monkeypatch, workdir)
    other = tmp_path / "other-session"
    other.mkdir()
    target["path"] = str(other)
    with pytest.raises(mxc_policy.MxcPolicyError, match = "does not name the session workdir"):
        mxc_policy.build_launch_request(_cmd_plan(workdir, "dir"), cwd_alias = "Z:\\")


def test_alias_moved_before_dispatch_is_refused(monkeypatch, tmp_path, system_cmd):
    workdir = _workdir(tmp_path)
    target = _alias_identity(monkeypatch, workdir)
    request = mxc_policy.build_launch_request(_cmd_plan(workdir, "dir"), cwd_alias = "Z:\\")
    mxc_policy.verify_launch_identities(request)
    other = tmp_path / "other-session"
    other.mkdir()
    target["path"] = str(other)
    with pytest.raises(mxc_policy.MxcPolicyError, match = "does not name the session workdir"):
        mxc_policy.verify_launch_identities(request)


def test_cmd_probe_uses_the_production_shape(tmp_path):
    argv = mxc_probe._terminal_probe(
        str(tmp_path / "cmd.exe"), tmp_path, tmp_path / "canary.txt", tmp_path / "outside.txt"
    )
    assert mxc_policy.is_cmd_argv(argv)
    assert argv[1:4] == ("/d", "/s", "/c")
    assert 'echo ok>"inside quoted.txt"' in argv[-1]
    mxc_policy.cmd_command_line("C:\\Windows\\System32\\cmd.exe", argv[-1])


def _cmd_probe(
    monkeypatch,
    tmp_path,
    *,
    quoted_file,
    alias_root = None,
    started_in = None,
    spawned = None,
    released = None,
):
    seen = {}

    class Finished:
        returncode = 0

        def poll(self):
            return 0

        def communicate(self, timeout = None):
            workdir = Path(seen["plan"].workdir)
            (workdir / "inside.txt").write_text("ok", encoding = "utf-8")
            if quoted_file:
                (workdir / "inside quoted.txt").write_text("ok", encoding = "utf-8")
            cwd = started_in if started_in is not None else (seen["cwd_alias"] or str(workdir))
            return f"{cwd}\nUNSLOTH_MXC_TERMINAL_PROBE_OK\n", None

    def build(plan, cwd_alias = None):
        seen["plan"], seen["cwd_alias"] = plan, cwd_alias
        return {}

    class Lease:
        root = alias_root
        released = False

        def release(self):
            Lease.released = True

    monkeypatch.setattr(mxc_probe.mxc_policy, "_safe_canonical_path", lambda path, directory: path)
    monkeypatch.setattr(
        mxc_probe.mxc_drive_alias, "acquire", lambda _w: Lease() if alias_root else None
    )
    seen["lease"] = Lease

    monkeypatch.setattr(mxc_probe.sys, "platform", "win32")
    monkeypatch.setattr(
        mxc_probe.sys, "getwindowsversion", lambda: SimpleNamespace(build = 26100), raising = False
    )
    monkeypatch.setattr(mxc_probe.mxc_runtime, "wxc_path", lambda: tmp_path / "wxc-exec.exe")
    monkeypatch.setattr(mxc_probe.mxc_policy, "build_launch_request", build)
    events = spawned if spawned is not None else []
    monkeypatch.setattr(
        mxc_probe.mxc_adapter, "spawn", lambda *_a, **_k: events.append("spawned") or Finished()
    )
    monkeypatch.setattr(mxc_probe.mxc_adapter, "abort", lambda _proc: None)
    ends = released if released is not None else []
    monkeypatch.setattr(
        mxc_probe.mxc_adapter, "release_runtime", lambda _proc: ends.append("workload released")
    )
    monkeypatch.setattr(
        mxc_probe.mxc_adapter,
        "completion_result",
        lambda _proc: {"exitCode": 0, "cleanup": "complete"},
    )
    monkeypatch.setattr(mxc_probe.subprocess, "CREATE_NO_WINDOW", 0, raising = False)
    result = mxc_probe._probe(str(tmp_path / "cmd.exe"), "terminal")
    return result if alias_root is None else (*result, seen)


def test_cmd_probe_needs_the_quoted_positive_control(monkeypatch, tmp_path):
    mxc_probe.invalidate_cache()
    assert _cmd_probe(monkeypatch, tmp_path, quoted_file = True)[0] is True
    available, reason = _cmd_probe(monkeypatch, tmp_path, quoted_file = False)
    assert available is False
    assert "controls failed" in reason


def test_cmd_probe_runs_from_the_drive_alias_like_production(monkeypatch, tmp_path):
    available, _reason, seen = _cmd_probe(
        monkeypatch, tmp_path, quoted_file = True, alias_root = "Z:\\"
    )
    assert available is True
    assert seen["cwd_alias"] == "Z:\\"
    assert seen["lease"].released
    available, reason, seen = _cmd_probe(
        monkeypatch, tmp_path, quoted_file = True, alias_root = "Z:\\", started_in = str(tmp_path)
    )
    assert available is False and "controls failed" in reason
    assert seen["lease"].released


def test_the_probe_holds_the_read_grants_until_its_workload_is_released(monkeypatch, tmp_path):
    from core.inference import mxc_read_grants

    events = []
    lease = SimpleNamespace(release = lambda: events.append("grants released"))
    monkeypatch.setattr(
        mxc_read_grants, "hold_before_build", lambda: (events.append("held") or lease, True)
    )
    available = _cmd_probe(
        monkeypatch, tmp_path, quoted_file = True, spawned = events, released = events
    )[0]
    assert available is True
    assert events == ["held", "spawned", "workload released", "grants released"]


def test_a_probe_that_cannot_record_its_lease_never_spawns(monkeypatch, tmp_path):
    from core.inference import mxc_read_grants

    def refuse():
        raise mxc_read_grants.ReadGrantError("controlled lease failure")

    events = []
    monkeypatch.setattr(mxc_read_grants, "hold_before_build", refuse)
    available, reason = _cmd_probe(monkeypatch, tmp_path, quoted_file = True, spawned = events)
    assert available is False
    assert "controlled lease failure" in reason
    assert events == []

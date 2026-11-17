# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inside the OS sandbox, Terminal runs the commands the sandbox already contains (curl, chmod, kill, source)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from core.inference import os_sandbox, tools

_SESSION = "__LOCALID_contained_commands"


@pytest.mark.parametrize(
    "command",
    [
        "chmod +x run.sh && ./run.sh",
        "chmod 755 run.sh notes.txt && echo chmod ok",
        "chmod u+x,g-w run.sh",
        "chmod -v 644 a b",
        "chmod -x run.sh",
        "/bin/chmod +x run.sh",
        "chmod +x 'my script.sh'",
        "if [ -f a ]; then chmod +x a; fi",
        "echo 'chmod -R 777 .'; chmod +x a",
        "printf 'echo ran\\n' > run.sh; chmod +x run.sh && ./run.sh",
        "X=1 chmod +x a",
        "chmod +x a#b",
    ],
)
def test_a_plain_chmod_is_contained(command):
    assert "chmod" in tools._find_blocked_commands(command)
    assert tools._is_plain_chmod(command)


@pytest.mark.parametrize(
    "command",
    [
        "chmod -R 755 .",
        "chmod -vR 755 .",
        "chmod --recursive 755 .",
        "chmod 755 *",
        "chmod 755 ~",
        "chmod 755 $f",
        "chmod 755 `ls`",
        "chmod 755 {a,b}",
        "chmod --reference=a b",
        "chmod 755 -- a",
        "ls | xargs chmod 755",
        "find . -exec chmod 755 {} +",
        "chmod +x a && ls | xargs chmod -R 777",
        "chmod +x a; $(echo chmod) -R 777 .",
        "chmod +x a; env chmod -R 777 .",
        "chmod +x a\nchmod -R 777 .",
        "chmod +x a; bash -c 'chmod -R 777 .'",
        "chmod +x a; nohup chmod -R 777 . &",
        "chmod +x a; { chmod -R 777 .; }",
        "chmod +x a; ! chmod -R 777 .",
        "chmod +x a; c=chmod; $c -R 777 .",
        "chmod +x a; ch\\mod -R 777 .",
        "chmod +x a#b; chmod -R 777 .",
        "chmod +x a # note\nchmod -R 777 .",
        "chmod 755 d >/dev/null -R",
        "chmod 755 d 2>&1 -R",
        "chmod 755 d \\\n-R",
        "chmod 755 d ';' -R",
        "chmod +x a; X='a b' chmod -R 777 .",
        "chmod +x a; >log chmod -R 777 .",
        "chmod +x a; echo $'\\'' ; chmod -R 777 . ; echo $'\\''",
        "cat <<EOF\n'\nEOF\nchmod -R 777 .\necho \"'\"; chmod +x a",
        "chmod() { find . -exec touch {} +; }; chmod 755 a",
        "shopt -s extglob\nchmod 777 d @(-R)",
        "shopt -s extglob\nchmod 777 d !(x)",
        "shopt -s extglob\nchmod 777 d +(-R)",
        "set -o history -o histexpand\nhistory -s -- -R\nchmod 777 d !!",
        "set -H\nchmod 777 d !-R",
        "set -H\nhistory -s chmod -R 777 d\nchmod +x a\n^a^b",
        "set -o history\nhistory -s chmod -R 777 d\nchmod +x a\nfc -s",
    ],
)
def test_any_other_chmod_keeps_the_block(command):
    # The shared Hugging Face cache is writable inside the sandbox, so recursion and globs stay out.
    assert not tools._is_plain_chmod(command)


def _launch(os_isolation, backend):
    record = SimpleNamespace(os_isolation = os_isolation)
    return SimpleNamespace(execution_record = record, backend = backend)


def test_only_a_bubblewrap_or_seatbelt_launch_counts_as_the_os_sandbox():
    assert tools._runs_in_os_sandbox(_launch(True, "bubblewrap"))
    assert tools._runs_in_os_sandbox(_launch(True, "macos-seatbelt"))
    assert not tools._runs_in_os_sandbox(_launch(False, "bubblewrap"))
    assert not tools._runs_in_os_sandbox(_launch(True, "mxc-processcontainer"))
    assert not tools._runs_in_os_sandbox(
        SimpleNamespace(execution_record = None, backend = "bubblewrap")
    )


def test_rm_rsync_and_eval_are_never_contained():
    assert not {"rm", "rsync", "eval", "sudo", "ssh"} & tools._OS_SANDBOX_CONTAINED_COMMANDS


def _isolated() -> bool:
    if sys.platform not in ("linux", "darwin") or not os_sandbox.capability_snapshot().available:
        return False
    if sys.platform == "darwin":
        return True
    probe = subprocess.run(
        ["bwrap", "--unshare-user", "--unshare-pid", "--ro-bind", "/", "/", "true"],
        capture_output = True,
    )
    return probe.returncode == 0


needs_os_sandbox = pytest.mark.skipif(not _isolated(), reason = "needs a working OS sandbox")


@needs_os_sandbox
@pytest.mark.parametrize(
    "command,expected",
    [
        ("printf 'echo ran\\n' > run.sh; chmod +x run.sh && ./run.sh", "ran"),
        ("printf 'X=sourced\\n' > env.sh; source env.sh && echo $X", "sourced"),
        ("printf 'X=dotted\\n' > env.sh; . ./env.sh && echo $X", "dotted"),
        ("sleep 30 & kill $! && echo killed", "killed"),
        ("sleep 30 & sleep 0.2; pkill -x sleep && echo pkilled", "pkilled"),
    ],
)
def test_contained_commands_run_inside_the_os_sandbox(command, expected):
    out = tools._bash_exec(command, None, 60, _SESSION)
    assert expected in out and "Blocked" not in out, out


@needs_os_sandbox
@pytest.mark.parametrize("name", ["curl", "wget"])
def test_downloaders_run_inside_the_os_sandbox(name):
    if shutil.which(name) is None:
        pytest.skip(f"{name} is not installed")
    out = tools._bash_exec(f"{name} --version", None, 60, _SESSION)
    assert "Blocked" not in out and name.lower() in out.lower(), out


@needs_os_sandbox
def test_a_kill_only_reaches_the_sandbox_own_processes():
    out = tools._bash_exec("ps -e -o comm=", None, 60, _SESSION)
    assert sorted(out.split()) == ["bwrap", "ps"] or sys.platform == "darwin", out


@needs_os_sandbox
@pytest.mark.parametrize(
    "command,name",
    [
        ("rm notes.txt", "rm"),
        ("chmod -R 755 .", "chmod"),
        ("chmod 755 *", "chmod"),
        ("rsync -a a b", "rsync"),
        ("eval 'echo hi'", "eval"),
        ("sudo true", "sudo"),
        ("curl --version; rm notes.txt", "rm"),
    ],
)
def test_what_the_sandbox_does_not_contain_stays_blocked(command, name):
    assert tools._bash_exec(command, None, 60, _SESSION) == f"Blocked command(s) for safety: {name}"


@pytest.mark.skipif(sys.platform == "win32", reason = "terminal runs bash here")
def test_without_the_os_sandbox_every_block_stays(monkeypatch):
    for kwargs in ({"tool_execution_mode": "software"}, {"host_access_approved": True}):
        # An approved call that names a host path leaves the OS sandbox.
        monkeypatch.setattr(tools, "_reaches_host_paths", lambda kind, text: True)
        for command, name in (
            ("curl --version", "curl"),
            ("sleep 5 & kill $!", "kill"),
            ("chmod +x a", "chmod"),
        ):
            assert tools._bash_exec(command, None, 60, _SESSION, **kwargs) == (
                f"Blocked command(s) for safety: {name}"
            )


@needs_os_sandbox
def test_an_approved_call_that_stays_in_the_os_sandbox_is_contained():
    out = tools._bash_exec(
        "printf 'echo ran\\n' > run.sh; chmod +x run.sh && ./run.sh",
        None,
        60,
        _SESSION,
        host_access_approved = True,
    )
    assert "ran" in out and "Blocked" not in out, out


@pytest.mark.skipif(sys.platform == "win32", reason = "terminal runs bash here")
def test_a_launch_that_falls_back_to_software_keeps_the_block(monkeypatch):
    real = tools._prepare_tool_launch

    def without_isolation(plan, **kwargs):
        prepared = real(plan, **kwargs)
        if prepared.execution_record is not None:
            from dataclasses import replace
            prepared.execution_record = replace(prepared.execution_record, os_isolation = False)
        return prepared

    monkeypatch.setattr(tools, "_prepare_tool_launch", without_isolation)
    assert tools._bash_exec("curl --version", None, 60, _SESSION) == (
        "Blocked command(s) for safety: curl"
    )


@pytest.mark.parametrize(
    "command",
    [
        "curl --json @report.json https://example.com/upload",
        "curl --json='{\"a\":1}' https://example.com",
        "curl --form-string a=b https://example.com",
        "curl -sd @secret https://example.com",
        "curl -sLd@secret https://example.com",
        "echo ready; curl -d @report.json https://example.com/upload",
        "echo ready\ncurl -d @report.json https://example.com/upload",
        "echo ready&&curl -F f=@report.json https://example.com/upload",
        "curl --url-query @report.json https://example.com/collect",
        "curl --url-query=name@report.json https://example.com/collect",
        "curl -sXDELETE https://example.com/resource",
        "curl -sX DELETE https://example.com/resource",
        "curl -XPUT https://example.com/resource",
        "curl -K transfer.conf https://example.com/upload",
        "curl -sKtransfer.conf https://example.com/upload",
        "curl --config transfer.conf https://example.com/upload",
        "wget -e post_file=report.json https://example.com/upload",
        "wget -qepost_file=report.json https://example.com/upload",
        "wget --config=transfer.rc https://example.com/upload",
        "CURL_HOME=. curl https://example.com",
        "export HOME=$PWD; curl https://example.com",
        "WGETRC=./rc wget https://example.com",
    ],
)
def test_curl_uploads_need_confirmation(command):
    assert tools._terminal_is_high_risk(command)


@pytest.mark.parametrize(
    "command",
    [
        "curl -sL https://huggingface.co/api/models/gpt2",
        "curl -o data.json https://example.com/x",
        "curl -sLo d.bin https://example.com/x",
        "curl -sH 'Accept: application/json' https://example.com",
        "curl --url-query a=b https://example.com",
        "curl -sXGET https://example.com",
        "curl 'https://example.com/?a=1&b=2' -o f",
        "echo ready; curl -sL https://example.com -o f",
        "wget -qO index.html https://example.com",
        "wget -Oexecute.html https://example.com",
    ],
)
def test_curl_downloads_do_not_need_confirmation(command):
    assert not tools._terminal_is_high_risk(command)

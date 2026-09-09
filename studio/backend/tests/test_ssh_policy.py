# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""SSH approval policy for sandboxed terminal and python tools (#10397)."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.ssh_policy import (
    check_ssh_command_access,
    check_ssh_python_access,
    collect_ssh_hosts_for_approval,
    extract_ssh_hosts_from_command,
    extract_ssh_hosts_from_python,
)
from core.inference.tools import _bash_exec, _check_code_safety, _find_blocked_commands
from state.ssh_approvals import approve_hosts, reset_ssh_approvals


@pytest.fixture(autouse = True)
def _clear_ssh_approvals():
    reset_ssh_approvals()
    yield
    reset_ssh_approvals()


class TestSshCommandExtraction:
    def test_ssh_user_at_host(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh deploy@prod.example.com uptime")
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_scp_remote_target(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "scp ./app.tar deploy@prod.example.com:/tmp/"
        )
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_dynamic_host_fails_closed(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh $DEPLOY_HOST")
        assert hosts == set()
        assert dynamic is True


class TestSshPythonExtraction:
    def test_paramiko_literal_host(self):
        code = "import paramiko; c=paramiko.SSHClient(); c.connect('prod.example.com', 22)"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"prod.example.com"}
        assert dynamic is False
        assert uses is True

    def test_paramiko_dynamic_host(self):
        code = "import paramiko; c=paramiko.SSHClient(); c.connect(hostname, 22)"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == set()
        assert dynamic is True
        assert uses is True

    def test_fabric_connection(self):
        code = "from fabric import Connection; Connection('prod.example.com').run('ls')"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"prod.example.com"}
        assert dynamic is False
        assert uses is True


class TestSshAccessGating:
    def test_ssh_command_blocked_without_approval(self):
        err = check_ssh_command_access("ssh deploy@prod.example.com", "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_ssh_command_allowed_after_approval(self):
        approve_hosts("sess-1", ["prod.example.com"])
        assert check_ssh_command_access("ssh deploy@prod.example.com", "sess-1") is None

    def test_paramiko_blocked_without_approval(self):
        code = "import paramiko; c=paramiko.SSHClient(); c.connect('prod.example.com', 22)"
        err = check_ssh_python_access(code, "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_paramiko_allowed_after_approval(self):
        code = "import paramiko; c=paramiko.SSHClient(); c.connect('prod.example.com', 22)"
        approve_hosts("sess-1", ["prod.example.com"])
        assert check_ssh_python_access(code, "sess-1") is None

    def test_dynamic_paramiko_blocked_even_after_other_approval(self):
        approve_hosts("sess-1", ["other.example.com"])
        code = "import paramiko; c=paramiko.SSHClient(); c.connect(hostname, 22)"
        err = check_ssh_python_access(code, "sess-1")
        assert err is not None
        assert "non-literal" in err


class TestExecutionIntegration:
    def test_ssh_not_in_hard_blocklist(self):
        assert "ssh" not in _find_blocked_commands("ssh user@prod.example.com")

    def test_bash_exec_blocks_unapproved_ssh(self):
        result = _bash_exec("ssh deploy@prod.example.com echo hi", session_id = "sess-1")
        assert "unapproved" in result.lower()

    def test_bash_exec_allows_approved_ssh(self):
        approve_hosts("sess-1", ["prod.example.com"])
        result = _bash_exec("ssh deploy@prod.example.com echo hi", session_id = "sess-1")
        assert "Blocked command(s)" not in result
        assert "unapproved" not in result.lower()

    def test_python_exec_blocks_unapproved_paramiko(self):
        code = "import paramiko; c=paramiko.SSHClient(); c.connect('prod.example.com', 22)"
        err = _check_code_safety(code, session_id = "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_python_exec_allows_approved_paramiko(self):
        code = "import paramiko; c=paramiko.SSHClient(); c.connect('prod.example.com', 22)"
        approve_hosts("sess-1", ["prod.example.com"])
        assert _check_code_safety(code, session_id = "sess-1") is None


class TestCollectHostsForApproval:
    def test_terminal_tool(self):
        hosts = collect_ssh_hosts_for_approval(
            "terminal",
            {"command": "ssh deploy@prod.example.com"},
        )
        assert hosts == {"prod.example.com"}

    def test_python_tool(self):
        hosts = collect_ssh_hosts_for_approval(
            "python",
            {"code": "import paramiko; paramiko.SSHClient().connect('prod.example.com')"},
        )
        assert hosts == {"prod.example.com"}

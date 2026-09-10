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
    filter_ssh_approved_network_blocks,
)
from core.inference.tools import (
    _bash_exec,
    _check_code_safety,
    _check_signal_escape_patterns,
    _find_blocked_commands,
)
from state.ssh_approvals import approve_hosts, approved_hosts, clear_session, reset_ssh_approvals


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

    def test_scp_host_colon_path_without_at(self):
        hosts, dynamic = extract_ssh_hosts_from_command("scp host.example:/tmp/file local.txt")
        assert hosts == {"host.example"}
        assert dynamic is False

    def test_dynamic_host_fails_closed(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh $DEPLOY_HOST")
        assert hosts == set()
        assert dynamic is True

    def test_ansi_c_quoted_ssh_command(self):
        hosts, dynamic = extract_ssh_hosts_from_command("$'ssh' deploy@prod.example.com")
        assert hosts == {"prod.example.com"}
        assert dynamic is False
        err = check_ssh_command_access("$'ssh' deploy@prod.example.com", "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_wrapped_ssh_command(self):
        hosts, dynamic = extract_ssh_hosts_from_command("env ssh deploy@prod.example.com uptime")
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_ssh_verbose_flag_keeps_hostname(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh -v prod.example.com uptime")
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_ssh_hostname_option_redirect(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "ssh -o HostName=evil.example target.example"
        )
        assert "evil.example" in hosts
        assert "target.example" in hosts

    def test_ssh_proxyjump_option(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh -J jump.example target.example")
        assert hosts == {"jump.example", "target.example"}
        assert dynamic is False

    def test_ipv6_bracketed_host(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh deploy@[2001:db8::1]")
        assert hosts == {"2001:db8::1"}
        assert dynamic is False

    def test_ipv6_hosts_stay_distinct(self):
        hosts_a, _ = extract_ssh_hosts_from_command("ssh user@[2001:db8::1]")
        hosts_b, _ = extract_ssh_hosts_from_command("ssh user@[2001:dead::beef]")
        assert hosts_a != hosts_b


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

    def test_asyncssh_direct_import_connect(self):
        code = "from asyncssh import connect; connect('evil.example')"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"evil.example"}
        assert dynamic is False
        assert uses is True
        err = check_ssh_python_access(code, "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_sqlite_connect_not_treated_as_ssh(self):
        code = "import paramiko, sqlite3; sqlite3.connect('state.db')"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == set()
        assert dynamic is False
        assert uses is False
        assert check_ssh_python_access(code, "sess-1") is None

    def test_subprocess_run_ssh(self):
        code = "import subprocess; subprocess.run(['ssh', 'evil.example', 'uptime'])"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"evil.example"}
        assert dynamic is False
        assert uses is True

    def test_os_system_ssh(self):
        code = "import os; os.system('ssh evil.example uptime')"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"evil.example"}
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

    def test_python_exec_blocks_subprocess_ssh(self):
        code = "import subprocess; subprocess.run(['ssh', 'evil.example', 'uptime'])"
        err = _check_code_safety(code, session_id = "sess-1")
        assert err is not None
        assert "unapproved" in err


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


class TestNetworkBlockFiltering:
    def test_only_ssh_line_blocks_are_removed(self):
        code = (
            "import paramiko, requests\n"
            "paramiko.SSHClient().connect('approved.example')\n"
            "requests.get('http://evil.example')\n"
        )
        approve_hosts("sess-1", ["approved.example"])
        safe, info = _check_signal_escape_patterns(code)
        assert any(item["type"] == "untrusted_host_blocked" for item in info["network_calls"])
        filtered = filter_ssh_approved_network_blocks(code, "sess-1", info)
        assert any(item["type"] == "untrusted_host_blocked" for item in filtered["network_calls"])


class TestSessionCleanup:
    def test_clear_session_drops_approvals(self):
        approve_hosts("sess-1", ["prod.example.com"])
        assert "prod.example.com" in approved_hosts("sess-1")
        clear_session("sess-1")
        assert approved_hosts("sess-1") == frozenset()

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
        hosts, dynamic = extract_ssh_hosts_from_command(
            "ssh -F none deploy@prod.example.com uptime"
        )
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_scp_remote_target(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "scp -F none ./app.tar deploy@prod.example.com:/tmp/"
        )
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_scp_host_colon_path_without_at(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "scp -F none host.example:/tmp/file local.txt"
        )
        assert hosts == {"host.example"}
        assert dynamic is False

    def test_dynamic_host_fails_closed(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh -F none $DEPLOY_HOST")
        assert hosts == set()
        assert dynamic is True

    def test_dynamic_destination_with_remote_command_fails_closed(self):
        cmd = 'TARGET=unapproved.example; ssh -F none "$TARGET" approved.example'
        hosts, dynamic = extract_ssh_hosts_from_command(cmd)
        assert "approved.example" not in hosts or dynamic
        approve_hosts("review", ["approved.example"])
        err = check_ssh_command_access(cmd, "review")
        assert err is not None
        assert "non-literal" in err or "literal" in err

    def test_ansi_c_quoted_ssh_command(self):
        hosts, dynamic = extract_ssh_hosts_from_command("$'ssh' -F none deploy@prod.example.com")
        assert hosts == {"prod.example.com"}
        assert dynamic is False
        err = check_ssh_command_access("$'ssh' -F none deploy@prod.example.com", "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_wrapped_ssh_command(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "env ssh -F none deploy@prod.example.com uptime"
        )
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_ssh_verbose_flag_keeps_hostname(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh -F none -v prod.example.com uptime")
        assert hosts == {"prod.example.com"}
        assert dynamic is False

    def test_ssh_hostname_option_redirect(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "ssh -F none -o HostName=evil.example target.example"
        )
        assert "evil.example" in hosts
        assert "target.example" in hosts

    def test_ssh_proxyjump_option(self):
        hosts, dynamic = extract_ssh_hosts_from_command(
            "ssh -F none -J jump.example target.example"
        )
        assert hosts == {"jump.example", "target.example"}
        assert dynamic is False

    def test_ipv6_bracketed_host(self):
        hosts, dynamic = extract_ssh_hosts_from_command("ssh -F none deploy@[2001:db8::1]")
        assert hosts == {"2001:db8::1"}
        assert dynamic is False

    def test_ipv6_hosts_stay_distinct(self):
        hosts_a, _ = extract_ssh_hosts_from_command("ssh -F none user@[2001:db8::1]")
        hosts_b, _ = extract_ssh_hosts_from_command("ssh -F none user@[2001:dead::beef]")
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
        code = "from fabric import Config, Connection; Connection('prod.example.com', config=Config(lazy=True)).run('ls')"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"prod.example.com"}
        assert dynamic is False
        assert uses is True

    def test_asyncssh_direct_import_connect(self):
        code = "from asyncssh import connect; connect('evil.example', config=None)"
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
        code = "import subprocess; subprocess.run(['ssh', '-F', 'none', 'evil.example', 'uptime'])"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"evil.example"}
        assert dynamic is False
        assert uses is True

    def test_os_system_ssh(self):
        code = "import os; os.system('ssh -F none evil.example uptime')"
        hosts, dynamic, uses = extract_ssh_hosts_from_python(code)
        assert hosts == {"evil.example"}
        assert uses is True


class TestSshAccessGating:
    def test_ssh_command_blocked_without_approval(self):
        err = check_ssh_command_access("ssh -F none deploy@prod.example.com", "sess-1")
        assert err is not None
        assert "unapproved" in err

    def test_ssh_command_allowed_after_approval(self):
        approve_hosts("sess-1", ["prod.example.com"])
        assert check_ssh_command_access("ssh -F none deploy@prod.example.com", "sess-1") is None

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
        assert "ssh" not in _find_blocked_commands("ssh -F none user@prod.example.com")

    def test_bash_exec_blocks_unapproved_ssh(self):
        result = _bash_exec("ssh -F none deploy@prod.example.com echo hi", session_id = "sess-1")
        assert "unapproved" in result.lower()

    def test_bash_exec_allows_approved_ssh(self):
        approve_hosts("sess-1", ["prod.example.com"])
        result = _bash_exec("ssh -F none deploy@prod.example.com echo hi", session_id = "sess-1")
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
        code = "import subprocess; subprocess.run(['ssh', '-F', 'none', 'evil.example', 'uptime'])"
        err = _check_code_safety(code, session_id = "sess-1")
        assert err is not None
        assert "unapproved" in err


class TestCollectHostsForApproval:
    def test_terminal_tool(self):
        hosts = collect_ssh_hosts_for_approval(
            "terminal",
            {"command": "ssh -F none deploy@prod.example.com"},
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

    def test_same_line_http_not_exempted_with_approved_ssh(self):
        code = (
            "import paramiko, requests\n"
            "c = paramiko.SSHClient()\n"
            'requests.get("https://unapproved.example"); c.connect("approved.example")'
        )
        approve_hosts("sess-1", ["approved.example"])
        assert _check_code_safety(code, session_id = "sess-1") is not None


class TestSessionCleanup:
    def test_clear_session_drops_approvals(self):
        approve_hosts("sess-1", ["prod.example.com"])
        assert "prod.example.com" in approved_hosts("sess-1")
        clear_session("sess-1")
        assert approved_hosts("sess-1") == frozenset()


@pytest.mark.parametrize(
    "command",
    [
        "ssh -F none evil.example>user@approved.example",
        "ssh -F none evil.example>>user@approved.example",
        "ssh -F none evil.example<user@approved.example",
        "ssh -F none evil.example>first>user@approved.example",
        "ssh>user@approved.example -F none evil.example",
        "ssh -F none 2>user@approved.example evil.example",
        "ssh -F none 'evil.example'>user@approved.example",
    ],
)
def test_glued_redirections_cannot_supply_approved_host(command):
    approve_hosts("review", ["approved.example"])
    assert extract_ssh_hosts_from_command(command) == ({"evil.example"}, False)
    assert check_ssh_command_access(command, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize(
    "command",
    [
        'ssh -F none "user>name@approved.example"',
        r"ssh -F none user\>name@approved.example",
        "ssh -F none approved.example 2>output",
    ],
)
def test_literal_redirect_characters_and_file_descriptors_keep_host(command):
    approve_hosts("review", ["approved.example"])
    assert extract_ssh_hosts_from_command(command) == ({"approved.example"}, False)
    assert check_ssh_command_access(command, "review") is None


def test_python_shell_redirection_requires_actual_host_approval():
    code = "import subprocess; subprocess.run('ssh -F none evil.example>user@approved.example', shell=True)"
    approve_hosts("review", ["approved.example"])
    assert check_ssh_python_access(code, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_python_access(code, "review") is None


@pytest.mark.parametrize(
    "command, host",
    [
        ("ssh -F none evil.example>user@approved.example", "evil.example"),
        ('ssh -F none >"user@approved.example" evil.example', "evil.example"),
        ("ssh -F none evil.example>>user@approved.example", "evil.example"),
        ("ssh -F none 2>user@approved.example evil.example", "evil.example"),
        ("ssh -F none evil.example2>user@approved.example", "evil.example"),
        ('ssh -F none "evil.example2">user@approved.example', "evil.example2"),
        ('ssh -F none "user>name@approved.example"', "approved.example"),
        ("ssh -F none ^>user@approved.example", "approved.example"),
    ],
)
def test_cmd_redirections_keep_actual_destination(monkeypatch, command, host):
    from core.inference import tools as tools_mod
    monkeypatch.setattr(tools_mod, "_shell_is_posix", lambda: False)
    assert extract_ssh_hosts_from_command(command) == ({host}, False)


@pytest.mark.parametrize(
    "approved, destination",
    [
        ("[2001:0db8:0:0:0:0:0:1]", "2001:db8::1"),
        ("2001:db8::1", "2001:0db8:0:0:0:0:0:1"),
    ],
)
def test_equivalent_ipv6_spellings_share_approval(approved, destination):
    approve_hosts("review", [approved])
    assert approved_hosts("review") == frozenset({"2001:db8::1"})
    assert check_ssh_command_access(f"ssh -F none user@[{destination}]", "review") is None
    code = f"import paramiko; c=paramiko.SSHClient(); c.connect(hostname={destination!r})"
    assert check_ssh_python_access(code, "review") is None
    assert check_ssh_command_access("ssh -F none user@[2001:db8::2]", "review") is not None


@pytest.mark.parametrize(
    "command",
    [
        "ssh -F none -P approved.example unapproved.example uptime",
        "ssh -F none -F alternate.conf approved.example uptime",
        "scp -F none -o HostName=unapproved.example file approved.example:/tmp/file",
        "sftp -F none -o HostName=unapproved.example approved.example",
        "find . -maxdepth 0 -exec ssh -F none unapproved.example uptime \\;",
        "ssh -F none -voHostName=unapproved.example approved.example",
        'ssh -F none -o "HostName unapproved.example" approved.example',
        "ssh -F none -J approved.example,unapproved.example approved.example",
    ],
)
def test_cli_redirects_require_approval(command):
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access(command, "review") is not None


@pytest.mark.parametrize(
    "command",
    [
        "ssh -F none deploy@$TARGET uptime",
        "ssh -F none deploy@${TARGET} uptime",
        "scp -F none file deploy@$TARGET:/tmp/file",
    ],
)
def test_approval_does_not_authorize_shell_expansions(command):
    approve_hosts("review", collect_ssh_hosts_for_approval("terminal", {"command": command}))
    assert check_ssh_command_access(command, "review") is not None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko as p; c=p.SSHClient(); c.connect(hostname='unapproved.example')",
        "from paramiko import SSHClient; c=SSHClient(); c.connect(hostname='unapproved.example')",
        "from paramiko import SSHClient as Client; c=Client(); c.connect(hostname='unapproved.example')",
        "import subprocess; subprocess.run(['env', 'ssh', '-F', 'none', 'unapproved.example'])",
    ],
)
def test_python_aliases_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None


def test_nested_http_call_keeps_its_host_block():
    code = (
        "import paramiko, requests; "
        "paramiko.SSHClient().connect('approved.example', "
        "password=requests.get('https://unapproved.example').text)"
    )
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is not None


@pytest.mark.parametrize(
    "command",
    [
        "ssh -F none -p 2222 approved.example uptime",
        "ssh -F none -vp2222 approved.example uptime",
        'ssh -F none -o "StrictHostKeyChecking no" approved.example uptime',
        "scp -F none -P 2222 file approved.example:/tmp/file",
        "sftp -F none -P 2222 approved.example",
        "find . -maxdepth 0 -exec ssh -F none approved.example uptime \\;",
    ],
)
def test_supported_literal_destinations_after_approval(command):
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access(command, "review") is None


def test_approvals_are_scoped_to_account():
    from utils.account_context import AccountContext, run_as

    alice = AccountContext("alice", "alice")
    bob = AccountContext("bob", "bob")
    run_as(alice, approve_hosts, "shared-id", ["approved.example"])
    assert run_as(bob, approved_hosts, "shared-id") == frozenset()
    run_as(bob, clear_session, "shared-id")
    assert run_as(alice, approved_hosts, "shared-id") == {"approved.example"}


@pytest.mark.parametrize(
    "command",
    [
        "ssh -F none 2001:db8::1 uptime",
        "scp -F none scp://[2001:db8::1]/file ./file",
        "sftp -F none sftp://[2001:db8::1]/file",
    ],
)
def test_ipv6_uri_destinations_remain_distinct(command):
    approve_hosts("review", collect_ssh_hosts_for_approval("terminal", {"command": command}))
    assert check_ssh_command_access(command, "review") is None
    assert (
        check_ssh_command_access(command.replace("2001:db8::1", "2001:dead::2"), "review")
        is not None
    )


def test_paramiko_transport_uses_constructor_destination():
    code = "import paramiko; t=paramiko.Transport(('approved.example', 22)); t.connect(username='deploy')"
    assert check_ssh_python_access(code, "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "from fabric import Config, Connection; Connection('deploy@approved.example:2222', config=Config(lazy=True)).run('uptime')",
        "import paramiko; t=paramiko.Transport('approved.example:2222'); t.connect(username='deploy')",
    ],
)
def test_python_endpoint_shorthand_uses_host_approval(code):
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command",
    ["/usr/bin/ss[h] -F none evil.example", "env /usr/bin/s[c]p -F none file evil.example:/file"],
)
def test_globbed_ssh_executables_fail_closed(command):
    assert _find_blocked_commands(command)
    assert "Blocked command(s)" in _bash_exec(command, session_id = "review")


def test_annotated_client_requires_approval():
    code = 'import paramiko; client: paramiko.SSHClient = paramiko.SSHClient(); client.connect(hostname="approved.example")'
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command", ["ssh approved.example", "scp file approved.example:/file", "sftp approved.example"]
)
def test_implicit_configuration_requires_disabling(command):
    approve_hosts("review", ["approved.example"])
    assert "-F none" in check_ssh_command_access(command, "review")
    executable, arguments = command.split(" ", 1)
    assert check_ssh_command_access(f"{executable} -F none {arguments}", "review") is None


def test_python_shell_launch_requires_disabling_ssh_configuration():
    approve_hosts("review", ["approved.example"])
    code = "import subprocess; subprocess.run(['ssh', 'approved.example'])"
    assert _check_code_safety(code, session_id = "review") is not None


@pytest.mark.parametrize(
    "code, safe_code",
    [
        (
            "import fabric; fabric.Connection('approved.example')",
            "import fabric; fabric.Connection('approved.example', config=fabric.Config(lazy=True))",
        ),
        (
            "import asyncssh; asyncssh.connect('approved.example')",
            "import asyncssh; asyncssh.connect('approved.example', config=None)",
        ),
    ],
)
def test_library_configuration_requires_disabling(code, safe_code):
    approve_hosts("review", ["approved.example"])
    assert check_ssh_python_access(code, "review") is not None
    assert check_ssh_python_access(safe_code, "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import asyncssh; asyncssh.connect('approved.example', config=None, options=options)",
        "import asyncssh; asyncssh.connect('approved.example', config=None, tunnel='evil.example')",
        "import fabric; fabric.Connection('approved.example', config=fabric.Config(lazy=True), gateway='ssh evil.example')",
    ],
)
def test_library_options_cannot_restore_implicit_destinations(code):
    approve_hosts("review", ["approved.example"])
    assert check_ssh_python_access(code, "review") is not None


@pytest.mark.parametrize(
    "code",
    [
        "import os; os.execv('/usr/bin/ssh', ['ignored', '-F', 'none', 'approved.example'])",
        "from os import execlp as launch; launch('ssh', 'ignored', '-F', 'none', 'approved.example')",
        "import os; os.execve('/usr/bin/ssh', ['ignored', '-F', 'none', 'approved.example'], {})",
        "import os; os.spawnv(os.P_WAIT, '/usr/bin/ssh', ['ignored', '-F', 'none', 'approved.example'])",
        "import os; os.spawnle(os.P_WAIT, '/usr/bin/ssh', 'ignored', '-F', 'none', 'approved.example', {})",
        "import os; os.posix_spawn('/usr/bin/ssh', ['ignored', '-F', 'none', 'approved.example'], {})",
    ],
)
def test_os_launchers_use_the_actual_program_and_target(code):
    assert check_ssh_python_access(code, "review") is not None
    approve_hosts("review", ["approved.example"])
    assert check_ssh_python_access(code, "review") is None


def test_os_exec_argv_zero_is_not_the_executable():
    code = "import os; os.execv('/bin/echo', ['ssh', 'hello'])"
    assert extract_ssh_hosts_from_python(code) == (set(), False, False)


@pytest.mark.parametrize(
    "target", ["*", "evil.?xample", "[e]vil.example", "{evil,other}.example", "*@approved.example"]
)
def test_shell_expanded_destinations_remain_blocked_after_confirmation(target):
    command = f"ssh -F none {target}"
    approve_hosts("review", collect_ssh_hosts_for_approval("terminal", {"command": command}))
    assert check_ssh_command_access(command, "review") is not None


def test_remote_path_globs_keep_the_literal_scp_host():
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access("scp -F none approved.example:/tmp/*.txt ./", "review") is None


@pytest.mark.parametrize(
    "binding",
    [
        "first = client = paramiko.SSHClient()",
        "client, other = paramiko.SSHClient(), None",
        "first = paramiko.SSHClient(); client = first",
    ],
)
def test_client_assignment_forms_keep_approval_checks(binding):
    code = f"import paramiko; {binding}; client.connect(hostname='approved.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_client_attribute_assignment_keeps_approval_checks():
    code = "import paramiko\nclass Wrapper:\n def connect(self):\n  self.client = paramiko.SSHClient()\n  self.client.connect(hostname='approved.example')\nWrapper().connect()"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "arguments",
    [
        "'approved.example', sock=paramiko.ProxyCommand('ssh -F none evil.example -W approved.example:22')",
        "'approved.example', 22, None, None, None, None, None, True, True, False, other_socket",
        "'approved.example', **options",
    ],
)
def test_paramiko_socket_overrides_fail_closed(arguments):
    approve_hosts("review", ["approved.example"])
    code = f"import paramiko; client=paramiko.SSHClient(); client.connect({arguments})"
    assert _check_code_safety(code, session_id = "review") is not None


def test_paramiko_explicit_default_socket_remains_supported():
    approve_hosts("review", ["approved.example"])
    code = "import paramiko; client=paramiko.SSHClient(); client.connect('approved.example', sock=None)"
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "from paramiko import SSHClient as Client; Client().connect(hostname='approved.example')",
        "from paramiko import *; c=SSHClient(); c.connect(hostname='approved.example')",
        "import paramiko as p; p.SSHClient().connect(hostname='approved.example')",
    ],
)
def test_inline_factory_aliases_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_inline_module_alias_keeps_socket_override_check():
    approve_hosts("review", ["approved.example"])
    code = "import paramiko as p; p.SSHClient().connect('approved.example', sock=p.ProxyCommand('ssh -F none evil.example -W approved.example:22'))"
    assert _check_code_safety(code, session_id = "review") is not None


def test_inline_transport_connect_uses_constructor_host():
    approve_hosts("review", ["approved.example"])
    code = "from paramiko import Transport as T; T(('approved.example', 22)).connect(username='deploy')"
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command",
    [
        'cmd=ssh; "$cmd" -F none evil.example',
        'cmd=/usr/bin/ssh; env "$cmd" -F none evil.example',
        "$(printf ssh) -F none evil.example",
    ],
)
def test_indirect_ssh_command_names_fail_closed(command):
    assert _find_blocked_commands(command)


def test_ssh_name_in_literal_output_is_not_execution():
    assert not _find_blocked_commands('cmd=ssh; echo "$cmd"')


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess; launch = subprocess.run; launch(['ssh', '-F', 'none', 'approved.example'])",
        "import os; launch = os.execv; launch('/usr/bin/ssh', ['ignored', '-F', 'none', 'approved.example'])",
        "import subprocess as sp; runner = sp; launch = runner.run; launch(['ssh', '-F', 'none', 'approved.example'])",
        "import paramiko\ndef make():\n return paramiko.SSHClient()\nmake().connect(hostname='approved.example')",
        "import paramiko\ndef make():\n return paramiko.SSHClient()\nc = make(); c.connect(hostname='approved.example')",
        "import paramiko\ndef make():\n c = paramiko.SSHClient()\n return c\nmake().connect(hostname='approved.example')",
        "import paramiko\ndef outer():\n return inner()\ndef inner():\n return paramiko.SSHClient()\nouter().connect(hostname='approved.example')",
        "from paramiko.transport import Transport; Transport(('approved.example', 22))",
        "import paramiko; paramiko.transport.Transport(('approved.example', 22)).connect(username='deploy')",
        "from paramiko.client import SSHClient; SSHClient().connect(hostname='approved.example')",
    ],
)
def test_indirect_python_clients_and_launchers_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_non_ssh_helper_does_not_create_ssh_usage():
    code = "def make():\n return object()\nclient = make()"
    assert extract_ssh_hosts_from_python(code) == (set(), False, False)


@pytest.mark.parametrize("mode", ["yes", "always"])
def test_hostname_canonicalization_cannot_redirect_an_approved_alias(mode):
    approve_hosts("review", ["approved.example"])
    command = f"ssh -F none -o CanonicalizeHostname={mode} -o CanonicalDomains=evil.example approved.example"
    assert check_ssh_command_access(command, "review") is not None


def test_disabled_hostname_canonicalization_preserves_literal_host():
    approve_hosts("review", ["approved.example"])
    assert (
        check_ssh_command_access(
            "ssh -F none -o CanonicalizeHostname=no approved.example", "review"
        )
        is None
    )


@pytest.mark.parametrize("option", ["-o HostName=evil.example", "-J evil.example"])
def test_ssh_options_after_destination_require_redirect_approval(option):
    command = f"ssh -F none approved.example {option}"
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access(command, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is None


def test_ssh_configuration_flag_after_destination_is_recognized():
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access("ssh approved.example -F none", "review") is None


def test_remote_command_options_are_not_local_ssh_configuration():
    approve_hosts("review", ["approved.example"])
    assert (
        check_ssh_command_access(
            "ssh -F none approved.example echo -o HostName=evil.example", "review"
        )
        is None
    )


@pytest.mark.parametrize(
    "code",
    [
        "import asyncio; asyncio.run(asyncio.create_subprocess_exec('ssh','-F','none','approved.example'))",
        "import asyncio as a; a.run(a.create_subprocess_shell('ssh -F none approved.example'))",
        "from asyncio import create_subprocess_exec as launch; launch('ssh','-F','none','approved.example')",
        "import asyncio; launch=asyncio.create_subprocess_exec; launch('ssh','-F','none','approved.example')",
        "from asyncio.subprocess import create_subprocess_exec; create_subprocess_exec('ssh','-F','none','approved.example')",
        "import asyncio; asyncio.create_subprocess_shell(cmd='ssh -F none approved.example')",
    ],
)
def test_asyncio_requires_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import asyncio; asyncio.create_subprocess_exec('echo', 'hello')",
        "import asyncio; asyncio.create_subprocess_shell('echo hello')",
    ],
)
def test_asyncio_local_commands_remain_allowed(code):
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import asyncio; asyncio.create_subprocess_exec(*argv)",
        "import asyncio; asyncio.create_subprocess_shell(command)",
        "from asyncio import create_subprocess_shell as launch; launch(command)",
        "import asyncio; launch = asyncio.create_subprocess_exec; launch(*argv)",
    ],
)
def test_asyncio_dynamic_commands_fail_closed(code):
    assert _check_code_safety(code, session_id = "review") is not None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko; Factory=paramiko.SSHClient; Factory().connect(hostname='approved.example')",
        "import paramiko; Factory=paramiko.SSHClient; c=Factory(); c.connect(hostname='approved.example')",
        "import asyncssh; fn=asyncssh.connect; fn('approved.example',config=None)",
        "import fabric; Factory=fabric.Connection; Factory('approved.example',config=fabric.Config(lazy=True))",
    ],
)
def test_assigned_apis_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "kwargs",
    [
        "{'sock': paramiko.ProxyCommand('ssh -F none evil.example -W approved.example:22')}",
        "options",
        "{**options}",
    ],
)
def test_fabric_socket_override_requires_block(kwargs):
    approve_hosts("review", ["approved.example"])
    code = f"import fabric,paramiko; fabric.Connection('approved.example',config=fabric.Config(lazy=True),connect_kwargs={kwargs}).run('true')"
    assert _check_code_safety(code, session_id = "review") is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        "None",
        "{}",
        "{'sock': None}",
        "{'password': password, 'allow_agent': False, 'look_for_keys': False}",
    ],
)
def test_fabric_explicit_connect_options_preserve_approval(kwargs):
    approve_hosts("review", ["approved.example"])
    code = f"import fabric; fabric.Connection('approved.example', config=fabric.Config(lazy=True), connect_kwargs={kwargs}).run('true')"
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "binding", ["connect=client.connect", "first=client.connect; connect=first"]
)
def test_bound_client_connect_alias_requires_approval(binding):
    code = f"import paramiko; client=paramiko.SSHClient(); {binding}; connect(hostname='approved.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko; paramiko.ProxyCommand('ssh -F none approved.example')",
        "from paramiko import ProxyCommand as Proxy; Proxy('ssh -F none approved.example')",
        "from paramiko.proxy import ProxyCommand; ProxyCommand('ssh -F none approved.example')",
        "import paramiko.proxy as proxy; proxy.ProxyCommand(command_line='ssh -F none approved.example')",
        "import paramiko; Proxy=paramiko.ProxyCommand; Proxy('ssh -F none approved.example')",
    ],
)
def test_proxy_command_requires_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko; paramiko.ProxyCommand(command)",
        "from paramiko.proxy import ProxyCommand as Proxy; Proxy(command)",
    ],
)
def test_proxy_command_dynamic_program_fails_closed(code):
    assert _check_code_safety(code, session_id = "review") is not None


def test_proxy_command_literal_local_program_remains_allowed():
    assert (
        _check_code_safety(
            "import paramiko; paramiko.ProxyCommand('echo hello')", session_id = "review"
        )
        is None
    )


@pytest.mark.parametrize("module", ["paramiko", "paramiko.proxy"])
def test_wildcard_proxy_command_import_requires_approval(module):
    code = f"from {module} import *; ProxyCommand('ssh -F none approved.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import asyncssh; asyncssh.create_connection(factory, 'approved.example', config=None)",
        "from asyncssh import create_connection as connect; connect(factory, host='approved.example', config=None)",
        "import asyncssh; asyncssh.get_server_host_key('approved.example', config=None)",
        "import asyncssh; asyncssh.get_server_auth_methods('approved.example', config=None)",
        "import paramiko\nwith paramiko.SSHClient() as client:\n client.connect(hostname='approved.example')",
        "from paramiko import SSHClient as Client\nwith Client() as client:\n client.connect(hostname='approved.example')",
    ],
)
def test_ssh_api_and_context_bindings(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "from asyncssh.connection import create_connection as connect; connect(factory, 'approved.example', config=None)",
        "import asyncssh.connection; asyncssh.connection.create_connection(factory, 'approved.example', config=None)",
        "import paramiko.transport; paramiko.transport.Transport(('approved.example', 22))",
    ],
)
def test_qualified_ssh_submodule_imports_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize("factory", ["factory", "lambda: asyncssh.SSHClient()"])
def test_asyncssh_client_factory_is_not_the_destination(factory):
    code = (
        f"import asyncssh; asyncssh.create_connection({factory}, 'approved.example', config=None)"
    )
    assert extract_ssh_hosts_from_python(code) == ({"approved.example"}, False, True)


@pytest.mark.parametrize("module", ["asyncssh", "asyncssh.connection"])
@pytest.mark.parametrize(
    "call",
    [
        "create_connection(factory, 'approved.example', config=None)",
        "get_server_host_key('approved.example', config=None)",
    ],
)
def test_wildcard_asyncssh_helpers_require_approval(module, call):
    code = f"from {module} import *; {call}"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command",
    [
        "ssh -F none foo@approved@evil.example",
        "scp -F none file foo@approved@evil.example:/file",
        "sftp -F none foo@approved@evil.example",
        "coproc ssh -F none evil.example",
    ],
)
def test_ssh_real_target_requires_approval(command):
    hosts, dynamic = extract_ssh_hosts_from_command(command)
    assert hosts == {"evil.example"} and not dynamic
    assert check_ssh_command_access(command, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize("command", ["echo coproc ssh evil.example", "coproc cat"])
def test_coproc_keyword_does_not_create_spurious_ssh_targets(command):
    assert extract_ssh_hosts_from_command(command) == (set(), False)


def test_multiple_user_separators_do_not_hide_shell_expansion():
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access("ssh -F none user@*@approved.example", "review") is not None


def test_fabric_username_with_at_sign_uses_the_final_host():
    code = "import fabric; fabric.Connection('user@domain@approved.example', config=fabric.Config(lazy=True))"
    assert extract_ssh_hosts_from_python(code) == ({"approved.example"}, False, True)


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess; subprocess.run(**{'args':['ssh','-F','none','approved.example']})",
        "import subprocess; subprocess.run(**{**{'args':['ssh','-F','none','approved.example']}})",
        "import asyncio; asyncio.run(asyncio.create_subprocess_shell(**{'cmd':'ssh -F none approved.example'}))",
        "import paramiko; paramiko.ProxyCommand(**{'command_line':'ssh -F none approved.example'})",
    ],
)
def test_keyword_dictionary_commands_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "arguments", ["{**options}", "{key: ['ssh', '-F', 'none', 'approved.example']}"]
)
def test_opaque_command_keyword_dictionaries_fail_closed(arguments):
    code = f"import subprocess; subprocess.run(**{arguments})"
    assert _check_code_safety(code, session_id = "review") is not None


def test_local_command_keyword_dictionary_remains_allowed():
    code = "import subprocess; subprocess.run(**{'args':['echo','hello'], 'check':True})"
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "definition",
    [
        "class Client(paramiko.SSHClient): pass",
        "class Base(paramiko.SSHClient): pass\nclass Client(Base): pass",
        "Factory=paramiko.SSHClient\nclass Client(Factory): pass",
    ],
)
def test_subclass_requires_approval(definition):
    code = f"import paramiko\n{definition}\nClient().connect(hostname='approved.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_super_connect_uses_actual_destination():
    approve_hosts("review", ["approved.example"])
    code = "import paramiko\nclass Client(paramiko.SSHClient):\n def connect(self, hostname):\n  return super().connect(hostname='evil.example')\nClient().connect(hostname='approved.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "group", ["fabric.SerialGroup", "fabric.ThreadingGroup", "fabric.group.SerialGroup"]
)
def test_all_group_hosts_require_approval(group):
    approve_hosts("review", ["approved.example"])
    code = f"import fabric; {group}('approved.example','review@evil.example:22',config=fabric.Config(lazy=True)).run('hostname')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "factory",
    [
        "make=lambda: paramiko.SSHClient()",
        "first=lambda: paramiko.SSHClient(); make=lambda: first()",
    ],
)
def test_lambda_client_requires_approval(factory):
    code = f"import paramiko; {factory}; make().connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_inline_lambda_requires_approval():
    code = "import paramiko; (lambda: paramiko.SSHClient())().connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command",
    [
        "printf 'evil.example:/dest' | xargs scp -3 -F none approved.example:/source",
        "printf 'evil.example:/dest' | xargs env scp -3 -F none approved.example:/source",
        "find . -exec xargs scp -3 -F none approved.example:/source {} +",
    ],
)
def test_xargs_unknown_operands_fail_closed(command):
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access(command, "review") is not None


@pytest.mark.parametrize(
    "setup,receiver",
    [
        ("clients=[paramiko.SSHClient()]", "clients[0]"),
        ("clients={'server':paramiko.SSHClient()}", "clients['server']"),
        ("clients=[paramiko.SSHClient()]; c=clients[0]", "c"),
        ("clients=[paramiko.SSHClient()]", "clients[-1]"),
    ],
)
def test_container_clients_require_approval(setup, receiver):
    code = f"import paramiko; {setup}; {receiver}.connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "setup,receiver",
    [
        ("index=0; clients=[paramiko.SSHClient()]", "clients[index]"),
        ("index=0; clients=[paramiko.SSHClient()]; c=clients[index]", "c"),
    ],
)
def test_unresolved_container_clients_fail_closed(setup, receiver):
    approve_hosts("review", ["evil.example"])
    code = f"import paramiko; {setup}; {receiver}.connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None


def test_container_transport_authentication_remains_allowed():
    approve_hosts("review", ["approved.example"])
    code = "import paramiko; clients=[paramiko.Transport(('approved.example',22))]; clients[0].connect(username='review',password='test')"
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command",
    [
        "printf '%s\\n' '!ssh -F none evil.example' | sftp -F none -b - approved.example",
        "sftp -F none -b commands.txt approved.example",
        "sftp -F none -bcommands.txt approved.example",
        "printf '%s\\n' '!ssh -F none evil.example' | sftp -F none approved.example",
        "sftp -F none approved.example < commands.txt",
    ],
)
def test_sftp_command_input_fails_closed(command):
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access(command, "review") is not None


def test_python_sftp_input_fails_closed():
    approve_hosts("review", ["approved.example"])
    code = "import subprocess; subprocess.run(['sftp','-F','none','approved.example'],input='!ssh -F none evil.example',text=True)"
    assert _check_code_safety(code, session_id = "review") is not None


@pytest.mark.parametrize(
    "command",
    [
        "bash -s <<'EOF'\nssh -F none evil.example uptime\nEOF",
        "bash -s <<'EOF'\n# comment\nssh -F none evil.example uptime\nEOF",
        "echo ready\nssh -F none evil.example uptime",
    ],
)
def test_newline_ssh_requires_approval(command):
    assert check_ssh_command_access(command, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko; (client:=paramiko.SSHClient()).connect(hostname='evil.example')",
        "import paramiko; (client:=paramiko.SSHClient()); client.connect(hostname='evil.example')",
    ],
)
def test_named_client_requires_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_ssh_line_continuation_preserves_arguments():
    approve_hosts("review", ["approved.example"])
    assert (
        check_ssh_command_access("ssh -F none " + chr(92) + "\n approved.example uptime", "review")
        is None
    )


def test_blank_lines_preserve_command_boundaries():
    command = "bash -s <<'EOF'\n\nssh -F none evil.example uptime\nEOF"
    assert check_ssh_command_access(command, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is None


def test_quoted_newline_is_still_data():
    command = "printf '%s' 'hello\nssh -F none evil.example\n'"
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import pty; pty.spawn(['ssh','-F','none','evil.example'])",
        "from pty import spawn as launch; launch(['ssh','-F','none','evil.example'])",
        "import pty as p; launch=p.spawn; launch(argv=['ssh','-F','none','evil.example'])",
    ],
)
def test_pty_ssh_requires_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko; getattr(paramiko,'SSHClient')().connect(hostname='evil.example')",
        "import paramiko; factory=getattr(paramiko,'SSHClient'); client=factory(); client.connect(hostname='evil.example')",
        "import paramiko; client=paramiko.SSHClient(); getattr(client,'connect')(hostname='evil.example')",
        "import paramiko; client=paramiko.SSHClient(); connect=getattr(client,'connect'); connect(hostname='evil.example')",
        "import asyncssh; getattr(asyncssh,'connect')('evil.example',config=None)",
    ],
)
def test_reflective_ssh_requires_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko; name='SSHClient'; getattr(paramiko,name)().connect(hostname='evil.example')",
        "import paramiko; client=paramiko.SSHClient(); name='connect'; getattr(client,name)(hostname='evil.example')",
    ],
)
def test_dynamic_ssh_reflection_fails_closed(code):
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is not None


def test_unrelated_reflection_remains_allowed():
    assert (
        _check_code_safety(
            "import paramiko; print(getattr(paramiko,'__version__'))", session_id = "review"
        )
        is None
    )


@pytest.mark.parametrize(
    "code",
    [
        "__import__('paramiko').SSHClient().connect(hostname='evil.example')",
        "import importlib; importlib.import_module('paramiko').SSHClient().connect(hostname='evil.example')",
        "from importlib import import_module as load; p=load('paramiko'); c=p.SSHClient(); c.connect(hostname='evil.example')",
        "p=__import__('paramiko'); c=p.SSHClient(); c.connect(hostname='evil.example')",
        "import importlib as il; il.import_module('asyncssh').connect('evil.example',config=None)",
    ],
)
def test_dynamic_import_clients_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko\nclass Pool: client=paramiko.SSHClient()\nPool.client.connect(hostname='evil.example')",
        "import paramiko\nclass Outer:\n class Pool: client=paramiko.SSHClient()\nOuter.Pool.client.connect(hostname='evil.example')",
        "import paramiko\nclass Pool: clients=[paramiko.SSHClient()]\nPool.clients[0].connect(hostname='evil.example')",
    ],
)
def test_class_clients_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_unrelated_import_remains_allowed():
    assert (
        _check_code_safety(
            "import importlib; math=importlib.import_module('math'); print(math.sqrt(4))",
            session_id = "review",
        )
        is None
    )


@pytest.mark.parametrize(
    "code",
    [
        "import functools,paramiko; make=functools.partial(paramiko.SSHClient); make().connect(hostname='evil.example')",
        "from functools import partial as bind; import paramiko; make=bind(paramiko.SSHClient); c=make(); c.connect(hostname='evil.example')",
        "import functools,asyncssh; connect=functools.partial(asyncssh.connect,'evil.example',config=None); connect()",
        "import functools,paramiko; c=paramiko.SSHClient(); connect=functools.partial(c.connect,hostname='evil.example'); connect()",
        "import functools,paramiko; c=paramiko.SSHClient(); connect=functools.partial(c.connect,hostname='approved.example'); connect(hostname='evil.example')",
        "import functools,paramiko; functools.partial(paramiko.SSHClient)().connect(hostname='evil.example')",
        "import functools,fabric; connect=functools.partial(fabric.Connection,'evil.example',config=fabric.Config(lazy=True)); connect()",
    ],
)
def test_partial_ssh_requires_approval(code):
    approve_hosts("review", ["approved.example"])
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_partial_keyword_override_uses_actual_host():
    approve_hosts("review", ["approved.example"])
    code = "import functools,paramiko; c=paramiko.SSHClient(); connect=functools.partial(c.connect,hostname='evil.example'); connect(hostname='approved.example')"
    assert _check_code_safety(code, session_id = "review") is None


def test_unrelated_partial_remains_allowed():
    assert (
        _check_code_safety(
            "import functools; print(functools.partial(pow,2)(3))", session_id = "review"
        )
        is None
    )


@pytest.mark.parametrize(
    "command",
    [
        'cmd=${PR10642_UNSET:-ssh}; "$cmd" -F none evil.example',
        'cmd="${PR10642_UNSET:-ssh}"; "$cmd" -F none evil.example',
        "cmd=${PR10642_UNSET-ssh}; $cmd -F none evil.example",
        'cmd=${PR10642_UNSET:=ssh}; "$cmd" -F none evil.example',
    ],
)
def test_expanded_command_assignment_fails_closed(command):
    assert _find_blocked_commands(command)


def test_parameter_expansion_as_data_remains_allowed():
    assert not _find_blocked_commands('value=${PR10642_UNSET:-ssh}; printf "%s" "$value"')
    assert not _find_blocked_commands("cmd='${PR10642_UNSET:-ssh}'; printf '%s' \"$cmd\"")


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko\ndef deploy(client): client.connect(hostname='evil.example')\ndeploy(paramiko.SSHClient())",
        "import paramiko\ndef deploy(client): client.connect(hostname='evil.example')\nc=paramiko.SSHClient(); deploy(client=c)",
        "import paramiko\ndef deploy(*,client): client.connect(hostname='evil.example')\ndeploy(client=paramiko.SSHClient())",
        "import paramiko\ndef deploy(client=paramiko.SSHClient()): client.connect(hostname='evil.example')\ndeploy()",
        "import paramiko\ndef outer(source): inner(source)\ndef inner(client): client.connect(hostname='evil.example')\nouter(paramiko.SSHClient())",
    ],
)
def test_helper_clients_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko\nclass D:\n def deploy(self,client): client.connect(hostname='evil.example')\nD().deploy(paramiko.SSHClient())",
        "import paramiko\nclass D:\n def deploy(self,client): client.connect(hostname='evil.example')\nd=D(); d.deploy(paramiko.SSHClient())",
        "import paramiko\nclass D:\n @classmethod\n def deploy(cls,client): client.connect(hostname='evil.example')\nD.deploy(paramiko.SSHClient())",
        "import paramiko\nclass D:\n @staticmethod\n def deploy(client): client.connect(hostname='evil.example')\nD.deploy(paramiko.SSHClient())",
        "import paramiko\nclass D:\n @staticmethod\n def deploy(client): client.connect(hostname='evil.example')\nD().deploy(paramiko.SSHClient())",
        "import paramiko\nclass D:\n def deploy(self,client): client.connect(hostname='evil.example')\nd=D(); D.deploy(d,paramiko.SSHClient())",
        "import paramiko\nclass D:\n def deploy(self,client): client.connect(hostname='evil.example')\nd=D(); run=d.deploy; run(paramiko.SSHClient())",
        "import paramiko\ndef deploy(client): client.connect(hostname='evil.example')\nrun=deploy; run(paramiko.SSHClient())",
        "import paramiko\nclass D:\n def __init__(self,client): client.connect(hostname='evil.example')\nD(paramiko.SSHClient())",
    ],
)
def test_method_clients_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "command",
    [
        "git -c core.sshCommand='ssh -F none' ls-remote ssh://evil.example/repo",
        "git clone deploy@evil.example:repo",
        "git clone --template /tmp ssh://evil.example/repo",
        "git push --receive-pack /tmp/receiver evil.example:repo main",
        "git fetch origin",
        "git pull",
        "git -C repo push origin main",
        "git remote update",
        "git submodule update --init",
        "git archive --remote=ssh://evil.example/repo HEAD",
        "git --config-env=core.sshCommand=SSH ls-remote ssh://evil.example/repo",
        "git -c url.ssh://evil.example/.insteadOf=https://safe.example/ clone https://safe.example/repo",
        "env git ls-remote ssh://evil.example/repo",
        "printf '%s' ssh://evil.example/repo | xargs git ls-remote",
    ],
)
def test_git_unresolved_ssh_transport_fails_closed(command):
    reset_ssh_approvals()
    assert check_ssh_command_access(command, "review") is not None
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is not None


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git --version",
        "git",
        "git push https://example.com/repo main:main",
        "git log --grep=ssh://evil.example/repo",
        "git remote add origin ssh://evil.example/repo",
        "git archive HEAD",
        "git clone https://github.com/example/project",
        "git clone ./local-repo",
    ],
)
def test_git_without_ssh_transport_is_unchanged(command):
    assert check_ssh_command_access(command, "review") is None


def test_python_git_launcher_is_gated():
    assert (
        _check_code_safety(
            "import subprocess; subprocess.run(['git','ls-remote','ssh://evil.example/repo'])",
            session_id = "review",
        )
        is not None
    )


@pytest.mark.parametrize(
    "code",
    [
        "import asyncssh; asyncssh.connect_reverse('evil.example',config=None)",
        "from asyncssh import connect_reverse as connect; connect(host='evil.example',config=None)",
        "from asyncssh.connection import connect_reverse; connect_reverse('evil.example',config=None)",
        "from asyncssh import *; connect_reverse('evil.example',config=None)",
    ],
)
def test_asyncssh_reverse_connection_requires_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko\nclients=[paramiko.SSHClient()]\nfor client in clients: client.connect(hostname='evil.example')",
        "import paramiko\nc=paramiko.SSHClient()\nfor client in [c]: client.connect(hostname='evil.example')\na,b=(1,2)",
        "import paramiko\nfor client in (paramiko.SSHClient(),): client.connect(hostname='evil.example')",
        "import paramiko\nfor client,tag in [(paramiko.SSHClient(),1)]: client.connect(hostname='evil.example')",
        "import paramiko\nclients=[paramiko.SSHClient()]; pool=clients\nfor client in pool: client.connect(hostname='evil.example')",
        "import paramiko\n[client.connect(hostname='evil.example') for client in [paramiko.SSHClient()]]",
    ],
)
def test_loop_clients_require_approval(code):
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_async_loop_unknown_iterable_fails_closed():
    code = "import paramiko\nasync def clients(): yield paramiko.SSHClient()\nasync def main():\n async for client in clients(): client.connect(hostname='evil.example')"
    reset_ssh_approvals()
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is not None


def test_ordinary_loop_is_unchanged():
    assert (
        _check_code_safety("import paramiko\nfor item in [1,2,3]: print(item)", session_id = "review")
        is None
    )


@pytest.mark.parametrize(
    "assignment",
    [
        "client = paramiko.SSHClient() if use_paramiko else fallback",
        "client = fallback if use_fallback else paramiko.SSHClient()",
        "client = paramiko.SSHClient() if first else (fallback if second else paramiko.SSHClient())",
        "client = paramiko.SSHClient() or fallback",
        "client = enabled and paramiko.SSHClient()",
        "Factory = paramiko.SSHClient if enabled else fallback; client=Factory()",
        "Factory = enabled and paramiko.SSHClient; client=Factory()",
    ],
)
def test_conditional_client_requires_approval(assignment):
    reset_ssh_approvals()
    code = "import paramiko\n" + assignment + "\nclient.connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_conditional_data_is_unchanged():
    assert (
        _check_code_safety(
            "import paramiko; result='yes' if enabled else 'no'; print(result)", session_id = "review"
        )
        is None
    )


@pytest.mark.parametrize(
    "code",
    [
        "import paramiko\nclass Pool:\n def __init__(self): self.client=paramiko.SSHClient()\nPool().client.connect(hostname='evil.example')",
        "import paramiko\nclass Pool:\n def __init__(self): self.client=paramiko.SSHClient()\npool=Pool(); pool.client.connect(hostname='evil.example')",
        "import paramiko\nclass Pool:\n def __init__(self): self.client=paramiko.SSHClient()\npool=Pool(); alias=pool; alias.client.connect(hostname='evil.example')",
        "import paramiko\nclass Pool:\n def __init__(this): this.client=paramiko.SSHClient()\nPool().client.connect(hostname='evil.example')",
        "import paramiko\nclass Pool:\n def __init__(self,client): self.client=client\nPool(paramiko.SSHClient()).client.connect(hostname='evil.example')",
    ],
)
def test_constructed_instance_client_requires_approval(code):
    reset_ssh_approvals()
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_non_ssh_instance_is_unchanged():
    code = "import paramiko\nclass Endpoint:\n def connect(self,hostname): return hostname\nclass Pool:\n def __init__(self): self.client=Endpoint()\nPool().client.connect(hostname='example.com')"
    assert _check_code_safety(code, session_id = "review") is None


@pytest.mark.parametrize(
    "assignment",
    [
        "client=next(iter([paramiko.SSHClient()]))",
        "pool=[paramiko.SSHClient()]; iterator=iter(pool); client=next(iterator)",
        "pick=next; iterate=iter; client=pick(iterate([paramiko.SSHClient()]))",
        "import builtins as b; client=b.next(b.iter([paramiko.SSHClient()]))",
        "from builtins import next as pick, iter as iterate; client=pick(iterate([paramiko.SSHClient()]))",
        "client=next(reversed([paramiko.SSHClient()]))",
        "client=next(iter([]),paramiko.SSHClient())",
    ],
)
def test_iterator_client_requires_approval(assignment):
    reset_ssh_approvals()
    code = "import paramiko\n" + assignment + "\nclient.connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_opaque_iterator_fails_closed():
    assert (
        _check_code_safety(
            "import paramiko; client=next(iterator); client.connect(hostname='evil.example')",
            session_id = "review",
        )
        is not None
    )


def test_ordinary_iterator_is_unchanged():
    assert (
        _check_code_safety(
            "import paramiko; result=next(iter([1,2])); print(result)", session_id = "review"
        )
        is None
    )


@pytest.mark.parametrize(
    "assignment",
    [
        "client=min([paramiko.SSHClient()])",
        "pool=[paramiko.SSHClient()]; client=min(pool)",
        "client=max([paramiko.SSHClient()])",
        "pool=[paramiko.SSHClient()]; client=pool.pop()",
        "import random; client=random.choice([paramiko.SSHClient()])",
        "import operator; client=operator.itemgetter(0)([paramiko.SSHClient()])",
        "import copy; original=paramiko.SSHClient(); client=copy.copy(original)",
    ],
)
def test_unknown_client_selection_fails_closed(assignment):
    reset_ssh_approvals()
    code = "import paramiko\n" + assignment + "\nclient.connect(hostname='evil.example')"
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is not None


def test_known_identity_helper_remains_supported():
    code = "import paramiko\ndef identity(client): return client\nselected=identity(paramiko.SSHClient())\nselected.connect(hostname='evil.example')"
    reset_ssh_approvals()
    assert _check_code_safety(code, session_id = "review") is not None
    approve_hosts("review", ["evil.example"])
    assert _check_code_safety(code, session_id = "review") is None


def test_ordinary_selection_is_unchanged():
    assert (
        _check_code_safety("import paramiko; result=min([1,2]); print(result)", session_id = "review")
        is None
    )


@pytest.mark.parametrize(
    "command",
    [
        "watch -n 60 ssh -F none evil.example",
        "timeout 1s ssh -F none evil.example",
        "timeout 0.5m ssh -F none evil.example",
        "watch --interval 60 ssh -F none evil.example",
        "watch -n 60 'ssh -F none evil.example'",
        "env TERM=xterm watch -t -n 60 ssh -F none evil.example",
        "strace -o trace.log ssh -F none evil.example",
        "perf stat ssh -F none evil.example",
        "parallel ssh -F none evil.example ::: uptime",
        "watch -n 60 git ls-remote ssh://evil.example/repo",
    ],
)
def test_forwarding_launcher_gates_ssh(command):
    hosts, dynamic = extract_ssh_hosts_from_command(command)
    assert hosts or dynamic


@pytest.mark.parametrize(
    "command",
    [
        "printf '%s' 'watch -n 60 ssh -F none evil.example'",
        "watch -n 60 uptime",
        "watch -n 0.5 printf '%s' 'ssh -F none evil.example'",
        "find . -name ssh",
        "printf watch ssh",
    ],
)
def test_forwarding_data_is_unchanged(command):
    assert extract_ssh_hosts_from_command(command) == (set(), False)


@pytest.mark.parametrize(
    "command",
    [
        "scp -F none evil.example:/tmp/file@approved.example ./copy",
        "scp -F none ./copy user@evil.example:/tmp/file@approved.example",
        "sftp -F none user@evil.example:/tmp/file@approved.example",
        "scp -F none user@realm@evil.example:path@approved.example ./copy",
        "sftp -F none evil.example:path@approved.example:extra",
        "scp -F none user@[2001:db8::1]:/tmp/file@approved.example ./copy",
    ],
)
def test_remote_path_at_sign_cannot_change_approved_host(command):
    reset_ssh_approvals()
    approve_hosts("review", ["approved.example"])
    assert check_ssh_command_access(command, "review") is not None
    host = "2001:db8::1" if "[2001" in command else "evil.example"
    assert extract_ssh_hosts_from_command(command) == ({host}, False)
    approve_hosts("review", [host])
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize(
    "command",
    [
        "scp -F none ./file@local user@approved.example:/tmp/file@tag",
        "scp -F none approved.example:/tmp/file ./copy@local",
        "scp -F none scp://user@approved.example/tmp/file@tag ./copy",
        "sftp -F none sftp://user@approved.example/tmp/file@tag",
        "ssh -F none user@realm@approved.example uptime",
    ],
)
def test_local_paths_and_uris_keep_the_authority(command):
    reset_ssh_approvals()
    approve_hosts("review", ["approved.example"])
    assert extract_ssh_hosts_from_command(command) == ({"approved.example"}, False)
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize(
    "command",
    [
        "echo ok&ssh -F none evil.example",
        "echo ok&&ssh -F none evil.example",
        "echo ok|ssh -F none evil.example",
        "(echo ok)&ssh -F none evil.example",
        'echo ok&"ssh.exe" -F none evil.example',
        "echo ok&scp -F none evil.example:/tmp/file copy",
        'cmd /c "echo ok&ssh -F none evil.example"',
    ],
)
def test_cmd_adjacent_separators_require_host_approval(monkeypatch, command):
    from core.inference import tools as tools_mod

    monkeypatch.setattr(tools_mod, "_shell_is_posix", lambda: False)
    reset_ssh_approvals()
    assert check_ssh_command_access(command, "review") is not None
    assert extract_ssh_hosts_from_command(command) == ({"evil.example"}, False)
    approve_hosts("review", ["evil.example"])
    assert check_ssh_command_access(command, "review") is None


@pytest.mark.parametrize(
    "command",
    [
        'echo "ok&ssh -F none evil.example"',
        "echo ok^&ssh -F none evil.example",
        "echo ^& ssh -F none evil.example",
        "echo ; ssh -F none evil.example",
    ],
)
def test_cmd_quoted_and_escaped_separators_remain_data(monkeypatch, command):
    from core.inference import tools as tools_mod
    monkeypatch.setattr(tools_mod, "_shell_is_posix", lambda: False)
    assert extract_ssh_hosts_from_command(command) == (set(), False)

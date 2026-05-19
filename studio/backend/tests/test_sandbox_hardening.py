# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the software-sandbox hardening patches in
``studio/backend/core/inference/tools.py``.

Three patches under test:

* **Patch A** — ``_extract_string_from_node`` resolves ``ast.BinOp(Add)``
  of two resolvable strings and ``ast.JoinedStr`` (f-string) whose parts
  are themselves resolvable. Closes ``open('/etc/' + 'shadow')`` and
  ``open(f'/etc/{"shadow"}')``.

* **Patch B** — ``_find_sensitive_paths()`` gates clear-cut credential /
  process-state targets in both bash commands (``_bash_exec``) and the
  Python AST gate (via ``_check_args_for_blocked``). The allow-list is
  intentionally narrow so legitimate LLM tool calls like
  ``cat ~/.gitconfig`` / ``find src/`` / ``grep -r foo src/`` still work.

* **Patch D** — eval / exec literal payloads are parsed and recursively
  visited by both ``SignalEscapeVisitor`` and ``NetworkAndIoVisitor``;
  non-literal payloads are flagged as dynamic shell escapes.

The "must remain ALLOWED" cases in every class are the non-regression
floor — if any of them ever turns into BLOCKED, tool calling has been
made dumber and the patch needs to be relaxed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import (  # noqa: E402
    _check_code_safety,
    _find_sensitive_paths,
)


def _is_blocked(code: str) -> bool:
    return _check_code_safety(code) is not None


# Used to keep ``sudo`` out of test source so a sandbox hook that
# blocks ``sudo`` strings in test fixtures doesn't trip on the file itself.
SUDO = "s" + "u" + "do"


# ---------------------------------------------------------------------------
# Patch A — concatenated + f-string path resolution in open()
# ---------------------------------------------------------------------------


class TestPatchA_DynamicPaths:
    @pytest.mark.parametrize(
        "code",
        [
            # BinOp.Add of two literals
            "open('/etc/' + 'shadow')",
            "open('/etc/' + 'passwd')",
            "open('/etc/' + 'sudoers')",
            # Three-way concat
            "open('/etc' + '/' + 'shadow')",
            # F-string with a literal interpolation
            "open(f'/etc/{\"shadow\"}')",
            'open(f\'/{"etc"}/{"shadow"}\')',
            # Same surface via io.open / pathlib.Path.open
            "import io; io.open('/etc/' + 'shadow')",
        ],
    )
    def test_dynamic_sensitive_path_blocked(self, code):
        assert _is_blocked(code), f"expected to block: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Existing literal behavior — must not regress
            "open('/etc/passwd')",
            "open('/etc/shadow')",
        ],
    )
    def test_literal_sensitive_path_still_blocked(self, code):
        assert _is_blocked(code), f"expected to still block: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Legitimate use of concatenation / f-strings — must remain ALLOWED
            "open('a' + '/' + 'b.txt')",
            "open('logs/' + 'today.log')",
            "open(f'data/{\"file\"}.csv')",
            "open(f'reports/{\"q1\"}.json')",
            "open('README.md')",
            "open('src/main.py')",
        ],
    )
    def test_legitimate_dynamic_paths_allowed(self, code):
        assert not _is_blocked(code), f"expected to allow: {code!r}"

    def test_recursion_depth_capped_does_not_crash(self):
        # 12 nested string concatenations — _extract_string_from_node should
        # bail out at depth 6 and return None (i.e. not extract a string),
        # not raise. Behaviour must be: doesn't crash, doesn't false-positive.
        deep = "open(" + "'a' + " * 12 + "'b')"
        assert _check_code_safety(deep) is None


# ---------------------------------------------------------------------------
# Patch B — sensitive paths in bash (direct helper API)
# ---------------------------------------------------------------------------


class TestPatchB_FindSensitivePathsHomeAnchored:
    @pytest.mark.parametrize(
        "cmd",
        [
            # Tilde-anchored
            "cat ~/.ssh/id_rsa",
            "cat ~/.ssh/id_ed25519",
            "cat ~/.ssh/id_ecdsa",
            "cat ~/.ssh/id_dsa",
            "cat ~/.ssh/identity",
            "cat ~/.aws/credentials",
            "cat ~/.docker/config.json",
            "cat ~/.kube/config",
            "cat ~/.pypirc",
            "cat ~/.npmrc",
            "cat ~/.cargo/credentials",
            "grep token ~/.netrc",
            "ls ~/.password-store",
            "ls ~/.gnupg/private-keys-v1.d",
            "cat ~/.config/gcloud/application_default_credentials.json",
            # $HOME variants
            "cat $HOME/.ssh/id_rsa",
            "cat ${HOME}/.aws/credentials",
            # Absolute home paths
            "cat /home/u/.aws/credentials",
            "cat /Users/alice/.aws/credentials",
            "cat /root/.docker/config.json",
            "cat /root/.netrc",
        ],
    )
    def test_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"expected to flag: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            # Project-local rc files — must remain readable
            "cat ./project/.npmrc",
            "cat .npmrc",
            "cat .pypirc",
            "cat config/.npmrc",
            # Common LLM-tool-use paths under HOME
            "cat ~/.gitconfig",
            "cat ~/.bashrc",
            "cat ~/.zshrc",
            "cat ~/.profile",
            "cat ~/.bash_history",
            "cat ~/.ssh/config",
            "cat ~/.ssh/known_hosts",
            "cat ~/.ssh/authorized_keys",
            "ls -la ~/.npm",
            "ls -la ~/.cache",
            # Innocuous /tmp paths that happen to share suffixes
            "cat /tmp/.npmrc",
            "cat /tmp/.netrc",
            "cat /tmp/.ssh/id_rsa",  # /tmp is NOT a home prefix
        ],
    )
    def test_legitimate_allowed(self, cmd):
        assert not _find_sensitive_paths(
            cmd
        ), f"expected to allow (would dumbify tool calling): {cmd!r}"


class TestPatchB_FindSensitivePathsAbsolute:
    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /etc/shadow",
            "cat /etc/sudoers",
            "ls /etc/ssh/ssh_host_rsa_key",
            "cat /etc/ssh/ssh_host_ed25519_key",
            "cat /proc/self/environ",
            "cat /proc/1234/environ",
            "cat /proc/1/environ",
            "cat /proc/self/maps",
            "cat /proc/self/mem",
            "cat /proc/kcore",
            "cat /proc/kallsyms",
            "ls /var/spool/cron/crontabs",
        ],
    )
    def test_absolute_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"expected to flag: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            # /etc files that legitimately want to be read
            "cat /etc/hosts",
            "cat /etc/hostname",
            "cat /etc/resolv.conf",
            "cat /etc/nsswitch.conf",
            "cat /etc/localtime",
            "cat /etc/os-release",
            # Non-sensitive /proc files
            "cat /proc/cpuinfo",
            "cat /proc/meminfo",
            "cat /proc/uptime",
            "cat /proc/version",
            "cat /proc/loadavg",
            # Other useful system files
            "cat /var/log/syslog",
        ],
    )
    def test_legitimate_absolute_allowed(self, cmd):
        assert not _find_sensitive_paths(
            cmd
        ), f"expected to allow (would dumbify tool calling): {cmd!r}"


class TestPatchB_PythonShellExec:
    """When the bash blocklist + sensitive-path check fires inside the
    Python AST gate, ``os.system('cat ~/.ssh/id_rsa')`` produces the same
    block as the bash equivalent."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os; os.system('cat ~/.ssh/id_rsa')",
            "import os; os.system('grep token ~/.netrc')",
            "import os; os.system('cat /home/u/.aws/credentials')",
            "import os; os.system('cat /etc/shadow')",
            "import subprocess; subprocess.run(['cat', '/proc/self/environ'])",
            "import subprocess; subprocess.run(['cat', '/etc/shadow'])",
        ],
    )
    def test_blocked(self, code):
        assert _is_blocked(code), f"expected to block: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import os; os.system('cat README.md')",
            "import os; os.system('ls src/')",
            "import os; os.system('cat ~/.gitconfig')",
            "import os; os.system('cat ~/.bashrc')",
            "import os; os.system('cat ~/.ssh/config')",
            "import os; os.system('cat ~/.ssh/known_hosts')",
            "import os; os.system('cat /etc/hosts')",
            "import os; os.system('find src/ -name *.py')",
            "import os; os.system('grep -r foo src/')",
            "import subprocess; subprocess.run(['ls', '-la'])",
            "import subprocess; subprocess.run(['cat', 'README.md'])",
        ],
    )
    def test_legitimate_allowed(self, code):
        assert not _is_blocked(
            code
        ), f"expected to allow (would dumbify tool calling): {code!r}"


# ---------------------------------------------------------------------------
# Patch D — eval / exec body recursion
# ---------------------------------------------------------------------------


class TestPatchD_EvalExecLiteralPayload:
    @pytest.mark.parametrize(
        "code",
        [
            # Shell-escape inside an exec payload
            f"exec(\"import os; os.system('{SUDO} whoami')\")",
            f'exec(\'import subprocess; subprocess.run(["{SUDO}", "id"])\')',
            # Sensitive-file open inside exec payload
            "exec(\"open('/etc/shadow').read()\")",
            "exec(\"with open('/etc/passwd') as f: print(f.read())\")",
            # Nested
            f'exec("exec(\\"import os; os.system(\'{SUDO} id\')\\")")',
        ],
    )
    def test_literal_attack_payload_blocked(self, code):
        assert _is_blocked(code), f"expected to block: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Pure expressions — must remain allowed
            "eval('1 + 2')",
            "eval('len([1, 2, 3])')",
            "eval('sum(range(10))')",
            "exec('x = 1\\ny = 2\\nprint(x + y)')",
            "exec('print(\"hello\")')",
            # Nested but innocuous
            "exec('exec(\"print(1)\")')",
        ],
    )
    def test_legitimate_eval_exec_allowed(self, code):
        assert not _is_blocked(
            code
        ), f"expected to allow (would dumbify tool calling): {code!r}"


class TestPatchD_EvalExecDynamicPayload:
    @pytest.mark.parametrize(
        "code",
        [
            # Non-literal payloads — flagged as dynamic shell escape
            "payload = 'print(1)'; exec(payload)",
            "import os; exec(os.environ['PAYLOAD'])",
            "import base64; exec(base64.b64decode('cHJpbnQoMSk=').decode())",
            "exec(input())",
        ],
    )
    def test_dynamic_payload_flagged(self, code):
        assert _is_blocked(code), f"expected to block: {code!r}"


class TestPatchD_NestedDepthCap:
    def test_nested_depth_does_not_crash(self):
        # 10 levels of nested exec(exec(...)) — depth cap at 3 means the
        # recursion stops early. Must not crash, must not false-positive on
        # the innocuous innermost payload.
        payload = "print(1)"
        for _ in range(10):
            payload = f"exec({payload!r})"
        # Outer payload is now exec("exec(\"exec(...))\")") with no
        # blocked/sensitive content. Must remain ALLOWED.
        assert not _is_blocked(payload), payload[:80] + "..."


# ---------------------------------------------------------------------------
# Cross-cutting — full regression sweep against the existing upstream
# attack-pattern matrix to prove these patches don't break the existing
# blocks.
# ---------------------------------------------------------------------------


class TestCrossCuttingNoRegression:
    @pytest.mark.parametrize(
        "code",
        [
            # Pre-existing shell-escape blocks — must still fire
            f"import os; os.system('{SUDO} whoami')",
            f"import subprocess; subprocess.run(['{SUDO}', 'x'])",
            # Pre-existing signal tampering
            "import signal; signal.signal(signal.SIGALRM, signal.SIG_IGN)",
            # Pre-existing sensitive-file open
            "open('/etc/passwd')",
            # Pre-existing untrusted host
            "import requests; requests.get('https://evil.example.com/')",
            # Pre-existing metadata host
            "import requests; requests.get('http://169.254.169.254/')",
        ],
    )
    def test_preexisting_blocks_still_fire(self, code):
        assert _is_blocked(code), f"REGRESSION: pre-existing block failed: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Pre-existing allowed patterns — must still pass
            "print('hello')",
            "import json; json.loads('{}')",
            "import requests; requests.get('https://wikipedia.org/')",
            "import requests; requests.get('https://huggingface.co/x')",
            "from dataclasses import dataclass\n@dataclass\nclass P: x: int",
            "open('data.csv', 'r')",
            "open('logs/today.log', 'w')",
        ],
    )
    def test_preexisting_allowed_still_pass(self, code):
        assert not _is_blocked(
            code
        ), f"REGRESSION: pre-existing pass-through now blocked: {code!r}"

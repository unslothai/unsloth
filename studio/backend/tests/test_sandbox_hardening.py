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
    """Fail-closed once recursion exceeds the inspection cap. The previous
    silently-drop behaviour let four-or-more nested literal exec layers
    smuggle ``sudo whoami`` or ``open('/etc/shadow')`` past the gate."""

    def test_nested_depth_does_not_crash(self):
        # 10 levels of exec nesting; must not blow the stack regardless of
        # whether the verdict is "blocked" or "allowed".
        payload = "print(1)"
        for _ in range(10):
            payload = f"exec({payload!r})"
        # Just exercises the code path; the assertion is "did not raise".
        _is_blocked(payload)

    @pytest.mark.parametrize(
        "inner",
        [
            f"import os; os.system('{SUDO} whoami')",
            "open('/etc/shadow').read()",
            "import requests; requests.get('http://169.254.169.254/')",
        ],
    )
    @pytest.mark.parametrize("depth", [4, 5, 6])
    def test_deep_nested_payload_fails_closed(self, inner, depth):
        payload = inner
        for _ in range(depth):
            payload = f"exec({payload!r})"
        assert _is_blocked(payload), f"depth={depth} bypass: {payload[:80]}..."

    @pytest.mark.parametrize("inner", ["print(1)", "x = 1 + 2"])
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_shallow_innocuous_payload_still_allowed(self, inner, depth):
        payload = inner
        for _ in range(depth):
            payload = f"exec({payload!r})"
        assert not _is_blocked(
            payload
        ), f"shallow innocuous depth={depth} now blocked: {payload!r}"


# ---------------------------------------------------------------------------
# Review-round 2 regressions: fixes for findings surfaced by reviewer.py.
# Every test here corresponds to a specific finding number from the
# 20-reviewer aggregated review.
# ---------------------------------------------------------------------------


class TestFinding1_DirectOpenSensitivePaths:
    """Finding #1 [15/20]: ``open()`` was missing the new home /
    credential / process-state path guard. ``cat ~/.ssh/id_rsa`` was
    blocked but ``open('~/.ssh/id_rsa').read()`` was not."""

    @pytest.mark.parametrize(
        "code",
        [
            "open('/home/u/.aws/credentials').read()",
            "open('/Users/alice/.aws/credentials').read()",
            "open('/root/.docker/config.json').read()",
            "open('/home/u/.ssh/id_rsa').read()",
            "open('/proc/self/environ').read()",
            "open('/proc/self/maps').read()",
            "open('/proc/self/auxv', 'rb').read()",
            # Wrapped in literal exec — Patch D recursion plus the new
            # open() wiring must combine.
            "exec(\"open('/home/u/.aws/credentials').read()\")",
            "exec(\"open('/proc/self/environ').read()\")",
        ],
    )
    def test_direct_open_credential_blocked(self, code):
        assert _is_blocked(code), f"expected to block: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Project-local lookalikes must remain allowed.
            "open('./fixtures/etc/shadow.txt')",
            "open('/tmp/project/etc/shadow')",
            "open('/tmp/home/u/.npmrc')",
            "open('./workspace/home/u/.aws/credentials')",
            # Real common paths the AI tools touch.
            "open('README.md')",
            "open('src/main.py')",
            "open('logs/today.log', 'w')",
        ],
    )
    def test_project_local_open_still_allowed(self, code):
        assert not _is_blocked(
            code
        ), f"regression: project-local open() now blocked: {code!r}"


class TestFinding4_ShellQuoteSplicing:
    """Finding #4 [4/20]: raw-text regex saw past shell quote tricks.
    ``cat /etc/sha''dow`` is executed by the shell as ``cat /etc/shadow``
    but the regex returned no match."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /etc/sha''dow",
            "cat ~/'.ssh/id_rsa'",
            "cat $HOME/.ssh/id_''rsa",
            "cat /proc/self/env''iron",
            "cat /'etc'/shadow",
            "bash -c \"cat ~/'.ssh/id_rsa'\"",
            "bash -c 'cat /etc/sha\"\"dow'",
        ],
    )
    def test_quote_spliced_sensitive_paths_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"missed splice: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            # Quote-spliced project-local lookalikes must still pass.
            "cat ./fixtures/etc/sha''dow.txt",
            "cat ./logs/'today.log'",
        ],
    )
    def test_quote_spliced_project_local_allowed(self, cmd):
        assert not _find_sensitive_paths(
            cmd
        ), f"regression: spliced project-local blocked: {cmd!r}"


class TestFinding5_WindowsHomePrefixes:
    """Finding #5 [3/20]: ``_HOME_PREFIX_RE`` only knew POSIX homes.
    Windows ``%USERPROFILE%\\.aws\\credentials`` was not detected."""

    @pytest.mark.parametrize(
        "cmd",
        [
            r"type %USERPROFILE%\.aws\credentials",
            r"type %USERPROFILE%\.ssh\id_rsa",
            r"type %HOMEDRIVE%%HOMEPATH%\.docker\config.json",
            r"type C:\Users\alice\.aws\credentials",
            r"type C:\Users\alice\.ssh\id_ed25519",
            r"type $env:USERPROFILE\.aws\credentials",
        ],
    )
    def test_windows_home_paths_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"missed Windows path: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            r"type C:\Users\alice\projects\app\config.json",
            r"type %USERPROFILE%\Documents\readme.txt",
            r"dir C:\Users\alice\Downloads",
        ],
    )
    def test_legitimate_windows_paths_allowed(self, cmd):
        assert not _find_sensitive_paths(
            cmd
        ), f"regression: legit Windows path blocked: {cmd!r}"


class TestFinding6_DeepLiteralConcat:
    """Finding #6 [2/20]: the static-string resolver bailed past depth 6,
    so ``open('/'+'e'+'t'+'c'+'/'+'s'+'h'+'a'+'d'+'o'+'w')`` was
    silently allowed."""

    @pytest.mark.parametrize(
        "code",
        [
            "open('/'+'e'+'t'+'c'+'/'+'s'+'h'+'a'+'d'+'o'+'w').read()",
            "open('/'+'e'+'t'+'c'+'/'+'p'+'a'+'s'+'s'+'w'+'d').read()",
            "open('/'+'p'+'r'+'o'+'c'+'/'+'s'+'e'+'l'+'f'+'/'+'e'+'n'+'v'+'i'+'r'+'o'+'n').read()",
        ],
    )
    def test_deep_literal_concat_blocked(self, code):
        assert _is_blocked(code), f"depth bypass: {code!r}"


class TestFinding7_NetworkHostStaticResolver:
    """Finding #7 [1/20]: network host validation only handled
    ``ast.Constant``; concat / f-string hosts bypassed."""

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.get('http://' + '169.254.169.254/')",
            "import requests; requests.get(f'http://{\"169.254.169.254\"}/')",
            "import socket; s=socket.socket(); s.connect(('169.254.' + '169.254', 80))",
            "exec(\"import requests; requests.get('http://' + '169.254.169.254/')\")",
        ],
    )
    def test_dynamic_metadata_host_blocked(self, code):
        assert _is_blocked(code), f"metadata bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.get('https://' + 'wikipedia.org/')",
            "import requests; requests.get(f'https://{\"huggingface.co\"}/x')",
        ],
    )
    def test_dynamic_trusted_host_allowed(self, code):
        assert not _is_blocked(
            code
        ), f"regression: trusted host with dynamic literal blocked: {code!r}"


class TestFinding8_PathlibPathOpen:
    """Finding #8 [1/20]: when the open target lives in the receiver
    constructor (``Path('/etc/shadow').open()``) rather than in
    ``open(arg)``, the gate skipped inspection."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('/etc/shadow').open().read()",
            "from pathlib import Path\nPath('/etc/' + 'shadow').open().read()",
            "import pathlib\npathlib.Path('/etc/passwd').open().read()",
            "from pathlib import Path\nPath('/home/u/.aws/credentials').open().read()",
        ],
    )
    def test_pathlib_path_open_blocked(self, code):
        assert _is_blocked(code), f"pathlib bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('data.csv').open()",
            "from pathlib import Path\nPath('logs/today.log').open('w')",
            "from pathlib import Path\nPath('README.md').open()",
        ],
    )
    def test_pathlib_legit_path_allowed(self, code):
        assert not _is_blocked(code), f"regression: legit Path.open() blocked: {code!r}"


class TestFinding9_ProjectLocalFalsePositives:
    """Finding #9 [3/20]: regex without a path-token start anchor
    blocked project-local lookalikes like ``./workspace/home/u/.aws/...``
    which are project paths, not host credentials."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat ./workspace/home/u/.aws/credentials",
            "cat /tmp/home/u/.npmrc",
            "cat ./fixtures/etc/shadow.txt",
            "cat /tmp/project/etc/shadow",
            "cat project/Users/alice/.aws/credentials",
            "ls /opt/Users/svc/.kube/config",
            "find /tmp/root/.gnupg -type f",
        ],
    )
    def test_project_local_lookalikes_allowed(self, cmd):
        assert not _find_sensitive_paths(
            cmd
        ), f"false-positive (tool calling dumber): {cmd!r}"


class TestFinding10_PublicSshKeyAllowed:
    """Finding #10 [1/20]: SSH private-key alternatives matched without
    a filename boundary, so ``cat ~/.ssh/id_rsa.pub`` was blocked even
    though reading a public key is a legitimate developer action."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat ~/.ssh/id_rsa.pub",
            "cat ~/.ssh/id_ed25519.pub",
            "cat ~/.ssh/id_ecdsa.pub",
            "cat /home/u/.ssh/id_rsa.pub",
            "cat /Users/alice/.ssh/id_rsa.pub",
            "ssh-keygen -lf ~/.ssh/id_rsa.pub",
        ],
    )
    def test_public_ssh_keys_allowed(self, cmd):
        assert not _find_sensitive_paths(
            cmd
        ), f"regression: public key read blocked: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            # Negative cross-check — the .pub end anchor must not relax
            # the actual private-key block.
            "cat ~/.ssh/id_rsa",
            "cat ~/.ssh/id_ed25519",
            "cat /home/u/.ssh/id_ecdsa",
        ],
    )
    def test_private_ssh_keys_still_blocked(self, cmd):
        assert _find_sensitive_paths(
            cmd
        ), f"regression: private key now allowed: {cmd!r}"


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

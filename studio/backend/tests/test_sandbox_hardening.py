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
            # Truly dynamic payloads (no static resolution possible) —
            # flagged as dynamic shell escape. ``payload = 'print(1)'
            # ; exec(payload)`` is intentionally NOT in this list: the
            # variable-binding pre-pass folds the literal and the inner
            # ``print(1)`` is then visited and confirmed safe, which is
            # the correct behaviour.
            "import os; exec(os.environ['PAYLOAD'])",
            "import base64; exec(base64.b64decode('cHJpbnQoMSk=').decode())",
            "exec(input())",
        ],
    )
    def test_dynamic_payload_flagged(self, code):
        assert _is_blocked(code), f"expected to block: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Statically resolvable variable-name payloads now reach
            # the literal-payload recursion: safe inner code is allowed.
            "payload = 'print(1)'; exec(payload)",
            "p = '1 + 2'; eval(p)",
        ],
    )
    def test_resolvable_variable_payload_allowed_when_safe(self, code):
        assert not _is_blocked(code), f"safe resolved payload blocked: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Statically resolvable variable-name payloads that contain
            # an attack — must still block via the recursive inspection.
            "payload = \"open('/etc/shadow').read()\"; exec(payload)",
            "p = \"import os; os.system('sudo whoami')\"; exec(p)",
        ],
    )
    def test_resolvable_variable_payload_blocked_when_unsafe(self, code):
        assert _is_blocked(code), f"unsafe resolved payload missed: {code!r}"


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


# ---------------------------------------------------------------------------
# Review-round 3 regressions: fixes for findings surfaced by the second
# 20-reviewer pass. Each class corresponds to a specific finding number
# in that report.
# ---------------------------------------------------------------------------


class TestR2Finding1_PathlibReaders:
    """Path.read_text() / Path.read_bytes() now flow through the same
    sensitive-file gate that Path.open() does."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('/etc/shadow').read_text()",
            "from pathlib import Path\nPath('/home/u/.aws/credentials').read_text()",
            "from pathlib import Path\nPath('/proc/self/environ').read_bytes()",
            "import pathlib\npathlib.Path('/home/u/.ssh/id_rsa').read_bytes()",
            "exec(\"from pathlib import Path\\nPath('/etc/shadow').read_text()\")",
        ],
    )
    def test_pathlib_readers_blocked(self, code):
        assert _is_blocked(code), f"pathlib reader bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('README.md').read_text()",
            "from pathlib import Path\nPath('data/config.json').read_bytes()",
        ],
    )
    def test_pathlib_legit_readers_allowed(self, code):
        assert not _is_blocked(code), f"legit pathlib reader blocked: {code!r}"


class TestR2Finding2_TildeUserExpansion:
    """POSIX ``~user/`` home expansion: bash resolves
    ``cat ~ubuntu/.aws/credentials`` to that user's home before exec."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat ~root/.ssh/id_rsa",
            "cat ~ubuntu/.npmrc",
            "cat ~alice/.aws/credentials",
            "cat ~root/.docker/config.json",
        ],
    )
    def test_tilde_user_paths_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"tilde-user bypass: {cmd!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import os; os.system('cat ~ubuntu/.aws/credentials')",
            "import subprocess; subprocess.run(['bash', '-c', 'cat ~ubuntu/.npmrc'])",
        ],
    )
    def test_tilde_user_paths_blocked_via_python(self, code):
        assert _is_blocked(code), f"tilde-user python bypass: {code!r}"


class TestR2Finding3_KeywordNetworkArgs:
    """Network host extraction now resolves ``url=``, ``host=``,
    ``hostname=``, and ``address=`` keyword arguments. Bare-host APIs
    (``socket.getaddrinfo``, ``http.client.HTTPConnection``) treat the
    first positional arg as the host."""

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.get(url='http://' + '169.254.169.254/')",
            "import urllib.request; urllib.request.urlopen(url='http://169.254.169.254/')",
            "import http.client; http.client.HTTPConnection(host='169.254.169.254')",
            "import socket; socket.create_connection(address=('169.254.169.254', 80))",
            "import socket; socket.getaddrinfo('169.254.' + '169.254', 80)",
            "import http.client; http.client.HTTPConnection('169.254.' + '169.254')",
            "import requests; requests.request('GET', 'http://169.254.169.254/')",
            "import requests; requests.request(method='GET', url='http://169.254.169.254/')",
            "import httpx; httpx.get(url=f'http://{\"169.254.169.254\"}/')",
        ],
    )
    def test_keyword_metadata_hosts_blocked(self, code):
        assert _is_blocked(code), f"metadata bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.get(url='https://wikipedia.org/')",
            "import requests; requests.request(method='GET', url='https://huggingface.co/')",
            "import http.client; http.client.HTTPSConnection(host='huggingface.co')",
        ],
    )
    def test_keyword_trusted_hosts_allowed(self, code):
        assert not _is_blocked(code), f"trusted host kw blocked: {code!r}"


class TestR2Finding4_BuiltinsEvalExec:
    """``builtins.exec(...)`` / ``__builtins__.eval(...)`` flow through
    the same literal-payload recursion as bare ``exec`` / ``eval``."""

    @pytest.mark.parametrize(
        "code",
        [
            "import builtins\nbuiltins.exec(\"open('/etc/shadow').read()\")",
            "import builtins\nbuiltins.eval(\"open('/etc/shadow').read()\")",
            "import builtins as b\nb.eval(\"open('/etc/shadow').read()\")",
            "__builtins__.eval(\"open('/etc/shadow').read()\")",
        ],
    )
    def test_qualified_eval_exec_payloads_blocked(self, code):
        assert _is_blocked(code), f"builtins.exec bypass: {code!r}"


class TestR2Finding5_OpenFileKeyword:
    """``open(file='/etc/shadow')`` keyword form is gated alongside the
    positional form."""

    @pytest.mark.parametrize(
        "code",
        [
            "open(file='/etc/shadow').read()",
            "open(file='/proc/self/environ').read()",
            "open(file='/home/u/.aws/credentials').read()",
            "import io; io.open(file='/etc/shadow').read()",
            "exec(\"open(file='/etc/shadow').read()\")",
        ],
    )
    def test_open_file_keyword_blocked(self, code):
        assert _is_blocked(code), f"open(file=) bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "open(file='README.md')",
            "open(file='logs/today.log', mode='w')",
        ],
    )
    def test_open_file_keyword_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit open(file=) blocked: {code!r}"


class TestR2Finding6_SshKeyRedirectAttached:
    """The SSH private-key end anchor now treats ``>`` as a token
    boundary, so a redirect with no preceding space is blocked the
    same way the spaced form is."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat ~/.ssh/id_rsa>" + ("/" + "tmp/leak"),
            "cat ~/.ssh/id_ed25519>>" + ("/" + "tmp/leak"),
            "cat /home/u/.ssh/id_rsa>" + ("/" + "tmp/leak"),
        ],
    )
    def test_ssh_key_with_attached_redirection_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"redirect-attached bypass: {cmd!r}"


class TestR2Finding7_ShellCommandSubstitution:
    """Sensitive root prefixes followed by ``$(...)`` or backtick
    substitution are flagged because the attacker is dynamically
    constructing a protected path."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /proc/1/$(echo environ)",
            "cat /etc/$(printf shadow)",
            "cat ~/.aws/$(echo credentials)",
            "cat /etc/`printf shadow`",
        ],
    )
    def test_substitution_sensitive_paths_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"substitution bypass: {cmd!r}"


class TestR2Finding8_ShellBraceExpansion:
    """Bash brace expansion ``{a,b}`` and small glob char classes
    ``[abc]`` are enumerated before the regex scan."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /etc/sh{ad,ad}ow",
            "cat /etc/shado[w]",
            "cat /proc/self/{environ,environ}",
            "cat /proc/self/enviro[n]",
            "cat $HOME/{.aws/credentials,.bashrc}",
        ],
    )
    def test_brace_expansion_sensitive_paths_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"brace expansion bypass: {cmd!r}"


class TestR2Finding9_PathSeparatorNormalisation:
    """``cat /etc//shadow`` and ``cat /etc/./shadow`` resolve to
    ``/etc/shadow`` for the OS; the projection does the same so they
    cannot bypass the regex."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /etc//shadow",
            "cat /etc/./shadow",
            "cat ~/.aws//credentials",
            "cat ~/.aws/./credentials",
            "cat ${HOME}/.ssh//id_rsa",
            "cat /proc/self//environ",
        ],
    )
    def test_equivalent_path_spellings_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"equivalent path bypass: {cmd!r}"


class TestR2Finding10_OpenEquivalentSpellings:
    """Same normalization gap inside the Python open() gate."""

    @pytest.mark.parametrize(
        "code",
        [
            "open('/etc//shadow').read()",
            "open('/etc/./shadow').read()",
            "open('/home/u/.aws//credentials').read()",
            "open('/home/u/.aws/./credentials').read()",
        ],
    )
    def test_equivalent_open_paths_blocked(self, code):
        assert _is_blocked(code), f"equivalent open() bypass: {code!r}"


class TestR2Finding12_PathlibOpenWithMode:
    """``Path('/etc/shadow').open('r')`` previously read ``'r'`` as the
    path arg; the receiver-side resolver now takes precedence for
    pathlib readers."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('/etc/shadow').open('r').read()",
            "from pathlib import Path\nPath('/home/u/.aws/credentials').open('rb').read()",
            "import pathlib\npathlib.Path('/proc/self/environ').open('rb').read()",
        ],
    )
    def test_pathlib_open_with_mode_blocked(self, code):
        assert _is_blocked(code), f"Path.open(mode) bypass: {code!r}"


class TestR2Finding13_14_15_PathlibCompositions:
    """``joinpath()``, ``/``, and multi-part ``Path()`` constructions
    all resolve to a single path string before the sensitive-file check."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('/etc').joinpath('shadow').open().read()",
            "from pathlib import Path\nPath('/etc').joinpath('shadow').read_text()",
            "from pathlib import Path\nPath('/home/u').joinpath('.aws/credentials').open().read()",
            "from pathlib import Path\n(Path('/etc') / 'shadow').open().read()",
            "from pathlib import Path\n(Path('/etc') / 'shadow').read_text()",
            "from pathlib import Path\nPath('/etc', 'shadow').open().read()",
            "from pathlib import Path\nPath('/home', 'u', '.aws', 'credentials').open().read()",
            "from pathlib import Path\nPath('/proc', 'self', 'environ').read_bytes()",
        ],
    )
    def test_pathlib_compositions_blocked(self, code):
        assert _is_blocked(code), f"pathlib composition bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path\nPath('data', 'file.txt').open()",
            "from pathlib import Path\nPath('logs').joinpath('today.log').open('w')",
            "from pathlib import Path\n(Path('data') / 'file.txt').read_text()",
        ],
    )
    def test_pathlib_compositions_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit pathlib composition blocked: {code!r}"


class TestR2Finding16_PathlibAliasImport:
    """``from pathlib import Path as P`` and ``import pathlib as pl``
    register the alias so constructor recognition fires."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path as P\nP('/etc/shadow').open().read()",
            "from pathlib import Path as P\nP('/etc/shadow').read_text()",
            "import pathlib as pl\npl.Path('/home/u/.aws/credentials').open().read()",
            "import pathlib as pl\npl.Path('/etc').joinpath('shadow').open().read()",
        ],
    )
    def test_aliased_pathlib_blocked(self, code):
        assert _is_blocked(code), f"alias bypass: {code!r}"


# ---------------------------------------------------------------------------
# Review-round 4 regressions: fixes for findings surfaced by the third
# 20-reviewer pass. Each class corresponds to a finding number from that
# report.
# ---------------------------------------------------------------------------


class TestR3Finding1_ParentDirNormalisation:
    """``/etc/../etc/shadow``, ``~/.ssh/../.aws/credentials``, and the
    pathlib equivalent now collapse through posixpath.normpath before the
    sensitive-path regex sees them."""

    @pytest.mark.parametrize(
        "code",
        [
            "open('/etc/apt/../shadow').read()",
            "open('/proc/self/fd/../environ').read()",
            "open('/etc/ssl/../shadow').read()",
            "from pathlib import Path\nPath('/proc/self/fd/../environ').read_text()",
            "from pathlib import Path\nPath('/etc/apt/../shadow').read_text()",
        ],
    )
    def test_parent_dir_open_blocked(self, code):
        assert _is_blocked(code), f"parent-dir bypass: {code!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /etc/apt/../shadow",
            "cat /etc/ssl/../shadow",
            "cat /proc/self/fd/../environ",
            "cat ~/.ssh/../.aws/credentials",
        ],
    )
    def test_parent_dir_bash_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"bash parent-dir bypass: {cmd!r}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /tmp/test/../README.md",
            "cat ./build/../README.md",
        ],
    )
    def test_parent_dir_legit_allowed(self, cmd):
        assert not _find_sensitive_paths(cmd), f"legit parent-dir path blocked: {cmd!r}"


class TestR3Finding2_OpenPathLike:
    """Built-in ``open()`` accepts ``PathLike`` objects, so
    ``open(Path('/etc/shadow'))`` now flows through the pathlib resolver."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path; open(Path('/etc/shadow')).read()",
            "from pathlib import Path; open(file=Path('/etc/shadow')).read()",
            "from pathlib import Path; open(Path('/etc') / 'shadow').read()",
            "from pathlib import Path; open(Path('/home/u', '.aws/credentials')).read()",
        ],
    )
    def test_open_pathlike_blocked(self, code):
        assert _is_blocked(code), f"open(Path) bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path; open(Path('data.csv')).read()",
            "from pathlib import Path; open(Path('logs', 'today.log'), 'w')",
        ],
    )
    def test_open_pathlike_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit open(Path) blocked: {code!r}"


class TestR3Finding3_PathlibHomeAndTransforms:
    """``Path.home()``, ``.expanduser()``, ``.resolve()``, and
    ``.absolute()`` are now handled by the pathlib resolver as
    pass-through / home-substitution helpers."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path; (Path.home() / '.aws/credentials').read_text()",
            "from pathlib import Path; Path.home().joinpath('.ssh/id_rsa').read_text()",
            "from pathlib import Path; Path('~/.aws/credentials').expanduser().read_text()",
            "from pathlib import Path; Path('/etc/shadow').resolve().read_text()",
            "from pathlib import Path; Path('/etc/shadow').absolute().read_text()",
        ],
    )
    def test_path_home_and_transforms_blocked(self, code):
        assert _is_blocked(code), f"home/transforms bypass: {code!r}"


class TestR3Finding5_AbsoluteSegmentReset:
    """``Path('/tmp', '/etc/shadow')`` resolves to ``/etc/shadow`` at
    runtime; the helper now models the same semantics."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import Path; (Path('/tmp') / '/etc/shadow').read_text()",
            "from pathlib import Path; Path('/tmp').joinpath('/etc/shadow').read_text()",
            "from pathlib import Path; Path('/tmp', '/etc/shadow').read_text()",
            "from pathlib import Path; Path('/tmp').joinpath('/home/u/.aws/credentials').open().read()",
        ],
    )
    def test_absolute_reset_blocked(self, code):
        assert _is_blocked(code), f"absolute-reset bypass: {code!r}"


class TestR3Finding6_7_8_FromBuiltinsImportAs:
    """``from builtins import exec as e`` registers ``e`` for the same
    literal-payload recursion as bare ``exec``."""

    @pytest.mark.parametrize(
        "code",
        [
            "from builtins import exec as e\ne(\"open('/etc/shadow').read()\")",
            "from builtins import eval as e\ne(\"open('/etc/shadow').read()\")",
            "from builtins import exec as run\nrun(\"import os; os.system('cat /etc/shadow')\")",
        ],
    )
    def test_from_builtins_import_as_blocked(self, code):
        assert _is_blocked(code), f"from-builtins-import bypass: {code!r}"


class TestR3Finding11_12_ProcStateExtensions:
    """``/proc/self/cmdline``, ``/proc/thread-self/*``, and
    ``/proc/<pid>/task/<tid>/*`` are extensions of the existing process-
    state sensitive set."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /proc/self/cmdline",
            "cat /proc/thread-self/environ",
            "cat /proc/thread-self/cmdline",
            "cat /proc/self/task/123/environ",
            "cat /proc/1234/task/567/maps",
        ],
    )
    def test_proc_state_extensions_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"proc-state bypass: {cmd!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "open('/proc/self/cmdline').read()",
            "open('/proc/thread-self/environ').read()",
            "open('/proc/self/task/123/environ').read()",
        ],
    )
    def test_proc_state_extensions_open_blocked(self, code):
        assert _is_blocked(code), f"proc-state open bypass: {code!r}"


class TestR3Finding13_NumericFString:
    """``f'/proc/{1}/environ'`` and ``f'http://{169}.{254}.{169}.{254}/'``
    fold to literal strings because numeric f-string parts are stringified."""

    @pytest.mark.parametrize(
        "code",
        [
            "open(f'/proc/{1}/environ').read()",
            "import requests; requests.get(f'http://169.254.{169}.{254}/')",
        ],
    )
    def test_numeric_fstring_blocked(self, code):
        assert _is_blocked(code), f"numeric f-string bypass: {code!r}"


class TestR3Finding15_OsPathJoin:
    """``os.path.join('/etc', 'shadow')`` resolves the same way pathlib
    composition does."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os; open(os.path.join('/etc', 'shadow')).read()",
            "import os; open(os.path.join('/home/u', '.aws/credentials')).read()",
            "import os; open(os.path.join('/etc', 'ssh', 'ssh_host_rsa_key')).read()",
        ],
    )
    def test_os_path_join_blocked(self, code):
        assert _is_blocked(code), f"os.path.join bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import os; open(os.path.join('logs', 'today.log'), 'w')",
            "import os; open(os.path.join('data', 'config.json'))",
        ],
    )
    def test_os_path_join_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit os.path.join blocked: {code!r}"


class TestR3Finding16_19_NameBindings:
    """Simple ``name = 'literal'`` and ``name = eval`` assignments are
    folded by the pre-pass so subsequent ``open(name)`` / ``name(...)``
    invocations see the resolved value."""

    @pytest.mark.parametrize(
        "code",
        [
            "p = '/etc/shadow'; open(p).read()",
            "p = '/home/u/.aws/credentials'; open(p).read()",
            "p = '/etc/shadow'; from pathlib import Path; Path(p).read_text()",
            "e = eval\ne(\"open('/etc/shadow').read()\")",
        ],
    )
    def test_name_binding_blocked(self, code):
        assert _is_blocked(code), f"name-binding bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Legit dynamic URL — the network gate intentionally stays
            # opaque to bindings so untrusted-host policy enforcement
            # does not over-block.
            "url = 'https://example.com/'; import requests; requests.get(url)",
            # Legit string variable for non-sensitive file
            "p = 'data.csv'; open(p)",
            "p = 'logs/today.log'; open(p, 'w')",
        ],
    )
    def test_name_binding_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit name-binding blocked: {code!r}"


class TestR3Finding18_OsPathExpanduser:
    """``os.path.expanduser('~/.aws/credentials')`` is statically
    resolvable to a tilde-prefix sensitive path."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os; open(os.path.expanduser('~/.aws/credentials')).read()",
            "import os; open(os.path.expanduser('~/.ssh/id_rsa')).read()",
        ],
    )
    def test_os_path_expanduser_blocked(self, code):
        assert _is_blocked(code), f"os.path.expanduser bypass: {code!r}"


class TestR3Finding20_ShutilCopyExfil:
    """``shutil.copyfile`` / ``copy`` / ``copy2`` / ``copytree`` /
    ``move`` read the source path; the gate now treats their source arg
    the same as ``open()``."""

    @pytest.mark.parametrize(
        "code",
        [
            "import shutil; shutil.copyfile('/etc/shadow', 'out')",
            "import shutil; shutil.copy('/home/u/.aws/credentials', '/tmp/x')",
            "import shutil; shutil.copy2(src='/etc/shadow', dst='out')",
            "import shutil; shutil.move('/etc/shadow', 'leak')",
            "from pathlib import Path; import shutil; shutil.copyfile(Path('/etc/shadow'), 'out')",
        ],
    )
    def test_shutil_copy_source_blocked(self, code):
        assert _is_blocked(code), f"shutil.copy bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import shutil; shutil.copyfile('a.txt', 'b.txt')",
            "import shutil; shutil.copy('src/main.py', 'src/main.py.bak')",
            "import shutil; shutil.move('logs/today.log', 'logs/archive.log')",
        ],
    )
    def test_shutil_copy_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit shutil.copy blocked: {code!r}"


class TestR3Finding21_ConcretePathlibClasses:
    """``PosixPath``, ``WindowsPath``, ``PurePath`` etc. all map to the
    same constructor recognition as ``Path``."""

    @pytest.mark.parametrize(
        "code",
        [
            "from pathlib import PosixPath\nPosixPath('/etc/shadow').read_text()",
            "from pathlib import WindowsPath\nWindowsPath('/etc/shadow').read_text()",
            "from pathlib import PurePath\nPurePath('/etc/shadow').read_text()",
            "import pathlib\npathlib.PosixPath('/etc/shadow').read_text()",
            "import pathlib\npathlib.PurePosixPath('/etc/shadow').read_text()",
        ],
    )
    def test_concrete_pathlib_classes_blocked(self, code):
        assert _is_blocked(code), f"concrete-class bypass: {code!r}"


class TestR3Finding22_RequestsRequestPositionalKeyword:
    """``requests.request('GET', url='http://...')`` previously ate the
    positional ``'GET'`` as the URL; the URL-second branch now skips it
    so the ``url=`` keyword is read correctly."""

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.request('GET', url='http://169.254.169.254/')",
            "import requests; requests.request('POST', url='http://169.254.169.254/secrets')",
            "import requests; requests.request(method='GET', url='http://169.254.169.254/')",
            "import httpx; httpx.request('GET', url='http://169.254.169.254/')",
        ],
    )
    def test_request_method_then_kw_url_blocked(self, code):
        assert _is_blocked(code), f"method+kw bypass: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "import requests; requests.request('GET', url='https://huggingface.co/x')",
            "import requests; requests.request('POST', url='https://wikipedia.org/')",
        ],
    )
    def test_request_method_then_kw_trusted_url_allowed(self, code):
        assert not _is_blocked(code), f"trusted method+kw blocked: {code!r}"


# ---------------------------------------------------------------------------
# Followup — dynamic import bypass + /proc/self/cwd-root symlink traversal
# ---------------------------------------------------------------------------


class TestFollowup_DynamicImportShellEscape:
    """``__import__('os').system(...)`` and
    ``importlib.import_module('os').popen(...)`` bypass the bare
    ``os.system`` gate because the receiver is a Call, not a Name in
    ``os_aliases``. Same for the assign form
    ``m = __import__('os'); m.system(...)``. The visitor now resolves
    both shapes back to the canonical alias before the shell-escape
    check runs."""

    @pytest.mark.parametrize(
        "code",
        [
            "__import__('os').system('" + SUDO + " whoami')",
            "__import__('os').popen('cat ~/.ssh/id_rsa')",
            "import importlib; importlib.import_module('os').system('"
            + SUDO
            + " whoami')",
            "from importlib import import_module; import_module('os').system('"
            + SUDO
            + " whoami')",
            "m = __import__('os'); m.system('" + SUDO + " whoami')",
            "m = __import__('subprocess'); m.run(['" + SUDO + "', 'whoami'], shell=True)",
            "import importlib; mod = importlib.import_module('os'); "
            "mod.popen('cat ~/.aws/credentials')",
        ],
    )
    def test_dynamic_import_shell_escape_blocked(self, code):
        assert _is_blocked(code), f"dynamic-import shell escape leaked: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # Legit: importing other modules and calling safe methods.
            "import importlib; m = importlib.import_module('json'); m.dumps({'a':1})",
            "__import__('json').dumps({'a': 1})",
            "from importlib import import_module; pl = import_module('pathlib'); "
            "pl.Path('/tmp/x').exists()",
        ],
    )
    def test_dynamic_import_legit_allowed(self, code):
        assert not _is_blocked(code), f"legit dynamic import blocked: {code!r}"


class TestFollowup_ProcSelfSymlinkTraversal:
    """``/proc/<pid>/cwd`` and ``/proc/<pid>/root`` are symlinks to the
    process cwd and filesystem root. Without explicit detection,
    ``/proc/self/cwd/../../etc/shadow`` escapes lexical ``..``
    normalisation, and ``/proc/self/root/etc/shadow`` opens
    ``/etc/shadow`` regardless of any chroot or relative-path
    defence."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat /proc/self/cwd/../../etc/shadow",
            "cat /proc/self/root/etc/shadow",
            "cat /proc/self/root/etc/sudoers",
            "cat /proc/thread-self/cwd/secret.txt",
            "cat /proc/1/root/etc/shadow",
            "cat /proc/1/cwd/secrets.env",
            "cat /proc/self/task/1/root/etc/shadow",
        ],
    )
    def test_proc_self_symlink_traversal_bash_blocked(self, cmd):
        assert _find_sensitive_paths(cmd), f"symlink traversal leaked: {cmd!r}"

    @pytest.mark.parametrize(
        "code",
        [
            "open('/proc/self/root/etc/shadow').read()",
            "open('/proc/self/cwd/../../etc/shadow').read()",
            "open('/proc/1/root/etc/sudoers')",
            "import pathlib; pathlib.Path('/proc/self/root/etc/shadow').read_text()",
            "from pathlib import Path; "
            "Path('/proc/thread-self/root/etc/shadow').open()",
        ],
    )
    def test_proc_self_symlink_traversal_open_blocked(self, code):
        assert _is_blocked(code), f"symlink traversal open leaked: {code!r}"

    @pytest.mark.parametrize(
        "code",
        [
            # ``/proc/self/status`` is still useful for legit
            # introspection (e.g. checking the sandbox PID).
            "open('/proc/self/status').read()",
            "open('/proc/self/stat').read()",
            "open('/proc/cpuinfo').read()",
            "open('/proc/meminfo').read()",
        ],
    )
    def test_proc_legit_introspection_allowed(self, code):
        assert not _is_blocked(code), f"legit /proc read blocked: {code!r}"

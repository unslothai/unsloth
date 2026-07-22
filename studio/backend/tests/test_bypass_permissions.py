# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Bypass Permissions (skip confirmation + disable sandbox).

Covers the secret-name classifier, the two env builders, the
``disable_sandbox`` branch of ``_python_exec`` / ``_bash_exec`` (which env is
used, which pre-exec is used, and that safety checks / the blocklist are
skipped), the request-model default, the confirm-vs-bypass precedence rule the
route enforces, and that the agentic loop forwards ``disable_sandbox`` while
never gating under bypass.

Run with: ``PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_bypass_permissions.py -q``
"""

import io
import os
import sys

import pytest

import core.inference.tools as tools
from core.inference.tools import (
    _bash_exec,
    _build_bypass_env,
    _build_safe_env,
    _is_cred_location_env_name,
    _is_secret_env_name,
    _is_secret_env_value,
    _python_exec,
)
from core.inference.safetensors_agentic import run_safetensors_tool_loop

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason = "preexec_fn / setsid are POSIX-only"
)


# ── secret-name classifier ──────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "MY_DB_PASSWORD",
        "x_api_key",
        "SOME_PRIVATE_KEY",
        "LD_PRELOAD",
    ],
)
def test_secret_names_are_flagged(name):
    assert _is_secret_env_name(name) is True


@pytest.mark.parametrize(
    "name", ["PATH", "HOME", "LANG", "TERM", "PWD", "SHELL", "HOSTVAR", "MY_VAR"]
)
def test_benign_names_are_not_flagged(name):
    assert _is_secret_env_name(name) is False


# ── env builders ────────────────────────────────────────────────────


def test_bypass_env_keeps_benign_strips_secret_repoints_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOSTVAR", "benign-123")
    monkeypatch.setenv("HF_TOKEN", "secret-abc")
    env = _build_bypass_env(str(tmp_path))
    assert env.get("HOSTVAR") == "benign-123"  # full host env inherited
    assert "HF_TOKEN" not in env  # ...minus secrets
    assert env["HOME"] == str(tmp_path)  # $HOME-based cred lookups defused
    assert env["TMPDIR"] == str(tmp_path)


def test_safe_env_excludes_host_and_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("HOSTVAR", "benign-123")
    monkeypatch.setenv("HF_TOKEN", "secret-abc")
    env = _build_safe_env(str(tmp_path))
    assert "HOSTVAR" not in env  # whitelist build -> host vars never reach child
    assert "HF_TOKEN" not in env


# ── Popen kwargs capture (no real execution) ────────────────────────


class _FakeProc:
    """A subprocess.Popen double for the drain path (``tools._drain_process_output``):
    a readable ``stdout`` pipe yielding the fake output then EOF, plus
    ``wait()`` / ``poll()`` / ``pid``. The pid is non-existent so
    ``_capture_process_group``'s ``os.getpgid`` returns None; ``wait`` returns
    immediately so the drain never kills.
    """

    returncode = 0
    # Unlikely-to-exist pid: os.getpgid raises ProcessLookupError (caught) -> None.
    pid = 2**22

    def __init__(self):
        # Readable stdout: iter(readline, "") yields "FAKEOUT" then hits EOF.
        self.stdout = io.StringIO("FAKEOUT")

    def wait(self, timeout = None):
        return 0

    def poll(self):
        return 0

    def kill(self):
        pass


@pytest.fixture
def captured_popen(monkeypatch):
    cap = {}

    def fake_popen(cmd, **kwargs):
        cap["cmd"] = cmd
        cap["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(tools.subprocess, "Popen", fake_popen)
    return cap


@_POSIX_ONLY
def test_python_sandboxed_uses_sandbox_preexec_and_safe_env(captured_popen, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-abc")
    _python_exec("print(1)", None, 5, "t", disable_sandbox = False)
    assert captured_popen["kwargs"]["preexec_fn"] is tools._sandbox_preexec
    assert "HF_TOKEN" not in captured_popen["kwargs"]["env"]


@_POSIX_ONLY
def test_python_bypass_uses_bypass_preexec_and_bypass_env(captured_popen, monkeypatch):
    monkeypatch.setenv("HOSTVAR", "benign-xyz")
    monkeypatch.setenv("HF_TOKEN", "secret-abc")
    _python_exec("print(1)", None, 5, "t", disable_sandbox = True)
    assert captured_popen["kwargs"]["preexec_fn"] is tools._bypass_preexec
    env = captured_popen["kwargs"]["env"]
    assert env.get("HOSTVAR") == "benign-xyz"
    assert env.get("PYTHONIOENCODING") == "utf-8"
    assert "HF_TOKEN" not in env


def test_bash_blocklist_enforced_when_sandboxed(captured_popen):
    out = _bash_exec("rm -rf /", None, 5, "t", disable_sandbox = False)
    assert "Blocked" in out
    assert "cmd" not in captured_popen  # never reached Popen


@pytest.mark.parametrize(
    "command",
    [
        'python -S -c "import boto3"',
        'python -E -c "import boto3"',
        'python -I -c "import boto3"',
        'python --no-site -c "import boto3"',
        'python --ignore-environment -c "import boto3"',
        'python --isolated -c "import boto3"',
        'env -u PYTHONPATH python -c "import boto3"',
        'env --unset=UNSLOTH_STUDIO_SANDBOXED python3 -c "import boto3"',
        'env -i python -c "import boto3"',
        'PYTHONPATH= python -c "import boto3"',
        'unset PYTHONPATH; python -c "import boto3"',
        'export UNSLOTH_STUDIO_SANDBOXED=0; python -c "import boto3"',
        'uv run python -S -c "import boto3"',
        'bash -lc "python -I -c import\\ boto3"',
        'env -S "python -S -c import\\ boto3"',
        'env --split-string="python -I -c import\\ boto3"',
        'env -u PYTHONPATH sh -c "python -c import\\ boto3"',
        'command sh -c "python -I -c import\\ boto3"',
        'find . -exec python -S -c "import boto3" ;',
        # A find/fd -exec that hides the interpreter behind a nested shell must
        # still be recursed into, not left as an opaque exec target.
        'find . -exec sh -c "python -S -c import\\ boto3" ;',
        'find . -type f -execdir bash -c "python -I -c import\\ boto3" ;',
        'python$IFS-S -c "import boto3"',
        "python -c \"import subprocess; subprocess.run(['python','-S','-c','import boto3'])\"",
        'python -c "import os; os.system(\\"python -S -c \'import boto3\'\\")"',
        'echo ok\npython -S -c "import boto3"',
        'timeout 1 env -i python -c "import boto3"',
        'find . -exec env -i python -c "import boto3" ;',
        "bash <<'EOF'\npython -S -c \"import boto3\"\nEOF",
        "bash <<< 'python -S -c \"import boto3\"'",
        'if true; then python -S -c "import boto3"; fi',
        'for x in 1; do python -S -c "import boto3"; done',
        "python <<'PY'\nimport subprocess\nsubprocess.run(['python','-S','-c','import boto3'])\nPY",
        'python \\\n-S -c "import boto3"',
        'env -uPYTHONPATH python -c "import boto3"',
        (
            "python -c \"import os,subprocess; os.environ.pop('PYTHONPATH',None); "
            "os.environ['UNSLOTH_STUDIO_SANDBOXED']='0'; "
            "subprocess.run(['python','-c','import boto3'])\""
        ),
        # shell=True passes a sequence's first element to /bin/sh -c, so the
        # embedded ``python -S`` is shell input, not a shlex-joined argv word.
        'python -c "import subprocess; subprocess.run([\'python -S -c \\"import boto3\\"\'], shell=True)"',
        # A launcher rebound by assignment (``r = subprocess.run``) still spawns
        # an unguarded child.
        "python -c \"import subprocess; r = subprocess.run; r(['python','-S','-c','import boto3'])\"",
        # A command word supplied entirely by a defaulted parameter expansion
        # (``${PYTHON:-python}`` with PYTHON unset) still runs ``python -S``.
        '${PYTHON:-python} -S -c "import boto3"',
        # A quoted here-doc delimiter containing a hyphen is still a here-doc; its
        # Python body must be parsed as stdin code.
        "python <<'PY-EOF'\nimport subprocess\nsubprocess.run(['python','-S','-c','import boto3'])\nPY-EOF",
        # argv elements assembled from concatenated literals fold to ``python``.
        "python -c \"import subprocess; subprocess.run(['py'+'thon','-S','-c','import boto3'])\"",
        # ``os.environ |= {...}`` (PEP 584) can clear PYTHONPATH for the child.
        (
            "python -c \"import os,subprocess; os.environ |= {'PYTHONPATH': ''}; "
            "subprocess.run(['python','-c','import boto3'])\""
        ),
        # GNU env's lone ``-`` implies ``-i`` (clear environment).
        'env - /usr/bin/python -c "import boto3"',
        # A child launch hidden inside a static ``exec`` string payload.
        (
            "python -c \"exec('import subprocess; "
            'subprocess.run([\\"python\\",\\"-S\\",\\"-c\\",\\"import boto3\\"])\')"'
        ),
        # Non-subprocess child launchers (pty.spawn / asyncio) skip sitecustomize
        # in the child too.
        "python -c \"import pty; pty.spawn(['python','-S','-c','import boto3'])\"",
        (
            'python -c "import asyncio; '
            "asyncio.create_subprocess_exec('python','-S','-c','import boto3')\""
        ),
    ],
)
def test_bash_blocks_python_startup_guard_bypasses(captured_popen, command):
    out = _bash_exec(command, None, 5, "t", disable_sandbox = False)
    assert "cannot disable the Studio runtime guard" in out
    assert "cmd" not in captured_popen


@pytest.mark.parametrize(
    "command",
    [
        'python -c "print(1)"',
        "python script.py -S",
        "echo python -S",
        "python -c \"print('-S')\"",
        "python -c \"import subprocess; subprocess.run(['python','-c','print(1)'])\"",
        "python <<'PY'\nprint(1)\nPY",
        'bash <<< "echo safe"',
        "cat <<< 'python -S -c \"import boto3\"'",
        "echo then python -S",
        'python \\\n-c "print(1)"',
        # A guard-neutral env merge (non-guard key) must still auto-run.
        "python -c \"import os; os.environ |= {'MYVAR': '1'}\"",
        # A dynamic argv element (sys.executable) is not a foldable literal, so a
        # legitimate self-relaunch is not misclassified as a bypass.
        "python -c \"import sys, subprocess; subprocess.run([sys.executable, '-c', 'print(1)'])\"",
        # shell=True with an entirely benign script.
        "python -c \"import subprocess; subprocess.run(['echo hi'], shell=True)\"",
        # A rebound launcher that spawns a guarded (no -S) child.
        "python -c \"import subprocess; r = subprocess.run; r(['python','-c','print(1)'])\"",
    ],
)
def test_bash_allows_python_without_startup_guard_bypass(captured_popen, command):
    out = _bash_exec(command, None, 5, "t", disable_sandbox = False)
    assert out == "FAKEOUT"
    assert "cmd" in captured_popen


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        ('py -3.12 -I -c "import boto3"', True),
        ('C:\\Python312\\python.exe -S -c "import boto3"', True),
        ('set PYTHONPATH= & python -c "import boto3"', True),
        ('cmd /c "python -E -c import boto3"', True),
        ('python.exe -c "print(1)"', False),
        ("echo python -S", False),
    ],
)
def test_python_startup_guard_windows_command_parsing(monkeypatch, command, blocked):
    monkeypatch.setattr(tools.sys, "platform", "win32")
    assert tools._sandbox_python_startup_bypasses_guard(command) is blocked


def test_bash_blocklist_skipped_when_bypassed(captured_popen):
    out = _bash_exec("rm -rf /", None, 5, "t", disable_sandbox = True)
    assert out == "FAKEOUT"  # blocklist skipped -> reached (faked) execution
    assert captured_popen["cmd"][0] in ("bash", "cmd")


@_POSIX_ONLY
def test_bash_bypass_uses_bypass_preexec(captured_popen, monkeypatch):
    # bypass inherits benign host vars; clear so we assert _bash_exec adds none.
    monkeypatch.delenv("PYTHONIOENCODING", raising = False)
    _bash_exec("echo hi", None, 5, "t", disable_sandbox = True)
    assert captured_popen["kwargs"]["preexec_fn"] is tools._bypass_preexec
    assert "PYTHONIOENCODING" not in captured_popen["kwargs"]["env"]


# ── real end-to-end python execution under bypass ───────────────────


@_POSIX_ONLY
def test_python_bypass_real_exec_sees_host_env_but_not_secret(monkeypatch):
    monkeypatch.setenv("HOSTVAR", "benign-xyz")
    monkeypatch.setenv("HF_TOKEN", "secret-pqr")
    code = (
        "import os;"
        "print('H=' + str(os.environ.get('HOSTVAR')),"
        " 'T=' + str(os.environ.get('HF_TOKEN')))"
    )
    out = _python_exec(code, None, 30, "test-bypass", disable_sandbox = True)
    assert "H=benign-xyz" in out  # unrestricted: real host var visible
    assert "T=None" in out  # ...but the secret was stripped
    assert "secret-pqr" not in out


# ── _bypass_preexec is setsid-only (no rlimits) ─────────────────────


@_POSIX_ONLY
def test_bypass_preexec_only_sets_session(monkeypatch):
    calls = {"setsid": 0}
    monkeypatch.setattr(
        tools.os, "setsid", lambda: calls.__setitem__("setsid", calls["setsid"] + 1)
    )
    # _resource must not be touched by the bypass pre-exec.
    if tools._resource is not None:
        monkeypatch.setattr(
            tools._resource,
            "setrlimit",
            lambda *a, **k: pytest.fail("bypass pre-exec must not set rlimits"),
        )
    tools._bypass_preexec()
    assert calls["setsid"] == 1


# ── request model default ───────────────────────────────────────────


def test_request_model_bypass_default_false():
    from models.inference import ChatCompletionRequest
    assert ChatCompletionRequest.model_fields["bypass_permissions"].default is False


# ── confirm-vs-bypass precedence (mirrors the route rule) ───────────


@pytest.mark.parametrize(
    "confirm,bypass,effective_confirm",
    [
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, True, False),
    ],
)
def test_confirm_precedence_rule(confirm, bypass, effective_confirm):
    # The route computes: confirm_tool_calls = confirm and not bypass.
    assert (bool(confirm) and not bool(bypass)) is effective_confirm


# ── agentic loop forwards disable_sandbox, never gates under bypass ──

_DEFAULT_TOOLS = [
    {"type": "function", "function": {"name": "python"}},
    {"type": "function", "function": {"name": "web_search"}},
]


def _tool_call(name, args_json):
    return f'<tool_call>{{"name": "{name}", "arguments": {args_json}}}</tool_call>'


def _multi_turn(turns):
    it = iter(turns)

    def _gen(_messages):
        try:
            yield next(it)
        except StopIteration:
            return

    return _gen


def test_loop_forwards_disable_sandbox_and_does_not_gate():
    seen = []

    def fake_exec(
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        seen.append(disable_sandbox)
        return f"RAN[{name}]"

    events = list(
        run_safetensors_tool_loop(
            single_turn = _multi_turn([_tool_call("python", '{"code": "x"}'), "done"]),
            messages = [{"role": "user", "content": "hi"}],
            tools = _DEFAULT_TOOLS,
            execute_tool = fake_exec,
            session_id = "s",
            confirm_tool_calls = False,  # route forces this off under bypass
            bypass_permissions = True,
        )
    )
    assert seen == [True]  # disable_sandbox threaded through
    starts = [e for e in events if e["type"] == "tool_start"]
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert starts[0]["approval_id"] == ""


def test_loop_bypass_overrides_confirm_for_direct_callers():
    # Even if a direct internal caller passes confirm_tool_calls=True, bypass
    # must suppress the confirm gate at the loop level (not only at the route).
    def fake_exec(
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        return f"RAN[{name}]"

    events = list(
        run_safetensors_tool_loop(
            single_turn = _multi_turn([_tool_call("python", '{"code": "x"}'), "done"]),
            messages = [{"role": "user", "content": "hi"}],
            tools = _DEFAULT_TOOLS,
            execute_tool = fake_exec,
            session_id = "s",
            confirm_tool_calls = True,  # raw caller leaves this on...
            bypass_permissions = True,  # ...but bypass must still win
        )
    )
    starts = [e for e in events if e["type"] == "tool_start"]
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert starts[0]["approval_id"] == ""


def test_gguf_loop_confirm_gate_respects_bypass():
    # The GGUF loop needs a live llama-server, so (per the other llama_cpp
    # tests) assert via AST that its _needs_confirm gate applies the bypass
    # precedence, mirroring the safetensors behavioral test above.
    import ast
    import inspect
    import textwrap

    llama_cpp = pytest.importorskip("core.inference.llama_cpp")
    src = textwrap.dedent(
        inspect.getsource(llama_cpp.LlamaCppBackend.generate_chat_completion_with_tools)
    )
    gates = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "needs_confirm" for t in node.targets)
    ]
    assert gates, "could not find the needs_confirm gate in the GGUF loop"
    names = {n.id for g in gates for n in ast.walk(g.value) if isinstance(n, ast.Name)}
    assert "confirm_tool_calls" in names
    assert "bypass_permissions" in names  # bypass must suppress the GGUF gate


# ── broker / capability env vars are stripped (regression) ──────────


@pytest.mark.parametrize(
    "name",
    ["SSH_AUTH_SOCK", "SSH_AGENT_PID", "GPG_AGENT_INFO", "GNUPGHOME", "KUBECONFIG"],
)
def test_broker_capability_names_are_flagged(name):
    # Not secrets by value, but they hand the child the operator's live agent
    # (ssh/gpg) or kube credentials, so bypass mode must drop them.
    assert _is_secret_env_name(name) is True


# ── credential-bearing URL values stripped regardless of name ───────


@pytest.mark.parametrize(
    "value",
    [
        "https://user:s3cr3t@feed.example.invalid/simple",  # user:pass@
        "https://ghp_deadbeef@github.com/org/private.git",  # token-only@
        "https://__token__@pypi.example.invalid/simple",
        "https://ghp_1234:@npm.pkg.github.com/simple",  # empty password
        "postgres://dbuser:dbpass@db.example.invalid/app",
    ],
)
def test_url_userinfo_values_are_flagged(value):
    assert _is_secret_env_value(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/simple",  # no userinfo
        "http://proxy.corp.example:8080",  # benign proxy
        "https://pypi.corp.example/simple",  # benign internal index
        "redis://localhost:6379/0",  # no creds
        "https://example.invalid/path?ref=a@b",  # '@' only in query, not userinfo
    ],
)
def test_non_credential_url_values_are_not_flagged(value):
    assert _is_secret_env_value(value) is False


def test_url_userinfo_value_is_stripped_even_with_benign_name(monkeypatch, tmp_path):
    # NAME dodges the classifier, but the VALUE embeds userinfo -> must go.
    monkeypatch.setenv("MY_FEED", "https://user:s3cr3t@feed.example.invalid/simple")
    monkeypatch.setenv("REPO_URL", "https://ghp_deadbeef@github.com/org/private.git")
    # A URL without credentials is harmless and should be kept.
    monkeypatch.setenv("PLAIN_URL", "https://example.invalid/simple")
    env = _build_bypass_env(str(tmp_path))
    assert "MY_FEED" not in env
    assert "REPO_URL" not in env
    assert env.get("PLAIN_URL") == "https://example.invalid/simple"


def test_bypass_env_keeps_noncredential_proxy_and_index_urls(monkeypatch, tmp_path):
    # Benign routing/config vars must survive bypass mode (proxy-only or
    # internal-index networks); only credentialed values are dropped.
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp.example:8080")
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.corp.example/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://user:token@pypi.example.invalid/simple")
    env = _build_bypass_env(str(tmp_path))
    assert env["HTTP_PROXY"] == "http://proxy.corp.example:8080"
    assert env["PIP_INDEX_URL"] == "https://pypi.corp.example/simple"
    assert "PIP_EXTRA_INDEX_URL" not in env  # this one carries credentials


# ── AWS IMDS-disable hardening flag is kept (regression) ────────────


def test_aws_imds_disable_flag_is_kept_but_creds_stripped(monkeypatch, tmp_path):
    # AWS_EC2_METADATA_DISABLED is a non-secret opt-out: dropping it would let a
    # bypassed boto/AWS-CLI call fall back to the instance role via IMDS even
    # though the operator disabled that path. Keep it; drop the real creds.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shhh")
    assert _is_secret_env_name("AWS_EC2_METADATA_DISABLED") is False
    assert _is_secret_env_name("AWS_ACCESS_KEY_ID") is True
    env = _build_bypass_env(str(tmp_path))
    assert env.get("AWS_EC2_METADATA_DISABLED") == "true"
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


# ── connection-string env vars are stripped (regression) ────────────


@pytest.mark.parametrize(
    "name",
    [
        "SQLCONNSTR_DB",  # Azure App Service injected connection strings
        "MYSQLCONNSTR_DB",
        "SQLAZURECONNSTR_DB",
        "POSTGRESQLCONNSTR_DB",
        "CUSTOMCONNSTR_CACHE",
        "WEBSITE_CONTENTAZUREFILECONNECTIONSTRING",
    ],
)
def test_connection_string_names_are_flagged(name):
    assert _is_secret_env_name(name) is True


@pytest.mark.parametrize(
    "value",
    [
        "Server=tcp:db;Database=app;User ID=u;Password=p@ss;",  # ADO.NET
        "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=abc123==;",  # storage
        "Endpoint=sb://x;SharedAccessKeyName=n;SharedAccessKey=zzz=",  # Service Bus
    ],
)
def test_connection_string_values_are_flagged(value):
    assert _is_secret_env_value(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "Server=tcp:db;Database=app;User ID=u;",  # no password field
        "Endpoint=sb://x;SharedAccessKeyName=n",  # key NAME only, no secret
        "AccountName=x;EndpointSuffix=core.windows.net",  # no AccountKey
    ],
)
def test_connection_string_noncredential_values_are_not_flagged(value):
    assert _is_secret_env_value(value) is False


def test_connection_string_value_stripped_even_with_benign_name(monkeypatch, tmp_path):
    # NAME dodges the classifier, but the VALUE is a credentialed conn string.
    monkeypatch.setenv("APP_DB", "Server=tcp:db;Database=app;User ID=u;Password=p@ss;")
    monkeypatch.setenv("SQLCONNSTR_DB", "DefaultEndpointsProtocol=https;AccountKey=abc==")
    env = _build_bypass_env(str(tmp_path))
    assert "APP_DB" not in env  # value-based catch
    assert "SQLCONNSTR_DB" not in env  # name-based catch


# ── temp dirs repointed on every platform (regression) ──────────────


def test_bypass_env_repoints_all_temp_vars(monkeypatch, tmp_path):
    # Windows tempfile honours TEMP/TMP, not TMPDIR; all three must repoint.
    monkeypatch.setenv("TEMP", "/host/tmp")
    monkeypatch.setenv("TMP", "/host/tmp")
    env = _build_bypass_env(str(tmp_path))
    assert env["TMPDIR"] == str(tmp_path)
    assert env["TEMP"] == str(tmp_path)
    assert env["TMP"] == str(tmp_path)


# ── credential-location redirect vars are dropped (regression) ──────────
# Vars that point SDKs at the real home/cache/config (cached tokens), e.g.
# HF_HOME which startup always sets -> the live leak the HOME repoint missed.


@pytest.mark.parametrize(
    "name",
    [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "NETRC",
        "BOTO_CONFIG",
        "PIP_CONFIG_FILE",
        "CLOUDSDK_CONFIG",
        "KAGGLE_CONFIG_DIR",
        "DOCKER_CONFIG",
        "WANDB_DIR",
        "WANDB_CONFIG_DIR",
        "NPM_CONFIG_USERCONFIG",
        "NPM_CONFIG_GLOBALCONFIG",
        "YARN_RC_FILENAME",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "CARGO_HOME",
        "RCLONE_CONFIG",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "BASH_ENV",
        "HOMEDRIVE",
        "HOMEPATH",
    ],
)
def test_cred_location_names_are_flagged(name):
    assert _is_cred_location_env_name(name) is True


@pytest.mark.parametrize("name", ["PATH", "HOME", "LANG", "PWD", "MY_VAR"])
def test_benign_names_not_flagged_as_cred_location(name):
    assert _is_cred_location_env_name(name) is False


def test_bypass_env_drops_hf_home_so_cached_token_unreachable(monkeypatch, tmp_path):
    # The live leak: startup sets HF_HOME at the real cache, whose $HF_HOME/token
    # holds the operator's token. Repointing HOME does not stop huggingface_hub
    # from reading $HF_HOME/token, so HF_HOME must be dropped in bypass mode.
    real_cache = tmp_path / "real_hf_cache"
    real_cache.mkdir()
    (real_cache / "token").write_text("hf_cachedOperatorToken")
    monkeypatch.setenv("HF_HOME", str(real_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(real_cache / "hub"))
    env = _build_bypass_env(str(tmp_path))
    assert "HF_HOME" not in env  # dropped -> HF falls back to $HOME/.cache (empty)
    assert "HF_HUB_CACHE" not in env


def test_bypass_env_hf_token_resolves_outside_real_cache(monkeypatch, tmp_path):
    # End-to-end: even when HF_HOME and XDG_CACHE_HOME both point at the real
    # cache, the bypass env must make huggingface_hub resolve the token under the
    # workdir (guards the XDG fallback chain, not just "HF_HOME absent").
    pytest.importorskip("huggingface_hub")
    import subprocess

    real_cache = tmp_path / "real_hf"
    real_cache.mkdir()
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    monkeypatch.setenv("HF_HOME", str(real_cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(real_cache))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(real_cache))
    env = _build_bypass_env(str(workdir))
    token_path = subprocess.run(
        [
            sys.executable,
            "-c",
            "import huggingface_hub.constants as c; print(c.HF_TOKEN_PATH)",
        ],
        env = env,
        capture_output = True,
        text = True,
    ).stdout.strip()
    assert str(real_cache) not in token_path  # never the operator's cache
    assert token_path.startswith(str(workdir))  # resolved under the sandbox


def test_bypass_env_drops_credential_config_path_vars(monkeypatch, tmp_path):
    # NETRC / BOTO_CONFIG / PIP_CONFIG_FILE point clients at real credential
    # files before $HOME, so they must not survive into the bypassed child.
    monkeypatch.setenv("NETRC", "/home/op/.netrc")
    monkeypatch.setenv("PGPASSFILE", "/home/op/.pgpass")
    monkeypatch.setenv("BOTO_CONFIG", "/home/op/.boto")
    monkeypatch.setenv("PIP_CONFIG_FILE", "/home/op/.pip/pip.conf")
    env = _build_bypass_env(str(tmp_path))
    assert "NETRC" not in env
    assert "PGPASSFILE" not in env
    assert "BOTO_CONFIG" not in env
    assert "PIP_CONFIG_FILE" not in env


def test_bypass_env_strips_npm_auth_and_mysql_pwd(monkeypatch, tmp_path):
    # NPM_CONFIG__AUTH (npm _auth, base64) and MYSQL_PWD dodge the URL-value
    # check and the PASSWD marker, but must still be dropped.
    monkeypatch.setenv("NPM_CONFIG__AUTH", "aGVsbG86c2VjcmV0")
    monkeypatch.setenv("MYSQL_PWD", "db-password")
    assert _is_secret_env_name("NPM_CONFIG__AUTH") is True
    assert _is_secret_env_name("MYSQL_PWD") is True
    env = _build_bypass_env(str(tmp_path))
    assert "NPM_CONFIG__AUTH" not in env
    assert "MYSQL_PWD" not in env


@_POSIX_ONLY
def test_bash_bypass_does_not_source_bash_env(monkeypatch, tmp_path):
    # bash -c sources $BASH_ENV for non-interactive shells; an operator startup
    # file could re-export stripped secrets, so a real bypass call must not see it.
    startup = tmp_path / "startup.sh"
    startup.write_text("export RECOVERED=leaked\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    out = _bash_exec("echo R=$RECOVERED", None, 30, "bash-env-test", disable_sandbox = True)
    assert "R=leaked" not in out  # BASH_ENV dropped -> startup not sourced
    assert "R=" in out


def test_bypass_env_repoints_windows_profile_vars(monkeypatch, tmp_path):
    # On Windows, SDKs read cached creds under USERPROFILE/APPDATA/LOCALAPPDATA,
    # not $HOME. Set ones are repointed at the workdir; HOMEDRIVE/HOMEPATH drop.
    monkeypatch.setenv("USERPROFILE", "/host/profile")
    monkeypatch.setenv("APPDATA", "/host/profile/AppData/Roaming")
    monkeypatch.setenv("LOCALAPPDATA", "/host/profile/AppData/Local")
    monkeypatch.setenv("HOMEDRIVE", "C:")
    monkeypatch.setenv("HOMEPATH", "\\Users\\op")
    env = _build_bypass_env(str(tmp_path))
    assert env["USERPROFILE"] == str(tmp_path)
    assert env["APPDATA"] == str(tmp_path)
    assert env["LOCALAPPDATA"] == str(tmp_path)
    assert "HOMEDRIVE" not in env
    assert "HOMEPATH" not in env


def test_bypass_env_does_not_add_unset_windows_profile_vars(monkeypatch, tmp_path):
    # Only repoint Windows profile vars that were actually set (no pollution on
    # Linux/macOS where they are absent).
    monkeypatch.delenv("USERPROFILE", raising = False)
    monkeypatch.delenv("APPDATA", raising = False)
    monkeypatch.delenv("LOCALAPPDATA", raising = False)
    env = _build_bypass_env(str(tmp_path))
    assert "USERPROFILE" not in env
    assert "APPDATA" not in env
    assert "LOCALAPPDATA" not in env


# ── parent /proc env-leak hardening (regression) ────────────────────


@_POSIX_ONLY
def test_bypass_exec_hardens_parent_proc_env(monkeypatch, captured_popen):
    # Stripping the child env is not enough: a same-UID child can read the
    # parent's /proc environ. The exec paths must invoke the parent hardening
    # when (and only when) the sandbox is disabled.
    calls = {"n": 0}

    def fake_harden():
        calls["n"] += 1
        return True

    monkeypatch.setattr(tools, "_harden_parent_against_proc_env_leak", fake_harden)
    _python_exec("print(1)", None, 5, "t", disable_sandbox = True)
    _bash_exec("echo hi", None, 5, "t", disable_sandbox = True)
    assert calls["n"] == 2

    calls["n"] = 0
    _python_exec("print(1)", None, 5, "t", disable_sandbox = False)
    _bash_exec("echo hi", None, 5, "t", disable_sandbox = False)
    assert calls["n"] == 0  # never hardened on the sandboxed path


def test_bypass_exec_fails_closed_when_hardening_fails(monkeypatch, captured_popen):
    # If the parent cannot be hardened (e.g. prctl denied), the unsandboxed
    # child must NOT run - otherwise the parent environ stays readable.
    monkeypatch.setattr(tools, "_harden_parent_against_proc_env_leak", lambda: False)
    out_py = _python_exec("print(1)", None, 5, "t", disable_sandbox = True)
    out_sh = _bash_exec("echo hi", None, 5, "t", disable_sandbox = True)
    assert "refusing bypass execution" in out_py
    assert "refusing bypass execution" in out_sh
    assert "cmd" not in captured_popen  # never reached Popen


@_POSIX_ONLY
def test_proc_env_unreadable_after_hardening():
    # Mechanism check: after hardening, a same-UID child can no longer read the
    # parent process /proc environ. Restores the dumpable flag afterwards so the
    # process-global state does not leak into later tests.
    import subprocess

    if tools._libc is None:
        pytest.skip("no libc/prctl available")
    pid = os.getpid()
    probe = (
        "try:\n"
        f"    open('/proc/{pid}/environ', 'rb').read()\n"
        "    print('READABLE')\n"
        "except PermissionError:\n"
        "    print('DENIED')\n"
    )
    prev_dumpable = tools._libc.prctl(3, 0, 0, 0, 0)  # PR_GET_DUMPABLE
    prev_guard = tools._parent_proc_hardened
    try:
        # Establish a clean readable baseline: another test may have already
        # cleared the dumpable flag on this process.
        tools._libc.prctl(4, 1, 0, 0, 0)  # PR_SET_DUMPABLE = 1
        before = subprocess.run(
            [sys.executable, "-c", probe], capture_output = True, text = True
        ).stdout.strip()
        if before != "READABLE":
            pytest.skip("/proc already restricted in this environment")

        tools._parent_proc_hardened = False
        assert tools._harden_parent_against_proc_env_leak() is True

        after = subprocess.run(
            [sys.executable, "-c", probe], capture_output = True, text = True
        ).stdout.strip()
        assert after == "DENIED"
    finally:
        if prev_dumpable in (0, 1):
            try:
                tools._libc.prctl(4, prev_dumpable, 0, 0, 0)
            except (OSError, AttributeError):
                pass
        tools._parent_proc_hardened = prev_guard


# ── Anthropic request model declares the field (regression) ─────────


def test_anthropic_request_model_bypass_default_false():
    # Omitting the field on the Anthropic path must default to False rather than
    # raising AttributeError (extra='allow' does not set absent attributes).
    from models.inference import AnthropicMessagesRequest

    assert AnthropicMessagesRequest.model_fields["bypass_permissions"].default is False
    req = AnthropicMessagesRequest(model = "x", messages = [], max_tokens = 8)
    assert bool(req.bypass_permissions) is False

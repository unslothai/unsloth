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

import sys

import pytest

import core.inference.tools as tools
from core.inference.tools import (
    _bash_exec,
    _build_bypass_env,
    _build_safe_env,
    _is_secret_env_name,
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
    returncode = 0

    def communicate(self, timeout = None):
        return ("FAKEOUT", None)

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
def test_python_sandboxed_uses_sandbox_preexec_and_safe_env(
    captured_popen, monkeypatch
):
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
    assert "HF_TOKEN" not in env


def test_bash_blocklist_enforced_when_sandboxed(captured_popen):
    out = _bash_exec("rm -rf /", None, 5, "t", disable_sandbox = False)
    assert "Blocked" in out
    assert "cmd" not in captured_popen  # never reached Popen


def test_bash_blocklist_skipped_when_bypassed(captured_popen):
    out = _bash_exec("rm -rf /", None, 5, "t", disable_sandbox = True)
    assert out == "FAKEOUT"  # blocklist skipped -> reached (faked) execution
    assert captured_popen["cmd"][0] in ("bash", "cmd")


@_POSIX_ONLY
def test_bash_bypass_uses_bypass_preexec(captured_popen):
    _bash_exec("echo hi", None, 5, "t", disable_sandbox = True)
    assert captured_popen["kwargs"]["preexec_fn"] is tools._bypass_preexec


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

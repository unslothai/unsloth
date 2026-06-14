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
    monkeypatch.setenv(
        "PIP_EXTRA_INDEX_URL", "https://user:token@pypi.example.invalid/simple"
    )
    env = _build_bypass_env(str(tmp_path))
    assert env["HTTP_PROXY"] == "http://proxy.corp.example:8080"
    assert env["PIP_INDEX_URL"] == "https://pypi.corp.example/simple"
    assert "PIP_EXTRA_INDEX_URL" not in env  # this one carries credentials


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
    monkeypatch.setenv("BOTO_CONFIG", "/home/op/.boto")
    monkeypatch.setenv("PIP_CONFIG_FILE", "/home/op/.pip/pip.conf")
    env = _build_bypass_env(str(tmp_path))
    assert "NETRC" not in env
    assert "BOTO_CONFIG" not in env
    assert "PIP_CONFIG_FILE" not in env


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

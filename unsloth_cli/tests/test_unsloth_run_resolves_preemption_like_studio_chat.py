# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth run` (and `unsloth studio run`) has no preemption or exact concurrency path of its own.

Preemption follows ``UNSLOTH_LLAMA_ADMISSION_PREEMPT`` (auto when unset: on only once the
server confirms its own parking), ``UNSLOTH_LLAMA_PREEMPT_MODE`` and the kill switch, and exact
concurrency follows ``UNSLOTH_LLAMA_EXACT_CONCURRENCY`` then the stored setting then off. All of
that is read by the Studio server the CLI starts, through the same load route the chat UI uses.
These pin that the CLI neither writes nor drops those variables and that its load request names
neither setting, so a headless run lands on exactly the defaults Studio and Desktop chat get.
"""

from __future__ import annotations

import json
import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_POLICY_ENV = (
    "UNSLOTH_LLAMA_ADMISSION_PREEMPT",
    "UNSLOTH_LLAMA_PREEMPT_MODE",
    "UNSLOTH_LLAMA_EXACT_CONCURRENCY",
    "LLAMA_EXACT_CONCURRENCY",
    "UNSLOTH_LLAMA_ADMISSION_CONTROL",
    "UNSLOTH_LLAMA_ADMISSION_KV_BUDGET",
)


def _studio():
    from unsloth_cli.commands import studio as studio_mod

    return studio_mod


def _run_in_venv(monkeypatch, tmp_path, args, *, alias):
    """Drive the in-venv branch up to the load request; return (env at server start, load kwargs)."""
    import typer

    studio_mod = _studio()
    fake_venv = tmp_path / "studio" / "venv" / "unsloth_studio"
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)
    from unsloth_cli import _tool_policy as tool_policy

    monkeypatch.setattr(tool_policy, "resolve_tool_policy", lambda host, flag, yes, silent: False)
    seen = {}

    class _App:
        class state:
            server_port = 8889
            server_request_host = "127.0.0.1"

    def run_server(**_kwargs):
        seen["env"] = {name: studio_mod.os.environ.get(name) for name in _POLICY_ENV}
        return _App()

    backend = types.ModuleType("studio.backend.run")
    backend.run_server = run_server
    backend._server = object()
    backend._graceful_shutdown = lambda server: None
    monkeypatch.setitem(sys.modules, "studio.backend.run", backend)
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", backend)
    monkeypatch.setattr(studio_mod, "_wait_for_server", lambda port, **_kwargs: True)
    monkeypatch.setattr(studio_mod, "_create_api_key_inprocess", lambda name: "sk-unsloth-test")

    def stop_at_load(**kwargs):
        seen["load"] = kwargs
        raise RuntimeError("stopped at load")

    monkeypatch.setattr(studio_mod, "_load_model_via_http", stop_at_load)
    if alias:
        from unsloth_cli import app
    else:
        app = typer.Typer()
        app.command(
            context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
        )(studio_mod.run)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return seen


_MODEL = ["--model", "unsloth/Qwen3-1.7B-GGUF"]


@pytest.mark.parametrize("alias", [True, False], ids = ["unsloth-run", "studio-run"])
@pytest.mark.parametrize(
    "inherited",
    [
        {},
        {"UNSLOTH_LLAMA_ADMISSION_PREEMPT": "0"},
        {"UNSLOTH_LLAMA_ADMISSION_PREEMPT": "1", "UNSLOTH_LLAMA_PREEMPT_MODE": "studio"},
        {"UNSLOTH_LLAMA_EXACT_CONCURRENCY": "on"},
    ],
    ids = ["unset", "kill-switch", "studio-mode", "exact-on"],
)
def test_the_server_starts_with_the_callers_policy_env(monkeypatch, tmp_path, alias, inherited):
    for name in _POLICY_ENV:
        monkeypatch.delenv(name, raising = False)
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    args = (["run"] if alias else []) + _MODEL
    seen = _run_in_venv(monkeypatch, tmp_path, args, alias = alias)
    assert "env" in seen, "the server never started"
    assert seen["env"] == {name: inherited.get(name) for name in _POLICY_ENV}
    assert "load" in seen, "the run never reached its load request"


def test_the_load_request_names_no_preemption_or_exact_setting(monkeypatch):
    """The UI's load request leaves both to the server too, so both resolve identically."""
    studio_mod = _studio()
    captured = {}

    def urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return BytesIO(b'{"model": "owner/model-GGUF"}')

    monkeypatch.setattr(studio_mod, "_direct_urlopen", urlopen)
    studio_mod._load_model_via_http(
        port = 8888,
        api_key = "sk-test",
        model = "owner/model-GGUF",
        gguf_variant = "Q4_K_M",
        max_seq_length = 0,
        load_in_4bit = True,
        llama_extra_args = ["--ctx-size", "8192"],
    )
    named = {key for key in captured["body"] if "exact" in key or "preempt" in key}
    assert named == set()


def test_a_reexeced_child_inherits_the_policy_env(monkeypatch):
    from unsloth_cli import _studio_runtime_gate as gate

    for name in _POLICY_ENV:
        monkeypatch.setenv(name, f"value-of-{name}")
    child = gate.runtime_gate_child_environment()
    assert {name: child.get(name) for name in _POLICY_ENV} == {
        name: f"value-of-{name}" for name in _POLICY_ENV
    }

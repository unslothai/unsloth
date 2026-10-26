# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parking llama-server from before the stream park notices is silent for a park, headers
included, so every reader with a deadline would have to tell that silence from a stall on its own:
the Windows header wait cannot, and Deep Research ended a healthy park as a timeout. Studio
therefore does not turn that build's parking on, on any platform: the launch names
``--preempt-ram 0`` and the pausing is Studio's. A notice build (``--preempt-async`` in its help,
unslothai/llama.cpp#269) keeps the budget, and so does a budget the user named."""

from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

import pytest

from core.inference.llama_cpp import LlamaCppBackend

from .test_gpu_init_crash_message import _run_cpu_fallback_load

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_SILENT_PARKING_BUILD = {"found": True, "supports_preempt_ram": True, "supports_kv_unified": True}
_NOTICE_BUILD = dict(_SILENT_PARKING_BUILD, supports_preempt_async = True)


def _launch(
    monkeypatch,
    tmp_path,
    caps,
    *,
    platform,
    extra_args = None,
):
    load_model = LlamaCppBackend.load_model
    # A simulated CPU launch: a real torch import would run `ldconfig -p` through the one-launch
    # Popen fake and use up its return codes before parking is reached.
    monkeypatch.setitem(sys.modules, "torch", None)

    def _load(self, intent, *args, **kwargs):
        self.probe_server_capabilities = lambda _binary = None: dict(caps)
        return load_model(self, dataclasses.replace(intent, n_parallel = 4), *args, **kwargs)

    monkeypatch.setattr(LlamaCppBackend, "load_model", _load)
    backend, loaded, launches, _ = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [None],
        vulkan = False,
        platform = platform,
        extra_args = extra_args,
    )
    assert loaded is True
    assert len(launches) == 1
    return backend, launches[0][0]


def _budgets(cmd) -> list[str]:
    return [cmd[i + 1] for i, token in enumerate(cmd[:-1]) if token == "--preempt-ram"]


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_a_silent_parking_build_is_not_asked_to_park(monkeypatch, tmp_path, platform):
    backend, cmd = _launch(monkeypatch, tmp_path, _SILENT_PARKING_BUILD, platform = platform)

    assert "--kv-unified" in cmd
    assert _budgets(cmd) == ["0"], (
        "Studio turned on parking for a build that parks in silence, which a reader with a "
        "deadline cannot tell from a stall"
    )
    assert backend._server_preempts_kv is False
    assert backend.server_may_park is False


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_a_notice_build_still_parks(monkeypatch, tmp_path, platform):
    backend, cmd = _launch(monkeypatch, tmp_path, _NOTICE_BUILD, platform = platform)

    assert len(_budgets(cmd)) == 1 and int(_budgets(cmd)[0]) > 0
    assert backend._server_preempts_kv is True


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_a_budget_the_user_named_is_kept(monkeypatch, tmp_path, platform):
    backend, cmd = _launch(
        monkeypatch,
        tmp_path,
        _SILENT_PARKING_BUILD,
        platform = platform,
        extra_args = ["--preempt-ram", "2048"],
    )

    assert _budgets(cmd) == ["2048"]
    assert backend._server_preempts_kv is True


_NEEDS_BASH = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _fake_server(path: Path, help_text: str) -> Path:
    path.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{help_text}\nEOF\n")
    path.chmod(0o755)
    return path


def _probe(path: Path) -> dict:
    with LlamaCppBackend._capability_cache_lock:
        LlamaCppBackend._capability_cache.clear()
        LlamaCppBackend._capability_retry_after.clear()
    return LlamaCppBackend.probe_server_capabilities(str(path))


_PARKING_HELP = (
    "--preempt-ram N                         with a unified KV cache, park a slot in host RAM\n"
    "-kvu,  --kv-unified, -no-kvu, --no-kv-unified\n"
    "                                        use single unified KV buffer\n"
)
_NOTICE_HELP = _PARKING_HELP + (
    "--preempt-async, --no-preempt-async     copy a parked sequence out of and back into the KV\n"
)


@_NEEDS_BASH
def test_the_probe_tells_a_notice_build_from_a_silent_one(tmp_path):
    silent = _probe(_fake_server(tmp_path / "silent-llama-server", _PARKING_HELP))
    notices = _probe(_fake_server(tmp_path / "notice-llama-server", _NOTICE_HELP))

    assert silent["supports_preempt_ram"] is True
    assert silent["supports_preempt_async"] is False
    assert notices["supports_preempt_ram"] is True
    assert notices["supports_preempt_async"] is True

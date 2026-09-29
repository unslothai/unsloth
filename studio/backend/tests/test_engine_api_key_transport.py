# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The engine key stays out of argv (readable by any local user via the process list), and
vLLM's key guards every route, not only /v1, /v2 and /inference."""

import importlib.util
import runpy
import sys
import types
from pathlib import Path

import pytest

from core.inference.engine_adapters import ADAPTERS

INFERENCE = Path(__file__).resolve().parents[1] / "core" / "inference"
KEY = "studio-secret-key"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("parallelism", ["tensor", "data"])
def test_the_key_is_never_on_the_command_line(engine, parallelism):
    command = ADAPTERS[engine].command(
        "/env/bin/python", "org/m", 8123, KEY, 4096, 0.9, 2, options = {"parallelism": parallelism}
    )
    assert KEY not in " ".join(command)
    assert "--api-key" not in command
    name = "VLLM_API_KEY" if engine == "vllm" else "UNSLOTH_ENGINE_API_KEY"
    assert ADAPTERS[engine].key_environment(KEY) == {name: KEY}


def test_vllm_starts_through_the_guarding_launcher():
    command = ADAPTERS["vllm"].command("/env/bin/python", "org/m", 8123, KEY, 4096, 0.9)
    assert command[2].endswith("vllm_server.py")
    assert command[3] == "vllm.entrypoints.openai.api_server"


def _fake_vllm(monkeypatch, **attrs):
    server_utils = types.SimpleNamespace(**attrs)
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.serve", "vllm.entrypoints.serve.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.entrypoints.serve.utils"].server_utils = server_utils
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.serve.utils.server_utils", server_utils)
    return server_utils


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", INFERENCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_launcher_guards_every_path(monkeypatch):
    server_utils = _fake_vllm(monkeypatch, GUARDED_PREFIX = ("/v1", "/v2", "/inference"))
    _load("vllm_server")
    assert server_utils.GUARDED_PREFIX == ("/",)
    assert "/invocations".startswith(server_utils.GUARDED_PREFIX)


def test_the_launcher_refuses_a_vllm_whose_guard_moved(monkeypatch):
    _fake_vllm(monkeypatch)
    with pytest.raises(SystemExit):
        _load("vllm_server")


def test_sglang_reads_the_key_from_its_environment(monkeypatch):
    torchao_utils = types.SimpleNamespace(
        apply_torchao_config_to_model = lambda *a, **k: None, proj_filter = None
    )
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["sglang.srt.layers"].torchao_utils = torchao_utils
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.torchao_utils", torchao_utils)
    ran = {}
    monkeypatch.setattr(runpy, "run_module", lambda name, **k: ran.update(argv = list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["sglang_server.py", "--model-path", "org/m"])
    monkeypatch.setenv("UNSLOTH_ENGINE_API_KEY", KEY)
    runpy.run_path(str(INFERENCE / "sglang_server.py"), run_name = "__main__")
    assert ran["argv"][-2:] == ["--api-key", KEY]

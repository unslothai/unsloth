# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth run --engine vllm|sglang`: the load payload, re-exec forwarding, and the
install-on-first-use consent."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest
import typer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unsloth_cli.commands import studio as studio_mod


def test_engine_options_are_registered_on_run():
    sig = inspect.signature(studio_mod.run)
    assert set(sig.parameters["engine"].default.param_decls) == {"--engine"}
    assert sig.parameters["engine"].default.default == "auto"
    assert set(sig.parameters["engine_precision"].default.param_decls) == {"--engine-precision"}


def test_reexec_forwards_the_engine():
    src = inspect.getsource(studio_mod.run)
    assert 'args.extend(["--engine", engine, "--engine-precision", engine_precision])' in src


def _capture_payload(monkeypatch, **kwargs):
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status": "loaded", "model": "m"}'

    def fake_urlopen(req, timeout = None):
        import json
        seen.update(json.loads(req.data))
        return _Resp()

    monkeypatch.setattr(studio_mod, "_direct_urlopen", fake_urlopen)
    import unsloth_cli._inference as inf

    monkeypatch.setattr(inf, "raise_for_deferred_error", lambda url, body: body)
    monkeypatch.setattr(inf, "require_completed_padded_body", lambda url, body: body)
    studio_mod._load_model_via_http(8888, "k", "m", None, 4096, False, **kwargs)
    return seen


def test_default_load_payload_has_no_engine(monkeypatch):
    payload = _capture_payload(monkeypatch)
    assert "engine" not in payload and "engine_precision" not in payload


def test_engine_load_payload(monkeypatch):
    payload = _capture_payload(monkeypatch, engine = "vllm", engine_precision = "fp8")
    assert payload["engine"] == "vllm" and payload["engine_precision"] == "fp8"


class _FakeInstall:
    def __init__(
        self,
        row,
        finish = "success",
    ):
        self.row = dict(row)
        self.started = 0
        self.finish = finish

    def status(self, engine):
        job = {"state": self.finish if self.started else "idle", "message": "done"}
        return {**self.row, "job": job}

    def support_reason(self, engine):
        return self.row["unsupported_reason"]

    def start_install(self, engine):
        self.started += 1
        return self.status(engine)


@pytest.fixture
def fake_install(monkeypatch):
    def make(row, finish = "success"):
        fake = _FakeInstall(row, finish)
        core = types.ModuleType("core")
        inference = types.ModuleType("core.inference")
        inference.engine_install = fake
        core.inference = inference
        monkeypatch.setitem(sys.modules, "core", core)
        monkeypatch.setitem(sys.modules, "core.inference", inference)
        monkeypatch.setitem(sys.modules, "core.inference.engine_install", fake)
        return fake

    return make


_MISSING = {
    "version": "0.30.0",
    "installed": False,
    "current": False,
    "unsupported_reason": None,
    "download_bytes": 1073741824,
}


def test_installed_engine_is_not_reinstalled(fake_install):
    fake = fake_install({**_MISSING, "installed": True, "current": True})
    studio_mod._ensure_engine_installed("vllm", yes = False, silent = True)
    assert fake.started == 0


def test_restored_engine_is_loaded_as_is(fake_install):
    fake = fake_install({**_MISSING, "installed": True, "current": False, "restored": True})
    studio_mod._ensure_engine_installed("vllm", yes = True, silent = True)
    assert fake.started == 0


def test_missing_engine_without_a_terminal_needs_yes(fake_install, monkeypatch):
    fake = fake_install(_MISSING)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(RuntimeError, match = "--yes"):
        studio_mod._ensure_engine_installed("vllm", yes = False, silent = True)
    assert fake.started == 0


def test_declined_prompt_installs_nothing(fake_install, monkeypatch):
    fake = fake_install(_MISSING)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    asked = []
    monkeypatch.setattr(typer, "confirm", lambda text, default = False: asked.append(text) or False)
    with pytest.raises(RuntimeError, match = "not installed"):
        studio_mod._ensure_engine_installed("vllm", yes = False, silent = True)
    assert fake.started == 0
    assert asked == ["Install vllm 0.30.0 (about 1.0 GiB to download)?"]


def test_yes_installs_and_waits(fake_install):
    fake = fake_install(_MISSING)
    studio_mod._ensure_engine_installed("vllm", yes = True, silent = True)
    assert fake.started == 1


def test_failed_install_is_an_error(fake_install):
    fake_install(_MISSING, finish = "error")
    with pytest.raises(RuntimeError, match = "did not finish"):
        studio_mod._ensure_engine_installed("vllm", yes = True, silent = True)


def test_unsupported_host_is_refused_before_installing(fake_install):
    fake = fake_install({**_MISSING, "unsupported_reason": "vLLM needs an NVIDIA GPU."})
    with pytest.raises(RuntimeError, match = "NVIDIA"):
        studio_mod._ensure_engine_installed("vllm", yes = True, silent = True)
    assert fake.started == 0


def test_windows_asks_before_turning_on_wsl(fake_install, monkeypatch):
    fake = fake_install({**_MISSING, "host": "wsl", "wsl": {"state": "missing", "distro": None}})
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    shown = []
    monkeypatch.setattr(typer, "echo", lambda text = "", **k: shown.append(text))
    monkeypatch.setattr(typer, "confirm", lambda text, default = False: False)
    with pytest.raises(RuntimeError):
        studio_mod._ensure_engine_installed("vllm", yes = False, silent = True)
    assert fake.started == 0
    assert any("administrator (UAC) prompt" in line and "WSL2" in line for line in shown)


@pytest.mark.parametrize(
    ("wsl", "expected"),
    [
        ({"state": "restart_required"}, "Restart Windows"),
        ({"state": "ready", "distro": "UnslothStudio"}, "private WSL2 environment"),
    ],
)
def test_wsl_notice_follows_its_state(wsl, expected):
    assert expected in studio_mod._engine_install_notice("vllm", {"host": "wsl", "wsl": wsl})


def test_linux_has_no_wsl_notice():
    assert studio_mod._engine_install_notice("vllm", {"host": "local"}) is None

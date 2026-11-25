# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import httpx
import pytest

from core.inference import npu_backend as nb


class Runtime:
    def __init__(self, available, state):
        self.available = available
        self.state = state
        self.running = False
        self.requests = []

    def start(self):
        self.running = True

    def is_alive(self):
        return self.running

    def close(self):
        self.running = False

    def request(self, method, path, **kwargs):
        self.requests.append((method, path))
        assert (method, path) == ("GET", "/v1/system-info")
        return httpx.Response(
            200,
            json = {
                "devices": {"amd_npu": {"available": self.available, "error": "Driver unavailable"}},
                "recipes": {"flm": {"backends": {"npu": {"state": self.state}}}},
            },
        )


@pytest.fixture
def configure(tmp_path, monkeypatch):
    exe = tmp_path / "flm"
    exe.write_bytes(b"executable")
    monkeypatch.setenv(nb.FLM_BINARY_ENV, str(exe))
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: {"supported": True, "present": False})
    monkeypatch.setattr(
        nb.LemonadeNpuBackend,
        "_validate",
        lambda self: pytest.fail("custom runtime called built-in validation"),
    )
    monkeypatch.setattr(nb.LemonadeNpuBackend, "_flm_version", lambda self, binary = None: "1.2.3")
    lemond = tmp_path / "lemond"
    monkeypatch.setattr(
        nb,
        "_installer_module",
        lambda: SimpleNamespace(install = lambda *a, **kw: lemond, installed_lemond = lambda *a: lemond),
    )
    backend = nb.LemonadeNpuBackend(root = tmp_path)

    def with_runtime(available = True, state = "installed"):
        runtime = Runtime(available, state)
        monkeypatch.setattr(backend, "_server_for", lambda path: runtime)
        return backend, runtime, exe

    return with_runtime


def test_enable_uses_custom_runtime_without_installing_fastflowlm(configure):
    backend, runtime, exe = configure()
    assert backend._flm_binary() == exe
    assert backend.enable()["ready"]
    assert runtime.requests and set(runtime.requests) == {("GET", "/v1/system-info")}

    # A changed executable at the same path must be checked again after restart.
    assert backend._validated_install() is not None
    exe.write_bytes(b"replacement executable")
    assert backend._validated_install() is None


def test_custom_runtime_refuses_an_unavailable_device(configure):
    backend, runtime, _ = configure(available = False)
    with pytest.raises(nb.NpuError, match = "Driver unavailable"):
        backend.enable()
    assert not runtime.is_alive()
    assert not backend.status()["ready"]


def test_custom_runtime_requires_an_installed_backend(configure):
    backend, _, _ = configure(state = "action_required")
    with pytest.raises(nb.NpuError, match = "configured NPU runtime is not ready"):
        backend.enable()
    assert not backend.status()["ready"]

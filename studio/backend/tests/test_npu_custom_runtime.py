# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from contextlib import contextmanager
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
                "devices": {
                    "amd_npu": {"available": self.available, "error": "Driver unavailable"}
                },
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
        lambda: SimpleNamespace(
            install = lambda *a, **kw: lemond, installed_lemond = lambda *a: lemond
        ),
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


def test_custom_runtime_preserves_model_file_discovery(configure, monkeypatch):
    backend, _, exe = configure()
    files = nb.FlmModelFiles(folder = backend._flm_model_dir / "downloaded-model", files = ())

    def read(binary, model_dir, checkpoint):
        assert (binary, model_dir, checkpoint) == (exe, backend._flm_model_dir, "model:small")
        return files

    monkeypatch.setattr(nb, "read_model_files", read)
    assert backend._model_files("model:small", backend._flm_binary()) is files


def test_custom_runtime_delegates_download_without_reading_the_file_list(configure, monkeypatch):
    backend, runtime, _ = configure()
    backend.enable()
    model = SimpleNamespace(id = "model-FLM", checkpoint = "model:small", downloaded = False)
    monkeypatch.setattr(backend, "_model", lambda model_id: model)
    monkeypatch.setattr(
        backend, "_model_files", lambda *args: pytest.fail("download read the custom file list")
    )

    @contextmanager
    def stream(method, path, *, json_body):
        assert (method, path) == ("POST", "/v1/pull")
        assert json_body["model_name"] == model.id
        yield httpx.Response(200, text = 'event: complete\ndata: {"status":"complete"}\n')

    monkeypatch.setattr(runtime, "stream", stream, raising = False)
    assert list(backend.download(model.id))[-1] == {
        "event": "complete",
        "model": model.id,
        "percent": 100,
    }

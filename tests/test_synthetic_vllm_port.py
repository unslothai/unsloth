# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vLLM port is no longer fixed at 8000, where Studio listens in the Unsloth Docker image."""

import socket

import pytest

import unsloth.dataprep.synthetic as synthetic
from unsloth.dataprep.synthetic import SyntheticDataKit


@pytest.fixture
def busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # All interfaces, as Studio binds in the Docker image: BSD lets a wildcard bind share a port with a
        # loopback-only listener, so a 127.0.0.1 listener would not read as taken on macOS.
        s.bind(("", 0))
        s.listen()
        yield s.getsockname()[1]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_default_port_is_kept_when_free(monkeypatch):
    port = _free_port()
    monkeypatch.setattr(SyntheticDataKit, "port", port)
    assert synthetic._pick_vllm_port(None) == port


def test_default_port_moves_when_taken(monkeypatch, busy_port):
    monkeypatch.setattr(SyntheticDataKit, "port", busy_port)
    picked = synthetic._pick_vllm_port(None)
    assert picked != busy_port
    assert synthetic._port_is_free(picked)


def test_explicit_port_is_used_as_given(busy_port):
    assert synthetic._pick_vllm_port(busy_port) == busy_port
    assert synthetic._pick_vllm_port(str(busy_port)) == busy_port


def test_check_vllm_status_queries_the_given_port(monkeypatch):
    import requests

    urls = []

    class _Response:
        status_code = 200

    def _get(url, **kwargs):
        urls.append(url)
        return _Response()

    monkeypatch.setattr(requests, "get", _get)
    assert SyntheticDataKit.check_vllm_status(8123) is True
    assert SyntheticDataKit.check_vllm_status() is True
    assert urls == ["http://localhost:8123/metrics", "http://localhost:8000/metrics"]


_REAL_AWAIT_METRICS = SyntheticDataKit._await_metrics_endpoint


def _built_kit(monkeypatch, tmp_path, **kwargs):
    """Run __init__ with the model download, vLLM and the waits stubbed out."""
    import transformers

    class _Loaded:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    monkeypatch.setattr(transformers, "AutoConfig", _Loaded)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Loaded)
    load_vllm = lambda **kwargs: {"model": kwargs["model_name"], "max_model_len": 2048}
    monkeypatch.setattr(
        synthetic,
        "_load_vllm_utils",
        lambda: (load_vllm, lambda **kwargs: None, lambda **kwargs: None),
    )
    argv = []

    class _Popen:
        stdout = stderr = None

        def __init__(self, args, **kwargs):
            argv.extend(args)

    monkeypatch.setattr(synthetic.subprocess, "Popen", _Popen)
    monkeypatch.setattr(synthetic, "PipeCapture", lambda *args, **kwargs: None)
    monkeypatch.setattr(SyntheticDataKit, "_await_vllm_server", lambda self, timeout: None)
    monkeypatch.setattr(SyntheticDataKit, "_await_metrics_endpoint", lambda self: None)
    monkeypatch.setattr(SyntheticDataKit, "cleanup", lambda self: None)
    monkeypatch.chdir(tmp_path)
    kit = SyntheticDataKit(model_name = "some/model", **kwargs)
    kit.prepare_qa_generation(output_folder = str(tmp_path / "data"))
    config = (tmp_path / "synthetic_data_kit_config.yaml").read_text(encoding = "utf-8")
    return kit, argv, config


def test_server_and_config_use_the_explicit_port(monkeypatch, tmp_path):
    kit, argv, config = _built_kit(monkeypatch, tmp_path, port = 8123)
    assert kit.port == 8123
    assert argv[argv.index("--port") + 1] == "8123"
    assert 'api_base: "http://localhost:8123/v1"' in config
    assert "port: 8123 " in config
    assert "8000" not in config


def test_server_and_config_follow_a_moved_default(monkeypatch, tmp_path, busy_port):
    monkeypatch.setattr(SyntheticDataKit, "port", busy_port)
    kit, argv, config = _built_kit(monkeypatch, tmp_path)
    assert kit.port != busy_port
    assert argv[argv.index("--port") + 1] == str(kit.port)
    assert f'api_base: "http://localhost:{kit.port}/v1"' in config


def test_server_binds_loopback_by_default(monkeypatch, tmp_path):
    _kit, argv, _config = _built_kit(monkeypatch, tmp_path, port = 8123)
    # vLLM's own default is every interface, and the server runs with no API key.
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_a_wider_bind_is_opt_in(monkeypatch, tmp_path):
    _kit, argv, _config = _built_kit(monkeypatch, tmp_path, port = 8123, host = "0.0.0.0")
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    assert argv.count("--host") == 1


@pytest.mark.parametrize(
    "host, client",
    [
        ("127.0.0.1", "localhost"),
        ("0.0.0.0", "localhost"),
        ("::1", "[::1]"),
        ("192.168.1.20", "192.168.1.20"),
    ],
)
def test_the_kit_reaches_its_own_bind_address(monkeypatch, tmp_path, host, client):
    kit, _argv, config = _built_kit(monkeypatch, tmp_path, port = 8123, host = host)
    assert f'api_base: "http://{client}:8123/v1"' in config
    seen = []
    monkeypatch.setattr(
        SyntheticDataKit, "check_vllm_status", staticmethod(lambda *a: seen.append(a) or True)
    )
    _REAL_AWAIT_METRICS(kit)
    assert seen == [(8123, client)]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
import importlib.util
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_EXTERNAL_PROVIDER_PATH = (
    Path(__file__).resolve().parent.parent / "core/inference/external_provider.py"
)


def _load_external_provider_module():
    spec = importlib.util.spec_from_file_location(
        "external_provider_under_test",
        _EXTERNAL_PROVIDER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_http_client_ignores_unsupported_proxy_scheme(monkeypatch):
    ep_mod = _load_external_provider_module()
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("trust_env") is not False:
                raise ValueError("Unknown scheme for proxy URL URL('socks4://127.0.0.1:12345')")

    monkeypatch.setattr(ep_mod.httpx, "AsyncClient", FakeAsyncClient)

    client = ep_mod._create_shared_http_client()

    assert isinstance(client, FakeAsyncClient)
    assert calls == [{}, {"trust_env": False}]


def test_shared_http_client_ignores_missing_socksio(monkeypatch):
    ep_mod = _load_external_provider_module()
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("trust_env") is not False:
                raise ImportError(
                    "Using SOCKS proxy, but the 'socksio' package is not installed. "
                    "Make sure to install httpx using `pip install httpx[socks]`."
                )

    monkeypatch.setattr(ep_mod.httpx, "AsyncClient", FakeAsyncClient)

    client = ep_mod._create_shared_http_client()

    assert isinstance(client, FakeAsyncClient)
    assert calls == [{}, {"trust_env": False}]


def test_shared_http_client_reraises_other_value_errors(monkeypatch):
    ep_mod = _load_external_provider_module()

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            raise ValueError("different httpx setup error")

    monkeypatch.setattr(ep_mod.httpx, "AsyncClient", FakeAsyncClient)

    try:
        ep_mod._create_shared_http_client()
    except ValueError as exc:
        assert str(exc) == "different httpx setup error"
    else:
        raise AssertionError("expected ValueError")

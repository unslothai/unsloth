import os, sys

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

from routes.inference import _llama_auth_headers


class _WithKey:
    _api_key = "k_secret_123"


class _NoKey:
    _api_key = None


class _Missing:
    pass


class _EmptyKey:
    _api_key = ""


def test_returns_bearer_header_when_api_key_set():
    headers = _llama_auth_headers(_WithKey())
    assert headers == {"Authorization": "Bearer k_secret_123"}


def test_returns_none_when_api_key_none():
    assert _llama_auth_headers(_NoKey()) is None


def test_returns_none_when_api_key_attribute_missing():
    assert _llama_auth_headers(_Missing()) is None


def test_returns_none_when_api_key_empty_string():
    # Empty string is falsy; the helper should treat it as absent.
    assert _llama_auth_headers(_EmptyKey()) is None

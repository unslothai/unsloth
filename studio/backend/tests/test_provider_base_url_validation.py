# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""External-provider base URL validation (SSRF hardening).

The backend fetches the provider base URL on the caller's behalf with their
decrypted API key attached, so the URL is server-side egress under caller
control. These tests pin both halves of the policy: every endpoint a user can
configure today keeps working (plain http, loopback, LAN, odd ports, query
strings), while shapes that can never be a provider -- non-http(s) schemes,
embedded credentials, cloud metadata services -- are refused.
"""

import importlib.util
import socket
from pathlib import Path

import pytest

_PROVIDERS_PATH = Path(__file__).resolve().parents[1] / "core" / "inference" / "providers.py"
_SPEC = importlib.util.spec_from_file_location("provider_registry_for_test", _PROVIDERS_PATH)
_providers = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_providers)

validate_provider_base_url = _providers.validate_provider_base_url
PROVIDER_REGISTRY = _providers.PROVIDER_REGISTRY
BLOCK_PRIVATE_ENV = _providers._BLOCK_PRIVATE_ENV


# Every base URL a user can reach today: shipped registry defaults, the local
# server presets, LAN gateways, docker-compose hostnames, query strings.
_SUPPORTED = [
    "http://localhost:11434/v1",
    "http://localhost:8080/v1",
    "http://127.0.0.1:8080/v1",
    "http://127.0.0.1:1",
    "http://192.168.1.50:8000/v1",
    "http://10.1.2.3:8000/v1",
    "http://my_ollama:11434/v1",
    "http://llama.test",
    "https://my-vllm-server.com/v1",
    "https://my-resource.openai.azure.com/openai/v1",
    "https://gw.example/v1?tenant=a",
    # A numeric host that canonicalizes to a public address is untouched.
    "http://1681207502/v1",
    # A DNS name is not link-local just because it starts with those digits.
    "http://169.254.gateway.example.com/v1",
    "https://[2606:4700:4700::1111]/v1",
    # Self-hosted gateways behind basic auth keep working.
    "https://user:pass@gw.example/v1",
] + [info["base_url"] for info in PROVIDER_REGISTRY.values() if info["base_url"]]


@pytest.fixture(autouse = True)
def _default_policy(monkeypatch):
    """Default deployment: the private-address opt-in is off."""
    monkeypatch.delenv(BLOCK_PRIVATE_ENV, raising = False)


@pytest.mark.parametrize("url", _SUPPORTED)
def test_supported_base_urls_are_unchanged(url):
    assert validate_provider_base_url(url) == url


@pytest.mark.parametrize("url", _SUPPORTED)
def test_validation_is_idempotent(url):
    once = validate_provider_base_url(url)
    assert validate_provider_base_url(once) == once


def test_trailing_slash_and_whitespace_are_normalized():
    assert validate_provider_base_url("  http://127.0.0.1:8080/v1/  ") == "http://127.0.0.1:8080/v1"


def test_no_dns_lookup_on_the_default_path(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("validation must not resolve DNS by default")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    assert validate_provider_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    "url, error",
    [
        ("file:///etc/passwd", "http or https"),
        ("gopher://example.com/", "http or https"),
        ("data:text/plain,hi", "http or https"),
        ("http://exa mple.com/v1", "invalid characters"),
        ("http://example.com\n/v1", "invalid characters"),
        ("http://example.com\\@evil.com/v1", "invalid characters"),
        ("https:///v1", "hostname"),
        ("", "required"),
        ("   ", "required"),
    ],
)
def test_rejected_url_shapes(url, error):
    with pytest.raises(ValueError, match = error):
        validate_provider_base_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://169.254.169.254./latest/meta-data/",
        "http://[::ffff:169.254.169.254]/latest/meta-data/",
        "http://169.254.170.2/v2/credentials",
        "http://169.254.170.23/v1/credentials",
        "http://[fd00:ec2::254]/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata/computeMetadata/v1/",
        "http://100.100.100.200/latest/meta-data/",
        # Userinfo does not disguise the real host.
        "http://api.openai.com@169.254.169.254/latest/meta-data/",
        # Legacy numeric spellings the resolver maps to 169.254.169.254.
        "http://2852039166/latest/meta-data/",
        "http://0xA9FEA9FE/latest/meta-data/",
        "http://0251.0376.0251.0376/latest/meta-data/",
        "http://169.254.43518/latest/meta-data/",
    ],
)
def test_cloud_metadata_endpoints_are_always_refused(url, monkeypatch):
    with pytest.raises(ValueError, match = "metadata"):
        validate_provider_base_url(url)
    # Also refused with the private-address opt-in on.
    monkeypatch.setenv(BLOCK_PRIVATE_ENV, "1")
    with pytest.raises(ValueError, match = "metadata"):
        validate_provider_base_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434/v1",
        "http://192.168.1.50:8000/v1",
        "http://10.1.2.3:8000/v1",
    ],
)
def test_private_targets_blocked_only_with_the_opt_in(url, monkeypatch):
    # Default: allowed (this is the normal local-provider flow).
    assert validate_provider_base_url(url) == url

    monkeypatch.setenv(BLOCK_PRIVATE_ENV, "1")
    # Names resolve to loopback; conftest blocks real resolution, and IP
    # literals never reach the resolver.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match = "private address"):
        validate_provider_base_url(url)


def test_public_targets_still_allowed_with_the_opt_in(monkeypatch):
    monkeypatch.setenv(BLOCK_PRIVATE_ENV, "1")
    assert validate_provider_base_url("https://1.1.1.1/v1") == "https://1.1.1.1/v1"

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )
    assert validate_provider_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

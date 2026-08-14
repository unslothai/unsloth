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
    # An internationalized host is left alone.
    "https://例え.テスト/v1",
    # A neighbour of the metadata address is an ordinary host.
    "http://[fd00:ec2::255]/v1",
    "http://[fd20:ce::255]/v1",
    "https://[2606:4700:4700::1111]/v1",
    # Self-hosted gateways behind basic auth keep working.
    "https://user:pass@gw.example/v1",
] + [info["base_url"] for info in PROVIDER_REGISTRY.values() if info["base_url"]]


@pytest.fixture(autouse = True)
def _default_policy(monkeypatch):
    """Default deployment: the private-address opt-in is off."""
    monkeypatch.delenv(BLOCK_PRIVATE_ENV, raising = False)
    # The metadata lookup caches its verdict per hostname; a stale entry would
    # carry one test's stubbed resolver into the next.
    _providers._metadata_dns_cache.clear()
    yield
    _providers._metadata_dns_cache.clear()


@pytest.mark.parametrize("url", _SUPPORTED)
def test_supported_base_urls_are_unchanged(url):
    assert validate_provider_base_url(url) == url


@pytest.mark.parametrize("url", _SUPPORTED)
def test_validation_is_idempotent(url):
    once = validate_provider_base_url(url)
    assert validate_provider_base_url(once) == once


def test_trailing_slash_and_whitespace_are_normalized():
    assert validate_provider_base_url("  http://127.0.0.1:8080/v1/  ") == "http://127.0.0.1:8080/v1"


def test_no_dns_lookup_for_shipped_providers_or_ip_literals(monkeypatch):
    """The common path stays resolver-free: shipped hosts and IP literals."""
    # Recorded rather than raised: the lookup runs on a worker thread, where an
    # exception is swallowed into a warning and would never fail this test.
    calls = []

    def _record(host, port, *args, **kwargs):
        calls.append(host)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", _record)
    assert validate_provider_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert validate_provider_base_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"
    assert validate_provider_base_url("http://[fd00:ec2::255]/v1") == "http://[fd00:ec2::255]/v1"
    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://metadata-alias.attacker.test/latest/meta-data",
        "https://metadata-alias.attacker.test/v1",
        # Userinfo and a trailing dot do not hide the name that gets resolved.
        "http://api.openai.com@metadata-alias.attacker.test/v1",
        "http://metadata-alias.attacker.test./v1",
    ],
)
def test_dns_alias_of_a_metadata_address_is_refused(url, monkeypatch):
    """A caller-controlled name pointing at the metadata service is metadata."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))],
    )
    with pytest.raises(ValueError, match = "metadata"):
        validate_provider_base_url(url)


def test_dns_alias_verdict_is_cached(monkeypatch):
    """Repeat validation of the same host does not re-resolve it."""
    calls = []

    def _record(host, port, *args, **kwargs):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _record)
    for _ in range(3):
        assert validate_provider_base_url("https://gw.example/v1") == "https://gw.example/v1"
    assert len(calls) == 1


def test_unresolvable_names_stay_allowed(monkeypatch):
    """docker-compose / service-discovery names resolve in the client's netns."""

    def _unresolvable(*args, **kwargs):
        raise socket.gaierror("not resolvable here")

    monkeypatch.setattr(socket, "getaddrinfo", _unresolvable)
    assert validate_provider_base_url("http://my_ollama:11434/v1") == "http://my_ollama:11434/v1"


def test_a_slow_resolver_does_not_stall_validation(monkeypatch):
    """A resolver that never answers is abandoned, and the URL is allowed."""
    import time as _time

    monkeypatch.setattr(_providers, "_DNS_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _time.sleep(30))
    started = _time.monotonic()
    assert validate_provider_base_url("http://slow.example/v1") == "http://slow.example/v1"
    assert _time.monotonic() - started < 5


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
        # IDNA label separators: httpx encodes the host through idna, which
        # splits on all of these, so they dial 169.254.169.254.
        "http://169。254。169。254/latest/meta-data/",
        "http://169．254．169．254/latest/meta-data/",
        "http://169｡254｡169｡254/latest/meta-data/",
        "http://169.254.169.254。/latest/meta-data/",
        "http://metadata。google。internal/computeMetadata/v1/",
        # Equivalent spellings of the same IPv6 metadata address.
        "http://[fd00:0ec2:0000:0000:0000:0000:0000:0254]/latest/meta-data/",
        "http://[fd00:ec2::0.0.2.84]/latest/meta-data/",
        "http://[FD00:EC2::254]/latest/meta-data/",
        "http://[0:0:0:0:0:ffff:a9fe:a9fe]/latest/meta-data/",
        # Google's IPv6 metadata address on IPv6-only VMs.
        "http://[fd20:ce::254]/computeMetadata/v1/",
        "http://[fd20:0ce:0:0:0:0:0:254]/computeMetadata/v1/",
        # A scope id keeps the address unequal while dialling the same host.
        "http://[fd00:ec2::254%250]/latest/meta-data/",
        "http://[fd00:ec2::254%25eth0]/latest/meta-data/",
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

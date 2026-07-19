# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Canonical website access policies for server-side web tools."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit

_DOMAIN_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MAX_DOMAINS_PER_LIST = 100


def normalize_domain(value: Any) -> str:
    domain = str(value or "").strip().lower()
    if not domain:
        raise ValueError("Website domains cannot be empty")
    if any(ord(char) < 32 for char in domain) or any(
        char in domain for char in ("\\", "/", "@", "?", "#")
    ):
        raise ValueError(f"Invalid website domain: {value!r}")
    bracketed = domain.startswith("[") and domain.endswith("]")
    if domain.startswith("[") != domain.endswith("]"):
        raise ValueError(f"Invalid website domain: {value!r}")
    domain = (domain[1:-1] if bracketed else domain).rstrip(".")
    try:
        return ipaddress.ip_address(domain).compressed
    except ValueError:
        pass
    if ":" in domain:
        raise ValueError("Website limits must contain domains without schemes or ports")
    numeric_parts = domain.split(".")
    if len(numeric_parts) <= 4 and all(
        re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", part) for part in numeric_parts
    ):
        raise ValueError("Non-canonical numeric IP hostnames are not allowed")
    try:
        ascii_domain = domain.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError(f"Invalid website domain: {value!r}") from exc
    if len(ascii_domain) > 253 or not all(
        _DOMAIN_LABEL.fullmatch(label) for label in ascii_domain.split(".")
    ):
        raise ValueError(f"Invalid website domain: {value!r}")
    return ascii_domain


def normalize_website_policy(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {"allowedDomains": [], "blockedDomains": []}
    if not isinstance(value, dict):
        raise ValueError("websitePolicy must be an object")
    unknown = set(value) - {"allowedDomains", "blockedDomains"}
    if unknown:
        raise ValueError(f"Unsupported websitePolicy fields: {', '.join(sorted(unknown))}")

    normalized: dict[str, list[str]] = {}
    for key in ("allowedDomains", "blockedDomains"):
        raw_domains = value.get(key, [])
        if not isinstance(raw_domains, list):
            raise ValueError(f"{key} must be a list")
        if len(raw_domains) > _MAX_DOMAINS_PER_LIST:
            raise ValueError(f"{key} supports at most {_MAX_DOMAINS_PER_LIST} domains")
        domains: list[str] = []
        for raw_domain in raw_domains:
            domain = normalize_domain(raw_domain)
            if domain not in domains:
                domains.append(domain)
        normalized[key] = domains
    return normalized


def _matches_domain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def hostname_allowed(hostname: str, policy: dict[str, Any] | None) -> bool:
    try:
        host = normalize_domain(hostname)
        normalized = normalize_website_policy(policy)
    except ValueError:
        return False
    blocked = normalized["blockedDomains"]
    if any(_matches_domain(host, domain) for domain in blocked):
        return False
    allowed = normalized["allowedDomains"]
    return not allowed or any(_matches_domain(host, domain) for domain in allowed)


def check_url_access(url: str, policy: dict[str, Any] | None) -> tuple[bool, str, str]:
    """Return ``(allowed, reason, canonical_hostname)`` for an HTTP(S) URL."""
    if not isinstance(url, str) or not url.strip():
        return False, "Blocked: URL is empty.", ""
    candidate = url.strip()
    if any(char.isspace() or ord(char) < 32 for char in candidate) or "\\" in candidate:
        return False, "Blocked: URL contains invalid characters.", ""
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in ("http", "https"):
            return False, "Blocked: only http/https URLs are allowed.", ""
        if parsed.username is not None or parsed.password is not None or "%" in parsed.netloc:
            return False, "Blocked: URL credentials or encoded hostnames are not allowed.", ""
        hostname = normalize_domain(parsed.hostname)
        _ = parsed.port
    except (TypeError, ValueError):
        return False, "Blocked: URL has an invalid hostname or port.", ""
    if not hostname_allowed(hostname, policy):
        return False, f"Blocked: website access policy disallows {hostname}.", hostname
    return True, "", hostname


def website_policy_prompt(policy: dict[str, Any] | None) -> str:
    normalized = normalize_website_policy(policy)
    allowed = normalized["allowedDomains"]
    blocked = normalized["blockedDomains"]
    if not allowed and not blocked:
        return ""
    lines = ["Website access limits are enforced by the application."]
    if allowed:
        lines.append(
            "Only search or fetch these domains and their subdomains: "
            + ", ".join(allowed)
            + ". Do not propose, cite, or attempt any other website."
        )
    if blocked:
        lines.append(
            "Never search or fetch these domains or their subdomains: " + ", ".join(blocked) + "."
        )
    lines.append("Blocked search results are unavailable; do not try to work around these limits.")
    return "\n".join(lines)


def scope_search_query(query: str, policy: dict[str, Any] | None) -> str:
    allowed = normalize_website_policy(policy)["allowedDomains"]
    if not allowed or len(allowed) > 8:
        return query
    site_filter = " OR ".join(f"site:{domain}" for domain in allowed)
    return f"{query} ({site_filter})"

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The owner's network rules: which websites the web tools, Python and Terminal may reach."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlsplit

from loggers import get_logger

logger = get_logger(__name__)

SETTING_KEY = "network_firewall"
MODES = ("open", "ask", "allowlist")
MAX_SITES = 200
# Allowed until the owner edits the list. An entry covers its subdomains, so brands that host anyone's content
# (google.com, microsoft.com) are named by the one host that is meant, and github.io and readthedocs are left out.
DEFAULT_ALLOWED_SITES = (
    # packages and code
    "pypi.org",
    "files.pythonhosted.org",
    "download.pytorch.org",
    "github.com",
    "githubusercontent.com",
    "huggingface.co",
    "hf.co",
    "registry.npmjs.org",
    "www.npmjs.com",
    "crates.io",
    # search
    "www.google.com",
    "www.bing.com",
    "duckduckgo.com",
    # reference
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "wiktionary.org",
    "www.britannica.com",
    "openlibrary.org",
    "www.openstreetmap.org",
    # documentation
    "python.org",
    "developer.mozilla.org",
    "developer.apple.com",
    "learn.microsoft.com",
    "docs.docker.com",
    "pytorch.org",
    "tensorflow.org",
    "numpy.org",
    "pandas.pydata.org",
    "scipy.org",
    "scikit-learn.org",
    "matplotlib.org",
    "fastapi.tiangolo.com",
    "starlette.io",
    # research
    "arxiv.org",
    "scholar.google.com",
    "openreview.net",
    "semanticscholar.org",
    "biorxiv.org",
    "medrxiv.org",
    "ncbi.nlm.nih.gov",
    # questions and answers
    "stackoverflow.com",
    "stackexchange.com",
    "askubuntu.com",
    "superuser.com",
    "serverfault.com",
    # standards
    "www.w3.org",
    "tools.ietf.org",
    "datatracker.ietf.org",
    "www.rfc-editor.org",
    # news
    "www.bbc.com",
    "www.bbc.co.uk",
    "www.reuters.com",
    "apnews.com",
    "www.nature.com",
    "www.science.org",
    # public data
    "data.gov",
    "www.census.gov",
    "nasa.gov",
    "www.cdc.gov",
    "www.nih.gov",
    "www.who.int",
    "api.weather.gov",
    "worldtimeapi.org",
)
_CACHE_SECONDS = 2.0
CHECKPOINT_FAILED = (
    "Not run: Settings > Security limits which websites Python and Terminal may reach, and the "
    "network checkpoint for this call could not start."
)
# Sent to one website in one chat before the chat is asked whether more may go.
UPLOAD_BUDGET_BYTES = 1_000_000

_OPEN = ("open", (), ())
# (read at, (mode, allowed domains, blocked domains))
_cache: "tuple[float, tuple[str, tuple[str, ...], tuple[str, ...]]] | None" = None


def get_network_firewall() -> dict[str, Any]:
    from storage.studio_db import get_app_setting
    from utils.account_context import OWNER, run_as

    stored = run_as(OWNER, get_app_setting, SETTING_KEY, None) or {}
    mode = stored.get("mode")
    return {
        "mode": mode if mode in MODES else "open",
        "allowed_sites": list(stored.get("allowed_sites", DEFAULT_ALLOWED_SITES)),
        "blocked_sites": list(stored.get("blocked_sites", [])),
    }


def _clean_sites(values: Any, label: str) -> list[str]:
    from core.inference.web_access_policy import normalize_domain

    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list.")
    sites: list[str] = []
    for item in values:
        text = item.strip() if isinstance(item, str) else ""
        if not text:
            continue
        try:
            host = urlsplit(text).hostname if "://" in text else text
            site = normalize_domain((host or "").removeprefix("*.").lstrip("."))
        except ValueError:
            raise ValueError(
                f"{text[:80]!r} is not a website. Use a name like example.com."
            ) from None
        if site not in sites:
            sites.append(site)
    if len(sites) > MAX_SITES:
        raise ValueError(f"{label}: at most {MAX_SITES} websites.")
    return sites


def set_network_firewall(
    *,
    mode: Any = None,
    allowed_sites: Any = None,
    blocked_sites: Any = None,
) -> dict[str, Any]:
    """Fields left as None keep their stored value."""
    value = get_network_firewall()
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"{str(mode)[:80]!r} is not a network mode.")
        value["mode"] = mode
    if allowed_sites is not None:
        value["allowed_sites"] = _clean_sites(allowed_sites, "Allowed websites")
    if blocked_sites is not None:
        value["blocked_sites"] = _clean_sites(blocked_sites, "Blocked websites")
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    global _cache
    run_as(OWNER, upsert_app_settings, {SETTING_KEY: value})
    _cache = None
    return value


def add_site(kind: str, host: str) -> dict[str, Any]:
    current = get_network_firewall()
    site = _clean_sites([host], "Websites")[0]
    other = "blocked_sites" if kind == "allowed_sites" else "allowed_sites"
    return set_network_firewall(
        **{
            kind: [*current[kind], site],
            other: [item for item in current[other] if item != site],
        }
    )


def _policy() -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_SECONDS:
        return _cache[1]
    try:
        stored = get_network_firewall()
    except Exception:  # noqa: BLE001 - the tools keep working on the last known settings
        logger.warning("Could not read the network firewall settings", exc_info = True)
        return _cache[1] if _cache is not None else _OPEN
    policy = (
        stored["mode"],
        tuple(stored["allowed_sites"]),
        tuple(stored["blocked_sites"]),
    )
    _cache = (now, policy)
    return policy


def _covers(domains: tuple[str, ...], host: str) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def blocked_domains() -> tuple[str, ...]:
    return _policy()[2]


def panel_allowed_domains(chat_domains) -> list[str]:
    """The browser panel takes an empty list as allow-all, so Allowed only narrows the chat's list to ours."""
    mode, allowed, _ = _policy()
    if mode != "allowlist":
        return list(chat_domains)
    if not chat_domains:
        both = list(allowed)
    else:
        both = [d for d in chat_domains if _covers(allowed, d)]
        both += [d for d in allowed if _covers(tuple(chat_domains), d) and d not in both]
    # RFC 6761 reserves .invalid, so an empty intersection still opens nothing.
    return both or ["invalid"]


def blocks(host: str) -> bool:
    return _covers(_policy()[2], host.lower().rstrip("."))


def confines_code() -> bool:
    return _policy()[0] != "open"


def decide(host: str) -> str:
    mode, allowed, blocked = _policy()
    host = host.lower().rstrip(".")
    if _covers(blocked, host):
        return "deny"
    if mode == "open" or _covers(allowed, host):
        return "allow"
    return "ask" if mode == "ask" else "deny"

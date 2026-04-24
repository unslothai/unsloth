# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import tempfile
import threading
from pathlib import Path
from typing import Optional

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import SeedReader

from .config import GitHubRepoSeedSource
from .scraper import ScrapeConfig, materialize_to_jsonl


# In-process cache mapping a stable config signature to the JSONL materialization
# path. A single recipe job invokes the seed reader multiple times (validation,
# preview, per-column sampling), and the default flow re-scrapes the repo on
# every call: for a 2-repo preview that is ~15s of redundant GitHub GraphQL
# traffic before any generation fires. Memoize the materialization so the second
# and third passes reuse the file the first pass wrote. Cache key excludes the
# raw token and uses a short SHA-256 digest so token values never hit memory
# twice and token rotation invalidates cleanly.
_SCRAPE_CACHE: dict[tuple, str] = {}
_SCRAPE_CACHE_LOCK = threading.Lock()


def _scrape_cache_key(cfg: ScrapeConfig) -> tuple:
    token_digest = hashlib.sha256(
        (cfg.token or "").encode("utf-8"),
    ).hexdigest()[:16]
    return (
        tuple(cfg.repos),
        tuple(cfg.item_types),
        cfg.limit,
        bool(cfg.include_comments),
        cfg.max_comments_per_item,
        token_digest,
    )


def _lookup_cached_scrape(key: tuple) -> Optional[str]:
    with _SCRAPE_CACHE_LOCK:
        path = _SCRAPE_CACHE.get(key)
    if path and Path(path).exists():
        return path
    # Stale entry (tmp cleanup, user restarted, ...); drop it so the caller
    # materializes a fresh file rather than returning a dangling path.
    if path:
        with _SCRAPE_CACHE_LOCK:
            _SCRAPE_CACHE.pop(key, None)
    return None


def _store_cached_scrape(key: tuple, path: str) -> None:
    with _SCRAPE_CACHE_LOCK:
        _SCRAPE_CACHE[key] = path


class GitHubRepoSeedReader(SeedReader[GitHubRepoSeedSource]):
    def create_duckdb_connection(self):
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        out_dir = Path(tempfile.gettempdir()) / "studio-github-repo-seed"
        cfg = ScrapeConfig(
            repos = list(self.source.repos),
            token = self.source.token,
            item_types = list(self.source.item_types),
            limit = self.source.limit,
            include_comments = self.source.include_comments,
            max_comments_per_item = self.source.max_comments_per_item,
        )
        cache_key = _scrape_cache_key(cfg)
        cached_path = _lookup_cached_scrape(cache_key)
        if cached_path is not None:
            return cached_path
        path = materialize_to_jsonl(cfg, out_dir)
        _store_cached_scrape(cache_key, str(path))
        return str(path)

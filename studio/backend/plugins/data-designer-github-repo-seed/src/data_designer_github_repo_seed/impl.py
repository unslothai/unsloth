# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import tempfile
from pathlib import Path

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import SeedReader

from .config import GitHubRepoSeedSource
from .scraper import ScrapeConfig, materialize_to_jsonl


class GitHubRepoSeedReader(SeedReader[GitHubRepoSeedSource]):
    def create_duckdb_connection(self):
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        out_dir = Path(tempfile.gettempdir()) / "studio-github-repo-seed"
        cfg = ScrapeConfig(
            repos=list(self.source.repos),
            token=self.source.token,
            item_types=list(self.source.item_types),
            limit=self.source.limit,
            include_comments=self.source.include_comments,
            max_comments_per_item=self.source.max_comments_per_item,
        )
        path = materialize_to_jsonl(cfg, out_dir)
        return str(path)

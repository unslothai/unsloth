# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

from __future__ import annotations

from pathlib import Path

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import SeedReader

from .chunking import materialize_unstructured_seed_dataset
from .config import UnstructuredSeedSource


class UnstructuredSeedReader(SeedReader[UnstructuredSeedSource]):
    def create_duckdb_connection(self):
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        path, _ = materialize_unstructured_seed_dataset(
            source_path = Path(self.source.path),
            chunk_size = self.source.chunk_size,
            chunk_overlap = self.source.chunk_overlap,
        )
        return str(path)

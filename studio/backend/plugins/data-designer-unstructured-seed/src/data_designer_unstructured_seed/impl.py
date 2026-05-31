# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.seed_reader import SeedReader

from .config import UnstructuredSeedSource


class UnstructuredSeedReader(SeedReader[UnstructuredSeedSource]):
    def create_duckdb_connection(self):
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        from .chunking import materialize_multi_file_unstructured_seed
        import json as json_mod

        file_entries: list[tuple[Path, str]] = []
        for p in self.source.paths:
            path_obj = Path(p)
            file_id = path_obj.name.replace(".extracted.txt", "")
            meta_path = path_obj.parent / f"{file_id}.meta.json"
            orig_name = path_obj.name
            if meta_path.exists():
                try:
                    meta = json_mod.loads(meta_path.read_text())
                    orig_name = meta.get("original_filename", path_obj.name)
                except (json_mod.JSONDecodeError, OSError):
                    pass
            file_entries.append((path_obj, orig_name))

        path, _ = materialize_multi_file_unstructured_seed(
            file_entries = file_entries,
            chunk_size = self.source.chunk_size,
            chunk_overlap = self.source.chunk_overlap,
        )
        return str(path)

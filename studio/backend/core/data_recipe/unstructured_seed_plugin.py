from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from .unstructured_seed import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    materialize_unstructured_seed_dataset,
    resolve_chunking,
)

try:
    import data_designer.lazy_heavy_imports as lazy
    from data_designer.config.seed_source import SeedSource
    from data_designer.engine.resources.seed_reader import SeedReader
except ImportError:  # pragma: no cover
    lazy = None

    class SeedSource:  # type: ignore[no-redef]
        pass

    class SeedReader:  # type: ignore[no-redef]
        @classmethod
        def __class_getitem__(cls, _item):
            return cls


class UnstructuredSeedSource(SeedSource):
    seed_type: Literal["unstructured"] = "unstructured"
    path: str = Field(..., min_length=1)
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    @field_validator("path", mode="after")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"Unstructured seed path is not a file: {path}")
        return value

    @field_validator("chunk_size", mode="after")
    @classmethod
    def _validate_chunk_size(cls, value: int) -> int:
        size, _ = resolve_chunking(value, 0)
        return size

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def _validate_chunk_overlap(cls, value: int, info) -> int:
        size = info.data.get("chunk_size", cls.model_fields["chunk_size"].default)
        _, overlap = resolve_chunking(size, value)
        return overlap


class UnstructuredSeedReader(SeedReader[UnstructuredSeedSource]):
    def create_duckdb_connection(self):
        if lazy is None:
            raise RuntimeError("data_designer is not available")
        return lazy.duckdb.connect()

    def get_dataset_uri(self) -> str:
        path, _ = materialize_unstructured_seed_dataset(
            source_path=Path(self.source.path),
            chunk_size=self.source.chunk_size,
            chunk_overlap=self.source.chunk_overlap,
        )
        return str(path)


def ensure_unstructured_seed_plugin_registered() -> None:
    try:
        from data_designer.plugins.plugin import Plugin, PluginType
        from data_designer.plugins.registry import PluginRegistry
    except ImportError:
        return

    registry = PluginRegistry()
    if registry.plugin_exists("unstructured"):
        return

    plugin = Plugin(
        impl_qualified_name="core.data_recipe.unstructured_seed_plugin.UnstructuredSeedReader",
        config_qualified_name="core.data_recipe.unstructured_seed_plugin.UnstructuredSeedSource",
        plugin_type=PluginType.SEED_READER,
    )
    registry._plugins[plugin.name] = plugin  # type: ignore[attr-defined]

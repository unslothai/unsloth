# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys
from pathlib import Path

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inference_routes


def _write_source_page(path: Path, title: str, source_ref: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        "---\n"
        f"title: {title}\n"
        f"source_ref: {source_ref}\n"
        "---\n\n"
        f"# {title}\n",
        encoding = "utf-8",
    )


def test_archive_stale_keeps_distinct_sources_with_same_basename(
    tmp_path: Path,
    monkeypatch,
):
    raw_root = tmp_path / "raw"
    source_a = raw_root / "repoA" / "README.md"
    source_b = raw_root / "repoB" / "README.md"
    source_a.parent.mkdir(parents = True, exist_ok = True)
    source_b.parent.mkdir(parents = True, exist_ok = True)
    source_a.write_text("repo a", encoding = "utf-8")
    source_b.write_text("repo b", encoding = "utf-8")

    sources_dir = tmp_path / "wiki" / "sources"
    _write_source_page(
        sources_dir / "repoa-readme.md",
        "repoA README",
        str(source_a.resolve()),
    )
    _write_source_page(
        sources_dir / "repob-readme.md",
        "repoB README",
        str(source_b.resolve()),
    )

    monkeypatch.setattr(inference_routes, "_WIKI_VAULT_ROOT", tmp_path)

    report = inference_routes._archive_stale_wiki_pages(
        dry_run = True,
        keep_recent_chat = 1,
        keep_recent_per_source = 1,
    )

    # Distinct source identities must not be grouped together just because
    # they share a basename like README.md.
    assert report["moved_count"] == 0
    assert report["errors"] == []

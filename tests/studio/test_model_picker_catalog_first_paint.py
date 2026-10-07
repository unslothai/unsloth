# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-paint contract for the Images / Video model pickers.

Recommended used to render the Hugging Face listing alone, so a task-scoped
picker sat on a spinner for a round trip with its curated models already in
memory, and the bottom spinner stayed up while pages remained rather than while
one was in flight.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PICKERS_TSX = (
    REPO / "studio/frontend/src/features/model-picker/components/model-selector/pickers.tsx"
)


def _source() -> str:
    return PICKERS_TSX.read_text(encoding = "utf-8")


def test_curated_catalog_becomes_rows_without_a_request():
    source = _source()
    seed_start = source.index("const catalogSeedRows = useMemo<HfModelResult[]>(")
    seed_end = source.index("const recommendedRows = useMemo(", seed_start)
    seed = source[seed_start:seed_end]

    assert "dedupe(models.map((model) => model.id))" in seed
    assert "recommendedSearch" not in seed
    assert "if (!task) return [];" in seed
    # Same format policy as the listing rows, so nothing vanishes when one lands.
    assert "taskCatalogFormatMatches(" in seed
    assert "matchesFormatFilter(id, isG, formatFilter)" in seed
    assert "isRecommendable: isRecommendableFormat(r.id, r.isGguf, isMac)," in source
    # Device fit reads the catalog size, not an id "<n>B" guess.
    assert "curatedSizeBytes: catalog ? curatedSizeBytesFor(id, catalog) : undefined," in seed


def test_listing_takes_over_each_id_once_it_reports_it():
    source = _source()
    rows_start = source.index("const recommendedRows = useMemo(")
    rows_end = source.index("const recommendedMeta = useMemo(", rows_start)
    rows = source[rows_start:rows_end]

    # Seeds go only to rows that survived `keep`.
    call = rows[rows.index("orderRecommendedRows({") :]
    call = call[: call.index("\n    });")]
    # Read as a set of arguments, one per line, so additions do not break this.
    args = {line.strip() for line in call.splitlines()[1:]}
    for arg in (
        "seeds: catalogSeedRows,",
        "results: recommendedSearch.results,",
        "keep,",
        "deviceFiltered,",
        "fits,",
    ):
        assert arg in args, f"orderRecommendedRows is no longer passed `{arg}`: {sorted(args)}"
    assert "recommendedSearch.results.map((r) => r.id)" not in rows


def test_bottom_spinner_shows_only_while_a_page_is_in_flight():
    source = _source()
    start = source.index("{recommendedHasMore && (")
    block = source[start : start + 700]

    assert '<div ref={recommendedSentinelRef} className="h-px" />' in block
    assert "{recommendedIsLoadingMore ? (" in block

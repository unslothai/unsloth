# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-paint contract for the Images / Video model pickers.

Recommended used to render the Hugging Face listing alone, so a task-scoped
picker whose curated models were already in memory sat on a spinner for a whole
round trip, and the bottom spinner stayed up for as long as pages remained
rather than while one was in flight.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PICKERS_TSX = REPO / "studio/frontend/src/features/model-picker/components/model-selector/pickers.tsx"


def _source() -> str:
    return PICKERS_TSX.read_text(encoding = "utf-8")


def test_curated_catalog_becomes_rows_without_a_request():
    source = _source()
    seed_start = source.index("const catalogSeedRows = useMemo<HfModelResult[]>(")
    seed_end = source.index("const recommendedRows = useMemo(", seed_start)
    seed = source[seed_start:seed_end]

    # Built from the `models` prop (catalogToModelOptions of IMAGE/VIDEO_CATALOG),
    # never from a fetch result.
    assert "dedupe(models.map((model) => model.id))" in seed
    assert "recommendedSearch" not in seed
    # Chat has no catalog: leave it on the listing.
    assert "if (!task) return [];" in seed
    # Same format policy the listing rows get, so nothing vanishes when it lands.
    assert "isRecommendableFormat(id, isG, isMac)" in seed
    assert "matchesFormatFilter(id, isG, formatFilter)" in seed


def test_listing_takes_over_each_id_once_it_reports_it():
    source = _source()
    rows_start = source.index("const recommendedRows = useMemo(")
    rows_end = source.index("const recommendedMeta = useMemo(", rows_start)
    rows = source[rows_start:rows_end]

    assert "const listed = new Set(recommendedSearch.results.map((r) => r.id));" in rows
    # A listed id renders the listing's row; only an unlisted one keeps its seed.
    assert "if (listedRow) {" in rows
    assert "} else if (!listed.has(seed.id) && (!deviceFiltered || fits(seed))) {" in rows
    # Curated first, then whatever else the listing found, each id once.
    assert (
        "return [...curated, ...rows.filter((r) => !curatedIds.has(r.id))];" in rows
    )
    assert "catalogSeedRows," in rows


def test_bottom_spinner_shows_only_while_a_page_is_in_flight():
    source = _source()
    start = source.index("{recommendedSearch.hasMore && (")
    block = source[start:start + 700]

    # The sentinel still mounts on hasMore so infinite scroll keeps paging...
    assert "<div ref={recommendedSentinelRef} className=\"h-px\" />" in block
    # ...but the spinner is gated on the in-flight flag.
    assert "{recommendedSearch.isLoadingMore ? (" in block

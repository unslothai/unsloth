# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from unforgettable.constants import (
    ACTIVE_RETRIEVE_STATUSES,
    PROVENANCE_WEIGHT,
    SEARCH_FTS_SCAN_CAP,
    typology_class,
)

from .db import get_connection
from .records import get_record

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "run",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "with",
    }
)


def _match_query(query: str) -> str:
    toks = _TOKEN.findall(query.lower())
    content = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
    if not content:
        content = [t for t in toks if len(t) > 1]
    if not content:
        return ""
    return " AND ".join(f'"{t}"' for t in content)


def search_records(
    query: str,
    *,
    top_k: int = 6,
    kinds: Optional[Iterable[str]] = None,
    provenances: Optional[Iterable[str]] = None,
    statuses: Optional[Iterable[str]] = None,
    namespace_id: Optional[str] = None,
    db_path = None,
) -> list[dict[str, Any]]:
    """FTS search, then bias by provenance weight. Default: active only."""
    match = _match_query(query)
    if not match:
        return []
    wanted_status = set(statuses) if statuses is not None else set(ACTIVE_RETRIEVE_STATUSES)
    wanted_kinds = set(kinds) if kinds is not None else None
    wanted_prov = set(provenances) if provenances is not None else None
    conn = get_connection(db_path)
    try:
        scan_cap = min(max(int(top_k) * 8, int(top_k)), SEARCH_FTS_SCAN_CAP)
        rows = conn.execute(
            "SELECT record_id, rank FROM record_fts WHERE record_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, scan_cap),
        ).fetchall()
    finally:
        conn.close()
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        rec = get_record(row["record_id"], db_path = db_path)
        if rec is None:
            continue
        if rec["status"] not in wanted_status:
            continue
        if namespace_id and rec["namespace_id"] != namespace_id:
            continue
        if wanted_kinds is not None and rec["kind"] not in wanted_kinds:
            continue
        if wanted_prov is not None and rec["provenance"] not in wanted_prov:
            continue
        fts_rank = float(row["rank"])
        weight = PROVENANCE_WEIGHT.get(rec["provenance"], 9)
        scored.append((typology_class(rec), weight, fts_rank, rec))
    scored.sort(key = lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in scored[:top_k]]

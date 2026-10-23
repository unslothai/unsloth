# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory streams after Park et al. 2023, "Generative Agents" (arXiv:2304.03442), section 4.

Defaults use the released simulation's weights (recency 0.5 / importance 2 / relevance 3, not the paper's equal
weights) but decay recency by elapsed hours at the paper's 0.995/hour; ``recency_mode = "rank"`` reproduces release.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

STREAM_KIND = "memory_stream"
RECORD_KINDS = ("observation", "reflection", "plan")
DEFAULT_WEIGHTS = {"recency": 0.5, "importance": 2.0, "relevance": 3.0}
DEFAULT_DECAY = 0.995
MAX_RECORDS = 5000

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
# Scripts written without spaces (Chinese, Japanese, Korean) also match on character pairs.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_STOP = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that "
    "these those i you he she they we me him her them us my your his their our what who whom which when where why "
    "how do does did not no so than then there here about into over after before just".split()
)


def empty_stream() -> dict:
    return {"kind": STREAM_KIND, "records": [], "since_reflection": 0.0, "next_id": 0}


def as_stream(value: Any) -> dict:
    """A stream from a block input: a stream, a bare record list, or nothing. Never shares the input's lists."""
    if value is None or value == "":
        return empty_stream()
    if isinstance(value, list):
        value = {"records": value}
    if not isinstance(value, dict) or not isinstance(value.get("records", []), list):
        raise ValueError("Expected a memory stream")
    records = [dict(r) for r in value.get("records") or [] if isinstance(r, dict) and r.get("text")]
    next_id = value.get("next_id")
    if not isinstance(next_id, int) or next_id < len(records):
        next_id = max([len(records)] + [_id_number(r.get("id")) + 1 for r in records])
    # Recall, Reflect and Agent Summary address records by id: a hand-written list may have none.
    seen: set = set()
    for r in records:
        if not isinstance(r.get("id"), str) or not r["id"] or r["id"] in seen:
            r["id"] = f"m{next_id}"
            next_id += 1
        seen.add(r["id"])
    return {
        "kind": STREAM_KIND,
        "records": records,
        "since_reflection": float(value.get("since_reflection") or 0.0),
        "next_id": next_id,
    }


def _id_number(record_id: Any) -> int:
    m = re.fullmatch(r"m(\d+)", str(record_id or ""))
    return int(m.group(1)) if m else -1


def clamp_importance(value: Any, default: int = 5) -> int:
    try:
        return max(1, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def new_record(
    stream: dict, text: str, kind: str, importance: int, now: float, **extra: Any
) -> dict:
    record = {
        "id": f"m{stream['next_id']}",
        "text": text,
        "kind": kind if kind in RECORD_KINDS else "observation",
        "t_created": now,
        "t_access": now,
        "importance": clamp_importance(importance),
    }
    record.update({k: v for k, v in extra.items() if v is not None})
    stream["next_id"] += 1
    return record


def tokens(text: str) -> set[str]:
    words = {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}
    for w in [w for w in words if _CJK.search(w)]:
        words.update(w[i : i + 2] for i in range(len(w) - 1))
    return words


def keyword_relevance(query: str, text: str) -> float:
    """Word-overlap fallback when no embedding model is available: |q & t| / |q|."""
    q = tokens(query)
    return len(q & tokens(text)) / len(q) if q else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def normalise(values: list[float]) -> list[float]:
    """Min-max to [0, 1]; a term with no spread ranks nothing, so it contributes 0 to every candidate."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def recency(records: list[dict], now: float, decay: float, mode: str) -> list[float]:
    if mode == "rank":
        # The released simulation: newest access first, decay ** rank (it uses 0.99).
        order = sorted(range(len(records)), key = lambda i: -float(records[i].get("t_access") or 0))
        out = [0.0] * len(records)
        for rank, i in enumerate(order):
            out[i] = decay**rank
        return out
    # Time 0 is a real time (a game clock starts there), so only a missing stamp counts as "now".
    return [
        decay ** max(0.0, now - (now if r.get("t_access") is None else float(r["t_access"])))
        for r in records
    ]


def score(
    records: list[dict],
    *,
    now: float,
    query: str,
    query_vec: Optional[list[float]] = None,
    weights: Optional[dict] = None,
    decay: float = DEFAULT_DECAY,
    recency_mode: str = "hours",
) -> list[dict]:
    """Each record's normalised components and total; relevance is cosine when every vector is known, else keywords."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    rec = normalise(recency(records, now, decay, recency_mode))
    imp = normalise([float(r.get("importance") or 0) for r in records])
    if query_vec is not None and all(isinstance(r.get("embedding"), list) for r in records):
        raw_rel = [cosine(query_vec, r["embedding"]) for r in records]
    else:
        raw_rel = [keyword_relevance(query, str(r.get("text") or "")) for r in records]
    rel = normalise(raw_rel)
    return [
        {
            "recency": rec[i],
            "importance": imp[i],
            "relevance": rel[i],
            "score": w["recency"] * rec[i] + w["importance"] * imp[i] + w["relevance"] * rel[i],
        }
        for i in range(len(records))
    ]


def top(records: list[dict], scores: list[dict], k: int) -> list[int]:
    """Indices of the k best records; ties go to the newer record so the order is stable."""
    order = sorted(
        range(len(records)),
        key = lambda i: (-scores[i]["score"], -float(records[i].get("t_created") or 0), i),
    )
    return order[: max(0, k)]


def cap(stream: dict, max_records: int) -> None:
    """Drop the least important, least recently used records past the cap; a record a reflection cites stays."""
    records = stream["records"]
    # Dropping a reflection frees what it cited, so protection is recomputed until the cap holds.
    while len(records) > max_records:
        cited = {c for r in records for c in r.get("cites") or []}
        removable = sorted(
            (r for r in records if r["id"] not in cited),
            key = lambda r: (r.get("importance") or 0, float(r.get("t_access") or 0)),
        )
        if not removable:
            break
        drop = {r["id"] for r in removable[: len(records) - max_records]}
        records = [r for r in records if r["id"] not in drop]
    stream["records"] = records


def context_text(records: Iterable[dict]) -> str:
    return "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(records))


def round_vector(vec: Any) -> Optional[list[float]]:
    # Five decimals keep cosine ranking intact and the stream about 3x smaller in saves and run history.
    if not isinstance(vec, list) or not vec:
        return None
    return [round(float(x), 5) for x in vec]

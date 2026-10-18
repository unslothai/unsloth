# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The voice decision log: one JSONL row per judge / filler / interject / addressed verdict, plus the outcome
the loop reports later. Rows hold only what training needs (the transcript the model got, its
verdict, the outcome, timing): no audio, no reply text, no paths."""

import json
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MODELS = ("judge", "filler", "interject", "addressed")
# Matches JUDGE_SEND_THRESHOLD in the frontend client; a judge row stores p_complete as confidence.
JUDGE_SEND_THRESHOLD = 0.65

# A verdict is wrong when the outcome contradicts it. "restart" has no derivable error signal yet
# (the reply changing topic is not observable), so only ignore->repeated and queue->reasked count for
# interject; a quiet filler has none either. For addressed, a turn held back as other / reading that
# the user then says again was for the assistant; an answered turn that gets cut ("not you") is
# logged as the outcome "cut" but not counted, since a barge-in has other causes.
ERROR_KINDS: dict[str, dict[tuple[str, str], str]] = {
    "judge": {
        ("finished", "continued_quickly"): "finished_but_continued",
        ("hold", "flushed"): "held_but_flushed",
    },
    "interject": {
        ("ignore", "repeated"): "ignored_but_repeated",
        ("queue", "reasked"): "queued_but_reasked",
    },
    "filler": {("say", "echoed"): "echoed", ("say", "too_late"): "too_late"},
    "addressed": {
        ("other", "repeated"): "missed_address",
        ("reading", "repeated"): "missed_address",
    },
}

_LOCK = threading.Lock()


def log_dir() -> Path:
    try:
        from utils.paths.storage_roots import studio_root
        root = studio_root()
    except Exception:  # noqa: BLE001 - the legacy root is the fallback everywhere else too
        root = Path.home() / ".unsloth" / "studio"
    return Path(os.environ.get("UNSLOTH_VOICE_LOG_DIR") or (root / "voice"))


def log_path() -> Path:
    return log_dir() / "laya-log.jsonl"


def outcomes_path() -> Path:
    # Outcomes live in a side file keyed by row id and merge on read: an append never rewrites the
    # log under a concurrent judge call.
    return log_dir() / "laya-outcomes.jsonl"


def _append_line(path: Path, row: dict) -> None:
    with _LOCK:
        path.parent.mkdir(parents = True, exist_ok = True)
        with open(path, "a", encoding = "utf-8") as f:
            f.write(json.dumps(row, ensure_ascii = False) + "\n")


def _read_lines(path: Path) -> list[dict]:
    try:
        with open(path, encoding = "utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def judge_verdict(p_complete: float) -> str:
    return "finished" if p_complete >= JUDGE_SEND_THRESHOLD else "hold"


def append(
    model: str,
    inputs: dict,
    verdict: str,
    confidence: Optional[float],
    latency_ms: float,
    *,
    thread_id: Optional[str] = None,
    source: str = "app",
) -> str:
    row_id = uuid.uuid4().hex
    row = {
        "id": row_id,
        "t": datetime.now(timezone.utc).isoformat(timespec = "milliseconds"),
        "model": model,
        "thread_id": thread_id or None,
        "source": source,
        "input": inputs,
        "verdict": verdict,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "latency_ms": round(float(latency_ms), 2),
        "outcome": None,
    }
    _append_line(log_path(), row)
    return row_id


def set_outcome(
    row_id: str,
    outcome: str,
    detail: Optional[dict] = None,
) -> None:
    _append_line(
        outcomes_path(),
        {"id": row_id, "outcome": outcome, "detail": detail or None, "at": time.time()},
    )


def read_rows() -> list[dict]:
    """Every log row with its latest outcome merged in."""
    rows = _read_lines(log_path())
    outcomes: dict[str, dict] = {}
    for o in _read_lines(outcomes_path()):
        if o.get("id"):
            outcomes[o["id"]] = o
    for row in rows:
        o = outcomes.get(row.get("id", ""))
        if o:
            row["outcome"] = o.get("outcome")
            if o.get("detail"):
                row["outcome_detail"] = o["detail"]
    return rows


def error_kind(row: dict) -> Optional[str]:
    outcome = row.get("outcome")
    if not outcome:
        return None
    return ERROR_KINDS.get(row.get("model", ""), {}).get((str(row.get("verdict")), str(outcome)))


def disagrees(row: dict) -> bool:
    return error_kind(row) is not None


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def stats(rows: Optional[list[dict]] = None) -> dict:
    rows = read_rows() if rows is None else rows
    per: dict[str, Any] = {}
    for model in MODELS:
        mine = [r for r in rows if r.get("model") == model]
        judged = [r for r in mine if r.get("outcome")]
        counts = {k: 0 for k in ERROR_KINDS[model].values()}
        for r in judged:
            k = error_kind(r)
            if k:
                counts[k] += 1
        wrong = sum(counts.values())
        lat = [
            float(r["latency_ms"]) for r in mine if isinstance(r.get("latency_ms"), (int, float))
        ]
        per[model] = {
            "verdicts": len(mine),
            "with_outcome": len(judged),
            "accuracy": None if not judged else round((len(judged) - wrong) / len(judged), 4),
            "errors": counts,
            "latency_p50_ms": _percentile(lat, 0.5),
            "latency_p95_ms": _percentile(lat, 0.95),
        }
    try:
        size = log_path().stat().st_size + (
            outcomes_path().stat().st_size if outcomes_path().exists() else 0
        )
    except OSError:
        size = 0
    return {"models": per, "rows": len(rows), "log_bytes": size, "path": str(log_path())}


def _training_label(row: dict) -> str:
    """The label a trainer should learn for this row: the verdict when the outcome agreed with it, the
    corrected one when it did not."""
    model, verdict = row.get("model"), str(row.get("verdict"))
    wrong = disagrees(row)
    if model == "judge":
        finished = (verdict == "finished") != wrong
        return "complete" if finished else "incomplete"
    if model == "interject":
        return "restart" if wrong else verdict
    if model == "filler":
        return "none" if wrong else str((row.get("input") or {}).get("class") or verdict)
    if model == "addressed":
        return "assistant" if wrong else verdict
    return verdict


def _training_text(row: dict) -> str:
    inp = row.get("input") or {}
    return str(inp.get("text") or inp.get("utterance") or "")


def to_training_row(row: dict) -> dict:
    out = {
        "text": _training_text(row),
        "label": _training_label(row),
        "model": row.get("model"),
        "verdict": row.get("verdict"),
        "outcome": row.get("outcome"),
        "confidence": row.get("confidence"),
        "t": row.get("t"),
        "thread_id": row.get("thread_id"),
    }
    inp = row.get("input") or {}
    for key in ("prompt", "prev_q", "elapsed_s", "prev"):
        if key in inp:
            out[key] = inp[key]
    return out


def export_rows(
    rows: Optional[list[dict]] = None,
    *,
    model: str = "all",
    only: str = "disagreements",
    thread_id: Optional[str] = None,
    since: Optional[str] = None,
    mix: bool = False,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    rows = read_rows() if rows is None else rows
    if model != "all":
        rows = [r for r in rows if r.get("model") == model]
    if thread_id:
        rows = [r for r in rows if r.get("thread_id") == thread_id]
    if since:
        rows = [r for r in rows if str(r.get("t") or "") >= since]
    if only == "all":
        picked = rows
    else:
        picked = [r for r in rows if disagrees(r)]
        if mix:
            # One random agreeing row per disagreement keeps a fine-tune from only ever seeing corrections.
            agreeing = [r for r in rows if r.get("outcome") and not disagrees(r)]
            draw = rng or random
            extra = draw.sample(agreeing, min(len(agreeing), len(picked))) if agreeing else []
            picked = sorted(picked + extra, key = lambda r: str(r.get("t") or ""))
    return [to_training_row(r) for r in picked]


def to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii = False) + "\n" for r in rows)


def _rewrite(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")
    os.replace(tmp, path)


def clear() -> int:
    with _LOCK:
        n = len(_read_lines(log_path()))
        for p in (log_path(), outcomes_path()):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    return n


def delete_thread_rows(thread_ids) -> int:
    """Drop every row logged for the given chats; the outcomes side file follows."""
    ids = {str(t) for t in thread_ids or [] if t}
    if not ids:
        return 0
    with _LOCK:
        rows = _read_lines(log_path())
        keep = [r for r in rows if r.get("thread_id") not in ids]
        if len(keep) == len(rows):
            return 0
        gone = {r.get("id") for r in rows if r.get("thread_id") in ids}
        _rewrite(log_path(), keep)
        if outcomes_path().exists():
            _rewrite(
                outcomes_path(),
                [o for o in _read_lines(outcomes_path()) if o.get("id") not in gone],
            )
    return len(rows) - len(keep)

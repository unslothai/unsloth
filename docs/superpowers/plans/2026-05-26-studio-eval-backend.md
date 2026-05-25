# Studio Eval Backend (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Studio Eval **backend** — a pluggable metric registry, a dependency-injected eval runner, persistence, an `EvalJobManager` (training-like lifecycle), and `/api/eval` routes — so a model can be evaluated over a dataset and scored, with run history. (The Eval **frontend** is a separate follow-on plan.)

**Architecture:** New framework under `studio/backend/eval/` beside the existing `json_score/`. Metrics are small plugins built from per-run config. The runner is a pure loop taking an injected `generate` callable + a pre-loaded `examples` list, so it's unit-testable without a model. The job manager runs the runner in a background thread, reusing the existing inference path; results persist to two new SQLite tables mirroring `training_runs`/`training_metrics`. Routes mirror training's start/progress(SSE)/cancel/history shape.

**Tech Stack:** Python 3.13 (Studio venv), FastAPI, Pydantic, SQLite (`storage/studio_db.py`), `datasets` (HF), `rapidfuzz` (already present via `json_score`), pytest + FastAPI `TestClient`.

**Spec:** `docs/superpowers/specs/2026-05-26-studio-eval-framework-design.md`

**Conventions for every new `.py` file:** start with Studio's AGPL header (exactly two lines):

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
```

**Imports:** flat, as the Studio backend runs with `studio/backend/` on `sys.path` — `from eval.metrics... import`, `from storage import studio_db`, `from models import ...`, `from routes.eval import router`. Within the `eval` package use relative imports.

**Environment / running tests:** use `~/.unsloth/studio/unsloth_studio/bin/python` (the Studio venv — has fastapi/httpx/pytest/datasets/rapidfuzz). NEVER use bare `python`. Run with explicit paths from the repo root:
```bash
~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_<area>.py -v
```
A repo-root conftest may print an `unsloth` banner to stderr — that's noise; read the pytest summary. `studio/backend/tests/conftest.py` puts the backend root on `sys.path`, enabling flat imports.

**File map (created/modified):**
- `studio/backend/eval/metrics/__init__.py`, `base.py`, `exact_match.py`, `text_similarity.py`, `json_document.py`, `registry.py`
- `studio/backend/eval/runner.py`
- `studio/backend/eval/dataset.py` (load examples from HF/local)
- `studio/backend/eval/inference_adapter.py` (bind the orchestrator backend to the runner's `generate`)
- `studio/backend/eval/jobs.py` (`EvalJobManager`)
- `studio/backend/models/eval.py` (Pydantic request/response models)
- `studio/backend/routes/eval.py`
- modify `storage/studio_db.py` (eval tables + helpers + reconciliation), `models/__init__.py`, `routes/__init__.py`, `main.py`
- remove `studio/backend/routes/scoring.py` + `studio/backend/models/scoring.py` + their registration (superseded). The `json_score` library stays.
- tests: `studio/backend/tests/test_eval_metrics.py`, `test_eval_runner.py`, `test_eval_dataset.py`, `test_eval_jobs.py`, `test_eval_db.py`, `test_eval_routes.py`

**Note on two integration seams (flagged, not placeholders):** the inference adapter (Task 8) wraps `core.inference.get_inference_backend()` whose `generate_chat_response` runs in a subprocess; and the DB tests rely on pointing `studio_db` at a temp database. Each task below gives the concrete code plus the exact verification command.

---

## Task 1: eval_runs / eval_results tables + DB helpers

**Files:**
- Modify: `studio/backend/storage/studio_db.py`
- Test: `studio/backend/tests/test_eval_db.py`

- [ ] **Step 1: Find the schema + DB-path hooks**

Open `studio/backend/storage/studio_db.py` and locate: the module-level DB path constant (used by `get_connection()`), the `_ensure_schema()` function (the `CREATE TABLE IF NOT EXISTS` block), and `cleanup_orphaned_runs()`. Confirm `get_connection()` sets `row_factory = sqlite3.Row` and `PRAGMA foreign_keys = ON`. Note the exact DB-path variable name (e.g. `DB_PATH` / `_DB_PATH`) — the test monkeypatches it.

- [ ] **Step 2: Write the failing test**

`studio/backend/tests/test_eval_db.py` (AGPL header, then):

```python
import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """studio_db pointed at a throwaway sqlite file.

    IMPORTANT: never reload the module against the default path (that would
    create/mutate the user's real studio.db). Repoint the path and reset any
    cached connection / once-flag so the next get_connection() opens the temp DB
    and re-runs _ensure_schema there. Adjust the attribute names below to match
    what Step 1 found (or use monkeypatch.setenv if the path comes from an env var).
    """
    from storage import studio_db

    monkeypatch.setattr(studio_db, "DB_PATH", tmp_path / "studio.db", raising=True)
    for attr in ("_CONNECTION", "_conn", "_connection"):
        if hasattr(studio_db, attr):
            monkeypatch.setattr(studio_db, attr, None, raising=False)
    for flag in ("_SCHEMA_READY", "_schema_initialized", "_INITIALIZED"):
        if hasattr(studio_db, flag):
            monkeypatch.setattr(studio_db, flag, False, raising=False)
    return studio_db


def test_create_and_get_eval_run(db):
    db.create_eval_run(
        id="run1", model_identifier="hf/model", dataset_ref="data.jsonl",
        metric_name="exact_match", config_json="{}", started_at="2026-05-26T00:00:00",
        num_examples=2,
    )
    run = db.get_eval_run("run1")
    assert run["id"] == "run1"
    assert run["status"] == "running"
    assert run["metric_name"] == "exact_match"
    assert run["num_examples"] == 2


def test_insert_results_and_finish(db):
    db.create_eval_run(
        id="run2", model_identifier="m", dataset_ref="d", metric_name="exact_match",
        config_json="{}", started_at="2026-05-26T00:00:00", num_examples=2,
    )
    db.insert_eval_result(run_id="run2", idx=0, input_text="a", prediction_text="x",
                          reference_text="x", score=1.0, breakdown_json=None, error=None)
    db.insert_eval_result(run_id="run2", idx=1, input_text="b", prediction_text="y",
                          reference_text="z", score=0.0, breakdown_json=None, error=None)
    db.finish_eval_run(id="run2", status="completed", ended_at="2026-05-26T00:01:00",
                       avg_score=0.5, error_message=None)
    run = db.get_eval_run("run2")
    assert run["status"] == "completed"
    assert run["avg_score"] == 0.5
    results = db.get_eval_results("run2", limit=10, offset=0)
    assert results["total"] == 2
    assert [r["idx"] for r in results["results"]] == [0, 1]
    assert results["results"][0]["score"] == 1.0


def test_list_eval_runs_newest_first(db):
    db.create_eval_run(id="r_old", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T00:00:00", num_examples=1)
    db.create_eval_run(id="r_new", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T01:00:00", num_examples=1)
    listing = db.list_eval_runs(limit=50, offset=0)
    assert listing["total"] == 2
    assert listing["runs"][0]["id"] == "r_new"  # newest first


def test_cleanup_marks_running_eval_interrupted(db):
    db.create_eval_run(id="stuck", model_identifier="m", dataset_ref="d",
                       metric_name="exact_match", config_json="{}",
                       started_at="2026-05-26T00:00:00", num_examples=1)
    db.cleanup_orphaned_runs()
    assert db.get_eval_run("stuck")["status"] == "interrupted"
```

> If Step 1 shows the path variable is not named `DB_PATH`, update the two `monkeypatch.setattr(..., "DB_PATH", ...)` lines to the real name. This is the only spot the test depends on internals.

- [ ] **Step 3: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_db.py -v`
Expected: FAIL — `AttributeError: module 'storage.studio_db' has no attribute 'create_eval_run'`.

- [ ] **Step 4: Add the tables to `_ensure_schema()`**

Inside `_ensure_schema()` in `studio/backend/storage/studio_db.py`, after the existing `CREATE TABLE` statements, add:

```python
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            model_identifier TEXT NOT NULL,
            dataset_ref TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            num_examples INTEGER,
            avg_score REAL,
            error_message TEXT,
            display_name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_results (
            run_id TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            input_text TEXT,
            prediction_text TEXT,
            reference_text TEXT,
            score REAL,
            breakdown_json TEXT,
            error TEXT,
            PRIMARY KEY (run_id, idx)
        )
        """
    )
```

- [ ] **Step 5: Add the helper functions**

Add these module-level functions to `studio/backend/storage/studio_db.py` (mirror the existing `create_run`/`finish_run`/`list_runs` style — each opens `get_connection()`, commits, closes):

```python
def create_eval_run(
    id: str, model_identifier: str, dataset_ref: str, metric_name: str,
    config_json: str, started_at: str, num_examples: Optional[int],
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO eval_runs
               (id, status, model_identifier, dataset_ref, metric_name,
                config_json, started_at, num_examples)
               VALUES (?, 'running', ?, ?, ?, ?, ?, ?)""",
            (id, model_identifier, dataset_ref, metric_name, config_json,
             started_at, num_examples),
        )
        conn.commit()
    finally:
        conn.close()


def insert_eval_result(
    run_id: str, idx: int, input_text: str, prediction_text: str,
    reference_text: str, score: Optional[float], breakdown_json: Optional[str],
    error: Optional[str],
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO eval_results
               (run_id, idx, input_text, prediction_text, reference_text,
                score, breakdown_json, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, idx) DO UPDATE SET
                 input_text=excluded.input_text,
                 prediction_text=excluded.prediction_text,
                 reference_text=excluded.reference_text,
                 score=excluded.score,
                 breakdown_json=excluded.breakdown_json,
                 error=excluded.error""",
            (run_id, idx, input_text, prediction_text, reference_text,
             score, breakdown_json, error),
        )
        conn.commit()
    finally:
        conn.close()


def finish_eval_run(
    id: str, status: str, ended_at: str, avg_score: Optional[float],
    error_message: Optional[str],
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE eval_runs
               SET status=?, ended_at=?, avg_score=?, error_message=?
               WHERE id=?""",
            (status, ended_at, avg_score, error_message, id),
        )
        conn.commit()
    finally:
        conn.close()


def get_eval_run(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM eval_runs WHERE id=?", (id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_eval_runs(limit: int = 50, offset: int = 0) -> dict:
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY started_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {"runs": [dict(r) for r in rows], "total": total}
    finally:
        conn.close()


def get_eval_results(run_id: str, limit: int = 100, offset: int = 0) -> dict:
    conn = get_connection()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM eval_results WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE run_id=? ORDER BY idx ASC LIMIT ? OFFSET ?",
            (run_id, limit, offset),
        ).fetchall()
        return {"results": [dict(r) for r in rows], "total": total}
    finally:
        conn.close()
```

- [ ] **Step 6: Extend `cleanup_orphaned_runs()` to reconcile eval runs**

In `cleanup_orphaned_runs()`, after the existing `UPDATE training_runs ...`, add (reuse the same `conn`):

```python
        conn.execute(
            """UPDATE eval_runs
               SET status='interrupted',
                   error_message='Server restarted during eval',
                   ended_at=?
               WHERE status='running'""",
            (datetime.now(timezone.utc).isoformat(),),
        )
```
(Keep the single `conn.commit()` at the end of the function.)

- [ ] **Step 7: Run to verify it passes**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_db.py -v`
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add studio/backend/storage/studio_db.py studio/backend/tests/test_eval_db.py
git commit -m "$(printf 'feat(studio/eval): eval_runs + eval_results tables and DB helpers\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: Metric base + registry + exact_match

**Files:**
- Create: `studio/backend/eval/metrics/__init__.py`, `base.py`, `exact_match.py`, `registry.py`
- Test: `studio/backend/tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

`studio/backend/tests/test_eval_metrics.py` (AGPL header, then):

```python
import pytest

from eval.metrics.registry import make_scorer, list_metrics
from eval.metrics.base import MetricResult


def test_exact_match_basic():
    score = make_scorer("exact_match", {})
    assert score("Yes", "Yes").score == 1.0
    assert score("Yes", "No").score == 0.0


def test_exact_match_case_insensitive_and_strip():
    score = make_scorer("exact_match", {"case_insensitive": True, "strip": True})
    assert score("  yes ", "YES").score == 1.0


def test_exact_match_strip_default_true():
    score = make_scorer("exact_match", {})
    assert score("yes ", "yes").score == 1.0  # trailing space stripped


def test_make_scorer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        make_scorer("nope", {})


def test_list_metrics_includes_exact_match_schema():
    metrics = {m["name"]: m for m in list_metrics()}
    assert "exact_match" in metrics
    em = metrics["exact_match"]
    assert em["reference_kind"] == "text"
    field_names = {f["name"] for f in em["config_fields"]}
    assert {"case_insensitive", "strip"} <= field_names


def test_metric_result_shape():
    r = MetricResult(score=0.5)
    assert r.score == 0.5 and r.breakdown is None and r.error is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_metrics.py -v`
Expected: FAIL — import error (`eval.metrics` missing).

- [ ] **Step 3: Implement base.py**

`studio/backend/eval/metrics/__init__.py`: AGPL header only.

`studio/backend/eval/metrics/base.py` (AGPL header, then):

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# A scorer compares one prediction string against a reference value.
Scorer = Callable[[str, Any], "MetricResult"]


@dataclass
class MetricResult:
    score: float                      # in [0, 1]
    breakdown: Optional[dict] = None  # e.g. serialized ScoreNode for JSON
    error: Optional[str] = None       # per-example issue (e.g. unparseable JSON)


@dataclass
class ConfigField:
    name: str
    type: str          # "bool" | "float" | "string" | "json"
    default: Any
    label: str


@dataclass
class MetricSpec:
    name: str
    label: str
    reference_kind: str            # "text" | "json"
    config_fields: list[ConfigField]
    build: Callable[[dict], Scorer]  # config -> scorer closure
```

- [ ] **Step 4: Implement exact_match.py**

`studio/backend/eval/metrics/exact_match.py` (AGPL header, then):

```python
from __future__ import annotations

from typing import Any

from .base import ConfigField, MetricResult, MetricSpec, Scorer


def _build(config: dict) -> Scorer:
    case_insensitive = bool(config.get("case_insensitive", False))
    strip = bool(config.get("strip", True))

    def score(prediction: str, reference: Any) -> MetricResult:
        a = "" if prediction is None else str(prediction)
        b = "" if reference is None else str(reference)
        if strip:
            a, b = a.strip(), b.strip()
        if case_insensitive:
            a, b = a.lower(), b.lower()
        return MetricResult(score=1.0 if a == b else 0.0)

    return score


SPEC = MetricSpec(
    name="exact_match",
    label="Exact match",
    reference_kind="text",
    config_fields=[
        ConfigField("case_insensitive", "bool", False, "Case-insensitive"),
        ConfigField("strip", "bool", True, "Trim whitespace"),
    ],
    build=_build,
)
```

- [ ] **Step 5: Implement registry.py**

`studio/backend/eval/metrics/registry.py` (AGPL header, then):

```python
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .base import MetricSpec, Scorer
from . import exact_match

_SPECS: dict[str, MetricSpec] = {
    exact_match.SPEC.name: exact_match.SPEC,
}


def register(spec: MetricSpec) -> None:
    _SPECS[spec.name] = spec


def make_scorer(name: str, config: dict) -> Scorer:
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unknown metric {name!r}. Known: {sorted(_SPECS)}")
    return spec.build(config or {})


def list_metrics() -> list[dict]:
    out = []
    for spec in _SPECS.values():
        out.append({
            "name": spec.name,
            "label": spec.label,
            "reference_kind": spec.reference_kind,
            "config_fields": [asdict(f) for f in spec.config_fields],
        })
    return out
```

- [ ] **Step 6: Run to verify it passes**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/eval/metrics studio/backend/tests/test_eval_metrics.py
git commit -m "$(printf 'feat(studio/eval): metric base, registry, exact_match metric\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: text_similarity + json_document metrics

**Files:**
- Create: `studio/backend/eval/metrics/text_similarity.py`, `json_document.py`
- Modify: `studio/backend/eval/metrics/registry.py`
- Test: `studio/backend/tests/test_eval_metrics.py`

- [ ] **Step 1: Append failing tests**

Append to `studio/backend/tests/test_eval_metrics.py`:

```python
def test_text_similarity_identical_and_different():
    score = make_scorer("text_similarity", {"threshold": 0.5})
    assert score("Acme Corporation", "Acme Corporation").score == 1.0
    assert score("abcdefgh", "zzzzzzzz").score == 0.0  # below threshold -> 0


def test_json_document_perfect_and_partial():
    schema = {"total": {"type": "money"}, "currency": "categorical"}
    score = make_scorer("json_document", {"schema": schema})
    gt = {"total": 100, "currency": "USD"}
    perfect = score('{"total": 100, "currency": "USD"}', gt)
    assert perfect.score == 1.0
    assert perfect.breakdown is not None and "children" in perfect.breakdown
    partial = score('{"total": 90, "currency": "EUR"}', gt)
    assert abs(partial.score - 0.45) < 1e-9  # money .9 + categorical 0, /2


def test_json_document_reference_as_json_string():
    score = make_scorer("json_document", {})
    # reference may arrive as a JSON string (dataset column) -> parsed
    r = score('{"a": "x"}', '{"a": "x"}')
    assert r.score == 1.0


def test_json_document_unparseable_prediction_is_error_zero():
    score = make_scorer("json_document", {})
    r = score("the model refused", {"a": "x"})
    assert r.score == 0.0 and r.error is not None


def test_json_document_bad_reference_is_error_zero():
    score = make_scorer("json_document", {})
    r = score('{"a": "x"}', "not json")
    assert r.score == 0.0 and r.error is not None


def test_text_similarity_and_json_in_registry():
    names = {m["name"] for m in list_metrics()}
    assert {"text_similarity", "json_document"} <= names
    kinds = {m["name"]: m["reference_kind"] for m in list_metrics()}
    assert kinds["json_document"] == "json"
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_metrics.py -k "text_similarity or json_document" -v`
Expected: FAIL — `Unknown metric 'text_similarity'`.

- [ ] **Step 3: Implement text_similarity.py**

`studio/backend/eval/metrics/text_similarity.py` (AGPL header, then):

```python
from __future__ import annotations

from typing import Any

from json_score.comparators import string_comparator
from .base import ConfigField, MetricResult, MetricSpec, Scorer


def _build(config: dict) -> Scorer:
    threshold = float(config.get("threshold", 0.5))
    cmp = string_comparator(threshold=threshold)

    def score(prediction: str, reference: Any) -> MetricResult:
        ref = "" if reference is None else str(reference)
        return MetricResult(score=float(cmp(ref, prediction)))

    return score


SPEC = MetricSpec(
    name="text_similarity",
    label="Text similarity (ANLS)",
    reference_kind="text",
    config_fields=[
        ConfigField("threshold", "float", 0.5, "Similarity threshold"),
    ],
    build=_build,
)
```

> `string_comparator` lives at `studio/backend/eval/json_score/comparators.py`; the flat import is `from json_score.comparators import string_comparator`.

- [ ] **Step 4: Implement json_document.py**

`studio/backend/eval/metrics/json_document.py` (AGPL header, then):

```python
from __future__ import annotations

import json
from typing import Any

from json_score import json_anls_score, score_from_text
from json_score.core import ScoreNode
from .base import ConfigField, MetricResult, MetricSpec, Scorer


def _serialize(node: ScoreNode) -> dict:
    out: dict[str, Any] = {"score": node.score, "n_leaves": node.n_leaves}
    if node.note is not None:
        out["note"] = node.note
    if node.matched_option is not None:
        out["matched_option"] = node.matched_option
    children = node.children
    if isinstance(children, dict):
        out["children"] = {k: _serialize(v) for k, v in children.items()}
    elif isinstance(children, list):
        out["children"] = [_serialize(v) for v in children]
    return out


def _build(config: dict) -> Scorer:
    schema = config.get("schema")
    default_comparator = config.get("default_comparator", "string")

    def score(prediction: str, reference: Any) -> MetricResult:
        # Reference column values arrive as strings; parse to JSON.
        ref = reference
        if isinstance(reference, str):
            try:
                ref = json.loads(reference)
            except (ValueError, TypeError):
                return MetricResult(score=0.0, error="reference is not valid JSON")
        try:
            value, node = score_from_text(
                ref, prediction, schema,
                default_comparator=default_comparator, return_key_scores=True,
            )
        except (ValueError, TypeError) as exc:
            return MetricResult(score=0.0, error=f"scoring failed: {exc}")
        err = None if node.note != "unparseable prediction" else "unparseable prediction"
        return MetricResult(score=value, breakdown=_serialize(node), error=err)

    return score


SPEC = MetricSpec(
    name="json_document",
    label="JSON document score",
    reference_kind="json",
    config_fields=[
        ConfigField("schema", "json", None, "Field schema (optional)"),
        ConfigField("default_comparator", "string", "string", "Default comparator"),
    ],
    build=_build,
)
```

> `score_from_text(ground_truth, raw_text, schema=None, *, default_comparator, return_key_scores)` returns `(score, ScoreNode)` when `return_key_scores=True`; unparseable predictions yield a `ScoreNode` with `note == "unparseable prediction"` and score `0.0` (from Task 7 of the json_score work).

- [ ] **Step 5: Register both metrics**

In `studio/backend/eval/metrics/registry.py`, update the imports and `_SPECS`:

```python
from . import exact_match, text_similarity, json_document

_SPECS: dict[str, MetricSpec] = {
    exact_match.SPEC.name: exact_match.SPEC,
    text_similarity.SPEC.name: text_similarity.SPEC,
    json_document.SPEC.name: json_document.SPEC,
}
```

- [ ] **Step 6: Run the full metric suite**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_metrics.py -v`
Expected: all pass (Task 2 + Task 3).

- [ ] **Step 7: Commit**

```bash
git add studio/backend/eval/metrics studio/backend/tests/test_eval_metrics.py
git commit -m "$(printf 'feat(studio/eval): text_similarity and json_document metrics\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Eval runner (dependency-injected, GPU-free)

**Files:**
- Create: `studio/backend/eval/runner.py`, `studio/backend/eval/__init__.py`
- Test: `studio/backend/tests/test_eval_runner.py`

- [ ] **Step 1: Write the failing test**

`studio/backend/tests/test_eval_runner.py` (AGPL header, then):

```python
from eval.runner import run_eval, EvalSummary
from eval.metrics.registry import make_scorer


def _echo_generate(reference_by_input):
    # stub: returns the prediction we want for a given input message text
    def generate(messages, system_prompt, **gen):
        user = messages[-1]["content"]
        return reference_by_input.get(user, "")
    return generate


def test_runs_all_examples_and_averages():
    examples = [("inA", "x"), ("inB", "y")]
    generate = _echo_generate({"inA": "x", "inB": "WRONG"})
    collected = []
    summary = run_eval(
        examples=examples,
        generate=generate,
        scorer=make_scorer("exact_match", {}),
        system_prompt="", template=None, gen_params={},
        should_cancel=lambda: False,
        on_result=lambda idx, res, pred, inp, ref: collected.append((idx, res.score)),
    )
    assert isinstance(summary, EvalSummary)
    assert summary.num_scored == 2
    assert abs(summary.avg_score - 0.5) < 1e-9
    assert summary.status == "completed"
    assert collected == [(0, 1.0), (1, 0.0)]


def test_template_renders_input():
    seen = {}
    def generate(messages, system_prompt, **gen):
        seen["content"] = messages[-1]["content"]
        return "x"
    run_eval(
        examples=[("world", "x")], generate=generate,
        scorer=make_scorer("exact_match", {}),
        system_prompt="sys", template="Hello {input}!", gen_params={},
        should_cancel=lambda: False, on_result=lambda *a: None,
    )
    assert seen["content"] == "Hello world!"


def test_cancellation_stops_early():
    calls = {"n": 0}
    def generate(messages, system_prompt, **gen):
        calls["n"] += 1
        return "x"
    summary = run_eval(
        examples=[("a", "x"), ("b", "x"), ("c", "x")],
        generate=generate, scorer=make_scorer("exact_match", {}),
        system_prompt="", template=None, gen_params={},
        should_cancel=lambda: calls["n"] >= 1,  # cancel after first generate
        on_result=lambda *a: None,
    )
    assert summary.status == "cancelled"
    assert summary.num_scored == 1


def test_generation_error_does_not_abort():
    def generate(messages, system_prompt, **gen):
        if messages[-1]["content"] == "boom":
            raise RuntimeError("kaboom")
        return "x"
    scores = []
    summary = run_eval(
        examples=[("ok", "x"), ("boom", "x"), ("ok", "x")],
        generate=generate, scorer=make_scorer("exact_match", {}),
        system_prompt="", template=None, gen_params={},
        should_cancel=lambda: False,
        on_result=lambda idx, res, *a: scores.append((idx, res.score, res.error)),
    )
    assert summary.num_scored == 3
    assert scores[1][1] == 0.0 and scores[1][2] is not None  # errored example -> 0 + error
    assert summary.status == "completed"
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_runner.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement runner.py**

`studio/backend/eval/__init__.py`: AGPL header only.

`studio/backend/eval/runner.py` (AGPL header, then):

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .metrics.base import MetricResult, Scorer


@dataclass
class EvalSummary:
    status: str           # "completed" | "cancelled"
    num_scored: int
    avg_score: float


def run_eval(
    *,
    examples: list[tuple[str, Any]],          # (input_text, reference)
    generate: Callable[..., str],             # (messages, system_prompt, **gen) -> text
    scorer: Scorer,
    system_prompt: str,
    template: str | None,
    gen_params: dict,
    should_cancel: Callable[[], bool],
    on_result: Callable[[int, MetricResult, str, str, Any], None],
) -> EvalSummary:
    """Pure eval loop. Model/dataset are injected so this is unit-testable.

    For each example: render the prompt, generate, score, and report via
    on_result(idx, result, prediction, input_text, reference). A generation or
    scoring error becomes a score-0 result with an error note (never aborts).
    Cancellation is checked before each example and takes effect immediately.
    """
    total = 0.0
    scored = 0
    status = "completed"
    for idx, (input_text, reference) in enumerate(examples):
        if should_cancel():
            status = "cancelled"
            break
        content = template.format(input=input_text) if template else input_text
        messages = [{"role": "user", "content": content}]
        try:
            prediction = generate(messages, system_prompt, **gen_params)
            result = scorer(prediction, reference)
        except Exception as exc:  # generation/scoring failure -> errored example
            prediction = ""
            result = MetricResult(score=0.0, error=f"{type(exc).__name__}: {exc}")
        total += result.score
        scored += 1
        on_result(idx, result, prediction, input_text, reference)
    avg = (total / scored) if scored else 0.0
    return EvalSummary(status=status, num_scored=scored, avg_score=avg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/eval/runner.py studio/backend/eval/__init__.py studio/backend/tests/test_eval_runner.py
git commit -m "$(printf 'feat(studio/eval): dependency-injected eval runner\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: Dataset loader (HF + local → examples)

**Files:**
- Create: `studio/backend/eval/dataset.py`
- Test: `studio/backend/tests/test_eval_dataset.py`

- [ ] **Step 1: Write the failing test (local file path)**

`studio/backend/tests/test_eval_dataset.py` (AGPL header, then):

```python
import json

import pytest

from eval.dataset import load_eval_examples, DatasetRef


def test_load_local_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = [{"q": "1+1?", "a": "2"}, {"q": "2+2?", "a": "4"}, {"q": "x", "a": "y"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    examples = load_eval_examples(ref, input_col="q", reference_col="a", limit=2)
    assert examples == [("1+1?", "2"), ("2+2?", "4")]  # first 2 only


def test_load_local_limit_none_returns_all(tmp_path):
    p = tmp_path / "data.jsonl"
    rows = [{"q": "a", "a": "1"}, {"q": "b", "a": "2"}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    examples = load_eval_examples(ref, input_col="q", reference_col="a", limit=None)
    assert len(examples) == 2


def test_missing_column_raises(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"q": "a", "a": "1"}))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    with pytest.raises(ValueError, match="column"):
        load_eval_examples(ref, input_col="missing", reference_col="a", limit=10)


def test_reference_dict_preserved(tmp_path):
    # JSON-valued reference column stays structured (becomes a JSON string via the loader)
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"q": "extract", "a": {"total": 5}}))
    ref = DatasetRef(is_local=True, path=str(p), name=None, split="train")
    examples = load_eval_examples(ref, input_col="q", reference_col="a", limit=10)
    # dict reference is preserved as a python object (the JSON metric handles either)
    assert examples[0][1] == {"total": 5}
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_dataset.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement dataset.py**

`studio/backend/eval/dataset.py` (AGPL header, then):

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetRef:
    is_local: bool
    path: Optional[str]     # local file path (when is_local)
    name: Optional[str]     # HF repo id (when not is_local)
    split: str = "train"
    subset: Optional[str] = None


def _coerce(value: Any) -> Any:
    # text columns -> str; structured (dict/list) references pass through.
    if isinstance(value, (dict, list)) or value is None:
        return value
    return str(value)


def load_eval_examples(
    ref: DatasetRef, *, input_col: str, reference_col: str, limit: Optional[int],
) -> list[tuple[str, Any]]:
    """Load (input, reference) pairs from an HF repo or a local file.

    Returns the first `limit` rows (all rows when limit is None).
    """
    from datasets import load_dataset

    if ref.is_local:
        path = Path(ref.path)
        suffix = path.suffix.lower()
        fmt = {".jsonl": "json", ".json": "json", ".csv": "csv",
               ".parquet": "parquet"}.get(suffix)
        if fmt is None:
            raise ValueError(f"Unsupported local dataset file type: {suffix!r}")
        ds = load_dataset(fmt, data_files=str(path), split=ref.split)
    else:
        ds = load_dataset(ref.name, ref.subset, split=ref.split)

    cols = set(ds.column_names)
    for col in (input_col, reference_col):
        if col not in cols:
            raise ValueError(
                f"column {col!r} not in dataset (have: {sorted(cols)})"
            )

    n = len(ds) if limit is None else min(limit, len(ds))
    sliced = ds.select(range(n))
    return [
        (str(row[input_col]) if row[input_col] is not None else "",
         _coerce(row[reference_col]))
        for row in sliced
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_dataset.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/eval/dataset.py studio/backend/tests/test_eval_dataset.py
git commit -m "$(printf 'feat(studio/eval): dataset loader for HF + local files\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: Pydantic models

**Files:**
- Create: `studio/backend/models/eval.py`
- Modify: `studio/backend/models/__init__.py`
- Test: covered indirectly by route tests (Task 9); a small import test here.

- [ ] **Step 1: Write the failing test**

`studio/backend/tests/test_eval_models.py` (AGPL header, then):

```python
from models import (
    EvalStartRequest, EvalRunSummary, EvalRunDetail, EvalResultRow,
    EvalProgress, MetricInfo,
)


def test_eval_start_request_defaults():
    req = EvalStartRequest(
        model_identifier="hf/m",
        dataset={"is_local": True, "path": "d.jsonl", "split": "train"},
        input_column="q", reference_column="a",
        metric_name="exact_match",
    )
    assert req.limit == 100
    assert req.metric_config == {}
    assert req.temperature == 0.0
    assert req.max_new_tokens == 256
    assert req.system_prompt == ""


def test_eval_progress_roundtrip():
    p = EvalProgress(run_id="r", status="running", done=1, total=2, avg_score=0.5)
    assert p.model_dump()["done"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_models.py -v`
Expected: FAIL — `cannot import name 'EvalStartRequest'`.

- [ ] **Step 3: Implement models/eval.py**

`studio/backend/models/eval.py` (AGPL header, then):

```python
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalDatasetRef(BaseModel):
    is_local: bool
    path: Optional[str] = None
    name: Optional[str] = None
    split: str = "train"
    subset: Optional[str] = None


class EvalStartRequest(BaseModel):
    model_identifier: str
    dataset: EvalDatasetRef
    input_column: str
    reference_column: str
    metric_name: str
    metric_config: dict = Field(default_factory=dict)
    system_prompt: str = ""
    template: Optional[str] = None
    limit: Optional[int] = 100          # None = all rows
    max_new_tokens: int = 256
    temperature: float = 0.0


class EvalLastResult(BaseModel):
    idx: int
    score: float
    error: Optional[str] = None


class EvalProgress(BaseModel):
    run_id: str
    status: str
    done: int
    total: int
    avg_score: float
    eta_sec: Optional[float] = None
    last_result: Optional[EvalLastResult] = None


class EvalRunSummary(BaseModel):
    id: str
    status: str
    model_identifier: str
    dataset_ref: str
    metric_name: str
    started_at: str
    ended_at: Optional[str] = None
    num_examples: Optional[int] = None
    avg_score: Optional[float] = None
    display_name: Optional[str] = None


class EvalResultRow(BaseModel):
    idx: int
    input_text: Optional[str] = None
    prediction_text: Optional[str] = None
    reference_text: Optional[str] = None
    score: Optional[float] = None
    breakdown: Optional[dict] = None
    error: Optional[str] = None


class EvalRunDetail(BaseModel):
    run: EvalRunSummary
    results: list[EvalResultRow]
    total_results: int


class MetricConfigField(BaseModel):
    name: str
    type: str
    default: Any = None
    label: str


class MetricInfo(BaseModel):
    name: str
    label: str
    reference_kind: str
    config_fields: list[MetricConfigField]
```

- [ ] **Step 4: Register exports**

In `studio/backend/models/__init__.py`, add an import block and extend `__all__`:

```python
from .eval import (
    EvalDatasetRef,
    EvalStartRequest,
    EvalProgress,
    EvalLastResult,
    EvalRunSummary,
    EvalResultRow,
    EvalRunDetail,
    MetricInfo,
    MetricConfigField,
)
```
and append those names to `__all__`.

- [ ] **Step 5: Run to verify it passes**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_models.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/models/eval.py studio/backend/models/__init__.py studio/backend/tests/test_eval_models.py
git commit -m "$(printf 'feat(studio/eval): pydantic request/response models\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: EvalJobManager

**Files:**
- Create: `studio/backend/eval/jobs.py`
- Test: `studio/backend/tests/test_eval_jobs.py`

The manager owns run lifecycle + in-memory progress (for SSE) + DB persistence. It takes an injectable `run_fn` (default wires the real runner+inference+dataset in Task 8) so it's testable with a stub.

- [ ] **Step 1: Write the failing test**

`studio/backend/tests/test_eval_jobs.py` (AGPL header, then):

```python
import time

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # Same isolation as test_eval_db.py: repoint the path + reset cached state;
    # never reload against the real studio.db. Adapt attr names to Task 1 Step 1.
    from storage import studio_db
    monkeypatch.setattr(studio_db, "DB_PATH", tmp_path / "studio.db", raising=True)
    for attr in ("_CONNECTION", "_conn", "_connection"):
        if hasattr(studio_db, attr):
            monkeypatch.setattr(studio_db, attr, None, raising=False)
    for flag in ("_SCHEMA_READY", "_schema_initialized", "_INITIALIZED"):
        if hasattr(studio_db, flag):
            monkeypatch.setattr(studio_db, flag, False, raising=False)
    return studio_db


def _wait(mgr, run_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        st = mgr.get(run_id)["status"]
        if st in ("completed", "cancelled", "error", "interrupted"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_start_runs_and_persists(db):
    from eval.jobs import EvalJobManager
    from models import EvalStartRequest

    # stub run_fn: scores 2 examples, calls on_result, returns summary-like dict
    def fake_run(config, *, on_result, should_cancel):
        from eval.runner import EvalSummary
        on_result(0, 1.0, "p0", "i0", "r0", None, None)
        on_result(1, 0.0, "p1", "i1", "r1", None, "bad")
        return EvalSummary(status="completed", num_scored=2, avg_score=0.5)

    mgr = EvalJobManager(run_fn=fake_run)
    req = EvalStartRequest(
        model_identifier="m", dataset={"is_local": True, "path": "d.jsonl"},
        input_column="q", reference_column="a", metric_name="exact_match",
    )
    run_id = mgr.start(req)
    assert _wait(mgr, run_id) == "completed"
    run = db.get_eval_run(run_id)
    assert run["avg_score"] == 0.5 and run["status"] == "completed"
    assert db.get_eval_results(run_id)["total"] == 2


def test_concurrent_start_rejected(db):
    from eval.jobs import EvalJobManager, EvalBusyError
    from models import EvalStartRequest

    def slow_run(config, *, on_result, should_cancel):
        from eval.runner import EvalSummary
        for _ in range(50):
            if should_cancel():
                break
            time.sleep(0.02)
        return EvalSummary(status="completed", num_scored=0, avg_score=0.0)

    mgr = EvalJobManager(run_fn=slow_run)
    req = EvalStartRequest(model_identifier="m",
                           dataset={"is_local": True, "path": "d.jsonl"},
                           input_column="q", reference_column="a",
                           metric_name="exact_match")
    run_id = mgr.start(req)
    with pytest.raises(EvalBusyError):
        mgr.start(req)
    mgr.cancel(run_id)
    assert _wait(mgr, run_id) in ("cancelled", "completed")


def test_cancel_sets_status(db):
    from eval.jobs import EvalJobManager
    from models import EvalStartRequest

    def slow_run(config, *, on_result, should_cancel):
        from eval.runner import EvalSummary
        n = 0
        while not should_cancel() and n < 100:
            time.sleep(0.02); n += 1
        return EvalSummary(status="cancelled", num_scored=n, avg_score=0.0)

    mgr = EvalJobManager(run_fn=slow_run)
    req = EvalStartRequest(model_identifier="m",
                           dataset={"is_local": True, "path": "d.jsonl"},
                           input_column="q", reference_column="a",
                           metric_name="exact_match")
    run_id = mgr.start(req)
    time.sleep(0.05)
    mgr.cancel(run_id)
    assert _wait(mgr, run_id) == "cancelled"
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_jobs.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement jobs.py**

`studio/backend/eval/jobs.py` (AGPL header, then):

```python
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from storage import studio_db

# run_fn(config, *, on_result, should_cancel) -> EvalSummary
# on_result(idx, score, prediction, input_text, reference, breakdown, error)
RunFn = Callable[..., Any]


class EvalBusyError(RuntimeError):
    """Raised when an eval is already running (single shared model)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalJobManager:
    def __init__(self, run_fn: RunFn):
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._active_run_id: Optional[str] = None
        self._cancel = threading.Event()
        # in-memory progress for SSE: run_id -> dict
        self._progress: dict[str, dict] = {}
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._active_run_id is not None

    def start(self, req) -> str:
        with self._lock:
            if self._active_run_id is not None:
                raise EvalBusyError("an eval is already running")
            run_id = uuid.uuid4().hex
            self._active_run_id = run_id
            self._cancel.clear()
        total = req.limit if req.limit is not None else 0
        dataset_ref = req.dataset.path or req.dataset.name or "?"
        studio_db.create_eval_run(
            id=run_id, model_identifier=req.model_identifier,
            dataset_ref=str(dataset_ref), metric_name=req.metric_name,
            config_json=req.model_dump_json(), started_at=_now(),
            num_examples=req.limit,
        )
        self._progress[run_id] = {
            "run_id": run_id, "status": "running", "done": 0,
            "total": total, "avg_score": 0.0, "last_result": None,
        }
        self._thread = threading.Thread(
            target=self._run, args=(run_id, req), daemon=True
        )
        self._thread.start()
        return run_id

    def _run(self, run_id: str, req) -> None:
        running_total = {"sum": 0.0, "n": 0}

        def on_result(idx, score, prediction, input_text, reference,
                      breakdown, error):
            studio_db.insert_eval_result(
                run_id=run_id, idx=idx, input_text=str(input_text),
                prediction_text=str(prediction),
                reference_text=reference if isinstance(reference, str)
                else json.dumps(reference),
                score=score,
                breakdown_json=json.dumps(breakdown) if breakdown else None,
                error=error,
            )
            running_total["sum"] += score or 0.0
            running_total["n"] += 1
            prog = self._progress[run_id]
            prog["done"] = running_total["n"]
            prog["avg_score"] = running_total["sum"] / running_total["n"]
            prog["last_result"] = {"idx": idx, "score": score, "error": error}

        status = "error"
        avg = 0.0
        err_msg = None
        try:
            summary = self._run_fn(
                req, on_result=on_result,
                should_cancel=lambda: self._cancel.is_set(),
            )
            status = summary.status
            avg = summary.avg_score
        except Exception as exc:  # model load / dataset / fatal error
            err_msg = f"{type(exc).__name__}: {exc}"
        finally:
            studio_db.finish_eval_run(
                id=run_id, status=status, ended_at=_now(),
                avg_score=avg, error_message=err_msg,
            )
            prog = self._progress.get(run_id, {})
            prog["status"] = status
            prog["avg_score"] = avg
            with self._lock:
                self._active_run_id = None

    def cancel(self, run_id: str) -> bool:
        if self._active_run_id != run_id:
            return False
        self._cancel.set()
        return True

    def get(self, run_id: str) -> Optional[dict]:
        """Live progress if active, else the persisted run summary."""
        if run_id in self._progress and self._active_run_id == run_id:
            return dict(self._progress[run_id])
        run = studio_db.get_eval_run(run_id)
        if run is None:
            return self._progress.get(run_id)
        # merge last known progress (post-finish)
        prog = self._progress.get(run_id, {})
        return {
            "run_id": run_id, "status": run["status"],
            "done": prog.get("done", 0), "total": prog.get("total", 0),
            "avg_score": run["avg_score"] if run["avg_score"] is not None
            else prog.get("avg_score", 0.0),
            "last_result": prog.get("last_result"),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_jobs.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/eval/jobs.py studio/backend/tests/test_eval_jobs.py
git commit -m "$(printf 'feat(studio/eval): EvalJobManager (threaded lifecycle + persistence)\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: Inference adapter + production run_fn wiring

**Files:**
- Create: `studio/backend/eval/inference_adapter.py`
- Modify: `studio/backend/eval/jobs.py` (add a default production `run_fn`)
- Test: `studio/backend/tests/test_eval_inference_adapter.py`

This is the live seam. `build_eval_run_fn()` returns a `run_fn` that: loads the model via the inference backend, loads dataset examples (Task 5), builds the scorer (Task 3), binds `generate`, and calls `run_eval` (Task 4). Generation goes through `get_inference_backend().generate_chat_response(...)` which yields cumulative text chunks — collect by keeping the last chunk.

**Known scope limit (GGUF):** this adapter targets the standard transformers/unsloth backend (`get_inference_backend()`). GGUF/llama-server models load + generate through a *different* backend (`get_llama_cpp_backend()`, see Explore findings / `routes/inference.py` which branches on `config.is_gguf`). Phase-1 eval supports **non-GGUF** models; add a `config.is_gguf` branch in `ensure_model_loaded`/`make_generate` as a fast-follow if GGUF eval is needed. The `ensure_model_loaded` guard below raises `ValueError` on a GGUF config, which the job records as the run's `error_message` (shown in the UI) — failing fast with a clear message rather than mis-generating.

- [ ] **Step 1: Write the failing test (adapter is injectable; test with a fake backend)**

`studio/backend/tests/test_eval_inference_adapter.py` (AGPL header, then):

```python
from eval.inference_adapter import make_generate, collect_generation


class _FakeBackend:
    def __init__(self):
        self.active_model_name = None
        self.loaded = []
    def generate_chat_response(self, messages, system_prompt, **kw):
        # yields cumulative chunks like the real backend
        text = "hello world"
        acc = ""
        for tok in text.split():
            acc = (acc + " " + tok).strip()
            yield acc


def test_collect_generation_takes_final_cumulative_chunk():
    gen = (c for c in ["a", "a b", "a b c"])
    assert collect_generation(gen) == "a b c"


def test_make_generate_returns_text():
    backend = _FakeBackend()
    generate = make_generate(backend, max_new_tokens=16, temperature=0.0)
    out = generate([{"role": "user", "content": "hi"}], "")
    assert out == "hello world"
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_inference_adapter.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement inference_adapter.py**

`studio/backend/eval/inference_adapter.py` (AGPL header, then):

```python
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


def collect_generation(chunks: Iterable[str]) -> str:
    """generate_chat_response yields CUMULATIVE text; the last chunk is the full output."""
    final = ""
    for chunk in chunks:
        final = chunk
    if isinstance(final, str) and final.startswith("Error: "):
        raise RuntimeError(final)
    return final


def make_generate(backend, *, max_new_tokens: int, temperature: float) -> Callable[..., str]:
    def generate(messages: list, system_prompt: str, **_: Any) -> str:
        chunks = backend.generate_chat_response(
            messages=messages, system_prompt=system_prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        return collect_generation(chunks)
    return generate


def ensure_model_loaded(backend, model_identifier: str, *, hf_token: Optional[str] = None) -> None:
    """Load the model into the shared backend if it isn't already active.

    VERIFY-LIVE: confirm against routes/inference.py:706-710 — the load path is
    ModelConfig.from_identifier(...) then backend.load_model(config, ...). Match
    the param names used there (max_seq_length, load_in_4bit, etc.).
    """
    if getattr(backend, "active_model_name", None) == model_identifier:
        return
    from utils.models import ModelConfig
    config = ModelConfig.from_identifier(model_id=model_identifier, hf_token=hf_token,
                                         gguf_variant=None)
    if getattr(config, "is_gguf", False):
        raise ValueError("GGUF eval not yet supported; choose a transformers/unsloth model.")
    ok = backend.load_model(config, max_seq_length=2048, dtype=None,
                            load_in_4bit=True, hf_token=hf_token,
                            trust_remote_code=False, gpu_ids=None)
    if not ok:
        raise RuntimeError(f"failed to load model {model_identifier!r}")
```

- [ ] **Step 4: Add the production `run_fn` to jobs.py**

Append to `studio/backend/eval/jobs.py`:

```python
def build_eval_run_fn() -> RunFn:
    """Production run_fn: loads model + dataset, scores via the real inference path."""
    from core.inference import get_inference_backend
    from .dataset import DatasetRef, load_eval_examples
    from .inference_adapter import ensure_model_loaded, make_generate
    from .metrics.registry import make_scorer
    from .runner import run_eval

    def run_fn(req, *, on_result, should_cancel):
        backend = get_inference_backend()
        ensure_model_loaded(backend, req.model_identifier)
        ref = DatasetRef(
            is_local=req.dataset.is_local, path=req.dataset.path,
            name=req.dataset.name, split=req.dataset.split, subset=req.dataset.subset,
        )
        examples = load_eval_examples(
            ref, input_col=req.input_column,
            reference_col=req.reference_column, limit=req.limit,
        )
        generate = make_generate(backend, max_new_tokens=req.max_new_tokens,
                                 temperature=req.temperature)
        scorer = make_scorer(req.metric_name, req.metric_config)

        def _on_result(idx, result, prediction, input_text, reference):
            on_result(idx, result.score, prediction, input_text, reference,
                      result.breakdown, result.error)

        return run_eval(
            examples=examples, generate=generate, scorer=scorer,
            system_prompt=req.system_prompt, template=req.template,
            gen_params={}, should_cancel=should_cancel, on_result=_on_result,
        )

    return run_fn
```

- [ ] **Step 5: Run to verify the adapter tests pass**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_inference_adapter.py -v`
Expected: 2 passed.

- [ ] **Step 6: VERIFY-LIVE the load signature**

Open `studio/backend/routes/inference.py` around the load handler (≈ lines 561–710) and confirm `ModelConfig.from_identifier(...)` + `backend.load_model(...)` argument names match `ensure_model_loaded`. Adjust the kwargs in `inference_adapter.py` if the real signature differs (e.g. different `from_identifier` kwarg names). No test change needed (the unit test uses a fake backend); this guards the production path. Document any change in the commit message.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/eval/inference_adapter.py studio/backend/eval/jobs.py studio/backend/tests/test_eval_inference_adapter.py
git commit -m "$(printf 'feat(studio/eval): inference adapter + production run_fn wiring\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 9: Routes + registration; remove Score route

**Files:**
- Create: `studio/backend/routes/eval.py`
- Modify: `studio/backend/routes/__init__.py`, `studio/backend/main.py`
- Remove: `studio/backend/routes/scoring.py`, `studio/backend/models/scoring.py`, their registration in `routes/__init__.py`, `models/__init__.py`, `main.py`, and the scoring route test `studio/backend/tests/test_json_score_route.py`
- Test: `studio/backend/tests/test_eval_routes.py`

- [ ] **Step 1: Write the failing test**

`studio/backend/tests/test_eval_routes.py` (AGPL header, then):

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # Same isolation as test_eval_db.py: repoint the path + reset cached state;
    # never reload against the real studio.db. Adapt attr names to Task 1 Step 1.
    from storage import studio_db
    monkeypatch.setattr(studio_db, "DB_PATH", tmp_path / "studio.db", raising=True)
    for attr in ("_CONNECTION", "_conn", "_connection"):
        if hasattr(studio_db, attr):
            monkeypatch.setattr(studio_db, attr, None, raising=False)
    for flag in ("_SCHEMA_READY", "_schema_initialized", "_INITIALIZED"):
        if hasattr(studio_db, flag):
            monkeypatch.setattr(studio_db, flag, False, raising=False)
    return studio_db


def _client(monkeypatch):
    # Inject a stub job manager so routes don't touch real inference.
    import eval.jobs as jobs
    from eval.runner import EvalSummary
    from auth.authentication import get_current_subject
    import routes.eval as eval_routes

    def fake_run(req, *, on_result, should_cancel):
        on_result(0, 1.0, "p", "i", "r", None, None)
        return EvalSummary(status="completed", num_scored=1, avg_score=1.0)

    mgr = jobs.EvalJobManager(run_fn=fake_run)
    monkeypatch.setattr(eval_routes, "get_eval_manager", lambda: mgr)

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(eval_routes.router, prefix="/api/eval")
    return TestClient(app), mgr


def test_list_metrics(db, monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/api/eval/metrics")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["metrics"]}
    assert {"exact_match", "text_similarity", "json_document"} <= names


def test_start_then_run_appears_in_history(db, monkeypatch):
    client, mgr = _client(monkeypatch)
    body = {
        "model_identifier": "m",
        "dataset": {"is_local": True, "path": "d.jsonl", "split": "train"},
        "input_column": "q", "reference_column": "a", "metric_name": "exact_match",
        "limit": 1,
    }
    r = client.post("/api/eval/start", json=body)
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    import time
    end = time.time() + 5
    while time.time() < end and db.get_eval_run(run_id)["status"] == "running":
        time.sleep(0.02)

    runs = client.get("/api/eval/runs").json()
    assert any(run["id"] == run_id for run in runs["runs"])
    detail = client.get(f"/api/eval/runs/{run_id}").json()
    assert detail["run"]["avg_score"] == 1.0
    assert detail["total_results"] == 1


def test_unknown_run_404(db, monkeypatch):
    client, _ = _client(monkeypatch)
    assert client.get("/api/eval/runs/does-not-exist").status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_routes.py -v`
Expected: FAIL — `No module named 'routes.eval'`.

- [ ] **Step 3: Implement routes/eval.py**

`studio/backend/routes/eval.py` (AGPL header, then):

```python
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loggers import get_logger

from auth.authentication import get_current_subject
from eval.jobs import EvalBusyError, EvalJobManager, build_eval_run_fn
from eval.metrics.registry import list_metrics
from models import (EvalProgress, EvalResultRow, EvalRunDetail, EvalRunSummary,
                    EvalStartRequest, MetricInfo)
from storage import studio_db

logger = get_logger(__name__)
router = APIRouter()

_MANAGER: EvalJobManager | None = None


def get_eval_manager() -> EvalJobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = EvalJobManager(run_fn=build_eval_run_fn())
    return _MANAGER


@router.get("/metrics")
async def get_metrics(current_subject: str = Depends(get_current_subject)):
    return {"metrics": [MetricInfo(**m).model_dump() for m in list_metrics()]}


@router.post("/start")
async def start_eval(payload: EvalStartRequest,
                     current_subject: str = Depends(get_current_subject)):
    mgr = get_eval_manager()
    try:
        run_id = mgr.start(payload)
    except EvalBusyError:
        raise HTTPException(status_code=409, detail="An eval is already running.")
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"run_id": run_id}


@router.post("/cancel/{run_id}")
async def cancel_eval(run_id: str, current_subject: str = Depends(get_current_subject)):
    ok = get_eval_manager().cancel(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="No active eval with that id.")
    return {"cancelled": True}


@router.get("/runs")
async def list_runs(limit: int = 50, offset: int = 0,
                    current_subject: str = Depends(get_current_subject)):
    data = studio_db.list_eval_runs(limit=limit, offset=offset)
    return {"runs": [EvalRunSummary(**r).model_dump() for r in data["runs"]],
            "total": data["total"]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, limit: int = 100, offset: int = 0,
                  current_subject: str = Depends(get_current_subject)):
    run = studio_db.get_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found.")
    res = studio_db.get_eval_results(run_id, limit=limit, offset=offset)
    rows = []
    for r in res["results"]:
        rows.append(EvalResultRow(
            idx=r["idx"], input_text=r["input_text"],
            prediction_text=r["prediction_text"], reference_text=r["reference_text"],
            score=r["score"],
            breakdown=json.loads(r["breakdown_json"]) if r["breakdown_json"] else None,
            error=r["error"],
        ))
    return EvalRunDetail(run=EvalRunSummary(**run), results=rows,
                         total_results=res["total"]).model_dump()


@router.get("/progress/{run_id}")
async def stream_progress(run_id: str, request: Request,
                          current_subject: str = Depends(get_current_subject)):
    mgr = get_eval_manager()

    async def gen():
        yield "retry: 3000\n\n"
        last_done = -1
        while True:
            if await request.is_disconnected():
                break
            prog = mgr.get(run_id)
            if prog is None:
                break
            if prog["done"] != last_done or prog["status"] != "running":
                payload = EvalProgress(**{
                    "run_id": run_id, "status": prog["status"],
                    "done": prog.get("done", 0), "total": prog.get("total", 0),
                    "avg_score": prog.get("avg_score", 0.0),
                    "last_result": prog.get("last_result"),
                }).model_dump_json()
                event = "complete" if prog["status"] != "running" else "progress"
                yield f"id: {prog.get('done', 0)}\nevent: {event}\ndata: {payload}\n\n"
                last_done = prog["done"]
                if prog["status"] != "running":
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Register the router; remove the scoring route**

In `studio/backend/routes/__init__.py`: add `from routes.eval import router as eval_router` and `"eval_router"` to `__all__`; **remove** the `scoring_router` import + `__all__` entry.

In `studio/backend/main.py`: add `eval_router` to the `from routes import (...)` block and `app.include_router(eval_router, prefix = "/api/eval", tags = ["eval"])`; **remove** the `scoring_router` import + its `include_router` line.

In `studio/backend/models/__init__.py`: **remove** the `ScoreRequest`/`ScoreResponse` import + `__all__` entries.

Delete files:
```bash
git rm studio/backend/routes/scoring.py studio/backend/models/scoring.py studio/backend/tests/test_json_score_route.py
```

- [ ] **Step 5: Run to verify routes pass + nothing references scoring**

Run:
```bash
~/.unsloth/studio/unsloth_studio/bin/python -m pytest studio/backend/tests/test_eval_routes.py -v
grep -rn "scoring_router\|routes.scoring\|models.scoring\|ScoreRequest" studio/backend --include=*.py
```
Expected: 4 passed; grep prints nothing.

- [ ] **Step 6: Verify the whole app imports with the new router**

Run:
```bash
cd studio/backend && ~/.unsloth/studio/unsloth_studio/bin/python -c "import sys; sys.path.insert(0,'.'); import main; print('app OK', any(r.path.startswith('/api/eval') for r in main.app.routes))"
```
Expected: `app OK True` (after the unsloth banner). If import fails, fix the offending reference.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/routes/eval.py studio/backend/routes/__init__.py studio/backend/main.py studio/backend/models/__init__.py studio/backend/tests/test_eval_routes.py
git add -u studio/backend/routes/scoring.py studio/backend/models/scoring.py studio/backend/tests/test_json_score_route.py
git commit -m "$(printf 'feat(studio/eval): /api/eval routes; remove superseded scoring route\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 10: Startup reconciliation + full backend suite

**Files:**
- Modify: (none new — confirm `cleanup_orphaned_runs()` from Task 1 runs at startup)
- Test: full backend eval suite

- [ ] **Step 1: Confirm reconciliation runs at boot**

In `studio/backend/main.py` find where `cleanup_orphaned_runs()` is called at startup (≈ line 243). Since Task 1 extended that function to also reconcile `eval_runs`, no new wiring is needed. Confirm by reading the call site.

- [ ] **Step 2: Run the entire eval backend suite**

Run:
```bash
~/.unsloth/studio/unsloth_studio/bin/python -m pytest \
  studio/backend/tests/test_eval_db.py \
  studio/backend/tests/test_eval_metrics.py \
  studio/backend/tests/test_eval_runner.py \
  studio/backend/tests/test_eval_dataset.py \
  studio/backend/tests/test_eval_models.py \
  studio/backend/tests/test_eval_jobs.py \
  studio/backend/tests/test_eval_inference_adapter.py \
  studio/backend/tests/test_eval_routes.py \
  studio/backend/tests/test_json_score_comparators.py \
  studio/backend/tests/test_json_score_schema.py \
  studio/backend/tests/test_json_score_core.py \
  studio/backend/tests/test_json_score_api.py -v
```
Expected: all pass (the json_score library suite still green; the removed `test_json_score_route.py` is gone).

- [ ] **Step 3: Commit (if any reconciliation tweak was needed)**

```bash
git add -A studio/backend
git commit -m "$(printf 'chore(studio/eval): confirm startup reconciliation of stale eval runs\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')" || echo "nothing to commit"
```

---

## Done (backend)

The Eval backend is complete: pluggable metrics, a DI runner, persistence, a threaded job manager, the inference/dataset adapters, and `/api/eval` routes (start/progress-SSE/cancel/runs/runs-detail/metrics). The manual Score route is removed.

**Live end-to-end check (manual, needs a small model + the running Studio backend):** start the backend (`run.py`), `POST /api/eval/start` with a tiny local JSONL dataset + `exact_match`, watch `GET /api/eval/progress/{run_id}` stream, and confirm `GET /api/eval/runs/{run_id}` shows results. (Full UI verification is the frontend plan.)

**Next:** the **Eval frontend** plan — `/eval` page (model/dataset/metric config), live progress view, results + breakdown, sidebar Recents, and removing the `document-score` panel.

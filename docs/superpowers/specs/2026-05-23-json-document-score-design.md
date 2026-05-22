# Type-Aware JSON Document Score — Design

**Date:** 2026-05-23
**Status:** Approved (brainstorming) — pending implementation plan

## Problem

When evaluating an LLM that generates structured JSON (e.g. invoice/document
extraction), a flat text-similarity score is the wrong tool: different fields
demand different comparison rules. A monetary `total` should be judged by
numeric closeness, a `currency` code by exact match, a free-text vendor name by
edit-distance similarity, and a date by format-agnostic equality. We want a
single **document-level score** that scores each field with the rule that field
deserves, plus a per-field breakdown for reporting.

## Goal & Non-Goals

**Goal:** An offline **evaluation metric** that, given a ground-truth JSON, a
predicted JSON, and an explicit per-field schema, returns:
- a scalar document score in `[0, 1]`, and
- an optional per-field breakdown (which field/line-item dragged the score down).

**Non-goals (YAGNI for this iteration):**
- Using the metric as an RL/GRPO reward (the scalar output is reward-compatible,
  but no trainer integration is in scope now).
- Per-field weighting — aggregation is **equal-weight** (ANLS\* default).
- Inferring comparators from values — the schema is **explicit**.

## Background: ANLS\* and the gap it leaves

[`anls_star`](https://github.com/deepopinion/anls_star_metric) (ANLS\*, "A
Universal Document Processing Metric for Generative LLMs") already solves the
hard structural parts of document scoring:
- recursive traversal of **nested dicts**,
- **Hungarian (optimal bipartite) matching** of list items so order doesn't matter,
- penalties for **missing** and **hallucinated** keys/items,
- `None` and tuple-of-options handling,
- a hierarchical `ScoreNode` breakdown (`return_key_scores=True`).

**The gap:** every leaf is scored with normalized **Levenshtein**, including
numbers. `"1234.56"` vs `"1230.00"` gets partial credit from character edits,
which is meaningless for money. The package exposes **no extension point** for
custom leaf comparators.

## Approach (decided)

**Reimplement the ANLS\* core with pluggable, schema-driven leaf comparators.**

Own a small, well-specified implementation of the ANLS\* algorithm (dict
matching with missing/hallucinated penalties, list Hungarian matching,
`None`/type handling, recursive equal-weight aggregation), but drive leaf
comparison from an explicit schema so money/currency/date are scored correctly.

Rejected alternatives:
- **Wrap/monkey-patch `anls_star`** — no public hook; would depend on private
  internals, and Levenshtein is baked in at the leaf so money still can't be
  scored correctly.
- **Fork `anls_star`** — maintenance burden, and its schema-*less* design
  (structure inferred from ground truth) fights our explicit-schema design.

We keep `anls_star` as a **test-only** dependency to cross-check our
reimplementation on pure-string documents.

## Module layout

```
unsloth/eval/json_score/
├── __init__.py        # exports json_anls_score, score_from_text, Schema helpers
├── comparators.py     # money, numeric, categorical, string(ANLS), date + registry
├── schema.py          # parse/normalize the user schema into internal nodes
├── core.py            # recursive traversal, list matching, aggregation, ScoreNode
└── api.py             # json_anls_score(), score_from_text()
```

Tests live alongside under `tests/` (see Testing).

## Schema format

A plain nested dict that mirrors the document shape and annotates each leaf with
a comparator. Objects are dicts; an **array is a one-element list** holding the
item-schema. A leaf is either a shorthand comparator name (string) or a dict
`{"type": <name>, **params}`.

```python
schema = {
    "invoice_no": "categorical",
    "total":      {"type": "money", "rel_tol": 0.0},
    "currency":   "categorical",
    "issue_date": {"type": "date", "granularity": "day"},
    "vendor":     {"name": "string", "vat_id": "categorical"},   # nested object
    "line_items": [                                              # array of objects
        {"description": "string", "qty": "numeric", "price": {"type": "money"}}
    ],
}
```

Fields present in the data but **absent from the schema** fall back to a
configurable `default_comparator` (default `string`, matching ANLS\* behavior),
so the schema only needs to call out fields needing special treatment.

**Acceptable alternatives.** In the **ground-truth data** (not the schema), a
`tuple` marks a set of acceptable values for that node — e.g.
`"name": ("Acme Inc", "Acme Incorporated")`. The schema node is unchanged; the
tuple wraps alternatives that each match that node. A `list` remains an actual
array (Hungarian-matched). JSON has no tuple type, so alternatives only appear
when ground truth is constructed in Python — exactly when they're wanted.

`schema.py` normalizes this user-facing dict into internal node objects
(`LeafNode(comparator, params)`, `ObjectNode(fields)`, `ArrayNode(item)`).

## Comparators

Each comparator returns a score in `[0, 1]` and owns its own type coercion.
Uncoercible input → `0.0`. Registered by name in a registry.

| name | rule | params |
|---|---|---|
| `money` | `1 − |a−b| / max(|a|,|b|,ε)`, clamped to `[0,1]`; within `rel_tol`/`abs_tol` → `1.0` | `rel_tol`, `abs_tol` |
| `numeric` | same formula as `money` (semantic alias) | `rel_tol`, `abs_tol` |
| `categorical` | exact equal → `1.0` else `0.0` | `case_insensitive`, `strip` |
| `string` | ANLS normalized Levenshtein; `0` below threshold τ | `threshold` (default `0.5`) |
| `date` | parse both (format-agnostic), equal/within tolerance → `1.0` | `granularity` (`day`/`month`/`year`), `day_tol` |

## Core algorithm

A single recursive function over `(gt, pred, schema_node)` returning
`ScoreResult(score: float, n_leaves: int, node: ScoreNode)`. Equal weighting
falls out: a subtree's score is the mean of its leaf scores, and `n_leaves` lets
parents average correctly across differently-sized children.

Dispatch by schema node kind:

**Leaf** (schema names a comparator):
- Both `None` → `1.0`; exactly one `None` → `0.0` (missing/hallucinated penalty).
- Otherwise call the comparator; uncoercible value → `0.0`.
- Contributes `1` leaf.

**Object** (`ObjectNode`):
- Walk the **union of keys** in `gt` and `pred`.
- Key in both → recurse.
- Key in `gt`, missing from `pred` → that subtree scores `0` (counts its expected leaves).
- Key in `pred`, not in `gt` (**hallucinated**) → penalized as one extra `0`-scoring leaf.
- Subtree score = leaf-count-weighted mean of children:
  `Σ(child.score · child.n_leaves) / Σ(child.n_leaves)`, with `n_leaves` =
  `Σ child.n_leaves` (so deeper, bushier branches count proportionally).

**Array** (`ArrayNode`):
- Build a cost matrix of **pairwise document scores** between every `gt` item and
  every `pred` item, each computed recursively via the item-schema (matching is
  field-aware, not Levenshtein-on-strings).
- Solve optimal assignment with `scipy.optimize.linear_sum_assignment` (maximize
  total) — order-independent.
- Array score = `sum(matched pair scores) / max(len(gt), len(pred))` → unmatched
  items on **either** side (missing and extra) are penalized.

**Options** (ground-truth value is a non-empty `tuple` of alternatives — checked
before kind dispatch, since it's data-driven): score the prediction against each
option under the same schema node and take the **max**;
`score = max(_score(opt, pred, node) for opt in gt)`, with `n_leaves` taken from
the selected (max-scoring) option. The `ScoreNode` records which alternative
matched. An empty tuple is invalid and scores `0`.

**Type mismatch** (schema expects object/array but data is scalar, or vice-versa)
→ subtree scores `0`, with a note recorded on the `ScoreNode`.

## Data flow

```
raw model text ──(score_from_text, optional)──▶ parsed JSON ─┐
ground-truth JSON ───────────────────────────────────────────┤
schema ──(schema.py normalize)──▶ internal nodes ─────────────┘
                                   │
                                   ▼
                        core._score(gt, pred, node)
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                                ▼
          float document score          ScoreNode tree (per-field breakdown)
```

## Public API

```python
def json_anls_score(
    ground_truth, prediction, schema=None, *,
    default_comparator="string",
    return_key_scores=False,
) -> float | tuple[float, "ScoreNode"]:
    ...

def score_from_text(
    ground_truth, raw_text, schema=None, **kwargs
) -> float | tuple[float, "ScoreNode"]:
    """Extract JSON from raw model output, then score. Unparseable → 0.0."""
```

- `schema=None` → every leaf uses `default_comparator` (pure ANLS\*-style
  behavior; useful as a baseline and for the reference cross-check).
- `return_key_scores=True` → returns `(score, ScoreNode)`. `ScoreNode` has
  `.score`, `.children` (dict for objects, list for arrays), and `.note` (set on
  mismatches).

## Error handling & edge cases

| Situation | Behavior |
|---|---|
| Unparseable prediction text (`score_from_text`) | document score `0.0` (tolerant extraction first: strip code fences / trailing prose) |
| Leaf value can't be coerced for its comparator | `0.0` for that leaf |
| Schema/data kind mismatch | subtree `0.0` + note |
| `money` with both values `0` | `1.0` |
| `money` with `gt=0, pred≠0` | uses ε → `→ 0.0` |
| Empty object/array on both sides | `1.0` (vacuously correct) |
| `gt` empty, `pred` non-empty | penalized as hallucinations |
| Field in schema but absent from both gt & pred | not counted (no leaf) |
| Ground-truth value is a `tuple` of alternatives | best alternative wins (max) |
| Empty alternatives tuple | invalid → `0.0` |

## Testing strategy

Pytest, alongside the module.

1. **Per-comparator unit tests** — money tolerance, sign, and zero edge cases;
   categorical case-insensitivity/strip; string ANLS at/around threshold τ; date
   across formats (`2024-01-05` == `05/01/2024`) and day-tolerance.
2. **Core traversal tests** — nested objects; missing key; hallucinated key; list
   **order-invariance** (shuffled `line_items` → same score); list length mismatch
   penalty; `None` handling; type mismatch; tuple-of-alternatives (max over
   options, including options at object/array level).
3. **Reference cross-check** — with an all-`string` schema, our score must match
   `anls_star.anls_score` on a battery of documents (validates our reimplemented
   matching/aggregation against the published metric). `anls_star` is a
   dev/test-only dependency.
4. **End-to-end** — a realistic invoice (totals, currency, dates, line items)
   with a hand-verified expected score and per-field breakdown.

## Dependencies

- `rapidfuzz` — fast, permissively-licensed Levenshtein for the `string` comparator.
- `scipy` — `optimize.linear_sum_assignment` for list matching (likely already a
  transitive dependency).
- `python-dateutil` — format-agnostic date parsing.
- `anls_star` — **test-only**, for the reference cross-check.
```

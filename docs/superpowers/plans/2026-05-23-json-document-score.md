# Type-Aware JSON Document Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a field-aware document scorer that evaluates LLM-generated JSON against ground truth, scoring each field with a schema-chosen comparator (money/numeric, categorical, string/ANLS, date) and aggregating with ANLS\*-style traversal into one document score plus a per-field breakdown.

**Architecture:** A self-contained package `unsloth/eval/json_score/` reimplements the ANLS\* core (nested-dict traversal, Hungarian list matching, missing/hallucinated penalties, `None`/tuple-of-alternatives handling, equal-weight aggregation) but dispatches each leaf to a pluggable, schema-selected comparator. Aggregation is **global-leaf-equal**: every leaf comparison counts the same in the document score. The scorer submodules are torch-free; importing them runs `unsloth/__init__.py`, so the full unsloth install is required at import time. Comparator deps ship as a new `unsloth[eval]` optional extra.

**Tech Stack:** Python, `rapidfuzz` (Levenshtein), `scipy` (`linear_sum_assignment`), `numpy`, `python-dateutil`; `pytest`; `anls_star` (test-only cross-check).

**Spec:** `docs/superpowers/specs/2026-05-23-json-document-score-design.md`

**Conventions for every new `.py` file:** start with the repo's Apache-2.0 header (copy the 13-line header block from the top of `unsloth/__init__.py`). It is omitted from the code blocks below for brevity — add it as the first lines of each created file.

**Running tests:** `pyproject.toml` sets `testpaths = ["tests/security"]`, so bare `pytest` will **not** discover these tests. Always pass the explicit path, e.g. `python -m pytest tests/test_json_score_comparators.py -v`. `pythonpath = ["."]` is already configured, and `tests/conftest.py` provides the GPU-free harness so `import unsloth` works without an accelerator.

**Divergence from `anls_star` (intentional):** this metric is not a drop-in clone. It (a) penalizes hallucinated dict keys (one zero-leaf each) where `anls_star` ignores extra dict keys, and (b) weights every leaf equally across the whole document where `anls_star` may weight per nesting level. The Task 8 cross-check therefore only asserts equality on documents where every scheme agrees (identical/flat).

---

## Task 0: Environment setup

**Files:** none (environment only)

- [ ] **Step 1: Ensure the unsloth dev stack + scoring deps are installed**

The scorer lives under the `unsloth` package, so tests must be able to `import unsloth`. Install the full stack once (if not already), then the scoring deps:

```bash
pip install -e ".[huggingface]"        # full unsloth stack (torch, unsloth_zoo, ...)
pip install rapidfuzz scipy numpy python-dateutil anls_star
```

- [ ] **Step 2: Verify imports work**

Run:

```bash
python -c "import unsloth; import rapidfuzz, scipy, numpy, dateutil, anls_star; print('deps OK')"
```

Expected: prints `deps OK` (after unsloth's import banner). If `import unsloth` fails, the dev stack is not installed — re-run Step 1.

---

## Task 1: Package skeleton + `eval` extra + money/numeric comparator

**Files:**
- Create: `unsloth/eval/__init__.py`
- Create: `unsloth/eval/json_score/__init__.py`
- Create: `unsloth/eval/json_score/comparators.py`
- Modify: `pyproject.toml` (add `eval` optional extra)
- Test: `tests/test_json_score_comparators.py`

- [ ] **Step 1: Create empty namespace inits**

`unsloth/eval/__init__.py` — keep empty (do NOT import json_score here, so `import unsloth.eval` does not require scipy):

```python
# (Apache header only — intentionally empty so importing unsloth.eval stays light)
```

`unsloth/eval/json_score/__init__.py` — leave a placeholder for now (public exports added in Task 7):

```python
# (Apache header) Public API is wired up in api.py and re-exported here in a later task.
```

- [ ] **Step 2: Add the `eval` optional extra to pyproject.toml**

In `pyproject.toml`, inside the `[project.optional-dependencies]` block (after the `triton = [...]` entry near line 66), add:

```toml
eval = [
    "rapidfuzz",
    "scipy",
    "numpy",
    "python-dateutil",
]
```

- [ ] **Step 3: Write the failing test for the money comparator**

`tests/test_json_score_comparators.py`:

```python
from unsloth.eval.json_score.comparators import get_comparator


def test_money_exact_match():
    cmp = get_comparator("money")
    assert cmp(1234.56, 1234.56) == 1.0


def test_money_proportional_difference():
    cmp = get_comparator("money")
    # 1 - |100-90|/max(100,90) = 1 - 10/100 = 0.9
    assert abs(cmp(100, 90) - 0.9) < 1e-9


def test_money_both_zero_is_perfect():
    cmp = get_comparator("money")
    assert cmp(0, 0) == 1.0


def test_money_zero_vs_nonzero_is_zero():
    cmp = get_comparator("money")
    assert cmp(0, 5) == 0.0


def test_money_within_abs_tol():
    cmp = get_comparator("money", abs_tol=0.5)
    assert cmp(100.0, 100.4) == 1.0


def test_money_within_rel_tol():
    cmp = get_comparator("money", rel_tol=0.05)
    assert cmp(100.0, 96.0) == 1.0  # 4% diff <= 5%


def test_money_parses_currency_string():
    cmp = get_comparator("money")
    assert cmp("$1,234.56", 1234.56) == 1.0


def test_money_uncoercible_is_zero():
    cmp = get_comparator("money")
    assert cmp("abc", 10) == 0.0


def test_numeric_is_money_alias():
    assert get_comparator("numeric")(100, 90) == get_comparator("money")(100, 90)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_json_score_comparators.py -v`
Expected: collection/import error — `ModuleNotFoundError` or `cannot import name 'get_comparator'`.

- [ ] **Step 5: Implement comparators.py with the number parser, money comparator, and registry**

`unsloth/eval/json_score/comparators.py`:

```python
from __future__ import annotations

import re
from typing import Any, Callable

Comparator = Callable[[Any, Any], float]  # returns a score in [0, 1]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _to_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None  # bools are not monetary/numeric values
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = _NUM_RE.search(x.replace(",", "").strip())
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def money_comparator(rel_tol: float = 0.0, abs_tol: float = 0.0) -> Comparator:
    def score(gt: Any, pred: Any) -> float:
        a, b = _to_number(gt), _to_number(pred)
        if a is None or b is None:
            return 0.0
        diff = abs(a - b)
        if diff <= abs_tol:
            return 1.0
        denom = max(abs(a), abs(b))
        if denom == 0.0:
            return 1.0  # both ~zero
        if rel_tol and diff / denom <= rel_tol:
            return 1.0
        return max(0.0, 1.0 - diff / denom)

    return score


_REGISTRY: dict[str, Callable[..., Comparator]] = {
    "money": money_comparator,
    "numeric": money_comparator,  # semantic alias
}


def is_comparator(name: str) -> bool:
    return name in _REGISTRY


def get_comparator(name: str, **params: Any) -> Comparator:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown comparator {name!r}. Known: {sorted(_REGISTRY)}"
        )
    try:
        return _REGISTRY[name](**params)
    except TypeError as e:
        raise ValueError(f"Invalid params for comparator {name!r}: {e}") from e
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_json_score_comparators.py -v`
Expected: 9 passed.

- [ ] **Step 7: Commit**

```bash
git add unsloth/eval pyproject.toml tests/test_json_score_comparators.py
git commit -m "feat(json_score): package skeleton, eval extra, money comparator"
```

---

## Task 2: categorical, string, and date comparators

**Files:**
- Modify: `unsloth/eval/json_score/comparators.py`
- Test: `tests/test_json_score_comparators.py`

- [ ] **Step 1: Write the failing tests (append to the test file)**

Append to `tests/test_json_score_comparators.py`:

```python
def test_categorical_exact():
    cmp = get_comparator("categorical")
    assert cmp("USD", "USD") == 1.0
    assert cmp("USD", "EUR") == 0.0


def test_categorical_case_insensitive_and_strip():
    cmp = get_comparator("categorical", case_insensitive=True, strip=True)
    assert cmp("USD", " usd ") == 1.0


def test_categorical_non_string_equality():
    cmp = get_comparator("categorical")
    assert cmp(True, True) == 1.0
    assert cmp(5, 5) == 1.0


def test_string_identical():
    cmp = get_comparator("string")
    assert cmp("Acme Inc", "Acme Inc") == 1.0


def test_string_below_threshold_is_zero():
    cmp = get_comparator("string", threshold=0.5)
    assert cmp("abcdefgh", "zzzzzzzz") == 0.0


def test_string_high_similarity_kept():
    cmp = get_comparator("string", threshold=0.5)
    assert cmp("Acme Incorporated", "Acme Incorporatd") > 0.9


def test_string_handles_none_as_empty():
    cmp = get_comparator("string")
    assert cmp(None, None) == 1.0


def test_date_format_agnostic_equal():
    cmp = get_comparator("date")
    assert cmp("2024-01-15", "Jan 15, 2024") == 1.0
    assert cmp("2024-01-15", "15 January 2024") == 1.0


def test_date_different_days():
    cmp = get_comparator("date")
    assert cmp("2024-01-15", "2024-01-16") == 0.0


def test_date_day_tolerance():
    cmp = get_comparator("date", day_tol=1)
    assert cmp("2024-01-15", "2024-01-16") == 1.0


def test_date_granularity_month():
    cmp = get_comparator("date", granularity="month")
    assert cmp("2024-01-15", "2024-01-28") == 1.0
    assert cmp("2024-01-15", "2024-02-15") == 0.0


def test_date_unparseable_is_zero():
    cmp = get_comparator("date")
    assert cmp("not a date", "2024-01-15") == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_json_score_comparators.py -k "categorical or string or date" -v`
Expected: FAIL — `Unknown comparator 'categorical'` (etc.).

- [ ] **Step 3: Implement the three comparators and register them**

In `unsloth/eval/json_score/comparators.py`, add the date import near the top (after `import re`):

```python
from datetime import date, datetime
```

Add the three comparator factories (after `money_comparator`):

```python
def categorical_comparator(case_insensitive: bool = False, strip: bool = False) -> Comparator:
    def score(gt: Any, pred: Any) -> float:
        a, b = gt, pred
        if isinstance(a, str) and isinstance(b, str):
            if strip:
                a, b = a.strip(), b.strip()
            if case_insensitive:
                a, b = a.lower(), b.lower()
        return 1.0 if a == b else 0.0

    return score


def string_comparator(threshold: float = 0.5) -> Comparator:
    from rapidfuzz.distance import Levenshtein

    def score(gt: Any, pred: Any) -> float:
        a = "" if gt is None else str(gt)
        b = "" if pred is None else str(pred)
        if not a and not b:
            return 1.0
        sim = Levenshtein.normalized_similarity(a, b)  # 1 - dist/max(len), in [0,1]
        return sim if sim >= threshold else 0.0

    return score


def _to_date(x: Any) -> datetime | None:
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if isinstance(x, str):
        try:
            from dateutil import parser as _dateparser

            # fixed default so absent components are deterministic
            return _dateparser.parse(x, default=datetime(2000, 1, 1))
        except (ValueError, OverflowError, TypeError):
            return None
    return None


def date_comparator(granularity: str = "day", day_tol: int = 0) -> Comparator:
    def score(gt: Any, pred: Any) -> float:
        da, db = _to_date(gt), _to_date(pred)
        if da is None or db is None:
            return 0.0
        if granularity == "year":
            return 1.0 if da.year == db.year else 0.0
        if granularity == "month":
            return 1.0 if (da.year, da.month) == (db.year, db.month) else 0.0
        delta = abs((da.date() - db.date()).days)
        return 1.0 if delta <= day_tol else 0.0

    return score
```

Extend the registry:

```python
_REGISTRY: dict[str, Callable[..., Comparator]] = {
    "money": money_comparator,
    "numeric": money_comparator,  # semantic alias
    "categorical": categorical_comparator,
    "string": string_comparator,
    "date": date_comparator,
}
```

- [ ] **Step 4: Run the full comparator suite**

Run: `python -m pytest tests/test_json_score_comparators.py -v`
Expected: all tests pass (Task 1 + Task 2).

- [ ] **Step 5: Commit**

```bash
git add unsloth/eval/json_score/comparators.py tests/test_json_score_comparators.py
git commit -m "feat(json_score): categorical, string (ANLS), and date comparators"
```

---

## Task 3: Schema normalization

**Files:**
- Create: `unsloth/eval/json_score/schema.py`
- Test: `tests/test_json_score_schema.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_json_score_schema.py`:

```python
import pytest

from unsloth.eval.json_score.schema import (
    ArrayNode,
    LeafNode,
    ObjectNode,
    normalize_schema,
)


def test_shorthand_string_is_leaf():
    node = normalize_schema("money")
    assert node == LeafNode("money", {})


def test_leaf_dict_with_type_and_params():
    node = normalize_schema({"type": "money", "rel_tol": 0.01})
    assert node == LeafNode("money", {"rel_tol": 0.01})


def test_object_schema():
    node = normalize_schema({"name": "string", "vat_id": "categorical"})
    assert node == ObjectNode(
        {"name": LeafNode("string", {}), "vat_id": LeafNode("categorical", {})}
    )


def test_array_schema():
    node = normalize_schema(["string"])
    assert node == ArrayNode(LeafNode("string", {}))


def test_nested_schema():
    node = normalize_schema(
        {"vendor": {"name": "string"}, "items": [{"price": "money"}]}
    )
    assert node == ObjectNode(
        {
            "vendor": ObjectNode({"name": LeafNode("string", {})}),
            "items": ArrayNode(ObjectNode({"price": LeafNode("money", {})})),
        }
    )


def test_empty_array_schema_is_error():
    with pytest.raises(ValueError, match="single-element list"):
        normalize_schema([])


def test_multi_element_array_schema_is_error():
    with pytest.raises(ValueError, match="single-element list"):
        normalize_schema(["string", "money"])


def test_unknown_comparator_is_error():
    with pytest.raises(ValueError, match="Unknown comparator"):
        normalize_schema("not_a_comparator")


def test_bad_leaf_param_is_error():
    with pytest.raises(ValueError, match="Invalid params"):
        normalize_schema({"type": "money", "bogus": 1})
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_json_score_schema.py -v`
Expected: FAIL — module/import error.

- [ ] **Step 3: Implement schema.py**

`unsloth/eval/json_score/schema.py` (plain dataclasses — the default `__eq__` compares all fields, including the `params` dict, which is exactly what the tests assert):

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .comparators import get_comparator, is_comparator


@dataclass
class LeafNode:
    comparator: str
    params: dict = field(default_factory=dict)


@dataclass
class ObjectNode:
    fields: dict  # str -> Node


@dataclass
class ArrayNode:
    item: "Node"


Node = Union[LeafNode, ObjectNode, ArrayNode]


def normalize_schema(raw: Any) -> Node:
    """Convert a user-facing schema (str / dict / single-element list) into nodes.

    Rules:
      - str                         -> LeafNode(name)
      - dict with "type" -> known comparator -> LeafNode(type, params)
      - any other dict              -> ObjectNode (values recursively normalized)
      - single-element list         -> ArrayNode (the element is the item-schema)
    A dict whose only structural cue is a field literally named "type" mapping to
    a comparator name will be read as a leaf — a documented limitation.
    """
    if isinstance(raw, str):
        if not is_comparator(raw):
            raise ValueError(
                f"Unknown comparator {raw!r}. Use a registered comparator name."
            )
        return LeafNode(raw, {})

    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                "array schema must be a single-element list [item_schema]"
            )
        return ArrayNode(normalize_schema(raw[0]))

    if isinstance(raw, dict):
        t = raw.get("type")
        if isinstance(t, str) and is_comparator(t):
            params = {k: v for k, v in raw.items() if k != "type"}
            get_comparator(t, **params)  # validate name + params early
            return LeafNode(t, params)
        return ObjectNode({k: normalize_schema(v) for k, v in raw.items()})

    raise TypeError(f"Invalid schema node: {raw!r}")
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_json_score_schema.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add unsloth/eval/json_score/schema.py tests/test_json_score_schema.py
git commit -m "feat(json_score): schema normalization into Leaf/Object/Array nodes"
```

---

## Task 4: Core — ScoreNode, leaf scoring, leaf counting, and the dispatcher

**Files:**
- Create: `unsloth/eval/json_score/core.py`
- Test: `tests/test_json_score_core.py`

This task defines `_score`, the recursive dispatcher, even though its object/array branches call `_score_object`/`_score_array`/`_mismatch` (added in Tasks 5–6). Python resolves those names only when the branch executes, so importing `core.py` and running this task's leaf/options tests works without them.

- [ ] **Step 1: Write the failing tests**

`tests/test_json_score_core.py`:

```python
from unsloth.eval.json_score.core import (
    ScoreNode,
    _leaf_count,
    _score,
    _score_leaf,
)
from unsloth.eval.json_score.schema import normalize_schema


def test_leaf_both_none_is_perfect():
    node = _score_leaf(None, None, "string", {})
    assert node.score == 1.0 and node.n_leaves == 1


def test_leaf_one_none_is_zero():
    node = _score_leaf("x", None, "string", {})
    assert node.score == 0.0 and node.note == "missing or hallucinated"


def test_leaf_uses_comparator():
    node = _score_leaf(100, 90, "money", {})
    assert abs(node.score - 0.9) < 1e-9 and node.n_leaves == 1


def test_leaf_count_scalar():
    assert _leaf_count(5, None) == 1


def test_leaf_count_object():
    assert _leaf_count({"a": 1, "b": {"c": 2, "d": 3}}, None) == 3


def test_leaf_count_array():
    # each array item counts as one slot (matches _score_array), not inner leaves
    assert _leaf_count([{"a": 1}, {"a": 2, "b": 3}], None) == 2


def test_leaf_count_tuple_uses_first_option():
    assert _leaf_count(("x", "y", "z"), None) == 1


def test_dispatch_leaf():
    node = _score(100, 90, normalize_schema("money"), "string")
    assert abs(node.score - 0.9) < 1e-9


def test_options_tuple_takes_best_alternative():
    schema = normalize_schema("string")
    node = _score(
        ("Acme Inc", "Acme Incorporated"), "Acme Incorporated", schema, "string"
    )
    assert node.score == 1.0 and node.matched_option == 1


def test_options_empty_tuple_is_zero():
    node = _score((), "anything", normalize_schema("string"), "string")
    assert node.score == 0.0 and node.note == "empty alternatives tuple"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_json_score_core.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Implement core.py (ScoreNode, _leaf_count, _score_leaf, _score)**

`unsloth/eval/json_score/core.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .comparators import get_comparator
from .schema import ArrayNode, LeafNode, Node, ObjectNode


@dataclass
class ScoreNode:
    """A node in the per-field score breakdown.

    score:          mean score over this subtree's leaves, in [0, 1]
    n_leaves:       number of leaf comparisons this subtree contributes
    children:       dict (objects) | list (arrays) | None (leaves)
    note:           explanation set on penalties / mismatches
    matched_option: index into a ground-truth alternatives tuple, when applicable
    """

    score: float
    n_leaves: int
    children: Any = None  # dict | list | None
    note: str | None = None
    matched_option: int | None = None


def _leaf_count(value: Any, node: Node | None) -> int:
    """Number of leaf comparisons in `value` under schema `node`."""
    if isinstance(value, tuple):
        return _leaf_count(value[0], node) if value else 1
    if isinstance(node, ObjectNode) or (node is None and isinstance(value, dict)):
        if not isinstance(value, dict):
            return 1
        total = 0
        for k, v in value.items():
            child = node.fields.get(k) if isinstance(node, ObjectNode) else None
            total += _leaf_count(v, child)
        return total
    if isinstance(node, ArrayNode) or (node is None and isinstance(value, list)):
        if not isinstance(value, list):
            return 1
        # each array item is one slot, matching _score_array's n_leaves = max(len)
        return len(value)
    return 1


def _score_leaf(gt: Any, pred: Any, comparator: str, params: dict) -> ScoreNode:
    if gt is None and pred is None:
        return ScoreNode(1.0, 1)
    if (gt is None) != (pred is None):
        return ScoreNode(0.0, 1, note="missing or hallucinated")
    cmp = get_comparator(comparator, **params)
    return ScoreNode(float(cmp(gt, pred)), 1)


def _score(gt: Any, pred: Any, node: Node | None, default: str) -> ScoreNode:
    # Options: a ground-truth tuple lists acceptable alternatives -> take the best
    if isinstance(gt, tuple):
        if not gt:
            return ScoreNode(0.0, 1, note="empty alternatives tuple")
        best: ScoreNode | None = None
        best_i = -1
        for i, opt in enumerate(gt):
            cand = _score(opt, pred, node, default)
            if best is None or cand.score > best.score:
                best, best_i = cand, i
        assert best is not None
        best.matched_option = best_i
        return best

    # Object: schema says object, or no schema and the data is a dict
    if isinstance(node, ObjectNode) or (node is None and isinstance(gt, dict)):
        if not isinstance(gt, dict) or not isinstance(pred, dict):
            return _mismatch(gt, pred, node)
        return _score_object(gt, pred, node, default)

    # Array: schema says array, or no schema and the data is a list
    if isinstance(node, ArrayNode) or (node is None and isinstance(gt, list)):
        if not isinstance(gt, list) or not isinstance(pred, list):
            return _mismatch(gt, pred, node)
        return _score_array(gt, pred, node, default)

    # Leaf
    if isinstance(node, LeafNode):
        return _score_leaf(gt, pred, node.comparator, node.params)
    return _score_leaf(gt, pred, default, {})
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_json_score_core.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add unsloth/eval/json_score/core.py tests/test_json_score_core.py
git commit -m "feat(json_score): ScoreNode, leaf scoring, leaf counting, dispatcher"
```

---

## Task 5: Core — object scoring and type mismatch

**Files:**
- Modify: `unsloth/eval/json_score/core.py`
- Test: `tests/test_json_score_core.py`

- [ ] **Step 1: Write the failing tests (append)**

Append to `tests/test_json_score_core.py`:

```python
from unsloth.eval.json_score.core import _mismatch, _score_object


def test_object_all_correct():
    schema = normalize_schema({"total": "money", "currency": "categorical"})
    gt = {"total": 100, "currency": "USD"}
    pred = {"total": 100, "currency": "USD"}
    node = _score_object(gt, pred, schema, "string")
    assert node.score == 1.0 and node.n_leaves == 2


def test_object_one_field_wrong():
    schema = normalize_schema({"total": "money", "currency": "categorical"})
    gt = {"total": 100, "currency": "USD"}
    pred = {"total": 100, "currency": "EUR"}
    node = _score_object(gt, pred, schema, "string")
    assert node.score == 0.5  # one of two leaves correct


def test_object_missing_key_penalized_proportionally():
    schema = normalize_schema(
        {"total": "money", "vendor": {"name": "string", "vat": "categorical"}}
    )
    gt = {"total": 100, "vendor": {"name": "Acme", "vat": "X1"}}
    pred = {"total": 100}  # whole vendor subtree (2 leaves) missing
    node = _score_object(gt, pred, schema, "string")
    # 3 leaves total, 1 correct -> 1/3
    assert abs(node.score - (1 / 3)) < 1e-9 and node.n_leaves == 3


def test_object_hallucinated_key_costs_one_leaf():
    schema = normalize_schema({"total": "money"})
    gt = {"total": 100}
    pred = {"total": 100, "extra": "junk"}
    node = _score_object(gt, pred, schema, "string")
    # 1 correct leaf + 1 hallucinated zero-leaf -> 1/2
    assert node.score == 0.5 and node.n_leaves == 2


def test_object_empty_both_is_perfect():
    node = _score_object({}, {}, normalize_schema({}), "string")
    assert node.score == 1.0 and node.n_leaves == 0


def test_dispatch_infers_object_when_no_schema():
    node = _score({"a": "x"}, {"a": "x"}, None, "string")
    assert node.score == 1.0 and node.n_leaves == 1


def test_options_at_object_level():
    schema = normalize_schema({"name": "string"})
    gt = ({"name": "A"}, {"name": "B"})
    node = _score(gt, {"name": "B"}, schema, "string")
    assert node.score == 1.0 and node.matched_option == 1


def test_mismatch_schema_object_data_scalar():
    schema = normalize_schema({"a": "string", "b": "string"})
    node = _mismatch(7, "x", schema)
    assert node.score == 0.0 and node.note == "type mismatch"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_json_score_core.py -k "object or mismatch" -v`
Expected: FAIL — `cannot import name '_score_object'`.

- [ ] **Step 3: Implement _mismatch and _score_object**

Add to `unsloth/eval/json_score/core.py`:

```python
def _mismatch(gt: Any, pred: Any, node: Node | None) -> ScoreNode:
    n = _leaf_count(gt, node) if gt is not None else 1
    return ScoreNode(0.0, max(n, 1), note="type mismatch")


def _score_object(gt: dict, pred: dict, node: Node | None, default: str) -> ScoreNode:
    children: dict[str, ScoreNode] = {}
    total_sum, total_n = 0.0, 0
    keys = list(dict.fromkeys([*gt.keys(), *pred.keys()]))  # ordered union
    for k in keys:
        child_node = node.fields.get(k) if isinstance(node, ObjectNode) else None
        if k in gt and k in pred:
            cn = _score(gt[k], pred[k], child_node, default)
        elif k in gt:  # present in gt, missing from prediction
            cn = ScoreNode(
                0.0, _leaf_count(gt[k], child_node), note="missing in prediction"
            )
        else:  # present in prediction only -> hallucinated, one zero-leaf
            cn = ScoreNode(0.0, 1, note="hallucinated")
        children[k] = cn
        total_sum += cn.score * cn.n_leaves
        total_n += cn.n_leaves
    if total_n == 0:
        return ScoreNode(1.0, 0, children=children)
    return ScoreNode(total_sum / total_n, total_n, children=children)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_json_score_core.py -v`
Expected: all tests pass (Task 4 + Task 5).

- [ ] **Step 5: Commit**

```bash
git add unsloth/eval/json_score/core.py tests/test_json_score_core.py
git commit -m "feat(json_score): object scoring with missing/hallucinated penalties"
```

---

## Task 6: Core — array scoring with Hungarian matching

**Files:**
- Modify: `unsloth/eval/json_score/core.py`
- Test: `tests/test_json_score_core.py`

- [ ] **Step 1: Write the failing tests (append)**

Append to `tests/test_json_score_core.py`:

```python
from unsloth.eval.json_score.core import _score_array


def _items_schema():
    return normalize_schema([{"desc": "string", "price": "money"}])


def test_array_order_invariant():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}, {"desc": "Banana", "price": 20}]
    pred = [{"desc": "Banana", "price": 20}, {"desc": "Apple", "price": 10}]
    node = _score_array(gt, pred, schema, "string")
    assert node.score == 1.0 and node.n_leaves == 2


def test_array_extra_predicted_item_penalized():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}]
    pred = [{"desc": "Apple", "price": 10}, {"desc": "Ghost", "price": 99}]
    node = _score_array(gt, pred, schema, "string")
    # 1 matched slot scores 1.0, divided by max(1, 2) = 2 -> 0.5
    assert node.score == 0.5 and node.n_leaves == 2


def test_array_missing_item_penalized():
    schema = _items_schema()
    gt = [{"desc": "Apple", "price": 10}, {"desc": "Banana", "price": 20}]
    pred = [{"desc": "Apple", "price": 10}]
    node = _score_array(gt, pred, schema, "string")
    assert node.score == 0.5 and node.n_leaves == 2


def test_array_empty_both_is_perfect():
    node = _score_array([], [], _items_schema(), "string")
    assert node.score == 1.0 and node.n_leaves == 0


def test_dispatch_infers_array_when_no_schema():
    node = _score([1, 2], [2, 1], None, "string")
    assert node.score == 1.0 and node.n_leaves == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_json_score_core.py -k array -v`
Expected: FAIL — `cannot import name '_score_array'`.

- [ ] **Step 3: Implement _score_array**

Add to `unsloth/eval/json_score/core.py`:

```python
def _score_array(gt: list, pred: list, node: Node | None, default: str) -> ScoreNode:
    item_node = node.item if isinstance(node, ArrayNode) else None
    n_g, n_p = len(gt), len(pred)
    slots = max(n_g, n_p)
    if slots == 0:
        return ScoreNode(1.0, 0, children=[])
    if n_g == 0 or n_p == 0:
        # all hallucinated or all missing
        return ScoreNode(0.0, slots, children=[])

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # full ScoreNode for every (gt_i, pred_j) pair; matching maximises mean score
    node_matrix = [
        [_score(g, p, item_node, default) for p in pred] for g in gt
    ]
    score_matrix = np.array([[n.score for n in row] for row in node_matrix])
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    matched = list(zip(row_ind.tolist(), col_ind.tolist()))
    total = sum(score_matrix[i][j] for i, j in matched)
    children = [node_matrix[i][j] for i, j in matched]
    return ScoreNode(float(total) / slots, slots, children=children)
```

- [ ] **Step 4: Run to verify they pass, then run the whole core suite**

Run: `python -m pytest tests/test_json_score_core.py -v`
Expected: all core tests pass (Tasks 4–6).

- [ ] **Step 5: Commit**

```bash
git add unsloth/eval/json_score/core.py tests/test_json_score_core.py
git commit -m "feat(json_score): Hungarian-matched array scoring"
```

---

## Task 7: Public API — json_anls_score and score_from_text

**Files:**
- Create: `unsloth/eval/json_score/api.py`
- Modify: `unsloth/eval/json_score/__init__.py`
- Test: `tests/test_json_score_api.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_json_score_api.py`:

```python
from unsloth.eval.json_score import json_anls_score, score_from_text
from unsloth.eval.json_score.core import ScoreNode


SCHEMA = {
    "total": {"type": "money", "rel_tol": 0.0},
    "currency": "categorical",
    "issue_date": {"type": "date"},
    "line_items": [{"desc": "string", "qty": "numeric", "price": "money"}],
}


def _gt():
    return {
        "total": 130,
        "currency": "USD",
        "issue_date": "2024-01-15",
        "line_items": [
            {"desc": "Apple", "qty": 1, "price": 10},
            {"desc": "Banana", "qty": 2, "price": 60},
        ],
    }


def test_perfect_score():
    assert json_anls_score(_gt(), _gt(), SCHEMA) == 1.0


def test_returns_float_by_default():
    out = json_anls_score(_gt(), _gt(), SCHEMA)
    assert isinstance(out, float)


def test_return_key_scores_gives_breakdown():
    score, node = json_anls_score(_gt(), _gt(), SCHEMA, return_key_scores=True)
    assert score == 1.0
    assert isinstance(node, ScoreNode)
    assert "total" in node.children


def test_schema_none_defaults_to_string():
    # both sides identical strings -> perfect under default string comparator
    gt = {"a": "hello", "b": "world"}
    assert json_anls_score(gt, gt) == 1.0


def test_score_from_text_strips_code_fence():
    raw = '```json\n{"currency": "USD"}\n```'
    assert score_from_text({"currency": "USD"}, raw, {"currency": "categorical"}) == 1.0


def test_score_from_text_unparseable_is_zero():
    assert score_from_text(_gt(), "the model refused to answer", SCHEMA) == 0.0


def test_score_from_text_unparseable_breakdown():
    score, node = score_from_text(
        _gt(), "no json here", SCHEMA, return_key_scores=True
    )
    assert score == 0.0 and node.note == "unparseable prediction"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_json_score_api.py -v`
Expected: FAIL — `cannot import name 'json_anls_score'`.

- [ ] **Step 3: Implement api.py**

`unsloth/eval/json_score/api.py`:

```python
from __future__ import annotations

import json
import re
from typing import Any

from .core import ScoreNode, _leaf_count, _score
from .schema import Node, normalize_schema

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def json_anls_score(
    ground_truth: Any,
    prediction: Any,
    schema: Any = None,
    *,
    default_comparator: str = "string",
    return_key_scores: bool = False,
):
    """Score a predicted JSON document against ground truth.

    Returns a float in [0, 1], or (float, ScoreNode) if return_key_scores=True.
    """
    node = normalize_schema(schema) if schema is not None else None
    result = _score(ground_truth, prediction, node, default_comparator)
    if return_key_scores:
        return result.score, result
    return result.score


def _extract_json(text: Any) -> Any | None:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    s = text.strip()
    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except (ValueError, TypeError):
                continue
    return None


def score_from_text(
    ground_truth: Any,
    raw_text: Any,
    schema: Any = None,
    *,
    default_comparator: str = "string",
    return_key_scores: bool = False,
):
    """Extract JSON from raw model output, then score. Unparseable -> 0.0."""
    pred = _extract_json(raw_text)
    if pred is None:
        node: Node | None = normalize_schema(schema) if schema is not None else None
        n = max(_leaf_count(ground_truth, node), 1)
        zero = ScoreNode(0.0, n, note="unparseable prediction")
        if return_key_scores:
            return 0.0, zero
        return 0.0
    return json_anls_score(
        ground_truth,
        pred,
        schema,
        default_comparator=default_comparator,
        return_key_scores=return_key_scores,
    )
```

- [ ] **Step 4: Wire up the public exports**

Replace the contents of `unsloth/eval/json_score/__init__.py` (keep the Apache header) with:

```python
from .api import json_anls_score, score_from_text
from .core import ScoreNode
from .schema import ArrayNode, LeafNode, ObjectNode, normalize_schema

__all__ = [
    "json_anls_score",
    "score_from_text",
    "ScoreNode",
    "normalize_schema",
    "LeafNode",
    "ObjectNode",
    "ArrayNode",
]
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest tests/test_json_score_api.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add unsloth/eval/json_score/api.py unsloth/eval/json_score/__init__.py tests/test_json_score_api.py
git commit -m "feat(json_score): public API json_anls_score and score_from_text"
```

---

## Task 8: Reference cross-check and end-to-end invoice test

**Files:**
- Test: `tests/test_json_score_reference.py`

- [ ] **Step 1: Write the cross-check + end-to-end tests**

`tests/test_json_score_reference.py`:

```python
import pytest

from unsloth.eval.json_score import json_anls_score

anls_star = pytest.importorskip("anls_star")


# Only documents where global-leaf-equal aggregation and anls_star's scheme
# agree: identical docs, a flat single-level dict with one differing value, and
# reordered lists. Hallucinated keys and non-perfect nested objects are
# intentionally NOT cross-checked (this metric diverges there by design).
REFERENCE_CASES = [
    ({"a": "hello", "b": "world"}, {"a": "hello", "b": "world"}),       # 1.0
    ({"a": "hello", "b": "world"}, {"a": "hello", "b": "earth"}),       # flat, b differs
    (
        {"items": ["alpha", "beta", "gamma"]},
        {"items": ["gamma", "alpha", "beta"]},                          # reorder -> 1.0
    ),
    (
        {"v": {"name": "Acme", "id": "X1"}, "n": "Bob"},
        {"v": {"name": "Acme", "id": "X1"}, "n": "Bob"},                # nested, perfect
    ),
]


@pytest.mark.parametrize("gt, pred", REFERENCE_CASES)
def test_matches_anls_star_on_agreeing_documents(gt, pred):
    ours = json_anls_score(gt, pred)  # schema=None -> all string/ANLS
    theirs = anls_star.anls_score(gt, pred)
    assert abs(ours - theirs) < 1e-6, f"ours={ours} theirs={theirs}"


def test_end_to_end_invoice_breakdown():
    schema = {
        "total": {"type": "money"},
        "currency": "categorical",
        "issue_date": {"type": "date"},
        "vendor": {"name": "string", "vat_id": "categorical"},
        "line_items": [{"desc": "string", "qty": "numeric", "price": "money"}],
    }
    gt = {
        "total": 100,
        "currency": "USD",
        "issue_date": "2024-01-15",
        "vendor": {"name": "Acme Inc", "vat_id": "X1"},
        "line_items": [
            {"desc": "Apple", "qty": 1, "price": 40},
            {"desc": "Banana", "qty": 2, "price": 60},
        ],
    }
    pred = {
        "total": 90,                       # money: 1 - 10/100 = 0.9
        "currency": "EUR",                 # categorical wrong: 0.0
        "issue_date": "Jan 15, 2024",      # date correct: 1.0
        "vendor": {"name": "Acme Inc", "vat_id": "X1"},  # 1.0, 1.0 (2 leaves)
        "line_items": [
            {"desc": "Apple", "qty": 1, "price": 40},    # perfect
            {"desc": "Banana", "qty": 2, "price": 60},   # perfect
        ],
    }
    # Per-child (score, n_leaves) contributions to the top object:
    #   total       (0.9, 1) -> 0.9
    #   currency    (0.0, 1) -> 0.0
    #   issue_date  (1.0, 1) -> 1.0
    #   vendor      (1.0, 2) -> 2.0
    #   line_items  (1.0, 2) -> 2.0   (2 perfect slots)
    # sum = 5.9 over n_leaves = 7
    expected = 5.9 / 7
    score, node = json_anls_score(gt, pred, schema, return_key_scores=True)
    assert abs(score - expected) < 1e-9
    assert node.children["currency"].score == 0.0
    assert abs(node.children["total"].score - 0.9) < 1e-9
```

- [ ] **Step 2: Run to verify behavior**

Run: `python -m pytest tests/test_json_score_reference.py -v`
Expected: cross-check cases pass (or skip with "could not import 'anls_star'" if not installed); `test_end_to_end_invoice_breakdown` passes.

If a cross-check case fails by a tiny margin, the cause is almost certainly a string-normalization detail (e.g. `anls_star` lower-casing) — adjust the *test data* to use already-identical or clearly-different lowercase strings. Do NOT loosen the `1e-6` tolerance, and do NOT add hallucinated-key or non-perfect-nested cases (they diverge by design — see the module docstring at the top of this plan).

- [ ] **Step 3: Run the full json_score test suite**

Run: `python -m pytest tests/test_json_score_comparators.py tests/test_json_score_schema.py tests/test_json_score_core.py tests/test_json_score_api.py tests/test_json_score_reference.py -v`
Expected: all pass (reference cross-check may skip if `anls_star` absent).

- [ ] **Step 4: Commit**

```bash
git add tests/test_json_score_reference.py
git commit -m "test(json_score): anls_star cross-check and end-to-end invoice"
```

---

## Done

The metric is importable as:

```python
from unsloth.eval.json_score import json_anls_score, score_from_text
```

with the schema/comparator/aggregation behavior defined in the spec.
```

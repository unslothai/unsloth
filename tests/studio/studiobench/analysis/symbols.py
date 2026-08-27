# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The symbol bridge: recover react-dom function names without measuring a dev build.

THE PROBLEM, stated exactly. React ships `cjs/react-dom-client.production.js`
and `cjs/react-dom-profiling.profiling.js` ALREADY MINIFIED. Vite never sees the
original identifiers, so `keep_fnames` has nothing to keep and a source map maps
`Zk` to the character offset in react-dom's own pre-minified file where `Zk` is
also what it is called. Source maps and `keep_fnames` recover OUR component
names beautifully and recover nothing at all inside react-dom. The original name
really is `Zk`. This is not a build misconfiguration and no build flag fixes it.

THE SOLUTION. React's DEVELOPMENT build has real names. Use it strictly as a
DICTIONARY and never as a measurement, because a dev build has different code,
different fast paths, extra warnings and different allocation behaviour: a
millisecond measured there describes a program nobody ships.

The join key is the EXACT CALL-COUNT VECTOR. Run the identical fixture at two or
three small rungs against both builds under precise coverage, and for each
function record `(count at rung 1, count at rung 2, ...)`. Invocation counts are
a semantic invariant across build modes: `cloneChildFibers` is called the same
number of times whichever bundle you loaded, because it is called once per
sibling per render either way, and that is a property of the algorithm rather
than of the minifier. So a function with vector `(340, 3400, 34000)` in dev and
a function with the same vector in prod are the same function.

THE THREE WAYS THIS GOES WRONG, each closed by a rule below:

1. **Collisions.** Many trivial functions share a vector, especially small
   integers like `(1, 1, 1)`. A vector that is not UNIQUE WITHIN ITS OWN BUILD
   is unusable, so it is recorded as ambiguous and never guessed. This is the
   rule that keeps the bridge honest, and it discards a lot.
2. **A bridge that is confidently wrong.** Validated against ANCHORS: our own
   app components are independently named on BOTH sides through source maps, so
   they must map to themselves. If any anchor maps to something else, the whole
   bridge is discarded and the run degrades to unnamed frames with
   `symbol_bridge: failed`. Not the bad anchor, the WHOLE bridge, because a
   bridge that mislabels one function it can check will mislabel others it
   cannot.
3. **Dev timings leaking.** `assert_no_measurements` refuses any float in the
   persisted artefact, and the artefact schema carries no duration fields at all.

The result is persisted as `symbols/react-dom@<version>-<bundle-sha>.json`. The
bundle SHA is part of the key because a bridge is only valid for the exact bytes
it was built against; a new build renames everything.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import CellFailure

FAILED = "failed"
OK = "ok"
NOT_BUILT = "not_built"

# A function whose counts are all this small carries almost no information and
# collides with everything. Excluded from matching, and the exclusion is
# reported so the coverage of the bridge is visible.
MIN_INFORMATIVE_COUNT = 2

# Above this share of resolved mappings being name-identical (`push` -> `push`),
# the two arms are almost certainly the same build. Some identity is EXPECTED
# and healthy: React's release minifier leaves scheduler entry points such as
# `push`, `peek` and `performWorkUntilDeadline` alone, and `keepNames` preserves
# others, so a genuine bridge measured on a real React 19.2.4 pair came in at
# roughly 0.1. A same-build pair comes in at exactly 1.0, so the two regimes are
# not close and this threshold is not delicate.
MAX_IDENTITY_MAPPING_FRACTION = 0.5

# NOTE FOR THE NEXT PERSON, because this is the obvious idea and it DOES NOT
# WORK. You cannot detect "is this the development build?" by looking at how
# long the function names are. Measured on the real pair: the development bundle
# had a median react-dom function-name length of 19 with 97.6% of names at four
# characters or more, and the PRODUCTION bundle had a median of 19 with 95.4%.
# React's own minifier renames only part of react-dom, `keepNames` preserves the
# rest, and V8 reports an inferred name for much of what is left. Name shape
# tells you nothing. The checks below are structural instead.


@dataclass(frozen = True)
class FunctionVector:
    """A function identified by its exact call counts across the rung ladder."""

    build: str  # "dev" or "prod"
    url: str
    function_name: str
    script_id: str
    start_offset: int
    end_offset: int
    counts: tuple[int, ...]

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.url, self.start_offset, self.end_offset)

    @property
    def informative(self) -> bool:
        return max(self.counts, default = 0) >= MIN_INFORMATIVE_COUNT and any(self.counts)

    def label(self) -> str:
        return f"{self.function_name or '(anonymous)'}@{self.url}[{self.start_offset}:{self.end_offset}]"


@dataclass
class Bridge:
    """A persisted prod-symbol to dev-name mapping, or an honest failure."""

    status: str = NOT_BUILT
    react_version: str = ""
    bundle_sha: str = ""
    rungs: tuple[str, ...] = ()
    # prod function key -> dev function name
    mapping: dict[str, str] = field(default_factory = dict)
    # prod function key -> the count vector that matched it, for auditing
    evidence: dict[str, list[int]] = field(default_factory = dict)
    ambiguous_prod: list[str] = field(default_factory = list)
    ambiguous_dev: list[str] = field(default_factory = list)
    unmatched_prod: int = 0
    # Resolved functions whose dev name equals their prod name. Expected to be
    # small and non-zero; near 100% means both arms were the same build.
    identity_mappings: int = 0
    anchors_checked: int = 0
    anchor_failures: list[str] = field(default_factory = list)
    failure_reason: str = ""

    @staticmethod
    def prod_key(url: str, start_offset: int, end_offset: int) -> str:
        # The URL is reduced to its basename because an Unsloth install serves the
        # same bundle from a hashed path that changes per install, while the
        # offsets inside the bundle do not.
        return f"{os.path.basename(url)}:{start_offset}:{end_offset}"

    def resolve(self, url: str, start_offset: int, end_offset: int) -> str | None:
        if self.status != OK:
            return None
        return self.mapping.get(self.prod_key(url, start_offset, end_offset))

    def resolve_frame_label(
        self, url: str, start_offset: int, end_offset: int, fallback: str
    ) -> str:
        name = self.resolve(url, start_offset, end_offset)
        return f"{name} (bridged)" if name else fallback

    def summary(self) -> dict[str, Any]:
        return {
            "symbol_bridge": self.status,
            "react_version": self.react_version,
            "bundle_sha": self.bundle_sha[:12],
            "rungs": list(self.rungs),
            "resolved_functions": len(self.mapping),
            "ambiguous_prod": len(self.ambiguous_prod),
            "ambiguous_dev": len(self.ambiguous_dev),
            "unmatched_prod": self.unmatched_prod,
            "identity_mappings": self.identity_mappings,
            "anchors_checked": self.anchors_checked,
            "anchor_failures": self.anchor_failures,
            "failure_reason": self.failure_reason,
        }

    def to_json(self) -> dict[str, Any]:
        doc = {
            "schema": "studiobench.symbols/1",
            "status": self.status,
            "react_version": self.react_version,
            "bundle_sha": self.bundle_sha,
            "rungs": list(self.rungs),
            "mapping": dict(self.mapping),
            "evidence": {k: list(v) for k, v in self.evidence.items()},
            "ambiguous_prod": list(self.ambiguous_prod),
            "ambiguous_dev": list(self.ambiguous_dev),
            "unmatched_prod": self.unmatched_prod,
            "identity_mappings": self.identity_mappings,
            "anchors_checked": self.anchors_checked,
            "anchor_failures": list(self.anchor_failures),
            "failure_reason": self.failure_reason,
        }
        assert_no_measurements(doc)
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> "Bridge":
        return cls(
            status = str(doc.get("status", NOT_BUILT)),
            react_version = str(doc.get("react_version", "")),
            bundle_sha = str(doc.get("bundle_sha", "")),
            rungs = tuple(doc.get("rungs") or ()),
            mapping = dict(doc.get("mapping") or {}),
            evidence = {k: list(v) for k, v in (doc.get("evidence") or {}).items()},
            ambiguous_prod = list(doc.get("ambiguous_prod") or ()),
            ambiguous_dev = list(doc.get("ambiguous_dev") or ()),
            unmatched_prod = int(doc.get("unmatched_prod", 0)),
            identity_mappings = int(doc.get("identity_mappings", 0)),
            anchors_checked = int(doc.get("anchors_checked", 0)),
            anchor_failures = list(doc.get("anchor_failures") or ()),
            failure_reason = str(doc.get("failure_reason", "")),
        )

    def filename(self) -> str:
        return f"react-dom@{self.react_version or 'unknown'}-{self.bundle_sha[:12] or 'nosha'}.json"

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok = True)
        path = os.path.join(directory, self.filename())
        with open(path, "w", encoding = "utf-8") as fh:
            json.dump(self.to_json(), fh, indent = 2, sort_keys = True)
        return path

    @classmethod
    def load(cls, path: str) -> "Bridge":
        with open(path, "r", encoding = "utf-8") as fh:
            return cls.from_json(json.load(fh))


def assert_no_measurements(payload: Any, path: str = "bridge") -> None:
    """Refuse any float anywhere in a bridge artefact.

    The dev build is a dictionary, never a measurement. This is the mechanical
    guarantee behind that sentence: the artefact can hold names and integers and
    nothing else, so there is no field a duration could sit in even by accident.
    """
    if isinstance(payload, bool):
        return
    if isinstance(payload, float):
        raise CellFailure(
            "dev_measurement_leak",
            f"{path} holds a float ({payload!r}). The development build is a "
            "dictionary, not a measurement; no dev millisecond may enter a table.",
        )
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            assert_no_measurements(v, f"{path}.{k}")
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_no_measurements(v, f"{path}[{i}]")


def bundle_sha(source: str | bytes) -> str:
    data = source.encode("utf-8") if isinstance(source, str) else source
    return hashlib.sha256(data).hexdigest()


def vectors_from_snapshots(
    snapshots: Sequence[Any],
    build: str,
    *,
    url_filter: str | None = None,
) -> list[FunctionVector]:
    """Turn one build's per-rung coverage snapshots into count vectors.

    `snapshots` are `instruments.coverage.CoverageSnapshot` objects in RUNG
    ORDER, and the order must be identical for both builds or the vectors
    describe different experiments and every match is spurious.

    A function missing from a rung contributes a 0 at that position rather than
    being dropped, because "never called at the small rung" is itself part of
    the signature.
    """
    if not snapshots:
        return []
    per_rung: list[dict[tuple[str, int, int], Any]] = []
    for snap in snapshots:
        table: dict[tuple[str, int, int], Any] = {}
        for f in snap.functions:
            if url_filter and url_filter not in f.url:
                continue
            table[(f.url, f.start_offset, f.end_offset)] = f
        per_rung.append(table)

    all_keys: set[tuple[str, int, int]] = set()
    for table in per_rung:
        all_keys |= set(table)

    out: list[FunctionVector] = []
    for key in sorted(all_keys):
        counts = tuple(int(table[key].count) if key in table else 0 for table in per_rung)
        sample = next(table[key] for table in per_rung if key in table)
        out.append(
            FunctionVector(
                build = build,
                url = key[0],
                function_name = str(sample.function_name or ""),
                script_id = str(sample.script_id),
                start_offset = key[1],
                end_offset = key[2],
                counts = counts,
            )
        )
    return out


def _index_unique(
    vectors: Iterable[FunctionVector],
) -> tuple[dict[tuple[int, ...], FunctionVector], list[str]]:
    """Index by count vector, keeping only vectors unique within this build."""
    buckets: dict[tuple[int, ...], list[FunctionVector]] = {}
    for v in vectors:
        if not v.informative:
            continue
        buckets.setdefault(v.counts, []).append(v)
    unique: dict[tuple[int, ...], FunctionVector] = {}
    ambiguous: list[str] = []
    for counts, group in buckets.items():
        if len(group) == 1:
            unique[counts] = group[0]
        else:
            ambiguous.append(
                f"{list(counts)} shared by {len(group)}: " + ", ".join(g.label() for g in group[:4])
            )
    return unique, ambiguous


def build_bridge(
    dev_snapshots: Sequence[Any],
    prod_snapshots: Sequence[Any],
    *,
    rungs: Sequence[str],
    react_version: str,
    bundle_source: str | bytes,
    anchor_names: Sequence[str],
    react_url_filter: str | None = None,
    anchor_url_filter: str | None = None,
) -> Bridge:
    """Match prod functions to dev names by exact call-count vector equality.

    `anchor_names` are functions that are independently named on BOTH sides,
    which in practice means our own app components: source maps and `keep_fnames`
    recover those in the production build, and the development build has them
    too. They are the control. If an anchor does not map to itself the bridge is
    discarded whole.
    """
    if len(dev_snapshots) != len(prod_snapshots):
        return Bridge(
            status = FAILED,
            react_version = react_version,
            bundle_sha = bundle_sha(bundle_source),
            rungs = tuple(rungs),
            failure_reason = (
                f"{len(dev_snapshots)} dev rungs against {len(prod_snapshots)} prod rungs; "
                "the two builds must run the identical ladder or the vectors describe "
                "different experiments"
            ),
        )
    if len(dev_snapshots) < 2:
        return Bridge(
            status = FAILED,
            react_version = react_version,
            bundle_sha = bundle_sha(bundle_source),
            rungs = tuple(rungs),
            failure_reason = (
                f"{len(dev_snapshots)} rung(s); a single-rung vector is one integer and "
                "collides with everything. Two or three rungs are the minimum."
            ),
        )

    bridge = Bridge(
        status = OK,
        react_version = react_version,
        bundle_sha = bundle_sha(bundle_source),
        rungs = tuple(rungs),
    )

    # ---- anchor validation, on our own app code, both sides named ----------
    dev_all = vectors_from_snapshots(dev_snapshots, "dev", url_filter = anchor_url_filter)
    prod_all = vectors_from_snapshots(prod_snapshots, "prod", url_filter = anchor_url_filter)
    dev_by_name: dict[str, list[FunctionVector]] = {}
    for v in dev_all:
        if v.function_name:
            dev_by_name.setdefault(v.function_name, []).append(v)
    prod_by_name: dict[str, list[FunctionVector]] = {}
    for v in prod_all:
        if v.function_name:
            prod_by_name.setdefault(v.function_name, []).append(v)

    for anchor in anchor_names:
        dev_hits = [v for v in dev_by_name.get(anchor, []) if v.informative]
        prod_hits = [v for v in prod_by_name.get(anchor, []) if v.informative]
        if not dev_hits or not prod_hits:
            bridge.anchor_failures.append(
                f"{anchor}: present in dev={len(dev_hits)} prod={len(prod_hits)} with a usable "
                "count vector; an anchor that does not run on both sides cannot validate anything"
            )
            continue
        bridge.anchors_checked += 1
        dev_counts = {v.counts for v in dev_hits}
        prod_counts = {v.counts for v in prod_hits}
        if not (dev_counts & prod_counts):
            bridge.anchor_failures.append(
                f"{anchor}: dev vectors {sorted(dev_counts)} vs prod vectors {sorted(prod_counts)} "
                "do not intersect, so call counts are NOT invariant across these two builds and "
                "the entire matching premise is false here"
            )

    if bridge.anchor_failures or bridge.anchors_checked == 0:
        bridge.status = FAILED
        bridge.mapping.clear()
        bridge.evidence.clear()
        bridge.failure_reason = (
            "anchor validation failed; discarding the whole bridge rather than the bad anchors, "
            "because a bridge that mislabels a function it can check will mislabel functions it cannot. "
            + ("no anchors were usable" if bridge.anchors_checked == 0 else "")
        )
        return bridge

    # ---- the two arms must actually be two different builds -----------------
    #
    # THE FAILURE THIS CLOSES. If both arms are pointed at the same server (a
    # port collision, a reused base URL, a dev server that quietly served the
    # production dist), every count vector matches trivially, EVERY ANCHOR
    # PASSES because an anchor genuinely does map to itself, and the bridge
    # reports `ok` with a mapping of `Zk` to `Zk`. That is a clean-looking run
    # that proves nothing, and it would put "bridged" next to a minified name.
    # The anchor check cannot catch it; it is the one thing anchors are blind
    # to, because anchors test invariance and same-build is perfectly invariant.
    dev_all_keys = {v.key for v in dev_all if v.informative}
    prod_all_keys = {v.key for v in prod_all if v.informative}
    if dev_all_keys and dev_all_keys == prod_all_keys:
        bridge.status = FAILED
        bridge.failure_reason = (
            "both arms observed an IDENTICAL set of functions at identical byte offsets in "
            "identically named scripts, so they were serving the same bundle. A bridge built "
            "from one build maps minified names to themselves and reads as a success. Check "
            "that the dev and prod arms are on different ports and different installs."
        )
        return bridge

    # ---- the actual matching, restricted to react-dom ----------------------
    dev_react = vectors_from_snapshots(dev_snapshots, "dev", url_filter = react_url_filter)
    prod_react = vectors_from_snapshots(prod_snapshots, "prod", url_filter = react_url_filter)
    dev_index, dev_ambiguous = _index_unique(dev_react)
    prod_index, prod_ambiguous = _index_unique(prod_react)
    bridge.ambiguous_dev = dev_ambiguous
    bridge.ambiguous_prod = prod_ambiguous

    matched = 0
    identical_names = 0
    for counts, prod_fn in prod_index.items():
        dev_fn = dev_index.get(counts)
        if dev_fn is None or not dev_fn.function_name:
            continue
        key = Bridge.prod_key(prod_fn.url, prod_fn.start_offset, prod_fn.end_offset)
        bridge.mapping[key] = dev_fn.function_name
        bridge.evidence[key] = list(counts)
        if prod_fn.function_name == dev_fn.function_name:
            identical_names += 1
        matched += 1
    bridge.unmatched_prod = len(prod_index) - matched
    bridge.identity_mappings = identical_names

    if matched == 0:
        bridge.status = FAILED
        bridge.failure_reason = (
            "no prod function had a count vector that was unique in prod AND unique in dev AND "
            "equal across the two. Widen the rung ladder so vectors carry more information."
        )
        return bridge

    # Second guard on the same failure, for the case where the two arms differ
    # in offsets but are still the same code (a rebuild of one tree, say). If
    # nearly every resolved name maps to itself, no minification was undone and
    # the bridge is not doing anything.
    fraction = identical_names / matched
    if fraction > MAX_IDENTITY_MAPPING_FRACTION:
        bridge.status = FAILED
        bridge.mapping.clear()
        bridge.evidence.clear()
        bridge.failure_reason = (
            f"{identical_names} of {matched} resolved functions map to their own name "
            f"({fraction * 100:.0f}%, limit {MAX_IDENTITY_MAPPING_FRACTION * 100:.0f}%). A bridge "
            "that recovers the names it already had has undone no minification, which means both "
            "arms were built the same way. A genuine dev-to-prod pair sits near 10%, because only "
            "the scheduler entry points keep their names in both."
        )
    return bridge


def apply_bridge(
    bridge: Bridge, frames: Sequence[tuple[str, str, int, int]]
) -> dict[tuple[str, str, int, int], str]:
    """Resolve `(label, url, start, end)` tuples to bridged names where possible."""
    out: dict[tuple[str, str, int, int], str] = {}
    for label, url, start, end in frames:
        name = bridge.resolve(url, start, end)
        if name:
            out[(label, url, start, end)] = name
    return out

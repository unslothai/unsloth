# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Hunt the volatile behind a null-control parity mismatch: WHICH BYTES MOVED, not how many.

`sweep/ui_parity.py` has pointed at `parity_null_control.py --hunt` since it was written, and the
tool did not exist. This is it, and the gap it fills is the difference between fixing a
normalisation gap and guessing at one.

THE PROBLEM WITH A DIGEST. `msg22(assistant):7400->7400c` says two DOMs of identical length hashed
differently. It cannot say what changed, so every response to it is a guess, and a wrong guess
added to `parity.js` is worse than the mismatch: a normaliser that erases something real turns the
instrument into a decoration that always passes. `scene/parity.js` has carried a `raw` option for
exactly this since it was written, returning the NORMALISED signature text beside the digest, and
`studiobench --parity-raw` is what asks for it.

So: run a base-vs-base null control with `--parity-raw`, point this at it, and it prints the exact
substring that differs between the two arms, in context, with the characters that moved marked.
Then the question "is this genuinely volatile, or is the normaliser dropping something it should
keep" is answerable by reading rather than by argument.

    python -m tests.studio.studiobench --tier fast --rungs 100K --reps 2 --parity-raw \\
        --attach http://127.0.0.1:5399 --attach-b http://127.0.0.1:5400 \\
        --ab <the same ref> --out outputs/null
    python -m tests.studio.studiobench.sweep.parity_null_control --hunt outputs/null

WHAT IT REFUSES. A payload recorded without `--parity-raw` carries digests only, and a hunt over it
would report "no differences found" from a run that had plenty. That is reported as a refusal, not
as a clean result.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.sweep.ui_parity import collect, shards_of  # noqa: E402

#: How much context to print either side of a differing run of characters.
CONTEXT = 90


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def common_suffix(a: str, b: str, floor: int) -> int:
    """Longest common suffix, not overrunning `floor` characters already matched as a prefix."""
    i = 0
    while i < (min(len(a), len(b)) - floor) and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return i


def first_divergence(base: str, treat: str) -> dict | None:
    """The one differing region between two normalised signatures, in context.

    Prefix/suffix trimming rather than a real diff: the two strings are two serialisations of the
    same page, so a mismatch is almost always ONE substitution, and naming it exactly beats a
    hunk-based diff that has to be read. When there are several the first one is still the one to
    fix, and `--hunt` aggregates across every pair anyway.
    """
    if base == treat:
        return None
    pre = common_prefix(base, treat)
    suf = common_suffix(base, treat, pre)
    return {
        "at": pre,
        "base": base[pre : len(base) - suf],
        "treat": treat[pre : len(treat) - suf],
        "before": base[max(0, pre - CONTEXT) : pre],
        "after": base[len(base) - suf : len(base) - suf + CONTEXT],
        "same_length": len(base) == len(treat),
    }


#: Patterns that name a differing region, so a hunt over hundreds of pairs reports a handful of
#: MECHANISMS rather than hundreds of substrings. Order matters: first match wins.
SHAPES = (
    ("a bare integer", re.compile(r"^\d+$")),
    ("an integer inside a longer token", re.compile(r"^[\w:.\-/]*\d+[\w:.\-/]*$")),
    ("a short hex run", re.compile(r"^[0-9a-f]{4,15}$", re.I)),
    ("a base62 / nanoid-shaped token", re.compile(r"^[0-9A-Za-z_-]{6,}$")),
    ("an ISO-ish timestamp", re.compile(r"^\d{4}-\d{2}-\d{2}")),
    ("whitespace only", re.compile(r"^\s*$")),
)


def shape_of(region: str) -> str:
    for name, pattern in SHAPES:
        if pattern.match(region):
            return name
    if len(region) > 400:
        return "a large block of content (likely stream progress, not a volatile)"
    return "unclassified"


def raw_pairs(paths: list[Path]) -> tuple[list[tuple], int, int]:
    """[(action, cell, kind, index, base raw, treatment raw)] over every pair that carries raw."""
    got = collect(paths)
    out, with_raw, without = [], 0, 0
    for (_shard, rung, rep, _sid, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            continue
        bp, tp = sides["base"].get("parity"), sides["treatment"].get("parity")
        if not isinstance(bp, dict) or not isinstance(tp, dict):
            continue
        if not (bp.get("parity_attempted") and tp.get("parity_attempted")):
            continue
        cell = f"{rung} {rep}"
        if "raw" not in bp or "raw" not in tp:
            without += 1
            continue
        with_raw += 1
        out.append((action, cell, "thread", -1, bp["raw"], tp["raw"]))
        for i, (bm, tm) in enumerate(zip(bp.get("messages", []), tp.get("messages", []))):
            if "raw" in bm and "raw" in tm:
                out.append(
                    (action, cell, f"msg{i}({bm.get('role', '?')})", i, bm["raw"], tm["raw"])
                )
        for i, (bo, to) in enumerate(zip(bp.get("overlays", []), tp.get("overlays", []))):
            if "raw" in bo and "raw" in to:
                out.append((action, cell, f"overlay[{bo.get('sel', i)}]", i, bo["raw"], to["raw"]))
    return out, with_raw, without


def hunt(paths: list[Path], limit: int = 12) -> int:
    pairs, with_raw, without = raw_pairs(paths)
    if not pairs:
        print(
            f"\nNO RAW SIGNATURES in {len(paths)} payload(s) ({without} pair(s) carried digests "
            f"only).\nRe-run the null control with --parity-raw; without it a hunt can only "
            f"report that\ntwo digests differ, which is the thing you already knew."
        )
        return 2

    findings = []
    for action, cell, kind, _i, base, treat in pairs:
        d = first_divergence(base, treat)
        if d is not None:
            d.update(action = action, cell = cell, kind = kind)
            findings.append(d)

    print(f"\nHUNT over {len(pairs)} raw pair(s) from {len(paths)} shard(s)")
    print(f"  pairs carrying raw signatures: {with_raw}")
    print(f"  differing regions found:       {len(findings)}")
    if not findings:
        print("\n  Every raw signature matched. This null control has no volatile to hunt.")
        return 0

    by_shape: dict[str, list] = collections.defaultdict(list)
    for f in findings:
        by_shape[f"{shape_of(f['base'])} -> {shape_of(f['treat'])}"].append(f)

    print("\n  BY SHAPE, which is the thing to fix rather than the individual bytes:")
    for shape, group in sorted(by_shape.items(), key = lambda kv: -len(kv[1])):
        same = sum(1 for f in group if f["same_length"])
        print(f"    {len(group):>4}x  {shape}   ({same} of them same-length)")

    print(f"\n  THE FIRST {min(limit, len(findings))} REGIONS, in context:")
    for f in findings[:limit]:
        print(
            f"\n    {f['action']} {f['cell']} {f['kind']} at offset {f['at']}"
            f"{'  [same length]' if f['same_length'] else ''}"
        )
        print(f"      ...{f['before'][-CONTEXT:]}")
        print(f"      base      >>>{f['base'][:200]}<<<")
        print(f"      treatment >>>{f['treat'][:200]}<<<")
        print(f"      ...{f['after']}")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("payloads", nargs = "+", help = "null-control output dirs (globs allowed)")
    ap.add_argument(
        "--hunt",
        action = "store_true",
        help = "name the bytes that differ between the two arms, rather than counting mismatches",
    )
    ap.add_argument("--limit", type = int, default = 12, help = "how many regions to print")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.payloads:
        paths.extend(shards_of(pattern))
    if not paths:
        print(f"no payload found for {args.payloads}")
        return 2
    if not args.hunt:
        print("nothing to do without --hunt (this tool has one job)")
        return 2
    return hunt(paths, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

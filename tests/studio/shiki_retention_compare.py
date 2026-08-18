# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paired base/head comparison of the retained-heap slopes.

Ratios are the result and absolute megabytes are context, so this pairs rep N of base with rep N
of head, takes the ratio inside the pair, and reports the MEDIAN of those ratios. The runs are
interleaved base, head, base, head, so a pair is two runs a few minutes apart rather than two
blocks half an hour apart on a host at load average 60.

Before any of that it censuses the fixtures: the page generates its sources from a seeded LCG and
reports an FNV-1a hash of each, and a pair whose hashes disagree is not a pair. That is a
SystemExit, not a warning.

Produce the inputs by running `tests/studio/playwright_shiki_retention.py` alternately in two
checkouts, base first, with `PW_ART_DIR` pointing both at the same directory and `SMOKE_SD_LABEL`
set to `base-rN` / `head-rN`:

    for rep in 1 2 3; do
      for arm in base head; do
        ( cd "$WORKTREE_FOR_$arm" \
          && SMOKE_SD_LABEL="$arm-r$rep" PW_ART_DIR=/abs/path/to/out \
             python tests/studio/playwright_shiki_retention.py )
      done
    done

Then:

    python tests/studio/shiki_retention_compare.py /abs/path/to/out
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(directory: Path, mode: str) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(directory.glob(f"shiki-retention-{mode}-*.json")):
        report = json.loads(path.read_text())
        label = report["label"]
        arm, _, rep = label.partition("-r")
        runs.setdefault(arm, {})[rep] = report
    return runs


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/sd_paired")
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    runs = load(directory, mode)
    if set(runs) != {"base", "head"}:
        raise SystemExit(f"expected base and head runs, found {sorted(runs)}")
    reps = sorted(set(runs["base"]) & set(runs["head"]), key = int)
    if not reps:
        raise SystemExit("no rep is present in both arms")

    # Fixture census, AFTER both arms exist and for every rep that will be compared.
    for rep in reps:
        base_cells = runs["base"][rep]["cells"]
        head_cells = runs["head"][rep]["cells"]
        if set(base_cells) != set(head_cells):
            raise SystemExit(
                f"rep {rep}: arms ran different cells, {sorted(base_cells)} vs {sorted(head_cells)}"
            )
        for name, cell in base_cells.items():
            other = head_cells[name]
            if cell["fixture_hash"] != other["fixture_hash"]:
                raise SystemExit(
                    f"rep {rep} {name}: fixture mismatch, "
                    f"{cell['fixture_hash']} vs {other['fixture_hash']}"
                )
            for field in ("chars", "kind", "tick_ms"):
                if cell.get(field) != other.get(field):
                    raise SystemExit(
                        f"rep {rep} {name}: {field} differs, {cell.get(field)} vs {other.get(field)}"
                    )
    print(f"fixture census: {len(reps)} pair(s) agree exactly on every cell")

    cells = sorted(runs["base"][reps[0]]["cells"])
    print()
    header = f"  {'cell':<20}{'base MB/fence':>15}{'head MB/fence':>15}{'ratio':>10}{'R^2 b/h':>16}"
    print(header)
    for name in cells:
        base_values = [runs["base"][r]["cells"][name]["slope_mb_per_fence"] for r in reps]
        head_values = [runs["head"][r]["cells"][name]["slope_mb_per_fence"] for r in reps]
        # Guard the ratio: a control arm sits near zero in both arms and a raw division there
        # prints a number that looks like a result and is noise.
        ratios = [
            (b / h) if abs(h) > 0.02 else float("nan") for b, h in zip(base_values, head_values)
        ]
        finite = [r for r in ratios if r == r]
        ratio = f"{statistics.median(finite):.2f}x" if finite else "n/a"
        r2_base = statistics.median(runs["base"][r]["cells"][name]["r2"] for r in reps)
        r2_head = statistics.median(runs["head"][r]["cells"][name]["r2"] for r in reps)
        print(
            f"  {name:<20}{statistics.median(base_values):>15.2f}"
            f"{statistics.median(head_values):>15.2f}{ratio:>10}"
            f"{f'{r2_base:.3f}/{r2_head:.3f}':>16}"
        )
    print()
    print(f"  reps paired: {', '.join(reps)}   (median of paired ratios, not ratio of medians)")
    for name in cells:
        pairs = [
            f"{runs['base'][r]['cells'][name]['slope_mb_per_fence']:.2f}/"
            f"{runs['head'][r]['cells'][name]['slope_mb_per_fence']:.2f}"
            for r in reps
        ]
        print(f"  {name:<20}{'  '.join(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

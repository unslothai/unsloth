# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Turn a red parity verdict into something a reviewer can judge in ten seconds.

A red verdict currently hands a reader `msg22(assistant):17334->17334c` and nothing else. Two hex
digests and a character count cannot answer the only question anybody has, which is whether the
change is real; so the cost of a false alarm is that somebody re-runs the job, and the cost of a
true one is that somebody has to reproduce it by hand. This pairs the two arms' screenshots for
exactly the actions the verdict called out.

WHAT MAKES A PAIR HONEST, and every one of these is a way a clean-looking pair proves nothing:

  IT MUST NAME ITS SIDE.        Both arms share one fixture, one film and one seeded thread. The
                                image carries nothing that says which build it is, so the arm, the
                                action and the cell are burned into the picture rather than left
                                in the filename, where a copy or a reorder loses them.
  IT MUST NOT BE SCALED.        Equalising two heights by scaling makes the shorter half larger
                                and the pair looks retouched; a genuine one-line difference then
                                reads as a layout change. Both halves are padded onto one canvas.
  SCROLL MUST BE COMPARABLE.    Two shots at different offsets look exactly like a UI change. The
                                offsets are recorded at capture time and printed on each half, and
                                a pair whose offsets disagree is labelled MISMATCHED rather than
                                presented as a comparison.
  A MISSING HALF IS NOT A PAIR. An action that ran on one arm and not the other has one image.
                                That is reported as a missing half, never silently rendered as a
                                single picture the reader will take for both.

Pillow is optional. Without it the shots are still collected and copied out under their labelled
names, and the absence of the composite is stated rather than skipped over.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.sweep.ui_parity import (  # noqa: E402
    compare_all,
    corroborated,
    is_unstable,
    shards_of,
    unstable_set,
)

BANNER = 46  # px of label strip above each half
GUTTER = 14  # px between the two halves
BG = (24, 24, 27)
FG = (244, 244, 245)
ACCENT = (248, 113, 113)


def _pil():
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415
        return Image, ImageDraw
    except Exception:  # noqa: BLE001
        return None, None


def shot_index(paths: list[Path]) -> dict:
    """{(cell_id, action, arm): {"file": name, "scroll": int}} from the payload's own rows.

    Read from the payload rather than by globbing the directory, so a file whose name happens to
    look right but was written by another run cannot be picked up.
    """
    out: dict = {}
    for path in paths:
        for line in path.read_text(encoding = "utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "action":
                continue
            parity = row.get("parity") or {}
            if not isinstance(parity, dict) or not parity.get("shot"):
                continue
            cid = row.get("cell_id") or ""
            arm = "treatment" if ".treatment." in cid else "base"
            out[(cid, row.get("action"), arm)] = {
                "file": parity["shot"],
                "scroll": parity.get("shot_scroll_top", -1),
            }
    return out


def differing_actions(
    result_paths: list[Path],
    null_paths: list[Path],
    min_reps: int = 1,
) -> list[dict]:
    """The STABLE differences, i.e. exactly what turned the verdict red. Not every mismatch.

    An excused difference is noise the null control already accounted for, and shooting it would
    bury the one picture that matters under a dozen that do not.

    THE SAME THRESHOLD THE VERDICT USED, for the same reason. When the verdict runs at
    `--min-reps 2`, a stable difference seen in one repetition of two is explicitly UNCORROBORATED
    and does not turn the gate red. Illustrating it anyway would put pictures of one-off flakes in
    the same artifact as the change that actually failed the job, and the reader has no way to
    tell which is which -- so the artifact would bury the finding it exists to show, which is the
    same failure as not producing it.
    """
    unstable, _derived, _checks = unstable_set(null_paths or None)
    out = []
    # `compare_all` returns (results, capture tally); only the results are wanted here.
    results, _tally = compare_all(result_paths)
    stable = [
        (action, shard, cell, r.get("moved", []))
        for action, shard, cell, r in results
        if r["verdict"] == P.DIFFER and not is_unstable(unstable, action, cell)
    ]
    firm, _weak = corroborated(stable, min_reps)
    for action, shard, cell, moved in firm:
        out.append({"action": action, "shard": shard, "cell": cell, "moved": moved})
    return out


def _label(
    draw,
    x,
    y,
    w,
    text,
    sub,
    accent = False,
):
    draw.rectangle([x, y, x + w, y + BANNER], fill = (63, 63, 70) if not accent else (69, 26, 26))
    draw.text((x + 10, y + 6), text, fill = ACCENT if accent else FG)
    draw.text((x + 10, y + 24), sub, fill = FG)


def composite(before: Path, after: Path, out: Path, meta: dict) -> bool:
    Image, ImageDraw = _pil()
    if Image is None:
        return False
    a, b = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
    # PADDED onto one canvas, never scaled: see the module docstring.
    w = max(a.width, b.width)
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (w * 2 + GUTTER, h + BANNER), BG)
    canvas.paste(a, (0, BANNER))
    canvas.paste(b, (w + GUTTER, BANNER))
    draw = ImageDraw.Draw(canvas)
    mismatched = meta["before_scroll"] != meta["after_scroll"]
    sub = f"{meta['action']}  {meta['cell']}  scrollTop={meta['before_scroll']}"
    _label(draw, 0, 0, w, "BEFORE  (base = merge base)", sub)
    sub2 = f"{meta['action']}  {meta['cell']}  scrollTop={meta['after_scroll']}"
    _label(draw, w + GUTTER, 0, w, "AFTER  (treatment = head)", sub2, accent = True)
    if mismatched:
        draw.text(
            (10, h + BANNER - 18),
            "SCROLL OFFSETS DIFFER -- this pair is NOT a like-for-like comparison",
            fill = ACCENT,
        )
    canvas.save(out)
    return True


def build(
    result_dir: Path,
    null_dir: Path,
    shots_dir: Path,
    out_dir: Path,
    min_reps: int = 1,
) -> int:
    result_paths = shards_of(str(result_dir))
    null_paths = shards_of(str(null_dir)) if null_dir else []
    if not result_paths:
        print(f"no result payload under {result_dir}")
        return 2
    diffs = differing_actions(result_paths, null_paths, min_reps)
    index = shot_index(result_paths)
    out_dir.mkdir(parents = True, exist_ok = True)

    if not diffs:
        print("\nno STABLE differences to illustrate; the verdict was not red on structure.")
        return 0

    print(f"\nSCREENSHOT EVIDENCE for {len(diffs)} stable difference(s)")
    made = missing = 0
    for d in diffs:
        # `cell` is "<rung> <rep>", and a cell id is "<rung>.<arm>.<rep>".
        rung, rep = d["cell"].split(" ", 1)
        base_id, treat_id = f"{rung}.base.{rep}", f"{rung}.treatment.{rep}"
        b = index.get((base_id, d["action"], "base"))
        t = index.get((treat_id, d["action"], "treatment"))
        if not b or not t:
            missing += 1
            have = "base" if b else ("treatment" if t else "neither")
            print(f"  {d['action']:<26} {d['cell']}: MISSING HALF (have {have}); not a pair")
            for side in (b, t):
                if side:
                    src = shots_dir / side["file"]
                    if src.exists():
                        shutil.copy2(src, out_dir / src.name)
            continue
        bp, tp = shots_dir / b["file"], shots_dir / t["file"]
        if not (bp.exists() and tp.exists()):
            missing += 1
            print(f"  {d['action']:<26} {d['cell']}: shot file absent on disk; not a pair")
            continue
        stem = f"{d['action']}__{rung}_{rep}"
        ok = composite(
            bp,
            tp,
            out_dir / f"{stem}__composite.png",
            {
                "action": d["action"],
                "cell": d["cell"],
                "before_scroll": b["scroll"],
                "after_scroll": t["scroll"],
            },
        )
        shutil.copy2(bp, out_dir / f"{stem}__BEFORE_base.png")
        shutil.copy2(tp, out_dir / f"{stem}__AFTER_treatment.png")
        made += 1
        note = "" if ok else "  (no composite: Pillow is not installed)"
        scroll = (
            ""
            if b["scroll"] == t["scroll"]
            else f"  SCROLL MISMATCH {b['scroll']} vs {t['scroll']}"
        )
        print(f"  {d['action']:<26} {d['cell']}: {', '.join(d['moved'][:2])}{note}{scroll}")

    print(f"\n  {made} pair(s) written to {out_dir}, {missing} incomplete")
    return 0


def prune(payload_dir: Path, shots_dir: Path) -> int:
    """Delete every shot for an action whose two arms AGREED. Run on the arm, before upload.

    The images are the only large thing this job produces, and on a clean run every one of them
    is a picture of two pages that matched. Deciding this here needs only the arm's own payload,
    because it is the question "did the two arms differ", not "was the difference excused" -- the
    second needs the null control and belongs to the verdict job. So a few excused pairs survive
    this and are dropped there, which is the right way round: erasing an image on the arm cannot
    be undone, and the verdict job can always ignore one.

    An action with no digest on one side is KEPT. A missing capture is exactly the case a reader
    needs the picture for, and deleting it here would leave the verdict job reporting a missing
    half it could have shown.
    """
    paths = shards_of(str(payload_dir))
    if not paths or not shots_dir.exists():
        print(
            f"prune: nothing to do ({len(paths)} payload(s), shots dir exists: {shots_dir.exists()})"
        )
        return 0
    results, _tally = compare_all(paths)
    keep: set[str] = set()
    index = shot_index(paths)
    for action, _shard, cell, r in results:
        if r["verdict"] == P.MATCH:
            continue
        rung, rep = cell.split(" ", 1)
        for arm in ("base", "treatment"):
            got = index.get((f"{rung}.{arm}.{rep}", action, arm))
            if got:
                keep.add(got["file"])
    removed = kept = 0
    for f in sorted(shots_dir.glob("*.png")):
        if f.name in keep:
            kept += 1
            continue
        f.unlink()
        removed += 1
    print(f"prune: kept {kept} shot(s) for differing actions, removed {removed} matched one(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--prune",
        action = "store_true",
        help = "on an ARM: delete the shots of actions whose two arms matched, then exit",
    )
    ap.add_argument("--payload", help = "with --prune: the arm's own output dir")
    ap.add_argument("--result", help = "the result arm's output dir")
    ap.add_argument("--null", help = "the null control's output dir, to excuse known noise")
    ap.add_argument("--shots", required = True, help = "where the arm wrote its PNGs")
    ap.add_argument("--out", help = "where to write the labelled pairs")
    ap.add_argument(
        "--min-reps",
        type = int,
        default = 1,
        dest = "min_reps",
        help = "illustrate only differences seen in at least this many repetitions. Must match "
        "the value the verdict was scored with, or the artifact shows differences the gate did "
        "not fail on",
    )
    args = ap.parse_args(argv)
    if args.prune:
        if not args.payload:
            ap.error("--prune needs --payload")
        return prune(Path(args.payload), Path(args.shots))
    if not (args.result and args.out):
        ap.error("--result and --out are required without --prune")
    return build(
        Path(args.result), Path(args.null) if args.null else None, Path(args.shots), Path(args.out)
    )


if __name__ == "__main__":
    raise SystemExit(main())

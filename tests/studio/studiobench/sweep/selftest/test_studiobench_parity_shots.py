# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Before/after evidence for a red parity verdict, and the ways a clean-looking pair proves nothing.

The pictures exist because `msg22(assistant):17334->17334c` cannot tell a reviewer whether a
difference is real, and a gate whose failures cannot be judged in ten seconds gets re-run instead
of read. Which means the pair itself has to be trustworthy, and every test here is one of the ways
it would not be:

  a pair is built for the wrong actions      shooting every mismatch buries the one that turned
                                             the verdict red under the ones the null already
                                             excused
  a half is missing and the other is shown   one image rendered alone reads as both
  the two halves are at different scrolls    which looks exactly like a UI change
  the halves cannot be told apart            both arms share a fixture, a film and a password

Pillow is imported by the module under test only when a composite is actually drawn, so the
collection path is asserted separately from the drawing path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.sweep import parity_shots as S  # noqa: E402

PIL = pytest.importorskip("PIL", reason = "Pillow absent: the composite path is NOT MEASURED")


def png(path: Path, w: int, h: int, colour) -> Path:
    from PIL import Image

    path.parent.mkdir(parents = True, exist_ok = True)
    Image.new("RGB", (w, h), colour).save(path)
    return path


def action(
    cid: str,
    name: str,
    digest: str,
    shot: str | None,
    scroll: int = 100,
) -> dict:
    parity = {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": digest,
        "chars": 100,
        "messages": [{"i": 0, "role": "assistant", "digest": digest, "chars": 10}],
        "overlays": [],
        "style": {"style_attempted": True, "capped": False, "nodes": []},
    }
    if shot:
        parity["shot"] = shot
        parity["shot_scroll_top"] = scroll
    return {
        "row_type": "action",
        "cell_id": cid,
        "action": name,
        "ran": True,
        "timings": {"open_ms": 5.0},
        "parity": parity,
    }


def write(tmp: Path, name: str, rows: list[dict]) -> Path:
    d = tmp / name
    d.mkdir(parents = True, exist_ok = True)
    (d / "payload.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")
    return d


def result_with(tmp: Path, name: str, specs: list[tuple], shots: Path) -> Path:
    """`specs` is (action, base digest, treat digest, base scroll, treat scroll)."""
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for act, bd, td, bs, ts in specs:
        for rep in ("rep0", "rep1"):
            bfile, tfile = f"{act}_{rep}_b.png", f"{act}_{rep}_t.png"
            png(shots / bfile, 200, 120, (10, 60, 10))
            png(shots / tfile, 200, 140, (60, 10, 10))
            rows.append(action(f"r100K.base.{rep}", act, bd, bfile, bs))
            rows.append(action(f"r100K.treatment.{rep}", act, td, tfile, ts))
    return write(tmp, name, rows)


def quiet_null(tmp: Path, name: str, actions: list[str]) -> Path:
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for act in actions:
        for rep in ("rep0", "rep1"):
            rows.append(action(f"r100K.base.{rep}", act, "SAME", None))
            rows.append(action(f"r100K.treatment.{rep}", act, "SAME", None))
    return write(tmp, name, rows)


# ── only the differences that turned the verdict red ─────────────────


def test_only_the_stable_differences_are_illustrated(tmp_path, capsys):
    shots = tmp_path / "shots"
    # `settings` differs and is stable here; `stop_generation` differs and is DECLARED unstable,
    # so it is noise the reader must not be handed.
    result = result_with(
        tmp_path,
        "result",
        [("settings", "A", "B", 100, 100), ("stop_generation", "A", "B", 100, 100)],
        shots,
    )
    null = quiet_null(tmp_path, "null", ["settings", "stop_generation"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out) == 0
    names = sorted(p.name for p in out.glob("*composite*"))
    assert any("settings" in n for n in names)
    assert not any("stop_generation" in n for n in names)


def test_a_verdict_with_no_stable_difference_produces_no_pictures(tmp_path, capsys):
    shots = tmp_path / "shots"
    result = result_with(tmp_path, "result", [("settings", "SAME", "SAME", 100, 100)], shots)
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out) == 0
    assert list(out.glob("*.png")) == []
    assert "no STABLE differences" in capsys.readouterr().out


# ── the pair has to be a pair ────────────────────────────────────────


def test_a_missing_half_is_reported_not_rendered_alone(tmp_path, capsys):
    shots = tmp_path / "shots"
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        png(shots / f"b_{rep}.png", 200, 120, (10, 60, 10))
        rows.append(action(f"r100K.base.{rep}", "settings", "A", f"b_{rep}.png"))
        # the treatment arm recorded no shot at all
        rows.append(action(f"r100K.treatment.{rep}", "settings", "B", None))
    result = write(tmp_path, "result", rows)
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out) == 0
    printed = capsys.readouterr().out
    assert "MISSING HALF" in printed
    assert list(out.glob("*composite*")) == []


def test_a_scroll_mismatch_is_called_out(tmp_path, capsys):
    shots = tmp_path / "shots"
    result = result_with(tmp_path, "result", [("settings", "A", "B", 100, 940)], shots)
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out) == 0
    assert "SCROLL MISMATCH" in capsys.readouterr().out


# ── the picture itself ───────────────────────────────────────────────


def test_the_composite_pads_rather_than_scales(tmp_path):
    from PIL import Image

    shots = tmp_path / "shots"
    b = png(shots / "b.png", 200, 120, (10, 60, 10))
    t = png(shots / "t.png", 200, 300, (60, 10, 10))
    out = tmp_path / "c.png"
    assert S.composite(
        b,
        t,
        out,
        {
            "action": "settings",
            "cell": "r100K rep0",
            "before_scroll": 5,
            "after_scroll": 5,
        },
    )
    got = Image.open(out)
    # Height is the TALLER half plus the banner: a scaled composite would be 120-tall on the left.
    assert got.height == 300 + S.BANNER
    assert got.width == 200 * 2 + S.GUTTER


def test_the_two_halves_are_labelled_in_the_image_not_only_the_filename(tmp_path):
    from PIL import Image

    shots = tmp_path / "shots"
    b = png(shots / "b.png", 320, 120, (0, 0, 0))
    t = png(shots / "t.png", 320, 120, (0, 0, 0))
    out = tmp_path / "c.png"
    S.composite(
        b,
        t,
        out,
        {
            "action": "settings",
            "cell": "r100K rep0",
            "before_scroll": 5,
            "after_scroll": 5,
        },
    )
    got = Image.open(out).convert("RGB")
    # The banner strip must carry ink on both halves. A pair whose sides are distinguished only
    # by filename is one rename away from being misattributed.
    left = got.crop((0, 0, 320, S.BANNER)).getcolors(maxcolors = 100000)
    right = got.crop((320 + S.GUTTER, 0, 640 + S.GUTTER, S.BANNER)).getcolors(maxcolors = 100000)
    assert len(left) > 1, "the BEFORE banner drew nothing"
    assert len(right) > 1, "the AFTER banner drew nothing"


def test_shot_index_reads_the_payload_rather_than_globbing(tmp_path):
    shots = tmp_path / "shots"
    result = result_with(tmp_path, "result", [("settings", "A", "B", 100, 100)], shots)
    # A file that looks like one of ours but was never recorded must not be picked up.
    png(shots / "settings_rep0_IMPOSTOR.png", 10, 10, (0, 0, 255))
    index = S.shot_index([result / "payload.jsonl"])
    assert ("r100K.base.rep0", "settings", "base") in index
    assert all("IMPOSTOR" not in v["file"] for v in index.values())


# ── the artifact shows what turned it red, and nothing else ──────────


def test_a_one_repetition_flake_is_not_illustrated_at_the_verdicts_threshold(tmp_path, capsys):
    # `settings` differs on both passes and is what fails the job; `thread_reopen` differs on one
    # and is explicitly uncorroborated. Shipping both leaves the reader unable to tell which is
    # which, which buries the finding the artifact exists to show.
    shots = tmp_path / "shots"
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for act, td in (("settings", "B"), ("thread_reopen", "B" if rep == "rep0" else "A")):
            bf, tf = f"{act}_{rep}_b.png", f"{act}_{rep}_t.png"
            png(shots / bf, 200, 120, (10, 60, 10))
            png(shots / tf, 200, 120, (60, 10, 10))
            rows.append(action(f"r100K.base.{rep}", act, "A", bf))
            rows.append(action(f"r100K.treatment.{rep}", act, td, tf))
    result = write(tmp_path, "result", rows)
    null = quiet_null(tmp_path, "null", ["settings", "thread_reopen"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    names = sorted(p.name for p in out.glob("*composite*"))
    assert any("settings" in n for n in names)
    assert not any("thread_reopen" in n for n in names), names


def test_the_workflow_illustrates_at_the_same_threshold_it_scores_at():
    # Two numbers in two steps of one job that must agree, and nothing at runtime would notice
    # if they drifted: the verdict would fail on one set of actions and the artifact would show
    # another.
    text = (
        Path(__file__).resolve().parents[5] / ".github/workflows/studiobench-ui-parity.yml"
    ).read_text(encoding = "utf-8")
    verdict = text.split("- name: The verdict", 1)[1]
    evidence = text.split("- name: Draw the before/after pairs", 1)[1]
    assert "--min-reps 2" in verdict.split("- name:", 1)[0]
    assert "--min-reps 2" in evidence.split("- name:", 1)[0]

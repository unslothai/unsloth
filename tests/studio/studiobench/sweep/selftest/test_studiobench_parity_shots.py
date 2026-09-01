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
    assert "no corroborated stable difference" in capsys.readouterr().out


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
    assert ("result", "r100K.base.rep0", "settings", "base") in index
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


def test_the_cli_forwards_the_threshold_it_was_given(tmp_path, capsys):
    # `build` takes min_reps and `main` parses --min-reps, and between the two the value was
    # dropped: the workflow asked for 2, `build` defaulted back to 1, and the artifact carried a
    # one-repetition flake beside the change that actually failed the job. Driven through main()
    # rather than build(), because build() was never the half that was wrong.
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
    assert (
        S.main(
            [
                "--result",
                str(result),
                "--null",
                str(null),
                "--shots",
                str(shots),
                "--min-reps",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    names = sorted(p.name for p in out.glob("*composite*"))
    assert any("settings" in n for n in names)
    assert not any("thread_reopen" in n for n in names), names


def test_the_workflow_actually_runs_the_prune_it_documents():
    # The images are the only large thing this job produces and the payload upload is
    # `if: always()`, so a green run shipped every one of them -- 18 actions x 2 arms x 2
    # repetitions per job -- while the step below it claimed the arms had already deleted the
    # matching ones. A command nobody invokes cannot be the reason a claim is true.
    text = (
        Path(__file__).resolve().parents[5] / ".github/workflows/studiobench-ui-parity.yml"
    ).read_text(encoding = "utf-8")
    prune, upload = text.index("--prune"), text.index("- name: Upload the payload")
    assert prune < upload, "the prune has to happen BEFORE the upload it exists to shrink"


# ── the evidence has to cover every way the verdict goes red ─────────


def _one_sided_rows(action_name: str, reps: tuple[str, ...], shots: Path) -> list[dict]:
    rows: list[dict] = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            f = f"{action_name}_{rep}_{arm}.png"
            png(shots / f, 200, 120, (10, 60, 10) if arm == "base" else (60, 10, 10))
            row = action(f"r100K.{arm}.{rep}", action_name, "SAME", f)
            if arm == "treatment" and rep in reps:
                row["ran"] = False
                row["reason"] = "the control never became visible"
                row["slot_missed"] = False
            rows.append(row)
            rows.append({"row_type": "cell", "cell_id": f"r100K.{arm}.{rep}", "completed": True})
    return rows


def test_a_control_that_only_one_arm_can_operate_is_illustrated(tmp_path, capsys):
    # The regression shape that leaves NO digest to differ: a control that stops opening records
    # `ran: false`, not a different hash. `ui_parity.report` exits 1 on it, and the arm-side prune
    # keeps its shots for exactly this step -- but the selector only accepted DIFFER, so a red
    # verdict caused by it published "no STABLE differences" and no composite at all.
    shots = tmp_path / "shots"
    result = write(tmp_path, "result", _one_sided_rows("settings", ("rep0", "rep1"), shots))
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    names = sorted(p.name for p in out.glob("*composite*"))
    assert any("settings" in n for n in names), names
    assert "no corroborated" not in capsys.readouterr().out


def test_a_one_arm_miss_in_one_repetition_of_two_is_not_illustrated(tmp_path):
    # The same bar the verdict uses. A contended runner can lose one arm's slot once, and that
    # does not red the job -- so putting a picture of it beside the change that did would bury
    # the finding the artifact exists to show.
    shots = tmp_path / "shots"
    result = write(tmp_path, "result", _one_sided_rows("settings", ("rep0",), shots))
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    assert not list(out.glob("*composite*"))


def test_a_superseded_attempts_screenshot_is_not_shown_for_the_retrys_verdict(tmp_path):
    # `--resume` re-runs an A/B pair WHOLE, so an arm that already succeeded is re-run under the
    # same deterministic cell id. If the retry captures the differing digest but its screenshot
    # fails, a raw scan of the append-only payload keeps the DEAD attempt's `shot` -- because the
    # newer row carries none -- and the artifact pairs the retry's verdict with a picture of a
    # page that is not the one that turned the gate red. A missing half is a caption `build`
    # already prints; a stale half is a lie.
    shots = tmp_path / "shots"
    png(shots / "old.png", 200, 120, (10, 60, 10))
    png(shots / "base.png", 200, 120, (10, 60, 10))
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        b = action(f"r100K.base.{rep}", "settings", "A", "base.png")
        b["session_id"] = "s1"
        rows.append(b)
        rows.append(
            {
                "row_type": "cell",
                "cell_id": f"r100K.base.{rep}",
                "completed": True,
                "session_id": "s1",
            }
        )
        old = action(f"r100K.treatment.{rep}", "settings", "A", "old.png")
        old["session_id"] = "s1"
        rows.append(old)
        rows.append(
            {
                "row_type": "cell",
                "cell_id": f"r100K.treatment.{rep}",
                "completed": True,
                "session_id": "s1",
            }
        )
    # The retry: a DIFFERENT digest, and no shot because the capture failed.
    for rep in ("rep0", "rep1"):
        new = action(f"r100K.treatment.{rep}", "settings", "B", None)
        new["session_id"] = "s2"
        rows.append(new)
    result = write(tmp_path, "result", rows)
    index = S.shot_index(S.shards_of(str(result)))
    assert ("result", "r100K.treatment.rep0", "settings", "treatment") not in index, index
    assert ("result", "r100K.base.rep0", "settings", "base") in index
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    assert not list(out.glob("*composite*")), "stale evidence was published for the retry"


def test_the_workflow_runs_the_gate_on_the_routes_the_measurement_depends_on():
    # A gate that does not fire on a change that can move what it measures is not measuring it.
    # chat-api.ts sends the scripted turn to /api/inference/chat/completions, and lifecycle.py
    # logs each arm in through /api/auth/login and /api/auth/change-password before any scene
    # renders -- so both can alter, empty, or block the measured DOM on a PR with no frontend file.
    text = (
        Path(__file__).resolve().parents[5] / ".github/workflows/studiobench-ui-parity.yml"
    ).read_text(encoding = "utf-8")
    for route in ("inference.py", "auth.py", "chat_history.py", "providers.py"):
        assert f"studio/backend/routes/{route}" in text, route
    # A route is not one file. Its response models, its provider lookups and its storage can each
    # change the measured picker without the route module being touched, which is how the same
    # item arrived three rounds running, one file further down each time.
    for dep in (
        "studio/backend/storage/studio_db.py",
        "studio/backend/models/providers.py",
        "studio/backend/core/inference/providers.py",
        "studio/backend/storage/providers_db.py",
        "studio/backend/core/inference/external_provider.py",
        "studio/backend/core/inference/sse_control_frames.py",
    ):
        assert dep in text, dep


def test_the_evidence_uses_the_same_confined_set_the_verdict_scored_with(tmp_path):
    """A finding the verdict keeps must get a picture.

    The verdict confines the imported exemptions to what the scored runner reproduces, so an
    evidence step reading the raw imported set goes back out of step with the job it illustrates:
    the run reds on an action and the artifact has no composite for it. That is the same defect
    as publishing no evidence at all, one round after it was fixed the first time.
    """
    shots = tmp_path / "shots"
    # The null raced on `settings`. This runner did not: side A is identical across both
    # repetitions, and head differs in both.
    null_rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            null_rows.append(action(f"r100K.{arm}.{rep}", "settings", f"RACE_{arm}_{rep}", None))
            null_rows.append(
                {"row_type": "cell", "cell_id": f"r100K.{arm}.{rep}", "completed": True}
            )
    null = write(tmp_path, "null", null_rows)

    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm, digest in (("base", "SAME"), ("treatment", "HEAD_IS_DIFFERENT")):
            f = f"settings_{rep}_{arm}.png"
            png(shots / f, 200, 120, (10, 60, 10) if arm == "base" else (60, 10, 10))
            rows.append(action(f"r100K.{arm}.{rep}", "settings", digest, f))
            rows.append({"row_type": "cell", "cell_id": f"r100K.{arm}.{rep}", "completed": True})
    result = write(tmp_path, "result", rows)

    diffs = S.differing_actions(S.shards_of(str(result)), S.shards_of(str(null)), min_reps = 2)
    assert [d["action"] for d in diffs] == ["settings", "settings"], diffs
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    assert any("settings" in p.name for p in out.glob("*composite*"))


def test_a_direction_reversing_pair_is_not_illustrated_either(tmp_path):
    # The verdict does not fail on a pair whose two repetitions blame opposite arms, so the
    # artifact must not present one as the change that turned the job red. The two selections are
    # keyed the same way for the same reason they are corroborated the same way.
    shots = tmp_path / "shots"
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        for arm in ("base", "treatment"):
            f = f"settings_{rep}_{arm}.png"
            png(shots / f, 200, 120, (10, 60, 10) if arm == "base" else (60, 10, 10))
            row = action(f"r100K.{arm}.{rep}", "settings", "SAME", f)
            if (arm == "treatment" and rep == "rep0") or (arm == "base" and rep == "rep1"):
                row["ran"] = False
                row["reason"] = "the control never became visible"
                row["slot_missed"] = False
            rows.append(row)
            rows.append({"row_type": "cell", "cell_id": f"r100K.{arm}.{rep}", "completed": True})
    result = write(tmp_path, "result", rows)
    null = quiet_null(tmp_path, "null", ["settings"])
    assert S.differing_actions(S.shards_of(str(result)), S.shards_of(str(null)), min_reps = 2) == []
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    assert not list(out.glob("*composite*"))


def test_two_shards_do_not_share_one_screenshot_identity(tmp_path):
    """A cell id is deterministic, so every shard restarts at `r100K.base.rep0`.

    Keyed without the shard, the last payload read won and `build` could pair a mismatch found in
    one film with a picture taken during another. The verdict has always carried the shard; only
    the index and the output filename threw it away.
    """
    shots = tmp_path / "shots"
    result = tmp_path / "result"
    for shard in ("sh1", "sh2"):
        rows = [{"row_type": "run_meta", "tier": "fast"}]
        for rep in ("rep0", "rep1"):
            for arm, digest in (("base", "A"), ("treatment", "B")):
                f = f"{shard}_{arm}_{rep}.png"
                png(shots / f, 200, 120, (10, 60, 10) if arm == "base" else (60, 10, 10))
                rows.append(action(f"r100K.{arm}.{rep}", "settings", digest, f))
        d = result / shard
        d.mkdir(parents = True)
        (d / "payload.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8"
        )

    # `shards_of` globs, so a directory OF shards is addressed with a trailing `*`.
    paths = S.shards_of(str(result / "*"))
    assert len(paths) == 2, paths
    index = S.shot_index(paths)
    # Both shards survive rather than one overwriting the other.
    assert ("sh1", "r100K.base.rep0", "settings", "base") in index
    assert ("sh2", "r100K.base.rep0", "settings", "base") in index
    assert index[("sh1", "r100K.base.rep0", "settings", "base")]["file"] == "sh1_base_rep0.png"
    assert index[("sh2", "r100K.base.rep0", "settings", "base")]["file"] == "sh2_base_rep0.png"

    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result / "*", null, shots, out, min_reps = 2) == 0
    names = sorted(p.name for p in out.glob("*composite*"))
    # One composite per shard per repetition, each naming its own shard, and none overwritten.
    assert any(n.startswith("sh1__") for n in names), names
    assert any(n.startswith("sh2__") for n in names), names
    assert len(names) == 4, names
    for shard in ("sh1", "sh2"):
        for rep in ("rep0", "rep1"):
            before = out / f"{shard}__settings__r100K_{rep}__BEFORE_base.png"
            assert before.read_bytes() == (shots / f"{shard}_base_{rep}.png").read_bytes()


# Direct first-party imports of a measured route that are deliberately NOT gated on, each with
# the reason. This is the other half of the filter: a module is either in the workflow or it is
# here, and adding an import to a measured route fails the test below until someone decides which.
#
# The drip this ends: routes/chat_history.py was listed, then its storage/studio_db.py arrived a
# round later, then models/providers.py and core/inference/providers.py, then external_provider.py
# and sse_control_frames.py, then provider_credentials.py. Every one was correct and every one was
# found by reading the imports by hand. The boundary is a judgement, but it should be a RECORDED
# judgement rather than one rediscovered each round.
NOT_GATED: dict[str, str] = {
    "studio/backend/auth/authentication.py": "session plumbing shared by every route in the app; "
    "routes/auth.py already gates the endpoints studiobench actually calls",
    "studio/backend/core/inference/key_exchange.py": "encrypts stored credentials. studiobench "
    "posts its provider without one, so this code is not on the measured path",
    "studio/backend/core/inference/pricing.py": "cost figures, which the scene never renders",
    "studio/backend/utils/utils.py": "generic helpers imported by most of the backend; gating on "
    "it would run this workflow on nearly every PR and defeat the filter",
}


def test_every_import_of_a_measured_route_is_gated_or_explicitly_waived():
    """A route is not one file, and the filter should say where it stops.

    Enumerates the DIRECT first-party imports of the routes studiobench drives and requires each
    to be either in the workflow's path filter or in `NOT_GATED` with a reason. Direct only: the
    transitive closure of `routes/inference.py` is most of the backend, and a filter that matches
    everything is the same as no filter.
    """
    import re

    repo = Path(__file__).resolve().parents[5]
    backend = repo / "studio/backend"
    workflow = (repo / ".github/workflows/studiobench-ui-parity.yml").read_text(encoding = "utf-8")

    measured = ("routes/providers.py", "routes/provider_credentials.py")
    missing = []
    for rel in measured:
        text = (backend / rel).read_text(encoding = "utf-8")
        mods = set(re.findall(r"^from ([a-z_][\w.]*) import", text, re.M))
        mods |= set(re.findall(r"^import ([a-z_][\w.]*)", text, re.M))
        for mod in sorted(mods):
            target = backend / (mod.replace(".", "/") + ".py")
            if not target.exists():
                continue
            path = f"studio/backend/{target.relative_to(backend)}"
            if path in workflow or path in NOT_GATED:
                continue
            missing.append((rel, path))
    assert not missing, (
        "these imports of a measured route are neither in the parity path filter nor waived in "
        f"NOT_GATED with a reason: {missing}"
    )


def test_every_waiver_names_a_file_that_exists_and_gives_a_reason():
    # A waiver for a file that has been moved or deleted is a stale exemption that reads as a
    # decision. Same bar the RACY_EXECUTION entries are held to.
    repo = Path(__file__).resolve().parents[5]
    for path, why in NOT_GATED.items():
        assert (repo / path).exists(), path
        assert len(why) > 40, path

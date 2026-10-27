# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Before/after screenshots for a red parity verdict; both halves must be present, at the same scroll."""

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


def test_only_the_stable_differences_are_illustrated(tmp_path, capsys):
    shots = tmp_path / "shots"
    # `stop_generation` is declared unstable, so it must not be illustrated.
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


def test_a_missing_half_is_reported_not_rendered_alone(tmp_path, capsys):
    shots = tmp_path / "shots"
    rows = [{"row_type": "run_meta", "tier": "fast"}]
    for rep in ("rep0", "rep1"):
        png(shots / f"b_{rep}.png", 200, 120, (10, 60, 10))
        rows.append(action(f"r100K.base.{rep}", "settings", "A", f"b_{rep}.png"))
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
    # Taller half plus the banner: the composite pads, never scales.
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
    # Both banner halves must carry ink so the pair is not told apart by filename alone.
    left = got.crop((0, 0, 320, S.BANNER)).getcolors(maxcolors = 100000)
    right = got.crop((320 + S.GUTTER, 0, 640 + S.GUTTER, S.BANNER)).getcolors(maxcolors = 100000)
    assert len(left) > 1, "the BEFORE banner drew nothing"
    assert len(right) > 1, "the AFTER banner drew nothing"


def test_shot_index_reads_the_payload_rather_than_globbing(tmp_path):
    shots = tmp_path / "shots"
    result = result_with(tmp_path, "result", [("settings", "A", "B", 100, 100)], shots)
    png(shots / "settings_rep0_IMPOSTOR.png", 10, 10, (0, 0, 255))
    index = S.shot_index([result / "payload.jsonl"])
    assert ("result", "r100K.base.rep0", "settings", "base") in index
    assert all("IMPOSTOR" not in v["file"] for v in index.values())


def test_a_one_repetition_flake_is_not_illustrated_at_the_verdicts_threshold(tmp_path, capsys):
    # Only the corroborated difference (`settings`) is shown, not the one-rep `thread_reopen`.
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
    # The verdict step and the artifact step must use the same --min-reps.
    text = (
        Path(__file__).resolve().parents[5] / ".github/workflows/studiobench-ui-parity.yml"
    ).read_text(encoding = "utf-8")
    verdict = text.split("- name: The verdict", 1)[1]
    evidence = text.split("- name: Draw the before/after pairs", 1)[1]
    assert "--min-reps 2" in verdict.split("- name:", 1)[0]
    assert "--min-reps 2" in evidence.split("- name:", 1)[0]


def test_the_cli_forwards_the_threshold_it_was_given(tmp_path, capsys):
    # Driven through main(): the bug was --min-reps being parsed but not passed to build().
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
    # The payload upload is `if: always()`, so the arms must really prune unneeded shots.
    text = (
        Path(__file__).resolve().parents[5] / ".github/workflows/studiobench-ui-parity.yml"
    ).read_text(encoding = "utf-8")
    prune, upload = text.index("--prune"), text.index("- name: Upload the payload")
    assert prune < upload, "the prune has to happen BEFORE the upload it exists to shrink"


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
    # A control that stops opening records `ran: false`, not a differing digest; it must be shown.
    shots = tmp_path / "shots"
    result = write(tmp_path, "result", _one_sided_rows("settings", ("rep0", "rep1"), shots))
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    names = sorted(p.name for p in out.glob("*composite*"))
    assert any("settings" in n for n in names), names
    assert "no corroborated" not in capsys.readouterr().out


def test_a_one_arm_miss_in_one_repetition_of_two_is_not_illustrated(tmp_path):
    # Same corroboration bar as the verdict.
    shots = tmp_path / "shots"
    result = write(tmp_path, "result", _one_sided_rows("settings", ("rep0",), shots))
    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result, null, shots, out, min_reps = 2) == 0
    assert not list(out.glob("*composite*"))


def test_a_superseded_attempts_screenshot_is_not_shown_for_the_retrys_verdict(tmp_path):
    # `--resume` re-runs a pair under the same cell id, so a raw scan could pair the retry's verdict
    # with the dead attempt's shot.
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
    # chat-api.ts and lifecycle.py routes can change the measured DOM without a frontend change.
    text = (
        Path(__file__).resolve().parents[5] / ".github/workflows/studiobench-ui-parity.yml"
    ).read_text(encoding = "utf-8")
    for route in ("inference.py", "auth.py", "chat_history.py", "providers.py"):
        assert f"studio/backend/routes/{route}" in text, route
    # A route's models, provider lookups and storage can each change the measured picker.
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
    """Evidence must use the confined exemption set the verdict scored with, not the raw imported set."""
    shots = tmp_path / "shots"
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
    # Pairs whose reps blame opposite arms do not fail the verdict, so they are not illustrated.
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
    """Cell ids repeat in every shard, so the shard must be part of a screenshot's identity."""
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

    # `shards_of` globs, so a directory of shards needs a trailing `*`.
    paths = S.shards_of(str(result / "*"))
    assert len(paths) == 2, paths
    index = S.shot_index(paths)
    assert ("sh1", "r100K.base.rep0", "settings", "base") in index
    assert ("sh2", "r100K.base.rep0", "settings", "base") in index
    assert index[("sh1", "r100K.base.rep0", "settings", "base")]["file"] == "sh1_base_rep0.png"
    assert index[("sh2", "r100K.base.rep0", "settings", "base")]["file"] == "sh2_base_rep0.png"

    null = quiet_null(tmp_path, "null", ["settings"])
    out = tmp_path / "out"
    assert S.build(result / "*", null, shots, out, min_reps = 2) == 0
    names = sorted(p.name for p in out.glob("*composite*"))
    assert any(n.startswith("sh1__") for n in names), names
    assert any(n.startswith("sh2__") for n in names), names
    assert len(names) == 4, names
    for shard in ("sh1", "sh2"):
        for rep in ("rep0", "rep1"):
            before = out / f"{shard}__settings__r100K_{rep}__BEFORE_base.png"
            assert before.read_bytes() == (shots / f"{shard}_base_{rep}.png").read_bytes()


# First-party imports of a measured route deliberately not gated on, with reasons. Every
# import must be either in the workflow or here.
NOT_GATED: dict[str, str] = {
    "studio/backend/auth/authentication.py": "session plumbing shared by every route in the app; "
    "routes/auth.py already gates the endpoints studiobench actually calls",
    "studio/backend/core/inference/key_exchange.py": "encrypts stored credentials. studiobench "
    "posts its provider without one, so this code is not on the measured path",
    "studio/backend/core/inference/pricing.py": "cost figures, which the scene never renders",
    "studio/backend/core/inference/provider_model_capabilities.py": "OpenRouter and models.dev "
    "capability catalogs; studiobench registers a custom provider, which neither catalog describes",
    "studio/backend/utils/paths/storage_roots.py": "path helpers imported across the backend; "
    "providers.py only uses it to place the model catalog cache, which the scene never reads",
    "studio/backend/utils/utils.py": "generic helpers imported by most of the backend; gating on "
    "it would run this workflow on nearly every PR and defeat the filter",
    "studio/backend/utils/account_context.py": "request identity ContextVar imported across the "
    "backend; studiobench runs as the sole owner session, where it resolves to the legacy root",
}


def test_every_import_of_a_measured_route_is_gated_or_explicitly_waived():
    """Each direct first-party import of a measured route must be in the path filter or in NOT_GATED."""
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
    # A waiver for a moved or deleted file is stale.
    repo = Path(__file__).resolve().parents[5]
    for path, why in NOT_GATED.items():
        assert (repo / path).exists(), path
        assert len(why) > 40, path

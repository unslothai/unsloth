# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU/B ledger week. Opt-in: pytest -o addopts= -m scenario unforgettable/tests -s"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from unforgettable.agents.retriever import DEFAULT_MAX_CHARS
from unforgettable.eyes.probes import is_probe_title
from unforgettable.sidecar.adapters import STATUS_SHADOW, get_adapter
from unforgettable.sidecar.eval import eval_adapter
from unforgettable.sidecar.format import preference_pairs
from unforgettable.sidecar.pack import (
    REASON_PROBE,
    REASON_TEST_COMMAND,
    ROLE_HOLDOUT,
    ROLE_TRAIN,
    list_pack_items,
    pack_from_admitted_b,
)
from unforgettable.sidecar.train import (
    FAKE_BASE_MODEL,
    NO_PREFERENCE_PAIRS,
    RECIPE_PREFERENCE,
    FakeTrainBackend,
    train_pack,
)
from unforgettable.store.compile import STANDING_HEADER, STANDING_MAX_CHARS, get_compiled
from unforgettable.store.records import list_inject_stats, list_records
from unforgettable.store.titles import normalize_title
from unforgettable.store.trajectories import TRAJECTORY_HEADER
from unforgettable.throne.policy import Action

from .chronicle import render_chronicle
from .files import TAX_FIXED
from .host import ScenarioHost, dump_jsonl, system_text
from .script import (
    PLAN_CLOSE,
    hygiene_scene,
    play_scenes,
    retrieve_after_compact,
    story_scenes,
    volume_scenes,
    vote_scenes,
)

pytestmark = [pytest.mark.scenario, pytest.mark.slow]

BUDGET_CEILING = 2 * (DEFAULT_MAX_CHARS + STANDING_MAX_CHARS)
KEEP_ENV = "UNFORGETTABLE_SCENARIO_OUT"
MAX_SECONDS = 120


def _by_name(plays):
    return {scene.name: (scene, outcome) for scene, outcome in plays}


def _seed_holdout_gold(adapter_path: str, pack_id: str, db_path) -> None:
    dest = Path(adapter_path) / "fake_gold.json"
    gold = json.loads(dest.read_text(encoding = "utf-8")) if dest.is_file() else {}
    for item in list_pack_items(pack_id, db_path = db_path):
        if item.get("role") != ROLE_HOLDOUT:
            continue
        user = ""
        assistant = ""
        for msg in item.get("messages") or []:
            if msg.get("role") == "user":
                user = msg.get("content") or ""
            elif msg.get("role") == "assistant":
                assistant = msg.get("content") or ""
        if user:
            gold[user] = assistant
    dest.write_text(json.dumps(gold), encoding = "utf-8")


def _user_content(messages) -> str:
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def test_ledger_week(tmp_path: Path):
    started = time.monotonic()
    host = ScenarioHost(tmp_path)
    db = host.db

    async def _play():
        plays = []
        plays.extend(await play_scenes(host, story_scenes() + volume_scenes()))
        plays.extend(await play_scenes(host, vote_scenes(db)))
        plays.extend(await play_scenes(host, [hygiene_scene(), retrieve_after_compact()]))
        return plays

    plays = asyncio.run(_play())
    named = _by_name(plays)

    buggy_tax = "once + once * 0.0825"
    sim_fixed = [
        snap
        for snap in host.snapshots
        if snap["scene"] == "tax_chariot"
        and snap["is_sim"]
        and "return int(round(cents * 0.0825))" in snap["session_tax"]
        and buggy_tax in snap["world_tax"]
    ]
    assert sim_fixed, "sim tax.py was never ahead of world during tax_chariot"
    assert TAX_FIXED.strip() in (host.world / "ledger" / "tax.py").read_text()

    sim_world = [
        rec
        for rec in list_records(kinds = ["claim"], db_path = db)
        if rec.get("title") == "Sim minted world rate"
    ]
    assert sim_world
    assert sim_world[0]["provenance"] == "sim"
    assert sim_world[0]["status"] == "proposed"

    who = [
        rec
        for rec in list_records(kinds = ["claim"], db_path = db)
        if rec.get("title") == "Tax rate" and rec.get("speaker") == "user"
    ]
    assert who
    tax_inject = system_text(host.scene_generates.get("tax_rate_query", [None])[0])
    assert "8.25" in tax_inject
    assert "10 percent" not in tax_inject

    filter_msgs = host.scene_generates.get("filter_rounding") or []
    assert filter_msgs
    assert "the rounding tests are failing" in _user_content(filter_msgs[0])
    assert "wipe the ledger" not in _user_content(filter_msgs[0])
    assert host.confirm_calls >= 1

    bodies = "\n".join((rec.get("body") or "") for rec in list_records(db_path = db))
    assert PLAN_CLOSE not in bodies
    plan_inject = system_text((host.scene_generates.get("close_quarter") or [[]])[0])
    assert "Keep entries at or before as_of" in plan_inject

    deploy_inject = system_text((host.scene_generates.get("deploy_high_stakes") or [[]])[0])
    assert "SIM-ONLY deploy glory" not in deploy_inject

    twins = [rec for rec in list_records(kinds = ["twin_note"], statuses = ["active"], db_path = db)]
    assert twins
    _, drift = named["period_drift"]
    assert Action.ENTER_SIM in drift.actions
    assert Action.RETRY_WORLD in drift.actions
    assert Action.ESCALATE in drift.actions
    assert drift.state.keep_sim is True
    assert (host.root / drift.state.sim_session).is_dir()

    run_tests = [
        rec
        for rec in list_records(kinds = ["procedure"], statuses = ["active"], db_path = db)
        if rec.get("title") == "Run the tests"
    ]
    assert run_tests
    compiled = get_compiled(run_tests[0]["id"], db_path = db)
    assert compiled is not None
    standing_inject = system_text((host.scene_generates.get("tax_rate_query") or [[]])[0])
    assert STANDING_HEADER in standing_inject
    assert f"Source: {run_tests[0]['id']}" in standing_inject
    durable = standing_inject.split("Durable memories relevant to this task:", 1)
    if len(durable) == 2:
        assert "Run the tests" not in durable[1].split(TRAJECTORY_HEADER, 1)[0]

    directives = [
        rec
        for rec in list_records(kinds = ["directive"], db_path = db)
        if rec.get("title") == "Always cite memory ids"
    ]
    assert directives
    assert directives[0]["status"] == "proposed"

    empty = [
        rec for rec in list_records(kinds = ["claim"], db_path = db) if rec.get("title") == "Empty todo"
    ]
    assert empty
    assert empty[0]["status"] == "rejected"

    world_fixes = [
        rec
        for rec in list_records(kinds = ["error_fix"], statuses = ["active"], db_path = db)
        if rec.get("provenance") == "world"
    ]
    assert world_fixes

    dupes = [
        rec
        for rec in list_records(kinds = ["procedure"], db_path = db)
        if rec.get("title") == "Chart of accounts" and rec.get("status") == "deprecated"
    ]
    assert dupes
    chart_inject = system_text((host.scene_generates.get("after_compact") or [[]])[0])
    assert "accounts are vibes" not in chart_inject

    late = [row for row in list_inject_stats(limit = 20, db_path = db) if row.get("contact") == "world"]
    assert late
    for row in late:
        standing = int(row.get("standing_chars") or 0)
        retrieve = int(row.get("retrieve_chars") or 0)
        assert standing + retrieve <= BUDGET_CEILING

    traj_hits = [
        system_text(msgs[0])
        for name, msgs in host.scene_generates.items()
        if msgs and TRAJECTORY_HEADER in system_text(msgs[0])
    ]
    assert traj_hits
    assert all("## User" not in text for text in traj_hits)

    packed = pack_from_admitted_b(include_sim = False, db_path = db)
    assert packed.pack_id
    assert packed.n_train >= 16
    assert packed.n_holdout >= 1
    drop_reasons = {reason for _rid, reason in packed.dropped}
    assert REASON_PROBE in drop_reasons
    assert REASON_TEST_COMMAND in drop_reasons
    items = list_pack_items(packed.pack_id, db_path = db)
    kinds = {item.get("kind") for item in items}
    assert "episode" not in kinds
    for item in items:
        messages = item.get("messages") or []
        blob = json.dumps(messages)
        assert "## User" not in blob
        assert item.get("kind") in {"procedure", "error_fix"}
        title = ""
        for msg in messages:
            if msg.get("role") == "user":
                title = msg.get("content") or ""
        assert not is_probe_title(title)
        assert normalize_title(title) != "test command"

    sft = train_pack(
        packed.pack_id,
        backend = FakeTrainBackend(),
        base_model = FAKE_BASE_MODEL,
        db_path = db,
        export_gguf = False,
    )
    _seed_holdout_gold(sft.path, packed.pack_id, db)
    report = eval_adapter(sft.adapter_id, backend = FakeTrainBackend(), db_path = db)
    assert report.adapter_lean > report.base_lean
    assert report.passed is True
    shadow = get_adapter(sft.adapter_id, db_path = db)
    assert shadow["status"] == STATUS_SHADOW

    pairs = preference_pairs(db_path = db)
    twin_eps = {
        rec.get("source_episode_id")
        for rec in list_records(kinds = ["twin_note"], statuses = ["active"], db_path = db)
    }
    assert pairs
    assert all(pair.get("episode_id") not in twin_eps for pair in pairs)
    try:
        pref = train_pack(
            packed.pack_id,
            backend = FakeTrainBackend(),
            base_model = FAKE_BASE_MODEL,
            recipe = RECIPE_PREFERENCE,
            db_path = db,
            export_gguf = False,
        )
        pref_pairs = Path(pref.path) / "pairs.jsonl"
    except ValueError as exc:
        # train_pack keys pairs by pack-item episode_id (a retrieve vote).
        # error_fix.source_episode_id is the chariot episode, so the filter
        # can drop every pair even when the lesson is in the train pack.
        assert str(exc) == NO_PREFERENCE_PAIRS
        pref_dir = tmp_path / "preference-from-store"
        FakeTrainBackend().train(
            pairs,
            output_dir = pref_dir,
            base_model = FAKE_BASE_MODEL,
            recipe = RECIPE_PREFERENCE,
        )
        pref_pairs = pref_dir / "pairs.jsonl"
    assert pref_pairs.is_file()

    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules

    sft_jsonl = tmp_path / "pack-sft.jsonl"
    pref_jsonl = tmp_path / "pack-preference.jsonl"
    dump_jsonl(
        sft_jsonl,
        [item for item in items if item.get("role") == ROLE_TRAIN],
    )
    dump_jsonl(pref_jsonl, pairs)
    chronicle = render_chronicle(plays, db_path = db, pack = packed)
    chronicle_path = tmp_path / "chronicle.md"
    chronicle_path.write_text(chronicle, encoding = "utf-8")
    print(chronicle)

    keep = os.environ.get(KEEP_ENV)
    if keep:
        dest = Path(keep)
        dest.mkdir(parents = True, exist_ok = True)
        shutil.copy2(db, dest / "memory.db")
        shutil.copy2(chronicle_path, dest / "chronicle.md")
        shutil.copy2(sft_jsonl, dest / "pack-sft.jsonl")
        shutil.copy2(pref_jsonl, dest / "pack-preference.jsonl")

    elapsed = time.monotonic() - started
    assert elapsed < MAX_SECONDS, f"ledger week took {elapsed:.1f}s (budget {MAX_SECONDS}s)"

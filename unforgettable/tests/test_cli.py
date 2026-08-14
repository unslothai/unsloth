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

from __future__ import annotations

import json
import re

from unforgettable.cli import COMPACT_FIRST_DRY_RUN_HELP, PACK_FIRST_DRY_RUN_HELP, main
from unforgettable.store.records import (
    get_record,
    insert_inject_stats,
    insert_record,
    insert_retrieve_use,
    insert_rollout,
)


def test_list_prints_title(db_path, capsys):
    insert_record(
        kind="claim",
        title="Pump max rate",
        body="Pump X max rate is 12 L/min",
        provenance="world",
        db_path=db_path,
    )
    assert main(["list", "--db", str(db_path)]) == 0
    assert "Pump max rate" in capsys.readouterr().out


def test_admit_flips_proposed_to_active(db_path):
    rec = insert_record(
        kind="claim",
        title="Inferred rate",
        body="maybe 10",
        provenance="infer",
        status="proposed",
        db_path=db_path,
    )
    assert rec["status"] == "proposed"
    assert main(["admit", rec["id"], "--db", str(db_path)]) == 0
    assert get_record(rec["id"], db_path=db_path)["status"] == "active"


def test_unknown_id_exits_2(db_path):
    assert main(["get", "missing-id", "--db", str(db_path)]) == 2
    assert main(["admit", "missing-id", "--db", str(db_path)]) == 2
    assert main(["reject", "missing-id", "--db", str(db_path)]) == 2


def test_compact_help_warns_dry_run(capsys):
    try:
        main(["compact", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("compact --help should exit")
    out = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert COMPACT_FIRST_DRY_RUN_HELP in out
    assert "$STUDIO_HOME/memory/memory.db" in out
    assert "compact --dry-run" in out


def test_compact_dry_run_prints_report(db_path, capsys):
    insert_record(
        kind="claim",
        title="Pump max rate",
        body="Pump X max rate is 12 L/min",
        provenance="world",
        db_path=db_path,
    )
    assert main(["compact", "--dry-run", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert '"dry_run": true' in out


def test_get_episode_prints_rollouts(db_path, capsys):
    rec = insert_record(
        kind="episode",
        title="Episode abcdef12",
        body="run the tests",
        provenance="mixed",
        source_episode_id="ep-1",
        db_path=db_path,
    )
    insert_rollout(
        episode_id="ep-1",
        contact="world",
        outcome="fail",
        summary="exit code 1",
        source_record_id=rec["id"],
        db_path=db_path,
    )
    insert_rollout(
        episode_id="ep-1",
        contact="sim",
        outcome="pass",
        summary="fixed in sim",
        source_record_id=rec["id"],
        db_path=db_path,
    )
    assert main(["get", rec["id"], "--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "episode"
    grades = {(row["contact"], row["outcome"]) for row in payload["rollouts"]}
    assert grades == {("world", "fail"), ("sim", "pass")}


def test_probes_db_prints_fixture_title(db_path, capsys):
    insert_record(
        kind="procedure",
        title="Probe: old login",
        body="echo login\n",
        provenance="human",
        db_path=db_path,
    )
    assert main(["probes", "--db", str(db_path)]) == 0
    assert "Probe: old login" in capsys.readouterr().out


def test_probes_run_world_failing_command_exits_1(tmp_path, db_path):
    insert_record(
        kind="procedure",
        title="Probe: broken",
        body="false\n",
        provenance="human",
        db_path=db_path,
    )
    world = tmp_path / "world"
    world.mkdir()
    assert main(["probes", "--run", "--world", str(world), "--db", str(db_path)]) == 1


def test_compile_uncompile_round_trip_and_probe_refused(db_path, capsys):
    rec = insert_record(
        kind="procedure",
        title="How we run the formatter",
        body="Always run ruff then pytest.",
        provenance="world",
        db_path=db_path,
    )
    probe = insert_record(
        kind="procedure",
        title="Probe: old login",
        body="echo login\n",
        provenance="human",
        db_path=db_path,
    )
    assert main(["compile", rec["id"], "--db", str(db_path)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["source_record_id"] == rec["id"]
    assert first["explicit"] == 1
    assert main(["compile", rec["id"], "--db", str(db_path)]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["source_record_id"] == rec["id"]
    assert again["explicit"] == 1
    assert main(["compiled", "--db", str(db_path)]) == 0
    listed = capsys.readouterr().out
    assert rec["id"][:8] in listed
    assert rec["title"] in listed
    assert "yes" in listed
    assert main(["uncompile", rec["id"], "--db", str(db_path)]) == 0
    dropped = json.loads(capsys.readouterr().out)
    assert dropped["source_record_id"] == rec["id"]
    assert main(["compiled", "--db", str(db_path)]) == 0
    assert rec["title"] not in capsys.readouterr().out
    assert main(["compile", probe["id"], "--db", str(db_path)]) == 2


def test_rollouts_contact_sim_and_db(db_path, capsys):
    rec = insert_record(
        kind="episode",
        title="Episode abcdef12",
        body="run the tests",
        provenance="mixed",
        source_episode_id="ep-1",
        db_path=db_path,
    )
    insert_rollout(
        episode_id="ep-1",
        contact="world",
        outcome="fail",
        summary="exit code 1",
        source_record_id=rec["id"],
        db_path=db_path,
    )
    insert_rollout(
        episode_id="ep-1",
        contact="sim",
        outcome="pass",
        summary="fixed in sim",
        source_record_id=rec["id"],
        db_path=db_path,
    )
    assert main(["rollouts", "--contact", "sim", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "sim" in out
    assert "pass" in out
    assert "fixed in sim" in out
    assert "ep-1" in out
    assert "exit code 1" not in out
    assert "world" not in out


def test_load_prints_char_split_columns(db_path, capsys):
    insert_inject_stats(
        episode_id="oldold01-load-stats",
        contact="sim",
        standing_chars=1,
        retrieve_chars=2,
        trajectory_chars=3,
        total_chars=6,
        compiled_ids="aa,bb",
        retrieved_ids="rec-1",
        db_path=db_path,
    )
    insert_inject_stats(
        episode_id="abcdef12-load-stats",
        contact="world",
        standing_chars=12,
        retrieve_chars=34,
        trajectory_chars=0,
        total_chars=46,
        compiled_ids="",
        retrieved_ids="rec-1",
        db_path=db_path,
    )
    assert main(["load", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "standing" in out
    assert "retrieve" in out
    assert "traj" in out
    assert "total" in out
    assert "n_compiled" in out
    rows = [line.split() for line in out.splitlines() if line and not line.startswith("-")]
    header, newest, older = rows[0], rows[1], rows[2]
    assert header == [
        "episode",
        "contact",
        "standing",
        "retrieve",
        "traj",
        "total",
        "n_compiled",
    ]
    assert newest == ["abcdef12", "world", "12", "34", "0", "46", "0"]
    assert older == ["oldold01", "sim", "1", "2", "3", "6", "2"]


def test_pack_dry_run_json_has_n_train(db_path, capsys):
    rec = insert_record(
        kind="procedure",
        title="How we run the formatter",
        body="Always run ruff then pytest.",
        provenance="world",
        db_path=db_path,
    )
    insert_retrieve_use(
        episode_id="ep-1",
        record_id=rec["id"],
        contact="world",
        db_path=db_path,
    )
    insert_rollout(
        episode_id="ep-1",
        contact="world",
        outcome="pass",
        summary="ok",
        db_path=db_path,
    )
    assert main(["pack", "--dry-run", "--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "n_train" in payload
    assert payload["n_train"] == 1
    assert payload["dry_run"] is True
    assert payload["pack_id"] is None


def test_packs_lists_wet_pack(db_path, capsys):
    rec = insert_record(
        kind="procedure",
        title="How we run the formatter",
        body="Always run ruff then pytest.",
        provenance="world",
        db_path=db_path,
    )
    insert_retrieve_use(
        episode_id="ep-1",
        record_id=rec["id"],
        contact="world",
        db_path=db_path,
    )
    insert_rollout(
        episode_id="ep-1",
        contact="world",
        outcome="pass",
        summary="ok",
        db_path=db_path,
    )
    assert main(["pack", "--db", str(db_path)]) == 0
    packed = json.loads(capsys.readouterr().out)
    assert packed["n_train"] == 1
    assert packed["pack_id"]
    assert main(["packs", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert packed["pack_id"][:8] in out
    assert "n_train" in out
    assert "1" in out


def test_pack_and_packs_help_include_db(capsys):
    for cmd in ("pack", "packs"):
        try:
            main([cmd, "--help"])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError(f"{cmd} --help should exit")
        out = capsys.readouterr().out
        assert "--db" in out
        if cmd == "pack":
            assert PACK_FIRST_DRY_RUN_HELP in out
            assert "--dry-run" in out


def _voted_procedures(db_path, n: int = 4) -> None:
    for i in range(n):
        rec = insert_record(
            kind="procedure",
            title=f"Playbook {i}",
            body=f"steps {i}",
            provenance="world",
            db_path=db_path,
        )
        insert_retrieve_use(
            episode_id=f"ep-{i}",
            record_id=rec["id"],
            contact="world",
            db_path=db_path,
        )
        insert_rollout(
            episode_id=f"ep-{i}",
            contact="world",
            outcome="pass",
            summary="ok",
            db_path=db_path,
        )


def test_train_fake_exits_0_and_adapters_lists_shadow(db_path, capsys):
    _voted_procedures(db_path)
    assert main(["pack", "--db", str(db_path)]) == 0
    packed = json.loads(capsys.readouterr().out)
    assert packed["n_train"] >= 4
    assert main(["train", "--backend", "fake", "--db", str(db_path)]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained["backend"] == "fake"
    assert trained["adapter_id"]
    assert main(["adapters", "--db", str(db_path)]) == 0
    listed = capsys.readouterr().out
    assert trained["adapter_id"][:8] in listed
    assert "shadow" in listed


def test_train_unsloth_without_base_exits_2(db_path):
    assert main(["train", "--backend", "unsloth", "--db", str(db_path)]) == 2
    import sys

    assert "torch" not in sys.modules


def test_rollback_works(db_path, capsys):
    _voted_procedures(db_path)
    assert main(["pack", "--db", str(db_path)]) == 0
    capsys.readouterr()
    assert main(["train", "--backend", "fake", "--db", str(db_path)]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert main(["promote", trained["adapter_id"], "--force", "--db", str(db_path)]) == 0
    capsys.readouterr()
    assert main(["rollback", "--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "discarded"
    assert payload["id"] == trained["adapter_id"]
    assert main(["rollback", "--db", str(db_path)]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty == {"promoted": None}


def test_train_adapters_rollback_promote_help_include_db(capsys):
    for cmd in ("train", "adapters", "rollback", "promote"):
        try:
            main([cmd, "--help"])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError(f"{cmd} --help should exit")
        out = capsys.readouterr().out
        assert "--db" in out
        if cmd == "promote":
            assert "--force" in out


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

from unforgettable.cli import COMPACT_FIRST_DRY_RUN_HELP, main
from unforgettable.store.records import get_record, insert_record, insert_rollout


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

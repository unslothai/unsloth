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

import asyncio
from pathlib import Path

from unforgettable.cli import main
from unforgettable.eyes.probes import MAX_EPISODE_PROBES, is_probe_title, list_probes, run_probes
from unforgettable.loop.context import EpisodeRequest
from unforgettable.loop.episode import run
from unforgettable.store.records import get_record, insert_record, list_admissions
from unforgettable.tests.test_episode import FakeHost, _fail_world, _ok


def _insert_probe(
    db_path,
    title: str,
    body: str,
    *,
    status: str = "active",
):
    return insert_record(
        kind = "procedure",
        title = title,
        body = body,
        provenance = "human",
        status = status,
        db_path = db_path,
    )


def _probe_notes(db_path) -> list[str]:
    return [
        row["reason"]
        for row in list_admissions(db_path = db_path, limit = 200)
        if str(row.get("reason") or "").startswith("probe:")
    ]


def _instant_ok(
    session_id,
    name,
    arguments,
    timeout = None,
    on_chunk = None,
) -> str:
    return "ok\n"


class NoRunActionHost(FakeHost):
    run_action = None


def test_is_probe_title_prefix_and_case():
    assert is_probe_title("Probe: old login")
    assert is_probe_title("probe: case")
    assert is_probe_title("  PROBE: spaced")
    assert not is_probe_title("Not a probe")
    assert not is_probe_title("probe old login")


def test_list_probes_prefix_case_and_skips_non_probe(db_path):
    _insert_probe(db_path, "Probe: old login", "echo old\n")
    _insert_probe(db_path, "probe: case", "echo case\n")
    _insert_probe(db_path, "Not a probe", "echo no\n")
    _insert_probe(db_path, "Probe: proposed", "echo maybe\n", status = "proposed")
    rows = list_probes(db_path = db_path)
    titles = {row["title"] for row in rows}
    assert "Probe: old login" in titles
    assert "probe: case" in titles
    assert "Not a probe" not in titles
    assert "Probe: proposed" not in titles
    by_title = {row["title"]: row["command"] for row in rows}
    assert by_title["Probe: old login"] == "echo old"
    assert by_title["probe: case"] == "echo case"


def test_run_probes_grades_logs_does_not_deprecate(tmp_path: Path, db_path: Path):
    world = tmp_path / "world"
    world.mkdir()
    (world / "app.py").write_text("print('world')\n")
    ok = _insert_probe(db_path, "Probe: old login", "echo ok\n")
    bad = _insert_probe(db_path, "Probe: broken", "false\n")
    results = run_probes(world = world, db_path = db_path)
    outcomes = {row["title"]: row["outcome"] for row in results}
    assert outcomes["Probe: old login"] == "pass"
    assert outcomes["Probe: broken"] == "fail"
    notes = _probe_notes(db_path)
    assert "probe: Probe: old login pass" in notes
    assert "probe: Probe: broken fail" in notes
    assert get_record(ok["id"], db_path = db_path)["status"] == "active"
    assert get_record(bad["id"], db_path = db_path)["status"] == "active"


def test_cli_run_grades_logs_does_not_deprecate(tmp_path: Path, db_path: Path):
    world = tmp_path / "world"
    world.mkdir()
    rec = _insert_probe(db_path, "Probe: old login", "echo ok\n")
    assert main(["probes", "--run", "--world", str(world), "--db", str(db_path)]) == 0
    notes = _probe_notes(db_path)
    assert "probe: Probe: old login pass" in notes
    assert get_record(rec["id"], db_path = db_path)["status"] == "active"


def test_episode_probes_at_most_three_notes(tmp_path: Path):
    host = FakeHost(
        tmp_path,
        [_fail_world(), _ok("fixed in sim", "sim"), _ok("works in world", "world")],
        run_action = _instant_ok,
    )
    for i in range(MAX_EPISODE_PROBES + 1):
        _insert_probe(host.db, f"Probe: p{i}", "echo ok\n")
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(messages = [{"role": "user", "content": "run the tests"}]),
        )
    )
    assert outcome.state.sim_session is not None
    assert len(_probe_notes(host.db)) == MAX_EPISODE_PROBES
    assert host.last_run_action_kwargs is not None
    assert host.last_run_action_kwargs["session_id"] != host.calls[1]


def test_episode_probes_skip_without_run_action(tmp_path: Path):
    host = NoRunActionHost(
        tmp_path,
        [_fail_world(), _ok("fixed in sim", "sim"), _ok("works in world", "world")],
    )
    _insert_probe(host.db, "Probe: old login", "echo ok\n")
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(messages = [{"role": "user", "content": "run the tests"}]),
        )
    )
    assert outcome.state.sim_session is not None
    assert _probe_notes(host.db) == []


def test_episode_probes_skip_without_sim(tmp_path: Path):
    host = FakeHost(tmp_path, [_ok("ok", "world")], run_action = _instant_ok)
    _insert_probe(host.db, "Probe: old login", "echo ok\n")
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(messages = [{"role": "user", "content": "all good"}]),
        )
    )
    assert outcome.state.sim_session is None
    assert _probe_notes(host.db) == []
    assert host.last_run_action_kwargs is None


def test_episode_run_probes_passes_request_on_chunk(tmp_path: Path):
    host = FakeHost(
        tmp_path,
        [_fail_world(), _ok("fixed in sim", "sim"), _ok("works in world", "world")],
        run_action = _instant_ok,
    )
    _insert_probe(host.db, "Probe: old login", "echo ok\n")

    async def on_chunk(data: bytes) -> None:
        return None

    asyncio.run(
        run(
            host,
            EpisodeRequest(
                messages = [{"role": "user", "content": "run the tests"}],
                on_chunk = on_chunk,
            ),
        )
    )
    assert host.last_run_action_kwargs is not None
    assert host.last_run_action_kwargs["on_chunk"] is on_chunk


def test_cli_run_does_not_pass_on_chunk(tmp_path: Path, db_path: Path, monkeypatch):
    seen: dict = {}

    def fake_run_probes(
        *,
        world,
        host = None,
        db_path = None,
        limit = None,
        on_chunk = None,
    ):
        seen["host"] = host
        seen["on_chunk"] = on_chunk
        return []

    monkeypatch.setattr("unforgettable.cli.run_probes", fake_run_probes)
    world = tmp_path / "world"
    world.mkdir()
    assert main(["probes", "--run", "--world", str(world), "--db", str(db_path)]) == 0
    assert seen["host"] is None
    assert seen["on_chunk"] is None

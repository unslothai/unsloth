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

from unforgettable.store.db import get_connection
from unforgettable.store.records import insert_record, insert_rollout
from unforgettable.store.trajectories import (
    TRAJECTORY_HEADER,
    TRAJECTORY_MAX_ROWS,
    format_trajectories,
    retrieve_trajectories,
)

_LOGIN_USER = "please never leak this unique paragraph xyzzyplugh"
_LOGIN_BODY = f"## User\n{_LOGIN_USER}\n\nxylophone login widget crashed"
_FMT_BODY = "## User\nrun the rustc formatter\n\nquetzal rustc formatter crashed"


def _set_rollout_created_at(rollout_id: str, created_at: str, db_path) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE rollouts SET created_at = ? WHERE id = ?",
            (created_at, rollout_id),
        )
        conn.commit()
    finally:
        conn.close()


def _episode_with_rollout(
    db_path,
    *,
    source_episode_id: str,
    title: str,
    body: str,
    contact: str,
    outcome: str,
    summary: str,
):
    rec = insert_record(
        kind = "episode",
        title = title,
        body = body,
        provenance = "mixed",
        source_episode_id = source_episode_id,
        db_path = db_path,
    )
    rollout = insert_rollout(
        episode_id = source_episode_id,
        contact = contact,
        outcome = outcome,
        summary = summary,
        source_record_id = rec["id"],
        db_path = db_path,
    )
    return rec, rollout


def test_query_matches_episode_user_text_not_other(db_path):
    login, login_rollout = _episode_with_rollout(
        db_path,
        source_episode_id = "ep-login1",
        title = "Episode login",
        body = _LOGIN_BODY,
        contact = "world",
        outcome = "fail",
        summary = "traceback in app.py",
    )
    _episode_with_rollout(
        db_path,
        source_episode_id = "ep-fmt000",
        title = "Episode formatter",
        body = _FMT_BODY,
        contact = "world",
        outcome = "pass",
        summary = "tests: pytest",
    )
    rows = retrieve_trajectories("xylophone login widget", db_path = db_path)
    assert [row["id"] for row in rows] == [login_rollout["id"]]
    assert rows[0]["episode_record_id"] == login["id"]
    assert rows[0]["episode_id"] == "ep-login1"


def test_empty_query_returns_newest(db_path):
    _, older = _episode_with_rollout(
        db_path,
        source_episode_id = "ep-old000",
        title = "Episode old",
        body = "older xylophone run",
        contact = "world",
        outcome = "pass",
        summary = "older pass",
    )
    _, newer = _episode_with_rollout(
        db_path,
        source_episode_id = "ep-new000",
        title = "Episode new",
        body = "newer xylophone run",
        contact = "world",
        outcome = "pass",
        summary = "newer pass",
    )
    _set_rollout_created_at(older["id"], "2024-01-01T00:00:00+00:00", db_path)
    _set_rollout_created_at(newer["id"], "2024-06-01T00:00:00+00:00", db_path)
    rows = retrieve_trajectories("", db_path = db_path)
    assert rows
    assert rows[0]["id"] == newer["id"]


def test_sim_contact_ranks_sim_fail_above_world_pass(db_path):
    _, world_pass = _episode_with_rollout(
        db_path,
        source_episode_id = "ep-world1",
        title = "Episode world",
        body = "shared xylophone rehearsal",
        contact = "world",
        outcome = "pass",
        summary = "world passed",
    )
    _, sim_fail = _episode_with_rollout(
        db_path,
        source_episode_id = "ep-sim000",
        title = "Episode sim",
        body = "shared xylophone rehearsal",
        contact = "sim",
        outcome = "fail",
        summary = "sim failed",
    )
    rows = retrieve_trajectories("", contact = "sim", db_path = db_path)
    assert [row["id"] for row in rows] == [sim_fail["id"], world_pass["id"]]


def test_high_stakes_drops_sim_rollouts(db_path):
    _, world = _episode_with_rollout(
        db_path,
        source_episode_id = "ep-world2",
        title = "Episode world",
        body = "shared xylophone rehearsal",
        contact = "world",
        outcome = "fail",
        summary = "world failed",
    )
    _episode_with_rollout(
        db_path,
        source_episode_id = "ep-sim001",
        title = "Episode sim",
        body = "shared xylophone rehearsal",
        contact = "sim",
        outcome = "pass",
        summary = "sim passed",
    )
    rows = retrieve_trajectories("", high_stakes = True, db_path = db_path)
    assert [row["id"] for row in rows] == [world["id"]]
    assert all(row["contact"] != "sim" for row in rows)


def test_format_trajectories_omits_episode_body(db_path):
    rec, rollout = _episode_with_rollout(
        db_path,
        source_episode_id = "ep8hex01",
        title = "Episode login",
        body = _LOGIN_BODY,
        contact = "world",
        outcome = "fail",
        summary = "traceback in app.py",
    )
    rows = retrieve_trajectories("xylophone", db_path = db_path)
    text = format_trajectories(rows)
    assert text.startswith(TRAJECTORY_HEADER)
    assert f"- [{rollout['episode_id'][:8]}] world/fail: traceback in app.py" in text
    assert _LOGIN_USER not in text
    assert rec["body"] not in text
    assert "xyzzyplugh" not in text


def test_retrieve_trajectories_caps_at_two(db_path):
    for i in range(3):
        _, rollout = _episode_with_rollout(
            db_path,
            source_episode_id = f"ep-cap{i:03d}",
            title = f"Episode cap {i}",
            body = f"shared xylophone cap {i}",
            contact = "world",
            outcome = "pass",
            summary = f"cap {i}",
        )
        _set_rollout_created_at(rollout["id"], f"2024-01-0{i + 1}T00:00:00+00:00", db_path)
    rows = retrieve_trajectories("", db_path = db_path)
    assert len(rows) == TRAJECTORY_MAX_ROWS
    assert len(rows) == 2

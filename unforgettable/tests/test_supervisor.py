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

from unforgettable.loop.context import EpisodeRequest
from unforgettable.store.records import insert_record, list_admissions
from unforgettable.supervisor import (
    PLANNER_ON,
    VOTER_ADVISORY,
    VOTER_BINDING,
    VOTER_OFF,
    VOTE_ALLOW,
    VOTE_DENY,
    VOTE_ABSTAIN,
    HttpSupervisor,
    SupervisorConfig,
    coerce_planner_flag,
    config_from_env,
    config_from_mapping,
    parse_mine,
    parse_vote,
    planner_block,
    planner_is_on,
    request_vote_sync,
    resolve_supervisor_host,
    should_vote,
    voter_blocks,
)


class _ScriptedHost:
    def __init__(self, text: str, exc: Exception | None = None):
        self.text = text
        self.exc = exc
        self.calls = []

    async def supervise(self, purpose, messages, *, model=None, max_tokens=400):
        self.calls.append(
            {
                "purpose": purpose,
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        if self.exc is not None:
            raise self.exc
        return self.text


def test_parse_vote_json_and_bare_token():
    vote = parse_vote('{"decision":"deny","reason":"secret"}')
    assert vote.decision == VOTE_DENY
    assert vote.reason == "secret"
    fenced = parse_vote("```json\n{\"decision\":\"allow\",\"reason\":\"ok\"}\n```")
    assert fenced.decision == VOTE_ALLOW
    bare = parse_vote("abstain not sure")
    assert bare.decision == VOTE_ABSTAIN
    assert "not sure" in bare.reason
    empty = parse_vote("")
    assert empty.decision == VOTE_ABSTAIN


def test_parse_mine_existing_and_drafts():
    raw = json.dumps(
        [
            {"id": "abc", "decision": "allow", "reason": "good"},
            {"kind": "procedure", "title": "Run fmt", "body": "ruff format"},
            {"kind": "episode", "title": "skip me", "body": "no"},
            {"title": "no kind"},
        ]
    )
    items = parse_mine(raw)
    assert items[0]["id"] == "abc"
    assert items[0]["decision"] == VOTE_ALLOW
    assert items[1]["kind"] == "procedure"
    assert items[1]["title"] == "Run fmt"
    assert all(item.get("kind") != "episode" for item in items if item.get("id") is None)


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("UNFORGETTABLE_VOTER", "advisory")
    monkeypatch.setenv("UNFORGETTABLE_PLANNER", "external")
    monkeypatch.setenv("UNFORGETTABLE_VOTER_MODEL", "big-voter")
    monkeypatch.setenv("UNFORGETTABLE_SUPERVISOR_URL", "http://127.0.0.1:9/s")
    cfg = config_from_env()
    assert cfg.voter == VOTER_ADVISORY
    assert cfg.planner == PLANNER_ON
    assert cfg.voter_model == "big-voter"
    assert cfg.url.endswith("/s")


def test_config_unknown_voter_is_off(monkeypatch):
    monkeypatch.setenv("UNFORGETTABLE_VOTER", "maybe")
    monkeypatch.delenv("UNFORGETTABLE_PLANNER", raising=False)
    cfg = config_from_env()
    assert cfg.voter == VOTER_OFF
    assert cfg.planner == "off"


def test_planner_is_request_scoped(monkeypatch):
    monkeypatch.setenv("UNFORGETTABLE_PLANNER", "on")
    assert planner_is_on(EpisodeRequest(messages=[], planner=None)) is False
    assert planner_is_on(EpisodeRequest(messages=[], planner="on")) is True
    assert planner_is_on(EpisodeRequest(messages=[], planner="off")) is False
    assert coerce_planner_flag(True) == "on"
    assert coerce_planner_flag(False) == "off"


def test_should_vote_skips_episode():
    assert should_vote({"kind": "error_fix"}) is True
    assert should_vote({"kind": "episode"}) is False


def test_voter_blocks_only_when_binding_deny():
    deny = parse_vote('{"decision":"deny","reason":"no"}')
    allow = parse_vote('{"decision":"allow","reason":"yes"}')
    binding = SupervisorConfig(voter=VOTER_BINDING)
    advisory = SupervisorConfig(voter=VOTER_ADVISORY)
    assert voter_blocks(deny, force=False, config=binding) is True
    assert voter_blocks(deny, force=True, config=binding) is False
    assert voter_blocks(deny, force=False, config=advisory) is False
    assert voter_blocks(allow, force=False, config=binding) is False


def test_request_vote_logs_and_parses(db_path):
    rec = insert_record(
        kind="error_fix",
        title="Use pytest",
        body="run pytest",
        provenance="infer",
        status="proposed",
        db_path=db_path,
    )
    host = _ScriptedHost('{"decision":"deny","reason":"too vague"}')
    cfg = SupervisorConfig(voter=VOTER_BINDING, voter_model="judge")
    vote = request_vote_sync(rec, host=host, config=cfg, db_path=db_path)
    assert vote.decision == VOTE_DENY
    assert host.calls[0]["purpose"] == "vote"
    assert host.calls[0]["model"] == "judge"
    log = list_admissions(db_path=db_path)
    assert any(row["decision"] == "voter:deny" for row in log)


def test_request_vote_off_and_missing_host(db_path):
    rec = {"id": "x", "kind": "claim", "title": "t", "body": "b"}
    off = request_vote_sync(rec, host=None, config=SupervisorConfig(voter=VOTER_OFF))
    assert off.reason == "voter off"
    missing = request_vote_sync(
        rec, host=None, config=SupervisorConfig(voter=VOTER_ADVISORY)
    )
    assert missing.reason == "no supervisor"
    skip = request_vote_sync(
        {"kind": "episode", "title": "e"},
        host=_ScriptedHost("allow"),
        config=SupervisorConfig(voter=VOTER_ADVISORY),
    )
    assert skip.reason == "bookkeeping skip"


def test_request_vote_host_failure_is_abstain():
    host = _ScriptedHost("", exc=RuntimeError("down"))
    vote = request_vote_sync(
        {"kind": "claim", "title": "t", "body": "b"},
        host=host,
        config=SupervisorConfig(voter=VOTER_BINDING),
    )
    assert vote.decision == VOTE_ABSTAIN
    assert "down" in vote.reason


def test_planner_block_empty():
    assert planner_block("") == ""
    text = planner_block("1. run tests")
    assert "Supervisor plan" in text
    assert "1. run tests" in text


def test_http_supervisor_posts_and_reads_text(monkeypatch):
    seen = {}

    class _Resp:
        def read(self):
            return b'{"text":"allow it"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=30):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(
        "unforgettable.supervisor.urllib.request.urlopen", fake_urlopen
    )
    host = HttpSupervisor("http://voter.example/s", timeout=5)
    import asyncio

    text = asyncio.run(
        host.supervise("vote", [{"role": "user", "content": "x"}], model="big")
    )
    assert text == "allow it"
    assert seen["url"] == "http://voter.example/s"
    assert seen["body"]["purpose"] == "vote"
    assert seen["body"]["model"] == "big"
    assert seen["timeout"] == 5


def test_resolve_supervisor_host(monkeypatch):
    monkeypatch.delenv("UNFORGETTABLE_SUPERVISOR_URL", raising=False)
    assert resolve_supervisor_host() is None
    monkeypatch.setenv("UNFORGETTABLE_SUPERVISOR_URL", "http://127.0.0.1/s")
    host = resolve_supervisor_host()
    assert isinstance(host, HttpSupervisor)
    assert host.url.endswith("/s")


def test_config_from_mapping_overlays_env(monkeypatch):
    monkeypatch.setenv("UNFORGETTABLE_VOTER", "advisory")
    monkeypatch.delenv("UNFORGETTABLE_SUPERVISOR_URL", raising=False)
    overlaid = config_from_mapping(
        {"voter": "binding", "supervisor_url": "http://127.0.0.1/s"}
    )
    assert overlaid.voter == VOTER_BINDING
    assert overlaid.url == "http://127.0.0.1/s"
    kept = config_from_mapping({})
    assert kept.voter == VOTER_ADVISORY
    empty = config_from_mapping(None)
    assert empty.voter == VOTER_ADVISORY

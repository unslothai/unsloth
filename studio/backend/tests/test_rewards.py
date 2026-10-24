# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training import rewards
from routes.rewards import router

GOOD = "<reasoning>\n48 + 24 = 72\n</reasoning>\n<answer>\n72\n</answer>"


@pytest.fixture
def user_root(tmp_path, monkeypatch):
    root = tmp_path / "rewards"
    monkeypatch.setattr(rewards, "_user_root", lambda: root)
    return root


@pytest.fixture
def client(user_root):
    app = FastAPI()
    app.include_router(router, prefix = "/api/rewards")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app)


def _md(
    name: str,
    body: str,
    kind: str = "rule",
) -> str:
    return f"---\nname: {name}\nkind: {kind}\ndescription: test\n---\n{body}"


def test_bundled_rewards_all_parse(user_root):
    records = rewards.list_rewards()
    names = {r["name"] for r in records}
    assert {"strict-xml-format", "soft-xml-format", "exact-answer", "numeric-close"} <= names
    assert all(r["valid"] for r in records), [r["error"] for r in records if not r["valid"]]


def test_bundled_rewards_score_a_correct_and_a_wrong_reply(user_root):
    specs = {r["name"]: r for r in rewards.list_rewards()}
    assert rewards.score_rule(specs["strict-xml-format"]["rule"], GOOD) == 0.5
    assert rewards.score_rule(specs["exact-answer"]["rule"], GOOD, "72") == 2.0
    assert rewards.score_rule(specs["exact-answer"]["rule"], GOOD, "73") == 0.0
    assert rewards.score_rule(specs["numeric-close"]["rule"], "<answer>75</answer>", "72") == 1.5
    assert rewards.score_rule(specs["numeric-close"]["rule"], "no tags", "72") == -2.0
    assert rewards.score_rule(specs["strict-xml-format"]["rule"], "It's 72") == 0.0


@pytest.mark.parametrize(
    "body, text, reference, expected",
    [
        (
            "type: regex\nmode: search\npattern: 'foo'\nscore: {match: 1, miss: -1}",
            "a foo b",
            None,
            1.0,
        ),
        ("type: regex\npattern: 'foo'\nscore: {match: 1, miss: -1}", "a foo b", None, -1.0),
        (
            "type: exact_match\nnormalize: [strip, lower, remove_commas]\nscore: {match: 2}",
            " 1,000 ",
            "1000",
            2.0,
        ),
        (
            "type: exact_match\nextract: {regex: 'is (\\d+)'}\nscore: {match: 2}",
            "it is 7",
            "7",
            2.0,
        ),
        ("type: numeric\nbands: [{within: 0.0, score: 1}]\nelse: -1", "0", "0", 1.0),
        (
            "type: json_schema\nschema: {required: [tool]}\nscore: {match: 1, miss: -1}",
            '```json\n{"tool": "x"}\n```',
            None,
            1.0,
        ),
        (
            "type: json_schema\nschema: {required: [tool]}\nscore: {match: 1, miss: -1}",
            "{nope",
            None,
            -1.0,
        ),
        ("type: length\nmax_chars: 3\nscore: {over: -1, under: 0}", "abcd", None, -1.0),
    ],
)
def test_rule_types(body, text, reference, expected):
    spec = rewards.parse_reward_markdown(_md("t", body))
    assert rewards.score_rule(spec["rule"], text, reference) == expected


@pytest.mark.parametrize(
    "raw, message",
    [
        ("type: regex\npattern: x", "frontmatter"),
        (_md("t", "type: shell\ncmd: rm"), "type must be one of"),
        (_md("t", "type: regex\npattern: '('"), "Invalid regex"),
        (_md("t", "type: regex\npattern: x", kind = "python"), "named 'reward'"),
        (_md("t", "import os\nimport sys", kind = "python"), "named 'reward'"),
        (_md("t", "type: regex\npattern: x", kind = "shell"), "kind must be"),
        (_md("Bad Name", "type: regex\npattern: x"), "lowercase"),
        (_md("t", "type: numeric\nbands: []"), "bands"),
    ],
)
def test_invalid_rewards_are_refused(raw, message):
    with pytest.raises(rewards.RewardError, match = message):
        rewards.parse_reward_markdown(raw)


def test_bundled_rewards_read_gsm8k_style_references(user_root):
    specs = {r["name"]: r for r in rewards.list_rewards()}
    gsm8k = "Natalia sold 48/2 = 24 clips in May.\n#### 72"
    assert rewards.score_rule(specs["exact-answer"]["rule"], GOOD, gsm8k) == 2.0
    assert rewards.score_rule(specs["numeric-close"]["rule"], GOOD, gsm8k) == 3.0


def test_make_reward_func_reads_the_reference_column_and_conversational_completions():
    spec = rewards.parse_reward_markdown(
        _md(
            "exact",
            "type: exact_match\nextract: {between: ['<answer>', '</answer>']}\nscore: {match: 2}",
        )
    )
    func = rewards.make_reward_func(spec)
    assert func.__name__ == "exact"
    scores = func(
        prompts = [None, None, None],
        completions = [[{"role": "assistant", "content": GOOD}], "<answer>5</answer>", "none"],
        answer = ["72", "6", "1"],
    )
    assert scores == [2.0, 0.0, 0.0]


def test_import_export_roundtrip_and_shadowing(client, user_root):
    raw = _md("exact-answer", "type: exact_match\nscore: {match: 9}")
    created = client.post("/api/rewards", json = {"markdown": raw})
    assert created.status_code == 201, created.text
    assert (user_root / "exact-answer" / "REWARD.md").is_file()
    assert client.post("/api/rewards", json = {"markdown": raw}).status_code == 409

    listed = [r for r in client.get("/api/rewards").json() if r["name"] == "exact-answer"]
    assert [(r["source"], r["shadowed"]) for r in listed] == [("user", False), ("bundled", True)]
    assert rewards.get_reward("exact-answer")["rule"]["score"]["match"] == 9.0

    exported = client.get("/api/rewards/exact-answer/export").json()["markdown"]
    assert (
        rewards.parse_reward_markdown(exported)["rule"]
        == rewards.get_reward("exact-answer")["rule"]
    )

    assert client.delete("/api/rewards/exact-answer").status_code == 204
    assert client.delete("/api/rewards/strict-xml-format").status_code == 404


def test_preview_scores_and_weights(client):
    response = client.post(
        "/api/rewards/preview",
        json = {
            "rewards": [{"name": "strict-xml-format", "weight": 2}, {"name": "exact-answer"}],
            "completion": GOOD,
            "reference": "72",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["weighted"] for s in body["scores"]] == [1.0, 2.0]
    assert body["total"] == 3.0


def test_bundled_root_is_packaged():
    pyproject = Path(rewards.__file__).parents[4] / "pyproject.toml"
    assert "backend/core/training/bundled_rewards/**/*.md" in pyproject.read_text("utf-8")


def test_a_backtracking_regex_rule_gives_up_instead_of_pinning_the_thread(monkeypatch):
    # (a+)+$ against a long near-match backtracks for minutes; a Try it request would hold a worker.
    import time

    monkeypatch.setattr(rewards, "REGEX_TIMEOUT_SECONDS", 0.2)
    rule = rewards.validate_rule(
        {"type": "regex", "pattern": "(a+)+$", "mode": "search", "score": {"match": 1, "miss": 0}}
    )
    started = time.monotonic()
    assert rewards.score_rule(rule, "a" * 40 + "!") == 0
    assert time.monotonic() - started < 5
    assert rewards.score_rule(rule, "aaa") == 1


def test_preview_reads_each_rules_compare_to_column_like_training():
    rule = {
        "type": "exact_match",
        "extract": None,
        "compare_to": "label",
        "score": {"match": 1, "miss": 0},
    }
    spec = {"name": "label-match", "kind": "rule", "rule": rewards.validate_rule(rule)}
    row = {"answer": "72", "label": "yes"}
    trained = rewards.make_reward_func(spec)(
        completions = ["yes"], **{k: [v] for k, v in row.items()}
    )
    scores, _ = rewards.preview_scores([spec], "yes", reference = "72", row = row)
    assert scores[0]["score"] == trained[0] == 1


@pytest.mark.parametrize("bands", [[None], ["x"], [None, {"within": 1, "score": 1}]])
def test_numeric_bands_must_all_be_mappings(bands):
    with pytest.raises(rewards.RewardError, match = "bands"):
        rewards.validate_rule({"type": "numeric", "extract": None, "bands": bands})


def test_a_backtracking_extract_regex_gives_up_too(monkeypatch):
    import time

    monkeypatch.setattr(rewards, "REGEX_TIMEOUT_SECONDS", 0.2)
    rule = rewards.validate_rule(
        {
            "type": "exact_match",
            "extract": {"regex": "((a+)+)$"},
            "score": {"match": 1, "miss": 0},
        }
    )
    started = time.monotonic()
    assert rewards.score_rule(rule, "a" * 40 + "!", "aaa") == rule["missing"]
    assert time.monotonic() - started < 5
    assert rewards.score_rule(rule, "aaa", "aaa") == 1


def test_json_schema_types_are_checked_or_refused():
    def rule(kind):
        return rewards.validate_rule(
            {"type": "json_schema", "schema": {"type": kind}, "score": {"match": 1, "miss": 0}}
        )

    assert rewards.score_rule(rule("integer"), "3") == 1
    assert rewards.score_rule(rule("integer"), '"three"') == 0
    assert rewards.score_rule(rule("integer"), "3.5") == 0
    assert rewards.score_rule(rule("boolean"), "true") == 1
    assert rewards.score_rule(rule("boolean"), "1") == 0
    assert rewards.score_rule(rule("number"), "true") == 0
    assert rewards.score_rule(rule("null"), "null") == 1
    with pytest.raises(rewards.RewardError, match = "schema.type"):
        rule("objcet")


def test_json_schema_refuses_nan_huge_ints_and_a_scalar_required():
    def rule(kind, **schema):
        return rewards.validate_rule(
            {
                "type": "json_schema",
                "schema": {"type": kind, **schema},
                "score": {"match": 1, "miss": 0},
            }
        )

    # Python's json.loads takes NaN / Infinity, which are not JSON.
    assert rewards.score_rule(rule("number"), "NaN") == 0
    assert rewards.score_rule(rule("number"), "-Infinity") == 0
    assert rewards.score_rule(rule("number"), "1e999") == 0
    assert rewards.score_rule(rule("number"), "2.5") == 1
    # A whole number past float range is still an integer, and must not raise.
    assert rewards.score_rule(rule("integer"), "9" * 400) == 1
    for bad in ("id", False, None, 0, ""):
        with pytest.raises(rewards.RewardError, match = "required"):
            rule("object", required = bad)

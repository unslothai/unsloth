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
        (_md("t", "type: regex\npattern: x", kind = "python"), "Python rewards are not supported"),
        (_md("t", "import os\nimport sys", kind = "python"), "Python rewards are not supported"),
        (_md("Bad Name", "type: regex\npattern: x"), "lowercase"),
        (_md("t", "type: numeric\nbands: []"), "bands"),
        (_md("t", "type: numeric\nbands: [1, 2]"), "band"),
        (_md("t", "type: exact_match\ncompare_to: completions"), "TRL reserves"),
        (
            _md("t", "type: json_schema\nextract: {regex: '(.*)'}\nschema: {type: [object]}"),
            "schema.type",
        ),
        (
            _md("t", "type: json_schema\nextract: {regex: '(.*)'}\nschema: {type: integer}"),
            "schema.type",
        ),
        (
            _md("t", "type: json_schema\nextract: {regex: '(.*)'}\nschema: {required: [[a, b]]}"),
            "schema.required",
        ),
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


def test_bundled_strict_format_scores_whitespace_runs_quickly(user_root):
    import time

    specs = {r["name"]: r for r in rewards.list_rewards()}
    ws = "\n" * 150
    text = (
        f"<reasoning>\nSo 16 - 3 - 4 = 9.\n{ws}</reasoning>\n<answer>\n{ws}9{ws}</answer>\nThanks"
    )
    start = time.perf_counter()
    assert rewards.score_rule(specs["strict-xml-format"]["rule"], text) == 0.0
    assert time.perf_counter() - start < 0.5
    assert (
        rewards.score_rule(
            specs["strict-xml-format"]["rule"], "<reasoning>a</reasoning>\n<answer>9</answer>"
        )
        == 0.5
    )


def test_catastrophic_user_regex_times_out_as_a_miss(monkeypatch):
    import time

    pytest.importorskip("regex")
    monkeypatch.setattr(rewards, "MATCH_TIMEOUT_S", 0.2)
    rule = rewards.parse_reward_markdown(_md("t", "type: regex\npattern: '(a+)+$'\nmode: search"))[
        "rule"
    ]
    start = time.perf_counter()
    assert rewards.score_rule(rule, "a" * 40 + "!") == rule["score"]["miss"]
    assert time.perf_counter() - start < 5


def test_regex_rule_is_refused_without_the_timeout_engine(monkeypatch):
    monkeypatch.setattr(rewards, "_rx", None)
    with pytest.raises(rewards.RewardError, match = "regex"):
        rewards.parse_reward_markdown(_md("t", "type: regex\npattern: '(a+)+$'"))
    with pytest.raises(rewards.RewardError, match = "regex"):
        rewards._match("(a+)+$", "a" * 40 + "!")


@pytest.mark.parametrize("pattern", ["a{10000000}", "((a{1000}){1000}){1000}"])
def test_regex_repeats_that_unroll_huge_are_refused_before_compiling(pattern):
    pytest.importorskip("regex")
    with pytest.raises(rewards.RewardError, match = "repeat"):
        rewards.parse_reward_markdown(_md("t", f"type: regex\npattern: '{pattern}'"))
    with pytest.raises(rewards.RewardError, match = "repeat"):
        rewards._match(pattern, "a")
    assert rewards._match("[a-z]{2,64}", "ab")


def test_yaml_aliases_and_merge_keys_are_refused_before_they_expand(user_root):
    rewards.import_reward(_md("big", "type: length\nmax_chars: 9"))
    marker = "x" * 17_000
    aliased = (
        "---\nname: big\ndescription: d\n---\ntype: numeric\n"
        f"extract: {{between: [&s '{marker}', *s]}}\n"
        "reference_extract: {between: [*s, *s]}\nbands: [{within: 0, score: 1}]\n"
    )
    with pytest.raises(rewards.RewardError, match = "alias"):
        rewards.import_reward(aliased, overwrite = True)
    assert rewards.get_reward("big")["rule"]["type"] == "length"
    merges = "m0: &m0 {x: 1}\n" + "".join(
        f"m{i}: &m{i} {{<<: [*m{i - 1}, *m{i - 1}]}}\n" for i in range(1, 25)
    )
    with pytest.raises(rewards.RewardError, match = "alias"):
        rewards.parse_reward_markdown(f"---\nname: x\n{merges}---\ntype: length\nmax_chars: 9\n")


def test_an_empty_extracted_reference_still_matches_an_empty_answer():
    between = {"between": ["<answer>", "</answer>"]}
    rule = rewards.validate_rule(
        {
            "type": "exact_match",
            "extract": between,
            "reference_extract": between,
            "score": {"match": 2.0, "miss": 0.0},
        }
    )
    assert rewards.score_rule(rule, "<answer></answer>", "<answer></answer>") == 2.0
    assert rewards.score_rule(rule, "<answer>72</answer>", "72") == 2.0


def test_import_refuses_a_reward_that_grows_past_the_limit_once_saved(user_root, monkeypatch):
    rewards.import_reward(_md("big", "type: length\nmax_chars: 9"))
    monkeypatch.setattr(rewards, "MAX_REWARD_MD_BYTES", 120)
    with pytest.raises(rewards.RewardError, match = "once saved"):
        rewards.import_reward(
            "---\nname: big\ndescription: d\n---\ntype: exact_match\n", overwrite = True
        )
    monkeypatch.setattr(rewards, "MAX_REWARD_MD_BYTES", 64 * 1024)
    assert rewards.get_reward("big")["rule"]["type"] == "length"


def test_import_refuses_to_write_through_a_linked_reward(user_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = rewards._user_root()
    root.mkdir(parents = True, exist_ok = True)
    (root / "linked").symlink_to(outside, target_is_directory = True)
    with pytest.raises(rewards.RewardError, match = "link"):
        rewards.import_reward(_md("linked", "type: regex\npattern: x"), overwrite = True)
    assert not (outside / "REWARD.md").exists()


def test_a_linked_library_cannot_delete_outside_the_account(tmp_path, monkeypatch):
    from utils import paths
    from utils.paths import storage_roots

    account = tmp_path / "accounts" / "a1"
    account.mkdir(parents = True)
    outside = tmp_path / "elsewhere"
    (outside / "victim").mkdir(parents = True)
    (outside / "victim" / "REWARD.md").write_text("keep", encoding = "utf-8")
    (account / "rewards").symlink_to(outside, target_is_directory = True)
    monkeypatch.setattr(storage_roots, "is_owner_context", lambda: False)
    monkeypatch.setattr(storage_roots, "workspace_root", lambda: account)
    monkeypatch.setattr(paths, "workspace_root", lambda: account)
    monkeypatch.setattr(storage_roots, "project_workspaces_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(storage_roots, "tmp_root", lambda: tmp_path / "tmp")

    with pytest.raises(rewards.RewardError):
        rewards.delete_reward("victim")
    assert (outside / "victim" / "REWARD.md").read_text(encoding = "utf-8") == "keep"
    assert all(r["source"] == "bundled" for r in rewards.list_rewards())


def test_alias_tree_name_is_refused_before_expansion():
    levels = ["l0: &l0 [xxxxxxxx, xxxxxxxx]"] + [
        f"l{i}: &l{i} [*l{i - 1}, *l{i - 1}]" for i in range(1, 40)
    ]
    raw = "---\n" + "\n".join(levels) + "\nname: *l39\nkind: rule\n---\ntype: regex\npattern: a\n"
    with pytest.raises(rewards.RewardError, match = "alias"):
        rewards.parse_reward_markdown(raw)


def test_import_refuses_a_full_library_but_still_replaces(user_root, monkeypatch):
    monkeypatch.setattr(rewards, "MAX_REWARDS_PER_ROOT", 2)
    for name in ("a-one", "a-two"):
        rewards.import_reward(_md(name, "type: exact_match"))
    with pytest.raises(rewards.RewardError, match = "library is full"):
        rewards.import_reward(_md("z-three", "type: exact_match"))
    rewards.import_reward(_md("a-two", "type: exact_match\nscore: {match: 3}"), overwrite = True)
    assert rewards.get_reward("a-two")["rule"]["score"]["match"] == 3.0


def test_failed_overwrite_keeps_the_saved_reward(user_root, monkeypatch):
    rewards.import_reward(_md("keep-me", "type: exact_match\nscore: {match: 4}"))

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rewards.os, "replace", boom)
    with pytest.raises(OSError):
        rewards.import_reward(
            _md("keep-me", "type: exact_match\nscore: {match: 5}"), overwrite = True
        )
    assert rewards.get_reward("keep-me")["rule"]["score"]["match"] == 4.0
    assert [p.name for p in (user_root / "keep-me").iterdir()] == ["REWARD.md"]


def test_json_rule_misses_on_nan_and_infinity():
    rule = rewards.validate_rule(
        {
            "type": "json_schema",
            "schema": {"type": "object", "required": ["answer"]},
            "score": {"match": 1, "miss": -1},
        }
    )
    for text in ('{"answer": NaN}', '{"answer": Infinity}', '{"answer": -Infinity}'):
        assert rewards.score_rule(rule, text, None) == -1.0
    assert rewards.score_rule(rule, '{"answer": 1}', None) == 1.0


def test_json_rule_misses_on_deeply_nested_output():
    rule = rewards.validate_rule(
        {
            "type": "json_schema",
            "schema": {"type": "array", "required": []},
            "score": {"match": 1, "miss": -1},
        }
    )
    assert rewards.score_rule(rule, "[" * 10_000, None) == -1.0


@pytest.mark.parametrize("column", ["chosen", "rejected", "messages", "label"])
def test_compare_to_refuses_trl_dataset_keys(column):
    with pytest.raises(rewards.RewardError, match = "reserves that name"):
        rewards.validate_rule({"type": "exact_match", "compare_to": column})


def test_failed_first_save_leaves_no_folder_and_can_be_retried(user_root, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(rewards.os, "replace", boom)
        with pytest.raises(OSError):
            rewards.import_reward(_md("fresh", "type: exact_match"))
    assert not (user_root / "fresh").exists()
    rewards.import_reward(_md("fresh", "type: exact_match"))
    assert rewards.get_reward("fresh")["name"] == "fresh"


def test_json_schema_refuses_keywords_it_does_not_enforce():
    with pytest.raises(rewards.RewardError, match = "properties"):
        rewards.validate_rule(
            {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"answer": {"type": "number"}}},
            }
        )


def test_regex_rules_keep_and_apply_their_extraction():
    from core.training.rewards import score_rule, validate_rule

    rule = validate_rule(
        {
            "type": "regex",
            "pattern": "[0-9]+",
            "mode": "fullmatch",
            "extract": {"between": ["<answer>", "</answer>"]},
            "score": {"match": 1.0, "miss": 0.0},
        }
    )
    assert rule["extract"] == {"between": ["<answer>", "</answer>"]}
    assert score_rule(rule, "<reasoning>2+2=4</reasoning><answer>4</answer>") == 1.0
    assert score_rule(rule, "<answer>four</answer>") == 0.0
    assert score_rule(rule, "4") == 0.0  # no answer tags: nothing to match
    plain = validate_rule(
        {"type": "regex", "pattern": "[0-9]+", "score": {"match": 1.0, "miss": 0.0}}
    )
    assert "extract" not in plain and score_rule(plain, "4") == 1.0


def test_regex_extraction_is_explained():
    from core.training.rewards import score_rule, validate_rule

    rule = validate_rule(
        {
            "type": "regex",
            "pattern": "[0-9]+",
            "extract": {"between": ["<answer>", "</answer>"]},
            "score": {"match": 1.0, "miss": 0.0},
        }
    )
    hit, miss = {}, {}
    assert score_rule(rule, "<answer>4</answer>", explanation = hit) == 1.0
    assert hit["reason"] == "matched" and hit["response"] == "4"
    assert score_rule(rule, "4", explanation = miss) == 0.0
    assert miss["reason"] == "missing_response"


def test_rules_that_score_the_whole_reply_refuse_an_extraction_step():
    rule = {
        "type": "length",
        "max_chars": 10,
        "score": {"over": -1, "under": 1},
        "extract": {"between": ["<answer>", "</answer>"]},
    }
    with pytest.raises(rewards.RewardError, match = "length rules take no extract"):
        rewards.validate_rule(rule)


def test_fullmatch_regex_keeps_boundary_whitespace_a_pattern_requires():
    from core.training.rewards import score_rule, validate_rule

    rule = validate_rule(
        {
            "type": "regex",
            "pattern": "<answer>\\n[0-9]+\\n</answer>\\n",
            "mode": "fullmatch",
            "score": {"match": 1.0, "miss": 0.0},
        }
    )
    assert score_rule(rule, "<answer>\n4\n</answer>\n") == 1.0
    loose = validate_rule(
        {
            "type": "regex",
            "pattern": "[0-9]+",
            "mode": "fullmatch",
            "score": {"match": 1.0, "miss": 0.0},
        }
    )
    assert score_rule(loose, "  4\n") == 1.0  # stray padding still tolerated


def test_json_schema_scoring_stays_fast_on_an_unclosed_fence():
    import time

    from core.training.rewards import score_rule, validate_rule

    rule = validate_rule(
        {"type": "json_schema", "schema": {"type": "object"}, "score": {"match": 1.0, "miss": 0.0}}
    )
    assert score_rule(rule, '```json\n{"a": 1}\n```') == 1.0
    start = time.perf_counter()
    assert score_rule(rule, "```json\n" + " " * 65000) == 0.0
    assert time.perf_counter() - start < 1.0

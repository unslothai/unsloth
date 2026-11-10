# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import copy

import pytest

from core.flows import loop as loop_rounds
from core.flows.compiler import CompileError, compile_flow, parse_document
from core.flows.engine import FlowEngine
from core.flows.specs import SPECS_BY_TYPE, inp, out, spec

from ._flows_helpers import doc, edge, node, run_async

COLUMN = {"name": "ticket", "column_type": "sampler", "sampler_type": "uuid", "params": {}}
# Distinct tickets: near-copies seed as one failure, so templated rows would collapse.
HELD = [
    "my invoice shows a charge I never made",
    "the parcel has been stuck in customs for weeks",
    "I want my money back for the broken blender",
    "cannot sign in after the password reset email",
    "someone changed my account email without asking",
    "where do I print a return label for these shoes",
    "billed twice for the annual plan this morning",
    "the courier says delivered but nothing arrived",
]


@pytest.fixture(autouse = True)
def fake_specs(monkeypatch):
    monkeypatch.setitem(
        SPECS_BY_TYPE,
        "test.bench",
        spec(
            "test.bench",
            "Benchmark",
            "data",
            "",
            inputs = [inp("checkpoint", "checkpoint"), inp("dataset", "dataset")],
            outputs = [out("score", "number"), out("results", "json"), out("failures", "json")],
            side_effect = True,
        ),
    )


def loop_doc(gen_rows = 4, **loop_params):
    return doc(
        [
            node("loop", "logic.loop", group = "L", **loop_params),
            node("gen", "recipe.generate", group = "R", rows = gen_rows),
            node("train", "train.finetune", group = "L", holdout_percent = 20),
            node("bench", "test.bench", group = "L"),
            node("col", "recipe.column", group = "R", column = COLUMN),
        ],
        [
            edge("gen", "dataset", "train", "dataset"),
            edge("train", "checkpoint", "bench", "checkpoint"),
            edge("train", "holdout", "bench", "dataset"),
            edge("bench", "failures", "gen", "seed", kind = "feedback"),
        ],
        groups = [
            {"id": "L", "label": "Loop", "kind": "loop", "rect": [0, 0, 900, 500]},
            {"id": "R", "label": "Recipe", "kind": "recipe", "rect": [20, 40, 300, 300]},
        ],
    )


def _plan(d):
    parsed, errors = parse_document(d)
    assert not errors, errors
    return compile_flow(parsed)


def test_loop_body_takes_the_fed_recipe_group_and_stays_acyclic():
    plan = _plan(loop_doc())
    assert set(plan.bodies["loop"]) == {"gen", "col", "train", "bench"}
    assert plan.feedback["loop"]["target"] == "gen"
    assert "seed" not in plan.inputs["gen"]
    assert plan.order.index("gen") < plan.order.index("train") < plan.order.index("bench")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["edges"].pop(), "Draw the dashed feeds-the-next-round wire"),
        (
            lambda d: d["edges"].append(edge("train", "holdout", "gen", "seed")),
            "takes one connection",
        ),
        (
            lambda d: d["edges"][-1].update(source = "train", source_port = "metrics"),
            "scores the model",
        ),
        (lambda d: d["edges"][-1].update(source_port = "results"), "scores the model"),
        (
            lambda d: d["edges"][-1].update(target = "train", target_port = "dataset"),
            "recipe's seed input",
        ),
        (
            lambda d: d["nodes"][3].pop("group"),
            "runs inside one Loop group",
        ),
        (lambda d: d["edges"].pop(2), "needs held-out rows"),
        (lambda d: d["nodes"][3].update(disabled = True), "its Loop scores every round with it"),
        (lambda d: d["edges"].pop(1), "so each round scores the model it just trained"),
        (lambda d: d["nodes"][2]["params"].pop("holdout_percent"), "Hold out % above 0"),
        (lambda d: d["nodes"][2]["params"].update(holdout_percent = 0), "Hold out % above 0"),
        (lambda d: d["nodes"][2]["params"].update(holdout_percent = 0.5), "Hold out % above 0"),
        (
            lambda d: d["edges"][2].update(source = "gen", source_port = "dataset"),
            "needs held-out rows",
        ),
        (
            lambda d: (
                d["nodes"].append(node("after", "output.result")),
                d["edges"].append(edge("train", "metrics", "after", "value")),
            ),
            "only feeds its Loop",
        ),
    ],
)
def test_bad_loops_are_refused(change, message):
    d = loop_doc()
    change(d)
    parsed, errors = parse_document(d)
    assert not errors, errors
    with pytest.raises(CompileError) as exc:
        compile_flow(parsed)
    assert any(message in i["message"] for i in exc.value.issues), exc.value.issues


def test_split_is_disjoint_covering_and_stable():
    rows = [{"i": i} for i in range(16)]
    probe, test = loop_rounds.split_holdout(rows, 50, 7)
    assert len(probe) == len(test) == 8
    assert sorted(r["i"] for r in probe + test) == list(range(16))
    assert loop_rounds.split_holdout(rows, 50, 7) == (probe, test)
    assert loop_rounds.split_holdout(rows[:2], 99, 1)[1]


def test_seed_rows_rewrite_failures_then_fresh():
    rows = loop_rounds.seed_rows(
        [{"prompt": "a", "reference": "b", "output": "c"}], 4, 50, per_failure = 2
    )
    assert [r["origin"] for r in rows] == ["failure", "failure", "fresh", "fresh"]
    assert rows[0] == {
        "origin": "failure",
        "failed_prompt": "a",
        "failed_reference": "b",
        "failed_output": "c",
    }
    assert set(rows[3]) == set(rows[0]) and rows[3]["failed_prompt"] == ""
    assert all(r["origin"] == "fresh" for r in loop_rounds.seed_rows([], 3, 50))
    assert loop_rounds.as_percent(0.875) == 87.5 and loop_rounds.as_percent(88) == 88.0
    assert loop_rounds.as_percent({"accuracy": 1}) == 100.0 and loop_rounds.as_percent("x") is None
    # Benchmark answers in percent: 0.5 is half a percent, not 50.
    assert (
        loop_rounds.as_percent(0.5, percent = True) == 0.5
        and loop_rounds.as_percent(87.5, percent = True) == 87.5
    )


def test_few_failures_never_flood_the_seed_with_near_copies():
    failures = [
        {
            "prompt": "My card was charged twice for one order!",
            "reference": "billing",
            "output": "refunds",
        },
        {
            "prompt": "my card was charged twice for one order",
            "reference": "billing",
            "output": "refunds",
        },
        {
            "prompt": "My card was charged twice for 1 order.",
            "reference": "billing",
            "output": "shipping",
        },
        {
            "prompt": "the package never arrived at my door",
            "reference": "shipping",
            "output": "returns",
        },
    ]
    rows = loop_rounds.seed_rows(failures, 60, 50)
    fed = [r for r in rows if r["origin"] == "failure"]
    # Three near-copies count once: two distinct failures, two rewrites each, the other 56 rows fresh.
    assert len(fed) == 4 and len(rows) == 60
    assert sorted({r["failed_prompt"] for r in fed}) == [
        "My card was charged twice for one order!",
        "the package never arrived at my door",
    ]
    # The same prompt filed under a different queue is a different failure.
    moved = failures + [
        {"prompt": "the package never arrived at my door", "reference": "returns", "output": "x"}
    ]
    assert len([r for r in loop_rounds.seed_rows(moved, 60, 50) if r["origin"] == "failure"]) == 6
    # Many distinct failures still fill the share, never more.
    many = [{"prompt": t, "reference": "billing", "output": "x"} for t in HELD]
    assert len([r for r in loop_rounds.seed_rows(many, 8, 25) if r["origin"] == "failure"]) == 2
    assert (
        next(p for p in SPECS_BY_TYPE["logic.loop"]["params"] if p["id"] == "failure_percent")[
            "default"
        ]
        == 25
    )


class LoopEngine(FlowEngine):
    async def _upload_rows(self, ctx, rows, name):
        return {"kind": "dataset", "source": "local", "ref": name, "rows": len(rows), "data": rows}

    async def _dataset_rows(self, ctx, value):
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        return await super()._dataset_rows(ctx, value)


def _engine(leak = False, test_pass = None):
    """Gen records its seed; the model 'learns' every prompt that reached its seed as a failure."""
    seen: list[list[dict]] = []
    starts: list = []
    scored: list[tuple[str, list[str]]] = []

    async def gen(ctx):
        rows = ctx.inputs["seed"]["data"]
        seen.append(rows)
        return {"dataset": {"kind": "dataset", "source": "local", "ref": f"gen{len(seen)}"}}

    async def train(ctx):
        starts.append(ctx.inputs.get("start"))
        learned = {r["failed_prompt"] for rows in seen for r in rows if r["origin"] == "failure"}
        held = {
            "kind": "dataset",
            "source": "local",
            "ref": "held",
            "rows": len(HELD),
            "data": [{"prompt": p, "reference": "billing"} for p in HELD],
        }
        return {
            "checkpoint": {
                "kind": "checkpoint",
                "name": f"run{len(seen)}",
                "learned": sorted(learned),
            },
            "holdout": held,
            "metrics": {},
        }

    async def bench(ctx):
        rows = ctx.inputs["dataset"]["data"]
        learned = set(ctx.inputs["checkpoint"]["learned"])
        scored.append((ctx.inputs["dataset"]["split"], [r["prompt"] for r in rows]))
        failed = [r for r in rows if r["prompt"] not in learned]
        if ctx.inputs["dataset"]["split"] == "test" and test_pass is not None:
            passed = test_pass[len([s for s in scored if s[0] == "test"]) - 1]
            return {"score": passed, "failures": []}
        if leak:
            failed = failed + [
                {"prompt": HELD[i], "reference": "billing", "output": "x"} for i in range(8)
            ]
        return {
            "score": 1 - len(failed) / len(rows),
            "failures": [{**r, "output": "wrong"} for r in failed],
        }

    eng = LoopEngine()
    eng.handlers = {"recipe.generate": gen, "train.finetune": train, "test.bench": bench}
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    eng.starts = starts
    return eng, seen, scored


async def _run(eng, d):
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = {})
    await asyncio.wait_for(rs.task, timeout = 20)
    return rs


@run_async
async def test_probe_failures_feed_the_next_round_and_test_rows_never_do():
    eng, seen, scored = _engine()
    rs = await _run(eng, loop_doc(rounds = 3, target = 100, failure_percent = 50))
    assert rs.status == "completed", rs.error
    assert all(r["origin"] == "fresh" for r in seen[0])
    probe = scored[0][1]
    test = scored[1][1]
    assert scored[0][0] == "probe" and scored[1][0] == "test" and not set(probe) & set(test)
    assert sorted(probe + test) == sorted(HELD)
    # Same split every round.
    assert all(s[1] == (probe if s[0] == "probe" else test) for s in scored)
    fed = [r["failed_prompt"] for rows in seen[1:] for r in rows if r["origin"] == "failure"]
    assert fed and set(fed) <= set(probe) and not set(fed) & set(test)
    history = rs.nodes["loop"].outputs["history"]
    assert [h["round"] for h in history["rounds"]] == [1, 2, 3]
    assert [h["probe_score"] for h in history["rounds"]] == [0, 50, 100]
    assert history["rounds"][1]["rows_from_failures"] == 2 and history["stopped"] == "max_rounds"
    assert all(h["test_score"] == 0 for h in history["rounds"])


@run_async
async def test_a_test_row_in_the_probe_failures_never_reaches_the_seed():
    eng, seen, _ = _engine(leak = True)
    rs = await _run(eng, loop_doc(gen_rows = 16, rounds = 2, failure_percent = 100))
    assert rs.status == "completed", rs.error
    _, test = loop_rounds.split_holdout([{"prompt": p} for p in HELD], 50, 3407)
    fed = {r["failed_prompt"] for r in seen[1] if r["origin"] == "failure"}
    assert fed and not fed & {t["prompt"] for t in test}


@run_async
async def test_feedback_off_seeds_fresh_rows_every_round():
    eng, seen, _ = _engine()
    rs = await _run(eng, loop_doc(rounds = 3, feedback = False))
    assert rs.status == "completed", rs.error
    assert len(seen) == 3 and all(r["origin"] == "fresh" for rows in seen for r in rows)
    assert rs.nodes["loop"].outputs["history"]["feedback"] is False


@run_async
async def test_stops_at_the_target_test_score_and_keeps_the_best_checkpoint():
    eng, seen, _ = _engine(test_pass = [0.25, 0.75, 1.0])
    rs = await _run(eng, loop_doc(rounds = 5, target = 70))
    assert rs.status == "completed", rs.error
    outputs = rs.nodes["loop"].outputs
    assert len(seen) == 2 and outputs["history"]["stopped"] == "target"
    assert outputs["score"] == 75.0 and outputs["checkpoint"]["name"] == "run2"


def test_benchmark_failures_map_to_the_seed_columns():
    failure = {
        "ticket": "I was charged twice",
        "queue": "billing",
        "model_output": "shipping",
        "reason": "no match",
    }
    assert loop_rounds.failure_row(failure, "ticket", "queue") == {
        "origin": "failure",
        "failed_prompt": "I was charged twice",
        "failed_reference": "billing",
        "failed_output": "shipping",
    }
    lm_eval = {"question": "q", "target": "a", "model_output": "b"}
    assert loop_rounds.failure_row(lm_eval)["failed_reference"] == "a"
    assert loop_rounds.failure_row({"model_output": "x"}) is None
    # Benchmark scored against the label name (or a structured answer's text): that is fed back.
    labelled = {
        "text": "great film",
        "label": 1,
        "model_output": "negative",
        "scored_reference": "positive",
    }
    assert loop_rounds.failure_row(labelled, "text", "label")["failed_reference"] == "positive"
    squad = {
        "question": "Capital of France?",
        "answers": {"text": ["Paris"], "answer_start": [0]},
        "model_output": "Lyon",
        "scored_reference": "Paris",
    }
    assert loop_rounds.failure_row(squad)["failed_reference"] == "Paris"


def test_the_real_benchmark_node_can_close_the_loop():
    d = loop_doc()
    d["nodes"][3]["type"] = "train.benchmark"
    parsed, errors = parse_document(d)
    assert not errors, errors
    plan = compile_flow(parsed)
    assert plan.feedback["loop"]["source"] == "bench"


def test_a_recipe_names_the_local_model_to_load_before_each_round():
    from core.flows.nodes.data import local_model

    recipe = {
        "model_providers": [{"name": "local", "is_local": True}, {"name": "api"}],
        "model_configs": [
            {
                "alias": "writer",
                "provider": "local",
                "model": "unsloth/Qwen3-1.7B-GGUF",
                "gguf_variant": "Q4_K_M",
            },
            {"alias": "unused", "provider": "local", "model": "other/model"},
        ],
        "columns": [{"name": "ticket", "column_type": "llm-text", "model_alias": "writer"}],
    }
    assert local_model(recipe) == {"repo": "unsloth/Qwen3-1.7B-GGUF", "variant": "Q4_K_M"}
    recipe["model_configs"][0]["provider"] = "api"
    assert local_model(recipe) is None


def test_a_recipe_step_validates_against_the_seed_it_runs_on(monkeypatch, tmp_path):
    from core.flows.compiler import RECIPE_PARAM
    from core.flows.nodes import data

    from .test_flows_recipe_groups import _compiled, _flow, _routes
    from .test_flows_recipes import _Ctx

    monkeypatch.setattr(data, "POLL_S", 0)
    seed = tmp_path / "seed.jsonl"
    seed.write_text('{"origin": "fresh"}\n')
    monkeypatch.setattr(data, "local_seed_path", lambda ref: str(seed))
    routes, _ = _routes(["job1"])
    checked = []

    def recording(method, path, params, body):
        if path == "/api/data-recipe/validate":
            checked.append(copy.deepcopy(body))
        return routes(method, path, params, body)

    ctx = _Ctx(
        recording,
        params = {RECIPE_PARAM: _compiled(_flow()), "rows": 1},
        inputs = {"seed": {"kind": "dataset", "source": "local", "ref": "loop-seed"}},
    )
    asyncio.run(data.recipe_group(ctx))
    assert checked[0]["recipe"]["seed_config"]["source"] == {
        "seed_type": "local",
        "path": str(seed),
    }


def test_a_loop_refuses_an_lm_eval_task_on_its_benchmark():
    d = loop_doc()
    d["nodes"][3]["type"] = "train.benchmark"
    d["nodes"][3]["params"]["task"] = "HellaSwag"
    parsed, errors = parse_document(d)
    assert not errors, errors
    with pytest.raises(CompileError, match = "set Benchmark's Task to Recipe held-out") as raised:
        compile_flow(parsed)
    # The feedback wire is drawn; only its task is wrong.
    assert not any("Draw the dashed" in i["message"] for i in raised.value.issues)


@run_async
async def test_round_two_continues_from_round_ones_checkpoint():
    eng, seen, _ = _engine()
    rs = await _run(eng, loop_doc(rounds = 3))
    assert rs.status == "completed", rs.error
    assert eng.starts[0] is None
    assert eng.starts[1]["name"] == "run1" and eng.starts[2]["name"] == "run2"
    rounds = rs.nodes["loop"].outputs["history"]["rounds"]
    assert [r["continued_from"] for r in rounds] == [None, "run1", "run2"]
    assert rs.nodes["loop"].outputs["history"]["continue"] is True


@run_async
async def test_continue_off_starts_every_round_from_the_base_model():
    eng, seen, _ = _engine()
    rs = await _run(eng, loop_doc(rounds = 2, continue_rounds = False))
    assert rs.status == "completed", rs.error
    assert eng.starts == [None, None]
    assert rs.nodes["loop"].outputs["history"]["continue"] is False


@run_async
async def test_a_loop_that_raises_fails_the_run():
    eng, _, _ = _engine()
    rs = await _run(eng, loop_doc(rounds = 2, target = "high"))
    assert rs.status == "failed"
    assert rs.nodes["loop"].status == "failed"
    assert "Loop failed" in rs.nodes["loop"].error


def test_fine_tune_continues_a_wired_checkpoint_as_an_adapter(monkeypatch):
    from core.flows.nodes import training

    from .test_flows_train_node import _Ctx as TrainCtx, _recipe_routes

    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = TrainCtx(_recipe_routes(started))
    ctx.inputs["start"] = {
        "kind": "checkpoint",
        "name": "run1",
        "base_model": "unsloth/Qwen3-0.6B",
        "output_dir": "/out/run1",
    }
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert started[0]["model_name"] == "/out/run1" and started[0]["model_format"] == "adapter"


def test_fine_tune_continues_the_checkpoint_fine_tune_emits(monkeypatch):
    # The carried value is Fine-tune's own output: a run name and no directory.
    from types import SimpleNamespace

    from core.flows.nodes import training

    from .test_flows_train_node import _Ctx as TrainCtx, _recipe_routes

    async def resolve(caller, name):
        assert name == "run1"
        return SimpleNamespace(path = "/out/run1", peft_type = "LORA")

    monkeypatch.setattr("studio_mcp.checkpoints.resolve", resolve)
    started = []
    ctx = TrainCtx(_recipe_routes(started))
    ctx.inputs["start"] = {"kind": "checkpoint", "name": "run1", "base_model": "unsloth/Qwen3-0.6B"}
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert started[0]["model_name"] == "/out/run1" and started[0]["model_format"] == "adapter"


def test_fine_tune_continues_a_full_finetune_checkpoint_as_a_plain_model(monkeypatch):
    from types import SimpleNamespace

    from core.flows.nodes import training

    from .test_flows_train_node import _Ctx as TrainCtx, _recipe_routes

    async def resolve(caller, name):
        return SimpleNamespace(path = "/out/full1", peft_type = None)

    monkeypatch.setattr("studio_mcp.checkpoints.resolve", resolve)
    started = []
    ctx = TrainCtx(_recipe_routes(started))
    ctx.inputs["start"] = {
        "kind": "checkpoint",
        "name": "full1",
        "base_model": "unsloth/Qwen3-0.6B",
    }
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert started[0]["model_name"] == "/out/full1" and "model_format" not in started[0]


@run_async
async def test_identical_seed_rows_reuse_one_upload_across_runs(monkeypatch):
    from core.flows.nodes import data

    monkeypatch.setattr(data, "local_seed_path", lambda ref: str(ref))
    eng, seen, _ = _engine()
    names = []
    upload = eng._upload_rows

    async def counted(ctx, rows, name):
        names.append(name)
        return await upload(ctx, rows, name)

    eng._upload_rows = counted
    on = await _run(eng, loop_doc(rounds = 2))
    off = await _run(eng, loop_doc(rounds = 2, feedback = False))
    assert on.status == off.status == "completed"
    # Round 1 is all fresh rows in both runs: one upload, the same seed dataset for both.
    assert names.count("loop-loop-round-1") == 1
    assert seen[0] == seen[2]
    split = lambda rs: rs.nodes["loop"].outputs["history"]["split"]
    assert split(on) and split(on) == split(off)


@run_async
async def test_the_best_checkpoint_is_the_scored_fine_tunes_not_another_in_the_loop():
    eng, _, _ = _engine(test_pass = [0.25, 0.75])
    train = eng.handlers["train.finetune"]

    async def either(ctx):
        found = await train(ctx)
        if ctx.node_id == "other":
            found["checkpoint"] = {**found["checkpoint"], "name": "unscored"}
        return found

    eng.handlers["train.finetune"] = either
    d = loop_doc(rounds = 2, target = 100)
    d["nodes"].insert(1, node("other", "train.finetune", group = "L"))
    d["edges"].append(edge("gen", "dataset", "other", "dataset"))
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert rs.nodes["loop"].outputs["checkpoint"]["name"] == "run2"
    # Each round's history names the start of the trainer it scored, not the other trainer's.
    rounds = rs.nodes["loop"].outputs["history"]["rounds"]
    assert rounds[1]["continued_from"] not in (None, "unscored")


def test_shared_boilerplate_in_test_rows_does_not_hide_probe_failures():
    shared = "Answer the following question carefully."
    test_rows = [
        {"instruction": shared, "input": "What is the capital of France today?", "output": "Paris"},
        {"instruction": shared, "input": "What is the capital of Spain today?", "output": "Madrid"},
    ]
    probe = {
        "instruction": shared,
        "input": "What is the capital of Italy today?",
        "output": "Rome",
    }
    leaked = {"instruction": shared, "input": "What is the capital of Spain today?", "output": "x"}
    assert loop_rounds.drop_test_rows([probe, leaked], test_rows) == [probe]


def test_an_alpaca_failure_is_fed_back_with_its_input():
    row = loop_rounds.failure_row(
        {
            "instruction": "Translate to French",
            "input": "cat",
            "output": "chat",
            "model_output": "dog",
        }
    )
    assert row["failed_prompt"] == "Translate to French\n\ncat"
    plain = loop_rounds.failure_row(
        {"instruction": "Say hi", "input": "", "output": "hi", "model_output": "x"}
    )
    assert plain["failed_prompt"] == "Say hi"


def test_a_test_row_repeated_in_the_split_still_bans_its_question():
    question = "What is the capital city of the country France?"
    test_rows = [{"q": question, "a": "Paris"}, {"q": question, "a": "Paris"}]
    assert loop_rounds.drop_test_rows([{"q": question, "model_output": "x"}], test_rows) == []


def test_cpt_continuing_a_4bit_checkpoint_still_loads_in_4bit(monkeypatch):
    # model_name is the adapter folder by now; the base it was trained on sets the precision.
    from core.flows.nodes import training

    from .test_flows_train_node import _Ctx as TrainCtx, _recipe_routes

    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = TrainCtx(_recipe_routes(started))
    ctx.params["training_type"] = "cpt"
    ctx.params["base_model"] = "unsloth/Qwen3-0.6B-bnb-4bit"
    ctx.inputs["start"] = {
        "kind": "checkpoint",
        "name": "run1",
        "base_model": "unsloth/Qwen3-0.6B-bnb-4bit",
        "output_dir": "/out/run1",
    }
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert started[0]["model_name"] == "/out/run1" and started[0]["load_in_4bit"] is True


def test_arithmetic_failures_whose_answers_differ_only_in_sign_stay_apart():
    failures = [
        {"prompt": "What is -5 + 4?", "reference": "-1", "output": "1"},
        {"prompt": "What is 5 - 4?", "reference": "1", "output": "-1"},
        # The same question with only its punctuation changed is still one failure.
        {"prompt": "what is -5 + 4", "reference": "-1", "output": "9"},
    ]
    kept = loop_rounds.distinct([loop_rounds.failure_row(f, "", "") for f in failures])
    assert [r["failed_reference"] for r in kept] == ["-1", "1"]


def test_a_single_alpaca_test_row_bans_only_its_own_question():
    shared = "Translate the following sentence to French."
    test_rows = [{"instruction": shared, "input": "The weather is lovely today.", "output": "x"}]
    probe = {"instruction": shared, "input": "Where is the train station?", "output": "y"}
    leaked = {"instruction": shared, "input": "The weather is lovely today.", "output": "z"}
    # A Benchmark failure names the question it asked: instruction plus input.
    asked = {
        "question": f"{shared}\n\nThe weather is lovely today.",
        "target": "t",
        "model_output": "o",
    }
    assert loop_rounds.drop_test_rows([probe, leaked, asked], test_rows) == [probe]


@run_async
async def test_on_a_mac_each_lora_round_trains_from_the_base_model(monkeypatch):
    from core.flows import engine as engine_mod

    # Apple Silicon's trainer refuses to continue a LoRA adapter: round 2 must not ask it to.
    monkeypatch.setattr(engine_mod, "_adapters_restart", lambda: True)
    eng, _, _ = _engine()
    rs = await _run(eng, loop_doc(rounds = 3))
    assert rs.status == "completed", rs.error
    assert eng.starts == [None, None, None]
    assert rs.nodes["loop"].outputs["history"]["continue"] is False


@run_async
async def test_a_test_score_with_errored_rows_neither_wins_nor_meets_the_target():
    eng, _, _ = _engine(test_pass = [1.0, 0.5])
    bench = eng.handlers["test.bench"]
    tests = []

    async def flaky(ctx):
        found = await bench(ctx)
        if ctx.inputs["dataset"]["split"] == "test":
            tests.append(1)
            # Round 1's test lost a row to an HTTP error, so its 100% is over the rows that answered.
            found["results"] = {"errored": 1 if len(tests) == 1 else 0}
        return found

    eng.handlers["test.bench"] = flaky
    rs = await _run(eng, loop_doc(rounds = 2, target = 100))
    assert rs.status == "completed", rs.error
    history = rs.nodes["loop"].outputs["history"]
    assert len(history["rounds"]) == 2 and history["rounds"][0]["test_errored"] == 1
    assert rs.nodes["loop"].outputs["checkpoint"]["name"] == "run2"


def test_a_rare_answer_label_in_the_test_split_does_not_hide_a_probe_failure():
    test_rows = [
        {
            "prompt": "My router keeps dropping the connection at night.",
            "label": "technical_support_escalation",
        },
        {"prompt": "How do I change the billing address on my account?", "label": "billing"},
    ]
    probe = {
        "prompt": "The app crashes whenever I open the settings page.",
        "label": "technical_support_escalation",
        "model_output": "billing",
    }
    leaked = {"prompt": "How do I change the billing address on my account?", "model_output": "x"}
    assert loop_rounds.drop_test_rows([probe, leaked], test_rows) == [probe]


def test_a_short_test_question_asked_again_never_reaches_the_seed():
    test_rows = [{"prompt": "Billed twice?", "response": "Refund the duplicate"}] * 3
    leaked = {"prompt": "Billed twice?", "response": "Refund the duplicate", "model_output": "x"}
    probe = {"prompt": "Card declined?", "response": "Retry", "model_output": "y"}
    assert loop_rounds.drop_test_rows([leaked, probe], test_rows) == [probe]


def test_the_loop_reads_benchmarks_max_rows_within_its_range():
    from core.flows.engine import _scorer_row_limit

    bench = SPECS_BY_TYPE["train.benchmark"]
    assert _scorer_row_limit(bench, {}) == 20
    assert _scorer_row_limit(bench, {"max_rows": 5}) == 5
    assert _scorer_row_limit(bench, {"max_rows": 10_000}) == 200
    assert _scorer_row_limit(SPECS_BY_TYPE["test.bench"], {}) is None


@run_async
async def test_each_split_is_cut_once_to_what_the_score_step_reads(monkeypatch):
    from core.flows.specs import P

    monkeypatch.setitem(
        SPECS_BY_TYPE,
        "test.bench",
        {
            **SPECS_BY_TYPE["test.bench"],
            "params": [P("max_rows", "int", 2, "Max rows", min = 1, max = 200)],
        },
    )
    eng, _, scored = _engine()
    rs = await _run(eng, loop_doc(rounds = 2, target = 100))
    assert rs.status == "completed", rs.error
    assert scored and all(len(rows) <= 2 for _, rows in scored)
    # Every round scores the same cut rows, and records that count.
    tests = [rows for split, rows in scored if split == "test"]
    assert all(rows == tests[0] for rows in tests)
    assert all(h["test_rows"] == 2 for h in rs.nodes["loop"].outputs["history"]["rounds"])


@run_async
async def test_a_score_over_part_of_the_test_split_neither_wins_nor_meets_the_target():
    eng, _, _ = _engine()
    real = eng.handlers["test.bench"]

    async def partial(ctx):
        found = await real(ctx)
        if ctx.inputs["dataset"]["split"] == "test":
            # Read only one of the test rows, and passed it.
            return {**found, "score": 1.0, "results": {"scored": 1, "errored": 0}}
        return found

    eng.handlers["test.bench"] = partial
    rs = await _run(eng, loop_doc(rounds = 2, target = 100))
    assert rs.status == "completed", rs.error
    history = rs.nodes["loop"].outputs["history"]
    assert history["stopped"] != "target"
    assert all(r.get("test_errored") for r in history["rounds"])


@run_async
async def test_later_rounds_train_without_the_frozen_test_questions():
    eng, _, scored = _engine()
    real = eng.handlers["train.finetune"]
    excluded = []

    async def train(ctx):
        excluded.append(ctx.inputs.get(loop_rounds.EXCLUDE_INPUT))
        return await real(ctx)

    eng.handlers["train.finetune"] = train
    rs = await _run(eng, loop_doc(rounds = 3, target = 100))
    assert rs.status == "completed", rs.error
    test = sorted({p for split, rows in scored if split == "test" for p in rows})
    # Round 1 makes the split; every later round is told its test questions.
    assert excluded[0] is None
    assert excluded[1:] and all(
        e == {"column": "", "system": "", "questions": test} for e in excluded[1:]
    )


def test_a_column_name_with_spaces_reads_the_same_column_benchmark_scored():
    row = {"ticket": "My payment was charged twice", "queue": "billing", "model_output": "x"}
    found = loop_rounds.failure_row(row, " ticket ", " queue ")
    assert found["failed_prompt"] == "My payment was charged twice"
    assert found["failed_reference"] == "billing"
    assert loop_rounds.questions([row], " ticket ") == ["My payment was charged twice"]


@run_async
async def test_a_recipe_step_with_no_saved_row_count_is_seeded_for_its_default():
    # An imported flow may omit rows: the step makes its catalog default (5), so the seed holds 5.
    eng, seen, _ = _engine()
    d = loop_doc(rounds = 2, target = 100, failure_percent = 25)
    gen = next(n for n in d["nodes"] if n["id"] == "gen")
    gen["params"].pop("rows")
    rs = await _run(eng, d)
    assert rs.status == "completed", rs.error
    assert [len(rows) for rows in seen] == [5, 5]


@run_async
async def test_the_loop_reads_questions_from_the_column_the_score_step_asked():
    eng, _, _ = _engine()
    real_bench, real_train = eng.handlers["test.bench"], eng.handlers["train.finetune"]
    excluded = []

    async def bench(ctx):
        # Benchmark's failures report the column it asked when its own setting is blank.
        found = await real_bench(ctx)
        return {**found, "failures": {"rows": found["failures"], "prompt_column": "prompt"}}

    async def train(ctx):
        excluded.append(ctx.inputs.get(loop_rounds.EXCLUDE_INPUT))
        return await real_train(ctx)

    eng.handlers["test.bench"], eng.handlers["train.finetune"] = bench, train
    rs = await _run(eng, loop_doc(rounds = 2, target = 100))
    assert rs.status == "completed", rs.error
    assert excluded[1]["column"] == "prompt" and excluded[1]["questions"]


def test_sharegpt_turns_are_read_as_benchmark_asked_them():
    row = {
        "prompt": [{"from": "human", "value": "What is the capital of France?"}],
        "completion": [{"from": "gpt", "value": "Paris"}],
        "model_output": "Lyon",
    }
    found = loop_rounds.failure_row(row)
    assert found["failed_prompt"] == "What is the capital of France?"
    assert found["failed_reference"] == "Paris"
    assert loop_rounds.questions([row]) == ["What is the capital of France?"]
    # The frozen test question keeps the same row out of training.
    assert (
        loop_rounds.without_questions(
            [row], {"column": "", "questions": loop_rounds.questions([row])}
        )
        == []
    )


def test_a_failure_seeds_the_question_benchmark_asked_but_is_known_by_its_row():
    failure = {
        "context": "The parcel is being held in customs.",
        "question": "Where is the parcel?",
        "answer": "customs",
        "model_output": "home",
        "scored_prompt": "The parcel is being held in customs.\nWhere is the parcel?",
    }
    found = loop_rounds.failure_row(failure)
    assert found["failed_prompt"] == failure["scored_prompt"]
    # A test row carries no scored question: both are known by the same column.
    test_row = {k: v for k, v in failure.items() if k not in ("model_output", "scored_prompt")}
    assert loop_rounds.questions([failure]) == loop_rounds.questions([test_row])


def test_a_mapped_question_is_known_by_all_its_columns():
    spec = {"columns": ["context", "question"], "join": "turns", "inputs": [], "labels": None}
    rows = [
        {"context": f"Ticket {i}: my card was charged twice", "question": "Which support queue?"}
        for i in range(3)
    ]
    frozen = loop_rounds.questions(rows[:1], spec)
    # Another ticket asking the same fixed question is not the frozen test row.
    exclude = {"column": spec, "questions": frozen}
    assert loop_rounds.without_questions(rows, exclude) == rows[1:]
    # A later round's own rows, without those columns, are never taken for it.
    fresh = [{"prompt": "x", "response": "y"}]
    assert loop_rounds.without_questions(fresh, exclude) == fresh


def test_a_scored_question_seeds_even_when_no_column_looks_like_one():
    failure = {
        "context": "Card charged twice",
        "ticket": "Refund please",
        "queue": "billing",
        "model_output": "sales",
        "scored_prompt": "user: Card charged twice\n\nuser: Refund please",
        "scored_reference": "billing",
    }
    found = loop_rounds.failure_row(failure)
    assert found["failed_prompt"] == failure["scored_prompt"]
    assert found["failed_reference"] == "billing"


@run_async
async def test_the_loop_knows_test_rows_by_the_mapped_columns_benchmark_reports():
    eng, _, _ = _engine()
    real_bench, real_train = eng.handlers["test.bench"], eng.handlers["train.finetune"]
    spec = {"columns": ["prompt"], "join": "lines", "inputs": [], "labels": None}
    excluded = []

    async def bench(ctx):
        found = await real_bench(ctx)
        return {**found, "failures": {"rows": found["failures"], "prompt_column": spec}}

    async def train(ctx):
        excluded.append(ctx.inputs.get(loop_rounds.EXCLUDE_INPUT))
        return await real_train(ctx)

    eng.handlers["test.bench"], eng.handlers["train.finetune"] = bench, train
    rs = await _run(eng, loop_doc(rounds = 2, target = 100))
    assert rs.status == "completed", rs.error
    assert excluded[1]["column"] == spec and excluded[1]["questions"]


def test_rows_sharing_a_question_are_told_apart_by_their_system_message():
    rows = [
        {"system": f"Reply with {c}", "prompt": "What label?"} for c in ("RED", "BLUE", "GREEN")
    ]
    frozen = loop_rounds.questions(rows[:1], "", "system")
    exclude = {"column": "", "system": "system", "questions": frozen}
    assert loop_rounds.without_questions(rows, exclude) == rows[1:]
    # The same row as a probe failure is still the test row, and is dropped.
    failure = {**rows[0], "model_output": "BLUE"}
    assert loop_rounds.drop_test_rows([failure], rows[:1], "", "system") == []
    # A conversation prompt's own system turn tells rows apart too.
    chats = [
        {
            "prompt": [
                {"role": "system", "content": f"Reply with {c}"},
                {"role": "user", "content": "What label?"},
            ]
        }
        for c in ("RED", "BLUE")
    ]
    assert (
        loop_rounds.without_questions(
            chats, {"column": "prompt", "questions": loop_rounds.questions(chats[:1], "prompt")}
        )
        == chats[1:]
    )


@run_async
async def test_the_loop_knows_test_rows_by_the_system_column_benchmark_reports():
    eng, _, _ = _engine()
    real_bench, real_train = eng.handlers["test.bench"], eng.handlers["train.finetune"]
    excluded = []

    async def bench(ctx):
        found = await real_bench(ctx)
        return {**found, "failures": {"rows": found["failures"], "system_column": "sys"}}

    async def train(ctx):
        excluded.append(ctx.inputs.get(loop_rounds.EXCLUDE_INPUT))
        return await real_train(ctx)

    eng.handlers["test.bench"], eng.handlers["train.finetune"] = bench, train
    rs = await _run(eng, loop_doc(rounds = 2, target = 100))
    assert rs.status == "completed", rs.error
    assert excluded[1]["system"] == "sys"


def test_several_system_columns_tell_rows_apart():
    rows = [
        {"policy": "Use the secret.", "secret": f"The word is {w}.", "prompt": "What is the word?"}
        for w in ("BLUE", "RED")
    ]
    system = ["policy", "secret"]
    exclude = {
        "column": "",
        "system": system,
        "questions": loop_rounds.questions(rows[:1], "", system),
    }
    assert loop_rounds.without_questions(rows, exclude) == rows[1:]


def test_a_prompt_s_assistant_prefill_tells_rows_apart():
    spec = {"kind": "column", "column": "prompt", "whole": False}
    rows = [
        {
            "prompt": [
                {"role": "user", "content": "Complete this sequence."},
                {"role": "assistant", "content": prefill},
            ],
            "answer": answer,
        }
        for prefill, answer in (("2, 4, 6,", "8"), ("3, 6, 9,", "12"))
    ]
    exclude = {"column": spec, "questions": loop_rounds.questions(rows[:1], spec)}
    assert loop_rounds.without_questions(rows, exclude) == rows[1:]
    # The failure seed keeps the prefill too.
    found = loop_rounds.failure_row({**rows[0], "model_output": "10"}, spec)
    assert "2, 4, 6," in found["failed_prompt"]

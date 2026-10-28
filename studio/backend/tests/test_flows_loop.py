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
HELD = [f"held-out support ticket number {i}: my invoice is wrong" for i in range(8)]


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
            node("train", "train.finetune", group = "L"),
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
        (lambda d: d["edges"].append(edge("train", "holdout", "gen", "seed")), "takes one connection"),
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
        (lambda d: d["edges"].pop(1), "so each round scores the model it just trained"),
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
    rows = loop_rounds.seed_rows([{"prompt": "a", "reference": "b", "output": "c"}], 4, 50)
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
        held = {"kind": "dataset", "source": "local", "ref": "held", "rows": len(HELD), "data": [{"prompt": p, "reference": "billing"} for p in HELD]}
        return {"checkpoint": {"kind": "checkpoint", "name": f"run{len(seen)}", "learned": sorted(learned)}, "holdout": held, "metrics": {}}

    async def bench(ctx):
        rows = ctx.inputs["dataset"]["data"]
        learned = set(ctx.inputs["checkpoint"]["learned"])
        scored.append((ctx.inputs["dataset"]["split"], [r["prompt"] for r in rows]))
        failed = [r for r in rows if r["prompt"] not in learned]
        if ctx.inputs["dataset"]["split"] == "test" and test_pass is not None:
            passed = test_pass[len([s for s in scored if s[0] == "test"]) - 1]
            return {"score": passed, "failures": []}
        if leak:
            failed = failed + [{"prompt": HELD[i], "reference": "billing", "output": "x"} for i in range(8)]
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
    rs = await _run(eng, loop_doc(rounds = 3, target = 100))
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
    failure = {"ticket": "I was charged twice", "queue": "billing", "model_output": "shipping", "reason": "no match"}
    assert loop_rounds.failure_row(failure, "ticket", "queue") == {
        "origin": "failure",
        "failed_prompt": "I was charged twice",
        "failed_reference": "billing",
        "failed_output": "shipping",
    }
    lm_eval = {"question": "q", "target": "a", "model_output": "b"}
    assert loop_rounds.failure_row(lm_eval)["failed_reference"] == "a"
    assert loop_rounds.failure_row({"model_output": "x"}) is None


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
            {"alias": "writer", "provider": "local", "model": "unsloth/Qwen3-1.7B-GGUF", "gguf_variant": "Q4_K_M"},
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
    assert checked[0]["recipe"]["seed_config"]["source"] == {"seed_type": "local", "path": str(seed)}



def test_a_loop_refuses_an_lm_eval_task_on_its_benchmark():
    d = loop_doc()
    d["nodes"][3]["type"] = "train.benchmark"
    d["nodes"][3]["params"]["task"] = "HellaSwag"
    parsed, errors = parse_document(d)
    assert not errors, errors
    with pytest.raises(CompileError, match = "set Benchmark's Task to Recipe held-out"):
        compile_flow(parsed)



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


def test_fine_tune_continues_a_wired_checkpoint_as_an_adapter(monkeypatch):
    from core.flows.nodes import training

    from .test_flows_train_node import _Ctx as TrainCtx, _recipe_routes

    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = TrainCtx(_recipe_routes(started))
    ctx.inputs["start"] = {"kind": "checkpoint", "name": "run1", "base_model": "unsloth/Qwen3-0.6B", "output_dir": "/out/run1"}
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert started[0]["model_name"] == "/out/run1" and started[0]["model_format"] == "adapter"


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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from pathlib import Path

import pytest

from core.flows.nodes import data, training
from core.flows.recipe_compile import RecipeCompileError, compile_recipe, normalise
from core.flows.recipe_import import COLUMN_TYPE, import_recipe, column_references
from core.flows.registry import FlowNodeError

LEARNING = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "features"
    / "data-recipes"
    / "learning-recipes"
)
RECIPES = sorted(LEARNING.glob("*.json"))


class _Ctx:
    def __init__(
        self,
        routes,
        params = None,
        inputs = None,
    ):
        self.routes = routes
        self.params = {"recipe_id": "r1", "mode": "full", "rows": 10, **(params or {})}
        self.inputs = inputs or {}
        self.calls = []
        self.hooks = []
        self.seed = 7
        self.caller = None

    async def call(
        self,
        method,
        path,
        json = None,
        params = None,
        files = None,
        **_,
    ):
        self.calls.append((method, path, params, json, files))
        return self.routes(method, path, params, json)

    async def ensure_model(self, kind, choice, options = None):
        self.loaded = choice

    def on_cancel(self, fn):
        self.hooks.append(fn)

    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def check_cancelled(self):
        pass


def _recipe_routes(job_ids, artifacts):
    started = iter(job_ids)
    saved = {"recipe": {"columns": [{"name": "q", "column_type": "expression", "expr": "x"}]}}

    def routes(method, path, params, body):
        if path == "/api/data-recipe/recipes/r1":
            return {"payload": saved}
        if path == "/api/data-recipe/jobs":
            return {"job_id": next(started)}
        if path.endswith("/status"):
            job = path.split("/")[-2]
            return {"status": "completed", "rows": 3, "artifact_path": artifacts.get(job)}
        if path.endswith("/dataset"):
            return {"dataset": [{"q": 1}], "total": 1}
        raise AssertionError(path)

    return routes


@pytest.fixture(autouse = True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(data, "POLL_S", 0)


def test_two_recipes_then_finetune_trains_on_the_first_recipes_parquet():
    artifacts = {
        "job1": "/x/assets/datasets/recipes/recipe_one",
        "job2": "/x/assets/datasets/recipes/recipe_two",
    }
    routes = _recipe_routes(["job1", "job2"], artifacts)
    first = asyncio.run(data.recipe(_Ctx(routes)))["dataset"]
    second = asyncio.run(data.recipe(_Ctx(routes)))["dataset"]
    assert first["ref"] == "recipes/recipe_one/parquet-files" and first["job_id"] == "job1"
    assert second["ref"] == "recipes/recipe_two/parquet-files"

    def train_routes(method, path, params, body):
        if path == "/api/train/start":
            raise RuntimeError(f"started with {body['local_datasets']}")
        raise AssertionError(f"the first recipe's rows were re-read through {path}")

    ctx = _Ctx(train_routes, params = {"base_model": "unsloth/tiny"}, inputs = {"dataset": first})
    with pytest.raises(RuntimeError, match = "recipes/recipe_one/parquet-files"):
        asyncio.run(training.finetune(ctx))


def test_a_preview_recipe_is_saved_while_its_rows_are_still_in_memory(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: "ref:preview")
    base = _recipe_routes(["job9"], {})

    def routes(method, path, params, body):
        if path == "/api/hub/datasets/upload":
            return {"stored_path": "/cache/recipe-job9.jsonl"}
        return base(method, path, params, body)

    out = asyncio.run(data.recipe(_Ctx(routes, params = {"mode": "preview"})))
    assert out["dataset"]["source"] == "local" and out["dataset"]["ref"] == "ref:preview"


def test_an_old_job_id_ref_still_trains_through_the_rows():
    assert not data.recipe_artifact("job1")
    assert data.recipe_artifact("recipes/recipe_one/parquet-files")
    assert not data.recipe_artifact("recipes/../etc/parquet-files")


def test_an_upstream_hub_dataset_becomes_the_recipes_seed():
    routes = _recipe_routes(["job1"], {"job1": "/r/recipe_a"})
    inner = routes

    def with_seed(method, path, params, body):
        if path == "/api/data-recipe/seed/inspect":
            assert body["dataset_name"] == "org/data" and body["split"] == "test"
            return {"resolved_path": "datasets/org/data/**/*.parquet"}
        return inner(method, path, params, body)

    seed = {"kind": "dataset", "source": "hf", "ref": "org/data", "split": "test"}
    ctx = _Ctx(with_seed, inputs = {"seed": seed})
    asyncio.run(data.recipe(ctx))
    sent = [c[3] for c in ctx.calls if c[1] == "/api/data-recipe/jobs"][0]["recipe"]["seed_config"]
    assert sent == {
        "sampling_strategy": "ordered",
        "source": {"seed_type": "hf", "path": "datasets/org/data/**/*.parquet"},
    }


def test_a_recipe_dataset_seed_reads_its_parquet(tmp_path, monkeypatch):
    folder = tmp_path / "parquet-files"
    folder.mkdir()
    (folder / "batch_00000.parquet").write_bytes(b"x")
    monkeypatch.setattr("utils.paths.resolve_dataset_path", lambda ref: folder)
    assert data.local_seed_path("recipes/r/parquet-files") == f"{folder}/*.parquet"
    monkeypatch.setattr("utils.paths.resolve_dataset_path", lambda ref: tmp_path / "gone")
    with pytest.raises(FlowNodeError, match = "no longer available"):
        data.local_seed_path("recipes/gone/parquet-files")


def test_provider_env_names_survive_but_secrets_do_not():
    body = data.declarative_recipe(
        {
            "recipe": {
                "columns": [],
                "model_providers": [
                    {
                        "name": "openai",
                        "api_key": "sk-live",
                        "api_key_env": "OPENAI_API_KEY",
                        "extra_headers": {"Authorization": "Bearer x"},
                    }
                ],
            }
        }
    )
    provider = body["model_providers"][0]
    assert provider == {"name": "openai", "api_key_env": "OPENAI_API_KEY"}


@pytest.mark.parametrize("path", RECIPES, ids = [p.stem for p in RECIPES])
def test_every_learning_recipe_round_trips(path):
    payload = json.loads(path.read_text())
    graph = import_recipe(payload)
    assert graph["group"]["kind"] == "recipe"
    columns = [n for n in graph["nodes"] if n["type"] == COLUMN_TYPE]
    assert len(columns) == len(payload["recipe"]["columns"])
    assert [n["type"] for n in graph["nodes"]].count("recipe.generate") == 1
    assert compile_recipe(graph) == normalise(payload)


def test_there_are_nine_learning_recipes():
    assert len(RECIPES) == 9


def test_reference_edges_follow_the_prompts():
    payload = {
        "recipe": {
            "columns": [
                {"name": "topic", "column_type": "sampler", "sampler_type": "category"},
                {"name": "q", "column_type": "llm-text", "prompt": "Ask about {{ topic }}"},
            ]
        }
    }
    graph = import_recipe(payload)
    assert [(e["source"], e["target"]) for e in graph["edges"]] == [
        ("recipe_col_1", "recipe_col_2")
    ]


def test_compile_refuses_loops_and_duplicate_names():
    loop = {
        "recipe": {
            "columns": [
                {"name": "a", "column_type": "expression", "expr": "{{ b }}"},
                {"name": "b", "column_type": "expression", "expr": "{{ a }}"},
            ]
        }
    }
    with pytest.raises(RecipeCompileError, match = "loop"):
        compile_recipe(import_recipe(loop))
    twice = import_recipe(
        {"recipe": {"columns": [{"name": "a", "column_type": "expression", "expr": "1"}] * 2}}
    )
    with pytest.raises(RecipeCompileError, match = "share the name"):
        compile_recipe(twice)


def test_every_column_an_expression_reads_is_a_dependency():
    cols = {"seed", "x", "y", "when", "gap", "level"}
    expr = {"name": "x", "expr": "{{ seed + y }} {% if level > 2 %}hi{% endif %}"}
    assert column_references(expr, cols) == ["seed", "y", "level"]
    timedelta = {"name": "gap", "params": {"reference_column_name": "when", "dt_min": 1}}
    assert column_references(timedelta, cols) == ["when"]
    loop = {
        "recipe": {
            "columns": [
                {"name": "seed", "column_type": "expression", "expr": "1"},
                {"name": "x", "column_type": "expression", "expr": "{{ seed + y }}"},
                {"name": "y", "column_type": "expression", "expr": "{{ x }}"},
            ]
        }
    }
    with pytest.raises(RecipeCompileError, match = "loop"):
        compile_recipe(import_recipe(loop))


def test_a_managed_account_never_sends_a_provider_env_name(monkeypatch):
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: True)
    body = data.declarative_recipe(
        {"columns": [], "model_providers": [{"name": "openai", "api_key_env": "OPENAI_API_KEY"}]}
    )
    assert body["model_providers"] == [{"name": "openai"}]


def test_a_hub_seed_fails_early_when_the_job_would_refuse_it(monkeypatch):
    seed = {"kind": "dataset", "source": "hf", "ref": "imdb"}
    ctx = _Ctx(_recipe_routes(["job1"], {}), inputs = {"seed": seed})
    with pytest.raises(FlowNodeError, match = "owner/imdb"):
        asyncio.run(data.seed_config(ctx, seed, None))
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: True)
    with pytest.raises(FlowNodeError, match = "Hugging Face token"):
        asyncio.run(data.seed_config(ctx, {**seed, "ref": "org/data"}, None))
    assert not ctx.calls


def test_attributes_filters_and_loop_names_are_not_columns():
    cols = {"topic", "field", "lower", "rows", "r"}
    column = {
        "name": "x",
        "prompt": "{{ topic.field | lower }} {% for r in rows %}{{ r }}{% endfor %}",
    }
    assert column_references(column, cols) == ["topic", "rows"]


def test_a_conditional_sampler_reads_its_rule_columns():
    sampler = {
        "name": "plan",
        "column_type": "sampler",
        "sampler_type": "subcategory",
        "params": {"category": "tier", "values": {}},
        "conditional_params": {"region == 'US'": {"category": "zone", "values": {}}},
    }
    assert column_references(sampler, {"tier", "region", "zone"}) == ["region", "tier", "zone"]
    loop = {
        "recipe": {
            "columns": [
                {
                    **sampler,
                    "params": {"values": ["a"]},
                    "conditional_params": {"age > 3": {"values": ["b"]}},
                },
                {"name": "age", "column_type": "expression", "expr": "{{ plan }}"},
            ]
        }
    }
    with pytest.raises(RecipeCompileError):
        compile_recipe(import_recipe(loop))

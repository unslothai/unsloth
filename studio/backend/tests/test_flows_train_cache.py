# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine
from core.flows.nodes import training

from ._flows_helpers import doc, edge, node, run_async


@pytest.fixture
def world(monkeypatch):
    """A Hub revision per dataset and the run folders Studio still has."""
    from studio_mcp import checkpoints

    state = {"sha": "rev1", "folders": set(), "trained": [], "asked": []}

    def revision(dataset, token):
        if state["sha"] is None:
            raise OSError("offline")
        return state["sha"]

    async def resolve(caller, name):
        if name not in state["folders"]:
            raise RuntimeError("gone")
        return name

    monkeypatch.setattr(training, "_hub_revision", revision)
    monkeypatch.setattr(checkpoints, "resolve", resolve)
    return state


def _engine(state) -> FlowEngine:
    async def hub(ctx):
        return {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/data", "split": "train"}}

    async def tune(ctx):
        name = f"run-{len(state['trained']) + 1}"
        state["trained"].append(name)
        state["folders"].add(name)
        return {"checkpoint": {"kind": "checkpoint", "name": name}, "metrics": {}, "holdout": None}

    async def chat(ctx):
        state["asked"].append(ctx.params.get("prompt"))
        return {"text": ctx.inputs["checkpoint"]["name"]}

    eng = FlowEngine()
    eng.handlers = {"data.hf_dataset": hub, "train.finetune": tune, "train.chat_checkpoint": chat}
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


def _flow(prompt = "q1", **tune):
    return doc(
        [
            node("data", "data.hf_dataset", repo = "org/data"),
            node("tune", "train.finetune", base_model = "unsloth/Qwen3-0.6B", **tune),
            node("chat", "train.chat_checkpoint", prompt = prompt),
        ],
        [
            edge("data", "dataset", "tune", "dataset"),
            edge("tune", "checkpoint", "chat", "checkpoint"),
        ],
    )


async def _run(eng, d, **options):
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = options)
    await asyncio.wait_for(rs.task, timeout = 20)
    assert rs.status == "completed", rs.error
    return rs


@run_async
async def test_a_downstream_change_reuses_the_adapter(world):
    eng = _engine(world)
    await _run(eng, _flow("q1"))
    rs = await _run(eng, _flow("q2"))
    assert world["trained"] == ["run-1"] and world["asked"] == ["q1", "q2"]
    assert rs.nodes["tune"].status == "cached"


@run_async
async def test_retrain_settings_data_and_lost_folders_train_again(world):
    eng = _engine(world)
    await _run(eng, _flow())
    await _run(eng, _flow(), force_nodes = ["tune"])  # Re-roll / Retrain
    assert world["trained"] == ["run-1", "run-2"]
    await _run(eng, _flow(max_steps = 30))  # a training setting
    assert len(world["trained"]) == 3
    world["sha"] = "rev2"  # the Hub dataset changed
    await _run(eng, _flow(max_steps = 30))
    assert len(world["trained"]) == 4
    world["folders"].clear()  # the adapter folder was deleted
    await _run(eng, _flow(max_steps = 30))
    assert len(world["trained"]) == 5
    await _run(eng, _flow(max_steps = 30))
    assert len(world["trained"]) == 5


@run_async
async def test_an_unreachable_hub_trains_instead_of_guessing(world):
    eng = _engine(world)
    world["sha"] = None
    await _run(eng, _flow())
    await _run(eng, _flow())
    assert len(world["trained"]) == 2


def test_local_datasets_hash_their_bytes(tmp_path, monkeypatch):
    folder = tmp_path / "parquet-files"
    folder.mkdir()
    (folder / "a.parquet").write_bytes(b"one")
    first = training._content_hash(f"{folder}/*.parquet")
    (folder / "a.parquet").write_bytes(b"two")
    assert training._content_hash(f"{folder}/*.parquet") != first


@run_async
async def test_a_reward_saved_again_under_its_name_trains_again(world, monkeypatch):
    from core.training import rewards

    rule = {"answer": "exact"}
    monkeypatch.setattr(rewards, "get_reward", lambda name: {"kind": "rule", "rule": dict(rule, name = name)})
    eng = _engine(world)
    await _run(eng, _flow(objective = "grpo"))
    await _run(eng, _flow(objective = "grpo"))
    assert len(world["trained"]) == 1
    rule["answer"] = "numeric"
    await _run(eng, _flow(objective = "grpo"))
    assert len(world["trained"]) == 2

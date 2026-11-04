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

    state = {"sha": "rev1", "model_sha": "m1", "folders": set(), "trained": [], "asked": []}

    def revision(dataset, token):
        if state["sha"] is None:
            raise OSError("offline")
        return state["sha"]

    async def resolve(caller, name):
        if name not in state["folders"]:
            raise RuntimeError("gone")
        return name

    def model_revision(repo, token):
        if state["model_sha"] is None:
            raise OSError("offline")
        return state["model_sha"]

    monkeypatch.setattr(training, "_hub_revision", revision)
    monkeypatch.setattr(training, "_hub_model_revision", model_revision)
    monkeypatch.setattr(training, "_hub_adapter_base", lambda repo, revision, token: None)
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
            node("tune", "train.finetune", **{"base_model": "unsloth/Qwen3-0.6B", **tune}),
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
    monkeypatch.setattr(
        rewards, "get_reward", lambda name: {"kind": "rule", "rule": dict(rule, name = name)}
    )
    eng = _engine(world)
    await _run(eng, _flow(objective = "grpo"))
    await _run(eng, _flow(objective = "grpo"))
    assert len(world["trained"]) == 1
    rule["answer"] = "numeric"
    await _run(eng, _flow(objective = "grpo"))
    assert len(world["trained"]) == 2


@run_async
async def test_new_weights_at_a_local_base_model_path_train_again(world, tmp_path):
    import os

    base = tmp_path / "my-model"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"old")
    eng = _engine(world)
    await _run(eng, _flow(base_model = str(base)))
    await _run(eng, _flow(base_model = str(base)))
    assert len(world["trained"]) == 1
    (base / "model.safetensors").write_bytes(b"new weights")
    os.utime(base / "model.safetensors", ns = (1, 1))
    await _run(eng, _flow(base_model = str(base)))
    assert len(world["trained"]) == 2


@run_async
async def test_new_weights_pushed_to_a_hub_base_model_train_again(world):
    eng = _engine(world)
    await _run(eng, _flow())
    await _run(eng, _flow())
    assert len(world["trained"]) == 1
    world["model_sha"] = "m2"
    await _run(eng, _flow())
    assert len(world["trained"]) == 2
    # A base revision the Hub cannot confirm trains again rather than guessing.
    world["model_sha"] = None
    await _run(eng, _flow())
    assert len(world["trained"]) == 3


@run_async
async def test_a_continued_adapter_trains_again_when_the_base_under_it_changes(world, tmp_path):
    import json
    import os

    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"old")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    eng = _engine(world)
    await _run(eng, _flow(base_model = str(adapter)))
    await _run(eng, _flow(base_model = str(adapter)))
    assert len(world["trained"]) == 1
    (base / "model.safetensors").write_bytes(b"new weights")
    os.utime(base / "model.safetensors", ns = (1, 1))
    await _run(eng, _flow(base_model = str(adapter)))
    assert len(world["trained"]) == 2
    # A Hub base under the adapter: a new revision trains again too.
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    await _run(eng, _flow(base_model = str(adapter)))
    await _run(eng, _flow(base_model = str(adapter)))
    assert len(world["trained"]) == 3
    world["model_sha"] = "m2"
    await _run(eng, _flow(base_model = str(adapter)))
    assert len(world["trained"]) == 4


@run_async
async def test_a_new_revision_of_the_4bit_mirror_the_loader_fetches_trains_again(
    world, monkeypatch
):
    from utils.models import unsloth_mirror

    revisions = {"unsloth/Qwen3-0.6B": "m1", "unsloth/Qwen3-0.6B-unsloth-bnb-4bit": "q1"}
    monkeypatch.setattr(training, "_hub_model_revision", lambda repo, token: revisions[repo])
    monkeypatch.setattr(
        unsloth_mirror,
        "unsloth_public_mirror",
        lambda name, mode: f"{name}-unsloth-bnb-4bit" if mode else None,
    )
    eng = _engine(world)
    await _run(eng, _flow())
    await _run(eng, _flow())
    assert len(world["trained"]) == 1
    revisions["unsloth/Qwen3-0.6B-unsloth-bnb-4bit"] = "q2"
    await _run(eng, _flow())
    assert len(world["trained"]) == 2
    # A 16-bit run never fetches the 4-bit mirror, so its changes do not retrain it.
    await _run(eng, _flow(load_in_4bit = False))
    revisions["unsloth/Qwen3-0.6B-unsloth-bnb-4bit"] = "q3"
    await _run(eng, _flow(load_in_4bit = False))
    assert len(world["trained"]) == 3


@run_async
async def test_a_hub_lora_trains_again_when_the_base_it_names_changes(world, monkeypatch):
    revisions = {"me/my-lora": "a1", "org/base": "b1"}
    monkeypatch.setattr(training, "_hub_model_revision", lambda repo, token: revisions[repo])
    monkeypatch.setattr(
        training,
        "_hub_adapter_base",
        lambda repo, revision, token: "org/base" if repo == "me/my-lora" else None,
    )
    eng = _engine(world)
    await _run(eng, _flow(base_model = "me/my-lora"))
    await _run(eng, _flow(base_model = "me/my-lora"))
    assert len(world["trained"]) == 1
    revisions["org/base"] = "b2"
    await _run(eng, _flow(base_model = "me/my-lora"))
    assert len(world["trained"]) == 2


@run_async
async def test_a_base_model_picked_by_inventory_handle_is_fingerprinted_through_it(
    world, tmp_path, monkeypatch
):
    import os

    from hub.utils import host_paths

    base = tmp_path / "my-model"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"old")
    monkeypatch.setattr(
        host_paths, "resolve_host_path_reference", lambda v: str(base) if v == "ref:mine" else None
    )
    eng = _engine(world)
    await _run(eng, _flow(base_model = "ref:mine"))
    await _run(eng, _flow(base_model = "ref:mine"))
    assert len(world["trained"]) == 1
    (base / "model.safetensors").write_bytes(b"new weights")
    os.utime(base / "model.safetensors", ns = (1, 1))
    await _run(eng, _flow(base_model = "ref:mine"))
    assert len(world["trained"]) == 2
    # A handle that no longer resolves cannot be told apart, so it trains again.
    await _run(eng, _flow(base_model = "ref:gone"))
    await _run(eng, _flow(base_model = "ref:gone"))
    assert len(world["trained"]) == 4


@run_async
async def test_a_flow_parameter_read_only_downstream_does_not_retrain(world):
    eng = _engine(world)
    await _run(eng, _flow(), params = {"question": "First question"})
    await _run(eng, _flow(), params = {"question": "First question"})
    await _run(eng, _flow(), params = {"question": "Different question"})
    assert len(world["trained"]) == 1


@run_async
async def test_a_changed_module_in_a_local_models_subfolder_trains_again(world, tmp_path):
    import os

    base = tmp_path / "embedder"
    (base / "2_Dense").mkdir(parents = True)
    (base / "model.safetensors").write_bytes(b"w")
    (base / "2_Dense" / "model.safetensors").write_bytes(b"dense")
    eng = _engine(world)
    await _run(eng, _flow(base_model = str(base)))
    await _run(eng, _flow(base_model = str(base)))
    assert len(world["trained"]) == 1
    (base / "2_Dense" / "model.safetensors").write_bytes(b"new dense weights")
    os.utime(base / "2_Dense" / "model.safetensors", ns = (1, 1))
    await _run(eng, _flow(base_model = str(base)))
    assert len(world["trained"]) == 2


def test_rows_that_name_image_files_are_never_reused(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = tmp_path / "vision.jsonl"
    rows.write_text('{"image": "/data/cat.png", "text": "a cat"}\n')
    # Training reads the PNG, which the rows' hash never sees: train again rather than reuse.
    assert training._content_hash(str(rows)) is None
    folder = tmp_path / "pq"
    folder.mkdir()
    pd.DataFrame([{"image": "photos/Cat.JPG", "text": "a cat"}]).to_parquet(folder / "a.parquet")
    assert training._content_hash(f"{folder}/*.parquet") is None
    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"q": "What is a png file?", "a": "An image format."}\n')
    assert training._content_hash(str(plain))


def test_parquet_image_columns_that_name_files_are_never_reused(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = tmp_path / "pq"
    folder.mkdir()
    image = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({"image": pa.array([{"bytes": None, "path": "sample.png"}], type = image)})
    pq.write_table(table, folder / "a.parquet")
    assert training._content_hash(f"{folder}/*.parquet") is None
    embedded = pa.table({"image": pa.array([{"bytes": b"\x89PNG", "path": None}], type = image)})
    pq.write_table(embedded, folder / "a.parquet")
    assert training._content_hash(f"{folder}/*.parquet")


def test_a_relative_local_base_model_is_fingerprinted(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.chdir(tmp_path)
    (tmp_path / "my-model").mkdir()
    (tmp_path / "my-model" / "model.safetensors").write_bytes(b"one")

    class Ctx:
        caller = None
        params: dict = {}

    first = asyncio.run(training._model_material(Ctx(), "./my-model"))
    assert first and first["base_model_files"]
    (tmp_path / "my-model" / "model.safetensors").write_bytes(b"three")
    assert asyncio.run(training._model_material(Ctx(), "./my-model")) != first
    # A relative path that is not there cannot be identified: train again.
    assert asyncio.run(training._model_material(Ctx(), "./gone")) is None


def test_a_repo_named_folder_on_disk_is_fingerprinted_not_looked_up(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "model").mkdir(parents = True)
    (tmp_path / "owner" / "model" / "model.safetensors").write_bytes(b"one")
    monkeypatch.setattr(
        training, "_hub_model_revision", lambda *a: pytest.fail("looked up the Hub")
    )

    class Ctx:
        caller = None
        params: dict = {}

    first = asyncio.run(training._model_material(Ctx(), "owner/model"))
    (tmp_path / "owner" / "model" / "model.safetensors").write_bytes(b"three")
    assert (
        first["base_model_files"]
        and asyncio.run(training._model_material(Ctx(), "owner/model")) != first
    )


def test_an_on_device_pick_keys_on_the_snapshot_it_trains_from(tmp_path, monkeypatch):
    import asyncio

    from core.training import training as core_training

    async def material(ctx, *args):
        return {}

    monkeypatch.setattr(training, "dataset_material", material)
    monkeypatch.setattr(training, "_model_material", material)
    snapshots = {"now": str(tmp_path / "snapshots" / "aaa")}
    monkeypatch.setattr(
        core_training, "resolve_training_model_load_target", lambda values: snapshots["now"]
    )
    cache = {"known_cached": True, "local_path": str(tmp_path)}

    class Ctx:
        caller = None
        inputs: dict = {}
        params = {"base_model": "owner/model", "base_model_cache": cache}

    first = asyncio.run(training.finetune_cache_material(Ctx()))
    assert first["base_model_snapshot"] == "aaa"
    # The cache moved to another downloaded snapshot: train again.
    snapshots["now"] = str(tmp_path / "snapshots" / "bbb")
    assert asyncio.run(training.finetune_cache_material(Ctx()))["base_model_snapshot"] == "bbb"
    # The copy is gone and training falls back to the Hub: also a different key.
    snapshots["now"] = "owner/model"
    assert asyncio.run(training.finetune_cache_material(Ctx()))["base_model_snapshot"] is None
    # No On device pick: the key is unchanged from before.
    Ctx.params = {"base_model": "owner/model"}
    assert "base_model_snapshot" not in asyncio.run(training.finetune_cache_material(Ctx()))


def test_a_local_vision_set_trains_again_since_its_images_are_not_hashed(tmp_path, monkeypatch):
    import asyncio

    async def material(ctx, *args):
        return {"rows": "same"}

    monkeypatch.setattr(training, "dataset_material", material)
    monkeypatch.setattr(training, "_model_material", material)

    class Ctx:
        caller = None
        inputs = {"dataset": {"kind": "dataset", "source": "local", "ref": "uploads/a/scans.jsonl"}}
        params = {"base_model": "owner/model", "task": "vision"}

    # scan.jfif replaced in place leaves the rows the same: nothing to key on, so no reuse.
    assert asyncio.run(training.finetune_cache_material(Ctx())) is None
    Ctx.params = {**Ctx.params, "task": "audio"}
    assert asyncio.run(training.finetune_cache_material(Ctx())) is None
    # A Hub vision set is keyed on its revision, which covers the images it holds; text is unchanged.
    Ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/scans"}}
    assert asyncio.run(training.finetune_cache_material(Ctx())) is not None
    Ctx.inputs = {"dataset": {"kind": "dataset", "source": "local", "ref": "uploads/a/t.jsonl"}}
    Ctx.params = {**Ctx.params, "task": "text"}
    assert asyncio.run(training.finetune_cache_material(Ctx())) is not None

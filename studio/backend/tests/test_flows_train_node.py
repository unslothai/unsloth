# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest

from core.flows.nodes import training
from core.flows.registry import FlowNodeError
from core.flows.specs import SPECS_BY_TYPE


class _Ctx:
    def __init__(self, routes, **params):
        self.routes, self.calls, self.partials = routes, [], []
        self.inputs = {"dataset": {"kind": "dataset", "source": "recipe", "ref": "job1"}}
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.finetune"]["params"]}
        self.params.update(params)
        self.seed, self.caller = 1, object()

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

    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def partial(self, port, value, **k):
        self.partials.append((port, value))


ROWS = [{"q": f"q{i}", "a": f"a{i}"} for i in range(20)]


def _recipe_routes(started):
    def routes(method, path, params, body):
        if path.endswith("/jobs/job1/dataset"):
            off = params["offset"]
            return {"dataset": ROWS[off : off + 500], "total": len(ROWS)}
        if path == "/api/hub/datasets/upload":
            return {"stored_path": f"/cache/{len(started)}.jsonl"}
        if path == "/api/train/start":
            started.append(body)
            raise RuntimeError("started")
        raise AssertionError(path)

    return routes


def _uploads(ctx):
    return {
        c[4]["file"][0]: [json.loads(line) for line in c[4]["file"][1].decode().splitlines()]
        for c in ctx.calls
        if c[1] == "/api/hub/datasets/upload"
    }


def test_holdout_rows_never_reach_training_and_match_the_recipe_card_split(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, prompt_column = "q", response_column = "a")
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    files = _uploads(ctx)
    train = next(rows for name, rows in files.items() if name.endswith("-train.jsonl"))
    held = next(rows for name, rows in files.items() if name.endswith("-holdout.jsonl"))
    assert len(train) == 18 and len(held) == 2
    assert not {r["q"] for r in train} & {r["q"] for r in held}
    ids = training.holdout_row_ids(20, 10, 3407)
    assert [r["q"] for r in held] == [f"q{i}" for i in ids]
    assert ids == training.holdout_row_ids(20, 10, 3407)
    body = started[0]
    assert body["custom_format_mapping"] == {"q": "user", "a": "assistant"}
    assert len(body["local_datasets"]) == 1 and "hf_dataset" not in body


def test_no_holdout_keeps_the_old_path(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started))
    with pytest.raises(RuntimeError):
        asyncio.run(training.finetune(ctx))
    assert len(_uploads(ctx)) == 1 and "custom_format_mapping" not in started[0]


def test_one_column_alone_is_refused():
    ctx = _Ctx(_recipe_routes([]), prompt_column = "q")
    with pytest.raises(FlowNodeError, match = "both"):
        asyncio.run(training.finetune(ctx))


def test_holdout_refuses_a_dataset_too_small_to_split(monkeypatch):
    monkeypatch.setattr(training, "holdout_row_ids", lambda *a: [])
    ctx = _Ctx(_recipe_routes([]), holdout_percent = 10)
    with pytest.raises(FlowNodeError, match = "too few"):
        asyncio.run(training.finetune(ctx))


def test_holdout_row_ids_keep_at_least_one_row_and_never_all():
    assert training.holdout_row_ids(1, 50, 1) == []
    assert len(training.holdout_row_ids(3, 1, 1)) == 1
    assert len(training.holdout_row_ids(2, 50, 1)) == 1
    assert training.holdout_row_ids(100, 0, 1) == []


def test_loss_history_is_downsampled_for_the_node():
    assert training.downsample(list(range(10))) == list(range(10))
    small = training.downsample(list(range(1000)), 50)
    assert len(small) == 50 and small[0] == 0 and small[-1] == 999


def test_live_metrics_stream_to_the_node_as_a_partial(monkeypatch):
    import studio_mcp.tools.jobs as jobs

    async def run_folder(caller, out):
        return "run-1", None

    monkeypatch.setattr(jobs, "_run_folder", run_folder)
    monkeypatch.setattr(training, "POLL_S", 0)
    ctx = _Ctx(None)
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/ds"}}
    ctx.on_cancel = lambda fn: None
    ctx.check_cancelled = lambda: None
    polls = iter(["training", "completed"])

    def routes(method, path, params, body):
        if path == "/api/train/start":
            return {"job_id": "j1"}
        if path == "/api/train/status":
            phase = next(polls)
            return {
                "job_id": "j1",
                "training_phase": phase,
                "phase": phase,
                "details": {"step": 2, "total_steps": 4, "loss": 1.5, "output_dir": "/x/run-1"},
            }
        if path == "/api/train/metrics":
            return {"loss_history": [2.0, 1.5]}
        raise AssertionError(path)

    ctx.routes = routes
    out = asyncio.run(training.finetune(ctx))
    assert ctx.partials and ctx.partials[0][0] == "metrics"
    assert ctx.partials[0][1]["loss_history"] == [2.0, 1.5]
    assert out["metrics"]["loss_history"] == [2.0, 1.5]
    assert out["holdout"] is None
    assert out["checkpoint"]["name"] == "run-1" and "output_dir" not in out["checkpoint"]


def test_holdout_splits_only_the_sliced_rows(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, dataset_slice_end = 9)
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    files = _uploads(ctx)
    train = next(rows for name, rows in files.items() if name.endswith("-train.jsonl"))
    held = next(rows for name, rows in files.items() if name.endswith("-holdout.jsonl"))
    assert len(train) == 9 and len(held) == 1
    assert {r["q"] for r in train + held} == {f"q{i}" for i in range(10)}
    assert "dataset_slice_end" not in started[0]


def test_holdout_reads_hub_datasets_with_the_requests_token(monkeypatch):
    seen = {}

    def load_rows(
        dataset,
        token,
        end,
        start = 0,
    ):
        seen["token"] = token
        return ROWS

    monkeypatch.setattr(training, "_load_rows", load_rows)
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: False)
    ctx = _Ctx(_recipe_routes([]), holdout_percent = 10)
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/gated"}}
    ctx.caller = type("C", (), {"hf_token": "hf_req"})()
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert seen["token"] == "hf_req"


def test_grpo_sends_objective_rewards_and_roles(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(
        _recipe_routes(started),
        objective = "grpo",
        prompt_column = "q",
        answer_column = "a",
        num_generations = 6,
        rewards = [{"name": "exact-answer", "weight": 2}, {"name": " ", "weight": 1}],
    )
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    body = started[0]
    assert body["objective"] == "grpo" and body["grpo_num_generations"] == 6
    assert body["grpo_rewards"] == [{"name": "exact-answer", "weight": 2.0}]
    assert body["custom_format_mapping"] == {"q": "prompt", "a": "answer"}
    assert "<answer>" in body["rl_system_prompt"]


def test_dpo_maps_chosen_and_rejected_and_ignores_the_response_column(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(
        _recipe_routes(started),
        objective = "dpo",
        response_column = "unused",
        chosen_column = "q",
        rejected_column = "a",
        beta = 0.2,
    )
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    body = started[0]
    assert body["objective"] == "dpo" and body["rl_beta"] == 0.2
    assert body["custom_format_mapping"] == {"q": "chosen", "a": "rejected"}
    assert "grpo_rewards" not in body


def test_grpo_without_rewards_is_refused():
    ctx = _Ctx(_recipe_routes([]), objective = "grpo", rewards = [])
    with pytest.raises(FlowNodeError, match = "reward"):
        asyncio.run(training.finetune(ctx))


def test_reward_ceiling_is_the_best_weighted_sum():
    assert (
        training.reward_ceiling(
            [{"name": "exact-answer", "weight": 1.0}, {"name": "soft-xml-format", "weight": 1.0}]
        )
        == 2.5
    )
    # A negative weight's best is its rule's lowest score.
    assert training.reward_ceiling([{"name": "numeric-close", "weight": -1.0}]) == 2.0
    assert training.reward_ceiling([{"name": "no-such-reward", "weight": 1.0}]) is None


def test_reward_history_reads_the_status_rl_log():
    payload = {
        "metric_history": {
            "rl": [{"step": 1, "reward": 0.25}, {"step": 2, "kl": 0.1}, {"step": 3, "reward": 1.5}]
        }
    }
    assert training.reward_history(payload) == [0.25, 1.5]
    assert training.reward_metrics([0.25, 1.5], 2.5) == {
        "reward": 1.5,
        "reward_start": 0.25,
        "reward_history": [0.25, 1.5],
        "reward_max": 2.5,
    }
    assert training.reward_history({}) == [] and training.reward_metrics([], 2.5) == {}


def test_a_long_reward_log_is_a_trailing_mean_so_start_and_end_are_not_one_noisy_step():
    # 0, 2.5, 0, 2.5 ... for 100 steps then a steady 2.5: one step alone says 0 or 2.5.
    raw = [2.5 * (i % 2) for i in range(100)] + [2.5] * 100
    payload = {"metric_history": {"rl": [{"step": i + 1, "reward": r} for i, r in enumerate(raw)]}}
    history = training.reward_history(payload)
    assert history[0] == 1.25 and history[-1] == 2.5
    assert len(history) == training.HISTORY_POINTS and max(history) <= 2.5


def test_one_column_for_both_roles_is_refused():
    ctx = _Ctx(_recipe_routes([]), prompt_column = "q", response_column = "q")
    with pytest.raises(FlowNodeError, match = "different columns"):
        asyncio.run(training.finetune(ctx))


def test_polls_read_the_loss_from_the_status_payload():
    assert training.status_losses({"metric_history": {"loss": [2.0, 1.0]}}) == [2.0, 1.0]
    assert training.status_losses({"metric_history": None}) == []
    assert training.status_losses(None) == []


def test_hub_rows_with_timestamps_still_upload(monkeypatch):
    import datetime

    import datasets

    rows = [
        {"q": f"q{i}", "a": f"a{i}", "at": datetime.datetime(2026, 1, 1, i), "x": float("nan")}
        for i in range(10)
    ]
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, streaming = False, **k: datasets.Dataset.from_list(rows).to_iterable_dataset()
        if streaming
        else datasets.Dataset.from_list(rows),
    )
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda *a, **k: 1 / 0)
    loaded = training._load_rows({"source": "hf", "ref": "org/ds"}, None, None)
    assert loaded[0]["at"] == "2026-01-01 00:00:00" and loaded[0]["x"] is None
    json.dumps(loaded, allow_nan = False)


def test_chat_with_checkpoint_puts_back_the_chat_model_it_replaced(monkeypatch):
    from types import SimpleNamespace

    from studio_mcp import loading, trained_models

    loads = []

    async def load_llm(caller, ctx, **kw):
        loads.append((kw["model"], kw["variant"]))

    async def resolve_by(*a, **k):
        return SimpleNamespace(ref = "/out/run-1", is_lora = True)

    monkeypatch.setattr(loading, "load_llm", load_llm)
    monkeypatch.setattr(trained_models, "resolve_by", resolve_by)
    ctx = _Ctx(
        lambda m, p, q, b: {"model_identifier": "unsloth/Qwen3-1.7B-GGUF", "gguf_variant": "Q4_K_M"}
    )
    ctx.inputs = {"checkpoint": {"kind": "checkpoint", "name": "run-1"}, "prompt": "hi"}
    ctx.params = {"max_tokens": 8}
    ctx.resident, ctx.set_status = {}, lambda *a, **k: None

    async def llm_text(**k):
        return "hello"

    ctx.llm_text = llm_text
    out = asyncio.run(training.chat_checkpoint(ctx))
    assert out == {"text": "hello"}
    assert loads == [("/out/run-1", None), ("unsloth/Qwen3-1.7B-GGUF", "Q4_K_M")]


def test_hub_holdout_never_downloads_more_than_it_can_use(monkeypatch):
    from types import SimpleNamespace

    import datasets

    def no_download(
        *a,
        streaming = False,
        **k,
    ):
        assert streaming, "a full split download"
        return SimpleNamespace(take = lambda n: [{"q": i} for i in range(n)])

    monkeypatch.setattr(datasets, "load_dataset", no_download)
    info = SimpleNamespace(splits = {"train": SimpleNamespace(num_examples = 900_000)})
    monkeypatch.setattr(
        datasets, "load_dataset_builder", lambda *a, **k: SimpleNamespace(info = info)
    )
    with pytest.raises(FlowNodeError, match = "900000 rows"):
        training._load_rows({"source": "hf", "ref": "org/big"}, None, None)
    assert len(training._load_rows({"source": "hf", "ref": "org/big"}, None, 9)) == 10


def test_grpo_without_saved_rewards_or_prompt_uses_the_node_defaults():
    config = {}
    rewards = training.rl_config({"objective": "grpo"}, "grpo", config)
    assert [r["name"] for r in rewards] == ["exact-answer", "soft-xml-format"]
    assert "<answer>" in config["rl_system_prompt"] and config["grpo_num_generations"] == 4
    config = {}
    training.rl_config({"system_prompt": ""}, "grpo", config)
    assert "rl_system_prompt" not in config


def test_grpo_steps_take_one_whole_prompt_group_unless_the_batch_is_set():
    config = {}
    training.rl_config({"num_generations": 8}, "grpo", config)
    assert (config["batch_size"], config["gradient_accumulation_steps"]) == (8, 1)
    config = {}
    training.rl_config({"batch_size": 2, "gradient_accumulation_steps": 4}, "grpo", config)
    assert "batch_size" not in config and "gradient_accumulation_steps" not in config
    config = {}
    training.rl_config({}, "dpo", config)
    assert "batch_size" not in config


def test_a_coded_route_error_reaches_the_node_as_its_message():
    from core.flows.caller import route_detail
    body = b'{"detail": {"code": "hf_model_verification_failed", "message": "The Hugging Face model could not be verified."}}'
    assert route_detail(body) == "The Hugging Face model could not be verified."


def _hub_body(**params):
    started = []

    def routes(method, path, query, body):
        if path == "/api/train/start":
            started.append(body)
            raise RuntimeError("started")
        raise AssertionError(path)

    ctx = _Ctx(routes, **params)
    ctx.inputs = {
        "dataset": {"kind": "dataset", "source": "hf", "ref": "org/data", "split": "train"}
    }
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    return started[0]


def test_model_group_sends_method_precision_and_project():
    cpt = _hub_body(training_type = "cpt", embedding_learning_rate = 2e-5, project_name = " books ")
    assert cpt["training_type"] == "Continued Pretraining" and cpt["load_in_4bit"] is False
    assert cpt["embedding_learning_rate"] == 2e-5 and cpt["project_name"] == "books"
    assert cpt["target_modules"][-2:] == ["embed_tokens", "lm_head"]
    assert _hub_body(training_type = "cpt", base_model = "unsloth/x-bnb-4bit")["load_in_4bit"] is True
    full = _hub_body(training_type = "full", load_in_4bit = True, embedding_learning_rate = 1e-5)
    assert full["training_type"] == "Full Finetuning" and full["load_in_4bit"] is False
    assert (
        full["use_lora"] is False and full["target_modules"] == [] and full["offload_layers"] == 0
    )
    assert "embedding_learning_rate" not in full
    assert _hub_body(load_in_4bit = False)["load_in_4bit"] is False


def test_dataset_group_sends_slices_eval_split_and_format():
    body = _hub_body(
        dataset_slice_start = 10,
        dataset_slice_end = 99,
        eval_split = "test",
        format_type = "raw",
        train_on_completions = True,
    )
    assert (body["dataset_slice_start"], body["dataset_slice_end"]) == (10, 99)
    assert body["eval_split"] == "test" and body["format_type"] == "raw"
    assert body["train_on_completions"] is False
    assert "dataset_streaming" not in _hub_body(dataset_streaming = True, objective = "grpo")
    assert _hub_body(dataset_streaming = True)["dataset_streaming"] is True


def test_holdout_starts_at_the_train_split_start(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, dataset_slice_start = 10)
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    rows = [r for f in _uploads(ctx).values() for r in f]
    assert len(rows) == 10 and {r["q"] for r in rows} == {f"q{i}" for i in range(10, 20)}
    assert "dataset_slice_start" not in started[0]


def test_lora_group_sends_dropout_variant_modules_and_vision_layers():
    body = _hub_body(
        lora_r = 32,
        lora_alpha = 64,
        lora_dropout = 0.1,
        lora_variant = "dora",
        target_modules = ["q_proj", " ", "v_proj"],
        finetune_vision_layers = False,
    )
    assert (body["lora_r"], body["lora_alpha"], body["lora_dropout"]) == (32, 64, 0.1)
    assert (body["use_rslora"], body["use_loftq"], body["use_dora"]) == (False, False, True)
    assert body["target_modules"] == ["q_proj", "v_proj"]
    assert body["finetune_vision_layers"] is False and body["finetune_mlp_modules"] is True


def test_optimization_group_sends_optimizer_scheduler_and_batch():
    body = _hub_body(
        optim = "adamw_torch_fused",
        lr_scheduler_type = "cosine",
        batch_size = 2,
        gradient_accumulation_steps = 16,
        weight_decay = 0.01,
    )
    assert (body["optim"], body["lr_scheduler_type"]) == ("adamw_torch_fused", "cosine")
    assert (body["batch_size"], body["gradient_accumulation_steps"], body["weight_decay"]) == (
        2,
        16,
        0.01,
    )
    grpo = _hub_body(objective = "grpo", num_generations = 8)
    assert (grpo["batch_size"], grpo["gradient_accumulation_steps"]) == (8, 1)


def test_schedule_group_sends_warmup_checkpoints_eval_and_seed():
    body = _hub_body(
        warmup_steps = 10, save_steps = 50, eval_steps = 0.1, early_stopping_patience = 3, random_seed = 42
    )
    assert (body["warmup_steps"], body["save_steps"], body["eval_steps"]) == (10, 50, 0.1)
    assert body["early_stopping_patience"] == 3 and body["random_seed"] == 42
    assert _hub_body(early_stopping_patience = 3)["early_stopping_patience"] == 0
    assert _hub_body()["random_seed"] == 1


def test_memory_group_sends_packing_checkpointing_and_offload():
    body = _hub_body(
        packing = True,
        gradient_checkpointing = "true",
        offload_layers = "auto",
        offload_vram_gb = 12.5,
        prefetch_depth = "auto",
    )
    assert body["packing"] is True and body["gradient_checkpointing"] == "true"
    assert (body["offload_layers"], body["offload_vram_gb"], body["prefetch_depth"]) == (
        "auto",
        12.5,
        "auto",
    )
    counted = _hub_body(
        offload_layers = "count", offload_count = 12, offload_vram_gb = 8, prefetch_depth = "3"
    )
    assert (counted["offload_layers"], counted["prefetch_depth"]) == (
        12,
        3,
    ) and "offload_vram_gb" not in counted
    assert _hub_body(offload_layers = "auto", gradient_checkpointing = "none")["offload_layers"] == 0
    assert _hub_body(packing = True, objective = "dpo")["packing"] is False
    assert _hub_body(task = "vision", vision_image_size = 512)["vision_image_size"] == 512


def test_rl_group_sends_clip_thinking_masking_and_prompt_length():
    grpo = _hub_body(
        objective = "grpo",
        epsilon_high = 0.28,
        enable_thinking = True,
        mask_truncated_completions = True,
        max_prompt_length = 256,
    )
    assert grpo["grpo_epsilon_high"] == 0.28 and grpo["grpo_enable_thinking"] is True
    assert grpo["grpo_mask_truncated_completions"] is True and grpo["rl_max_prompt_length"] == 256
    dpo = _hub_body(objective = "dpo", max_prompt_length = 128, epsilon_high = 0.28)
    assert dpo["rl_max_prompt_length"] == 128 and "grpo_epsilon_high" not in dpo


def test_drafter_group_sends_the_drafter_only_for_sft():
    body = _hub_body(
        drafter_enabled = True,
        drafter_data_source = "regenerate",
        drafter_max_examples = 500,
        drafter_epochs = 2.5,
        drafter_draft_steps = 4,
    )
    assert body["drafter"] == {
        "enabled": True,
        "data_source": "regenerate",
        "max_examples": 500,
        "epochs": 2.5,
        "learning_rate": None,
        "draft_steps": 4,
    }
    assert "drafter" not in _hub_body(drafter_enabled = True, objective = "grpo")
    assert "drafter" not in _hub_body(drafter_enabled = True, training_type = "cpt")
    assert "drafter" not in _hub_body()


def test_every_setting_reaches_a_valid_training_request():
    from models.training import TrainingStartRequest

    body = _hub_body(
        training_type = "cpt",
        offload_layers = "auto",
        offload_vram_gb = 10,
        drafter_enabled = True,
        eval_steps = 0.05,
        early_stopping_patience = 2,
        random_seed = 7,
    )
    TrainingStartRequest.model_validate(body)
    TrainingStartRequest.model_validate(
        _hub_body(objective = "grpo", epsilon_high = 0.28, max_prompt_length = 64)
    )


def test_hub_eval_split_survives_the_hold_out(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: False)
    splits = []

    def load_rows(
        dataset,
        token,
        end,
        start = 0,
    ):
        splits.append(dataset.get("split"))
        return ROWS if dataset.get("split") == "train" else ROWS[:5]

    monkeypatch.setattr(training, "_load_rows", load_rows)
    monkeypatch.setattr(
        training,
        "_eval_rows",
        lambda dataset, token, split: load_rows({**dataset, "split": split}, token, None),
    )
    started = []
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, eval_split = "test", eval_steps = 0.1)
    ctx.inputs = {
        "dataset": {"kind": "dataset", "source": "hf", "ref": "org/data", "split": "train"}
    }
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert splits == ["train", "test"]
    evals = next(rows for name, rows in _uploads(ctx).items() if name.endswith("-test-eval.jsonl"))
    assert len(evals) == 5 and len(started[0]["local_eval_datasets"]) == 1
    splits.clear()
    started.clear()
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, eval_split = "train", eval_steps = 0.1)
    ctx.inputs = {
        "dataset": {"kind": "dataset", "source": "hf", "ref": "org/data", "split": "train"}
    }
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert splits == ["train"] and "local_eval_datasets" not in started[0]


def test_one_column_for_two_rl_roles_is_refused():
    with pytest.raises(FlowNodeError, match = "own column"):
        training.rl_roles({"prompt_column": "q", "answer_column": "q"}, "grpo")
    with pytest.raises(FlowNodeError, match = "own column"):
        training.rl_roles(
            {"prompt_column": "p", "chosen_column": "c", "rejected_column": "c"}, "dpo"
        )
    assert training.rl_roles({"prompt_column": "p", "chosen_column": "c"}, "dpo") == {
        "p": "prompt",
        "c": "chosen",
    }


def test_hub_hold_out_streams_a_late_slice_and_caps_unknown_sizes(monkeypatch):
    from types import SimpleNamespace

    import datasets

    class Stream:
        def __init__(
            self,
            n,
            skipped = 0,
        ):
            self.n, self.skipped = n, skipped

        def skip(self, k):
            return Stream(self.n, self.skipped + k)

        def take(self, k):
            return [{"q": i} for i in range(self.skipped, min(self.n, self.skipped + k))]

    size = {"n": 300_000}
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, streaming = False, **k: Stream(size["n"])
        if streaming
        else pytest.fail("full download"),
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset_builder",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no metadata")),
    )
    hub = {"source": "hf", "ref": "org/big", "split": "train"}
    rows = training._load_rows(hub, None, 250_099, 250_000)
    assert [r["q"] for r in rows][:2] == [250_000, 250_001] and len(rows) == 100
    with pytest.raises(FlowNodeError, match = "More than 200000"):
        training._load_rows(hub, None, None)
    size["n"] = 50
    assert len(training._load_rows(hub, None, None)) == 50


def test_a_zero_learning_rate_reaches_the_training_api():
    assert _hub_body(learning_rate = 0)["learning_rate"] == "0"


def test_cpt_keeps_a_single_listed_embedding_target():
    body = _hub_body(training_type = "cpt", target_modules = ["q_proj", "lm_head"])
    assert body["target_modules"] == ["q_proj", "lm_head"]


def test_chat_with_checkpoint_turns_the_adapter_back_on_after_a_benchmark(monkeypatch):
    from types import SimpleNamespace

    from core.flows import caller as flow_caller
    from core.flows.nodes import _common
    from studio_mcp import loading, trained_models

    server = {"use_adapter": True}

    async def stream_sse(caller, path, body):
        # Like the inference server: the adapter switch sticks until a request sets it again.
        if body.get("use_adapter") is not None:
            server["use_adapter"] = body["use_adapter"]
        yield {
            "choices": [
                {"delta": {"content": "SELECT 1;" if server["use_adapter"] else "Sure! ```sql"}}
            ]
        }

    async def noop(*a, **k):
        return None

    async def resolve_by(*a, **k):
        return SimpleNamespace(ref = "/out/run-1", is_lora = True)

    monkeypatch.setattr(flow_caller, "stream_sse", stream_sse)
    monkeypatch.setattr(_common, "ensure_model", noop)
    monkeypatch.setattr(_common, "chat_model_name", lambda ctx: "m")
    monkeypatch.setattr(loading, "load_llm", noop)
    monkeypatch.setattr(trained_models, "resolve_by", resolve_by)
    ctx = _Ctx(lambda m, p, q, b: {})
    ctx.inputs = {"checkpoint": {"kind": "checkpoint", "name": "run-1"}, "prompt": "list users"}
    ctx.params = {"max_tokens": 8}
    ctx.resident, ctx.set_status = {}, lambda *a, **k: None
    ctx.on_cancel = ctx.text_delta = lambda *a, **k: None
    ctx.check_cancelled = lambda: None
    ctx.llm_text = lambda **k: _common.llm_text(ctx, **k)

    async def benchmark_judge():
        async for _ in stream_sse(None, "/v1/chat/completions", {"use_adapter": False}):
            pass

    asyncio.run(benchmark_judge())
    assert server["use_adapter"] is False
    assert asyncio.run(training.chat_checkpoint(ctx)) == {"text": "SELECT 1;"}


def test_chat_with_checkpoint_asks_with_the_trained_system_prompt(monkeypatch):
    from types import SimpleNamespace

    from studio_mcp import loading, trained_models

    async def noop(*a, **k):
        return None

    async def resolve_by(*a, **k):
        return SimpleNamespace(ref = "/out/run-1", is_lora = True)

    monkeypatch.setattr(loading, "load_llm", noop)
    monkeypatch.setattr(trained_models, "resolve_by", resolve_by)
    asked = []

    def chat(**params):
        ctx = _Ctx(lambda m, p, q, b: {})
        ctx.inputs = {
            "checkpoint": {
                "kind": "checkpoint",
                "name": "run-1",
                "system_prompt": "Answer in SQL.",
            },
            "prompt": "list users",
        }
        ctx.params = {"max_tokens": 8, **params}
        ctx.resident, ctx.set_status = {}, lambda *a, **k: None

        async def llm_text(**k):
            asked.append(k["system"])
            return "ok"

        ctx.llm_text = llm_text
        asyncio.run(training.chat_checkpoint(ctx))

    chat()
    chat(system_prompt = "Be brief.")
    assert asked == ["Answer in SQL.", "Be brief."]


def test_the_checkpoint_carries_the_one_system_prompt_training_used():
    grpo = {"objective": "grpo", "rl_system_prompt": "Use <answer> tags."}
    assert training.trained_system_prompt(grpo) == "Use <answer> tags."
    sft = {
        "custom_format_mapping": {
            "q": "user",
            "a": "assistant",
            "__system_prompt": " Answer in SQL. ",
        }
    }
    assert training.trained_system_prompt(sft) == "Answer in SQL."
    assert (
        training.trained_system_prompt({"custom_format_mapping": {"s": "system", "q": "user"}})
        == ""
    )


def test_the_evaluation_split_is_not_held_to_the_hold_out_cap(monkeypatch):
    import datasets

    big = datasets.Dataset.from_dict({"q": list(range(training.HOLDOUT_MAX_ROWS + 5))})
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, split = None, **k: big if split == "test" else pytest.fail(split),
    )
    rows = training._eval_rows({"ref": "org/data"}, None, "test")
    assert len(rows) == training.HOLDOUT_MAX_ROWS + 5

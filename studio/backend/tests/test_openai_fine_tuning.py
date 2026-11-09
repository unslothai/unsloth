# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import asyncio
import json
import threading

import httpx
import pytest
from fastapi import FastAPI

openai = pytest.importorskip("openai")

from auth.authentication import authenticated_via_api_key, get_current_subject  # noqa: E402
from models.training import TrainingJobResponse  # noqa: E402
from routes import openai_fine_tuning, training as training_routes  # noqa: E402
from storage import studio_db  # noqa: E402
import core.training.training as _training_core  # noqa: E402

_REAL_TRAINS_ALONE = openai_fine_tuning._trains_alone

CHATS = (
    b'{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}\n'
    b'{"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "See you"}]}\n'
)


class _Status:
    def __init__(
        self,
        job_id = "",
        phase = "idle",
    ):
        self.job_id = job_id
        self.phase = phase
        self.start_request_state = None
        self.error = None
        self.message = ""


@pytest.fixture
def studio(monkeypatch):
    calls = {"start": [], "stop": []}
    live = {"status": _Status()}

    async def start_training(request, current_subject, via_api_key):
        calls["start"].append(request)
        job_id = "job_20261021_101500_ab12cd34" + (
            "" if len(calls["start"]) == 1 else f"{len(calls['start'])}"
        )
        live["status"] = _Status(job_id, "loading_model")
        return TrainingJobResponse(job_id = job_id, status = "queued", message = "queued")

    async def stop_training(body, current_subject):
        calls["stop"].append(body)
        return training_routes.TrainingStopResponse(status = "stopped", message = "Stop requested")

    async def get_training_status(current_subject, via_api_key):
        return live["status"]

    monkeypatch.setattr(openai_fine_tuning, "_STARTED", {})
    monkeypatch.setattr(openai_fine_tuning, "_CANCELLING", set())
    monkeypatch.setattr(openai_fine_tuning, "_BATCH_OF", {})
    monkeypatch.setattr(openai_fine_tuning, "_BATCHES", {})
    monkeypatch.setattr(training_routes, "start_training", start_training)
    monkeypatch.setattr(training_routes, "stop_training", stop_training)
    monkeypatch.setattr(training_routes, "get_training_status", get_training_status)

    app = FastAPI()
    app.include_router(openai_fine_tuning.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[authenticated_via_api_key] = lambda: True
    client = openai.AsyncOpenAI(
        api_key = "sk-unsloth-test",
        base_url = "http://studio.test/v1",
        http_client = httpx.AsyncClient(transport = httpx.ASGITransport(app = app)),
        max_retries = 0,
    )
    return client, calls, live


def _run(coro):
    return asyncio.run(coro)


def _finished_run(
    tmp_path,
    training_file,
    status = "completed",
    error = None,
):
    output_dir = tmp_path / "unsloth_Qwen3-0.6B_1761041700"
    (output_dir / "checkpoint-2").mkdir(parents = True)
    config = {
        "model_name": "unsloth/Qwen3-0.6B",
        "local_datasets": [str(training_file)],
        "num_epochs": 2,
        "batch_size": 4,
        "learning_rate": "0.0001",
        "random_seed": 7,
        "project_name": "support-bot",
    }
    run_id = "job_20261021_101500_ab12cd34"
    studio_db.create_run(
        id = run_id,
        model_name = config["model_name"],
        dataset_name = str(training_file),
        config_json = json.dumps(config),
        started_at = "2026-10-21T10:15:00+00:00",
        total_steps = 4,
        output_dir = str(output_dir),
    )
    studio_db.insert_metrics_batch(
        run_id,
        [
            {
                "step": step,
                "loss": 2.0 - step / 4,
                "num_tokens": 100 * step,
                "elapsed_seconds": step,
            }
            for step in range(1, 5)
        ],
    )
    studio_db.finish_run(
        id = run_id,
        status = status,
        ended_at = "2026-10-21T10:20:00+00:00",
        final_step = 4,
        final_loss = 1.0,
        duration_seconds = 300.0,
        output_dir = str(output_dir),
        error_message = error,
    )
    return run_id


@pytest.mark.parametrize("mode", ["mapped", "alpaca", "raw"])
def test_unused_weighted_messages_do_not_block_other_training_data(studio, mode):
    client, calls, _live = studio
    row = {
        "messages": [{"role": "assistant", "content": "CONTEXT_ONLY", "weight": 0}],
        "question": "Question",
        "answer": "TARGET",
    }
    extra = {}
    if mode == "mapped":
        extra["custom_format_mapping"] = {"question": "user", "answer": "assistant"}
    elif mode == "alpaca":
        row.update(instruction = "Question", output = "TARGET")
    else:
        row["text"] = "TARGET"
        extra["format_type"] = "raw"

    async def flow():
        uploaded = await client.files.create(
            file = ("rows.jsonl", (json.dumps(row) + "\n").encode()), purpose = "fine-tune"
        )
        return await _create(client, uploaded.id, extra_body = extra)

    job = _run(flow())
    assert job.status == "validating_files"
    assert len(calls["start"]) == 1


@pytest.mark.parametrize("mode", ["cpt", "sparse_alpaca"])
@pytest.mark.parametrize("file_param", ["training_file", "validation_file"])
def test_dataset_selection_ignores_unused_message_weights(studio, tmp_path, mode, file_param):
    from datasets import load_dataset
    from utils.datasets.dataset_utils import format_dataset

    client, calls, _live = studio
    weighted = [{"role": "assistant", "content": "CONTEXT_ONLY", "weight": 0}]
    if mode == "cpt":
        rows = [{"text": "TARGET", "messages": weighted}]
        extra = {"training_type": "Continued Pretraining"}
    else:
        rows = [
            {"instruction": "Question", "output": "TARGET", "messages": weighted},
            {"output": "TARGET", "messages": weighted},
        ]
        extra = {}
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    source = tmp_path / "selected.jsonl"
    source.write_bytes(content)
    dataset = load_dataset(
        "json", data_files = str(source), split = "train", cache_dir = str(tmp_path / "cache")
    )
    formatted = format_dataset(dataset, format_type = "raw" if mode == "cpt" else "auto")
    assert formatted["final_format"] == ("raw_text" if mode == "cpt" else "alpaca")

    async def flow():
        uploaded = await client.files.create(file = ("selected.jsonl", content), purpose = "fine-tune")
        plain = await client.files.create(file = ("plain.jsonl", CHATS), purpose = "fine-tune")
        kwargs = {"training_file": plain.id, file_param: uploaded.id}
        return await client.fine_tuning.jobs.create(
            model = "unsloth/Qwen3-0.6B", extra_body = extra, **kwargs
        )

    job = _run(flow())
    assert job.status == "validating_files"
    assert len(calls["start"]) == 1


def test_a_mapping_that_selects_weighted_messages_is_still_refused(studio):
    client, calls, _live = studio
    rows = b'{"messages":[{"role":"assistant","content":"CONTEXT_ONLY","weight":0}]}\n'

    async def flow():
        uploaded = await client.files.create(file = ("rows.jsonl", rows), purpose = "fine-tune")
        with pytest.raises(openai.BadRequestError):
            await _create(
                client, uploaded.id, extra_body = {"custom_format_mapping": {"messages": "assistant"}}
            )

    _run(flow())
    assert calls["start"] == []


def test_exported_unit_weight_messages_still_train(studio, tmp_path):
    from datasets import Dataset

    client, calls, _live = studio
    exported = tmp_path / "export.jsonl"
    Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Bye"},
                    {"role": "assistant", "content": "Goodbye", "weight": 1},
                ]
            }
        ]
    ).to_json(exported)
    rows = exported.read_bytes()
    assert json.loads(rows)["messages"][1]["weight"] is None

    async def flow():
        uploaded = await client.files.create(file = ("export.jsonl", rows), purpose = "fine-tune")
        return await _create(client, uploaded.id)

    job = _run(flow())
    assert job.status == "validating_files"
    assert len(calls["start"]) == 1


@pytest.mark.parametrize("weight, escaped", [(0, False), (0.5, False), (0, True)])
def test_a_file_excluding_an_assistant_message_by_weight_is_refused(studio, weight, escaped):
    client, calls, _live = studio
    rows = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "CONTEXT_ONLY", "weight": weight},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "TARGET", "weight": 1},
                ]
            }
        ).encode()
        + b"\n"
    )
    if escaped:
        rows = rows.replace(b'"weight"', b'"weigh\\u0074"')
        assert b'"weight"' not in rows and json.loads(rows)["messages"][1]["weight"] == 0

    async def flow():
        weighted = await client.files.create(
            file = ("weighted.jsonl", CHATS + rows), purpose = "fine-tune"
        )
        plain = await client.files.create(file = ("plain.jsonl", CHATS), purpose = "fine-tune")
        errors = []
        for kwargs in (
            {"training_file": weighted.id},
            {"training_file": plain.id, "validation_file": weighted.id},
        ):
            try:
                await client.fine_tuning.jobs.create(model = "unsloth/Qwen3-0.6B", **kwargs)
            except openai.BadRequestError as exc:
                errors.append(exc.response.text)
        return errors

    errors = _run(flow())
    assert len(errors) == 2
    assert "training_file line 3" in errors[0] and "validation_file line 3" in errors[1]
    assert calls["start"] == []


def test_a_file_with_only_weight_one_messages_still_trains(studio):
    client, calls, _live = studio
    rows = (
        b'{"messages": [{"role": "user", "content": "Hi"}, '
        b'{"role": "assistant", "content": "Hello!", "weight": 1}]}\n'
    )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", rows), purpose = "fine-tune")
        return await _create(client, uploaded.id)

    job = _run(flow())
    assert job.status == "validating_files"
    assert len(calls["start"]) == 1


def test_upload_then_create_starts_a_studio_run(studio):
    client, calls, _live = studio

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        second = await client.files.create(file = ("more.jsonl", CHATS), purpose = "fine-tune")
        listed = await client.files.list()
        first_page = await client.files.list(limit = 1)
        oldest_first = await client.files.list(order = "asc")
        job = await client.fine_tuning.jobs.create(
            model = "unsloth/Qwen3-0.6B",
            training_file = uploaded.id,
            validation_file = uploaded.id,
            suffix = "support-bot",
            seed = 7,
            method = {
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "n_epochs": 2,
                        "batch_size": 4,
                        "learning_rate_multiplier": 0.5,
                    }
                },
            },
            integrations = [{"type": "wandb", "wandb": {"project": "support"}}],
            extra_body = {"lora_r": 32, "max_seq_length": 4096},
        )
        retrieved = await client.fine_tuning.jobs.retrieve(job.id)
        return uploaded, second, listed, first_page, oldest_first, job, retrieved

    uploaded, second, listed, first_page, oldest_first, job, retrieved = _run(flow())
    assert uploaded.id.startswith("file-") and uploaded.filename == "chats.jsonl"
    assert uploaded.bytes == len(CHATS)
    assert {f.id for f in listed.data} == {uploaded.id, second.id}
    assert len(first_page.data) == 1 and first_page.has_more
    assert [f.id for f in oldest_first.data] == [f.id for f in listed.data][::-1]

    [request] = calls["start"]
    assert request.model_name == "unsloth/Qwen3-0.6B"
    assert request.training_type == "LoRA/QLoRA" and request.format_type == "auto"
    assert request.local_datasets[0].endswith("_chats.jsonl")
    assert open(request.local_datasets[0], "rb").read() == CHATS
    assert (request.num_epochs, request.batch_size, request.random_seed) == (2, 4, 7)
    assert float(request.learning_rate) == 1e-4
    assert request.local_eval_datasets == request.local_datasets and request.eval_steps > 0
    assert request.enable_wandb and request.wandb_project == "support"
    assert (request.lora_r, request.max_seq_length, request.project_name) == (
        32,
        4096,
        "support-bot",
    )

    assert job.id == "ftjob_20261021_101500_ab12cd34"
    assert job.training_file == uploaded.id

    assert retrieved.status == "validating_files"
    assert (retrieved.model, retrieved.training_file) == ("unsloth/Qwen3-0.6B", uploaded.id)


def test_finished_run_reads_as_a_succeeded_job(studio, tmp_path):
    client, _calls, _live = studio
    training_file = tmp_path / ("0123456789abcdef0123456789abcdef" + "_chats.jsonl")
    _finished_run(tmp_path, training_file)

    async def flow():
        job = await client.fine_tuning.jobs.retrieve("ftjob_20261021_101500_ab12cd34")
        jobs = await client.fine_tuning.jobs.list()
        events = await client.fine_tuning.jobs.list_events(job.id, limit = 100)
        checkpoints = await client.fine_tuning.jobs.checkpoints.list(job.id)
        return job, jobs, events, checkpoints

    job, jobs, events, checkpoints = _run(flow())
    assert job.status == "succeeded"
    assert job.fine_tuned_model == "unsloth_Qwen3-0.6B_1761041700"
    assert job.training_file == "file-0123456789abcdef0123456789abcdef"
    assert job.trained_tokens == 400 and job.seed == 7
    assert job.hyperparameters.n_epochs == 2 and job.hyperparameters.learning_rate_multiplier == 0.5
    assert job.finished_at - job.created_at == 300
    assert [j.id for j in jobs.data] == [job.id]

    messages = [e.message for e in events.data]
    assert messages[0] == "The job has successfully completed"
    assert messages[-1] == "Fine-tuning job started"
    assert "Step 4/4: training loss=1.0000" in messages
    metrics = [e for e in events.data if e.type == "metrics"]
    assert [e.data["step"] for e in metrics] == [4, 3, 2, 1]

    [checkpoint] = checkpoints.data
    assert checkpoint.step_number == 2
    assert checkpoint.fine_tuned_model_checkpoint == "unsloth_Qwen3-0.6B_1761041700/checkpoint-2"
    assert checkpoint.metrics.train_loss == 1.5


def test_failed_run_keeps_host_paths_from_an_api_key(studio, tmp_path):
    client, _calls, _live = studio
    training_file = tmp_path / ("0123456789abcdef0123456789abcdef" + "_chats.jsonl")
    _finished_run(
        tmp_path,
        training_file,
        status = "error",
        error = f"No such file: {tmp_path}/secret/model.safetensors",
    )

    job = _run(client.fine_tuning.jobs.retrieve("ftjob_20261021_101500_ab12cd34"))
    assert job.status == "failed" and job.fine_tuned_model is None
    assert str(tmp_path) not in job.error.message


def test_cancel_stops_the_running_job(studio):
    client, calls, live = studio
    live["status"] = _Status("job_20261021_101500_ab12cd34", "training")
    studio_db.create_run(
        id = "job_20261021_101500_ab12cd34",
        model_name = "unsloth/Qwen3-0.6B",
        dataset_name = "chats.jsonl",
        config_json = "{}",
        started_at = "2026-10-21T10:15:00+00:00",
        total_steps = 4,
    )

    async def flow():
        cancelled = await client.fine_tuning.jobs.cancel("ftjob_20261021_101500_ab12cd34")

        return cancelled, await client.fine_tuning.jobs.retrieve(cancelled.id)

    job, retrieved = _run(flow())
    assert job.status == "cancelled" and retrieved.status == "cancelled"
    [stop] = calls["stop"]
    assert stop.expected_job_id == "job_20261021_101500_ab12cd34" and stop.save is True


def test_refusals_are_openai_errors(studio):
    client, calls, _live = studio

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        errors = []
        for kwargs in (
            {"training_file": "file-missing"},
            {"training_file": uploaded.id, "method": {"type": "dpo"}},
            {"training_file": uploaded.id, "method": {"type": "supervised", "supervised": "x"}},
            {"training_file": uploaded.id, "extra_body": {"local_datasets": ["/etc/passwd"]}},
        ):
            try:
                await client.fine_tuning.jobs.create(model = "unsloth/Qwen3-0.6B", **kwargs)
            except openai.APIStatusError as exc:
                errors.append(exc.status_code)
        try:
            await client.files.create(file = ("notes.txt", b"x"), purpose = "assistants")
        except openai.BadRequestError:
            errors.append(400)
        try:
            await client.fine_tuning.jobs.retrieve("ftjob_unknown")
        except openai.NotFoundError:
            errors.append(404)
        try:
            await client.fine_tuning.jobs.pause("ftjob_unknown")
        except openai.BadRequestError:
            errors.append(400)
        return errors

    assert _run(flow()) == [404, 400, 400, 400, 400, 404, 400]
    assert calls["start"] == []


def test_list_pages_through_the_whole_history(studio, monkeypatch):
    client, _calls, _live = studio
    runs = [{"id": f"job_{i:03d}", "status": "completed", "config_json": "{}"} for i in range(450)]
    monkeypatch.setattr(
        openai_fine_tuning,
        "list_runs",
        lambda limit, offset: {"runs": runs[offset : offset + limit], "total": len(runs)},
    )
    monkeypatch.setattr(openai_fine_tuning, "_trained_tokens", lambda run_id: None)

    async def every_id():
        return [job.id async for job in client.fine_tuning.jobs.list(limit = 100)]

    assert _run(every_id()) == [f"ftjob_{i:03d}" for i in range(450)]


def test_a_live_job_cursor_still_reaches_the_history(studio, tmp_path):
    client, _calls, live = studio
    _finished_run(tmp_path, tmp_path / ("0123456789abcdef0123456789abcdef" + "_chats.jsonl"))
    live["status"] = _Status("job_20261021_111500_cd34ef56", "loading_model")

    async def every_id():
        return [job.id async for job in client.fine_tuning.jobs.list(limit = 1)]

    assert _run(every_id()) == ["ftjob_20261021_111500_cd34ef56", "ftjob_20261021_101500_ab12cd34"]


def test_a_legacy_non_finite_config_still_lists(studio, tmp_path):
    client, _calls, _live = studio
    run_id = _finished_run(
        tmp_path, tmp_path / ("0123456789abcdef0123456789abcdef" + "_chats.jsonl")
    )
    conn = studio_db.get_connection()
    conn.execute(
        "UPDATE training_runs SET config_json = ? WHERE id = ?",
        ('{"model_name": "unsloth/Qwen3-0.6B", "learning_rate": Infinity}', run_id),
    )
    conn.commit()
    conn.close()

    async def listed():
        return await client.fine_tuning.jobs.list()

    assert [j.status for j in _run(listed()).data] == ["succeeded"]


def test_cancel_before_the_run_row_sticks(studio):
    client, _calls, live = studio
    live["status"] = _Status("job_20261021_101500_ab12cd34", "loading_model")

    async def flow():
        await client.fine_tuning.jobs.cancel("ftjob_20261021_101500_ab12cd34")
        return await client.fine_tuning.jobs.retrieve("ftjob_20261021_101500_ab12cd34")

    assert _run(flow()).status == "cancelled"


def test_an_overflowed_grad_norm_keeps_events_readable(studio, tmp_path):
    client, _calls, _live = studio
    run_id = _finished_run(
        tmp_path, tmp_path / ("0123456789abcdef0123456789abcdef" + "_chats.jsonl")
    )
    studio_db.insert_metrics_batch(
        run_id, [{"step": 2, "loss": float("nan"), "grad_norm": float("inf")}]
    )

    async def flow():
        events = await client.fine_tuning.jobs.list_events(f"ft{run_id}", limit = 100)
        checkpoints = await client.fine_tuning.jobs.checkpoints.list(f"ft{run_id}")
        return events, checkpoints

    events, checkpoints = _run(flow())
    [step2] = [e for e in events.data if e.type == "metrics" and e.data["step"] == 2]
    assert step2.data["grad_norm"] is None
    assert checkpoints.data[0].step_number == 2


@pytest.fixture
def queue(studio, monkeypatch):
    client, calls, live = studio
    monkeypatch.setattr(openai_fine_tuning, "_QUEUE", [])
    monkeypatch.setattr(openai_fine_tuning, "_ALIASES", {})
    monkeypatch.setattr(openai_fine_tuning, "_RUN_ALIAS", {})
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_RUN", {})
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_CREATED", {})
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_CREATED_NS", {})
    monkeypatch.setattr(openai_fine_tuning, "_aliases_loaded", set())
    monkeypatch.setattr(openai_fine_tuning, "_ensure_drainer", lambda: None)

    monkeypatch.setattr(openai_fine_tuning, "_multi_lora_available", lambda: False)
    gpu = {"busy": True}
    monkeypatch.setattr(openai_fine_tuning, "_training_busy", lambda: gpu["busy"])
    return client, calls, live, gpu


def _create(client, file_id, **kwargs):
    return client.fine_tuning.jobs.create(
        model = "unsloth/Qwen3-0.6B", training_file = file_id, **kwargs
    )


def test_a_busy_gpu_queues_jobs_and_starts_them_in_order(queue):
    client, calls, _live, gpu = queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        first = await _create(client, uploaded.id, suffix = "first")
        second = await _create(client, uploaded.id, suffix = "second")
        listed = await client.fine_tuning.jobs.list()
        events = await client.fine_tuning.jobs.list_events(second.id)
        await openai_fine_tuning._drain_once()
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        gpu["busy"] = True
        await openai_fine_tuning._drain_once()
        after_first = [await client.fine_tuning.jobs.retrieve(j.id) for j in (first, second)]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return first, second, listed, events, after_first

    first, second, listed, events, (started, waiting) = _run(flow())
    assert first.status == second.status == "queued"
    newest = (second, first)
    assert [j.id for j in listed.data[:2]] == [job.id for job in newest]
    assert events.data[0].message == "Waiting for the GPU: position 2 in the queue"
    assert [r.project_name for r in calls["start"]] == ["first", "second"]

    assert started.id == first.id and started.status == "validating_files"
    assert waiting.status == "queued"


def test_a_queued_job_can_be_cancelled_and_never_starts(queue):
    client, calls, _live, gpu = queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        cancelled = await client.fine_tuning.jobs.cancel(job.id)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return cancelled, await client.fine_tuning.jobs.retrieve(job.id)

    cancelled, retrieved = _run(flow())
    assert cancelled.status == retrieved.status == "cancelled"
    assert calls["start"] == []


def test_a_refused_start_fails_the_queued_job(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def refuse(request, current_subject, via_api_key):
        return TrainingJobResponse(
            job_id = "", status = "error", message = "The model is gated.", error = "gated"
        )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", refuse)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return await client.fine_tuning.jobs.retrieve(job.id), openai_fine_tuning._QUEUE

    job, remaining = _run(flow())
    assert job.status == "failed" and job.error.message == "The model is gated."
    assert remaining == []


def test_another_account_cannot_see_or_cancel_a_queued_job(queue, monkeypatch):
    client, _calls, _live, _gpu = queue
    from utils.account_context import AccountContext

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(
            openai_fine_tuning,
            "current_account_id",
            lambda: AccountContext("bob", "bob").account_id,
        )
        statuses = []
        for call in (client.fine_tuning.jobs.retrieve, client.fine_tuning.jobs.cancel):
            try:
                await call(job.id)
            except openai.NotFoundError:
                statuses.append(404)
        listed = await client.fine_tuning.jobs.list()
        return statuses, [j.id for j in listed.data]

    statuses, listed = _run(flow())
    assert statuses == [404, 404] and listed == []


def _forget_aliases(monkeypatch):
    monkeypatch.setattr(openai_fine_tuning, "_RUN_ALIAS", {})
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_RUN", {})
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_CREATED", {})
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_CREATED_NS", {})
    monkeypatch.setattr(openai_fine_tuning, "_aliases_loaded", set())


def test_a_started_queued_job_keeps_its_id_after_a_restart(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        _forget_aliases(monkeypatch)
        retrieved = await client.fine_tuning.jobs.retrieve(job.id)
        listed = await client.fine_tuning.jobs.list()

        paged = await client.fine_tuning.jobs.list(after = job.id)
        return job, retrieved, listed, paged

    job, retrieved, listed, paged = _run(flow())
    assert retrieved.id == job.id and retrieved.status == "validating_files"
    assert [j.id for j in listed.data] == [job.id]
    assert paged.data == [] and not paged.has_more


def test_an_alias_table_from_an_older_build_loads_and_gains_the_ns_column(queue):
    table = openai_fine_tuning._ALIAS_TABLE
    conn = studio_db.get_connection()
    conn.execute(
        f"CREATE TABLE {table} "
        "(alias TEXT PRIMARY KEY, run_id TEXT NOT NULL, account_id TEXT, created_at INTEGER)"
    )
    conn.execute(f"INSERT INTO {table} VALUES ('ftjob_qold', 'run_old', NULL, 100)")
    conn.commit()
    conn.close()

    openai_fine_tuning._load_aliases()
    openai_fine_tuning._ALIAS_CREATED_NS["ftjob_qnew"] = 7
    openai_fine_tuning._save_alias("ftjob_qnew", "run_new", None, 100)
    conn = studio_db.get_connection()
    rows = conn.execute(f"SELECT alias, created_ns FROM {table} ORDER BY alias").fetchall()
    conn.close()
    assert openai_fine_tuning._ALIAS_RUN["ftjob_qold"] == ("run_old", None)
    assert [tuple(r) for r in rows] == [("ftjob_qnew", 7), ("ftjob_qold", None)]


def test_jobs_created_within_one_clock_tick_list_newest_submitted_first(queue, monkeypatch):
    client, _calls, _live, _gpu = queue
    monkeypatch.setattr(openai_fine_tuning, "_now", lambda: 1_700_000_000)
    monkeypatch.setattr(openai_fine_tuning.time, "time_ns", lambda: 1_700_000_000 * 10**9)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = s) for s in "abc"]
        return jobs, await client.fine_tuning.jobs.list()

    jobs, listed = _run(flow())
    assert [j.id for j in listed.data] == [j.id for j in jobs[::-1]]


def test_a_job_started_directly_lists_ahead_of_an_older_same_second_job(queue, monkeypatch):
    client, _calls, _live, gpu = queue
    monkeypatch.setattr(openai_fine_tuning, "_now", lambda: 1_700_000_000)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        queued = await _create(client, uploaded.id, suffix = "queued")
        await client.fine_tuning.jobs.cancel(queued.id)
        gpu["busy"] = False
        direct = await _create(client, uploaded.id, suffix = "direct")
        first = await client.fine_tuning.jobs.list(limit = 1)
        _forget_aliases(monkeypatch)
        restarted = await client.fine_tuning.jobs.list(limit = 1)
        return direct, first, restarted

    direct, first, restarted = _run(flow())
    assert not direct.id.startswith("ftjob_q")
    assert first.data[0].id == restarted.data[0].id == direct.id


def test_cancelling_a_job_whose_start_is_in_flight_stops_it(queue, monkeypatch):
    client, calls, live, gpu = queue
    release = asyncio.Event()
    entered = asyncio.Event()
    start = training_routes.start_training

    async def slow_start(request, current_subject, via_api_key):
        entered.set()
        await release.wait()
        return await start(request, current_subject = current_subject, via_api_key = via_api_key)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", slow_start)
        gpu["busy"] = False
        drain = asyncio.ensure_future(openai_fine_tuning._drain_once())
        await entered.wait()
        cancelled = await client.fine_tuning.jobs.cancel(job.id)
        release.set()
        await drain
        return cancelled, await client.fine_tuning.jobs.retrieve(job.id)

    cancelled, retrieved = _run(flow())
    assert cancelled.status == retrieved.status == "cancelled"
    assert [b.expected_job_id for b in calls["stop"]] == [live["status"].job_id]


def test_a_queued_failure_hides_host_paths_from_an_api_key(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def refuse(request, current_subject, via_api_key):
        message = "Dataset not found: /home/alice/.unsloth/studio/datasets/chats.jsonl"
        return TrainingJobResponse(job_id = "", status = "error", message = message, error = message)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", refuse)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return await client.fine_tuning.jobs.list_events(job.id)

    events = _run(flow())
    assert events.data[0].level == "error" and "/home/alice" not in events.data[0].message


def test_a_diffusion_run_winning_the_gpu_keeps_the_job_queued(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def diffusion_first(request, current_subject, via_api_key):
        return TrainingJobResponse(
            job_id = "",
            status = "error",
            message = "Diffusion training already active",
            error = "Diffusion training already active",
        )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", diffusion_first)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return await client.fine_tuning.jobs.retrieve(job.id)

    assert _run(flow()).status == "queued"


def test_two_creates_at_once_start_one_and_queue_the_other(queue, monkeypatch):
    client, calls, _live, gpu = queue
    gpu["busy"] = False
    start = training_routes.start_training

    async def start_and_take_the_gpu(request, current_subject, via_api_key):
        await asyncio.sleep(0.01)
        gpu["busy"] = True
        return await start(request, current_subject = current_subject, via_api_key = via_api_key)

    monkeypatch.setattr(training_routes, "start_training", start_and_take_the_gpu)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        return await asyncio.gather(_create(client, uploaded.id), _create(client, uploaded.id))

    jobs = _run(flow())
    assert sorted(j.status for j in jobs) == ["queued", "validating_files"]
    assert len(calls["start"]) == 1


def test_a_job_cancelled_before_it_started_does_not_hide_newer_jobs(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        monkeypatch.setattr(openai_fine_tuning, "_now", lambda: 1_700_000_000)
        old = await _create(client, uploaded.id)
        await client.fine_tuning.jobs.cancel(old.id)
        gpu["busy"] = False
        newer = await _create(client, uploaded.id)
        listed = await client.fine_tuning.jobs.list()
        return old, newer, listed, await client.fine_tuning.jobs.retrieve(old.id)

    old, newer, listed, retrieved = _run(flow())
    assert listed.data[0].id == newer.id
    assert retrieved.status == "cancelled"


def test_a_queued_job_keeps_its_creation_time_and_event_cursor(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        monkeypatch.setattr(openai_fine_tuning, "_now", lambda: 1_700_000_000)
        job = await _create(client, uploaded.id)
        events = await client.fine_tuning.jobs.list_events(job.id)
        again = await client.fine_tuning.jobs.list_events(job.id, after = events.data[0].id)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        _forget_aliases(monkeypatch)
        return job, again, await client.fine_tuning.jobs.retrieve(job.id)

    job, again, started = _run(flow())
    assert again.data == []
    assert started.created_at == job.created_at == 1_700_000_000


def test_an_inference_request_in_flight_never_fails_a_queued_job(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def chat_in_flight(request, current_subject, via_api_key):
        raise openai_fine_tuning.HTTPException(
            status_code = 409,
            detail = "Cannot start training over the API while an inference request is in progress.",
        )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", chat_in_flight)
        gpu["busy"] = False
        for _ in range(openai_fine_tuning.QUEUE_IDLE_CONFLICTS + 2):
            await openai_fine_tuning._drain_once()
        return await client.fine_tuning.jobs.retrieve(job.id)

    assert _run(flow()).status == "queued"


def test_a_job_id_store_failure_keeps_the_started_job(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    def locked():
        raise studio_db.sqlite3.OperationalError("database is locked")

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        await client.fine_tuning.jobs.list()
        monkeypatch.setattr(openai_fine_tuning, "_alias_db", locked)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return job, await client.fine_tuning.jobs.retrieve(job.id)

    job, started = _run(flow())
    assert started.id == job.id and started.status == "validating_files"


def test_queue_ids_are_stored_and_loaded_per_account(queue, monkeypatch):
    client, _calls, _live, gpu = queue
    from utils.account_context import AccountContext

    bob = AccountContext("bob", "bob")
    reads, writes = [], []
    alias_db = openai_fine_tuning._alias_db

    def spy():
        reads.append(openai_fine_tuning.current_account_id())
        return alias_db()

    save = openai_fine_tuning._save_alias

    def save_spy(*args):
        writes.append(openai_fine_tuning.current_account_id())
        return save(*args)

    monkeypatch.setattr(openai_fine_tuning, "_alias_db", spy)
    monkeypatch.setattr(openai_fine_tuning, "_save_alias", save_spy)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        # The drainer runs in whatever account context it was created in, not the job's.
        openai_fine_tuning._QUEUE[0].account = bob
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        await client.fine_tuning.jobs.list()
        monkeypatch.setattr(openai_fine_tuning, "current_account_id", lambda: bob.account_id)
        await client.fine_tuning.jobs.list()
        return job

    _run(flow())
    assert writes == [bob.account_id]

    assert bob.account_id in reads and len(set(reads)) == 2


def _transient_409(request, current_subject, via_api_key):
    raise openai_fine_tuning.HTTPException(
        status_code = 409,
        detail = "Cannot start training over the API while an inference request is in progress.",
    )


def test_a_cancel_during_a_start_that_hits_a_transient_conflict_ends_the_job(queue, monkeypatch):
    client, calls, _live, gpu = queue
    release, entered = asyncio.Event(), asyncio.Event()

    async def slow_conflict(request, current_subject, via_api_key):
        entered.set()
        await release.wait()
        _transient_409(request, current_subject, via_api_key)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", slow_conflict)
        gpu["busy"] = False
        drain = asyncio.ensure_future(openai_fine_tuning._drain_once())
        await entered.wait()
        await client.fine_tuning.jobs.cancel(job.id)
        release.set()
        await drain
        return await client.fine_tuning.jobs.retrieve(job.id), list(openai_fine_tuning._QUEUE)

    job, remaining = _run(flow())
    assert job.status == "cancelled" and remaining == []


def test_a_ui_run_winning_the_gpu_keeps_the_queued_job_waiting(queue, monkeypatch):
    client, _calls, _live, gpu = queue

    async def lost_to_ui(request, current_subject, via_api_key):
        gpu["busy"] = True
        return TrainingJobResponse(
            job_id = "",
            status = "error",
            message = "Training did not start.",
            error = "subprocess_start_failed",
        )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", lost_to_ui)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return await client.fine_tuning.jobs.retrieve(job.id)

    assert _run(flow()).status == "queued"


def test_a_create_during_a_chat_request_is_queued_not_refused(queue, monkeypatch):
    client, _calls, _live, gpu = queue
    gpu["busy"] = False

    async def chat_in_flight(request, current_subject, via_api_key):
        _transient_409(request, current_subject, via_api_key)

    monkeypatch.setattr(training_routes, "start_training", chat_in_flight)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        return await _create(client, uploaded.id)

    assert _run(flow()).status == "queued"


def test_a_sidecar_swap_conflict_queues_and_a_crash_after_cancel_stays_cancelled(
    queue, monkeypatch
):
    client, _calls, _live, gpu = queue
    release, entered = asyncio.Event(), asyncio.Event()

    async def sidecar_swap(request, current_subject, via_api_key):
        raise openai_fine_tuning.HTTPException(
            status_code = 409,
            detail = "A transformers installation is replacing the latest sidecar; retry when it completes.",
        )

    async def crash_after_cancel(request, current_subject, via_api_key):
        entered.set()
        await release.wait()
        raise RuntimeError("worker spawn blew up")

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        monkeypatch.setattr(training_routes, "start_training", sidecar_swap)
        gpu["busy"] = False
        queued = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "start_training", crash_after_cancel)
        drain = asyncio.ensure_future(openai_fine_tuning._drain_once())
        await entered.wait()
        await client.fine_tuning.jobs.cancel(queued.id)
        release.set()
        await drain
        return queued, await client.fine_tuning.jobs.retrieve(queued.id)

    queued, after = _run(flow())
    assert queued.status == "queued" and after.status == "cancelled"


def _entry(
    tmp_path,
    name = "a",
    subject = "unsloth",
    **fields,
):
    from utils.account_context import current_account

    dataset = tmp_path / f"{name}.jsonl"
    dataset.write_bytes(CHATS)
    config = {
        "model_name": "unsloth/Qwen3-0.6B",
        "training_type": "LoRA/QLoRA",
        "format_type": "auto",
        "local_datasets": [str(dataset)],
        **fields,
    }
    return openai_fine_tuning._Queued(
        alias = f"ftjob_q{name}",
        account = current_account(),
        subject = subject,
        via_api_key = True,
        request = openai_fine_tuning.TrainingStartRequest.model_validate(config),
        created_at = 1_700_000_000,
    )


@pytest.fixture
def batching(monkeypatch):
    monkeypatch.setattr(openai_fine_tuning, "_QUEUE", [])
    monkeypatch.setattr(openai_fine_tuning, "_multi_lora_available", lambda: True)
    monkeypatch.setattr(_training_core, "should_use_mlx_training_backend", lambda: False)
    monkeypatch.delenv("UNSLOTH_STUDIO_FT_BATCH_MAX", raising = False)
    return openai_fine_tuning._QUEUE


def test_jobs_differing_only_in_member_fields_batch_in_queue_order(batching, tmp_path):
    head = _entry(tmp_path, "a")
    other = _entry(
        tmp_path,
        "b",
        learning_rate = "1e-4",
        num_epochs = 3,
        batch_size = 4,
        lora_r = 32,
        lora_alpha = 64,
        random_seed = 7,
        project_name = "other",
        max_steps = 30,
        weight_decay = 0.05,
        warmup_steps = 10,
        packing = False,
    )
    odd = _entry(tmp_path, "c", max_seq_length = 4096)
    last = _entry(tmp_path, "d", lora_r = 8, warmup_ratio = 0.1)
    batching.extend([head, other, odd, last])
    assert openai_fine_tuning._batch_for(head) == [head, other, last]


@pytest.mark.parametrize(
    "fields",
    [
        {"resume_from_checkpoint": "/runs/old"},
        {"eval_steps": 0.1},
        {"lora_dropout": 0.05},
        {"use_dora": True},
        {"use_loftq": True},
        {"enable_wandb": True},
        {"enable_tensorboard": True},
        {"lr_scheduler_type": "reduce_lr_on_plateau"},
        {"offload_layers": "auto"},
        {"offload_layers": 4},
        {"training_type": "Full Finetuning"},
        {"training_type": "Continued Pretraining"},
        {"is_dataset_image": True},
        {"is_embedding": True},
        {"optim": "adamw_torch_fused"},
        {"gpu_ids": [0, 1]},
        {"start_request_id": "retry-1"},
    ],
)
def test_an_excluded_setting_trains_alone(batching, tmp_path, fields):
    head = _entry(tmp_path, "a", **fields)
    other = _entry(tmp_path, "b", **fields)
    batching.extend([head, other])
    assert openai_fine_tuning._batch_for(head) == [head]

    plain = _entry(tmp_path, "c")
    batching.insert(0, plain)
    assert openai_fine_tuning._batch_for(plain) == [plain]


@pytest.mark.parametrize(
    "fields",
    [
        {"gradient_accumulation_steps": 4},
        {"load_in_4bit": False},
        {"model_name": "unsloth/Qwen3-1.7B"},
        {"target_modules": ["q_proj"]},
        {"lr_scheduler_type": "cosine"},
        {"optim": "adamw_torch"},
        {"use_rslora": True},
    ],
)
def test_shared_settings_must_match(batching, tmp_path, fields):
    head = _entry(tmp_path, "a")
    other = _entry(tmp_path, "b", **fields)
    batching.extend([head, other])
    assert openai_fine_tuning._batch_for(head) == [head]


def test_another_subject_or_account_is_never_batched(batching, tmp_path):
    import dataclasses

    head = _entry(tmp_path, "a")
    other_subject = _entry(tmp_path, "b", subject = "bob")
    other_account = _entry(tmp_path, "c")
    other_account.account = dataclasses.replace(head.account, account_id = "acct-bob")
    batching.extend([head, other_subject, other_account])
    assert openai_fine_tuning._batch_for(head) == [head]


def test_the_cap_and_the_kill_switch(batching, tmp_path, monkeypatch):
    entries = [_entry(tmp_path, name) for name in "abcdef"]
    batching.extend(entries)
    assert openai_fine_tuning._batch_for(entries[0]) == entries[:4]
    monkeypatch.setenv("UNSLOTH_STUDIO_FT_BATCH_MAX", "2")
    assert openai_fine_tuning._batch_for(entries[0]) == entries[:2]
    monkeypatch.setenv("UNSLOTH_STUDIO_FT_BATCH_MAX", "1")
    assert openai_fine_tuning._batch_for(entries[0]) == entries[:1]
    monkeypatch.setenv("UNSLOTH_STUDIO_FT_BATCH_MAX", "many")
    assert openai_fine_tuning._batch_for(entries[0]) == entries[:4]


def test_without_the_kernels_every_job_trains_alone(batching, tmp_path, monkeypatch):
    entries = [_entry(tmp_path, name) for name in "ab"]
    batching.extend(entries)
    monkeypatch.setattr(openai_fine_tuning, "_multi_lora_available", lambda: False)
    assert openai_fine_tuning._batch_for(entries[0]) == entries[:1]


def test_the_kernel_lookup_reads_the_package_on_disk_once(tmp_path, monkeypatch):
    import importlib.machinery

    from routes.openai_fine_tuning import _multi_lora_available

    package = tmp_path / "unsloth"
    (package / "kernels").mkdir(parents = True)
    lookups = []

    def find_spec(name):
        lookups.append(name)
        spec = importlib.machinery.ModuleSpec(name, None, is_package = True)
        spec.submodule_search_locations = [str(package)]
        return spec

    monkeypatch.setattr(openai_fine_tuning.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(openai_fine_tuning, "_multi_lora", None)
    assert _multi_lora_available() is False
    (package / "kernels" / "multi_lora.py").write_text("")
    assert _multi_lora_available() is False
    monkeypatch.setattr(openai_fine_tuning, "_multi_lora", None)
    assert _multi_lora_available() is True
    # Only the top-level package is looked up: finding a submodule would import unsloth (and torch).
    assert lookups == ["unsloth", "unsloth"]


class _BatchBackend:
    def __init__(self):
        self.current_job_id = ""
        self.running: list[str] = []
        self.stopped: list[str] = []

    def batch_member_ids(self):
        return list(self.running)

    def stop_member(self, run_id):
        if run_id not in self.running:
            return False
        self.stopped.append(run_id)
        return True


@pytest.fixture
def batch_queue(queue, monkeypatch):
    import core.training

    client, calls, live, gpu = queue
    monkeypatch.setattr(openai_fine_tuning, "_multi_lora_available", lambda: True)

    async def batchable_model(request):
        return False

    monkeypatch.setattr(openai_fine_tuning, "_trains_alone", batchable_model)
    monkeypatch.setattr(_training_core, "should_use_mlx_training_backend", lambda: False)
    monkeypatch.delenv("UNSLOTH_STUDIO_FT_BATCH_MAX", raising = False)
    backend = _BatchBackend()
    monkeypatch.setattr(core.training, "get_training_backend", lambda: backend)
    calls["batch"] = []

    async def start_training_batch(requests, current_subject, via_api_key):
        calls["batch"].append((list(requests), current_subject, via_api_key))
        backend.current_job_id = "job_20261021_101500_batch000"
        run_ids = [f"job_20261021_101500_member0{i}" for i in range(len(requests))]
        backend.running = list(run_ids)
        live["status"] = _Status(backend.current_job_id, "loading_model")
        return [
            TrainingJobResponse(job_id = run_id, status = "queued", message = "queued")
            for run_id in run_ids
        ]

    monkeypatch.setattr(training_routes, "start_training_batch", start_training_batch)
    return client, calls, live, gpu, backend


def test_compatible_queued_jobs_start_as_one_batch(batch_queue):
    client, calls, live, gpu, _backend = batch_queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [
            await _create(client, uploaded.id, suffix = "a", seed = 1),
            await _create(client, uploaded.id, suffix = "b", hyperparameters = {"n_epochs": 3}),
            await _create(client, uploaded.id, suffix = "solo", extra_body = {"use_dora": True}),
            await _create(client, uploaded.id, suffix = "c", extra_body = {"lora_r": 64}),
        ]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        retrieved = [await client.fine_tuning.jobs.retrieve(job.id) for job in jobs]
        listed = await client.fine_tuning.jobs.list()
        return jobs, retrieved, listed

    jobs, retrieved, listed = _run(flow())
    [(requests, subject, via_api_key)] = calls["batch"]
    assert [r.project_name for r in requests] == ["a", "b", "c"]
    assert (subject, via_api_key) == ("unsloth", True)
    assert calls["start"] == []
    batched = [retrieved[0], retrieved[1], retrieved[3]]

    assert [j.id for j in batched] == [jobs[0].id, jobs[1].id, jobs[3].id]
    assert [j.status for j in batched] == ["validating_files"] * 3
    assert [j.suffix for j in batched] == ["a", "b", "c"]
    assert batched[0].seed == 1 and batched[1].hyperparameters.n_epochs == 3
    assert retrieved[2].status == "queued"
    run_ids = [openai_fine_tuning._ALIAS_RUN[j.id][0] for j in batched]
    assert run_ids == [f"job_20261021_101500_member0{i}" for i in range(3)]

    listed_ids = [j.id for j in listed.data]
    assert sorted(listed_ids) == sorted(j.id for j in jobs)
    assert live["status"].job_id not in {openai_fine_tuning._run_id(i) for i in listed_ids}


def test_listing_pages_through_a_batch_that_is_still_loading(batch_queue):
    client, _calls, _live, gpu, _backend = batch_queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = s) for s in "abc"]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        seen, after = [], None
        for _ in range(5):
            page = await client.fine_tuning.jobs.list(
                limit = 1, **({"after": after} if after else {})
            )
            seen += [j.id for j in page.data]
            if not page.has_more:
                break
            after = page.data[-1].id
        return jobs, seen

    jobs, seen = _run(flow())
    assert sorted(seen) == sorted(j.id for j in jobs) and len(seen) == 3


def test_a_job_that_waited_for_a_batch_keeps_its_place_once_it_starts(batch_queue, monkeypatch):
    client, _calls, _live, gpu, _backend = batch_queue
    clock = {"t": 1_700_000_000}
    monkeypatch.setattr(openai_fine_tuning, "_now", lambda: clock["t"])

    def history_row(run_id, started_at):
        studio_db.create_run(
            id = run_id,
            model_name = "unsloth/Qwen3-0.6B",
            dataset_name = "chats.jsonl",
            config_json = json.dumps({"model_name": "unsloth/Qwen3-0.6B"}),
            started_at = started_at,
            total_steps = 4,
        )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = []
        for suffix in "abc":
            clock["t"] += 10
            extra = {"use_dora": True} if suffix == "b" else {}
            jobs.append(await _create(client, uploaded.id, suffix = suffix, extra_body = extra))
        first = await client.fine_tuning.jobs.list(limit = 1)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        for run_id in openai_fine_tuning._BATCHES["job_20261021_101500_batch000"]:
            history_row(run_id, "2026-11-04T00:00:00+00:00")
        await openai_fine_tuning._drain_once()
        history_row("job_20261021_101500_ab12cd34", "2026-11-04T00:10:00+00:00")
        seen, after = [j.id for j in first.data], first.data[-1].id
        for _ in range(5):
            page = await client.fine_tuning.jobs.list(limit = 1, after = after)
            seen += [j.id for j in page.data]
            if not page.has_more:
                break
            after = page.data[-1].id
        return jobs, seen

    jobs, seen = _run(flow())
    assert seen[0] == jobs[2].id
    assert sorted(seen) == sorted(j.id for j in jobs)


def test_paging_stays_complete_when_a_loading_batch_gets_its_history_rows(batch_queue, monkeypatch):
    client, _calls, live, gpu, backend = batch_queue
    clock = {"t": 1_700_000_000}
    monkeypatch.setattr(openai_fine_tuning, "_now", lambda: clock["t"])
    starts = []

    async def start_training_batch(requests, current_subject, via_api_key):
        backend.current_job_id = f"job_20261021_10150{len(starts)}_batch"
        run_ids = [f"{backend.current_job_id}_member{i}" for i in range(len(requests))]
        starts.append(run_ids)
        backend.running = list(run_ids)
        live["status"] = _Status(backend.current_job_id, "loading_model")
        return [
            TrainingJobResponse(job_id = run_id, status = "queued", message = "queued")
            for run_id in run_ids
        ]

    monkeypatch.setattr(training_routes, "start_training_batch", start_training_batch)

    def history_rows(run_ids, started_at):
        for run_id in run_ids:
            studio_db.create_run(
                id = run_id,
                model_name = "unsloth/Qwen3-0.6B",
                dataset_name = "chats.jsonl",
                config_json = json.dumps({"model_name": "unsloth/Qwen3-0.6B"}),
                started_at = started_at,
                total_steps = 4,
            )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = []
        for suffix in "abcd":
            clock["t"] += 10
            model = "unsloth/Qwen3-1.7B" if suffix in "bd" else "unsloth/Qwen3-0.6B"
            jobs.append(
                await client.fine_tuning.jobs.create(
                    model = model, training_file = uploaded.id, suffix = suffix
                )
            )
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        history_rows(starts[0], "2026-11-04T00:00:00+00:00")
        await openai_fine_tuning._drain_once()
        first = await client.fine_tuning.jobs.list(limit = 2)
        history_rows(starts[1], "2026-11-04T00:10:00+00:00")
        rest = await client.fine_tuning.jobs.list(limit = 10, after = first.data[-1].id)
        return jobs, [j.id for j in first.data] + [j.id for j in rest.data]

    jobs, seen = _run(flow())
    assert seen == [jobs[i].id for i in (3, 2, 1, 0)]


@pytest.mark.parametrize(
    "model_kind",
    [
        "diffusion",
        "lora_adapter",
        "moe",
        "moe_text_config",
        "moe_router_aux_loss",
        "mamba",
        "jamba",
        "lfm2",
        "ssm_keys",
        "dense",
        "hybrid_gated_deltanet",
        "qwen3_5_moe",
        "qwen3_moe",
        "qwen3_moe_on_transformers_4",
        "qwen3_moe_without_expert_kernels",
    ],
)
def test_only_checked_model_kinds_batch(batch_queue, monkeypatch, model_kind):
    import utils.models.diffusion_lm as diffusion_lm
    import utils.models.model_config as model_config
    import utils.transformers_version as transformers_version

    monkeypatch.setattr(
        transformers_version,
        "get_transformers_tier",
        lambda *a, **k: "457" if model_kind == "qwen3_moe_on_transformers_4" else "default",
    )
    monkeypatch.setattr(
        openai_fine_tuning,
        "_multi_lora_experts_available",
        lambda: model_kind != "qwen3_moe_without_expert_kernels",
    )

    client, calls, _live, gpu, _backend = batch_queue
    monkeypatch.setattr(openai_fine_tuning, "_trains_alone", _REAL_TRAINS_ALONE)
    monkeypatch.setattr(
        diffusion_lm,
        "diffusion_lm_profile_for_model",
        lambda *a, **k: "dream" if model_kind == "diffusion" else None,
    )
    monkeypatch.setattr(
        model_config,
        "get_base_model_from_lora_identifier",
        lambda *a, **k: "unsloth/Qwen3-0.6B" if model_kind == "lora_adapter" else None,
    )
    configs = {
        "moe": {"model_type": "deepseek_v3", "n_routed_experts": 256},
        "moe_text_config": {"text_config": {"num_local_experts": 8}},
        "moe_router_aux_loss": {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_experts": 256,
                "output_router_logits": True,
            },
        },
        "mamba": {"model_type": "mamba2", "state_size": 128},
        "jamba": {
            "model_type": "jamba",
            "num_experts": 16,
            "layers_block_type": ["mamba", "attention"],
        },
        "lfm2": {"model_type": "lfm2", "layer_types": ["conv", "full_attention"]},
        "ssm_keys": {"model_type": "custom_hybrid", "mamba_d_state": 64},
        "hybrid_gated_deltanet": {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "layer_types": ["linear_attention", "full_attention"],
            },
        },
        "qwen3_5_moe": {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_experts": 256,
                "layer_types": ["linear_attention", "full_attention"],
            },
        },
        "qwen3_moe": {"model_type": "qwen3_moe", "num_experts": 128},
        "qwen3_moe_on_transformers_4": {"model_type": "qwen3_moe", "num_experts": 128},
        "qwen3_moe_without_expert_kernels": {"model_type": "qwen3_moe", "num_experts": 128},
    }
    monkeypatch.setattr(
        model_config,
        "_read_raw_config_json",
        lambda *a, **k: configs.get(model_kind, {"model_type": "qwen3", "num_experts": None}),
    )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        for s in "ab":
            await _create(client, uploaded.id, suffix = s)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()

    _run(flow())
    if model_kind in ("dense", "hybrid_gated_deltanet", "qwen3_5_moe", "qwen3_moe"):
        assert len(calls["batch"]) == 1 and calls["start"] == []
    else:
        assert calls["batch"] == []
        assert [r.project_name for r in calls["start"]] == ["a"]


def test_mlx_jobs_never_batch(batch_queue, monkeypatch):
    client, calls, _live, gpu, _backend = batch_queue
    monkeypatch.setattr(_training_core, "should_use_mlx_training_backend", lambda: True)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        for s in "ab":
            await _create(client, uploaded.id, suffix = s)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()

    _run(flow())
    assert calls["batch"] == []
    assert [r.project_name for r in calls["start"]] == ["a"]


def test_batch_members_stay_validating_until_their_first_step(batch_queue):
    client, _calls, live, gpu, _backend = batch_queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = s) for s in "ab"]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        seen = []

        for phase in ("loading_model", "training", "configuring", "loading_dataset", "training"):
            live["status"].phase = phase
            seen.append([(await client.fine_tuning.jobs.retrieve(j.id)).status for j in jobs])
        live["status"].phase = "error"
        failed = [(await client.fine_tuning.jobs.retrieve(j.id)).status for j in jobs]
        return seen, failed

    seen, failed = _run(flow())
    assert seen == [["validating_files"] * 2] * 5
    assert failed == ["failed"] * 2


def test_a_single_or_incompatible_job_starts_exactly_as_before(batch_queue):
    client, calls, _live, gpu, _backend = batch_queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        first = await _create(client, uploaded.id, suffix = "first", extra_body = {"use_dora": True})
        second = await _create(client, uploaded.id, suffix = "second")
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        queued = list(openai_fine_tuning._QUEUE)
        await openai_fine_tuning._drain_once()
        return first, second, queued

    first, second, queued = _run(flow())
    assert calls["batch"] == []
    assert [q.alias for q in queued] == [second.id]
    assert [(r.project_name, r.use_dora) for r in calls["start"]] == [
        ("first", True),
        ("second", False),
    ]
    assert queued[0].request is calls["start"][1]


def test_a_refused_batch_starts_the_head_alone(batch_queue, monkeypatch):
    client, calls, _live, gpu, _backend = batch_queue

    async def refuse(requests, current_subject, via_api_key):
        calls["batch"].append(requests)
        raise openai_fine_tuning.HTTPException(status_code = 400, detail = "Dataset not found")

    monkeypatch.setattr(training_routes, "start_training_batch", refuse)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        first = await _create(client, uploaded.id, suffix = "first")
        second = await _create(client, uploaded.id, suffix = "second")
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return (
            first,
            second,
            [await client.fine_tuning.jobs.retrieve(j.id) for j in (first, second)],
        )

    first, second, (started, waiting) = _run(flow())
    assert len(calls["batch"]) == 1
    assert [r.project_name for r in calls["start"]] == ["first"]
    assert started.id == first.id and started.status == "validating_files"
    assert waiting.status == "queued"


def test_a_busy_gpu_keeps_the_whole_batch_queued(batch_queue, monkeypatch):
    client, calls, _live, gpu, _backend = batch_queue

    async def busy(requests, current_subject, via_api_key):
        calls["batch"].append(requests)
        return [
            TrainingJobResponse(
                job_id = "",
                status = "error",
                message = "Training is already in progress.",
                error = "Training already active",
            )
            for _ in requests
        ]

    monkeypatch.setattr(training_routes, "start_training_batch", busy)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = name) for name in ("a", "b")]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return [await client.fine_tuning.jobs.retrieve(j.id) for j in jobs]

    assert [j.status for j in _run(flow())] == ["queued", "queued"]
    assert calls["start"] == [] and len(openai_fine_tuning._QUEUE) == 2


def test_cancelling_a_batched_job_stops_only_that_adapter(batch_queue):
    client, calls, _live, gpu, backend = batch_queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = name) for name in ("a", "b", "c")]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        cancelled = await client.fine_tuning.jobs.cancel(jobs[1].id)
        after = [await client.fine_tuning.jobs.retrieve(j.id) for j in jobs]
        return jobs, cancelled, after

    jobs, cancelled, after = _run(flow())
    assert backend.stopped == ["job_20261021_101500_member01"]
    assert calls["stop"] == []
    assert cancelled.status == "cancelled"
    assert [j.status for j in after] == ["validating_files", "cancelled", "validating_files"]


def test_cancelling_the_last_running_batched_job_stops_the_run(batch_queue):
    client, calls, live, gpu, backend = batch_queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = name) for name in ("a", "b")]
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        backend.running = ["job_20261021_101500_member01"]
        cancelled = await client.fine_tuning.jobs.cancel(jobs[1].id)
        return cancelled, await client.fine_tuning.jobs.retrieve(jobs[1].id)

    cancelled, retrieved = _run(flow())
    assert backend.stopped == []
    [stop] = calls["stop"]

    assert stop.expected_job_id == live["status"].job_id == "job_20261021_101500_batch000"
    assert cancelled.status == retrieved.status == "cancelled"


def test_a_batched_job_cancelled_while_its_start_was_in_flight_stops_alone(
    batch_queue, monkeypatch
):
    client, calls, _live, gpu, backend = batch_queue
    start = training_routes.start_training_batch
    release, entered = asyncio.Event(), asyncio.Event()

    async def slow(requests, current_subject, via_api_key):
        entered.set()
        await release.wait()
        return await start(requests, current_subject = current_subject, via_api_key = via_api_key)

    monkeypatch.setattr(training_routes, "start_training_batch", slow)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = name) for name in ("a", "b")]
        gpu["busy"] = False
        drain = asyncio.ensure_future(openai_fine_tuning._drain_once())
        await entered.wait()
        await client.fine_tuning.jobs.cancel(jobs[0].id)
        release.set()
        await drain
        return [await client.fine_tuning.jobs.retrieve(j.id) for j in jobs]

    assert [j.status for j in _run(flow())] == ["cancelled", "validating_files"]
    assert backend.stopped == ["job_20261021_101500_member00"] and calls["stop"] == []


def test_start_training_batch_runs_one_start_carrying_every_member(monkeypatch):
    seen = []

    async def prepare(request, via_api_key):
        seen.append(("prepare", request.project_name))

    async def admit(request, current_subject, via_api_key):
        seen.append(("admit", request.project_name))
        return training_routes._AdmittedTrainingRequest(
            training_kwargs = {"project_name": request.project_name, "lora_r": request.lora_r},
            hf_token = None,
            resume_run = None,
        )

    async def start_training(request, current_subject, via_api_key):
        members = training_routes._batch_members_for_start({"project_name": request.project_name})
        seen.append(("start", request.project_name, members))
        return TrainingJobResponse(job_id = "job_batch", status = "queued", message = "queued")

    monkeypatch.setattr(training_routes, "_prepare_training_request", prepare)
    monkeypatch.setattr(training_routes, "_admit_training_request", admit)
    monkeypatch.setattr(training_routes, "start_training", start_training)
    requests = [
        openai_fine_tuning.TrainingStartRequest(
            model_name = "unsloth/Qwen3-0.6B",
            training_type = "LoRA/QLoRA",
            format_type = "auto",
            project_name = name,
            lora_r = r,
        )
        for name, r in (("a", 8), ("b", 16), ("c", 32))
    ]
    responses = _run(
        training_routes.start_training_batch(requests, current_subject = "unsloth", via_api_key = True)
    )
    assert seen[:4] == [("prepare", "b"), ("admit", "b"), ("prepare", "c"), ("admit", "c")]
    [(_, first, members)] = seen[4:]
    assert first == "a"
    run_ids = [r.job_id for r in responses]
    assert len(set(run_ids)) == 3 and "job_batch" not in run_ids
    assert all(r.status == "queued" for r in responses)
    assert members == [
        {"run_id": run_ids[0], "project_name": "a"},
        {"run_id": run_ids[1], "project_name": "b", "lora_r": 16},
        {"run_id": run_ids[2], "project_name": "c", "lora_r": 32},
    ]
    assert training_routes._batch_members_for_start({"project_name": "a"}) is None


def test_start_training_batch_refuses_resume_and_reports_a_busy_gpu_per_member(monkeypatch):
    async def busy(request, current_subject, via_api_key):
        return TrainingJobResponse(
            job_id = "job_running",
            status = "error",
            message = "Training is already in progress.",
            error = "Training already active",
        )

    async def admit(request, current_subject, via_api_key):
        return training_routes._AdmittedTrainingRequest({}, None, None)

    async def prepare(request, via_api_key):
        return None

    monkeypatch.setattr(training_routes, "_prepare_training_request", prepare)
    monkeypatch.setattr(training_routes, "_admit_training_request", admit)
    monkeypatch.setattr(training_routes, "start_training", busy)

    def request(**fields):
        return openai_fine_tuning.TrainingStartRequest(
            model_name = "unsloth/Qwen3-0.6B",
            training_type = "LoRA/QLoRA",
            format_type = "auto",
            **fields,
        )

    with pytest.raises(ValueError):
        _run(
            training_routes.start_training_batch(
                [request(), request(resume_from_checkpoint = "/runs/old")],
                current_subject = "unsloth",
                via_api_key = True,
            )
        )
    responses = _run(
        training_routes.start_training_batch(
            [request(), request()], current_subject = "unsloth", via_api_key = True
        )
    )
    assert [(r.status, r.error, r.job_id) for r in responses] == [
        ("error", "Training already active", "job_running")
    ] * 2


@pytest.mark.parametrize(
    "model", ["//attacker.example/share/model", "\\\\attacker.example\\share\\model"]
)
def test_batch_probe_never_touches_a_managed_share_path(batch_queue, monkeypatch, model):
    import utils.models.diffusion_lm as diffusion_lm
    import utils.models.model_config as model_config
    from utils.account_context import AccountContext

    client, calls, _live, gpu, _backend = batch_queue
    monkeypatch.setattr(openai_fine_tuning, "_trains_alone", _REAL_TRAINS_ALONE)
    touched = []

    def record(name):
        def probe(model_name, *a, **k):
            touched.append((name, model_name))
            return {"model_type": "qwen3"} if name == "config" else None

        return probe

    monkeypatch.setattr(diffusion_lm, "diffusion_lm_profile_for_model", record("diffusion"))
    monkeypatch.setattr(model_config, "get_base_model_from_lora_identifier", record("lora"))
    monkeypatch.setattr(model_config, "_read_raw_config_json", record("config"))
    managed = AccountContext("acct-managed-1", "alice")
    monkeypatch.setattr(openai_fine_tuning, "current_account", lambda: managed)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        for s in "ab":
            await client.fine_tuning.jobs.create(
                model = model,
                training_file = uploaded.id,
                suffix = s,
                extra_body = {"packing": False},
            )
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()

    _run(flow())
    assert touched == []
    assert calls["batch"] == []


def test_batch_probe_unchanged_for_the_owner(batch_queue, monkeypatch):
    import utils.models.diffusion_lm as diffusion_lm
    import utils.models.model_config as model_config

    client, calls, _live, gpu, _backend = batch_queue
    monkeypatch.setattr(openai_fine_tuning, "_trains_alone", _REAL_TRAINS_ALONE)
    touched = []
    monkeypatch.setattr(
        diffusion_lm, "diffusion_lm_profile_for_model", lambda m, *a, **k: touched.append(m)
    )
    monkeypatch.setattr(model_config, "get_base_model_from_lora_identifier", lambda *a, **k: None)
    monkeypatch.setattr(
        model_config, "_read_raw_config_json", lambda *a, **k: {"model_type": "qwen3"}
    )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        for s in "ab":
            await _create(client, uploaded.id, suffix = s, extra_body = {"packing": False})
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()

    _run(flow())
    assert touched and len(calls["batch"]) == 1


@pytest.mark.parametrize("trains_alone", [False, True])
@pytest.mark.parametrize("cancelled", [0, 1])
def test_a_job_cancelled_during_the_batch_model_probe_never_starts(
    batch_queue, monkeypatch, cancelled, trains_alone
):
    client, calls, _live, gpu, _backend = batch_queue
    jobs = []

    async def probe_then_cancel(request):
        cancelled_job = await client.fine_tuning.jobs.cancel(jobs[cancelled].id)
        assert cancelled_job.status == "cancelled"
        return trains_alone

    monkeypatch.setattr(openai_fine_tuning, "_trains_alone", probe_then_cancel)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs.extend([await _create(client, uploaded.id, suffix = s) for s in "ab"])
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        return [(await client.fine_tuning.jobs.retrieve(j.id)).status for j in jobs]

    statuses = _run(flow())
    started = [r.project_name for r in calls["start"]] + [
        r.project_name for requests, *_ in calls["batch"] for r in requests
    ]
    assert "ab"[cancelled] not in started
    assert statuses[cancelled] == "cancelled"
    assert calls["stop"] == []


@pytest.mark.parametrize("persisted", [False, True])
def test_a_cancelled_queued_job_is_still_listed_after_a_newer_job_starts(
    queue, monkeypatch, tmp_path, persisted
):
    client, _calls, _live, gpu = queue

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        monkeypatch.setattr(openai_fine_tuning, "_now", lambda: 1_700_000_000)
        old = await _create(client, uploaded.id)
        await client.fine_tuning.jobs.cancel(old.id)
        before = await client.fine_tuning.jobs.list()
        gpu["busy"] = False
        newer = await _create(client, uploaded.id)
        if persisted:
            _finished_run(tmp_path, tmp_path / "chats.jsonl")
        after = [job.id async for job in client.fine_tuning.jobs.list(limit = 1)]
        retrieved = await client.fine_tuning.jobs.retrieve(old.id)
        return old, newer, before, after, retrieved

    old, newer, before, after, retrieved = _run(flow())
    assert [(job.id, job.status) for job in before.data] == [(old.id, "cancelled")]
    assert retrieved.status == "cancelled"
    assert after == [newer.id, old.id]


def test_a_job_that_starts_during_a_list_request_is_listed_once(queue, monkeypatch):
    client, _calls, live, gpu = queue
    status_read = training_routes.get_training_status

    async def start_mid_request(current_subject, via_api_key):
        if openai_fine_tuning._QUEUE:
            gpu["busy"] = False
            await openai_fine_tuning._drain_once()
        return await status_read(current_subject = current_subject, via_api_key = via_api_key)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        job = await _create(client, uploaded.id)
        monkeypatch.setattr(training_routes, "get_training_status", start_mid_request)
        page = await client.fine_tuning.jobs.list(limit = 1)
        return job, page

    job, page = _run(flow())
    assert [(j.id, j.status) for j in page.data] == [(job.id, "validating_files")]
    assert page.has_more is False


def test_a_cursor_on_a_job_that_starts_during_the_request_still_pages(queue, monkeypatch):
    client, _calls, _live, gpu = queue
    status_read = training_routes.get_training_status

    def history_row(run_id, started_at):
        studio_db.create_run(
            id = run_id,
            model_name = "unsloth/Qwen3-0.6B",
            dataset_name = "chats.jsonl",
            config_json = json.dumps({"model_name": "unsloth/Qwen3-0.6B"}),
            started_at = started_at,
            total_steps = 4,
        )

    async def start_mid_request(current_subject, via_api_key):
        if openai_fine_tuning._QUEUE:
            gpu["busy"] = False
            await openai_fine_tuning._drain_once()
            history_row("job_20261021_101500_ab12cd34", "2026-11-04T00:00:00+00:00")
        return await status_read(current_subject = current_subject, via_api_key = via_api_key)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        history_row("job_20261001_000000_old", "2026-10-01T00:00:00+00:00")
        job = await _create(client, uploaded.id)
        first = await client.fine_tuning.jobs.list(limit = 1)
        monkeypatch.setattr(training_routes, "get_training_status", start_mid_request)
        second = await client.fine_tuning.jobs.list(after = job.id, limit = 1)
        return job, first, second

    job, first, second = _run(flow())
    assert [j.id for j in first.data] == [job.id]
    assert [j.id for j in second.data] == ["ftjob_20261001_000000_old"]


def test_a_newer_cancelled_job_is_not_skipped_when_an_older_one_starts_between_pages(
    queue, monkeypatch
):
    client, _calls, _live, gpu = queue
    clock = {"t": 1_700_000_000}
    monkeypatch.setattr(openai_fine_tuning, "_now", lambda: clock["t"])

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        older = await _create(client, uploaded.id)
        clock["t"] += 10
        newer = await _create(client, uploaded.id)
        await client.fine_tuning.jobs.cancel(newer.id)
        first = await client.fine_tuning.jobs.list(limit = 1)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        rest = [j.id async for j in client.fine_tuning.jobs.list(after = first.data[0].id, limit = 1)]
        return older, newer, [j.id for j in first.data] + rest

    older, newer, walked = _run(flow())
    assert sorted(walked) == sorted([older.id, newer.id])


def test_a_batch_lists_in_the_same_order_before_and_after_its_history_rows_exist(monkeypatch):
    started = "2026-11-04T00:00:00+00:00"
    monkeypatch.setattr(
        openai_fine_tuning, "_RUN_ALIAS", {"run_a": "ftjob_qa", "run_c": "ftjob_qc"}
    )
    monkeypatch.setattr(openai_fine_tuning, "_ALIAS_CREATED", {"ftjob_qa": 100, "ftjob_qc": 300})
    waiting = [{"id": "ftjob_qb", "created_at": 200}]
    live = {
        "run_c": {"id": "ftjob_qc", "created_at": 300},
        "run_a": {"id": "ftjob_qa", "created_at": 100},
    }

    def order(rows, live_rows):
        monkeypatch.setattr(openai_fine_tuning, "_iter_runs", lambda: iter(rows))
        return [
            openai_fine_tuning._RUN_ALIAS.get(row["id"], row["id"]) if is_run else row["id"]
            for row, is_run in openai_fine_tuning._ordered_jobs(waiting, live_rows)
        ]

    persisted = [{"id": "run_a", "started_at": started}, {"id": "run_c", "started_at": started}]
    assert order([], live) == order(persisted, {}) == ["ftjob_qc", "ftjob_qb", "ftjob_qa"]


def test_a_cursor_on_a_batch_member_that_starts_during_the_request_still_pages(
    batch_queue, monkeypatch, tmp_path
):
    client, _calls, live, gpu, _backend = batch_queue
    previous = _finished_run(tmp_path, tmp_path / "chats.jsonl")
    live["status"] = _Status(previous, "completed")
    get_run = openai_fine_tuning.get_run
    reached, released = threading.Event(), threading.Event()

    def get_run_held_once(run_id):
        if run_id == previous and not reached.is_set():
            reached.set()
            released.wait(timeout = 30)
        return get_run(run_id)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = s) for s in "ab"]
        first = await client.fine_tuning.jobs.list(limit = 1)
        monkeypatch.setattr(openai_fine_tuning, "get_run", get_run_held_once)
        second = asyncio.ensure_future(
            client.fine_tuning.jobs.list(after = first.data[0].id, limit = 1)
        )
        await asyncio.to_thread(reached.wait, 30)
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        released.set()
        page = await second
        seen = [j.id for j in first.data + page.data]
        while page.has_more:
            page = await client.fine_tuning.jobs.list(after = page.data[-1].id, limit = 1)
            seen += [j.id for j in page.data]
        return jobs, seen

    jobs, seen = _run(flow())
    newest = jobs[::-1]
    assert seen == [job.id for job in newest] + ["ftjob_20261021_101500_ab12cd34"]


def test_a_batch_lists_under_its_job_ids_while_its_start_is_being_saved(batch_queue, monkeypatch):
    client, _calls, _live, gpu, _backend = batch_queue
    save_alias = openai_fine_tuning._save_alias
    saving, release = threading.Event(), threading.Event()

    def save_alias_held(*args):
        saving.set()
        release.wait(timeout = 30)
        save_alias(*args)

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = [await _create(client, uploaded.id, suffix = s) for s in "ab"]
        monkeypatch.setattr(openai_fine_tuning, "_save_alias", save_alias_held)
        gpu["busy"] = False
        drain = asyncio.ensure_future(openai_fine_tuning._drain_once())
        await asyncio.to_thread(saving.wait, 30)
        listed = await client.fine_tuning.jobs.list()
        release.set()
        await drain
        return jobs, listed

    jobs, listed = _run(flow())
    newest = jobs[::-1]
    assert [(j.id, j.status) for j in listed.data] == [
        (job.id, "validating_files") for job in newest
    ]


@pytest.mark.parametrize("step", [10, 0])
@pytest.mark.parametrize("persisted", [False, True, "restart"])
def test_a_job_batched_past_keeps_its_place_when_it_starts_between_pages(
    batch_queue, monkeypatch, persisted, step
):
    client, calls, live, gpu, _backend = batch_queue
    clock = {"t": 1_700_000_000}
    monkeypatch.setattr(openai_fine_tuning, "_now", lambda: clock["t"])

    def history_row(run_id, started_at):
        studio_db.create_run(
            id = run_id,
            model_name = "unsloth/Qwen3-0.6B",
            dataset_name = "chats.jsonl",
            config_json = json.dumps({"model_name": "unsloth/Qwen3-0.6B"}),
            started_at = started_at,
            total_steps = 4,
        )

    async def flow():
        uploaded = await client.files.create(file = ("chats.jsonl", CHATS), purpose = "fine-tune")
        jobs = []
        for suffix, extra in (("a", {}), ("b", {"use_dora": True}), ("c", {})):
            jobs.append(await _create(client, uploaded.id, suffix = suffix, extra_body = extra))
            clock["t"] += step
        gpu["busy"] = False
        await openai_fine_tuning._drain_once()
        for i in range(2):
            history_row(f"job_20261021_101500_member0{i}", "2026-11-04T00:00:00+00:00")
        first = await client.fine_tuning.jobs.list(limit = 1)
        listed = [j.id async for j in client.fine_tuning.jobs.list(limit = 1)]
        await openai_fine_tuning._drain_once()
        if persisted:
            history_row("job_20261021_101500_ab12cd34", "2026-11-04T01:00:00+00:00")
            live["status"] = _Status()
        if persisted == "restart":
            _forget_aliases(monkeypatch)
        after = first.data[0].id
        rest = [j.id async for j in client.fine_tuning.jobs.list(after = after, limit = 1)]
        return jobs, listed, [j.id for j in first.data] + rest

    (a, b, c), listed, walked = _run(flow())
    assert [r.project_name for r in calls["batch"][0][0]] == ["a", "c"]
    assert [r.project_name for r in calls["start"]] == ["b"]
    assert walked == listed == [c.id, b.id, a.id]


def test_history_lists_by_creation_time_reading_only_what_a_page_needs(monkeypatch):
    runs = [
        {"id": f"run{i}", "started_at": f"2026-11-04T00:{i // 60:02d}:{i % 60:02d}+00:00"}
        for i in range(300)
    ]
    created = {run["id"]: openai_fine_tuning._unix(run["started_at"]) for run in runs}
    created["run_waited"] = created["run0"] - 1
    history = [{"id": "run_waited", "started_at": "2026-11-04T01:00:00+00:00"}] + runs[::-1]
    pulled = []

    def iter_runs():
        for run in history:
            pulled.append(run["id"])
            yield run

    monkeypatch.setattr(openai_fine_tuning, "_iter_runs", iter_runs)
    monkeypatch.setattr(openai_fine_tuning, "_RUN_ALIAS", {i: f"ftjob_q{i}" for i in created})
    monkeypatch.setattr(
        openai_fine_tuning, "_ALIAS_CREATED", {f"ftjob_q{i}": t for i, t in created.items()}
    )

    jobs = openai_fine_tuning._ordered_jobs([], {})
    assert [next(jobs)[0]["id"] for _ in range(3)] == ["run299", "run298", "run297"]
    assert len(pulled) <= 6
    assert [row["id"] for row, _ in jobs][-2:] == ["run0", "run_waited"]


def test_a_queue_poll_reads_one_history_row_when_the_backlog_waited(monkeypatch):
    history = [
        {"id": f"run{i}", "started_at": f"2026-11-04T{i // 60:02d}:{i % 60:02d}:00+00:00"}
        for i in range(300, 0, -1)
    ]
    t0 = openai_fine_tuning._unix("2026-11-04T00:00:00+00:00")
    pulled = []

    def iter_runs():
        for run in history:
            pulled.append(run["id"])
            yield run

    monkeypatch.setattr(openai_fine_tuning, "_iter_runs", iter_runs)
    monkeypatch.setattr(
        openai_fine_tuning, "_RUN_ALIAS", {r["id"]: f"ftjob_{r['id']}" for r in history}
    )
    monkeypatch.setattr(
        openai_fine_tuning, "_ALIAS_CREATED", {f"ftjob_{r['id']}": t0 for r in history}
    )
    pending = [{"id": f"ftjob_q{i}", "created_at": t0 + 20_000 - i} for i in range(2)]

    page = openai_fine_tuning._list_job_page(pending, {}, None, None, 1, False)
    assert [job["id"] for job in page["data"]] == ["ftjob_q0"]
    assert page["has_more"] and pulled == ["run300"]

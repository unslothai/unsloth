# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The OpenAI fine-tuning API, driven by the real ``openai`` SDK against the router in-process.

Training itself is stubbed at the ``/api/train`` handlers the router calls; the upload directory and the
training history are the real ones, in the per-test Studio home."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

openai = pytest.importorskip("openai")

from auth.authentication import authenticated_via_api_key, get_current_subject  # noqa: E402
from models.training import TrainingJobResponse  # noqa: E402
from routes import openai_fine_tuning, training as training_routes  # noqa: E402
from storage import studio_db  # noqa: E402

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
        live["status"] = _Status("job_20261021_101500_ab12cd34", "loading_model")
        return TrainingJobResponse(
            job_id = "job_20261021_101500_ab12cd34", status = "queued", message = "queued"
        )

    async def stop_training(body, current_subject):
        calls["stop"].append(body)
        return training_routes.TrainingStopResponse(status = "stopped", message = "Stop requested")

    async def get_training_status(current_subject, via_api_key):
        return live["status"]

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
    # No history row until the worker is up: the live training status answers instead.
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
        # The worker is still finishing its step, so the row still reads running.
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

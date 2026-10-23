# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import contextlib
import json
import time

import httpx
import pytest
from fastmcp.exceptions import ToolError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from studio_mcp import video_queue

from .mcp_harness import call_tool, fake_studio, make_caller, openai_error, run_tool, served

BUSY = openai_error(video_queue.BUSY_MESSAGE, 409, type = "conflict_error")
MODEL_CHANGED = openai_error(
    "The requested video model changed before generation was reserved.",
    409,
    type = "conflict_error",
    param = "model",
    code = "model_changed",
)
GPU_BUSY = JSONResponse(
    {"error": "gpu_busy", "message": "Another account is generating.", "retry_after": 0},
    status_code = 409,
    headers = {"Retry-After": "0"},
)
ETA = 30.0


@pytest.fixture(autouse = True)
def fresh_queue(monkeypatch):
    monkeypatch.setattr(video_queue, "POLL_S", 0.01)
    monkeypatch.setattr(video_queue, "RETRY_S", 0.01)
    video_queue._reset()
    yield
    video_queue._reset()


class OneVideoStudio:
    """A fake Unsloth Studio that makes one video at a time: a POST while one runs is the busy 409, and a video runs until the test calls ``finish``. ``answers`` are POST answers given first, in order."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.active = None
        self.made = 0
        self.done = set()
        self.app = fake_studio(
            {
                ("POST", "/v1/videos"): self._create,
                ("GET", "/v1/videos/{video_id}"): self._get,
                ("GET", video_queue.PROGRESS): self._progress,
            }
        )

    def _create(self, request, body):
        if self.answers:
            return self.answers.pop(0)
        if self.active is not None:
            return BUSY
        self.made += 1
        self.active = f"video-{self.made}"
        return {
            "id": self.active,
            "status": "queued",
            "progress": 0,
            "prompt": json.loads(body)["prompt"],
        }

    def _get(self, request, body):
        video_id = request.path_params["video_id"]
        status = "completed" if video_id in self.done else "in_progress"
        return {"id": video_id, "object": "video", "status": status, "progress": 50}

    def _progress(self, request, body):
        return {"active": self.active is not None, "eta_seconds": ETA if self.active else None}

    def finish(self):
        self.done.add(self.active)
        self.active = None

    def prompts(self):
        return [json.loads(c[3])["prompt"] for c in self.app.state.calls if c[0] == "POST"]


@contextlib.contextmanager
def open_client(monkeypatch, fake):
    # One client for the whole test: the queue's background task runs on its event loop.
    from mcp_server import create_studio_mcp
    with TestClient(served(create_studio_mcp(), fake.app, monkeypatch = monkeypatch)) as http:
        yield http


def generate(http, prompt):
    return call_tool(http, "generate_video", {"prompt": prompt, "seconds": "4"})[
        "structuredContent"
    ]


def get_video(http, video_id):
    result = call_tool(http, "get_job", {"kind": "video", "id": video_id})
    assert result["isError"] is False, result
    return result["structuredContent"]


def wait_until(http, video_id, check):
    deadline = time.monotonic() + 30.0
    while True:
        job = get_video(http, video_id)
        if check(job) or time.monotonic() > deadline:
            return job
        time.sleep(0.02)


def test_a_shot_list_runs_in_order_one_video_after_another(monkeypatch):
    fake = OneVideoStudio()
    with open_client(monkeypatch, fake) as http:
        a, b, c = generate(http, "A"), generate(http, "B"), generate(http, "C")
        assert a["id"] == "video-1" and a["queue_position"] is None
        assert [b["status"], b["queue_position"], c["status"], c["queue_position"]] == [
            "queued",
            1,
            "queued",
            2,
        ]
        assert b["id"].startswith("vq_") and b["seconds"] == "4"
        # C queues behind B without trying: only B met the busy answer.
        assert fake.prompts() == ["A", "B"]

        fake.finish()
        job_b = wait_until(http, b["id"], lambda job: job["status"] != "queued")
        assert job_b["id"] == "video-2" and job_b["status"] == "in_progress"
        assert fake.prompts() == ["A", "B", "B"]
        assert get_video(http, c["id"])["queue_position"] == 1

        fake.finish()
        job_c = wait_until(http, c["id"], lambda job: job["id"] == "video-3")
        assert job_c["id"] == "video-3"
        assert fake.prompts() == ["A", "B", "B", "C"]


def test_a_queued_video_reports_when_it_should_start(monkeypatch):
    fake = OneVideoStudio()
    with open_client(monkeypatch, fake) as http:
        generate(http, "A")
        b, c = generate(http, "B"), generate(http, "C")
        # Next in line starts when the running video's denoise ends; each later one a clip later.
        assert b["eta_s"] == ETA
        assert c["eta_s"] > ETA
        job_b, job_c = get_video(http, b["id"]), get_video(http, c["id"])
    assert [job_b["status"], job_b["queue_position"], job_b["eta_s"]] == ["queued", 1, ETA]
    assert job_c["queue_position"] == 2 and job_c["eta_s"] > job_b["eta_s"]


def test_cancelling_a_waiting_video_withdraws_it_and_moves_the_rest_up(monkeypatch):
    fake = OneVideoStudio()
    with open_client(monkeypatch, fake) as http:
        generate(http, "A")
        b, c = generate(http, "B"), generate(http, "C")
        result = call_tool(http, "cancel", {"kind": "video", "id": b["id"]})["structuredContent"]
        assert result["cancelled"] is True
        assert result["message"] == "Removed the queued video before it started."
        assert get_video(http, c["id"])["queue_position"] == 1
        assert get_video(http, b["id"])["status"] == "cancelled"

        fake.finish()
        job_c = wait_until(http, c["id"], lambda job: job["status"] != "queued")
        assert job_c["id"] == "video-2"
        # B met the busy answer once and was never sent again.
        assert fake.prompts() == ["A", "B", "C"]
        again = call_tool(http, "cancel", {"kind": "video", "id": c["id"]})["structuredContent"]
    assert again["cancelled"] is False
    assert (
        again["message"]
        == "It already started as video-2; cancel(kind='video') stops the running video."
    )


def test_the_queue_waits_out_gpu_busy_and_fails_a_refused_video_with_its_reason(monkeypatch):
    refused = openai_error("Unsupported size 1x1.", 400, param = "size")
    fake = OneVideoStudio()
    with open_client(monkeypatch, fake) as http:
        generate(http, "A")
        b, c = generate(http, "B"), generate(http, "C")
        fake.answers = [GPU_BUSY, refused]
        fake.finish()
        job_b = wait_until(http, b["id"], lambda job: job["status"] != "queued")
        job_c = wait_until(http, c["id"], lambda job: job["status"] != "queued")
    assert job_b["status"] == "failed"
    assert job_b["error"] == "Invalid arguments: Unsupported size 1x1."
    # The gpu_busy try is retried, and the next video still goes after the refused one.
    assert fake.prompts() == ["A", "B", "B", "B", "C"]
    assert job_c["id"] == "video-2"


def test_a_settled_record_drops_its_request_and_caller(monkeypatch):
    # The body can carry tens of MiB of reference images and the caller its token; a settled record
    # only answers get_job, so neither may outlive the queue's last use of them.
    refused = openai_error("Unsupported size 1x1.", 400, param = "size")
    fake = OneVideoStudio()
    with open_client(monkeypatch, fake) as http:
        generate(http, "A")
        b, c, d = generate(http, "B"), generate(http, "C"), generate(http, "D")
        assert call_tool(http, "cancel", {"kind": "video", "id": d["id"]})["structuredContent"][
            "cancelled"
        ]
        fake.answers = [refused]
        fake.finish()
        job_b = wait_until(http, b["id"], lambda job: job["status"] != "queued")
        job_c = wait_until(http, c["id"], lambda job: job["status"] != "queued")
        job_d = get_video(http, d["id"])
        items = {item.id: item for item in video_queue._items.values()}
    assert [job_b["status"], job_c["id"], job_d["status"]] == ["failed", "video-2", "cancelled"]
    assert {items[q["id"]].status for q in (b, c, d)} == {"failed", "submitted", "cancelled"}
    for q in (b, c, d):
        item = items[q["id"]]
        assert (item.body, item.hints, item.caller) == (None, None, None)


@pytest.mark.parametrize(
    "answer, text",
    [
        (MODEL_CHANGED, "[model_changed] The requested video model changed"),
        (GPU_BUSY, "GPU busy: Another account is generating. Retry after 0 s."),
        (openai_error("No video model is loaded.", 503), "Load a video model with load_model"),
    ],
)
def test_other_refusals_raise_as_before_and_queue_nothing(monkeypatch, answer, text):
    result, studio = run_tool(
        monkeypatch, {("POST", "/v1/videos"): answer}, "generate_video", {"prompt": "A"}
    )
    assert result["isError"] is True
    assert text in result["content"][0]["text"]
    assert [c[1] for c in studio.state.calls] == ["/v1/videos"]
    assert not video_queue._items and not video_queue._pumps


def test_a_queue_id_belongs_to_its_account():
    owner = make_caller(account_id = "owner")
    item = video_queue._enqueue(owner, {"prompt": "A"}, None)
    for call in (video_queue.lookup, video_queue.cancel):
        with pytest.raises(Exception, match = "No such video job"):
            call("someone-else", item.id)
    assert video_queue.lookup("owner", item.id) is item


def test_an_unknown_queue_id_reads_as_no_such_video_job(monkeypatch):
    result, studio = run_tool(monkeypatch, {}, "get_job", {"kind": "video", "id": "vq_nope"})
    assert result["isError"] is True
    assert "No such video job" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_a_full_queue_is_refused():
    owner = make_caller(account_id = "owner")
    for n in range(video_queue.MAX_WAITING):
        video_queue._enqueue(owner, {"prompt": str(n)}, None)
    with pytest.raises(Exception, match = "already queued"):
        video_queue._enqueue(owner, {"prompt": "one more"}, None)


def test_the_busy_message_matches_unsloth_studio():
    from core.inference.video_families import VIDEO_GENERATION_BUSY_MSG
    assert video_queue.BUSY_MESSAGE == VIDEO_GENERATION_BUSY_MSG


def test_a_video_that_starts_while_its_place_is_read_is_not_reported_queued(monkeypatch):
    item = video_queue.QueuedVideo(id = "vq_x", account_id = "a", caller = make_caller(), body = {})

    async def starts_meanwhile(queued):
        queued.status, queued.video_id = "submitted", "video-9"
        return 1, 0.0

    monkeypatch.setattr(video_queue, "_position_eta", starts_meanwhile)
    status = asyncio.run(video_queue.job_status(item))
    # The status read before the await goes with the position read during it.
    assert (status.status, status.queue_position) == ("queued", 1)


def test_the_queue_id_of_a_video_still_running_outlives_the_record_cap(monkeypatch):
    caller = make_caller(account_id = "a")
    running = video_queue.QueuedVideo(id = "vq_run", account_id = "a", caller = caller, body = {})
    running.status, running.video_id = "submitted", "video-1"
    video_queue._items["a:vq_run"] = running
    video_queue._running["a"] = ("video-1", 0.0)
    for n in range(video_queue.CAPACITY + 5):
        done = video_queue.QueuedVideo(id = f"vq_{n}", account_id = "b", caller = caller, body = {})
        done.status = "cancelled"
        video_queue._items[f"b:vq_{n}"] = done
    video_queue._evict()
    assert video_queue.lookup("a", "vq_run") is running
    assert len(video_queue._items) == video_queue.CAPACITY


def test_a_second_video_never_overtakes_one_entering_the_queue(monkeypatch):
    busy = httpx.Response(409, json = {"error": {"code": None, "message": video_queue.BUSY_MESSAGE}})
    started = httpx.Response(200, json = {"id": "video-2", "status": "queued"})
    posted = []

    async def forward(caller, method, path, **kwargs):
        posted.append(kwargs["json_body"]["prompt"])
        if len(posted) == 1:
            # The running video ends while the first, refused, is still on its way into the queue.
            await asyncio.sleep(0.05)
            return busy
        return started

    async def no_eta(item):
        return 1, None

    monkeypatch.setattr(video_queue, "forward", forward)
    monkeypatch.setattr(video_queue, "_ensure_pump", lambda account_id: None)
    monkeypatch.setattr(video_queue, "_position_eta", no_eta)
    caller = make_caller(account_id = "a")

    async def both():
        first = asyncio.create_task(video_queue.post_or_queue(caller, {"prompt": "A"}))
        await asyncio.sleep(0)
        second = asyncio.create_task(video_queue.post_or_queue(caller, {"prompt": "B"}))
        return await first, await second

    first, second = asyncio.run(both())
    assert (first.status, second.status) == ("queued", "queued")
    assert posted == ["A"]
    assert [item.body["prompt"] for item in video_queue._waiting["a"]] == ["A", "B"]


def test_the_last_queued_video_is_followed_to_its_end(monkeypatch):
    fake = OneVideoStudio()
    with open_client(monkeypatch, fake) as http:
        generate(http, "A")
        b = generate(http, "B")
        fake.finish()
        wait_until(http, b["id"], lambda job: job["id"] == "video-2")
        fake.finish()
        deadline = time.monotonic() + 10.0
        while video_queue._running and time.monotonic() < deadline:
            time.sleep(0.02)
        # Nothing waits on the finished clip any more, so its record can be dropped at the cap.
        assert video_queue._running == {}
        assert video_queue._ended(video_queue.lookup(*next(iter(video_queue._items)).split(":", 1)))


def test_the_queue_has_a_cap_across_accounts():
    for n in range(video_queue.CAPACITY):
        caller = make_caller(account_id = f"acct-{n}")
        video_queue._enqueue(caller, {"prompt": "x"}, None)
    with pytest.raises(ToolError, match = "already queued"):
        video_queue._enqueue(make_caller(account_id = "late"), {"prompt": "x"}, None)


def test_a_failed_status_read_does_not_end_the_wait_on_a_running_video(monkeypatch):
    reads = iter([None, "stall", {"status": "in_progress"}, {"status": "completed"}])

    async def try_json(caller, path, **kwargs):
        read = next(reads)
        if read == "stall":
            await asyncio.sleep(10)
        return read

    monkeypatch.setattr(video_queue, "try_json", try_json)
    monkeypatch.setattr(video_queue, "POLL_S", 0)
    monkeypatch.setattr(video_queue, "READ_S", 0.05)
    video_queue._running["a"] = ("video-1", 0.0)
    asyncio.run(video_queue._wait_for_running(make_caller(account_id = "a")))
    # It waited through a failed read and a stalled one to the video's real end.
    assert next(reads, "drained") == "drained" and "a" not in video_queue._running


def test_a_stalled_eta_read_leaves_the_eta_unknown(monkeypatch):
    async def stalled(caller, position):
        await asyncio.sleep(10)

    monkeypatch.setattr(video_queue, "_eta", stalled)
    monkeypatch.setattr(video_queue, "READ_S", 0.05)
    item = video_queue._enqueue(make_caller(account_id = "a"), {"prompt": "x"}, None)
    assert asyncio.run(video_queue._position_eta(item)) == (1, None)


def test_a_call_scoped_caller_is_told_to_retry_instead_of_queued():
    # The chat bridge's key is revoked when its call returns, so a clip queued for later would fail with 401.
    fake = OneVideoStudio()
    agent = make_caller(studio_app = fake.app)
    chat = make_caller(studio_app = fake.app, call_scoped = True)

    async def run():
        first = await video_queue.post_or_queue(agent, {"prompt": "A"})
        with pytest.raises(ToolError) as refused:
            await video_queue.post_or_queue(chat, {"prompt": "B"})
        return first, str(refused.value)

    first, refused = asyncio.run(run())
    assert first["id"] == "video-1"
    assert refused.startswith("[gpu_busy]") and "Try again once it finishes" in refused, refused
    assert not video_queue._waiting.get("owner") and not video_queue._pumps
    assert fake.prompts() == ["A", "B"]

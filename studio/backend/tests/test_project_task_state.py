# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.agent_workspace import task_state as state


@pytest.fixture
def task_db(tmp_path, monkeypatch):
    path = tmp_path / "tasks.db"

    def connect():
        conn = sqlite3.connect(path, timeout = 10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    with connect() as conn:
        conn.execute("CREATE TABLE chat_projects(id TEXT PRIMARY KEY, archived INTEGER DEFAULT 0)")
        conn.executemany("INSERT INTO chat_projects(id) VALUES(?)", [("one",), ("two",)])
    clock = [100000]
    monkeypatch.setattr(state, "get_connection", connect)
    monkeypatch.setattr(state, "_now", lambda: clock[0])
    return connect, clock


def root(**kwargs):
    return state.create_task("one", "Inspect the project", {"model": "local"}, **kwargs)


def running_root(**kwargs):
    row = root(**kwargs)
    return state.claim_task("one", row["id"], "root-owner")


def test_task_round_trip_and_private_ownership(task_db):
    row = running_root()
    assert "owner" not in row and "lease_expires_at" not in row
    finished = state.finish_task(row["id"], "root-owner", "completed", result = {"output": "Done"})
    assert finished["result"] == {"output": "Done"}
    assert state.get_task("one", row["id"]) == finished
    events = state.task_events("one", row["id"])
    assert [e["kind"] for e in events] == ["queued", "running", "completed"]
    assert [e["revision"] for e in events] == [1, 2, 3]
    assert state.task_events("one", row["id"], after = events[1]["sequence"]) == events[2:]


def test_only_one_concurrent_claim_wins(task_db):
    row = root()
    barrier = threading.Barrier(8)

    def claim(index):
        barrier.wait(timeout = 5)
        try:
            state.claim_task("one", row["id"], str(index))
            return True
        except state.TaskStateError:
            return False

    with ThreadPoolExecutor(max_workers = 8) as pool:
        assert sum(pool.map(claim, range(8))) == 1


def test_dispatch_reservation_is_exclusive_and_keeps_queue_alive(task_db):
    _, clock = task_db
    row = root()
    state.reserve_dispatch("one", row["id"], "dispatch")
    with pytest.raises(state.TaskStateError):
        state.reserve_dispatch("one", row["id"], "second")
    with pytest.raises(state.TaskStateError):
        state.claim_task("one", row["id"], "second")
    clock[0] += state.LEASE_MS - 1
    assert state.heartbeat(row["id"], "dispatch")
    clock[0] += state.LEASE_MS - 1
    assert state.claim_task("one", row["id"], "dispatch")["status"] == "running"


@pytest.mark.parametrize("claimed", [False, True])
def test_expiry_does_not_replay_and_failed_renewal_keeps_recovery(task_db, claimed):
    connect, clock = task_db
    row = running_root() if claimed else root()
    clock[0] += state.LEASE_MS
    with pytest.raises(state.TaskStateError):
        state.heartbeat(row["id"], "root-owner")
    # Inspect directly: a later successful accessor must not be necessary to
    # commit recovery performed during the refused heartbeat.
    with connect() as conn:
        stored = conn.execute(
            "SELECT * FROM studio_project_tasks WHERE id=?", (row["id"],)
        ).fetchone()
    assert stored["status"] == "interrupted" and stored["owner"] is None
    retry = state.retry_task("one", row["id"])
    assert retry["status"] == "queued" and retry["retryOf"] == row["id"]
    assert retry["attempt"] == 2 and retry["rootId"] == retry["id"]


def test_parent_and_child_expiry_are_idempotent(task_db):
    _, clock = task_db
    parent = running_root(child_limit = 2, child_budget = 20)
    child = state.create_child(
        parent["id"], "root-owner", "Review", {}, role = "reviewer", max_output_tokens = 10
    )
    state.claim_task("one", child["id"], "child-owner")
    queued = state.create_child(
        parent["id"], "root-owner", "Implement", {}, role = "implementer", max_output_tokens = 10
    )
    clock[0] += state.LEASE_MS
    state.reconcile_expired_tasks()
    before = {r["id"]: r for r in state.list_tasks("one")}
    assert before[parent["id"]]["status"] == "interrupted"
    assert before[child["id"]]["status"] == "interrupted"
    assert before[queued["id"]]["status"] in {"cancelled", "interrupted"}
    state.reconcile_expired_tasks()
    assert {r["id"]: r for r in state.list_tasks("one")} == before
    for task in before.values():
        revisions = [e["revision"] for e in state.task_events("one", task["id"])]
        assert len(revisions) == len(set(revisions))


def test_cancel_cascades_and_parent_waits_for_children(task_db):
    parent = running_root(child_limit = 1, child_budget = 10)
    child = state.create_child(
        parent["id"], "root-owner", "Review", {}, role = "reviewer", max_output_tokens = 10
    )
    state.claim_task("one", child["id"], "child-owner")
    cancelled = state.cancel_task("one", parent["id"])
    assert cancelled["status"] == "cancelling"
    assert not state.heartbeat(child["id"], "child-owner")
    with pytest.raises(state.TaskStateError, match = "settle"):
        state.finish_task(parent["id"], "root-owner", "completed")
    assert state.finish_task(child["id"], "child-owner", "completed")["status"] == "cancelled"
    assert state.finish_task(parent["id"], "root-owner", "completed")["status"] == "cancelled"
    assert state.cancel_task("one", parent["id"])["revision"] == cancelled["revision"] + 1


def test_parent_retry_waits_for_live_children_after_parent_expires(task_db):
    _, clock = task_db
    parent = running_root(child_limit = 1, child_budget = 10)
    child = state.create_child(
        parent["id"], "root-owner", "Review", {}, role = "reviewer", max_output_tokens = 10
    )
    state.claim_task("one", child["id"], "child-owner")
    clock[0] += state.LEASE_MS - 1
    state.heartbeat(child["id"], "child-owner")
    clock[0] += 1
    with pytest.raises(state.TaskStateError, match = "settle"):
        state.retry_task("one", parent["id"])
    state.finish_task(child["id"], "child-owner", "cancelled")
    assert state.retry_task("one", parent["id"])["attempt"] == 2


def test_child_reservations_are_atomic_and_retries_consume_budget(task_db):
    parent = running_root(child_limit = 1, child_budget = 20)
    barrier = threading.Barrier(2)

    def create(_):
        barrier.wait(timeout = 5)
        try:
            return state.create_child(
                parent["id"], "root-owner", "Review", {}, role = "reviewer", max_output_tokens = 10
            )
        except state.TaskStateError:
            return None

    with ThreadPoolExecutor(max_workers = 2) as pool:
        children = [r for r in pool.map(create, range(2)) if r]
    assert len(children) == 1
    child = children[0]
    state.cancel_task("one", child["id"])
    retry = state.retry_task("one", child["id"], parent_owner = "root-owner")
    state.cancel_task("one", retry["id"])
    with pytest.raises(state.TaskStateError, match = "budget"):
        state.retry_task("one", retry["id"], parent_owner = "root-owner")
    updated = state.get_task("one", parent["id"])
    assert updated["childCount"] == 1 and updated["childAllocated"] == 20


def test_children_cannot_delegate(task_db):
    parent = running_root(child_limit = 1, child_budget = 10)
    child = state.create_child(
        parent["id"], "root-owner", "Review", {}, role = "reviewer", max_output_tokens = 10
    )
    state.claim_task("one", child["id"], "child-owner")
    with pytest.raises(state.TaskStateError, match = "cannot delegate"):
        state.create_child(child["id"], "child-owner", "More", {}, role = "reviewer")


def test_failed_enqueue_rolls_back_budget_and_events(task_db, monkeypatch):
    parent = running_root(child_limit = 1, child_budget = 10)
    monkeypatch.setattr(state, "MAX_ACTIVE_TASKS", 1)
    with pytest.raises(state.TaskStateError, match = "queue is full"):
        state.create_child(
            parent["id"], "root-owner", "Review", {}, role = "reviewer", max_output_tokens = 10
        )
    assert state.get_task("one", parent["id"]) == parent
    assert [e["kind"] for e in state.task_events("one", parent["id"])] == ["queued", "running"]


def test_retries_are_explicit_single_successors_with_attempt_cap(task_db):
    row = root()
    for attempt in (2, 3):
        state.cancel_task("one", row["id"])
        retry = state.retry_task("one", row["id"])
        with pytest.raises(state.TaskStateError):
            state.retry_task("one", row["id"])
        assert retry["attempt"] == attempt
        row = retry
    state.cancel_task("one", row["id"])
    with pytest.raises(state.TaskStateError, match = "attempt limit"):
        state.retry_task("one", row["id"])


def test_project_scope_and_delete_cascade(task_db):
    connect, _ = task_db
    row = root()
    for operation in (state.get_task, state.cancel_task, state.retry_task, state.task_events):
        with pytest.raises(state.TaskStateError):
            operation("two", row["id"])
    with connect() as conn:
        conn.execute("DELETE FROM chat_projects WHERE id='one'")
        assert conn.execute("SELECT COUNT(*) FROM studio_project_task_events").fetchone()[0] == 0
    assert state.list_tasks("one") == []


def test_retirement_fences_creation_and_only_owner_can_release(task_db):
    _, clock = task_db
    row = running_root()
    state.begin_project_retirement("one", "retire-owner")
    assert state.get_task("one", row["id"])["cancelRequested"]
    state.finish_project_retirement("one", "wrong-owner")
    with pytest.raises(state.TaskStateError, match = "retirement"):
        root()
    clock[0] += state.LEASE_MS - 1
    state.renew_project_retirement("one", "retire-owner")
    clock[0] += state.LEASE_MS - 1
    with pytest.raises(state.TaskStateError, match = "retirement"):
        root()
    state.finish_project_retirement("one", "retire-owner")
    assert root()["status"] == "queued"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_output_tokens": True},
        {"max_output_tokens": 32769},
        {"child_limit": 1},
        {"child_budget": 1},
        {"child_limit": 9, "child_budget": 1},
    ],
)
def test_invalid_limits_are_refused(task_db, kwargs):
    with pytest.raises(state.TaskStateError):
        root(**kwargs)


@pytest.mark.parametrize(
    "instruction",
    ["", "\x00", "\ud800", "é" * 32769],
    ids = ["empty", "nul", "surrogate", "utf8-byte-bound"],
)
def test_instruction_bounds(task_db, instruction):
    with pytest.raises(state.TaskStateError):
        state.create_task("one", instruction, {})


def test_snapshot_is_copied_and_bounded(task_db):
    original = {"model": "first"}
    row = state.create_task("one", "Task", original)
    original["model"] = "second"
    row["snapshot"]["model"] = "third"
    assert state.get_task("one", row["id"])["snapshot"] == {"model": "first"}
    with pytest.raises(state.TaskStateError):
        state.create_task("one", "Task", {"bad": float("nan")})
    with pytest.raises(state.TaskStateError):
        state.create_task("one", "Task", {"huge": "é" * 65536})


@pytest.fixture
def runners(task_db):
    from core.agent_workspace.task_runner import ProjectTaskRunner

    created = []

    def create(executor, **kwargs):
        runner = ProjectTaskRunner(executor, heartbeat_seconds = 0.01, drain_seconds = 0.2, **kwargs)
        created.append(runner)
        return runner

    yield create
    for runner in created:
        assert runner.shutdown(timeout = 3), "Task test leaked an executor"


def test_all_parent_workers_can_wait_for_children_without_starvation(runners):
    parents = threading.Barrier(2)
    children = threading.Barrier(2)

    def execute(context):
        if context.task["parentId"]:
            children.wait(timeout = 3)
            return {"review": context.task["instruction"]}
        parents.wait(timeout = 3)
        child = context.delegate("Inspect files", role = "reviewer", max_output_tokens = 10)
        result = context.wait_child(child["id"], timeout = 3)
        assert result is not None and result["status"] == "completed"
        return result["result"]

    runner = runners(execute, root_workers = 2, child_workers = 2)
    tasks = [runner.submit("one", "Parent", {}, child_limit = 1, child_budget = 10) for _ in range(2)]
    for task in tasks:
        result = runner.wait("one", task["id"], timeout = 5)
        assert result["status"] == "completed"
        assert result["result"] == {"review": "Inspect files"}


def test_cancelled_queued_task_never_executes(runners):
    started, release = threading.Event(), threading.Event()
    calls = []

    def execute(context):
        calls.append(context.task["instruction"])
        started.set()
        assert release.wait(3)
        return {}

    runner = runners(execute, root_workers = 1)
    first = runner.submit("one", "First", {})
    assert started.wait(3)
    second = runner.submit("one", "Second", {})
    runner.cancel("one", second["id"])
    release.set()
    assert runner.wait("one", first["id"], timeout = 3)["status"] == "completed"
    assert runner.wait("one", second["id"], timeout = 3)["status"] == "cancelled"
    assert calls == ["First"]


def test_parent_failure_cancels_and_drains_its_child(runners):
    child_started = threading.Event()

    def execute(context):
        if context.task["parentId"]:
            child_started.set()
            assert context.cancel_event.wait(3)
            context.check()
        context.delegate("Child", role = "implementer", max_output_tokens = 10)
        assert child_started.wait(3)
        raise RuntimeError("secret-value-must-not-be-persisted")

    runner = runners(execute)
    parent = runner.submit("one", "Parent", {}, child_limit = 1, child_budget = 10)
    result = runner.wait("one", parent["id"], timeout = 3)
    assert result["status"] == "failed"
    assert "secret-value" not in str(result)
    assert state.list_children("one", parent["id"])[0]["status"] == "cancelled"


def test_worker_child_retry_obeys_original_budget(runners):
    def execute(context):
        task = context.task
        if task["parentId"]:
            if task["attempt"] == 1:
                raise RuntimeError("Try again")
            return {"attempt": task["attempt"]}
        child = context.delegate("Child", role = "reviewer", max_output_tokens = 10)
        assert context.wait_child(child["id"], timeout = 3)["status"] == "failed"
        retried = context.retry_child(child["id"])
        return context.wait_child(retried["id"], timeout = 3)["result"]

    runner = runners(execute)
    task = runner.submit("one", "Parent", {}, child_limit = 1, child_budget = 20)
    result = runner.wait("one", task["id"], timeout = 3)
    assert result["status"] == "completed"
    assert result["result"] == {"attempt": 2}
    assert result["childCount"] == 1 and result["childAllocated"] == 20


def test_retry_does_not_run_automatically(runners):
    calls = []

    def execute(context):
        calls.append(context.task["attempt"])
        if context.task["attempt"] == 1:
            raise RuntimeError("Failed")
        return {}

    runner = runners(execute)
    task = runner.submit("one", "Task", {})
    assert runner.wait("one", task["id"], timeout = 3)["status"] == "failed"
    assert calls == [1]
    retried = runner.retry("one", task["id"])
    assert runner.wait("one", retried["id"], timeout = 3)["status"] == "completed"
    assert calls == [1, 2]


def test_starting_runner_never_replays_an_expired_task(task_db, runners):
    _, clock = task_db
    task = running_root()
    clock[0] += state.LEASE_MS
    calls = []
    runner = runners(lambda context: calls.append(context.task) or {})
    assert runner.wait("one", task["id"], timeout = 3)["status"] == "interrupted"
    assert calls == []


def test_expired_worker_cannot_publish_result_or_overlap_retry(task_db, runners):
    _, clock = task_db
    started, release = threading.Event(), threading.Event()

    def execute(context):
        started.set()
        assert release.wait(3)
        return {"must_not_publish": True}

    runner = runners(execute)
    task = runner.submit("one", "Task", {})
    assert started.wait(3)
    clock[0] += state.LEASE_MS
    assert state.get_task("one", task["id"])["status"] == "interrupted"
    with pytest.raises(state.TaskStateError, match = "have not stopped"):
        runner.retry("one", task["id"])
    release.set()
    result = runner.wait("one", task["id"], timeout = 3)
    assert result["status"] == "interrupted" and result["result"] is None


def test_shutdown_reports_a_worker_that_has_not_stopped(runners):
    started, release = threading.Event(), threading.Event()
    cancelled_seen = threading.Event()

    def execute(context):
        started.set()
        assert context.cancel_event.wait(3)
        cancelled_seen.set()
        assert release.wait(3)
        return {}

    runner = runners(execute)
    task = runner.submit("one", "Task", {})
    assert started.wait(3)
    try:
        assert runner.shutdown(timeout = 0.05) is False
        with pytest.raises(state.TaskStateError, match = "shutting down"):
            runner.submit("one", "More", {})
        assert cancelled_seen.wait(3)
        assert state.get_task("one", task["id"])["status"] in {"running", "cancelling"}
    finally:
        release.set()
    assert runner.shutdown(timeout = 3)
    assert state.get_task("one", task["id"])["status"] == "cancelled"


def test_deadline_cancels_running_executor(runners):
    def execute(context):
        assert context.cancel_event.wait(3)
        context.check()

    runner = runners(execute)
    task = runner.submit("one", "Task", {}, timeout = 0.05)
    assert runner.wait("one", task["id"], timeout = 3)["status"] == "cancelled"


def test_project_retirement_signals_existing_worker(runners):
    started = threading.Event()

    def execute(context):
        started.set()
        assert context.cancel_event.wait(3)
        context.check()

    runner = runners(execute)
    task = runner.submit("one", "Task", {})
    assert started.wait(3)
    state.begin_project_retirement("one", "retire")
    assert runner.wait("one", task["id"], timeout = 3)["status"] == "cancelled"
    assert not state.project_has_active_tasks("one")
    state.finish_project_retirement("one", "retire")


@pytest.mark.parametrize("result", [None, [], {"large": "x" * (1024 * 1024)}])
def test_invalid_executor_result_is_terminal_failure(runners, result):
    runner = runners(lambda _: result)
    task = runner.submit("one", "Task", {})
    outcome = runner.wait("one", task["id"], timeout = 3)
    assert outcome["status"] == "failed" and outcome["result"] is None


def test_context_changes_do_not_change_captured_snapshot(runners):
    def execute(context):
        context.task["snapshot"]["model"] = "different"
        return {"model": context.check()["snapshot"]["model"]}

    runner = runners(execute)
    task = runner.submit("one", "Task", {"model": "original"})
    assert runner.wait("one", task["id"], timeout = 3)["result"] == {"model": "original"}


def test_late_submission_during_shutdown_is_cancelled(task_db, runners, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    create = state.create_task

    def blocked_create(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return create(*args, **kwargs)

    monkeypatch.setattr(state, "create_task", blocked_create)
    runner = runners(lambda _: pytest.fail("Closed runner executed a task"))
    with ThreadPoolExecutor(max_workers = 1) as pool:
        submission = pool.submit(runner.submit, "one", "Late", {})
        assert entered.wait(3)
        try:
            assert runner.shutdown(timeout = 0.3)
        finally:
            release.set()
        with pytest.raises(state.TaskStateError, match = "shutting down"):
            submission.result(timeout = 3)
    assert state.list_tasks("one")[0]["status"] == "cancelled"


def test_parent_drain_timeout_does_not_claim_project_idle(runners):
    child_started, release = threading.Event(), threading.Event()

    def execute(context):
        if context.task["parentId"]:
            child_started.set()
            assert release.wait(3)
            return {}
        context.delegate("Child", role = "reviewer", max_output_tokens = 10)
        assert child_started.wait(3)
        return {}

    runner = runners(execute)
    parent = runner.submit("one", "Parent", {}, child_limit = 1, child_budget = 10)
    try:
        result = runner.wait("one", parent["id"], timeout = 2)
        assert result["status"] == "interrupted"
        assert not runner.project_is_idle("one")
        with pytest.raises(state.TaskStateError, match = "have not stopped"):
            runner.retry("one", parent["id"])
    finally:
        release.set()
    child = state.list_children("one", parent["id"])[0]
    assert runner.wait("one", child["id"], timeout = 3)["status"] == "cancelled"
    assert runner.project_is_idle("one")


def test_archived_project_cancels_task_and_fences_submission(task_db, runners):
    connect, _ = task_db
    started = threading.Event()

    def execute(context):
        started.set()
        assert context.cancel_event.wait(3)
        context.check()

    runner = runners(execute)
    task = runner.submit("one", "Task", {})
    assert started.wait(3)
    with connect() as conn:
        conn.execute("UPDATE chat_projects SET archived=1 WHERE id='one'")
    assert runner.wait("one", task["id"], timeout = 3)["status"] == "cancelled"
    with pytest.raises(state.TaskStateError, match = "Project not found"):
        runner.submit("one", "More", {})


def test_shutdown_after_project_deletion_still_stops_executor(task_db, runners):
    connect, _ = task_db
    started = threading.Event()

    def execute(context):
        started.set()
        assert context.cancel_event.wait(3)
        return {}

    runner = runners(execute)
    runner.submit("one", "Task", {})
    assert started.wait(3)
    with connect() as conn:
        conn.execute("DELETE FROM chat_projects WHERE id='one'")
    assert runner.shutdown(timeout = 3)


def test_project_deletion_cascades_child_retries_and_events(task_db):
    connect, _ = task_db
    parent = running_root(child_limit = 1, child_budget = 20)
    child = state.create_child(
        parent["id"], "root-owner", "Child", {}, role = "reviewer", max_output_tokens = 10
    )
    state.cancel_task("one", child["id"])
    state.retry_task("one", child["id"], parent_owner = "root-owner")
    with connect() as conn:
        conn.execute("DELETE FROM chat_projects WHERE id='one'")
        assert conn.execute("SELECT COUNT(*) FROM studio_project_tasks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM studio_project_task_events").fetchone()[0] == 0


def test_task_tables_integrate_with_real_studio_storage():
    from storage import studio_db

    studio_db.upsert_chat_project(
        {
            "id": "integration",
            "name": "Task integration",
            "createdAt": 1000,
            "updatedAt": 1000,
        }
    )
    task = state.create_task("integration", "Inspect", {"model": "local"})
    state.claim_task("integration", task["id"], "integration-owner")
    state.finish_task(task["id"], "integration-owner", "completed", result = {"output": "Done"})
    assert state.list_tasks("integration")[0]["result"] == {"output": "Done"}
    studio_db.delete_chat_project("integration", delete_files = False)
    assert state.list_tasks("integration") == []


def test_retirement_can_fence_an_already_archived_project(task_db):
    connect, _ = task_db
    with connect() as conn:
        conn.execute("UPDATE chat_projects SET archived=1 WHERE id='one'")
    state.begin_project_retirement("one", "retire")
    state.renew_project_retirement("one", "retire")
    with pytest.raises(state.TaskStateError):
        root()
    state.finish_project_retirement("one", "retire")


def test_executor_system_exit_does_not_remove_worker_capacity(runners):
    def execute(context):
        if context.task["instruction"] == "Exit":
            raise SystemExit(1)
        return {"output": "Still working"}

    runner = runners(execute, root_workers = 1)
    first = runner.submit("one", "Exit", {})
    assert runner.wait("one", first["id"], timeout = 3)["status"] == "failed"
    second = runner.submit("one", "Continue", {})
    assert runner.wait("one", second["id"], timeout = 3)["result"] == {"output": "Still working"}

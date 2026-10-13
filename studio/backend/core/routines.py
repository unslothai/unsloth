# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small background dispatcher for durable, account-scoped Studio routines."""

from __future__ import annotations

import threading
from contextlib import closing
from typing import Any

from starlette.requests import Request

from core.training.account_jobs import job_accounts
from loggers import get_logger
from models.data_recipe import RecipePayload
from storage import routines_db as db
from utils.account_context import account_thread, run_as

logger = get_logger(__name__)


class RoutineSupervisor:
    def __init__(self, app: Any):
        self.app = app
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_accounts: set[str] = set()
        self._seen_accounts: set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target = self._loop, name = "studio-routines", daemon = True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout = 2)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                accounts = job_accounts()
                for account in accounts:
                    if account.account_id not in self._seen_accounts:
                        run_as(account, db.fail_interrupted_runs)
                        self._seen_accounts.add(account.account_id)
                    with self._lock:
                        busy = account.account_id in self._active_accounts
                    if busy:
                        continue
                    run_as(account, db.claim_due)
                    queued = run_as(account, db.claim_queued_run)
                    if queued is None:
                        continue
                    with self._lock:
                        self._active_accounts.add(account.account_id)
                    account_thread(
                        account = account,
                        target = self._execute,
                        args = (account.account_id, queued),
                        name = f"studio-routine-{account.account_id}",
                        daemon = True,
                    ).start()
            except Exception:
                logger.exception("Routine scheduling failed")
            self.stop_event.wait(2)

    def _request(self) -> Request:
        port = getattr(self.app.state, "server_port", None) or 8888
        return Request({
            "type": "http", "app": self.app, "method": "POST",
            "path": "/api/data-recipe/jobs", "raw_path": b"/api/data-recipe/jobs",
            "headers": [], "query_string": b"", "scheme": "http",
            "server": ("127.0.0.1", port),
        })

    def _wait_job(self, kind: str, job_id: str) -> dict:
        while not self.stop_event.wait(2):
            if kind == "recipe":
                from core.data_recipe.jobs.manager import get_job_manager
                job = get_job_manager().get_status(job_id)
                if job and job["status"] in {"completed", "failed", "cancelled"}:
                    return job
            else:
                from core.rag.folder_sync import get_job
                from core.rag.account_db import get_connection as rag_connection
                job = get_job(job_id)
                if job and job["status"] in {"completed", "failed"}:
                    # A request made during an active folder scan is folded into a successor.
                    # Wait for that newer job before advancing the routine.
                    if job.get("successor_kind"):
                        continue
                    with closing(rag_connection()) as conn:
                        newer = conn.execute(
                            "SELECT id FROM linked_folder_sync_jobs WHERE folder_id=? "
                            "ORDER BY created_at DESC LIMIT 1", (job["folder_id"],),
                        ).fetchone()
                    if newer is not None and newer["id"] != job_id:
                        job_id = newer["id"]
                        continue
                    return job
            if job is None:
                raise RuntimeError("The underlying job is no longer available")
        raise RuntimeError("Studio stopped before the job finished")

    def _start_step(self, step: dict) -> str:
        if step["kind"] == "recipe":
            from routes.data_recipe.jobs import create_job
            from utils.account_context import current_account
            payload = RecipePayload.model_validate(step["recipe"])
            return create_job(
                payload, self._request(),
                credential = (current_account().username, None), via_api_key = False,
            )["job_id"]
        from core.rag.folder_sync import request_sync
        return request_sync(step["folder_id"])

    def _execute(self, account_id: str, run: dict) -> None:
        statuses = [
            {"kind": step["kind"], "label": step["label"], "status": "pending"}
            for step in run["steps"]
        ]
        try:
            db.update_run(run["id"], status = "running", step_statuses = statuses)
            for index, step in enumerate(run["steps"]):
                if self.stop_event.is_set():
                    return
                statuses[index]["status"] = "running"
                db.update_run(run["id"], status = "running", current_step = index,
                              step_statuses = statuses)
                job_id = self._start_step(step)
                statuses[index]["job_id"] = job_id
                db.update_run(run["id"], status = "running", step_statuses = statuses)
                result = self._wait_job(step["kind"], job_id)
                if result["status"] != "completed":
                    raise RuntimeError(str(result.get("error") or f"{step['label']} failed")[:1000])
                statuses[index]["status"] = "completed"
                if step["kind"] == "recipe":
                    statuses[index]["artifact_path"] = result.get("artifact_path")
                else:
                    statuses[index]["summary"] = {
                        key: result[key] for key in ("added", "changed", "deleted", "failed")
                        if key in result
                    }
                db.update_run(run["id"], status = "running", step_statuses = statuses)
            db.update_run(run["id"], status = "completed", current_step = len(statuses),
                          step_statuses = statuses)
        except Exception as exc:
            logger.exception("Routine run %s failed", run["id"])
            active = next((item for item in statuses if item["status"] == "running"), None)
            if active:
                active["status"] = "failed"
                active["error"] = str(exc)[:1000]
            db.update_run(run["id"], status = "failed", step_statuses = statuses,
                          error = str(exc)[:1000])
        finally:
            with self._lock:
                self._active_accounts.discard(account_id)

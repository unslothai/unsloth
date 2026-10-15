# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from .factory_base import Factory, seeder

TITLE = "scheduled-task-matrix-title"
EDITED = "scheduled-task-matrix-edited"


@seeder("scheduled-task")
def seed_scheduled_task(account) -> dict[str, str]:
    from storage import scheduled_tasks_db
    from utils.account_context import run_as

    now = scheduled_tasks_db.now_ms()
    task = run_as(
        account,
        scheduled_tasks_db.create_task,
        {
            "id": "scheduled-task-matrix",
            "ownerSubject": account.username,
            "title": TITLE,
            "prompt": "scheduled task matrix prompt",
            "schedule": {"kind": "daily", "time": "09:00"},
            "timezone": "UTC",
            "enabled": False,
            "nextRunAt": None,
            "createdAt": now,
            "updatedAt": now,
        },
    )
    return {"task_id": task["id"]}


FACTORIES = {
    "routes.scheduled_tasks:GET:/{task_id}": Factory(
        "scheduled-task", fragment = TITLE, absent = TITLE
    ),
    "routes.scheduled_tasks:PATCH:/{task_id}": Factory(
        "scheduled-task", {"title": EDITED}, fragment = EDITED, absent = TITLE
    ),
    "routes.scheduled_tasks:DELETE:/{task_id}": Factory(
        "scheduled-task", success = 204, absent = TITLE
    ),
    "routes.scheduled_tasks:POST:/{task_id}/run": Factory(
        "scheduled-task", success = 202, fragment = '"status":"queued"', absent = TITLE
    ),
}

SKIPPED: dict[str, str] = {}

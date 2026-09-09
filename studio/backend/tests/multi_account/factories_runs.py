# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable Deep Research run routes and the public per-checkpoint preview routes."""

import json

from .factory_base import Factory, seeder

MARKER = "runs-matrix-sentinel"
THREAD_ID = "runs-thread"
USER_MESSAGE_ID = "runs-user-message"
RUN_ID = "runs-research-run"
PLAN = {"title": MARKER, "steps": [{"title": MARKER, "query": MARKER}]}
UPDATE_PLAN_BODY = {"plan": PLAN, "expectedRevision": 0}
# Filled by the seeder from the product's own canonical hash, so approve matches the stored plan.
APPROVE_BODY = {"planRevision": 1, "planHash": ""}

PREVIEW_RUN = "runs-preview-run"
PREVIEW_CHECKPOINT = "checkpoint-1"
UNSUPPORTED_PART = "runs-unsupported-part"
PREVIEW_CHAT_BODY = {
    "model": PREVIEW_RUN,
    "messages": [{"role": "user", "content": [{"type": UNSUPPORTED_PART, "value": MARKER}]}],
}
PREVIEW_RUN_KEY: dict = {}
PREVIEW_CHECKPOINT_KEY: dict = {}


def _link_holder(seeded, actor: str):
    from auth import storage
    from utils.account_context import OWNER

    if actor == "unauthenticated":
        return None
    if actor == "owner":
        return OWNER
    if actor == "wrong":
        return storage.get_account("bob")
    return seeded


def _create_research_run(account) -> None:
    from storage import research_runs_db as db
    from storage.studio_db import upsert_chat_message, upsert_chat_thread
    from utils.account_context import run_as

    run_as(
        account,
        upsert_chat_thread,
        {
            "id": THREAD_ID,
            "title": MARKER,
            "modelType": "base",
            "modelId": "local/model",
            "createdAt": 1000,
            "updatedAt": 1000,
        },
    )
    run_as(
        account,
        upsert_chat_message,
        {
            "id": USER_MESSAGE_ID,
            "threadId": THREAD_ID,
            "role": "user",
            "content": [{"type": "text", "text": MARKER}],
            "createdAt": 1000,
        },
    )
    run_as(
        account,
        db.create_run,
        run_id = RUN_ID,
        owner_subject = account.username,
        thread_id = THREAD_ID,
        user_message_id = USER_MESSAGE_ID,
        assistant_message_id = None,
        config = {"model": "local/model", "question": MARKER},
    )


@seeder("runs-research")
def seed_research_run(account) -> dict[str, str]:
    _create_research_run(account)
    return {"run_id": RUN_ID}


@seeder("runs-research-planned")
def seed_planned_research_run(account) -> dict[str, str]:
    from storage import research_runs_db as db
    from utils.account_context import run_as

    _create_research_run(account)
    planned = run_as(account, db.set_plan, RUN_ID, PLAN, 0)
    APPROVE_BODY["planRevision"] = planned["planRevision"]
    APPROVE_BODY["planHash"] = planned["planHash"]
    return {"run_id": RUN_ID}


@seeder("runs-research-stopped")
def seed_stopped_research_run(account) -> dict[str, str]:
    """A terminal run whose events are all committed, so its event stream ends at once."""
    from storage import research_runs_db as db
    from utils.account_context import run_as

    seed_planned_research_run(account)
    run_as(account, db.request_cancel, RUN_ID)
    return {"run_id": RUN_ID}


@seeder("runs-preview")
def seed_preview_run(account, actor: str = "right") -> dict[str, str]:
    from utils.paths import outputs_root
    from utils.account_context import run_as
    from utils.preview_token import sign_preview_ref

    run = run_as(account, outputs_root) / PREVIEW_RUN
    (run / PREVIEW_CHECKPOINT).mkdir(parents = True, exist_ok = True)
    run.joinpath("adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": MARKER}), encoding = "utf-8"
    )
    run.joinpath(PREVIEW_CHECKPOINT, "adapter_config.json").write_text("{}", encoding = "utf-8")
    holder = _link_holder(account, actor)
    PREVIEW_RUN_KEY.clear()
    PREVIEW_CHECKPOINT_KEY.clear()
    if holder is not None:
        PREVIEW_RUN_KEY["k"] = sign_preview_ref(PREVIEW_RUN, holder)
        PREVIEW_CHECKPOINT_KEY["k"] = sign_preview_ref(
            f"{PREVIEW_RUN}/{PREVIEW_CHECKPOINT}", holder
        )
    return {"run": PREVIEW_RUN, "checkpoint": PREVIEW_CHECKPOINT}


_LINK_REASON = (
    "public share link: the signed capability is the credential, so each actor presents its own "
    "link and only the minting account's opens the run"
)
_CHAT_REASON = (
    "covered only to the pre-load parameter guard: an unsupported content part is refused before "
    "the preview lock, so no checkpoint is ever loaded"
)


def _preview(fragment: str, query: dict, **overrides) -> Factory:
    return Factory(
        "runs-preview",
        fragment = fragment,
        query = query,
        unauthenticated = (404,),
        deactivated = (404,),
        **{"reason": _LINK_REASON, **overrides},
    )


FACTORIES = {
    "routes.research_runs:GET:/{run_id}": Factory("runs-research", fragment = MARKER),
    "routes.research_runs:PUT:/{run_id}/plan": Factory(
        "runs-research", UPDATE_PLAN_BODY, fragment = '"status":"awaiting_approval"'
    ),
    "routes.research_runs:POST:/{run_id}/approve": Factory(
        "runs-research-planned", APPROVE_BODY, fragment = '"status":"queued"'
    ),
    "routes.research_runs:POST:/{run_id}/cancel": Factory(
        "runs-research", fragment = '"status":"cancelling"'
    ),
    "routes.research_runs:POST:/{run_id}/retry": Factory(
        "runs-research-stopped", fragment = '"status":"awaiting_approval"'
    ),
    "routes.research_runs:GET:/{run_id}/events": Factory(
        "runs-research-stopped", fragment = "event: run.cancelled"
    ),
    "routes.research_runs:POST:/{run_id}/events": Factory(
        "runs-research-stopped", fragment = "event: run.cancelled"
    ),
    "routes.preview:GET:/{run}": _preview(PREVIEW_RUN, PREVIEW_RUN_KEY),
    "routes.preview:GET:/{run}/{checkpoint}": _preview(
        f"{PREVIEW_RUN}/{PREVIEW_CHECKPOINT}", PREVIEW_CHECKPOINT_KEY
    ),
    "routes.preview:GET:/{run}/v1/models": _preview(f'"id":"{PREVIEW_RUN}"', PREVIEW_RUN_KEY),
    "routes.preview:GET:/{run}/{checkpoint}/v1/models": _preview(
        f'"id":"{PREVIEW_RUN}/{PREVIEW_CHECKPOINT}"', PREVIEW_CHECKPOINT_KEY
    ),
    "routes.preview:POST:/{run}/v1/chat/completions": _preview(
        UNSUPPORTED_PART,
        PREVIEW_RUN_KEY,
        body = PREVIEW_CHAT_BODY,
        success = 400,
        reason = f"{_LINK_REASON}; {_CHAT_REASON}",
    ),
    "routes.preview:POST:/{run}/{checkpoint}/v1/chat/completions": _preview(
        UNSUPPORTED_PART,
        PREVIEW_CHECKPOINT_KEY,
        body = PREVIEW_CHAT_BODY,
        success = 400,
        reason = f"{_LINK_REASON}; {_CHAT_REASON}",
    ),
}

SKIPPED: dict = {}

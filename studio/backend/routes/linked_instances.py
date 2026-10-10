# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CRUD and health checks for linked Unsloth Studio instances. Owner-only, UI session only."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import linked_instances
from hub.services.models import account_access
from models.linked_instances import (
    LinkedInstance,
    LinkedInstanceCreate,
    LinkedInstanceInfo,
    LinkedInstanceStatus,
    LinkedInstanceUpdate,
)
from routes.provider_credentials import require_ui_session
from storage import linked_instances_db

router = APIRouter()


def _require_owner_ui(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> None:
    require_ui_session(via_api_key)
    account_access.require_installation_owner()


async def _normalized_url(base_url: str) -> str:
    try:
        return await asyncio.to_thread(linked_instances.normalize_base_url, base_url)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


async def _status(instance: dict) -> LinkedInstanceStatus:
    result = await linked_instances.probe(instance)
    prefix = f"{linked_instances.MODEL_PREFIX}{instance['name']}/"
    return LinkedInstanceStatus(
        id = instance["id"],
        online = result["online"],
        error = result["error"],
        models = [prefix + m["id"] for m in result["models"]],
        loaded = [prefix + m["id"] for m in result["models"] if m.get("loaded")],
        latency_ms = result["latency_ms"],
    )


@router.get("", response_model = list[LinkedInstance], dependencies = [Depends(_require_owner_ui)])
async def list_linked_instances():
    return await asyncio.to_thread(linked_instances_db.list_instances)


@router.get(
    "/status", response_model = list[LinkedInstanceStatus], dependencies = [Depends(_require_owner_ui)]
)
async def linked_instances_status():
    instances = await asyncio.to_thread(linked_instances_db.list_instances)
    return await asyncio.gather(*(_status(i) for i in instances))


@router.get(
    "/info", response_model = list[LinkedInstanceInfo], dependencies = [Depends(_require_owner_ui)]
)
async def linked_instances_info():
    instances = await asyncio.to_thread(linked_instances_db.list_instances)
    results = await asyncio.gather(*(linked_instances.fetch_info(i) for i in instances))
    return [LinkedInstanceInfo(id = i["id"], **r) for i, r in zip(instances, results)]


@router.post("", response_model = LinkedInstance, dependencies = [Depends(_require_owner_ui)])
async def create_linked_instance(payload: LinkedInstanceCreate):
    base_url = await _normalized_url(payload.base_url)
    try:
        return await asyncio.to_thread(
            linked_instances_db.create_instance, payload.name, base_url, payload.api_key.strip()
        )
    except linked_instances_db.DuplicateName as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


@router.patch(
    "/{instance_id}", response_model = LinkedInstance, dependencies = [Depends(_require_owner_ui)]
)
async def update_linked_instance(instance_id: str, payload: LinkedInstanceUpdate):
    base_url = await _normalized_url(payload.base_url) if payload.base_url is not None else None
    try:
        updated = await asyncio.to_thread(
            linked_instances_db.update_instance,
            instance_id,
            name = payload.name,
            base_url = base_url,
            api_key = payload.api_key.strip() if payload.api_key else None,
            allow_tools = payload.allow_tools,
        )
    except linked_instances_db.DuplicateName as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code = 404, detail = "Linked instance not found")
    linked_instances.forget(instance_id)
    return updated


@router.delete("/{instance_id}", dependencies = [Depends(_require_owner_ui)])
async def delete_linked_instance(instance_id: str):
    if not await asyncio.to_thread(linked_instances_db.delete_instance, instance_id):
        raise HTTPException(status_code = 404, detail = "Linked instance not found")
    linked_instances.forget(instance_id)
    return {"deleted": True}


@router.api_route(
    "/{instance_id}/proxy/{path:path}",
    methods = ["GET", "POST", "PATCH", "DELETE"],
    dependencies = [Depends(_require_owner_ui)],
    include_in_schema = False,
)
async def proxy_linked_instance(instance_id: str, path: str, request: Request):
    if not linked_instances.proxy_allowed(request.method, path):
        raise HTTPException(status_code = 404, detail = "Not available on a linked instance")
    instance = await asyncio.to_thread(linked_instances_db.get_instance, instance_id)
    if instance is None:
        raise HTTPException(status_code = 404, detail = "Linked instance not found")
    return await linked_instances.proxy(request, instance, path)


@router.post(
    "/{instance_id}/test",
    response_model = LinkedInstanceStatus,
    dependencies = [Depends(_require_owner_ui)],
)
async def test_linked_instance(instance_id: str):
    instance = await asyncio.to_thread(linked_instances_db.get_instance, instance_id)
    if instance is None:
        raise HTTPException(status_code = 404, detail = "Linked instance not found")
    linked_instances.forget(instance_id)
    return await _status(instance)

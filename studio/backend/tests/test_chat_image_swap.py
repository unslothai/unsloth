# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, Request

from core.inference import chat_image_swap, gpu_arbiter
from core.inference.chat_image_swap import generate_and_restore
from models.inference import ChatImageSwapRequest, InferenceStatusResponse, LoadRequest
from routes import inference


@pytest.mark.parametrize("gguf", [False, True], ids = ["transformers", "gguf"])
@pytest.mark.parametrize("failure", [None, "load", "render", "restore", "native lease"])
def test_swap_lifecycle(monkeypatch, gguf, failure):
    resident = "text"
    token = None
    events = []
    recipe = LoadRequest(
        model_path = "text",
        max_seq_length = 4096,
        gguf_variant = "Q4_K_M" if gguf else None,
        native_path_lease = "one-time-lease" if failure == "native lease" else None,
    )
    monkeypatch.setattr(chat_image_swap, "text_load_to_restore", lambda: (recipe, True))
    payload = ChatImageSwapRequest(
        text_id = "text",
        image_load = {"model_path": "image"},
        prompt = "A lighthouse",
    )
    monkeypatch.setattr(
        inference,
        "get_status",
        AsyncMock(
            return_value = InferenceStatusResponse(
                active_model = "text",
                loaded = ["text"],
                loading = [],
                is_gguf = gguf,
                inference = {},
            )
        ),
    )
    monkeypatch.setattr(inference, "_loaded_satisfies", lambda model: model == resident)
    monkeypatch.setattr(
        inference, "_loaded_slot_ident", lambda: resident if resident == "text" else None
    )
    monkeypatch.setattr(
        gpu_arbiter,
        "residents",
        lambda: (
            {}
            if resident is None
            else {"chat" if resident == "text" else "diffusion": frozenset({0})}
        ),
    )

    async def status(**kwargs):
        return SimpleNamespace(
            loaded = resident == "image",
            load_id = token,
            resident_load_id = token,
            repo_id = "image",
            family = "fixture",
            base_repo = None,
            model_dump = lambda: {},
        )

    async def unload_text(req, subject):
        nonlocal resident
        assert req.model_path == resident == "text"
        assert not req.force_cancel_active
        events.append("unload text")
        resident = None

    async def load_image(req, subject):
        nonlocal resident, token
        assert resident is None
        token = req.load_token
        events.append("load image")
        if failure == "load":
            raise RuntimeError("load")
        resident = "image"

    async def render(req, *, pinned_load):
        assert (
            resident == "image" and req.prompt == payload.prompt and pinned_load.repo_id == resident
        )
        events.append("render")
        if failure == "render":
            raise RuntimeError("render")
        return SimpleNamespace(images = [SimpleNamespace(id = "saved")])

    async def unload_image(req, subject, via_api_key):
        nonlocal resident, token
        assert req.expected_load_id == token
        events.append("unload image")
        resident = token = None

    async def restore(req, request, subject, *, user_initiated):
        nonlocal resident
        inference._check_switch_precondition(request)
        assert resident is None and req.model_path == "text"
        assert req.max_seq_length == 4096 and req.gguf_variant == recipe.gguf_variant
        events.append("restore")
        if failure == "restore":
            raise RuntimeError("restore")
        resident = "text"

    for name, method in {
        "diffusion_status": status,
        "_unload_model_impl": unload_text,
        "load_diffusion_model_gated": load_image,
        "_generate_diffusion_image_core": render,
        "unload_diffusion_model": unload_image,
        "load_model_gated": restore,
    }.items():
        monkeypatch.setattr(inference, name, method)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request({"type": "http", "headers": []}, receive)
    if failure == "native lease":
        with pytest.raises(HTTPException, match = "file picker") as error:
            asyncio.run(generate_and_restore(payload, request, "owner"))
        assert error.value.status_code == 409
        assert events == []
        assert resident == "text"
        return
    if failure:
        with pytest.raises((RuntimeError, HTTPException), match = failure):
            asyncio.run(generate_and_restore(payload, request, "owner"))
    else:
        assert asyncio.run(generate_and_restore(payload, request, "owner")).images[0].id == "saved"
    assert events == [
        "unload text",
        "load image",
        *([] if failure == "load" else ["render"]),
        "unload image",
        "restore",
    ]
    assert resident == (None if failure == "restore" else "text")

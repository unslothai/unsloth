# SPDX-License-Identifier: AGPL-3.0-only
"""Render between two completed chat requests. Keep the restore on the server if the tab closes."""

import asyncio
import time
import uuid

from fastapi import HTTPException, Request

from core.inference.generation_ticket import GenerationTicket, generation_ticket
from utils.account_context import current_account_id


_active = {}
_last_text_load = None


def _text_runtime():
    from routes import inference

    llama = inference.get_llama_cpp_backend()
    if getattr(llama, "is_loaded", False):
        return (
            "gguf",
            getattr(llama, "_process", None),
            id(getattr(llama, "last_load_intent", None)),
        )
    backend = inference._peek_inference_backend()
    name = getattr(backend, "active_model_name", None)
    if name:
        return (
            "transformers",
            getattr(backend, "_proc", None),
            id(getattr(backend, "models", {}).get(name)),
        )
    return None


def remember_text_load(request, *, user_initiated):
    global _last_text_load
    # Keep the actual launch request in server memory. Never retain a user's Hub credential.
    _last_text_load = (
        _text_runtime(),
        request.model_copy(update = {"hf_token": None}),
        user_initiated,
    )


def text_load_to_restore():
    runtime = _text_runtime()
    if runtime is not None and _last_text_load and _last_text_load[0] == runtime:
        return _last_text_load[1:]
    raise HTTPException(
        status_code = 409,
        detail = "Select the text model in Unsloth once so its load settings can be restored.",
    )


def cancel(swap_id):
    ticket = _active.get((current_account_id(), swap_id))
    if ticket is not None:
        ticket.stop()
    return ticket is not None


async def generate_and_restore(request, http_request, subject):
    ticket = GenerationTicket()
    key = (current_account_id(), request.swap_id or uuid.uuid4().hex)
    if _active.setdefault(key, ticket) is not ticket:
        raise HTTPException(
            status_code = 409, detail = "An image swap with this ID is already running."
        )
    try:
        return await _run(request, http_request, subject, ticket)
    finally:
        _active.pop(key, None)


async def _run(request, http_request, subject, ticket):
    from core.inference import gpu_arbiter, llama_keepwarm, media_keepwarm
    from core.inference.chat_swap_kv import preparing_load
    from core.inference.diffusion_families import load_identity
    from core.inference.media_switch_locks import gpu_switch_lock
    from core.inference.generate_image_tool import _build_request, offers_txt2img
    from models.inference import MediaUnloadRequest, UnloadRequest
    from routes import inference

    def busy(message):
        return HTTPException(status_code = 409, detail = message)

    def require_empty():
        if (
            inference._loaded_slot_ident()
            or inference.model_slots.busy()
            or gpu_arbiter.residents()
        ):
            raise busy("Another model took the slot during the image call.")

    async def image_status():
        return await inference.diffusion_status(current_subject = subject, via_api_key = False)

    token = uuid.uuid4().hex
    image_load = request.image_load.model_copy(update = {"load_token": token})
    # A manual load arriving between the last status read and restore wins. The existing loader
    # checks this precondition again while holding the actual model lifecycle gate.
    restore_request = Request(dict(http_request.scope))
    inference.require_for_model_switch(restore_request.scope, require_empty)
    async with gpu_switch_lock():
        chat = await inference.get_status(subject)
        recipe, user_initiated = text_load_to_restore()
        if recipe.native_path_lease:
            # The desktop file picker grants a single load. The already-used lease
            # cannot restore the chat model once it has been unloaded.
            raise busy("Models opened through the desktop file picker cannot use image swapping.")
        text_load = recipe.model_copy(
            update = {
                "force_cancel_active": False,
                "force_reload": False,
                "alongside": False,
                "load_request_id": None,
                **(
                    {"gguf_variant": chat.gguf_variant}
                    if chat.is_gguf and chat.gguf_variant
                    else {}
                ),
                **({"load_in_4bit": chat.load_in_4bit} if chat.load_in_4bit is not None else {}),
            }
        )
        if (
            request.text_id not in (chat.active_model, chat.model_identifier)
            or not await asyncio.to_thread(inference._loaded_satisfies, text_load.model_path)
            or len(chat.loaded) != 1
            or chat.loading
            or chat.is_audio
            or chat.is_diffusion
            or chat.is_npu
            or chat.is_mlx
            or chat.engine != "auto"
        ):
            raise busy(
                "The selected GGUF or Transformers chat model must be the only loaded model."
            )
        before = await image_status()
        if before.loaded or before.load_id or set(gpu_arbiter.residents()) - {gpu_arbiter.CHAT}:
            raise busy("Another model already occupies the image or GPU slot.")
        if ticket.cancelled() or await http_request.is_disconnected():
            raise busy("The chat disconnected before the model swap.")
        await inference._unload_model_impl(UnloadRequest(model_path = request.text_id), subject)
        saved = None
        try:
            async with (
                llama_keepwarm.model_load_gate(),
                llama_keepwarm.inference_lifecycle_gate(),
                media_keepwarm.admission_gate(gpu_arbiter.DIFFUSION),
                media_keepwarm.admission_gate(gpu_arbiter.VIDEO),
            ):
                require_empty()
                if inference.active_generations.count(include_external = False):
                    raise busy("Another chat is running.")
                try:
                    with preparing_load("image", token):
                        await inference.load_diffusion_model_gated(image_load, subject)
                    deadline = time.monotonic() + 600
                    while True:
                        if ticket.cancelled():
                            raise busy("Image generation was stopped.")
                        state = await image_status()
                        if state.loaded and state.load_id == state.resident_load_id == token:
                            break
                        progress = await inference.diffusion_load_progress(subject, False)
                        if progress.phase == "error":
                            raise HTTPException(
                                status_code = 503, detail = progress.error or "Image load failed."
                            )
                        if state.load_id != token or time.monotonic() >= deadline:
                            raise busy("The selected image model did not finish loading.")
                        await asyncio.sleep(0.25)
                    identity = load_identity(state.repo_id, state.base_repo, state.family)
                    status = state.model_dump()
                    if not offers_txt2img(status):
                        raise busy("The selected image model only supports editing.")
                    options = request.model_dump(
                        include = {"width", "height", "seed", "negative_prompt"}, exclude_none = True
                    )
                    with generation_ticket(ticket):
                        saved = await inference._generate_diffusion_image_core(
                            _build_request(options, request.prompt, status), pinned_load = identity
                        )
                finally:
                    await inference.unload_diffusion_model(
                        MediaUnloadRequest(expected_load_id = token),
                        subject,
                        False,
                    )
        finally:
            try:
                require_empty()
                await inference.load_model_gated(
                    text_load, restore_request, subject, user_initiated = user_initiated
                )
            except Exception as exc:
                detail = getattr(exc, "detail", str(exc))
                gallery = (
                    f" Image {saved.images[0].id} is saved in the gallery."
                    if saved and saved.images
                    else ""
                )
                raise HTTPException(
                    status_code = 503,
                    detail = f"Could not restore {request.text_id}: {detail}.{gallery}",
                ) from exc
        return saved

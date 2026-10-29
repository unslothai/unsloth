# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a request may load the model it names, and the coded refusal when it may not.

The routes resolve the target and run the load; this decides, from the request's ``LoadPolicy``
and what the load would do to the models already loaded, whether it runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from core.inference.load_policy import LoadPolicy, current_load_policy

# The Unsloth Studio page where a user loads each kind of model.
PAGES = {
    "llm": "Chat",
    # The arbiter's name for the chat slot.
    "chat": "Chat",
    "tts": "Audio",
    "audio": "Audio",
    "stt": "Audio",
    "image": "Image",
    "video": "Video",
}
_WHAT = {
    "llm": "chat model",
    "tts": "text-to-speech model",
    "audio": "music model",
    "stt": "speech-to-text model",
    "image": "image model",
    "video": "video model",
}


@dataclass(frozen = True)
class Load:
    """The load may run; ``beside`` keeps the models already loaded."""

    beside: bool


@dataclass(frozen = True)
class Refusal:
    code: str
    status: int
    message: str
    kind: Optional[str] = None
    model: Optional[str] = None
    page: Optional[str] = None

    def headers(self) -> dict:
        from utils.api_errors import problem_headers
        return problem_headers(self.code, kind = self.kind, model = self.model, page = self.page)

    def http_exception(self, path: Optional[str]):
        """/v1 answers in its envelope; other routes keep a plain detail string, as the UI reads them."""
        from fastapi import HTTPException
        from utils.api_errors import error_body_for_path, wants_api_error_envelope

        detail = (
            error_body_for_path(
                path, self.message, status = self.status, code = self.code, param = "model"
            )
            if isinstance(path, str) and wants_api_error_envelope(path)
            else self.message
        )
        return HTTPException(status_code = self.status, detail = detail, headers = self.headers())


Outcome = Union[Load, Refusal]


MULTI_MODEL_SETTING = "Keep multiple models loaded (Settings > System > Model memory)"
# (why it is refused, what the user can change so a retry works), for a credential that may only load beside.
SWITCH_OFF_FOR_ACCOUNT = (
    "'Switch model by request' is off for this account",
    "have the owner turn it on in Settings > API",
)


def _why_it_would_unload() -> tuple[str, Optional[str]]:
    """Why a load would unload the chat model, and the setting that would keep it, by the current mode."""
    from utils.multi_model_settings import get_multi_model_mode

    try:
        mode = get_multi_model_mode()
    except Exception:  # noqa: BLE001 - unread: no setting to name
        mode = None
    if mode == "off":
        return (
            f"{MULTI_MODEL_SETTING} is Off, so loading it would unload the model serving this chat",
            "set it to Always",
        )
    if mode == "fit":
        return (
            f"{MULTI_MODEL_SETTING} is When they fit, and it does not fit beside the model serving "
            "this chat",
            "set it to Always",
        )
    return ("Loading it here would unload the model serving this chat", None)


def needs_user_load(
    kind: str,
    model: Optional[str],
    *,
    reason: Optional[tuple[str, Optional[str]]] = None,
) -> Refusal:
    """``reason`` (why, fix) overrides the multi-model one, for a refusal that setting would not lift."""
    page = PAGES.get(kind, "Chat")
    what = _WHAT.get(kind, "model")
    named = f"The {what} '{model}'" if model else f"A {what}"
    why, fix = reason or _why_it_would_unload()
    on_page = f"load it on the {page} page in Unsloth Studio"
    ask = (
        f"Ask the user to {fix} and then try again, or to {on_page}."
        if fix
        else f"Ask the user to {on_page}, then try again."
    )
    return Refusal(
        code = "needs_user_load",
        status = 403,
        message = f"{named} is not loaded. {why}. {ask}",
        kind = kind,
        model = model,
        page = page,
    )


def needs_download(kind: str, model: str) -> Refusal:
    """A named model that is not on disk, for a credential that may not start a download by loading it."""
    page = PAGES.get(kind, "Chat")
    return Refusal(
        code = "needs_download",
        status = 404,
        message = (
            f"The {_WHAT.get(kind, 'model')} '{model}' is not downloaded yet. Download it first, "
            f"or ask the user to load it on the {page} page in Unsloth Studio."
        ),
        kind = kind,
        model = model,
        page = page,
    )


def not_resident_with_switch_off(kind: str, model: str) -> Refusal:
    """A named model that is not loaded while "Switch model by request" is off."""
    page = PAGES.get(kind, "Chat")
    if current_load_policy() is not LoadPolicy.ANY:
        return needs_user_load(kind, model, reason = SWITCH_OFF_FOR_ACCOUNT)
    return Refusal(
        code = "not_loaded",
        status = 400,
        message = (
            f"The {_WHAT.get(kind, 'model')} '{model}' is not loaded, and 'Switch model by "
            f"request' is off. Load it on the {page} page in Unsloth Studio, or turn on 'Switch "
            "model by request' under Settings > API to load it from a request."
        ),
        kind = kind,
        model = model,
        page = page,
    )


def decide_load(
    *,
    kind: str,
    model: Optional[str],
    beside: bool,
    unloads_chat: bool,
    policy: Optional[LoadPolicy] = None,
) -> Outcome:
    """``unloads_chat``: the load, run as it would run, unloads a loaded chat model."""
    policy = current_load_policy() if policy is None else policy
    if policy is LoadPolicy.NEVER or (policy is LoadPolicy.BESIDE_ONLY and unloads_chat):
        return needs_user_load(kind, model)
    return Load(beside = beside)


def ensure_load(
    *,
    kind: str,
    model: Optional[str],
    beside: bool,
    unloads_chat: bool,
    path: Optional[str] = None,
) -> Load:
    """``decide_load`` that raises its refusal; ``path`` picks the error shape."""
    outcome = decide_load(kind = kind, model = model, beside = beside, unloads_chat = unloads_chat)
    if isinstance(outcome, Refusal):
        raise outcome.http_exception(path)
    return outcome


def media_load_keeps_chat(owner: str) -> bool:
    """True only when the image or video load provably leaves the chat model loaded."""
    from core.inference import gpu_arbiter
    from core.inference.media_switch_backends import load_takes_the_gpu
    from utils.multi_model_settings import get_multi_model_mode

    if not load_takes_the_gpu():
        return True
    # Unpriced (no need_mib), so When they fit counts as evicting: a doubtful case refuses.
    return not gpu_arbiter.would_evict(owner, policy = get_multi_model_mode())


def ensure_media_load(owner: str, kind: str, model: str, *, openai_errors: bool) -> None:
    """Refuses an image or video switch the request's policy does not allow."""
    policy = current_load_policy()
    if policy is LoadPolicy.ANY:
        return
    if policy is LoadPolicy.BESIDE_ONLY and media_load_keeps_chat(owner):
        return
    from core.inference.media_switch_errors import refuse

    refusal = needs_user_load(kind, model)
    exc = refuse(
        refusal.message,
        status_code = refusal.status,
        openai_errors = openai_errors,
        code = refusal.code,
    )
    exc.headers = refusal.headers()
    raise exc

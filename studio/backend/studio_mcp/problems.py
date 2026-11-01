# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A route's machine code for a failure, and what the agent should do next.

Routes name the state in the ``X-Unsloth-Problem`` header (``code k=v``) or in the code field of
their error body; nothing here reads the English. The next step is an action, rendered against
the tools the caller has, so a hint never names a tool the caller cannot call."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from core.inference.ensure_model import PAGES

PROBLEM_HEADER = "x-unsloth-problem"
_CODE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
# Fields the tag repeats, per code: the rest stay in the message.
TAG_FIELDS = {
    "not_loaded": ("kind",),
    "needs_user_load": ("kind", "page"),
    "needs_download": ("kind",),
    "multi_model_always": ("kind",),
}
WHAT = {
    "llm": "chat model",
    "tts": "text-to-speech or music model",
    "audio": "text-to-speech or music model",
    "stt": "speech-to-text model",
    "image": "image model",
    "video": "video model",
}
# load_model's kind for each not_loaded kind; a music model loads as tts.
LOAD_KINDS = {"tts": "tts", "audio": "tts", "stt": "stt", "image": "image", "video": "video"}
# multi_model_always names the slot as the arbiter does (chat); unload_model calls it llm.
UNLOAD_KINDS = {"chat": "llm", "audio": "audio", "image": "image", "video": "video"}

_UPDATE_AUDIO_RUNTIME = (
    "Update the audio runtime (`unsloth studio update`), then load the model again."
)


@dataclass(frozen = True)
class Problem:
    code: str
    fields: Mapping[str, str] = field(default_factory = dict)

    def tag_fields(self) -> dict[str, str]:
        keys = TAG_FIELDS.get(self.code, ())
        return {key: self.fields[key] for key in keys if key in self.fields}


def parse(value: Any) -> Optional[Problem]:
    """``code k=v ...`` as a route sent it; None for anything else."""
    if not isinstance(value, str):
        return None
    code, *pairs = value.split()
    if not _CODE.match(code):
        return None
    fields = dict(pair.split("=", 1) for pair in pairs if "=" in pair)
    return Problem(code, fields)


def from_headers(headers: Mapping[str, str]) -> Optional[Problem]:
    return parse(headers.get(PROBLEM_HEADER))


def caller_tools() -> frozenset[str]:
    """The tools the current caller's profile offers."""
    from mcp_server import surface
    from studio_mcp.caller import current_caller
    from studio_mcp.profile import AGENT

    try:
        profile = current_caller().profile
    except RuntimeError:
        profile = AGENT
    return surface(profile)


def _ask_user_load(kind: str, model: Optional[str] = None) -> str:
    page = PAGES.get(kind, "Chat")
    what = f"'{model}'" if model else f"a {WHAT.get(kind, 'model')}"
    return f"Ask the user to load {what} on the {page} page in Unsloth Studio, then try again."


def download_hint(kind: str, model: Optional[str], tools: frozenset[str]) -> str:
    if "download_model" not in tools or not model:
        return _ask_user_load(kind, model)
    return (
        f"Call download_model(model='{model}', kind='{kind}'), follow it with "
        "get_job(kind='download', id=<its id>), then try again."
    )


def load_hint(kind: str, tools: frozenset[str]) -> str:
    if "load_model" not in tools:
        return _ask_user_load(kind)
    if kind in ("llm", "chat"):
        return "Load a chat model with load_model first; list_models shows the downloaded ones."
    load_kind = LOAD_KINDS.get(kind, kind)
    article = "an" if WHAT.get(kind, "model")[0] in "aeiou" else "a"
    return f"Load {article} {WHAT.get(kind, 'model')} with load_model(kind='{load_kind}') first."


def unload_hint(kind: Optional[str], tools: frozenset[str]) -> str:
    unload = UNLOAD_KINDS.get(kind or "")
    if "unload_model" in tools:
        if unload is None:
            return (
                "Call unload_model(kind=…) for the model it names to free the GPU; "
                "retrying won't help."
            )
        return f"Call unload_model(kind='{unload}') to free the GPU; retrying won't help."
    return (
        "Ask the user to unload the model it names in Unsloth Studio, or to choose When they fit; "
        "retrying won't help."
    )


def wait_hint(tools: frozenset[str]) -> str:
    if "cancel" in tools:
        return (
            "Wait for the running chats to finish, or cancel a chat you started with"
            " cancel(kind='chat', id=<its cancel_id>)."
        )
    return "Wait for the running chats to finish, then try again."


def hint(problem: Problem, tools: frozenset[str]) -> Optional[str]:
    kind = problem.fields.get("kind")
    if problem.code == "not_loaded":
        return load_hint(kind or "llm", tools)
    if problem.code == "needs_user_load":
        # The route's message already says what the user must do and why.
        return None
    if problem.code == "needs_download":
        return download_hint(kind or "llm", problem.fields.get("model"), tools)
    if problem.code == "multi_model_always":
        return unload_hint(kind, tools)
    if problem.code == "active_generations":
        return wait_hint(tools)
    return None


def audio_hint(code: str, smaller_variants: list[str], tools: frozenset[str]) -> Optional[str]:
    """What to do about a coded audio runtime failure."""
    if code in ("runtime_missing", "family_unknown_to_runtime", "espeak_missing"):
        return _UPDATE_AUDIO_RUNTIME
    if code == "file_incomplete":
        return "Download the model again, or load another variant."
    loads = "load_model" in tools
    if code == "runtime_stopped":
        if loads:
            return (
                "Retry once; if it fails again, run studio_status and load_model the audio "
                "model again."
            )
        return (
            "Retry once; if it fails again, ask the user to reload the audio model in "
            "Unsloth Studio."
        )
    if code in ("out_of_memory", "gpu_unavailable"):
        if not loads:
            return "Ask the user to reload the audio model on the CPU in Unsloth Studio."
        variants = (
            f", or a smaller variant with load_model(variant=...): {', '.join(smaller_variants)}"
            if code == "out_of_memory" and smaller_variants
            else ""
        )
        return f"Load the audio model again with device='cpu'{variants}."
    return None


AUDIO_CODES = frozenset(
    (
        "out_of_memory",
        "gpu_unavailable",
        "runtime_missing",
        "family_unknown_to_runtime",
        "espeak_missing",
        "runtime_stopped",
        "file_incomplete",
    )
)

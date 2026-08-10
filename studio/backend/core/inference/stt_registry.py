# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One place that knows which dictation models are resident, and loads them.

The sidecars still own their processes: whisper.cpp serves GGML through
whisper-server, llama.cpp serves mtmd models, and Transformers loads in
process. What lives here is the lifecycle above them, so the orchestrator has a
single view of dictation the way it has one of chat, and Voice settings and
Model Hub cannot report different things about the same model.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

# Every engine a dictation model can be resident on. Order is the order an
# unload sweeps them, which matters only for logging.
STT_ENGINES = ("transformers", "gguf", "mtmd")

# Serialises load-then-release so two loads on different engines cannot leave both resident.
_load_lock = threading.Lock()


def sidecar_for(engine: str) -> Any:
    """The sidecar serving ``engine``. Transformers is the catch-all."""
    if engine == "mtmd":
        from core.inference.stt_mtmd_sidecar import get_mtmd_stt_sidecar
        return get_mtmd_stt_sidecar()
    if engine == "gguf":
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        return get_ggml_stt_sidecar()
    from core.inference.stt_sidecar import get_stt_sidecar

    return get_stt_sidecar()


def load(
    model: Optional[str],
    engine: str,
    request_cancel_event: Optional[threading.Event] = None,
) -> None:
    """Make ``model`` resident on ``engine``, then release every idle other engine.

    Dictation is one user-visible choice, so the engines are alternatives rather
    than slots: a Transformers Whisper and a llama.cpp Qwen3-ASR held at once
    doubles VRAM for the whole keep-alive window. An engine serving a request
    keeps its model and releases it on its own idle timer, since waiting for a
    transcription that may run for minutes would stall this load. Raises what
    the sidecar raises, before anything is released: a 409 for a model that is not
    downloaded must not cost the user the engine they were already using, which is the
    order `_load_locked` keeps for an in-engine switch.
    """
    with _load_lock:
        sidecar_for(engine).load(model, request_cancel_event = request_cancel_event)
        unload([name for name in STT_ENGINES if name != engine], wait = False)


def unload(engines: Optional[Sequence[str]] = None, *, wait: bool = True) -> list[str]:
    """Release every named engine (all of them by default), reporting refusals.

    Each is attempted even after a failure: more than one can hold memory at
    once after an engine switch, so stopping early would strand the rest.
    ``wait=False`` leaves a sidecar that is mid-request resident instead of
    blocking on it, for callers releasing engines they do not own.
    """
    failed: list[str] = []
    for name in STT_ENGINES if engines is None else engines:
        try:
            sidecar_for(name).unload(wait = wait)
        except Exception as exc:  # noqa: BLE001 - report after attempting all
            logger.warning("Failed to unload STT engine '%s': %s", name, exc)
            failed.append(name)
    return failed


def resident() -> dict:
    """What dictation currently holds, for the shared inference status.

    Never raises: a sidecar that cannot even be imported reports nothing rather
    than taking the status endpoint down with it.
    """
    for engine in STT_ENGINES:
        try:
            sidecar = sidecar_for(engine)
            model = sidecar.loaded_model
            if model:
                return {
                    "model": model,
                    "engine": engine,
                    "device": sidecar.device,
                    "loading": False,
                }
            if sidecar.is_loading():
                return {
                    "model": None,
                    "engine": engine,
                    "device": None,
                    "loading": True,
                }
        except Exception as exc:  # noqa: BLE001 - one engine must not hide the rest
            logger.debug("Could not inspect STT engine '%s': %s", engine, exc)
    return {"model": None, "engine": None, "device": None, "loading": False}

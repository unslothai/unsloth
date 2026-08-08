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
    """Make ``model`` resident on ``engine``. Raises what the sidecar raises."""
    sidecar_for(engine).load(model, request_cancel_event = request_cancel_event)


def unload(engines: Optional[Sequence[str]] = None) -> list[str]:
    """Release every named engine (all of them by default), reporting refusals.

    Each is attempted even after a failure: more than one can hold memory at
    once after an engine switch, so stopping early would strand the rest.
    """
    failed: list[str] = []
    for name in STT_ENGINES if engines is None else engines:
        try:
            sidecar_for(name).unload()
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

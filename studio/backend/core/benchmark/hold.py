# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scoring hold: while a benchmark scores the loaded model, nothing may swap that model."""

import contextlib
import threading
import time
import uuid
from typing import Optional

_lock = threading.Lock()
# Scorings in progress. They answer with whatever model is loaded, so a load or unload by anyone
# while one runs would score the later rows with another model under the first one's name.
_scorings = 0
# A scoring outliving this would be a hung worker; the hold is let go when the worker ends anyway.
SCORING_LEASE_S = 6 * 3600
LOAD_PATHS = ("/api/inference/load", "/api/inference/unload")


@contextlib.contextmanager
def _gpu_lease():
    """The Studio-wide GPU lease: training's start, a video load and every other residency change
    refuse instead of evicting the model being scored."""
    from core.inference import gpu_arbiter

    holder = f"benchmark-scoring:{uuid.uuid4().hex}"
    for attempt in range(40):
        try:
            lease = gpu_arbiter.operation_lease(
                "benchmark", holder, SCORING_LEASE_S, label = "A benchmark scoring"
            )
            lease.__enter__()
            break
        except gpu_arbiter.GpuLeaseHeldError as exc:
            # The arbiter mid-change reads as busy for a moment; another operation's lease stays held.
            if exc.label != "A model change" or attempt == 39:
                raise
            time.sleep(0.25)
    try:
        yield
    finally:
        lease.__exit__(None, None, None)


@contextlib.contextmanager
def scoring_hold():
    """While a scoring (held-out rows or an lm-eval task) runs, nothing may change the model it
    answers with. The scoring itself never loads."""
    global _scorings
    with _lock:
        _scorings += 1
    try:
        with _gpu_lease():
            yield
    finally:
        with _lock:
            _scorings -= 1


def scoring_active() -> bool:
    with _lock:
        return _scorings > 0


def loaded_identity() -> str:
    """The model Chat's main slot holds, with its quant: what a scoring answers with."""
    from routes.data_recipe.jobs import _loaded_local_model_identity

    _, model, variant = _loaded_local_model_identity()
    return f"{model}:{variant}" if variant else model


def check_resident(expected: Optional[str]) -> None:
    """Under the hold: the model a caller loaded is still the one loaded. Between its load and the
    hold another client could swap it, and the run would score that model under the first's name."""
    if expected and (now := loaded_identity()) != expected:
        raise RuntimeError(
            f"The model to score changed after it was loaded ({now or 'nothing'} is loaded now, "
            f"not {expected}). Run the benchmark again."
        )


class ScoringLoadGuardMiddleware:
    """Pure ASGI: refuses model loads and unloads while a scoring runs, whichever route they use."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            _scorings
            and scope.get("type") == "http"
            and scope.get("method") == "POST"
            and (path := scope.get("path") or "") in LOAD_PATHS
        ):
            from core.inference.llama_keepwarm import _carries_bearer_credentials

            # Uncredentialed calls go on to the route's own 401.
            if _carries_bearer_credentials(scope, path) and scoring_active():
                from starlette.responses import JSONResponse
                await JSONResponse(
                    {
                        "detail": "A benchmark is scoring this model. Stop the flow scoring it "
                        "or let it finish, then try again."
                    },
                    status_code = 409,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)

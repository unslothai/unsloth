# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-GPU arbiter for Unsloth's heavy GPU consumers.

The chat backends, diffusion, and video share one GPU. Before taking it each calls
``acquire_for(owner)``, which evicts the current other owner so two large models never sit in VRAM
at once. The arbiter only sequences ownership (freeing is each backend's teardown); eviction runs
under the lock, so a transfer is atomic vs other acquires.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

CHAT = "chat"
DIFFUSION = "diffusion"
VIDEO = "video"

_lock = threading.Lock()
_owner: Optional[str] = None
_owner_epoch = 0
# Account whose load put the current owner on the GPU, so routes can refuse to evict it.
_owner_account: Optional[str] = None
_prior_account: Optional[str] = None


class OwnerChangedError(RuntimeError):
    """The outgoing GPU owner changed after a pre-handoff capacity snapshot."""


def _evict_chat() -> None:
    import time

    from core.inference import get_inference_backend
    from routes.inference import get_llama_cpp_backend

    from core.inference.llama_cpp import chat_load_active

    llama = get_llama_cpp_backend()
    # is_active (process exists), not is_loaded (exists AND healthy): a chat model still starting up holds VRAM but is
    # not healthy. chat_load_active too, since an HF load has no process until its GGUF downloaded. unload_model sets
    # the cancel event the download loop polls, so it aborts.
    if llama.is_active or chat_load_active():
        llama.unload_model()
    orchestrator = get_inference_backend()
    if orchestrator.active_model_name:
        orchestrator.unload_model(orchestrator.active_model_name)
    # An in-flight safetensors load has no active_model_name yet (published only on success), so the unload above misses
    # it and it would finish onto the GPU we just granted away. cancel_load discards the loading marker BEFORE tearing
    # the worker down, and runs off the lifecycle gate.
    for pending in list(getattr(orchestrator, "loading_models", ()) or ()):
        orchestrator.cancel_load(pending)
    # Kill the subprocess too: its base CUDA context holds VRAM diffusion needs.
    orchestrator._shutdown_subprocess(timeout = 5.0)
    # The driver reclaims the killed VRAM asynchronously, so wait for it to settle before diffusion allocates, else a
    # warm handoff can transiently OOM.
    llama._wait_for_vram_settle(since_kill = time.monotonic())
    from hub.services.models.account_access import clear_resident

    clear_resident(CHAT)


def _evict_diffusion() -> None:
    # Unload whichever engine the router has active (diffusers or native sd.cpp).
    from core.inference.diffusion_engine_router import get_active_diffusion_engine
    get_active_diffusion_engine().unload()


def _evict_video() -> None:
    from core.inference.video import get_video_backend
    get_video_backend().unload()


# Patchable in tests via monkeypatch.setitem. Ownership is exclusive, so acquire_for's evict-the-current-owner
# generalises to any number of owners.
_EVICTORS = {CHAT: _evict_chat, DIFFUSION: _evict_diffusion, VIDEO: _evict_video}


class GpuOwnerBusyError(RuntimeError):
    """Raised when an ownership transfer is configured to refuse eviction."""

    def __init__(self, owner: str):
        self.owner = owner
        super().__init__(f"GPU is owned by {owner}")


class GpuBusyForAnotherAccountError(GpuOwnerBusyError):
    """Another account is generating on the resident model; routes answer 409 ``gpu_busy``."""

    def __init__(self, owner: str, active: int):
        self.active = active
        super().__init__(owner)

    @property
    def retry_after(self) -> int:
        from core.inference.llama_admission import estimate_gpu_retry_after
        return estimate_gpu_retry_after()

    def as_http_exception(self, path: Optional[str] = None):
        """A retryable refusal without another account's model or conversation ids."""
        from fastapi import HTTPException
        from utils.api_errors import error_body_for_path

        retry_after = self.retry_after
        message = "Another account is generating on the resident model. Retry after it finishes."
        detail = (
            error_body_for_path(path, message, status = 409, code = "gpu_busy", param = "model")
            if path and path.startswith("/v1/")
            else {"error": "gpu_busy", "message": message, "retry_after": retry_after}
        )
        return HTTPException(
            status_code = 409,
            detail = detail,
            headers = {"Retry-After": str(retry_after)},
        )


def other_accounts_active(account_id: str) -> int:
    from state import active_generations

    # Image and video jobs never enter active_generations, so ask their own trackers too.
    from hub.services.models.account_access import foreign_media_generations
    return active_generations.foreign_count(account_id) + foreign_media_generations(account_id)


def raise_if_other_accounts_active(account_id: Optional[str] = None) -> None:
    """Guard a destructive reload; call under the lifecycle gate before touching any backend."""
    from utils.account_context import current_account_id

    busy = other_accounts_active(account_id or current_account_id())
    if busy:
        raise GpuBusyForAnotherAccountError(_owner or CHAT, busy)


def require_no_foreign_generations(
    account_id: Optional[str] = None, *, path: Optional[str] = None
) -> None:
    try:
        raise_if_other_accounts_active(account_id)
    except GpuBusyForAnotherAccountError as exc:
        raise exc.as_http_exception(path) from exc


def acquire_for_request(
    owner: str,
    register = None,
    **kwargs,
) -> Any:
    try:
        return acquire_for(owner, register, **kwargs)
    except GpuBusyForAnotherAccountError as exc:
        raise exc.as_http_exception() from exc


def acquire_for(
    owner: str,
    register: Optional[Callable[[], Any]] = None,
    *,
    expected_current: Optional[tuple[Optional[str], int]] = None,
    allow_evict: bool = True,
    account_id: Optional[str] = None,
    replacing: bool = False,
) -> Any:
    """Make ``owner`` the sole GPU owner, evicting the other if it holds it.

    ``register`` runs under the arbiter lock as ownership transfers, so a competing acquire cannot
    evict this owner and let both loaders allocate VRAM at once.
    """
    global _owner, _owner_epoch, _owner_account, _prior_account
    if owner not in _EVICTORS:
        raise ValueError(f"unknown GPU owner: {owner!r}")
    from utils.account_context import current_account_id

    acting = account_id or current_account_id()
    with _lock:
        if expected_current is not None and (_owner, _owner_epoch) != expected_current:
            raise OwnerChangedError("The resident GPU model changed; retry the load.")
        if register is not None or replacing:
            raise_if_other_accounts_active(acting)
        if _owner is not None and _owner != owner:
            if not allow_evict:
                raise GpuOwnerBusyError(_owner)
            # Never evict an account mid-generation; the caller retries after its stream ends.
            busy = other_accounts_active(acting)
            if busy:
                raise GpuBusyForAnotherAccountError(_owner, busy)
            logger.info("gpu_arbiter: evicting %s for %s", _owner, owner)
            _EVICTORS[_owner]()
        # Records who LOADED the model; a plain re-assert must not hand it to whoever asked last.
        claims = _owner != owner or register is not None or replacing
        _owner = owner
        _owner_epoch += 1
        result = register() if register is not None else None
        # A raising registration loaded nothing and must not take residency.
        if claims:
            _prior_account, _owner_account = _owner_account, acting
        return result


def restore_owner_account(owner: str, account_id: Optional[str] = None) -> bool:
    """Hand residency back to the displaced account when a claim's load never committed."""
    global _owner_account, _prior_account
    from utils.account_context import current_account_id

    with _lock:
        acting = account_id or current_account_id()
        if _owner != owner or _owner_account != acting or _prior_account is None:
            return False
        _owner_account, _prior_account = _prior_account, None
        return True


def release(owner: str) -> None:
    """Drop ``owner``'s claim (no-op if it isn't the current owner)."""
    global _owner, _owner_epoch, _owner_account, _prior_account
    with _lock:
        if _owner == owner:
            _owner = None
            _owner_account = None
            _prior_account = None
            _owner_epoch += 1


def release_if(owner: str, predicate: Callable[[], bool]) -> bool:
    """Drop ``owner``'s claim only if it still holds it AND ``predicate()`` is true, atomically.

    A slow unload's idle check and its ``release`` must not straddle a concurrent same-owner load
    whose ``acquire_for(register=...)`` re-registers ownership under this lock; evaluating the
    predicate under the lock keeps them atomic so ``release`` never clears the newer claim.
    ``predicate`` must be quick and not re-enter the arbiter. Returns True iff ownership was dropped."""
    global _owner, _owner_epoch, _owner_account, _prior_account
    with _lock:
        if _owner != owner or not predicate():
            return False
        _owner = None
        _owner_account = None
        _prior_account = None
        _owner_epoch += 1
        return True


def current_owner() -> Optional[str]:
    return _owner


def owner_account() -> Optional[str]:
    return _owner_account


def owner_snapshot() -> tuple[Optional[str], int]:
    with _lock:
        return _owner, _owner_epoch

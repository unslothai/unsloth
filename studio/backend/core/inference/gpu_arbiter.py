# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU arbiter: default evicts other residents, 'always' refuses instead; eviction runs under lock"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

CHAT = "chat"
DIFFUSION = "diffusion"
VIDEO = "video"
# A Decision API llama.cpp server on the GPU (the PyTorch Clef worker is not arbitrated).
DECISIONS = "decisions"

# What each kind is called in refusals and eviction reports.
KIND_LABELS = {CHAT: "chat", DIFFUSION: "image", VIDEO: "video", DECISIONS: "decisions"}
# Residents that "always" protects; a Decision API server is still evicted as before.
_KEPT_UNDER_ALWAYS = frozenset({CHAT, DIFFUSION, VIDEO})

_lock = threading.Lock()
_owner: Optional[str] = None
_owner_epoch = 0
# Account whose load put the current owner on the GPU, so routes can refuse to evict it.
_owner_account: Optional[str] = None
_prior_account: Optional[str] = None
# Every resident and the physical GPU ids it uses (None = unknown, treat as every GPU).
_residents: dict[str, Optional[frozenset[int]]] = {}
# (account, prior account) for residents other than ``_owner``, whose pair lives in the globals above.
_accounts: dict[str, tuple[Optional[str], Optional[str]]] = {}
# What each image or video resident was admitted with under When they fit / Always (gpu_residency.MediaNeed).
_media_needs: dict[str, dict] = {}
# The price each media owner had before its newest load started (None: unpriced), put back by
# restore_media_need when that load fails and the model it was replacing stays loaded.
_media_need_rollback: dict[str, Optional[dict]] = {}

# Phrases the refusal copy always contains, for readers that lose the X-Unsloth-Refusal header.
MULTI_MODEL_REFUSAL_MARKERS = ("is set to Always", "is set to keep loaded")
MULTI_MODEL_REFUSAL_HEADERS = {"X-Unsloth-Refusal": "multi-model-always"}


class OwnerChangedError(RuntimeError):
    """The outgoing GPU owner changed after a pre-handoff capacity snapshot."""


class MultiModelAlwaysRefusal(RuntimeError):
    """A load would unload another model while Keep multiple models loaded is set to Always.

    Not a ``GpuMemoryShortError``: the chat load loop answers that by evicting. Waiting does not
    help, so routes answer 409 without Retry-After, with ``MULTI_MODEL_REFUSAL_HEADERS``.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.kind = kind
        self.model = model
        super().__init__(message)

    def as_http_exception(self, path: Optional[str] = None):
        from fastapi import HTTPException
        from utils.api_errors import error_body_for_path

        message = str(self)
        detail = (
            error_body_for_path(path, message, status = 409, code = "multi_model_always", param = "model")
            if path and path.startswith("/v1/")
            else message
        )
        return HTTPException(
            status_code = 409, detail = detail, headers = dict(MULTI_MODEL_REFUSAL_HEADERS)
        )


def always_refusal_message(
    kind: str,
    model: Optional[str] = None,
    *,
    is_owner: Optional[bool] = None,
) -> str:
    """The Always refusal copy. ``kind`` is chat, audio, image or video."""
    if is_owner is None:
        from utils.account_context import is_owner_context
        is_owner = is_owner_context()
    what = f"the {kind} model {model}" if model else f"the {kind} model"
    if is_owner:
        return (
            f"Loading this model would unload {what}, and Keep multiple models loaded is set to Always. "
            "Unload that model first, or choose When they fit."
        )
    return (
        f"Loading this model would unload {what}, which Unsloth Studio is set to keep loaded. "
        "Unload that model first, or ask the Unsloth Studio owner."
    )


def refusal_for(
    slots: Iterable[dict[str, Any]], fallback_kind: str = "chat"
) -> MultiModelAlwaysRefusal:
    """Build the refusal naming the first model a load would have unloaded (hidden from other accounts)."""
    shown = public_slots(slots)
    first = shown[0] if shown else {"kind": fallback_kind, "model": ""}
    kind, model = first.get("kind") or fallback_kind, first.get("model") or None
    return MultiModelAlwaysRefusal(always_refusal_message(kind, model), kind = kind, model = model)


def _evict_chat() -> None:
    import time

    from core.inference import get_inference_backend
    from core.inference.model_slots import unload_extra_models
    from routes.inference import get_llama_cpp_backend

    from core.inference.llama_cpp import chat_load_active

    unload_extra_models(strict = True)
    llama = get_llama_cpp_backend()
    # is_active, not is_loaded: a model still starting holds VRAM. chat_load_active covers HF GGUF downloads.
    if llama.is_active or chat_load_active():
        # unload_model clears the identity, so read it first for the pre-swap KV save.
        unloaded = _chat_identity(llama)
        llama.unload_model()
        _commit_swap_kv(unloaded)
    orchestrator = get_inference_backend()
    if orchestrator.active_model_name:
        unloaded = _mlx_chat_identity(orchestrator)
        orchestrator.unload_model(orchestrator.active_model_name)
        if unloaded is not None:
            _commit_swap_kv(unloaded)
    # In-flight loads lack active_model_name; cancel_load must drop the loading marker before teardown.
    for pending in list(getattr(orchestrator, "loading_models", ()) or ()):
        orchestrator.cancel_load(pending)
    # Kill the subprocess too: its base CUDA context holds VRAM diffusion needs.
    managed = getattr(orchestrator, "_managed_engine", None) is not None
    stopped = orchestrator._shutdown_subprocess(timeout = 5.0)
    if managed and stopped is False:
        raise RuntimeError(
            "The inference engine did not stop; GPU ownership cannot be transferred."
        )
    # The driver reclaims killed VRAM asynchronously; wait or a warm handoff can transiently OOM.
    llama._wait_for_vram_settle(since_kill = time.monotonic())
    from hub.services.models.account_access import clear_resident

    clear_resident(CHAT)


def _chat_identity(llama):
    try:
        from core.inference.llama_keepwarm import _loaded_identity
        return _loaded_identity(llama)
    except Exception:
        return None


def _mlx_chat_identity(orchestrator):
    try:
        from core.inference.chat_swap_kv import mlx_chat_identity
        return mlx_chat_identity(orchestrator)
    except Exception:
        return None


def _commit_swap_kv(unloaded) -> None:
    """Promote the chat KV saved before this media load; an eviction never fails on it."""
    try:
        from core.inference.chat_swap_kv import commit_after_chat_unload
        commit_after_chat_unload(unloaded)
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_arbiter: could not keep the chat KV: %s", exc)


def _evict_diffusion() -> None:
    # Unload whichever engine the router has active (diffusers or native sd.cpp).
    from core.inference.diffusion_engine_router import get_active_diffusion_engine

    # A launching upsampler warm start holds the engine's generation slot the unload waits on.
    _cancel_prompt_upsampler_warm_start()
    get_active_diffusion_engine().unload()
    _stop_prompt_upsampler()
    _release_mask_segmenter("image model evicted", handover = True)


def _cancel_prompt_upsampler_warm_start() -> None:
    module = sys.modules.get("core.inference.flux3_prompt_upsampler")
    if module is None:
        return
    try:
        module.cancel_warm_start()
    except Exception as exc:  # noqa: BLE001 - never block an eviction on this
        logger.warning("gpu_arbiter: could not cancel a prompt upsampler warm start: %s", exc)


def _stop_prompt_upsampler() -> None:
    """The FLUX.3 prompt upsampler serves the image model, so it leaves with every DIFFUSION claim: an evicted, a
    released or a replaced one. Only if Images ever started it."""
    module = sys.modules.get("core.inference.flux3_prompt_upsampler")
    if module is None:
        return
    try:
        module.stop_for_handover()
    except Exception as exc:  # noqa: BLE001 - never block an acquire on this
        logger.warning("gpu_arbiter: could not stop the FLUX.3 prompt upsampler: %s", exc)


class SegmenterStillRunningError(RuntimeError):
    """A mask from text still runs on the GPU after the release's wait: the GPU cannot change hands yet.
    ``blocks_training`` makes the core trainer abort its start on it rather than log and continue."""

    blocks_training = True


def _release_mask_segmenter(reason: str, *, handover: bool = False) -> None:
    """The Images text-to-mask segmenter (SAM 3) is auxiliary: it leaves whenever the GPU changes hands, an image model
    is evicted or released, or training starts. Only if Images ever loaded it; a call in flight finishes first.
    ``handover``: a call still on the GPU after the wait aborts the transfer, as an engine that did not stop does."""
    module = sys.modules.get("core.inference.diffusion_segment")
    if module is None:
        return
    try:
        module.release_segmenter(reason, wait = True, handover = handover)
    except Exception as exc:  # noqa: BLE001 - never block an acquire on anything else
        if handover and type(exc).__name__ == "SegmenterReleaseTimeoutError":
            raise SegmenterStillRunningError(
                "A mask from text is still running on the GPU; GPU ownership cannot be transferred."
            ) from exc
        logger.warning("gpu_arbiter: could not release the mask segmenter: %s", exc)


_handovers = 0


@contextmanager
def _handover_fence() -> Iterator[None]:
    global _handovers
    _handovers += 1
    try:
        yield
    finally:
        _handovers -= 1


def handover_in_progress() -> bool:
    """An acquire is handing the GPU to a new owner (evicting, or its load registering). Lock-free read."""
    return _handovers > 0


def release_mask_segmenter(reason: str) -> None:
    """Release the text-to-mask segmenter for a training start: an in-flight call finishes first, and one still on the
    GPU after the wait raises SegmenterStillRunningError so the start is refused instead of spawning beside it."""
    try:
        _release_mask_segmenter(reason, handover = True)
    except SegmenterStillRunningError as exc:
        raise SegmenterStillRunningError(
            "A mask from text is still running on the GPU; try again when it finishes."
        ) from exc


def _release_idle_video_servers(reason: str) -> None:
    try:
        from core.inference.video_minimax_h3 import release_h3_native_servers
        release_h3_native_servers(reason)
    except Exception as exc:  # noqa: BLE001 - never block an acquire on this
        logger.warning("gpu_arbiter: could not release the idle video sd-server: %s", exc)


def _release_idle_decision_worker() -> None:
    module = sys.modules.get("core.systemone.laya_runtime")
    if module is None:
        return
    try:
        module.evict_idle_worker()
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpu_arbiter: could not release the idle decision worker: %s", exc)


def _evict_video() -> None:
    from core.inference.video import get_video_backend
    get_video_backend().unload()


def _evict_decisions() -> None:
    from core.systemone.laya_runtime import evict_for_gpu
    evict_for_gpu()


# Patchable in tests via monkeypatch.setitem. Under the default policy acquire_for evicts every other resident, so this
# generalises to any number of owners.
_EVICTORS = {
    CHAT: _evict_chat,
    DIFFUSION: _evict_diffusion,
    VIDEO: _evict_video,
    DECISIONS: _evict_decisions,
}


def _chat_slots() -> list[dict[str, str]]:
    from routes.inference import chat_eviction_reports
    return chat_eviction_reports()


def _media_backend(owner: str) -> Any:
    if owner == DIFFUSION:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        return get_active_diffusion_engine()
    from core.inference.video import get_video_backend
    return get_video_backend()


def _media_loading(owner: str) -> bool:
    """Whether an image or video load is in flight; lock-safe under ``_lock``."""
    return bool(_media_backend(owner).loading_repo_ids())


def _chat_occupied() -> bool:
    """Whether a chat load is in flight, a kept slot failed to stop, or the primary cannot be read;
    unreadable counts as yes, so Always never evicts it blind."""
    try:
        from core.inference import model_slots
        from core.inference.llama_cpp import chat_load_active
        from routes.inference import _primary_chat_reports, get_llama_cpp_backend

        if chat_load_active() or model_slots.any_loading() or model_slots.stuck:
            return True
        model_slots.in_slot(None, _primary_chat_reports)
        # A llama-server mid-respawn is active but not loaded, so no report lists it.
        return bool(
            model_slots.in_slot(None, lambda: get_llama_cpp_backend().is_active)
            or any(slot.llama.is_active for slot in list(model_slots.slots))
        )
    except Exception:  # noqa: BLE001
        return True


def _media_occupied(owner: str) -> bool:
    """Whether an image or video model is loaded or loading. One that cannot be read counts as
    loaded, so Always never evicts it blind."""
    try:
        return _media_status(owner) is not None or _media_loading(owner)
    except Exception:  # noqa: BLE001
        return True


def _media_status(owner: str) -> Optional[dict]:
    """The loaded image or video model's status, or None when none is loaded."""
    backend = _media_backend(owner)
    status = backend.status() if hasattr(backend, "status") else None
    return status if isinstance(status, dict) and status.get("loaded") else None


def _diffusion_slots() -> list[dict[str, str]]:
    return _media_slot("image", DIFFUSION, _media_status(DIFFUSION))


def _video_slots() -> list[dict[str, str]]:
    return _media_slot("video", VIDEO, _media_status(VIDEO))


def refuse_replacing_media(
    owner: str, policy: str, requested: Iterable[Optional[str]]
) -> list[dict[str, Any]]:
    """The image or video model of its own kind a load of ``requested`` would replace. Under Always
    that is refused instead, as is replacing a load of another repo still in flight. Another file or
    setting of the same repo is allowed, as for chat. ``requested`` are the model paths the load
    reads, never a caller's display label, which nothing checks against the path. Safe under
    ``_lock``."""
    if owner not in (DIFFUSION, VIDEO):
        return []
    from core.inference.model_ids import hf_cache_repo_id

    # Only an HF cache snapshot path names its repo; any other local path is its own identity.
    def aliases(name: Optional[str]) -> set:
        return {alias for alias in (name, hf_cache_repo_id(name) if name else None) if alias}

    names = {alias for name in requested for alias in aliases(name)}
    try:
        status = _media_status(owner)
        loading = () if status is not None else _media_backend(owner).loading_repo_ids()
    except Exception:  # noqa: BLE001 - unreadable: Always keeps it, Off replaces it as before
        if policy == "always":
            raise refusal_for([], KIND_LABELS[owner])
        return []
    if status is not None:
        repo = status.get("repo_id")
        loaded = aliases(repo)
        if not loaded.isdisjoint(names):
            return []
        slots = resident_slots(owner)
        if policy == "always":
            raise refusal_for(slots, KIND_LABELS[owner])
        return slots
    # The in-flight load's own repo comes first; the rest are companions it reads, not its identity.
    if policy == "always" and loading and names.isdisjoint(aliases(loading[0])):
        raise refusal_for([], KIND_LABELS[owner])
    return []


def guarded_media_begin(
    owner: str, policy: str, requested: Iterable[Optional[str]], begin: Callable[[], Any]
) -> Callable[[], Any]:
    """``begin`` re-checked against a load that registered meanwhile, for use as an acquire's
    ``register`` (which runs under ``_lock``). Off and fit begin as before."""
    if policy != "always":
        return begin
    names = tuple(requested)

    def register() -> Any:
        refuse_replacing_media(owner, policy, names)
        return begin()

    return register


def begin_media_load(
    owner: str, policy: str, requested: Iterable[Optional[str]], begin: Callable[[], Any]
) -> Any:
    """``guarded_media_begin`` for a load that takes no GPU claim, under ``_lock`` all the same."""
    register = guarded_media_begin(owner, policy, requested, begin)
    with _lock:
        # This load reprices nothing, so an older load's rollback must not apply to its failure.
        _media_need_rollback.pop(owner, None)
    if register is begin:
        return begin()
    with _lock:
        return register()


def _media_slot(kind: str, owner: str, status: Optional[dict]) -> list[dict[str, str]]:
    if status is None:
        return []
    from core.inference.model_ids import public_model_id
    from hub.services.models.account_access import _resident_loader

    # display_repo_id is what the page shows; a bare repo_id can be a local path.
    model = status.get("display_repo_id") or public_model_id(status.get("repo_id")) or kind
    return [{"kind": kind, "model": str(model), "account": _resident_loader(owner)}]


# What an owner's eviction unloads, as {kind, model} with the public model id, plus the loading account under
# "account". Patchable like _EVICTORS. Read before the evictor runs, under _lock, so a reporter must never call back
# into the arbiter's locking functions. One that fails reports nothing rather than blocking the acquire.
_SLOT_REPORTERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    CHAT: _chat_slots,
    DIFFUSION: _diffusion_slots,
    VIDEO: _video_slots,
}

# Where an acquire without an explicit ``evicted`` list reports, so a route can collect what a nested load evicted.
eviction_sink: ContextVar[Optional[list]] = ContextVar("gpu_eviction_sink", default = None)


def resident_slots(owner: str) -> list[dict[str, Any]]:
    reporter = _SLOT_REPORTERS.get(owner)
    if reporter is None:
        return []
    try:
        return [dict(slot) for slot in reporter() or ()]
    except Exception as exc:  # noqa: BLE001 - a report must never block an acquire
        logger.warning("gpu_arbiter: could not list the %s models: %s", owner, exc)
        return []


def public_slots(slots: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """``{kind, model}`` for a caller: a managed account never sees what another account loaded."""
    try:
        from hub.services.models.account_access import managed_account
        from utils.account_context import current_account_id
        hide, me = managed_account(), current_account_id()
    except Exception:  # noqa: BLE001
        hide, me = False, None
    out = []
    for slot in slots:
        account = slot.get("account")
        model = (
            "" if hide and account is not None and account != me else str(slot.get("model") or "")
        )
        out.append({"kind": str(slot.get("kind") or "chat"), "model": model})
    return out


def note_evictions(slots: Iterable[dict[str, Any]], evicted: Optional[list] = None) -> None:
    """Report models a route unloaded itself, to ``evicted`` or the current eviction sink."""
    sink = evicted if evicted is not None else eviction_sink.get()
    if sink is not None:
        sink.extend(public_slots(slots))


def policy_kwargs(policy: str) -> dict[str, str]:
    """``acquire_for`` kwargs for a mode: none for "off", so callers and fakes see today's call."""
    return {} if policy == "off" else {"policy": policy}


def physical_gpu_ids(visible: Optional[Iterable[int]]) -> Optional[frozenset[int]]:
    """Map visible device ordinals to physical GPU ids (CUDA_VISIBLE_DEVICES=2,3 makes ordinal 0 GPU 2)."""
    if visible is None:
        return None
    ordinals = [int(i) for i in visible]
    try:
        from utils.hardware.hardware import get_parent_visible_gpu_ids
        parent = get_parent_visible_gpu_ids()
    except Exception:  # noqa: BLE001
        parent = []
    if not parent:
        return frozenset(ordinals)
    return frozenset(parent[i] if 0 <= i < len(parent) else i for i in ordinals)


def _view(owner: Any = ...) -> dict[str, Optional[frozenset[int]]]:
    """Residents, trusting ``_owner`` when a caller or test set it without going through the arbiter."""
    if owner is ...:
        owner = _owner
    if owner is None:
        return {}
    if owner not in _residents:
        return {owner: None}
    return dict(_residents)


def _sync_from_view() -> None:
    """Bring ``_residents`` and ``_accounts`` in line with ``_owner`` before a mutation. Caller holds ``_lock``."""
    global _residents
    view = _view()
    if view != _residents:
        _residents = view
    for stale in [name for name in _accounts if name not in view or name == _owner]:
        _accounts.pop(stale, None)


def _drop_resident(owner: str) -> None:
    """Remove ``owner`` and recompute ``_owner`` from the residents left. Caller holds ``_lock``."""
    global _owner, _owner_epoch, _owner_account, _prior_account
    _sync_from_view()
    if owner not in _residents:
        return
    del _residents[owner]
    _media_needs.pop(owner, None)
    _media_need_rollback.pop(owner, None)
    if owner == _owner:
        if _residents:
            _owner = next(reversed(_residents))
            _owner_account, _prior_account = _accounts.pop(_owner, (None, None))
        else:
            _owner = None
            _owner_account = None
            _prior_account = None
    else:
        _accounts.pop(owner, None)
    _owner_epoch += 1


def holds(owner: str) -> bool:
    """Whether ``owner`` has a model resident on a GPU. Lock-free, so register callbacks may call it."""
    return owner in _view(current_owner())


def residents() -> dict[str, Optional[frozenset[int]]]:
    """Every resident owner and its physical GPU ids (None when unknown). Lock-free like ``holds``."""
    return _view(current_owner())


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
    from core.inference.gpu_residency import ChatLoadInFlightRefusal
    try:
        return acquire_for(owner, register, **kwargs)
    except (GpuBusyForAnotherAccountError, ChatLoadInFlightRefusal) as exc:
        raise exc.as_http_exception() from exc


def acquire_for(
    owner: str,
    register: Optional[Callable[[], Any]] = None,
    *,
    expected_current: Optional[tuple[Optional[str], int]] = None,
    allow_evict: bool = True,
    account_id: Optional[str] = None,
    replacing: bool = False,
    alongside: bool = False,
    policy: str = "off",
    evicted: Optional[list] = None,
    gpu_ids: Optional[Iterable[int]] = None,
    need_mib: Optional[dict] = None,
    evict_first: bool = False,
    admit: Optional[Callable[[], None]] = None,
) -> Any:
    """Make ``owner`` resident, evicting every other resident under the default policy.

    ``register`` runs under the arbiter lock as residency transfers, so a competing acquire cannot
    evict this owner and let both loaders allocate VRAM at once.

    ``policy`` is the Keep multiple models loaded mode, read by the caller outside the lock. "off"
    and "fit" evict as before; "always" raises ``MultiModelAlwaysRefusal`` instead of evicting a
    chat, image or video model. ``alongside`` is unrelated: it only decides whether the owner
    account is re-claimed. ``evicted`` collects ``{kind, model}`` for each model an evictor
    unloads. ``gpu_ids`` are the visible device ordinals the owner will use, when known.

    Under "fit" and "always" only residents on the same GPUs can be evicted (gpu_residency.coreside_locked):
    ``need_mib`` is what an image or video load will hold per physical GPU (None: unknown, so it evicts first),
    and ``evict_first`` makes a chat load with no estimate before launch evict as "off" does.
    ``admit`` runs under the lock before anything is evicted; raising refuses the acquire untouched.
    """
    global _owner, _owner_epoch, _owner_account, _prior_account
    if owner not in _EVICTORS:
        raise ValueError(f"unknown GPU owner: {owner!r}")
    from utils.account_context import current_account_id

    acting = account_id or current_account_id()
    # The segmenter's VRAM goes to this owner: a mask call arriving before its register returns stays off the card.
    handover = owner != DIFFUSION or register is not None or replacing
    with _lock, _handover_fence() if handover else nullcontext():
        if admit is not None:
            admit()
        if expected_current is not None and (_owner, _owner_epoch) != expected_current:
            raise OwnerChangedError("The resident GPU model changed; retry the load.")
        coreside = policy != "off" and owner in _KEPT_UNDER_ALWAYS
        if (register is not None or replacing) and not coreside:
            raise_if_other_accounts_active(acting)
        elif (register is not None or replacing) and not (owner == CHAT and alongside):
            # Only another account's work on this model blocks reloading it; residents it would evict are
            # checked one by one.
            from core.inference.gpu_residency import foreign_on
            busy = foreign_on(owner, acting)
            if busy:
                raise GpuBusyForAnotherAccountError(owner, busy)
        _sync_from_view()
        prev_owner, prev_pair = _owner, (_owner_account, _prior_account)
        # Read before evicting: dropping the current owner promotes a parked resident and pops its accounts.
        parked = owner in _accounts
        others = [name for name in _residents if name != owner]
        if others and coreside:
            from core.inference.gpu_residency import coreside_locked
            coreside_locked(
                owner,
                policy = policy,
                need_mib = need_mib,
                gpu_ids = gpu_ids,
                allow_evict = allow_evict,
                evict_first = evict_first,
                acting = acting,
                evicted = evicted,
            )
            others = []
        if others:
            if not allow_evict:
                raise GpuOwnerBusyError(others[-1])
            if policy == "always" and owner in _KEPT_UNDER_ALWAYS:
                kept = [name for name in others if name in _KEPT_UNDER_ALWAYS]
                slots = [slot for name in kept for slot in resident_slots(name)]
                # A claim whose load failed holds nothing: evicting it unloads nothing.
                if slots or any(
                    _chat_occupied() if name == CHAT else _media_occupied(name) for name in kept
                ):
                    raise refusal_for(slots, KIND_LABELS[kept[0]])
            # Never evict an account mid-generation; the caller retries after its stream ends.
            busy = other_accounts_active(acting)
            if busy:
                raise GpuBusyForAnotherAccountError(others[-1], busy)
            sink = evicted if evicted is not None else eviction_sink.get()
            for name in others:
                logger.info("gpu_arbiter: evicting %s for %s", name, owner)
                unloading = resident_slots(name) if sink is not None else []
                _EVICTORS[name]()
                note_evictions(unloading, sink)
                _drop_resident(name)
        if owner != VIDEO and not (coreside and VIDEO in _residents):
            # A resident H3 sd-server sits outside every owner's teardown.
            _release_idle_video_servers(f"GPU acquired for {owner}")
        if owner != DECISIONS and (register is not None or replacing) and policy == "off":
            _release_idle_decision_worker()
        if owner != DIFFUSION or register is not None or replacing:
            # A load (or another kind taking the GPU) gets the segmenter's VRAM; an image model's re-assert keeps it.
            _release_mask_segmenter(f"GPU acquired for {owner}", handover = True)
        # Records who LOADED the model; a plain re-assert must not hand it to whoever asked last.
        # A resident parked behind a later claimant is a re-assert, not a new load.
        rejoins = prev_owner != owner and owner in _residents and parked
        claims = (prev_owner != owner and not rejoins) or (
            (register is not None or replacing) and not alongside
        )
        if owner != _owner:
            if _owner is not None:
                # Another resident stays loaded; park its accounts while ``owner`` is the last claimant.
                _accounts[_owner] = (_owner_account, _prior_account)
            if owner in _accounts:
                _owner_account, _prior_account = _accounts.pop(owner)
            elif prev_owner is None or prev_owner not in _residents:
                # The evicted owner's accounts carry over, as they did with a single owner.
                _owner_account, _prior_account = prev_pair
            else:
                _owner_account, _prior_account = None, None
        _owner = owner
        held_before = _residents.pop(owner, ...)
        known = physical_gpu_ids(gpu_ids)
        gpus_now = known if known is not None or not need_mib else frozenset(need_mib)
        if owner == CHAT and held_before is not ...:
            from core.inference import model_slots
            if register is None and not replacing and gpus_now is None:
                # A re-assert places nothing new.
                gpus_now = held_before
            elif alongside or model_slots.holds_vram():
                # Every chat model kept loaded counts, not just the newest; unknown (None) means every GPU.
                gpus_now = (
                    None if held_before is None or gpus_now is None else held_before | gpus_now
                )
        _residents[owner] = gpus_now
        _owner_epoch += 1
        try:
            result = register() if register is not None else None
        except BaseException:
            if held_before is not ...:
                # The model already loaded stays where it was.
                _residents[owner] = held_before
            raise
        if owner in (DIFFUSION, VIDEO) and register is not None:
            # The price media_held_mib() reports for this resident; unpriced residents are evicted to make room.
            _media_need_rollback[owner] = _media_needs.get(owner)
            if coreside and need_mib:
                _media_needs[owner] = need_mib
            else:
                _media_needs.pop(owner, None)
        # A raising registration loaded nothing and must not take residency.
        if claims:
            _prior_account, _owner_account = _owner_account, acting
        return result


def would_evict(
    owner: str,
    *,
    policy: str = "off",
    need_mib: Optional[dict] = None,
    gpu_ids: Optional[Iterable[int]] = None,
) -> bool:
    """Lock-free peek: will ``acquire_for(owner, policy = ..., need_mib = ..., gpu_ids = ...)`` unload
    the chat model? Never call it under ``_lock``."""
    if owner == CHAT or owner not in _EVICTORS:
        return False
    holds = globals().get("holds")
    if not (holds(CHAT) if callable(holds) else _owner == CHAT):
        return False
    if policy in ("fit", "always"):
        # Same target as gpu_residency.coreside_locked: chat on other GPUs is never touched.
        target = (
            physical_gpu_ids(gpu_ids)
            if gpu_ids is not None
            else (frozenset(need_mib) if need_mib else None)
        )
        chat_gpus = _view().get(CHAT)
        if chat_gpus is not None and target is not None and not set(chat_gpus) & set(target):
            return False
    if policy == "always":
        # Always refuses the load rather than evicting, except a claim whose load failed and holds nothing.
        try:
            from core.inference.gpu_residency import _claim_holds_anything
            return not _claim_holds_anything(CHAT)
        except Exception:  # noqa: BLE001
            return False
    if policy == "fit" and need_mib:
        # An unpriced load evicts first, as Off does; a priced one joins when it fits.
        try:
            from core.inference import gpu_residency
            fits = getattr(gpu_residency, "can_load_alongside", None)
        except ImportError:
            gpu_residency, fits = None, None
        if callable(fits):
            in_flight = getattr(gpu_residency, "chat_load_in_flight", None)
            if callable(in_flight) and in_flight():
                return True  # coreside_locked evicts a chat that is still loading
            unpriced = getattr(gpu_residency, "chat_growth_unpriced", None)
            if callable(unpriced) and unpriced():
                return True
            answer = fits(owner, need_mib)
            ok, info = answer if isinstance(answer, tuple) else (answer, {})
            return not (ok or _idle_media_frees_room(gpu_residency, owner, info))
    return True


def _idle_media_frees_room(gpu_residency, owner: str, info: dict) -> bool:
    """When they fit frees in gpu_residency._free_steps order and stops once the load fits: the
    idle image or video model on these GPUs goes before the active chat model. Predict that step
    with a margin, since the live re-check may find less freed than was priced; when unsure, save."""
    try:
        short = dict((info or {}).get("short_mib") or {})
        if not short or info.get("reason") != "short":
            return False
        from utils.account_context import current_account_id

        acting = current_account_id()
        for name, need in gpu_residency._media_records().items():
            if name == owner or any(gpu_residency._media_busy(name)):
                continue
            if gpu_residency.foreign_on(name, acting):
                continue
            for gpu in short:
                short[gpu] -= int(need.get(gpu, 0))
        margin = int(getattr(gpu_residency, "FIT_MARGIN_MIB", 1024))
        return all(left <= -margin for left in short.values())
    except Exception:  # noqa: BLE001
        return False


def restore_owner_account(owner: str, account_id: Optional[str] = None) -> bool:
    """Hand residency back to the displaced account when a claim's load never committed."""
    global _owner_account, _prior_account
    from utils.account_context import current_account_id

    with _lock:
        acting = account_id or current_account_id()
        if owner == _owner:
            if _owner_account != acting or _prior_account is None:
                return False
            _owner_account, _prior_account = _prior_account, None
            return True
        account, prior = _accounts.get(owner, (None, None))
        if owner not in _view() or account != acting or prior is None:
            return False
        _accounts[owner] = (prior, None)
        return True


def restore_media_need(owner: str) -> None:
    """Put back the price the newest load of ``owner`` overwrote, when that load failed and the
    model it was replacing stays loaded; otherwise the kept model is priced as the failed one."""
    with _lock:
        if owner not in _media_need_rollback:
            return
        previous = _media_need_rollback.pop(owner)
        if previous is None:
            _media_needs.pop(owner, None)
        else:
            _media_needs[owner] = previous


def release(owner: str) -> None:
    """Drop ``owner``'s residency (no-op if it isn't resident)."""
    with _lock:
        if owner == DIFFUSION and owner in _view():
            _stop_prompt_upsampler()
        if owner == DIFFUSION:
            # Training releases DIFFUSION before it starts, resident or not: the segmenter goes too.
            _release_mask_segmenter("image GPU released")
        _drop_resident(owner)


def release_if(owner: str, predicate: Callable[[], bool]) -> bool:
    """Drop ``owner``'s residency only if it still holds it AND ``predicate()`` is true, atomically.

    A slow unload's idle check and its ``release`` must not straddle a concurrent same-owner load
    whose ``acquire_for(register=...)`` re-registers ownership under this lock; evaluating the
    predicate under the lock keeps them atomic so ``release`` never clears the newer claim.
    ``predicate`` must be quick and not re-enter the arbiter. Returns True iff ownership was dropped."""
    with _lock:
        if owner not in _view() or not predicate():
            return False
        if owner == DIFFUSION:
            _stop_prompt_upsampler()
            _release_mask_segmenter("image GPU released")
        _drop_resident(owner)
        return True


def current_owner() -> Optional[str]:
    """The last claimant. Prefer ``holds()`` / ``residents()``: other models may be resident too."""
    return _owner


def owner_account(owner: Optional[str] = None) -> Optional[str]:
    """Account whose load put ``owner`` (default: the last claimant) on the GPU."""
    if owner is None or owner == _owner:
        return _owner_account
    return _accounts.get(owner, (None, None))[0]


def owner_snapshot() -> tuple[Optional[str], int]:
    with _lock:
        return _owner, _owner_epoch

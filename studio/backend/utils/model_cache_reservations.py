# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

import anyio
from fastapi import HTTPException

from utils.hf_repo_ids import hf_cache_repo_id, is_valid_repo_id


_T = TypeVar("_T")
_MAX_INFERENCE_LOAD_EPOCHS = 4096


def _default_variant_scopes_overlap(first: Optional[str], second: Optional[str]) -> bool:
    return first is None or second is None or first == second


class ModelCacheOperations:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._deleting: set[str] = set()
        # repo -> {owner: the variant that owner reads, None for the whole repo}.
        self._inference_loads: dict[str, dict[object, Optional[str]]] = {}
        self._inference_load_repos: dict[object, set[str]] = {}
        self._inference_load_epochs: OrderedDict[tuple[str, Optional[str]], int] = OrderedDict()
        self._inference_load_epoch_sequence = 0
        self._inference_load_epoch_floor = 0
        self._repository_owners: dict[str, object] = {}
        # repo -> watcher cleanup owners. A worker remains a cache writer while its
        # terminal watcher persists markers and removes abandoned partials, even though
        # the user-facing job state is already complete/cancelled/error.
        self._writer_cleanups: dict[str, set[object]] = {}
        self._active_writer_state: Optional[Callable[[str, Optional[str]], Optional[str]]] = None
        self._delete_blocked_by_active: Optional[Callable[[str, Optional[str]], bool]] = None
        self._variant_scopes_overlap: Callable[[Optional[str], Optional[str]], bool] = (
            _default_variant_scopes_overlap
        )

    @staticmethod
    def _repo_key(repo_id: str) -> str:
        return repo_id.strip().lower()

    def bind(
        self,
        active_writer_state: Callable[[str, Optional[str]], Optional[str]],
        delete_blocked_by_active: Callable[[str, Optional[str]], bool],
        variant_scopes_overlap: Callable[[Optional[str], Optional[str]], bool],
    ) -> None:
        with self.lock:
            self._active_writer_state = active_writer_state
            self._delete_blocked_by_active = delete_blocked_by_active
            self._variant_scopes_overlap = variant_scopes_overlap

    def cache_writer_conflict(
        self,
        repo_id: str,
        variant: Optional[str] = None,
    ) -> Optional[str]:
        """Why a writer of *repo_id*/*variant* cannot start right now, or None.

        The overlap rule the download registry already applies to its own jobs and to
        deletion: a whole-repo scope meets any other scope, a variant scope meets only the
        same checkpoint identity or alias. So a load reading one quantization leaves its siblings
        downloadable -- which is the promise :meth:`claim_inference_load` makes when it
        takes a variant at all.
        """
        repo = self._repo_key(repo_id)
        variant_key = (variant or "").strip().lower() or None
        with self.lock:
            if repo in self._repository_owners or self._writer_cleanups.get(repo):
                return "repository_owned"
            held_scopes = self._inference_loads.get(repo)
            if held_scopes and any(
                self._variant_scopes_overlap(held, variant_key) for held in held_scopes.values()
            ):
                return "inference_loading"
        return None

    def deletion_conflicts(self, repo_id: str) -> bool:
        """Whether a running delete owns the repository a writer would mutate."""
        repo = self._repo_key(repo_id)
        with self.lock:
            return repo in self._deleting

    def delete_admission_conflict(self, repo_id: str) -> Optional[str]:
        """Why a delete of *repo_id* cannot start right now, or None."""
        repo = self._repo_key(repo_id)
        with self.lock:
            return self._delete_admission_conflict_locked(repo)

    def _delete_admission_conflict_locked(self, repo: str) -> Optional[str]:
        if repo in self._repository_owners:
            return "repository_owned"
        # Whole-repo on purpose, unlike cache_writer_conflict's per-quant answer above: a
        # variant delete also unlinks the shared companions (mmproj, drafter) that a load of
        # any sibling quantization is reading, so no quant label bounds what it removes.
        if self._inference_loads.get(repo):
            return "inference_loading"
        if repo in self._deleting:
            return "deleting"
        if self._writer_cleanups.get(repo):
            return "downloading"
        if self._delete_blocked_by_active is not None and self._delete_blocked_by_active(
            repo, None
        ):
            return "downloading"
        return None

    def claim_repository_owner(self, repo_id: str, owner: object) -> tuple[bool, str]:
        repo = self._repo_key(repo_id)
        with self.lock:
            if repo in self._repository_owners:
                return False, "repository_owned"
            if self._writer_cleanups.get(repo):
                return False, "downloading"
            if self._inference_loads.get(repo):
                return False, "inference_loading"
            if repo in self._deleting:
                return False, "deleting"
            if self._active_writer_state is not None:
                state = self._active_writer_state(repo, None)
                if state is not None:
                    return False, state
            self._repository_owners[repo] = owner
        return True, "owned"

    def claim_inference_load(
        self,
        repo_ids: Sequence[str],
        owner: object,
        variant: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        """Publish *owner*'s hold on *repo_ids*, or report the first conflict.

        *variant* is the GGUF quantization this load reads, when it reads exactly one:
        a download of a sibling quant writes different blobs, so neither may hold the other
        off. A broader later claim widens an existing hold only after rechecking the newly
        covered scope under the registry lock.
        """
        variant_key = (variant or "").strip().lower() or None
        repos = tuple(
            dict.fromkeys(
                repo for repo in (self._repo_key(repo_id) for repo_id in repo_ids) if repo
            )
        )
        if not repos:
            return None
        with self.lock:
            owned = self._inference_load_repos.get(owner)
            scopes: dict[str, Optional[str]] = {}
            for repo in repos:
                owners = self._inference_loads.get(repo)
                if owners is None or owner not in owners:
                    scopes[repo] = variant_key
                    continue
                held = owners[owner]
                if held is not None and (
                    variant_key is None or not self._variant_scopes_overlap(held, variant_key)
                ):
                    scopes[repo] = None
            if not scopes:
                return None
            conflict = next((repo for repo in scopes if repo in self._deleting), None)
            if conflict is not None:
                return conflict, "deleting"
            conflict = next(
                (repo for repo in scopes if repo in self._repository_owners),
                None,
            )
            if conflict is not None:
                return conflict, "repository_owned"
            # A terminal worker still sweeping partials and markers is a live writer, even
            # though its job already reads as finished, so the active-writer callback below
            # reports nothing. Every other admission path already refuses on this hold.
            conflict = next((repo for repo in scopes if self._writer_cleanups.get(repo)), None)
            if conflict is not None:
                return conflict, "downloading"
            if self._active_writer_state is not None:
                for repo, scope in scopes.items():
                    state = self._active_writer_state(repo, scope)
                    if state is not None:
                        return repo, state
            if owned is None:
                owned = self._inference_load_repos.setdefault(owner, set())
            for repo, scope in scopes.items():
                self._inference_loads.setdefault(repo, {})[owner] = scope
                owned.add(repo)
                self._record_inference_load_epoch_locked(repo, scope)
        return None

    def _record_inference_load_epoch_locked(self, repo: str, scope: Optional[str]) -> None:
        self._inference_load_epoch_sequence += 1
        key = (repo, scope)
        self._inference_load_epochs[key] = self._inference_load_epoch_sequence
        self._inference_load_epochs.move_to_end(key)
        self._prune_inference_load_epochs_locked()

    def _prune_inference_load_epochs_locked(self) -> None:
        while len(self._inference_load_epochs) > _MAX_INFERENCE_LOAD_EPOCHS:
            _, epoch = self._inference_load_epochs.popitem(last = False)
            self._inference_load_epoch_floor = max(self._inference_load_epoch_floor, epoch)

    def inference_load_epoch(
        self,
        repo_id: str,
        variant: Optional[str] = None,
    ) -> int:
        repo = self._repo_key(repo_id)
        variant_key = (variant or "").strip().lower() or None
        with self.lock:
            key = (repo, variant_key)
            if key not in self._inference_load_epochs:
                self._inference_load_epochs[key] = self._inference_load_epoch_floor
                self._inference_load_epochs.move_to_end(key, last = False)
                self._prune_inference_load_epochs_locked()
            latest = self._inference_load_epoch_floor
            for (epoch_repo, scope), epoch in self._inference_load_epochs.items():
                if epoch_repo == repo and self._variant_scopes_overlap(scope, variant_key):
                    latest = max(latest, epoch)
            return latest

    def release_inference_load(self, owner: object) -> None:
        with self.lock:
            repos = self._inference_load_repos.pop(owner, set())
            for repo in repos:
                owners = self._inference_loads.get(repo)
                if owners is None or owner not in owners:
                    continue
                scope = owners.pop(owner)
                self._record_inference_load_epoch_locked(repo, scope)
                if not owners:
                    self._inference_loads.pop(repo, None)

    def release_repository_owner(self, repo_id: str, owner: object) -> bool:
        repo = self._repo_key(repo_id)
        with self.lock:
            if self._repository_owners.get(repo) is not owner:
                return False
            self._repository_owners.pop(repo, None)
        return True

    def begin_writer_cleanup(self, repo_id: str, owner: object) -> bool:
        """Keep terminal download cleanup mutually exclusive with writers/deletes.

        The caller establishes this hold while its job is still active, under the same
        lock used by delete and writer admission. Multiple sibling-quant watchers may
        clean up together; peer blob protection still bounds what each one removes.
        """
        repo = self._repo_key(repo_id)
        with self.lock:
            if repo in self._deleting or repo in self._repository_owners:
                return False
            self._writer_cleanups.setdefault(repo, set()).add(owner)
        return True

    def end_writer_cleanup(self, repo_id: str, owner: object) -> None:
        repo = self._repo_key(repo_id)
        with self.lock:
            owners = self._writer_cleanups.get(repo)
            if owners is None:
                return
            owners.discard(owner)
            if not owners:
                self._writer_cleanups.pop(repo, None)

    def begin_delete(self, repo_id: str) -> Optional[str]:
        """Reserve the repository for deletion under the cache-operation lock."""
        repo = self._repo_key(repo_id)
        with self.lock:
            conflict = self._delete_admission_conflict_locked(repo)
            if conflict is not None:
                return conflict
            self._deleting.add(repo)
        return None

    def end_delete(self, repo_id: str) -> None:
        repo = self._repo_key(repo_id)
        with self.lock:
            self._deleting.discard(repo)


_model_cache_operations = ModelCacheOperations()


def get_model_cache_operations() -> ModelCacheOperations:
    return _model_cache_operations


def _cache_repo_id(identifier: Optional[str]) -> Optional[str]:
    value = (identifier or "").strip()
    if not value:
        return None
    # A cache path names its repo exactly; only a bare request identifier gets the
    # loader's unsloth/ shorthand (ModelConfig.from_identifier), or a namespace-less Hub
    # repo such as bert-base-uncased would reserve a repo that does not exist.
    cached_repo_id = hf_cache_repo_id(value)
    if cached_repo_id is not None:
        return cached_repo_id if is_valid_repo_id(cached_repo_id) else None
    repo_id = value
    if not is_valid_repo_id(repo_id):
        return None
    try:
        if Path(repo_id).expanduser().exists():
            return None
    except OSError:
        return None
    if "/" not in repo_id:
        repo_id = f"unsloth/{repo_id}"
    return repo_id


class CacheReservationConflict(HTTPException):
    """409 for a cache scope another operation already holds.

    ``str()`` is the detail alone: a media load worker stamps whatever ended it into
    load-progress verbatim, and Starlette's ``__str__`` would put ``409: `` in front of
    the sentence shown there.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(status_code = 409, detail = detail)

    def __str__(self) -> str:
        return str(self.detail)


class InferenceLoadReservation:
    """Cache repositories one inference load is reading, held until it stops reading them.

    *variant* narrows the primary model's identities to a single GGUF quantization, the only
    scope they can be precise about; dependencies discovered later (LoRA base, audio codec,
    draft model) are claimed whole, as a load pulls all of them.
    """

    def __init__(
        self,
        *identifiers: Optional[str],
        variant: Optional[str] = None,
    ) -> None:
        self._owner = object()
        self._released = False
        self._primary_variant = variant
        self._claim(identifiers, variant)

    def add(self, *identifiers: Optional[str]) -> None:
        self._claim(identifiers, None)

    def add_primary(self, *identifiers: Optional[str]) -> None:
        self._claim(identifiers, self._primary_variant)

    def _claim(self, identifiers: Sequence[Optional[str]], variant: Optional[str]) -> None:
        if self._released:
            raise RuntimeError("Inference load reservation is already released")
        repo_ids = tuple(
            dict.fromkeys(
                repo_id
                for repo_id in (_cache_repo_id(value) for value in identifiers)
                if repo_id is not None
            )
        )
        conflict = get_model_cache_operations().claim_inference_load(repo_ids, self._owner, variant)
        if conflict is not None:
            conflict_key, conflict_state = conflict
            # The conflict names the internal lowercased key; show the caller's spelling.
            repo_id = next(
                (value for value in repo_ids if value.strip().lower() == conflict_key),
                conflict_key,
            )
            detail = (
                f"Cached model '{repo_id}' is being deleted. Wait for deletion to finish."
                if conflict_state == "deleting"
                else (
                    f"Cache files for '{repo_id}' are busy. "
                    "Wait for the active operation to finish."
                )
            )
            raise CacheReservationConflict(detail)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        get_model_cache_operations().release_inference_load(self._owner)


def reserve_inference_load(
    *identifiers: Optional[str], variant: Optional[str] = None
) -> InferenceLoadReservation:
    return InferenceLoadReservation(*identifiers, variant = variant)


async def wait_for_reserved_worker(work: Awaitable[_T]) -> _T:
    """Await *work* to completion, deferring cancellation until its worker has exited.

    A cache reservation is released by the coroutine that took it, so a client disconnect
    must not unwind past a worker still reading or unlinking those files. The cancellation
    is re-raised once the worker is done, so the caller still sees it.
    """
    worker = asyncio.ensure_future(work)
    cancellation: Optional[asyncio.CancelledError] = None
    while not worker.done():
        try:
            with anyio.CancelScope(shield = True):
                await asyncio.shield(worker)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            break
    if cancellation is not None:
        if not worker.cancelled():
            worker.exception()
        raise cancellation
    if not worker.cancelled():
        worker.exception()
    await anyio.lowlevel.checkpoint()
    return worker.result()


def run_reserved_inference_load(
    reservation: InferenceLoadReservation, target: Callable[..., Any], /, **kwargs
) -> None:
    try:
        target(**kwargs)
    finally:
        reservation.release()

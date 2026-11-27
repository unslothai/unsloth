# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One sd-server per card, started only once a request waits on a busy primary. Off: ``UNSLOTH_SD_CPP_REPLICAS=off``."""

from __future__ import annotations

import collections
import dataclasses
import os
import threading
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

REPLICAS_ENV = "UNSLOTH_SD_CPP_REPLICAS"
# Host RAM an sd-server process holds beside its weights (measured 1.3-1.5 GB anon per server on Kaggle T4).
PROCESS_HOST_MIB = 1536

_OFF = ("0", "1", "off", "false", "no", "disable", "disabled", "none")
_AUTO = ("", "auto", "on", "true", "yes")

PRIMARY = "primary"


def replica_setting() -> tuple[str, Optional[int]]:
    """``("off", None)``, ``("auto", None)`` (one server per usable card) or ``("count", N)`` (at most N servers)."""
    raw = (os.environ.get(REPLICAS_ENV) or "").strip().lower()
    if raw in _OFF:
        return "off", None
    if raw in _AUTO:
        return "auto", None
    try:
        count = int(raw)
    except ValueError:
        logger.warning("%s=%r is not off, auto or a server count; using auto", REPLICAS_ENV, raw)
        return "auto", None
    return ("off", None) if count <= 1 else ("count", count)


def host_spill_mib(
    diffusion_mib: int, text_encoder_mib: int, vae_mib: int, limit_mib: int
) -> Optional[int]:
    """MiB auto-fit leaves on the host for a one-card load; None when the diffusion model does not fit the card."""
    from .sd_cpp_multigpu import DENOISE_HEADROOM_MIB

    if diffusion_mib <= 0 or diffusion_mib + DENOISE_HEADROOM_MIB > limit_mib:
        return None
    if diffusion_mib + text_encoder_mib + vae_mib + DENOISE_HEADROOM_MIB <= limit_mib:
        return 0
    if diffusion_mib + vae_mib + DENOISE_HEADROOM_MIB <= limit_mib:
        return text_encoder_mib
    return text_encoder_mib + vae_mib


def replica_host_mib(spill_mib: int, mmap: bool) -> int:
    """--mmap shares the spill via the page cache; without it the spill is pinned per server."""
    return PROCESS_HOST_MIB + (0 if mmap else int(spill_mib))


@dataclasses.dataclass(frozen = True)
class ReplicaPrice:
    ordinal: int
    fits: bool
    mmap: bool = False
    spill_mib: int = 0
    host_mib: int = 0
    reason: str = ""


def price_replica(
    *,
    ordinal: int,
    limit_mib: Optional[int],
    diffusion_mib: int,
    text_encoder_mib: int,
    vae_mib: int,
    host_room_mib: Optional[int],
) -> ReplicaPrice:
    """Pinned if the host holds the spill, else ``--mmap``, else refused."""
    card = f"cuda:{ordinal}"
    if limit_mib is None:
        return ReplicaPrice(ordinal, False, reason = f"free memory of {card} is unknown")
    spill = host_spill_mib(diffusion_mib, text_encoder_mib, vae_mib, int(limit_mib))
    if spill is None:
        return ReplicaPrice(
            ordinal,
            False,
            reason = f"the diffusion model ({diffusion_mib} MiB) does not fit {card} ({limit_mib} MiB)",
        )
    if host_room_mib is None:
        return ReplicaPrice(ordinal, False, spill_mib = spill, reason = "host RAM is unknown")
    pinned = replica_host_mib(spill, False)
    if pinned <= host_room_mib:
        return ReplicaPrice(
            ordinal,
            True,
            spill_mib = spill,
            host_mib = pinned,
            reason = f"{card}: {spill} MiB on the host, {pinned} of {host_room_mib} MiB host RAM",
        )
    mapped = replica_host_mib(spill, True)
    if spill and mapped <= host_room_mib:
        return ReplicaPrice(
            ordinal,
            True,
            mmap = True,
            spill_mib = spill,
            host_mib = mapped,
            reason = (
                f"{card}: --mmap, host RAM holds {host_room_mib} MiB, not the {pinned} MiB a pinned "
                f"copy of the {spill} MiB spill needs"
            ),
        )
    return ReplicaPrice(
        ordinal,
        False,
        spill_mib = spill,
        reason = f"host RAM: {card} needs {mapped if spill else pinned} MiB, {host_room_mib} MiB free",
    )


def model_sizes(files: Any) -> tuple[int, int, int]:
    from .sd_cpp_multigpu import TEXT_ENCODER_FIELDS, file_mib
    return (
        file_mib(getattr(files, "diffusion_model", None)),
        sum(file_mib(getattr(files, f, None)) for f in TEXT_ENCODER_FIELDS),
        file_mib(getattr(files, "vae", None)),
    )


def host_room_mib() -> Optional[int]:
    """Host RAM a new server may take: cgroup-aware available less the small-host reserve."""
    from .diffusion_small_host import host_ram_mib, host_reserve_mib

    total, available = host_ram_mib()
    if available is None:
        return None
    return int(available) - host_reserve_mib(total)


def accelerator_ok(binary: Optional[str]) -> bool:
    """CUDA builds only: the per-card mask is CUDA_VISIBLE_DEVICES."""
    from .sd_cpp_engine import binary_accelerator
    return binary_accelerator(binary) == "cuda"


def cuda_mask_entry(ordinal: int) -> str:
    """The parent's CUDA_VISIBLE_DEVICES entry (index or UUID) for torch ordinal ``ordinal``."""
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    entries = [e.strip() for e in raw.split(",") if e.strip()] if raw else []
    if entries:
        return entries[int(ordinal)] if 0 <= int(ordinal) < len(entries) else str(ordinal)
    return str(int(ordinal))


@dataclasses.dataclass(frozen = True)
class ReplicaLoadPlan:
    primary: int
    cards: tuple[int, ...]
    drop_split: bool
    mode: str
    reason: str


def plan_for_load(
    files: Any,
    *,
    primary: Optional[int],
    offload: Any,
    split: Any,
    binary: Optional[str],
    device: str,
    off_torch: Any,
    server_mode: bool,
    limit_of: Optional[Callable[[int], Optional[int]]] = None,
    host_room: Optional[Callable[[], Optional[int]]] = None,
) -> Optional[ReplicaLoadPlan]:
    """Replica cards, None for one server. Auto only for one-card primaries; an explicit count may drop a companion
    split if the primary's spill and one replica fit the host; a block split always keeps."""
    from .diffusion_multigpu import current_selection, multigpu_all_enabled

    mode, cap = replica_setting()
    if mode == "off" or not server_mode or device != "cuda" or off_torch is not None:
        return None
    if list(offload or ()):
        return None
    selection = current_selection()
    if selection is None or not multigpu_all_enabled() or not selection.secondaries:
        return None
    if mode == "auto" and not selection.automatic:
        return None
    if primary is None and selection.automatic:
        primary = selection.primary
    if primary is None or int(primary) != int(selection.primary):
        return None
    if split is not None and getattr(split, "layer", False):
        logger.info("sd_cpp.replicas: one server (the diffusion model is cut across cards)")
        return None
    if not accelerator_ok(binary):
        return None
    if limit_of is None:
        from .sd_cpp_multigpu import card_limit_mib as limit_of
    host_room = host_room or host_room_mib
    diffusion, encoders, vae = model_sizes(files)
    cards = [int(o) for o in selection.secondaries if int(o) != int(primary)]
    if cap is not None:
        cards = cards[: max(0, cap - 1)]
    limits = {o: limit_of(o) for o in (int(primary), *cards)}
    fitting = [
        o
        for o in cards
        if limits[o] is not None and host_spill_mib(diffusion, encoders, vae, limits[o]) is not None
    ]
    if not fitting:
        logger.info("sd_cpp.replicas: one server (no other card holds the diffusion model)")
        return None
    drop = False
    if split is not None and mode != "count":
        # One-card servers cost host RAM at N=1, so dropping the companion split is opt-in.
        logger.info(
            "sd_cpp.replicas: keeping the companion split (one-card servers need %s=N)",
            REPLICAS_ENV,
        )
        return None
    if split is not None:
        room = host_room()
        own = price_replica(
            ordinal = int(primary),
            limit_mib = limits[int(primary)],
            diffusion_mib = diffusion,
            text_encoder_mib = encoders,
            vae_mib = vae,
            host_room_mib = room,
        )
        # The primary runs as a one-card Studio would (no --mmap), so its spill must fit pinned.
        if not own.fits or own.mmap:
            logger.info("sd_cpp.replicas: keeping the split (%s)", own.reason)
            return None
        first = price_replica(
            ordinal = fitting[0],
            limit_mib = limits[fitting[0]],
            diffusion_mib = diffusion,
            text_encoder_mib = encoders,
            vae_mib = vae,
            host_room_mib = None if room is None else room - own.host_mib,
        )
        if not first.fits:
            logger.info("sd_cpp.replicas: keeping the split (%s)", first.reason)
            return None
        drop = True
    reason = (
        f"one sd-server per card under load: cuda:{primary} + "
        + ", ".join(f"cuda:{o}" for o in fitting)
        + (" (instead of the companion split)" if drop else "")
    )
    logger.info("sd_cpp.replicas: %s", reason)
    return ReplicaLoadPlan(
        primary = int(primary), cards = tuple(fitting), drop_split = drop, mode = mode, reason = reason
    )


@dataclasses.dataclass
class _Lane:
    ordinal: int
    status: str = "planned"  # planned | spawning | ready | dead | refused
    reason: str = ""
    mmap: bool = False
    host_mib: int = 0
    server: Any = None
    backend: Any = None
    busy: bool = False
    dispatched: int = 0


def _replica_failure(exc: BaseException, server: Any) -> bool:
    """A failure of the replica itself (its process died or timed out, or its card ran out of memory, which the
    primary recovers from by restarting): the replica is retired and the request retried once elsewhere. Anything
    else (a cancel, a bad request, a replaced model, a job error on a healthy server) fails the same everywhere."""
    from . import sd_cpp_args
    from .diffusion_families import DIFFUSION_CANCELLED_MSG, DiffusionModelReplacedError

    if not isinstance(exc, RuntimeError) or isinstance(exc, DiffusionModelReplacedError):
        return False
    text = str(exc)
    if DIFFUSION_CANCELLED_MSG in text:
        return False
    if server is None or not server.is_alive():
        return True
    try:
        from .sd_cpp_backend import output_shows_capacity_failure
        return bool(sd_cpp_args.is_vram_refusal_text(text) or output_shows_capacity_failure(text))
    except Exception:  # noqa: BLE001 -- unknown keeps the request's own error
        return False


class ReplicaPool:
    """A load's extra servers plus the FIFO line in front of all its servers. ``owner`` provides ``_generate_impl``,
    ``_start_replica(state, ordinal, mmap)`` and ``_stop_server(server)``."""

    def __init__(self, owner: Any, state: Any, plan: ReplicaLoadPlan, token: int) -> None:
        self._owner = owner
        self.state = state
        self.plan = plan
        self.token = token
        self._cond = threading.Condition()
        self._line: collections.deque = collections.deque()
        self._primary_busy = False
        self._primary_dispatched = 0
        self._failovers = 0
        self._price_lock = threading.Lock()
        self.closed = False
        self.lanes = [_Lane(int(o)) for o in plan.cards]

    def run(self, kwargs: dict[str, Any]) -> Any:
        from core.inference.generation_ticket import bind_generation_cancel

        waiting = threading.Event()
        bind_generation_cancel(waiting)
        front = False
        retried = False
        while True:
            target = self._acquire(waiting, front)
            if target is None:
                return self._owner._generate_impl(**kwargs)
            if target == PRIMARY:
                try:
                    return self._owner._generate_impl(**kwargs)
                finally:
                    self._release(PRIMARY)
            lane = target
            server = lane.server
            try:
                return lane.backend._generate_impl(**kwargs)
            except BaseException as exc:
                if self.closed or waiting.is_set() or not _replica_failure(exc, server):
                    raise
                # Retired even past the one retry, so later requests stop landing on a server that failed.
                self._retire(lane, f"failed a render: {str(exc)[:300]}")
                if retried:
                    raise
                with self._cond:
                    self._failovers += 1
                logger.warning(
                    "sd_cpp.replicas: cuda:%s failed a render (%s); retrying it on another server",
                    lane.ordinal,
                    str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__,
                )
                retried = True
                front = True
            finally:
                self._release(lane)

    def _acquire(self, waiting: threading.Event, front: bool) -> Any:
        from .diffusion_families import DIFFUSION_CANCELLED_MSG
        with self._cond:
            if self.closed:
                return None
            if not front and not self._line and not self._primary_busy:
                return self._take_primary_locked()
            ticket = object()
            if front:
                self._line.appendleft(ticket)
            else:
                self._line.append(ticket)
            self._spawn_planned_locked()
            try:
                while True:
                    if waiting.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    if self.closed:
                        return None
                    if self._line[0] is ticket:
                        if not self._primary_busy:
                            return self._take_primary_locked()
                        lane = self._free_lane_locked()
                        if lane is not None:
                            lane.busy = True
                            lane.dispatched += 1
                            return lane
                    # Timed: cancels and idle replica deaths are polled, not signalled.
                    self._cond.wait(timeout = 0.1)
            finally:
                try:
                    self._line.remove(ticket)
                except ValueError:
                    pass
                self._cond.notify_all()

    def _take_primary_locked(self) -> str:
        self._primary_busy = True
        self._primary_dispatched += 1
        return PRIMARY

    def _free_lane_locked(self) -> Optional[_Lane]:
        for lane in self.lanes:
            if lane.status != "ready" or lane.busy:
                continue
            server = lane.server
            if server is None or not server.is_alive():
                self._mark_dead_locked(lane, "the server exited while idle")
                if server is not None:
                    # Already exited: only its client, scratch dir and atexit hook are left to release.
                    threading.Thread(
                        target = self._owner._stop_server, args = (server,), daemon = True
                    ).start()
                continue
            return lane
        return None

    def _release(self, target: Any) -> None:
        with self._cond:
            if target == PRIMARY:
                self._primary_busy = False
            else:
                target.busy = False
            self._cond.notify_all()

    def _spawn_planned_locked(self) -> None:
        for lane in self.lanes:
            if lane.status == "planned":
                lane.status = "spawning"
                threading.Thread(
                    target = self._spawn,
                    args = (lane,),
                    name = f"sd-server-replica-{lane.ordinal}",
                    daemon = True,
                ).start()

    def _spawn(self, lane: _Lane) -> None:
        from .sd_cpp_multigpu import card_limit_mib

        try:
            diffusion, encoders, vae = model_sizes(self.state.files)
            # Serialised: a replica still loading has not charged the host yet.
            with self._price_lock:
                room = host_room_mib()
                with self._cond:
                    pending = sum(
                        other.host_mib
                        for other in self.lanes
                        if other is not lane and other.status == "spawning"
                    )
                price = price_replica(
                    ordinal = lane.ordinal,
                    limit_mib = card_limit_mib(lane.ordinal),
                    diffusion_mib = diffusion,
                    text_encoder_mib = encoders,
                    vae_mib = vae,
                    host_room_mib = None if room is None else room - pending,
                )
                with self._cond:
                    if not price.fits:
                        lane.status, lane.reason = "refused", price.reason
                        logger.info(
                            "sd_cpp.replicas: no server on cuda:%s (%s)", lane.ordinal, price.reason
                        )
                        return
                    lane.mmap, lane.host_mib, lane.reason = price.mmap, price.host_mib, price.reason
            backend = self._owner._start_replica(self, lane, price.mmap)
        except Exception as exc:  # noqa: BLE001 -- a replica that cannot start leaves the primary serving
            server = lane.server
            if server is not None:
                # Reserved before it leaves the lane, so the tree never reads idle while it may still run.
                self._owner._reserve_replica_stop()
            with self._cond:
                if lane.server is server:
                    lane.server = None
                if lane.status == "spawning":
                    lane.status, lane.reason = "dead", f"failed to start: {str(exc)[:300]}"
                self._cond.notify_all()
            if not self.closed:
                logger.warning("sd_cpp.replicas: cuda:%s failed to start: %s", lane.ordinal, exc)
            if server is not None:
                self._owner._stop_reserved(server)
            return
        with self._cond:
            if self.closed or lane.status != "spawning":
                stale = backend
            else:
                stale = None
                lane.backend = backend
                lane.status = "ready"
                logger.info(
                    "sd_cpp.replicas: cuda:%s ready%s",
                    lane.ordinal,
                    " (--mmap)" if lane.mmap else "",
                )
            self._cond.notify_all()
        if stale is not None and lane.server is not None:
            self._owner._stop_server(lane.server)

    def publish_server(self, lane: _Lane, server: Any) -> bool:
        """Called by the owner before ``server.start``, so a close during the load stops it. False once closed."""
        with self._cond:
            if self.closed:
                return False
            lane.server = server
            return True

    def _mark_dead_locked(self, lane: _Lane, reason: str) -> Any:
        server = lane.server
        lane.status, lane.reason = "dead", reason
        lane.server = None
        return server

    def _retire(self, lane: _Lane, reason: str) -> None:
        server = lane.server
        if server is not None:
            # Reserved before it leaves the lane, and never under _cond: the owner's lock comes first.
            self._owner._reserve_replica_stop()
        with self._cond:
            if lane.server is server:
                self._mark_dead_locked(lane, reason)
            elif lane.status in ("ready", "spawning"):
                lane.status, lane.reason = "dead", reason
            self._cond.notify_all()
        if server is not None:
            threading.Thread(
                target = self._owner._stop_reserved,
                args = (server,),
                name = f"sd-server-replica-stop-{lane.ordinal}",
                daemon = True,
            ).start()

    def close(self) -> list[Any]:
        """Stop dispatching and cancel replica renders; returns the replica servers for the caller to stop."""
        with self._cond:
            self.closed = True
            servers = []
            for lane in self.lanes:
                if lane.server is not None:
                    servers.append(lane.server)
                    lane.server = None
                if lane.status in ("planned", "spawning", "ready"):
                    lane.status, lane.reason = "dead", "unloaded"
            self._cond.notify_all()
        for lane in self.lanes:
            backend = lane.backend
            cancel = getattr(backend, "_active_generate_cancel", None)
            if cancel is not None:
                cancel.set()
        return servers

    def wait_idle(self) -> None:
        """Returns once no replica render runs."""
        for lane in self.lanes:
            lock = getattr(lane.backend, "_generate_lock", None)
            if lock is not None:
                with lock:
                    pass

    def holds_server(self) -> bool:
        return any(lane.server is not None for lane in self.lanes)

    def lane_backends(self) -> list[Any]:
        return [lane.backend for lane in self.lanes if lane.backend is not None]

    def describe(self) -> dict[str, Any]:
        with self._cond:
            cards = [
                {
                    "card": f"cuda:{self.plan.primary}",
                    "role": "primary",
                    "status": "busy" if self._primary_busy else "ready",
                    "dispatched": self._primary_dispatched,
                }
            ]
            for lane in self.lanes:
                cards.append(
                    {
                        "card": f"cuda:{lane.ordinal}",
                        "role": "replica",
                        "status": "busy" if lane.busy and lane.status == "ready" else lane.status,
                        "mmap": lane.mmap,
                        "host_mib": lane.host_mib or None,
                        "dispatched": lane.dispatched,
                        "reason": lane.reason or None,
                    }
                )
            ready = 1 + sum(1 for lane in self.lanes if lane.status == "ready")
            return {
                "mode": self.plan.mode,
                "active": ready > 1,
                "servers": ready,
                "waiting": len(self._line),
                "in_flight": int(self._primary_busy)
                + sum(1 for lane in self.lanes if lane.busy and lane.status == "ready"),
                "failovers": self._failovers,
                "replaced_split": self.plan.drop_split,
                "cards": cards,
                "summary": self.plan.reason,
            }

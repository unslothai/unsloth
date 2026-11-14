# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Machines: pair workers, approve them, list and remove them."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject

logger = logging.getLogger(__name__)

PAIR_TTL_S = 600.0


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    # A pairing code lets a machine join the cluster, so an API key must not mint one.
    if via_api_key:
        raise HTTPException(status_code = 403, detail = "Pairing machines requires a UI session.")


router = APIRouter(
    dependencies = [
        Depends(get_current_subject),
        Depends(policy.require_owner),
        Depends(_require_ui_session),
    ]
)


def _link() -> Any:
    from core.inference import cluster_serving

    identity = cluster_serving._link_identity()
    if identity is None:
        raise HTTPException(status_code = 501, detail = "This install has no cluster support.")
    return identity


@dataclass
class _Pairing:
    id: str
    window: Any
    expires_at: float
    remote: bool
    state: str = "waiting"  # waiting | approval | paired | expired | refused | failed | cancelled
    name: Optional[str] = None
    words: Optional[List[str]] = None
    node_id: Optional[str] = None
    error: Optional[str] = None
    answer: Optional[asyncio.Future] = None
    task: Optional[asyncio.Task] = field(default = None, repr = False)

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "name": self.name,
            "words": self.words,
            "node_id": self.node_id,
            "error": self.error,
            "expires_at": self.expires_at,
            "remote": self.remote,
        }


_PAIRINGS: Dict[str, _Pairing] = {}
_PAIR_LOCK = asyncio.Lock()


def _end(
    p: _Pairing,
    state: str,
    error: Optional[str] = None,
) -> None:
    if p.answer is not None and not p.answer.done():
        p.answer.set_result(False)
    p.window.close()
    # Closing refuses any later commit; a worker already recorded stays paired.
    peer = getattr(p.window, "paired", None)
    if peer is not None:
        p.state, p.name, p.node_id, p.error = "paired", peer.name, peer.node_id, None
    elif p.state in ("waiting", "approval"):
        p.state, p.error = state, error


async def _run(p: _Pairing) -> None:
    from studio.cluster_link.pairing import PairingError
    try:
        peer = await p.window.wait(max(0.0, p.expires_at - time.time()))
    except (asyncio.TimeoutError, TimeoutError):
        _end(p, "expired")
    except PairingError as exc:
        _end(p, "refused" if p.state == "refused" else "failed", str(exc).strip().rstrip("."))
    except asyncio.CancelledError:
        _end(p, "cancelled")
        raise
    except Exception as exc:
        logger.info("pairing from Settings failed: %s", type(exc).__name__)
        _end(p, "failed", "pairing failed")
    else:
        p.state, p.name, p.node_id = "paired", peer.name, peer.node_id
        p.window.close()


def _get(pairing_id: str) -> _Pairing:
    p = _PAIRINGS.get(pairing_id)
    if p is None:
        raise HTTPException(status_code = 404, detail = "No such pairing.")
    return p


@router.get("/machines")
async def list_machines(check: bool = True) -> Dict[str, Any]:
    identity = _link()
    master = identity.master_identity(create = False)
    if master is None:
        worker = identity.worker_identity()
        return {"role": "worker" if worker is not None else "none", "machines": []}
    registry = identity.Registry().load()
    workers = [p for p in registry.workers() if not getattr(p, "removed", False)]
    if not check or not workers:
        rows = [
            {"name": p.name, "node_id": p.node_id, "state": "unknown", "where": None}
            for p in workers
        ]
        return {"role": "master", "machines": rows}
    from studio.cluster_link import status

    def run() -> List[Any]:
        return status.check_workers(
            master,
            registry,
            workers,
            authorize = lambda peer: identity.Registry().load().authorize(peer, "worker") is not None,
        )

    rows = await asyncio.to_thread(run)
    return {
        "role": "master",
        "machines": [
            {
                "name": r.name,
                "node_id": r.node_id,
                "state": r.state,
                "via": r.via,
                "reason": r.reason,
                "where": r.where,
                "devices": r.devices,
                "discovered": r.discovered,
            }
            for r in rows
        ],
    }


@router.delete("/machines/{node_id}")
async def remove_machine(node_id: str) -> Dict[str, Any]:
    identity = _link()
    if identity.master_identity(create = False) is None:
        raise HTTPException(status_code = 404, detail = "No machines are paired.")
    registry = identity.Registry().load()
    match = [p for p in registry.workers() if p.node_id == node_id and not p.removed]
    if not match:
        raise HTTPException(status_code = 404, detail = "No such machine.")
    with registry.transaction():
        removed = registry.remove(node_id)
    if not removed:
        raise HTTPException(status_code = 500, detail = f"Could not remove {match[0].name}.")
    from studio.cluster_link import remote

    await asyncio.to_thread(remote.drop, node_id)
    return {"removed": node_id, "name": match[0].name}


@router.post("/pair")
async def start_pairing() -> Dict[str, Any]:
    # One pairing at a time: replacing, starting and registering a window never interleave.
    async with _PAIR_LOCK:
        return await _start_pairing()


async def _start_pairing() -> Dict[str, Any]:
    identity = _link()
    from studio.cluster_link import PAIR_PORT_DEFAULT
    from studio.cluster_link.pairing import PairingError, PairingWindow

    for p in list(_PAIRINGS.values()):
        if p.state in ("waiting", "approval"):
            _end(p, "cancelled")
    try:
        master = identity.master_identity(create = True)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc))
    except ImportError:
        raise HTTPException(status_code = 501, detail = "Pairing needs the cryptography package.")

    from studio.cluster_link import remote

    await asyncio.to_thread(remote.ensure_helper)
    pairing_id = secrets.token_hex(8)
    holder: Dict[str, _Pairing] = {}

    async def approve(name: str, words: Optional[List[str]]) -> bool:
        p = holder["p"]
        p.state, p.name, p.words = "approval", name, words
        p.answer = asyncio.get_running_loop().create_future()
        return await p.answer

    window = None
    code = None
    for port in (PAIR_PORT_DEFAULT, 0):
        window = PairingWindow(master, host = "0.0.0.0", port = port, ttl = PAIR_TTL_S, approve = approve)
        try:
            code = await window.start()
            break
        except OSError:
            window.close()
        except PairingError as exc:
            window.close()
            raise HTTPException(status_code = 400, detail = str(exc))
        except BaseException:
            window.close()
            raise
    if code is None:
        raise HTTPException(status_code = 503, detail = "Could not open a pairing port.")
    p = _Pairing(id = pairing_id, window = window, expires_at = window.expires_at, remote = bool(window.tc))
    holder["p"] = p
    _PAIRINGS.clear()
    _PAIRINGS[pairing_id] = p
    p.task = asyncio.create_task(_run(p))
    # The code is only ever in this response body: never in a URL, a log line or another route.
    return {**p.public(), "code": code}


@router.get("/pair/{pairing_id}")
async def pairing_state(pairing_id: str) -> Dict[str, Any]:
    return _get(pairing_id).public()


def _answer(pairing_id: str, allow: bool) -> Dict[str, Any]:
    p = _get(pairing_id)
    if p.state != "approval" or p.answer is None or p.answer.done():
        raise HTTPException(status_code = 409, detail = "No machine is waiting for approval.")
    if not allow:
        p.state = "refused"
    p.answer.set_result(allow)
    return p.public()


@router.post("/pair/{pairing_id}/approve")
async def approve_pairing(pairing_id: str) -> Dict[str, Any]:
    return _answer(pairing_id, True)


@router.post("/pair/{pairing_id}/refuse")
async def refuse_pairing(pairing_id: str) -> Dict[str, Any]:
    return _answer(pairing_id, False)


@router.post("/pair/{pairing_id}/cancel")
async def cancel_pairing(pairing_id: str) -> Dict[str, Any]:
    p = _get(pairing_id)
    _end(p, "cancelled")
    return p.public()

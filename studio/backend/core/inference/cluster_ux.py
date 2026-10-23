# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""User-facing wording for a model served across paired cluster workers."""

from __future__ import annotations

from typing import Any, List, Optional

_GIB = float(2**30)
_LIST_HINT = "`unsloth cluster list` shows which workers are ready"


def _state() -> Any:
    try:
        from core.inference import cluster_serving
    except Exception:
        return None
    return cluster_serving._STATE


def _join(names: List[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _worker_label(endpoint: str, nodes: List[Any]) -> str:
    from core.inference import cluster_serving
    for node in nodes:
        try:
            if node.endpoint == endpoint or cluster_serving._same_link_worker(endpoint, node):
                return node.name or node.node_id or node.host
        except Exception:
            continue
    return endpoint.rsplit("@", 1)[-1]


def _machines(state: Any, plan: Any) -> List[str]:
    devices = list(getattr(plan, "devices", None) or [])
    nodes = list(getattr(state, "nodes", None) or [])
    workers: List[str] = []
    for device in devices:
        if device.endpoint is None:
            continue
        label = _worker_label(device.endpoint, nodes)
        if label not in workers:
            workers.append(label)
    local = any(device.endpoint is None for device in devices)
    return (["this machine"] if local else []) + workers


def _remote_workers(state: Any, plan: Any) -> List[str]:
    return [m for m in _machines(state, plan) if m != "this machine"]


def _split_plan(state: Any) -> Any:
    plan = getattr(state, "plan", None)
    if (
        state is None
        or getattr(state, "topology", None) != "layer_split"
        or plan is None
        or getattr(plan, "topology", None) != "layer_split"
    ):
        return None
    return plan


def split_summary() -> Optional[str]:
    """ "Split across this machine and ux-worker." for the served (or loading) split, else None."""
    state = _state()
    if getattr(state, "_passthrough", False):
        return None
    plan = _split_plan(state)
    if plan is None:
        return None
    machines = _machines(state, plan)
    if not _remote_workers(state, plan):
        return None
    return f"Split across {_join(machines)}."


def load_progress_message(llama_started: bool) -> Optional[str]:
    """What a planned cluster load is doing while the progress bar alone cannot say it."""
    state = _state()
    if (
        state is None
        or not getattr(state, "in_flight", False)
        or getattr(state, "_passthrough", False)
    ):
        return None
    plan = _split_plan(state)
    if plan is None:
        if llama_started or getattr(state, "plan", None) is not None:
            return None
        return "Checking your paired workers and planning where the model's layers go."
    workers = _remote_workers(state, plan)
    if not workers:
        return None
    if not llama_started:
        return f"Starting the model server, split across {_join(_machines(state, plan))}."
    return (
        f"Sending the model's layers to {_join(workers)} over the network and warming up. "
        "This can take a while."
    )


def lost_connection_message() -> Optional[str]:
    """The chat error for a dead llama-server when the model it served was split across workers."""
    state = _state()
    plan = _split_plan(state)
    if plan is None or not getattr(state, "block", None):
        return None
    workers = _remote_workers(state, plan)
    if not workers:
        return None
    named = workers[0] if len(workers) == 1 else "each worker"
    return (
        f"Lost connection to the model, which is split across {_join(_machines(state, plan))}. "
        f"Check that `unsloth worker` is still running on {named}, then reload the model."
    )


def dropped_worker_notice() -> Optional[str]:
    """After planning: the paired workers this load had to leave out, in words a user can act on."""
    state = _state()
    if state is None or getattr(state, "_passthrough", False):
        return None
    try:
        from core.inference import cluster_serving
        dropped = list(cluster_serving.LINK_DROPPED)
    except Exception:
        return None
    if not dropped:
        return None
    plan = _split_plan(state)
    held = set(_remote_workers(state, plan)) if plan is not None else set()
    missing = [name for name in dropped if name not in held]
    if not missing:
        return None
    named = _join(missing)
    verb = "was" if len(missing) == 1 else "were"
    where = (
        f"split across {_join(_machines(state, plan))} instead"
        if plan is not None and held
        else "running on this machine only"
    )
    return (
        f"Paired worker {named} {verb} not reachable, so this model is {where}. "
        f"Start `unsloth worker` on {named} ({_LIST_HINT}), then reload the model to use it."
    )


def does_not_fit_detail(exc: BaseException) -> Optional[str]:
    """The /load refusal for a model the planner found too big even across the cluster."""
    plan = getattr(exc, "cluster_plan", None)
    if plan is None:
        return None
    need = getattr(plan, "need_bytes", None)
    usable = getattr(plan, "usable_bytes", None)
    state = _state()
    machines = _machines(state, plan)
    fix = (
        "Use a smaller quant, a shorter context length or fewer parallel chats, or pair another "
        "machine with `unsloth cluster pair`."
    )
    if need is not None and usable is not None and machines:
        return (
            f"This model does not fit, even split across {_join(machines)}: with its context "
            f"cache it needs {need / _GIB:.1f} GiB, and {'they have' if len(machines) > 1 else 'it has'} "
            f"{usable / _GIB:.1f} GiB usable. {fix}"
        )
    reason = str(getattr(plan, "reason", "") or str(exc)).strip()
    if not reason:
        return None
    return f"This model cannot be split across your machines: {reason}. {fix}"

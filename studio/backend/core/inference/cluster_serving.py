# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serving one GGUF across N machines over llama.cpp RPC: any number of nodes, any mix of NVIDIA,
AMD and Apple Silicon, any OS that runs ``ggml-rpc-server``.

Opt-in only. With no cluster configured nothing here runs and ``spark_serving`` keeps its two-Spark
path exactly as it was. Configure with ``UNSLOTH_CLUSTER_NODES`` (comma list of
``[user@]host[:port]``) or ``<studio home>/cluster.json``::

    {"nodes": [{"host": "10.0.0.2", "rpc_port": 50052, "ssh_user": "me", "managed": true}]}

``managed`` nodes get their ``ggml-rpc-server`` started over ssh; the others are expected to run
``ggml-rpc-server -H <ip> -p <port>`` themselves (the way a Mac or a Windows box joins).

Placement comes from the coordinator's own ``llama-server --rpc ... --list-devices``: one
``RPC<k>`` per remote device in endpoint order, with its free memory, so a Metal, CUDA or ROCm
worker is priced by what its backend reports, without ssh.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV_NODES = "UNSLOTH_CLUSTER_NODES"
ENV_MANAGED = "UNSLOTH_CLUSTER_MANAGED"  # "1": start every node's rpc-server over ssh
ENV_TOPOLOGY = "UNSLOTH_CLUSTER_TOPOLOGY"  # single | layer_split
CONFIG_NAME = "cluster.json"
RPC_PORT_DEFAULT = 50052
_GIB = 1024**3
_MIB = 1024**2
# Compute buffers, the KV padding and the context llama.cpp keeps per device.
DEVICE_RESERVE_MIN_BYTES = 1 * _GIB
DEVICE_RESERVE_FRACTION = 0.08
# Whole layers: a device can be handed up to a layer more than its share of the weights.
LAYER_ROUNDING_MARGIN = 0.05
LIST_DEVICES_TIMEOUT_S = 60.0
WORKER_START_TIMEOUT_S = 60.0

_DEVICE_LINE = re.compile(r"^\s+(\S+): (.*) \((\d+) MiB, (\d+) MiB free\)\s*$")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")


class ClusterConfigError(ValueError):
    """The cluster configuration cannot be used as written."""


@dataclass(frozen = True)
class Node:
    host: str
    rpc_port: int = RPC_PORT_DEFAULT
    ssh_user: Optional[str] = None
    managed: bool = False

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.rpc_port}"

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}" if self.ssh_user else self.host


@dataclass
class Device:
    name: str
    description: str
    total_bytes: int
    free_bytes: int
    endpoint: Optional[str] = None  # None: on this machine

    @property
    def remote(self) -> bool:
        return self.endpoint is not None


@dataclass
class ClusterPlan:
    topology: str
    reason: str
    devices: List[Device] = field(default_factory = list)
    weights: List[float] = field(default_factory = list)
    need_bytes: Optional[float] = None
    usable_bytes: Optional[float] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
            "reason": self.reason,
            "need_gib": None if self.need_bytes is None else round(self.need_bytes / _GIB, 2),
            "usable_gib": None if self.usable_bytes is None else round(self.usable_bytes / _GIB, 2),
            "devices": [
                {
                    "name": d.name,
                    "endpoint": d.endpoint or "local",
                    "description": d.description,
                    "total_gib": round(d.total_bytes / _GIB, 2),
                    "free_gib": round(d.free_bytes / _GIB, 2),
                    "weight": round(w, 4),
                }
                for d, w in zip(self.devices, self.weights or [0.0] * len(self.devices))
            ],
        }


def _parse_node(text: str, *, managed: bool = False) -> Node:
    raw = text.strip()
    user = None
    if "@" in raw:
        user, raw = raw.split("@", 1)
        if not user:
            raise ClusterConfigError(f"empty ssh user in {text!r}")
    if raw.startswith("["):
        raise ClusterConfigError(f"IPv6 node addresses are not supported yet: {text!r}")
    host, sep, port_text = raw.partition(":")
    if not host or not _HOST_RE.match(host):
        raise ClusterConfigError(f"not a host name or IPv4 address: {text!r}")
    port = RPC_PORT_DEFAULT
    if sep:
        if not port_text.isdigit() or not (0 < int(port_text) < 65536):
            raise ClusterConfigError(f"not a port: {text!r}")
        port = int(port_text)
    return Node(host = host, rpc_port = port, ssh_user = user, managed = managed)


def parse_nodes_env(value: str, *, managed: bool = False) -> List[Node]:
    nodes = [_parse_node(part, managed = managed) for part in value.split(",") if part.strip()]
    return _dedupe(nodes)


def parse_nodes_json(data: Any) -> List[Node]:
    entries = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ClusterConfigError('cluster.json needs {"nodes": [...]}')
    nodes = []
    for entry in entries:
        if isinstance(entry, str):
            nodes.append(_parse_node(entry))
            continue
        if not isinstance(entry, dict) or not entry.get("host"):
            raise ClusterConfigError(f"a node needs a host: {entry!r}")
        node = _parse_node(str(entry["host"]), managed = bool(entry.get("managed", False)))
        port = entry.get("rpc_port", node.rpc_port)
        if not isinstance(port, int) or not (0 < port < 65536):
            raise ClusterConfigError(f"not a port: {entry!r}")
        user = entry.get("ssh_user", node.ssh_user)
        nodes.append(
            Node(host = node.host, rpc_port = port, ssh_user = user or None, managed = node.managed)
        )
    return _dedupe(nodes)


def _dedupe(nodes: List[Node]) -> List[Node]:
    seen = set()
    for node in nodes:
        if node.endpoint in seen:
            raise ClusterConfigError(f"{node.endpoint} is listed twice")
        seen.add(node.endpoint)
    if not nodes:
        raise ClusterConfigError("the cluster lists no nodes")
    return nodes


def _config_path() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import studio_root
        return studio_root() / CONFIG_NAME
    except Exception:
        return None


def configured() -> bool:
    """Whether the operator asked for a cluster at all; cheap, no file parsing beyond existence."""
    if (os.environ.get(ENV_NODES) or "").strip():
        return True
    path = _config_path()
    return bool(path is not None and path.is_file())


def load_nodes() -> List[Node]:
    value = (os.environ.get(ENV_NODES) or "").strip()
    if value:
        return parse_nodes_env(value, managed = (os.environ.get(ENV_MANAGED) or "") == "1")
    path = _config_path()
    if path is None or not path.is_file():
        raise ClusterConfigError("no cluster configured")
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        raise ClusterConfigError(f"{path} could not be read: {exc}") from exc
    return parse_nodes_json(data)


def parse_device_list(text: str, endpoints: Sequence[str]) -> List[Device]:
    """``llama-server --list-devices``. In the unslothai fork an RPC device's description is its
    endpoint; upstream builds may say more, so an endpoint found anywhere in it counts."""
    devices = []
    for line in (text or "").splitlines():
        m = _DEVICE_LINE.match(line)
        if not m:
            continue
        name, desc, total, free = m.group(1), m.group(2).strip(), int(m.group(3)), int(m.group(4))
        endpoint = None
        if name.startswith("RPC"):
            endpoint = next((e for e in endpoints if e in desc), desc)
        devices.append(Device(name, desc, total * _MIB, free * _MIB, endpoint))
    return devices


def list_devices(binary: str, endpoints: Sequence[str]) -> Tuple[List[Device], str]:
    """The devices the coordinator would see with these RPC servers, and any error text."""
    try:
        proc = subprocess.run(
            [binary, "--rpc", ",".join(endpoints), "--list-devices"],
            capture_output = True,
            text = True,
            timeout = LIST_DEVICES_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    devices = parse_device_list(proc.stdout, endpoints)
    error = "" if devices else ((proc.stderr or "").strip().splitlines() or ["no devices"])[-1]
    return devices, error[:300]


def usable_bytes(device: Device) -> float:
    reserve = max(DEVICE_RESERVE_MIN_BYTES, DEVICE_RESERVE_FRACTION * device.total_bytes)
    return max(0.0, float(device.free_bytes) - reserve)


def plan_cluster(
    model_bytes: Optional[float],
    kv_bytes: float,
    local_extra_bytes: float,
    devices: Sequence[Device],
    *,
    forced: Optional[str] = None,
) -> ClusterPlan:
    """single when this machine holds the load, else a layer split weighted by usable memory.

    The split prices weights and KV by share, which is what ``--tensor-split`` hands out, and
    charges sidecars (projector, drafter, adapters) to the coordinator, which loads them.
    """
    if model_bytes is None:
        return ClusterPlan("single", "model size unknown; not guessing, serving on this machine")
    local = [d for d in devices if not d.remote]
    remote = [d for d in devices if d.remote]
    need = float(model_bytes) + float(kv_bytes or 0)
    local_dev = local[0] if local else None
    local_usable = usable_bytes(local_dev) if local_dev else 0.0
    if forced != "layer_split" and local_dev and need + local_extra_bytes <= local_usable:
        return ClusterPlan(
            "single",
            f"fits this machine ({(need + local_extra_bytes) / _GIB:.1f} GiB of "
            f"{local_usable / _GIB:.1f} GiB usable)",
            need_bytes = need,
            usable_bytes = local_usable,
        )
    if forced == "single":
        return ClusterPlan("single", f"forced by {ENV_TOPOLOGY}=single", need_bytes = need)
    if not remote:
        return ClusterPlan("single", "no remote device answered; serving on this machine")
    # Remote first, this machine last: llama.cpp puts the output layer on the last device, so the
    # logits stay local and the wire carries only the hidden state.
    ordered = remote + ([local_dev] if local_dev else [])
    budgets = [usable_bytes(d) for d in remote]
    if local_dev:
        budgets.append(max(0.0, local_usable - local_extra_bytes))
    total = sum(budgets)
    if local_dev is None and local_extra_bytes > 0:
        return ClusterPlan(
            "refuse",
            "this machine has no accelerator to hold the projector / drafter the load names",
            need_bytes = need,
        )
    if total <= 0 or need * (1 + LAYER_ROUNDING_MARGIN) > total:
        return ClusterPlan(
            "refuse",
            f"the model with its KV needs {need / _GIB:.1f} GiB (+{LAYER_ROUNDING_MARGIN:.0%} for "
            f"whole layers) against {total / _GIB:.1f} GiB usable over {len(ordered)} devices; "
            "add a node, use a smaller quant, fewer users or a shorter context",
            ordered,
            need_bytes = need,
            usable_bytes = total,
        )
    weights = [b / total for b in budgets]
    keep = [(d, w) for d, w in zip(ordered, weights) if w > 0]
    return ClusterPlan(
        "layer_split",
        f"{need / _GIB:.1f} GiB over {len(keep)} devices weighted by usable memory "
        f"({total / _GIB:.1f} GiB usable)",
        [d for d, _ in keep],
        [w for _, w in keep],
        need_bytes = need,
        usable_bytes = total,
    )


def split_extra_args(endpoints: Sequence[str], plan: ClusterPlan) -> List[str]:
    weights = ",".join(f"{w:.4f}".rstrip("0").rstrip(".") or "0" for w in plan.weights)
    return [
        "--rpc",
        ",".join(endpoints),
        "--device",
        ",".join(d.name for d in plan.devices),
        "-sm",
        "layer",
        "--tensor-split",
        weights,
        "--cache-ram",
        "0",
    ]


# Blocks this module appended to a launch, so a reload that inherits the stored extras drops them
# instead of reading them as the caller's own --rpc.
_MANAGED_BLOCKS: List[List[str]] = []


def strip_managed_block(extra: Sequence[str]) -> List[str]:
    tokens = [str(t) for t in extra]
    for block in reversed(_MANAGED_BLOCKS):
        n = len(block)
        if n and len(tokens) >= n and tokens[-n:] == block:
            return tokens[:-n]
    return tokens


def _remember_block(block: List[str]) -> None:
    if block in _MANAGED_BLOCKS:
        _MANAGED_BLOCKS.remove(block)
    _MANAGED_BLOCKS.append(list(block))
    del _MANAGED_BLOCKS[:-8]


class ClusterServing:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.config_error: str = ""
        self.topology: str = "single"
        self.reason: str = "no load yet"
        self.plan: Optional[ClusterPlan] = None
        self.block: List[str] = []
        self.workers: Dict[str, Any] = {}  # endpoint -> spark_serving.PeerProcess
        self.last_error: str = ""
        self._passthrough: bool = False
        self._pre_load: Optional[Tuple[str, str, Optional[ClusterPlan], List[str]]] = None

    def refresh_nodes(self) -> List[Node]:
        try:
            self.nodes = load_nodes()
            self.config_error = ""
        except ClusterConfigError as exc:
            self.nodes, self.config_error = [], str(exc)
        return self.nodes

    async def _ensure_workers(self, *, restart: bool) -> Optional[str]:
        """Start (or restart) the managed nodes' rpc-servers. The error text, or None."""
        from core.inference import spark_serving as ss

        local_rpc = ss.rpc_server_binary()
        names = ([os.path.basename(local_rpc)] if local_rpc else []) + list(ss.rpc_server_names())
        for node in self.nodes:
            if not node.managed:
                continue
            running = self.workers.get(node.endpoint)
            if running is not None and running.alive and not restart:
                continue
            if running is not None:
                await running.stop()
                self.workers.pop(node.endpoint, None)
            candidates: List[str] = []
            for name in dict.fromkeys(names):
                candidates += [
                    c for c in ss.peer_binary_candidates(local_rpc, name) if c not in candidates
                ]
            rc, out, _err = await ss.ssh_run(node.ssh_target, ss.find_binary_script(candidates))
            binary = out.strip().splitlines()[-1] if out.strip() else "MISSING"
            if rc != 0 or binary == "MISSING":
                return f"{node.host} has no ggml-rpc-server at the bundle path"
            argv = ss.rpc_server_argv(binary, bind = node.host, port = node.rpc_port, cache = False)
            log_dir = ss._log_dir()
            process = ss.PeerProcess(
                "ggml-rpc-server",
                node.ssh_target,
                argv,
                log_dir / f"rpc-server-{node.host}-{node.rpc_port}.log" if log_dir else None,
            )
            await process.start()
            if not await ss.wait_for_own_port(
                process, node.host, node.rpc_port, WORKER_START_TIMEOUT_S
            ):
                tail = list(process.tail)[-3:]
                await process.stop()
                return f"ggml-rpc-server did not take {node.endpoint} (last output: {tail})"
            self.workers[node.endpoint] = process
        return None

    async def before_load(
        self,
        request: Any,
        n_parallel: int,
        *,
        inherited_extra_args: Optional[List[str]] = None,
        confirm_teardown: Any = None,
    ) -> Any:
        from core.inference import spark_serving as ss

        self._passthrough = (
            bool(getattr(request, "alongside", False))
            or bool(getattr(request, "force_alongside", False))
            or str(getattr(request, "engine", "auto") or "auto") != "auto"
        )
        if self._passthrough:
            return request
        self._pre_load = (self.topology, self.reason, self.plan, list(self.block))
        if not self.refresh_nodes():
            self.topology, self.reason = "single", f"cluster config: {self.config_error}"
            return request
        explicit = getattr(request, "llama_extra_args", None)
        effective = list(explicit if explicit is not None else (inherited_extra_args or []))
        if getattr(request, "gpu_ids", None):
            self.topology, self.reason = "single", "an explicit GPU selection replaces placement"
            return request
        if ss.argv_or_env_rpc(effective):
            self.topology, self.reason = "single", "the caller supplied --rpc; placement is theirs"
            return request
        restart = bool(getattr(request, "force_reload", False)) and any(
            w.alive for w in self.workers.values()
        )
        if restart and confirm_teardown is not None:
            confirm_teardown()  # refuses (409) before a managed worker serving a model is stopped
        failed = await self._ensure_workers(restart = restart)
        if failed:
            self.topology, self.reason, self.last_error = "single", failed, failed
            return request
        model_bytes, kv_bytes, local_extra = await self._size(request, n_parallel, effective, ss)
        binary = ss.llama_server_binary()
        if not binary:
            self.topology, self.reason = "single", "no local llama-server to plan with"
            return request
        endpoints = [n.endpoint for n in self.nodes]
        devices, error = await asyncio.to_thread(list_devices, binary, endpoints)
        missing = [e for e in endpoints if not any(d.endpoint == e for d in devices)]
        if missing:
            self.last_error = f"no device from {', '.join(missing)}" + (
                f": {error}" if error else ""
            )
            logger.warning("cluster serving: %s", self.last_error)
        forced = (os.environ.get(ENV_TOPOLOGY) or "").strip().lower().replace("-", "_") or None
        plan = plan_cluster(model_bytes, kv_bytes, local_extra, devices, forced = forced)
        self.plan = plan
        if plan.topology == "refuse":
            self.topology, self.reason = self._pre_load[0], self._pre_load[1]
            raise ss.SparkLoadDoesNotFit(plan.reason)
        self.topology, self.reason = plan.topology, plan.reason
        if plan.topology != "layer_split":
            self.block = []
            return request
        used = [e for e in endpoints if any(d.endpoint == e for d in plan.devices)]
        self.block = split_extra_args(used, plan)
        _remember_block(self.block)
        logger.info("cluster serving: layer split %s", " ".join(self.block))
        return ss.SparkServing._request_with(request, {"llama_extra_args": effective + self.block})

    async def _size(self, request, n_parallel, extras, ss) -> Tuple[Optional[float], float, float]:
        model_path = str(getattr(request, "model_path", "") or "")
        variant = getattr(request, "gguf_variant", None)
        local_file = ss.cached_repo_file(model_path, variant)
        size = ss.gguf_size_bytes(local_file)
        sidecars, _unknown = ss.sidecar_bytes(extras)
        named = {"--mmproj", "-mm", "--mmproj-url", "-mmu"}
        if not any(str(a).split("=", 1)[0] in named for a in extras):
            projector = ss.auto_projector_file(
                local_file, disable_vision = bool(getattr(request, "disable_vision", False))
            )
            sidecars += int(ss.gguf_size_bytes(projector) or 0) if projector else 0
        if size is None:
            size = await asyncio.to_thread(
                ss.remote_gguf_size_bytes, model_path, variant, getattr(request, "hf_token", None)
            )
        ctx, cache_type, cache_type_v = ss.effective_kv_settings(request, extras)
        kv = 0.0
        if local_file and ctx:
            kv = float(
                await asyncio.to_thread(
                    ss.estimate_kv_bytes, local_file, ctx, cache_type, cache_type_v
                )
                or 0.0
            )
        return (None if size is None else float(size)), kv, float(sidecars)

    async def after_load(self, llama_backend: Any) -> None:
        if self._passthrough:
            self._passthrough = False
            return
        self._pre_load = None
        argv = [
            str(a) for a in (getattr(getattr(llama_backend, "_process", None), "args", None) or [])
        ]
        if (
            self.block
            and argv[-len(self.block) :] != self.block
            and not _contains(argv, self.block)
        ):
            self.topology = "single"
            self.reason = "llama-server launched without the planned split"

    async def load_failed(self) -> None:
        if self._passthrough:
            self._passthrough = False
            return
        if self._pre_load is not None:
            self.topology, self.reason, self.plan, self.block = self._pre_load
            self._pre_load = None

    async def shutdown(self) -> None:
        for endpoint, process in list(self.workers.items()):
            try:
                await process.stop()
            except Exception:
                logger.warning("cluster serving: stopping %s failed", endpoint, exc_info = True)
        self.workers.clear()
        self.topology, self.block = "single", []

    def status(self) -> Dict[str, Any]:
        if not self.nodes and not self.config_error:
            self.refresh_nodes()
        return {
            "enabled": True,
            "mode": "cluster",
            "topology": self.topology,
            "reason": self.reason,
            "config_error": self.config_error or None,
            "nodes": [
                {
                    "host": n.host,
                    "rpc_port": n.rpc_port,
                    "managed": n.managed,
                    "worker": (
                        self.workers[n.endpoint].snapshot() if n.endpoint in self.workers else None
                    ),
                }
                for n in self.nodes
            ],
            "plan": self.plan.snapshot() if self.plan is not None else None,
            "split_args": list(self.block) or None,
            "last_error": self.last_error,
        }


def _contains(argv: List[str], block: List[str]) -> bool:
    n = len(block)
    return any(argv[i : i + n] == block for i in range(0, max(0, len(argv) - n) + 1))


_STATE: Optional[ClusterServing] = None


def state() -> ClusterServing:
    global _STATE
    if _STATE is None:
        _STATE = ClusterServing()
    return _STATE


def reset_for_tests() -> None:
    global _STATE
    _STATE = None
    _MANAGED_BLOCKS.clear()

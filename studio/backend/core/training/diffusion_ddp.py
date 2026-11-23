# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rank 0 spawns 1..N-1 as subprocesses (daemonic processes cannot have children); ranks share the
seeded global micro-batch (N GPUs at batch b = one at N*b) and all-reduce LoRA grads before
clipping; only rank 0 writes outputs; ranks 1..N-1 wait for its cache, weights, then join the
collective."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta
from typing import Any, Optional

log = logging.getLogger(__name__)

TIMEOUT_ENV = "UNSLOTH_DIFFUSION_DDP_TIMEOUT_S"
_ENV_RANK = "UNSLOTH_DIFFUSION_DDP_RANK"
_ENV_WORLD = "UNSLOTH_DIFFUSION_DDP_WORLD"
_ENV_PORT = "UNSLOTH_DIFFUSION_DDP_PORT"
_ENV_PARENT = "UNSLOTH_DIFFUSION_DDP_PARENT"
_GATE_PREFIX = "unsloth_ddp_rank0_"
_HOST = "127.0.0.1"
# Rank 0 encodes the whole dataset and loads the base before the others start; generous by design.
_LOAD_WAIT = timedelta(hours = 6)

_CURRENT: Optional["DistContext"] = None


def current() -> Optional["DistContext"]:
    """The context of a rank started by ``launch``; None in a single-device run."""
    return _CURRENT


def _timeout() -> timedelta:
    try:
        return timedelta(seconds = max(60, int(os.environ.get(TIMEOUT_ENV, "") or 1800)))
    except ValueError:
        return timedelta(seconds = 1800)


class DistContext:
    """One rank's view of the run: collectives over the LoRA parameters plus the per-run counters."""

    def __init__(
        self,
        rank: int,
        world: int,
        store = None,
        backend: str = "nccl",
    ):
        self.rank = int(rank)
        self.world = int(world)
        self.store = store
        self.backend = backend
        self.allreduce_calls = 0
        self.allreduce_bytes = 0
        self.procs: list = []
        self._monitor: Optional[threading.Thread] = None
        self._closing = False
        # Set once every rank has left the training loop together; before that a rank 0 return means abandon.
        self.finished = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def batch_slice(self, batch_size: int) -> slice:
        return slice(self.rank * batch_size, (self.rank + 1) * batch_size)

    # Gates before the first collective: "conditioning" once rank 0 has written the conditioning cache (the others then
    # read it warm), "transformer" once rank 0 has loaded its base (the others then load from page cache, in turn).
    def release(self, stage: str) -> None:
        if self.is_main and self.store is not None:
            self.store.set(f"{_GATE_PREFIX}{stage}", "1")

    def wait_for(self, stage: str) -> None:
        if not self.is_main and self.store is not None:
            self.store.wait([f"{_GATE_PREFIX}{stage}"], _LOAD_WAIT)

    def broadcast_params(self, params) -> None:
        import torch
        import torch.distributed as dist
        with torch.no_grad():
            for p in params:
                dist.broadcast(p.data, src = 0)

    def all_reduce_grads(
        self,
        params,
        loss = None,
    ):
        """Average every trainable gradient (None counts as zero) and the step loss in one flat fp32 all-reduce;
        returns the averaged loss (float), or None when ``loss`` was None."""
        import torch
        import torch.distributed as dist

        params = list(params)
        if not params:
            return None
        dev = params[0].device
        sizes = [p.numel() for p in params]
        flat = torch.empty(sum(sizes) + 1, dtype = torch.float32, device = dev)
        off = 0
        for p, n in zip(params, sizes):
            if p.grad is None:
                flat[off : off + n].zero_()
            else:
                flat[off : off + n].copy_(p.grad.reshape(-1))
            off += n
        if loss is None:
            flat[off].zero_()
        elif isinstance(loss, torch.Tensor):
            flat[off].copy_(loss.detach().reshape(()))
        else:
            flat[off].fill_(float(loss))
        dist.all_reduce(flat, op = dist.ReduceOp.SUM)
        flat.div_(self.world)
        off = 0
        for p, n in zip(params, sizes):
            g = flat[off : off + n].view_as(p)
            if p.grad is None:
                p.grad = g.to(p.dtype).clone()
            else:
                p.grad.copy_(g)
            off += n
        self.allreduce_calls += 1
        self.allreduce_bytes += flat.numel() * 4
        return None if loss is None else float(flat[off])

    def any_flag(
        self,
        flag: bool,
        device = None,
    ) -> bool:
        """True on every rank when any rank passed True (stop requests are polled on rank 0 only)."""
        import torch
        import torch.distributed as dist

        t = torch.tensor([1.0 if flag else 0.0], device = device or self._device())
        dist.all_reduce(t, op = dist.ReduceOp.SUM)
        return bool(t.item() > 0)

    def sync_rng(self) -> None:
        """Every rank continues from rank 0's torch CPU and device generator states. A resume restores each rank's
        device generator from the per-device states rank 0 saved, and rank 0 only ever advanced its own card's, so
        without this the other ranks would draw their noise from a restarted stream."""
        import torch
        import torch.distributed as dist

        dev = self._device()
        states = [torch.get_rng_state()]
        if dev.type == "cuda":
            states.append(torch.cuda.get_rng_state())
        synced = []
        for state in states:
            t = state.to(dev)
            dist.broadcast(t, src = 0)
            synced.append(t.cpu())
        torch.set_rng_state(synced[0])
        if dev.type == "cuda":
            torch.cuda.set_rng_state(synced[1])

    def gather(self, obj) -> list:
        import torch.distributed as dist

        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def agree_precision(self, mode: str) -> str:
        """Every rank trains the same base: the most memory-frugal mode any rank resolved (auto reads its own card)."""
        order = ("nf4", "int8", "fp8", "mxfp8", "bf16")
        picks = self.gather(mode)
        # Kept for the caller's warning: which rank resolved what.
        self.precision_picks = list(picks)
        known = [p for p in picks if p in order]
        if len(set(picks)) == 1 or not known:
            return picks[0]
        return min(known, key = order.index)

    # The base every rank actually trains once its conversions ran (their fallbacks are rank-local).
    _EFFECTIVE_PRECISIONS = ("nf4", "int8", "int8_w8a8", "fp8", "mxfp8", "bf16")

    def check_effective_precision(
        self,
        effective: str,
        layers: Optional[tuple] = None,
    ) -> None:
        """Refuse on every rank when the ranks ended on different bases. agree_precision fixes the pick, but the fp8 /
        W8A8 conversions fall back per rank (bf16, an nf4 reload, torchao weight-only int8); a mixed run would
        all-reduce LoRA grads taken through different base numerics without any error. One int64 per rank (three with
        ``layers``).
        ``layers``: ``lowp_layers_digest`` of the converted base; a layer that failed on one rank only stays bf16
        there under the same label, so the converted set is compared too."""
        import torch
        import torch.distributed as dist

        names = self._EFFECTIVE_PRECISIONS
        code = names.index(effective) if effective in names else len(names)
        values = [code] if layers is None else [code, int(layers[0]), int(layers[1])]
        mine = torch.tensor(values, dtype = torch.int64, device = self._device())
        out = [torch.empty_like(mine) for _ in range(self.world)]
        dist.all_gather(out, mine)
        rows = [[int(v) for v in t.reshape(-1).tolist()] for t in out]
        codes = [row[0] for row in rows]
        if len(set(codes)) > 1:
            picks = ", ".join(
                f"rank {r}: {names[c] if c < len(names) else 'unknown'}"
                for r, c in enumerate(codes)
            )
            raise RuntimeError(
                "Multi-GPU training stopped: the GPUs ended up training different base precisions after "
                f"conversion ({picks}). Set base_precision to one of them explicitly, or train on one GPU."
            )
        if layers is not None and len({tuple(row[1:]) for row in rows}) > 1:
            counts = ", ".join(f"rank {r}: {row[2]}" for r, row in enumerate(rows))
            raise RuntimeError(
                f"Multi-GPU training stopped: the GPUs converted different layers to {effective} (layers converted: "
                f"{counts}; a layer that fails on one GPU stays bf16 there). Use base_precision='bf16' or 'nf4', "
                "or train on one GPU."
            )

    def params_checksum(self, params) -> float:
        import torch
        with torch.no_grad():
            return float(sum(p.detach().double().abs().sum() for p in params))

    def _device(self):
        import torch
        if self.backend == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def stats(self) -> dict:
        return {
            "world_size": self.world,
            "allreduce_calls": self.allreduce_calls,
            "allreduce_mib": round(self.allreduce_bytes / (1 << 20), 1),
        }

    def close(self, failed: bool = False) -> None:
        """Rank 0: wait for (or, after a failure, stop) the other ranks, then leave the process group."""
        global _CURRENT
        self._closing = True
        # Rank 0 returning before the ranks finished together (a stop while caching) leaves them waiting on it.
        failed = failed or not self.finished
        for proc in self.procs:
            try:
                if failed:
                    proc.terminate()
                proc.wait(timeout = 10 if failed else 120)
            except Exception:  # noqa: BLE001 -- a stuck rank is killed, never waited on forever
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if not failed:
            # Off-thread and bounded: a communicator whose peers already exited must not hold up the trainer child.
            def _destroy():
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                except Exception:  # noqa: BLE001 -- teardown never fails a finished run
                    pass

            t = threading.Thread(target = _destroy, name = "diffusion-ddp-destroy", daemon = True)
            t.start()
            t.join(timeout = 30)
        if _CURRENT is self:
            _CURRENT = None


def lowp_layers_digest(model) -> tuple[int, int]:
    """(signed int64 digest, count) of the sorted name and format of every low-precision layer in ``model``."""
    import hashlib

    rows = sorted(
        f"{fqn}={mode}"
        for fqn, mod in model.named_modules()
        if (mode := getattr(mod, "_studio_lowp", None))
    )
    digest = hashlib.sha256("\n".join(rows).encode()).digest()[:8]
    return int.from_bytes(digest, "little", signed = True), len(rows)


def plan_world(cfg, device: str) -> tuple[int, Optional[str]]:
    """How many GPUs this run trains on, and why fewer than requested (None when as requested)."""
    want = int(getattr(cfg, "num_gpus", 1) or 1)
    if want <= 1:
        return 1, None
    if device != "cuda":
        return 1, "multi-GPU training needs CUDA GPUs; training on one device."
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_nccl_available():
            return (
                1,
                "multi-GPU training needs NCCL (or RCCL on ROCm), which this torch build lacks; training on one GPU.",
            )
        count = torch.cuda.device_count()
        props = [torch.cuda.get_device_properties(i) for i in range(min(want, count))]
    except Exception as exc:  # noqa: BLE001 -- an unprobeable host trains on one GPU
        return 1, f"multi-GPU training unavailable ({exc}); training on one GPU."
    from core.training.diffusion_block_swap import block_swap_request

    try:
        swap = block_swap_request(cfg)
    except ValueError:
        swap = None
    if swap:
        return 1, "multi-GPU training does not combine with block swap; training on one GPU."
    if count < 2:
        return 1, f"num_gpus={want} but only {count} GPU is visible; training on one GPU."
    if len({(p.name, p.total_memory) for p in props}) > 1:
        return 1, "multi-GPU training needs identical GPUs; these differ, so training on one GPU."
    world = min(want, count)
    if world < want:
        return world, f"num_gpus={want} but {count} GPUs are visible; training on {world}."
    return world, None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_HOST, 0))
        return s.getsockname()[1]


def _loopback_store(world: int):
    """Rank 0's store, listening on 127.0.0.1 only. A TCPStore server given just a port listens on every interface,
    so it hands the store a socket already bound to loopback (also no free-port race); the legacy backend is the one
    that accepts a listening fd on every supported torch. Returns (store, port)."""
    import torch.distributed as dist

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((_HOST, 0))
        sock.listen(max(16, 2 * world))
        port = sock.getsockname()[1]
        store = dist.TCPStore(
            _HOST,
            port,
            world,
            is_master = True,
            timeout = _timeout(),
            wait_for_workers = False,
            master_listen_fd = sock.fileno(),
            use_libuv = False,
        )
    except Exception as exc:  # noqa: BLE001 -- a torch without listen-fd support keeps the old store
        sock.close()
        log.warning(
            "multi-GPU training: loopback-only store unavailable (%s); using the default bind.", exc
        )
        port = _free_port()
        store = dist.TCPStore(
            _HOST, port, world, is_master = True, timeout = _timeout(), wait_for_workers = False
        )
        return store, port
    # The store's server owns the descriptor now.
    sock.detach()
    return store, port


def _config_payload(cfg) -> str:
    import dataclasses

    data = dataclasses.asdict(cfg)
    # The others must not each spawn ranks of their own.
    data["num_gpus"] = 1
    return json.dumps(data)


def _pin_loopback(env) -> None:
    """Every rank runs on this host, so NCCL's bootstrap sockets need only loopback, not every interface; a user's own
    NCCL_SOCKET_IFNAME wins."""
    if sys.platform.startswith("linux"):
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")


def _backend_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def launch(
    cfg,
    world: int,
    on_failure = None,
) -> DistContext:
    """Rank 0 (this process): start ranks 1..world-1 on GPUs 1..world-1 and join the process group with them.
    ``on_failure(message)`` runs when another rank dies, before this process exits (it would hang in a collective)."""
    global _CURRENT
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    store, port = _loopback_store(world)
    ctx = DistContext(0, world, store)
    payload = _config_payload(cfg)
    # Before the ranks' env is copied and before this process creates its communicator.
    _pin_loopback(os.environ)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_backend_dir(), *filter(None, [env.get("PYTHONPATH")])])
    for rank in range(1, world):
        env_r = {
            **env,
            _ENV_RANK: str(rank),
            _ENV_WORLD: str(world),
            _ENV_PORT: str(port),
            _ENV_PARENT: str(os.getpid()),
        }
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from core.training.diffusion_ddp import worker_main; worker_main()",
            ],
            stdin = subprocess.PIPE,
            env = env_r,
            # The caller's directory: rank 0 resolved relative data / model / cache / resume paths against it.
            cwd = os.getcwd(),
        )
        proc.stdin.write(payload.encode())
        proc.stdin.close()
        ctx.procs.append(proc)

    def _watch():
        while not ctx._closing:
            for rank, proc in enumerate(ctx.procs, start = 1):
                code = proc.poll()
                if code not in (None, 0) and not ctx._closing:
                    msg = f"multi-GPU training: rank {rank} exited with code {code}; stopping the run."
                    log.error(msg)
                    if on_failure is not None:
                        try:
                            on_failure(msg)
                        except Exception:  # noqa: BLE001
                            pass
                    time.sleep(2)
                    os._exit(1)
            time.sleep(1)

    ctx._monitor = threading.Thread(target = _watch, name = "diffusion-ddp-watch", daemon = True)
    ctx._monitor.start()
    try:
        dist.init_process_group("nccl", store = store, rank = 0, world_size = world, timeout = _timeout())
    except BaseException:
        # Caller never gets ctx: stop ranks and the watcher here, else the watcher os._exit()s.
        ctx.close(failed = True)
        raise
    _CURRENT = ctx
    return ctx


def worker_main() -> None:
    """Entry of ranks 1..N-1: read the config from stdin, join the group, train without events, exit."""
    global _CURRENT
    rank = int(os.environ[_ENV_RANK])
    world = int(os.environ[_ENV_WORLD])
    port = int(os.environ[_ENV_PORT])
    parent = int(os.environ.get(_ENV_PARENT, "0") or 0)
    payload = sys.stdin.read()

    def _parent_watch():
        while True:
            if parent and os.getppid() != parent:
                os._exit(1)
            time.sleep(2)

    threading.Thread(target = _parent_watch, name = "diffusion-ddp-parent", daemon = True).start()
    try:
        from utils.native_tls import activate_native_tls
        activate_native_tls()
    except Exception:  # noqa: BLE001 -- network tweaks never stop a rank
        pass
    try:
        from utils.happy_eyeballs import activate_happy_eyeballs
        activate_happy_eyeballs()
    except Exception:  # noqa: BLE001 -- network tweaks never stop a rank
        pass
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    store = dist.TCPStore(_HOST, port, world, is_master = False, timeout = _timeout())
    dist.init_process_group("nccl", store = store, rank = rank, world_size = world, timeout = _timeout())
    _CURRENT = DistContext(rank, world, store)
    from core.training.diffusion_dit_trainer import run_dit_lora_training
    from core.training.diffusion_train_common import _config_from_dict

    cfg = _config_from_dict(json.loads(payload))
    code = 0
    try:
        run_dit_lora_training(cfg)
    except BaseException:
        log.exception("multi-GPU training rank %d failed", rank)
        code = 1
    # No communicator teardown: NCCL's finalize waits on peers that may already be gone (rank 0 is still exporting
    # the adapter), and process exit frees the same resources.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def activate_for_tests(
    rank: int,
    world: int,
    store = None,
    backend: str = "gloo",
) -> DistContext:
    """Install a context for a process group the caller already initialised (CPU tests on gloo)."""
    global _CURRENT
    _CURRENT = DistContext(rank, world, store, backend = backend)
    return _CURRENT


def deactivate() -> None:
    global _CURRENT
    _CURRENT = None


__all__: list[Any] = [
    "DistContext",
    "current",
    "plan_world",
    "launch",
    "worker_main",
    "activate_for_tests",
    "deactivate",
]

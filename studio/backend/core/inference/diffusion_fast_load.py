# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cold-start I/O for the image load path, byte-for-byte neutral: ``start_load_prefetch`` warms the page
cache with the files the load reads next (parallel slice reads beat mmap page faults); ``fast_upload``
moves plain host tensors through a pinned ring and hands them to the existing ``.to`` via a
``TorchFunctionMode``, so placement semantics stay the module's own. Kill switches
``UNSLOTH_DIFFUSION_PREFETCH=0`` / ``UNSLOTH_DIFFUSION_FAST_UPLOAD=0``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from typing import Any, Iterable, Iterator, Optional, Sequence

PREFETCH_ENV = "UNSLOTH_DIFFUSION_PREFETCH"
FAST_UPLOAD_ENV = "UNSLOTH_DIFFUSION_FAST_UPLOAD"
_SWITCH_OFF = ("0", "off", "false", "no")

_PREFETCH_THREADS = 8
_PREFETCH_READ_BYTES = 64 << 20
_PREFETCH_MIN_BYTES = 256 << 20
_PREFETCH_THREAD_PREFIX = "unsloth-image-prefetch"

_UPLOAD_THREADS = 8
_UPLOAD_CHUNK_BYTES = 16 << 20
_UPLOAD_BUFFERS = 16
_UPLOAD_MIN_TENSOR_BYTES = 1 << 20
_UPLOAD_MIN_TOTAL_BYTES = 256 << 20

# variant=None: from_pretrained never opens *.fp16.safetensors-style twins.
_DEFAULT_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|model)(?:-\d{5}-of-\d{5})?\.safetensors$"
)
_DENOISER_COMPONENTS = ("transformer", "unet")


def _switch_on(env: str) -> bool:
    return (os.environ.get(env) or "").strip().lower() not in _SWITCH_OFF


def prefetch_enabled() -> bool:
    return _switch_on(PREFETCH_ENV)


def fast_upload_enabled() -> bool:
    return _switch_on(FAST_UPLOAD_ENV)


class LoadPrefetch:
    """Handle on a running prefetch: ``stop`` ends the reads at the next request boundary."""

    def __init__(
        self, files: list[str], need: int, threads: list[threading.Thread], stop: threading.Event
    ):
        self.files = files
        self.need = need
        self.threads = threads
        self._stop = stop

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> bool:
        for thread in self.threads:
            thread.join(timeout)
        return not any(thread.is_alive() for thread in self.threads)


def _uncached_bytes(path: str) -> int:
    from .video_ltx2 import _uncached_bytes as uncached
    return uncached(path)


def _available_host_mib() -> Optional[int]:
    try:
        from .diffusion_memory import _available_system_memory_mib
        return _available_system_memory_mib()
    except Exception:  # noqa: BLE001
        return None


def _on_rotational_disk(path: str) -> bool:
    """True only when sysfs says the file's block device spins: parallel slice reads would seek-thrash it."""
    try:
        st = os.stat(path)
        base = f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
        for queue in (os.path.join(base, "queue"), os.path.join(base, "..", "queue")):
            flag = os.path.join(queue, "rotational")
            if os.path.isfile(flag):
                with open(flag, encoding = "utf-8") as handle:
                    return handle.read().strip() == "1"
    except Exception:  # noqa: BLE001 - unknown (non-Linux, overlay, network): not rotational
        pass
    return False


def _slices(size: int, parts: int) -> list[tuple[int, int]]:
    """``parts`` contiguous [start, end) ranges covering ``size``, aligned to 1 MiB."""
    align = 1 << 20
    step = -(-size // parts)
    step = -(-step // align) * align
    return [(start, min(size, start + step)) for start in range(0, size, step)]


def start_prefetch(
    paths: Sequence[str],
    *,
    logger: Any = None,
    threads: int = _PREFETCH_THREADS,
    min_bytes: int = _PREFETCH_MIN_BYTES,
) -> Optional[LoadPrefetch]:
    """Read the uncached parts of ``paths`` into the page cache on worker threads, in the given order.

    None when switched off, when nothing needs it, or when the uncached bytes exceed half the available
    host RAM (the reads would evict what the load itself needs). Never raises."""
    if not prefetch_enabled():
        return None
    try:
        seen: set[str] = set()
        files: list[str] = []
        for path in paths:
            if not path:
                continue
            real = os.path.realpath(path)
            if real in seen or not os.path.isfile(real):
                continue
            seen.add(real)
            files.append(real)
        pending = []
        need = 0
        for path in files:
            missing = _uncached_bytes(path)
            if missing > 0:
                pending.append(path)
                need += missing
        if not pending or need < min_bytes:
            return None
        if any(_on_rotational_disk(path) for path in pending):
            if logger is not None:
                logger.info("diffusion.prefetch: skipped, the weights are on a rotational disk")
            return None
        available = _available_host_mib()
        if available is None or need > (int(available) << 20) // 2:
            if logger is not None:
                logger.info(
                    "diffusion.prefetch: skipped, %.1f GiB uncached against %s MiB available host RAM",
                    need / 2**30,
                    available,
                )
            return None
        threads = max(1, int(threads))
        # Thread i reads slice i of every file, files in load order, so the first file is warm first.
        plan: list[list[tuple[str, int, int]]] = [[] for _ in range(threads)]
        for path in pending:
            for index, (start, end) in enumerate(_slices(os.path.getsize(path), threads)):
                plan[index].append((path, start, end))
    except Exception as exc:  # noqa: BLE001 - a prefetch is only a hint
        if logger is not None:
            logger.debug("diffusion.prefetch: skipped (%r)", exc)
        return None

    stop = threading.Event()

    def _worker(jobs: list[tuple[str, int, int]]) -> None:
        buffer = bytearray(_PREFETCH_READ_BYTES)
        view = memoryview(buffer)
        try:
            for path, start, end in jobs:
                if stop.is_set():
                    return
                with open(path, "rb", buffering = 0) as handle:
                    handle.seek(start)
                    at = start
                    while at < end and not stop.is_set():
                        got = handle.readinto(view[: min(_PREFETCH_READ_BYTES, end - at)])
                        if not got:
                            break
                        at += got
        except Exception:  # noqa: BLE001 - a prefetch is only a hint
            pass

    workers = []
    for index, jobs in enumerate(plan):
        if not jobs:
            continue
        thread = threading.Thread(
            target = _worker, args = (jobs,), name = f"{_PREFETCH_THREAD_PREFIX}-{index}", daemon = True
        )
        thread.start()
        workers.append(thread)
    if logger is not None:
        logger.info(
            "diffusion.prefetch: warming %.1f GiB of %d file(s) on %d threads",
            need / 2**30,
            len(pending),
            len(workers),
        )
    return LoadPrefetch(pending, need, workers, stop)


def stop_prefetch(handle: Optional[LoadPrefetch]) -> None:
    if handle is not None:
        try:
            handle.stop()
        except Exception:  # noqa: BLE001
            pass


def _snapshot_dir(
    base: Optional[str], base_local_dir: Optional[str], cache_dir: Optional[str]
) -> Optional[str]:
    if base_local_dir and os.path.isdir(base_local_dir):
        return base_local_dir
    if base and os.path.isdir(base):
        return base
    if not base:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        for root in (cache_dir, None) if cache_dir else (None,):
            hit = try_to_load_from_cache(base, "model_index.json", cache_dir = root)
            if isinstance(hit, str):
                return os.path.dirname(hit)
    except Exception:  # noqa: BLE001 - uncached (the load downloads it) or an unreadable cache
        return None
    return None


def pipeline_component_files(
    snapshot: Optional[str],
    *,
    skip_denoiser: bool,
    skip_components: Iterable[str] = (),
) -> list[str]:
    """The weight files ``from_pretrained`` reads from a local diffusers snapshot: every component
    ``model_index.json`` names, text encoders first, then the VAE, then the denoiser. Cache only."""
    if not snapshot:
        return []
    try:
        with open(os.path.join(snapshot, "model_index.json"), "r", encoding = "utf-8") as handle:
            index = json.load(handle)
    except Exception:  # noqa: BLE001
        return []
    components = [
        name
        for name, value in index.items()
        if not name.startswith("_")
        and isinstance(value, (list, tuple))
        and len(value) == 2
        and value[0]
    ]

    def _rank(name: str) -> int:
        if name.startswith("text_encoder"):
            return 0
        if name in _DENOISER_COMPONENTS:
            return 2
        return 1

    skip = set(skip_components or ())
    files: list[str] = []
    for name in sorted(components, key = lambda n: (_rank(n), n)):
        if skip_denoiser and name in _DENOISER_COMPONENTS:
            continue
        if name in skip:
            continue
        folder = os.path.join(snapshot, name)
        if not os.path.isdir(folder):
            continue
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            continue
        for entry in entries:
            if _DEFAULT_WEIGHT_RE.match(entry):
                path = os.path.join(folder, entry)
                if os.path.isfile(path):
                    files.append(path)
    return files


def te_precast_components(fam: Any, base: str, te_quant_mode: Any, target: Any) -> frozenset:
    """Encoders a hosted pre-cast checkpoint replaces; a runtime cast still reads the dense shards."""
    try:
        from .diffusion_te_prequant import te_prequant_sources_for_base
        return frozenset(
            te_prequant_sources_for_base(fam, base, te_quant_mode = te_quant_mode, target = target)
        )
    except Exception:  # noqa: BLE001 - a prefetch is only a hint
        return frozenset()


def start_load_prefetch(
    fam: Any,
    base: Optional[str],
    *,
    base_local_dir: Optional[str] = None,
    prequant_scheme: Optional[str] = None,
    prequant_path_override: Optional[str] = None,
    prequant_base_repo: Optional[str] = None,
    text_encoders_replaced: Iterable[str] = (),
    cache_dir: Optional[str] = None,
    logger: Any = None,
) -> Optional[LoadPrefetch]:
    """Prefetch what a diffusers pipeline load of ``base`` will read. Never raises."""
    if not prefetch_enabled():
        return None
    try:
        paths: list[str] = []
        seeded = False
        if prequant_scheme:
            from .diffusion_denoiser_prequant import denoiser_prequant_source
            source = denoiser_prequant_source(
                fam,
                prequant_scheme,
                base_repo = prequant_base_repo or base,
                path_override = prequant_path_override,
            )
            if source is not None:
                location = None
                if getattr(source, "kind", None) == "repo":
                    from .diffusion_prequant import cached_checkpoint_path
                    location = cached_checkpoint_path(source, cache_dir = cache_dir)
                else:
                    location = getattr(source, "location", None)
                if location and os.path.isfile(str(location)):
                    paths.append(str(location))
                    seeded = True
        paths += pipeline_component_files(
            _snapshot_dir(base, base_local_dir, cache_dir),
            skip_denoiser = seeded,
            skip_components = text_encoders_replaced,
        )
        return start_prefetch(paths, logger = logger)
    except Exception as exc:  # noqa: BLE001 - a prefetch is only a hint
        if logger is not None:
            logger.debug("diffusion.prefetch: skipped (%r)", exc)
        return None


def _upload_candidates(modules: Sequence[Any], device: Any) -> list[Any]:
    """Plain, dense, contiguous CPU parameters and buffers worth a ring slot, each tensor once."""
    import torch

    out: list[Any] = []
    seen: set[int] = set()
    for module in modules:
        if module is None or not hasattr(module, "parameters"):
            continue
        tensors = list(module.parameters()) + list(module.buffers())
        for tensor in tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                continue  # subclasses (torchao) keep their own .to
            if tensor.device.type != "cpu" or tensor.is_sparse or tensor.is_quantized:
                continue
            if not tensor.is_contiguous() or tensor.storage_offset() < 0:
                continue
            if tensor.numel() * tensor.element_size() < _UPLOAD_MIN_TENSOR_BYTES:
                continue
            out.append(tensor)
    return out


def _ring_copy(sources: list[Any], device: Any) -> list[Any]:
    """Device copies of ``sources`` (byte-identical), read through a pinned ring on worker threads."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    targets = [torch.empty_like(src, device = device) for src in sources]
    jobs = []
    for src, dst in zip(sources, targets):
        nbytes = src.numel() * src.element_size()
        src8 = src.detach().reshape(-1).view(torch.uint8)
        dst8 = dst.view(-1).view(torch.uint8)
        for offset in range(0, nbytes, _UPLOAD_CHUNK_BYTES):
            jobs.append((src8, dst8, offset, min(_UPLOAD_CHUNK_BYTES, nbytes - offset)))
    if not jobs:
        return targets
    buffers = max(1, min(_UPLOAD_BUFFERS, len(jobs)))
    staging = [
        torch.empty(_UPLOAD_CHUNK_BYTES, dtype = torch.uint8, pin_memory = True) for _ in range(buffers)
    ]
    done: list[Any] = [None] * buffers
    stream = torch.cuda.Stream(device = device)

    def _fill(slot: int, src8: Any, offset: int, size: int) -> None:
        event = done[slot]
        if event is not None:
            event.synchronize()
        staging[slot][:size].copy_(src8[offset : offset + size])

    try:
        with ThreadPoolExecutor(
            max_workers = _UPLOAD_THREADS, thread_name_prefix = "unsloth-upload"
        ) as pool:
            pending = {}
            for index in range(buffers):
                src8, _dst8, offset, size = jobs[index]
                pending[index] = pool.submit(_fill, index, src8, offset, size)
            with torch.cuda.stream(stream):
                for index, (src8, dst8, offset, size) in enumerate(jobs):
                    slot = index % buffers
                    pending.pop(index).result()
                    dst8[offset : offset + size].copy_(staging[slot][:size], non_blocking = True)
                    event = torch.cuda.Event()
                    event.record(stream)
                    done[slot] = event
                    following = index + buffers
                    if following < len(jobs):
                        nsrc8, _ndst8, noffset, nsize = jobs[following]
                        pending[following] = pool.submit(_fill, slot, nsrc8, noffset, nsize)
            stream.synchronize()
    finally:
        try:
            stream.synchronize()
        except Exception:  # noqa: BLE001
            pass
        del staging
        try:
            torch._C._host_emptyCache()
        except Exception:  # noqa: BLE001 - older torch keeps the ring cached
            pass
    # The default stream must not read the targets before the side stream wrote them.
    torch.cuda.current_stream(device).wait_stream(stream)
    return targets


def _same_device(a: Any, b: Any) -> bool:
    import torch

    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    if a.type != "cuda":
        return a == b
    index_a = a.index if a.index is not None else torch.cuda.current_device()
    index_b = b.index if b.index is not None else torch.cuda.current_device()
    return index_a == index_b


@contextlib.contextmanager
def fast_upload(
    modules: Sequence[Any],
    device: Any,
    *,
    logger: Any = None,
) -> Iterator[int]:
    """Inside the block, ``Tensor.to(device)`` on the modules' plain CPU tensors returns a ring-uploaded copy.

    Wrap the existing placement call (``module.to(device)`` / ``pipe.to(device)``): it still decides
    what moves and how parameters are re-wrapped; only the bytes arrive faster. Yields how many tensors
    were staged (0 = the block runs exactly as without this wrapper)."""
    staged: dict[int, tuple[Any, Any]] = {}
    mode = None
    try:
        if fast_upload_enabled():
            import torch
            target = torch.device(device)
            if target.type == "cuda" and torch.cuda.is_available():
                if target.index is None:
                    target = torch.device("cuda", torch.cuda.current_device())
                sources = _upload_candidates(modules, target)
                total = sum(t.numel() * t.element_size() for t in sources)
                if sources and total >= _UPLOAD_MIN_TOTAL_BYTES:
                    with torch.no_grad():
                        copies = _ring_copy(sources, target)
                    staged = {id(src): (src, dst) for src, dst in zip(sources, copies)}
                    mode = _StagedTo(staged, target)
                    if logger is not None:
                        logger.info(
                            "diffusion.fast_upload: staged %.1f GiB in %d tensor(s) onto %s",
                            total / 2**30,
                            len(sources),
                            target,
                        )
    except Exception as exc:  # noqa: BLE001 - the stock .to() still runs inside the block
        if logger is not None:
            logger.warning("diffusion.fast_upload: falling back to the stock copy (%s)", exc)
        staged, mode = {}, None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    if mode is None:
        yield 0
        return
    with mode:
        yield len(staged)
    staged.clear()


def _staged_to_class():
    from torch.overrides import TorchFunctionMode
    class _StagedToMode(TorchFunctionMode):
        """Answers ``Tensor.to`` for a staged source with its device copy; every other call runs as usual."""

        def __init__(self, staged: dict[int, tuple[Any, Any]], device: Any):
            super().__init__()
            self.staged = staged
            self.device = device

        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            import torch

            kwargs = kwargs or {}
            if func is torch.Tensor.to and args:
                hit = self.staged.get(id(args[0]))
                if hit is not None and hit[0] is args[0]:
                    try:
                        device, dtype, _non_blocking, memory_format = torch._C._nn._parse_to(
                            *args[1:], **kwargs
                        )
                    except Exception:  # noqa: BLE001
                        device = None
                    if (
                        device is not None
                        and _same_device(device, self.device)
                        and (dtype is None or dtype == args[0].dtype)
                        and memory_format is None
                        and not kwargs.get("copy", False)
                    ):
                        out = hit[1]
                        if args[0].requires_grad and torch.is_grad_enabled():
                            out = out.requires_grad_(True)
                        return out
            return func(*args, **kwargs)

    return _StagedToMode


def _StagedTo(staged: dict[int, tuple[Any, Any]], device: Any) -> Any:
    return _staged_to_class()(staged, device)

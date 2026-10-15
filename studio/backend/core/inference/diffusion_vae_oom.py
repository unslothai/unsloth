# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recover a VAE decode that runs out of memory by offloading, never by shrinking tiles.

Wraps the instance ``_decode`` (under the decode-level wrappers, over the tiled decode). On an OOM, each step runs only
if the previous one still runs out of memory:

a. Free the decode's device without changing the math: Studio's resident offload groups (text encoders, denoiser
   blocks) stream again, captured CUDA graphs are dropped, and every other pipeline component still whole on the card
   moves to the host. The SAME decode is retried, so an untiled decode stays untiled and the output is unchanged.
b. An untiled decode retries tiled at the VAE's stock tile sizes: ComfyUI's single tiled retry.
c. The VAE's own weights stream from the host per layer (diffusers leaf-level group offload), still at stock tiles.

Tile sizes are never touched: tiles under the stock size bring back visible seams. Everything moved goes back where
it was once the decode returns or fails. Kill switch ``UNSLOTH_VAE_OOM_RETRY=0``.
"""

from __future__ import annotations

import functools
import gc
import os
import time
import weakref
from typing import Any, Callable, Optional

from . import decode_layers

OOM_RETRY_ENV = "UNSLOTH_VAE_OOM_RETRY"
# Host RAM left free when a component moves off the card for a decode.
HOST_MARGIN_MIB = 2048
_SESSION_ATTR = "_unsloth_vae_oom_session"
_PIPE_ATTR = "_unsloth_vae_oom_pipe"


def oom_retry_disabled() -> bool:
    return os.environ.get(OOM_RETRY_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _is_oom_one(exc: BaseException) -> bool:
    if any(cls.__name__ in ("OutOfMemoryError", "OutOfMemoryError_") for cls in type(exc).__mro__):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text or "tried to allocate" in text
    )


def is_oom(exc: BaseException) -> bool:
    """An allocator failure, also when wrapped: torch.compile re-raises an OOM inside a compiler error whose own
    message is generic, so ``__cause__`` / ``__context__`` are walked (bounded)."""
    seen: set = set()
    cur: Optional[BaseException] = exc
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if _is_oom_one(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def tile_attrs(vae: Any) -> dict:
    """The tile geometry a VAE exposes; the retry never changes it."""
    out = {}
    for axis in ("size", "height", "width"):
        for kind in ("tile_sample_min", "tile_latent_min", "tile_sample_stride", "tile_sample_min_overlap"):
            name = f"{kind}_{axis}"
            value = vae.__dict__.get(name, getattr(type(vae), name, None))
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                out[name] = value
    return out


def _empty_cache(device: Any) -> None:
    try:
        import torch

        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _release_cache(z: Any, vae: Any = None) -> None:
    try:
        # A causal decoder's frame cache (Wan / Qwen-Image) still holds the failed attempt's tensors.
        clear = getattr(vae, "clear_cache", None)
        if callable(clear):
            clear()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    _empty_cache(getattr(z, "device", None))


def _compiling() -> bool:
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _outer_gate_owns_untiled(vae: Any) -> bool:
    """video_vae_untiled's gate turns tiling off for its own attempt and retries tiled itself. Outer wrappers (the ROCm
    bf16 decode) expose it only through ``__wrapped__``."""
    fn = vae.__dict__.get("decode")
    for _ in range(8):
        if fn is None:
            return False
        if getattr(fn, "_unsloth_untiled_decode", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


# -- step (a): free the decode's device ------------------------------------------------------------------------


def _module_tensors(module: Any) -> list:
    return [*module.parameters(), *module.buffers()]


def _hooked(module: Any) -> bool:
    """Placed by an offload mechanism (accelerate model / sequential offload, diffusers group offload): that mechanism
    owns where its weights live."""
    for m in module.modules():
        if getattr(m, "_hf_hook", None) is not None:
            return True
        # Other diffusers hooks (first-block cache, an emptied registry) leave placement alone.
        hooks = getattr(getattr(m, "_diffusers_hook", None), "hooks", None) or {}
        if any("group_offloading" in str(name) for name in hooks):
            return True
    return False


def _whole_on(module: Any, device: Any) -> bool:
    tensors = _module_tensors(module)
    return bool(tensors) and all(t.device == device for t in tensors)


def _nbytes(module: Any) -> int:
    seen: set = set()
    total = 0
    for t in _module_tensors(module):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += int(t.numel()) * int(t.element_size())
    return total


def _host_free_mib() -> Optional[int]:
    try:
        from .diffusion_memory import _available_system_memory_mib
        return _available_system_memory_mib()
    except Exception:  # noqa: BLE001
        return None


def _holds_torchao(module: Any) -> bool:
    try:
        from .diffusion_memory import _holds_torchao_weights
        return bool(_holds_torchao_weights(module))
    except Exception:  # noqa: BLE001
        return False


def _return_to(module: Any, device: Any) -> None:
    try:
        module.to(device)
    except Exception:  # noqa: BLE001 - a weight Module.to cannot move: tensor by tensor, as the offload path does
        from .diffusion_memory import _return_module_tensors
        _return_module_tensors(module, device)


def _to_host(module: Any) -> None:
    module.to("cpu")


def _placement(module: Any, device: Any) -> str:
    """How a component sits on the card: "plain" (whole there, no hooks), "torchao_resident" (whole there, torchao
    weights, no hooks), "groups" (resident offload groups), or "managed" (an offload mechanism places it)."""
    if getattr(module, "_unsloth_resident_room", None):
        return "groups"
    if _hooked(module):
        return "managed"
    if not _whole_on(module, device):
        return "managed"
    return "torchao_resident" if _holds_torchao(module) else "plain"


def _call_restore(restore: Callable) -> None:
    try:
        restore(strict = True)
    except TypeError as exc:  # a restore without the strict keyword (older release path)
        if "strict" not in str(exc):
            raise
        restore()


_DEFERRED_ATTR = "_unsloth_vae_oom_deferred"
_RECORD_ATTR = "_unsloth_vae_oom_record"


class DeferredReturn:
    """The way back for state a decode's restore could not undo (the card filled up meanwhile).

    Registered on the module that has to come back: a forward pre-hook retries every pending undo before the module
    runs, and the module carries an accelerate-shaped ``_hf_hook`` whose ``execution_device`` is the card, so
    ``DiffusionPipeline._execution_device`` still resolves the card while the module waits on the host (diffusers
    reads that attribute for offloaded components; ``apply_forward_hook`` decode methods call ``pre_forward``). Each
    pending undo leaves only once it has succeeded; when none is left the hooks go. ``required``: the module cannot run
    where it is (weights on the host, no offload hooks), so a retry that still runs out of memory raises; otherwise
    (the VAE streams through its offload hooks, resident groups stream) it runs as it is and retries next time."""

    no_grad = False

    def __init__(self, module: Any, device: Any, required: bool) -> None:
        self.module_ref = weakref.ref(module)
        self.execution_device = device
        self.required = required
        self.pending: list = []  # (label, undo)
        self.handle = module.register_forward_pre_hook(self._pre_hook)
        self.owns_hf_hook = "_hf_hook" not in module.__dict__
        if self.owns_hf_hook:
            module._hf_hook = self
        module.__dict__[_DEFERRED_ATTR] = self

    # accelerate ModelHook surface
    def init_hook(self, module: Any) -> Any:
        return module

    def pre_forward(self, module: Any, *args: Any, **kwargs: Any) -> tuple:
        self.attempt()
        return args, kwargs

    def post_forward(self, module: Any, output: Any) -> Any:
        return output

    def detach_hook(self, module: Any) -> Any:
        return module

    def _pre_hook(self, module: Any, args: Any) -> None:
        self.attempt()

    def add(self, label: str, undo: Callable[[], None]) -> None:
        self.pending.append((label, undo))

    def attempt(self) -> bool:
        """Retry every pending undo in order; True once none is left."""
        while self.pending:
            label, undo = self.pending[0]
            try:
                undo()
            except Exception as exc:  # noqa: BLE001
                if not self.required:
                    return False  # it runs where it is (streamed); retried next time
                # its weights are on the host and nothing would move them: running it would fail on the wrong
                # device, so say why (an out-of-memory keeps "out of memory" in the text for the OOM handlers)
                module = self.module_ref()
                raise RuntimeError(
                    f"{type(module).__name__} ({label.split(':', 1)[-1]}) is still on the host after a VAE decode "
                    f"ran out of memory and cannot return to {self.execution_device}: "
                    + ("out of memory; free GPU memory and retry" if is_oom(exc) else f"{type(exc).__name__}: {exc}")
                ) from exc
            self.pending.pop(0)
        self.detach()
        return True

    def detach(self) -> None:
        module = self.module_ref()
        try:
            self.handle.remove()
        except Exception:  # noqa: BLE001
            pass
        if module is None:
            return
        if self.owns_hf_hook and module.__dict__.get("_hf_hook") is self:
            del module.__dict__["_hf_hook"]
        if module.__dict__.get(_DEFERRED_ATTR) is self:
            del module.__dict__[_DEFERRED_ATTR]


def _defer(module: Any, device: Any, label: str, undo: Callable[[], None], required: bool, owner: Any) -> None:
    """Hand ``undo`` to ``module``'s deferred return (one per module); ``owner`` (the VAE) records the module so
    uninstall can settle it."""
    deferred = module.__dict__.get(_DEFERRED_ATTR)
    if deferred is None:
        deferred = DeferredReturn(module, device, required)
    deferred.required = deferred.required or required
    deferred.add(label, undo)
    if owner is not None:
        record = owner.__dict__.setdefault(_RECORD_ATTR, {})
        refs = record.setdefault("deferred", [])
        if not any(r() is module for r in refs):
            refs.append(weakref.ref(module))


# Fault injection: tests set this to a callable(point) that may raise at every mutation point of the ladder.
_FAULT: Optional[Callable[[str], None]] = None


def _fault(point: str) -> None:
    if _FAULT is not None:
        _FAULT(point)


class Change:
    """One state change of a recovery. ``undo`` puts it back; when ``undo`` fails part way, ``settle`` leaves the
    module in a consistent working state and ``defer`` hands the way back to the module's DeferredReturn, whose
    retry is ``retry`` (undo, falling back to settle)."""

    def __init__(self, label: str, undo: Callable[[], None], settle: Callable[[], None] = lambda: None,
                 defer: Optional[Callable[[Callable[[], None], BaseException], None]] = None) -> None:  # fmt: skip
        self.label = label
        self.undo = undo
        self.settle = settle
        self.defer = defer

    def retry(self) -> None:
        try:
            _fault(self.label + ":undo")
            self.undo()
        except Exception:
            _fault(self.label + ":settle")
            self.settle()
            raise


class DecodeSession:
    """One OOM recovery as a transaction: every state change goes through ``apply`` (do, then record its Change),
    and ``restore`` rolls the record back in reverse order. A Change leaves the record only once its own undo
    succeeded or its module was settled into a consistent state with a deferred way back."""

    def __init__(self, vae: Any, device: Any, logger: Any = None) -> None:
        self.vae = vae
        self.device = device
        self.logger = logger
        self.freed = False
        self.vae_offloaded = False
        self.log: list = []  # Change, in the order applied

    @property
    def moved(self) -> list:
        return [c.label.split(":", 1)[1] for c in self.log if c.label.startswith("moved:")]

    def _note(self, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.warning(msg, *args)

    def apply(self, do: Callable[[], None], change: Change) -> bool:
        """Run ``do``; record ``change`` on success. A ``do`` that fails (part way, possibly) is rolled back now."""
        try:
            _fault(change.label + ":do")
            do()
        except Exception as exc:  # noqa: BLE001
            self._note("diffusion.vae_oom: %s failed (%s); rolled back", change.label, exc)
            self.rollback(change)
            return False
        self.log.append(change)
        return True

    def rollback(self, change: Change) -> bool:
        try:
            _fault(change.label + ":undo")
            change.undo()
            return True
        except Exception as exc:  # noqa: BLE001
            try:
                _fault(change.label + ":settle")
                change.settle()
            except Exception as exc2:  # noqa: BLE001 - the deferred return below retries from wherever it is
                self._note("diffusion.vae_oom: settling %s failed (%s)", change.label, exc2)
            if change.defer is not None:
                try:
                    change.defer(change.retry, exc)
                except Exception as exc3:  # noqa: BLE001
                    self._note("diffusion.vae_oom: deferring %s failed (%s)", change.label, exc3)
            return False

    def restore(self) -> None:
        while self.log:
            self.rollback(self.log.pop())
        self.vae_offloaded = False

    def free_others(self, pipe: Any) -> int:
        """Step (a). Returns how many things were released (0: nothing on the card to free)."""
        if self.freed or pipe is None or getattr(self.device, "type", None) != "cuda":
            return 0
        self.freed = True
        released = 0
        # Captured graphs pin their pool and the weights' addresses: drop them before any weight moves (the image and
        # video backends drop them on any OOM too). Nothing to undo: the next step re-captures. If dropping them
        # fails, no weight may move under a graph that could still replay: step (a) stops here.
        graphs = getattr(pipe, "_unsloth_cuda_graphs", ()) or ()
        if graphs:

            def reset() -> None:
                from . import diffusion_cuda_graph
                diffusion_cuda_graph.reset_all(graphs)

            if not self.apply(reset, Change("graphs_reset", lambda: None)):
                return 0
            released += 1
        # One dispatcher keyed on how each component is placed, each kind through Studio's own release path:
        #   torchao resident with no hooks -> hook_resident_denoiser (block offload hooks, all resident), then
        #   resident offload groups      -> release_resident_groups (stream them again), then
        #   plain modules whole on card  -> to the host;
        #   anything an offload mechanism already places (accelerate / streamed group offload) is left to it.
        self._hook_torchao_resident(pipe)
        released += self._release_groups(pipe)
        released += self._move_components(pipe)
        if self.moved:
            self._note("diffusion.vae_oom: moved %s to the host for the decode", ", ".join(self.moved))
        return released

    def _hook_torchao_resident(self, pipe: Any) -> bool:
        """A hookless resident torchao transformer cannot move whole (Module.to skips or breaks its weights): give it
        block offload hooks with every group resident (diffusion_memory.hook_resident_denoiser, the path an oversized
        request takes), so the group release below can stream it. The hooks stay afterwards, every group re-pinned,
        exactly as after an oversized request."""
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or _placement(transformer, self.device) != "torchao_resident":
            return False
        from . import diffusion_memory

        hooked: list = []

        def do() -> None:
            hooked.append(bool(diffusion_memory.hook_resident_denoiser(pipe, self.device, self.logger)))

        if not self.apply(do, Change("torchao_hooks", lambda: None)):
            return False
        return bool(hooked and hooked[0])

    def _release_groups(self, pipe: Any) -> int:
        """Resident offload groups (text encoders first, then the denoiser's blocks) stream again. A release that
        fails part way returns the restore for what it did release (diffusion_memory.release_resident_groups)."""
        from . import diffusion_memory

        got: list = []
        anchor = next(
            (m for m in (getattr(pipe, "components", {}) or {}).values() if getattr(m, "_unsloth_resident_room", None)),
            None,
        )  # fmt: skip
        if anchor is None:
            return 0

        def do() -> None:
            restore = diffusion_memory.release_resident_groups(
                pipe, 1 << 30, self.logger, reason = "a VAE decode that ran out of memory"
            )
            if restore is not None:
                got.append(restore)

        def undo() -> None:
            while got:
                # strict: an incomplete re-pin raises (the release path's re-pin otherwise swallows it), so the
                # settle / defer below runs and the restore is kept until every group is resident again
                _call_restore(got[-1])
                got.pop()

        def settle() -> None:
            # re-pinning failed part way: stream every group again, the load's flat plan
            diffusion_memory.release_resident_groups(pipe, 1 << 30, self.logger, reason = "a failed re-pin")

        def defer(retry: Callable[[], None], exc: BaseException) -> None:
            _defer(anchor, self.device, "resident_groups", retry, False, self.vae)

        change = Change("resident_groups", undo, settle, defer)
        if not self.apply(do, change):
            return 0
        if not got:
            self.log.remove(change)
            return 0
        return 1

    def _move_components(self, pipe: Any) -> int:
        """Components whole on the card with no offload hooks: to the host, largest first, while host RAM allows."""
        import torch

        candidates = []
        for name, comp in (getattr(pipe, "components", {}) or {}).items():
            if comp is self.vae or not isinstance(comp, torch.nn.Module):
                continue
            if _placement(comp, self.device) != "plain":
                continue
            candidates.append((_nbytes(comp), name, comp))
        candidates.sort(key = lambda c: -c[0])
        host = _host_free_mib()
        released = 0
        for size, name, comp in candidates:
            mib = size >> 20
            if host is not None and mib + HOST_MARGIN_MIB > host:
                self._note("diffusion.vae_oom: %s (%d MiB) stays on the card: host RAM is short", name, mib)
                continue
            if self.apply(lambda comp=comp: _to_host(comp), self._component_change(name, comp)):
                released += 1
                if host is not None:
                    host -= mib
        return released

    def _component_change(self, name: str, comp: Any) -> Change:
        device = self.device

        def settle() -> None:
            _to_host(comp)  # whole on the host: consistent, and the deferred return knows how to move it back
            _empty_cache(device)

        def defer(retry: Callable[[], None], exc: BaseException) -> None:
            self._note(
                "diffusion.vae_oom: %s could not return to the card after the decode (%s); it returns before its "
                "next forward", name, exc,
            )  # fmt: skip
            _defer(comp, device, f"moved:{name}", retry, True, self.vae)

        return Change(f"moved:{name}", lambda: _return_to(comp, device), settle, defer)

    def _component_undo(self, name: str, comp: Any) -> tuple:
        """(undo, defer) for a component moved to the host; kept for callers recording a move by hand."""
        change = self._component_change(name, comp)
        return change.retry, lambda exc: change.defer(change.retry, exc)

    def offload_vae(self) -> bool:
        """Step (c): the VAE's weights stream per layer from the host (diffusers leaf-level group offload)."""
        if self.vae_offloaded or getattr(self.device, "type", None) != "cuda" or _hooked(self.vae):
            return False
        vae, device = self.vae, self.device
        host, mib = _host_free_mib(), _nbytes(vae) >> 20
        if host is not None and mib + HOST_MARGIN_MIB > host:
            # the streamed copy lives in host RAM: never trade a decode OOM for a host OOM
            self._note("diffusion.vae_oom: %s (%d MiB) stays on the card: host RAM is short", type(vae).__name__, mib)
            return False

        def settle() -> None:
            _park_vae(vae, device)

        def defer(retry: Callable[[], None], exc: BaseException) -> None:
            self._note("diffusion.vae_oom: %s stays streamed from the host (%s); it returns at its next decode",
                       type(vae).__name__, exc)  # fmt: skip
            _defer(vae, device, "vae_offload", retry, False, vae)

        change = Change("vae_offload", lambda: _remove_vae_offload(vae, device), settle, defer)
        if not self.apply(lambda: _apply_vae_offload(vae, device), change):
            return False
        self.vae_offloaded = True
        self._note("diffusion.vae_oom: %s weights stream from the host for this decode", type(vae).__name__)
        return True


def _apply_vae_offload(vae: Any, device: Any) -> None:
    import torch
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(
        vae, onload_device = device, offload_device = torch.device("cpu"), offload_type = "leaf_level"
    )


def _remove_vae_offload(vae: Any, device: Any) -> None:
    from .diffusion_memory import _remove_group_offload_hooks
    _remove_group_offload_hooks(vae)
    _return_to(vae, device)


def _park_vae(vae: Any, device: Any) -> None:
    """Whole on the host with the offload hooks on: the VAE still decodes (streamed)."""
    from .diffusion_memory import _remove_group_offload_hooks
    _remove_group_offload_hooks(vae)
    _to_host(vae)
    _empty_cache(device)
    _apply_vae_offload(vae, device)


def _undo_vae_offload(vae: Any, device: Any) -> None:
    """Hooks off and weights back on the card; if that fails, the VAE is parked streamed (``_park_vae``) and the
    error raised, so the caller can retry later."""
    try:
        _remove_vae_offload(vae, device)
    except Exception:
        _park_vae(vae, device)
        raise


def settle_deferred(module: Any) -> bool:
    """Run ``module``'s pending deferred returns now; True when nothing is left pending."""
    deferred = getattr(module, "__dict__", {}).get(_DEFERRED_ATTR)
    if deferred is None:
        return True
    try:
        return deferred.attempt()
    except Exception:  # noqa: BLE001 - still out of memory: the deferred return stays
        return False


# -- the wrapper -------------------------------------------------------------------------------------------------


def _pipe_of(vae: Any) -> Any:
    ref = vae.__dict__.get(_PIPE_ATTR)
    return ref() if callable(ref) else None


def _device_of(z: Any) -> Any:
    return getattr(z, "device", None)


def free_first(vae: Any, z: Any = None) -> bool:
    """Step (a) for a Studio decode wrapper about to fall back to a cheaper decode on an OOM (wide tiles -> stock
    tiles, LTX-2 tiles -> the floor tile, untiled -> tiled): free the card inside the current decode's recovery
    session. True when something was freed, so the wrapper retries its own attempt first. Outside a decode scope, or
    with the retry switched off, or once this decode already freed the card: False (fall back as before)."""
    holder = getattr(vae, "__dict__", {}).get(_SESSION_ATTR)
    if holder is None or oom_retry_disabled() or _compiling():
        return False
    session = holder.get("session")
    if session is None:
        session = holder["session"] = DecodeSession(vae, _device_of(z), vae.__dict__.get("_unsloth_vae_oom_logger"))
    if session.freed:
        return False
    _release_cache(z, vae)
    freed = bool(session.free_others(_pipe_of(vae)))
    stats = vae.__dict__.get("_unsloth_vae_oom_stats")
    if isinstance(stats, dict):
        stats["freed_first"] = stats.get("freed_first", 0) + int(freed)
    return freed


_RETRY_LAYER = "vae_oom_retry"
_SCOPE_LAYER = "vae_oom_scope"


def _scope_session(vae: Any) -> Optional[dict]:
    return vae.__dict__.get(_SESSION_ATTR)


def _close_scope(vae: Any, holder: dict, z: Any) -> None:
    vae.__dict__.pop(_SESSION_ATTR, None)
    session = holder.get("session")
    if session is None:
        return
    t0 = time.perf_counter()
    session.restore()
    stats = vae.__dict__.get("_unsloth_vae_oom_stats")
    if isinstance(stats, dict):
        stats["restore_s"] = round(time.perf_counter() - t0, 3)
    _empty_cache(_device_of(z))


def install(vae: Any, logger: Any = None, pipe: Any = None) -> bool:
    """Wrap ``vae._decode`` with the offload-first OOM ladder, and the public ``decode`` with the scope one logical
    decode's recovery spans (a compiled decode's guard included). Idempotent. ``pipe``: the pipeline whose other
    components step (a) may move off the card (kept as a weak reference; a reinstall keeps the last one)."""
    if vae is None or oom_retry_disabled():
        return False
    if pipe is not None:
        try:
            vae.__dict__[_PIPE_ATTR] = weakref.ref(pipe)
        except TypeError:
            pass
    if not callable(getattr(vae, "_decode", None)):
        return False
    if decode_layers.find(vae, "_decode", _RETRY_LAYER) is not None:
        return True
    can_tile = callable(getattr(vae, "enable_tiling", None)) and hasattr(vae, "use_tiling")
    stats = {"calls": 0, "oom": 0, "freed_retry": 0, "tiled_retry": 0, "offload_retry": 0, "recovered": 0,
             "recovered_by": None, "failed": 0}  # fmt: skip

    def make_retry(inner: Callable, layer: Any) -> Callable:
        depth = [0]

        def _attempt(z: Any, args: tuple, kwargs: dict, tiled: bool) -> Any:
            prev = bool(getattr(vae, "use_tiling", False))
            vae._unsloth_vae_oom_active = True
            try:
                if can_tile:
                    vae.use_tiling = tiled
                return inner(z, *args, **kwargs)
            finally:
                if can_tile:
                    vae.use_tiling = prev
                vae._unsloth_vae_oom_active = False

        def _decode(z: Any, *args: Any, **kwargs: Any) -> Any:
            # Traced inside a compiled decode: a pass-through (no try/except in the graph; the compiled guard
            # retries). Nested (a wide-tile override decoding one tile through _decode): the outermost call retries.
            if _compiling() or oom_retry_disabled() or depth[0]:
                return inner(z, *args, **kwargs)
            depth[0] += 1
            holder = _scope_session(vae)
            owns = holder is None  # called without the public decode (no scope): this call is the logical decode
            if owns:
                holder = {"session": None}
                vae.__dict__[_SESSION_ATTR] = holder
                settle_deferred(vae)
            try:
                return _retrying(z, args, kwargs, holder)
            finally:
                depth[0] -= 1
                if owns:
                    _close_scope(vae, holder, z)
                    # reached without the scope: a later rewrap (a compile fallback restoring its eager decode, a
                    # speed layer run after load) dropped or covered it; put it back on top for the next decode
                    ensure_outermost(vae, logger)

        def _retrying(z: Any, args: tuple, kwargs: dict, holder: dict) -> Any:
            stats["calls"] += 1
            tiled_now = bool(getattr(vae, "use_tiling", False))
            steps: list = [("freed", tiled_now, False)]
            if can_tile and not tiled_now:
                steps.append(("tiled", True, False))
            steps.append(("offload", tiled_now or can_tile, True))
            recovered = holder.get("recovered")
            if recovered is not None:
                # A later slice / frame of a decode an earlier one recovered: the same mode (tiling as it recovered,
                # the VAE still streamed if it was), so one decode's samples come out of one path.
                start = [n for n, _, _ in steps].index(recovered) if recovered in [n for n, _, _ in steps] else 0
                name, tiled, _ = steps[start]
                try:
                    return _attempt(z, args, kwargs, tiled)
                except Exception as exc:  # noqa: BLE001
                    if not is_oom(exc):
                        raise
                    exc.__traceback__ = None
                    holder.setdefault("first_oom", exc)
                    stats["oom"] += 1
                steps = steps[start + 1:]
            else:
                gate_owned = False
                try:
                    return inner(z, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not is_oom(exc):
                        raise
                    gate_owned = not tiled_now and _outer_gate_owns_untiled(vae)
                    exc.__traceback__ = None
                    holder.setdefault("first_oom", exc)
                    stats["oom"] += 1
                if gate_owned:
                    # The untiled-when-fits gate retries tiled itself; the ladder still sees the OOM first: free the
                    # card (step a) and retry untiled once, then hand back to the gate.
                    if free_first(vae, z):
                        stats["freed_retry"] += 1
                        try:
                            out = inner(z, *args, **kwargs)
                            stats["recovered"] += 1
                            stats["recovered_by"] = "freed"
                            return out
                        except Exception as exc:  # noqa: BLE001
                            if not is_oom(exc):
                                raise
                            exc.__traceback__ = None
                    raise holder["first_oom"]
            # Outside the handler: the failed attempt's tensors are gone. One session per logical decode: a later
            # slice / frame of the same decode reuses what an earlier one freed.
            session = holder.get("session")
            if session is None:
                session = holder["session"] = DecodeSession(vae, _device_of(z), logger)
            for name, tiled, offload_vae in steps:
                _release_cache(z, vae)
                if name == "freed":
                    t0 = time.perf_counter()
                    freed = session.free_others(_pipe_of(vae))
                    stats["free_s"] = round(time.perf_counter() - t0, 3)
                    if not freed:
                        continue  # nothing (more) was freed: the same decode would fail the same way
                elif offload_vae and not session.offload_vae():
                    continue
                if logger is not None:
                    logger.warning(
                        "diffusion.vae_oom: %s decode ran out of memory; retrying %s",
                        type(vae).__name__,
                        {"freed": "with the rest of the pipeline off the card", "tiled": "tiled at stock tile sizes",
                         "offload": "with the VAE weights streamed from the host"}[name],
                    )  # fmt: skip
                stats[f"{name}_retry"] += 1
                failed = False
                try:
                    out = _attempt(z, args, kwargs, tiled)
                except Exception as exc:  # noqa: BLE001
                    if not is_oom(exc) and not offload_vae:
                        raise
                    if not is_oom(exc) and logger is not None:
                        # the streamed decode itself broke (an offload hook the VAE does not support): the render
                        # fails on the out-of-memory it already had, not on the fallback
                        logger.warning("diffusion.vae_oom: streamed %s decode failed (%s)", type(vae).__name__, exc)
                    exc.__traceback__ = None
                    failed = True
                if not failed:
                    stats["recovered"] += 1
                    stats["recovered_by"] = name
                    holder["recovered"] = name
                    return out
                stats["oom"] += 1
            stats["failed"] += 1
            _release_cache(z, vae)
            # Nothing recovered: the decode's own out-of-memory surfaces, with what was tried
            first = holder.get("first_oom")
            note = ("VAE decode ran out of memory even with the rest of the pipeline and the VAE weights off the "
                    "card; free GPU memory or lower the resolution")  # fmt: skip
            if first is None:
                raise RuntimeError(note)
            if callable(getattr(first, "add_note", None)) and note not in (getattr(first, "__notes__", None) or ()):
                first.add_note(note)
            raise first

        _decode._unsloth_vae_oom_retry = True
        _decode._unsloth_vae_oom_stats = stats
        return _decode

    decode_layers.wrap(vae, "_decode", _RETRY_LAYER, make_retry)
    vae._unsloth_vae_oom_stats = stats
    vae.__dict__["_unsloth_vae_oom_logger"] = logger
    _install_scope(vae, logger)
    if logger is not None:
        logger.info(
            "diffusion.vae_oom: %s decode recovers an out-of-memory by offloading, then tiling at stock sizes",
            type(vae).__name__,
        )
    return True


def _compiled_decode_owner(vae: Any) -> Any:
    """The object whose ``decode`` slot holds the speed layer's compiled decode guard (the VAE, or the slot an eager
    outer wrapper calls through), or None when the decode is not compiled whole."""
    outer = vae.__dict__.get("decode")
    for owner in (vae, getattr(outer, "_unsloth_decode_slot", None)):
        fn = getattr(owner, "__dict__", {}).get("decode") if owner is not None else None
        if getattr(fn, "_unsloth_compiled_decode", False) and callable(getattr(fn, "_unsloth_eager_decode", None)):
            return owner
    return None


# Decode wrappers that must sit BELOW the scope: a compiled decode guard (the graph would trace the scope as a
# pass-through and its guard re-raises the OOM) and the untiled-when-fits gate (it retries tiled itself).
_BELOW_SCOPE_MARKERS = ("_unsloth_compiled_decode", "_unsloth_untiled_decode")


def _scope_intact(vae: Any) -> bool:
    """The scope layer is active, reachable from the current public decode through the ``__wrapped__`` chain, and
    no compiled guard or untiled gate sits above it."""
    layer = vae.__dict__.get(_RECORD_ATTR, {}).get("scope")
    if layer is None or not layer.active:
        return False
    fn = vae.__dict__.get("decode")
    slot_owner = getattr(fn, "_unsloth_decode_slot", None)
    for _ in range(32):
        if fn is None:
            break
        if fn is layer.fn:
            return True
        if any(getattr(fn, m, False) for m in _BELOW_SCOPE_MARKERS) and not getattr(fn, "_unsloth_vae_oom_scope", False):
            return False
        nxt = getattr(fn, "__wrapped__", None)
        if nxt is None and slot_owner is not None:
            nxt, slot_owner = slot_owner.__dict__.get("decode"), None
        fn = nxt
    return False


def ensure_outermost(vae: Any, logger: Any = None) -> bool:
    """(Re)assert the OOM scope as the outermost decode layer after anything rewrapped ``vae.decode`` (a deferred
    speed profile, a VAE compile, a compile fallback). Called after every rewrap site; idempotent and cheap. True
    when the retry is installed and its scope is in place."""
    if vae is None or decode_layers.find(vae, "_decode", _RETRY_LAYER) is None or oom_retry_disabled():
        return False
    if _scope_intact(vae):
        return True
    record = vae.__dict__.setdefault(_RECORD_ATTR, {})
    decode_layers.unwrap(record.pop("scope", None))  # buried under the rewrap: now a pass-through
    return _install_scope(vae, logger)


def _install_scope(vae: Any, logger: Any = None) -> bool:
    """Wrap the public ``decode`` (or, for a decode compiled whole, the slot holding it): the outermost call opens
    the recovery scope every inner ``_decode`` (batch slices, frames, tiles) shares, and closes it, restoring what
    the recovery moved, once the whole decode returns. A compiled decode that runs out of memory frees the card and
    retries compiled first (same kernels, same output), then decodes on the eager path inside the same scope."""
    owner = _compiled_decode_owner(vae)
    compiled_slot = owner is not None
    owner = owner if compiled_slot else vae
    if decode_layers.find(owner, "decode", _SCOPE_LAYER) is not None:
        return True
    if not callable(getattr(owner, "decode", None)):
        return False

    def make_scope(inner: Callable, layer: Any) -> Callable:
        eager = getattr(inner, "_unsloth_eager_decode", None) if compiled_slot else None

        def _try(args: tuple, kwargs: dict) -> tuple:
            try:
                return True, inner(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc) or (
                    not getattr(vae, "use_tiling", False) and _outer_gate_owns_untiled(vae)
                ):
                    raise
                exc.__traceback__ = None
                return False, None

        def decode(*args: Any, **kwargs: Any) -> Any:
            if _compiling() or oom_retry_disabled() or _scope_session(vae) is not None:
                return inner(*args, **kwargs)
            holder: dict = {"session": None}
            vae.__dict__[_SESSION_ATTR] = holder
            z = args[0] if args else kwargs.get("z")
            try:
                # a VAE left streamed by an earlier restore that ran out of memory comes back first, if it fits now
                settle_deferred(vae)
                if eager is None:
                    return inner(*args, **kwargs)
                ok, out = _try(args, kwargs)
                if ok:
                    return out
                stats = vae.__dict__.get("_unsloth_vae_oom_stats")
                if isinstance(stats, dict):
                    stats["compiled_oom"] = stats.get("compiled_oom", 0) + 1
                session = holder["session"] = DecodeSession(vae, _device_of(z), logger)
                _release_cache(z, vae)
                if session.free_others(_pipe_of(vae)):
                    if logger is not None:
                        logger.warning("diffusion.vae_oom: compiled %s decode ran out of memory; retrying with the "
                                       "rest of the pipeline off the card", type(vae).__name__)  # fmt: skip
                    ok, out = _try(args, kwargs)
                    if ok:
                        if isinstance(stats, dict):
                            stats["recovered"] = stats.get("recovered", 0) + 1
                            stats["recovered_by"] = "freed_compiled"
                        return out
                    _release_cache(z, vae)
                if logger is not None:
                    logger.warning("diffusion.vae_oom: compiled %s decode ran out of memory; retrying eager",
                                   type(vae).__name__)  # fmt: skip
                return eager(*args, **kwargs)
            finally:
                _close_scope(vae, holder, z)

        if eager is not None:
            decode._unsloth_vae_oom_guard = True
            decode._unsloth_compiled_decode = True
            decode._unsloth_eager_decode = eager
        decode._unsloth_vae_oom_scope = True
        return decode

    layer = decode_layers.wrap(owner, "decode", _SCOPE_LAYER, make_scope)
    vae.__dict__.setdefault(_RECORD_ATTR, {})["scope"] = layer
    return True


def uninstall(vae: Any) -> None:
    """Remove everything ``install`` and its retries put in place: the ``_decode`` layer, the ``decode`` scope /
    compiled guard (cut out exactly when outermost, a pass-through under later wrappers), and every deferred return.
    A deferred return is settled first (its pending undo runs); one that still cannot run (the card is full) stays,
    since it is that module's only way back to the card."""
    record = vae.__dict__.get(_RECORD_ATTR, {})
    decode_layers.unwrap(record.pop("scope", None))
    for ref in record.pop("deferred", []):
        module = ref()
        if module is not None and not settle_deferred(module):
            record.setdefault("deferred", []).append(ref)
    if not record:
        vae.__dict__.pop(_RECORD_ATTR, None)
    layer = decode_layers.find(vae, "_decode", _RETRY_LAYER)
    if layer is None:
        return
    decode_layers.unwrap(layer)
    vae.__dict__.pop("_unsloth_vae_oom_stats", None)


__all__ = [
    "DecodeSession",
    "DeferredReturn",
    "ensure_outermost",
    "free_first",
    "settle_deferred",
    "OOM_RETRY_ENV",
    "install",
    "is_oom",
    "oom_retry_disabled",
    "tile_attrs",
    "uninstall",
]

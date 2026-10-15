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
                if self.required and is_oom(exc):
                    raise
                return False
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


class DecodeSession:
    """One OOM recovery as a transaction: every state change is an undo entry, undone in reverse order by
    ``restore``. An entry leaves the log only once its own undo succeeded; an undo that fails first puts its module
    back in a consistent, working state and is handed to the module's DeferredReturn, which retries it."""

    def __init__(self, vae: Any, device: Any, logger: Any = None) -> None:
        self.vae = vae
        self.device = device
        self.logger = logger
        self.freed = False
        self.vae_offloaded = False
        # (label, undo, defer): defer(exc) runs when undo raised, after undo left its module consistent
        self.log: list = []

    @property
    def moved(self) -> list:
        return [label.split(":", 1)[1] for label, _, _ in self.log if label.startswith("moved:")]

    def _note(self, msg: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.warning(msg, *args)

    def free_others(self, pipe: Any) -> int:
        """Step (a). Returns how many things were released (0: nothing on the card to free)."""
        if self.freed or pipe is None or getattr(self.device, "type", None) != "cuda":
            return 0
        self.freed = True
        released = 0
        # Captured graphs pin their pool and the weights' addresses: drop them before any weight moves (the image and
        # video backends drop them on any OOM too). Nothing to undo: the next step re-captures.
        graphs = getattr(pipe, "_unsloth_cuda_graphs", ()) or ()
        if graphs:
            try:
                from . import diffusion_cuda_graph
                diffusion_cuda_graph.reset_all(graphs)
                self.log.append(("graphs_reset", lambda: None, lambda exc: None))
                released += 1
            except Exception as exc:  # noqa: BLE001
                self._note("diffusion.vae_oom: dropping CUDA graphs failed (%s)", exc)
        # Resident offload groups (text encoders first, then the denoiser's blocks) stream again.
        try:
            from .diffusion_memory import release_resident_groups
            restore = release_resident_groups(
                pipe, 1 << 30, self.logger, reason = "a VAE decode that ran out of memory"
            )
        except Exception as exc:  # noqa: BLE001
            restore = None
            self._note("diffusion.vae_oom: releasing resident groups failed (%s)", exc)
        if restore is not None:
            anchor = next(
                (m for m in (getattr(pipe, "components", {}) or {}).values()
                 if getattr(m, "_unsloth_resident_room", None)),
                None,
            )  # fmt: skip

            def defer_groups(exc: BaseException, restore: Callable = restore, anchor: Any = anchor) -> None:
                # streamed groups are the load's flat plan: a working state; re-pinned before the next forward
                if anchor is not None:
                    _defer(anchor, self.device, "resident_groups", restore, False, self.vae)

            self.log.append(("resident_groups", restore, defer_groups))
            released += 1
        # Components whole on the card with no offload hooks: to the host, largest first, while host RAM allows.
        try:
            import torch

            candidates = []
            for name, comp in (getattr(pipe, "components", {}) or {}).items():
                if comp is self.vae or not isinstance(comp, torch.nn.Module):
                    continue
                if not _whole_on(comp, self.device) or _hooked(comp) or _holds_torchao(comp):
                    continue
                candidates.append((_nbytes(comp), name, comp))
            candidates.sort(key = lambda c: -c[0])
            host = _host_free_mib()
            for size, name, comp in candidates:
                mib = size >> 20
                if host is not None and mib + HOST_MARGIN_MIB > host:
                    self._note("diffusion.vae_oom: %s (%d MiB) stays on the card: host RAM is short", name, mib)
                    continue
                try:
                    _to_host(comp)
                except Exception as exc:  # noqa: BLE001 - put a half-moved module back whole
                    _return_to(comp, self.device)
                    self._note("diffusion.vae_oom: %s could not move to the host (%s)", name, exc)
                    continue
                self.log.append((f"moved:{name}", *self._component_undo(name, comp)))
                if host is not None:
                    host -= mib
                released += 1
        except Exception as exc:  # noqa: BLE001
            self._note("diffusion.vae_oom: moving components to the host failed (%s)", exc)
        if self.moved:
            self._note("diffusion.vae_oom: moved %s to the host for the decode", ", ".join(self.moved))
        return released

    def _component_undo(self, name: str, comp: Any) -> tuple:
        device = self.device

        def undo() -> None:
            try:
                _return_to(comp, device)
            except Exception:
                _to_host(comp)  # whole on the host: consistent, and the deferred return knows how to move it
                _empty_cache(device)
                raise

        def defer(exc: BaseException) -> None:
            self._note(
                "diffusion.vae_oom: %s could not return to the card after the decode (%s); it returns before its "
                "next forward", name, exc,
            )  # fmt: skip
            _defer(comp, device, f"moved:{name}", undo, True, self.vae)

        return undo, defer

    def offload_vae(self) -> bool:
        """Step (c): the VAE's weights stream per layer from the host (diffusers leaf-level group offload)."""
        if self.vae_offloaded or getattr(self.device, "type", None) != "cuda" or _hooked(self.vae):
            return False
        try:
            _apply_vae_offload(self.vae, self.device)
        except Exception as exc:  # noqa: BLE001
            try:
                _undo_vae_offload(self.vae, self.device)
            except Exception:  # noqa: BLE001
                pass
            self._note("diffusion.vae_oom: offloading the VAE failed (%s)", exc)
            return False
        self.vae_offloaded = True
        vae, device = self.vae, self.device

        def undo() -> None:
            _undo_vae_offload(vae, device)

        def defer(exc: BaseException) -> None:
            self._note("diffusion.vae_oom: %s stays streamed from the host (%s); it returns at its next decode",
                       type(vae).__name__, exc)  # fmt: skip
            _defer(vae, device, "vae_offload", undo, False, vae)

        self.log.append(("vae_offload", undo, defer))
        self._note("diffusion.vae_oom: %s weights stream from the host for this decode", type(self.vae).__name__)
        return True

    def restore(self) -> None:
        while self.log:
            label, undo, defer = self.log[-1]
            try:
                undo()
            except Exception as exc:  # noqa: BLE001
                self.log.pop()
                try:
                    defer(exc)
                except Exception as exc2:  # noqa: BLE001
                    self._note("diffusion.vae_oom: deferring %s failed (%s)", label, exc2)
                continue
            self.log.pop()
        self.vae_offloaded = False


def _apply_vae_offload(vae: Any, device: Any) -> None:
    import torch
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(
        vae, onload_device = device, offload_device = torch.device("cpu"), offload_type = "leaf_level"
    )


def _undo_vae_offload(vae: Any, device: Any) -> None:
    """Hooks off and weights back on the card; if the weights cannot all return, they go back to the host whole and the
    hooks are put back, so the VAE still decodes (streamed) and the caller can retry later."""
    from .diffusion_memory import _remove_group_offload_hooks
    _remove_group_offload_hooks(vae)
    try:
        _return_to(vae, device)
    except Exception:
        _to_host(vae)
        _empty_cache(device)
        _apply_vae_offload(vae, device)
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


def install(vae: Any, logger: Any = None, pipe: Any = None) -> bool:
    """Wrap ``vae._decode`` with the offload-first OOM ladder. Idempotent. ``pipe``: the pipeline whose other
    components step (a) may move off the card (kept as a weak reference; a reinstall keeps the last one)."""
    if vae is None or oom_retry_disabled():
        return False
    if pipe is not None:
        try:
            vae.__dict__[_PIPE_ATTR] = weakref.ref(pipe)
        except TypeError:
            pass
    inner = getattr(vae, "_decode", None)
    if not callable(inner):
        return False
    if getattr(inner, "_unsloth_vae_oom_retry", False):
        return True
    can_tile = callable(getattr(vae, "enable_tiling", None)) and hasattr(vae, "use_tiling")
    stats = {"calls": 0, "oom": 0, "freed_retry": 0, "tiled_retry": 0, "offload_retry": 0, "recovered": 0,
             "recovered_by": None, "failed": 0}  # fmt: skip
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

    @functools.wraps(inner)
    def _decode(z: Any, *args: Any, **kwargs: Any) -> Any:
        # Traced inside a compiled decode: a pass-through (no try/except in the graph; the compiled guard retries).
        # Nested (a wide-tile override decoding one tile through _decode): the outermost call owns the retry.
        if _compiling() or oom_retry_disabled() or depth[0]:
            return inner(z, *args, **kwargs)
        depth[0] += 1
        try:
            # a VAE left streamed by an earlier restore that ran out of memory comes back first, if it fits now
            settle_deferred(vae)
            return _retrying(z, args, kwargs)
        finally:
            depth[0] -= 1

    def _retrying(z: Any, args: tuple, kwargs: dict) -> Any:
        stats["calls"] += 1
        tiled_now = bool(getattr(vae, "use_tiling", False))
        try:
            return inner(z, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc) or (not tiled_now and _outer_gate_owns_untiled(vae)):
                raise
            exc.__traceback__ = None
            stats["oom"] += 1
        # Outside the handler: the failed attempt's tensors are gone.
        outer = vae.__dict__.get(_SESSION_ATTR)
        session = outer if outer is not None else DecodeSession(vae, _device_of(z), logger)
        vae.__dict__[_SESSION_ATTR] = session
        try:
            steps: list = []
            steps.append(("freed", tiled_now, False))
            if can_tile and not tiled_now:
                steps.append(("tiled", True, False))
            steps.append(("offload", tiled_now or can_tile, True))
            for name, tiled, offload_vae in steps:
                _release_cache(z, vae)
                if name == "freed":
                    t0 = time.perf_counter()
                    freed = session.free_others(_pipe_of(vae))
                    stats["free_s"] = round(time.perf_counter() - t0, 3)
                    if not freed:
                        continue  # nothing was freed: the same decode would fail the same way
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
                    return out
                stats["oom"] += 1
            stats["failed"] += 1
            _release_cache(z, vae)
            raise RuntimeError(
                "VAE decode ran out of memory even with the rest of the pipeline and the VAE weights off the card; "
                "free GPU memory or lower the resolution"
            )
        finally:
            if outer is None:
                vae.__dict__.pop(_SESSION_ATTR, None)
                t0 = time.perf_counter()
                session.restore()
                stats["restore_s"] = round(time.perf_counter() - t0, 3)
                _empty_cache(_device_of(z))

    _decode._unsloth_vae_oom_retry = True
    _decode._unsloth_vae_oom_stats = stats
    vae._unsloth_vae_oom_inner = inner
    vae._unsloth_vae_oom_stats = stats
    vae._decode = _decode
    _guard_compiled_decode(vae, logger)
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


def _guard_compiled_decode(vae: Any, logger: Any = None) -> bool:
    """A compiled ``decode`` traces the ``_decode`` retry as a pass-through, so a runtime OOM escapes it. Wrap the
    compiled callable: on an OOM (wrapped compiler errors included) free the rest of the pipeline and retry the
    compiled decode (same kernels, same output); still out of memory, decode on the eager path, where the ``_decode``
    ladder runs with that session. Installed after the speed layer compiled, so it sits directly over the compiled
    callable."""
    owner = _compiled_decode_owner(vae)
    if owner is None:
        return False
    compiled = owner.__dict__["decode"]
    if getattr(compiled, "_unsloth_vae_oom_guard", False):
        return True
    eager = compiled._unsloth_eager_decode

    def _try(fn: Callable, args: tuple, kwargs: dict) -> tuple:
        try:
            return True, fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc) or (
                not getattr(vae, "use_tiling", False) and _outer_gate_owns_untiled(vae)
            ):
                raise
            exc.__traceback__ = None
            return False, None

    @functools.wraps(compiled)
    def decode(*args: Any, **kwargs: Any) -> Any:
        if oom_retry_disabled() or vae.__dict__.get(_SESSION_ATTR) is not None:
            return compiled(*args, **kwargs)
        ok, out = _try(compiled, args, kwargs)
        if ok:
            return out
        z = args[0] if args else kwargs.get("z")
        stats = vae.__dict__.get("_unsloth_vae_oom_stats")
        if isinstance(stats, dict):
            stats["compiled_oom"] = stats.get("compiled_oom", 0) + 1
        session = DecodeSession(vae, _device_of(z), logger)
        vae.__dict__[_SESSION_ATTR] = session
        try:
            _release_cache(z, vae)
            if session.free_others(_pipe_of(vae)):
                if logger is not None:
                    logger.warning("diffusion.vae_oom: compiled %s decode ran out of memory; retrying with the "
                                   "rest of the pipeline off the card", type(vae).__name__)  # fmt: skip
                ok, out = _try(compiled, args, kwargs)
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
            vae.__dict__.pop(_SESSION_ATTR, None)
            session.restore()
            _empty_cache(_device_of(z))

    decode._unsloth_vae_oom_guard = True
    decode._unsloth_compiled_decode = True
    decode._unsloth_eager_decode = eager
    owner.decode = decode
    vae.__dict__.setdefault(_RECORD_ATTR, {})["guard"] = (owner, decode, compiled)
    return True


def uninstall(vae: Any) -> None:
    """Remove everything ``install`` and its retries put in place: the ``_decode`` wrapper, the compiled-decode guard,
    and every deferred return. A deferred return is settled first (its pending undo runs); one that still cannot run
    (the card is full) stays, since it is that module's only way back to the card."""
    record = vae.__dict__.get(_RECORD_ATTR, {})
    guard = record.pop("guard", None)
    if guard is not None:
        owner, decode, compiled = guard
        if owner.__dict__.get("decode") is decode:
            owner.__dict__["decode"] = compiled
    for ref in record.pop("deferred", []):
        module = ref()
        if module is not None and not settle_deferred(module):
            record.setdefault("deferred", []).append(ref)
    if not record:
        vae.__dict__.pop(_RECORD_ATTR, None)
    inner = vae.__dict__.pop("_unsloth_vae_oom_inner", None)
    if inner is None:
        return
    if getattr(inner, "__self__", None) is vae and getattr(inner, "__func__", None) is getattr(
        type(vae), "_decode", None
    ):
        vae.__dict__.pop("_decode", None)
    else:
        vae._decode = inner
    vae.__dict__.pop("_unsloth_vae_oom_stats", None)


__all__ = [
    "DecodeSession",
    "DeferredReturn",
    "settle_deferred",
    "OOM_RETRY_ENV",
    "install",
    "is_oom",
    "oom_retry_disabled",
    "tile_attrs",
    "uninstall",
]

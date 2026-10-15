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


class DecodeSession:
    """What one OOM recovery moved, and how to put it back. ``restore`` is idempotent."""

    def __init__(self, vae: Any, device: Any, logger: Any = None) -> None:
        self.vae = vae
        self.device = device
        self.logger = logger
        self.freed = False
        self.vae_offloaded = False
        self.moved: list = []  # (name, module) moved to the host
        self.restorers: list = []
        self.graphs_reset = False

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
        # video backends drop them on any OOM too); the next step re-captures.
        graphs = getattr(pipe, "_unsloth_cuda_graphs", ()) or ()
        if graphs:
            try:
                from . import diffusion_cuda_graph
                diffusion_cuda_graph.reset_all(graphs)
                self.graphs_reset = True
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
            self.restorers.append(restore)
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
                self.moved.append((name, comp))
                if host is not None:
                    host -= mib
                released += 1
        except Exception as exc:  # noqa: BLE001
            self._note("diffusion.vae_oom: moving components to the host failed (%s)", exc)
        if self.moved:
            self._note(
                "diffusion.vae_oom: moved %s to the host for the decode", ", ".join(n for n, _ in self.moved)
            )
        return released

    def offload_vae(self) -> bool:
        """Step (c): the VAE's weights stream per layer from the host (diffusers leaf-level group offload)."""
        if self.vae_offloaded or getattr(self.device, "type", None) != "cuda" or _hooked(self.vae):
            return False
        try:
            import torch
            from diffusers.hooks import apply_group_offloading

            apply_group_offloading(
                self.vae,
                onload_device = self.device,
                offload_device = torch.device("cpu"),
                offload_type = "leaf_level",
            )
        except Exception as exc:  # noqa: BLE001
            self._undo_vae_offload()
            self._note("diffusion.vae_oom: offloading the VAE failed (%s)", exc)
            return False
        self.vae_offloaded = True
        self._note("diffusion.vae_oom: %s weights stream from the host for this decode", type(self.vae).__name__)
        return True

    def _undo_vae_offload(self) -> None:
        from .diffusion_memory import _remove_group_offload_hooks
        _remove_group_offload_hooks(self.vae)
        _return_to(self.vae, self.device)

    def restore(self) -> None:
        if self.vae_offloaded:
            self.vae_offloaded = False
            try:
                self._undo_vae_offload()
            except Exception as exc:  # noqa: BLE001
                self._note("diffusion.vae_oom: returning the VAE to the card failed (%s)", exc)
        moved, self.moved = self.moved, []
        for name, comp in reversed(moved):
            try:
                _return_to(comp, self.device)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc):
                    raise
                # The card filled up during the decode (another process): leave the component on the host whole and
                # bring it back on its next forward.
                _to_host(comp)
                _empty_cache(self.device)
                _return_on_next_forward(comp, self.device, self.logger, name)
        restorers, self.restorers = self.restorers, []
        for restore in reversed(restorers):
            try:
                restore()
            except Exception as exc:  # noqa: BLE001
                self._note("diffusion.vae_oom: re-pinning resident groups failed (%s)", exc)


def _return_on_next_forward(module: Any, device: Any, logger: Any, name: str) -> None:
    if logger is not None:
        logger.warning(
            "diffusion.vae_oom: %s could not return to the card after the decode; it returns on its next forward",
            name,
        )
    handle: list = []

    def pre_hook(mod: Any, args: Any) -> None:
        for h in handle:
            h.remove()
        handle.clear()
        _return_to(mod, device)

    handle.append(module.register_forward_pre_hook(pre_hook))


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
                    if not is_oom(exc):
                        raise
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
    return True


def uninstall(vae: Any) -> None:
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
    "OOM_RETRY_ENV",
    "install",
    "is_oom",
    "oom_retry_disabled",
    "tile_attrs",
    "uninstall",
]

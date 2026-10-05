# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep as much of a streamed MiniMax-H3 denoiser on the device as the card allows: the top-level group first, then a
prefix of its block groups, re-fitted per request. Resident groups keep their pinned host copy, so demoting is free.

``UNSLOTH_H3_DIT_RESIDENT=0`` keeps everything streamed.
"""

from __future__ import annotations

import os
from typing import Any, Optional

H3_DIT_RESIDENT_ENV = "UNSLOTH_H3_DIT_RESIDENT"

# Decimal GB each phase of an H3 render needs on top of the resident set (B200 peaks at 12 / 16 / 24 / 40 GB caps).
#   running block + prefetched next block + compiled workspace
H3_STREAM_WINDOW_GB = 1.5
#   per million pixel-frames
H3_ACTIVATION_GB_PER_MPIXEL_FRAME = 0.08
#   fraction of the activations: 1-2 GB MLP buffers fragment the caching allocator (OOM at 1344x768x124 without it)
H3_FRAGMENTATION_FRACTION = 0.5
H3_VAE_DECODE_GB = 6.0
H3_PHASE_OVERHEAD_GB = 1.8
#   top-level group: 0.81 GB int8
H3_TOP_LEVEL_GB = 1.0


def h3_dit_resident_enabled() -> bool:
    return str(os.environ.get(H3_DIT_RESIDENT_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def h3_phase_need_gb(
    width: int,
    height: int,
    num_frames: int,
    *,
    te_streamed_gb: float,
    fragmentation: bool = True,
    top_gb: float = 0.0,
) -> float:
    """Device GB a request needs beyond the resident set: its largest phase. ``fragmentation`` is slack only the
    residency plan keeps; the refusal floor leaves it out so residency never refuses what the streamed tier admitted."""
    activations = H3_ACTIVATION_GB_PER_MPIXEL_FRAME * width * height * num_frames / 1_000_000
    slack = H3_FRAGMENTATION_FRACTION if fragmentation else 0.0
    denoise = H3_STREAM_WINDOW_GB + top_gb + activations * (1.0 + slack)
    return max(te_streamed_gb, denoise, H3_VAE_DECODE_GB) + H3_PHASE_OVERHEAD_GB


def _tensor_bytes(tensor: Any) -> int:
    try:
        from .diffusion_prequant import tensor_payload_bytes
        return int(tensor_payload_bytes(tensor))
    except Exception:  # noqa: BLE001
        return int(tensor.numel()) * int(tensor.element_size())


def _group_tensors(group: Any) -> list:
    out: list = []
    seen: set = set()
    for module in getattr(group, "modules", None) or ():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                out.append(tensor)
    for tensor in list(getattr(group, "parameters", None) or ()) + list(
        getattr(group, "buffers", None) or ()
    ):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            out.append(tensor)
    return out


def group_payload_bytes(group: Any) -> int:
    return sum(_tensor_bytes(t) for t in _group_tensors(group))


def _group_hook(module: Any) -> Any:
    try:
        from diffusers.hooks import group_offloading as go

        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            return None
        return registry.get_hook(getattr(go, "_GROUP_OFFLOADING", "group_offloading"))
    except Exception:  # noqa: BLE001
        return None


def h3_offload_groups(transformer: Any) -> tuple[Optional[Any], list]:
    """(top-level group, block groups in execution order) of a block-streamed H3 denoiser; (None, []) if not streamed."""
    top_hook = _group_hook(transformer)
    top = getattr(top_hook, "group", None)
    blocks: list = []
    seen: set = set()
    for block in getattr(transformer, "transformer_blocks", None) or ():
        group = getattr(_group_hook(block), "group", None)
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            blocks.append(group)
    return top, blocks


def _noop() -> None:
    return None


def _resident_onload(group: Any) -> Any:
    """A resident group's onload kicks the prefetch, else the first streamed block behind it is never prefetched."""
    prefetcher = getattr(group, "_unsloth_prefetcher", None)
    kick = getattr(prefetcher, "kick", None)
    if not callable(kick):
        return _noop

    def onload_(*_a: Any, **_k: Any) -> None:
        kick()

    return _outside_inference_mode(onload_)


def is_resident(group: Any) -> bool:
    return bool(getattr(group, "_unsloth_resident", False))


def make_resident(group: Any) -> None:
    """Onload ``group`` once and make its hooks' onload / offload no-ops, so it stays on the device."""
    import torch

    if is_resident(group):
        return
    # The instance attribute, not the class method: an instance override (pinned top group) is an onload path too.
    # Except the prefetcher's own: a group leaving the stream must not count against its in-flight window.
    prefetcher = getattr(group, "_unsloth_prefetcher", None)
    owns = getattr(prefetcher, "owns", None)
    if callable(owns) and owns(group):
        _outside_inference_mode(type(group).onload_)(group)
    else:
        group.onload_()
    stream = getattr(group, "stream", None)
    if stream is not None:
        stream.synchronize()
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    group._unsloth_saved_io = (group.__dict__.get("onload_"), group.__dict__.get("offload_"))
    group.onload_ = _resident_onload(group)
    group.offload_ = _noop
    group._unsloth_resident = True


def demote(group: Any) -> None:
    """Undo ``make_resident``: restore the streaming onload / offload and move the weights back to the host copy."""
    import torch

    if not is_resident(group):
        return
    saved_on, saved_off = getattr(group, "_unsloth_saved_io", (None, None))
    for name, saved in (("onload_", saved_on), ("offload_", saved_off)):
        if saved is not None:
            setattr(group, name, saved)
        else:
            group.__dict__.pop(name, None)
    group._unsloth_resident = False
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    group.offload_()


class H3Residency:
    """The resident set of one streamed H3 denoiser: the top-level group and a prefix of its block groups."""

    def __init__(
        self,
        transformer: Any,
        device: Any,
        *,
        logger: Any = None,
    ) -> None:
        self.top, self.blocks = h3_offload_groups(transformer)
        self.device = device
        self.logger = logger
        self.top_bytes = group_payload_bytes(self.top) if self.top is not None else 0
        self.block_bytes = [group_payload_bytes(g) for g in self.blocks]
        self.max_blocks = 0

    @property
    def usable(self) -> bool:
        return bool(self.blocks)

    def resident_blocks(self) -> int:
        n = 0
        for group in self.blocks:
            if not is_resident(group):
                break
            n += 1
        return n

    def resident_bytes(self) -> int:
        total = self.top_bytes if (self.top is not None and is_resident(self.top)) else 0
        return total + sum(b for g, b in zip(self.blocks, self.block_bytes) if is_resident(g))

    def plan(
        self,
        budget_bytes: int,
        *,
        cap: Optional[int] = None,
    ) -> tuple[bool, int]:
        """(top resident, number of resident blocks) that fit ``budget_bytes``; top first, then the block prefix."""
        if budget_bytes <= 0 or self.top is None and not self.blocks:
            return False, 0
        top = self.top is not None and self.top_bytes <= budget_bytes
        left = budget_bytes - (self.top_bytes if top else 0)
        n = 0
        limit = len(self.blocks) if cap is None else min(cap, len(self.blocks))
        while n < limit and self.block_bytes[n] <= left:
            left -= self.block_bytes[n]
            n += 1
        return top, n

    def apply(self, top: bool, n: int) -> None:
        """Demote first (frees the device), then promote."""
        import torch

        for i in range(len(self.blocks) - 1, n - 1, -1):
            demote(self.blocks[i])
        if self.top is not None and not top:
            demote(self.top)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.top is not None and top:
            make_resident(self.top)
        for i in range(n):
            make_resident(self.blocks[i])

    def fit(
        self,
        budget_bytes: int,
        *,
        initial: bool = False,
    ) -> tuple[bool, int]:
        """Re-fit the resident set to ``budget_bytes`` (the device bytes the resident set may occupy)."""
        top, n = self.plan(budget_bytes)
        self.apply(top, n)
        if initial:
            self.max_blocks = n
        if self.logger is not None:
            self.logger.info(
                "video.h3_residency: %d of %d denoiser blocks resident (+ top-level %s), %.2f GB on the device, "
                "budget %.2f GB",
                n,
                len(self.blocks),
                "resident" if top else "streamed",
                self.resident_bytes() / 1e9,
                budget_bytes / 1e9,
            )
        return top, n


def release_all(residency: "H3Residency", *, logger: Any = None) -> None:
    """Return every group of ``residency`` to streaming, including one whose onload failed partway. Never raises."""
    import torch

    groups = list(reversed(residency.blocks))
    if residency.top is not None:
        groups.append(residency.top)
    for group in groups:
        try:
            if is_resident(group):
                demote(group)
            else:
                group.offload_()
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning("video.h3_residency: could not release a group: %s", exc)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def h3_held_host_bytes(*modules: Any) -> dict[str, int]:
    """Host bytes the given components hold (tensors and group-offload host copies), pinned vs pageable, de-duplicated
    by storage; pinned bytes rounded up to a power of two as torch's host allocator reserves them."""
    seen: set = set()
    pinned = 0
    pageable = 0

    def _add(tensor: Any) -> None:
        nonlocal pinned, pageable
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if callable(flatten) and type(tensor).__name__ not in ("Tensor", "Parameter"):
            try:
                for name in flatten()[0]:
                    _add(getattr(tensor, name))
                return
            except Exception:  # noqa: BLE001
                pass
        try:
            if tensor.device.type != "cpu":
                return
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key in seen or storage.nbytes() == 0:
                return
            seen.add(key)
            nbytes = int(storage.nbytes())
            if tensor.is_pinned():
                pinned += 1 << (nbytes - 1).bit_length()
            else:
                pageable += nbytes
        except Exception:  # noqa: BLE001
            return

    for module in modules:
        if module is None:
            continue
        try:
            for tensor in list(module.parameters()) + list(module.buffers()):
                _add(tensor)
            for sub in module.modules():
                group = getattr(_group_hook(sub), "group", None)
                for tensor in (getattr(group, "cpu_param_dict", None) or {}).values():
                    _add(tensor)
        except Exception:  # noqa: BLE001
            continue
    return {"pinned": pinned, "pageable": pageable}


H3_COMPILE_BELOW_HOOKS_ENV = "UNSLOTH_H3_COMPILE_BELOW_HOOKS"


def _is_original_forward(fn: Any, module: Any) -> bool:
    return getattr(fn, "__self__", None) is module and getattr(fn, "__func__", None) is getattr(
        type(module), "forward", None
    )


def compile_blocks_below_offload_hooks(transformer: Any, logger: Any = None) -> int:
    """Compile each streamed block's own ``forward`` instead of ``_call_impl``, so the group-offload hooks stay eager.

    Traced hooks put residency and prefetch state into the guards: 40 graphs on the first render at 24 GB and more on
    every resident-set change. ``UNSLOTH_H3_COMPILE_BELOW_HOOKS=0`` keeps the old placement."""
    if str(os.environ.get(H3_COMPILE_BELOW_HOOKS_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return 0
    kwargs = getattr(transformer, "_unsloth_regional_compile_kwargs", None)
    if not isinstance(kwargs, dict):
        return 0
    import torch

    guard = getattr(transformer, "_unsloth_compile_guard", None)
    moved = 0
    for module in list(transformer.modules()):
        if getattr(module, "_compiled_call_impl", None) is None or _group_hook(module) is None:
            continue
        registry = getattr(module, "_diffusers_hook", None)
        fn_refs = list(getattr(registry, "_fn_refs", None) or ())
        target = next(
            (ref for ref in fn_refs if _is_original_forward(getattr(ref, "forward", None), module)),
            None,
        )
        if target is None:
            continue
        original = target.forward
        compiled = torch.compile(original, **kwargs)
        if guard is not None and callable(getattr(guard, "wrap", None)):
            target.forward = guard.wrap(compiled, original, transformer)
            guard.restores.append(lambda ref = target, fn = original: setattr(ref, "forward", fn))
        else:
            target.forward = compiled
        module._compiled_call_impl = None
        moved += 1
    if moved and logger is not None:
        logger.info(
            "video.h3_compile: %d streamed denoiser blocks compile below their offload hooks", moved
        )
    return moved


H3_TOP_GROUP_PIN_ENV = "UNSLOTH_H3_TOP_GROUP_PIN"


def pin_streamed_top_level_group(transformer: Any, logger: Any = None) -> bool:
    """Give a block-streamed denoiser's top-level group a pinned host copy and the blocks' copy stream.

    diffusers builds it without a stream, so every forward uploads it from pageable memory and copies it back (0.45 s
    of a 1.5 s step at 12 GB). #12389's top-level pin skips torchao weights, which this checkpoint has.
    ``UNSLOTH_H3_TOP_GROUP_PIN=0`` keeps diffusers' group."""
    if str(os.environ.get(H3_TOP_GROUP_PIN_ENV, "")).strip().lower() in ("0", "off", "false", "no"):
        return False
    from .video_minimax_h3_te import h3_te_pin_allowed

    top, blocks = h3_offload_groups(transformer)
    if not h3_te_pin_allowed(group_payload_bytes(top) if top is not None else 0):
        if logger is not None:
            logger.info(
                "video.h3_top_group: left as diffusers built it (pinning not allowed on this host)"
            )
        return False
    stream = next(
        (getattr(g, "stream", None) for g in blocks if getattr(g, "stream", None) is not None), None
    )
    why = None
    if top is None:
        why = "no top-level group"
    elif stream is None:
        why = "the blocks have no copy stream"
    elif getattr(top, "stream", None) is not None:
        why = "the top-level group already streams"
    elif getattr(top, "offload_to_disk_path", None) or not callable(
        getattr(top, "_init_cpu_param_dict", None)
    ):
        why = "disk offload / unknown diffusers group"
    elif getattr(top, "_unsloth_pinned_top", False) or is_resident(top):
        why = "another path already pinned or holds it"
    if why is not None:
        if logger is not None:
            logger.info("video.h3_top_group: left as diffusers built it (%s)", why)
        return False
    saved = (
        top.stream,
        top.low_cpu_mem_usage,
        top.record_stream,
        top.non_blocking,
        top.cpu_param_dict,
    )
    try:
        from .diffusion_pinned_arena import pinned_arena_for_group_offload

        top.stream = stream
        top.low_cpu_mem_usage = False
        top.record_stream = True
        # Blocking: nothing prefetches the top-level group.
        top.non_blocking = False
        with pinned_arena_for_group_offload():
            top.cpu_param_dict = top._init_cpu_param_dict()
    except Exception as exc:  # noqa: BLE001 -- keep diffusers' group as it was
        (
            top.stream,
            top.low_cpu_mem_usage,
            top.record_stream,
            top.non_blocking,
            top.cpu_param_dict,
        ) = saved
        if logger is not None:
            logger.warning(
                "video.h3_top_group: pinned copy refused, keeping the pageable group: %s", exc
            )
        return False
    if logger is not None:
        logger.info(
            "video.h3_top_group: the streamed denoiser's top-level group (%.2f GB) onloads from a pinned copy",
            group_payload_bytes(top) / 1e9,
        )
    return True


H3_STREAM_PREFETCH_ENV = "UNSLOTH_H3_STREAM_PREFETCH"


def h3_stream_prefetch_depth(block_bytes: list, default_depth: int) -> int:
    """Groups copied ahead: as many as ``H3_STREAM_WINDOW_GB`` (reserved by the residency plan) holds next to the
    running block, capped by ``default_depth``, at least 1."""
    largest = max(block_bytes) if block_bytes else 0
    if largest <= 0:
        return max(1, int(default_depth))
    fits = int(H3_STREAM_WINDOW_GB * 1e9 // largest) - 1
    return max(1, min(int(default_depth), fits))


def h3_stream_prefetch_window(top_bytes: int, block_bytes: list) -> int:
    """Bytes the prefetcher may hold on the device: the top-level group plus two blocks, diffusers' own footprint."""
    largest = max(block_bytes) if block_bytes else 0
    return int(max(0, top_bytes) + 2 * largest)


def install_h3_stream_prefetch(
    transformer: Any,
    device: Any,
    logger: Any = None,
) -> int:
    """Drive a block-streamed H3 denoiser's offload groups with the event-fenced prefetch (no host sync per onload).
    Run after the top-level pin, before the residency fit; returns groups covered. ``UNSLOTH_H3_STREAM_PREFETCH=0`` off."""
    if str(os.environ.get(H3_STREAM_PREFETCH_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return 0
    try:
        from .diffusion_offload_prefetch import install_group_prefetch, prefetch_depth
    except ImportError:
        return 0
    top, blocks = h3_offload_groups(transformer)
    if not blocks:
        return 0
    depth = h3_stream_prefetch_depth([group_payload_bytes(g) for g in blocks], prefetch_depth())
    from .diffusion_memory import _pinned_memory_capped

    if _pinned_memory_capped():
        depth = 1  # each in-flight group is pinned on the fly; diffusers' own path holds two near a ~1 GiB cap
    # The prefetcher refuses groups whose onload_ is replaced: lift the outside-inference_mode wrappers (torchao v1
    # int8 cannot be re-pointed inside it), install, then wrap its moves the same way.
    groups = _all_offload_groups(transformer)
    # A top-level group the H3 pin left unstreamed (kill switch, host refusal) keeps its wrapper, so the generic
    # top-group adoption cannot pin it.
    kept = top if top is not None and getattr(top, "stream", None) is None else None
    lifted = [g for g in groups if g is not kept and _lift_outside_inference_wrappers(g)]
    try:
        covered = install_group_prefetch(transformer, device, logger, depth = depth)
    except Exception:  # noqa: BLE001 -- install_group_prefetch never raises today; keep the wrappers either way
        covered = 0
    for group in lifted:
        for name in ("onload_", "offload_"):
            setattr(group, name, _outside_inference_mode(getattr(group, name)))
    if covered:
        from .diffusion_offload_prefetch import module_prefetcher

        prefetcher = module_prefetcher(transformer)
        for group in groups:
            if getattr(group, "_unsloth_prefetcher", None) is prefetcher:
                # owns() compares the instance onload_ with this
                group._unsloth_prefetch_onload = group.__dict__.get("onload_")
        if prefetcher is not None:
            # (depth + 1) x the 0.8 GB top-level group would leave no block ahead once that group streams (12 GB).
            prefetcher.window = h3_stream_prefetch_window(
                group_payload_bytes(top) if top is not None else 0,
                [group_payload_bytes(g) for g in blocks],
            )
            # end() puts groups copied ahead but never run back on the host copy; kick() queues copies
            for name in ("end", "kick"):
                setattr(prefetcher, name, _outside_inference_mode(getattr(prefetcher, name)))
    return covered


def _all_offload_groups(transformer: Any) -> list:
    out: list = []
    seen: set = set()
    for sub in transformer.modules():
        group = getattr(_group_hook(sub), "group", None)
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            out.append(group)
    return out


def _lift_outside_inference_wrappers(group: Any) -> bool:
    """Drop the instance onload_ / offload_ that only re-enter the class methods outside inference_mode."""
    attrs = getattr(group, "__dict__", {})
    if is_resident(group) or not any(name in attrs for name in ("onload_", "offload_")):
        return False
    for name in ("onload_", "offload_"):
        attrs.pop(name, None)
    return True


def _outside_inference_mode(fn: Any) -> Any:
    import torch

    def call(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(False), torch.no_grad():
            return fn(*args, **kwargs)

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(call) if callable(disable) else call


H3_VAE_PINNED_SWAP_ENV = "UNSLOTH_H3_VAE_PINNED_SWAP"


def install_pinned_swap(
    module: Any,
    *,
    logger: Any = None,
    label: str = "component",
) -> bool:
    """Make a ComponentsManager-rotated module (H3's VAEs) move by re-pointing at a pinned in-place host copy.

    Stock rotation uploads from pageable memory and copies back every park (3.0 s per VAE decode at 24 GB). A device
    move uploads from the pinned copy, a CPU move only re-points; dtype changes or mismatched tensors take stock
    ``nn.Module.to``. ``UNSLOTH_H3_VAE_PINNED_SWAP=0`` keeps stock moves."""
    if str(os.environ.get(H3_VAE_PINNED_SWAP_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return False
    if module is None or getattr(module, "_unsloth_pinned_swap", None) is not None:
        return False
    import torch

    from .video_minimax_h3_te import _module_payload_bytes, h3_te_pin_allowed, pin_module_in_place

    if not torch.cuda.is_available() or not h3_te_pin_allowed(_module_payload_bytes(module)):
        return False
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors or any(t.device.type != "cpu" for t in tensors):
        return False
    if any(type(t.data) is not torch.Tensor for t in tensors):
        return False
    try:
        pin_module_in_place(module, repoint = True)
    except Exception as exc:  # noqa: BLE001 -- stock moves
        if logger is not None:
            logger.warning("video.h3_pinned_swap: %s not pinned (%s)", label, exc)
        return False
    host: dict[int, Any] = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.device.type == "cpu" and tensor.is_pinned():
            host[id(tensor)] = tensor.data
    stock_to = module.to

    def _swap_to(*args: Any, **kwargs: Any) -> Any:
        try:
            device, dtype, _non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return stock_to(*args, **kwargs)
        if device is None or dtype is not None or memory_format is not None:
            return stock_to(*args, **kwargs)
        device = torch.device(device)
        fallback = False
        for tensor in list(module.parameters()) + list(module.buffers()):
            pinned = host.get(id(tensor))
            if pinned is None or pinned.dtype != tensor.dtype or pinned.shape != tensor.shape:
                fallback = True
                continue
            if device.type == "cpu":
                tensor.data = pinned
            elif tensor.device != device:
                tensor.data = pinned.to(device, non_blocking = True)
        if fallback:
            return stock_to(*args, **kwargs)
        return module

    module.to = _swap_to
    module._unsloth_pinned_swap = host
    if logger is not None:
        logger.info(
            "video.h3_pinned_swap: %s moves by re-pointing at a pinned host copy (%.2f GB)",
            label,
            sum(t.numel() * t.element_size() for t in host.values()) / 1e9,
        )
    return True

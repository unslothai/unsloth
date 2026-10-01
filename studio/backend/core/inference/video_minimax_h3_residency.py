# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep as much of a streamed MiniMax-H3 denoiser on the device as the card allows.

When the hosted int8 denoiser does not fit beside the rest of the pipeline it is group-offloaded block by block
(``diffusion_prequant.stream_prequantized_module``): every one of its 50 blocks crosses the bus on every denoise step,
and its top-level weights (token refiner, embedders, adaLN tables, output projection) form one group that diffusers
uploads synchronously from pageable memory and copies back after every forward. On a card with room to spare that is
pure waste. This module turns the spare room into residency:

  * the top-level group first (it is touched twice per step and has no copy stream at all),
  * then a PREFIX of the block groups, so the copy stream only has to keep up with the remaining tail.

A resident group keeps its pinned host copy, so demoting it again is free (re-point at the host copy) and the host
footprint is exactly what a fully streamed load already holds. The resident set is re-fitted before every generation
against the phase needs of THAT request (encode, denoise activations at its width x height x frames, VAE decode), so a
long clip on a small card demotes blocks instead of running out of memory, and the next short clip promotes them back.

``UNSLOTH_H3_DIT_RESIDENT=0`` keeps everything streamed (the behaviour before this existed).
"""

from __future__ import annotations

import os
from typing import Any, Optional

H3_DIT_RESIDENT_ENV = "UNSLOTH_H3_DIT_RESIDENT"

# Decimal GB the phases of an H3 render need on the device ON TOP of whatever is resident. Measured on a B200 with the
# process capped at 12 / 16 / 24 / 40 GB (set_per_process_memory_fraction + a mem_get_info shim), 960x544 at 124 and
# 345 frames and 1344x768x124, torch peak allocated per phase minus the resident bytes (outputs/h3_vs_comfy/lowvram,
# scripts/h3vc_lowvram_e2e.py). Measured at 960x544x124: encode 1.45 GiB, denoise 5.39 GiB (top-level group, two blocks
# and activations), VAE decode 6.70 GiB, the same at every budget.
#   streamed window: the running block + the prefetched next one (0.39 GB int8 each) + the compiled block's workspace
H3_STREAM_WINDOW_GB = 1.5
#   activations: GB per million pixel-frames (latent tokens scale with width x height x frames)
H3_ACTIVATION_GB_PER_MPIXEL_FRAME = 0.08
#   allocator fragmentation, as a fraction of the activations: the per-block MLP buffers are 1-2 GB each and the
#   caching allocator cannot hand back a segment with one live block in it. Measured reserved-but-unallocated at the
#   denoise peak: 4.6 GiB at 960x544x345, 5.5 GiB at 1344x768x124 (the OOM that set this term).
H3_FRAGMENTATION_FRACTION = 0.5
#   VAE decode: the rotated-in video + audio VAE (decoder pre-cast to fp16, t2va encoder dropped) + tiled decode
H3_VAE_DECODE_GB = 6.0
#   allocator / context slack on top of every phase
H3_PHASE_OVERHEAD_GB = 1.8
#   the denoiser's top-level group (token refiner, embedders, adaLN tables, output projections): 0.81 GB int8
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
    """Device GB one generation needs beyond the resident set: the largest of its encode, denoise and decode phases.

    ``fragmentation`` adds the allocator slack the RESIDENCY plan keeps free. The refusal floor leaves it out (a fully
    streamed render runs in whatever is left, exactly as before residency existed), so residency never refuses a clip
    the streamed tier admitted."""
    activations = H3_ACTIVATION_GB_PER_MPIXEL_FRAME * width * height * num_frames / 1_000_000
    slack = H3_FRAGMENTATION_FRACTION if fragmentation else 0.0
    # ``top_gb``: the top-level group when it is NOT resident; it is on the device for the denoise only.
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


def is_resident(group: Any) -> bool:
    return bool(getattr(group, "_unsloth_resident", False))


def make_resident(group: Any) -> None:
    """Onload ``group`` once and make its hooks' onload / offload no-ops, so it stays on the device."""
    import torch

    if is_resident(group):
        return
    # Through the CLASS methods: an instance override (e.g. the top-level pinned copy) is an onload path too, and
    # calling the instance attribute honours it.
    group.onload_()
    stream = getattr(group, "stream", None)
    if stream is not None:
        stream.synchronize()
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    group._unsloth_saved_io = (group.__dict__.get("onload_"), group.__dict__.get("offload_"))
    group.onload_ = _noop
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
        # Blocks the load itself made resident (sized for the family's largest preset), for the status line.
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
        # Uncapped: a request smaller than the one the load was sized for may hold more blocks than the load did.
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


def h3_held_host_bytes(*modules: Any) -> dict[str, int]:
    """Host bytes the given loaded components hold, split into pinned and pageable, de-duplicated by storage.

    Counts both the tensors the modules point at and the pinned host copies their group-offload groups keep
    (``cpu_param_dict``), so a resident group's host copy is counted once and a streamed group's is counted once.
    Pinned bytes are counted as the allocator reserves them (rounded up to a power of two per allocation), which is
    what the host actually loses. For the host-RAM preflight: on a repeat render every byte here is already held."""
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
    """Move a streamed denoiser's regional compile BELOW its group-offload hooks.

    ``Module.compile`` compiles ``_call_impl``, so dynamo traced diffusers' hook ``new_forward`` and every
    ``pre_forward`` / ``post_forward`` around the block. Their Python state became guards: whether the block's group is
    resident (its onload is a no-op), whether a prefetch target exists (``next_group`` is None until the first forward
    has traced the execution order), and the graph break at the disabled ``onload_`` left a resume frame whose inputs
    (``___stack0``) lost the unbacked ``temb`` / dynamic text-length marking. Measured on a 24 GB budget: 40 graphs on the
    first render, 10 more on the second (a 9.4 s first AND second step), and again whenever the resident set changed.

    Compiling the block's own ``forward`` (the innermost function the hook chain calls) leaves the hooks eager: the same
    copies are issued, nothing about them is guarded, and the block graph is the one a resident denoiser compiles.
    Uses the settings ``_compile_repeated_blocks`` compiled with and the DiT's compile guard. Returns the blocks moved.
    ``UNSLOTH_H3_COMPILE_BELOW_HOOKS=0`` keeps the old placement."""
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

    diffusers builds the top-level group (token refiner, embedders, adaLN tables, output projections: 0.81 GB of the
    int8 H3 denoiser) WITHOUT a stream, so whenever it is not resident every forward uploads it from pageable memory
    and its offload ``.to("cpu")``s every tensor back into fresh pageable buffers. Profiled at a 12 GB budget: 53
    device-to-pageable copies per step, 0.37 s of copy-back plus 0.08 s of pageable upload per 1.5 s step, with the
    GPU waiting on both. With a pinned copy the upload is a blocking copy from pinned memory (so the forward never
    reads a half-copied weight) and the offload only re-points each tensor at its host copy. torchao weights are
    pinned and restored by diffusers' own torchao paths, which is why #12389's top-level pin (plain tensors only) does
    not cover this checkpoint. ``UNSLOTH_H3_TOP_GROUP_PIN=0`` keeps diffusers' group."""
    if str(os.environ.get(H3_TOP_GROUP_PIN_ENV, "")).strip().lower() in ("0", "off", "false", "no"):
        return False
    top, blocks = h3_offload_groups(transformer)
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
        # The instance onload_ / offload_ wrappers (diffusion_prequant's inference-mode guard) call the class
        # methods, which read stream / cpu_param_dict, so they are not a reason to stop; these two are.
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
        # Blocking: nothing prefetches the top-level group, so the forward must not start before its copy lands.
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


H3_VAE_PINNED_SWAP_ENV = "UNSLOTH_H3_VAE_PINNED_SWAP"


def install_pinned_swap(
    module: Any,
    *,
    logger: Any = None,
    label: str = "component",
) -> bool:
    """Make a ComponentsManager-rotated module move between host and device by RE-POINTING at a pinned host copy.

    The rotation moves a component with ``module.to(device)`` and parks it with ``module.to("cpu")``: an upload from
    pageable memory, then a device-to-host copy into freshly allocated pageable memory, for weights a decode never
    changes. Once the conditioner and the denoiser stream, the rotation holds only MiniMax-H3's two VAEs, and they
    evict each other inside every render: measured at a 24 GB budget the video decode took 3.0 s and the audio decode
    3.0 s, against 1.08 s and 0.05 s on a card where both stay resident.

    Pins the module's CPU weights in place (``pin_module_in_place``: the pageable copy is replaced, so host RAM does
    not grow), then overrides ``to`` on the instance: a device move uploads each weight from its pinned copy without
    blocking the host, a CPU move re-points each weight at its pinned copy (no copy at all). Anything else (a dtype
    change, a weight whose dtype / shape no longer matches its pinned copy, a tensor added after the install) takes
    the stock ``nn.Module.to`` path. Returns True when installed. ``UNSLOTH_H3_VAE_PINNED_SWAP=0`` keeps stock moves."""
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

    from .video_minimax_h3_te import h3_te_pin_allowed, pin_module_in_place

    if not torch.cuda.is_available() or not h3_te_pin_allowed():
        return False
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors or any(t.device.type != "cpu" for t in tensors):
        # Only from a parked state: every weight is on the host and its host copy is the pinned one.
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
            # Whatever did not match moves the stock way; the matched ones are already where they belong.
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stream text-encoder layers through the card while a DiT training run encodes its captions.

The conditioning phase used to move each whole bf16 text encoder onto the card (Qwen3-VL 17.5 GB for Qwen-Image-2.1,
UMT5-XXL 11.4 GB for Wan2.2, Qwen3 8 GB for FLUX.2 klein 4B), which is more than the nf4 training loop needs, so the
encoder, not training, decided whether a run fit. Here the encoder's non-layer weights (embeddings, norms, any vision
tower) go to the card and its decoder layers stay on the host: each layer is copied in on a side stream ``DEPTH``
layers ahead and dropped after its forward. Compute still runs on the card with the same weights and kernels, so the
embeddings are bit-identical to the whole-module path.

``UNSLOTH_DIFFUSION_TRAIN_TE_STREAM``: ``auto`` (default) streams only when the encoders do not fit in free VRAM, and
then only as many layers as needed; ``on`` streams every layer; ``off`` is the whole-module move.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

TE_STREAM_ENV = "UNSLOTH_DIFFUSION_TRAIN_TE_STREAM"
TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")
DEPTH = 2
# Headroom for the encoder's activations and the CUDA workspace.
_MARGIN_BYTES = 3 << 29
_PIN_RESERVE_MIN_BYTES = 4 << 30
_PIN_RESERVE_FRACTION = 0.15
STREAMER_ATTR = "_unsloth_te_streamer"
LAST_ERROR: Optional[BaseException] = None


def te_stream_mode() -> str:
    raw = os.environ.get(TE_STREAM_ENV, "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "all"):
        return "on"
    return "auto"


def _tensor_bytes(t: Any) -> int:
    return t.numel() * t.element_size()


def _module_bytes(module: Any) -> int:
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            total += _tensor_bytes(t)
    return total


def _on_host(module: Any) -> bool:
    return all(p.device.type == "cpu" for p in module.parameters())


def _quantized(module: Any) -> bool:
    if getattr(module, "is_quantized", False) or getattr(module, "quantization_method", None):
        return True
    return any(type(p).__name__ in ("Params4bit", "Int8Params") for p in module.parameters())


def find_layers(module: Any) -> Optional[Any]:
    """The heaviest ModuleList of >= 2 weighted children (decoder layers / T5 blocks), else None."""
    import torch

    best, best_bytes = None, 0
    for mod in module.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
            sizes = [_module_bytes(c) for c in mod]
            if min(sizes) > 0 and sum(sizes) > best_bytes:
                best, best_bytes = mod, sum(sizes)
    return best


def _pin_budget() -> int:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return max(
            0, vm.available - max(_PIN_RESERVE_MIN_BYTES, int(vm.total * _PIN_RESERVE_FRACTION))
        )
    except Exception:  # noqa: BLE001 -- unknown RAM: copy from pageable memory
        return 0


_ALIGN = 512


def _aligned(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


class _Registered:
    """Page-locks ``nbytes`` of a pageable host buffer in place (exact size: the pinned allocator rounds every
    allocation up to a power of two); unregistered before the buffer can be freed."""

    def __init__(self, buf: Any, nbytes: int, stream: Any):
        import torch

        self.buf, self.ptr, self.stream = buf, None, stream
        rc = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {rc}")
        self.ptr = buf.data_ptr()

    def release(self) -> None:
        if self.ptr is not None:
            try:
                import torch

                # A prefetch still in flight reads the registered range.
                self.stream.synchronize()
                torch.cuda.cudart().cudaHostUnregister(self.ptr)
            except Exception:  # noqa: BLE001
                pass
            self.ptr = None

    def __del__(self):
        self.release()


class _Layer:
    __slots__ = ("params", "shapes", "sizes", "offsets", "start", "nbytes", "event", "resident")

    def __init__(self, layer: Any, start: int):
        self.params = [p for p in layer.parameters()]
        # Recorded up front: an evicted parameter is an empty placeholder.
        self.shapes = [tuple(p.shape) for p in self.params]
        self.sizes = [_tensor_bytes(p) for p in self.params]
        self.offsets, off = [], 0
        for n in self.sizes:
            self.offsets.append(off)
            off += _aligned(n)
        self.start, self.nbytes = start, off
        self.event = None
        self.resident = False


class TextEncoderStreamer:
    """Hooks on ``layers[i]`` for ``i in streamed``: fetch DEPTH layers ahead on a side stream, drop after forward.

    Each streamed layer's weights are packed back to back in one host arena (page-locked in place up to the pin
    budget), so a fetch is one host-to-device copy into one device buffer the parameters then view."""

    def __init__(
        self, encoder: Any, layers: Any, streamed: list[int], device: Any, pin_budget: int
    ):
        import torch

        self.order = sorted(streamed)
        owners: dict[int, int] = {}
        for li, layer in enumerate(layers):
            for p in layer.parameters():
                owners[id(p)] = owners.get(id(p), 0) + 1
        if any(n > 1 for n in owners.values()):
            # A weight shared by two layers would be evicted under the other: keep the whole-module move.
            raise ValueError("text encoder layers share parameters")
        self.device = torch.device(device)
        self.side = torch.cuda.Stream(device = self.device)
        self.handles = []
        self.fetches = 0
        self.pinned_bytes = 0
        self.blocks, total = [], 0
        for li in self.order:
            blk = _Layer(layers[li], total)
            self.blocks.append(blk)
            total += blk.nbytes
        streamed_ids = {id(m) for li in self.order for m in layers[li].modules()}
        with torch.no_grad():
            # Everything but the streamed layers' parameters lives on the card for the whole encode.
            for mod in encoder.modules():
                if id(mod) not in streamed_ids:
                    for _name, p in mod.named_parameters(recurse = False):
                        if p.device != self.device:
                            p.data = p.data.to(self.device)
                for name, b in list(mod.named_buffers(recurse = False)):
                    if b is not None and b.device != self.device:
                        mod._buffers[name] = b.to(self.device)
            self.arena = torch.empty(total, dtype = torch.uint8)
            for blk in self.blocks:
                for p, off, n in zip(blk.params, blk.offsets, blk.sizes):
                    region = self.arena[blk.start + off : blk.start + off + n]
                    region.view(p.dtype).view(p.shape).copy_(p.data)
                    # The host copy now lives in the arena: drop the original as each layer is packed.
                    p.data = torch.empty(0, dtype = p.dtype, device = self.device)
        # Pin whole layers only: a copy from a half-pinned range is staged synchronously.
        pin = 0
        for blk in self.blocks:
            if blk.start + blk.nbytes <= pin_budget:
                pin = blk.start + blk.nbytes
        self.registered = None
        if pin > 0:
            try:
                self.registered = _Registered(self.arena, pin, self.side)
                self.pinned_bytes = pin
            except Exception:  # noqa: BLE001 -- pageable copies: slower, same bytes
                self.registered = None
        self.pos = {li: k for k, li in enumerate(self.order)}
        for li in self.order:
            self.handles.append(layers[li].register_forward_pre_hook(self._pre(self.pos[li])))
            self.handles.append(layers[li].register_forward_hook(self._post(self.pos[li])))
        self._arm()

    def _fetch(self, k: int) -> None:
        import torch

        blk = self.blocks[k]
        if blk.resident:
            return
        cur = torch.cuda.current_stream(self.device)
        # Allocated on the compute stream, written on the side stream after everything already queued (including the
        # last reads of whatever block these bytes held before), read on the compute stream after the event.
        buf = torch.empty(blk.nbytes, dtype = torch.uint8, device = self.device)
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            buf.copy_(self.arena[blk.start : blk.start + blk.nbytes], non_blocking = True)
            blk.event = torch.cuda.Event()
            blk.event.record(self.side)
        for p, off, n, shape in zip(blk.params, blk.offsets, blk.sizes, blk.shapes):
            p.data = buf[off : off + n].view(p.dtype).view(shape)
        blk.resident = True
        self.fetches += 1

    def _drop(self, k: int) -> None:
        import torch

        blk = self.blocks[k]
        if not blk.resident:
            return
        for p in blk.params:
            p.data = torch.empty(0, dtype = p.dtype, device = self.device)
        blk.resident = False
        blk.event = None

    def _arm(self) -> None:
        for k in range(min(DEPTH, len(self.blocks))):
            self._fetch(k)

    def _pre(self, k: int):
        def hook(module, args):
            import torch

            self._fetch(k)
            ev = self.blocks[k].event
            if ev is not None:
                torch.cuda.current_stream(self.device).wait_event(ev)
            for j in range(k + 1, min(k + 1 + DEPTH, len(self.blocks))):
                self._fetch(j)

        return hook

    def _post(self, k: int):
        def hook(module, args, output):
            self._drop(k)
            if k == len(self.blocks) - 1:
                # The next caption's forward starts at the first streamed layer.
                self._arm()
            return output

        return hook

    def host_tensor(self, k: int, i: int) -> Any:
        """The host copy of parameter ``i`` of streamed layer ``k`` (a view into the arena)."""
        blk = self.blocks[k]
        start = blk.start + blk.offsets[i]
        return (
            self.arena[start : start + blk.sizes[i]].view(blk.params[i].dtype).view(blk.shapes[i])
        )

    def remove(self) -> None:
        import torch

        for h in self.handles:
            h.remove()
        self.handles = []
        for k in range(len(self.blocks)):
            self._drop(k)
        if self.registered is not None:
            self.registered.release()
            self.registered = None


def release_text_encoders(pipe: Any) -> None:
    """Unhook and unpin every streamed encoder of ``pipe`` (before the trainer drops them)."""
    for attr in TEXT_ENCODER_ATTRS:
        enc = getattr(pipe, attr, None)
        streamer = getattr(enc, STREAMER_ATTR, None) if enc is not None else None
        if streamer is not None:
            try:
                streamer.remove()
            except Exception:  # noqa: BLE001 -- the encoder is dropped either way
                pass


def _plan_resident(layer_bytes: list[int], budget: int) -> int:
    """Leading layers kept on the card so they plus the streaming window fit in ``budget``."""
    window = (DEPTH + 1) * (max(layer_bytes) if layer_bytes else 0)
    keep, used = 0, window
    for b in layer_bytes:
        if used + b > budget:
            break
        used += b
        keep += 1
    return keep


def place_text_encoders(pipe: Any, device: str) -> list[str]:
    """Stream the host-resident text encoders of ``pipe`` per ``te_stream_mode()``; returns the attrs it streamed.

    Encoders it does not stream are left for the caller's whole-module move: quantized or already-placed encoders,
    a non-CUDA device, ROCm under ``auto``, a layout with no layer list, or encoders that fit."""
    mode = te_stream_mode()
    if mode == "off" or device != "cuda":
        return []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        if mode == "auto" and getattr(torch.version, "hip", None):
            return []
        cands = []
        for attr in TEXT_ENCODER_ATTRS:
            enc = getattr(pipe, attr, None)
            if enc is None or getattr(enc, STREAMER_ATTR, None) is not None:
                continue
            if not _on_host(enc) or _quantized(enc):
                continue
            layers = find_layers(enc)
            if layers is None:
                continue
            cands.append((attr, enc, layers))
        if not cands:
            return []
        free, _total = torch.cuda.mem_get_info()
        need = sum(_module_bytes(enc) for _a, enc, _l in cands)
        if mode == "auto" and need + _MARGIN_BYTES <= free:
            return []
        budget = (
            free
            - _MARGIN_BYTES
            - sum(
                _module_bytes(enc) - sum(_module_bytes(layer) for layer in layers)
                for _a, enc, layers in cands
            )
        )
        from core.inference.host_pinning import host_pinning_allowed

        pin_budget = _pin_budget() if host_pinning_allowed() else 0
        streamed_attrs = []
        for attr, enc, layers in cands:
            sizes = [_module_bytes(layer) for layer in layers]
            keep = 0 if mode == "on" else _plan_resident(sizes, budget)
            if keep >= len(sizes):
                # Fits after all (an earlier encoder took the streaming): the caller moves it whole.
                budget -= sum(sizes)
                continue
            with torch.no_grad():
                for li in range(keep):
                    layers[li].to(device)
            streamer = TextEncoderStreamer(
                enc, layers, list(range(keep, len(sizes))), device, pin_budget
            )
            pin_budget = max(0, pin_budget - streamer.pinned_bytes)
            budget -= sum(sizes[:keep]) + (DEPTH + 1) * max(sizes)
            setattr(enc, STREAMER_ATTR, streamer)
            streamed_attrs.append(attr)
            logging.getLogger(__name__).info(
                "Streaming %d of %d %s layers through the GPU while encoding captions "
                "(%.1f GiB of text encoders, %.1f GiB free, %.1f GiB pinned).",
                len(sizes) - keep,
                len(sizes),
                attr,
                need / 2**30,
                free / 2**30,
                streamer.pinned_bytes / 2**30,
            )
        return streamed_attrs
    except Exception as exc:  # noqa: BLE001 -- an encoder half-streamed is undone; the caller moves the rest whole
        global LAST_ERROR
        LAST_ERROR = exc
        for attr in TEXT_ENCODER_ATTRS:
            enc = getattr(pipe, attr, None)
            streamer = getattr(enc, STREAMER_ATTR, None) if enc is not None else None
            if streamer is not None:
                _unstream(enc, streamer)
        return []


def _unstream(enc: Any, streamer: TextEncoderStreamer) -> None:
    """Put the host copies back so a whole-module ``.to(device)`` sees real weights."""
    streamer.remove()
    for k, blk in enumerate(streamer.blocks):
        for i, p in enumerate(blk.params):
            p.data = streamer.host_tensor(k, i).clone()
    try:
        delattr(enc, STREAMER_ATTR)
    except AttributeError:
        pass

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-encoder residency on a discrete CUDA card, re-decided per render.

The load places the encoders from the planner's measured peaks: whole on the card (``plain``), or partly in resident
offload groups (``groups``). This keeps that placement only while it still fits: after each prompt encode, the render's
priced peak at its own size and batch (the learned / seeded peak the planner uses, plus what the measured placement
reserved) and the planner's residency slack must fit in live free memory; when they do not (a bigger canvas, a batch,
another process took VRAM), the encoders stream from the host before the denoise. Another Studio load on the card asks
for the same room first (``lend``). The encoders come back before a later prompt's encode once they fit again.

Streaming reuses the existing mechanisms: leaf-level group offload from the host (the VAE out-of-memory path's apply,
the streamed-encoder tier's hooks) for ``plain`` encoders, ``release_resident_groups`` for ``groups``. The same weights
run either way, so outputs are bit-identical. ``UNSLOTH_DIFFUSION_TE_RESIDENCY=0`` keeps the load's placement for the
whole session, as before."""

from __future__ import annotations

import os
import threading
import time
import weakref
from typing import Any, Callable, Optional

RESIDENCY_ENV = "UNSLOTH_DIFFUSION_TE_RESIDENCY"
ATTR = "_unsloth_te_residency"
TEXT_ENCODER_PREFIX = "text_encoder"
# Host RAM kept free beside the streamed copy (the VAE out-of-memory path's margin).
HOST_MARGIN_MIB = 2048
KIND_PLAIN = "plain"
KIND_GROUPS = "groups"


def enabled() -> bool:
    return (os.environ.get(RESIDENCY_ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def _note(logger: Any, level: str, msg: str, *args: Any) -> None:
    if logger is not None:
        getattr(logger, level)(msg, *args)


def _encoders(pipe: Any) -> dict[str, Any]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {}
    return {
        str(name): module
        for name, module in (getattr(pipe, "components", {}) or {}).items()
        if str(name).startswith(TEXT_ENCODER_PREFIX) and isinstance(module, torch.nn.Module)
    }


def _holds_subclass_weights(module: Any) -> bool:
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
    except Exception:  # noqa: BLE001
        return False
    return any(
        is_traceable_wrapper_subclass(p.data if hasattr(p, "data") else p)
        for p in module.parameters()
    )


def slack_mib(total_mib: int) -> int:
    """The planner's residency slack (``_resident_dit_slack_mib``): 10% of the card, at least 1 GiB."""
    from .diffusion_memory import _RESIDENT_DIT_SLACK_FRACTION, _RESIDENT_DIT_SLACK_MIN_MIB
    return max(_RESIDENT_DIT_SLACK_MIN_MIB, int(int(total_mib) * _RESIDENT_DIT_SLACK_FRACTION))


def live_free_mib(device: Any) -> Optional[tuple[int, int]]:
    """``(free MiB, total MiB)`` on ``device``: driver free plus the allocator cache an ordinary allocation can reuse
    (not a CUDA graph pool's), less the user's VRAM Budget reserve, as ``release_for_decode`` reads it. Driver free is
    the load planner's reading (``trusted_mem_get_info``: Windows ROCm over-reports it). None when unreadable."""
    try:
        import torch

        from utils.hardware import trusted_mem_get_info

        from .diffusion_memory import _graph_held_cache_bytes, user_budget_reserve_bytes

        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        free, total = (int(x) for x in trusted_mem_get_info(dev))
        free -= user_budget_reserve_bytes(dev, total)
        cached = max(
            0,
            int(torch.cuda.memory_reserved(dev))
            - int(torch.cuda.memory_allocated(dev))
            - _graph_held_cache_bytes(dev.index),
        )
        return (free + cached) >> 20, total >> 20
    except Exception:  # noqa: BLE001
        return None


def render_need_mib(
    pipe: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    batch_size: int = 1,
    extra_mib: int = 0,
    measured_activation: Any = None,
    tiled: bool = False,
) -> Optional[int]:
    """MiB this render needs above the weights once the prompt is encoded (margin included). None when unpriced (the
    placement then stands).

    Encoders in resident groups: the planner's own quantity, its measured placement reserve plus what this request
    needs beyond it (the decode is priced by the decode lend, which streams the same groups first). Whole-resident
    encoders: the largest of that reserve, the learned / seeded peak for this family, quant and tier at this size and
    batch, and the calibrated per-phase row the placement was sized from (denoise and decode scaled by pixels, a tiled
    decode capped at its tile), plus a priced LoRA forward."""
    from .diffusion_memory import (
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        measured_image_runtime_mib,
    )

    residency = of(pipe)
    batch = max(1, int(batch_size or 1))
    reserve = getattr(pipe, "_unsloth_measured_reserve", None)
    reserved: Optional[int] = None
    if reserve:
        try:
            reserved = int(reserve[0]) + max(0, int(extra_mib or 0))
        except Exception:  # noqa: BLE001
            reserved = None
    if residency is not None and residency.kind == KIND_GROUPS:
        from .diffusion_memory import DECODE_LEND_ATTR, MEASURED_RESERVE_DENOISE_PHASE_INDEX

        phase = (
            reserve[MEASURED_RESERVE_DENOISE_PHASE_INDEX]
            if reserve and len(reserve) > MEASURED_RESERVE_DENOISE_PHASE_INDEX
            else None
        )
        lend = getattr(pipe, DECODE_LEND_ATTR, None)
        if reserved is not None and lend:
            # The decode lend streams these groups again when the decode is short, so price the denoise here: the
            # phase the placement was sized from, else the whole-render reserve less what it lent to the decode.
            try:
                denoise = int(phase) if phase else int(reserve[0]) - int(lend[0])
                return max(0, denoise) + max(0, int(extra_mib or 0))
            except Exception:  # noqa: BLE001
                return reserved
        return reserved
    candidates: list[int] = [] if reserved is None else [reserved]
    identity = getattr(pipe, "_unsloth_peak_identity", None)
    if isinstance(identity, tuple) and len(identity) == 3:
        try:
            measured = measured_image_runtime_mib(
                identity[0],
                identity[2],
                width = width,
                height = height,
                batch_size = batch,
                quant = identity[1],
            )
        except Exception:  # noqa: BLE001
            measured = None
        if measured:
            candidates.append(int(measured))
    if measured_activation is not None:
        try:
            w = max(64, int(width or DEFAULT_IMAGE_WIDTH))
            h = max(64, int(height or DEFAULT_IMAGE_HEIGHT))
            scale = max(1.0, w * h * batch / float(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT))
            denoise = int(measured_activation.denoise_mib * scale)
            decode = (
                int(measured_activation.tiled_decode_mib)
                if tiled
                else int(measured_activation.decode_mib * scale)
            )
            candidates.append(max(denoise, decode))
        except Exception:  # noqa: BLE001
            pass
    if candidates and getattr(pipe, "_unsloth_loras", None):
        try:
            from .diffusion_lora_memory import request_extra_mib as lora_request_extra_mib
            lora = int(
                lora_request_extra_mib(
                    pipe, width = width, height = height, batch_size = batch, extra_pixels = 0
                )
                or 0
            )
            candidates = [c + lora for c in candidates]
        except Exception:  # noqa: BLE001 - an unpriced LoRA forward: the figures above stand
            pass
    return max(candidates) if candidates else None


class EncoderResidency:
    """Keeps a pipeline's text encoders on the card while the render fits, streams them from the host when not."""

    def __init__(
        self,
        pipe: Any,
        device: Any,
        kind: str,
        logger: Any = None,
    ) -> None:
        import torch

        from .diffusion_vae_oom import _nbytes

        self._pipe = weakref.ref(pipe)
        self.device = torch.device(device)
        self.kind = kind
        self.logger = logger
        self.names = tuple(_encoders(pipe))
        self.mib = sum(_nbytes(m) for m in _encoders(pipe).values()) >> 20
        self.streamed = False
        # torchao / tensor-subclass encoder weights cannot move under inference_mode (aliasing check): renders that may
        # stream or restore them run under no_grad (diffusion._torchao_render_needs_no_grad).
        self.moves_need_grad_mode = any(
            _holds_subclass_weights(m) for m in _encoders(pipe).values()
        )
        self._restore_groups: Optional[Callable[..., None]] = None
        self.released_mib = 0
        self.releases = 0
        self.restores = 0
        self.decisions: list[dict] = []
        self._lock = threading.RLock()

    # ---- state ------------------------------------------------------------------------------------------------
    def _record(self, **entry: Any) -> dict:
        entry.setdefault("t", round(time.time(), 3))
        entry.setdefault("kind", self.kind)
        self.decisions.append(entry)
        del self.decisions[:-64]
        return entry

    @property
    def last(self) -> Optional[dict]:
        return self.decisions[-1] if self.decisions else None

    def _modules(self) -> dict[str, Any]:
        pipe = self._pipe()
        if pipe is None:
            return {}
        found = _encoders(pipe)
        return {n: found[n] for n in self.names if n in found}

    def resident_mib(self) -> int:
        """MiB of the encoders on the card now: whole (``plain``), or their groups pinned resident (``groups``)."""
        if self.kind != KIND_GROUPS:
            return 0 if self.streamed else int(self.mib)
        from .diffusion_memory import _offload_groups

        total = 0
        for module in self._modules().values():
            for group in _offload_groups(module) or []:
                if getattr(group, "_unsloth_resident", False):
                    total += int(getattr(group, "_unsloth_resident_bytes", 0) or 0)
        return total >> 20

    def _still_plain(self) -> bool:
        from .diffusion_vae_oom import _placement
        modules = self._modules()
        return bool(modules) and all(
            _placement(m, self.device) in ("plain", "torchao_resident") for m in modules.values()
        )

    # ---- moves ------------------------------------------------------------------------------------------------
    def release(
        self,
        reason: str,
        short_mib: Optional[int] = None,
    ) -> int:
        """Stream the encoders from the host; MiB freed on the card (0 when nothing moved)."""
        with self._lock:
            if self.kind == KIND_GROUPS:
                return self._release_groups(reason, short_mib)
            if self.streamed:
                return 0
            return self._release_plain(reason)

    def _release_plain(self, reason: str) -> int:
        import torch

        from .diffusion_vae_oom import _empty_cache, _host_free_mib, _return_to

        if not self._still_plain():
            self._record(
                action = "keep", why = "the encoders are no longer whole on the card", reason = reason
            )
            return 0
        host = _host_free_mib()
        if host is not None and self.mib + HOST_MARGIN_MIB > host:
            self._record(action = "keep", why = f"host RAM is short ({host} MiB free)", reason = reason)
            _note(self.logger, "warning", "diffusion.te_residency: %d MiB of text encoders stay on the card for %s: "
                  "host RAM is short (%d MiB free)", self.mib, reason, host)  # fmt: skip
            return 0
        started = time.perf_counter()
        modules = self._modules()
        done: list[Any] = []
        try:
            for module in modules.values():
                module.to("cpu")
                done.append(module)
                _apply_stream(module, self.device, self.logger)
        except Exception as exc:  # noqa: BLE001 - back as it was: whole on the card
            from .diffusion_memory import _remove_group_offload_hooks

            for module in done:
                try:
                    _remove_group_offload_hooks(module)
                    _return_to(module, self.device)
                except Exception:  # noqa: BLE001
                    pass
            self._record(action = "keep", why = f"streaming failed: {exc}", reason = reason)
            _note(
                self.logger,
                "warning",
                "diffusion.te_residency: could not stream the text encoders (%s)",
                exc,
            )
            return 0
        _empty_cache(torch.device(self.device))
        self.streamed = True
        self.released_mib = self.mib
        self.releases += 1
        seconds = time.perf_counter() - started
        self._record(action = "release", mib = self.mib, seconds = round(seconds, 3), reason = reason)
        _note(self.logger, "info", "diffusion.te_residency: %s (%d MiB) stream from the host for %s (%.2f s)",
              ", ".join(modules), self.mib, reason, seconds)  # fmt: skip
        return self.mib

    def _release_groups(self, reason: str, short_mib: Optional[int]) -> int:
        from .diffusion_memory import release_resident_groups, resident_group_mib

        pipe = self._pipe()
        if pipe is None:
            return 0
        before = resident_group_mib(pipe)
        need = int(short_mib) if short_mib else int(self.mib)
        restore = (
            release_resident_groups(
                pipe, need, self.logger, reason = reason, encoders_only = True, device = self.device
            )
            if self.resident_mib() > 0
            else None
        )
        if restore is None:
            self._record(action = "keep", why = "no resident encoder groups", reason = reason)
            return 0
        freed = max(0, before - resident_group_mib(pipe))
        held = self._restore_groups
        if held is None:
            self._restore_groups = restore
        else:
            # a further release while partly streamed: one restore re-pins both, the newest first

            def both(
                strict: bool = False,
                _first = restore,
                _then = held,
            ) -> None:
                try:
                    _first(strict = strict)
                finally:
                    _then(strict = strict)

            self._restore_groups = both
        self.streamed = True
        self.released_mib += freed
        self.releases += 1
        self._record(action = "release", mib = freed, reason = reason)
        return freed

    def restore(self, reason: str) -> bool:
        """Put the encoders back on the card; True when they are resident afterwards."""
        with self._lock:
            if not self.streamed:
                return True
            started = time.perf_counter()
            if self.kind == KIND_GROUPS:
                restore, self._restore_groups = self._restore_groups, None
                try:
                    if restore is not None:
                        restore(strict = True)
                except Exception as exc:  # noqa: BLE001 - the groups that did not fit keep streaming
                    self._record(action = "stay_streamed", why = str(exc), reason = reason)
                    _note(
                        self.logger,
                        "warning",
                        "diffusion.te_residency: re-pinning the encoder groups failed (%s)",
                        exc,
                    )
                    self.streamed = (
                        False  # what re-pinned is resident; the rest streams as the flat plan
                    )
                    return False
            else:
                ok, exc = _unstream(self._modules(), self.device)
                if not ok:
                    self._record(action = "stay_streamed", why = str(exc), reason = reason)
                    _note(
                        self.logger,
                        "warning",
                        "diffusion.te_residency: the text encoders keep streaming (%s)",
                        exc,
                    )
                    return False
            self.streamed = False
            self.restores += 1
            seconds = time.perf_counter() - started
            self._record(
                action = "restore", mib = self.released_mib, seconds = round(seconds, 3), reason = reason
            )
            _note(self.logger, "info", "diffusion.te_residency: %d MiB of text encoders back on the card for %s (%.2f s)",
                  self.released_mib, reason, seconds)  # fmt: skip
            self.released_mib = 0
            return True

    # ---- decisions --------------------------------------------------------------------------------------------
    def restore_slack_mib(self, total_mib: int) -> int:
        """Kept free beside the encoders and the render's priced peak before streamed encoders come back: the release
        slack plus the decode lend's slack again, so a render at the edge does not move them every prompt."""
        from .diffusion_memory import _DECODE_SLACK_MIB

        if self.kind == KIND_GROUPS:
            return self.release_slack_mib(total_mib) + int(_DECODE_SLACK_MIB)
        return slack_mib(total_mib)

    def release_slack_mib(self, total_mib: int) -> int:
        """Kept free beside the render's priced peak before resident encoders stream. Whole-resident encoders: the
        planner's residency slack. Resident groups were packed into the room the planner left after its own slack, so
        they keep the decode lend's slack (``release_for_decode``), else a steady render would stream them."""
        if self.kind == KIND_GROUPS:
            from .diffusion_memory import _DECODE_SLACK_MIB
            return int(_DECODE_SLACK_MIB)
        return slack_mib(total_mib)

    def before_encode(self, need_mib: Optional[int]) -> Optional[dict]:
        """Before a prompt encode: streamed encoders come back when they, this render's peak and the slack fit."""
        with self._lock:
            if not self.streamed:
                return None
            live = live_free_mib(self.device)
            if live is None or need_mib is None:
                return self._record(
                    action = "stay_streamed", why = "free memory or render peak unknown"
                )
            free, total = live
            want = int(self.released_mib) + int(need_mib) + self.restore_slack_mib(total)
            if free < want:
                return self._record(action = "stay_streamed", free_mib = free, need_mib = int(need_mib),
                                    want_mib = want, why = "the encoders, the render peak and the slack do not fit")  # fmt: skip
            self.restore(
                f"a render that fits with them resident ({free} MiB free, {want} MiB wanted)"
            )
            return self.last

    def after_encode(self, need_mib: Optional[int]) -> Optional[dict]:
        """Prompt encoded: keep resident encoders when the rest of the render still fits with the slack, else stream
        the shortfall from the host before the denoise allocates."""
        with self._lock:
            if self.streamed and (self.kind != KIND_GROUPS or self.resident_mib() <= 0):
                return None
            live = live_free_mib(self.device)
            if live is None or need_mib is None:
                return self._record(action = "keep", why = "free memory or render peak unknown")
            free, total = live
            want = int(need_mib) + self.release_slack_mib(total)
            if free >= want:
                return self._record(
                    action = "keep", free_mib = free, need_mib = int(need_mib), want_mib = want
                )
            self.release(
                f"a render that does not fit beside them ({free} MiB free, {want} MiB wanted)",
                short_mib = want - free,
            )
            entry = self.last or {}
            entry.update(free_mib = free, need_mib = int(need_mib), want_mib = want)
            return entry

    def lend(
        self,
        reason: str,
        at_least_mib: int = 0,
        need_mib: Optional[int] = None,
    ) -> int:
        """Another load wants the card: stream the encoders now (they come back at a later encode if they fit). Not
        when they hold less than ``at_least_mib``: the caller would have to free the whole model anyway. Not when
        ``need_mib`` (the other load's size) and the slack already fit in live free memory: nothing is short."""
        with self._lock:
            resident = self.resident_mib()
            if resident <= 0 or resident < int(at_least_mib or 0):
                return 0
            if need_mib is not None:
                live = live_free_mib(self.device)
                if live is not None and live[0] >= int(need_mib) + slack_mib(live[1]):
                    self._record(action = "keep", free_mib = live[0], need_mib = int(need_mib),
                                 why = "the other load fits beside them", reason = reason)  # fmt: skip
                    return 0
            freed = self.release(reason, short_mib = resident)
            if freed > 0 and self.kind == KIND_GROUPS:
                # the other load is usually another process: hand the freed blocks back to the driver
                from .diffusion_vae_oom import _empty_cache
                _empty_cache(self.device)
            return freed


def _apply_stream(module: Any, device: Any, logger: Any) -> None:
    """Leaf-level group offload from pageable host memory (no stream, no pinned copy): the VAE out-of-memory path's
    apply, with the streamed-encoder tier's hook fixes."""
    import torch
    from diffusers.hooks import apply_group_offloading

    from .diffusion_memory import (
        _apply_group_offloading_packed,
        _pin_vision_embedding_device,
        install_group_offload_buffer_restore,
        install_group_offload_hooks_eager,
    )

    install_group_offload_buffer_restore()
    install_group_offload_hooks_eager()
    kwargs = {
        "onload_device": torch.device(device),
        "offload_device": torch.device("cpu"),
        "offload_type": "leaf_level",
        "use_stream": False,
    }
    _apply_group_offloading_packed(apply_group_offloading, module, kwargs, logger)
    _pin_vision_embedding_device(module)


def _unstream(modules: dict[str, Any], device: Any) -> tuple[bool, Optional[BaseException]]:
    """Hooks off and weights back on the card; if that fails (out of memory), every encoder streams again."""
    import torch

    from .diffusion_memory import _remove_group_offload_hooks
    from .diffusion_vae_oom import _empty_cache, _return_to

    try:
        for module in modules.values():
            _remove_group_offload_hooks(module)
            _return_to(module, device)
        return True, None
    except Exception as exc:  # noqa: BLE001
        for module in modules.values():
            try:
                _remove_group_offload_hooks(module)
                module.to("cpu")
                _apply_stream(module, device, None)
            except Exception:  # noqa: BLE001
                pass
        _empty_cache(torch.device(device))
        return False, exc


def _placement_kind(pipe: Any, device: Any) -> Optional[str]:
    from .diffusion_vae_oom import _placement

    modules = list(_encoders(pipe).values())
    if not modules:
        return None
    kinds = {_placement(m, device) for m in modules}
    if kinds <= {"plain", "torchao_resident"}:
        return KIND_PLAIN
    if kinds == {"groups"}:
        return KIND_GROUPS
    return None


def _encoder_device(pipe: Any) -> Any:
    for module in _encoders(pipe).values():
        for tensor in [*module.parameters(), *module.buffers()]:
            if tensor.device.type == "cuda":
                return tensor.device
    groups = [getattr(m, "_unsloth_resident_device", None) for m in _encoders(pipe).values()]
    return next((g for g in groups if g is not None), None)


def maybe_install(
    pipe: Any,
    plan: Any,
    *,
    logger: Any = None,
) -> Optional[EncoderResidency]:
    """Attach the per-render residency check when the encoders sit on one discrete CUDA card, whole or in resident
    groups. Never raises."""
    try:
        if pipe is None or not enabled():
            return None
        memory = getattr(plan, "device_memory", None)
        if getattr(memory, "is_unified", False) or getattr(memory, "device", "cuda") != "cuda":
            return None
        from .diffusion_multigpu import current_selection

        if current_selection() is not None:
            return None  # a layer split places the encoders itself
        device = _encoder_device(pipe)
        if device is None or getattr(device, "type", None) != "cuda":
            return None
        kind = _placement_kind(pipe, device)
        if kind is None:
            return None
        residency = EncoderResidency(pipe, device, kind, logger = logger)
        if residency.mib <= 0:
            return None
        setattr(pipe, ATTR, residency)
        _note(logger, "info", "diffusion.te_residency: %s (%d MiB, %s) re-checked per render on %s",
              ", ".join(residency.names), residency.mib, kind, device)  # fmt: skip
        return residency
    except Exception as exc:  # noqa: BLE001 - the load's placement stands
        _note(logger, "warning", "diffusion.te_residency: not installed (%s)", exc)
        return None


def of(pipe: Any) -> Optional[EncoderResidency]:
    return getattr(pipe, ATTR, None) if pipe is not None else None


def before_encode(
    pipe: Any,
    need_mib: Optional[int],
    logger: Any = None,
) -> Optional[dict]:
    residency = of(pipe)
    if residency is None:
        return None
    try:
        return residency.before_encode(need_mib)
    except Exception as exc:  # noqa: BLE001 - the encoders run where they are
        _note(logger, "warning", "diffusion.te_residency: pre-encode check skipped (%s)", exc)
        return None


def after_encode(
    pipe: Any,
    need_mib: Optional[int],
    logger: Any = None,
) -> Optional[dict]:
    residency = of(pipe)
    if residency is None:
        return None
    try:
        return residency.after_encode(need_mib)
    except Exception as exc:  # noqa: BLE001 - the encoders stay where they are
        _note(logger, "warning", "diffusion.te_residency: post-encode check skipped (%s)", exc)
        return None


def lend(
    pipe: Any,
    reason: str,
    logger: Any = None,
    at_least_mib: int = 0,
    need_mib: Optional[int] = None,
) -> int:
    residency = of(pipe)
    if residency is None:
        return 0
    try:
        return residency.lend(reason, at_least_mib, need_mib = need_mib)
    except Exception as exc:  # noqa: BLE001 - the other load proceeds as before
        _note(logger, "warning", "diffusion.te_residency: lend skipped (%s)", exc)
        return 0


def lend_from_loaded_engine(
    reason: str,
    at_least_mib: int = 0,
    need_mib: Optional[int] = None,
) -> int:
    """For a load outside the image engine (dictation, a chat model beside an image model): the loaded image engine,
    if any and idle, streams its resident text encoders first. ``need_mib``: the other load's size when known (it
    lends nothing while that already fits). Never imports the engine; never raises. MiB freed."""
    import sys
    try:
        module = sys.modules.get("core.inference.diffusion")
        engine = getattr(module, "_diffusion_backend", None) if module is not None else None
        lend_vram = getattr(engine, "lend_text_encoder_vram", None)
        if not callable(lend_vram):
            return 0
        if need_mib is None:
            return int(lend_vram(reason, at_least_mib) or 0)
        return int(lend_vram(reason, at_least_mib, need_mib = need_mib) or 0)
    except Exception:  # noqa: BLE001 - the other load proceeds as before
        return 0

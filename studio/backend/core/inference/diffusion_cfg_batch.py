# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batches CFG branches into one forward for stateless denoisers only; a second call differing from
the batched inputs reruns that branch plain; not always bit-exact; UNSLOTH_DIFFUSION_CFG_BATCH=0
keeps one branch per call"""

from __future__ import annotations

import os
import sys
import time
import weakref
from typing import Any, Optional

CFG_BATCH_ENV = "UNSLOTH_DIFFUSION_CFG_BATCH"

# Forward keeps no state between calls, both branches share every per-step input but the text conditioning. Z-Image is
# absent on purpose: its pipeline already batches the branches (a list of latents, one call per step).
STATELESS_CLASSES = frozenset(
    {
        "Flux2Transformer2DModel",
        "FluxTransformer2DModel",
        "QwenImageTransformer2DModel",
        "WanTransformer3DModel",
    }
)

# Arguments that are one entry per sample (a doubled call needs both branches' entries); compared, not concatenated.
_PER_SAMPLE_LISTS = ("img_shapes",)
# (embeddings, mask) pairs a branch pads to its own prompt length: padded to the longer one with masked zeros.
_PADDED_TEXT = (("encoder_hidden_states", "encoder_hidden_states_mask"),)

_MEASURED_FRACTION = 0.92
_BATCHED_SAMPLES = 2
_PLAIN_STEPS = 2
# Compute-bound resident pair: skip the trial (B200 winners 0.03-0.06 s/pair; T4 FLUX.2-klein ~4.2 s, trial +17 s).
_RESIDENT_MAX_PAIR_S = 0.25
_MISS_LIMIT = 3
_OFF_TRANSIENT = "transient"


def cfg_batch_mode() -> str:
    """``off`` / ``auto`` (default: a listed class, resident or offloaded) / ``force``. Independent of the multi-GPU
    switches: a single card is the usual case."""
    raw = (os.environ.get(CFG_BATCH_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    if raw == "force":
        return "force"
    return "auto"


def _torch():
    import torch
    return torch


def _mg():
    from . import diffusion_multigpu as mg
    return mg


class _NotBatchable(Exception):
    pass


def _streamed(module: Any) -> bool:
    """Offload machinery anywhere in the denoiser: its weights live off the card between calls."""
    mg = _mg()
    for sub in module.modules():
        if mg._registry_kind(sub) == "offload" or getattr(sub, "_hf_hook", None) is not None:
            return True
        override = sub.__dict__.get("forward")
        if override is not None and mg._is_offload_object(override):
            return True
    return False


def _foreign_hook(module: Any) -> Optional[str]:
    """A non-offload diffusers hook on the denoiser or a block (a step cache)."""
    mg = _mg()
    for name, sub in module.named_modules():
        if mg._registry_kind(sub) == "other":
            return f"{name or '<root>'} carries {', '.join(sorted(sub._diffusers_hook.hooks))}"
    return None


# Set on a denoiser whose blocks Studio compiled (regional compile; also read when the compile moves below an
# offload hook, which clears the block's ``_compiled_call_impl``).
_COMPILE_MARKS = ("_unsloth_regional_compile_kwargs", "_unsloth_compile_guard")


def _compiled_callable(fn: Any) -> bool:
    for _ in range(4):
        if fn is None:
            return False
        if hasattr(fn, "_torchdynamo_orig_callable") or hasattr(fn, "_unsloth_compile_guard"):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def compiled_denoiser(module: Any) -> Optional[str]:
    """Why ``module`` runs compiled code, else None. A doubled batch there is a new input shape: the batch 1 -> 2 change
    compiles a second graph per compiled block, which is only acceptable inside the warm-up render of a shape."""
    try:
        subs = list(module.named_modules())
    except Exception:  # noqa: BLE001 - not a torch module (fakes): nothing compiled
        return None
    for mark in _COMPILE_MARKS:
        if mark in getattr(module, "__dict__", {}):
            return "the denoiser is compiled"
    for name, sub in subs:
        if getattr(sub, "_compiled_call_impl", None) is not None:
            return f"{name or 'the denoiser'} is compiled"
        registry = getattr(sub, "__dict__", {}).get("_diffusers_hook")
        for ref in list(getattr(registry, "_fn_refs", None) or ()):
            if _compiled_callable(getattr(ref, "forward", None)):
                return f"{name or 'the denoiser'} runs a compiled forward below its hooks"
    return None


def _offloaded(denoiser: Any, offload_policy: str) -> bool:
    return offload_policy in ("group", "streaming", "sequential") and _streamed(denoiser)


def _graph_count() -> int:
    """Dynamo's compiled-graph count (0 before Dynamo was ever imported)."""
    dynamo_utils = sys.modules.get("torch._dynamo.utils")
    try:
        return (
            int(dynamo_utils.counters["stats"]["unique_graphs"]) if dynamo_utils is not None else 0
        )
    except Exception:  # noqa: BLE001
        return 0


def _foreign_layer(module: Any) -> Optional[str]:
    """A non-offload diffusers hook (a step cache) or a root forward override that is not offload machinery."""
    mg = _mg()
    hooked = _foreign_hook(module)
    if hooked is not None:
        return hooked
    override = module.__dict__.get("forward")
    if override is not None and not mg._is_offload_object(override):
        return f"forward already wrapped by {type(override).__name__}"
    return None


def per_call_scaled(module: Any) -> Optional[str]:
    """A layer whose activation global scale comes from each call's amax: a doubled batch shares one scale across both
    branches, so it is not the same computation. ComfyUI's dynamic NVFP4 linear and torchao's NVFP4 tensor do this;
    Studio's own NVFP4 linear freezes its activation scale (``a_gsf``), and per-row scaling (Studio's fp8 / int8) is
    per token, so both are unaffected."""
    for name, sub in module.named_modules():
        cls = type(sub).__name__
        upper = cls.upper()
        if "NVFP4" in upper and "DYNAMIC" in upper:
            return f"{name}: {cls}"
        if "NVFP4" in upper and "a_gsf" not in sub._buffers and "input_scale" not in sub._buffers:
            return f"{name}: {cls} (no frozen activation scale)"
        for pname, p in sub.named_parameters(recurse = False):
            if type(p).__name__ == "NVFP4Tensor":
                return f"{name}.{pname}: NVFP4Tensor"
    return None


def _speed_off_requested(speed_mode: Optional[str]) -> bool:
    """The request asked for ``speed_mode="off"`` (unset is not a request): the bit-identical reference path."""
    return speed_mode is not None and str(speed_mode).strip().lower() == "off"


def cfg_batch_decline(
    pipe: Any,
    *,
    offload_policy: str,
    speed_mode: Optional[str] = None,
) -> Optional[str]:
    """Why both guidance branches cannot share one forward on this load, else None. ``speed_mode`` is the request's own
    value: an explicit ``off`` keeps one branch per call (a doubled batch can pick other tiles); ``force`` overrides."""
    mode = cfg_batch_mode()
    if mode == "off":
        return f"{CFG_BATCH_ENV}=0"
    if mode != "force" and _speed_off_requested(speed_mode):
        return f"speed_mode=off keeps one branch per call ({CFG_BATCH_ENV}=force batches anyway)"
    torch = _torch()
    denoiser = getattr(pipe, "transformer", None)
    if not isinstance(denoiser, torch.nn.Module):
        return "no DiT denoiser"
    if getattr(pipe, "transformer_2", None) is not None:
        return "two-expert denoiser"
    if getattr(denoiser, "_compiled_call_impl", None) is not None:
        return "denoiser is compiled whole"
    compiled = compiled_denoiser(denoiser)
    if compiled is not None and not _offloaded(denoiser, offload_policy):
        return f"{compiled} and resident"
    if getattr(pipe, "_unsloth_cuda_graphs", None):
        return "the denoiser replays CUDA graphs"
    foreign = _foreign_layer(denoiser)
    if foreign is not None:
        return foreign
    scaled = per_call_scaled(denoiser)
    if scaled is not None and mode != "force":
        return f"per-call activation scaling ({scaled})"
    if mode == "force":
        return None
    if type(denoiser).__name__ not in STATELESS_CLASSES or not type(denoiser).__module__.startswith(
        "diffusers."
    ):
        return f"{type(denoiser).__name__} is not on the stateless-forward list ({CFG_BATCH_ENV}=force opts it in)"
    if offload_policy == "model":
        # diffusers rebuilds model-offload hooks between renders, evicting the router: later renders would differ.
        return "model offload re-hooks the denoiser every render"
    return None


def cfg_pair_decline(
    pipe: Any,
    *,
    offload_policy: Optional[str] = None,
    graphs: bool = True,
) -> Optional[str]:
    """Why guidance branches cannot pair per streamed block, else None. ``graphs=False`` ignores armed CUDA graphs."""
    from .diffusion_cfg_pair import pair_decline

    torch = _torch()
    denoiser = getattr(pipe, "transformer", None)
    if not isinstance(denoiser, torch.nn.Module):
        return "no DiT denoiser"
    if getattr(pipe, "transformer_2", None) is not None:
        return "two-expert denoiser"
    if getattr(denoiser, "_compiled_call_impl", None) is not None:
        return "denoiser is compiled whole"
    if not _streamed(denoiser) or (
        offload_policy is not None and not _offloaded(denoiser, offload_policy)
    ):
        return "the denoiser stays on the card"
    if graphs and getattr(pipe, "_unsloth_cuda_graphs", None):
        return "the denoiser replays CUDA graphs"
    foreign = _foreign_layer(denoiser)
    if foreign is not None:
        return foreign
    return pair_decline(denoiser, denoiser.forward)


def _usable_free_mib(device: Any) -> Optional[int]:
    """Driver free plus reusable cache (graph pools excluded) on ``device``, less the VRAM Budget reserve."""
    torch = _torch()
    try:
        from .diffusion_memory import _user_budget_reserve_mib, cuda_reusable_cache_bytes

        free, total = torch.cuda.mem_get_info(device)
        idle = cuda_reusable_cache_bytes(device)
        return (int(free + idle) >> 20) - _user_budget_reserve_mib(int(total) >> 20)
    except Exception:  # noqa: BLE001 - unreadable: the out-of-memory backstop decides
        return None


VERDICTS_ENV = "UNSLOTH_DIFFUSION_CFG_BATCH_VERDICTS"
VERDICTS_FILE_ENV = "UNSLOTH_DIFFUSION_CFG_BATCH_VERDICTS_FILE"
_VERDICTS_FILE = "diffusion_cfg_batch_verdicts.json"
_VERDICTS_MAX = 512
# v1 keyed a bf16 load and its torchao INT8 / FP8 load alike
_VERDICTS_VERSION = 2
_BATCH = "batch"
_PLAIN = "plain"


def _verdicts_path() -> Optional[Any]:
    """The timing verdict store (next to the learned peaks), None when turned off or nowhere safe to write."""
    if (os.environ.get(VERDICTS_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    from pathlib import Path

    explicit = (os.environ.get(VERDICTS_FILE_ENV) or "").strip()
    if explicit:
        return Path(explicit)
    from . import diffusion_measured_peaks as peaks

    base = peaks.store_path()
    return None if base is None else base.parent / _VERDICTS_FILE


def _read_verdicts(path: Any) -> dict:
    import json

    from . import diffusion_measured_peaks as peaks

    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - absent or corrupt: an empty store
        return {}
    if (
        not isinstance(data, dict)
        or data.get("version") != _VERDICTS_VERSION
        or data.get("runtime") != peaks.runtime_stamp()
    ):
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _stored(entries: dict, key: str) -> Optional[dict]:
    entry = entries.get(key)
    if isinstance(entry, dict) and entry.get("verdict") in (_BATCH, _PLAIN):
        return entry
    return None


def stored_verdict(key: str) -> Optional[dict]:
    path = _verdicts_path()
    if path is None:
        return None
    try:
        return _stored(_read_verdicts(path), key)
    except Exception:  # noqa: BLE001
        return None


def _settle_verdict(key: str, verdict: dict) -> dict:
    """First writer wins: the stored verdict for ``key`` if any process measured one, else ``verdict`` (stored).
    A store that cannot be read or written leaves this process with its own measurement."""
    path = _verdicts_path()
    if path is None:
        return verdict
    from . import diffusion_measured_peaks as peaks

    try:
        with peaks._lock, peaks._file_lock(path):
            entries = _read_verdicts(path)
            stored = _stored(entries, key)
            if stored is not None:
                return stored
            entries[key] = verdict
            for stale in list(entries)[: max(0, len(entries) - _VERDICTS_MAX)]:
                entries.pop(stale, None)
            import json
            import tempfile

            path.parent.mkdir(parents = True, exist_ok = True)
            fd, tmp = tempfile.mkstemp(prefix = ".cfg_batch.", dir = str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding = "utf-8") as handle:
                    json.dump(
                        {
                            "version": _VERDICTS_VERSION,
                            "runtime": peaks.runtime_stamp(),
                            "entries": entries,
                        },
                        handle,
                        indent = 1,
                    )
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
    return verdict


def _card_name(latent: Any) -> Optional[str]:
    """The CUDA card a verdict was timed on (None off CUDA: nothing timed, nothing stored)."""
    if getattr(latent.device, "type", None) != "cuda":
        return None
    from .diffusion_measured_peaks import device_name

    return device_name(latent.device)


def _weight_kind(p: Any) -> str:
    """torchao ``quantize_`` keeps ``nn.Linear`` and a bf16 ``.dtype``: only the tensor class tells the loads apart."""
    data = getattr(p, "data", p)
    kind = type(data) if type(p).__name__ == "Parameter" else type(p)
    names = list(getattr(kind, "tensor_attribute_names", None) or ()) + list(
        getattr(kind, "optional_tensor_attribute_names", None) or ()
    )
    attrs = [f"{n}={getattr(data, n, None)!r}" for n in names if n != "block_size"]
    return f"{kind.__module__}.{kind.__qualname__}" + (f"({','.join(attrs)})" if attrs else "")


def _load_identity(
    denoiser: Any,
    offload_policy: str,
    speed_mode: Optional[str],
    transformer_quant: Optional[str] = None,
) -> Optional[str]:
    """What a doubled call's timing depends on in the load. Hashed: it is a store key, never read back."""
    import hashlib
    import json

    try:
        config = getattr(denoiser, "config", None)
        layers = sorted({f"{type(m).__module__}.{type(m).__name__}" for m in denoiser.modules()})
        params = list(denoiser.parameters())
        dtypes = sorted({str(p.dtype) for p in params})
        weights = sorted({_weight_kind(p) for p in params})
        blob = json.dumps(
            [
                f"{type(denoiser).__module__}.{type(denoiser).__name__}",
                dict(config) if config is not None else None,
                layers,
                dtypes,
                weights,
                None if transformer_quant is None else str(transformer_quant),
                offload_policy,
                speed_mode,
            ],
            sort_keys = True,
            default = str,
        )
    except Exception:  # noqa: BLE001 - unidentifiable: the verdict stays per load
        return None
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class _Shape:
    """Per request shape: warm-up, timing, verdict and the run of misses."""

    def __init__(self, key: Any = None) -> None:
        self.key = key
        # the timing verdict (_BATCH / _PLAIN), latched for the load and persisted: memory pressure never re-times it
        self.verdict: Optional[str] = None
        self.doubled_ran = False
        self.steps = 0
        self.plain_pair_s: Optional[float] = None
        self.batched_s: list[float] = []
        self.activation_mib: Optional[int] = None
        self.off: Optional[str] = None
        self.off_kind: Optional[str] = None
        self.misses = 0
        self.judged = False
        self.compiled_at_trial = False
        self.trial_render = 0
        self.render_steps = 0  # steps of this shape in the current render


class CfgBatchRouter:
    """Denoiser ``forward`` running both guidance branches of a step through one forward."""

    def __init__(
        self,
        module: Any,
        *,
        reason: str,
        logger: Any = None,
    ) -> None:
        self.module_ref = weakref.ref(module)
        # Under Dynamo tracing: pass straight through (no graph breaks, no doubled-shape recompile).
        self._is_compiling = getattr(getattr(_torch(), "compiler", None), "is_compiling", None) or (
            lambda: False
        )
        self.orig_forward = module.forward
        self.had_override = "forward" in module.__dict__
        self.prev_override = module.__dict__.get("forward")
        self.reason = reason
        self.logger = logger
        self.disabled: Optional[str] = None
        self.shapes: dict = {}
        self.stats = {
            "batched_steps": 0,
            "paired_steps": 0,
            "hits": 0,
            "misses": 0,
            "plain_calls": 0,
        }
        self.paired: Any = None
        self._logged_paired = 0
        self._reset_step()
        self.template: Optional[tuple] = None
        self.prev_latent: Any = None
        self.prev_calls = 0
        self.last_shape: Optional[_Shape] = None
        self.render_off: Optional[str] = None
        self.render_checked = False
        self.offloaded = False
        self.pending_disable: Optional[str] = None
        self.pair_t0: Optional[float] = None
        self.renders = 0  # completed renders: a doubled-batch compile is warm-up only in a shape's trial render
        self.render_compiled: Optional[str] = None
        self.identity: Optional[str] = (
            None  # the load's verdict-store identity (None: verdicts stay per load)
        )

    def _reset_step(self) -> None:
        self.branch = 0
        self.b0: Optional[tuple] = None
        self.kept: Optional[dict] = None
        self.shape: Optional[_Shape] = None

    def describe(self) -> dict[str, Any]:
        shape = self.shape or self.last_shape
        return {
            "armed_for": self.reason,
            "mode": "pair" if self.paired is not None else "batch",
            "disabled_reason": self.disabled,
            "shape_off": None if shape is None else shape.off,
            "verdict": None if shape is None else shape.verdict,
            "plain_pair_s": None
            if shape is None or shape.plain_pair_s is None
            else round(shape.plain_pair_s, 4),
            "batched_pair_s": None
            if shape is None or not shape.batched_s
            else round(min(shape.batched_s), 4),
            "displaced": self._displaced(),
            "render_off": self.render_off,
            **self.stats,
        }

    def end_render(self) -> None:
        """Drop every tensor held between calls (the kept half, the template's constants, the latent)."""
        paired = self.stats["paired_steps"] - self._logged_paired
        if paired and self.logger is not None:
            self._logged_paired = self.stats["paired_steps"]
            self.logger.info(
                "diffusion.cfg_pair: %d guided step(s) ran both branches through each streamed block "
                "(one upload per step); %d paired so far",
                paired,
                self._logged_paired,
            )
        if self.render_checked:
            self.renders += 1
        if self.pending_disable is not None:
            self.disable(self.pending_disable)
        self.pair_t0 = None
        self.kept = None
        self.template = None
        self.prev_latent = None
        self.prev_calls = 0
        self.render_off = None
        self.render_checked = False
        self._reset_step()
        for shape in self.shapes.values():
            shape.render_steps = 0
            if shape.off_kind == _OFF_TRANSIENT:
                # memory pressure is per render: retry on the next one, keeping a timing verdict already reached
                shape.off = shape.off_kind = None
                shape.judged = shape.verdict == _BATCH

    def disable(self, reason: str) -> None:
        if self.disabled is None:
            self.disabled = reason
            if self.logger is not None:
                self.logger.warning("diffusion.cfg_batch: off for this load (%s)", reason)
        self.kept = None
        self.template = None

    def _shape_off(
        self,
        shape: _Shape,
        reason: str,
        kind: Optional[str] = None,
    ) -> None:
        if shape.off is None:
            shape.off, shape.off_kind = reason, kind
            if self.logger is not None:
                self.logger.info("diffusion.cfg_batch: off for this request shape (%s)", reason)
        self.kept = None

    def _displaced(self) -> bool:
        module = self.module_ref()
        return module is None or module.__dict__.get("forward") is not self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_compiling():
            return self.orig_forward(*args, **kwargs)
        mg = _mg()
        latent = mg._latent_arg(args, kwargs)
        if self.disabled is not None or latent is None or self._displaced():
            return self._plain(args, kwargs)
        if not self.render_checked:
            # first call of a render: a request-time step cache may have hooked the blocks since install
            module = self.module_ref()
            self.render_off = _foreign_hook(module) if module is not None else None
            self.render_compiled = compiled_denoiser(module) if module is not None else None
            if self.render_compiled is not None:
                if not self.offloaded:
                    self.render_off = f"{self.render_compiled} and resident"
                for shape in self.shapes.values():
                    if shape.judged and not shape.compiled_at_trial and shape.off is None:
                        self._shape_off(shape, "the denoiser was compiled after this shape's trial")
            self.render_checked = True
        if self.render_off is not None:
            return self._plain(args, kwargs)
        if self.prev_latent is not None and latent is self.prev_latent:
            self.branch += 1
        else:
            self._new_step(args, kwargs, latent)
        shape = self.shape
        if self.branch == 0 and self.paired is not None:
            # the first plain pair records the second call's template
            if self.template is not None and shape.off is None and shape.render_steps > 1:
                out = self._pair(args, kwargs, latent, shape)
                if out is not None:
                    return out
            return self._plain(args, kwargs)
        if self.branch == 0:
            if (
                self.template is not None
                and shape.off is None
                and shape.judged
                and shape.render_steps > _PLAIN_STEPS
            ):
                out = self._batched(args, kwargs, latent, shape)
                if out is not None:
                    return out
            if shape.off is None and not shape.judged and shape.render_steps == _PLAIN_STEPS:
                self.pair_t0 = time.perf_counter() if self._sync(latent) else None
            if shape.off is None and not shape.judged and shape.activation_mib is None:
                return self._measured_plain(args, kwargs, latent, shape)
            return self._plain(args, kwargs)
        if self.branch == 1:
            if self.kept is not None:
                out = self._take(args, kwargs, shape)
                if out is not None:
                    return out
            if shape.off is not None:
                # skip recording on a plain shape: it is a host sync per step (2.5-3.5% on B200)
                return self._plain(args, kwargs)
            self._record_template(args, kwargs, latent)
            out = self._plain(args, kwargs)
            if self.paired is not None:
                return out
            if not shape.judged and self.disabled is None and shape.render_steps == _PLAIN_STEPS:
                t0, self.pair_t0 = self.pair_t0, None
                if shape.verdict is None:
                    stored = self._stored_verdict(shape, latent)
                    if stored is not None:
                        self._apply_verdict(shape, stored, stored = True)
                if shape.verdict == _BATCH:
                    # already priced (this load, or any process on this model, shape and card): never re-timed, so
                    # a borderline shape cannot flip; only the memory gate below can keep this render plain
                    if self._memory_ok(latent, shape):
                        self._judge(shape)
                    else:
                        self._shape_off(shape, "no room for a doubled call", _OFF_TRANSIENT)
                    return out
                if shape.off is not None:
                    return out
                if t0 is not None and not self.offloaded and self._sync(latent):
                    live = time.perf_counter() - t0
                    if live >= _RESIDENT_MAX_PAIR_S:
                        shape.plain_pair_s = live
                        self._decide(
                            shape,
                            latent,
                            _PLAIN,
                            f"compute-bound: a plain pair takes {live:.3f}s, where one doubled call cannot win "
                            f"{round((1 - _MEASURED_FRACTION) * 100)}%",
                        )
                        return out
                self._trial((args, kwargs), latent, shape)
            return out
        self.disable("more than two denoiser calls per step")
        return self._plain(args, kwargs)

    def _new_step(self, args: tuple, kwargs: dict, latent: Any) -> None:
        if self.kept is not None:
            self._miss(self.shape, "the second branch never came")
        if self.prev_latent is not None and self.branch == 0:
            self.template = (
                None  # the last step made one call (guidance ended or off): nothing to predict from
            )
        self._reset_step()
        self.prev_latent = latent
        self.b0 = (args, kwargs)
        mg = _mg()
        key = tuple(tuple(t.shape) for t in mg._iter_tensors((args, kwargs)))
        shape = self.shapes.get(key)
        if shape is None:
            shape = self.shapes[key] = _Shape(key)
        if shape is not self.last_shape:
            self.template = (
                None  # a template predicts its own shape's second call: one plain pair first
            )
        shape.steps += 1
        shape.render_steps += 1
        self.shape = self.last_shape = shape

    def _plain(self, args: tuple, kwargs: dict) -> Any:
        self.stats["plain_calls"] += 1
        return self.orig_forward(*args, **kwargs)

    def _sync(self, latent: Any) -> bool:
        if getattr(latent.device, "type", None) != "cuda":
            return False
        from .diffusion_cuda_graph import synchronize_outside_capture

        synchronize_outside_capture(latent.device)
        return True

    def _measured_plain(self, args: tuple, kwargs: dict, latent: Any, shape: _Shape) -> Any:
        """A plain first-branch call that records the activations it allocated (the doubled call's memory gate)."""
        torch = _torch()
        if getattr(latent.device, "type", None) != "cuda":
            return self._plain(args, kwargs)
        # never reset the peak counters (other measurements own them); a call that stayed under an
        # earlier peak leaves its activations unknown
        base = torch.cuda.memory_allocated(latent.device)
        before = torch.cuda.max_memory_allocated(latent.device)
        out = self._plain(args, kwargs)
        peak = torch.cuda.max_memory_allocated(latent.device)
        if peak > before:
            shape.activation_mib = int(peak - base) >> 20
        return out

    def _trial(self, second: tuple, latent: Any, shape: _Shape) -> None:
        """Price this shape once on the step's warm arguments; outputs are discarded so renders keep one path."""
        torch = _torch()
        first = self.b0
        if first is None:
            return
        if not self._sync(latent):
            if not self.offloaded and cfg_batch_mode() != "force":
                # A resident denoiser off CUDA (CPU, MPS): nothing can time the doubled call, and the gate only turns
                # batching on where it was measured to win, so the shape stays plain unless forced.
                self._shape_off(
                    shape,
                    f"no device timer on {getattr(latent.device, 'type', 'this device')} to price the doubled call "
                    f"({CFG_BATCH_ENV}=force batches anyway)",
                )
                return
            shape.verdict = _BATCH  # untimed (offloaded off CUDA): the same verdict every time, nothing to store
            self._judge(shape)
            return
        if not self._memory_ok(latent, shape):
            self._shape_off(shape, "no room for a doubled call", _OFF_TRANSIENT)
            return
        batch = int(latent.shape[0])
        plain_s: list[float] = []
        try:
            graphs = _graph_count()
            for _ in range(_BATCHED_SAMPLES):
                t0 = time.perf_counter()
                self.orig_forward(*first[0], **first[1])
                self.orig_forward(*second[0], **second[1])
                self._sync(latent)
                plain_s.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                predicted = self._predict(latent) if self.template is not None else second
                b_args, b_kwargs = merge_branches(first, predicted, batch)
                snapshots = _snapshot(predicted)
                split_output(self.orig_forward(*b_args, **b_kwargs), batch)
                _matches(snapshots, second)
                self._sync(latent)
                shape.batched_s.append(time.perf_counter() - t0)
                shape.doubled_ran = True
                del b_args, b_kwargs, snapshots, predicted
            shape.plain_pair_s = min(plain_s)
            if _graph_count() != graphs and self.render_compiled is None and not self.offloaded:
                self.disable(
                    "the doubled batch compiled a new graph (the denoiser runs compiled code)"
                )
                return
        except _NotBatchable as exc:
            self._shape_off(shape, f"not batchable: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - the shape stays plain; the render's own calls already ran
            if _is_oom(exc):
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                self._shape_off(shape, "out of memory on the timed doubled call", _OFF_TRANSIENT)
                return
            self.disable(f"the doubled call failed ({type(exc).__name__}: {exc})")
            return
        batched = min(shape.batched_s)
        if shape.plain_pair_s and batched > _MEASURED_FRACTION * shape.plain_pair_s:
            self._decide(
                shape,
                latent,
                _PLAIN,
                f"measured no gain: batched pair {batched:.4f}s vs plain pair {shape.plain_pair_s:.4f}s",
            )
            return
        self._decide(
            shape,
            latent,
            _BATCH,
            f"measured gain: batched pair {batched:.4f}s vs plain pair {shape.plain_pair_s or 0:.4f}s",
        )

    def _verdict_key(self, shape: _Shape, latent: Any) -> Optional[str]:
        """Store key of a timing verdict: the load's identity, the card, compiled or eager, the request shape.
        None off CUDA or for an unidentified load (the verdict then holds for this load only)."""
        card = None if self.identity is None else _card_name(latent)
        if not card:
            return None
        compiled = "compiled" if self.render_compiled is not None else "eager"
        module = self.module_ref()
        # Read live: set_attention_backend() changes neither the module classes nor the config.
        attention = (
            getattr(module, "_unsloth_attention_backend", None) if module is not None else None
        )
        return f"{self.identity}|{card}|{compiled}|{attention}|{shape.key}"

    def _stored_verdict(self, shape: _Shape, latent: Any) -> Optional[dict]:
        key = self._verdict_key(shape, latent)
        return None if key is None else stored_verdict(key)

    def _apply_verdict(self, shape: _Shape, entry: dict, *, stored: bool) -> None:
        shape.verdict = entry["verdict"]
        if shape.verdict == _PLAIN:
            reason = str(entry.get("reason") or "measured no gain")
            self._shape_off(shape, f"stored verdict: {reason}" if stored else reason)

    def _decide(self, shape: _Shape, latent: Any, verdict: str, reason: str) -> None:
        """Latch a measured verdict; the first one stored for this model, shape and card wins across processes."""
        entry = {"verdict": verdict, "reason": reason}
        key = self._verdict_key(shape, latent)
        settled = entry if key is None else _settle_verdict(key, entry)
        self._apply_verdict(shape, settled, stored = settled is not entry)
        if shape.verdict == _BATCH:
            if self._memory_ok(latent, shape):
                self._judge(shape)
            else:
                self._shape_off(shape, "no room for a doubled call", _OFF_TRANSIENT)

    def _judge(self, shape: _Shape) -> None:
        shape.judged = True
        shape.compiled_at_trial = self.render_compiled is not None
        shape.trial_render = self.renders

    def _record_template(self, args: tuple, kwargs: dict, latent: Any) -> None:
        """How the second call's arguments relate to the first's: the latent, a first-branch tensor (same object, or a
        small per-call scalar recomputed with equal bytes), or a constant of the second branch."""
        mg = _mg()
        torch = _torch()
        b0_leaves = dict(mg._leaves(self.b0)) if self.b0 is not None else {}
        b0_by_id = {id(v): p for p, v in b0_leaves.items() if isinstance(v, torch.Tensor)}
        kinds: dict = {}
        for path, leaf in mg._leaves((args, kwargs)):
            if isinstance(leaf, torch.Tensor):
                if leaf is latent:
                    kinds[path] = ("latent",)
                elif id(leaf) in b0_by_id:
                    kinds[path] = ("b0", b0_by_id[id(leaf)])
                elif (
                    isinstance(b0_leaves.get(path), torch.Tensor)
                    and leaf.numel() <= mg._SMALL_TENSOR_NUMEL
                    and mg._bitwise_equal(leaf, b0_leaves[path])
                ):
                    kinds[path] = ("b0", path)
                else:
                    kinds[path] = ("const", leaf)
            else:
                kinds[path] = ("value", leaf)
        self.template = ((args, kwargs), kinds)

    def _predict(self, latent: Any) -> tuple:
        mg = _mg()
        structure, kinds = self.template
        b0_leaves = dict(mg._leaves(self.b0))
        values: dict = {}
        for path, kind in kinds.items():
            if kind[0] == "latent":
                values[path] = latent
            elif kind[0] == "b0":
                if kind[1] not in b0_leaves:
                    raise _NotBatchable("the first branch changed structure")
                values[path] = b0_leaves[kind[1]]
            else:
                values[path] = kind[1]
        try:
            return mg._rebuild(structure, values)
        except KeyError:
            raise _NotBatchable("the first branch changed structure") from None

    def _memory_ok(self, latent: Any, shape: _Shape) -> bool:
        if shape.activation_mib is None or getattr(latent.device, "type", None) != "cuda":
            return True
        free = _usable_free_mib(latent.device)
        return free is None or free >= 2 * shape.activation_mib

    def _batched(self, args: tuple, kwargs: dict, latent: Any, shape: _Shape) -> Any:
        torch = _torch()
        if not self._memory_ok(latent, shape):
            # plain for the rest of this render (not one step here and there), batched again on the next one
            self._shape_off(shape, "no room for a doubled call", _OFF_TRANSIENT)
            return None
        try:
            second = self._predict(latent)
            b_args, b_kwargs = merge_branches((args, kwargs), second, int(latent.shape[0]))
            snapshots = _snapshot(second)
        except _NotBatchable as exc:
            self._shape_off(shape, f"not batchable: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 - the merge and snapshots allocate too
            if not _is_oom(exc):
                raise
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            self._shape_off(shape, "out of memory preparing the doubled call", _OFF_TRANSIENT)
            return None
        graphs = _graph_count()
        try:
            out = self.orig_forward(*b_args, **b_kwargs)
            mine, other = split_output(out, int(latent.shape[0]))
        except Exception as exc:  # noqa: BLE001 - the branch reruns plain
            del b_args, b_kwargs
            if _is_oom(exc):
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                self._shape_off(shape, "out of memory on the doubled call", _OFF_TRANSIENT)
                return None
            if isinstance(exc, _NotBatchable):
                self._shape_off(shape, f"output cannot be split per branch ({exc})")
                return None
            self.disable(f"the doubled call failed ({type(exc).__name__}: {exc})")
            return None
        if not shape.doubled_ran:
            # a stored verdict skipped the trial: this render's first doubled call is the shape's warm-up
            shape.doubled_ran = True
            shape.trial_render = self.renders
            if _graph_count() != graphs and self.render_compiled is None and not self.offloaded:
                self.pending_disable = (
                    "the doubled batch compiled a new graph (the denoiser runs compiled code)"
                )
        elif _graph_count() != graphs and self.renders > shape.trial_render:
            self.pending_disable = "the doubled batch compiled a graph after its warm-up render"
        self.kept = {
            "out": other,
            "snapshots": snapshots,
            "latent": latent,
        }
        self.stats["batched_steps"] += 1
        return mine

    def _pair(self, args: tuple, kwargs: dict, latent: Any, shape: _Shape) -> Any:
        from .diffusion_cfg_pair import NotPairable, root_forward

        torch = _torch()
        module = self.module_ref()
        root_fn = root_forward(module, self.orig_forward) if module is not None else None
        if root_fn is None:
            self.disable("the denoiser's own forward is no longer reachable below its hooks")
            return None
        try:
            second = self._predict(latent)
            snapshots = _snapshot(second)
        except _NotBatchable as exc:
            self._shape_off(shape, f"not pairable: {exc}")
            return None
        graphs = _graph_count()
        try:
            mine, other = self.paired.run(
                module, self.orig_forward, root_fn, (args, kwargs), second
            )
        except NotPairable as exc:
            self._shape_off(shape, f"not pairable: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 - the branch reruns plain
            if _is_oom(exc):
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                self._shape_off(shape, "out of memory on the paired call", _OFF_TRANSIENT)
                return None
            self.disable(f"the paired call failed ({type(exc).__name__}: {exc})")
            return None
        if _graph_count() != graphs:
            self.pending_disable = "the paired call compiled a new graph"
        self.kept = {"out": other, "snapshots": snapshots, "latent": latent}
        self.stats["paired_steps"] += 1
        return mine

    def _take(self, args: tuple, kwargs: dict, shape: _Shape) -> Any:
        kept, self.kept = self.kept, None
        if not _matches(kept["snapshots"], (args, kwargs)):
            self._miss(shape, "the second branch differed from what was batched")
            return None
        self.stats["hits"] += 1
        shape.misses = 0
        return kept["out"]

    def _miss(self, shape: Optional[_Shape], why: str) -> None:
        self.kept = None
        self.stats["misses"] += 1
        if shape is None:
            return
        shape.misses += 1
        if shape.misses >= _MISS_LIMIT:
            self._shape_off(shape, f"{shape.misses} consecutive misses ({why})")


def _is_oom(exc: BaseException) -> bool:
    try:
        from .diffusion_batched import is_oom_error
        return bool(is_oom_error(exc))
    except Exception:  # noqa: BLE001
        return "out of memory" in str(exc).lower()


def _snapshot(call: tuple) -> list:
    """The second branch's predicted arguments, tensors cloned: an in-place write between the two calls is caught."""
    mg = _mg()
    torch = _torch()
    return [
        (p, v.detach().clone() if isinstance(v, torch.Tensor) else v) for p, v in mg._leaves(call)
    ]


def _matches(snapshots: list, call: tuple) -> bool:
    mg = _mg()
    torch = _torch()
    actual = mg._leaves(call)
    if len(actual) != len(snapshots):
        return False
    tensors = []
    for (pa, va), (ps, vs) in zip(actual, snapshots):
        if pa != ps:
            return False
        if isinstance(vs, torch.Tensor):
            tensors.append((va, vs))
        elif not mg._same_value(va, vs):
            return False
    # Every tensor argument under one host read per device (once per guided step), not one torch.equal each.
    return mg._bitwise_equal_all(tensors)


def _pad_text(first: dict, second: dict) -> None:
    """Pad the shorter prompt's embeddings (and its mask) with masked zeros so both branches stack (Qwen-Image pads
    each prompt to its own length; the masked keys leave every image token's attention unchanged)."""
    torch = _torch()
    for emb_key, mask_key in _PADDED_TEXT:
        a, b = first.get(emb_key), second.get(emb_key)
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.dim() < 3:
            continue
        if a.shape[1] == b.shape[1]:
            continue
        if mask_key not in first and mask_key not in second:
            raise _NotBatchable(f"{emb_key}: prompt lengths differ and the model takes no mask")
        length = max(a.shape[1], b.shape[1])
        given = [
            m for m in (first.get(mask_key), second.get(mask_key)) if isinstance(m, torch.Tensor)
        ]
        dtype = given[0].dtype if given else torch.long
        for kw, emb in ((first, a), (second, b)):
            mask = kw.get(mask_key)
            if mask is None:
                mask = torch.ones(emb.shape[:2], dtype = dtype, device = emb.device)
            mask = mask.to(dtype)
            pad = length - emb.shape[1]
            if pad:
                emb = torch.cat((emb, emb.new_zeros((emb.shape[0], pad, *emb.shape[2:]))), dim = 1)
                mask = torch.cat((mask, mask.new_zeros((mask.shape[0], pad))), dim = 1)
            kw[emb_key] = emb
            kw[mask_key] = mask


def merge_branches(first_call: tuple, second_call: tuple, batch: int) -> tuple[tuple, dict]:
    """One call carrying both branches: per-sample tensors concatenated on dim 0, equal shared values passed once,
    per-sample lists joined, text padded to a common length when the model takes a mask."""
    mg = _mg()
    torch = _torch()
    args_a, kwargs_a = first_call
    args_b, kwargs_b = second_call
    kwargs_a, kwargs_b = dict(kwargs_a), dict(kwargs_b)
    lists = {}
    for key in _PER_SAMPLE_LISTS:
        if key in kwargs_a or key in kwargs_b:
            la, lb = kwargs_a.pop(key, None), kwargs_b.pop(key, None)
            if la is None and lb is None:
                lists[key] = None
                continue
            if not (isinstance(la, list) and isinstance(lb, list)) or la != lb:
                raise _NotBatchable(f"{key} differs per branch")
            lists[key] = list(la) + list(lb) if len(la) == batch else la
    _pad_text(kwargs_a, kwargs_b)
    first, second = mg._leaves((args_a, kwargs_a)), mg._leaves((args_b, kwargs_b))
    if len(first) != len(second):
        raise _NotBatchable("the branches pass different arguments")
    values: dict = {}
    for (path_a, a), (path_b, b) in zip(first, second):
        if path_a != path_b:
            raise _NotBatchable("the branches pass different arguments")
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                raise _NotBatchable(f"{path_a}: tensor in one branch only")
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise _NotBatchable(f"{path_a}: {tuple(a.shape)} vs {tuple(b.shape)}")
            if a.dim() >= 1 and a.shape[0] == batch:
                values[path_a] = torch.cat((a, b), dim = 0)
            elif a is b or mg._bitwise_equal(a, b):
                values[path_a] = a
            else:
                raise _NotBatchable(f"{path_a}: differs per branch without a batch dim")
        elif mg._same_value(a, b):
            values[path_a] = a
        else:
            raise _NotBatchable(f"{path_a}: differs per branch")
    b_args, b_kwargs = mg._rebuild((args_a, kwargs_a), values)
    b_kwargs = dict(b_kwargs)
    b_kwargs.update(lists)
    return b_args, b_kwargs


def split_output(
    out: Any,
    batch: int,
    depth: int = 0,
) -> tuple[Any, Any]:
    """(first, second) branch of a doubled output, split on a leading dim of ``2 * batch``."""
    torch = _torch()
    if isinstance(out, torch.Tensor):
        if out.dim() >= 1 and out.shape[0] == 2 * batch:
            return out[:batch], out[batch:]
        raise _NotBatchable(f"output of shape {tuple(out.shape)} has no doubled batch")
    if out is None or isinstance(out, (bool, int, float, str)):
        return out, out
    if depth > 4:
        raise _NotBatchable("output nests too deep")
    if isinstance(out, tuple) and not hasattr(out, "_fields"):
        pairs = [split_output(v, batch, depth + 1) for v in out]
        return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
    if isinstance(out, list):
        pairs = [split_output(v, batch, depth + 1) for v in out]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if isinstance(out, dict):
        pairs = {k: split_output(v, batch, depth + 1) for k, v in out.items()}
        return type(out)((k, v[0]) for k, v in pairs.items()), type(out)(
            (k, v[1]) for k, v in pairs.items()
        )
    fields = getattr(out, "__dataclass_fields__", None)
    if fields is not None:
        import dataclasses
        pairs = {name: split_output(getattr(out, name), batch, depth + 1) for name in fields}
        return (
            dataclasses.replace(out, **{k: v[0] for k, v in pairs.items()}),
            dataclasses.replace(out, **{k: v[1] for k, v in pairs.items()}),
        )
    raise _NotBatchable(f"output type {type(out).__name__}")


_ROUTERS: "weakref.WeakKeyDictionary[Any, CfgBatchRouter]" = weakref.WeakKeyDictionary()


def install_cfg_batch(
    pipe: Any,
    *,
    offload_policy: str,
    speed_mode: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Arm :class:`CfgBatchRouter` on ``pipe.transformer``; the decline reason, or None when armed. An optional lever:
    it never fails a load. ``speed_mode`` is the request's value (None = unset)."""
    try:
        return _install(
            pipe,
            offload_policy = offload_policy,
            speed_mode = speed_mode,
            transformer_quant = transformer_quant,
            logger = logger,
        )
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.info("diffusion.cfg_batch: not armed (%s: %s)", type(exc).__name__, exc)
        return f"{type(exc).__name__}: {exc}"


def _install(
    pipe: Any,
    *,
    offload_policy: str,
    speed_mode: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    if cfg_pair_decline(pipe, offload_policy = offload_policy) is None:
        from .diffusion_cfg_pair import PairedForward

        denoiser = pipe.transformer
        router = CfgBatchRouter(
            denoiser,
            reason = f"offloaded denoiser ({offload_policy}): both branches run each streamed block",
            logger = logger,
        )
        router.offloaded = True
        router.paired = PairedForward(denoiser)
        return _arm(pipe, denoiser, router, logger)
    reason = cfg_batch_decline(pipe, offload_policy = offload_policy, speed_mode = speed_mode)
    if reason is not None:
        if logger is not None:
            logger.info("diffusion.cfg_batch: not armed (%s)", reason)
        return reason
    denoiser = pipe.transformer
    router = CfgBatchRouter(
        denoiser,
        reason = "forced"
        if cfg_batch_mode() == "force"
        else (
            f"offloaded denoiser ({offload_policy})"
            if _offloaded(denoiser, offload_policy)
            else "resident denoiser"
        ),
        logger = logger,
    )
    router.offloaded = _offloaded(denoiser, offload_policy)
    router.identity = _load_identity(denoiser, offload_policy, speed_mode, transformer_quant)
    return _arm(pipe, denoiser, router, logger)


def _arm(pipe: Any, denoiser: Any, router: CfgBatchRouter, logger: Any) -> Optional[str]:
    try:
        _ROUTERS[pipe] = router
    except TypeError:
        return "pipeline cannot be tracked (not weak-referenceable)"
    denoiser.__dict__["forward"] = router
    if logger is not None:
        logger.info("diffusion.cfg_batch: armed (%s)", router.reason)
    return None


def router_for(pipe: Any) -> Optional[CfgBatchRouter]:
    if pipe is None:
        return None
    try:
        return _ROUTERS.get(pipe)
    except TypeError:
        return None


def end_cfg_batch_render(pipe: Any) -> None:
    """Per render: drop what the router holds between calls. Nothing to do when no router is armed."""
    router = router_for(pipe)
    if router is not None:
        try:
            router.end_render()
        except Exception:  # noqa: BLE001 - cleanup never masks the render's own outcome
            pass


def cfg_batch_status(pipe: Any) -> Optional[dict]:
    router = router_for(pipe)
    try:
        return None if router is None else router.describe()
    except Exception:  # noqa: BLE001
        return None


def uninstall_cfg_batch(pipe: Any) -> None:
    router = router_for(pipe)
    if router is None:
        return
    try:
        _ROUTERS.pop(pipe, None)
    except TypeError:
        pass
    router.end_render()
    module = router.module_ref()
    if module is not None and module.__dict__.get("forward") is router:
        if router.had_override:
            module.__dict__["forward"] = router.prev_override
        else:
            del module.__dict__["forward"]

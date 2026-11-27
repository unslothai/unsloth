# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit-scale fp8 GEMM: tensorwise _scaled_mm at scale 1, then the row/column scales in an fp32 epilogue.

Two uses. sm_89 precise accumulate, where it beats torchao row-wise (fast accumulate stays on torchao, no gain).
And any card whose torch build has no row-wise _scaled_mm (the Windows wheels: "Rowwise scaling is not currently
supported on your device"): a one-off per-card probe finds it, and every fp8 GEMM on that card then runs here, fast
accumulate included and at every row count. UNSLOTH_DIFFUSION_FP8_UNIT_SCALE=0 off, =fallback acts as if no card had
the row-wise kernel. Installed only where it can engage: installing tags the process-wide compile cache key.
On those fallback cards, sm_89 / sm_12x run the GEMM and epilogue as one fused Triton kernel instead
(diffusion_fp8_triton_gemm; UNSLOTH_DIFFUSION_FP8_TRITON_GEMM=0 off)."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

UNIT_SCALE_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE"
MIN_ROWS_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS"
# Unit-scale wins from M = 256 up (L4: row-wise 2.5-4x slower from 512); below, tens of microseconds either way.
_DEFAULT_MIN_ROWS = 256
# A card whose kernel self-check hit an OOM retries it from an eager call at most this often (seconds).
_TRITON_RETRY_S = 30.0
# Eager epilogue chunk size, avoids a full fp32 [M, N] buffer.
_EAGER_EPILOGUE_ELEMS = 1 << 24
_OFF_WORDS = ("0", "off", "false", "no", "none")
_ON_WORDS = ("1", "on", "true", "yes", "force")
FALLBACK_WORD = "fallback"
_ADA_CAPS = ((8, 9),)
# The tensorwise / row-wise fp8 GEMMs need sm_89+; older cards have no fp8 path to rescue.
_FP8_MIN_CAP = (8, 9)

_COUNTS = {"unit_scale": 0, "rowwise": 0, "triton_gemm": 0}
_STATE: dict = {
    "installed": False,
    "orig": None,
    "caps": {},
    "qualifying": frozenset(),
    "fallback": frozenset(),
    "fp32_acc": frozenset(),
    "acc_deferred": frozenset(),
    "forced_fallback": False,
    "triton_eager": frozenset(),
    "triton_compiled": frozenset(),
    # (card, output dtype, fast accumulate) the traced path sends to the kernel, frozen at install (the compile tag
    # names them).
    "triton_kinds": frozenset(),
    # Cards whose bf16 self-check / accuracy check hit an OOM: {"eager": set, "accuracy": set, "next": {card: t},
    # "resolved": bool}. Eager calls retry them; the next install rebuilds the dispatcher and compile tag.
    "triton_pending": None,
}
# device index -> True (row-wise kernel runs), False (only tensorwise runs), None (neither / unknown); per process.
_ROWWISE: dict = {}


def _mode() -> str:
    v = (os.environ.get(UNIT_SCALE_ENV) or "").strip().lower()
    if v in _OFF_WORDS:
        return "off"
    if v in _ON_WORDS:
        return "force"
    if v == FALLBACK_WORD:
        return FALLBACK_WORD
    return "auto"


def _min_rows() -> int:
    try:
        return max(1, int(os.environ.get(MIN_ROWS_ENV, "") or _DEFAULT_MIN_ROWS))
    except ValueError:
        return _DEFAULT_MIN_ROWS


def _compiling() -> bool:
    # Counters are eager-only: a module-global mutated inside a traced region makes Dynamo guard on it and recompile.
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - no torch.compiler: nothing is being traced
        return False


def _traced(t: Any) -> bool:
    """Dynamo, or the dispatch-level trace AOTAutograd runs through torchao's tensor subclass, where
    ``is_compiling()`` reads False: a fake (or functional-over-fake) operand. The eager chunk loop must never be traced,
    it specializes the graph on the row count."""
    if _compiling():
        return True
    try:
        from torch._subclasses.fake_tensor import is_fake
        return bool(is_fake(t))
    except Exception:  # noqa: BLE001 - no fake-tensor API: nothing traces at dispatch level
        return False


def _scale_epilogue(
    acc: Any, a_scale: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> Any:
    """``(acc * a_scale * b_scale + bias)`` in fp32, rounded once to ``output_dtype``. Eager runs it per row chunk
    in place (bitwise identical) to bound peak memory."""
    import torch

    if _traced(acc):
        out = acc * a_scale.float()
        out.mul_(b_scale.float())
        if bias is not None:
            out.add_(bias.float())
        return out.to(output_dtype)
    sa, sb = a_scale.float(), b_scale.float()
    bf = bias.float() if bias is not None else None
    rows = int(acc.shape[0])
    step = max(1, _EAGER_EPILOGUE_ELEMS // max(1, int(acc.shape[1])))
    out = acc if acc.dtype == output_dtype else torch.empty_like(acc, dtype = output_dtype)
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        tmp = acc[start:stop] * sa[start:stop]
        tmp.mul_(sb)
        if bf is not None:
            tmp.add_(bf)
        out[start:stop].copy_(tmp)
        # Free before the next chunk, else two are live.
        del tmp
    return out


def can_engage(fast_accum: Optional[bool]) -> bool:
    """The sm_89 speed path: False only for fast accumulate outside ``force``; None (unknown) installs. A card without
    the row-wise kernel engages whatever this says."""
    return _mode() in ("force", FALLBACK_WORD) or fast_accum is not True


def stats() -> dict:
    return {**_COUNTS, "installed": bool(_STATE["installed"])}


# What torch raises for a scaled_mm layout this build or card does not have ("Rowwise scaling is not currently
# supported on your device" on the Windows wheels). Anything else (OOM, a busy or lost device) is no verdict.
_UNSUPPORTED_WORDS = (
    "not currently supported",
    "not supported",
    "unsupported",
    "is only supported",
    # torch 2.10 sends sm_12x row-wise scales to cuBLASLt, which refuses them (Windows RTX 5090 wheel)
    "cublas_status_not_supported",
)


def _is_unsupported(exc: BaseException) -> bool:
    if _is_oom(exc):
        return False
    return isinstance(exc, (RuntimeError, NotImplementedError)) and any(
        w in str(exc).lower() for w in _UNSUPPORTED_WORDS
    )


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        named = tuple(
            t
            for t in (
                getattr(torch, "OutOfMemoryError", None),
                getattr(torch.cuda, "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )
        if named and isinstance(exc, named):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def _probe_rowwise(index: int) -> Optional[bool]:
    """True when the row-wise fp8 _scaled_mm runs on ``index``; False only when the tensorwise one runs and the
    row-wise one raises torch's "not supported"; None for anything else (no fp8 GEMM at all, an OOM, a transient
    error, a wrong answer), which is never cached or persisted. One tiny GEMM pair per card."""
    import torch

    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        return None

    def gemm(scale_a: Any, scale_b: Any) -> bool:
        out = torch._scaled_mm(a, a.t(), scale_a = scale_a, scale_b = scale_b, out_dtype = torch.bfloat16)
        out = out[0] if isinstance(out, tuple) else out  # torch < 2.5 returns (out, amax)
        return bool(torch.isfinite(out).all().item()) and float(out[0, 0]) == 32.0

    dev = f"cuda:{index}"
    try:
        a = torch.ones((16, 32), device = dev).to(torch.float8_e4m3fn)
        one = torch.ones((), device = dev, dtype = torch.float32)
        if not gemm(one, one):
            return None
    except Exception:  # noqa: BLE001 - no tensorwise fp8 GEMM, or no memory for it: nothing proven
        return None
    try:
        rows = torch.ones((16, 1), device = dev, dtype = torch.float32)
        cols = torch.ones((1, 16), device = dev, dtype = torch.float32)
        return True if gemm(rows, cols) else None
    except Exception as exc:  # noqa: BLE001
        return False if _is_unsupported(exc) else None


_PERSIST: list = []  # [Namespace or None], built once


def _namespace() -> Any:
    if not _PERSIST:
        try:
            from .diffusion_probe_cache import Namespace
            _PERSIST.append(
                Namespace(
                    "fp8_rowwise", ("diffusion_fp8_unit_scale.py", "diffusion_modelopt_fp8.py")
                )
            )
        except Exception:  # noqa: BLE001 - no cache: probe per process
            _PERSIST.append(None)
    return _PERSIST[0]


def _persisted_rowwise(index: int) -> Optional[bool]:
    ns = _namespace()
    try:
        found = ns.get(f"cuda:{index}", "rowwise") if ns is not None else None
    except Exception:  # noqa: BLE001 - a miss, never an error
        return None
    return found if isinstance(found, bool) else None


def _persist_rowwise(index: int, verdict: bool) -> None:
    ns = _namespace()
    try:
        if ns is not None:
            ns.put(f"cuda:{index}", "rowwise", bool(verdict))
    except Exception:  # noqa: BLE001 - a cache write never fails a load
        pass


def _has_context(index: int) -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch._C._cuda_hasPrimaryContext(index))
    except Exception:  # noqa: BLE001
        return False


def _probe_refused() -> bool:
    """ROCm (fnuz fp8, gfx versions in the CUDA capability API; Studio keeps it on GGUF) and a stubbed torchao are
    never probed: an fp8 GEMM there proves nothing and may not fail cleanly."""
    try:
        from core._torchao_stub import is_stubbed, torch_is_rocm
        return bool(torch_is_rocm() or is_stubbed("torchao"))
    except Exception:  # noqa: BLE001 - cannot tell: do not probe
        return True


def _current_device() -> Optional[int]:
    try:
        import torch
        return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None


def rowwise_verdicts(caps: dict) -> dict:
    """Row-wise verdicts for the fp8-capable cards in ``caps``: in-process, then the persisted probe table (keyed like
    the quant probe verdicts: torch, driver, card, env), then a probe. Only cards this process already uses (a CUDA
    context; the current device when none has one) are probed, so no context opens on a card it never touches, and
    each verdict belongs to the card that was probed. A card with no verdict keeps torchao's own path."""
    out: dict = {}
    if _probe_refused():
        return out
    unknown = []
    for index, cap in caps.items():
        if tuple(cap) < _FP8_MIN_CAP:
            continue
        if index in _ROWWISE:
            out[index] = _ROWWISE[index]
            continue
        persisted = _persisted_rowwise(index)
        if persisted is not None:
            out[index] = _ROWWISE[index] = persisted
            continue
        unknown.append(index)
    probe = [i for i in unknown if _has_context(i)]
    if not probe and not any(_has_context(i) for i in caps):
        current = _current_device()
        probe = [current] if current in unknown else []
    for index in probe:
        try:
            verdict = _probe_rowwise(index)
        except Exception:  # noqa: BLE001 - an unprobeable card keeps torchao's own path
            verdict = None
        out[index] = verdict
        if verdict is not None:
            # None (OOM, transient) is neither cached nor persisted: the next install probes again.
            _ROWWISE[index] = verdict
            _persist_rowwise(index, verdict)
    return out


_FP32_OUT: dict = {}  # device index -> whether the tensorwise fp8 GEMM writes fp32 there


def _fp32_out_runs(index: int) -> bool:
    if index not in _FP32_OUT:
        ok = False
        try:
            import torch

            dev = f"cuda:{index}"
            a = torch.ones((16, 32), device = dev).to(torch.float8_e4m3fn)
            one = torch.ones((), device = dev, dtype = torch.float32)
            out = torch._scaled_mm(a, a.t(), scale_a = one, scale_b = one, out_dtype = torch.float32)
            out = out[0] if isinstance(out, tuple) else out
            ok = out.dtype == torch.float32 and float(out[0, 0]) == 32.0
        except Exception:  # noqa: BLE001 - bf16 accumulator then
            ok = False
        _FP32_OUT[index] = ok
    return _FP32_OUT[index]


def fallback_cards() -> frozenset:
    """Cards whose fp8 GEMMs run on the unit-scale path because their row-wise kernel is missing (or forced)."""
    return frozenset(_STATE["fallback"]) if _STATE["installed"] else frozenset()


def gemm_path(device_index: Optional[int] = None) -> dict:
    """Which fp8 GEMM a quantized layer on ``device_index`` runs (status field). Never raises."""
    try:
        cards = fallback_cards()
        index = device_index if device_index is not None else (min(cards) if cards else None)
        rowwise = _ROWWISE.get(index) if index is not None else None
        if index is not None and index in cards:
            path = "unit_scale_fallback"
            reason = (
                f"{UNIT_SCALE_ENV}=fallback"
                if _STATE.get("forced_fallback")
                else "this torch build has no row-wise fp8 _scaled_mm on this card: tensorwise GEMM + row/column "
                "scale epilogue"
            )
        elif _STATE["installed"] and index is not None and index in _STATE["qualifying"]:
            path = "unit_scale_or_rowwise"
            reason = "precise accumulate runs the unit-scale GEMM from the row threshold; fast accumulate stays row-wise"
        else:
            path = "torchao_rowwise"
            reason = "torchao row-wise _scaled_mm"
        return {
            "path": path,
            "reason": reason,
            "device_index": index,
            "rowwise_kernel": rowwise,
            "accumulator": (
                ("fp32" if index in _STATE["fp32_acc"] else "bf16")
                if path != "torchao_rowwise"
                else None
            ),
            "calls": dict(_COUNTS),
            "triton_gemm": _triton_status(index) if path == "unit_scale_fallback" else None,
        }
    except Exception:  # noqa: BLE001 - status must never fail
        return {"path": None}


_CACHE_TAG_PREFIX = "unsloth_fp8_unit_scale="


def _triton_status(index: Any) -> dict:
    """Eager: the card takes the kernel; fast / precise accumulate: bf16 requests of that mode pass its accuracy
    check (eager, live); compiled_kinds: what a trace sends to it (frozen at install)."""
    eager = index in _STATE["triton_eager"]
    fast = precise = False
    if eager:
        from . import diffusion_fp8_triton_gemm as tg
        import torch

        fast = tg.accurate(index, torch.bfloat16, True)
        precise = tg.accurate(index, torch.bfloat16, False)
    return {
        "eager": eager,
        "compiled": index in _STATE["triton_compiled"],
        "fast_accumulate": bool(fast),
        "precise_accumulate": bool(precise),
        "compiled_kinds": _kind_names(k for k in _STATE["triton_kinds"] if k[0] == index),
    }


def _kind_names(kinds: Any) -> list:
    out = []
    for index, dtype, fast in kinds:
        short = {"bfloat16": "bf16", "float16": "fp16"}.get(
            str(dtype).replace("torch.", ""), str(dtype)
        )
        out.append(f"{index}{short}{'f' if fast else 'p'}")
    return sorted(out)


def _set_compile_cache_tag(tag: Optional[str]) -> None:
    """Key torch's FX/AOTAutograd caches on the dispatcher: the Dynamo graph is identical with or without the patch,
    so untagged, a graph cached on the other side of install / the kill switch is reused. ``None`` removes our tag."""
    try:
        import torch.compiler.config as _cc
    except Exception:  # noqa: BLE001 - torch without compiler config: no persistent compile cache to split
        return
    if not isinstance(getattr(_cc, "cache_key_tag", None), str):
        return
    from .diffusion_compile_config import set_compile_cache_tag

    try:
        set_compile_cache_tag(_CACHE_TAG_PREFIX, tag)
    except Exception:  # noqa: BLE001
        pass


def _device_caps() -> dict:
    import torch

    caps = {}
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                caps[i] = tuple(torch.cuda.get_device_capability(i))
    except Exception:  # noqa: BLE001 - no CUDA runtime: nothing qualifies
        return {}
    return caps


def _triton_cards(
    in_use: frozenset,
    caps: dict,
    logger: Any = None,
) -> tuple:
    """(eager cards, compiled cards, compiled kinds, module) for the fused Triton GEMM: measured arches, CUDA only,
    kill switch unset, bf16 self-check passed. A request kind (output dtype x fast / precise accumulate; Studio's
    hosted FP8 checkpoints are baked precise) takes the kernel only where the accuracy check passed it. Compiled kinds
    = (card, dtype, fast) a trace sends to it: self-checked dtypes only, since no check can run inside a trace. Cards
    used later rejoin on the next install."""
    none = (frozenset(), frozenset(), frozenset(), None)
    if not in_use:
        return none
    try:
        from core._torchao_stub import torch_is_rocm
        if torch_is_rocm():
            return none
        from . import diffusion_fp8_triton_gemm as tg
    except Exception:  # noqa: BLE001 - cannot tell / no module: the tensorwise fallback
        return none
    if not tg.enabled() or not tg.available():
        return none
    import torch

    eager, kinds = set(), set()
    for index in sorted(in_use):
        run_eager, run_compiled = tg.card_paths(caps.get(index, ()))
        if not run_eager or not tg.verify(index, torch.bfloat16):
            continue
        eager.add(index)
        tg.verify_accuracy(index)
        if run_compiled and tg.traced_available():
            for dtype in (torch.bfloat16, torch.float16):
                for fast in (True, False):
                    if tg.verified(index, dtype) and tg.accurate(index, dtype, fast):
                        kinds.add((index, dtype, fast))
    compiled = {k[0] for k in kinds}
    failed = tg.describe()["failed"]
    if logger is not None and (eager or failed):
        try:
            logger.info(
                "fp8: fused Triton row-wise GEMM on cuda:%s (compiled: %s; accuracy-checked kinds: %s; %s=0 "
                "disables)%s",
                ",".join(str(i) for i in sorted(eager)) or "-",
                ",".join(_kind_names(kinds)) or "-",
                {i: _triton_status_kinds(tg, i) for i in sorted(eager)},
                tg.ENV,
                f"; kept the tensorwise fallback: {failed}" if failed else "",
            )
        except Exception:  # noqa: BLE001
            pass
    return frozenset(eager), frozenset(compiled), frozenset(kinds), tg


def _triton_status_kinds(tg: Any, index: Any) -> str:
    import torch
    names = [
        f"{short}:{'fast' if fast else 'precise'}"
        for dtype, short in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"))
        for fast in (True, False)
        if tg.accurate(index, dtype, fast)
    ]
    return "+".join(names) or "none"


def install(logger: Any = None, *, fast_accum: Optional[bool] = None) -> bool:
    """Patch torchao addmm; idempotent, False when off, no qualifying GPU, or layout moved."""
    if _STATE["installed"]:
        # A card that started being used after install needs the dispatcher rebuilt: it lacks the row-wise kernel,
        # or it is a fallback card whose accumulator check was deferred while it was idle.
        if _STATE["orig"] is not None:
            pending = _STATE["triton_pending"]
            if any(_has_context(i) for i in _STATE["acc_deferred"]):
                uninstall()
            elif pending is not None and (
                pending["eager"] or pending["accuracy"] or pending["resolved"]
            ):
                # A kernel self-check deferred by an OOM: check again, and give a card that passed since its compiled
                # path and compile cache tag (graphs compiled before keep the tensorwise GEMM they traced).
                uninstall()
            elif _mode() == "auto" and not _STATE["forced_fallback"]:
                caps_now = _device_caps()
                missing = {i for i, ok in rowwise_verdicts(caps_now).items() if ok is False}
                if missing - set(_STATE["fallback"]):
                    uninstall()
        if _STATE["installed"]:
            from .diffusion_compile_config import apply
            apply()
            return True
    mode = _mode()
    if mode == "off":
        return False
    caps = _device_caps()
    forced_fallback = mode == FALLBACK_WORD
    if forced_fallback:
        fallback = frozenset(i for i, c in caps.items() if tuple(c) >= _FP8_MIN_CAP)
    else:
        fallback = frozenset(i for i, ok in rowwise_verdicts(caps).items() if ok is False)
    if not fallback and not can_engage(fast_accum):
        return False
    if mode == "auto" and not fallback and not any(c in _ADA_CAPS for c in caps.values()):
        return False
    try:
        from torchao.quantization.quantize_.workflows.float8 import float8_tensor as _f8t
    except Exception:  # noqa: BLE001 - torchao absent / stubbed / moved
        return False
    orig = getattr(_f8t, "addmm_float8_unwrapped_inference", None)
    if (
        orig is None or not callable(orig) or isinstance(orig, type)
    ):  # a torchao stub's sentinel class
        return False
    if getattr(orig, "_unsloth_unit_scale", False):
        _STATE.update(installed = True)
        return True
    min_rows = _min_rows()
    force = mode == "force"
    qualifying = frozenset(i for i, c in caps.items() if force or c in _ADA_CAPS) | fallback

    import torch

    unit_out_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    # Fallback cards take the tensorwise GEMM's fp32 output: no bf16 rounding before the scales, so eager output is
    # bitwise the row-wise kernel's (FLUX.2 Klein on G4). sm_89 keeps its faster bf16 accumulator.
    # Probed only on fallback cards in use, like the row-wise probe: a card known only from a persisted verdict has no
    # CUDA context here, so it keeps the bf16 accumulator (its tensorwise bf16 GEMM is what the probe proved) until it
    # is used, when the next install rebuilds the dispatcher with its own check.
    in_use = frozenset(i for i in fallback if _has_context(i))
    if not in_use and not any(_has_context(i) for i in caps):
        in_use = fallback & {_current_device()}
    fp32_acc = frozenset(i for i in sorted(in_use) if _fp32_out_runs(i))
    acc_deferred = fallback - in_use
    try:
        triton_eager, triton_compiled, triton_kinds, tg = _triton_cards(in_use, caps, logger)
    except Exception:  # noqa: BLE001 - the fused kernel is optional: the tensorwise route stays
        triton_eager, triton_compiled, triton_kinds, tg = (
            frozenset(),
            frozenset(),
            frozenset(),
            None,
        )
    # Live eager set: an OOM-deferred self-check that passes later joins it from an eager call, and eager calls read
    # the accuracy verdicts live. Traced calls read only the frozen install-time kinds, which the compile tag names.
    live_eager = set(triton_eager)
    pending = {"eager": set(), "accuracy": set(), "next": {}, "resolved": False}
    if tg is not None:
        try:
            for index in sorted(in_use):
                if index not in triton_eager:
                    if tg.card_paths(caps.get(index, ()))[0] and tg.check_pending(
                        index, torch.bfloat16
                    ):
                        pending["eager"].add(index)
                elif tg.accuracy_pending(index):
                    pending["accuracy"].add(index)
        except Exception:  # noqa: BLE001 - nothing to retry then
            pending["eager"].clear()
            pending["accuracy"].clear()

    def retry_pending(index: Any, device: Any) -> None:
        # Eager only, never inside a CUDA graph capture (the checks sync the host), at most every _TRITON_RETRY_S.
        now = time.monotonic()
        if now < pending["next"].get(index, 0.0) or tg._capturing(device):
            return
        pending["next"][index] = now + _TRITON_RETRY_S
        try:
            if index in pending["eager"]:
                if tg.verify(index, torch.bfloat16):
                    pending["eager"].discard(index)
                    live_eager.add(index)
                    pending["resolved"] = True
                    if tg.accuracy_pending(index):
                        pending["accuracy"].add(index)
                elif not tg.check_pending(index, torch.bfloat16):
                    pending["eager"].discard(index)
            if index in pending["accuracy"]:
                if tg.verify_accuracy(index):
                    pending["accuracy"].discard(index)
                    pending["resolved"] = True
                elif not tg.accuracy_pending(index):
                    pending["accuracy"].discard(index)
        except Exception:  # noqa: BLE001 - the retry is optional: the tensorwise route stays
            pending["eager"].discard(index)
            pending["accuracy"].discard(index)
        if logger is not None and pending["resolved"]:
            try:
                logger.info(
                    "fp8: deferred fused Triton GEMM check passed on cuda:%s (eager now; accuracy-checked kinds: "
                    "%s); compiled graphs take it from the next load",
                    index,
                    _triton_status_kinds(tg, index),
                )
            except Exception:  # noqa: BLE001
                pass

    def addmm_float8_unwrapped_inference(
        a_data: Any,
        a_scale: Any,
        b_data: Any,
        b_scale: Any,
        output_dtype: Any,
        output_scale: Optional[Any] = None,
        bias: Optional[Any] = None,
        use_fast_accum: bool = False,
    ) -> Any:
        # No row-wise kernel on this card: every row count and accumulate mode, as torchao would fail.
        missing = a_data.device.type == "cuda" and a_data.device.index in fallback
        # Each request kind (dtype x fast / precise accumulate) only where the accuracy check passed it.
        if missing and output_scale is None and tg is not None:
            index = a_data.device.index
            if _traced(a_data):
                # Traced: the install-time kinds only (no launch can run inside a trace). A launch failure since
                # can only turn the kernel off, never on.
                if (
                    (index, output_dtype, bool(use_fast_accum)) in triton_kinds
                    and tg.eligible(a_data, a_scale, b_data, b_scale, bias, output_dtype)
                    and not tg.card_failed(index)
                ):
                    return tg.traced_gemm(a_data, a_scale, b_data, b_scale, bias, output_dtype)
            else:
                if index in pending["eager"] or index in pending["accuracy"]:
                    retry_pending(index, a_data.device)
                if (
                    index in live_eager
                    and tg.accurate(index, output_dtype, use_fast_accum)
                    and tg.eligible(a_data, a_scale, b_data, b_scale, bias, output_dtype)
                ):
                    out = tg.eager_gemm(a_data, a_scale, b_data, b_scale, bias, output_dtype)
                    if out is not None:
                        _COUNTS["triton_gemm"] += 1
                        return out
        if (
            (missing or force or not use_fast_accum)
            and output_scale is None
            and (output_dtype in unit_out_dtypes if missing else output_dtype == torch.bfloat16)
            and a_data.device.type == "cuda"
            and a_data.device.index in qualifying
            and a_data.dim() == 2
            and b_data.dim() == 2
            and a_scale.dim() == 2
            and b_scale.dim() == 2
            and a_scale.shape[1] == 1
            and b_scale.shape[0] == 1
            and a_scale.shape[0] == a_data.shape[0]
            and b_scale.shape[1] == b_data.shape[1]
            and (missing or a_data.shape[0] >= min_rows)
        ):
            if not _compiling():
                _COUNTS["unit_scale"] += 1
            one = torch.ones((), device = a_data.device, dtype = torch.float32)
            acc = torch._scaled_mm(
                a_data,
                b_data,
                scale_a = one,
                scale_b = one,
                out_dtype = torch.float32 if a_data.device.index in fp32_acc else torch.bfloat16,
                use_fast_accum = use_fast_accum,
            )
            if isinstance(acc, tuple):  # torch < 2.5 returns (out, amax)
                acc = acc[0]
            return _scale_epilogue(acc, a_scale, b_scale, bias, output_dtype)
        if not _compiling():
            _COUNTS["rowwise"] += 1
        return orig(
            a_data,
            a_scale,
            b_data,
            b_scale,
            output_dtype,
            output_scale = output_scale,
            bias = bias,
            use_fast_accum = use_fast_accum,
        )

    addmm_float8_unwrapped_inference._unsloth_unit_scale = True  # type: ignore[attr-defined]
    addmm_float8_unwrapped_inference._unsloth_orig = orig  # type: ignore[attr-defined]
    _f8t.addmm_float8_unwrapped_inference = addmm_float8_unwrapped_inference
    _STATE.update(
        installed = True,
        orig = orig,
        caps = caps,
        qualifying = qualifying,
        fallback = fallback,
        fp32_acc = fp32_acc,
        acc_deferred = acc_deferred,
        forced_fallback = forced_fallback,
        triton_eager = live_eager,
        triton_compiled = triton_compiled,
        triton_kinds = triton_kinds,
        triton_pending = pending,
    )
    # Only a fallback install changes the tag: an sm_89 / force install keys exactly as before.
    # "+" between cards: "," and ";" separate whole tags in cache_key_tag.
    missing_tag = (
        f":rowwise-missing={'+'.join(str(i) for i in sorted(fallback))}"
        f":acc32={'+'.join(str(i) for i in sorted(fp32_acc))}"
        if fallback
        else ""
    )
    # The compiled graphs of a card on the fused kernel hold a triton_op the FX / AOT caches key without its source.
    if triton_compiled:
        missing_tag += (
            f":gemm=triton-{'+'.join(str(i) for i in sorted(triton_compiled))}-{tg.source_digest()}"
        )
        # The request kinds traced to the kernel: graphs from installs with other verdicts must not share a cache entry.
        missing_tag += f":kinds={'+'.join(_kind_names(triton_kinds))}"
    _set_compile_cache_tag(f"{min_rows}{':force' if force else ''}{missing_tag}")
    if logger is not None:
        try:
            if fallback:
                logger.warning(
                    "fp8: no row-wise fp8 _scaled_mm on cuda:%s (%s); running fp8 as a tensorwise GEMM + row/column "
                    "scale epilogue there (every row count, fast accumulate allowed)",
                    ",".join(str(i) for i in sorted(fallback)),
                    f"{UNIT_SCALE_ENV}=fallback" if forced_fallback else "probe",
                )
            if force or qualifying - fallback:
                logger.info(
                    "fp8: unit-scale GEMM + row/column scale epilogue on %s (rows >= %d, %s=0 disables)",
                    "every CUDA device" if force else "sm_89 precise accumulate",
                    min_rows,
                    UNIT_SCALE_ENV,
                )
        except Exception:  # noqa: BLE001
            pass
    return True


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    try:
        from torchao.quantization.quantize_.workflows.float8 import float8_tensor as _f8t
        if _STATE["orig"] is not None:
            _f8t.addmm_float8_unwrapped_inference = _STATE["orig"]
    finally:
        _STATE.update(
            installed = False,
            orig = None,
            caps = {},
            qualifying = frozenset(),
            fallback = frozenset(),
            fp32_acc = frozenset(),
            acc_deferred = frozenset(),
            forced_fallback = False,
            triton_eager = frozenset(),
            triton_compiled = frozenset(),
            triton_kinds = frozenset(),
            triton_pending = None,
        )
        _set_compile_cache_tag(None)

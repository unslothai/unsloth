# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sm_89 precise-accumulate fp8: unit-scale _scaled_mm beats torchao row-wise (fast accumulate stays
on torchao, no gain); UNSLOTH_DIFFUSION_FP8_UNIT_SCALE=0 off. Installed only where it can
engage: installing tags the process-wide compile cache key.

A card whose PyTorch build runs the tensorwise fp8 GEMM but not the row-wise one (the Windows wheels leave the
row-wise kernel out) has no other fp8 matmul: ``engage_rowwise_fallback`` routes every call on it here, at any row
count and either accumulate mode."""

from __future__ import annotations

import os
from typing import Any, Optional

UNIT_SCALE_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE"
MIN_ROWS_ENV = "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS"
# Unit-scale wins from M = 256 up (L4: row-wise 2.5-4x slower from 512); below, tens of microseconds either way.
_DEFAULT_MIN_ROWS = 256
# Eager epilogue chunk size, avoids a full fp32 [M, N] buffer.
_EAGER_EPILOGUE_ELEMS = 1 << 24
_OFF_WORDS = ("0", "off", "false", "no", "none")
_ON_WORDS = ("1", "on", "true", "yes", "force")
_ADA_CAPS = ((8, 9),)

_COUNTS = {"unit_scale": 0, "rowwise": 0}
# "fallback": cards with no row-wise fp8 GEMM, every call on them takes the unit-scale path.
_STATE: dict = {
    "installed": False,
    "orig": None,
    "caps": {},
    "qualifying": frozenset(),
    "force": False,
    "fallback": frozenset(),
    "tag_base": None,
}
# device index -> whether the card needs the row-wise fallback, probed once per process.
_FALLBACK_CHECKED: dict = {}


def _mode() -> str:
    v = (os.environ.get(UNIT_SCALE_ENV) or "").strip().lower()
    if v in _OFF_WORDS:
        return "off"
    if v in _ON_WORDS:
        return "force"
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


def _scale_epilogue(
    acc: Any, a_scale: Any, b_scale: Any, bias: Optional[Any], output_dtype: Any
) -> Any:
    """``(acc * a_scale * b_scale + bias)`` in fp32, rounded once to ``output_dtype``. Eager runs it per row chunk
    in place (bitwise identical) to bound peak memory."""
    import torch

    if _compiling():
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


def fallback_disabled() -> bool:
    return _mode() == "off"


def can_engage(fast_accum: Optional[bool]) -> bool:
    """False only for fast accumulate outside ``force``; None (unknown) installs."""
    return _mode() == "force" or fast_accum is not True


def stats() -> dict:
    return {
        **_COUNTS,
        "installed": bool(_STATE["installed"]),
        "rowwise_fallback_cards": sorted(_STATE["fallback"]),
    }


_CACHE_TAG_PREFIX = "unsloth_fp8_unit_scale="


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


def _torchao_float8_module() -> Any:
    try:
        from torchao.quantization.quantize_.workflows.float8 import float8_tensor as _f8t
    except Exception:  # noqa: BLE001 - torchao absent / stubbed / moved
        return None
    orig = getattr(_f8t, "addmm_float8_unwrapped_inference", None)
    if (
        orig is None or not callable(orig) or isinstance(orig, type)
    ):  # a torchao stub's sentinel class
        return None
    return _f8t


def _cache_tag() -> Optional[str]:
    base = _STATE["tag_base"]
    fallback = _STATE["fallback"]
    if fallback:
        cards = ",".join(str(i) for i in sorted(fallback))
        return f"{base or ''}:rowwise-fallback={cards}"
    return base


def install(logger: Any = None, *, fast_accum: Optional[bool] = None) -> bool:
    """Patch torchao addmm; idempotent, False when off, no qualifying GPU, or layout moved."""
    if _STATE["installed"]:
        from .diffusion_compile_config import apply
        apply()
        # Installed for the row-wise fallback alone, this call may still widen it to its qualifying cards.
        if _STATE["tag_base"] is not None:
            return True
    mode = _mode()
    if mode == "off" or not can_engage(fast_accum):
        return _STATE["installed"]
    caps = _device_caps()
    if mode == "auto" and not any(c in _ADA_CAPS for c in caps.values()):
        return _STATE["installed"]
    force = mode == "force"
    qualifying = frozenset(i for i, c in caps.items() if force or c in _ADA_CAPS)
    return _install_dispatcher(
        logger,
        qualifying = qualifying,
        caps = caps,
        force = force,
        what = "every CUDA device" if force else "sm_89 precise accumulate",
    )


def _install_dispatcher(
    logger: Any, *, qualifying: frozenset, caps: dict, force: bool, what: str
) -> bool:
    min_rows = _min_rows()
    # what="" is the row-wise fallback alone: its own tag suffix keys the cache, no unit-scale opt-in tag.
    tag_base = f"{min_rows}{':force' if force else ''}" if what else None
    if _STATE["installed"]:
        # Installed for the row-wise fallback alone: widen it to the cards this call qualifies.
        if what:
            _STATE.update(qualifying = qualifying, force = force, tag_base = tag_base)
            _set_compile_cache_tag(_cache_tag())
        return True
    _f8t = _torchao_float8_module()
    if _f8t is None:
        return False
    orig = _f8t.addmm_float8_unwrapped_inference
    if getattr(orig, "_unsloth_unit_scale", False):
        _STATE.update(installed = True)
        return True

    import torch

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
        # No row-wise kernel on this card: every shape the epilogue covers comes here, or the GEMM raises.
        fallback = a_data.device.type == "cuda" and a_data.device.index in _STATE["fallback"]
        force = _STATE["force"]
        if (
            (force or fallback or not use_fast_accum)
            and output_scale is None
            and output_dtype == torch.bfloat16
            and a_data.device.type == "cuda"
            and (fallback or a_data.device.index in _STATE["qualifying"])
            and a_data.dim() == 2
            and b_data.dim() == 2
            and a_scale.dim() == 2
            and b_scale.dim() == 2
            and a_scale.shape[1] == 1
            and b_scale.shape[0] == 1
            and a_scale.shape[0] == a_data.shape[0]
            and b_scale.shape[1] == b_data.shape[1]
            and (fallback or a_data.shape[0] >= min_rows)
        ):
            if not _compiling():
                _COUNTS["unit_scale"] += 1
            one = torch.ones((), device = a_data.device, dtype = torch.float32)
            acc = torch._scaled_mm(
                a_data,
                b_data,
                scale_a = one,
                scale_b = one,
                out_dtype = torch.bfloat16,
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
        force = force,
        tag_base = tag_base,
    )
    _set_compile_cache_tag(_cache_tag())
    if logger is not None and what:
        try:
            logger.info(
                "fp8: unit-scale GEMM + row/column scale epilogue on %s (rows >= %d, %s=0 disables)",
                what,
                min_rows,
                UNIT_SCALE_ENV,
            )
        except Exception:  # noqa: BLE001
            pass
    return True


def _device_index(device: Any) -> Optional[int]:
    """The CUDA ordinal ``device`` (int, ``"cuda:N"``, torch.device, None = current card) names, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if isinstance(device, int):
            return device
        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            return None
        return dev.index if dev.index is not None else int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - an unreadable device is not a card to patch
        return None


def _card_gemms(index: int) -> tuple[bool, bool]:
    """``(tensorwise, rowwise)``: whether each fp8 ``_scaled_mm`` runs on card ``index`` (the ModelOpt fp8 probe)."""
    from .diffusion_modelopt_fp8 import _GEMM_PROBE, ACT_DYNAMIC, ACT_STATIC, _probe_gemms

    if index not in _GEMM_PROBE:
        _GEMM_PROBE[index] = _probe_gemms(index)
    found = _GEMM_PROBE[index]
    return bool(found.get(ACT_STATIC)), bool(found.get(ACT_DYNAMIC))


def _is_rocm() -> bool:
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except Exception:  # noqa: BLE001
        return False


def rowwise_missing(device: Any = None) -> Optional[bool]:
    """True when the card runs the tensorwise fp8 GEMM but not the row-wise one, False when it runs the row-wise
    one or no fp8 GEMM at all, None when unknown (no CUDA card). Probed once per card."""
    index = _device_index(device)
    if index is None:
        return None
    if index not in _FALLBACK_CHECKED:
        try:
            tensorwise, rowwise = _card_gemms(index)
            # ROCm fp8 is a different encoding and has its own paths; the fallback is a CUDA build gap.
            _FALLBACK_CHECKED[index] = bool(tensorwise and not rowwise and not _is_rocm())
        except Exception:  # noqa: BLE001 - a probe that cannot run proves nothing either way
            return None
    return _FALLBACK_CHECKED[index]


def last_rowwise_verdict() -> Optional[bool]:
    """Whether any card probed so far lacked the row-wise fp8 GEMM (None: none probed). For refusal text only:
    never probes."""
    if not _FALLBACK_CHECKED:
        return None
    return any(_FALLBACK_CHECKED.values())


def engage_rowwise_fallback(device: Any = None, logger: Any = None) -> bool:
    """On a card with the tensorwise fp8 GEMM but no row-wise one, send every torchao fp8 matmul on it through the
    unit-scale GEMM (installing the dispatcher if needed); True when that card now takes the fallback. A no-op on
    cards with the row-wise kernel, so their compile cache key is untouched. The kill switch turns it off too."""
    if _mode() == "off":
        return False
    index = _device_index(device)
    if index is None:
        return False
    if index in _STATE["fallback"] and _STATE["installed"]:
        return True
    if not rowwise_missing(index):
        return False
    _STATE["fallback"] = _STATE["fallback"] | {index}
    if _STATE["installed"]:
        _set_compile_cache_tag(_cache_tag())
    elif not _install_dispatcher(
        logger,
        qualifying = frozenset(),
        caps = _device_caps(),
        force = False,
        what = "",
    ):
        _STATE["fallback"] = _STATE["fallback"] - {index}
        return False
    if logger is not None:
        try:
            logger.info(
                "fp8: cuda:%d has no row-wise fp8 _scaled_mm in this PyTorch build; running every fp8 matmul on it "
                "as a tensorwise GEMM + row/column scale epilogue (%s=0 disables)",
                index,
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
            force = False,
            fallback = frozenset(),
            tag_base = None,
        )
        _set_compile_cache_tag(None)

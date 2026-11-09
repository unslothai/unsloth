# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm GEMM selection for diffusion denoisers on the measured RDNA APUs (gfx1151): hipBLASLt's default pick at DiT
shapes runs well below the card's best solution, so torch's TunableOp benchmarks the hipBLASLt / rocBLAS candidates
once per GEMM shape and keeps the fastest. Tuning runs only inside the denoiser's forward (its shapes repeat every
step and every render at one resolution) and only on a call shaped like the previous one, so a render's first call
and a prefix-cache prefill (one per prompt, its length set by the prompt) never tune; a bounded per-shape budget;
text encoder / VAE GEMMs only look up what is already tuned. Results persist under the compile cache, keyed on arch + torch + HIP, and torch rejects a
file whose library versions differ (it re-tunes). Speed tiers only; UNSLOTH_DIFFUSION_ROCM_TUNABLEOP=0 keeps the
default pick."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

TUNABLEOP_ENV = "UNSLOTH_DIFFUSION_ROCM_TUNABLEOP"
# Measured archs only (AMD CI gfx1151 GEMM + end-to-end A/B); widen with evidence.
TUNED_ARCHES = frozenset({"gfx1151"})
# Per GEMM shape: at most this long benchmarking candidates, at most this many timed runs of each.
MAX_TUNING_MS = 30
MAX_TUNING_ITERS = 20
_HOOK_ATTR = "_unsloth_tunableop_hooks"

_LOCK = threading.Lock()
_STATE: dict = {"active": False, "depth": 0, "written": 0, "file": None, "last_key": None}


def disabled() -> bool:
    return (os.environ.get(TUNABLEOP_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _user_owns_tunableop() -> bool:
    """A user who set any PYTORCH_TUNABLEOP_* variable drives TunableOp themselves."""
    return any(k.startswith("PYTORCH_TUNABLEOP_") for k in os.environ)


def _arch(device: Any = None) -> Optional[str]:
    try:
        import torch

        props = torch.cuda.get_device_properties(
            device if device is not None else torch.cuda.current_device()
        )
        name = str(getattr(props, "gcnArchName", "") or "")
        return name.split(":", 1)[0] or None
    except Exception:  # noqa: BLE001
        return None


def eligible(device: Any = None) -> tuple:
    """(True, arch) where this module would engage, else (False, reason)."""
    if disabled():
        return False, f"{TUNABLEOP_ENV}=0"
    if _user_owns_tunableop():
        return False, "PYTORCH_TUNABLEOP_* set by the user"
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False, "no torch"
    if not is_rocm_torch(torch) or not torch.cuda.is_available():
        return False, "not ROCm"
    tunable = getattr(torch.cuda, "tunable", None)
    if tunable is None or not hasattr(tunable, "tuning_enable"):
        return False, "torch without TunableOp"
    arch = _arch(device)
    if arch not in TUNED_ARCHES:
        return False, f"arch {arch} not measured"
    return True, arch


def results_file(arch: str) -> Path:
    import torch

    from .diffusion_compile_cache import cache_root

    tag = "_".join(
        re.sub(r"[^A-Za-z0-9.]+", "-", str(x)) for x in (arch, torch.__version__, torch.version.hip)
    )
    return cache_root() / "tunableop" / f"{tag}.csv"


def _write(tunable: Any) -> None:
    """Persist new results in torch's own CSV layout (validators, then one line per tuned GEMM), atomically. torch
    itself writes only at interpreter exit, which a killed server never reaches."""
    try:
        results = list(tunable.get_results() or ())
        validators = list(tunable.get_validators() or ())
    except Exception:  # noqa: BLE001
        return
    if len(results) <= _STATE["written"] or not _STATE["file"]:
        return
    lines = [f"Validator,{k},{v}" for k, v in validators]
    lines += [",".join(str(x) for x in row) for row in results]
    path = Path(_STATE["file"])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding = "utf-8")
        os.replace(tmp, path)
        _STATE["written"] = len(results)
    except OSError:  # an unwritable cache only costs a re-tune next session
        try:
            tmp.unlink()
        except OSError:
            pass


def _call_key(args: Any, kwargs: Any) -> tuple:
    """Shapes of every tensor input plus the string-valued options (e.g. a prefix cache's extract / cached mode)."""
    import torch

    key = []
    for v in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(v, torch.Tensor):
            key.append(tuple(v.shape))
        elif isinstance(v, str):
            key.append(v)
    return tuple(key)


def _pre(_module: Any, args: Any, kwargs: Any) -> None:
    """Tune only on a call shaped like the one before it: the steady steps of a render. A render's first call (and a
    prefix-cache prefill, whose text length changes with every prompt) runs on what is already tuned, so a new prompt
    never pays a tuning pass for shapes it uses once."""
    import torch

    key = _call_key(args, kwargs)
    with _LOCK:
        if not _STATE["active"]:
            return
        _STATE["depth"] += 1
        if _STATE["depth"] == 1:
            steady = key == _STATE.get("last_key")
            _STATE["last_key"] = key
            _STATE["tuning"] = steady
            if steady:
                torch.cuda.tunable.tuning_enable(True)


def _post(_module: Any, _args: Any, _kwargs: Any, _out: Any) -> None:
    import torch
    with _LOCK:
        if not _STATE["active"] or _STATE["depth"] == 0:
            return
        _STATE["depth"] -= 1
        if _STATE["depth"] == 0 and _STATE.pop("tuning", False):
            torch.cuda.tunable.tuning_enable(False)
            _write(torch.cuda.tunable)


def _activate(arch: str, logger: Any = None) -> bool:
    import torch

    tunable = torch.cuda.tunable
    with _LOCK:
        if _STATE["active"]:
            return True
        path = results_file(arch)
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
        except OSError:
            return False
        tunable.set_filename(str(path), insert_device_ordinal = False)
        tunable.set_max_tuning_duration(MAX_TUNING_MS)
        tunable.set_max_tuning_iterations(MAX_TUNING_ITERS)
        tunable.tuning_enable(False)
        tunable.enable(True)
        if path.exists():
            try:
                tunable.read_file(str(path))
            except Exception:  # noqa: BLE001 - a stale / foreign file: torch re-tunes
                pass
        try:
            _STATE["written"] = len(tunable.get_results() or ())
        except Exception:  # noqa: BLE001
            _STATE["written"] = 0
        _STATE.update(active = True, depth = 0, file = str(path), last_key = None)
    if logger is not None:
        logger.info(
            "diffusion.rocm_tunableop: %s GEMMs pick the fastest hipBLASLt / rocBLAS solution per shape "
            "(tuned inside the denoiser, %d known), results %s",
            arch,
            _STATE["written"],
            path,
        )
    return True


def _denoisers(pipe: Any) -> list:
    out = []
    for name in ("transformer", "transformer_2", "unet"):
        mod = getattr(pipe, name, None)
        if mod is not None and callable(getattr(mod, "register_forward_pre_hook", None)):
            out.append(mod)
    return out


def install_for_pipe(
    pipe: Any,
    device: Any = None,
    logger: Any = None,
) -> bool:
    ok, why = eligible(device)
    if not ok:
        return False
    dits = _denoisers(pipe)
    if not dits or not _activate(why, logger):
        return False
    for dit in dits:
        if dit.__dict__.get(_HOOK_ATTR):
            continue
        dit.__dict__[_HOOK_ATTR] = (
            dit.register_forward_pre_hook(_pre, with_kwargs = True),
            dit.register_forward_hook(_post, with_kwargs = True, always_call = True),
        )
    return True


def uninstall(pipe: Any = None) -> None:
    """Back to torch's default pick (the hooks of ``pipe``'s denoisers removed; a later load re-reads the file)."""
    try:
        import torch
        tunable = torch.cuda.tunable
    except Exception:  # noqa: BLE001
        return
    with _LOCK:
        if _STATE["active"]:
            _write(tunable)
            try:
                tunable.tuning_enable(False)
                tunable.enable(False)
            except Exception:  # noqa: BLE001
                pass
        _STATE.update(active = False, depth = 0)
    for dit in _denoisers(pipe) if pipe is not None else ():
        for h in dit.__dict__.pop(_HOOK_ATTR, None) or ():
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass


def is_active() -> bool:
    return bool(_STATE["active"])

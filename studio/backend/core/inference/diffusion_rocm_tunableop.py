# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm TunableOp GEMM tables for diffusion denoisers on gfx1151 (hipBLASLt's default pick is 20-38% slow at DiT
shapes). Tuning costs 17-215 s per shape, so by default only READ the AMD CI tables in ``rocm_gemm_tables/`` whose
validators match exactly. ``UNSLOTH_DIFFUSION_ROCM_TUNABLEOP=tune`` also tunes on steady denoiser calls; ``=0`` off."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

TUNABLEOP_ENV = "UNSLOTH_DIFFUSION_ROCM_TUNABLEOP"
# Measured archs only; widen with evidence.
TUNED_ARCHES = frozenset({"gfx1151"})
MAX_TUNING_MS = 30
MAX_TUNING_ITERS = 20
_HOOK_ATTR = "_unsloth_tunableop_hooks"

_LOCK = threading.Lock()
_STATE: dict = {
    "active": False,
    "depth": 0,
    "written": 0,
    "file": None,
    "last_key": None,
    "tune": False,
}


_TABLE_DIR = Path(__file__).resolve().parent / "rocm_gemm_tables"


def _env() -> str:
    return (os.environ.get(TUNABLEOP_ENV) or "").strip().lower()


def disabled() -> bool:
    return _env() in ("0", "off", "false", "no")


def tuning_requested() -> bool:
    return _env() == "tune"


def _validators(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.strip().split(",", 2)
        if len(parts) == 3 and parts[0] == "Validator":
            out[parts[1]] = parts[2]
    return out


def shipped_tables(arch: str, validators: dict) -> list:
    """Tables whose validators all match: solution indices are only valid for the exact build that measured them."""
    out = []
    want = {str(k): str(v) for k, v in validators.items()}
    for path in sorted(_TABLE_DIR.glob(f"{arch}_*.csv")):
        try:
            have = _validators(path.read_text(encoding = "utf-8"))
        except OSError:
            continue
        if have and have == want:
            out.append(path)
    return out


def _user_owns_tunableop() -> bool:
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


def _device_count(torch: Any) -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def eligible(device: Any = None) -> tuple:
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
    # TunableOp is process-global and keyed on shape, not device: every visible card must share the arch.
    others = {_arch(i) for i in range(_device_count(torch))} - {arch}
    if others:
        return False, f"mixed archs {sorted(map(str, others))} beside {arch}"
    return True, arch


def results_file(arch: str) -> Path:
    import torch

    from .diffusion_compile_cache import cache_root

    # The HIP build string only labels the file (an f-string, as the ROCm gate guard expects); None on CUDA.
    hip = f"{getattr(torch.version, 'hip', None)}"
    tag = "_".join(re.sub(r"[^A-Za-z0-9.]+", "-", str(x)) for x in (arch, torch.__version__, hip))
    return cache_root() / "tunableop" / f"{tag}.csv"


def _write(tunable: Any) -> None:
    """Atomic write in torch's CSV layout; torch only writes at exit, which a killed server never reaches."""
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
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _call_key(args: Any, kwargs: Any) -> tuple:
    import torch

    key = []
    for v in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(v, torch.Tensor):
            key.append(tuple(v.shape))
        elif isinstance(v, str):
            key.append(v)
    return tuple(key)


def _pre(_module: Any, args: Any, kwargs: Any) -> None:
    """Tune only on a call shaped like the previous one, so one-off shapes (first call, prefix prefill) never tune."""
    import torch

    key = _call_key(args, kwargs)
    with _LOCK:
        if not _STATE["active"]:
            return
        _STATE["depth"] += 1
        if _STATE["depth"] == 1:
            steady = key == _STATE.get("last_key") and bool(_STATE.get("tune"))
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
        tune = tuning_requested()
        path = results_file(arch)
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
        except OSError:
            if tune:
                return False
        tunable.tuning_enable(False)
        tunable.enable(True)
        _prime(torch)
        try:
            validators = dict(tunable.get_validators() or ())
        except Exception:  # noqa: BLE001
            validators = {}
        tables = shipped_tables(arch, validators)
        cached = path.exists() and _validators(_read(path)) == {
            str(k): str(v) for k, v in validators.items()
        }
        if not tune and not tables and not cached:
            tunable.enable(False)
            return False
        tunable.set_filename(str(path), insert_device_ordinal = False)
        tunable.set_max_tuning_duration(MAX_TUNING_MS)
        tunable.set_max_tuning_iterations(MAX_TUNING_ITERS)
        for src in ([path] if cached else []) + tables:
            try:
                tunable.read_file(str(src))
            except Exception:  # noqa: BLE001 - a stale / foreign file: the default pick
                pass
        try:
            _STATE["written"] = len(tunable.get_results() or ())
        except Exception:  # noqa: BLE001
            _STATE["written"] = 0
        _STATE.update(active = True, depth = 0, file = str(path), last_key = None, tune = tune)
    if logger is not None:
        logger.info(
            "diffusion.rocm_tunableop: %s GEMMs use the measured fastest solution for %d shapes (%s)%s",
            arch,
            _STATE["written"],
            ", ".join(p.name for p in tables) or "local cache",
            f"; tuning new denoiser shapes into {path}" if tune else "",
        )
    return True


def _prime(torch: Any) -> None:
    """torch registers validators on its first tunable GEMM: run a tiny one before comparing them."""
    try:
        a = torch.ones(8, 8, dtype = torch.bfloat16, device = "cuda")
        (a @ a).sum().item()
    except Exception:  # noqa: BLE001 - validators then stay partial and no table matches
        pass


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding = "utf-8")
    except OSError:
        return ""


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
        if _STATE["active"]:  # a load on another arch must not inherit an earlier load's table
            uninstall(None)
        return False
    dits = _denoisers(pipe)
    if not dits or not _activate(why, logger):
        return False
    if not _STATE.get("tune"):
        return True
    for dit in dits:
        if dit.__dict__.get(_HOOK_ATTR):
            continue
        dit.__dict__[_HOOK_ATTR] = (
            dit.register_forward_pre_hook(_pre, with_kwargs = True),
            dit.register_forward_hook(_post, with_kwargs = True, always_call = True),
        )
    return True


def uninstall(pipe: Any = None) -> None:
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

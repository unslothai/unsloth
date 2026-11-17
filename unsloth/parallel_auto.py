# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL=auto: plain DDP when the model and its training state fit on every GPU, else native FSDP2.

    UNSLOTH_PARALLEL=auto torchrun --nproc-per-node 2 train.py

Opt-in. An explicit layout (UNSLOTH_PARALLEL="fsdp=1", UNSLOTH_FSDP2=1, ...) is untouched by this module.

The choice is made once per process, at the first Unsloth `from_pretrained` (load-time paths read UNSLOTH_FSDP2:
the 4-bit storage dtype, sharded loading), or at the first Trainer if no Unsloth load ran. It writes
UNSLOTH_FSDP2=0 (DDP) or 1 (native FSDP2), so every later reader sees an ordinary explicit launch. Every rank
sizes its own GPU and the ranks agree with one collective: DDP only when it fits on all of them.

Fits means weights + training state + activation margin <= the memory this rank can use, where
  * weights: the checkpoint's safetensors headers (no weights downloaded), at the bytes they take once loaded
    (4-bit / 8-bit quantize on load counts the quantized linear weights), or the loaded model's parameters;
  * training state, all of it per rank under DDP: gradient + DDP bucket copy + two AdamW moments per trainable
    parameter (LoRA r=16 on every linear weight before get_peft_model has run);
  * activation margin: _ACTIVATION_SHARE of the GPU. Activations are per rank under both layouts, so they only
    keep headroom for DDP's larger resident state.
Anything that cannot be sized picks FSDP2, the launcher default.
"""

from __future__ import annotations

import math
import os

__all__ = [
    "AUTO",
    "auto_requested",
    "check_auto_env",
    "decide",
    "training_state_bytes",
    "resolve_for_load",
    "resolve_for_trainer",
    "resolved_choice",
]

AUTO = "auto"
FLAG = "UNSLOTH_PARALLEL"
# Written beside UNSLOTH_FSDP2 so a child process (dataset workers, a re-exec) inherits the decision.
RESOLVED = "UNSLOTH_PARALLEL_AUTO_RESOLVED"
# Per-rank memory budget override in GiB (default: this GPU's free memory, or host RAM per local rank on CPU).
BUDGET = "UNSLOTH_PARALLEL_AUTO_GB"
DDP, FSDP2 = "ddp", "fsdp2"
# The activation share loader_utils._plan_fits_first_device keeps free when it plans a device map. In the M2
# Kaggle 2x T4 notebook sweep, DDP's training peak above its post-setup reserved memory was 0.97 GB median and
# 3.04 GB p90 (21% of a 14.56 GB T4).
_ACTIVATION_SHARE = 0.2
_LORA_RANK = 16  # get_peft_model's default r
_GiB = 1024**3
_TRUTHY = ("1", "true", "yes", "on")
_EMBEDDING_NAMES = ("embed", "wte", "wpe", "lm_head", "output_layer", "shared")

# {"choice": None | "ddp" | "fsdp2" | "single", "reason": str}
_STATE = {"choice": None, "reason": None}


def auto_requested(environ = None):
    environ = os.environ if environ is None else environ
    return str(environ.get(FLAG, "")).strip().lower() == AUTO


def resolved_choice():
    return _STATE["choice"]


def _world(environ):
    try:
        return int(environ.get("WORLD_SIZE", "1") or "1")
    except ValueError:
        return 1


def check_auto_env(environ = None):
    """`import unsloth` under UNSLOTH_PARALLEL=auto: refuse what already fixes the layout (ValueError on every
    rank alike), and adopt a decision inherited from a parent process."""
    environ = os.environ if environ is None else environ
    fsdp2 = str(environ.get("UNSLOTH_FSDP2", "")).strip().lower()
    inherited = str(environ.get(RESOLVED, "")).strip().lower()
    if inherited in (DDP, FSDP2) and fsdp2 == ("1" if inherited == FSDP2 else "0"):
        _STATE.update(choice = inherited, reason = f"inherited {RESOLVED}={inherited}")
        return
    if fsdp2:
        raise ValueError(
            f"Unsloth: {FLAG}=auto picks DDP or FSDP2 itself, but UNSLOTH_FSDP2={environ.get('UNSLOTH_FSDP2')} "
            f'already does. Use one of the two: unset UNSLOTH_FSDP2, or set {FLAG}="fsdp=1" / "fsdp=0".'
        )
    for name in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE"):
        if str(environ.get(name, "")).strip() not in ("", "1"):
            raise ValueError(
                f"Unsloth: {FLAG}=auto only chooses the data parallel mode (DDP or FSDP2), but "
                f"{name}={environ.get(name)} asks for another axis. Give the whole layout instead, e.g. "
                f'{FLAG}="tp=2,fsdp=1".'
            )
    if str(environ.get(BUDGET, "")).strip():
        _parse_budget(environ.get(BUDGET))
    if (
        str(environ.get("ACCELERATE_USE_FSDP", "")).strip().lower() in _TRUTHY
        or str(environ.get("FSDP_VERSION", "")).strip()
    ):
        raise ValueError(
            f"Unsloth: {FLAG}=auto under an `accelerate launch` FSDP config: accelerate already picked FSDP. "
            f"Launch with torchrun, or drop {FLAG}=auto."
        )


def training_state_bytes(
    trainable_numel,
    grad_itemsize = 4,
    optim = None,
):
    """What DDP keeps per trainable parameter on every rank: the gradient, its DDP bucket copy, and the optimizer
    moments (two fp32 AdamW moments; 8-bit optimizers two bytes)."""
    name = str(getattr(optim, "value", optim) or "").lower()
    moments = 2 if "8bit" in name else 8
    return int(trainable_numel) * (2 * int(grad_itemsize) + moments)


def decide(
    weights,
    state,
    capacity,
    total = None,
    share = _ACTIVATION_SHARE,
):
    """(fits, need) for one rank: fits when weights + state + share x total <= capacity."""
    total = capacity if total is None else total
    need = int(weights) + int(state) + int(share * total)
    return (capacity > 0 and need <= capacity), need


def _parse_budget(raw):
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise ValueError(f"Unsloth: {BUDGET}={raw!r} must be a number of GiB.") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Unsloth: {BUDGET}={raw!r} must be a positive number of GiB.")
    return int(value * _GiB)


def _local_cuda_device(torch):
    """cuda:LOCAL_RANK (the GPU this rank trains on) without making it current; the current device when each
    rank sees only its own GPU (LOCAL_RANK beyond the visible count)."""
    try:
        local = int(os.environ.get("LOCAL_RANK", ""))
    except ValueError:
        local = None
    if local is not None and 0 <= local < torch.cuda.device_count():
        return local
    return torch.cuda.current_device()


def _budget(held = 0):
    """(capacity, total) bytes for this rank: what it could use (free + what it already holds) and the device
    size the margin is a share of."""
    raw = str(os.environ.get(BUDGET, "")).strip()
    if raw:
        value = _parse_budget(raw)
        return value, value
    import torch

    if torch.cuda.is_available():
        device = _local_cuda_device(torch)
        free, total = torch.cuda.mem_get_info(device)
        return int(free + torch.cuda.memory_reserved(device)), int(total)
    try:
        local = max(int(os.environ.get("LOCAL_WORLD_SIZE", "1") or "1"), 1)
    except ValueError:
        local = 1
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.available + held) // local, int(vm.total) // local
    except Exception:
        return 0, 0


def _gather(entry):
    """[entry of every rank], creating the process group the Trainer would create if none exists yet."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl" if cuda else "gloo")
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, entry)
    return out


def _gib(n):
    return f"{n / _GiB:.2f} GiB" if n >= _GiB / 10 else f"{n / 1024**2:.1f} MiB"


def _resolve(estimate, install_trainer_init):
    """Run once: `estimate()` -> (weights, state, held); agree over the ranks; write
    UNSLOTH_FSDP2. Returns the choice."""
    if _STATE["choice"] is not None or not auto_requested():
        return _STATE["choice"]
    world = _world(os.environ)
    if world <= 1:
        _STATE.update(choice = "single", reason = "one process: nothing to choose")
        return _STATE["choice"]
    try:
        weights, state, held = estimate()
        capacity, total = _budget(held)
        fits, need = decide(weights, state, capacity, total)
        entry = dict(
            fits = bool(fits),
            need = need,
            capacity = capacity,
            weights = weights,
            state = state,
            margin = need - weights - state,
        )
    except Exception as exc:
        entry = dict(fits = False, why = f"could not size the model ({type(exc).__name__}: {exc})")
    entries = _gather(entry)
    choice = DDP if all(e.get("fits") for e in entries) else FSDP2
    reason = _describe(choice, entries)
    os.environ["UNSLOTH_FSDP2"] = "1" if choice == FSDP2 else "0"
    os.environ[RESOLVED] = choice
    _STATE.update(choice = choice, reason = reason)
    from .fsdp2_launch import apply_unsloth_fsdp2_launch_env

    apply_unsloth_fsdp2_launch_env()
    if install_trainer_init:
        from .fsdp2_launch import install_native_fsdp2_trainer_init
        install_native_fsdp2_trainer_init()
    if str(os.environ.get("RANK", "0")).strip() in ("", "0"):
        print(reason)
    return choice


def _describe(choice, entries):
    head = f"Unsloth: {FLAG}=auto picked {'DDP (UNSLOTH_FSDP2=0)' if choice == DDP else 'native FSDP2 (UNSLOTH_FSDP2=1)'}"
    first = entries[0]
    if choice == DDP:
        tight = min(entries, key = lambda e: e["capacity"] - e["need"])
        return (
            f"{head}: weights {_gib(first['weights'])} + training state {_gib(first['state'])} + activation "
            f"margin {_gib(first['margin'])} fits on every rank ({len(entries)} ranks, tightest has "
            f"{_gib(tight['capacity'])}). Set {FLAG}=\"fsdp=1\" to force FSDP2."
        )
    reasons = []
    for rank, e in enumerate(entries):
        if e.get("fits"):
            continue
        if e.get("why"):
            reasons.append(f"rank {rank}: {e['why']}")
        else:
            reasons.append(
                f"rank {rank} needs {_gib(e['need'])} (weights {_gib(e['weights'])} + training state "
                f"{_gib(e['state'])} + activation margin {_gib(e['margin'])}) but has {_gib(e['capacity'])}"
            )
    split = (
        " (the ranks disagree, so all take FSDP2)" if any(e.get("fits") for e in entries) else ""
    )
    return f"{head}{split}: {'; '.join(reasons)}. Set {FLAG}=\"fsdp=0\" to force DDP."


# --------------------------------------------------------------------------------------------------
# Sizing a checkpoint before it loads (top-level from_pretrained)
# --------------------------------------------------------------------------------------------------

_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}  # fmt: skip
_PACKED = ("U8", "I8")


def _checkpoint_tensors(
    name,
    token = None,
    revision = None,
    subfolder = "",
):
    """[(tensor name, safetensors dtype, shape)] from the safetensors headers only (no weights read)."""
    if os.path.isdir(name):
        import glob

        from safetensors import safe_open

        root = os.path.join(name, subfolder or "")
        files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no safetensors checkpoint in {root}")
        out = []
        for path in files:
            with safe_open(path, framework = "pt", device = "cpu") as f:
                for key in f.keys():
                    piece = f.get_slice(key)
                    out.append((key, str(piece.get_dtype()).upper(), tuple(piece.get_shape())))
        return out
    from huggingface_hub import get_safetensors_metadata

    if subfolder:
        raise ValueError(f"a Hub checkpoint in subfolder {subfolder!r} is not sized")
    meta = get_safetensors_metadata(name, revision = revision, token = token)
    return [
        (key, str(info.dtype).upper(), tuple(info.shape))
        for file_meta in meta.files_metadata.values()
        for key, info in file_meta.tensors.items()
    ]


def _is_quantizable(key, shape):
    """A linear weight bitsandbytes quantizes on load: 2-D, not an embedding / output head / norm."""
    low = key.lower()
    return (
        len(shape) == 2
        and low.endswith(".weight")
        and not any(part in low for part in _EMBEDDING_NAMES)
        and "norm" not in low
    )


def checkpoint_estimate(tensors, load_in_4bit = False, load_in_8bit = False, full_finetuning = False,
                        dtype_bytes = 2, lora_rank = _LORA_RANK):  # fmt: skip
    """(weights bytes once loaded, trainable parameter count) for safetensors header entries."""
    packed = any(dt in _PACKED for _, dt, _ in tensors)
    quantize = not packed and not full_finetuning and (load_in_4bit or load_in_8bit)
    per_quant = 0.53 if load_in_4bit else 1.03  # packed codes + absmax / scales
    weights, numel_total, lora = 0, 0, 0
    for key, dt, shape in tensors:
        numel = math.prod(shape) if shape else 1
        stored = _DTYPE_BYTES.get(dt, 2)
        if dt in _PACKED:
            weights += numel * stored
            numel_total += 2 * numel  # two 4-bit codes per byte
            if key.lower().endswith(".weight") and len(shape) >= 1:
                # packed [n / 2, 1]: in + out of a square layer of 2 n parameters
                lora += lora_rank * 2 * int(math.sqrt(2 * numel))
            continue
        # Floating-point weights load in the model dtype (an fp32 load upcasts a bf16 checkpoint); bitsandbytes
        # quant state beside packed weights keeps its stored dtype.
        quant_state = packed and any(
            t in key for t in ("absmax", "quant_map", "quant_state", "nested_")
        )
        loaded = dtype_bytes if dt.startswith(("F", "BF")) and not quant_state else stored
        numel_total += numel
        if _is_quantizable(key, shape):
            lora += lora_rank * (shape[0] + shape[1])
            if quantize:
                weights += int(numel * per_quant)
                continue
        weights += numel * loaded
    trainable = numel_total if full_finetuning else lora
    return int(weights), int(trainable)


def resolve_for_load(loader, args, kwargs):
    """First Unsloth `from_pretrained` under UNSLOTH_PARALLEL=auto: size the checkpoint before it loads."""
    if _STATE["choice"] is not None or not auto_requested():
        return _STATE["choice"]

    def estimate():
        import inspect

        bound = inspect.signature(loader).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        call.update(call.pop("kwargs", None) or {})
        name = call.get("model_name") or call.get("pretrained_model_name_or_path")
        if not isinstance(name, str):
            raise ValueError(f"model_name {name!r} is not a checkpoint name or path")
        if call.get("quantization_config") is not None or call.get("load_in_fp8") not in (
            None,
            False,
        ):
            # The loader's precision then depends on the config / the fp8 route: not sized here.
            raise ValueError("a quantization_config or load_in_fp8 load is not sized")
        dtype = call.get("dtype") or call.get("torch_dtype")
        dtype_bytes = 4 if "float32" in str(dtype) else 2
        full = bool(call.get("full_finetuning"))
        load_in_4bit = bool(call.get("load_in_4bit")) and not (
            call.get("load_in_8bit") or call.get("load_in_16bit") or full
        )
        if not call.get("use_exact_model_name"):
            # The repo the loader loads: get_model_name maps a 4-bit mirror to its float repo and back.
            from .models.loader_utils import get_model_name
            name = get_model_name(
                name,
                load_in_4bit = load_in_4bit,
                load_in_fp8 = False,
                token = call.get("token"),
                trust_remote_code = bool(call.get("trust_remote_code")),
                local_files_only = call.get("local_files_only", False),
                cache_dir = call.get("cache_dir", None),
                revision = call.get("revision"),
            )
        tensors = _checkpoint_tensors(
            name, token = call.get("token"), revision = call.get("revision"),
            subfolder = call.get("subfolder") or "",
        )  # fmt: skip
        from .models.mapper import bf16_repo_loads_16bit

        # The loader's own rule, on the mapped name as the loader applies it: a -bf16 Hub repo turns 4bit / 8bit off.
        load_in_16bit = bool(call.get("load_in_16bit")) or bf16_repo_loads_16bit(
            name, call.get("load_in_16bit")
        )
        weights, trainable = checkpoint_estimate(
            tensors,
            load_in_4bit = load_in_4bit and not load_in_16bit,
            load_in_8bit = bool(call.get("load_in_8bit")) and not load_in_16bit,
            full_finetuning = full,
            dtype_bytes = dtype_bytes,
        )
        state = training_state_bytes(trainable, grad_itemsize = dtype_bytes if full else 4)
        return weights, state, 0

    return _resolve(estimate, install_trainer_init = True)


# --------------------------------------------------------------------------------------------------
# Sizing a model already built (a Trainer whose model no Unsloth load made)
# --------------------------------------------------------------------------------------------------


def resolve_for_trainer(model, training_args = None):
    """Trainer.__init__ under UNSLOTH_PARALLEL=auto when no Unsloth load decided: size the built model."""
    if _STATE["choice"] is not None or not auto_requested() or model is None:
        return _STATE["choice"]

    def estimate():
        seen, weights, trainable, grad_itemsize = set(), 0, 0, 4
        for p in list(model.parameters()) + list(model.buffers()):
            if id(p) in seen:
                continue
            seen.add(id(p))
            weights += p.numel() * p.element_size()
            if getattr(p, "requires_grad", False):
                trainable += p.numel()
                grad_itemsize = p.element_size()
        state = training_state_bytes(
            trainable, grad_itemsize, getattr(training_args, "optim", None)
        )
        # Already resident: on CPU it is out of the available RAM _budget reads (CUDA counts reserved).
        held = weights
        return weights, state, held

    return _resolve(estimate, install_trainer_init = False)

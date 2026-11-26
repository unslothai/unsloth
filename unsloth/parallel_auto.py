# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL=auto: plain DDP when the model and its training state fit on every GPU, else native FSDP2.

Decided once per process (first Unsloth `from_pretrained`, else first Trainer) by writing UNSLOTH_FSDP2=0 / 1;
the ranks agree with one collective. Fits = weights + DDP training state + _ACTIVATION_SHARE of the GPU <= capacity.
Conservative by construction: only loads modelled exactly through the loader's own helpers are sized; anything
else (quantization_config, fp8, vLLM, FORCE_FLOAT32 / CUSTOM_DTYPE, unknown dtype, subfolder, unreadable
config / header) declines to FSDP2, the launcher default, with the reason in the rank 0 line.
"""

from __future__ import annotations

import importlib
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
RESOLVED = "UNSLOTH_PARALLEL_AUTO_RESOLVED"
BUDGET = "UNSLOTH_PARALLEL_AUTO_GB"
DDP, FSDP2 = "ddp", "fsdp2"
# Matches loader_utils._plan_fits_first_device. Kaggle 2x T4 sweep: DDP training peak above post-setup
# reserved was 0.97 GB median, 3.04 GB p90 (21% of a 14.56 GB T4).
_ACTIVATION_SHARE = 0.2
_LORA_RANK = 16  # get_peft_model's default r
_GiB = 1024**3
_TRUTHY = ("1", "true", "yes", "on")
_EMBEDDING_NAMES = ("embed", "wte", "wpe", "lm_head", "output_layer", "shared")

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
    # Before the inherited return: a parent never sets these, so a child that has one asks for another axis.
    for name in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "UNSLOTH_EP_SIZE"):
        if str(environ.get(name, "")).strip() not in ("", "1"):
            raise ValueError(
                f"Unsloth: {FLAG}=auto only chooses the data parallel mode (DDP or FSDP2), but "
                f"{name}={environ.get(name)} asks for another axis. Give the whole layout instead, e.g. "
                f'{FLAG}="tp=2,fsdp=1".'
            )
    # After it: a parent that picked FSDP2 exported UNSLOTH_FSDP2 and accelerate's FSDP env itself.
    if inherited in (DDP, FSDP2) and fsdp2 == ("1" if inherited == FSDP2 else "0"):
        _STATE.update(choice = inherited, reason = f"inherited {RESOLVED}={inherited}")
        return
    if fsdp2:
        raise ValueError(
            f"Unsloth: {FLAG}=auto picks DDP or FSDP2 itself, but UNSLOTH_FSDP2={environ.get('UNSLOTH_FSDP2')} "
            f'already does. Use one of the two: unset UNSLOTH_FSDP2, or set {FLAG}="fsdp=1" / "fsdp=0".'
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
    """Per trainable parameter under DDP: gradient + bucket copy + two AdamW moments (fp32, or 1 byte each 8-bit)."""
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
    """cuda:LOCAL_RANK without making it current; the current device if LOCAL_RANK exceeds the visible count."""
    try:
        local = int(os.environ.get("LOCAL_RANK", ""))
    except ValueError:
        local = None
    if local is not None and 0 <= local < torch.cuda.device_count():
        return local
    return torch.cuda.current_device()


def _budget(held = 0, held_cuda = 0):
    """(capacity, total) bytes for this rank: what it could use and the device size the margin is a share of.
    CUDA: free + the allocator's reusable cache (reserved - allocated) + `held_cuda`, the part of the sized model
    already resident there; other live tensors (a reference model loaded first) are not available."""
    raw = str(os.environ.get(BUDGET, "")).strip()
    if raw:
        value = _parse_budget(raw)
        return value, value
    import torch

    if torch.cuda.is_available():
        device = _local_cuda_device(torch)
        free, total = torch.cuda.mem_get_info(device)
        cache = max(torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device), 0)
        return int(free + cache + held_cuda), int(total)
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
        # Native FSDP2's own default group: NCCL on CUDA, gloo where NCCL is missing (Windows CUDA builds).
        from .models.fsdp2_native import _ensure_process_group

        cuda = torch.cuda.is_available()
        device = torch.device("cuda", _local_cuda_device(torch)) if cuda else torch.device("cpu")
        _ensure_process_group(device)
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, entry)
    return out


def _gib(n):
    return f"{n / _GiB:.2f} GiB" if n >= _GiB / 10 else f"{n / 1024**2:.1f} MiB"


def _resolve(estimate, install_trainer_init):
    """Run once: `estimate()` -> (weights, state, held); agree over ranks; write UNSLOTH_FSDP2."""
    if _STATE["choice"] is not None or not auto_requested():
        return _STATE["choice"]
    world = _world(os.environ)
    if world <= 1:
        _STATE.update(choice = "single", reason = "one process: nothing to choose")
        return _STATE["choice"]
    try:
        weights, state, held, *rest = estimate()
        capacity, total = _budget(held, *rest)
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
                        dtype_bytes = 2, lora_rank = _LORA_RANK, skip_modules = None,
                        hidden_size = None):  # fmt: skip
    """(weights bytes once loaded, trainable parameter count) for safetensors header entries. Quantizing on
    load needs `skip_modules`, the loader's llm_int8_skip_modules: a linear whose name contains one stays
    dense (a substring match keeps at least what the loader keeps)."""
    packed = any(dt in _PACKED for _, dt, _ in tensors)
    quantize = not packed and not full_finetuning and (load_in_4bit or load_in_8bit)
    if quantize and skip_modules is None:
        raise ValueError("quantizing on load is not sized without the loader's skip modules")
    per_quant = 0.53 if load_in_4bit else 1.03  # packed codes + absmax / scales
    weights, numel_total, lora = 0, 0, 0
    for key, dt, shape in tensors:
        numel = math.prod(shape) if shape else 1
        stored = _DTYPE_BYTES.get(dt, 2)
        if dt in _PACKED:
            weights += numel * stored
            numel_total += 2 * numel  # two 4-bit codes per byte
            if key.lower().endswith(".weight") and len(shape) >= 1:
                # packed [n / 2, 1] = 2n params; one side is hidden_size (square undercounts MLP / GQA).
                params = 2 * numel
                if hidden_size and params % hidden_size == 0:
                    lora += lora_rank * (hidden_size + params // hidden_size)
                else:
                    lora += lora_rank * 2 * int(math.sqrt(params))
            continue
        # Float weights load in the model dtype; bnb quant state beside packed weights keeps its stored dtype.
        quant_state = packed and any(
            t in key for t in ("absmax", "quant_map", "quant_state", "nested_")
        )
        loaded = dtype_bytes if dt.startswith(("F", "BF")) and not quant_state else stored
        numel_total += numel
        if _is_quantizable(key, shape):
            lora += lora_rank * (shape[0] + shape[1])
            module = key[: -len(".weight")]
            if quantize and not any(m in module for m in skip_modules):
                weights += int(numel * per_quant)
                continue
        weights += numel * loaded
    trainable = numel_total if full_finetuning else lora
    return int(weights), int(trainable)


def _peft_base(name, call, revision):
    """peft_config.base_model_name_or_path when `name` is a PEFT adapter, found the way the loaders find it
    (PeftConfig.from_pretrained; a failure means not an adapter), else None."""
    if os.path.isdir(name) and not os.path.isfile(os.path.join(name, "adapter_config.json")):
        return None
    try:
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(
            name,
            token = call.get("token"),
            revision = revision,
            local_files_only = call.get("local_files_only", False),
        )
    except Exception:
        return None
    return getattr(config, "base_model_name_or_path", None) or None


def _loader_precision():
    """(bfloat16 supported, FORCE_FLOAT32 model types): the loader's own values."""
    from .models._utils import is_bfloat16_supported
    from .models.loader import FORCE_FLOAT32

    return bool(is_bfloat16_supported()), [str(x).lower() for x in FORCE_FLOAT32]


def _config(name, call, revision):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(
        name,
        token = call.get("token"),
        revision = revision,
        trust_remote_code = bool(call.get("trust_remote_code")),
        local_files_only = call.get("local_files_only", False),
        cache_dir = call.get("cache_dir", None),
    )


def _model_types(name, call, revision):
    """The loader's model types for `name` (config.json only, read as the loader reads it)."""
    from unsloth_zoo.hf_utils import get_transformers_model_type

    trust = bool(call.get("trust_remote_code"))
    config = _config(name, call, revision)
    return [str(t).lower() for t in get_transformers_model_type(config, trust_remote_code = trust)]


def _hidden_size(name, call, revision):
    try:
        config = _config(name, call, revision)
        config = config.get_text_config() if hasattr(config, "get_text_config") else config
        value = getattr(config, "hidden_size", None)
        return int(value) if value else None
    except Exception:
        return None


def _quant_skip_modules(model_types):
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

    from .models.vision import _architecture_skip_modules
    return list(SKIP_QUANTIZATION_MODULES) + list(_architecture_skip_modules(model_types))


def _loaded_dtype(dtype, supports_bf16):
    """The dtype the loaders load float weights in: None / auto picks bfloat16 where supported, else
    float16; bfloat16 without support drops to float16."""
    text = "" if dtype is None else str(dtype).lower()
    if dtype is None or text == "auto":
        return "bfloat16" if supports_bf16 else "float16"
    for name in ("float32", "bfloat16", "float16"):
        if name in text:
            return "float16" if name == "bfloat16" and not supports_bf16 else name
    raise ValueError(f"dtype {dtype!r} is not sized")


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
        if call.get("fast_inference"):
            raise ValueError("vLLM (fast_inference) memory is not sized")
        for env in ("UNSLOTH_FORCE_FLOAT32", "UNSLOTH_FORCE_CUSTOM_DTYPE"):
            if os.environ.get(env, "") not in ("", "0"):
                raise ValueError(f"{env} changes the loaded dtype; not sized")
        supports_bf16, force_float32 = _loader_precision()
        loaded = _loaded_dtype(call.get("dtype") or call.get("torch_dtype"), supports_bf16)
        dtype_bytes = 4 if loaded == "float32" else 2
        full = bool(call.get("full_finetuning"))
        load_in_8bit = bool(call.get("load_in_8bit"))
        load_in_4bit = bool(call.get("load_in_4bit")) and not (
            load_in_8bit or call.get("load_in_16bit") or full
        )
        if not importlib.import_module("unsloth.device_type").ALLOW_BITSANDBYTES:
            # The loaders' own gate: bitsandbytes unusable here, so 4bit / 8bit are off before the name maps.
            load_in_4bit = load_in_8bit = False
        exact = bool(call.get("use_exact_model_name"))
        revision = call.get("revision")

        def mapped(repo, revision):
            if exact:
                return repo
            # The repo the loader loads: get_model_name maps a 4-bit mirror to its float repo and back.
            from .models.loader_utils import get_model_name
            return get_model_name(
                repo,
                load_in_4bit = load_in_4bit,
                load_in_fp8 = False,
                token = call.get("token"),
                trust_remote_code = bool(call.get("trust_remote_code")),
                local_files_only = call.get("local_files_only", False),
                cache_dir = call.get("cache_dir", None),
                revision = revision,
            )

        original, name = name, mapped(name, revision)
        if name.lower() != original.lower():
            revision = None  # a remapped repo does not carry the caller's ref
        adapter = []
        base = _peft_base(name, call, revision)
        if base is not None:
            # Before the base reaches a filesystem probe: a Hub adapter's UNC base, as the loaders refuse it.
            from types import SimpleNamespace

            from .models.loader import _refuse_network_share_adapter_base

            _refuse_network_share_adapter_base(name, SimpleNamespace(base_model_name_or_path = base))
            # A PEFT adapter: the loaders load peft_config.base_model_name_or_path (mapped, no ref) and the adapter.
            adapter = _checkpoint_tensors(name, token = call.get("token"), revision = revision)
            name, revision = mapped(base, None), None
        tensors = _checkpoint_tensors(
            name, token = call.get("token"), revision = revision,
            subfolder = "" if base is not None else call.get("subfolder") or "",
        )  # fmt: skip
        from .models.mapper import bf16_repo_loads_16bit

        # The loader's own rule, on the mapped name as the loader applies it: a -bf16 Hub repo turns 4bit / 8bit off.
        load_in_16bit = bool(call.get("load_in_16bit")) or bf16_repo_loads_16bit(
            name, call.get("load_in_16bit")
        )
        model_types = _model_types(name, call, revision)
        joined = ",".join(model_types) + ","
        if loaded == "float16" and any(
            f in joined or any(f == t.replace("-", "").replace("_", "") for t in model_types)
            for f in force_float32
        ):
            raise ValueError(
                f"{model_types[0]} loads in a forced float32 mode on float16; not sized"
            )
        if full:
            # FastBaseModel.post_patch_model: fp32 master weights unless bf16 and float32_mixed_precision is not True.
            fp32_master = loaded != "bfloat16" or call.get("float32_mixed_precision") is True
            dtype_bytes = 4 if fp32_master else 2
        quantize = (load_in_4bit or load_in_8bit) and not load_in_16bit
        if full and any(dt in _PACKED for _, dt, _ in tensors):
            # _dequantize_bitsandbytes_for_full_finetuning turns packed weights dense before training.
            raise ValueError("full finetuning a pre-quantized checkpoint dequantizes it; not sized")
        weights, trainable = checkpoint_estimate(
            tensors,
            load_in_4bit = load_in_4bit and not load_in_16bit,
            load_in_8bit = load_in_8bit and not load_in_16bit,
            full_finetuning = full,
            dtype_bytes = dtype_bytes,
            skip_modules = _quant_skip_modules(model_types) if quantize else None,
            hidden_size = _hidden_size(name, call, revision),
        )
        if adapter:
            # The adapter's own weights (float32 trainable copies) replace the r = 16 guess.
            numel = sum(math.prod(shape) if shape else 1 for _, _, shape in adapter)
            weights += 4 * numel
            trainable = trainable if full else numel
        elif not full:
            weights += 4 * trainable  # the fresh adapter's float32 parameters
        state = training_state_bytes(trainable, grad_itemsize = dtype_bytes if full else 4)
        return weights, state, 0

    return _resolve(estimate, install_trainer_init = True)


def resolve_for_trainer(model, training_args = None):
    """Trainer.__init__ under UNSLOTH_PARALLEL=auto when no Unsloth load decided: size the built model."""
    if _STATE["choice"] is not None or not auto_requested() or model is None:
        return _STATE["choice"]

    def estimate():
        seen, weights, trainable, on_cuda = set(), 0, 0, 0
        for p in list(model.parameters()) + list(model.buffers()):
            if id(p) in seen:
                continue
            seen.add(id(p))
            weights += p.numel() * p.element_size()
            if getattr(p, "is_cuda", False):
                on_cuda += p.numel() * p.element_size()
            if getattr(p, "requires_grad", False):
                # Per parameter: mixed trainable dtypes each keep their own gradient and bucket width.
                trainable += training_state_bytes(
                    p.numel(), p.element_size(), getattr(training_args, "optim", None)
                )
        state = trainable
        # Already resident: on CPU it is out of the available RAM _budget reads (CUDA counts reserved).
        held = weights
        return weights, state, held, on_cuda

    return _resolve(estimate, install_trainer_init = False)

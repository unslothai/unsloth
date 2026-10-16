# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch, ubatch, parallel slot and split-mode arguments."""

import math
import os
import re
import struct

from typing import (
    Iterable,
    Mapping,
    Optional,
    Sequence,
)
from core.inference.llama_server_args import (
    _flag_name,
    _SPLIT_MODE_FLAGS,
    extra_args_disable_mmproj,
    extra_args_image_max_tokens,
    extra_args_mmproj_auto,
)

from .draft_args import _extra_args_device
from .launch_flags import (
    _DEFAULT_LLAMA_N_BATCH,
    _DEFAULT_LLAMA_N_UBATCH,
    _GEMMA4V_CAUSAL_TEXT_N_EMBD,
    _MMPROJ_NON_CAUSAL_IMAGE_TOKENS,
    _MMPROJ_UNKNOWN_UBATCH,
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
)
from .log import logger
from .memory_policy import _TENSOR_SPLIT_FLAGS


def _emitted_n_batch(n_batch: Optional[int], n_parallel: int) -> Optional[int]:
    """--batch-size the launch actually emits for a requested ``n_batch``.

    llama-server aborts below 2 and below the slot count, so the loader raises the
    flag to ``max(slots, 2)``. llama.cpp then derives the micro-batch from the
    raised value (``cparams.n_ubatch = min(cparams.n_batch, n_ubatch or n_batch)``),
    so every VRAM budget must be sized from it, not from the requested number, or a
    small batch with many slots is budgeted at a micro-batch the launch never uses.
    None (llama.cpp defaults) stays None: nothing is emitted, so nothing is raised.
    """
    if n_batch is None:
        return None
    return max(int(n_batch), max(2, int(n_parallel or 1)))


def _with_parallel_slots(
    cmd: Sequence[str], slots: int, *, n_batch: Optional[int], supports_kv_unified: bool
) -> list[str]:
    """``cmd`` relaunched at ``slots``: Unsloth's own --parallel, its batch floor, and the
    --kv-unified a multi-slot launch adds. Each is emitted before the extras, so the
    first occurrence is the one to rewrite and a hand-typed flag still last-wins."""
    out = [str(a) for a in cmd]
    at = out.index("--parallel")
    out[at + 1] = str(slots)
    if n_batch is not None and "--batch-size" in out:
        b = out.index("--batch-size")
        out[b + 1] = str(_emitted_n_batch(n_batch, slots))
    if slots > 1 and supports_kv_unified and "--kv-unified" not in out and "-kvu" not in out:
        out[at + 2 : at + 2] = ["--kv-unified"]
    return out


def _extra_args_split_mode(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Last-wins ``-sm`` / ``--split-mode`` value, lowercased, or None."""
    source_env = os.environ if env is None else env
    found: Optional[str] = None
    raw = source_env.get("LLAMA_ARG_SPLIT_MODE")
    if raw and str(raw).strip():
        found = str(raw).strip().lower()
    args = [str(a) for a in extra_args] if extra_args else []
    for i, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in _SPLIT_MODE_FLAGS:
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        if value.strip():
            found = value.strip().lower()
    return found


def _extra_args_tensor_split(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[list[float]]:
    """Last-wins ``-ts`` / ``--tensor-split`` proportions, or None.

    llama.cpp copies these into ``splits`` verbatim and prefix-sums them
    (llama-model.cpp:1436-1447), so they ARE the row-ownership weights that
    ``split_weights_per_device`` carries. Malformed parses to None rather than
    raising, which puts the load back on the decline path.

    The delimiter set is llama.cpp's own ``[,/]+`` (common/arg.cpp:2791-2804) and
    nothing else: ``-ts 3/1`` is exactly ``-ts 3,1`` to the child, while inventing
    extra delimiters is the same bug pointing the other way -- ``std::stof`` stops
    at the first non-numeric character, so ``-ts 3;1`` is ``[3, 0]``, not
    ``[3, 1]``.
    """
    source_env = os.environ if env is None else env
    raw: Optional[str] = None
    env_value = source_env.get("LLAMA_ARG_TENSOR_SPLIT")
    if env_value and str(env_value).strip():
        raw = str(env_value).strip()
    args = [str(a) for a in extra_args] if extra_args else []
    for i, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in _TENSOR_SPLIT_FLAGS:
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        if value.strip():
            raw = value.strip()
    if raw is None:
        return None
    try:
        parts = [
            struct.unpack("=f", struct.pack("=f", float(p)))[0]
            for p in re.split(r"[,/]+", raw)
            if p.strip()
        ]
    except (OverflowError, ValueError):
        return None
    # isfinite, not just >= 0: float() takes "nan"/"inf", and NaN passes BOTH the
    # other tests (every comparison against NaN is false, sum() of a NaN list is
    # NaN). The caller's int(round(p * scale)) would then raise ValueError /
    # OverflowError with nothing catching it, aborting a load llama.cpp itself
    # merely degenerates on. Rejecting here routes it to the unparseable decline.
    if not parts or any(not math.isfinite(p) or p < 0 for p in parts) or sum(parts) <= 0:
        return None
    try:
        running = 0.0
        for part in parts:
            running = struct.unpack("=f", struct.pack("=f", running + part))[0]
    except OverflowError:
        return None
    if not math.isfinite(running):
        return None
    return parts


def _extra_args_n_parallel(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Serving slots a pass-through overrides to, or None when nothing does.

    Same precedence as the launched command line: extras are appended last
    (llama_cpp.py emits Unsloth's --parallel first), and llama.cpp is last-wins,
    so an extras value beats both the emitted flag and the env. 0 is rejected at
    parse time on the server path, so it is not a value anyone gets.
    """
    source_env = os.environ if env is None else env
    found: Optional[int] = None
    raw = source_env.get("LLAMA_ARG_N_PARALLEL")
    if raw:
        try:
            found = int(str(raw).strip())
        except (TypeError, ValueError):
            pass
    args = [str(a) for a in extra_args] if extra_args else []
    for i, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in ("-np", "--parallel"):
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        try:
            found = int(value.strip())
        except (TypeError, ValueError):
            continue
    return found


def _named_batch_sizes(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_batch: Optional[int] = None,
    n_ubatch: Optional[int] = None,
) -> tuple[int, int, bool, bool]:
    """Resolve batch sizes and whether each was explicitly set.

    Precedence is environment, first-class fields, then extra arguments.
    """
    values = {
        "batch": _DEFAULT_LLAMA_N_BATCH,
        "ubatch": _DEFAULT_LLAMA_N_UBATCH,
    }
    named = {"batch": False, "ubatch": False}
    source_env = os.environ if env is None else env
    for key, env_name in (
        ("batch", "LLAMA_ARG_BATCH"),
        ("ubatch", "LLAMA_ARG_UBATCH"),
    ):
        raw = source_env.get(env_name)
        if raw:
            try:
                values[key] = int(raw)
                named[key] = True
            except (TypeError, ValueError):
                pass

    if n_batch is not None:
        values["batch"] = int(n_batch)
        named["batch"] = True
    if n_ubatch is not None:
        values["ubatch"] = int(n_ubatch)
        named["ubatch"] = True

    args = [str(a) for a in extra_args] if extra_args else []
    flags = {
        "-b": "batch",
        "--batch-size": "batch",
        "-ub": "ubatch",
        "--ubatch-size": "ubatch",
    }
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        key = flags.get(flag)
        if key is None:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        try:
            values[key] = int(value)
            named[key] = True
        except (TypeError, ValueError):
            continue
    return values["batch"], values["ubatch"], named["batch"], named["ubatch"]


def _extra_args_n_ubatch(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_ctx: Optional[int] = None,
    *,
    n_batch: Optional[int] = None,
    n_ubatch: Optional[int] = None,
) -> Optional[int]:
    """Effective ubatch after llama.cpp normalizes it, or None at defaults."""
    _batch, _ubatch, _batch_named, _ubatch_named = _named_batch_sizes(
        extra_args, env, n_batch, n_ubatch
    )
    values = {"batch": _batch, "ubatch": _ubatch}
    if not (_batch_named or _ubatch_named):
        return None

    # common_params stores signed values, then llama_context_params converts
    # them to uint32_t. A zero ubatch means "use batch"; the context then caps
    # ubatch at batch size.
    batch = values["batch"] & 0xFFFFFFFF
    raw_ubatch = values["ubatch"]
    ubatch = batch if raw_ubatch == 0 else raw_ubatch & 0xFFFFFFFF
    effective = min(batch, ubatch)
    if n_ctx is not None and n_ctx > 0:
        effective = min(effective, n_ctx)
    return effective


def _read_gguf_embedding_length(path: Optional[str]) -> Optional[int]:
    """``{arch}.embedding_length``, cached, or None. Thin import-local wrapper."""
    if not path:
        return None
    try:
        from utils.models.gguf_metadata import read_gguf_embedding_length
        return read_gguf_embedding_length(str(path))
    except Exception as e:
        logger.debug(f"embedding length read failed: {e}")
        return None


def _unknown_projector_ubatch(
    extra_args: Optional[Iterable[str]] = None, env: Optional[Mapping[str, str]] = None
) -> int:
    """Return conservative headroom for an unclassified projector."""
    return max(_MMPROJ_UNKNOWN_UBATCH, extra_args_image_max_tokens(extra_args, env) or 0)


def _mmproj_required_ubatch(
    mmproj_path: Optional[str],
    n_embd_text: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the required micro-batch, or 0 when the default is sufficient.

    Inspect the image tower because audio-only projectors may still be marked as vision.
    """
    if not mmproj_path:
        return 0
    try:
        from utils.models.gguf_metadata import (
            mmproj_accepts_image,
            read_mmproj_vision_projector_type,
        )
        if not mmproj_accepts_image(mmproj_path):
            return 0
        family = (read_mmproj_vision_projector_type(mmproj_path) or "").strip().lower()
    except Exception as e:
        logger.debug(f"mmproj capability read failed: {e}")
        family = ""
    custom = extra_args_image_max_tokens(extra_args, env) or 0
    if not family:
        return _unknown_projector_ubatch(extra_args, env)
    if family not in _MMPROJ_NON_CAUSAL_IMAGE_TOKENS:
        return 0
    if family == "gemma4v" and n_embd_text in _GEMMA4V_CAUSAL_TEXT_N_EMBD:
        return 0
    # A custom image limit can exceed the family's built-in ceiling.
    ceiling = max(_MMPROJ_NON_CAUSAL_IMAGE_TOKENS[family], custom)
    return ceiling if ceiling > _DEFAULT_LLAMA_N_UBATCH else 0


def _launch_required_ubatch(
    mmproj_path: Optional[str],
    n_embd_text: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
    *,
    is_vision: bool = True,
    vision_off: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the largest micro-batch required by any possible launch projector.

    Taking the maximum avoids duplicating llama.cpp's projector precedence and only
    over-reserves when multiple image towers are supplied.
    """
    required = 0

    # Pass-through arguments override managed projector flags.
    override = _extra_args_device(extra_args, {"--mmproj", "-mm"})
    if override and not vision_off and not extra_args_disable_mmproj(extra_args):
        required = max(
            required, _mmproj_required_ubatch(str(override), n_embd_text, extra_args, env)
        )

    if not vision_off:
        # Inherited projectors survive --no-mmproj but not the vision switch.
        source_env = os.environ if env is None else env
        if (source_env.get("LLAMA_ARG_MMPROJ_URL") or "").strip():
            required = max(required, _unknown_projector_ubatch(extra_args, env))
        inherited = (source_env.get("LLAMA_ARG_MMPROJ") or "").strip()
        if inherited:
            required = max(
                required, _mmproj_required_ubatch(inherited, n_embd_text, extra_args, env)
            )

    if is_vision and not vision_off and not extra_args_disable_mmproj(extra_args):
        if mmproj_path:
            required = max(
                required, _mmproj_required_ubatch(str(mmproj_path), n_embd_text, extra_args, env)
            )

    if not vision_off and extra_args_mmproj_auto(extra_args, env):
        # llama-server's adjacent-projector search can disagree with ours.
        required = max(
            required,
            _mmproj_required_ubatch(str(mmproj_path), n_embd_text, extra_args, env)
            if mmproj_path
            else _unknown_projector_ubatch(extra_args, env),
        )

    return required


def _batch_ubatch_for_mmproj(
    required_ubatch: int,
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Raise an unset micro-batch, capped by the effective batch size."""
    if required_ubatch <= 0:
        return n_batch, n_ubatch
    batch, ubatch, batch_named, ubatch_named = _named_batch_sizes(
        extra_args, env, n_batch, n_ubatch
    )
    if ubatch_named:
        return n_batch, n_ubatch
    # Match llama_context_params, which converts the signed batch to uint32_t.
    batch &= 0xFFFFFFFF
    target = min(required_ubatch, batch)
    if target <= ubatch:
        return n_batch, n_ubatch
    return n_batch, target


def _embedding_batch_ubatch(
    n_ctx: int,
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Size an unset batch pair to the context for pooling that needs one micro-batch."""
    _, _, batch_named, ubatch_named = _named_batch_sizes(extra_args, env, n_batch, n_ubatch)
    if (batch_named and ubatch_named) or n_ctx <= _DEFAULT_LLAMA_N_UBATCH:
        return n_batch, n_ubatch
    # llama.cpp caps the micro-batch at the batch, so the unset side must grow too.
    return (n_batch if batch_named else n_ctx), (n_ubatch if ubatch_named else n_ctx)


# Micro-batch when MoE experts sit in host RAM. Prompt processing copies each
# layer's routed experts over PCIe once per micro-batch (ggml op offload), so 4x
# larger means 4x fewer copies. Measured on T4 / L4 / A100 with Qwen3.6-35B-A3B,
# gemma-4-26B-A4B, Qwen3.8-Flash-Next and GLM-5.3-Flash: prompt +60% to +260%,
# decode unchanged. 4096 was slower overall: its bigger compute buffer pushed
# expert layers off the GPU and cost decode. llama.cpp's default batch (2048)
# already covers it.
_MOE_SPILL_N_UBATCH = 2048


def _moe_spill_batch_ubatch(
    n_batch: Optional[int],
    n_ubatch: Optional[int],
    *,
    n_moe_layers: int,
    experts_on_host: bool,
    discrete_gpu: bool,
    user_named_batch: bool,
) -> tuple[Optional[int], Optional[int]]:
    """Raise an unset micro-batch to ``_MOE_SPILL_N_UBATCH`` when experts spill.

    Only for an MoE whose experts are (or may be) in host RAM next to a discrete GPU,
    the transfer-bound prompt path. A batch or micro-batch the user set (first-class
    field, pass-through flag or LLAMA_ARG_BATCH / LLAMA_ARG_UBATCH) always wins, and
    a larger projector value already in ``n_ubatch`` is kept, so this only raises.
    Read ``user_named_batch`` BEFORE the projector and embedding raises set ``n_ubatch``.
    """
    if user_named_batch or not experts_on_host or not discrete_gpu or n_moe_layers <= 0:
        return n_batch, n_ubatch
    current = _DEFAULT_LLAMA_N_UBATCH if n_ubatch is None else int(n_ubatch)
    if current >= _MOE_SPILL_N_UBATCH:
        return n_batch, n_ubatch
    # llama.cpp caps the micro-batch at the batch, so keep batch >= micro-batch.
    batch = _DEFAULT_LLAMA_N_BATCH if n_batch is None else int(n_batch)
    if batch < _MOE_SPILL_N_UBATCH:
        n_batch = _MOE_SPILL_N_UBATCH
    return n_batch, _MOE_SPILL_N_UBATCH


def _override_targets_host(value: str) -> bool:
    """Whether an ``--override-tensor`` value sends any tensor to host memory
    (``CPU``, ``CPU_REPACK``, ``CUDA_Host``, ...), not only to GPUs."""
    targets = [
        part.rsplit("=", 1)[-1].strip().lower() for part in str(value).split(",") if "=" in part
    ]
    return any(t.startswith("cpu") or t.endswith("_host") for t in targets)


def _expert_spill_places_tensors_on_cpu(
    extra_args: Optional[Iterable[str]] = None, env: Optional[Mapping[str, str]] = None
) -> bool:
    """Host placement from extras or env, for the expert-spill micro-batch raise.

    ``_args_place_tensors_on_cpu`` / ``_env_places_tensors_on_cpu`` count any ``-ot``,
    which suits their residency gates. Here an override counts only when it targets
    host memory: ``-ot exps=CUDA0`` keeps the weights on a GPU, so there is no copy to
    amortise, and on ``--fit off`` the larger compute buffer would go unpriced.
    """
    args = [str(a) for a in extra_args or ()]
    kept: list[str] = []
    i = 0
    while i < len(args):
        if _flag_name(args[i]) in {"-ot", "--override-tensor"}:
            _, eq, inline = args[i].partition("=")
            step = 1 if eq else 2
            if _override_targets_host(inline if eq else (args[i + 1] if i + 1 < len(args) else "")):
                kept.extend(args[i : i + step])
            i += step
            continue
        kept.append(args[i])
        i += 1
    source_env = dict(os.environ if env is None else env)
    if not _override_targets_host(source_env.get("LLAMA_ARG_OVERRIDE_TENSOR") or ""):
        source_env.pop("LLAMA_ARG_OVERRIDE_TENSOR", None)
    return _args_place_tensors_on_cpu(kept) or _env_places_tensors_on_cpu(source_env)

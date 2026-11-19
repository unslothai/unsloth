# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA ModelOpt FP8 diffusers transformers (per-tensor e4m3 W8A8), e.g. nvidia/Cosmos3-Nano-FP8 / -Edge-FP8.

Diffusers cannot load these without the nvidia-modelopt package. Runtimes per Linear: ``static`` / ``dynamic``
(W8A8 ``torch._scaled_mm``, tensorwise / rowwise activation scales, sm_89+), ``stored`` (fp8 codes widened per
layer), ``dequant`` (widened once at load). W8A8 follows the checkpoint's ``diffusion_step_policy``.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

FP8_E4M3_MAX = 448.0
# "static" (the checkpoint's calibrated input scale) or "dynamic" (per-row activation scales, Studio's fp8 runtime).
ACT_SCALE_ENV = "UNSLOTH_MODELOPT_FP8_ACT"
ACT_STATIC = "static"
ACT_DYNAMIC = "dynamic"
DEFAULT_ACT = ACT_STATIC
# Off: never take an fp8 GEMM for these checkpoints (weight-only on every card).
GEMM_ENV = "UNSLOTH_MODELOPT_FP8_GEMM"
# Off: ignore the checkpoint's diffusion_step_policy (W8A8 on every step and path; measurement only).
STEP_POLICY_ENV = "UNSLOTH_MODELOPT_FP8_STEP_POLICY"

RUNTIME_STATIC = "static"
RUNTIME_DYNAMIC = "dynamic"
RUNTIME_STORED = "stored"
RUNTIME_DEQUANT = "dequant"

MODELOPT_ATTR = "_unsloth_modelopt_fp8"

_QUANT_SUFFIXES = (
    ".weight_scale",
    ".input_scale",
    ".weight_quantizer._amax",
    ".weight_quantizer._scale",
    ".input_quantizer._amax",
)
_REQUIRED_SUFFIXES = (".weight_scale", ".input_scale")
# Calibration companions: consumed as a consistency check, never loaded.
_CALIBRATION_SUFFIXES = (
    ".weight_quantizer._amax",
    ".weight_quantizer._scale",
    ".input_quantizer._amax",
)
# Any other ModelOpt quantizer state (smoothquant / AWQ pre-scales, attention bmm quantizers, block scales) is a
# format this loader does not implement.
_FOREIGN_MARKERS = ("_quantizer.", "_pre_quant_scale", ".weight_scale_2", "bmm_quantizer")
_MAX_HEADER_BYTES = 256 * 1024 * 1024
_QUANT_CHUNK_ELEMENTS = 1 << 24
# Calibrated amax / 448 against the stored scale: ModelOpt writes the same float32 value; looser only for rounding.
_SCALE_RTOL = 1e-3
_DTYPE_BYTES = {
    "F8_E4M3": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
}


@dataclass
class ModelOptFp8Scan:
    """What a ModelOpt FP8 transformer folder holds; ``problems`` lists everything the loader refuses."""

    folder: str
    shards: tuple = ()
    # layer stem (no ``.weight``) -> (out_features, in_features)
    layers: dict = field(default_factory = dict)
    fp8_bytes: int = 0
    dense_bytes: int = 0
    ignore: tuple = ()
    quant_algo: str = ""
    producer: str = ""
    problems: list = field(default_factory = list)

    def resident_bytes(self, keep_fp8: bool) -> int:
        """Transformer bytes once loaded: fp8 codes kept (one byte each) or widened to bf16 (two)."""
        return int(self.dense_bytes + self.fp8_bytes * (1 if keep_fp8 else 2))


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _as_dict(raw: Any) -> Optional[dict]:
    """A ``quantization_config`` as written: a dict, or a JSON / Python-repr string of one."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip().startswith("{"):
        for parse in (json.loads, ast.literal_eval):
            try:
                value = parse(raw)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(value, dict):
                return value
    return None


def transformer_folder(path: Any) -> Optional[Path]:
    """The ``transformer/`` folder of a local diffusers pipeline folder (or the folder itself), else None."""
    if not path:
        return None
    root = Path(str(path)).expanduser()
    for candidate in (root / "transformer", root):
        if (candidate / "config.json").is_file():
            return candidate
    return None


def quantization_config(path: Any) -> Optional[dict]:
    """The transformer's ``quantization_config``, else the pipeline root's ``hf_quant_config.json``; None if neither."""
    folder = transformer_folder(path)
    if folder is None:
        return None
    config = _read_json(folder / "config.json") or {}
    found = _as_dict(config.get("quantization_config"))
    if found is None and folder.name == "transformer":
        found = _read_json(folder.parent / "hf_quant_config.json")
        if found is not None and isinstance(found.get("quantization"), dict):
            found = found["quantization"]
    return found


def is_modelopt_config(config: Optional[dict]) -> bool:
    if not isinstance(config, dict):
        return False
    producer = config.get("producer")
    name = producer.get("name") if isinstance(producer, dict) else None
    return (
        str(config.get("quant_method") or "").lower() == "modelopt"
        or str(name or "").lower() == "modelopt"
    )


def modelopt_problem(config: Optional[dict]) -> Optional[str]:
    """Why a ModelOpt ``quantization_config`` is not the per-tensor FP8 W8A8 format this loader runs, or None."""
    algo = str(config.get("quant_algo") or "").upper() if isinstance(config, dict) else ""
    if algo != "FP8":
        return (
            f"its ModelOpt quantization is {algo or 'undeclared'}; Unsloth loads only ModelOpt FP8 (per-tensor "
            "e4m3 weights and activations) checkpoints"
        )
    if config.get("weight_only"):
        return "it is a ModelOpt weight-only FP8 build; Unsloth loads only the W8A8 FP8_FP8 format"
    for key in ("block_quantize", "channel_quantize"):
        if config.get(key):
            return f"it declares ModelOpt {key} = {config.get(key)!r}; only per-tensor FP8 scales are supported"
    return None


def _shard_files(folder: Path) -> list:
    index = _read_json(folder / "diffusion_pytorch_model.safetensors.index.json")
    if index is not None and isinstance(index.get("weight_map"), dict):
        names = sorted(set(index["weight_map"].values()))
        return [folder / n for n in names]
    return sorted(folder.glob("*.safetensors"))


def _read_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if not 0 < length <= _MAX_HEADER_BYTES:
            raise ValueError(f"{path.name}: implausible safetensors header")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def _ignored(name: str, ignore: tuple) -> bool:
    """ModelOpt ``ignore`` patterns: a leading module path, ``*`` a wildcard suffix (``time_embedder*``)."""
    for pattern in ignore:
        stem = pattern.rstrip("*")
        if not stem:
            continue
        if pattern.endswith("*"):
            if name.startswith(stem):
                return True
        elif name == stem or name.startswith(stem + "."):
            return True
    return False


def hub_modelopt_root(
    repo_id: Any,
    *,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> Optional[str]:
    """The cached snapshot root holding a Hub repo's ModelOpt transformer (its config read first, the shards fetched
    only when that config is ModelOpt), for a load that was handed a hub id rather than a staged folder. None for a
    local path, a repo without a ModelOpt transformer, or an unreachable Hub."""
    if not repo_id or Path(str(repo_id)).expanduser().exists():
        return None
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        kwargs = {
            "token": token,
            "cache_dir": cache_dir,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = (
            _read_json(Path(hf_hub_download(str(repo_id), "transformer/config.json", **kwargs)))
            or {}
        )
        if not is_modelopt_config(_as_dict(config.get("quantization_config"))):
            return None
        return snapshot_download(
            str(repo_id),
            allow_patterns = ["transformer/*.json", "transformer/*.safetensors"],
            **kwargs,
        )
    except Exception:  # noqa: BLE001 -- the stock load then reports what it cannot read
        return None


def scan_modelopt_fp8(path: Any) -> Optional[ModelOptFp8Scan]:
    """Header scan of a local ModelOpt transformer folder (torch-free, no weights read). None when ``path`` has no
    ModelOpt quantization config; a scan with ``problems`` when it has one this loader cannot run."""
    folder = transformer_folder(path)
    config = quantization_config(path)
    if folder is None or not is_modelopt_config(config):
        return None
    producer = config.get("producer") if isinstance(config.get("producer"), dict) else {}
    ignore = tuple(
        str(p) for p in (config.get("ignore") or config.get("modules_to_not_convert") or ())
    )
    scan = ModelOptFp8Scan(
        folder = str(folder),
        ignore = ignore,
        quant_algo = str(config.get("quant_algo") or ""),
        producer = f"{producer.get('name', 'modelopt')} {producer.get('version', '')}".strip(),
    )
    problem = modelopt_problem(config) or parse_step_policy(config)[1]
    if problem:
        scan.problems.append(problem)
        return scan
    shards = _shard_files(folder)
    if not shards or not all(p.is_file() for p in shards):
        scan.problems.append("its transformer safetensors shards are not all on disk")
        return scan
    scan.shards = tuple(str(p) for p in shards)
    header: dict = {}
    for shard in shards:
        try:
            header.update(_read_header(shard))
        except Exception as exc:  # noqa: BLE001 -- reported as a refusal, never a crash
            scan.problems.append(f"{shard.name}: unreadable header ({exc})")
            return scan
    stems = set()
    for key in header:
        for suffix in _QUANT_SUFFIXES:
            if key.endswith(suffix):
                stems.add(key[: -len(suffix)])
                break
        else:
            if any(m in key for m in _FOREIGN_MARKERS):
                scan.problems.append(f"{key}: ModelOpt state this loader does not implement")
    for stem in sorted(stems):
        weight = header.get(stem + ".weight") or {}
        shape = weight.get("shape") or []
        if weight.get("dtype") != "F8_E4M3" or len(shape) != 2:
            scan.problems.append(
                f"{stem}: quantizer state on a {weight.get('dtype') or 'missing'} weight of shape {shape} "
                "(expected a 2-D F8_E4M3 Linear weight)"
            )
            continue
        for suffix in _REQUIRED_SUFFIXES:
            entry = header.get(stem + suffix)
            if entry is None:
                scan.problems.append(f"{stem}{suffix} is missing")
            elif entry.get("dtype") != "F32" or _numel(entry.get("shape")) != 1:
                scan.problems.append(
                    f"{stem}{suffix} is {entry.get('dtype')} {entry.get('shape')}; only a per-tensor float32 "
                    "scale is supported"
                )
        if _ignored(stem, ignore):
            scan.problems.append(f"{stem} is quantized but its quantization config excludes it")
        scan.layers[stem] = (int(shape[0]), int(shape[1]))
    for key, entry in header.items():
        size = _numel(entry.get("shape")) * _DTYPE_BYTES.get(str(entry.get("dtype")), 2)
        if entry.get("dtype") == "F8_E4M3":
            if key[: -len(".weight")] not in scan.layers or not key.endswith(".weight"):
                scan.problems.append(f"{key}: an fp8 tensor without its per-tensor scales")
            scan.fp8_bytes += size
        elif not any(key.endswith(s) for s in _QUANT_SUFFIXES):
            # dense tensors load at bf16 (2 bytes) whatever their stored width, as from_pretrained casts them
            scan.dense_bytes += _numel(entry.get("shape")) * 2
    if not scan.layers and not scan.problems:
        scan.problems.append("its ModelOpt config declares FP8 but no quantized Linear was found")
    return scan


def _numel(shape: Any) -> int:
    n = 1
    for dim in shape or ():
        n *= int(dim)
    return n


def modelopt_fp8_error(scan: Optional[ModelOptFp8Scan], name: str = "") -> Optional[str]:
    if scan is None or not scan.problems:
        return None
    shown = "; ".join(scan.problems[:3])
    more = f" (and {len(scan.problems) - 3} more)" if len(scan.problems) > 3 else ""
    return f"{name or 'This checkpoint'} cannot be loaded as an NVIDIA ModelOpt FP8 checkpoint: {shown}{more}."


def act_mode() -> str:
    raw = (os.environ.get(ACT_SCALE_ENV) or "").strip().lower()
    return raw if raw in (ACT_STATIC, ACT_DYNAMIC) else DEFAULT_ACT


def gemm_disabled() -> bool:
    return (os.environ.get(GEMM_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def step_policy_disabled() -> bool:
    return (os.environ.get(STEP_POLICY_ENV) or "").strip().lower() in ("0", "off", "false", "no")


# device index -> {"static": bool, "dynamic": bool}
_GEMM_PROBE: dict = {}


def _probe_gemms(index: int) -> dict:
    import torch

    found = {ACT_STATIC: False, ACT_DYNAMIC: False}
    if torch.cuda.get_device_capability(index) < (8, 9) or not hasattr(torch, "_scaled_mm"):
        return found
    dev = f"cuda:{index}"
    a = torch.ones((16, 32), device = dev).to(torch.float8_e4m3fn)
    b = torch.ones((16, 32), device = dev).to(torch.float8_e4m3fn)
    try:
        one = torch.ones((), device = dev, dtype = torch.float32)
        out = torch._scaled_mm(a, b.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        found[ACT_STATIC] = bool(torch.isfinite(out).all().item()) and float(out[0, 0]) == 32.0
    except Exception:  # noqa: BLE001 -- no tensorwise fp8 GEMM on this build / card
        pass
    try:
        rows = torch.ones((16, 1), device = dev, dtype = torch.float32)
        cols = torch.ones((1, 16), device = dev, dtype = torch.float32)
        out = torch._scaled_mm(a, b.t(), scale_a = rows, scale_b = cols, out_dtype = torch.bfloat16)
        found[ACT_DYNAMIC] = bool(torch.isfinite(out).all().item()) and float(out[0, 0]) == 32.0
    except Exception:  # noqa: BLE001 -- no rowwise fp8 GEMM on this build / card
        pass
    return found


def fp8_gemm_runs(
    target: Any = None,
    device: Any = None,
    act: Optional[str] = None,
) -> bool:
    """Whether the W8A8 ``torch._scaled_mm`` (tensorwise for ``static``, rowwise for ``dynamic``) runs on the target
    card: CUDA (not ROCm, whose fp8 is the fnuz encoding), sm_89 or newer, bfloat16 compute, and a one-off probe GEMM
    returns the exact answer."""
    if gemm_disabled():
        return False
    act = act or act_mode()
    try:
        import torch

        if getattr(target, "dtype", torch.bfloat16) is not torch.bfloat16:
            return False
        if getattr(target, "device", "cuda") != "cuda" or not torch.cuda.is_available():
            return False
        from .rocm_bf16 import is_rocm_torch

        if is_rocm_torch(torch):
            return False
        dev = torch.device(device or getattr(target, "torch_device", None) or "cuda")
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if index not in _GEMM_PROBE:
            _GEMM_PROBE[index] = _probe_gemms(index)
        return bool(_GEMM_PROBE[index].get(act))
    except Exception:  # noqa: BLE001
        return False


def modelopt_runtime(
    target: Any,
    *,
    keep: bool = True,
    fp8_backend: Optional[str] = None,
    fp8_storage: bool = False,
) -> str:
    """The runtime every quantized Linear of this load takes. The W8A8 GEMM wherever it runs (plain buffers, so a
    streamed plan keeps it too); else what Studio's rules for pre-quantized fp8 layers pick on this card and plan:
    ``fp8_backend`` (a resident or native fp8 runtime) / ``fp8_storage`` keep the codes stored, otherwise they are
    dequantized. ``keep=False`` (the user pinned the bf16 DiT): dequantize."""
    if not keep:
        return RUNTIME_DEQUANT
    act = act_mode()
    if fp8_gemm_runs(target, act = act):
        return RUNTIME_DYNAMIC if act == ACT_DYNAMIC else RUNTIME_STATIC
    if act == ACT_DYNAMIC and fp8_gemm_runs(target, act = ACT_STATIC):
        return RUNTIME_STATIC
    if fp8_backend is not None or fp8_storage:
        return RUNTIME_STORED
    return RUNTIME_DEQUANT


def runtime_keeps_fp8(runtime: str) -> bool:
    return runtime in (RUNTIME_STATIC, RUNTIME_DYNAMIC, RUNTIME_STORED)


# Per diffusers ``Cosmos3AttnProcessor.__call__`` / ``Cosmos3DecoderLayer.forward``: to_q/k/v, to_out, mlp run the
# reasoner (text) tokens; add_*_proj, to_add_out, mlp_moe_gen the generation (video) tokens. diffusers 0.42.1's
# ``_classify_cosmos3_linear`` swaps the attention halves; do not copy it.
_GENERATION_SUFFIXES = (
    ".self_attn.add_q_proj",
    ".self_attn.add_k_proj",
    ".self_attn.add_v_proj",
    ".self_attn.to_add_out",
)
_REASONER_SUFFIXES = (
    ".self_attn.to_q",
    ".self_attn.to_k",
    ".self_attn.to_v",
    ".self_attn.to_out",
)


def linear_path(name: str) -> Optional[str]:
    """``generation`` / ``reasoner``: the token stream a Cosmos3 MoT Linear runs on (video / text), else None."""
    if ".mlp_moe_gen." in name or name.endswith(_GENERATION_SUFFIXES):
        return "generation"
    if ".mlp." in name or name.endswith(_REASONER_SUFFIXES):
        return "reasoner"
    return None


@dataclass
class StepPolicy:
    """The checkpoint's ``quantization_config.runtime.diffusion_step_policy`` (schema 1, ``first_last_n``): the
    generation path runs W8A16 (bf16 activations) for the first / last N denoising iterations and W8A8 between; the
    reasoner path runs W8A16 throughout when ``reasoner`` is ``a16``. The schedule NVIDIA validated for these
    checkpoints (diffusers' Cosmos3 mixed precision, vLLM-Omni)."""

    first_steps: int = 0
    last_steps: int = 0
    overlap_a16: bool = True
    reasoner_a16: bool = False
    generation_a16: bool = False
    steps_seen: int = 0

    def set_step(self, step_index: Any, num_steps: Any) -> None:
        try:
            i, n = int(step_index), int(num_steps)
        except (TypeError, ValueError):
            self.generation_a16 = False
            return
        self.steps_seen += 1
        if n <= 1 or not 0 <= i < n:
            self.generation_a16 = False
            return
        first, last = i < self.first_steps, i >= n - self.last_steps
        self.generation_a16 = self.overlap_a16 if (first and last) else (first or last)

    def high_precision(self, path: Optional[str]) -> bool:
        if path == "reasoner":
            return self.reasoner_a16
        if path == "generation":
            return self.generation_a16
        return False

    def compile_key(self) -> tuple:
        """What a compiled FP8 Linear guards on (the path each branch takes); a CFG replica shares this object, so
        its warm-call signatures join it (diffusion_multigpu._shared_state_key)."""
        return (bool(self.generation_a16), bool(self.reasoner_a16))


_POLICY_FIELDS = (
    "schema_version",
    "type",
    "index_space",
    "scope",
    "default_mode",
    "first_steps",
    "last_steps",
    "overlap",
    "reasoner",
)
_MODES = ("a16", "native")


def parse_step_policy(config: Optional[dict]) -> tuple[Optional[StepPolicy], Optional[str]]:
    """``(policy, problem)`` from a ModelOpt ``quantization_config``: no policy declared is ``(None, None)`` (native
    W8A8 every step, as diffusers runs a checkpoint without one); a malformed one is a problem."""
    runtime = (config or {}).get("runtime") if isinstance(config, dict) else None
    policy = runtime.get("diffusion_step_policy") if isinstance(runtime, dict) else None
    if policy is None:
        return None, None
    if not isinstance(policy, dict):
        return None, "its diffusion_step_policy is not a mapping"
    missing = [f for f in _POLICY_FIELDS if f not in policy]
    if missing:
        return None, f"its diffusion_step_policy lacks {', '.join(missing)}"
    if (
        policy["schema_version"] != 1
        or policy["type"] != "first_last_n"
        or policy["index_space"] != "denoising_loop_iteration"
        or policy["default_mode"] != "native"
    ):
        return None, (
            "its diffusion_step_policy (schema "
            f"{policy['schema_version']}, {policy['type']}, {policy['index_space']}, default {policy['default_mode']}) "
            "is not the first_last_n schedule this loader runs"
        )
    try:
        first, last = policy["first_steps"], policy["last_steps"]
        modes = (first["mode"], last["mode"], policy["overlap"], policy["reasoner"])
        if any(m not in _MODES for m in modes) or first["mode"] != "a16" or last["mode"] != "a16":
            raise ValueError(modes)
        return (
            StepPolicy(
                first_steps = max(0, int(first["count"])),
                last_steps = max(0, int(last["count"])),
                overlap_a16 = policy["overlap"] == "a16",
                reasoner_a16 = policy["reasoner"] == "a16",
            ),
            None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"its diffusion_step_policy has unsupported steps / modes ({exc})"


def install_step_policy(model: Any, policy: StepPolicy) -> None:
    """Follow the denoising iteration: the Cosmos3 pipeline opens ``transformer.cache_context(name, step_index=i,
    num_inference_steps=n)`` around every transformer call, so wrap it to switch the policy before the call."""
    import contextlib

    original = model.cache_context

    @contextlib.contextmanager
    def cache_context(name: Any, **kwargs: Any):
        if "step_index" in kwargs:
            policy.set_step(kwargs.get("step_index"), kwargs.get("num_inference_steps"))
        with original(name, **kwargs):
            yield

    model.cache_context = cache_context
    model._unsloth_modelopt_step_policy = policy


@functools.lru_cache(maxsize = None)
def fp8_linear_class():
    """``ModelOptFp8Linear``, defined on first use so this module imports torch-free."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ModelOptFp8Linear(nn.Module):
        """A ModelOpt FP8 Linear: e4m3 codes with one weight scale, run as W8A8 ``torch._scaled_mm`` with the
        checkpoint's static input scale (``act="static"``, ModelOpt's per-tensor fp8 GEMM) or per-row dynamic
        activation scales (``act="dynamic"``), and as W8A16 (bf16 activations on the widened weight) where the step
        policy asks for it. The codes and fp32 scales are integer-view buffers, so a module-wide ``.to(dtype)``
        (offload hooks, pipeline casts) never widens or rounds them, and the module streams like any other. Off sm_89+
        CUDA (CPU tests, a moved module) the same quantized activations run through a widened GEMM."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            codes: Any,
            weight_scale: float,
            input_scale: float,
            bias: Any = None,
            compute_dtype: Any = torch.bfloat16,
            fast_accum: bool = True,
            act: str = ACT_STATIC,
            path: Optional[str] = None,
            policy: Optional[StepPolicy] = None,
        ):
            super().__init__()
            if codes.dtype != torch.float8_e4m3fn or tuple(codes.shape) != (
                out_features,
                in_features,
            ):
                raise ValueError(
                    f"ModelOptFp8Linear needs float8_e4m3fn codes of shape {(out_features, in_features)}"
                )
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.compute_dtype = compute_dtype
            self.fast_accum = bool(fast_accum)
            self.act = act if act in (ACT_STATIC, ACT_DYNAMIC) else ACT_STATIC
            self.path = path
            self.policy = policy
            self.register_buffer("weight_q", codes.contiguous().view(torch.uint8))
            self.register_buffer(
                "weight_scale",
                torch.tensor([float(weight_scale)], dtype = torch.float32).view(torch.int32),
            )
            self.register_buffer(
                "input_scale",
                torch.tensor([float(input_scale)], dtype = torch.float32).view(torch.int32),
            )
            if bias is not None:
                bias = nn.Parameter(bias.detach().to(compute_dtype), requires_grad = False)
            self.bias = bias
            self.gemm_calls = 0
            self.a16_calls = 0

        def dequantized_weight(self, dtype: Any) -> Any:
            codes = self.weight_q.view(torch.float8_e4m3fn).to(torch.float32)
            return (codes * self.weight_scale.view(torch.float32)).to(dtype)

        def _quantize_static(self, x2: Any) -> Any:
            """e4m3 codes of ``x / input_scale``, rounded once from float32, in row chunks so the float32 copy stays
            small next to the activation."""
            scale = self.input_scale.view(torch.float32)
            out = torch.empty(x2.shape, dtype = torch.float8_e4m3fn, device = x2.device)
            step = max(1, _QUANT_CHUNK_ELEMENTS // max(1, self.in_features))
            for start in range(0, x2.shape[0], step):
                out[start : start + step] = (
                    x2[start : start + step].to(torch.float32) / scale
                ).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            return out

        def _quantize_dynamic(self, x2: Any) -> tuple:
            """Per-row e4m3 codes and their float32 scales (row amax / 448, floored at 1e-12)."""
            codes = torch.empty(x2.shape, dtype = torch.float8_e4m3fn, device = x2.device)
            scales = torch.empty((x2.shape[0], 1), dtype = torch.float32, device = x2.device)
            step = max(1, _QUANT_CHUNK_ELEMENTS // max(1, self.in_features))
            for start in range(0, x2.shape[0], step):
                chunk = x2[start : start + step].to(torch.float32)
                s = chunk.abs().amax(dim = 1, keepdim = True).clamp_(min = 1e-12) / FP8_E4M3_MAX
                scales[start : start + step] = s
                codes[start : start + step] = (chunk / s).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            return codes, scales

        def _gemm(self, x2: Any) -> bool:
            if not (x2.is_cuda and x2.shape[0] > 0 and self.weight_q.is_cuda):
                return False
            probe = _GEMM_PROBE.get(x2.device.index)
            if probe is None and not torch.compiler.is_compiling():
                fp8_gemm_runs(device = x2.device, act = self.act)
                probe = _GEMM_PROBE.get(x2.device.index)
            return bool((probe or {}).get(self.act))

        def forward(self, x: Any) -> Any:
            shape = x.shape
            out_dtype = (
                x.dtype
                if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                else self.compute_dtype
            )
            x2 = x.reshape(-1, self.in_features)
            bias = self.bias
            if self.policy is not None and self.policy.high_precision(self.path):
                # W8A16: the activation stays in its dtype, the weight widens (the step policy's a16 mode)
                if not torch.compiler.is_compiling():
                    self.a16_calls += 1
                # exactly diffusers' Cosmos3 W8A16 (pipelines/cosmos/mixed_precision.py ``_w8a16_linear``, the policy's
                # reference runtime): the scale in the activation dtype, the weight widened and scaled in it
                scale = self.weight_scale.view(torch.float32).to(out_dtype)
                weight = self.weight_q.view(torch.float8_e4m3fn).to(out_dtype) * scale
                out = F.linear(
                    x2.to(out_dtype), weight, None if bias is None else bias.to(out_dtype)
                )
                return out.reshape(*shape[:-1], self.out_features)
            s_w = self.weight_scale.view(torch.float32)
            if self.act == ACT_DYNAMIC:
                xq, s_x = self._quantize_dynamic(x2)
            else:
                xq, s_x = self._quantize_static(x2), self.input_scale.view(torch.float32)
            if self._gemm(x2):
                if not torch.compiler.is_compiling():
                    self.gemm_calls += 1
                rowwise = self.act == ACT_DYNAMIC
                out = torch._scaled_mm(
                    xq,
                    self.weight_q.view(torch.float8_e4m3fn).t(),
                    scale_a = s_x if rowwise else s_x.reshape(()),
                    scale_b = s_w.reshape(1, 1).expand(1, self.out_features).contiguous()
                    if rowwise
                    else s_w.reshape(()),
                    bias = None if bias is None else bias.to(torch.bfloat16),
                    out_dtype = torch.bfloat16,
                    use_fast_accum = self.fast_accum,
                )
                out = out.to(out_dtype)
            else:
                xd = (xq.to(torch.float32) * s_x).to(out_dtype)
                out = F.linear(
                    xd,
                    self.dequantized_weight(out_dtype),
                    None if bias is None else bias.to(out_dtype),
                )
            return out.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, e4m3 W8A8 ({self.act} "
                f"activation scales), path={self.path}"
            )

    return ModelOptFp8Linear


def fp8_gemm_calls(model: Any) -> dict:
    """Engagement counters of a ModelOpt FP8 model (eager forwards): ``torch._scaled_mm`` calls and W8A16 calls."""
    total = {"gemm": 0, "a16": 0}
    for module in getattr(model, "modules", lambda: ())():
        if (
            hasattr(module, "gemm_calls")
            and hasattr(module, "input_scale")
            and hasattr(module, "weight_q")
        ):
            total["gemm"] += int(module.gemm_calls)
            total["a16"] += int(getattr(module, "a16_calls", 0))
    return total


def _scale_check(stem: str, scale: float, amax: Optional[float], what: str) -> None:
    if amax is None:
        return
    expected = float(amax) / FP8_E4M3_MAX
    if not (scale > 0 and abs(scale - expected) <= _SCALE_RTOL * max(abs(expected), 1e-30)):
        raise ValueError(
            f"{stem}: {what} {scale:.6g} does not match its calibrated amax / 448 ({expected:.6g}); the "
            "checkpoint is not the ModelOpt per-tensor FP8 layout"
        )


def load_modelopt_fp8_transformer(
    transformer_cls: Any,
    path: Any,
    *,
    scan: Optional[ModelOptFp8Scan] = None,
    dtype: Any = None,
    runtime: str = RUNTIME_DEQUANT,
    fast_accum: Optional[bool] = None,
    logger: Any = None,
) -> Any:
    """Build ``transformer_cls`` from a ModelOpt FP8 folder on the CPU, every quantized Linear on ``runtime``
    (``modelopt_runtime``). Raises ``ValueError`` for a folder it must refuse."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from .diffusion_init_buffers import is_fp32_init_table
    from .diffusion_transformer_quant import TQ_FP8, _resolve_fast_accum, mark_source_precision

    dtype = dtype or torch.bfloat16
    scan = scan if scan is not None else scan_modelopt_fp8(path)
    if scan is None:
        raise ValueError(f"{path} has no ModelOpt quantization config")
    problem = modelopt_fp8_error(scan, Path(str(path)).name)
    if problem:
        raise ValueError(problem)
    if runtime in (RUNTIME_STATIC, RUNTIME_DYNAMIC) and dtype is not torch.bfloat16:
        runtime = RUNTIME_STORED
    fast = _resolve_fast_accum(fast_accum)
    folder = Path(scan.folder)
    config = transformer_cls.load_config(str(folder))
    policy, _ = parse_step_policy(_as_dict(dict(config).get("quantization_config")))
    if step_policy_disabled():
        policy = None
    # diffusers must not see the ModelOpt config: its quantizer table (and the Cosmos3 pipeline's own FP8 step policy)
    # would look for ModelOpt's restored modules
    config = {k: v for k, v in dict(config).items() if k != "quantization_config"}
    with init_empty_weights():
        model = transformer_cls.from_config(config)
    keep_fp32 = getattr(transformer_cls, "_keep_in_fp32_modules", None) or []
    if isinstance(keep_fp32, str):
        keep_fp32 = [keep_fp32]

    def _dense_dtype(key: str) -> Any:
        return torch.float32 if any(m in key.split(".") for m in keep_fp32) else dtype

    modules = dict(model.named_modules())
    state: dict = {}
    layers: dict = {}  # stem -> {"weight": codes, "weight_scale": float, ...}
    suffixes = (".weight", *_QUANT_SUFFIXES)
    for shard in scan.shards:
        with safe_open(shard, framework = "pt", device = "cpu") as handle:
            for key in handle.keys():
                stem = next(
                    (
                        key[: -len(s)]
                        for s in suffixes
                        if key.endswith(s) and key[: -len(s)] in scan.layers
                    ),
                    None,
                )
                tensor = handle.get_tensor(key)
                if stem is not None:
                    part = key[len(stem) + 1 :]
                    layers.setdefault(stem, {})[part] = (
                        tensor if part == "weight" else float(tensor.float().reshape(-1)[0])
                    )
                    continue
                if tensor.is_floating_point():
                    tensor = tensor.to(_dense_dtype(key))
                state[key] = tensor

    built = {RUNTIME_STATIC: 0, RUNTIME_DYNAMIC: 0, RUNTIME_STORED: 0, RUNTIME_DEQUANT: 0}
    replacements: dict = {}
    gemm_cls = fp8_linear_class() if runtime in (RUNTIME_STATIC, RUNTIME_DYNAMIC) else None
    native_cls = None
    if runtime == RUNTIME_STORED:
        from .diffusion_native_quant import native_linear_class
        native_cls = native_linear_class()
    for stem, parts in sorted(layers.items()):
        module = modules.get(stem)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"{stem}: quantized in the checkpoint but not a Linear of {transformer_cls.__name__}"
            )
        codes = parts.get("weight")
        if codes is None or "weight_scale" not in parts or "input_scale" not in parts:
            raise ValueError(f"{stem}: incomplete ModelOpt FP8 tensors")
        if tuple(codes.shape) != (module.out_features, module.in_features):
            raise ValueError(
                f"{stem}: fp8 weight {tuple(codes.shape)}, the model expects "
                f"{(module.out_features, module.in_features)}"
            )
        codes = codes.view(torch.float8_e4m3fn) if codes.dtype == torch.uint8 else codes
        w_scale, x_scale = float(parts["weight_scale"]), float(parts["input_scale"])
        _scale_check(stem, w_scale, parts.get("weight_quantizer._amax"), "weight_scale")
        _scale_check(stem, x_scale, parts.get("input_quantizer._amax"), "input_scale")
        if "weight_quantizer._scale" in parts:
            _scale_check(
                stem, w_scale, parts["weight_quantizer._scale"] * FP8_E4M3_MAX, "weight_scale"
            )
        bias = state.pop(stem + ".bias", None)
        if gemm_cls is not None:
            replacements[stem] = gemm_cls(
                module.in_features,
                module.out_features,
                codes,
                w_scale,
                x_scale,
                bias = bias,
                compute_dtype = dtype,
                fast_accum = fast,
                act = ACT_DYNAMIC if runtime == RUNTIME_DYNAMIC else ACT_STATIC,
                path = linear_path(stem),
                policy = policy,
            )
        elif runtime == RUNTIME_STORED:
            dense = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias = bias is not None,
                device = "meta",
                dtype = dtype,
            )
            if bias is not None:
                dense.bias = torch.nn.Parameter(bias.to(dtype), requires_grad = False)
            replacements[stem] = native_cls(
                dense,
                TQ_FP8,
                codes = codes,
                scale = torch.full((module.out_features, 1), w_scale, dtype = torch.float32),
            )
        else:
            state[stem + ".weight"] = (codes.to(torch.float32) * w_scale).to(dtype)
            if bias is not None:
                state[stem + ".bias"] = bias
        built[runtime] += 1
        del parts["weight"]

    for stem, layer in replacements.items():
        parent_name, _, leaf = stem.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, layer)
    model_keys = set(model.state_dict().keys())
    missing, unexpected = model.load_state_dict(state, strict = False, assign = True)
    missing = [k for k in missing if k.rpartition(".")[0] not in replacements]
    ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    if ignore:
        import re
        unexpected = [k for k in unexpected if not any(re.search(p, k) for p in ignore)]
    name = folder.parent.name if folder.name == "transformer" else folder.name
    if missing:
        raise ValueError(
            f"{name} is missing {len(missing)} weight(s) of {transformer_cls.__name__} (e.g. {missing[0]!r})"
        )
    if unexpected:
        raise ValueError(
            f"{name} holds {len(unexpected)} tensor(s) {transformer_cls.__name__} has no place for "
            f"(e.g. {unexpected[0]!r})"
        )
    if any(p.device.type == "meta" for p in model.parameters()) or any(
        b.device.type == "meta" for b in model.buffers()
    ):
        raise ValueError(
            f"{transformer_cls.__name__}: weights left unloaded after the ModelOpt FP8 load"
        )
    loaded = set(state) | {k for k in model_keys if k.rpartition(".")[0] in replacements}
    del state
    # __init__ buffers follow the compute dtype except fp32 tables, as from_pretrained leaves them
    for module_name, module in model.named_modules():
        if module_name in replacements:
            continue
        for buffer_name, buffer in list(module._buffers.items()):
            full = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if full in loaded or is_fp32_init_table(module, buffer_name, buffer):
                continue
            if buffer is not None and buffer.is_floating_point():
                want = _dense_dtype(full)
                if buffer.dtype != want:
                    module._buffers[buffer_name] = buffer.to(want)
    model.eval()
    if gemm_cls is not None and policy is not None and hasattr(model, "cache_context"):
        install_step_policy(model, policy)
    kept = built[RUNTIME_STATIC] + built[RUNTIME_DYNAMIC] + built[RUNTIME_STORED]
    info = {
        "runtime": runtime,
        "fp8": kept,
        "dequantized": built[RUNTIME_DEQUANT],
        "producer": scan.producer,
        "layers": len(layers),
        "step_policy": None
        if policy is None or gemm_cls is None
        else {
            "first_steps": policy.first_steps,
            "last_steps": policy.last_steps,
            "reasoner_a16": policy.reasoner_a16,
        },
    }
    try:
        setattr(model, MODELOPT_ATTR, info)
        if kept:
            # Studio's quant passes treat it as already quantized; fp8 storage counts as stored layers
            model._unsloth_runtime_quant = TQ_FP8
            if runtime == RUNTIME_STORED:
                from .diffusion_comfy_quant import FP8_STORED_ATTR
                setattr(model, FP8_STORED_ATTR, built[RUNTIME_STORED])
        else:
            # widened from fp8 on load: requantising would compound the loss
            mark_source_precision(model, "fp8")
    except Exception:  # noqa: BLE001 -- markers are best-effort
        pass
    if logger is not None:
        logger.info(
            "diffusion.modelopt_fp8: %s loaded from a %s FP8 checkpoint: %d quantized Linears as %s%s; "
            "unquantized modules (%s) at their stored precision",
            transformer_cls.__name__,
            scan.producer or "ModelOpt",
            len(layers),
            {
                RUNTIME_STATIC: "W8A8 fp8 GEMMs (static input scales)",
                RUNTIME_DYNAMIC: "W8A8 fp8 GEMMs (dynamic per-row activation scales)",
                RUNTIME_STORED: "fp8 weights widened per layer in the forward",
                RUNTIME_DEQUANT: f"{str(dtype).replace('torch.', '')} weights dequantized at load",
            }[runtime],
            ""
            if info["step_policy"] is None
            else (
                f", W8A16 for the first {policy.first_steps} / last {policy.last_steps} denoising steps"
                + (" and every reasoner-path call" if policy.reasoner_a16 else "")
                + " (the checkpoint's step policy)"
            ),
            ", ".join(scan.ignore) or "none",
        )
    return model

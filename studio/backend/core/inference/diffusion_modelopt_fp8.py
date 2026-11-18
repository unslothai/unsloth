# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA ModelOpt FP8 diffusers transformers (``quant_algo`` FP8, ModelOpt ``FP8_FP8``: e4m3 weights and
activations, one static scale per tensor), as published in nvidia/Cosmos3-Nano-FP8 and nvidia/Cosmos3-Edge-FP8.

Every quantized Linear stores an e4m3 ``weight`` and scalar ``weight_scale`` / ``input_scale`` (amax / 448), plus the
calibration state ModelOpt restores (``weight_quantizer._amax`` / ``._scale``, ``input_quantizer._amax``), which is
checked against the scales and dropped. Diffusers cannot load these folders itself: the transformer config carries a
``quantization_config`` whose ``quant_method`` resolves to its ModelOpt quantizer, which needs the nvidia-modelopt
package (not a Studio dependency) and ModelOpt's restore hooks.

Runtimes, per Linear (``modelopt_runtime``):
  * ``static``: W8A8 ``torch._scaled_mm`` with the checkpoint's static input scale (NVIDIA's ModelOpt per-tensor GEMM),
    plain buffers, so it is resident or streamed alike; sm_89+ CUDA with a bfloat16 pipeline.
  * ``dynamic``: Studio's own fp8 weight (torchao per-row ``Float8Tensor``, dynamic per-row activations) on a resident
    plan, the runtime Studio's fp8 quant of the bf16 checkpoint uses.
  * ``stored``: fp8 codes kept, widened per layer in the forward (weight-only; Studio's native fp8 Linear), for cards
    without an fp8 GEMM when the plan streams the DiT or the host is tight.
  * ``dequant``: widened to the compute dtype once at load (weight-only; exactly the checkpoint's weights).
Modules ModelOpt left unquantized keep their stored precision, cast as a stock ``from_pretrained`` would.
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
# float32 elements per activation-quantization chunk (64 MiB of scratch).
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
    problem = modelopt_problem(config)
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


_GEMM_PROBE: dict = {}


def fp8_gemm_runs(target: Any = None, device: Any = None) -> bool:
    """Whether the static W8A8 ``torch._scaled_mm`` runs on the target card: CUDA (not ROCm, whose fp8 is the fnuz
    encoding), sm_89 or newer, bfloat16 compute, and a one-off probe GEMM succeeds."""
    if gemm_disabled():
        return False
    try:
        import torch

        if getattr(target, "dtype", torch.bfloat16) is not torch.bfloat16:
            return False
        if getattr(target, "device", "cuda") != "cuda" or not torch.cuda.is_available():
            return False
        if getattr(torch.version, "hip", None):
            return False
        dev = torch.device(device or getattr(target, "torch_device", None) or "cuda")
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if index in _GEMM_PROBE:
            return _GEMM_PROBE[index]
        ok = False
        if torch.cuda.get_device_capability(index) >= (8, 9) and hasattr(torch, "_scaled_mm"):
            try:
                a = torch.ones((16, 32), device = f"cuda:{index}").to(torch.float8_e4m3fn)
                b = torch.ones((16, 32), device = f"cuda:{index}").to(torch.float8_e4m3fn)
                one = torch.ones((), device = f"cuda:{index}", dtype = torch.float32)
                out = torch._scaled_mm(a, b.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
                ok = bool(torch.isfinite(out).all().item()) and float(out[0, 0]) == 32.0
            except Exception:  # noqa: BLE001 -- no fp8 GEMM on this build / card
                ok = False
        _GEMM_PROBE[index] = ok
        return ok
    except Exception:  # noqa: BLE001
        return False


def modelopt_runtime(
    target: Any,
    *,
    keep: bool = True,
    fp8_backend: Optional[str] = None,
    fp8_storage: bool = False,
) -> str:
    """The runtime every quantized Linear of this load takes. ``fp8_backend`` / ``fp8_storage``: what Studio's rules
    for pre-quantized fp8 layers pick on this card and plan (the resident fp8 runtime, fp8 storage); ``keep=False``
    (the user pinned the bf16 DiT): dequantize."""
    if not keep:
        return RUNTIME_DEQUANT
    act = act_mode()
    if act == ACT_STATIC and fp8_gemm_runs(target):
        # plain buffers: resident or streamed, as the plan decides
        return RUNTIME_STATIC
    if fp8_backend == "torchao" and not gemm_disabled():
        return RUNTIME_DYNAMIC if act == ACT_DYNAMIC else RUNTIME_STATIC
    if fp8_backend == "native" or fp8_storage:
        return RUNTIME_STORED
    return RUNTIME_DEQUANT


def runtime_keeps_fp8(runtime: str) -> bool:
    return runtime in (RUNTIME_STATIC, RUNTIME_DYNAMIC, RUNTIME_STORED)


@functools.lru_cache(maxsize = None)
def static_linear_class():
    """``ModelOptFp8Linear``, defined on first use so this module imports torch-free."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ModelOptFp8Linear(nn.Module):
        """W8A8 e4m3 Linear with per-tensor static scales: ``y = (q(x / s_x) @ q_w.T) * s_x * s_w + b``, as
        ModelOpt's per-tensor fp8 GEMM. The fp8 codes and fp32 scales are integer-view buffers, so a module-wide
        ``.to(dtype)`` (offload hooks, pipeline casts) never widens or rounds them. Off sm_89+ CUDA (CPU tests, a
        moved module) the same quantized activations run through a widened GEMM."""

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

        def dequantized_weight(self, dtype: Any) -> Any:
            codes = self.weight_q.view(torch.float8_e4m3fn).to(torch.float32)
            return (codes * self.weight_scale.view(torch.float32)).to(dtype)

        def quantize_input(self, x2: Any) -> Any:
            """e4m3 codes of ``x / input_scale``, rounded once from float32 (no bf16 double rounding), in row chunks
            so the float32 copy stays small next to the activation."""
            scale = self.input_scale.view(torch.float32)
            out = torch.empty(x2.shape, dtype = torch.float8_e4m3fn, device = x2.device)
            step = max(1, _QUANT_CHUNK_ELEMENTS // max(1, self.in_features))
            for start in range(0, x2.shape[0], step):
                out[start : start + step] = (
                    x2[start : start + step].to(torch.float32) / scale
                ).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            return out

        def _gemm(self, x2: Any) -> bool:
            if not (x2.is_cuda and x2.shape[0] > 0 and self.weight_q.is_cuda):
                return False
            ok = _GEMM_PROBE.get(x2.device.index)
            if ok is None and not torch.compiler.is_compiling():
                ok = fp8_gemm_runs(device = x2.device)
            return bool(ok)

        def forward(self, x: Any) -> Any:
            shape = x.shape
            out_dtype = (
                x.dtype
                if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                else self.compute_dtype
            )
            x2 = x.reshape(-1, self.in_features)
            xq = self.quantize_input(x2)
            s_x = self.input_scale.view(torch.float32)
            s_w = self.weight_scale.view(torch.float32)
            bias = self.bias
            if self._gemm(x2):
                if not torch.compiler.is_compiling():
                    self.gemm_calls += 1
                out = torch._scaled_mm(
                    xq,
                    self.weight_q.view(torch.float8_e4m3fn).t(),
                    scale_a = s_x.reshape(()),
                    scale_b = s_w.reshape(()),
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
            return f"in_features={self.in_features}, out_features={self.out_features}, w8a8=e4m3 static per-tensor"

    return ModelOptFp8Linear


def static_gemm_calls(model: Any) -> int:
    """Engagement counter: ``torch._scaled_mm`` calls the static fp8 Linears of ``model`` made (eager forwards)."""
    total = 0
    for module in getattr(model, "modules", lambda: ())():
        calls = getattr(module, "gemm_calls", None)
        if (
            isinstance(calls, int)
            and hasattr(module, "input_scale")
            and hasattr(module, "weight_q")
        ):
            total += calls
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
    for shard in scan.shards:
        with safe_open(shard, framework = "pt", device = "cpu") as handle:
            for key in handle.keys():
                stem = next(
                    (
                        key[: -len(s)]
                        for s in (".weight", *_QUANT_SUFFIXES)
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
    static_cls = static_linear_class() if runtime == RUNTIME_STATIC else None
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
        if runtime == RUNTIME_STATIC:
            replacements[stem] = static_cls(
                module.in_features,
                module.out_features,
                codes,
                w_scale,
                x_scale,
                bias = bias,
                compute_dtype = dtype,
                fast_accum = fast,
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
        elif runtime == RUNTIME_DYNAMIC:
            from .diffusion_comfy_quant import _fp8_tensor

            weight = _fp8_tensor(
                stem + ".weight", codes, torch.tensor([w_scale], dtype = torch.float32), dtype, fast
            )
            if weight is None:
                raise ValueError(f"{stem}: this install cannot build Studio's fp8 weight (torchao)")
            state[stem + ".weight"] = weight
            if bias is not None:
                state[stem + ".bias"] = bias
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
    # buffers the replacement Linears created themselves are already real
    missing = [k for k in missing if k.rpartition(".")[0] not in replacements]
    ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    if ignore:
        import re
        unexpected = [k for k in unexpected if not any(re.search(p, k) for p in ignore)]
    if missing:
        raise ValueError(
            f"{folder.parent.name or folder.name} is missing {len(missing)} weight(s) of "
            f"{transformer_cls.__name__} (e.g. {missing[0]!r})"
        )
    if unexpected:
        raise ValueError(
            f"{folder.parent.name or folder.name} holds {len(unexpected)} tensor(s) {transformer_cls.__name__} has no "
            f"place for (e.g. {unexpected[0]!r})"
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
    kept = built[RUNTIME_STATIC] + built[RUNTIME_DYNAMIC] + built[RUNTIME_STORED]
    info = {
        "runtime": runtime,
        "act": act_mode() if runtime in (RUNTIME_STATIC, RUNTIME_DYNAMIC) else None,
        "fp8": kept,
        "dequantized": built[RUNTIME_DEQUANT],
        "producer": scan.producer,
        "layers": len(layers),
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
            "diffusion.modelopt_fp8: %s loaded from a %s FP8 checkpoint: %d quantized Linears as %s, "
            "unquantized modules (%s) at their stored precision",
            transformer_cls.__name__,
            scan.producer or "ModelOpt",
            len(layers),
            {
                RUNTIME_STATIC: "W8A8 fp8 GEMMs (static input scales)",
                RUNTIME_DYNAMIC: "W8A8 fp8 GEMMs (dynamic per-row activation scales, torchao)",
                RUNTIME_STORED: "fp8 weights widened per layer in the forward",
                RUNTIME_DEQUANT: f"{str(dtype).replace('torch.', '')} weights dequantized at load",
            }[runtime],
            ", ".join(scan.ignore) or "none",
        )
    return model

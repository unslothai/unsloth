# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention for DiT inference (backend ``unsloth_int8``): Q.K^T and P.V on the int8 tensor cores.

One call is four CUDA kernel launches (``diffusion_int8_attention_cuda``): the per-channel statistics of K and V, the
V quantiser, the Q and K quantiser, and the attention kernel. It takes the strided ``[B, S, H, D]`` views the strided
processors already hold and returns ``[B, S, H, D]``; nothing is copied to a heads-first layout.

How the values are quantised (ideas from the cited papers, written from scratch here):

* Q and K are rotated by one fixed orthogonal map (signs, then a Hadamard transform over the head dim, at most 128
  channels) before rounding, which leaves Q.K^T unchanged and evens out each row's use of the int8 range (QuaRot,
  arXiv 2404.00456).
* K is first shifted by a per-channel mean over a sub-sample of the keys: the softmax does not see a shift common to
  all keys (SageAttention, arXiv 2410.02367). Q has one scale per row; K has one scale per 32 keys and tensor-core
  accumulator lane (SageAttention2, arXiv 2411.10958), so scores are rescaled without a per-key load.
* The probabilities are 8-bit integers too, rescaled per key tile so every tile uses the full range, and the softmax
  denominator is summed over the rounded values (INT-FlashAttention, arXiv 2409.16997).
* V is centred and divided by a per-channel gain taken over every key (both undone on the output), with one scale per
  key tile. A head whose V has an attention sink (a few keys far from the rest in most channels: the first text
  tokens of a joint sequence) takes the range without the stats chunk that holds it, clips that chunk in place and
  adds the far-out parts back exactly in a launch after the kernel, so the sink no longer costs every other key its
  resolution (CUDA kernel; the Triton kernel keeps the range over every key). A launch plan none of whose first
  calls had a sink runs the kernels without that path from then on, so calls without sinks cost what they did; one
  with sinks keeps the path only where it still beats torch's attention on its own inputs (a race at that call).
  A seed renders the same image on the first render of a load as on the next: a plan whose sink-path calls were used
  in a render keeps that path for the load, and the verdict it raced decides later loads before their first clip.

The kernel never replaces attention on its own: a strided processor runs it only after ``install_int8_attention``
found the kernel accurate on this card and, unless it was asked for by name, more than 2% faster than what the
processor ran before. Calls it cannot take (masks, other head dims, other devices) run native attention.

``UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL``: ``0`` never engages it, ``1`` engages it wherever it is accurate without
the race, unset races it where INT8 attention is already wanted (an INT8 checkpoint) or it was requested by name.

Where the CUDA kernel cannot be built or loaded on an NVIDIA card (no NVRTC, as on a Windows install without the CUDA
runtime wheels, or a failed compile, module load or first launch), the same calls run the Triton kernel of
``diffusion_int8_attention_triton`` instead (int8 Q.K^T, 16-bit P.V), when Triton imports and ``TRITON_KERNEL_CONFIGS``
has the card. That path passes its own accuracy check, and it races the incumbent even where the kernel was asked for
by name (only the env switch ``1`` skips that race), so it stands in only where it is faster. ``UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL``: ``auto`` (default),
``cuda`` (never Triton on NVIDIA) or ``triton`` (skip the CUDA kernel).
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

INT8_ATTENTION_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"
IMPL_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL"
BACKEND = "unsloth_int8"
_OP = "unsloth_studio::int8_attention_nhd"
HEAD_DIMS = (64, 128)
RACE_MARGIN = 0.98
RACE_TOKENS = (4096, 1024)
# (large, small) race tokens per DiT class, replacing RACE_TOKENS: H3 video runs 15k-45k tokens and A100 lost at 4096
# but won at 27k, so a 4096-token race would wrongly reject it.
FAMILY_RACE_TOKENS: dict = {
    "MiniMaxH3Transformer3DModel": (16384, 8192),
}
# the accuracy every card must reach against fp32 attention on the probe before the kernel may run at all
MIN_COSINE = 0.999
MAX_REL_L1 = 0.05
SHARP_GAIN = 5.0
MIN_SHARP_COSINE = 0.99
LOG2E = 1.4426950408889634

# Per CUDA capability and head dim: keys per tile (bn), warps (wpc), 16-row query tiles per warp (nq), smem stages,
# qreg (2 = Q from global memory: sm75 has 64 KB smem per SM), flush_tail placement and an optional register cap.
# qkw / pvw / spread pick the order the products and the tile copies are issued in (see the kernel source): they
# move no output bit, so they are per card too.
# long_keys / long: overrides from that many keys on (select_config). A card runs one block per SM, so a call of a
# few thousand tokens is a handful of blocks per SM and the last round leaves SMs idle: smaller blocks waste less of
# it, and bigger ones copy each tile for more rows on long calls.
# Measured per card; a card missing here keeps native. One bn per head dim, so the output does not depend on the card.
KERNEL_CONFIGS: dict = {
    (7, 5): {
        128: {"bn": 128, "wpc": 8, "nq": 1, "stages": 1, "qreg": 2, "qkw": 1, "pvw": 8},
        64: {"bn": 128, "wpc": 4, "nq": 2, "stages": 1, "qreg": 2},
    },
    (8, 0): {
        128: {
            "bn": 128,
            "wpc": 4,
            "nq": 1,
            "stages": 2,
            "qreg": 1,
            "long_keys": 4608,
            "long": {"wpc": 8},
        },
        64: {"bn": 128, "wpc": 4, "nq": 2, "stages": 2, "qreg": 1},
    },
    (8, 6): {
        128: {"bn": 128, "wpc": 8, "nq": 1, "stages": 2, "qreg": 1},
        64: {"bn": 128, "wpc": 4, "nq": 2, "stages": 2, "qreg": 1},
    },
    (8, 9): {
        128: {"bn": 128, "wpc": 12, "nq": 1, "stages": 2, "max_registers": 168},
        64: {
            "bn": 128,
            "wpc": 4,
            "nq": 2,
            "stages": 2,
            "qreg": 1,
            "long_keys": 4608,
            "long": {"wpc": 8},
        },
    },
    (12, 0): {
        128: {
            "bn": 128,
            "wpc": 12,
            "nq": 1,
            "stages": 2,
            "max_registers": 168,
            "flush_tail": 1,
            "qkw": 1,
            "pvw": 8,
            "spread": 5,
        },
        64: {"bn": 128, "wpc": 4, "nq": 2, "stages": 2, "qreg": 1},
    },
}
# ROCm runs the Triton kernel: block shape and P.V mode per gfx arch and head dim; a missing arch keeps native.
ROCM_KERNEL_CONFIGS: dict = {
    "gfx1151": {
        128: {
            "bm": 64,
            "bn": 32,
            "warps": 4,
            "stages": 1,
            "pv": "f16_t",
            "kscale": "tile",
            "long_keys": 8000,
            "long": {"bm": 128, "warps": 8},
        },
        64: {"bm": 128, "bn": 32, "warps": 4, "stages": 1, "pv": "f16_t", "kscale": "tile"},
    },
}
# Triton stand-in when the CUDA kernel cannot load, tuned on sm80 / sm89 / sm120 (sm86 = sm80, fits its 99 KB smem).
# 16-bit P.V beat int8 P.V on every card; sm75 has no entry (Triton cannot compile its int8 dot).
_TRITON_SM80: dict = {
    128: {"bm": 128, "bn": 32, "warps": 4, "stages": 3, "pv": "half_strided", "kscale": "tile"},
    64: {"bm": 64, "bn": 64, "warps": 4, "stages": 2, "pv": "half_strided", "kscale": "tile"},
}
TRITON_KERNEL_CONFIGS: dict = {
    (8, 0): _TRITON_SM80,
    (8, 6): _TRITON_SM80,
    (8, 9): {
        128: {"bm": 128, "bn": 64, "warps": 4, "stages": 2, "pv": "half_strided", "kscale": "tile"},
        64: {"bm": 128, "bn": 64, "warps": 4, "stages": 3, "pv": "half_strided", "kscale": "tile"},
    },
    (12, 0): {
        128: {"bm": 128, "bn": 32, "warps": 4, "stages": 3, "pv": "half_strided", "kscale": "tile"},
        64: {"bm": 128, "bn": 64, "warps": 4, "stages": 3, "pv": "half_strided", "kscale": "tile"},
    },
}
ROT_GROUP = (
    128  # channels per Hadamard block of the Q / K rotation (the head dim where that is smaller)
)
K_MEAN_STEP = 8  # the K mean is taken over every 8th key ...
K_MEAN_KEYS = 512  # ... of sequences long enough to leave this many
V_RANGE_STEP = 1  # the V range is taken over every key
V_RANGE_CHUNKS = 16  # row chunks per (batch, head) of the V range pass

_LOCK = threading.Lock()
_VERDICTS: dict = {}
_BROKEN: set = set()
_VERIFIED: set = set()
# (index, head dim, dtype, impl) whose long variant passed its own check; elsewhere long calls run the short one.
_LONG_VERIFIED: set = set()
_LONG_CHECKS: dict = {}
# (index, head dim, impl) whose long variant failed in a real call
_LONG_BROKEN: set = set()
# long probe: queries and heads against long_keys + LONG_PROBE_EXTRA keys (a ragged last key tile)
LONG_PROBE_QUERIES = 1000
LONG_PROBE_HEADS = 2
LONG_PROBE_EXTRA = 100
_LOADED: dict = {}
# (device index, head dim) -> why the CUDA kernel cannot run there (NVRTC, compile, module load or launch error)
_CUDA_FAILED: dict = {}
STATS = {"int8": 0, "native": 0, "triton": 0}
_OUT_KIND = {"torch.float16": 0, "torch.bfloat16": 1, "torch.float32": 2}
_KINDS: dict = {}  # torch dtype -> _OUT_KIND, filled on first use (torch is imported lazily)
# Card verdicts and long / mid race floors persist per card + toolchain + these sources; sink race verdicts persist
# per DiT class + checkpoint too (none for a DiT of no known checkpoint). UNSLOTH_DIFFUSION_PROBE_CACHE=0 turns it off.
PERSIST_SOURCES = (
    "diffusion_int8_attention.py",
    "diffusion_int8_attention_cuda.py",
    "diffusion_int8_attention_triton.py",
    "diffusion_int8_attention_dispatch.py",
    "diffusion_qwenimage21_attn.py",
)
_PERSISTED: list = [None]
WARM_TOKENS = 128


def _library_versions() -> dict:
    """Library versions the races depend on beyond torch / triton / the driver."""
    import torch

    from .diffusion_int8_attention_cuda import nvrtc_version
    from .diffusion_probe_cache import _version

    try:
        nvrtc = list(nvrtc_version())
    except Exception:  # noqa: BLE001
        nvrtc = None
    return {
        "diffusers": _version("diffusers"),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "nvrtc": nvrtc,
    }


def persisted() -> Any:
    if _PERSISTED[0] is None:
        from .diffusion_probe_cache import Namespace
        _PERSISTED[0] = Namespace("int8_attention", PERSIST_SOURCES, extra = _library_versions)
    return _PERSISTED[0]


def card_name(index: int) -> str:
    return f"cuda:{int(index)}"


def _dtype_kinds() -> dict:
    import torch

    # filled in one step: a concurrent first call never sees a partial table
    _KINDS.update({getattr(torch, name.split(".", 1)[1]): kind for name, kind in _OUT_KIND.items()})
    return _KINDS


def _env_mode() -> str:
    """``off`` | ``auto`` | ``force`` from ``UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL``."""
    raw = (os.environ.get(INT8_ATTENTION_ENV) or "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


INT8_ATTENTION_CHOICES = ("auto", "on", "off")


def normalize_int8_attention(value: Any) -> Optional[str]:
    """Fold a request to ``auto`` / ``on`` / ``off`` (None if unset); ValueError otherwise."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"int8_attention must be one of {INT8_ATTENTION_CHOICES}, got {value!r}")
    text = value.strip().lower()
    if not text:
        return None
    if text not in INT8_ATTENTION_CHOICES:
        raise ValueError(f"int8_attention must be one of {INT8_ATTENTION_CHOICES}, got {value!r}")
    return text


@dataclass(frozen = True)
class Int8Policy:
    """Resolved once per load so later renders never re-read the env. Precedence: per-load on/off > env switch >
    family default."""

    request: Optional[str] = None
    env: str = "auto"

    @property
    def mode(self) -> str:
        if self.request == "on":
            return "force"
        if self.request == "off":
            return "off"
        return self.env

    @property
    def env_force(self) -> bool:
        """The env switch ``1`` decided this load (it alone skips the Triton stand-in's race)."""
        return self.request not in ("on", "off") and self.env == "force"

    @property
    def source(self) -> str:
        if self.request in ("on", "off"):
            return "load"
        return "env" if self.env != "auto" else "default"


def resolve_policy(override: Any = None) -> Int8Policy:
    """Invalid requests read as none here: the request models and ``begin_load`` already refuse them."""
    if isinstance(override, Int8Policy):
        return override
    try:
        request = normalize_int8_attention(override)
    except ValueError:
        request = None
    return Int8Policy(request = request, env = _env_mode())


def mode(override: Any = None) -> str:
    """``off`` | ``auto`` | ``force`` from the load's policy, else the env switch."""
    if override is None:
        return _env_mode()
    return resolve_policy(override).mode


def env_forced(override: Any = None) -> bool:
    return resolve_policy(override).env_force


def impl_setting() -> str:
    """``auto`` | ``cuda`` | ``triton`` from ``UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL`` (NVIDIA only: ROCm always runs
    Triton)."""
    raw = (os.environ.get(IMPL_ENV) or "auto").strip().lower()
    return raw if raw in ("cuda", "triton") else "auto"


def requested(value: Any) -> bool:
    """Whether a load's attention request names this backend."""
    return isinstance(value, str) and value.strip().lower() == BACKEND


def kernel_config(capability: Any, head_dim: int) -> Optional[dict]:
    try:
        table = KERNEL_CONFIGS.get((int(capability[0]), int(capability[1])))
    except Exception:  # noqa: BLE001
        return None
    found = (table or {}).get(int(head_dim))
    return dict(found) if found else None


def _is_rocm() -> bool:
    import torch

    from .rocm_bf16 import is_rocm_torch
    return bool(is_rocm_torch(torch))


def _gfx_arch(index: int) -> str:
    import torch

    from .rocm_bf16 import normalize_gfx_arch
    return normalize_gfx_arch(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))


def rocm_kernel_config(arch: Any, head_dim: int) -> Optional[dict]:
    found = (ROCM_KERNEL_CONFIGS.get(str(arch)) or {}).get(int(head_dim))
    return dict(found) if found else None


def triton_kernel_config(capability: Any, head_dim: int) -> Optional[dict]:
    """The Triton config for an NVIDIA capability and head dim, or None."""
    try:
        table = TRITON_KERNEL_CONFIGS.get((int(capability[0]), int(capability[1])))
    except Exception:  # noqa: BLE001
        return None
    found = (table or {}).get(int(head_dim))
    return dict(found) if found else None


@lru_cache(maxsize = 1)
def triton_available() -> bool:
    """Whether the Triton kernel module has Triton to compile with (torch's own on Linux, triton-windows on
    Windows)."""
    try:
        from . import diffusion_int8_attention_triton as triton_kernel

        if triton_kernel.triton is None:
            return False
        # NVIDIA Triton 3.3 cannot compile the Q / K rotation ('tt.join' op); a rewrite that does was 1.3-2.9x
        # slower than native attention (L4, A100), so the race would keep native anyway.
        try:
            version = tuple(int(x) for x in triton_kernel.triton.__version__.split(".")[:2])
        except Exception:  # noqa: BLE001 -- unparsable: read as a current Triton
            return True
        return not (version == (3, 3) and not triton_kernel._rocm_build())
    except Exception:  # noqa: BLE001
        return False


def _serving(index: int, head_dim: int) -> tuple:
    """(implementation, config) for this device and head dim (``implementation``, ``device_config``), the device
    facts read once."""
    import torch

    if _is_rocm():
        config = rocm_kernel_config(_gfx_arch(index), head_dim)
        return ("triton", config) if config is not None else (None, None)
    setting = impl_setting()
    capability = torch.cuda.get_device_capability(index)
    if setting != "triton" and (int(index), int(head_dim)) not in _CUDA_FAILED:
        config = kernel_config(capability, head_dim)
        if config is not None:
            return "cuda", config
    if setting == "cuda":
        return None, None
    config = triton_kernel_config(capability, head_dim)
    if config is not None and triton_available():
        return "triton", config
    return None, None


def implementation(index: int, head_dim: int) -> Optional[str]:
    """``cuda`` | ``triton`` | None (native attention): what serves this device and head dim. NVIDIA runs the CUDA
    kernel unless it failed to build, load or launch here (or ``IMPL=triton``); then the Triton kernel where Triton
    imports and the capability has a Triton config. ROCm runs Triton where its gfx arch has a config."""
    return _serving(index, head_dim)[0]


def device_config(
    index: int,
    head_dim: int,
    impl: Optional[str] = None,
) -> Optional[dict]:
    """The kernel config for this device and head dim on the implementation that serves it (``implementation``):
    the CUDA or the Triton table by capability, the ROCm table by gfx arch."""
    import torch

    if impl is None:
        return _serving(index, head_dim)[1]
    if _is_rocm():
        return rocm_kernel_config(_gfx_arch(index), head_dim)
    capability = torch.cuda.get_device_capability(index)
    if impl == "triton":
        return triton_kernel_config(capability, head_dim)
    return kernel_config(capability, head_dim)


def kernel_defines(head_dim: int, config: dict) -> dict:
    defines = {
        "HD": int(head_dim),
        "BN": int(config["bn"]),
        "NQ": int(config["nq"]),
        "WPC": int(config["wpc"]),
        "STAGES": int(config["stages"]),
        "QREG": int(config.get("qreg", 0)),
    }
    for name in ("flush", "flush_tail", "sm75", "qkw", "pvw", "spread"):
        if name in config:
            defines[name.upper()] = int(config[name])
    return defines


def select_config(config: dict, keys: int) -> dict:
    """The kernel config of a call with ``keys`` keys: ``config`` with its ``long`` overrides applied from
    ``long_keys`` keys on, and without those two entries (what is left names one compiled kernel)."""
    long_keys = config.get("long_keys")
    chosen = (
        {**config, **config.get("long", {})} if long_keys and keys >= int(long_keys) else config
    )
    return {key: value for key, value in chosen.items() if key not in ("long_keys", "long")}


def shared_bytes(head_dim: int, config: dict) -> int:
    bn, wpc, nq = int(config["bn"]), int(config["wpc"]), int(config["nq"])
    stage = bn * (head_dim + 16) + head_dim * (bn + 16)
    # qreg 2 reads the Q operands from global memory: no Q rows in shared memory
    q_rows = 0 if int(config.get("qreg", 0)) == 2 else wpc * 16 * nq * (head_dim + 16)
    return int(config["stages"]) * stage + q_rows


def _pad(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


def stats_plan(n_keys: int) -> tuple:
    """(kstep, vstep, nvc, vrows): the K mean is taken over every kstep-th key (at least K_MEAN_KEYS of them where
    there are that many: any mean is a valid shift, it only has to be representative), the V range over every
    vstep-th key in nvc row chunks of vrows keys. vstep is 1: a channel whose range is underestimated costs every
    other channel of the tile its resolution (16 text-refiner tokens with the range from every 8th key: rel L1 0.55
    against fp64)."""
    kstep = max(1, min(K_MEAN_STEP, n_keys // K_MEAN_KEYS))
    vstep = max(1, int(V_RANGE_STEP))
    nvc = max(1, min(V_RANGE_CHUNKS, -(-n_keys // (256 * vstep))))
    vrows = -(-(-(-n_keys // nvc)) // vstep) * vstep
    return kstep, vstep, -(-n_keys // vrows), vrows


class Operands(dict):
    """``quantise``'s output: the int8 operands and their scales as views of one allocation, made when asked for."""

    def __missing__(self, key: str) -> Any:
        import torch

        plan = self["plan"]
        if key == "f32":
            view = self["i8"][plan["f32_at"] :].view(torch.float32)
        elif key == "layout":
            view = plan["layout"]
        else:
            offset, shape = plan["layout"][key]
            base = self["i8"] if key in ("qi", "ki", "vt") else self["f32"]
            view = base[offset : offset + math.prod(shape)].view(shape)
            if key in ("rcnt", "ridx"):
                view = view.view(torch.int32)
        self[key] = view
        return view


def _strides_aligned(x: Any) -> bool:
    """Whether the strides of ``x`` let the quantisers' 16-byte loads read it: last-dim stride 1 and every row a
    whole number of 16 bytes from the first (the first one's own alignment is the caller's to check)."""
    size, stride = x.element_size(), x.stride()
    return stride[3] == 1 and all((stride[d] * size) % 16 == 0 for d in range(3) if x.shape[d] > 1)


def _aligned(x: Any) -> Any:
    """``x`` with last-dim stride 1 and every row 16-byte aligned (what the quantisers' vector loads need)."""
    import torch

    if x.data_ptr() % 16 == 0 and _strides_aligned(x):
        return x
    # a copy even where the view is already contiguous: a fresh allocation starts aligned
    return x.clone(memory_format = torch.contiguous_format)


def _device_index(device: Any) -> int:
    import torch
    return device.index if device.index is not None else torch.cuda.current_device()


def _functions(
    index: int,
    head_dim: int,
    config: dict,
    sinks: bool = False,
) -> tuple:
    """(attention kernel, quantiser kernels, shared bytes), with or without the sink path; compiled on first use."""
    config = select_config(
        config, 0
    )  # a config with long overrides names its kernel for short sequences
    key = (index, head_dim, tuple(sorted(config.items())), ROT_GROUP, bool(sinks))
    found = _LOADED.get(key)
    if found is None:
        import torch

        from . import diffusion_int8_attention_cuda as cuda_kernel

        capability = tuple(torch.cuda.get_device_capability(index))
        shared = shared_bytes(head_dim, config)
        group = min(int(ROT_GROUP), head_dim)
        extra = {"SINKS": 1} if sinks else {}
        found = (
            cuda_kernel.load(
                index,
                capability,
                {**kernel_defines(head_dim, config), **extra},
                config.get("max_registers"),
                shared,
            ),
            cuda_kernel.load_quantisers(
                index,
                capability,
                {"HD": head_dim, "BN": int(config["bn"]), "RGX": group.bit_length() - 5, **extra},
            ),
            shared,
        )
        _LOADED[key] = found
    return found


def _raw_stream(index: int) -> int:
    """The current stream of device ``index`` as the driver's handle (the capture stream under a CUDA graph)."""
    import torch

    get = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if get is not None:
        return get(index)
    return torch.cuda.current_stream(index).cuda_stream


# A call's launch arithmetic (padded lengths, operand layout, grids, every argument that is not a pointer), keyed
# by everything it depends on: a short call is mostly this glue, so it is done once per call signature.
_PLAN_LIMIT = 512
RESID_THREADS = 256  # NTR of the quantiser source
RESID_BLOCKS = 128
# A plan's first SINK_LEARN_CALLS calls learn whether a head clipped V (attention sink); it races at SINK_DECIDE_AT.
# Undecided, a plan runs the sink path, which a graph records. A verdict other than the sink path is taken only where
# no call whose output was used (not a thrown-away graph warm-up) clipped on the sink path: else the plan keeps it for
# the load, so the first render repeats, and the verdict, kept for the bucket and persisted, decides the next plans
# and loads at their first clipped call, before its attention runs (_learn_now).
SINK_LEARN_CALLS = 32
SINK_DECIDE_AT = 48
# A race verdict is shared per shape, DECIDE_BUCKET token bucket and sinks flag (never across sinks / no sinks).
DECIDE_BUCKET = 256
# Plain path wins only if its error vs torch is <= this x the sink path's (Qwen-Image-2.1 ~1.03, Z-Image-Turbo ~1.6).
PLAIN_ERROR_SLACK = 1.15
# MODE_MAIN: no sink path (non-SINK_DITS, probes). MODE_RACE: sink path raced vs torch (auto SINK_DITS route).
# MODE_KERNELS: forced SINK_DITS request, races only the INT8 paths, never torch.
MODE_MAIN, MODE_RACE, MODE_KERNELS = 0, 1, 2
SINK_DITS = frozenset(
    {
        "Flux2Transformer2DModel",
        "Flux3Transformer2DModel",
        "FluxTransformer2DModel",
        "HunyuanVideo15Transformer3DModel",
        "QwenImage21Transformer2DModel",
        "ZImageTransformer2DModel",
    }
)
_RACE = (
    threading.local()
)  # .path: the path _median_ms's timed calls take on this thread (None outside it)
GRID_LIMIT = 65535  # blocks along y in one launch


def _plans() -> dict:
    """The launch plans, kept with the loaded kernels they hold (a plan does not outlive them)."""
    found = _LOADED.get("plans")
    if found is None:
        found = _LOADED["plans"] = {}
    return found


def _decisions() -> dict:
    """Sink decisions per ``plan["decision"]`` key."""
    found = _LOADED.get("decisions")
    if found is None:
        found = _LOADED["decisions"] = {}
    return found


# Set once a CUDA graph records a sink-route call: no later race or route change, the graph replays what it recorded.
_GRAPHED = [False]
# (DiT class, checkpoint) of the sink routes installed last (note_sink_checkpoint): persisted sink verdicts key on it.
_SINK_DIT: list = [None]
# (card, head dim, raise_floor) per raced route; a torch verdict raises the floor so those calls leave the op
_ROUTES: list = []


def register_route(index: int, head_dim: int, raise_floor: Callable) -> None:
    _ROUTES.append((int(index), int(head_dim), raise_floor))


def _drop_routes(plan: dict) -> None:
    _, Sq, Sk, _, D = plan["shape"]
    floor = (min(Sq, Sk) // DECIDE_BUCKET + 1) * DECIDE_BUCKET
    restored = raised = 0
    for index, head_dim, raise_floor in list(_ROUTES):
        if index == plan["index"] and head_dim == D:
            if raise_floor(floor):
                restored += 1
            else:
                raised += 1
    if restored or raised:
        _log().info(
            "diffusion.int8_attention: torch attention beat the sink path on cuda:%d at %d tokens: %d attention "
            "modules back on their stock processor, %d below %d tokens stay on %s",
            plan["index"],
            min(Sq, Sk),
            restored,
            raised,
            floor,
            BACKEND,
        )


def _raise_processor_floor(proc: Any, module: Any = None) -> Callable:
    """A processor that replaced a stock one for the kernel (``_unsloth_stock``) gives the module that one back, so
    its calls compile as they did before the install; any other raises its token floor."""

    def raise_floor(tokens: int) -> bool:
        stock = getattr(proc, "_unsloth_stock", None)
        if stock is not None and module is not None and getattr(module, "processor", None) is proc:
            module.processor = stock
            return True
        proc._unsloth_int8_min_tokens = max(
            int(getattr(proc, "_unsloth_int8_min_tokens", 0)), tokens
        )
        return False

    return raise_floor


def _config_key(config: dict) -> tuple:
    """``config`` by value, hashable (its ``long`` overrides are a dict)."""
    long = config.get("long")
    if long is not None:
        config = {**config, "long": tuple(sorted(long.items()))}
    return tuple(sorted(config.items()))


def _plan_key(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    scale: Optional[float],
    index: int,
    mode: int = MODE_MAIN,
) -> tuple:
    return (
        index,
        _config_key(config),
        q.shape,
        k.shape[1],
        q.dtype,
        k.dtype,
        v.dtype,
        q.stride(),
        k.stride(),
        v.stride(),
        scale,
        (K_MEAN_STEP, K_MEAN_KEYS, V_RANGE_STEP, V_RANGE_CHUNKS, GRID_LIMIT),
        int(mode),
    )


def _plan(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    scale: Optional[float],
    index: int,
    mode: int = MODE_MAIN,
) -> dict:
    """The launch plan for aligned ``[B, S, H, D]`` q / k / v on device ``index``."""
    from .diffusion_int8_attention_cuda import float_bits

    key = _plan_key(q, k, v, config, scale, index, mode)
    plans = _plans()
    plan = plans.get(key)
    if plan is not None:
        return plan
    decision = (
        index,
        _config_key(config),
        q.shape[0],
        q.shape[2],
        q.shape[3],
        q.dtype,
        scale,
        q.shape[1] // DECIDE_BUCKET,
        k.shape[1] // DECIDE_BUCKET,
        int(mode),
    )
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    config = select_config(config, Sk)
    bn, wpc, nq = int(config["bn"]), int(config["wpc"]), int(config["nq"])
    fn0, kernels0, shared = _functions(index, D, config, sinks = False)
    fn, kernels = (
        _functions(index, D, config, sinks = True)[:2] if mode != MODE_MAIN else (None, None)
    )
    # log2(e) turns the kernel's 2^x into e^x; the rotation adds sqrt(group) in norm to Q and to K
    mul = (1.0 / math.sqrt(D) if scale is None else float(scale)) * LOG2E / min(int(ROT_GROUP), D)
    n_qpad = _pad(Sq, 16 * nq * wpc)
    n_kpad = _pad(Sk, bn)
    tiles = n_kpad // bn
    BH = B * H
    kstep, vstep, nvc, vrows = stats_plan(Sk)
    layout, at = {}, 0
    for name, shape in (
        ("qi", (BH, n_qpad, D)),
        ("ki", (BH, n_kpad, D)),
        ("vt", (BH, tiles, D, bn)),
    ):
        layout[name] = (at, shape)
        at += math.prod(shape)
    f32_at, at = _pad(at, 16), 0
    for name, shape in (
        ("qs", (BH, n_qpad)),
        ("ks", (BH, n_kpad // 32, 4)),
        ("kc", (BH, D)),
        ("vs", (BH, tiles)),
        ("vc", (BH, D)),
        ("vg", (BH, D)),
        ("vpart", (BH, nvc, 2, D)),
        ("vsink", (BH, nvc, 2)),
        ("rcnt", (BH,)),
        ("ridx", (BH, vrows)),
        ("rres", (BH, vrows, D)),
        ("rml", (BH, n_qpad, 2)),
    ):
        layout[name] = (at, shape)
        at += math.prod(shape)
    off = {name: layout[name][0] for name in ("qi", "ki", "vt")}
    off.update(
        {
            name: f32_at + 4 * layout[name][0]
            for name in (
                "qs",
                "ks",
                "kc",
                "vs",
                "vc",
                "vg",
                "vpart",
                "vsink",
                "rcnt",
                "ridx",
                "rres",
                "rml",
            )
        }
    )
    q_kind, k_kind, v_kind = (_OUT_KIND[str(x.dtype)] for x in (q, k, v))
    qst, kst, vst = q.stride(), k.stride(), v.stride()
    stats_threads = 256  # NTS of the quantiser source
    stats_jobs = -(-n_qpad // (stats_threads // (D // 16))) + 1 + nvc
    kv_threads = max(bn * D // 64, 2 * D)  # NTKV: a V tile, or runs of 32 keys
    kv_jobs = tiles + n_kpad // (kv_threads // (D // 16))
    out_stride = (Sq * H * D, H * D, D)

    def grids(jobs: int) -> tuple:
        """((BH, jobs of this launch), first job) per launch: a grid dimension holds 65535 blocks"""
        return tuple(((BH, min(GRID_LIMIT, jobs - at)), at) for at in range(0, jobs, GRID_LIMIT))

    plan = {
        "index": index,
        "config": config,
        "layout": layout,
        "off": off,
        "f32_at": f32_at,
        "bytes": f32_at + 4 * at,
        "shape": (B, Sq, Sk, H, D),
        "pads": (n_qpad, n_kpad),
        "kernels": (
            (
                kernels["unsloth_i8stats_q"],
                kernels["unsloth_i8quant_kv"],
                fn,
                kernels["unsloth_i8resid"],
            )
            if kernels is not None
            else None
        ),
        "mode": int(mode),
        "kernels0": (kernels0["unsloth_i8stats_q"], kernels0["unsloth_i8quant_kv"], fn0),
        "calls": 0,
        "sinks": None,
        "decision": decision,
        "checks": [],
        "shared": shared,
        "stats_grids": grids(stats_jobs),
        "stats_threads": stats_threads,
        "stats_tail": (
            Sq,
            n_qpad,
            Sk,
            H,
            nvc,
            vrows,
            vstep,
            kstep,
            *qst[:3],
            *kst[:3],
            *vst[:3],
            float_bits(mul),
            q_kind,
            k_kind,
            v_kind,
        ),
        "kv_grids": grids(kv_jobs + 1),
        "kv_grids0": grids(kv_jobs),
        "kv_threads": kv_threads,
        "kv_tail": (Sk, n_kpad, H, nvc, vrows, vstep, *kst[:3], *vst[:3], k_kind, v_kind),
        "kv_tail0": (Sk, n_kpad, H, nvc, *kst[:3], *vst[:3], k_kind, v_kind),
        "attn_grid": (n_qpad // (16 * nq * wpc), BH),
        "attn_threads": 32 * wpc,
        "attn_tail": (Sq, n_qpad, Sk, n_kpad, H, *out_stride, _OUT_KIND[str(v.dtype)]),
        "resid_grid": (BH, max(1, min(RESID_BLOCKS, -(-Sq // (RESID_THREADS // 32 * 4))))),
        "resid_tail": (Sq, n_qpad, n_kpad, H, vrows, *out_stride, _OUT_KIND[str(v.dtype)]),
        "out_shape": (B, Sq, H, D),
    }
    if len(plans) >= _PLAN_LIMIT:
        plans.clear()
    plans[key] = plan
    return plan


def forget_plans() -> None:
    """Drop launch plans, sink decisions and routes: a new pipeline decides again."""
    _plans().clear()
    _decisions().clear()
    _ROUTES.clear()
    _GRAPHED[0] = False


def _find_plan(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    scale: Optional[float],
    index: int,
    mode: int = MODE_MAIN,
) -> tuple:
    """(plan, q, k, v): the call's launch plan and its operands with aligned strides (copies only where needed)."""
    # a plan is only ever made for aligned strides: finding one leaves the first rows' alignment to check
    plan = None
    if not (q.data_ptr() | k.data_ptr() | v.data_ptr()) & 15:
        plan = _plans().get(_plan_key(q, k, v, config, scale, index, mode))
    if plan is None:
        q, k, v = _aligned(q), _aligned(k), _aligned(v)
        plan = _plan(q, k, v, config, scale, index, mode)
    return plan, q, k, v


def _launch_quantisers(
    plan: dict, q: Any, k: Any, v: Any, base: int, stream: int, sinks: bool
) -> None:
    qp, kp, vp = q.data_ptr(), k.data_ptr(), v.data_ptr()
    off = plan["off"]
    stats_q, quant_kv = plan["kernels" if sinks else "kernels0"][:2]
    kc, vpart, vsink = base + off["kc"], base + off["vpart"], base + off["vsink"]
    sink = (vsink,) if sinks else ()
    args = (qp, kp, vp, kc, vpart, *sink, base + off["qi"], base + off["qs"]) + plan["stats_tail"]
    for grid, job in plan["stats_grids"]:
        stats_q.launch(grid, plan["stats_threads"], args + (job,), 0, stream)
    tile_args = (
        base + off["ki"],
        base + off["ks"],
        base + off["vt"],
        base + off["vs"],
        base + off["vc"],
        base + off["vg"],
    )
    if sinks:
        args = (kp, vp, kc, vpart, vsink) + tile_args
        args += (base + off["rcnt"], base + off["ridx"], base + off["rres"]) + plan["kv_tail"]
    else:
        args = (kp, vp, kc, vpart) + tile_args + plan["kv_tail0"]
    for grid, job in plan["kv_grids" if sinks else "kv_grids0"]:
        quant_kv.launch(grid, plan["kv_threads"], args + (job,), 0, stream)


def _launch_attention(plan: dict, base: int, out_ptr: int, stream: int, sinks: bool) -> None:
    off = plan["off"]
    args = (
        base + off["qi"],
        base + off["ki"],
        base + off["vt"],
        base + off["qs"],
        base + off["ks"],
        base + off["vs"],
        base + off["vg"],
        base + off["vc"],
        out_ptr,
    ) + plan["attn_tail"]
    if not sinks:
        plan["kernels0"][2].launch(
            plan["attn_grid"], plan["attn_threads"], args, plan["shared"], stream
        )
        return
    args += (base + off["rcnt"], base + off["rml"])
    plan["kernels"][2].launch(plan["attn_grid"], plan["attn_threads"], args, plan["shared"], stream)
    args = (
        base + off["qi"],
        base + off["ki"],
        base + off["qs"],
        base + off["ks"],
        base + off["vg"],
        base + off["rcnt"],
        base + off["ridx"],
        base + off["rres"],
        base + off["rml"],
        out_ptr,
    ) + plan["resid_tail"]
    plan["kernels"][3].launch(plan["resid_grid"], RESID_THREADS, args, 0, stream)


def _choose_path(plan: dict, race: Optional[Callable] = None) -> tuple:
    """Returns (path, call index): sink path until SINK_DECIDE_AT, then dropped if the first
    SINK_LEARN_CALLS calls never clipped V; race keeps it or falls to torch; native is torch;
    CUDA graph capture keeps the decision"""
    import torch

    forced = getattr(_RACE, "path", None)
    if forced is not None:
        return forced, None
    if plan.get("mode", MODE_RACE) == MODE_MAIN:
        return False, None
    if torch.cuda.is_current_stream_capturing():
        _GRAPHED[0] = True
        return (True if plan["sinks"] is None else plan["sinks"]), None
    n = plan["calls"]
    plan["calls"] = n + 1
    if plan["sinks"] is None and n >= SINK_DECIDE_AT:
        # its event waits and the race's are prohibited while another thread records a graph: decide on a later call
        with _outside_capture() as safe:
            if safe:
                seen = bool(plan.get("seen"))
                for check in plan["checks"]:
                    check[
                        0
                    ].synchronize()  # done: recorded SINK_DECIDE_AT - SINK_LEARN_CALLS calls ago or more
                    if bool(check[1].any()):
                        seen = True
                        if len(check) > 2 and check[2]:
                            plan["live_seen"] = True
                plan["checks"] = []
                # the bucket's verdict for plans that, like this one, had sinks (or none: no race either way)
                key = plan["decision"] + (seen,) if "decision" in plan else None
                known = _known_verdict(plan) if seen else None
                if not seen:
                    plan["sinks"] = (
                        False  # the same bits as the sink path a graph may have recorded
                    )
                elif _GRAPHED[0]:
                    # after a capture: the path the graph recorded for an undecided plan, no race, no route change
                    plan["sinks"] = True
                else:
                    # new to this load (raced, or persisted by an earlier load): it moves routes; the bucket keeps it
                    fresh = key is not None and key not in _decisions()
                    # never decided: the race checks accuracy on this model's own inputs (_pick_path)
                    verdict = known if known is not None else (True if race is None else race())
                    if fresh:
                        _decisions()[key] = verdict
                        if known is None and race is not None:
                            _persist_sink_verdict(plan, verdict)
                    if verdict is not True and plan.get("live_seen"):
                        # a render already used this plan's sink-path outputs where V clipped: keep the path for the
                        # load (the next render repeats the seed); the verdict decides the next plans and loads
                        verdict = True
                    plan["sinks"] = verdict
                    if verdict == "native" and fresh:
                        _drop_routes(plan)
    return (True if plan["sinks"] is None else plan["sinks"]), n


def _outside_capture() -> Any:
    """``hold_off_capture``: True while no other thread records a CUDA graph."""
    from .diffusion_cuda_graph import hold_off_capture
    return hold_off_capture()


def _sink_race(
    plan: dict,
    q: Any,
    k: Any,
    v: Any,
    scale: Optional[float],
    index: int,
    forced: bool = False,
) -> Any:
    """Timed on this call: faster of sink or plain INT8 if it beats torch by RACE_MARGIN, else
    native; forced: INT8 paths only, torch's attention neither timed nor taken."""
    import torch

    stream = _raw_stream(index)

    def kernels(sinks: bool) -> Callable:
        def run() -> Any:
            i8 = torch.empty(plan["bytes"], dtype = torch.int8, device = q.device)
            out = torch.empty(plan["out_shape"], dtype = v.dtype, device = q.device)
            _launch_quantisers(plan, q, k, v, i8.data_ptr(), stream, sinks)
            _launch_attention(plan, i8.data_ptr(), out.data_ptr(), stream, sinks)
            return out

        return run

    def torch_attention() -> Any:
        return native_attention(q, k, v) if scale is None else native_attention(q, k, v, scale)

    fns = {"sinks": kernels(True), "plain": kernels(False)}
    if not forced:
        fns = {"incumbent": torch_attention, **fns}
    try:
        ref = torch_attention().float()
        norm = float(ref.norm())
        err = {
            name: float((fns[name]().float() - ref).norm()) / max(norm, 1e-30)
            for name in ("sinks", "plain")
        }
        ref = None
        ms = _median_ms(fns)
    except torch.OutOfMemoryError:
        return False if forced else "native"
    path = _pick_path(ms, err, forced)
    _log().info(
        "diffusion.int8_attention: attention sinks in %s keys x %s heads, sink path %.4f ms (error %.4g), without it "
        "%.4f ms (error %.4g), torch %.4f ms: %s",
        plan["shape"][2],
        plan["shape"][3],
        ms["sinks"],
        err["sinks"],
        ms["plain"],
        err["plain"],
        ms.get("incumbent", float("nan")),
        {True: "the sink path", False: "the kernels without it"}.get(
            path, "torch's attention from here on"
        ),
    )
    plan["sink_race"] = {name: round(value, 4) for name, value in ms.items()}
    plan["sink_error"] = {name: round(value, 6) for name, value in err.items()}
    return path


def _pick_path(
    ms: dict,
    err: dict,
    forced: bool = False,
) -> Any:
    """Verdict from median ms and errors: a path must beat torch attention by RACE_MARGIN else
    torch; a near tie keeps the old path; non-finite errors never win; forced never uses torch"""
    finite = {name: math.isfinite(err[name]) for name in ("sinks", "plain")}
    if not finite["sinks"]:
        order = ["plain"] if finite["plain"] else []
    elif finite["plain"] and err["plain"] <= PLAIN_ERROR_SLACK * err["sinks"]:
        order = (
            ["sinks", "plain"] if ms["sinks"] < RACE_MARGIN * ms["plain"] else ["plain", "sinks"]
        )
    else:
        order = ["sinks"]
    paths = {"sinks": True, "plain": False}
    if forced:
        return paths[order[0]] if order else False
    for name in order:
        if ms[name] < RACE_MARGIN * ms["incumbent"]:
            return paths[name]
    return "native"


def _check_sinks(plan: dict, i8: Any, n: Optional[int]) -> None:
    """For one of the plan's first SINK_LEARN_CALLS calls: its clipped-key counts (rcnt) copied to pinned host memory
    behind the quantisers, read when the plan decides (_choose_path); no wait here."""
    import torch

    if n is None or n >= SINK_LEARN_CALLS or plan["sinks"] is not None:
        return
    # an event record is prohibited while another thread records a graph: that call goes unread
    with _outside_capture() as safe:
        if not safe:
            return
        at, (BH,) = plan["off"]["rcnt"], plan["layout"]["rcnt"][1]
        host = torch.empty(BH, dtype = torch.int32, pin_memory = True)
        host.copy_(i8[at : at + 4 * BH].view(torch.int32), non_blocking = True)
        event = torch.cuda.Event()
        event.record()
        # live: the call's output is used (not a thrown-away side-stream warm-up of a CUDA graph capture)
        plan["checks"].append((event, host, not _discarded_call()))


def _discarded_call() -> bool:
    """Whether this call runs in the side-stream warm-ups of a CUDA graph capture, whose outputs are thrown away."""
    try:
        from .diffusion_cuda_graph import step_recording
        return bool(step_recording())
    except Exception:  # noqa: BLE001
        return False


def _sink_verdict_key(plan: dict) -> str:
    return "sinks:" + repr((_SINK_DIT[0],) + tuple(plan["decision"][1:]))


def _known_verdict(plan: dict) -> Any:
    """The race verdict of a plan with sinks: its bucket's in this load, else one an earlier load persisted for this
    checkpoint, card and bucket; None if neither."""
    if "decision" not in plan:
        return None
    known = _decisions().get(plan["decision"] + (True,))
    if known is None and _SINK_DIT[0] is not None:
        try:
            known = persisted().get(card_name(plan["index"]), _sink_verdict_key(plan))
        except Exception:  # noqa: BLE001
            known = None
    return known if known in (True, False, "native") else None


def _persist_sink_verdict(plan: dict, verdict: Any) -> None:
    """A race that ran to the end (an out-of-memory race sets no ``sink_race``) keeps its verdict for later loads."""
    if "decision" not in plan or verdict not in (True, False, "native") or _SINK_DIT[0] is None:
        return
    if "sink_race" not in plan:
        return
    try:
        persisted().put(card_name(plan["index"]), _sink_verdict_key(plan), verdict)
    except Exception:  # noqa: BLE001
        pass


def _learn_now(plan: dict, i8: Any, n: Optional[int]) -> Any:
    """For one of an undecided plan's first SINK_LEARN_CALLS calls, after its sink-path quantisers, where the bucket
    already has a verdict (an earlier plan of this load, or a persisted one): whether a head clipped V, read now (one
    wait on this call's quantisers, until the first clip). The first clipped call takes the verdict before its
    attention runs; the calls before it clipped nothing, so the sink path gave them the bits the kernels without it
    give. Returns the path this call takes (True: the sink path, as quantised; False: the kernels without it;
    "native"), or None to leave the call to _check_sinks and the decision at SINK_DECIDE_AT (no verdict yet, or
    another thread records a CUDA graph)."""
    import torch

    if n is None or n >= SINK_LEARN_CALLS or plan["sinks"] is not None:
        return True
    known = _known_verdict(plan)
    if known is None or _GRAPHED[0]:
        return None
    with _outside_capture() as safe:
        if not safe:
            return None
        at, (BH,) = plan["off"]["rcnt"], plan["layout"]["rcnt"][1]
        clipped = bool(i8[at : at + 4 * BH].view(torch.int32).any())
        if not clipped:
            if n == SINK_LEARN_CALLS - 1 and not plan["checks"]:
                plan["sinks"] = (
                    False  # the same bits as the sink path this and the earlier calls took
                )
            return True
        if known == "native" and (n or _bucket_ran(plan)):
            # earlier INT8 outputs of this bucket were used: torch's attention would change them on the next render
            known = True
        fresh = plan["decision"] + (True,) not in _decisions()
        _decisions()[plan["decision"] + (True,)] = known
        plan["sinks"] = known
        plan["checks"] = []
        if known == "native" and fresh:
            _drop_routes(plan)
    return known


def _bucket_ran(plan: dict) -> bool:
    """Whether another plan of ``plan``'s decision bucket already ran a call this load."""
    bucket = plan.get("decision")
    return any(
        other is not plan and other.get("decision") == bucket and other.get("calls", 0)
        for other in _plans().values()
        if isinstance(other, dict)
    )


def _quantise_on_device(plan: dict, q: Any, k: Any, v: Any, i8: Any) -> bool:
    """The quantiser launches of one call on the current device and stream (``quantise``: no race, a plan with sinks
    keeps the path); whether it took the sink path."""
    path, n = _choose_path(plan)
    sinks = path is not False
    _launch_quantisers(plan, q, k, v, i8.data_ptr(), _raw_stream(plan["index"]), sinks)
    if sinks:
        _check_sinks(plan, i8, n)
    return sinks


def quantise(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    scale: Optional[float] = None,
    mode: int = MODE_MAIN,
) -> dict:
    """The kernel's inputs from ``[B, S, H, D]`` q / k / v (any strides on the first three dims); ``mode`` as for
    ``int8_attention_call``."""
    import torch

    dev = q.device
    index = _device_index(dev)
    plan, q, k, v = _find_plan(q, k, v, config, scale, index, mode)
    i8 = torch.empty(plan["bytes"], dtype = torch.int8, device = dev)
    if torch.cuda.current_device() == index:
        sinks = _quantise_on_device(plan, q, k, v, i8)
    else:
        with torch.cuda.device(index):
            sinks = _quantise_on_device(plan, q, k, v, i8)
    return Operands(
        i8 = i8,
        plan = plan,
        shape = plan["shape"],
        pads = plan["pads"],
        dtype = v.dtype,
        config = plan["config"],
        sinks = sinks,
        inputs = (q, k, v),  # kept alive until the launches that read them are done
    )


def _cuda_attention(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    scale: Optional[float],
    index: int,
    mode: int = MODE_MAIN,
) -> Any:
    """``attend(quantise(q, k, v, config, scale))`` in one pass, on the path the plan takes (_choose_path, with the
    sink race): one plan lookup, one stream lookup, no operand views. The scratch buffer and any aligned copies are
    freed on return; the caching allocator hands their memory out again only in stream order, after the launches that
    read it."""
    import torch

    plan, q, k, v = _find_plan(q, k, v, config, scale, index, mode)
    if torch.cuda.current_device() == index:
        return _attention_on_device(plan, q, k, v, scale, index)
    with torch.cuda.device(index):
        return _attention_on_device(plan, q, k, v, scale, index)


def _attention_on_device(
    plan: dict, q: Any, k: Any, v: Any, scale: Optional[float], index: int
) -> Any:
    """One call of ``plan`` on the current stream."""
    import torch

    path, n = _choose_path(
        plan,
        race = lambda: _sink_race(
            plan, q, k, v, scale, index, forced = plan.get("mode") == MODE_KERNELS
        ),
    )
    if path == "native":
        return native_attention(q, k, v) if scale is None else native_attention(q, k, v, scale)
    dev = q.device
    i8 = torch.empty(plan["bytes"], dtype = torch.int8, device = dev)
    out = torch.empty(plan["out_shape"], dtype = v.dtype, device = dev)
    stream = _raw_stream(index)
    _launch_quantisers(plan, q, k, v, i8.data_ptr(), stream, path)
    if path:
        decided = _learn_now(plan, i8, n)
        if decided is None:
            _check_sinks(plan, i8, n)
        elif decided == "native":
            return native_attention(q, k, v) if scale is None else native_attention(q, k, v, scale)
        elif decided is False:
            # this call clipped and the kernels without the sink path won: quantise it again for them
            _launch_quantisers(plan, q, k, v, i8.data_ptr(), stream, False)
            path = False
    _launch_attention(plan, i8.data_ptr(), out.data_ptr(), stream, path)
    return out


def kernel_function(
    device: Any,
    head_dim: int,
    config: dict,
    sinks: bool = False,
) -> Any:
    """The loaded CUDA kernel, with or without the sink path."""
    import torch
    return _functions(_device_index(torch.device(device)), head_dim, config, sinks)[0]


def attend(p: dict) -> Any:
    """Launch the kernel on ``quantise``'s output; ``[B, Sq, H, D]`` in the dtype of V."""
    import torch

    plan, i8 = p["plan"], p["i8"]
    index = plan["index"]
    out = torch.empty(plan["out_shape"], dtype = p["dtype"], device = i8.device)
    sinks = p.get("sinks", False)
    if torch.cuda.current_device() == index:
        _launch_attention(plan, i8.data_ptr(), out.data_ptr(), _raw_stream(index), sinks)
    else:
        with torch.cuda.device(index):
            _launch_attention(plan, i8.data_ptr(), out.data_ptr(), _raw_stream(index), sinks)
    return out


def _call_route(q: Any, k: Any, v: Any) -> tuple:
    """(reason, device index, implementation, config): why this call cannot run the kernel (None when it can), and
    what serves it. Every device fact is read once per call."""
    import torch

    if not (q.is_cuda and k.is_cuda and v.is_cuda) or not (q.device == k.device == v.device):
        return "not on one CUDA device", None, None, None
    if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape:
        return "not [B, S, H, D]", None, None, None
    B, Sq, H, D = q.shape
    if k.shape[0] != B or k.shape[2] != H or k.shape[3] != D or Sq < 1 or k.shape[1] < 1:
        return "q / k / v shapes differ", None, None, None
    if D not in HEAD_DIMS:
        return f"head dim {D}", None, None, None
    kinds = _KINDS or _dtype_kinds()
    if q.dtype not in kinds or k.dtype not in kinds or v.dtype not in kinds:
        return "dtype", None, None, None
    if B * H > 65535:
        return "batch * heads", None, None, None
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return "needs gradients", None, None, None
    index = q.device.index
    if index is None:
        index = torch.cuda.current_device()
    impl, config = _serving(index, D)
    if config is None:
        return "no kernel config for this card", index, None, None
    if (index, D) in _BROKEN:
        return "kernel failed on this card", index, impl, config
    return None, index, impl, config


def unsupported_reason(q: Any, k: Any, v: Any) -> Optional[str]:
    """Why this call cannot run the kernel (it then runs native attention), or None."""
    return _call_route(q, k, v)[0]


def _verified(
    q: Any,
    k: Any,
    v: Any,
    impl: Optional[str] = None,
) -> bool:
    """Whether the accuracy check passed for this call's card, head dim and operand dtypes on the implementation that
    serves it (the op runs the kernel only then: a check passed in one dtype, or by the other implementation, says
    nothing about this one)."""
    index = _device_index(q.device)
    impl = impl or implementation(index, int(q.shape[-1]))
    return impl is not None and _verified_at(index, q, k, v, impl)


def _verified_at(index: int, q: Any, k: Any, v: Any, impl: str) -> bool:
    head_dim = int(q.shape[-1])
    return all((index, head_dim, str(x.dtype), impl) in _VERIFIED for x in (q, k, v))


def _long_ready(index: int, head_dim: int, impl: Optional[str], q: Any, k: Any, v: Any) -> bool:
    """Whether the long variant passed its check for this call's card, head dim, dtypes and implementation (and has
    not failed in a call since)."""
    return (index, head_dim, impl) not in _LONG_BROKEN and all(
        (index, head_dim, str(x.dtype), impl) in _LONG_VERIFIED for x in (q, k, v)
    )


def _uses_long(config: Optional[dict], keys: int) -> bool:
    long_keys = (config or {}).get("long_keys")
    return bool(long_keys) and int(keys) >= int(long_keys)


def _run(
    q: Any,
    k: Any,
    v: Any,
    impl: Optional[str],
    config: Optional[dict] = None,
    scale: Optional[float] = None,
    mode: int = MODE_MAIN,
) -> Any:
    """The kernels of ``impl`` on a call ``unsupported_reason`` accepted (``mode``: the CUDA kernel's, as for
    ``int8_attention_call``). Without an explicit ``config``, a call long
    enough for the card's long variant runs it only where that variant passed its own check; elsewhere it runs the
    short config (a valid kernel at any length, only slower there)."""
    import torch

    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if not config:
        head_dim = int(q.shape[-1])
        config = device_config(index, head_dim, impl)
        if _uses_long(config, k.shape[1]) and not _long_ready(index, head_dim, impl, q, k, v):
            config = select_config(config, 0)
    if impl == "triton":
        from . import diffusion_int8_attention_triton as triton_kernel

        # the launches run under the card of q here, so the module's own same-card check is skipped
        with torch.no_grad(), torch.cuda.device(q.device):
            return triton_kernel._int8_attention(q, k, v, config, scale = scale)
    with torch.no_grad():
        return attend(quantise(q, k, v, config, scale, mode))


def int8_attention(
    q: Any,
    k: Any,
    v: Any,
    scale: Optional[float] = None,
    config: Optional[dict] = None,
    mode: int = MODE_MAIN,
) -> Any:
    """``softmax(q k^T * scale) v`` with int8 products. ``[B, Sq, H, D]``, ``[B, Sk, H, D]`` x 2 -> ``[B, Sq, H, D]``
    in the dtype of V. Raises where ``unsupported_reason`` says so; ``int8_attention_impl`` is the guarded form."""
    import torch

    reason = unsupported_reason(q, k, v)
    if reason is not None:
        raise ValueError(f"INT8 attention cannot take this call: {reason}")
    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    return _run(q, k, v, implementation(index, int(q.shape[-1])), config, scale, mode)


def native_attention(
    q: Any,
    k: Any,
    v: Any,
    scale: Optional[float] = None,
) -> Any:
    """torch's attention on the same ``[B, S, H, D]`` views."""
    import torch.nn.functional as F

    out = F.scaled_dot_product_attention(
        q.to(v.dtype).permute(0, 2, 1, 3),
        k.to(v.dtype).permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        scale = scale,
    )
    return out.permute(0, 2, 1, 3)


def int8_attention_impl(
    q: Any,
    k: Any,
    v: Any,
    mode: int = MODE_MAIN,
) -> Any:
    """The kernel where it can run, native attention elsewhere and after any failure (``mode``: as for
    ``int8_attention_call``)."""
    import torch

    reason, index, impl, config = _call_route(q, k, v)
    if reason is None and impl is not None and _verified_at(index, q, k, v, impl):
        if _uses_long(config, k.shape[1]) and not _long_ready(
            index, int(q.shape[-1]), impl, q, k, v
        ):
            config = select_config(
                config, 0
            )  # the long variant runs only where it passed its own check
        try:
            if impl == "triton":
                out = _run(q, k, v, impl, config)
            elif torch.is_grad_enabled():
                with torch.no_grad():
                    out = _cuda_attention(q, k, v, config, None, index, mode)
            else:
                out = _cuda_attention(q, k, v, config, None, index, mode)
            STATS["int8"] += 1
            if impl == "triton":
                STATS["triton"] += 1
            return out
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a JIT / launch failure keeps native attention on this card
            head_dim = int(q.shape[-1])
            if _uses_long(config, k.shape[1]) and _long_ready(index, head_dim, impl, q, k, v):
                # only the long variant failed: the verified short kernel keeps every call, long ones included
                _LONG_BROKEN.add((index, head_dim, impl))
                _log().warning(
                    "diffusion.int8_attention: long-sequence kernel failed, the short one runs long calls from here on: %s",
                    exc,
                )
            else:
                _BROKEN.add((index, head_dim))
                _log().warning(
                    "diffusion.int8_attention: kernel failed, native attention from here on: %s",
                    exc,
                )
    STATS["native"] += 1
    return native_attention(q, k, v)


def _tracing() -> bool:
    """Whether a dispatch mode (make_fx, FlopCounterMode), a functorch transform (vmap, grad) or the TorchScript
    tracer is active: their tensors look plain, and they must see the op, not raw launches (a trace would record the
    allocations but not the ctypes launches that fill them)."""
    import torch

    if torch.jit.is_tracing():
        return True
    stack = getattr(torch._C, "_len_torch_dispatch_stack", None)
    functorch = getattr(torch._C, "_are_functorch_transforms_active", None)
    return bool((stack is not None and stack()) or (functorch is not None and functorch()))


def int8_attention_call(
    q: Any,
    k: Any,
    v: Any,
    mode: int = MODE_MAIN,
) -> Any:
    """What a processor calls: the custom op while torch.compile traces (opaque, with a fake), and in eager the op's
    body directly, which skips the dispatcher's per-call cost (about a fifth of a short call's CPU time). Tensor
    subclasses, dispatch modes and functorch transforms always go through the op. ``mode`` (the route's): MODE_MAIN
    the kernels as before the attention-sink path, else the sink path with its race (a constant of the graph: another
    op)."""
    import torch

    if (
        type(q) is torch.Tensor
        and type(k) is torch.Tensor
        and type(v) is torch.Tensor
        and not torch.compiler.is_compiling()
        and not _tracing()
    ):
        return int8_attention_impl(q, k, v, mode) if mode else int8_attention_impl(q, k, v)
    if mode != MODE_MAIN:
        return torch.ops.unsloth_studio.int8_attention_nhd_sinks(q, k, v, int(mode))
    return torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)


def _log() -> Any:
    import logging
    return logging.getLogger(__name__)


@lru_cache(maxsize = None)
def int8_attention_op() -> Any:
    """Registered once per process: opaque to torch.compile, with a fake for shape propagation."""
    import torch

    # explicit schema: the annotations here are strings
    @torch.library.custom_op(
        _OP, mutates_args = (), schema = "(Tensor q, Tensor k, Tensor v) -> Tensor"
    )
    def int8_attention_nhd(q, k, v):
        # The fake's strides are part of the op contract: Inductor asserts them on the real output. The kernels
        # write a fresh contiguous [B, S, H, D]; native attention can hand back the permuted [B, H, S, D] layout
        # (math SDPA does), so it is copied to the fake's layout here (a no-op when it already is).
        return int8_attention_impl(q, k, v).contiguous()

    @int8_attention_nhd.register_fake
    def _(q, k, v):
        # contiguous [B, S, H, D] in V's dtype: the layout every real output above has
        return v.new_empty(q.shape)

    # the attention-sink routes (``mode``): another op, so the main one keeps its schema and its graphs
    @torch.library.custom_op(
        _OP + "_sinks", mutates_args = (), schema = "(Tensor q, Tensor k, Tensor v, int mode) -> Tensor"
    )
    def int8_attention_nhd_sinks(q, k, v, mode):
        return int8_attention_impl(q, k, v, int(mode)).contiguous()

    @int8_attention_nhd_sinks.register_fake
    def _(q, k, v, mode):
        return v.new_empty(q.shape)

    return int8_attention_nhd


def reference_attention(q: Any, k: Any, v: Any) -> Any:
    """fp32 attention on ``[B, S, H, D]``: what the kernel is checked against."""
    import torch.nn.functional as F

    out = F.scaled_dot_product_attention(
        q.float().permute(0, 2, 1, 3), k.float().permute(0, 2, 1, 3), v.float().permute(0, 2, 1, 3)
    )
    return out.permute(0, 2, 1, 3)


def probe_inputs(
    device: Any,
    dtype: Any,
    tokens: int,
    heads: int,
    head_dim: int,
    seed: int = 0,
) -> tuple:
    """q / k like RMS-normalised projections with a learned gain, v with per-channel spread and offset."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(1000 + seed)
    shape = (1, tokens, heads, head_dim)
    q = torch.randn(shape, generator = g) * 1.6
    k = torch.randn(shape, generator = g) * 1.6 + 0.4 * torch.randn(
        (1, 1, heads, head_dim), generator = g
    )
    v = torch.randn(shape, generator = g) * (0.5 + torch.rand((1, 1, heads, head_dim), generator = g))
    v = v + 0.5 * torch.randn((1, 1, heads, head_dim), generator = g)
    return tuple(x.to(device = device, dtype = dtype) for x in (q, k, v))


def accuracy(got: Any, ref: Any) -> dict:
    import torch.nn.functional as F
    got, ref = got.float(), ref.float()
    return {
        "cos": float(F.cosine_similarity(got.flatten(), ref.flatten(), dim = 0)),
        "rel_l1": float((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-30)),
        "max_abs": float((got - ref).abs().max()),
        "mean_abs": float((got - ref).abs().mean()),
    }


def _median_ms(fns: dict, rounds: int = 5) -> dict:
    """Median milliseconds per callable, timed with CUDA events in interleaved rounds after two warm calls each. The
    races time on probe inputs, which have no attention sinks: the kernel calls in here take the path a plan without
    sinks runs (_choose_path), and count toward no plan's decision."""
    import torch

    _RACE.path = False
    try:
        for fn in fns.values():
            fn()
            fn()
        times: dict = {name: [] for name in fns}
        for _ in range(rounds):
            for name, fn in fns.items():
                a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                times[name].append(a.elapsed_time(b))
    finally:
        _RACE.path = None
    return {name: sorted(ts)[len(ts) // 2] for name, ts in times.items()}


def _long_check(device: Any, dtype: Any, head_dim: int, impl: str) -> Optional[dict]:
    """The accuracy check of the card's long variant (``long_keys`` / ``long`` in its config), once per card, head
    dim, dtype and implementation, or None where the config has no long variant. It compiles that kernel here,
    eagerly, rather than on the first long call (which may be under CUDA graph capture). A pass lets long calls run
    it (``_LONG_VERIFIED``); a miss or a failure leaves them on the short config, which stays verified."""
    import torch

    index = _device_index(torch.device(device))
    hd = int(head_dim)
    try:
        config = device_config(index, hd, impl)
    except Exception:  # noqa: BLE001 -- no card to read a config for (the check run off CUDA)
        config = None
    if not (config or {}).get("long_keys"):
        return None
    key = (index, hd, str(dtype), impl)
    found = _LONG_CHECKS.get(key)
    if found is not None:
        return found
    keys = int(config["long_keys"]) + LONG_PROBE_EXTRA
    found = {"ok": False, "keys": keys}
    try:
        with torch.no_grad():
            q, k, v = probe_inputs(device, dtype, keys, LONG_PROBE_HEADS, hd, seed = keys)
            q = q[:, :LONG_PROBE_QUERIES]
            acc = accuracy(int8_attention(q, k, v, config = config), reference_attention(q, k, v))
            q, k = q * SHARP_GAIN, k * SHARP_GAIN
            sharp = int8_attention(q, k, v, config = config)
            finite = bool(torch.isfinite(sharp).all())
            sharp_cos = accuracy(sharp, reference_attention(q, k, v))["cos"] if finite else 0.0
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the long variant cannot run here: long calls keep the short kernel
        found.update(failed = True, error = repr(exc)[:300])
    else:
        found.update(cos = acc["cos"], rel_l1 = acc["rel_l1"], sharp_cos = sharp_cos)
        found["ok"] = bool(
            acc["cos"] >= MIN_COSINE
            and acc["rel_l1"] <= MAX_REL_L1
            and sharp_cos >= MIN_SHARP_COSINE
        )
    if found["ok"]:
        _LONG_VERIFIED.add(key)
    else:
        _log().warning(
            "diffusion.int8_attention: long-sequence kernel not verified on cuda:%d (head_dim=%d, %s), the short one "
            "runs long calls: %s",
            index,
            hd,
            impl,
            found.get("error") or {x: found.get(x) for x in ("cos", "rel_l1", "sharp_cos")},
        )
    _LONG_CHECKS[key] = found
    return found


def _check(
    device: Any,
    dtype: Any,
    heads: int,
    head_dim: int,
    incumbent: Optional[Callable],
    timed: bool,
    impl: str = "cuda",
) -> dict:
    """``ok``: the kernel runs here and matches fp32 attention on the probe, also with q and k scaled up. ``ms``: int8 and incumbent milliseconds
    per call at each RACE_TOKENS size (when ``timed``). ``min_tokens``: the sequence length from which the kernel
    is more than 2% faster than the incumbent, or None when it is not at the larger size. ``failed``: the kernel
    raised (it could not be built, loaded or launched) rather than missed the accuracy bar. ``impl`` is the
    implementation ``int8_attention`` runs here; a pass is recorded for it alone. ``long``: the check of the
    card's long variant (``_long_check``), where its config has one; it never changes ``ok``."""
    import torch

    out: dict = {"ok": False, "ms": {}, "min_tokens": None, "impl": impl}
    try:
        with torch.no_grad():
            q, k, v = probe_inputs(device, dtype, 1000, min(int(heads), 4), head_dim)
            acc = accuracy(int8_attention(q, k, v), reference_attention(q, k, v))
            q, k = q * SHARP_GAIN, k * SHARP_GAIN
            sharp = int8_attention(q, k, v)
            finite = bool(torch.isfinite(sharp).all())
            sharp_cos = accuracy(sharp, reference_attention(q, k, v))["cos"] if finite else 0.0
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 -- no kernel on this card
        out["error"] = repr(exc)[:300]
        out["failed"] = True
        return out
    out.update(acc)
    out["sharp_cos"] = sharp_cos
    out["ok"] = bool(
        acc["cos"] >= MIN_COSINE and acc["rel_l1"] <= MAX_REL_L1 and sharp_cos >= MIN_SHARP_COSINE
    )
    if out["ok"]:
        dev = torch.device(device)
        _VERIFIED.add((_device_index(dev), int(head_dim), str(dtype), impl))
        long = _long_check(device, dtype, head_dim, impl)
        if long is not None:
            out["long"] = long
    if not out["ok"] or not timed:
        return out
    wins = {}
    for tokens in RACE_TOKENS:
        with torch.no_grad():
            q, k, v = probe_inputs(device, dtype, tokens, heads, head_dim, seed = tokens)
            fns = {
                "incumbent": (lambda: incumbent(q, k, v))
                if incumbent
                else (lambda: native_attention(q, k, v)),
                "int8": lambda: int8_attention(q, k, v),
            }
            try:
                ms = _median_ms(fns)
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 -- an incumbent that cannot be timed is kept
                out["error"] = repr(exc)[:300]
                return out
        out["ms"][tokens] = {name: round(value, 4) for name, value in ms.items()}
        wins[tokens] = ms["int8"] < RACE_MARGIN * ms["incumbent"]
    large, small = RACE_TOKENS
    if wins[large]:
        out["min_tokens"] = 0 if wins[small] else (large + small) // 2
    return out


def _verdict_key(key: tuple) -> str:
    import json
    return json.dumps(list(key[1:]))


def _storable(found: dict) -> bool:
    """Reusable only if exception-free (failures may be transient) and, when timed and passing, has all race sizes."""
    if found.get("error") or found.get("failed") or (found.get("long") or {}).get("failed"):
        return False
    return not (
        found.get("ok") and found.get("timed") and len(found.get("ms") or {}) < len(RACE_TOKENS)
    )


def _persist_verdict(index: int, key: tuple, found: dict) -> None:
    if key[-1] is None or not _storable(dict(found, timed = key[-2])):
        return
    try:
        persisted().put(card_name(index), _verdict_key(key), found)
    except Exception:  # noqa: BLE001
        pass


def _warm(index: int, dtype: Any, head_dim: int, impl: str, found: dict) -> None:
    """Eagerly load the vouched kernels (before CUDA graph capture); raises if they no longer run here."""
    import torch

    device = torch.device("cuda", index)
    with torch.no_grad():
        q, k, v = probe_inputs(device, dtype, WARM_TOKENS, 1, head_dim)
        _run(q, k, v, impl, config = select_config(device_config(index, head_dim, impl), 0))
        if (found.get("long") or {}).get("ok"):
            config = device_config(index, head_dim, impl)
            keys = int(config["long_keys"]) + LONG_PROBE_EXTRA
            q, k, v = probe_inputs(device, dtype, keys, 1, head_dim, seed = keys)
            _run(q[:, :WARM_TOKENS], k, v, impl, config = config)
        torch.cuda.synchronize(device)


def _adopt_verdict(index: int, dtype: Any, head_dim: int, key: tuple) -> Optional[dict]:
    """The persisted verdict for ``key`` installed as ``_check`` would leave it, or None to re-check."""
    import torch

    impl = key[-1]
    if impl is None:
        return None
    try:
        stored = persisted().get(card_name(index), _verdict_key(key))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(stored, dict) or stored.get("impl") != impl:
        return None
    found = dict(stored, persisted = True)
    found["ms"] = {
        int(t): v for t, v in (found.get("ms") or {}).items()
    }  # JSON keys come back as strings
    if found.get("ok"):
        try:
            _warm(index, dtype, head_dim, impl, found)
        except torch.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001
            return None
        _VERIFIED.add((index, int(head_dim), str(dtype), impl))
        long = found.get("long")
        if isinstance(long, dict):
            long_key = (index, int(head_dim), str(dtype), impl)
            _LONG_CHECKS[long_key] = long
            if long.get("ok"):
                _LONG_VERIFIED.add(long_key)
    return found


def verdict(
    device: Any,
    dtype: Any,
    heads: int,
    head_dim: int,
    incumbent: Optional[Callable] = None,
    incumbent_name: str = "native",
    timed: bool = True,
) -> dict:
    """``_check`` cached per (device, dtype, heads, head dim, incumbent, timed, implementation). A CUDA kernel that
    raises in the check (no NVRTC, a failed compile, module load or launch) is recorded as failed on this card and
    the check runs again on the implementation that serves it now (Triton, where it can), whose verdict is
    returned."""
    import torch

    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    hd = int(head_dim)

    def key_of(impl: Optional[str]) -> tuple:
        return (index, str(dtype), int(heads), hd, incumbent_name, bool(timed), impl)

    found = _VERDICTS.get(key_of(implementation(index, hd)))
    if found is not None and not (found.get("impl") == "cuda" and found.get("failed")):
        return found
    with _LOCK:
        while True:
            impl = implementation(index, hd)
            key = key_of(impl)
            found = _VERDICTS.get(key)
            if found is None and impl is None:
                found = {
                    "ok": False,
                    "ms": {},
                    "min_tokens": None,
                    "impl": None,
                    "error": _CUDA_FAILED.get((index, hd)) or "no kernel for this card",
                }
                _VERDICTS[key] = found
            if found is None:
                with torch.cuda.device(index):
                    found = _adopt_verdict(index, dtype, hd, key)
                    if found is None:
                        found = _check(
                            torch.device("cuda", index),
                            dtype,
                            int(heads),
                            hd,
                            incumbent,
                            timed,
                            impl,
                        )
                        _persist_verdict(index, key, found)
                _VERDICTS[key] = found
            if impl == "cuda" and found.get("failed") and (index, hd) not in _CUDA_FAILED:
                _CUDA_FAILED[(index, hd)] = found.get("error")
                _log().warning(
                    "diffusion.int8_attention: CUDA kernel unavailable on cuda:%d (head_dim=%d): %s; %s",
                    index,
                    hd,
                    found.get("error"),
                    "trying the Triton kernel"
                    if implementation(index, hd) == "triton"
                    else "native attention",
                )
                continue
            return found


def _head_dim(attn: Any) -> Optional[int]:
    value = getattr(attn, "head_dim", None)
    if isinstance(value, int):
        return value
    try:
        return int(attn.to_out[0].in_features) // int(attn.heads)
    except Exception:  # noqa: BLE001
        return None


def int8_processors(transformer: Any) -> list:
    """The attention modules whose processor can route to the kernel (the strided processors)."""
    return [
        m
        for m in transformer.modules()
        if hasattr(type(getattr(m, "processor", None)), "_unsloth_int8") and hasattr(m, "heads")
    ]


def _sdpa_incumbent(backend: Any) -> tuple:
    """(name, callable) for a processor's native backend: torch attention pinned the way the processor pins it."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    value = str(getattr(backend, "value", backend)) if backend is not None else "native"
    kernel = {
        "_native_cudnn": SDPBackend.CUDNN_ATTENTION,
        "_native_flash": SDPBackend.FLASH_ATTENTION,
    }.get(value)
    if kernel is None:
        return "native", native_attention

    def run(q: Any, k: Any, v: Any) -> Any:
        with sdpa_kernel(kernel):
            out = F.scaled_dot_product_attention(
                q.to(v.dtype).permute(0, 2, 1, 3),
                k.to(v.dtype).permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
            )
        return out.permute(0, 2, 1, 3)

    return value, run


def _secondary_cards(primary: int, transformer: Any = None) -> tuple:
    """The other cards this DiT runs on in a multi-GPU load (the dispatch route checks the same set), or none."""
    try:
        from .diffusion_int8_attention_dispatch import _secondary_cards as cards
        dits = () if transformer is None else (transformer,)
        return tuple(c for c in cards(primary, dits) if int(c) != int(primary))
    except Exception:  # noqa: BLE001 - no selection: the load's card only
        return ()


def sink_mode(dit_name: str, forced: bool) -> int:
    """The ``mode`` a route of DiT class ``dit_name`` runs the kernel in: the attention-sink path only for SINK_DITS,
    the families it was measured on."""
    if dit_name not in SINK_DITS:
        return MODE_MAIN
    return MODE_KERNELS if forced else MODE_RACE


def note_sink_checkpoint(dit: Any) -> None:
    """The sink-family DiT whose routes are installed: persisted sink verdicts key on its class and checkpoint, since
    a verdict holds for the model whose inputs it raced on (a DiT of no known checkpoint persists none)."""
    name = type(dit).__name__
    if name not in SINK_DITS:
        return
    path = str(getattr(getattr(dit, "config", None), "_name_or_path", "") or "")
    _SINK_DIT[0] = (name, path) if path else None


def install_int8_attention(
    transformer: Any,
    *,
    device: Any,
    dtype: Any,
    modules: Optional[list] = None,
    incumbent: Optional[tuple] = None,
    force: bool = False,
    logger: Any = None,
    floor: Optional[Callable] = None,
    policy: Any = None,
) -> int:
    """Route strided attention processors to the INT8 kernel. Returns how many now run it.

    ``modules`` limits the attention modules considered (default: every one with a strided processor).
    ``incumbent`` is ``(name, callable(q, k, v))`` for what those modules run now (default: their native backend).
    The kernel must match fp32 attention on this card; unless ``force`` (or the env switch is ``1``) it must also be
    more than 2% faster than the incumbent on the timed probe. The Triton stand-in for a CUDA kernel that cannot load
    must be faster also under ``force`` (not under the env switch). ``floor(device, dtype, heads, head_dim, incumbent,
    name)`` -> ``(verdict, token floor or None)`` replaces the card race when not forced (``install_auto``: the
    dispatch route's races and token floor). ``policy``: the load's ``Int8Policy`` (or ``int8_attention`` request);
    None reads the env switch. Never raises."""
    try:
        return _install(
            transformer, device, dtype, modules, incumbent, force, logger, floor, policy
        )
    except Exception as exc:  # noqa: BLE001 -- an optimisation: the processors stay as they were
        if logger is not None:
            logger.info("diffusion.int8_attention: skipped (%s)", exc)
        return 0


def _install(
    transformer: Any,
    device: Any,
    dtype: Any,
    modules: Optional[list],
    incumbent: Optional[tuple],
    force: bool,
    logger: Any,
    floor: Optional[Callable] = None,
    policy: Any = None,
) -> int:
    import torch

    setting = mode(policy)
    if transformer is None or setting == "off":
        return 0
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return 0
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    if str(dtype) not in _OUT_KIND:
        return 0
    candidates = [
        m for m in int8_processors(transformer) if modules is None or any(m is x for x in modules)
    ]
    if not candidates:
        return 0
    env_force = setting == "force" and env_forced(policy)
    # a per-load "on" forces like a request by name: the Triton stand-in still races
    force = force or setting == "force"
    int8_attention_op()
    engaged = 0
    report: dict = {}
    for m in candidates:
        proc = m.processor
        head_dim = _head_dim(m)
        if head_dim is None or implementation(index, head_dim) is None:
            continue
        if getattr(proc, "_parallel_config", None) is not None:
            continue
        name, fn = (
            incumbent
            if incumbent is not None
            else _sdpa_incumbent(getattr(proc, "_attention_backend", None))
        )
        if floor is not None and not force:
            found, minimum = floor(dev, dtype, int(m.heads), head_dim, fn, name)
            report[(int(m.heads), head_dim, name)] = found
            proc._unsloth_int8_impl = found.get("impl")
            if minimum is None:
                continue
            proc._unsloth_int8 = True
            proc._unsloth_int8_min_tokens = int(minimum)
            proc._unsloth_int8_sinks = sink_mode(type(transformer).__name__, False)
            register_route(index, head_dim, _raise_processor_floor(proc, m))
            engaged += 1
            for card in _secondary_cards(index, transformer):
                if implementation(card, head_dim) is None:
                    continue
                report[(card, int(m.heads), head_dim, name)] = verdict(
                    torch.device("cuda", card), dtype, int(m.heads), head_dim, fn, name, timed = False
                )
            continue
        found = verdict(dev, dtype, int(m.heads), head_dim, fn, name, timed = not force)
        # a Triton stand-in races even when requested by name (only the env switch or IMPL=triton skip it)
        stand_in = (
            found.get("impl") == "triton"
            and not env_force
            and impl_setting() == "auto"
            and not _is_rocm()
        )
        if force and stand_in and found.get("ok"):
            found = verdict(dev, dtype, int(m.heads), head_dim, fn, name, timed = True)
        report[(int(m.heads), head_dim, name)] = found
        proc._unsloth_int8_impl = found.get("impl")
        if not found.get("ok"):
            continue
        if force and not stand_in:
            minimum = 0
        elif found.get("min_tokens") is None:
            continue
        else:
            minimum = int(found["min_tokens"])
        proc._unsloth_int8 = True
        proc._unsloth_int8_min_tokens = minimum
        proc._unsloth_int8_sinks = sink_mode(type(transformer).__name__, force)
        note_sink_checkpoint(transformer)
        if not force:
            register_route(index, head_dim, _raise_processor_floor(proc, m))
        engaged += 1
        # A layer split or CFG replica runs this processor on the selection's other cards too: the op runs the kernel
        # only on a card whose own accuracy check passed, so check each of them here (the race stays the primary's).
        for card in _secondary_cards(index, transformer):
            if implementation(card, head_dim) is None:
                continue
            report[(card, int(m.heads), head_dim, name)] = verdict(
                torch.device("cuda", card), dtype, int(m.heads), head_dim, fn, name, timed = False
            )
    if logger is not None:
        for key, found in report.items():
            card, heads, head_dim, name = key if len(key) == 4 else (index, *key)
            logger.info(
                "diffusion.int8_attention: cuda:%d heads=%d head_dim=%d vs %s: impl=%s ok=%s cos=%s ms=%s "
                "min_tokens=%s%s%s%s",
                card,
                heads,
                head_dim,
                name,
                found.get("impl"),
                found.get("ok"),
                found.get("cos"),
                found.get("ms"),
                found.get("min_tokens"),
                f" long_ok={found['long'].get('ok')}" if "long" in found else "",
                f" long race min_tokens={found['long_min_tokens']}"
                if "long_min_tokens" in found
                else (
                    f" 2048 race min_tokens={found['mid_min_tokens']}"
                    if "mid_min_tokens" in found
                    else ""
                ),
                " (forced)" if force else "",
            )
        if engaged:
            logger.info(
                "diffusion.int8_attention: %s (%s) on %d attention modules",
                BACKEND,
                "/".join(sorted(engaged_implementations(transformer))),
                engaged,
            )
    return engaged


def install_requested(
    pipe: Any,
    request: Any,
    target: Any,
    logger: Any = None,
    policy: Any = None,
) -> int:
    """For ``apply_attention_backend``: engage on the pipeline's DiT when the load asked for ``unsloth_int8`` by name
    or the env switch forces it (no race either way: the accuracy check still gates it; the Triton stand-in for a CUDA
    kernel that cannot load still races when only the name asked for it). ``policy``: the load's ``Int8Policy``, whose
    ``on`` forces it like the name. Returns the count."""
    import torch

    setting = mode(policy)
    if setting == "off" or not (setting == "force" or requested(request)):
        return 0
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return 0
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", "") or "")
    if not device.startswith("cuda"):
        return 0
    dtype = getattr(target, "dtype", None)
    if str(dtype) not in _OUT_KIND:
        # sm75 has no bf16: its pipelines run fp16
        index = _device_index(torch.device(device))
        dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability(index) >= (8, 0) else torch.float16
        )
    try:
        from .diffusion_flux3_attn import ensure_strided_processors
        ensure_strided_processors(pipe, target, logger)
    except Exception:  # noqa: BLE001 -- other families bring their own strided processors
        pass
    return install_int8_attention(
        transformer, device = device, dtype = dtype, force = True, logger = logger, policy = policy
    )


def install_auto(
    pipe: Any,
    target: Any,
    logger: Any = None,
    speed_active: bool = True,
    speed_unset_only: bool = False,
    policy: Any = None,
) -> int:
    """For ``apply_attention_backend`` with the switch unset: the strided processors of a DiT in the dispatch route's
    ``AUTO_DITS`` race their incumbent the way that route does (accuracy check, card race, the 2048 / long races,
    ``MIN_TOKENS`` floor; on ROCm the ``ROCM_AUTO_DITS`` of the card's gfx arch and the OS). Off with speed off
    (bit-identical) and off CUDA. Returns the count."""
    import torch

    if mode(policy) != "auto" or not speed_active:
        return 0
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return 0
    try:
        from . import diffusion_int8_attention_dispatch as dispatch
    except Exception:  # noqa: BLE001
        return 0
    if not dispatch.enabled():
        return 0
    if speed_unset_only and type(transformer).__name__ not in dispatch.SPEED_UNSET_DITS:
        return 0
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", "") or "")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0
    rocm = _is_rocm()
    listed = (
        dispatch.rocm_auto_dits(_device_index(torch.device(device))) if rocm else dispatch.AUTO_DITS
    )
    if type(transformer).__name__ not in listed:
        return 0
    dtype = getattr(target, "dtype", None)
    if str(dtype) not in ("torch.float16", "torch.bfloat16"):
        index = _device_index(torch.device(device))
        dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability(index) >= (8, 0) else torch.float16
        )
    try:
        from .diffusion_flux3_attn import ensure_strided_processors
        ensure_strided_processors(pipe, target, logger)
    except Exception:  # noqa: BLE001 -- other families bring their own strided processors
        pass

    cuda_only = not rocm and type(transformer).__name__ in dispatch.AUTO_CUDA_ONLY

    def floor(dev: Any, dt: Any, heads: int, head_dim: int, fn: Any, name: str) -> tuple:
        if cuda_only and implementation(_device_index(dev), head_dim) != "cuda":
            return {
                "ok": False,
                "impl": implementation(_device_index(dev), head_dim),
                "error": "CUDA kernel only",
            }, None
        found, minimum = dispatch._card_floor(dev, dt, heads, head_dim, fn, name, False, False)
        # an untried CUDA kernel reads "cuda" above; one that failed in the check falls back to Triton in the verdict
        if cuda_only and found.get("impl") != "cuda":
            return {**found, "ok": False, "error": "CUDA kernel only"}, None
        return found, minimum

    engaged = install_int8_attention(
        transformer, device = device, dtype = dtype, logger = logger, floor = floor, policy = policy
    )
    _give_back_unengaged(transformer)
    return engaged


def family_race_tokens(transformer: Any) -> Optional[tuple]:
    """``FAMILY_RACE_TOKENS`` of this DiT's class, or None (the family does not race automatically that way)."""
    tokens = FAMILY_RACE_TOKENS.get(type(transformer).__name__)
    return tuple(int(t) for t in tokens) if tokens else None


def install_family_auto(
    transformer: Any,
    *,
    device: Any,
    dtype: Any,
    policy: Any = None,
    logger: Any = None,
) -> int:
    """Policy ``auto``: race a ``FAMILY_RACE_TOKENS`` DiT's strided processors at its own sizes (CUDA kernel only:
    accuracy was measured on it); the token floor keeps short calls on the incumbent. Never raises."""
    if transformer is None or mode(policy) != "auto":
        return 0
    tokens = family_race_tokens(transformer)
    if tokens is None:
        return 0
    try:
        import torch

        from . import diffusion_int8_attention_dispatch as dispatch
        if _is_rocm():
            return 0
    except Exception:  # noqa: BLE001
        return 0

    def floor(dev: Any, dt: Any, heads: int, head_dim: int, fn: Any, name: str) -> tuple:
        if implementation(_device_index(dev), head_dim) != "cuda":
            return {
                "ok": False,
                "impl": implementation(_device_index(dev), head_dim),
                "error": "CUDA kernel only",
            }, None
        # the accuracy check alone: the family's own race below replaces the RACE_TOKENS one
        found = verdict(dev, dt, heads, head_dim, fn, name, timed = False)
        if not found.get("ok"):
            return found, None
        if found.get("impl") != "cuda":
            return {**found, "ok": False, "error": "CUDA kernel only"}, None
        minimum = dispatch.long_verdict(dev, dt, heads, head_dim, fn, name, tokens = tokens)
        found = {**found, "long_min_tokens": minimum}
        if minimum is None:
            return found, None
        return found, max(int(minimum), int(dispatch.MIN_TOKENS))

    with torch.no_grad():
        return install_int8_attention(
            transformer,
            device = device,
            dtype = dtype,
            logger = logger,
            floor = floor,
            policy = policy,
        )


def _pipe_dits(pipe: Any) -> list:
    pipes = list(pipe) if isinstance(pipe, (list, tuple)) else [pipe]
    out: list = []
    for p in pipes:
        try:
            from . import diffusion_int8_attention_dispatch as dispatch
            dits = dispatch._dits(p)
        except Exception:  # noqa: BLE001
            dits = []
        if not dits and getattr(p, "transformer", None) is not None:
            dits = [p.transformer]
        out.extend(
            d
            for d in dits
            if callable(getattr(d, "modules", None)) and not any(d is x for x in out)
        )
    return out


def _routed(dit: Any) -> list:
    try:
        from .diffusion_int8_attention_dispatch import Int8Route
    except Exception:  # noqa: BLE001
        return []
    modules = getattr(dit, "modules", None)
    if not callable(modules):
        return []
    return [
        b
        for m in modules()
        if isinstance(
            b := getattr(getattr(m, "processor", None), "_attention_backend", None), Int8Route
        )
    ]


def engaged_count(pipe: Any) -> int:
    """Census of attention modules running the kernel (strided + dispatch); never raises."""
    try:
        return sum(int8_attention_count(d) + len(_routed(d)) for d in _pipe_dits(pipe))
    except Exception:  # noqa: BLE001
        return 0


def resolved_entry(
    policy: Any,
    pipe: Any,
    *,
    reason: Optional[str] = None,
    supported: bool = True,
) -> tuple:
    """(explicit, engaged, reason, status) for ``build_resolved_record``; ``supported=False`` for sd.cpp, MLX, CPU."""
    p = resolve_policy(policy)
    count = engaged_count(pipe) if pipe is not None else 0
    engaged = "on" if count else "off"
    explicit = p.request
    if reason is None:
        if count:
            reason = f"{BACKEND} ({'/'.join(sorted(_pipe_impls(pipe))) or 'cuda'}) on {count} attention modules"
        elif not supported:
            reason = "this engine has no INT8 attention kernel"
        elif p.mode == "off":
            reason = "off for this load" if p.request == "off" else f"off by {INT8_ATTENTION_ENV}=0"
        elif p.mode == "force":
            reason = "requested, but the kernel did not pass its accuracy check on this card or cannot run here"
        else:
            reason = "not measured faster and accurate for this model and GPU; the selected attention runs"
    status_value = None
    if explicit == "on" and not count:
        status_value = "unsupported" if not supported else "fell_back"
    return (
        (explicit, engaged, reason, status_value) if status_value else (explicit, engaged, reason)
    )


def _pipe_impls(pipe: Any) -> set:
    impls: set = set()
    try:
        for dit in _pipe_dits(pipe):
            impls |= engaged_implementations(dit)
            impls |= {str(r.impl) for r in _routed(dit) if r.impl}
    except Exception:  # noqa: BLE001 - a census only
        pass
    return impls


def annotate_resolved(resolved: Any, policy: Any, pipe: Any) -> Any:
    entry = resolved.get("int8_attention") if isinstance(resolved, dict) else None
    if not isinstance(entry, dict):
        return resolved
    count = engaged_count(pipe) if pipe is not None else 0
    entry["engaged"] = bool(count)
    entry["modules"] = int(count)
    entry["impl"] = ("/".join(sorted(_pipe_impls(pipe))) or None) if count else None
    entry["mode"] = resolve_policy(policy).mode
    return resolved


def _give_back_unengaged(transformer: Any) -> None:
    """A strided processor ``install_auto`` put in for the race (``_unsloth_stock``) that did not engage gives the
    module the processor it replaced, so a lost race leaves the module as base had it (FLUX.3: its fused Q/K swap
    follows). That processor is marked raced: the dispatch route does not race it again."""
    for m in int8_processors(transformer):
        proc = m.processor
        stock = getattr(proc, "_unsloth_stock", None)
        if stock is None or getattr(proc, "_unsloth_int8", False):
            continue
        try:
            stock._unsloth_int8_raced = True
        except Exception:  # noqa: BLE001 -- a slotted processor: the route's cached verdict decides alike
            pass
        setter = getattr(m, "set_processor", None)
        if callable(setter):
            setter(stock)
        else:
            m.processor = stock


def remove_int8_attention(transformer: Any) -> int:
    """Every processor back to the attention it ran before ``install_int8_attention``."""
    count = 0
    for m in int8_processors(transformer) if transformer is not None else ():
        if getattr(m.processor, "_unsloth_int8", False):
            m.processor._unsloth_int8 = False
            count += 1
    return count


def int8_attention_count(transformer: Any) -> int:
    """Engagement census: attention modules whose processor routes to the kernel."""
    if transformer is None:
        return 0
    return sum(
        1 for m in int8_processors(transformer) if getattr(m.processor, "_unsloth_int8", False)
    )


# The backend a strided install raced (None: native), kept on the pipeline: the status names it once no module routes.
INCUMBENT_ATTR = "_unsloth_int8_incumbent"


def live_backend(pipe: Any, engaged: Any) -> Any:
    """``engaged`` while any module still routes to the kernel, else the incumbent after a sink race dropped them."""
    if engaged != BACKEND or pipe is None or not hasattr(pipe, INCUMBENT_ATTR):
        return engaged
    dits = [getattr(pipe, name, None) for name in ("transformer", "transformer_2")]
    if any(int8_attention_count(t) for t in dits if t is not None):
        return engaged
    try:
        from .diffusion_int8_attention_dispatch import routed_count
        if routed_count(pipe):
            return engaged
    except Exception:  # noqa: BLE001 - no dispatch route: the strided census decides
        pass
    return getattr(pipe, INCUMBENT_ATTR)


def engaged_implementations(transformer: Any) -> set:
    """The implementations (``cuda`` / ``triton``) the routed processors run."""
    if transformer is None:
        return set()
    return {
        str(getattr(m.processor, "_unsloth_int8_impl", None) or "cuda")
        for m in int8_processors(transformer)
        if getattr(m.processor, "_unsloth_int8", False)
    }


def status(transformer: Any = None, policy: Any = None) -> dict:
    """What the status panel and the logs report (``policy``: the loaded pipeline's ``Int8Policy``)."""
    p = resolve_policy(policy)
    return {
        "backend": BACKEND,
        "mode": p.mode,
        "requested": p.request,
        "source": p.source,
        "impl_setting": impl_setting(),
        "modules": int8_attention_count(transformer),
        "impl": sorted(engaged_implementations(transformer)),
        "calls": dict(STATS),
        "cuda_failed": {f"cuda:{k[0]} head_dim={k[1]}": v for k, v in _CUDA_FAILED.items()},
        "verdicts": {
            f"cuda:{k[0]} {k[1]} heads={k[2]} head_dim={k[3]} vs {k[4]}{' timed' if k[5] else ''} impl={k[6]}": {
                x: v.get(x)
                for x in (
                    "impl",
                    "ok",
                    "cos",
                    "rel_l1",
                    "sharp_cos",
                    "ms",
                    "min_tokens",
                    "error",
                    "long",
                )
                if x in v
            }
            for k, v in _VERDICTS.items()
        },
    }

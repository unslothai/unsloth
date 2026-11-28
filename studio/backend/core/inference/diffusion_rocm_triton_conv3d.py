# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm gfx11: route a video VAE's ``nn.Conv3d`` through the Triton conv3d vendored from ROCm/aiter instead of MIOpen.

gfx1151 Wan 2.2 5B VAE decode 1.61x, PSNR 54.7 dB vs MIOpen, but peak memory 11.8 -> 18.7 GiB, so each call checks
free memory first. ``Conv3d._conv_forward`` is wrapped per instance (covers ``WanCausalConv3d`` and the ``nn.Conv3d``
in ``LTX2VideoCausalConv3d``); unsupported calls stay on MIOpen, a shape that failed once is not retried, and OOM /
torch._dynamo errors are re-raised. ``UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D``: 1 on for any gfx11, 0 off, unset = only
``DEFAULT_ON_ARCHES``; ``UNSLOTH_DIFFUSION_ROCM_AITER_CONV3D=1`` is an alias for 1. 3x3x3 convs run the direct
(cblocked / general) kernels, bf16 per-conv error 0.70x MIOpen's; ``UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_WINOGRAD=1``
opts into upstream's H/W Winograd routes (faster on wide layers, but 1.45-1.48x MIOpen's error, so never by default).

Cold Triton cache: a call whose compiled variants are not on disk yet stays on MIOpen (counted ``fallback_cold``)
instead of compiling inline (135 s Linux / 277 s Windows on the first 480p Wan decode). ``prebuild`` (from the video
load, ``diffusion_vae_prebuild.maybe_kick_video``) compiles every layer's variants in a child process on tiny inputs
(the direct kernels take spatial sizes at runtime, so a tiny input compiles the binary a real decode loads) and lists
them in a manifest beside the Triton cache; calls route once their variant is listed. A variant first seen during a
decode (a tuned literal shape, an encoder conv) is built by a child after the decode goes quiet.
``UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_PREBUILD=0`` (or ``UNSLOTH_DIFFUSION_VAE_PREBUILD=0``) compiles inline instead.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

ENABLE_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D"
LEGACY_ENABLE_ENV = "UNSLOTH_DIFFUSION_ROCM_AITER_CONV3D"
SELFCHECK_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_SELFCHECK"
WINOGRAD_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_WINOGRAD"
PREBUILD_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_PREBUILD"
# Self-check bound on rel L2 vs fp32 F.conv3d: observed <= 3.4e-3 in bf16 on gfx1151 (bf16 output rounding alone is
# ~1e-3); a broken tile, mask or stride is O(1).
SELFCHECK_REL_L2 = 1e-2
# Archs / VAE classes routed with the env unset (measured end to end).
DEFAULT_ON_ARCHES: tuple = ("gfx1151",)
DEFAULT_ON_VAES: tuple = ("AutoencoderKLWan",)
# Archs whose kernel tables were tuned (ROCm/aiter PR #6973); other gfx11 archs borrow them (explicit opt-in only).
TUNED_ARCHES = ("gfx1151",)
# Free memory kept back beyond the workspace: the larger of these.
MEM_MARGIN_BYTES = 512 << 20
MEM_MARGIN_FRACTION = 0.03
# mem_get_info is a driver call (~0.2 ms): only calls whose workspace could matter pay it.
MEM_CHECK_MIN_BYTES = 64 << 20
# Measured Triton releases (Linux 3.6, Windows triton-windows 3.8); older ones stay on MIOpen.
MIN_TRITON = (3, 6)
_INT32_MAX = 2**31 - 1
_BLOCK = 64

COUNTS = {
    "routed": 0,
    "fallback_error": 0,
    "fallback_shape": 0,
    "fallback_memory": 0,
    "fallback_cold": 0,
}
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
_LOCK = threading.Lock()
_STATE: dict = {}
_FAILED: set = set()
_DEVICE_OK: dict = {}
# Prebuild bookkeeping (one process): variant keys the manifest lists, call -> variant key, variants waiting for a
# child, per-variant child attempts, manifest state.
_READY: set = set()
_VKEYS: dict = {}
_PENDING: dict = {}
_ATTEMPTS: dict = {}
_PB: dict = {}
_PB_LOCK = threading.RLock()
# A decode's cold variants go to one child once no new one appeared for this long.
KICK_DEBOUNCE_S = 3.0
# Manifest re-read at most this often while calls are cold.
MANIFEST_POLL_S = 1.0
# A variant a child failed to build twice stays on MIOpen in this process.
MAX_ATTEMPTS = 2
# Free memory the prebuild child needs (its HIP context plus the largest literal-shape input).
PREBUILD_MIN_FREE = 2 << 30
# Compile budget of one child (Windows compiles about 2x slower than Linux); the manifest keeps partial progress.
PREBUILD_TIMEOUT_S = 1500.0


def reset_counts() -> None:
    for k in COUNTS:
        COUNTS[k] = 0


def _env_mode() -> Optional[bool]:
    raw = (os.environ.get(ENABLE_ENV) or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    if (os.environ.get(LEGACY_ENABLE_ENV) or "").strip().lower() in _TRUE:
        return True
    return None


def winograd_enabled() -> bool:
    """Opt-in only: the Winograd routes exceed MIOpen's bf16 error by ~1.5x, the direct routes stay below it."""
    return (os.environ.get(WINOGRAD_ENV) or "").strip().lower() in _TRUE


def arch_enabled(arch: str) -> bool:
    """Whether a card of ``arch`` (e.g. ``gfx1151``, ``gfx1151:xnack-``) is routed under the current env."""
    arch = (arch or "").split(":")[0].strip().lower()
    if not arch.startswith("gfx11"):
        return False
    mode = _env_mode()
    if mode is None:
        return arch in DEFAULT_ON_ARCHES
    return mode


def requested() -> bool:
    """Cheap pre-check: False when no arch can be on (env 0, or env unset with no default arch)."""
    mode = _env_mode()
    return bool(DEFAULT_ON_ARCHES) if mode is None else mode


def _device_arch(torch: Any, index: Optional[int]) -> str:
    try:
        props = torch.cuda.get_device_properties(
            index if index is not None else torch.cuda.current_device()
        )
        return str(getattr(props, "gcnArchName", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _device_ok(torch: Any, index: Optional[int]) -> bool:
    key = index if index is not None else -1
    ok = _DEVICE_OK.get(key)
    if ok is None:
        from .rocm_bf16 import is_rocm_torch
        ok = bool(is_rocm_torch(torch)) and arch_enabled(_device_arch(torch, index))
        _DEVICE_OK[key] = ok
    return ok


def _kernels() -> Optional[Any]:
    """The vendored conv3d module, or None when Triton (or its HIP backend) is missing."""
    with _LOCK:
        if "mod" in _STATE:
            return _STATE["mod"]
        mod = None
        if not _triton_ok():
            _STATE["import_error"] = f"Triton older than {MIN_TRITON} or missing"
        else:
            try:
                from .rocm_triton_conv3d import conv3d as mod
            except Exception as exc:  # noqa: BLE001 - no Triton: MIOpen stays
                mod = None
                _STATE["import_error"] = repr(exc)[:300]
        _STATE["mod"] = mod
        return mod


def _route_name(mod: Any, x_shape, w_shape, stride, padding, dilation) -> Any:
    N, C, D, H, W = x_shape
    K, _c, T, R, S = w_shape
    return mod._resolve_route(T, R, S, stride, dilation, N, C, D, H, W, K, "ncdhw", padding = padding)


def plan(
    mod: Any,
    x_shape,
    w_shape,
    elt: int,
    stride,
    padding,
    dilation,
    contiguous: bool = True,
) -> tuple:
    """(route, extra bytes the call allocates beyond MIOpen's output, largest element count any kernel indexes).
    The route follows ``mod.ALLOW_WINOGRAD`` (set by ``install`` from ``WINOGRAD_ENV``).

    y = N*K*OD*P*Q*elt (output, also MIOpen's); copy of x when not contiguous; per route:
      1x1x1      : the squeezed weight
      general    : K-major weight pack  K*ceil(C*T*R*S/64)*64*elt
      cblocked   : NCDHWc input pack    N*ceil(C/64)*64*D*H*W*elt + weight pack K*27*C_pad*elt
      winograd   : V 36*N*D*th*tw*C_pad*2 + M 36*N*OD*th*tw*K*4 + filter 3*36*K*C_pad*2 (+ its fp32 transform
                   3*36*K*C*4), th = ceil(P/4), tw = ceil(Q/4); the cblocked variant adds the NCDHWc input pack.
    """
    N, C, D, H, W = x_shape
    K, _c, T, R, S = w_shape
    OD, P, Q = mod._out_dhw(D, H, W, T, R, S, stride, padding, dilation)
    route = _route_name(mod, x_shape, w_shape, stride, padding, dilation)
    names = route.name
    c_pad = -(-C // _BLOCK) * _BLOCK
    x_numel = N * C * D * H * W
    y_numel = N * K * OD * P * Q
    extra = y_numel * elt + (0 if contiguous else x_numel * elt)
    biggest = max(x_numel, y_numel)
    if names == "ONE_X_ONE_X_ONE":
        extra += K * C * elt
    elif names == "GENERAL":
        extra += K * (-(-(C * T * R * S) // _BLOCK) * _BLOCK) * elt
    else:
        blocked = N * c_pad * D * H * W
        if names in ("CBLOCKED_NCDHW", "WINOGRAD_HW_CBLOCKED"):
            extra += blocked * elt
            biggest = max(biggest, blocked)
        if names == "CBLOCKED_NCDHW":
            extra += K * 27 * c_pad * elt
        else:
            tiles = ((P + 3) // 4) * ((Q + 3) // 4)
            v = 36 * N * D * tiles * c_pad
            m = 36 * N * OD * tiles * K
            extra += v * 2 + m * 4 + 3 * 36 * K * c_pad * 2 + 3 * 36 * K * C * 4
            biggest = max(biggest, v, m)
    return route, extra, biggest


def _free_bytes(torch: Any, device: Any) -> int:
    # Driver-free memory only: reserved-but-unallocated blocks are not counted (private pools, other allocators).
    free, total = torch.cuda.mem_get_info(device)
    return int(free), int(total)


def _selfcheck(torch: Any, mod: Any, device: Any, dtype: Any) -> bool:
    """One small conv per route against F.conv3d in fp32, once per (device, dtype); a miss keeps MIOpen."""
    key = ("selfcheck", str(device), str(dtype), bool(getattr(mod, "ALLOW_WINOGRAD", False)))
    if key in _STATE:
        return _STATE[key]
    ok = True
    if (os.environ.get(SELFCHECK_ENV) or "").strip().lower() not in _FALSE:
        import torch.nn.functional as F

        gen = torch.Generator(device = "cpu").manual_seed(0)
        cases = (  # 1x1x1, general (small C), cblocked, strided cblocked, (3,1,1) general, Winograd-eligible
            ((1, 32, 2, 6, 7), (48, 32, 1, 1, 1), (0, 0, 0), (1, 1, 1)),
            ((1, 8, 3, 6, 6), (16, 8, 3, 3, 3), (1, 1, 1), (1, 1, 1)),
            ((1, 64, 3, 6, 6), (64, 64, 3, 3, 3), (0, 1, 1), (1, 1, 1)),
            ((1, 64, 3, 9, 9), (64, 64, 3, 3, 3), (0, 1, 1), (1, 2, 2)),
            ((1, 32, 5, 6, 6), (32, 32, 3, 1, 1), (1, 0, 0), (2, 1, 1)),
            ((1, 160, 3, 28, 28), (96, 160, 3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )
        try:
            with torch.no_grad():
                for xs, ws, pad, st in cases:
                    x = torch.randn(xs, generator = gen).to(device = device, dtype = dtype)
                    w = (torch.randn(ws, generator = gen) * 0.05).to(device = device, dtype = dtype)
                    b = torch.randn(ws[0], generator = gen).to(device = device, dtype = dtype)
                    y = mod.conv3d(x, w, b, st, pad, (1, 1, 1))
                    ref = F.conv3d(x.float(), w.float(), b.float(), stride = st, padding = pad)
                    rel = float((y.float() - ref).norm() / ref.norm())
                    if not (y.shape == ref.shape and rel < SELFCHECK_REL_L2):
                        ok = False
                        _STATE["selfcheck_error"] = f"{xs}x{ws}: rel L2 {rel:.3g}"
                        break
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - no Triton HIP backend, compile failure, ...
            ok = False
            _STATE["selfcheck_error"] = repr(exc)[:300]
    _STATE[key] = ok
    return ok


def _capturing(torch: Any) -> bool:
    """Never run the first-use self-check or a Triton compile inside CUDA / HIP graph capture."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _triton_ok() -> bool:
    try:
        import triton
        parts = tuple(int(x) for x in str(triton.__version__).split(".")[:2])
        return parts >= MIN_TRITON
    except Exception:  # noqa: BLE001 - no Triton, or an unparsable stub
        return False


def _on_gpu(t: Any) -> bool:
    return bool(t.is_cuda)


def _reraise(torch: Any, exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return type(exc).__module__.startswith("torch._dynamo")


def _make_conv_forward(module: Any, original: Any, logger: Any):
    import torch

    def _conv_forward(input, weight, bias):
        if (
            torch.compiler.is_compiling()
            or not _on_gpu(input)
            or _capturing(torch)
            or input.dim() != 5
            or input.dtype not in (torch.bfloat16, torch.float16)
            or weight.dtype != input.dtype
            or (bias is not None and bias.dtype != input.dtype)
            or module.groups != 1
            or module.padding_mode != "zeros"
            or not isinstance(module.padding, tuple)
            or (torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad))
            or not _device_ok(torch, input.device.index)
        ):
            return original(input, weight, bias)
        mod = _kernels()
        if mod is None:
            return original(input, weight, bias)
        gated = _gated()
        if gated and not _selfcheck_ready(mod, input):
            COUNTS["fallback_cold"] += 1
            return original(input, weight, bias)
        if not _selfcheck(torch, mod, input.device, input.dtype):
            return original(input, weight, bias)
        stride, padding, dilation = (
            tuple(module.stride),
            tuple(module.padding),
            tuple(module.dilation),
        )
        key = (
            tuple(input.shape),
            tuple(weight.shape),
            stride,
            padding,
            dilation,
            input.dtype,
            input.device.index,
        )
        if key in _FAILED:
            COUNTS["fallback_error"] += 1
            return original(input, weight, bias)
        try:
            _route, extra, biggest = plan(
                mod,
                tuple(input.shape),
                tuple(weight.shape),
                input.element_size(),
                stride,
                padding,
                dilation,
                contiguous = input.is_contiguous(),
            )
        except Exception:  # noqa: BLE001 - shape the dispatcher rejects (non-positive output, ...)
            COUNTS["fallback_shape"] += 1
            return original(input, weight, bias)
        if biggest > _INT32_MAX:
            COUNTS["fallback_shape"] += 1
            return original(input, weight, bias)
        if extra >= MEM_CHECK_MIN_BYTES:
            free, total = _free_bytes(torch, input.device)
            if extra + max(MEM_MARGIN_BYTES, int(total * MEM_MARGIN_FRACTION)) > free:
                COUNTS["fallback_memory"] += 1
                return original(input, weight, bias)
        if gated and not _variant_ready(mod, input, weight, bias, key, logger):
            COUNTS["fallback_cold"] += 1
            return original(input, weight, bias)
        try:
            y = mod.conv3d(input, weight, bias, stride, padding, dilation)
        except Exception as exc:  # noqa: BLE001 - MIOpen for this shape from now on
            if _reraise(torch, exc):
                raise
            _FAILED.add(key)
            COUNTS["fallback_error"] += 1
            if logger is not None:
                logger.warning(
                    "diffusion.rocm_triton_conv3d: Triton conv3d failed for x=%s w=%s, MIOpen from now on: %s",
                    key[0],
                    key[1],
                    repr(exc)[:300],
                )
            return original(input, weight, bias)
        COUNTS["routed"] += 1
        return y

    _conv_forward._unsloth_triton_conv3d = True
    return _conv_forward


def prebuild_enabled() -> bool:
    """Cold calls wait on a prebuild child (default) instead of compiling inline."""
    if (os.environ.get(PREBUILD_ENV) or "").strip().lower() in _FALSE:
        return False
    try:
        from . import diffusion_vae_prebuild
    except Exception:  # noqa: BLE001
        return False
    return diffusion_vae_prebuild.enabled()


def _source_digest() -> str:
    import hashlib

    h = hashlib.sha256()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rocm_triton_conv3d")
    for name in sorted(os.listdir(root)):
        if name.endswith(".py"):
            with open(os.path.join(root, name), "rb") as f:
                h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()[:16]


def _manifest_path() -> Optional[str]:
    """Manifest beside the Triton cache, one per (kernel source, torch, HIP, Triton, arch, TRITON_* env): wiping the
    cache drops it, and anything that changes Triton's own cache key starts a new one."""
    with _PB_LOCK:
        if "path" in _PB:
            return _PB["path"]
        path = None
        try:
            import hashlib
            import json

            import torch
            import triton

            from . import diffusion_vae_prebuild

            root = diffusion_vae_prebuild._triton_cache_dir()
            if root:
                key = json.dumps(
                    [
                        _source_digest(),
                        str(torch.__version__),
                        str(triton.__version__),
                        _device_arch(torch, None).split(":")[0],
                        sorted((k, v) for k, v in os.environ.items() if k.startswith("TRITON_")),
                    ]
                )
                digest = hashlib.sha256(key.encode()).hexdigest()[:32]
                path = os.path.join(
                    root, diffusion_vae_prebuild._MARKER_DIR, f"rocm_conv3d_{digest}.json"
                )
        except Exception:  # noqa: BLE001 - no Triton / cache dir: compile inline as before
            path = None
        _PB["path"] = path
        return path


def _gated() -> bool:
    return prebuild_enabled() and _manifest_path() is not None


def _read_manifest(path: str) -> dict:
    import json
    try:
        with open(path, "r", encoding = "utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001 - absent, or mid-replace on Windows: next poll
        pass
    return {}


def _refresh(force: bool = False) -> None:
    """Re-read the manifest when it changed (at most every ``MANIFEST_POLL_S`` unless forced)."""
    path = _manifest_path()
    if path is None:
        return
    now = time.monotonic()
    with _PB_LOCK:
        if not force and now - _PB.get("polled", -1e9) < MANIFEST_POLL_S:
            return
        _PB["polled"] = now
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            return
        if not force and mtime == _PB.get("mtime"):
            return
        data = _read_manifest(path)
        _PB["mtime"] = mtime
        _READY.update(k for k in data.get("keys", []) if isinstance(k, str))
        _PB.setdefault("selfcheck", {}).update(
            {k: bool(v) for k, v in (data.get("selfcheck") or {}).items()}
        )


def _merge_manifest(
    path: str,
    keys = (),
    selfcheck: Optional[dict] = None,
) -> None:
    import json

    os.makedirs(os.path.dirname(path), exist_ok = True)
    data = _read_manifest(path)
    merged = sorted(set(data.get("keys", [])) | set(keys))
    checks = dict(data.get("selfcheck") or {})
    checks.update(selfcheck or {})
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding = "utf-8") as f:
        json.dump({"keys": merged, "selfcheck": checks}, f)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader holds it open
            time.sleep(0.05)
    try:
        os.remove(tmp)
    except OSError:
        pass


def _sc_key(mod: Any, dtype: Any) -> str:
    return f"{dtype}|{bool(getattr(mod, 'ALLOW_WINOGRAD', False))}"


def _selfcheck_ready(mod: Any, input: Any) -> bool:
    """Gated: the self-check runs in this process only once a child compiled (and passed) it."""
    import torch

    state_key = (
        "selfcheck",
        str(input.device),
        str(input.dtype),
        bool(getattr(mod, "ALLOW_WINOGRAD", False)),
    )
    if state_key in _STATE:
        return True
    sk = _sc_key(mod, input.dtype)
    if sk not in _PB.get("selfcheck", {}):
        _refresh()
    ok = _PB.get("selfcheck", {}).get(sk)
    if ok is None:
        _note_pending(
            "selfcheck|" + sk,
            {
                "selfcheck": str(input.dtype).replace("torch.", ""),
                "device": _index(torch, input.device),
            },
        )
        return False
    if not ok:
        _STATE[state_key] = False  # the child's self-check failed: MIOpen, nothing compiled here
    return True


def _index(torch: Any, device: Any) -> int:
    return int(device.index if device.index is not None else torch.cuda.current_device())


_DIRECT_ROUTES = ("ONE_X_ONE_X_ONE", "GENERAL", "CBLOCKED_NCDHW")


def _variant_key(
    mod: Any, x_shape, w_shape, stride, padding, dilation, dtype, has_bias, aligned
) -> str:
    """Key of the compiled variants ``mod.conv3d`` launches for this call, computed without compiling.

    The direct kernels take N / D / H / W / OD / P / Q / M as runtime ints (``do_not_specialize``), so a call needs
    the binaries of any other call with the same route, channels, kernel, stride, padding, dilation, dtype, bias
    presence, tile configs (the table's ``any`` entry or a tuned literal shape), input pointer alignment and, for the
    NCDHWc pack, whether ``D*H*W`` is a multiple of 16 (Triton specialises that int). Winograd keeps spatial
    constexprs: its key is the exact shape."""
    import json

    from .rocm_triton_conv3d._config import format_prepack_shape_key_3d, format_shape_key_3d
    from .rocm_triton_conv3d.conv3d_1x1x1 import _get_config as config_1x1x1
    from .rocm_triton_conv3d.conv3d_3x3x3 import _get_config_cblocked as config_cblocked
    from .rocm_triton_conv3d.conv3d_general import _get_config as config_general
    from .rocm_triton_conv3d.ncx_to_cblocked import _get_config_3d as config_prepack

    x_shape, w_shape = tuple(int(v) for v in x_shape), tuple(int(v) for v in w_shape)
    stride, padding, dilation = tuple(stride), tuple(padding), tuple(dilation)
    N, C, D, H, W = x_shape
    K, _c, T, R, S = w_shape
    route = mod._resolve_route(
        T, R, S, stride, dilation, N, C, D, H, W, K, "ncdhw", padding = padding
    ).name
    head = [
        route,
        C,
        K,
        T,
        R,
        S,
        stride,
        padding,
        dilation,
        str(dtype),
        bool(has_bias),
        bool(aligned),
    ]
    if route not in _DIRECT_ROUTES:
        return json.dumps(head + [list(x_shape)])
    OD, P, Q = mod._out_dhw(D, H, W, T, R, S, stride, padding, dilation)
    M = N * OD * P * Q
    tail = None
    if route == "ONE_X_ONE_X_ONE":  # the launcher's key: T = R = S = 1, dilation 1
        key = format_shape_key_3d(N, C, D, H, W, K, 1, 1, 1, *stride, *padding, 1, 1, 1)
        configs = [config_1x1x1(shape_key = key, M = M, variants = ("ncdhw",))]
    elif route == "GENERAL":
        key = format_shape_key_3d(N, C, D, H, W, K, T, R, S, *stride, *padding, *dilation)
        configs = [config_general(shape_key = key, M = M, variants = ("ncdhw",))]
    else:
        key = format_shape_key_3d(N, C, D, H, W, K, 3, 3, 3, *stride, *padding, *dilation)
        pack = format_prepack_shape_key_3d(N, C, D, H, W, _BLOCK)
        configs = [config_prepack(shape_key = pack, M = D * H * W), config_cblocked(shape_key = key, M = M)]
        tail = (D * H * W) % 16 == 0
    return json.dumps(head + [[sorted(c.items()) for c in configs], tail])


def _literal_inputs(mod: Any, w_shape, stride, padding, dilation) -> list:
    """Input shapes of the tuned literal table entries matching this layer (a config other than ``any``)."""
    from .rocm_triton_conv3d._config import _table

    K, C, T, R, S = (int(v) for v in w_shape)
    want = dict(zip(("C", "K", "T", "R", "S"), (C, K, T, R, S)))
    want.update(zip(("sd", "sh", "sw"), stride))
    want.update(zip(("pd", "ph", "pw"), padding))
    want.update(zip(("dd", "dh", "dw"), dilation))
    out: list = []
    for name in ("CONV3D-1X1X1", "CONV3D-GENERAL", "CONV3D-3X3X3-CBLOCKED", "CONV3D-PREPACK"):
        table = _table(name) or {}
        for section in ("shapes_ncdhw", "shapes"):
            for key in table.get(section, {}):
                try:
                    f = {
                        k.strip(): int(v) for k, _, v in (p.partition("=") for p in key.split(","))
                    }
                except ValueError:
                    continue
                if name == "CONV3D-PREPACK":
                    if f.get("C") != C or (T, R, S) != (3, 3, 3):
                        continue
                elif any(f.get(k) != v for k, v in want.items()):
                    continue
                shape = (f["N"], C, f["D"], f["H"], f["W"])
                if shape not in out:
                    out.append(shape)
    return out


def _tiny_inputs(x_shape, w_shape, stride, padding, dilation):
    """Small inputs of the same layer (N=1, a few output voxels): same variant key as most real calls."""
    _n, C = x_shape[0], x_shape[1]
    K, _c, *ks = w_shape
    base = [max(1, d * (k - 1) + 1 - 2 * p) for k, p, d in zip(ks, padding, dilation)]
    for a in range(2):
        for b in range(4):
            for c in range(4):
                yield (1, C, base[0] + a, base[1] + b, base[2] + c)


def _spec(x_shape, w_shape, stride, padding, dilation, dtype, has_bias, aligned, device) -> dict:
    return {
        "x": list(x_shape),
        "w": list(w_shape),
        "stride": list(stride),
        "padding": list(padding),
        "dilation": list(dilation),
        "dtype": str(dtype).replace("torch.", ""),
        "bias": bool(has_bias),
        "aligned": bool(aligned),
        "device": int(device),
    }


def _shrink(mod: Any, spec: dict, vkey: str) -> dict:
    args = (tuple(spec["stride"]), tuple(spec["padding"]), tuple(spec["dilation"]))
    for x in _tiny_inputs(spec["x"], spec["w"], *args):
        try:
            if (
                _variant_key(
                    mod,
                    x,
                    spec["w"],
                    *args,
                    "torch." + spec["dtype"],
                    spec["bias"],
                    spec["aligned"],
                )
                == vkey
            ):
                return dict(spec, x = list(x))
        except Exception:  # noqa: BLE001 - a tiny shape the dispatcher rejects
            continue
    return spec


def _variant_ready(mod: Any, input: Any, weight: Any, bias: Any, key: tuple, logger: Any) -> bool:
    """Gated: True when this call's compiled variants are listed in the manifest; else queue them for a child."""
    import torch

    aligned = (not input.is_contiguous()) or input.data_ptr() % 16 == 0
    ck = key + (bias is not None, aligned)
    vk = _VKEYS.get(ck)
    if vk is None:
        try:
            vk = _variant_key(
                mod, key[0], key[1], key[2], key[3], key[4], input.dtype, bias is not None, aligned
            )
        except Exception:  # noqa: BLE001 - unknown variant: MIOpen, never an inline compile
            vk = ""
        _VKEYS[ck] = vk
    if not vk:
        return False
    if vk in _READY:
        return True
    _refresh()
    if vk in _READY:
        return True
    spec = _spec(
        key[0],
        key[1],
        key[2],
        key[3],
        key[4],
        input.dtype,
        bias is not None,
        aligned,
        _index(torch, input.device),
    )
    _note_pending(vk, spec, mod = mod, logger = logger)
    return False


def _note_pending(
    vk: str,
    spec: dict,
    mod: Any = None,
    logger: Any = None,
) -> None:
    with _PB_LOCK:
        if vk in _PENDING:
            # A kick skipped for low free memory leaves its batch pending with no timer: re-arm on later cold calls.
            if _PB.get("timer") is None and not _PB.get("live"):
                _arm_timer()
            return
        if vk in _READY or vk in _PB.get("inflight", ()):
            return
        if _ATTEMPTS.get(vk, 0) >= MAX_ATTEMPTS:
            return
        if mod is not None and "x" in spec:
            spec = _shrink(mod, spec, vk)
        _PENDING[vk] = spec
        if logger is not None:
            _PB["logger"] = logger
        _arm_timer()


def _arm_timer() -> None:
    with _PB_LOCK:
        old = _PB.get("timer")
        if old is not None:
            old.cancel()
        t = threading.Timer(KICK_DEBOUNCE_S, _kick)
        t.daemon = True
        _PB["timer"] = t
        t.start()


def _kick(logger: Any = None) -> bool:
    """Hand the pending variants of one device to a prebuild child (one child at a time). Never raises."""
    try:
        import torch

        from . import diffusion_vae_prebuild

        with _PB_LOCK:
            _PB["timer"] = None
            logger = logger or _PB.get("logger")
            if _PB.get("live") or not _PENDING:
                return False
            device = next(iter(_PENDING.values()))["device"]
            free, _total = torch.cuda.mem_get_info(device)
            if free < PREBUILD_MIN_FREE:
                return False  # stays pending: the next cold call re-arms
            batch = {k: s for k, s in _PENDING.items() if s["device"] == device}
            for k in batch:
                del _PENDING[k]
                _ATTEMPTS[k] = _ATTEMPTS.get(k, 0) + 1
            mod = _kernels()
            job = {
                "name": "rocm conv3d",
                "owner": "video",
                "device": device,
                "manifest": _manifest_path(),
                "winograd": bool(getattr(mod, "ALLOW_WINOGRAD", False)),
                "selfcheck": sorted({s["selfcheck"] for s in batch.values() if "selfcheck" in s}),
                "specs": [dict(s, key = k) for k, s in batch.items() if "x" in s],
            }
            _PB["live"] = True
            _PB["inflight"] = set(batch)
        ok = diffusion_vae_prebuild._spawn(
            job,
            logger,
            target = (__name__, "_prebuild_child"),
            timeout = PREBUILD_TIMEOUT_S,
            on_exit = _child_exit,
        )
        if not ok:
            _child_exit(None)
        return ok
    except Exception as exc:  # noqa: BLE001 - a prebuild, never a failed decode
        with _PB_LOCK:
            _PB["live"] = False
            _PB["inflight"] = set()
        if logger is not None:
            logger.info(
                "diffusion.rocm_triton_conv3d: prebuild skipped (%s: %s)", type(exc).__name__, exc
            )
        return False


def _child_exit(_exitcode: Any = None) -> None:
    with _PB_LOCK:
        _PB["live"] = False
        _PB["inflight"] = set()
    _refresh(force = True)
    with _PB_LOCK:
        for k in [k for k in _PENDING if k in _READY]:
            del _PENDING[k]
        if _PENDING:
            _arm_timer()


def cancel_pending() -> None:
    """Unload: no new child for this VAE's variants (``diffusion_vae_prebuild.cancel_all`` kills a live one)."""
    with _PB_LOCK:
        t = _PB.get("timer")
        if t is not None:
            t.cancel()
        _PB["timer"] = None
        _PENDING.clear()


def _ordered_convs(vae: Any) -> list:
    """Wrapped convs, decoder side first (the first render decodes; an encoder runs only for image input)."""
    named = list(getattr(vae, "named_modules", lambda: [])())
    wrapped = [
        (n, m)
        for n, m in named
        if getattr(m.__dict__.get("_conv_forward"), "_unsloth_triton_conv3d", False)
    ]
    return [
        m for n, m in wrapped if not n.startswith("encoder") and not n.startswith("quant_conv")
    ] + [m for n, m in wrapped if n.startswith("encoder") or n.startswith("quant_conv")]


def prebuild(vae: Any, logger: Any = None) -> int:
    """After the VAE is placed: queue every wrapped conv's variants (tiny inputs plus tuned literal shapes) for a
    prebuild child and start it. Returns the number of variants queued. Never blocks, never raises."""
    try:
        if vae is None or not _gated():
            return 0
        import torch

        mod = _kernels()
        if mod is None:
            return 0
        _refresh(force = True)
        queued: dict = {}
        for m in _ordered_convs(vae):
            w = getattr(m, "weight", None)
            if w is None or not _on_gpu(w) or w.dtype not in (torch.bfloat16, torch.float16):
                continue
            if not _device_ok(torch, w.device.index):
                continue
            device = _index(torch, w.device)
            sk = _sc_key(mod, w.dtype)
            if sk not in _PB.get("selfcheck", {}):
                queued.setdefault(
                    "selfcheck|" + sk,
                    {"selfcheck": str(w.dtype).replace("torch.", ""), "device": device},
                )
            args = (tuple(m.stride), tuple(m.padding), tuple(m.dilation))
            has_bias = m.bias is not None
            shapes = list(_tiny_inputs((1, w.shape[1]), tuple(w.shape), *args))
            shapes += _literal_inputs(mod, tuple(w.shape), *args)
            for x in shapes:
                try:
                    vk = _variant_key(mod, x, tuple(w.shape), *args, w.dtype, has_bias, True)
                except Exception:  # noqa: BLE001
                    continue
                if vk in _READY or vk in queued:
                    continue
                queued[vk] = _spec(x, tuple(w.shape), *args, w.dtype, has_bias, True, device)
        with _PB_LOCK:
            for vk, spec in queued.items():
                if vk not in _PENDING and _ATTEMPTS.get(vk, 0) < MAX_ATTEMPTS:
                    _PENDING[vk] = spec
            if logger is not None:
                _PB["logger"] = logger
        n = sum(1 for k in queued if not k.startswith("selfcheck|"))
        if queued:
            _kick(logger)
            if logger is not None:
                logger.info(
                    "diffusion.rocm_triton_conv3d: prebuilding %d conv3d variants in a child", n
                )
        return n
    except Exception as exc:  # noqa: BLE001 - a prebuild, never a failed load
        if logger is not None:
            logger.info(
                "diffusion.rocm_triton_conv3d: prebuild skipped (%s: %s)", type(exc).__name__, exc
            )
        return 0


def _child_device(torch: Any, index: int) -> Any:
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _prebuild_child(job: dict) -> None:
    """Child side: compile each variant on zeros (no weights), self-check first; the manifest grows per variant."""
    import torch

    dev = _child_device(torch, int(job["device"]))
    mod = _kernels()
    if mod is None:
        return
    mod.ALLOW_WINOGRAD = bool(job.get("winograd"))
    path = job["manifest"]
    for name in job.get("selfcheck", []):
        dtype = getattr(torch, name)
        ok = _selfcheck(torch, mod, dev, dtype)
        _merge_manifest(path, selfcheck = {_sc_key(mod, dtype): ok})
    with torch.no_grad():
        for spec in job.get("specs", []):
            dtype = getattr(torch, spec["dtype"])
            x_shape, w_shape = tuple(spec["x"]), tuple(spec["w"])
            args = (tuple(spec["stride"]), tuple(spec["padding"]), tuple(spec["dilation"]))
            try:
                numel = 1
                for v in x_shape:
                    numel *= v
                if spec.get("aligned", True):
                    x = torch.zeros(x_shape, device = dev, dtype = dtype)
                else:  # one element past a fresh allocation: the misaligned variant
                    x = torch.zeros(numel + 8, device = dev, dtype = dtype)[1 : numel + 1].view(x_shape)
                aligned = x.data_ptr() % 16 == 0
                if (
                    _variant_key(mod, x_shape, w_shape, *args, dtype, spec["bias"], aligned)
                    != spec["key"]
                ):
                    continue
                w = torch.zeros(w_shape, device = dev, dtype = dtype)
                b = torch.zeros(w_shape[0], device = dev, dtype = dtype) if spec["bias"] else None
                mod.conv3d(x, w, b, *args)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                del x, w, b
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception:  # noqa: BLE001 - this variant stays on MIOpen
                continue
            _merge_manifest(path, keys = (spec["key"],))


def install(vae: Any, logger: Any = None) -> int:
    """Wrap every eligible ``nn.Conv3d`` of ``vae`` (per instance). Returns the number wrapped; 0 when off, not ROCm
    gfx11, or Triton is missing. The device is checked again at call time (speed optims run before placement)."""
    if vae is None or not requested() or not callable(getattr(vae, "modules", None)):
        return 0
    if _env_mode() is None and type(vae).__name__ not in DEFAULT_ON_VAES:
        return 0
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch
    except Exception:  # noqa: BLE001
        return 0
    if not is_rocm_torch(torch) or not torch.cuda.is_available():
        return 0
    arch = _device_arch(torch, None)
    if not arch_enabled(arch):
        return 0
    if _kernels() is None:
        if logger is not None:
            logger.info(
                "diffusion.rocm_triton_conv3d: Triton conv3d unavailable (%s)",
                _STATE.get("import_error"),
            )
        return 0
    mod = _kernels()
    mod.ALLOW_WINOGRAD = winograd_enabled()
    n = 0
    for m in vae.modules():
        if not isinstance(m, torch.nn.Conv3d) or m.groups != 1 or m.padding_mode != "zeros":
            continue
        current = m._conv_forward
        if getattr(current, "_unsloth_triton_conv3d", False):
            continue
        m._conv_forward = _make_conv_forward(m, current, logger)
        n += 1
    if n and logger is not None:
        tuned = arch.split(":")[0] in TUNED_ARCHES
        logger.info(
            "diffusion.rocm_triton_conv3d: %d %s Conv3d routed to the Triton conv3d on %s%s%s",
            n,
            type(vae).__name__,
            arch,
            "" if tuned else " (gfx1151 kernel tables)",
            " (Winograd routes on)" if mod.ALLOW_WINOGRAD else "",
        )
    return n


def uninstall(vae: Any) -> int:
    """Unwrap ``vae`` and drop the weight packs (they also go when their weight is freed: weak references)."""
    cancel_pending()
    n = 0
    for m in getattr(vae, "modules", lambda: [])():
        fwd = m.__dict__.get("_conv_forward")
        if getattr(fwd, "_unsloth_triton_conv3d", False):
            del m.__dict__["_conv_forward"]
            n += 1
    if n and _STATE.get("mod") is not None:
        try:
            from .rocm_triton_conv3d._prepack import clear_conv3d_weight_pack_caches
            clear_conv3d_weight_pack_caches()
        except Exception:  # noqa: BLE001 - stand-in module / no Triton: nothing cached
            pass
    return n

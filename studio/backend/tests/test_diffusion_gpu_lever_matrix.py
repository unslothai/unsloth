# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A PR that silently costs any GPU a speed lever fails here, on a CPU runner, including on future GPUs.

An external tester re-runs Studio against a reference stack on an RTX 6000 Ada (sm89) and an RTX 3090 (sm86) after
each batch of merges, and almost every round found a GPU x model cell that got slower because a load-time decision
changed: a per-arch policy table (#19098 turned H3 video Auto INT8 attention off on sm89 only: the Ada cell lost 25%),
a race at the wrong shape, a fast path no longer armed, a plan that ignored the VRAM fit or the link. Those decisions
are pure functions of the card, so this file resolves them for every hardware profile in ``gpu_profiles.py`` (real
cards plus synthetic future archs) with torch mocked to that card, and diffs the result against a reviewed golden,
``fixtures/diffusion_gpu_lever_golden.json``:

(a) the lever table: every arch-gated policy function (INT8 attention / GEMM / ConvRot / grouped / CUTLASS tables,
    the H3 attention gates, VAE strips, quant ladders, NVFP4 routes, ...) per profile, head dim and dtype: engaged or
    off, the config it engages with, a reason code;
(b) the speed layer per profile: every registered image / video family through ``install_patches`` +
    ``run_speed_layer`` (the parity harness's entry) at the default speed tier and the dtype the profile resolves.

A lever lost (engaged to off, a config gone, a cell gone) fails as ``GPU LEVER REGRESSION <profile>.<family>.<lever>:
expected X, got Y``; gains and config changes fail too, so every change is reviewed. Update an intended change with
``UNSLOTH_UPDATE_GPU_GOLDEN=1 UNSLOTH_GPU_GOLDEN_REASON="..."`` (the reason and the changed cells go to the golden's
``changes`` ledger; an empty reason refuses). Both tables run in one child process with diffusers hidden (Backend CI
has none), so every host resolves the same golden; the diffusers patch layer stays with test_diffusion_autoopt_parity.

Future archs: every lever a synthetic profile's parent engages must engage on the child or be listed in
``FUTURE_ARCH_EXCLUSIONS`` with a reason. The incident replays at the end assert, one per tester regression class,
the decision that would have caught it.
"""

from __future__ import annotations

import contextlib as _contextlib
import itertools
import os
import subprocess
import sys
import types as _types
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

_TESTS = Path(__file__).resolve().parent
_BACKEND = _TESTS.parent
for _p in (str(_BACKEND), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_profiles import GIB, PROFILES, GpuProfile, clean_policy_env, mock_profile  # noqa: E402

# One worker builds the matrix once (a module fixture per xdist worker would rebuild it on each).
pytestmark = pytest.mark.xdist_group("gpu_lever_matrix")


def lever(
    on: Any,
    cfg: Any = None,
    why: Optional[str] = None,
) -> dict:
    """One golden cell: engaged or off, the config it engages with (compact), a stable reason code."""
    out: dict = {"on": bool(on)}
    if cfg is not None:
        out["cfg"] = fmt(cfg)
    if why:
        out["why"] = str(why)
    return out


def fmt(cfg: Any) -> Any:
    """Compact, stable text for a config: tuples join with x, dicts sort as k=v, nested values recurse."""
    if cfg is None or isinstance(cfg, (bool, int, float, str)):
        return cfg
    if isinstance(cfg, dict):
        return ",".join(f"{k}={fmt(cfg[k])}" for k in sorted(cfg, key = str))
    if isinstance(cfg, (tuple, list, set, frozenset)):
        items = sorted(cfg, key = str) if isinstance(cfg, (set, frozenset)) else list(cfg)
        return (
            "x".join(str(fmt(v)) for v in items)
            if all(isinstance(v, (int, float)) for v in items)
            else "[" + ";".join(str(fmt(v)) for v in items) + "]"
        )
    return repr(cfg)


def nvidia(p: GpuProfile) -> bool:
    return p.vendor == "nvidia"


# A probe takes a profile (called inside ``mock_profile(p)`` and ``clean_policy_env()``) and returns
# ``{"<family>.<lever>": lever(...)}``. It returns the SAME keys for every profile (a key missing on some profile is
# a lost cell) and never raises: non-applicable platforms get ``lever(False, why = "n/a:<platform>")``.
LEVER_PROBES: list = []


def probe(fn: Callable) -> Callable:
    LEVER_PROBES.append(fn)
    return fn


def _na_code(p: GpuProfile) -> str:
    return "n/a:" + (
        "rocm" if p.rocm else "mps" if p.mps else "cpu" if p.vendor == "cpu" else p.vendor
    )


def _probe_target(p: GpuProfile) -> Any:
    """The load target a profile resolves to (device, dtype, ordinal), built like diffusion_device would."""
    import types

    import torch

    if p.cuda:
        bf16 = p.rocm or (p.capability or (0, 0)) >= (8, 0)
        return types.SimpleNamespace(
            device = "cuda",
            backend = "rocm" if p.rocm else "cuda",
            dtype = torch.bfloat16 if bf16 else torch.float16,
            ordinal = 0,
            torch_device = "cuda:0",
        )
    if p.mps:
        return types.SimpleNamespace(
            device = "mps", backend = "mps", dtype = torch.bfloat16, ordinal = None, torch_device = "mps"
        )
    return types.SimpleNamespace(
        device = "cpu", backend = "cpu", dtype = torch.float32, ordinal = None, torch_device = "cpu"
    )


@probe
def probe_int8_attention(p: GpuProfile) -> dict:
    """Which INT8 attention kernel serves this card per head dim (the CUDA table, the Triton stand-in, ROCm Triton)."""
    from core.inference import diffusion_int8_attention as i8

    out: dict = {}
    for hd in (64, 128):
        if not p.cuda:
            out[f"any.int8_attn.serving.hd{hd}"] = lever(False, why = _na_code(p))
            out[f"any.int8_attn.triton_table.hd{hd}"] = lever(False, why = _na_code(p))
            continue
        impl = i8.implementation(0, hd)
        cfg = i8.device_config(0, hd)
        why = None if impl else "no_kernel_config"
        nearest = getattr(i8, "nearest_kernel_config", None)
        if impl == "cuda" and nearest is not None:
            src = nearest(p.capability, hd, p.smem_optin)[0]
            if src is not None and tuple(src) != tuple(p.capability):
                why = f"nearest:sm{src[0]}{src[1]}"  # an unlisted arch offered a measured card's config (raced)
        if (
            impl is None
            and not p.rocm  # ROCm serves its own Triton table; NVRTC never builds for it
            and nearest is not None
            and nearest(p.capability, hd, p.smem_optin)[0] is not None
        ):
            from core.inference import diffusion_int8_attention_cuda as i8c
            if not i8c.nvrtc_arch_supported(p.capability):
                why = "no_nvrtc_target"  # a measured config fits, but the pinned NVRTC cannot build for the arch
        out[f"any.int8_attn.serving.hd{hd}"] = lever(
            impl is not None, cfg = f"{impl}:{fmt(cfg)}" if impl else None, why = why
        )
        if p.rocm:
            out[f"any.int8_attn.triton_table.hd{hd}"] = lever(False, why = "n/a:rocm")
        else:
            tri = i8.triton_kernel_config(p.capability, hd)
            out[f"any.int8_attn.triton_table.hd{hd}"] = lever(
                tri is not None, cfg = tri, why = None if tri else "no_triton_config"
            )
    return out


@probe
def probe_int8_gemm(p: GpuProfile) -> dict:
    """The fused INT8 GEMM: arch tile, unlisted-arch race, CUTLASS prebuilt, TMA tile, per-shape rules; the install
    gate is ``diffusion_int8_gemm.install``'s (CUDA, not ROCm, prebuilt or Triton, then the arch / unlisted gates)."""
    import torch

    from core.inference import diffusion_int8_cutlass as cut
    from core.inference import diffusion_int8_gemm as g
    from core.inference.rocm_bf16 import is_rocm_torch

    keys = (
        "install_gate",
        "arch_tile",
        "unlisted_race",
        "cutlass_prebuilt",
        "cutlass_family",
        "tma_tile",
        "shape_tiles",
    )
    if not p.cuda:
        return {f"any.int8_gemm.{k}": lever(False, why = _na_code(p)) for k in keys}
    cap = tuple(torch.cuda.get_device_capability(0))
    rocm = is_rocm_torch(torch)
    arch = g.arch_config(cap)
    unl = g.unlisted_wanted(cap)
    pre = g.cutlass_wanted(cap)
    gemm, rotq = g._gates(cap)
    out: dict = {}
    if rocm:
        out["any.int8_gemm.install_gate"] = lever(False, why = "rocm")
    elif not pre and not g._triton_version_ok():
        out["any.int8_gemm.install_gate"] = lever(False, why = "no_triton")
    else:
        routes = [
            n
            for n, on in (("arch", gemm), ("unlisted", unl), ("cutlass", pre), ("rotq", rotq))
            if on
        ]
        # The cell is the real installer's outcome on a denoiser with one eligible int8 Linear (its mode, kernel
        # availability, arch gates and eligible-Linear scan, and the first-forward swap it arms); cfg names the routes.
        installed = _int8_gemm_installs() if routes else False
        out["any.int8_gemm.install_gate"] = lever(
            bool(routes) and installed,
            cfg = "+".join(routes) or None,
            why = None if (routes and installed) else ("installs_nothing" if routes else "no_route"),
        )
    out["any.int8_gemm.arch_tile"] = lever(
        arch is not None, cfg = arch, why = None if arch else "no_arch_tile"
    )
    if g.unlisted_arch(cap) and not rocm:
        why = None if unl else ("unlisted_skip" if cap in g._UNLISTED_SKIP else "triton38_sm8x")
    else:
        why = "rocm" if rocm else "listed_arch" if arch is not None else "below_sm80"
    out["any.int8_gemm.unlisted_race"] = lever(unl and not rocm, why = why)
    fam = cut.family(cap) if not rocm else None
    # The prebuilt wheel counts as installed under the mock (Studio's installer puts it on NVIDIA); a real host
    # without it reads ``no_wheel`` here, not ``no_family``.
    no_pre = "rocm" if rocm else ("no_family" if cut.family(cap) not in (75, 80) else "no_wheel")
    out["any.int8_gemm.cutlass_prebuilt"] = lever(pre, why = None if pre else no_pre)
    out["any.int8_gemm.cutlass_family"] = lever(
        fam is not None, cfg = fam, why = None if fam else "no_family"
    )
    tma = g.tma_config(cap)
    out["any.int8_gemm.tma_tile"] = lever(
        tma is not None and not rocm, cfg = tma, why = None if tma else "no_tma_tile"
    )
    tiles = g.shape_tiles(cap)
    out["any.int8_gemm.shape_tiles"] = lever(
        bool(tiles) and not rocm,
        cfg = len(tiles) if tiles else None,
        why = None if tiles else "no_shape_rules",
    )
    return out


class _PerRow:
    pass


_PerRow.__name__ = "PerRow"


def _int8_weight(n: int, k: int) -> Any:
    """A torchao ``Int8Tensor`` as the INT8 transformer quant leaves it (dynamic symmetric per-row activation quant,
    bf16 per-row scale), duck-typed so the fixture needs no torchao on the host: only its fields are read."""
    import enum

    import torch

    mapping = enum.Enum("MappingType", "SYMMETRIC ASYMMETRIC").SYMMETRIC
    w = type("Int8Tensor", (), {})()
    w.qdata = torch.zeros(n, k, dtype = torch.int8)
    w.scale = torch.ones(n, 1, dtype = torch.bfloat16)
    w.block_size = [1, k]
    w.act_quant_kwargs = _types.SimpleNamespace(
        mapping_type = mapping, granularity = _PerRow(), reduce_range = False
    )
    w.act_pre_scale = w.act_quant_scale = w.act_quant_zero_point = w.zero_point = None
    return w


def _int8_gemm_installs() -> bool:
    """``diffusion_int8_gemm.install()`` on a stub denoiser holding one eligible int8 Linear (weights not on the card
    yet, as at load): True when it counts the Linear and arms the first-forward swap."""
    from torch import nn

    from core.inference import diffusion_int8_gemm as g
    from core.inference.diffusion_int8_fused import cancel_first_call

    stub = nn.Sequential(nn.Linear(256, 256, bias = False))
    stub[0]._parameters["weight"] = _int8_weight(256, 256)
    try:
        n = int(g.install(stub) or 0)
    except Exception:  # noqa: BLE001 - an installer that raises installs nothing
        return False
    armed = "int8_gemm" in stub.__dict__.get("_unsloth_first_call_hooks", {})
    cancel_first_call(stub, "int8_gemm")
    return n > 0 and armed


# Production GEMM shapes (M rows, N, K): H3 video fused QKV and FFN down at a 960x544x124 render, Qwen-Image-2.1 MLP
# in / down at 1024 px, and a short call below the per-shape rules' row floor.
GEMM_SHAPES = {
    "h3_qkv": (19318, 21504, 5376),
    "h3_ffn_down": (19318, 5376, 14336),
    "q21_mlp_in": (4096, 12288, 3072),
    "q21_mlp_down": (4096, 3072, 12288),
    "short": (512, 3072, 3072),
}


@probe
def probe_int8_gemm_tiles(p: GpuProfile) -> dict:
    """The tile ``tile_for`` serves each production shape with on a listed arch, after the real ``device_config``
    with every on-device probe passing (bit-exact on the real card): a moved shape rule changes the cell. Unlisted
    archs race their tile at load (``int8_gemm.unlisted_race``), so they record that instead."""
    keys = [f"any.int8_gemm.tile.{name}" for name in GEMM_SHAPES]
    if p.vendor != "nvidia":
        return {k: lever(False, why = _na_code(p)) for k in keys}
    import torch

    from core.inference import diffusion_int8_gemm as g

    cap = tuple(torch.cuda.get_device_capability(0))
    if g.arch_config(cap) is None:
        why = "raced_at_load" if g.unlisted_wanted(cap) else "no_route"
        return {k: lever(False, why = why) for k in keys}
    saved = {
        name: dict(getattr(g, name))
        if isinstance(getattr(g, name), dict)
        else set(getattr(g, name))
        for name in ("_DEVICE_CFG", "_DEVICE_TILES", "_DEVICE_TMA", "_UNLISTED_CARDS")
    }
    pins = {
        "_kernels": lambda: _types.SimpleNamespace(),
        "_op": lambda: object(),
        "_settled_probe": lambda index, cfg: True,
        "_stored_cfg": lambda index, cap, want: None,
        "_store_cfg": lambda *a, **k: None,
        "_tma_probe": lambda index, capability: g.tma_config(capability),
    }
    old = {k: getattr(g, k) for k in pins}
    try:
        for k, v in pins.items():
            setattr(g, k, v)
        for name in saved:
            getattr(g, name).clear()
        g.device_config(0)
        out = {}
        for name, (m, n, k) in GEMM_SHAPES.items():
            tile = g.tile_for(0, m, n, k)
            tma = g.tma_tile_for(0, m)
            out[f"any.int8_gemm.tile.{name}"] = lever(
                tile is not None,
                cfg = (f"{fmt(tile)}" + (f" tma={fmt(tma)}" if tma else "")) if tile else None,
            )
        return out
    finally:
        for k, v in old.items():
            setattr(g, k, v)
        for name, value in saved.items():
            target = getattr(g, name)
            target.clear()
            target.update(value)


@probe
def probe_grouped_int8_gemm(p: GpuProfile) -> dict:
    """The fused grouped (MoE / GGUF-style) INT8 GEMM tiles per arch."""
    from core.inference import diffusion_grouped_int8_gemm as gg

    if not p.cuda or p.rocm:
        return {
            "any.grouped_int8_gemm.tiles": lever(False, why = _na_code(p) if not p.rocm else "rocm")
        }
    rules = gg.arch_tiles(p.capability)
    return {
        "any.grouped_int8_gemm.tiles": lever(
            bool(rules),
            cfg = [fmt((r[0],) + tuple(r[1])) for r in rules] if rules else None,
            why = None if rules else "no_arch_tiles",
        )
    }


@probe
def probe_convrot(p: GpuProfile) -> dict:
    """ConvRot fused rotation + act quant: dense rotq_i8 tile, unlisted-arch rotq_h16 race, tf32x3 second stage,
    per-K tiles."""
    from core.inference import diffusion_convrot_quant as cq

    keys = ("rotq_tile", "unlisted_h16", "h16_tf32x3", "k_tiles")
    if not p.cuda or p.rocm:
        why = "rocm" if p.rocm else _na_code(p)
        return {f"any.convrot.{k}": lever(False, why = why) for k in keys}
    cap = tuple(p.capability)
    cfg = cq.rotquant_config(cap)
    unl = cq.unlisted_wanted(cap)
    tf = cap in cq._H16_TF32X3_ARCHS
    kt = cq.rotq_k_tiles(cap)
    return {
        "any.convrot.rotq_tile": lever(
            cfg is not None, cfg = cfg, why = None if cfg else "no_arch_tile"
        ),
        "any.convrot.unlisted_h16": lever(
            unl, why = None if unl else ("listed_arch" if cfg is not None else "below_sm80")
        ),
        "any.convrot.h16_tf32x3": lever(tf, why = None if tf else "ieee_fma"),
        "any.convrot.k_tiles": lever(
            bool(kt), cfg = len(kt) if kt else None, why = None if kt else "no_k_rules"
        ),
    }


@probe
def probe_cutlass_sm75(p: GpuProfile) -> dict:
    """The CUTLASS native-epilogue (fp16, Turing) path keys on family 75."""
    from core.inference import diffusion_int8_cutlass as cut

    if not p.cuda or p.rocm:
        return {
            "any.cutlass.native_fp16_epilogue": lever(False, why = "rocm" if p.rocm else _na_code(p))
        }
    fam = cut.family(p.capability)
    return {
        "any.cutlass.native_fp16_epilogue": lever(fam == 75, why = None if fam == 75 else "not_sm75")
    }


@probe
def probe_native_w8a8_cuda(p: GpuProfile) -> dict:
    """NativeWeightOnlyLinear W8A8 on CUDA: what production's ``diffusion_native_w8a8_cuda.prepare()`` decides for the
    card (its capability gate, Triton kernels, custom op and rotation mode). Only the on-card bit-compare probe and the
    kernel build are pinned as passing, like a race on the real card; the follow-up GEMM arming is not part of it."""
    if not p.cuda or p.rocm:
        return {"any.native_w8a8.cap_gate": lever(False, why = "rocm" if p.rocm else _na_code(p))}
    import types

    from core.inference import diffusion_convrot_quant as cq
    from core.inference import diffusion_native_w8a8_cuda as nw

    saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in nw._STATE.items()}
    for k in ("devices", "on", "tiles"):
        nw._STATE[k] = {}
    with (
        _pinned(
            nw,
            _probe = lambda index, cfg, rules: (cfg, rules),
            _op = lambda: object(),
            _norm_op = lambda: None,
            prepare_gemm = lambda index: False,
        ),
        _pinned(cq, _kernels = lambda: types.SimpleNamespace() if p.triton else None),
    ):
        try:
            on = bool(nw.prepare("cuda:0"))  # an int device needs a CUDA build of torch
            why = nw._STATE["devices"].get(0)
        except Exception as exc:  # noqa: BLE001 - recorded, never read as engaged
            on, why = False, f"error {type(exc).__name__}"
        finally:
            nw._STATE.clear()
            nw._STATE.update(saved)
    if not on and why is None:
        why = "disabled"  # prepare() returns before probing: its kill switch, or the rotation kernels off
    return {"any.native_w8a8.cap_gate": lever(on, why = None if on else _code(why, _W8A8_CODES))}


_W8A8_CODES = (
    ("no bf16 activations", "below_sm80"),
    ("Triton", "no_triton"),
    ("custom_op", "no_custom_op"),
    ("switched off", "rotq_off"),
    ("codes differ", "probe_mismatch"),
    ("disabled", "disabled"),
)


@probe
def probe_segment_quant(p: GpuProfile) -> dict:
    """Segmenter dynamic quant: ``unsupported_reason`` per scheme and what ``resolve_precision`` picks for a tight
    load (auto ladder), as resolved; an unrecognised refusal is recorded off, never assumed on."""
    import torch

    from core.inference import diffusion_segment_quant as sq

    codes = (
        ("compute capability", "below_min_cap"),
        ("ROCm", "rocm"),
        ("CUDA only", "not_cuda"),
        ("no bf16", "no_bf16"),
        ("stub", "torchao_stub"),
    )
    device = "cuda:0" if p.cuda else ("mps" if p.mps else "cpu")
    out: dict = {}
    for prec in (sq.PRECISION_FP8, sq.PRECISION_INT8):
        try:
            why = sq.unsupported_reason(prec, device, torch.bfloat16)
        except Exception as exc:  # noqa: BLE001 - recorded, never read as engaged
            why = f"error {type(exc).__name__}"
        code = _code(why, codes)
        if code == "not_cuda" and not p.cuda:
            # the production refusal, under the code the golden has always used for the platform
            code = _na_code(p)
        out[f"any.segment_quant.{prec}"] = lever(why is None, why = code)
    try:
        chosen, _why = sq.resolve_precision(device, torch.bfloat16, memory_tight = True)
    except Exception as exc:  # noqa: BLE001
        chosen = f"error {type(exc).__name__}"
    out["any.segment_quant.auto_tight"] = lever(chosen not in (sq.PRECISION_BF16,), cfg = chosen)
    return out


@probe
def probe_transformer_quant(p: GpuProfile) -> dict:
    """Denoiser quant: dense torchao support, the auto scheme tier, the hosted-checkpoint auto order, the fp16-card
    (Turing) int8 linear, the consumer heuristic behind fp16 accumulation and the RDNA explicit W8A8 arch gate."""
    from core.inference import diffusion_transformer_quant as tq

    target = _probe_target(p)
    out: dict = {}
    dense = tq.dense_transformer_supported(target)
    out["any.transformer_quant.dense_torchao"] = lever(
        dense,
        why = None if dense else ("rocm" if p.rocm else _na_code(p) if not p.cuda else "no_bf16"),
    )
    cap = tq._capability(0) if p.cuda else None
    order = tq._auto_scheme_order(None, "cuda", cap) if (dense and cap is not None) else ()
    out["any.transformer_quant.auto_tier"] = lever(
        bool(order), cfg = "+".join(order) or None, why = None if order else "no_tier"
    )
    hosted = tq.hosted_auto_scheme_order(None, cap) if (dense and cap is not None) else ()
    out["any.transformer_quant.hosted_auto"] = lever(
        bool(hosted), cfg = "+".join(hosted) or None, why = None if hosted else "bf16"
    )
    import torch
    import types

    fp16_target = types.SimpleNamespace(**{**vars(target), "dtype": torch.float16})
    turing = tq.fp16_card_int8_host(fp16_target)
    out["any.transformer_quant.fp16_card_int8"] = lever(
        turing, why = None if turing else ("not_turing" if nvidia(p) else _na_code(p))
    )
    if nvidia(p):
        consumer = tq._is_consumer_gpu(0)
        out["any.transformer_quant.consumer_gpu"] = lever(
            consumer, why = None if consumer else "datacenter_or_pro"
        )
    else:
        out["any.transformer_quant.consumer_gpu"] = lever(False, why = _na_code(p))
    rdna = p.rocm and tq._rdna_explicit_w8a8_arch_measured(target)
    out["any.transformer_quant.rdna_explicit_w8a8_arch"] = lever(
        rdna, why = None if rdna else ("unmeasured_gfx" if p.rocm else _na_code(p))
    )
    return out


@probe
def probe_te_prequant(p: GpuProfile) -> dict:
    """Hosted int8 ConvRot text encoder in float16 compute: per family, on a CUDA fp16 target."""
    import types

    import torch

    from core.inference import diffusion_te_prequant as tp

    out: dict = {}
    for fam in ("qwen-image-2.1", "flux.3-klein"):
        target = _probe_target(p)
        if p.cuda:
            target = types.SimpleNamespace(**{**vars(target), "dtype": torch.float16})
        on = tp._fp16_family_target(fam, target)
        out[f"{fam}.te_int8_fp16"] = lever(
            on, why = None if on else ("family_unvalidated" if p.cuda else _na_code(p))
        )
    return out


@probe
def probe_nvfp4(p: GpuProfile) -> dict:
    """NVFP4: whether a request is rerouted (pre-Blackwell) and to what, the flashinfer arch set and jit-cache
    provider, the MXFP8 tensor-core gate."""
    from core.inference import diffusion_comfy_block as cb
    from core.inference import diffusion_nvfp4_install as ni
    from core.inference import diffusion_nvfp4_ops as no
    from core.inference import diffusion_nvfp4_route as nr

    out: dict = {}
    saved = nr._smi_visible_capabilities
    nr._smi_visible_capabilities = lambda torch: ()
    try:
        prof = nr._probe_device_profile(True)
    except Exception as exc:  # noqa: BLE001
        prof = None
        kind = f"error {type(exc).__name__}"
    finally:
        nr._smi_visible_capabilities = saved
    kind = prof.kind if prof is not None else kind
    if prof is not None and prof.kind == nr.NVIDIA:
        cap, routes = nr.load_capability(prof)
        out["any.nvfp4.native"] = lever(not routes, why = None if not routes else "pre_blackwell")
        dit_i = nr._dit_target("image", cap)
        dit_v = nr._dit_target("video", cap)
        out["any.nvfp4.reroute_image"] = lever(
            routes and dit_i == nr.INT8,
            cfg = dit_i if routes else None,
            why = None if routes else "native",
        )
        out["any.nvfp4.reroute_video"] = lever(
            routes and dit_v == nr.INT8,
            cfg = dit_v if routes else None,
            why = None if routes else "native",
        )
    else:
        out["any.nvfp4.native"] = lever(False, why = f"kind:{kind}")
        out["any.nvfp4.reroute_image"] = lever(False, why = f"kind:{kind}")
        out["any.nvfp4.reroute_video"] = lever(False, why = f"kind:{kind}")
    cap = no._device_capability(0) if nvidia(p) else None
    fi = cap is not None and tuple(cap) in no.NVFP4_FLASHINFER_CAPS
    out["any.nvfp4.flashinfer_arch"] = lever(
        fi, why = None if fi else ("arch_not_in_set" if cap else _na_code(p))
    )
    jit = ni.FLASHINFER_JIT_CACHE_ARCH.get(tuple(cap)) if cap else None
    out["any.nvfp4.jit_cache_provider"] = lever(
        jit is not None, cfg = jit, why = None if jit else ("no_provider" if cap else _na_code(p))
    )
    # MXFP8: the capability gate of mxfp8_runtime_reason; its on-device probe after the gate cannot run on CPU, so
    # anything past the gate counts as passing it. The probe cache is restored so no verdict leaks out.
    target = _probe_target(p)
    snap = dict(getattr(cb, "_MX_PROBE", {}))
    try:
        if hasattr(cb, "_MX_PROBE"):
            cb._MX_PROBE[0] = (
                None  # the on-device probe passes (it runs on the real card); only the gate is asked
            )
        why = cb.mxfp8_runtime_reason(target)
    except Exception:  # noqa: BLE001 - raised by the device probe, after the gate
        why = None
    finally:
        if hasattr(cb, "_MX_PROBE"):
            cb._MX_PROBE.clear()
            cb._MX_PROBE.update(snap)
    if why and "MXFP8 tensor cores" in why:
        out["any.mxfp8.tensor_cores"] = lever(False, why = "below_sm100")
    elif why and "not CUDA" in why:
        out["any.mxfp8.tensor_cores"] = lever(False, why = _na_code(p))
    elif why:  # any other refusal is recorded, never read as passing the gate
        out["any.mxfp8.tensor_cores"] = lever(False, why = "other")
    elif p.rocm:
        out["any.mxfp8.tensor_cores"] = lever(True, why = "rocm_cap_passes")
    else:
        out["any.mxfp8.tensor_cores"] = lever(True)
    return out


@_contextlib.contextmanager
def _pinned(obj: Any, **attrs: Any):
    """Temporarily set ``attrs`` on ``obj`` (host-dependent answers a profile must not inherit)."""
    missing = object()
    old = {k: getattr(obj, k, missing) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                try:
                    delattr(obj, k)
                except AttributeError:
                    pass
            else:
                setattr(obj, k, v)


def _na(p: GpuProfile) -> Optional[dict]:
    """The off cell for a platform the policy does not serve (None where it does: CUDA / ROCm)."""
    if p.cuda:
        return None
    return lever(False, why = f"n/a:{p.vendor if p.vendor != 'apple' else 'mps'}")


def _device_target(p: GpuProfile) -> Any:
    """``resolve_diffusion_device_target()`` for the profile: Studio's detected-device globals (``utils.hardware``
    caches the host's device type and IS_ROCM) are pinned to the profile so the host never decides."""
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from utils import hardware as uh
    from utils.hardware import hardware as hw

    dev = uh.DeviceType.CUDA if p.cuda else uh.DeviceType.CPU
    with _pinned(uh, get_device = lambda: dev), _pinned(hw, IS_ROCM = bool(p.rocm)):
        return resolve_diffusion_device_target()


def _code(reason: Any, table: tuple) -> Optional[str]:
    """Short stable code for a policy's reason text: the first ``(substring, code)`` that matches, else ``other``."""
    if not reason:
        return None
    text = str(reason)
    for needle, code in table:
        if needle in text:
            return code
    return "other"


_H3_GATE_CODES = (
    ("speed_mode=off", "speed_off"),
    ("picked explicitly", "explicit_backend"),
    ("Auto keeps it off for MiniMax-H3 video on", "arch_off"),
    ("hosted INT8 denoiser", "not_evidenced_config"),
)


@probe
def probe_h3_video_attention(p: GpuProfile) -> dict:
    """H3 video's default load (Auto attention, speed on, hosted INT8 denoiser, no adapters): does Auto race INT8
    attention (and with which exact start), and which SDPA backend the strided processor runs."""
    keys = (
        "minimax-h3.int8_attn_auto",
        "minimax-h3.sdpa_backend",
        "minimax-h3.strided_processor",
        "minimax-h3.flash_sdpa",
        "minimax-h3.int8_exact_policy",
    )
    off = _na(p)
    if off is not None:
        return {k: off for k in keys}
    import torch

    from core.inference import diffusion_int8_attention as i8
    from core.inference import diffusion_attention as da
    from core.inference.video_minimax_h3_attn import (
        cuda_capability,
        h3_attention_backend,
        h3_int8_exact_kernel,
        h3_int8_load_gate,
        strided_install_backend,
    )

    out: dict = {}
    gate = h3_int8_load_gate(
        mode = i8.resolve_policy(None).mode,
        attention_backend = None,
        auto_attention = da._auto_request(None),
        speed_on = True,
        hosted_int8 = True,
        comfy_checkpoint = False,
        adapters = False,
        capability = cuda_capability("cuda:0"),
    )
    on = gate.auto or gate.forced
    out[keys[0]] = lever(
        on,
        cfg = f"exact_steps={gate.exact_steps}" if on else None,
        why = ("forced" if gate.forced else "auto_race")
        if on
        else _code(gate.reason, _H3_GATE_CODES),
    )
    target = _device_target(p)
    # ROCm's Auto flash pick needs an installed, verified flash_attn: pinned present + verified, so the cell is the
    # policy's answer for a working wheel, not the host's.
    with _pinned(
        da, _flash_attn_installed = lambda: True, _rocm_flash_attn_runs = lambda *a, **k: True
    ):
        selected = da.select_attention_backend(target, None, speed_active = True)
        backend = h3_attention_backend(
            selected,
            requested = None,
            target = _types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16),
        )
    out[keys[1]] = lever(
        backend is not None, cfg = str(getattr(backend, "value", backend)) if backend else None
    )
    # Flash over cuDNN is its own lever (H3_FLASH_FASTER_ARCHS): losing it on a measured arch is a regression.
    out["minimax-h3.flash_sdpa"] = lever(str(getattr(backend, "value", backend)) == "_native_flash")
    out[keys[2]] = lever(strided_install_backend(backend))
    # Exact-start steps + kernel move clip time on every card (#19344): keep them a golden diff.
    out[keys[4]] = _h3_exact_policy_cell(gate, backend)
    return out


# hidden 7168 / 56 heads
_H3_HEAD_DIM = 128


def _h3_exact_policy_cell(gate: Any, backend: Any) -> dict:
    """What the load installs, not what the gate allows: needs the strided processor and a CUDA INT8 race (never ROCm)."""
    from core.inference import diffusion_int8_attention as i8
    from core.inference.video_minimax_h3_attn import h3_int8_exact_kernel, strided_install_backend

    if not gate.auto:
        why = "forced_every_step" if gate.forced else "int8_off"
    elif gate.exact_steps <= 0:
        why = "int8_every_step"
    elif not strided_install_backend(backend):
        why = "inactive:no_strided_processor"
    elif i8._is_rocm():
        why = "inactive:rocm_no_race"
    elif i8.implementation(0, _H3_HEAD_DIM) != "cuda":
        why = "inactive:no_cuda_kernel"
    else:
        return lever(True, cfg = {"steps": gate.exact_steps, "kernel": h3_int8_exact_kernel(backend)})
    return lever(False, cfg = {"steps": 0, "kernel": "none"}, why = why)


_H3I_CODES = (
    ("bf16 requested", "bf16_requested"),
    ("torchao unavailable", "no_torchao"),
    ("no int8 or fp8 rung", "no_rung"),
    ("auto: int8", "auto_int8"),
    ("auto: fp8", "auto_fp8"),
    ("auto order", "auto_order"),
    ("fits", "fit"),
)


@probe
def probe_h3_image_scheme(p: GpuProfile) -> dict:
    """H3-Image's Auto DiT weight scheme at the profile's VRAM / host RAM (2048 px), and whether explicit fp8 runs."""
    keys = ("minimax-h3-image.dit_scheme_auto", "minimax-h3-image.fp8_explicit")
    off = _na(p)
    if off is not None:
        return {k: off for k in keys}
    from core.inference.diffusion_minimax_h3_image import (
        explicit_scheme_refusal,
        resolve_dit_scheme,
    )

    scheme, reason = resolve_dit_scheme(
        "auto", device = "cuda:0", vram_gb = p.vram_gb, host_gb = p.host_ram_gb
    )
    refusal = explicit_scheme_refusal("fp8", "cuda:0")
    return {
        keys[0]: lever(scheme is not None, cfg = scheme, why = _code(reason, _H3I_CODES)),
        keys[1]: lever(
            refusal is None,
            why = "needs_sm89" if refusal and "8.9" in refusal else ("refused" if refusal else None),
        ),
    }


@probe
def probe_vae_decode(p: GpuProfile) -> dict:
    """Strip decode on every decode (measured cards), the KL strip path, the strip size, the folded upsample, the
    16-bit decode (vs fp32 on cards without tensor cores) and the Wan VAE frame pairing."""
    keys = (
        "any.vae_strips_always",
        "any.vae_kl_strips_always",
        "any.vae_strip_elems",
        "any.vae_upsample_fold",
        "any.vae_16bit_decode",
        "any.video_vae_pair_frames",
    )
    off = _na(p)
    if off is not None:
        return {k: off for k in keys}
    import torch

    from core.inference import diffusion_vae_finite as vf
    from core.inference import diffusion_vae_strips as vs
    from core.inference import video_vae_untiled as vu

    for cached in (vs._card_strip_elems, vs._card_cap):
        cached.cache_clear()
    dev = torch.device("cuda", 0)
    try:
        elems = vs.strip_elems(dev)
        x = _types.SimpleNamespace(is_cuda = True, device = dev)
        target = _device_target(p)
        out = {
            keys[0]: lever(vs.always_on(dev)),
            keys[1]: lever(vs.kl_always_on(dev)),
            keys[2]: lever(True, cfg = f"2^{int(elems).bit_length() - 1}"),
            keys[3]: lever(vs._fold_on(x)),
            keys[4]: lever(
                not vf.prefers_fp32_decode(target),
                why = "no_tensor_cores" if vf.prefers_fp32_decode(target) else None,
            ),
            keys[5]: lever(vu._pair_frames_enabled(dev)),
        }
    finally:
        for cached in (vs._card_strip_elems, vs._card_cap):
            cached.cache_clear()
    return out


@probe
def probe_device_target(p: GpuProfile) -> dict:
    """The device / backend / compute dtype a diffusion load resolves, and the ROCm bf16 Wan VAE decode arch gate."""
    from core.inference import diffusion_device as dd

    target = _device_target(p)
    dtype = str(getattr(target, "dtype", None)).replace("torch.", "")
    out = {
        "any.device_target": lever(
            target.device != "cpu",
            cfg = f"{target.device}/{target.backend}/{dtype}",
        ),
        "any.compile_default": lever(
            bool(getattr(target, "supports_default_torch_compile", False))
        ),
    }
    if p.mps:
        # Studio's other MPS branch: the bf16 probe fails (before macOS 14), so the load runs fp32, never fp16.
        with _pinned(dd, _mps_supports_bfloat16 = lambda torch_mod: False):
            fp32 = _device_target(p)
        dtype = str(getattr(fp32, "dtype", None)).replace("torch.", "")
        out["any.device_target_fp32"] = lever(
            fp32.device != "cpu", cfg = f"{fp32.device}/{fp32.backend}/{dtype}"
        )
    # A request with no speed_mode compiles from the deferred profile only where the dtype allows it: fp16 cards (T4)
    # compile on an explicit tier only, so their Studio-default load runs eager (real T4 load, 2026-11-24).
    from core.inference.diffusion_speed import fp16_compile_explicit_only

    explicit_only = target.device != "cpu" and fp16_compile_explicit_only(target)
    out["any.unset_speed_may_compile"] = lever(
        target.device != "cpu" and not explicit_only,
        why = "fp16_explicit_tier_only" if explicit_only else None,
    )
    if p.rocm:
        import torch
        arch = dd._rocm_bf16_decode_arch(torch, target)
        out["any.rocm_vae_bf16_decode"] = lever(arch is not None)
    else:
        out["any.rocm_vae_bf16_decode"] = lever(
            False, why = f"n/a:{p.vendor if p.vendor != 'apple' else 'mps'}"
        )
    return out


class Float8Tensor:  # the class NAME is what ``_fp8_rowwise_weight`` reads
    def __init__(self) -> None:
        self.scale = _types.SimpleNamespace(numel = lambda: 64)
        self.act_quant_kwargs = {"granularity": "per_row"}
        self.device = _types.SimpleNamespace(type = "cuda")


@probe
def probe_fp8_rowwise_graph(p: GpuProfile) -> dict:
    """A row-wise fp8 denoiser may be CUDA-graphed (``fp8_rowwise_capture_refusal``), at a pinned torch 2.10 (the
    hang window; torch >= 2.12 never refuses), so the cell does not follow the host's torch."""
    key = "any.cuda_graph_fp8_rowwise_torch210"
    off = _na(p)
    if off is not None:
        return {key: off}
    import sys as _sys

    import torch

    from core.inference import diffusion_cuda_graph as cg

    module = _types.SimpleNamespace(parameters = lambda: [Float8Tensor()])
    with _pinned(torch, __version__ = "2.10.0"), _pinned(_sys, platform = "linux"):
        refusal = cg.fp8_rowwise_capture_refusal([module], "cuda:0")
    return {key: lever(refusal is None, why = "streamk_hang" if refusal else None)}


@probe
def probe_inductor_reduction_filter(p: GpuProfile) -> dict:
    """Families that pin inductor's reduction configs on measured archs (``filter_reduction_configs_archs``)."""
    from core.inference.diffusion_compile_config import family_filters_reductions
    from core.inference.diffusion_families import _FAMILIES as image
    from core.inference.video_families import _FAMILIES as video

    out: dict = {}
    for fam in tuple(image) + tuple(video):
        # Only the arch-listed ones: ``filter_reduction_configs`` (every arch) is no per-profile decision.
        if getattr(fam, "filter_reduction_configs_archs", None) and not getattr(
            fam, "filter_reduction_configs", False
        ):
            out[f"{fam.name}.inductor_reduction_filter"] = lever(family_filters_reductions(fam))
    return out


_LTX_CODES = (
    ("not a CUDA load", "not_cuda"),
    ("bfloat16", "bf16_card"),
    ("not promoted", "fp16_not_promoted"),
    ("ROCm", "rocm"),
    ("no int8 tensor cores", "no_int8_tensor_cores"),
    ("SMALL_HOST=0", "switch_off"),
    ("probe failed", "probe_failed"),
)


@probe
def probe_ltx25_lowram(p: GpuProfile) -> dict:
    """LTX-2.5's small-host int8 route (fp16 cards): can it serve this card at all."""
    key = "ltx-2.5.lowram_int8_route"
    from core.inference.video_families import _FAMILIES as video
    from core.inference.video_ltx25_lowram import lowram_ineligible

    fam = next(f for f in video if f.name == "ltx-2.5")
    why = lowram_ineligible(fam, _device_target(p), model_kind = "pipeline")
    return {key: lever(why is None, why = _code(why, _LTX_CODES))}


@probe
def probe_sd_cpp_quant_cublas(p: GpuProfile) -> dict:
    """sd.cpp's Q8_0 BF16-cuBLAS route row threshold (None keeps int8 MMQ)."""
    key = "any.sd_cpp_q8_cublas"
    if not nvidia(p):
        return {key: lever(False, why = f"n/a:{p.vendor if p.vendor != 'apple' else 'mps'}")}
    from core.inference.sd_cpp_args import quant_cublas_min_batch

    rows = quant_cublas_min_batch(tuple(p.capability))
    return {key: lever(rows is not None, cfg = rows, why = None if rows is not None else "mmq")}


@probe
def probe_auto_attention(p: GpuProfile) -> dict:
    """Studio's Auto attention pick for a default (speed on) load, with ROCm flash_attn pinned present + verified."""
    key = "any.attn_auto_backend"
    from core.inference import diffusion_attention as da

    target = _device_target(p)
    with _pinned(
        da, _flash_attn_installed = lambda: True, _rocm_flash_attn_runs = lambda *a, **k: True
    ):
        backend = da.select_attention_backend(target, None, speed_active = True)
    return {
        key: lever(
            backend is not None, cfg = str(getattr(backend, "value", backend)) if backend else None
        )
    }


@probe
def probe_cfg_batch(p: GpuProfile) -> dict:
    """CFG batching: which compiled resident classes batch, and whether a measured-slower entry names this card."""
    from core.inference import diffusion_cfg_batch as cb

    slower = sorted(
        f"{cls}/{mode}"
        for (cls, card, mode) in cb.MEASURED_SLOWER
        if str(card).lower() in p.name.lower()
    )
    return {
        "any.cfg_batch_compiled_resident": lever(
            bool(cb.COMPILED_RESIDENT_CLASSES), cfg = fmt(sorted(cb.COMPILED_RESIDENT_CLASSES))
        ),
        "any.cfg_batch_card_allowed": lever(not slower, cfg = fmt(slower) if slower else None),
    }


# Every registered image / video family through the real optimizer (``install_patches`` + ``run_speed_layer``, the
# parity harness's entry), at the default speed tier and the dtype the profile resolves. Stub denoisers carry the
# family's real class name, module and ``_repeated_blocks`` from ``inputs.denoiser_blocks`` in the golden, so the run
# is the same with or without diffusers installed (Backend CI has none; the whole matrix runs with it hidden).
AUTOOPT_SPEED = "default"


def _mps_target(bf16_probe_ok: bool) -> Any:
    """Studio's MPS target (``_mps_or_cpu_target``) with its on-device bf16 probe pinned: bf16 when the probe passes
    (macOS 14+), fp32 when it fails, never fp16. The allocator cap (a real-device call) is left out; the env it would
    set is restored."""
    import torch

    from core.inference import diffusion_device as dd

    env = dd.MPS_HIGH_WATERMARK_ENV
    saved_env = os.environ.get(env)
    saved = (dd._mps_supports_bfloat16, dd.apply_mps_memory_cap)
    dd._mps_supports_bfloat16 = lambda torch_mod: bf16_probe_ok
    dd.apply_mps_memory_cap = lambda torch_mod, logger = None: None
    try:
        return dd._mps_or_cpu_target(torch)
    finally:
        dd._mps_supports_bfloat16, dd.apply_mps_memory_cap = saved
        if saved_env is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = saved_env


def _autoopt_target(p: GpuProfile) -> Any:
    import torch

    from core.inference import diffusion_device as dd

    if p.cuda:
        return dd._cuda_or_rocm_target(torch, is_rocm = p.rocm)
    if p.mps:
        return _mps_target(True)
    return dd._cpu_target(torch)


def _autoopt_targets(p: GpuProfile) -> list:
    """``[(key suffix, target)]``: one target per dtype branch Studio's resolver can take on the profile. MPS has two:
    the bf16 probe passing (the default row) or failing (``autoopt_fp32`` rows)."""
    if p.mps:
        return [("", _mps_target(True)), ("_fp32", _mps_target(False))]
    return [("", _autoopt_target(p))]


def _stub_denoiser(fam: Any, blocks: dict) -> Any:
    import torch

    name = fam.transformer_class
    meta = blocks.get(name) or {}

    def compile_repeated_blocks(self, **kwargs):
        if not getattr(self, "_repeated_blocks", None):
            raise ValueError("_repeated_blocks is empty")
        self.compile_calls.append(dict(kwargs))

    def compile(self, **kwargs):
        self.compile_calls.append({"whole": True, **kwargs})

    ns = {
        "_repeated_blocks": list(meta.get("blocks") or []),
        "_no_split_modules": list(meta.get("no_split") or []),
        "compile_repeated_blocks": compile_repeated_blocks,
        "forward": lambda self, *a, **k: None,
        "__module__": meta.get("module") or "studio_gpu_lever_stub",
    }
    if name == "UNet2DConditionModel":
        ns["compile"] = compile
        del ns["compile_repeated_blocks"]
    module = type(name, (torch.nn.Module,), ns)()
    module.compile_calls = []
    return module


def _autoopt_pipe(backend: str, fam: Any, blocks: dict) -> Any:
    pipe = _types.SimpleNamespace()
    setattr(
        pipe,
        getattr(fam, "denoiser_attr", "transformer") or "transformer",
        _stub_denoiser(fam, blocks),
    )
    if backend == "video" and getattr(fam, "is_moe", False):
        pipe.transformer_2 = _stub_denoiser(fam, blocks)
    return pipe


def _stub_denoisers(pipe: Any) -> list:
    """The denoisers ``_autoopt_pipe`` built (one, or a MoE's two experts): the views the load must cover."""
    names = ("transformer", "unet", "transformer_2")
    return [getattr(pipe, n) for n in names if getattr(pipe, n, None) is not None]


def _speed_layer_extra(patch: Any, p: GpuProfile) -> None:
    """What the parity harness pins besides the device: unsloth_zoo's patch helpers, graph install, Triton kernels."""
    from core.inference import diffusion_cuda_graph, diffusion_gguf_compile, diffusion_patch_backend

    stash: dict = {}

    def _patch_function(
        target,
        attr,
        new_fn,
        match_level = "relaxed",
        force = False,
    ):
        stash.setdefault((target, attr), getattr(target, attr))
        setattr(target, attr, new_fn)
        return True

    def _restore_original(target, attr):
        if (target, attr) not in stash:
            return False
        setattr(target, attr, stash.pop((target, attr)))
        return True

    patch.setattr(
        diffusion_patch_backend,
        "_HELPERS",
        {"patch": _patch_function, "restore": _restore_original},
    )
    fake_gguf_utils = _types.ModuleType("diffusers.quantizers.gguf.utils")
    fake_gguf_utils.dequantize_gguf_tensor = lambda tensor: tensor
    patch.setattr(diffusion_gguf_compile, "_gguf_utils", lambda: fake_gguf_utils)
    patch.setattr(diffusion_gguf_compile, "_compiled_dequant_installed", False)
    patch.setattr(diffusion_gguf_compile, "_compile_error", None)
    # install_cuda_graphs runs for real: it only wraps each denoiser (the first step captures), so the cell reads
    # what it armed, as the load does.
    if hasattr(diffusion_cuda_graph, "release_cached_outside_capture"):
        patch.setattr(diffusion_cuda_graph, "release_cached_outside_capture", lambda *a, **k: None)
    kernel = (lambda *a, **k: None) if (p.cuda and p.triton) else None
    for mod_name, attr, value in (
        ("diffusion_flux2_rope", "_kernel", lambda: kernel),
        ("diffusion_flux3_rope", "_kernel", lambda: kernel),
        ("video_wan_fused", "_kernels", lambda: ({} if kernel else None)),
        ("diffusion_rocm_fused", "_is_nvidia_linux", lambda: p.vendor == "nvidia"),
    ):
        try:
            mod = __import__(f"core.inference.{mod_name}", fromlist = [attr])
            patch.setattr(mod, attr, value)
        except Exception:  # noqa: BLE001
            pass


def family_target(backend: str, fam: Any, target: Any) -> Any:
    """The load target with the compute dtype the image load resolves for the family (fp16-incompatible families
    promote fp16 to fp32: a real T4 Qwen-Image load ran fp32), as ``diffusion._resolve_diffusion_compute_dtype``."""
    if backend != "image" or getattr(target, "dtype", None) is None:
        return target
    from core.inference.diffusion import _resolve_diffusion_compute_dtype

    dtype = _resolve_diffusion_compute_dtype(fam, target.dtype)
    if dtype == target.dtype:
        return target
    import dataclasses

    if dataclasses.is_dataclass(target):
        return dataclasses.replace(target, dtype = dtype)
    return _types.SimpleNamespace(**{**vars(target), "dtype": dtype})


# The host-flip children resolve one profile per platform class (each host input is profile-independent; these
# cover every vendor, consumer / datacenter, fp16 and bf16, and both future-arch parents).
FLIP_PROFILES = ("t4", "rtx3090", "l4", "rtx5090", "b200", "gfx1151", "mps", "cpu")


def _collected_profiles() -> list:
    only = os.environ.get("GPU_LEVER_PROFILES")
    keys = [k for k in only.split(",") if k] if only else list(PROFILES)
    return [PROFILES[k] for k in keys]


# The arming's own labels (diffusion_cuda_graph.arm_after_placement) for the placement each graph recorded under.
_ARMING_LABELS = (
    ("model offload over a stable device range", "model_stable"),
    ("block-streamed", "group"),
    ("model offload", "model"),
    ("resident", "resident"),
)


class CpuOffload:
    """What accelerate's ``add_hook_to_module`` leaves on a module under ``enable_model_cpu_offload``: a hook named
    ``CpuOffload`` and the module's own forward kept as ``_old_forward`` (the arming wraps the replay under it)."""

    def __init__(self, execution_device: str = "cuda:0"):
        self.execution_device = execution_device


def place_model_offload(pipe: Any) -> None:
    """Placement for an offloaded load, after the speed layer as in the loaders: each denoiser gets accelerate's
    whole-module CPU offload hook (Studio's model-offload policy)."""
    for module in _stub_denoisers(pipe):
        module._old_forward = module.forward
        module._hf_hook = CpuOffload()


def armed_after_placement(pipe: Any, views: list) -> Optional[str]:
    """What the loader's post-placement step installs for an offloaded load: ``arm_graphs_after_placement`` (the call
    the image and video loaders make once placement is final) on denoisers an offload hook really moves, never the
    pending flag the speed layer leaves. Returns the placement the graphs recorded under (the arming's own label:
    ``model_stable``, ``model``, ``group`` or ``resident``), else None. Block-streamed copies recorded in the graph
    and the replay itself need a real card: the GPU test ``test_diffusion_offload_cuda_graph.py``."""
    from core.inference.diffusion_speed import arm_graphs_after_placement

    place_model_offload(pipe)
    branch = None
    for view in views:
        try:
            armed = bool(arm_graphs_after_placement(pipe, dict(view)).get("cuda_graph"))
        except Exception:  # noqa: BLE001 - recorded as not armed
            continue
        if armed:
            reason = str(getattr(pipe, "_unsloth_cuda_graph_reason", "") or "")
            branch = next((code for text, code in _ARMING_LABELS if text in reason), "other")
    try:
        from core.inference.diffusion_cuda_graph import uninstall_all
        uninstall_all(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())
    except Exception:  # noqa: BLE001
        pass
    return branch


def autoopt_cell(
    p: GpuProfile, backend: str, fam: Any, blocks: dict, target: Any, offload: bool
) -> dict:
    """One speed-layer cell, inside ``mock_profile(p, extra = _speed_layer_extra)``: the family through the parity
    harness's entry point at the family's resolved dtype, resident or offloaded."""
    import test_diffusion_autoopt_parity as par
    from core.inference import diffusion_speed as ds

    fam_target = family_target(backend, fam, target)
    dtype = str(fam_target.dtype).replace("torch.", "")
    pipe = _autoopt_pipe(backend, fam, blocks)
    flags = ds.snapshot_backend_flags()
    try:
        result = par.autoopt_entry(backend, pipe, fam_target, fam, AUTOOPT_SPEED, offload, False)
        views = list(result["views"])
        engaged = {k for view in views for k, v in view.items() if v}
        engaged |= {f"patch:{k}" for k, v in result["patches"].items() if v}
        # A dual-DiT MoE (Wan 2.2 A14B) runs the speed layer once per expert: each expert's engagement is recorded on
        # its own (the union above could hide a lever lost on one expert), and a view missing is a lost cell.
        experts = _stub_denoisers(pipe)
        if len(views) != len(experts):
            return {
                "on": False,
                "why": f"views:{len(views)}/{len(experts)}",
                "cfg": f"dtype={dtype}",
            }
        per_expert = None
        if len(experts) > 1:
            per_expert = [
                sorted(k for k, v in view.items() if v and not (offload and k == "cuda_graph"))
                for view in views
            ]
        branch = None
        if offload:
            engaged.discard("cuda_graph")
            branch = armed_after_placement(pipe, result["views"])
            if branch:
                engaged.add("cuda_graph_after_placement")
        engaged = sorted(engaged)
        cell = {"on": bool(engaged), "engaged": engaged, "cfg": f"dtype={dtype}"}
        if branch:
            cell["arming"] = branch
        if per_expert is not None:
            cell["experts"] = per_expert
        return cell
    except Exception as exc:  # noqa: BLE001 - recorded: a load that now raises is a lost cell
        return {"on": False, "why": f"error:{type(exc).__name__}"}
    finally:
        ds.restore_backend_flags(flags)
        par._teardown()


def collect_autoopt(blocks: dict) -> dict:
    import test_diffusion_autoopt_parity as par

    cells: dict = {}
    for p in _collected_profiles():
        with clean_policy_env(), mock_profile(p, extra = _speed_layer_extra):
            # offload: the streamed / offloaded tier (8-16 GB cards), where graphs arm after placement (R3 / R4): the
            # cell records what arm_graphs_after_placement installs (cuda_graph_after_placement) and the branch that
            # armed (``arming``), not the speed layer's pending flag
            for suffix, target in _autoopt_targets(p):
                for (backend, fam), offload in itertools.product(par._families(), (False, True)):
                    key = f"{p.key}.{fam.name}.autoopt{suffix}" + ("_offload" if offload else "")
                    cells[key] = autoopt_cell(p, backend, fam, blocks, target, offload)
    return cells


def collect_levers() -> dict:
    cells: dict = {}
    for p in _collected_profiles():
        with clean_policy_env(), mock_profile(p):
            for fn in LEVER_PROBES:
                for key, value in fn(p).items():
                    cells[f"{p.key}.{key}"] = value
    return cells


def denoiser_blocks_from_diffusers() -> dict:
    """``{class: {module, blocks, no_split}}`` from the installed diffusers / Studio's in-tree classes (update only)."""
    import test_diffusion_autoopt_parity as par

    out: dict = {}
    for _backend, fam in par._families():
        name = fam.transformer_class
        cls = par._real_class(name)
        if cls is None:
            continue
        out[name] = {
            "module": cls.__module__,
            "blocks": list(getattr(cls, "_repeated_blocks", None) or []),
            "no_split": list(getattr(cls, "_no_split_modules", None) or []),
        }
    return out


def collect_all(blocks: dict) -> dict:
    """Both tables, in THIS process (the test runs it in a child with diffusers hidden: ``_collect_in_child``)."""
    import time

    t0 = time.perf_counter()
    cells = collect_levers()
    t1 = time.perf_counter()
    cells.update(collect_autoopt(blocks))
    t2 = time.perf_counter()
    return {"cells": cells, "seconds": {"levers": round(t1 - t0, 2), "autoopt": round(t2 - t1, 2)}}


# Host differences the golden must NOT see (Backend CI's runners, a dev box, Windows, torch floor to latest): each is
# simulated in its own child and must resolve the identical matrix (test_the_matrix_is_host_independent).
HOST_FLIPS: dict = {
    # torchao missing or failing to import (TorchAO 0.18 needs torch >= 2.11)
    "no_torchao": "sys.modules['torchao'] = None",
    # Triton missing (CPU torch from pip on Windows / macOS)
    "no_triton": "sys.modules['triton'] = None",
    # Windows: torch_compile_runtime_available() and other sys.platform reads
    "win32": "import torch; sys.platform = 'win32'",
    "darwin": "import torch; sys.platform = 'darwin'",
    # torch before allow_fp16_accumulation (Backend CI's floor, 2.4)
    "old_torch_matmul": (
        "import torch\n"
        "class _M:\n"
        "    def __init__(self, m): object.__setattr__(self, '_m', m)\n"
        "    def __getattr__(self, k):\n"
        "        if k == 'allow_fp16_accumulation': raise AttributeError(k)\n"
        "        return getattr(self._m, k)\n"
        "    def __setattr__(self, k, v): setattr(self._m, k, v)\n"
        "torch.backends.cuda.matmul = _M(torch.backends.cuda.matmul)"
    ),
    # torch before the e8m0 dtype (< 2.7)
    # torch's own compiler modules bind the dtype at import, so load them first: a real old torch
    # imports cleanly, it only lacks the attribute the production probes read.
    "old_torch_dtypes": (
        "import torch, torch._dynamo, torch._inductor\n"
        "if hasattr(torch, 'float8_e8m0fnu'): delattr(torch, 'float8_e8m0fnu')"
    ),
    # a macOS runner whose real MPS device crashes on first use (torch 2.15 on an Apple Paravirtual device): any MPS
    # tensor, MPS copy, torch.mps reading or Metal.framework load kills the process, so the matrix must touch none
    "mps_device_crashes": (
        "import os, ctypes, torch\n"
        "def _die(*a, **k):\n"
        "    os.kill(os.getpid(), 11)\n"
        "def _mps_arg(a, k):\n"
        "    return any('mps' in str(x) for x in (*a, *k.values()) if isinstance(x, (str, torch.device)))\n"
        "def _guard(fn):\n"
        "    def run(*a, **k):\n"
        "        if _mps_arg(a, k):\n"
        "            _die()\n"
        "        return fn(*a, **k)\n"
        "    return run\n"
        "for _n in ('ones', 'zeros', 'empty', 'randn', 'rand', 'tensor', 'full', 'arange', 'ones_like', 'zeros_like', 'empty_like'):\n"
        "    setattr(torch, _n, _guard(getattr(torch, _n)))\n"
        "for _n in ('to', 'copy_'):\n"
        "    setattr(torch.Tensor, _n, _guard(getattr(torch.Tensor, _n)))\n"
        "torch.Tensor.mps = _die\n"
        "for _n in ('recommended_max_memory', 'driver_allocated_memory', 'current_allocated_memory', 'synchronize', 'empty_cache'):\n"
        "    if hasattr(torch.mps, _n): setattr(torch.mps, _n, _die)\n"
        "_cdll = ctypes.CDLL\n"
        "class _CDLL(_cdll):\n"
        "    def __init__(self, name, *a, **k):\n"
        "        if 'Metal' in str(name): _die()\n"
        "        super().__init__(name, *a, **k)\n"
        "ctypes.CDLL = _CDLL"
    ),
    # a host without the CUDA driver library (every CPU CI runner, macOS, Windows without a card)
    "no_cuda_driver": (
        "import ctypes\n"
        "_cdll = ctypes.CDLL\n"
        "class _NoDriver(_cdll):\n"
        "    def __init__(self, name, *a, **k):\n"
        "        if any(t in str(name) for t in ('libcuda', 'nvcuda')):\n"
        "            raise OSError(f'{name}: not found')\n"
        "        super().__init__(name, *a, **k)\n"
        "ctypes.CDLL = _NoDriver"
    ),
    # accelerate absent (the profiles' offload hook is the guard's own, never the installed accelerate's)
    "no_accelerate": "import sys\nsys.modules['accelerate'] = None",
    # an Intel XPU visible on the host
    "host_xpu": "import torch; torch.xpu.is_available = lambda: True; torch.xpu.device_count = lambda: 1",
    # a host with diffusers installed (the golden hides it; the flip un-hides nothing, it must stay hidden)
}

_CHILD = """
import json, os, sys
sys.modules["diffusers"] = None  # what Backend CI has; every host resolves the same golden
flip = os.environ.get("GPU_LEVER_HOST_FLIP")
if flip:
    exec(json.loads(os.environ["GPU_LEVER_HOST_FLIP_CODE"]))
sys.path[:0] = [{backend!r}, {tests!r}]
import test_diffusion_gpu_lever_matrix as m
blocks = json.loads(sys.stdin.read())
out = m.collect_all(blocks)
sys.stdout.write("\\n@@GPU_LEVER_JSON@@" + json.dumps(out))
"""


def _child_cmd() -> list:
    return [sys.executable, "-c", _CHILD.format(backend = str(_BACKEND), tests = str(_TESTS))]


# Flip sets the host-independence test runs, one child each (all at once): a bare Windows CPU host, and macOS.
HOST_FLIP_SETS: dict = {
    "windows_cpu_bare": (
        "no_torchao",
        "no_triton",
        "old_torch_matmul",
        "old_torch_dtypes",
        "host_xpu",
        "win32",
        "no_cuda_driver",
        "no_accelerate",
    ),
    "macos": ("no_triton", "darwin", "mps_device_crashes", "no_cuda_driver"),
}


def _flip_code(flip: str) -> str:
    names = HOST_FLIP_SETS.get(flip) or (flip,)
    return "\n".join(HOST_FLIPS[n] for n in names)


def _child_env(flip: Optional[str] = None, profiles: Optional[Any] = None) -> dict:
    import json

    from gpu_profiles import child_env_base

    env = (
        child_env_base()
    )  # an allowlist: no inherited UNSLOTH_* lever or import-time opt-in reaches the child
    if profiles:
        env["GPU_LEVER_PROFILES"] = ",".join(profiles)
    if flip:
        env["GPU_LEVER_HOST_FLIP"] = flip
        env["GPU_LEVER_HOST_FLIP_CODE"] = json.dumps(_flip_code(flip))
        env["GPU_LEVER_PROFILES"] = ",".join(FLIP_PROFILES)
    return env


def _child_result(proc: Any, stdout: str, stderr: str) -> dict:
    import json

    marker = "@@GPU_LEVER_JSON@@"
    if proc.returncode != 0 or marker not in stdout:
        raise AssertionError(
            f"the lever matrix child exited {proc.returncode}:\n{stdout[-3000:]}\n{stderr[-6000:]}"
        )
    return json.loads(stdout.rsplit(marker, 1)[1])


def _collect_in_child(
    blocks: dict,
    flip: Optional[str] = None,
    profiles: Optional[Any] = None,
) -> dict:
    import json
    import subprocess

    proc = subprocess.run(
        _child_cmd(),
        input = json.dumps(blocks),
        capture_output = True,
        text = True,
        cwd = str(_BACKEND),
        env = _child_env(flip, profiles),
        timeout = 600,
    )
    return _child_result(proc, proc.stdout, proc.stderr)


def collect_flips(blocks: dict, flips: Any) -> dict:
    """``{flip: matrix}``, every flip in its own child, all at once."""
    import json
    import subprocess

    procs = {
        flip: subprocess.Popen(
            _child_cmd(),
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            cwd = str(_BACKEND),
            env = _child_env(flip),
        )
        for flip in flips
    }
    out = {}
    for flip, proc in procs.items():
        stdout, stderr = proc.communicate(json.dumps(blocks), timeout = 600)
        out[flip] = _child_result(proc, stdout, stderr)
    return out


GOLDEN = _TESTS / "fixtures" / "diffusion_gpu_lever_golden.json"
GOLDEN_SCHEMA = 1
UPDATE_ENV = "UNSLOTH_UPDATE_GPU_GOLDEN"
REASON_ENV = "UNSLOTH_GPU_GOLDEN_REASON"
_HOW_TO_UPDATE = (
    "If this change is intended, regenerate the golden from studio/backend and say why (the reason and the changed "
    "cells are appended to its `changes` ledger, which review reads):\n"
    f'    {UPDATE_ENV}=1 {REASON_ENV}="<what changed and the evidence>" '
    "python -m pytest tests/test_diffusion_gpu_lever_matrix.py -k golden\n"
    "A lost lever on a real profile is a speed regression on that card until a benchmark says otherwise."
)


def load_golden(path: Path = GOLDEN) -> dict:
    import json
    if not path.exists():
        return {"schema": GOLDEN_SCHEMA, "version": 0, "inputs": {}, "changes": [], "cells": {}}
    return json.loads(path.read_text(encoding = "utf-8"))


def _engaged_set(cell: Optional[dict]) -> set:
    if not cell:
        return set()
    if "engaged" in cell:
        return set(cell["engaged"])
    return {"<lever>"} if cell.get("on") else set()


def _show(cell: Optional[dict]) -> str:
    if cell is None:
        return "<no cell>"
    if not cell.get("on"):
        return "off" + (f" ({cell['why']})" if cell.get("why") else "")
    return "on" + (f" [{cell['cfg']}]" if cell.get("cfg") is not None else "")


def diff_cells(want: dict, got: dict) -> list:
    """``[(kind, cell id, expected, got)]``, kind REGRESSION (a lever lost, a config gone, a cell gone), GAIN (a lever
    engaged that was off, a new cell) or CHANGE (still engaged, different config / reason)."""
    out: list = []
    for cid in sorted(set(want) | set(got)):
        w, g = want.get(cid), got.get(cid)
        if w == g:
            continue
        if g is None:
            out.append(("REGRESSION", cid, _show(w), "cell gone"))
            continue
        if w is None:
            out.append(("GAIN", cid, "<no cell>", _show(g)))
            continue
        if "engaged" in w or "engaged" in g:
            lost = sorted(_engaged_set(w) - _engaged_set(g))
            won = sorted(_engaged_set(g) - _engaged_set(w))
            for item in lost:
                out.append(("REGRESSION", f"{cid}.{item}", "engaged", "off"))
            for item in won:
                out.append(("GAIN", f"{cid}.{item}", "off", "engaged"))
            rest_w = {k: v for k, v in w.items() if k not in ("engaged", "on")}
            rest_g = {k: v for k, v in g.items() if k not in ("engaged", "on")}
            if rest_w != rest_g:
                out.append(("CHANGE", cid, str(rest_w), str(rest_g)))
            continue
        if w.get("on") and not g.get("on"):
            out.append(("REGRESSION", cid, _show(w), _show(g)))
        elif g.get("on") and not w.get("on"):
            out.append(("GAIN", cid, _show(w), _show(g)))
        elif w.get("on") and w.get("cfg") is not None and g.get("cfg") is None:
            out.append(("REGRESSION", cid, _show(w), "config None"))
        else:
            out.append(("CHANGE", cid, _show(w), _show(g)))
    return out


def _split_cell(cid: str) -> tuple:
    """``profile.family.lever...`` -> (profile, family, lever)."""
    profile, _, rest = cid.partition(".")
    family, _, lever_name = rest.partition(".")
    return profile, family, lever_name


def format_diffs(diffs: list) -> str:
    """One line per first divergence ``GPU LEVER <KIND> <profile>.<family>.<lever>: expected X, got Y``; cells with the
    same (kind, family.lever, expected, got) are grouped under it with the other profiles listed."""
    groups: dict = {}
    for kind, cid, exp, got in diffs:
        profile, family, lever_name = _split_cell(cid)
        groups.setdefault((kind, family, lever_name, exp, got), []).append(profile)
    order = {"REGRESSION": 0, "CHANGE": 1, "GAIN": 2}
    lines = []
    counts = {k: sum(1 for d in diffs if d[0] == k) for k in order}
    lines.append(
        f"{counts['REGRESSION']} lever regression(s), {counts['CHANGE']} change(s), {counts['GAIN']} gain(s) "
        f"against {GOLDEN.name}:"
    )
    for (kind, family, lever_name, exp, got), profiles in sorted(
        groups.items(), key = lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2], kv[1][0])
    ):
        first = profiles[0]
        line = f"GPU LEVER {kind} {first}.{family}.{lever_name}: expected {exp}, got {got}"
        if len(profiles) > 1:
            line += f"  (same on: {', '.join(profiles[1:])})"
        lines.append(line)
    if counts["GAIN"] and not counts["REGRESSION"]:
        lines.append("Gains fail too so every policy change is reviewed: update the golden.")
    lines.append(_HOW_TO_UPDATE)
    return "\n".join(lines)


def _dump_golden(data: dict) -> str:
    """Sorted, one cell per line (reviewable diffs)."""
    import json

    head = {k: data[k] for k in ("schema", "version", "about") if k in data}
    parts = ["{"]
    for k, v in head.items():
        parts.append(f" {json.dumps(k)}: {json.dumps(v)},")
    parts.append(' "profiles": {')
    prof = data.get("profiles", {})
    parts.append(",\n".join(f"  {json.dumps(k)}: {json.dumps(prof[k])}" for k in sorted(prof)))
    parts.append(" },")
    parts.append(f' "inputs": {json.dumps(data.get("inputs", {}), sort_keys = True)},')
    parts.append(' "changes": [')
    parts.append(
        ",\n".join(f"  {json.dumps(c, sort_keys = True)}" for c in data.get("changes", []))
    )
    parts.append(" ],")
    parts.append(' "cells": {')
    cells = data.get("cells", {})
    parts.append(
        ",\n".join(
            f"  {json.dumps(k)}: {json.dumps(cells[k], sort_keys = True)}" for k in sorted(cells)
        )
    )
    parts.append(" }")
    parts.append("}")
    return "\n".join(parts) + "\n"


def _profile_summary(p: GpuProfile) -> str:
    bits = [p.name, p.arch, p.segment, f"{p.vram_gb:g} GB", f"host {p.host_ram_gb:g} GB", p.link]
    if p.parent:
        bits.append(f"synthetic, parent {p.parent}")
    return " | ".join(bits)


def write_golden(
    got: dict,
    reason: str,
    blocks: dict,
    path: Path = GOLDEN,
    old: Optional[dict] = None,
) -> dict:
    """Write ``got`` as the new golden, appending ``reason`` and the changed cells to ``changes``. Refuses an empty
    reason. Returns the written document."""
    import datetime

    reason = (reason or "").strip()
    if not reason:
        raise ValueError(
            f"{REASON_ENV} is empty: say why the levers changed before updating the golden"
        )
    old = load_golden(path) if old is None else old
    diffs = diff_cells(old.get("cells", {}), got)
    changed = [f"{kind} {cid}: {exp} -> {g}" for kind, cid, exp, g in diffs]
    if not old.get("cells"):
        changed = [f"initial capture: {len(got)} cells"]
    entry = {
        "version": int(old.get("version", 0)) + 1,
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "reason": reason,
        "regressions": sum(1 for d in diffs if d[0] == "REGRESSION"),
        "gains": sum(1 for d in diffs if d[0] == "GAIN"),
        "changes": sum(1 for d in diffs if d[0] == "CHANGE"),
        "changed": changed[:300]
        + ([f"... and {len(changed) - 300} more"] if len(changed) > 300 else []),
    }
    data = {
        "schema": GOLDEN_SCHEMA,
        "version": entry["version"],
        "about": "Speed levers each GPU profile engages (tests/test_diffusion_gpu_lever_matrix.py); update only "
        f"with {UPDATE_ENV}=1 and {REASON_ENV}.",
        "profiles": {p.key: _profile_summary(p) for p in PROFILES.values()},
        "inputs": {"denoiser_blocks": blocks},
        "changes": list(old.get("changes", [])) + [entry],
        "cells": got,
    }
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(_dump_golden(data), encoding = "utf-8")
    tmp.replace(path)
    return data


@pytest.fixture(scope = "module")
def golden() -> dict:
    data = load_golden()
    if not data.get("cells") and os.environ.get(UPDATE_ENV) != "1":
        pytest.fail(f"{GOLDEN} has no cells; create it with {UPDATE_ENV}=1 {REASON_ENV}=...")
    return data


@pytest.fixture(scope = "module")
def matrix(golden) -> dict:
    blocks = dict(golden.get("inputs", {}).get("denoiser_blocks") or {})
    if os.environ.get(UPDATE_ENV) == "1":
        try:
            blocks.update(denoiser_blocks_from_diffusers())
        except Exception:  # noqa: BLE001 - keep the recorded table where diffusers does not import
            pass
    # The plain matrix and every host-flip set, each in its own child, all at once (one wall-clock wait).
    runs = collect_flips(blocks, [None, *HOST_FLIP_SETS])
    out = runs.pop(None)
    out["blocks"] = blocks
    out["flips"] = runs
    return out


def test_gpu_lever_golden(golden, matrix):
    """Every profile x lever (a) and profile x family x speed-layer optim (b) matches the reviewed golden."""
    got = matrix["cells"]
    assert len(got) > 1000, f"only {len(got)} cells collected: the matrix lost its probes"
    profiles = {cid.split(".", 1)[0] for cid in got}
    assert profiles == set(PROFILES), f"profiles without cells: {sorted(set(PROFILES) - profiles)}"
    if os.environ.get(UPDATE_ENV) == "1":
        reason = os.environ.get(REASON_ENV, "")
        if not reason.strip():
            pytest.fail(f'{UPDATE_ENV}=1 refuses to write without {REASON_ENV}="<why>"')
        if diff_cells(golden.get("cells", {}), got) or blocks_changed(golden, matrix):
            write_golden(got, reason, matrix["blocks"], old = golden)
        return
    diffs = diff_cells(golden["cells"], got)
    assert not diffs, format_diffs(diffs)


def blocks_changed(golden: dict, matrix: dict) -> bool:
    return (golden.get("inputs", {}).get("denoiser_blocks") or {}) != matrix["blocks"]


def test_every_probe_returns_the_same_cells_on_every_profile(matrix):
    by_profile: dict = {}
    for cid in matrix["cells"]:
        profile, _, rest = cid.partition(".")
        # a profile's extra dtype branch (MPS: autoopt_fp32) repeats its default rows, it adds no other cell
        rest = rest.replace(".autoopt_fp32", ".autoopt").replace(
            ".device_target_fp32", ".device_target"
        )
        by_profile.setdefault(profile, set()).add(rest)
    sets = list(by_profile.values())
    assert sets and all(s == sets[0] for s in sets), {
        p: sorted(s ^ sets[0])[:10] for p, s in by_profile.items() if s != sets[0]
    }


def test_golden_is_sorted_versioned_and_reasoned():
    golden = load_golden()
    cells = golden["cells"]
    assert golden["schema"] == GOLDEN_SCHEMA and int(golden["version"]) >= 1
    assert list(cells) == sorted(cells)
    assert golden["changes"] and golden["changes"][-1]["version"] == golden["version"]
    assert all(str(c.get("reason", "")).strip() for c in golden["changes"])
    on_disk = (
        load_golden()
    )  # re-read: an update run in this session rewrote it after the fixture loaded
    assert GOLDEN.read_text(encoding = "utf-8") == _dump_golden(
        on_disk
    ), "the golden was hand-edited out of format"
    assert set(golden["profiles"]) == set(PROFILES)


def test_the_matrix_is_host_independent(matrix):
    """The same matrix on any host: without torchao or Triton, on a torch older than allow_fp16_accumulation, with an
    Intel XPU visible, on Windows (sys.platform win32) and on macOS. Each flip set runs in its own child; a diff here
    is a host dependency the mock does not pin (pin it in gpu_profiles.mock_profile, never in the golden). Isolate one
    flip with collect_flips(blocks, ["<name>"]) over HOST_FLIPS."""
    flipped = matrix["flips"]
    assert set(flipped) == set(HOST_FLIP_SETS)
    bad = {}
    for name, out in flipped.items():
        want = {k: v for k, v in matrix["cells"].items() if k.split(".", 1)[0] in FLIP_PROFILES}
        assert want and set(k.split(".", 1)[0] for k in out["cells"]) == set(FLIP_PROFILES)
        diffs = diff_cells(want, out["cells"])
        if diffs:
            bad[name] = format_diffs(diffs).splitlines()[:12]
    assert not bad, "the lever matrix follows the host, not the profile:\n" + "\n".join(
        f"[{name}] " + "\n  ".join(lines) for name, lines in bad.items()
    )


def test_a_mixed_card_host_resolves_each_ordinal_on_its_own_card():
    """``mock_profile(..., ordinals=...)``: device 1 answers as its own profile (a load pinned to cuda:1 must route by
    cuda:1, not by card 0)."""
    import torch

    from core.inference import diffusion_int8_attention as i8
    from core.inference.video_minimax_h3_attn import cuda_capability

    with clean_policy_env(), mock_profile(PROFILES["rtx6000_ada"], ordinals = {1: PROFILES["t4"]}):
        assert torch.cuda.device_count() == 2
        assert torch.cuda.get_device_capability(0) == (8, 9) and torch.cuda.get_device_capability(
            "cuda:1"
        ) == (7, 5)
        assert torch.cuda.get_device_name(torch.device("cuda", 1)) == PROFILES["t4"].name
        assert cuda_capability("cuda:1") == (7, 5)
        assert i8.device_config(0, 128) == i8.KERNEL_CONFIGS[(8, 9)][128]
        assert i8.device_config(1, 128) == i8.KERNEL_CONFIGS[(7, 5)][128]


def test_mock_profile_keeps_dynamo_importable():
    """Replaced torch functions keep ``__wrapped__`` (Dynamo inspects them at import, torch >= 2.14)."""
    import torch
    with mock_profile(PROFILES["mps"]):
        assert hasattr(torch.backends.mps.is_available, "__wrapped__")
        assert hasattr(torch.cuda.get_device_capability, "__wrapped__")
        import torch._dynamo  # noqa: F401


def test_streamed_graph_cell_is_what_the_arming_installs(monkeypatch):
    """The offloaded cell reads the post-placement arming's result: an arming that installs nothing loses the lever
    even though the speed layer left its pending flag set."""
    import test_diffusion_autoopt_parity as par

    from core.inference import diffusion_cuda_graph as cg
    from core.inference import diffusion_speed as ds

    fam = next(f for b, f in par._families() if f.name == "flux.2-klein")
    blocks = load_golden()["inputs"]["denoiser_blocks"]
    p = PROFILES["l4"]

    def run() -> Optional[str]:
        with clean_policy_env(), mock_profile(p, extra = _speed_layer_extra):
            pipe = _autoopt_pipe("image", fam, blocks)
            flags = ds.snapshot_backend_flags()
            try:
                res = par.autoopt_entry(
                    "image", pipe, _autoopt_target(p), fam, AUTOOPT_SPEED, True, False
                )
                assert getattr(pipe, "_unsloth_cuda_graph_after_placement", False)
                return armed_after_placement(pipe, res["views"])
            finally:
                ds.restore_backend_flags(flags)
                par._teardown()

    assert run() == "model_stable"
    monkeypatch.setattr(
        cg, "arm_after_placement", lambda pipe, logger = None, **k: ([], "nothing installed")
    )
    assert run() is None
    monkeypatch.undo()
    # an armer that refuses every denoiser an offload hook moves (resident ones would still arm)
    monkeypatch.setattr(cg, "offload_placement", lambda module: (None, "refused under offload"))
    assert run() is None


def _klein_cell(offload: bool = False) -> dict:
    import test_diffusion_autoopt_parity as par

    backend, fam = next((b, f) for b, f in par._families() if f.name == "flux.2-klein")
    p = PROFILES["l4"]
    blocks = load_golden()["inputs"]["denoiser_blocks"]
    with clean_policy_env(), mock_profile(p, extra = _speed_layer_extra):
        return autoopt_cell(p, backend, fam, blocks, _autoopt_target(p), offload)


def test_speed_cells_read_the_real_compile_gate_and_graph_installer(monkeypatch):
    """The speed layer's compile decision and resident graph install are production's own (under the pinned host),
    so a change that turns either off loses the lever in the cell instead of hiding behind a pinned True."""
    import functools

    from core.inference import diffusion_cuda_graph as cg
    from core.inference import diffusion_speed as ds

    base = _klein_cell()
    assert {"compiled", "cuda_graph"} <= set(base["engaged"]), base
    no_compile = functools.lru_cache(maxsize = 1)(lambda: False)
    monkeypatch.setattr(ds, "torch_compile_runtime_available", no_compile)
    assert "compiled" not in _klein_cell()["engaged"]
    monkeypatch.undo()
    monkeypatch.setattr(cg, "install_cuda_graphs", lambda pipe, logger = None, **k: ())
    assert "cuda_graph" not in _klein_cell()["engaged"]


_INHERITED_STUDIO_ENV = {
    "UNSLOTH_DISABLE_CUDA_GRAPH": "1",  # read per call: every graph lever off
    "UNSLOTH_STUDIO_FLUX3_FP16": "1",  # read at import: FLUX.3 on fp16 cards changes dtype
}


def test_inherited_studio_settings_cannot_move_the_golden(monkeypatch):
    """An operator's Studio settings in the shell running the tests reach neither the matrix child (it starts from an
    allowlisted environment, before any production import) nor the in-process cells (``clean_policy_env`` clears
    every UNSLOTH_* lever and opt-out)."""
    for k, v in _INHERITED_STUDIO_ENV.items():
        monkeypatch.setenv(k, v)
    golden = load_golden()
    keys = ("t4", "l4")
    got = _collect_in_child(golden["inputs"]["denoiser_blocks"], profiles = keys)["cells"]
    want = {k: v for k, v in golden["cells"].items() if k.split(".", 1)[0] in keys}
    assert not diff_cells(want, got), format_diffs(diff_cells(want, got))
    assert "cuda_graph" in _klein_cell()["engaged"]


def test_native_w8a8_cell_is_productions_prepare(monkeypatch):
    """The native W8A8 cell is ``diffusion_native_w8a8_cuda.prepare()``'s decision for the card: a policy change there
    (the rotation kernels switched off, the module's kill switch) moves the cell."""
    from core.inference import diffusion_convrot_quant as cq

    p = PROFILES["l4"]

    def cell() -> dict:
        with clean_policy_env(), mock_profile(p):
            return probe_native_w8a8_cuda(p)["any.native_w8a8.cap_gate"]

    assert cell() == {"on": True}
    with clean_policy_env(), mock_profile(PROFILES["t4"]):
        assert probe_native_w8a8_cuda(PROFILES["t4"])["any.native_w8a8.cap_gate"] == {
            "on": False,
            "why": "below_sm80",
        }
    monkeypatch.setattr(cq, "rotquant_mode", lambda: "off")
    assert cell() == {"on": False, "why": "disabled"}


def test_int8_gemm_install_cell_is_the_installers_outcome(monkeypatch):
    """install_gate is what diffusion_int8_gemm.install() does on a denoiser with an eligible int8 Linear: an installer
    that installs nothing (or arms no first-forward swap) turns the cell off even with every route's policy on."""
    from core.inference import diffusion_int8_gemm as g

    p = PROFILES["l4"]

    def cell() -> dict:
        with clean_policy_env(), mock_profile(p):
            return probe_int8_gemm(p)["any.int8_gemm.install_gate"]

    assert cell() == {"on": True, "cfg": "arch+cutlass+rotq"}
    monkeypatch.setattr(g, "install", lambda *a, **k: 0)
    assert cell() == {"on": False, "cfg": "arch+cutlass+rotq", "why": "installs_nothing"}
    monkeypatch.undo()
    monkeypatch.setattr(g, "candidates", lambda *a, **k: 0)  # no eligible Linear found
    assert cell()["on"] is False


def test_mps_speed_rows_follow_studios_dtype_resolver(matrix):
    """Studio's MPS target is bf16 when its on-device bf16 probe passes and fp32 when it fails, never fp16: the
    default MPS rows carry bf16 and the ``autoopt_fp32`` rows the fallback branch."""
    import test_diffusion_autoopt_parity as par

    cells = matrix["cells"]
    for _backend, fam in par._families():
        for off in ("", "_offload"):
            assert cells[f"mps.{fam.name}.autoopt{off}"].get("cfg") == "dtype=bfloat16", (
                fam.name,
                off,
            )
            assert cells[f"mps.{fam.name}.autoopt_fp32{off}"].get("cfg") == "dtype=float32", (
                fam.name,
                off,
            )


def test_mps_device_target_rows_cover_both_resolver_branches(matrix):
    """The MPS profile declares macOS 14+ (the bf16 probe passes): the load target is bf16, and the probe-fails branch
    is its own row in fp32. Neither reads the host's MPS device."""
    cells = matrix["cells"]
    assert cells["mps.any.device_target"] == {"on": True, "cfg": "mps/mps/bfloat16"}
    assert cells["mps.any.device_target_fp32"] == {"on": True, "cfg": "mps/mps/float32"}


def test_mock_profile_never_touches_the_hosts_mps_device(monkeypatch):
    """Under any profile every Metal / MPS reading Studio makes is pinned: the bf16 probe, torch.mps memory readings and
    sync, the VAE's Metal sync and the working-set reader. A host whose MPS device crashes on first use (a paravirtual
    macOS runner) must never reach it; the macOS host flip runs the whole matrix that way."""
    import torch

    from core.inference import diffusion_device as dd
    from utils.hardware import hardware as hw

    def crash(*_a, **_k):
        raise AssertionError("touched the host's MPS device")

    monkeypatch.setattr(dd, "_mps_supports_bfloat16", crash)
    monkeypatch.setattr(dd, "mps_synchronize", crash)
    monkeypatch.setattr(hw, "_metal_recommended_working_set_bytes", crash)
    for name in ("recommended_max_memory", "driver_allocated_memory", "synchronize"):
        if hasattr(torch.mps, name):
            monkeypatch.setattr(torch.mps, name, crash)
    for key in ("mps", "cpu", "l4"):
        p = PROFILES[key]
        with clean_policy_env(), mock_profile(p):
            assert dd._mps_supports_bfloat16(torch) is bool(p.mps)
            dd.mps_synchronize()
            assert hw._metal_recommended_working_set_bytes() == (
                int(p.vram_gb * GIB) if p.mps else 0
            )
            if p.mps:
                # torch < 2.5 has no recommended_max_memory (Studio then sizes from system RAM)
                if hasattr(torch.mps, "recommended_max_memory"):
                    assert torch.mps.recommended_max_memory() == int(p.vram_gb * GIB)
                assert torch.mps.driver_allocated_memory() == 0
            if p.mps:
                assert _device_target(p).dtype == torch.bfloat16


def test_streamed_cells_name_the_arming_branch(matrix):
    """Every offloaded cell that records graphs after placement says which placement they recorded under, read from
    the real arming on a denoiser the CPU-offload hook moves (Wan 2.2 5B, which records offloaded steps only, arms
    there too)."""
    armed = {
        k: v
        for k, v in matrix["cells"].items()
        if k.endswith("_offload") and "cuda_graph_after_placement" in (v.get("engaged") or ())
    }
    assert armed
    assert all(v.get("arming") in ("model_stable", "model", "group") for v in armed.values()), armed
    assert any(".wan2.2-ti2v-5b." in k for k in armed)


def test_offload_only_family_arms_on_its_offloaded_denoiser(monkeypatch):
    """Wan records offloaded steps only: on the offloaded denoiser the arming records under the model-offload
    placement, and an armer that refuses offloaded denoisers loses the lever."""
    import test_diffusion_autoopt_parity as par

    from core.inference import diffusion_cuda_graph as cg
    from core.inference import diffusion_speed as ds

    backend, fam = next((b, f) for b, f in par._families() if f.name == "wan2.2-ti2v-5b")
    blocks = load_golden()["inputs"]["denoiser_blocks"]
    p = PROFILES["l4"]

    def run() -> Optional[str]:
        with clean_policy_env(), mock_profile(p, extra = _speed_layer_extra):
            pipe = _autoopt_pipe(backend, fam, blocks)
            flags = ds.snapshot_backend_flags()
            try:
                res = par.autoopt_entry(
                    backend,
                    pipe,
                    family_target(backend, fam, _autoopt_target(p)),
                    fam,
                    AUTOOPT_SPEED,
                    True,
                    False,
                )
                return armed_after_placement(pipe, res["views"])
            finally:
                ds.restore_backend_flags(flags)
                par._teardown()

    assert run() == "model_stable"
    monkeypatch.setattr(cg, "offload_placement", lambda module: (None, "refused under offload"))
    assert run() is None


# Every loader that places a denoiser and then arms its graphs: (module, function). The arming call is inline in these
# (a full diffusers load drives it); the cells above run the same call on an offloaded stub.
_ARMING_LOADERS = (
    ("diffusion.py", "load_pipeline"),
    ("diffusion.py", "_engage_deferred_profile"),
    ("video.py", "load_pipeline"),
    ("video.py", "_load_h3_modular_pipeline"),
)


def test_every_loader_arms_graphs_after_placement():
    """Deleting a loader's ``arm_graphs_after_placement`` call drops whole-step graphs from its streamed renders; each
    loader must still make it, after its speed layer."""
    import ast
    for fname, func in _ARMING_LOADERS:
        tree = ast.parse((_BACKEND / "core" / "inference" / fname).read_text(encoding = "utf-8"))
        node = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func
            ),
            None,
        )
        assert node is not None, f"{fname}:{func} is gone"
        calls = [
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None)) == "arm_graphs_after_placement"
        ]
        assert calls, f"{fname}:{func} no longer arms graphs after placement"
        speed = [
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None))
            in ("run_speed_layer", "apply_speed_optims", "_apply_video_speed")
        ]
        if speed:
            assert max(calls) > min(speed), f"{fname}:{func} arms before its speed layer"


def test_gemm_tile_cells_follow_the_shape_rules(monkeypatch):
    """Moving a per-shape rule (here sm120's H3 fused-QKV N range) changes the tile cell for that production shape."""
    from core.inference import diffusion_int8_gemm as g

    p = PROFILES["rtx5090"]
    with clean_policy_env(), mock_profile(p):
        before = probe_int8_gemm_tiles(p)["any.int8_gemm.tile.h3_qkv"]
    rules = list(g._SHAPE_TILES[(12, 0)])
    first = rules[0]
    rules[0] = (65536, 65536, first[2], first[3], first[4])
    monkeypatch.setitem(g._SHAPE_TILES, (12, 0), tuple(rules))
    with clean_policy_env(), mock_profile(p):
        after = probe_int8_gemm_tiles(p)["any.int8_gemm.tile.h3_qkv"]
    assert before["on"] and before != after, (before, after)


def test_segment_quant_records_an_unexpected_refusal_as_off(monkeypatch):
    from core.inference import diffusion_segment_quant as sq

    p = PROFILES["l4"]
    real = sq.unsupported_reason

    def ada_refuses(
        precision,
        device,
        dtype = None,
    ):
        return (
            "fp8 refused on this card for a new reason"
            if precision == sq.PRECISION_FP8
            else real(precision, device, dtype)
        )

    with clean_policy_env(), mock_profile(p):
        assert probe_segment_quant(p)["any.segment_quant.fp8"]["on"] is True
        monkeypatch.setattr(sq, "unsupported_reason", ada_refuses)
        assert probe_segment_quant(p)["any.segment_quant.fp8"] == {"on": False, "why": "other"}


def test_an_update_run_passes_its_own_format_check(tmp_path):
    """The documented update writes a new version; the format check reads the file, not a stale copy."""
    path = tmp_path / "golden.json"
    write_golden({"t4.any.x": lever(True)}, "first", {}, path = path)
    stale = load_golden(path)
    write_golden({"t4.any.x": lever(False, why = "gone")}, "second", {}, path = path)
    assert path.read_text(encoding = "utf-8") == _dump_golden(load_golden(path))
    assert path.read_text(encoding = "utf-8") != _dump_golden(stale)


def test_golden_writer_refuses_an_empty_reason_and_ledgers_the_change(tmp_path):
    path = tmp_path / "golden.json"
    base = {"t4.any.x": lever(True, cfg = (1, 2)), "t4.any.y": lever(False, why = "n/a")}
    with pytest.raises(ValueError):
        write_golden(base, "  ", {}, path = path)
    first = write_golden(base, "initial", {}, path = path)
    assert first["version"] == 1
    second = write_golden(
        {"t4.any.x": lever(False, why = "gone")}, "drop x on purpose", {}, path = path
    )
    assert second["version"] == 2 and second["changes"][-1]["reason"] == "drop x on purpose"
    assert second["changes"][-1]["regressions"] == 2  # x off, y gone
    assert load_golden(path)["cells"] == {"t4.any.x": lever(False, why = "gone")}


def test_diff_flags_losses_as_regressions_and_gains_for_review():
    want = {
        "l4.minimax-h3.int8_attn_auto": lever(True, cfg = "exact_steps=4"),
        "h100.any.int8_attn.serving.hd128": lever(False, why = "no_kernel_config"),
        "l4.flux.1.autoopt": {
            "on": True,
            "engaged": ["compiled", "cuda_graph"],
            "cfg": "dtype=bfloat16",
        },
        "l4.any.gone": lever(True),
    }
    got = {
        "l4.minimax-h3.int8_attn_auto": lever(False, why = "arch_off"),
        "h100.any.int8_attn.serving.hd128": lever(True, cfg = "cuda:bn=128"),
        "l4.flux.1.autoopt": {"on": True, "engaged": ["compiled"], "cfg": "dtype=bfloat16"},
    }
    kinds = {(k, c) for k, c, _e, _g in diff_cells(want, got)}
    assert ("REGRESSION", "l4.minimax-h3.int8_attn_auto") in kinds
    assert ("REGRESSION", "l4.flux.1.autoopt.cuda_graph") in kinds
    assert ("REGRESSION", "l4.any.gone") in kinds
    assert ("GAIN", "h100.any.int8_attn.serving.hd128") in kinds
    text = format_diffs(diff_cells(want, got))
    assert (
        "GPU LEVER REGRESSION l4.minimax-h3.int8_attn_auto: expected on [exact_steps=4], got off (arch_off)"
        in text
    )
    assert UPDATE_ENV in text and REASON_ENV in text


def test_mock_profile_restores_torch():
    import torch

    before = (torch.cuda.is_available, torch.cuda.get_device_capability, torch.version.hip)
    with mock_profile(PROFILES["gfx1151"]):
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 5)
        assert torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.startswith(
            "gfx1151"
        )
    with mock_profile(PROFILES["mps"]):
        assert not torch.cuda.is_available() and torch.backends.mps.is_available()
    assert (torch.cuda.is_available, torch.cuda.get_device_capability, torch.version.hip) == before


# Every lever a synthetic profile's parent engages must engage on the child (same resources, newer arch), unless
# listed here with the reason it cannot follow yet. Keys are ``<child>.<family>.<lever>`` fnmatch patterns. A stale
# entry (the child engages it now, or the parent no longer does) fails, so the list only shrinks.
_NEEDS = "not raced, so not enabled blind: needs {} hardware to measure"
FUTURE_ARCH_EXCLUSIONS: dict = {
    # INT8 GEMM / ConvRot: the child keeps the lever through the unlisted-arch RACE (int8_gemm.install_gate stays
    # on: _FALLBACK_CONFIG, which is sm120's tile, and _UNLISTED_RACE_TILES race cuBLAS; convrot.unlisted_h16 races
    # the stock rotation). Copying sm120's listed entry instead would skip that race, so only the table cell drops.
    "sm1[23][01].any.int8_gemm.arch_tile": "served by the unlisted-arch race (install_gate on); a listed tile skips it",
    "sm1[23][01].any.convrot.rotq_tile": "served by the unlisted rotq_h16 race (convrot.unlisted_h16 on)",
    "sm1[23][01].any.convrot.k_tiles": "per-K rules of the listed rotq tile; the unlisted rotq_h16 race serves",
    "sm1[23][01].any.int8_gemm.tile.*": "served by the unlisted-arch race (its tile is picked on the card at load)",
    "sm1[23][01].any.int8_gemm.shape_tiles": _NEEDS.format("sm121 / sm130")
    + " (per-shape tiles of sm120)",
    "sm1[23][01].any.int8_gemm.tma_tile": _NEEDS.format("sm121 / sm130")
    + " (host-TMA kernel, sm120 only)",
    # The Triton INT8 attention stand-in serves only when the CUDA kernel fails to build; the CUDA kernel itself
    # follows by the nearest-family fallback (int8_attn.serving stays on).
    "sm1[23][01].any.int8_attn.triton_table.*": _NEEDS.format("sm121 / sm130")
    + " (Triton stand-in tiles)",
    # The fallback compiles for the card's own arch, and the NVRTC the profiles declare (13.0) targets sm75..sm121.
    "sm130.any.int8_attn.serving.*": "the pinned NVRTC cannot target sm130 (the fallback is gated on the toolchain)",
    "sm130.minimax-h3.int8_exact_policy": "no INT8 attention kernel serves sm130 (see int8_attn.serving), so Auto "
    "engages no INT8 attention and installs no exact start",
    "sm1[23][01].any.grouped_int8_gemm.tiles": _NEEDS.format("sm121 / sm130")
    + " (spill-measured grouped tiles)",
    "sm1[123][01].any.nvfp4.*": _NEEDS.format("sm110 / sm121 / sm130")
    + " (FlashInfer NVFP4 cubins ship per arch)",
    "sm110.any.convrot.h16_tf32x3": _NEEDS.format("sm110")
    + " (tf32x3 vs IEEE second stage; IEEE still runs)",
    "sm1[0123][0-3].*.inductor_reduction_filter": _NEEDS.format("the new arch")
    + " (compile filters per measured arch)",
    "sm1[13]0.any.sd_cpp_q8_cublas": _NEEDS.format("sm110 / sm130")
    + " (sd.cpp cuBLAS vs MMQ crossover by major)",
    "sm1[23][01].any.vae_*": _NEEDS.format("sm121 / sm130")
    + " (VAE strip / fold decode measured per card)",
    # ROCm: tables keyed by the exact gfx ISA; gfx12 is not gfx11 with more of it.
    "gfx1250.any.int8_attn.serving.*": _NEEDS.format("gfx1250")
    + " (ROCm Triton INT8 attention tiles per gfx ISA)",
    "gfx1250.any.transformer_quant.rdna_explicit_w8a8_arch": _NEEDS.format("gfx1250")
    + " (RDNA W8A8 rows)",
    "gfx1250.any.vae_upsample_fold": _NEEDS.format("gfx1250")
    + " (MIOpen fold measured on gfx1151)",
    "gfx1250.*.inductor_reduction_filter": _NEEDS.format("gfx1250"),
}


def _child_drops(cells: dict) -> list:
    """``[(child cell id, parent cell id, item)]`` for every parent lever the child does not engage."""
    drops = []
    for p in PROFILES.values():
        if not p.parent:
            continue
        prefix = p.parent + "."
        for pid, pcell in cells.items():
            if not pid.startswith(prefix):
                continue
            cid = p.key + "." + pid[len(prefix) :]
            lost = _engaged_set(pcell) - _engaged_set(cells.get(cid))
            for item in sorted(lost):
                drops.append((cid if item == "<lever>" else f"{cid}.{item}", pid, item))
    return drops


def _excluded(cid: str) -> Optional[str]:
    import fnmatch
    for pattern, reason in FUTURE_ARCH_EXCLUSIONS.items():
        if fnmatch.fnmatchcase(cid, pattern):
            return pattern
    return None


def test_future_archs_keep_every_parent_lever(matrix):
    drops = _child_drops(matrix["cells"])
    unexplained = [(cid, pid) for cid, pid, _item in drops if _excluded(cid) is None]
    assert not unexplained, (
        "a future arch loses a lever its parent engages (same resources, newer arch):\n"
        + "\n".join(
            f"GPU LEVER REGRESSION {cid}: expected engaged (as {pid}), got off"
            for cid, pid in unexplained
        )
        + "\nFix the policy (raced levers: nearest-family fallback) or add a FUTURE_ARCH_EXCLUSIONS entry with why."
    )


def test_future_arch_exclusions_are_not_stale(matrix):
    drops = {cid for cid, _pid, _item in _child_drops(matrix["cells"])}
    used = {_excluded(cid) for cid in drops}
    stale = sorted(set(FUTURE_ARCH_EXCLUSIONS) - used)
    assert (
        not stale
    ), f"FUTURE_ARCH_EXCLUSIONS entries no child needs any more (delete them): {stale}"
    assert all(str(r).strip() for r in FUTURE_ARCH_EXCLUSIONS.values())


# One test per regression an external tester's benchmark found (rounds R4-R7), each asserting the production
# decision that would have caught it on a CPU runner. Every replay calls the real policy function; where a test
# double stands in, it stands in for hardware or for diffusers, never for the decision under test.
def _h3_video_default_gate(p: GpuProfile):
    """The H3 video load's INT8 attention gate for Studio's default request on profile ``p``: the request fields
    come from ``VideoLoadRequest`` defaults, the card capability from the same helper video.py reads it with, and
    the hosted INT8 ConvRot denoiser (what Auto loads for H3 on CUDA)."""
    from core.inference import diffusion_int8_attention as i8
    from core.inference.diffusion_attention import _auto_request
    from core.inference.video_minimax_h3_attn import cuda_capability, h3_int8_load_gate
    from models.inference import VideoLoadRequest

    fields = VideoLoadRequest.model_fields
    backend = fields["attention_backend"].default
    policy = i8.resolve_policy(fields["int8_attention"].default)
    with mock_profile(p):
        capability = cuda_capability("cuda:0")
    gate = h3_int8_load_gate(
        mode = policy.mode,
        attention_backend = backend,
        auto_attention = _auto_request(backend),
        speed_on = True,
        hosted_int8 = True,
        comfy_checkpoint = False,
        adapters = bool(fields["loras"].default),
        capability = capability,
    )
    return gate, capability


@pytest.mark.parametrize("key", ["rtx6000_ada", "l4", "rtx3090", "a10g"])
def test_incident_r7_19098_h3_video_auto_int8_on_sm89_and_sm86(key, monkeypatch):
    """R7, #19098 (fixed by #19344): an sm89 deny-list turned H3 video Auto INT8 attention off and the tester's
    RTX 6000 Ada 960x544x124 cell got 25% slower ("INT8 attention ... not eligible: 0"). Auto must race it on sm89
    and sm86 with the 4-step exact start, and an arch deny-list must make this test fail."""
    from core.inference import video_minimax_h3_attn as h3
    with clean_policy_env():
        assert (8, 9) not in h3.H3_INT8_AUTO_OFF_ARCHS and (8, 6) not in h3.H3_INT8_AUTO_OFF_ARCHS
        gate, cap = _h3_video_default_gate(PROFILES[key])
        assert cap == PROFILES[key].capability
        assert gate.auto and not gate.forced and gate.reason is None, gate
        assert gate.exact_steps == h3.H3_INT8_EXACT_START_STEPS == 4, gate
        # The replay bites: #19098's deny-list refuses Auto on sm89 (and only there).
        monkeypatch.setattr(h3, "H3_INT8_AUTO_OFF_ARCHS", frozenset({(8, 9)}))
        replay, _ = _h3_video_default_gate(PROFILES[key])
        assert replay.auto is (cap != (8, 9)), replay
        if cap == (8, 9):
            assert replay.exact_steps == 0 and "sm89" in (replay.reason or ""), replay


@pytest.mark.parametrize("key", ["rtx3090", "rtx6000_ada", "l4", "a100_40"])
def test_h3_exact_start_policy_is_a_golden_cell(key, golden, monkeypatch):
    """R8 (#19344): the 4-step exact start cost the 3090 11% and Ada 5%; step count or kernel changes must diff the golden."""
    from core.inference import video_minimax_h3_attn as h3

    p = PROFILES[key]
    cid = f"{key}.minimax-h3.int8_exact_policy"

    def cell():
        with clean_policy_env(), mock_profile(p):
            return probe_h3_video_attention(p)["minimax-h3.int8_exact_policy"]

    base = cell()
    assert base == golden["cells"][cid], (base, golden["cells"][cid])
    assert base["on"] and f"steps={h3.H3_INT8_EXACT_START_STEPS}" in base["cfg"], base
    for steps in (h3.H3_INT8_EXACT_START_STEPS - 1, h3.H3_INT8_EXACT_START_STEPS + 1):
        with monkeypatch.context() as m:
            m.setattr(h3, "H3_INT8_EXACT_START_STEPS", steps)
            changed = cell()
        diffs = diff_cells({cid: golden["cells"][cid]}, {cid: changed})
        assert diffs and cid in format_diffs(diffs), (steps, changed)
    with monkeypatch.context() as m:
        m.setattr(h3, "H3_FLASH_FASTER_ARCHS", h3.H3_FLASH_FASTER_ARCHS ^ {tuple(p.capability)})
        moved = cell()
    diffs = diff_cells({cid: golden["cells"][cid]}, {cid: moved})
    assert diffs and cid in format_diffs(diffs), (moved, base)
    monkeypatch.setenv(h3.H3_INT8_EXACT_STEPS_ENV, "0")
    assert cell() == base


@pytest.mark.parametrize(
    "key, active",
    [("l4", True), ("rtx3090", True), ("t4", False), ("gfx1151", False)],
)
def test_h3_exact_policy_cell_records_what_the_load_installs(key, active, golden):
    """T4 (no strided processor) and ROCm (no race) are inactive even though the gate allows 4 steps."""
    from core.inference import video_minimax_h3_attn as h3

    p = PROFILES[key]
    with clean_policy_env(), mock_profile(p):
        cell = probe_h3_video_attention(p)["minimax-h3.int8_exact_policy"]
        gate, _ = _h3_video_default_gate(p)
    assert gate.auto and gate.exact_steps == h3.H3_INT8_EXACT_START_STEPS, gate
    assert cell == golden["cells"][f"{key}.minimax-h3.int8_exact_policy"], cell
    if active:
        assert cell["on"] and f"steps={h3.H3_INT8_EXACT_START_STEPS}" in cell["cfg"], cell
    else:
        assert not cell["on"] and cell["cfg"] == "kernel=none,steps=0", cell
        assert cell["why"].startswith("inactive:"), cell


def test_incident_r5_18248_h3_video_int8_attention_auto_by_default():
    """R5, #18004 left H3 video out of Auto INT8 attention (Ada / 3090 -2 to -5%; forcing it was 20% / 37% faster);
    #18248 put it on by default and R6 measured +27% Ada, +57% 3090. Auto is the default policy, H3 video races at its
    own long-sequence points (a 4096-token race rejected A100) and every CUDA card in the matrix gets the race."""
    from core.inference import diffusion_int8_attention as i8
    with clean_policy_env():
        assert i8.resolve_policy(None).mode == "auto"
        large, small = i8.FAMILY_RACE_TOKENS["MiniMaxH3Transformer3DModel"]
        assert large >= 16384 and small >= 8192 and large > max(i8.RACE_TOKENS)
        for key, p in PROFILES.items():
            if p.vendor != "nvidia":
                continue
            gate, _ = _h3_video_default_gate(p)
            assert gate.auto and gate.exact_steps == 4, (key, gate)
        # ROCm: the gate gets no CUDA capability and Auto does not race there (a documented CUDA-only limit).
        gate, cap = _h3_video_default_gate(PROFILES["gfx1151"])
        assert cap is None


def test_incident_r5_18105_int8_gemm_race_at_the_long_row_bucket(monkeypatch):
    """R5, #17486 / #18105 / #18149: the CUTLASS INT8 GEMM race ran at 512 / 4096 rows in short bursts while H3 video
    calls every GEMM at 19318 rows; the 4096-row winner was 1.2-1.55x slower there (Ada -7 to -10%, L4 -43%). A
    19318-row call must take the 16384-row race's tile, timed sustained."""
    from core.inference import diffusion_int8_cutlass as cut

    h3_rows = 19318
    with clean_policy_env():
        points = cut.race_points()
        assert 16384 in points, points
        entry = {512: "small", 4096: "mid", 16384: "long"}
        assert cut.slot(h3_rows, entry) == "long"
        assert cut.slot(4096, entry) == "mid" and cut.slot(512, entry) == "small"
        assert cut.sustained_for(16384) and cut.sustained_for(h3_rows)
        decides, _drift = cut.sustained_decides(16384, {"today": 10.0}, {"today": 10.0}, "today")
        assert decides
        # The replay bites: without the long race (pre-#18105) the plan has no 16384-row point.
        monkeypatch.setenv(cut.HUGE_RACE_ENV, "0")
        assert 16384 not in cut.race_points()


def test_incident_r9_cutlass_winners_are_paired_with_todays_path(monkeypatch):
    """R9: on the RTX 6000 Ada (300 W) the CUTLASS tiles won every MiniMax-H3 weight shape timed alone, so 313 int8
    Linears ran them (R8: 52); the video step ran 6.13 s instead of 5.40 s at the same 291 W (SM clock 1240 -> 1035 MHz)
    and H3-Image 2048^2 lost 5% the same way. The 16384-row race (the slot video row counts take) checks its sustained
    winners in pairs with today's path, whose clock both set; image row counts keep their race as before."""
    from core.inference import diffusion_int8_cutlass as cut
    with clean_policy_env():
        for rows in (16384, 19318):
            assert cut.paired_for(rows), rows
        for rows in (512, 4160, 4608):
            assert not cut.paired_for(rows), rows
        monkeypatch.setenv(cut.PAIRED_ENV, "0")
        assert not cut.paired_for(16384)


@pytest.mark.parametrize("steps", [1, 8, 20, 30])
def test_incident_r4_17362_h3_steps_equal_evaluations(steps, monkeypatch):
    """R3 / R4, #17362: H3 video ran 19 denoiser passes for 20 requested steps (the tester's +1 to +4% was 1 to 5% behind
    at equal passes). diffusers' H3 grid of n points (terminal 0 included) runs n - 1 evaluations, so Studio asks for
    steps + 1; PDD keeps steps + 1 too. Checked on the real scheduler when this diffusers has it."""
    from core.inference import video_minimax_h3 as h3
    from core.inference.video_minimax_h3_pdd import h3_pdd_num_inference_steps

    with clean_policy_env():
        points = h3.h3_num_inference_steps(steps)
        assert points == steps + 1
        assert h3_pdd_num_inference_steps(steps) == steps + 1
        try:
            from diffusers import MiniMaxH3Scheduler
        except Exception:  # noqa: BLE001 - CI has no diffusers; the grid arithmetic above is the contract
            MiniMaxH3Scheduler = None
        if MiniMaxH3Scheduler is not None:
            sched = MiniMaxH3Scheduler(shift = 12.0)
            sched.set_timesteps(points)
            assert len(sched.timesteps) == steps
        # The replay bites: the pre-#17362 grid (kill switch) is one evaluation short.
        monkeypatch.setenv(h3.H3_STEP_EVALUATIONS_ENV, "0")
        assert h3.h3_num_inference_steps(steps) == steps


# Hosted INT8 H3-Image DiT / text encoder GB and the prefetch streamer's in-flight members (as the stream-layout
# suite plans them).
_H3I_STREAM = dict(dit_gb = 21.9, te_gb = 27.2, inflight = 2)


def _h3i_planner_replay() -> None:
    """R5, #18183: H3-Image 2048^2 on the tester's RTX 6000 Ada capped to 8 GB went out of memory at step 1 (5.52 GiB
    of session weights left 1.46 GiB) while the reference stack rendered in 8.1 s; the 3090 on PCIe 3.0 x4 streamed 19
    of 50 blocks and was 18% slower because the plan never priced the link. A card gets a layout that fits it, is
    refused at load, and a slow link keeps more blocks resident."""
    from core.inference import diffusion_minimax_h3_image as h3i
    with clean_policy_env():
        # The reported load: 7.73 GB free, 80.5 GB host.
        plan = h3i.plan_placement(vram_gb = 7.73, host_gb = 80.53, **_H3I_STREAM)
        assert plan.placement == h3i.PLACE_STREAM
        # Today's layout (session weights resident) is what ran out of memory: a parked one must be chosen.
        assert plan.parked and plan.layout in (
            h3i.LAYOUT_PARKED,
            h3i.LAYOUT_PARKED_VAE,
        ), plan.reasons
        assert plan.host_need_gb is not None and plan.host_need_gb <= 80.53
        # Every budget from 4 to 24 GB: a stream / resident layout, or a refusal naming the free memory it needs.
        for tenth in range(40, 245, 5):
            vram = tenth / 10
            try:
                got = h3i.plan_placement(vram_gb = vram, host_gb = 80.53, **_H3I_STREAM)
            except RuntimeError as exc:
                assert "free GPU memory" in str(exc), (vram, exc)
                assert vram < 7.73, f"{vram} GB refused though 7.73 GB renders"
                continue
            assert got.placement in h3i.PLACEMENTS, (vram, got)
        # The 3090 on PCIe 3.0 x4 (3 GiB/s): the priced plan parks and keeps more blocks than the unpriced one.
        p = PROFILES["rtx3090"]
        assert p.link == "pcie3x4"
        unpriced = h3i.plan_placement(vram_gb = 25.06, host_gb = 84.44, **_H3I_STREAM)
        slow = h3i.plan_placement(
            vram_gb = 25.06, host_gb = 84.44, h2d_gib_s = 3.0, block_tops = 220 * 0.67, **_H3I_STREAM
        )
        assert slow.parked and slow.dit_resident_blocks > unpriced.dit_resident_blocks, slow.reasons
        # The replay bites: today's layout regardless of fit (pre-#18183) leaves the 8 GB card unparked.
        real = h3i._choose_layout
        h3i._choose_layout = lambda candidates, **_k: (candidates[0], "")
        try:
            bad = h3i.plan_placement(vram_gb = 7.73, host_gb = 80.53, **_H3I_STREAM)
        finally:
            h3i._choose_layout = real
        assert bad.layout == h3i.LAYOUT_TODAY and bad.dit_resident_blocks == 0


def test_incident_r5_18183_h3_image_8gb_plan_fits_or_refuses():
    """R5, #18183 (see ``_h3i_planner_replay``). The planner module imports diffusers when present, and a broken
    optional diffusers (a lazy import against a mismatched huggingface_hub) must not fail the replay: it runs in a child
    with diffusers hidden, as the matrix and Backend CI do."""
    code = (
        "import sys; sys.modules['diffusers'] = None; "
        f"sys.path[:0] = [{str(_BACKEND)!r}, {str(_TESTS)!r}]; "
        "import test_diffusion_gpu_lever_matrix as m; m._h3i_planner_replay(); print('REPLAY_OK')"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        timeout = 600,
        env = _child_env(None),
    )
    assert run.returncode == 0 and "REPLAY_OK" in run.stdout, (
        run.stdout[-2000:] + run.stderr[-6000:]
    )


class _Spec:
    def __init__(self, name, load):
        self.name = name
        self.load = load
        self.type_hint = None
        self.pretrained_model_name_or_path = "unsloth/MiniMax-H3"
        self.subfolder = name


class _ModularPipe:
    """What diffusers' ModularPipeline.load_components does with a component that fails: logs and registers None."""

    def __init__(self, specs):
        self._component_specs = {s.name: s for s in specs}
        for s in specs:
            setattr(self, s.name, None)

    def load_components(self, **kwargs):
        for name, spec in self._component_specs.items():
            try:
                value = spec.load(**kwargs)
            except Exception:  # noqa: BLE001 - diffusers swallows it
                value = None
            setattr(self, name, value)


def test_incident_r5_18157_missing_component_fails_the_load():
    """R5, #18157: a MiniMax-H3 VAE shard that failed to download (disk full, then a 404 on the fallback) still let
    the load succeed; the first render failed with ``AttributeError: 'NoneType' ... 'config'``. The strict loader
    fails the load naming the component, and the H3 video load path uses it with no optional VAE."""
    import ast
    import errno
    import inspect

    from core.inference import video
    from core.inference.diffusion_component_fetch import ComponentFetchError, load_components_strict

    def vae_load(**_kw):
        raise OSError(errno.ENOSPC, "No space left on device")

    specs = [_Spec("transformer", lambda **_kw: object()), _Spec("vae", vae_load)]
    # The replay bites: diffusers alone loads "successfully" with vae None (the R5 failure).
    plain = _ModularPipe(specs)
    plain.load_components()
    assert plain.vae is None and plain.transformer is not None
    with pytest.raises(ComponentFetchError) as info:
        load_components_strict(_ModularPipe(specs), cache_dir = None)
    assert info.value.component == "vae" and info.value.kind == "disk_full", info.value
    # The video load calls it, never with the VAE (or any H3 component) optional.
    tree = ast.parse(inspect.getsource(video))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "load_components_strict"
    ]
    assert calls, "video.py no longer loads components strictly"
    for call in calls:
        assert not any(kw.arg == "optional" for kw in call.keywords), ast.unparse(call)


@pytest.mark.parametrize("family", ["minimax-h3-image", "flux.3-klein"])
def test_incident_r6_upsampler_never_starts_split_on_small_cards(family, monkeypatch, tmp_path):
    """R6, #17792 (fixed by #18627): the default-on prompt upsampler sized ``-ngl`` to free VRAM, so on the tester's
    8 / 12 / 16 GB Ada budgets the 6 GB GGUF split onto the CPU and captions took 507-628 s against a 7.8 s render.
    For every free-memory reading a render's upsampler starts with every layer on the GPU, or not at all."""
    import torch

    from core.inference import flux3_prompt_upsampler as pup
    from core.inference import llama_cpp, stt_mtmd_sidecar, stt_sidecar
    from utils import process_lifetime

    model = tmp_path / "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    with open(model, "wb") as fh:
        fh.truncate(6 * 1024**3)  # sparse: only its size is read
    state = {"free": 0, "argv": []}

    class _Started(Exception):
        pass

    def _popen(
        cmd,
        env = None,
        **_kw,
    ):
        state["argv"].append(cmd)
        raise _Started

    monkeypatch.delenv(pup.ENV_ALLOW_PARTIAL_OFFLOAD, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False)
    )
    monkeypatch.setattr(pup, "_resident_cuda_ordinal", lambda: 0)
    monkeypatch.setattr(
        pup, "_accelerator_memory_mib", lambda o = None: (state["free"], 48 * 1024, "cuda")
    )
    monkeypatch.setattr(pup, "_gpu_unowned", lambda: False)
    monkeypatch.setattr(pup, "_image_model_owns_gpu", lambda: True)
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    monkeypatch.setattr(stt_mtmd_sidecar, "_llama_server_child_env", lambda b: {})
    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: False)
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: False)
    monkeypatch.setattr(process_lifetime, "spawn_on_lifetime_thread", lambda fn: fn())
    monkeypatch.setattr(pup.subprocess, "Popen", _popen)
    assert pup.supports_family(family) and pup.resolve_enhance(family, None)  # on by default

    def render_start(free_mib):
        state["free"] = free_mib
        token = pup._GENERATE_NEEDS_FULL_OFFLOAD.set(True)
        try:
            pup.PromptUpsamplerSidecar()._spawn_server("/x/llama-server", str(model), None)
        except _Started:
            argv = state["argv"][-1]
            return int(argv[argv.index("-ngl") + 1])
        except pup.UpsamplerUnavailableError:
            return None
        finally:
            pup._GENERATE_NEEDS_FULL_OFFLOAD.reset(token)
        raise AssertionError("the start neither spawned nor refused")

    started = {}
    for free_mib in range(0, 16 * 1024 + 1, 512):
        started[free_mib] = render_start(free_mib)
        assert started[free_mib] in (None, 99), (free_mib, started[free_mib])
    assert started[16 * 1024] == 99, "a card that fits the GGUF must still enhance"
    # The replay bites: the old split start (#17792 behaviour, now behind the opt-in) runs part on the CPU.
    monkeypatch.setenv(pup.ENV_ALLOW_PARTIAL_OFFLOAD, "1")
    assert render_start(4096) not in (None, 99)


@pytest.mark.parametrize(
    "key,flash", [("a100_40", True), ("l4", True), ("rtx6000_ada", True), ("rtx5090", False)]
)
def test_incident_r4_h3_flash_over_cudnn_where_measured(key, flash):
    """R4: H3 per-step 2-5% behind on the tester's cards; on sm89 the attention pick is flash over cuDNN (measured
    faster at H3's shape), sm120 keeps cuDNN (ties or loses). sm86 (the 3090) is NOT in ``H3_FLASH_FASTER_ARCHS`` on
    main, unmeasured, and keeps cuDNN: R4 named that as one cause, so it is pinned in the lever golden, not here."""
    import types

    from core.inference.video_minimax_h3_attn import h3_attention_backend

    p = PROFILES[key]
    with clean_policy_env(), mock_profile(p):
        import torch
        target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
        got = h3_attention_backend(
            "_native_cudnn", capability = p.capability, requested = None, target = target
        )
    assert got == ("_native_flash" if flash else "_native_cudnn"), (key, got)


def test_incident_r1_12874_convrot_default_has_a_fused_route_on_every_arch():
    """R1, #12874 (fixed by #14988): INT8 ConvRot became Qwen-Image-2.1's default while sm86 had neither an
    ``_ARCH_CONFIG`` GEMM tile nor a ``_ROTQ_CONFIG`` rotation tile, so the 3090 ran stock int8 plus a separate
    Hadamard pass: 22% slower (15.99 vs 12.41 s at 1024^2). Wherever a family defaults ConvRot on, every sm80+
    NVIDIA profile must get a fused rotation (its tile, or the raced unlisted rotq_h16 path) and a fused / raced
    GEMM route; B200-class cards skip the Triton GEMM on purpose (measured slower than cuBLAS, ``_UNLISTED_SKIP``)."""
    from core.inference import diffusion_convrot_quant as cq
    from core.inference import diffusion_int8_gemm as g
    from core.inference import diffusion_transformer_quant as tq

    def routes(p: GpuProfile) -> tuple:
        with clean_policy_env(), mock_profile(p):
            cap = p.capability
            rotation = cq.rotquant_config(cap) is not None or cq.unlisted_wanted(cap)
            gemm = g.arch_config(cap) is not None or g.unlisted_wanted(cap) or g.cutlass_wanted(cap)
            return rotation, gemm

    with clean_policy_env():
        families = sorted(
            f for f in tq._INT8_FAMILY_CONVROT_DEFAULT_ON if tq.int8_convrot_enabled(f)
        )
    assert "qwen-image-2.1" in families
    cards = [p for p in PROFILES.values() if nvidia(p) and p.capability >= (8, 0)]
    assert any(p.capability == (8, 6) for p in cards)
    for p in cards:
        rotation, gemm = routes(p)
        assert rotation, f"{p.key}: ConvRot defaults on for {families} with no fused rotation"
        assert (
            gemm or tuple(p.capability) in g._UNLISTED_SKIP
        ), f"{p.key}: ConvRot on, no fused / raced GEMM"
    # The replay bites: without #14988's unlisted-arch route the 3090 is back to stock rotation.
    from unittest import mock

    with mock.patch.object(cq, "unlisted_mode", lambda: "off"):
        assert routes(PROFILES["rtx3090"])[0] is False


def test_incident_r1_14979_learned_peak_keeps_the_encoders_resident(tmp_path, monkeypatch):
    """R1, #14979 / #15048: the learned activation peak (per GPU name) only ever rose; after one Qwen-Image-2.1 render
    the 3090's reserve went 2304 -> 6912 MiB and later loads kept 4877 of 8917 MiB of text encoders resident, so every
    new prompt streamed the rest (+0.9-1.3 s, +6-10%). With the per-phase peaks a later load on the 24 GB card keeps
    every encoder resident."""
    import core.inference.diffusion_measured_peaks as mp
    import core.inference.diffusion_memory as dm

    p = PROFILES["rtx3090"]
    with clean_policy_env(), mock_profile(p):
        monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
        mp._cache.update(path = None, mtime = None, data = None)
        monkeypatch.setattr(dm, "_free_reading_untrusted", lambda: False, raising = False)
        monkeypatch.setattr(mp, "current_device_name", lambda: p.name)
        monkeypatch.setattr(mp, "device_name", lambda dev: p.name)
        loaded = {
            "transformer": (6922, "dit"),
            "text_encoder": (8917, "text_encoder"),
            "vae": (254, "other"),
        }
        monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(loaded))
        monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
        monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: "torchao:Int8Tensor")
        total = int(p.vram_gb * 1024)
        budget = 21013  # what the tester's auto load logged on this card
        memory = dm.DeviceMemory(
            "cuda", "cuda", "discrete_vram", budget + dm._reserve_mib("discrete_vram", total), total
        )
        plan = dm.plan_diffusion_memory(
            target = _types.SimpleNamespace(supports_model_cpu_offload = True),
            device_memory = memory,
            runtime_headroom_mib = dm.estimate_image_runtime_mib(
                width = None, height = None, family = "qwen-image-2.1"
            ),
            requested_mode = None,
            model_dense_mib = 6922 + 8917 + 254,
            companion_dense_mib = 8917 + 254,
            text_encoder_dense_mib = 8917,
        )
        if plan.offload_policy == dm.OFFLOAD_MODEL:
            plan = dm.torchao_streaming_plan(plan)
        pipe = _types.SimpleNamespace(_unsloth_phase_seeds = True)

        def refine() -> Any:
            return dm.refine_plan_from_loaded_weights(
                pipe, plan, family = "qwen-image-2.1", speed_mode = "default"
            )

        first = refine()
        assert first.resident_text_encoder_mib == 8917
        pad = 4187  # a render's peak: 1864 MiB decode (encode + steps ~900) plus the streamed-window pad
        mp.record(
            family = "qwen-image-2.1",
            quant = "torchao:Int8Tensor",
            speed_mode = "default",
            width = 1024,
            height = 1024,
            peak_mib = 1864 + pad,
            phases = {
                mp.PHASE_PRE_DECODE: 900 + pad,
                mp.PHASE_DECODE: 1864 + pad,
                mp.PHASE_DECODE_UNPADDED: 1864,
                mp.PHASE_UNPADDED: 1864,
            },
        )
        assert (
            refine().resident_text_encoder_mib == 8917
        ), "a learned peak shrank the encoder residency"
        # The replay bites: the whole-render peak alone (pre-#14979) keeps only part of the encoders resident.
        monkeypatch.setenv(mp.PHASE_PEAKS_ENV, "0")
        whole = refine()
        assert (
            whole.estimates.get("flat_resident_text_encoder_mib", whole.resident_text_encoder_mib)
            < 8917
        )


def test_incident_r2_15642_pin_budget_inside_a_memorymax_scope(tmp_path, monkeypatch):
    """R2, #15642: inside a MemoryMax=70G scope ``_pin_budget_mib()`` read 7.7 GB one day and 0 the next for the same
    live set (mapped checkpoint pages credited only on the inactive LRU list), so H3 at 12 GB streamed from unpinned
    memory: 9% slower. The budget is the same, and not 0, on either list."""
    from core.inference import diffusion_memory as dm
    from utils import host_memory

    root, proc = tmp_path / "cgroup", tmp_path / "proc_self_cgroup"
    root.mkdir()
    monkeypatch.setattr(host_memory, "CGROUP_ROOT", str(root))
    monkeypatch.setattr(host_memory, "PROC_SELF_CGROUP", str(proc))
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_CGROUP_ROOT", str(root), raising = False)
    monkeypatch.setattr(llama_cpp, "_PROC_SELF_CGROUP", str(proc), raising = False)
    gib = 1024**3

    def budget(inactive: int, active: int) -> int:
        proc.write_text("0::/app.slice/run-x.scope\n", encoding = "utf-8")
        d = root / "app.slice" / "run-x.scope"
        d.mkdir(parents = True, exist_ok = True)
        (d / "memory.max").write_text(str(70 * gib), encoding = "utf-8")
        (d / "memory.current").write_text(str(60 * gib), encoding = "utf-8")
        stat = dict(
            anon = 40 * gib,
            file_mapped = 20 * gib,
            file_dirty = 0,
            file_writeback = 0,
            inactive_file = inactive,
            active_file = active,
        )
        (d / "memory.stat").write_text(
            "".join(f"{k} {v}\n" for k, v in stat.items()), encoding = "utf-8"
        )
        return dm._pin_budget_mib()

    p = dataclasses_replace(PROFILES["rtx6000_ada"], host_ram_gb = 94.0)
    with clean_policy_env(), mock_profile(p):
        on_inactive, on_active = budget(20 * gib, 0), budget(0, 20 * gib)
        assert on_inactive == on_active > 0, (on_inactive, on_active)
        # The replay bites: without the mapped-page credit the active-list reading drops to 0.
        monkeypatch.setenv(dm.PIN_CREDIT_MAPPED_ENV, "0")
        assert budget(0, 20 * gib) == 0


def dataclasses_replace(p: GpuProfile, **changes: Any) -> GpuProfile:
    import dataclasses
    return dataclasses.replace(p, **changes)


@pytest.mark.parametrize("speed", ["default", "max"])
def test_incident_r2_16764_h3_int8_vae_decoder_on_by_default(speed):
    """R2, #16764: the H3 INT8 video decoder was opt-in only (``UNSLOTH_H3_VAE_INT8=1``) and decode took 13-17 s of the
    render (full card 3%, 3090 5% slower). It is a lossless-enough lever (LPIPS 0.0004-0.0013, 30-33% faster decode)
    and is on at the default and max tiers without any env."""
    from core.inference import video_minimax_h3_vae as hv
    with clean_policy_env():
        assert hv.LEVER_INT8_DECODER in hv.plan_h3_vae_levers(speed)
        assert hv.LEVER_INT8_DECODER not in hv.plan_h3_vae_levers(
            "off"
        )  # speed off stays bit-identical


@pytest.mark.parametrize(
    "key", ["rtx3090", "a10g", "l4", "rtx6000_ada", "t4", "a100_40", "rtx5090"]
)
def test_incident_r3_stream_tier_keeps_cuda_graphs(matrix, key):
    """R3 / R4: the 8 GB streaming tier ran with CUDA graphs off. Offloaded / streamed loads arm graphs after
    placement on every family that is capture-safe; H3 video stays off by measurement (its streamed group onloads
    wait on the host, which a graph cannot record), so it is not asserted here. The cell is what the loader's
    post-placement arming installs; families that record offloaded steps only (Wan) need a real offload hook, which
    a CPU stub does not carry (test_diffusion_offload_cuda_graph covers their replay on a GPU)."""
    cells = matrix["cells"]
    for fam in (
        "flux.1",
        "flux.2-klein",
        "flux.3-klein",
        "z-image",
        "minimax-h3-image",
    ):
        resident = cells[f"{key}.{fam}.autoopt"]
        streamed = cells[f"{key}.{fam}.autoopt_offload"]
        if "cuda_graph" in resident.get("engaged", ()):
            assert "cuda_graph_after_placement" in streamed.get("engaged", ()), (key, fam, streamed)


def test_a_developer_opt_out_never_reaches_the_cells(monkeypatch):
    # every env name a lever module reads to switch a lever off is cleared while cells are collected
    import os

    monkeypatch.setenv("UNSLOTH_DISABLE_FP16_ACCUM", "1")
    with clean_policy_env():
        assert "UNSLOTH_DISABLE_FP16_ACCUM" not in os.environ
    assert os.environ["UNSLOTH_DISABLE_FP16_ACCUM"] == "1"

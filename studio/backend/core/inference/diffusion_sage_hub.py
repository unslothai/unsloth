# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loads SageAttention 2 from the Hugging Face kernels hub without the ``kernels`` package.

transformers < 5.13 can only import next to kernels < 0.15 (utils/kernels_compat.py), and those releases read
the hub's model-type mirror of ``kernels-community/sage-attention``, whose builds stop at torch 2.12. The current
builds live in the kernel-type repo, so this fetches the one build variant this torch needs from there (the
mirror is the fallback), checks its file digests and CUDA archs, and imports it under a private module name.
transformers' own ``kernels`` import is never touched."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.util
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

SAGE_HUB_REPO = "kernels-community/sage-attention"
_MODULE_PREFIX = "_unsloth_sage_hub_"


def _torch_build() -> Optional[dict]:
    """``{"torch": (2, 11), "cuda": (12, 8), "cxx11": True}`` for a CUDA torch, else None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    cuda = getattr(torch.version, "cuda", None)
    if not cuda or is_rocm_torch(torch):
        return None
    tv = re.match(r"(\d+)\.(\d+)", torch.__version__)
    cv = re.match(r"(\d+)\.(\d+)", str(cuda))
    if tv is None or cv is None:
        return None
    try:
        cxx11 = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception:  # noqa: BLE001
        cxx11 = True
    return {
        "torch": (int(tv.group(1)), int(tv.group(2))),
        "cuda": (int(cv.group(1)), int(cv.group(2))),
        "cxx11": cxx11,
    }


def _platform_tag() -> Optional[str]:
    if sys.platform != "linux":
        return None
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return f"{machine}-linux" if machine in ("x86_64", "aarch64") else None


def candidate_variants(build: Optional[dict], plat: Optional[str]) -> list[str]:
    """Build directory names this torch can load, best first: exact CUDA, then older minors of the same major
    (a cu130 torch runs cu128-era builds of CUDA 13 only, so the major must match). Both naming schemes are
    listed: ``torch211-cxx11-cu128-x86_64-linux`` and, from kernel-builder 0.18, ``torch211-cu128-x86_64-linux``."""
    if build is None or plat is None:
        return []
    major, minor = build["cuda"]
    t = "torch{}{}".format(*build["torch"])
    out: list[str] = []
    for m in range(minor, -1, -1):
        cu = f"cu{major}{m}"
        if build["cxx11"]:
            out.append(f"{t}-cxx11-{cu}-{plat}")
        out.append(f"{t}-{cu}-{plat}")
    return out


def _arch_ok(archs: Any, capability: Optional[tuple[int, int]]) -> bool:
    """CUDA forward compatibility: an ``8.0`` build runs on 8.6 / 8.9, ``9.0a`` only on 9.0. Unknown -> True,
    so the accuracy probe decides."""
    if capability is None or not isinstance(archs, list) or not archs:
        return True
    for arch in archs:
        m = re.fullmatch(r"(\d+)\.(\d+)([a-z]?)", str(arch).strip())
        if m is None:
            return True
        major, minor, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
        if suffix:
            if (major, minor) == capability:
                return True
        elif major == capability[0] and minor <= capability[1]:
            return True
    return False


def _digest_mismatch(variant_dir: Path, metadata: dict) -> Optional[str]:
    """The first file whose sha256 differs from the build's metadata digest, or None (also when it has none)."""
    digest = metadata.get("digest")
    if not isinstance(digest, dict) or digest.get("algorithm") != "sha256":
        return None
    files = digest.get("files")
    if not isinstance(files, dict):
        return None
    for rel, want in files.items():
        path = variant_dir / rel
        try:
            have = base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()
        except OSError:
            return f"{rel} is missing"
        if have != want:
            return f"{rel} does not match its sha256 digest"
    return None


def _kernel_repo_type() -> Optional[str]:
    try:
        from huggingface_hub import constants
    except Exception:  # noqa: BLE001
        return None
    kernel = getattr(constants, "REPO_TYPE_KERNEL", None)
    return kernel if kernel in getattr(constants, "REPO_TYPES_WITH_KERNEL", ()) else None


def _list_variants(repo: str, repo_type: Optional[str], revision: str) -> Optional[set[str]]:
    """Build directories at ``revision``; None when the Hub cannot be asked (offline: the cache decides)."""
    from huggingface_hub import HfApi

    kwargs = {"repo_type": repo_type} if repo_type else {}
    try:
        files = HfApi().list_repo_files(repo, revision = revision, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if "RevisionNotFound" in type(exc).__name__ or "RepositoryNotFound" in type(exc).__name__:
            return set()
        return None
    return {p.split("/")[1] for p in files if p.startswith("build/") and p.count("/") >= 2}


def _download(
    repo: str, repo_type: Optional[str], revision: str, variant: str, pattern: str, offline: bool
) -> Optional[Path]:
    from huggingface_hub import snapshot_download

    kwargs = {"repo_type": repo_type} if repo_type else {}
    try:
        root = snapshot_download(
            repo,
            revision = revision,
            allow_patterns = [f"build/{variant}/{pattern}"],
            local_files_only = offline,
            **kwargs,
        )
    except Exception:  # noqa: BLE001
        return None
    path = Path(root) / "build" / variant
    return path if path.is_dir() else None


def _current_capability() -> Optional[tuple[int, int]]:
    try:
        import torch
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability())[:2]
    except Exception:  # noqa: BLE001
        pass
    return None


def _import_variant(path: Path) -> Any:
    name = _MODULE_PREFIX + "{:x}".format(ctypes.c_size_t(hash(str(path.resolve()))).value)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations = [str(path)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


# "0": call the hub ``sageattn`` as shipped on sm89 too (it synchronizes the device after every attention call)
SM89_NO_SYNC_ENV = "UNSLOTH_SAGE_SM89_NO_SYNC"


def _sm89_parts(module: Any) -> Optional[tuple]:
    """``(per_thread_int8, per_channel_fp8, attn_op)`` of the build's sm89 route, or None when it has other names."""
    core = getattr(module, "core", None)
    parts = (
        getattr(core, "per_thread_int8_triton", None),
        getattr(core, "per_channel_fp8", None),
        getattr(core, "sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", None),
    )
    if not getattr(core, "SM89_ENABLED", False) or not all(callable(p) for p in parts):
        return None
    return parts


def _on_sm89(t: Any) -> bool:
    import torch
    return bool(t.is_cuda) and tuple(torch.cuda.get_device_capability(t.device)) == (8, 9)


def without_sm89_sync(module: Any, sageattn: Any) -> Any:
    """``sageattn`` with its sm89 route run here without the ``torch.cuda.synchronize()`` the hub build issues after
    every launch: same ops, arguments and order as its ``sageattn_qk_int8_pv_fp8_cuda`` (``fp32+fp16`` accumulation),
    so the output is bit-identical, but the host no longer drains every stream (a streamed DiT's weight prefetch too)
    once per attention call. Calls outside that route (other archs, causal, LSE, odd head dims, grouped heads) take
    ``sageattn``."""
    parts = _sm89_parts(module)
    if parts is None:
        return sageattn
    per_thread_int8, per_channel_fp8, attn_op = parts
    import contextlib
    import os

    import torch

    def _sm89(
        q: Any,
        k: Any,
        v: Any,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        sm_scale: Any = None,
        return_lse: bool = False,
        **kwargs: Any,
    ) -> Any:
        seq_dim = 1 if tensor_layout == "NHD" else 2
        if (
            kwargs
            or is_causal
            or return_lse
            or tensor_layout not in ("NHD", "HND")
            or (os.environ.get(SM89_NO_SYNC_ENV) or "1").strip().lower()
            in ("0", "off", "false", "no")
            or not _on_sm89(q)
            or q.dtype not in (torch.float16, torch.bfloat16)
            or not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device)
            or q.size(-1) not in (64, 128)
            or not (k.size(-1) == v.size(-1) == q.size(-1))
            or not (q.stride(-1) == k.stride(-1) == v.stride(-1) == 1)
            # grouped heads: the stock call broadcasts the K mean
            or q.size(3 - seq_dim) != k.size(3 - seq_dim)
        ):
            return sageattn(
                q,
                k,
                v,
                tensor_layout = tensor_layout,
                is_causal = is_causal,
                sm_scale = sm_scale,
                return_lse = return_lse,
                **kwargs,
            )
        if sm_scale is None:
            sm_scale = q.size(-1) ** -0.5
        # The build's CUDA ops launch on the device's default stream whatever stream is current (its Triton
        # quantisers follow the current one), so the whole call runs on the default stream, ordered against the
        # caller's stream by events instead of a host sync.
        side = None
        if v.is_cuda:
            current = torch.cuda.current_stream(v.device)
            default = torch.cuda.default_stream(v.device)
            if current != default:
                side = (current, default)
                default.wait_stream(current)
                for t in (q, k, v):
                    t.record_stream(default)
        device = torch.cuda.device(v.device) if v.is_cuda else contextlib.nullcontext()
        stream = torch.cuda.stream(side[1]) if side is not None else contextlib.nullcontext()
        with device, stream:
            km = k.mean(dim = seq_dim, keepdim = True)
            q_int8, q_scale, k_int8, k_scale = per_thread_int8(
                q, k, km, tensor_layout = tensor_layout, BLKQ = 128, WARPQ = 32, BLKK = 64, WARPK = 64
            )
            o = torch.empty(q.size(), dtype = q.dtype, device = q.device)
            v_fp8, v_scale, _ = per_channel_fp8(
                v, tensor_layout = tensor_layout, scale_max = 2.25, smooth_v = False
            )
            layout = 0 if tensor_layout == "NHD" else 1
            attn_op(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, layout, 0, 3, sm_scale, 0)
        if side is not None:
            side[0].wait_stream(side[1])
            o.record_stream(side[0])
        return o

    _sm89.__wrapped__ = sageattn  # type: ignore[attr-defined]
    return _sm89


def load_sageattn(
    versions: tuple[int, ...],
    capability: Optional[tuple[int, int]] = None,
    repo: str = SAGE_HUB_REPO,
) -> tuple[Optional[Any], str]:
    """``(sageattn, "")`` from the first version whose build loads here, else ``(None, why)``. Never raises."""
    try:
        if capability is None:
            capability = _current_capability()
        variants = candidate_variants(_torch_build(), _platform_tag())
        if not variants:
            return None, "the hub builds need CUDA torch on Linux x86_64 or aarch64"
        sources = []
        kernel_type = _kernel_repo_type()
        if kernel_type is not None:
            sources += [(kernel_type, v) for v in versions]
        sources += [(None, v) for v in versions]
        errors = []
        for repo_type, version in sources:
            where = f"{'kernel' if repo_type else 'model'} repo v{version}"
            revision = f"v{version}"
            listed = _list_variants(repo, repo_type, revision)
            picked = [v for v in variants if listed is None or v in listed]
            if not picked:
                errors.append(f"{where}: no build for {variants[0]}")
                continue
            offline = listed is None
            path = variant = None
            for variant in picked:
                # metadata.json alone first, so an unsupported card or torch never downloads the build.
                path = _download(repo, repo_type, revision, variant, "metadata.json", offline)
                if path is not None:
                    break
            if path is None:
                errors.append(f"{where}: could not download {picked[0]}")
                continue
            try:
                metadata = json.loads((path / "metadata.json").read_text(encoding = "utf-8"))
            except Exception:  # noqa: BLE001 - old builds may ship none
                metadata = {}
            if metadata.get("python-depends"):
                errors.append(f"{where}: needs Python packages {metadata['python-depends']}")
                continue
            archs = (metadata.get("backend") or {}).get("archs")
            if not _arch_ok(archs, capability):
                errors.append(
                    f"{where}: built for CUDA archs {', '.join(map(str, archs))}, not sm{capability[0]}{capability[1]}"
                )
                continue
            path = _download(repo, repo_type, revision, variant, "*", offline)
            if path is None or not (path / "__init__.py").is_file():
                errors.append(f"{where}: could not download {variant}")
                continue
            bad = _digest_mismatch(path, metadata)
            if bad is not None:
                errors.append(f"{where} {variant}: {bad}")
                continue
            try:
                module = _import_variant(path)
                fn = getattr(module, "sageattn", None)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{where} {variant}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if callable(fn):
                return without_sm89_sync(module, fn), ""
            errors.append(f"{where} {variant}: no sageattn in the build")
        return None, "; ".join(errors)
    except Exception as exc:  # noqa: BLE001 - never break a load
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"

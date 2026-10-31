# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Low-RAM load route, used only when a dense load would not fit RAM; UNSLOTH_DIFFUSION_SMALL_HOST=0|1."""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from .rocm_bf16 import is_rocm_torch

SMALL_HOST_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST"
HOST_RAM_CHECK_ENV = "UNSLOTH_DIFFUSION_HOST_RAM_CHECK"
# test hook: caps every host reading the guards take (total, available, commit)
HOST_RAM_LIMIT_ENV = "UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB"
SMALL_HOST_ATTR = "_unsloth_small_host"

_MIB = 1024 * 1024
_STREAM_MIN_MIB = 512
_HOST_RESERVE_MIN_MIB = 3072
_HOST_RESERVE_FRACTION = 0.10
# Host RAM the fp8-storage rule (``diffusion_comfy_quant.fp8_storage_wanted``) keeps free when it lets a single file
# widen. The single-file refusal reserves at most this much, so it never refuses a widened load that rule chose.
FP8_STORAGE_HOST_RESERVE_MIB = 4096
# fp16 T5 keeps ``wo`` in fp32 (``_keep_in_fp32_modules``): 9.5 GB stored, 11.0 GB loaded.
_CONVERT_MARGIN = 1.15
_INT8_MIN_ELEMENTS = 1 << 22
# DiT denoisers the route stores as int8; a UNet (mostly convs) converts dense like any other non-encoder component
INT8_DENOISER_NAMES = ("transformer", "transformer_2", "unconditional_transformer")


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().lower()


def small_host_disabled() -> bool:
    return _env(SMALL_HOST_ENV) in ("0", "off", "false", "no")


def small_host_forced() -> bool:
    return _env(SMALL_HOST_ENV) in ("1", "on", "true", "yes", "force")


def host_ram_check_disabled() -> bool:
    return _env(HOST_RAM_CHECK_ENV) in ("0", "off", "false", "no")


# Int8 spill: the route's int8 DiT weights live in a memory-mapped scratch file, not anonymous host RAM.
# Auto only when the in-RAM route does not fit; 0 = off, 1 = always.
SPILL_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_SPILL"
SPILL_DIR_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_SPILL_DIR"
_SPILL_DISK_MARGIN_MIB = 2048
# host bytes quantisation holds at once (bias, scale, D2H staging)
_SPILL_TRANSIENT_MIB = 256
_SPILL_ALIGN = 64
_SPILL_PREFIX = "unsloth-int8-spill-"
# a spill here would be RAM again
_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs", "hugetlbfs", "rootfs"})


def spill_disabled() -> bool:
    return _env(SPILL_ENV) in ("0", "off", "false", "no")


# A first load fetches the weights before the host-RAM decision, which reads their headers.
DOWNLOAD_FIRST_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_DOWNLOAD_FIRST"


def download_first_enabled() -> bool:
    return _env(DOWNLOAD_FIRST_ENV) not in ("0", "off", "false", "no")


def spill_forced() -> bool:
    return _env(SPILL_ENV) in ("1", "on", "true", "yes", "force")


def spill_dir() -> Path:
    raw = (os.environ.get(SPILL_DIR_ENV) or "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    try:
        from utils.paths.storage_roots import cache_root as studio_cache_root
        return studio_cache_root() / "diffusion_int8_spill"
    except Exception:  # noqa: BLE001
        return Path.home() / ".cache" / "unsloth" / "diffusion_int8_spill"


def _fs_type(path: Path) -> Optional[str]:
    try:
        real = os.path.realpath(path)
        best, kind = "", None
        with open("/proc/mounts", encoding = "utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                inside = real == mnt or real.startswith(mnt.rstrip("/") + "/")
                if inside and len(mnt) >= len(best):
                    best, kind = mnt, parts[2]
        return kind
    except Exception:  # noqa: BLE001
        return None


def spill_capacity_mib(directory: Any = None) -> Optional[int]:
    """Free MiB for the spill, None if off, not Linux, RAM-backed or unwritable."""
    if spill_disabled() or not sys.platform.startswith("linux"):
        return None
    try:
        import shutil

        d = Path(directory) if directory is not None else spill_dir()
        # nearest existing ancestor: no directory is created unless the spill runs
        while not d.exists() and d.parent != d:
            d = d.parent
        kind = _fs_type(d)
        if kind is None or kind in _RAM_FILESYSTEMS or not os.access(d, os.W_OK):
            return None
        return int(shutil.disk_usage(d).free // _MIB)
    except Exception:  # noqa: BLE001
        return None


def _sweep_stale_spills(directory: Path) -> None:
    try:
        for f in directory.glob(f"{_SPILL_PREFIX}*.bin"):
            try:
                pid = int(f.name[len(_SPILL_PREFIX) :].split("-", 1)[0])
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            try:
                f.unlink()
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass


class Int8Spill:
    """Preallocated, unlinked scratch file mapped MAP_SHARED; ``store`` returns a file-backed copy of an int8 tensor."""

    def __init__(self, directory: Any, nbytes: int):
        import torch

        d = Path(directory)
        d.mkdir(parents = True, exist_ok = True)
        _sweep_stale_spills(d)
        self.nbytes = max(_SPILL_ALIGN, int(nbytes))
        path = d / f"{_SPILL_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:8]}.bin"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            try:
                # reserve now: ENOSPC under a shared mapping is SIGBUS
                os.posix_fallocate(fd, 0, self.nbytes)
            finally:
                os.close(fd)
            self.flat = torch.from_file(str(path), shared = True, size = self.nbytes, dtype = torch.int8)
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        self.offset = 0

    def store(self, q: Any) -> Any:
        n = int(q.numel())
        start = -(-self.offset // _SPILL_ALIGN) * _SPILL_ALIGN
        if start + n > self.nbytes:
            raise RuntimeError("int8 spill file sized too small")
        view = self.flat[start : start + n].view(q.shape)
        view.copy_(q)
        self.offset = start + n
        return view


def spill_bytes_for(numels: list) -> int:
    return sum(-(-int(n) // _SPILL_ALIGN) * _SPILL_ALIGN for n in numels)


_ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _safetensors_bytes_by_dtype(path: Path, keep: tuple[str, ...] = ()) -> dict[str, int]:
    return _header_sizes(path, keep)[0]


def _header_sizes(path: Path, keep: tuple[str, ...] = ()) -> tuple[dict[str, int], int]:
    """(bytes per dtype plus "kept", largest embedding bytes)."""
    from .diffusion_component_sizes import read_safetensors_header

    out: dict[str, int] = {}
    embed = 0
    header = read_safetensors_header(path, max_bytes = 256 * _MIB)
    if header is None:  # unreadable header: no claim about this file
        return out, embed
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = meta["data_offsets"]
            key = _ST_DTYPES.get(str(meta.get("dtype")), str(meta.get("dtype")).lower())
            out[key] = out.get(key, 0) + int(end) - int(start)
            if keep and any(f".{k}." in f".{name}." for k in keep):
                out["kept"] = out.get("kept", 0) + int(end) - int(start)
            if _is_embedding_name(name):
                embed = max(embed, int(end) - int(start))
        except Exception:  # noqa: BLE001
            continue
    return out, embed


def _is_embedding_name(name: str) -> bool:
    leaf = name.rsplit(".", 2)
    return "embed" in name or (len(leaf) >= 2 and leaf[-2] == "shared")


@dataclass(frozen = True)
class StoredComponent:
    name: str
    mib: int
    dtype: str  # dominant stored float dtype ("bfloat16", "float16", "float32", ...)
    # stored MiB of ``_keep_in_fp32_modules`` tensors: they convert to fp32 on the host even on the route
    kept_fp32_mib: int = 0
    library: str = ""
    # largest embedding (text encoders), converted on the host
    embed_mib: int = 0


def _keep_fp32_from_config(sub: Path) -> tuple[str, ...]:
    try:
        arch = (
            json.loads((sub / "config.json").read_text(encoding = "utf-8")).get("architectures")
            or [None]
        )[0]
        if not arch:
            return ()
        import transformers

        return _keep_fp32_patterns(getattr(transformers, str(arch), None))
    except Exception:  # noqa: BLE001
        return ()


def _weight_files(sub: Path) -> list[Path]:
    files = sorted(sub.glob("*.safetensors"))
    # ``model.fp16.safetensors`` next to ``model.safetensors`` is a variant from_pretrained does not read
    plain = [f for f in files if "." not in f.name[: -len(".safetensors")]]
    return plain or files


def snapshot_weights_complete(
    snapshot: Any,
    skip: Any = (),
    variant: Optional[str] = None,
) -> bool:
    """False when a ``model_index.json`` component lacks weights or index-listed shards; True without an index.
    ``skip`` names caller-supplied components; ``variant`` checks only that layout's files."""
    try:
        root = Path(snapshot)
        index = json.loads((root / "model_index.json").read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return True
    for name, entry in index.items():
        if name.startswith("_") or name in skip:
            continue
        if not (isinstance(entry, list) and len(entry) >= 2 and entry[0] and entry[1]):
            continue
        sub = root / name
        if not sub.is_dir():
            return False
        files = list(sub.glob("*.safetensors")) + list(sub.glob("*.bin"))
        shard_indexes = list(sub.glob("*.index.json"))
        if variant:
            marks = (f".{variant}.", f".{variant}-")
            files = [f for f in files if any(m in f.name for m in marks)]
            shard_indexes = [
                f for f in sub.glob("*.index*.json") if any(m in f.name for m in marks)
            ]
        if not files and (sub / "config.json").is_file():
            return False  # a model folder without weights; tokenizer / scheduler folders have no config.json
        for shard_index in shard_indexes:
            try:
                shards = set(
                    json.loads(shard_index.read_text(encoding = "utf-8"))
                    .get("weight_map", {})
                    .values()
                )
            except Exception:  # noqa: BLE001
                return False
            if any(not (sub / shard).is_file() for shard in shards):
                return False
    return True


def stored_components(snapshot: Any) -> dict[str, StoredComponent]:
    """Per component: stored MiB and dominant dtype, from safetensors headers only."""
    out: dict[str, StoredComponent] = {}
    try:
        root = Path(snapshot)
        if not root.is_dir():
            return out
        try:
            index = json.loads((root / "model_index.json").read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001
            index = {}
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            totals: dict[str, int] = {}
            embed = 0
            keep = _keep_fp32_from_config(sub) if sub.name.startswith("text_encoder") else ()
            for f in _weight_files(sub):
                sizes, largest = _header_sizes(f, keep)
                embed = max(embed, largest)
                for k, v in sizes.items():
                    totals[k] = totals.get(k, 0) + v
            kept = totals.pop("kept", 0)
            if not totals:
                continue
            dtype = max(totals.items(), key = lambda kv: kv[1])[0]
            entry = index.get(sub.name)
            library = str(entry[0]) if isinstance(entry, list) and entry and entry[0] else ""
            out[sub.name] = StoredComponent(
                sub.name,
                -(-sum(totals.values()) // _MIB),
                dtype,
                -(-kept // _MIB),
                library,
                -(-embed // _MIB) if sub.name.startswith("text_encoder") else 0,
            )
    except Exception:  # noqa: BLE001 - no claim
        return {}
    return out


def resolve_snapshot_dir(repo_or_dir: Any, cache_dir: Any = None) -> Optional[Path]:
    """Local diffusers dir for a repo id (cache only) or path."""
    try:
        p = Path(str(repo_or_dir)).expanduser()
        if p.is_dir():
            return p
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(str(repo_or_dir), "model_index.json", cache_dir = cache_dir)
        if isinstance(hit, str):
            return Path(hit).parent
    except Exception:  # noqa: BLE001
        return None
    return None


def _host_ram_limit_mib() -> int:
    try:
        return max(0, int((os.environ.get(HOST_RAM_LIMIT_ENV) or "").strip()))
    except ValueError:
        return 0


def _capped(value: Optional[int], limit: int) -> Optional[int]:
    if limit <= 0:
        return value
    return limit if value is None else min(value, limit)


def host_ram_mib() -> tuple[Optional[int], Optional[int]]:
    """(total, available) host MiB, both capped by an enforcing cgroup and the ``HOST_RAM_LIMIT_ENV`` test hook."""
    from .diffusion_memory import _available_system_memory_mib, _host_ram_capacity_mib

    limit = _host_ram_limit_mib()
    return _capped(_host_ram_capacity_mib(), limit), _capped(_available_system_memory_mib(), limit)


def host_commit_mib() -> Optional[int]:
    """Windows commit headroom MiB (None elsewhere or unreadable), capped by the ``HOST_RAM_LIMIT_ENV`` test hook."""
    from .diffusion_memory import _windows_commit_headroom_mib

    headroom = _windows_commit_headroom_mib()
    return None if headroom is None else _capped(headroom, _host_ram_limit_mib())


def host_reserve_mib(total_mib: Optional[int]) -> int:
    return max(_HOST_RESERVE_MIN_MIB, int((total_mib or 0) * _HOST_RESERVE_FRACTION))


def single_file_host_reserve_mib(total_mib: Optional[int]) -> int:
    """Reserve of the single-file host RAM refusal: the small-host reserve, capped at the fp8-storage rule's. Hosts
    up to 40 GB are unchanged; above, 10% would refuse loads the fp8-storage rule widens because they fit."""
    return min(host_reserve_mib(total_mib), FP8_STORAGE_HOST_RESERVE_MIB)


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _itemsize(name: str) -> int:
    return {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1}.get(
        name, 2
    )


@dataclass(frozen = True)
class SmallHostDecision:
    engaged: bool
    reason: str
    available_mib: Optional[int] = None
    total_mib: Optional[int] = None
    dense_host_mib: int = 0
    route_host_mib: int = 0
    commit_mib: Optional[int] = None
    dense_commit_mib: int = 0
    # component -> stored dtype it loads at (memory-mapped)
    storage_dtypes: dict[str, str] = field(default_factory = dict)
    refuse: Optional[str] = None
    int8_spill: bool = False
    spill_route_host_mib: int = 0


def decide_small_host(
    components: dict[str, StoredComponent],
    compute_dtype: Any,
    *,
    device: str,
    host_total_mib: Optional[int],
    host_available_mib: Optional[int],
    lora_active: bool = False,
    host_commit_mib: Optional[int] = None,
    pread_libraries: tuple[str, ...] = (),
    spill_free_mib: Optional[int] = None,
) -> SmallHostDecision:
    """Whether this pipeline load takes the small-host route. Only CUDA loads converting large bf16 components
    (fp16 / fp32 compute) change; same-dtype loads already stay memory-mapped. ``host_commit_mib`` (Windows)
    also routes a dense load whose commit need does not fit: on Windows commit exhaustion is an access
    violation in native code, not a Python error (#13432, #14513). ``spill_free_mib``: ``spill_capacity_mib``, None = no spill."""
    if small_host_disabled():
        return SmallHostDecision(False, f"{SMALL_HOST_ENV}=0")
    if str(device) != "cuda":
        return SmallHostDecision(False, "not a CUDA load")
    compute = _dtype_name(compute_dtype)
    if compute not in ("float16", "float32"):
        return SmallHostDecision(False, f"compute dtype {compute} needs no conversion")
    converted = {
        name: comp
        for name, comp in components.items()
        if comp.dtype == "bfloat16" and comp.mib >= _STREAM_MIN_MIB
    }
    if not converted:
        return SmallHostDecision(False, "no large bf16 component to convert")
    dense = int(
        sum(c.mib * _itemsize(compute) / _itemsize(c.dtype) for c in converted.values())
        * _CONVERT_MARGIN
    )
    # encoders stay memory-mapped except fp32-kept layers; a DiT lands as int8
    route = sum(
        c.mib // 2
        if n in INT8_DENOISER_NAMES
        else c.kept_fp32_mib * 4 // _itemsize(c.dtype)
        if n.startswith("text_encoder")
        else c.mib * _itemsize(compute) // _itemsize(c.dtype)
        for n, c in converted.items()
    )
    reserve = host_reserve_mib(host_total_mib)
    # Windows charges each copy-on-write safetensors view to commit until the load ends: stored bytes plus the
    # converted copy; a component read through ``pread`` (``pread_libraries``) holds no view. Linux / macOS do not
    # enforce commit.
    dense_commit = dense + int(
        sum(c.mib for c in converted.values() if c.library not in pread_libraries) * _CONVERT_MARGIN
    )
    storage = {name: comp.dtype for name, comp in converted.items()}
    forced = small_host_forced()
    if host_available_mib is None and not forced:
        return SmallHostDecision(False, "host RAM unreadable", dense_host_mib = dense)
    commit = None if host_commit_mib is None else int(host_commit_mib)
    commit_fits = commit is None or dense_commit + reserve <= commit
    fits = (
        host_available_mib is not None
        and dense + reserve <= int(host_available_mib)
        and commit_fits
    )
    commit_note = (
        ""
        if commit is None
        else f", {commit} MiB commit headroom (dense needs ~{dense_commit} MiB)"
    )
    extra = {"commit_mib": commit, "dense_commit_mib": dense_commit}
    if fits and not forced:
        return SmallHostDecision(
            False,
            f"dense {compute} load ({dense} MiB) fits {host_available_mib} MiB available host RAM{commit_note}",
            host_available_mib,
            host_total_mib,
            dense,
            route,
            **extra,
        )
    if lora_active and not forced:
        # the int8 denoiser cannot carry adapters
        return SmallHostDecision(
            False,
            f"LoRA adapters need the dense denoiser (dense load ~{dense} MiB, {host_available_mib} MiB available"
            f"{commit_note})",
            host_available_mib,
            host_total_mib,
            dense,
            route,
            **extra,
        )
    spill, spill_route = _spill_choice(
        converted, compute, route, reserve, host_available_mib, spill_free_mib
    )
    need = spill_route if spill else route
    refuse = None
    if (
        host_available_mib is not None
        and need + reserve > int(host_available_mib)
        and not host_ram_check_disabled()
    ):
        refuse = _refusal(need, reserve, host_available_mib, "even the low-memory route")
    why = (
        f"{SMALL_HOST_ENV}=1"
        if forced and fits
        else f"dense {compute} load needs ~{dense} MiB + {reserve} MiB reserve of host RAM, "
        f"{host_available_mib} MiB available{commit_note}"
    )
    return SmallHostDecision(
        True,
        why,
        host_available_mib,
        host_total_mib,
        dense,
        route,
        storage_dtypes = storage,
        refuse = refuse,
        int8_spill = spill,
        spill_route_host_mib = spill_route if spill else 0,
        **extra,
    )


def _spill_choice(
    converted: dict[str, StoredComponent],
    compute: str,
    route: int,
    reserve: int,
    available: Optional[int],
    spill_free_mib: Optional[int],
) -> tuple[bool, int]:
    """(spill, host MiB of the spilled route). Auto only when the in-RAM route does not fit."""
    dit = sum(c.mib // 2 for n, c in converted.items() if n in INT8_DENOISER_NAMES)
    if not dit or spill_free_mib is None or spill_disabled():
        return False, route
    short = available is not None and route + reserve > int(available)
    if not (short or spill_forced()) or dit + _SPILL_DISK_MARGIN_MIB > int(spill_free_mib):
        return False, route
    # route omits the converted encoder embedding
    embed = sum(
        c.embed_mib * _itemsize(compute) // _itemsize(c.dtype)
        for n, c in converted.items()
        if n.startswith("text_encoder")
    )
    return True, route - dit + embed + _SPILL_TRANSIENT_MIB


def _refusal(need: int, reserve: int, available: Optional[int], what: str) -> str:
    return (
        f"Not enough system RAM to load this model: {what} needs about {(need + reserve) / 1024:.1f} GB "
        f"of host memory and {((available or 0)) / 1024:.1f} GB is available. Close other applications, "
        f"pick a smaller or GGUF model, or set {HOST_RAM_CHECK_ENV}=0 to try anyway."
    )


def torch_dtype_map(decision: SmallHostDecision, compute_dtype: Any) -> Any:
    """``torch_dtype`` for ``DiffusionPipeline.from_pretrained``: stored dtype for the routed components."""
    if not decision.engaged or not decision.storage_dtypes:
        return compute_dtype
    import torch

    mapping: dict[str, Any] = {"default": compute_dtype}
    for name, dtype in decision.storage_dtypes.items():
        mapping[name] = getattr(torch, dtype)
    return mapping


INT8_ACT_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_INT8_ACT"
# Measured on a T4 (LPIPS inside the fp32-vs-bf16 spread); qwen-image-edit shares the DiT but is unmeasured.
INT8_ACT_FAMILIES = frozenset({"qwen-image"})
# torch._int_mm: rows M > 16, K and N multiples of 8.
_INT8_ACT_MIN_ROWS = 17
_INT8_ACT_ALIGN = 8
_INT8_ACT_DEVICE_OK: dict[int, bool] = {}
_INT8_ACT_COUNTS = {"int8_act": 0, "dequant": 0}


def int8_act_disabled() -> bool:
    return _env(INT8_ACT_ENV) in ("0", "off", "false", "no")


def int8_act_family(fam: Any) -> bool:
    """Whether ``fam``'s int8-stored denoiser may run W8A8 (still gated per device and per call)."""
    if int8_act_disabled():
        return False
    name = str(getattr(fam, "name", fam) or "").strip().lower()
    return name in INT8_ACT_FAMILIES


def int8_act_counts() -> dict[str, int]:
    return dict(_INT8_ACT_COUNTS)


def int8_act_device_ok(device: Any) -> bool:
    """sm_75 CUDA only (fp32-promoted, int8 tensor cores; on sm_80+ bf16 eager W8A8 does not win; ROCm capabilities
    differ), and ``torch._int_mm`` must return the exact integer product there."""
    import torch

    try:
        dev = torch.device(device)
        if dev.type != "cuda" or is_rocm_torch(torch):
            return False
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        return False
    hit = _INT8_ACT_DEVICE_OK.get(idx)
    if hit is not None:
        return hit
    ok = False
    try:
        if tuple(torch.cuda.get_device_capability(idx)) == (7, 5) and hasattr(torch, "_int_mm"):
            g = torch.Generator(device = "cpu").manual_seed(0)
            a = torch.randint(-127, 128, (32, 64), generator = g, dtype = torch.int8)
            b = torch.randint(-127, 128, (48, 64), generator = g, dtype = torch.int8)
            want = a.to(torch.int64) @ b.to(torch.int64).t()
            got = torch._int_mm(a.to(dev), b.to(dev).t())
            ok = bool(torch.equal(got.cpu().to(torch.int64), want))
    except Exception:  # noqa: BLE001 - no usable int8 GEMM: keep the dequantised path
        ok = False
    _INT8_ACT_DEVICE_OK[idx] = ok
    return ok


def _int8_linear_class():
    import torch
    import torch.nn.functional as F

    class Int8WeightLinear(torch.nn.Module):
        """Linear with int8 per-output-channel weights, dequantised to the input dtype for each forward.

        ``act_int8``: eligible float32 CUDA calls quantise activations per row and run ``torch._int_mm`` instead."""

        def __init__(self, qweight, scale, bias, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.qweight = torch.nn.Parameter(qweight, requires_grad = False)
            self.scale = torch.nn.Parameter(scale, requires_grad = False)
            self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad = False)
            self.act_int8 = False

        def _int8_act_ok(self, x) -> bool:
            if not self.act_int8 or x.dtype is not torch.float32 or not x.is_cuda:
                return False
            if self.in_features % _INT8_ACT_ALIGN or self.out_features % _INT8_ACT_ALIGN:
                return False
            if x.numel() // max(1, self.in_features) < _INT8_ACT_MIN_ROWS:
                return False
            return int8_act_device_ok(x.device)

        def _forward_int8_act(self, x):
            x2 = x.reshape(-1, self.in_features)
            xs = torch.linalg.vector_norm(x2, ord = float("inf"), dim = 1, keepdim = True)
            xs = xs.clamp_min_(1e-12).div_(127.0)
            xq = torch.div(x2, xs).round_().clamp_(-127, 127).to(torch.int8)
            acc = torch._int_mm(xq, self.qweight.t())
            y = torch.mul(acc, xs).mul_(self.scale.reshape(1, -1).to(torch.float32))
            if self.bias is not None:
                y.add_(self.bias.to(torch.float32))
            return y.reshape(*x.shape[:-1], self.out_features)

        def forward(self, x):
            if self._int8_act_ok(x):
                _INT8_ACT_COUNTS["int8_act"] += 1
                return self._forward_int8_act(x)
            if self.act_int8:
                _INT8_ACT_COUNTS["dequant"] += 1
            scale = self.scale
            if scale.dtype == x.dtype:
                # int8 * float promotes: cast + scale in one pass, bit-identical (int8 -> fp16/fp32 is exact).
                w = torch.mul(self.qweight, scale)
            else:
                w = self.qweight.to(x.dtype) * scale.to(x.dtype)
            return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

        def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, int8 weight"

    return Int8WeightLinear


_INT8_CLS = None


def int8_linear_class():
    global _INT8_CLS
    if _INT8_CLS is None:
        _INT8_CLS = _int8_linear_class()
    return _INT8_CLS


def quantize_int8_weight_(
    module: Any,
    *,
    compute_dtype: Any,
    work_device: Any,
    keep_device: Any = "cpu",
    act_int8: bool = False,
    spill_dir: Any = None,
) -> dict[str, int]:
    """Replace every large ``nn.Linear`` with per-row int8 weights in place, one Linear at a time (quantised on
    ``work_device``, stored on ``keep_device``); everything else casts to ``compute_dtype``. ``act_int8`` lets the
    new Linears run W8A8 where ``Int8WeightLinear`` allows it. ``spill_dir``: int8 weights go to an ``Int8Spill``
    there (same bytes, file-backed)."""
    import torch

    cls = int8_linear_class()
    work = torch.device(work_device)
    keep = torch.device(keep_device)
    stats = {"linears": 0, "dense_linears": 0, "int8_bytes": 0}
    targets = [
        (name, child) for name, child in module.named_modules() if type(child) is torch.nn.Linear
    ]
    spill = None
    if spill_dir is not None:
        sizes = [
            lin.weight.numel() for _, lin in targets if lin.weight.numel() >= _INT8_MIN_ELEMENTS
        ]
        spill = Int8Spill(spill_dir, spill_bytes_for(sizes))
    for name, lin in targets:
        if lin.weight.numel() < _INT8_MIN_ELEMENTS:
            stats["dense_linears"] += 1
            continue
        with torch.no_grad():
            w = lin.weight.detach().to(work, non_blocking = False).float()
            scale = w.abs().amax(dim = 1, keepdim = True).clamp_min(1e-12) / 127.0
            q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
            bias = None if lin.bias is None else lin.bias.detach().to(keep, compute_dtype)
            new = cls(
                q.to(keep) if spill is None else spill.store(q),
                scale.to(compute_dtype).to(keep),
                bias,
                lin.in_features,
                lin.out_features,
            )
            new.act_int8 = bool(act_int8)
            del w, q, scale
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, new)
        stats["linears"] += 1
        stats["int8_bytes"] += new.qweight.numel()
        if spill is not None:
            stats["spilled_bytes"] = stats.get("spilled_bytes", 0) + new.qweight.numel()
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, cls):
                continue
            for pname, p in list(sub.named_parameters(recurse = False)):
                if p.dtype == torch.bfloat16 and p.dtype != compute_dtype:
                    p.data = p.data.to(keep, compute_dtype)
                elif p.device != keep:
                    p.data = p.data.to(keep)
            for bname, b in list(sub.named_buffers(recurse = False)):
                if b is not None and b.dtype == torch.bfloat16 and b.dtype != compute_dtype:
                    sub._buffers[bname] = b.to(keep, compute_dtype)
    if work.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def _keep_fp32_patterns(module: Any) -> tuple[str, ...]:
    pats = getattr(module, "_keep_in_fp32_modules", None) or ()
    strict = getattr(module, "_keep_in_fp32_modules_strict", None) or ()
    return tuple(str(p) for p in (*pats, *strict))


def cast_resident_(module: Any, device: Any, compute_dtype: Any) -> None:
    """Move ``module`` to ``device`` one tensor at a time, cast as the dense load would (fp32 for kept modules)."""
    import torch

    dev = torch.device(device)
    keep = _keep_fp32_patterns(module)
    with torch.no_grad():
        for mname, sub in module.named_modules():
            want = torch.float32 if keep and any(k in mname for k in keep) else compute_dtype
            for pname, p in list(sub.named_parameters(recurse = False)):
                dt = want if p.dtype == torch.bfloat16 else p.dtype
                p.data = p.data.to(dev).to(dt)
            for bname, b in list(sub.named_buffers(recurse = False)):
                if b is None:
                    continue
                dt = want if b.dtype == torch.bfloat16 else b.dtype
                sub._buffers[bname] = b.to(dev).to(dt)


def prepare_streamed_encoder_(module: Any, compute_dtype: Any) -> int:
    """Layerwise-cast a memory-mapped bf16 encoder: Linear / Embedding weights keep their stored dtype, everything
    else converts now, including the owner of the first float parameter (``module.dtype`` sets the prompt-embedding
    dtype). Returns the MiB left memory-mapped."""
    import torch
    from diffusers.hooks import apply_layerwise_casting

    keep = _keep_fp32_patterns(module)
    streamable = (torch.nn.Linear, torch.nn.Embedding)

    def _kept(mname: str) -> bool:
        return bool(keep) and any(k in mname for k in keep)

    def _want(mname: str) -> Any:
        return torch.float32 if _kept(mname) else compute_dtype

    def _convert(sub: Any, want: Any) -> None:
        # fp32 buffers (RoPE ``inv_freq``) stay fp32, as in the dense load
        for pname, p in list(sub.named_parameters(recurse = False)):
            if p.dtype == torch.bfloat16 and p.dtype != want:
                p.data = p.data.to(want)
        for bname, b in list(sub.named_buffers(recurse = False)):
            if b is not None and b.dtype == torch.bfloat16 and b.dtype != want:
                sub._buffers[bname] = b.to(want)

    first_owner = None
    for pname, p in module.named_parameters():
        if p.is_floating_point():
            first_owner = pname.rpartition(".")[0]
            break
    streamed = 0
    with torch.no_grad():
        for mname, sub in module.named_modules():
            want = _want(mname)
            # kept-fp32 Linears convert now: T5 casts its input to ``wo.weight.dtype`` before the hook runs
            if (
                type(sub) in streamable
                and not _kept(mname)
                and mname != first_owner
                and sub.weight.dtype != want
            ):
                storage = sub.weight.dtype
                streamed += sub.weight.numel() * sub.weight.element_size()
                apply_layerwise_casting(
                    sub,
                    storage_dtype = storage,
                    compute_dtype = want,
                    skip_modules_pattern = None,
                    skip_modules_classes = None,
                )
                continue
            _convert(sub, want)
    return streamed // _MIB


def module_dtype_ok(module: Any, compute_dtype: Any) -> bool:
    try:
        return getattr(module, "dtype", compute_dtype) == compute_dtype
    except Exception:  # noqa: BLE001
        return True


def mark(pipe: Any, info: dict) -> None:
    try:
        setattr(pipe, SMALL_HOST_ATTR, dict(info))
    except Exception:  # noqa: BLE001
        pass


def engaged_on(pipe: Any) -> Optional[dict]:
    return getattr(pipe, SMALL_HOST_ATTR, None)


ENCODER_PREFETCH_ENV = "UNSLOTH_DIFFUSION_SMALL_HOST_PREFETCH"
ENCODER_PREFETCH_ATTR = "_unsloth_encoder_prefetch"
# Pinned staging ring: the only host bytes this adds.
_STAGE_SLOT_BYTES = 32 * _MIB
_STAGE_SLOTS = 4
# Device bytes copied ahead: a few groups (T5-XXL MLP weight 80 MiB, Qwen2.5-VL 130 MiB).
_PREFETCH_MAX_BYTES = 384 * _MIB
_PREFETCH_MIN_BYTES = 128 * _MIB


def encoder_prefetch_disabled() -> bool:
    return _env(ENCODER_PREFETCH_ENV) in ("0", "off", "false", "no")


def _group_tensors(group: Any) -> list:
    """(tensor, host source) pairs of a diffusers offload group, in the order diffusers onloads them."""
    out: list = []
    seen: set[int] = set()
    cpu = getattr(group, "cpu_param_dict", None) or {}
    for t in (
        [p for m in group.modules for p in m.parameters()]
        + [b for m in group.modules for b in m.buffers()]
        + list(getattr(group, "parameters", None) or [])
        + list(getattr(group, "buffers", None) or [])
    ):
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append((t, cpu.get(t, t)))
    return out


def _prefetch_budget_bytes(device: Any) -> int:
    """Bytes of encoder groups the prefetcher may hold on ``device`` ahead of compute."""
    try:
        import torch

        from .diffusion_memory import user_budget_reserve_bytes

        free, total = torch.cuda.mem_get_info(device)
        reserve = user_budget_reserve_bytes(device, total)
        free = max(0, int(free) - reserve)
        if reserve > 0 and free < _PREFETCH_MIN_BYTES:
            return 0
        return max(_PREFETCH_MIN_BYTES, min(_PREFETCH_MAX_BYTES, free // 8))
    except Exception:  # noqa: BLE001
        return _PREFETCH_MIN_BYTES


def _prefetch_budget_lowered(device: Any) -> bool:
    """True when the user's VRAM Budget keeps a reserve on ``device``."""
    try:
        import torch

        from .diffusion_memory import user_budget_reserve_bytes

        _free, total = torch.cuda.mem_get_info(device)
        return user_budget_reserve_bytes(device, total) > 0
    except Exception:  # noqa: BLE001
        return False


class _EncoderPrefetcher:
    """Copies a memory-mapped encoder's offload groups ahead of its forward, in the previous forward's order, on a
    worker thread through a pinned ring (diffusers pins each group on the calling thread). Off-order groups and worker
    errors fall back to a synchronous copy."""

    def __init__(self, module: Any, groups: list, device: Any):
        import threading

        import torch

        self.module = module
        self.groups = groups
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self.stream = None  # side stream, created on the first copy
        self.compute = None  # the stream the encoder runs on, read when a forward begins
        self.order: list = []  # group ids in the last forward's onload order
        self.seen: list = []
        self.active = False
        self.cond = threading.Condition()
        self.ready: dict = {}  # id(group) -> (device tensors, event)
        self.inflight = 0
        self.budget = _PREFETCH_MIN_BYTES
        self.strict = False  # lowered Budget: an ahead group must fit the window, else it copies synchronously
        self.skipped: dict = {}  # group id -> times the worker left to a synchronous copy
        self.pos = 0
        self.stop = False
        self.worker = None
        self.error: Optional[BaseException] = None
        self.stage: list = []
        self.stage_events: list = []
        self.slot = 0
        # The worker and a synchronous onload copy (a group skipped under a lowered Budget) share the staging ring.
        self.copy_lock = threading.Lock()
        self.stats = {"prefetched": 0, "sync": 0, "passes": 0}
        self.by_id = {id(g): g for g in groups}

    def _side_stream(self) -> Any:
        import torch
        if self.stream is None:
            self.stream = torch.cuda.Stream(device = self.device)
        return self.stream

    def _ensure_stage(self) -> None:
        import torch
        if not self.stage:
            # inference_mode is thread-local: the worker cannot write inference tensors made on the forward thread.
            with torch.inference_mode(False):
                self.stage = [
                    torch.empty(_STAGE_SLOT_BYTES, dtype = torch.uint8, pin_memory = True)
                    for _ in range(_STAGE_SLOTS)
                ]
            self.stage_events = [None] * _STAGE_SLOTS

    def _copy_group(self, group: Any) -> tuple:
        """Device copies of ``group``'s host tensors, issued on the side stream. Returns (tensors, event, bytes)."""
        # Held across staging write, H2D enqueue and slot event so the other thread cannot reuse a slot in between.
        with self.copy_lock:
            return self._copy_group_locked(group)

    def _copy_group_locked(self, group: Any) -> tuple:
        import torch

        self._ensure_stage()
        stream = self._side_stream()
        compute = self.compute or torch.cuda.current_stream(self.device)
        moved: list = []
        nbytes = 0
        with torch.cuda.stream(stream):
            for _t, src in _group_tensors(group):
                n = int(src.numel()) * int(src.element_size())
                if src.device.type != "cpu" or n == 0 or not src.is_contiguous() or src.is_pinned():
                    with torch.cuda.stream(compute):
                        moved.append(
                            src.to(
                                self.device,
                                non_blocking = src.device.type != "cpu" or src.is_pinned(),
                            )
                        )
                    continue
                nbytes += n
                # Copy waits on the compute stream so it covers the last user of a freed block.
                with torch.cuda.stream(compute):
                    dst = torch.empty(src.shape, dtype = src.dtype, device = self.device)
                    queued = torch.cuda.Event()
                    queued.record(compute)
                stream.wait_event(queued)
                sb = src.reshape(-1).view(torch.uint8)
                db = dst.reshape(-1).view(torch.uint8)
                off = 0
                while off < n:
                    k = min(_STAGE_SLOT_BYTES, n - off)
                    i = self.slot
                    self.slot = (i + 1) % _STAGE_SLOTS
                    ev = self.stage_events[i]
                    if ev is not None:
                        ev.synchronize()
                    buf = self.stage[i][:k]
                    buf.copy_(sb[off : off + k])
                    db[off : off + k].copy_(buf, non_blocking = True)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    self.stage_events[i] = ev
                    off += k
                # A dropped entry (halt, exception) frees dst before the copy lands.
                dst.record_stream(stream)
                moved.append(dst)
            done = torch.cuda.Event()
            done.record(stream)
        return moved, done, nbytes

    def _group_bytes(self, group: Any) -> int:
        """Host bytes :meth:`_copy_group` would stage for ``group``."""
        n = 0
        for _t, src in _group_tensors(group):
            if src.device.type == "cpu" and src.is_contiguous() and not src.is_pinned():
                n += int(src.numel()) * int(src.element_size())
        return n

    def _attach(self, group: Any, moved: list, done: Any) -> None:
        import torch

        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(done)
        if getattr(group, "stream", None) is not None:
            # diffusers fences its own-stream prefetch only in the next group's onload_, which this replaces.
            cur.wait_stream(group.stream)
        for (t, _src), dev in zip(_group_tensors(group), moved):
            t.data = dev
            dev.record_stream(cur)

    def _run(self, order: list) -> None:
        import torch
        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            for gid in order:
                group = self.by_id.get(gid)
                need = self._group_bytes(group) if self.strict and group is not None else 0
                with self.cond:
                    if need > self.budget:
                        self.skipped[gid] = self.skipped.get(gid, 0) + 1
                        self.cond.notify_all()
                        continue
                    # A group the forward onloads twice waits for its first copy to be taken.
                    while not self.stop and (
                        (
                            self.inflight + need > self.budget
                            if self.strict
                            else self.inflight > 0 and self.inflight >= self.budget
                        )
                        or gid in self.ready
                    ):
                        self.cond.wait(0.05)
                    if self.stop:
                        return
                if group is None:
                    continue
                moved, done, n = self._copy_group(group)
                with self.cond:
                    if self.stop:
                        return
                    self.ready[gid] = (moved, done, n)
                    self.inflight += n
                    self.cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - every group falls back to a synchronous copy
            with self.cond:
                self.error = exc
                self.cond.notify_all()

    def _halt(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        if self.worker is not None:
            self.worker.join()
        self.worker = None
        with self.cond:
            self.ready.clear()
            self.skipped.clear()
            self.inflight = 0

    def begin(self) -> None:
        import threading

        import torch

        self._halt()
        self.stop = False
        self.error = None
        self.seen = []
        self.pos = 0
        self.active = True
        self.stats["passes"] += 1
        self.compute = (
            torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        )
        if not self.order:
            return  # first forward: synchronous copies, and it records the order
        self.budget = _prefetch_budget_bytes(self.device)
        if self.budget <= 0:
            return  # no budgeted room: every group copies synchronously, as on the first forward
        self.strict = _prefetch_budget_lowered(self.device)
        self._order_after_compute()
        self.worker = threading.Thread(
            target = self._run, args = (list(self.order),), daemon = True, name = "unsloth-encoder-prefetch"
        )
        self.worker.start()

    def _order_after_compute(self) -> None:
        """The worker's copies must not overtake work the compute stream already queued (the prompt's input ids)."""
        import torch
        self._side_stream().wait_stream(torch.cuda.current_stream(self.device))

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        self._halt()
        if self.seen:
            self.order = list(self.seen)

    def onload(self, group: Any) -> None:
        gid = id(group)
        if self.active:
            self.seen.append(gid)
        entry = None
        if self.worker is not None:
            expected = self.order[self.pos] if self.pos < len(self.order) else None
            if expected == gid:
                self.pos += 1
                with self.cond:
                    while (
                        gid not in self.ready
                        and not self.skipped.get(gid)
                        and self.error is None
                        and self.worker.is_alive()
                    ):
                        self.cond.wait(0.05)
                    entry = self.ready.pop(gid, None)
                    if entry is None and self.skipped.get(gid):
                        self.skipped[gid] -= 1
                    if entry is not None:
                        self.inflight -= entry[2]
                        self.cond.notify_all()
            else:
                # Off the recorded order: stop prefetching for this forward; the next one re-records the order.
                self._halt()
        if entry is None:
            moved, done, _n = self._copy_group(group)
            self.stats["sync"] += 1
        else:
            moved, done, _n = entry
            self.stats["prefetched"] += 1
        self._attach(group, moved, done)


def install_encoder_prefetch(
    module: Any,
    device: Any,
    logger: Any = None,
) -> int:
    """Swap each streamed offload group's ``onload_`` of a small-host text encoder for the prefetching copy. Groups made
    resident are left alone. Returns the number of groups covered (0: unchanged, e.g. kill switch or no CUDA)."""
    if encoder_prefetch_disabled():
        return 0
    try:
        import torch

        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return 0
        from .diffusion_memory import GROUP_OFFLOAD_PIN_ENV, _offload_groups

        if _env(GROUP_OFFLOAD_PIN_ENV) in ("0", "off", "false", "no"):
            return 0  # the ring is pinned memory: the user's "pin nothing" override wins
        from .host_pinning import host_pinning_allowed

        if not host_pinning_allowed(device):
            return 0  # integrated GPU: pinned memory only takes RAM from the GPU

        groups = [
            g
            for g in _offload_groups(module)
            if not getattr(g, "_unsloth_resident", False)
            and getattr(g, "stream", None) is not None
            and not getattr(g, "offload_to_disk_path", None)
        ]
        if not groups:
            return 0
        pf = _EncoderPrefetcher(module, groups, device)
        pf._ensure_stage()  # a pinned allocation failing here keeps diffusers' onload, not a failed render
        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        for group in groups:

            def onload_(
                *_a: Any,
                _g: Any = group,
                **_k: Any,
            ) -> None:
                pf.onload(_g)

            group.onload_ = disable(onload_) if callable(disable) else onload_
        module.register_forward_pre_hook(lambda *_a, **_k: pf.begin())
        module.register_forward_hook(lambda *_a, **_k: pf.end(), always_call = True)
        setattr(module, ENCODER_PREFETCH_ATTR, pf)
        if logger is not None:
            logger.info(
                "diffusion.small_host: %s streams %d offload groups through a prefetching copy",
                type(module).__name__,
                len(groups),
            )
        return len(groups)
    except Exception as exc:  # noqa: BLE001 - keep diffusers' own onload
        if logger is not None:
            logger.warning("diffusion.small_host: encoder prefetch unavailable (%s)", exc)
        return 0

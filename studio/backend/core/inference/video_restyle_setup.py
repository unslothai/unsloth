# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-click Restyle setup: the pinned sd.cpp build, then NVIDIA's Cosmos3-Nano-Sim-Transfer weights in the
single-file layout sd-server reads, written to Studio's cache.

The transformer shards are merged the way the sd.cpp fork's ``scripts/convert_cosmos3.py`` does (sorted names,
``lm_head`` and the understanding-path final norm dropped, bytes copied unchanged). The VAE is Wan-AI's own Wan2.2
release (``Wan2.2_VAE.pth``, fp32, original tensor names, which is what sd.cpp reads for Cosmos3) stored as fp16:
tensor for tensor the file the fork's docs use, so renders match the fork's validated setup. The bf16 diffusers VAE in
NVIDIA's repo is the same network rounded to bf16, and that rounding alone moved a 4-step Restyle to 21.7 dB PSNR
against the fork's setup. A card with too little memory for bf16 gets sd-cli's q8_0 conversion."""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Transfer"
DIR_ENV = "UNSLOTH_COSMOS3_TRANSFER_DIR"
SETUP_DIRNAME = "cosmos3_sim_transfer"
RECORD_NAME = "setup.json"
RECORD_VERSION = 1
TRANSFORMER_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
# Wan-AI's original Wan2.2 VAE (fp32 state dict, loaded weights-only).
VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
VAE_FILE = "Wan2.2_VAE.pth"
VAE_TENSORS = 196
MODEL_BF16 = "cosmos3_nano_sim_transfer_bf16.safetensors"
MODEL_Q8 = "cosmos3_nano_sim_transfer_q8_0.gguf"
VAE_OUT = "wan2.2_vae.safetensors"
# Tensors the video path never reads (the fork's converter drops the same two).
UNUSED_TENSORS = frozenset({"lm_head.weight", "norm.weight"})
_COPY_CHUNK = 8 * 1024 * 1024
_GIB = 1024**3
# Whole-card peaks measured at 832x480, 121 frames on an RTX PRO 6000: 39.8 GiB bf16, 28.95 GiB q8_0.
BF16_MIN_VRAM_GIB = 42.0
Q8_MIN_VRAM_GIB = 31.0
# q8_0 stores 34 bytes per 32 bf16 weights (64 bytes).
_Q8_RATIO = 34 / 64
_DISK_MARGIN = 2 * _GIB

_TERMINAL = ("done", "error", "cancelled")


class SetupError(RuntimeError):
    """Shown to the user."""


class SetupCancelled(RuntimeError):
    pass


def setup_dir() -> Path:
    raw = (os.environ.get(DIR_ENV) or "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    from utils.paths.storage_roots import cache_root

    return cache_root() / SETUP_DIRNAME


def _read_record(root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    root = setup_dir() if root is None else root
    try:
        record = json.loads((root / RECORD_NAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("version") == RECORD_VERSION else None


def installed_files(root: Optional[Path] = None) -> Optional[tuple[str, str]]:
    """(transformer, vae) a finished setup wrote, when both are still there at their recorded sizes."""
    root = setup_dir() if root is None else root
    record = _read_record(root)
    if record is None:
        return None
    out = []
    for key in ("model", "vae"):
        entry = record.get(key) or {}
        path = root / str(entry.get("file") or "")
        try:
            if not entry.get("file") or path.stat().st_size != int(entry.get("bytes", -1)):
                return None
        except OSError:
            return None
        out.append(str(path))
    return out[0], out[1]


# ---- hardware -------------------------------------------------------------------------------------------------


def _metal_working_set() -> Optional[int]:
    from utils.hardware import is_apple_silicon

    if not is_apple_silicon():
        return None
    from core.inference.mlx_diffusion_backend import working_set_bytes

    return working_set_bytes()


def _sized_plan(vram: float, short: str) -> dict[str, Any]:
    """``short``: the refusal's first sentence, with ``need`` and ``have`` (GB)."""
    if vram >= BF16_MIN_VRAM_GIB:
        return {"weights": "bf16", "vram_gib": round(vram, 1), "refusal": None}
    if vram >= Q8_MIN_VRAM_GIB:
        return {"weights": "q8_0", "vram_gib": round(vram, 1), "refusal": None}
    return {
        "weights": None,
        "vram_gib": round(vram, 1),
        "refusal": (
            short.format(need = f"{Q8_MIN_VRAM_GIB:.0f}", have = f"{vram:.1f}")
            + " The model uses about 29 GB with 8-bit weights and about 40 GB in bf16."
        ),
    }


def hardware_plan(
    *,
    platform: Optional[str] = None,
    devices: Any = "probe",
    working_set: Any = "probe",
) -> dict[str, Any]:
    """{weights: "bf16" | "q8_0" | None, refusal: str | None, vram_gib}. ``devices`` is
    utils.hardware.nvidia.get_cuda_device_info()'s answer: [] no CUDA device, None unknown (not refused).
    On a Mac, ``working_set`` is Metal's recommended working set in bytes, None without Apple Silicon."""
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        if working_set == "probe":
            working_set = _metal_working_set()
        if not working_set:
            return {
                "weights": None,
                "vram_gib": None,
                "refusal": "Restyle on a Mac needs Apple Silicon and a native (arm64) Unsloth install.",
            }
        # The working set (2/3 to 3/4 of RAM) bounds the whole footprint, measured at 121 frames in bf16 on an M3 Max:
        # 29.2 GiB on Metal + 11.1 GiB for a CPU-side VAE, or 41.4 GiB all on Metal with an IM2COL_3D build.
        return _sized_plan(
            int(working_set) / _GIB,
            "Restyle needs at least {need} GB of GPU memory; this Mac's GPU can use {have} GB.",
        )
    if devices == "probe":
        try:
            from utils.hardware.nvidia import get_cuda_device_info
            devices = get_cuda_device_info()
        except Exception as exc:  # noqa: BLE001 -- unknown is not a refusal
            logger.info("restyle setup: GPU probe failed: %s", exc)
            devices = None
    if devices is None:
        return {"weights": "bf16", "vram_gib": None, "refusal": None}
    if not devices:
        return {
            "weights": None,
            "vram_gib": None,
            "refusal": "Restyle needs an NVIDIA GPU with CUDA; none was found on this machine.",
        }
    first = devices[0]
    if first is None:
        return {"weights": "bf16", "vram_gib": None, "refusal": None}
    # sd-server renders on the first visible CUDA device.
    return _sized_plan(
        int(first[2]) / _GIB,
        "Restyle needs a GPU with at least {need} GB of memory; this one has {have} GB.",
    )


# ---- safetensors merge and rename ------------------------------------------------------------------------------


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SetupError(f"{path.name} is not a safetensors file.")
        size = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


Entry = tuple[str, dict[str, Any], Path, int]


def _entries(
    sources: Iterable[Path],
    rename: Callable[[str], str] = lambda name: name,
    drop: frozenset[str] = frozenset(),
) -> list[Entry]:
    entries: list[Entry] = []
    seen: set[str] = set()
    for path in sources:
        header, data_start = read_safetensors_header(path)
        for name, info in header.items():
            if name in drop:
                continue
            new = rename(name)
            if new in seen:
                raise SetupError(f"Tensor {new} appears twice in the source files.")
            seen.add(new)
            entries.append((new, info, path, data_start))
    entries.sort(key = lambda e: e[0])
    return entries


def write_safetensors(
    entries: list[Entry],
    out: Path,
    *,
    cancel: Optional[threading.Event] = None,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> int:
    """Stream ``entries`` into one safetensors file at ``out`` (via ``out.partial``), then verify it; returns the
    payload size. Memory stays at one copy buffer."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, info, _, _ in entries:
        begin, end = info["data_offsets"]
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + end - begin],
        }
        offset += end - begin
    header_bytes = json.dumps(header, separators = (",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    partial = out.with_name(out.name + ".partial")
    done = 0
    try:
        with open(partial, "wb") as dst:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
            for name, info, path, data_start in entries:
                begin, end = info["data_offsets"]
                with open(path, "rb") as src:
                    src.seek(data_start + begin)
                    remaining = end - begin
                    while remaining > 0:
                        if cancel is not None and cancel.is_set():
                            raise SetupCancelled("cancelled")
                        chunk = src.read(min(_COPY_CHUNK, remaining))
                        if not chunk:
                            raise SetupError(f"{path.name} is truncated at tensor {name}.")
                        dst.write(chunk)
                        remaining -= len(chunk)
                        done += len(chunk)
                        if on_bytes is not None:
                            on_bytes(done)
        _verify_written(partial, header, 8 + len(header_bytes) + offset)
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok = True)
        raise
    return offset


def _verify_written(path: Path, expected: dict[str, Any], size: int) -> None:
    if path.stat().st_size != size:
        raise SetupError(f"{path.name}: wrote {path.stat().st_size} bytes, expected {size}.")
    got, _ = read_safetensors_header(path)
    want = {k: v for k, v in expected.items() if k != "__metadata__"}
    if got != want:
        raise SetupError(f"{path.name}: the written header does not match the source tensors.")


def merge_shards(
    shards: Iterable[Path],
    out: Path,
    *,
    cancel: Optional[threading.Event] = None,
    on_bytes: Optional[Callable[[int], None]] = None,
) -> int:
    """The fork's ``convert_cosmos3.py``: every shard's tensors but the unused two, sorted, bytes unchanged."""
    entries = _entries(shards, drop = UNUSED_TENSORS)
    if not entries:
        raise SetupError("The Cosmos3 transformer shards hold no tensors.")
    return write_safetensors(entries, out, cancel = cancel, on_bytes = on_bytes)


def index_shards(index_path: Path) -> list[str]:
    index = json.loads(Path(index_path).read_text(encoding = "utf-8"))
    return sorted(set(index["weight_map"].values()))


def convert_vae(
    src: Path,
    out: Path,
    *,
    cancel: Optional[threading.Event] = None,
) -> int:
    """Wan2.2_VAE.pth (fp32, original names) -> fp16 safetensors at ``out`` (via ``out.partial``); returns the payload
    size. Loaded weights-only, so the pickle can only hold tensors."""
    import torch
    from safetensors.torch import save_file

    state = torch.load(str(src), map_location = "cpu", weights_only = True, mmap = True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or len(state) != VAE_TENSORS or "conv1.weight" not in state:
        raise SetupError(f"{VAE_REPO}/{VAE_FILE} is not the Wan2.2 VAE Studio expects.")
    half: dict[str, Any] = {}
    for name, tensor in state.items():
        if cancel is not None and cancel.is_set():
            raise SetupCancelled("cancelled")
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise SetupError(f"{VAE_FILE}: tensor {name} is not a float tensor.")
        wide = tensor.float()
        if not torch.isfinite(wide).all() or wide.abs().max().item() > 65504:
            raise SetupError(f"{VAE_FILE}: a weight in {name} is outside fp16 range.")
        half[name] = wide.to(torch.float16).contiguous()
    partial = out.with_name(out.name + ".partial")
    try:
        save_file(half, str(partial), metadata = {"format": "pt"})
        header, _ = read_safetensors_header(partial)
        if len(header) != VAE_TENSORS or any(v.get("dtype") != "F16" for v in header.values()):
            raise SetupError(f"{out.name}: the written VAE does not match its source.")
        os.replace(partial, out)
    except BaseException:
        partial.unlink(missing_ok = True)
        raise
    return sum(t.numel() * 2 for t in half.values())


# ---- the setup job --------------------------------------------------------------------------------------------


def _hub_blob_dir(repo_id: str = MODEL_ID) -> Optional[Path]:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        return (
            Path(get_hf_cache_paths().hub_cache)
            / ("models--" + repo_id.replace("/", "--"))
            / "blobs"
        )
    except Exception:  # noqa: BLE001 -- progress only
        return None


def _blob_bytes(
    blobs: Optional[Path],
    sha256: Optional[str],
    *,
    final: bool = False,
) -> int:
    """Bytes of the cached blob, or of its partial download unless ``final``."""
    if blobs is None or not sha256:
        return 0
    for name in (sha256,) if final else (sha256, sha256 + ".incomplete"):
        try:
            return (blobs / name).stat().st_size
        except OSError:
            continue
    return 0


def _outputs(root: Path) -> list[Path]:
    """Every file setup writes in ``root`` before its record, finished or partial."""
    names = (MODEL_BF16, MODEL_Q8, VAE_OUT)
    return [root / n for n in names] + [root / (n + ".partial") for n in names]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _free_bytes(path: Path) -> tuple[Optional[int], Optional[int]]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free, os.stat(probe).st_dev
    except OSError:
        return None, None


def _hub_repo_dir(repo_id: str = MODEL_ID) -> Optional[Path]:
    blobs = _hub_blob_dir(repo_id)
    return blobs.parent if blobs is not None else None


def _was_cached(
    repo_id: str, filename: str, revision: Optional[str], repo_dir: Optional[Path]
) -> bool:
    """Whether a file was already in the Hub cache holding ``repo_dir`` before setup ran (the user's own download is
    not ours to drop)."""
    if repo_dir is None:
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(
            repo_id, filename, cache_dir = repo_dir.parent, revision = revision
        )
        return isinstance(hit, str)
    except Exception:  # noqa: BLE001 -- unknown counts as cached, so nothing is removed
        return True


def _remove_cached(path: str, repo_dir: Optional[Path] = None) -> None:
    """Drop a shard this setup downloaded from Studio's Hub cache: the snapshot entry, and its blob unless any other
    snapshot link of the repo still resolves to it (blobs are content-addressed, so another revision may hold the same
    blob under a different file name). A path outside Studio's cache for this repo is left alone."""
    repo_dir = _hub_repo_dir() if repo_dir is None else repo_dir
    if repo_dir is None:
        return
    p = Path(path)
    # hf_hub_download resolves its cache_dir, so its paths may name the repo under the resolved cache root.
    for root in (repo_dir, repo_dir.parent.resolve() / repo_dir.name):
        snapshots = root / "snapshots"
        if p.is_relative_to(snapshots):
            break
    else:
        logger.info("restyle setup: %s is outside Studio's cache for that repo; kept", path)
        return
    try:
        target = p.resolve() if p.is_symlink() else None
        p.unlink(missing_ok = True)
        if target is None or not target.is_relative_to(repo_dir.resolve() / "blobs"):
            return
        for other in snapshots.rglob("*"):
            if other.is_symlink() and other.resolve() == target:
                return
        target.unlink(missing_ok = True)
    except OSError as exc:
        logger.info("restyle setup: could not remove %s: %s", path, exc)


class RestyleSetup:
    """At most one setup at a time; every outcome lands in ``progress()``."""

    def __init__(
        self,
        *,
        download: Optional[Callable[..., str]] = None,
        repo_files: Optional[
            Callable[[str, Optional[str]], tuple[str, list[dict[str, Any]]]]
        ] = None,
        ensure_engine: Optional[Callable[[threading.Event], str]] = None,
        quantize: Optional[Callable[[Path, Path, threading.Event], None]] = None,
        plan: Optional[Callable[[], dict[str, Any]]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state: dict[str, Any] = {"active": False, "phase": None}
        self._download = download
        self._repo_files = repo_files
        self._ensure_engine = ensure_engine
        self._quantize = quantize
        self._plan = plan
        self._thread: Optional[threading.Thread] = None

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def active(self) -> bool:
        with self._lock:
            return bool(self._state.get("active"))

    def _update(self, **fields: Any) -> None:
        with self._lock:
            self._state = {**self._state, **fields}

    def cancel(self) -> bool:
        with self._lock:
            if not self._state.get("active"):
                return False
            self._cancel.set()
            self._state = {**self._state, "cancelling": True}
        return True

    def start(
        self,
        *,
        hf_token: Any = None,
        need_engine: bool = True,
        need_weights: bool = True,
        background: bool = True,
    ) -> dict[str, Any]:
        plan = (self._plan or hardware_plan)()
        if plan.get("refusal"):
            raise SetupError(plan["refusal"])
        with self._lock:
            if self._state.get("active"):
                raise SetupError("Restyle setup is already running.")
            self._cancel.clear()
            self._state = {
                "active": True,
                "phase": "engine" if need_engine else "download",
                "weights": plan.get("weights") or "bf16",
                "bytes_done": 0,
                "bytes_total": None,
                "error": None,
                "cancelling": False,
                "started_at": time.time(),
            }
        args = (hf_token, need_engine, need_weights, plan.get("weights") or "bf16")
        if not background:
            self._run(*args)
            return self.progress()
        self._thread = threading.Thread(
            target = self._run, args = args, name = "video-restyle-setup", daemon = True
        )
        self._thread.start()
        return self.progress()

    def _check(self) -> None:
        if self._cancel.is_set():
            raise SetupCancelled("cancelled")

    def _run(
        self, hf_token: Optional[str], need_engine: bool, need_weights: bool, weights: str
    ) -> None:
        try:
            if need_engine:
                self._update(phase = "engine")
                (self._ensure_engine or _ensure_engine)(self._cancel)
            self._check()
            if need_weights and installed_files() is None:
                self._fetch_weights(hf_token, weights)
            self._update(active = False, phase = "done")
        except SetupCancelled:
            self._update(active = False, phase = "cancelled", cancelling = False)
        except Exception as exc:  # noqa: BLE001 -- surfaced through progress
            if self._cancel.is_set():
                self._update(active = False, phase = "cancelled", cancelling = False)
                return
            logger.warning("restyle setup failed: %s", exc)
            message = str(exc) if isinstance(exc, SetupError) else f"Restyle setup failed: {exc}"
            try:
                # Every account can read the status; a mirror URL in an error may carry credentials.
                from utils.log_redaction import redact_log_text
                message = redact_log_text(message)
            except Exception:  # noqa: BLE001
                pass
            self._update(active = False, phase = "error", error = message)

    def _fetch_weights(self, hf_token: Optional[str], weights: str) -> None:
        root = setup_dir()
        root.mkdir(parents = True, exist_ok = True)
        self._update(phase = "download")
        repo_files = self._repo_files or _repo_files
        revision, files = repo_files(MODEL_ID, hf_token)
        vae_revision, vae_files = repo_files(VAE_REPO, hf_token)
        by_name = {f["name"]: f for f in files}
        vae_entry = next((f for f in vae_files if f["name"] == VAE_FILE), None)
        download = self._download or _download
        index_path = download(
            MODEL_ID, TRANSFORMER_INDEX, hf_token, revision = revision, cancel_event = self._cancel
        )
        shard_names = ["transformer/" + name for name in index_shards(Path(index_path))]
        if any(name not in by_name for name in shard_names) or vae_entry is None:
            raise SetupError(
                f"{MODEL_ID} or {VAE_REPO} no longer has the transformer shards and VAE Studio expects."
            )
        # (repo, revision, file) in download order; the VAE source comes last.
        wanted = [(MODEL_ID, revision, by_name[name]) for name in shard_names]
        wanted.append((VAE_REPO, vae_revision, vae_entry))
        sizes = [int(f.get("size") or 0) for _, _, f in wanted]
        total = sum(sizes)
        transformer_bytes = sum(sizes[:-1])
        blobs = _hub_blob_dir()
        cached = sum(
            size
            for (repo, _, f), size in zip(wanted, sizes)
            if size > 0 and _blob_bytes(_hub_blob_dir(repo), f.get("sha256"), final = True) >= size
        )
        # Outputs go to .partial and replace their file only once verified, so an existing copy
        # keeps its space meanwhile; unrecorded leftovers are removed first; recorded outputs are
        # kept, not counted free.
        if not (root / RECORD_NAME).exists():
            for leftover in _outputs(root):
                leftover.unlink(missing_ok = True)
        out_bytes = transformer_bytes + sizes[-1] // 2
        if weights == "q8_0":
            out_bytes += int(transformer_bytes * _Q8_RATIO)
        self._check_disk(blobs, total - cached, root, out_bytes)
        # Read the cache once: a cache-home change mid-setup must not make the new cache's files count as ours.
        repo_dirs = {repo: _hub_repo_dir(repo) for repo, _, _ in wanted}
        ours = {
            (repo, f["name"])
            for repo, rev, f in wanted
            if not _was_cached(repo, f["name"], rev, repo_dirs[repo])
        }
        self._update(bytes_total = total, bytes_done = cached)
        local: dict[str, Path] = {}
        done = 0
        for (repo, rev, entry), size in zip(wanted, sizes):
            self._check()
            stop = threading.Event()

            def tick(
                base: int = done,
                sha: Optional[str] = entry.get("sha256"),
                where: Optional[Path] = _hub_blob_dir(repo),
            ) -> None:
                while not stop.wait(1.0):
                    self._update(bytes_done = min(total, base + _blob_bytes(where, sha)))

            ticker = threading.Thread(target = tick, daemon = True)
            ticker.start()
            try:
                local[entry["name"]] = Path(
                    download(
                        repo,
                        entry["name"],
                        hf_token,
                        revision = rev,
                        cancel_event = self._cancel,
                    )
                )
            finally:
                stop.set()
                ticker.join(timeout = 5)
            done += size
            self._update(bytes_done = done)
        self._check()
        try:
            self._write_outputs(root, local, shard_names, transformer_bytes, weights, revision)
        except BaseException:
            # No record means no usable setup: drop the ~30 GB of outputs instead of leaving them unreachable.
            if not (root / RECORD_NAME).exists():
                for leftover in _outputs(root):
                    leftover.unlink(missing_ok = True)
            raise
        # The outputs are verified, so the sources this run downloaded (about 33 GB) are no longer needed.
        for repo, _, entry in wanted:
            if (repo, entry["name"]) in ours:
                _remove_cached(str(local[entry["name"]]), repo_dirs[repo])

    def _write_outputs(
        self,
        root: Path,
        local: dict[str, Path],
        shard_names: list[str],
        transformer_bytes: int,
        weights: str,
        revision: str,
    ) -> None:
        self._update(phase = "merge", bytes_done = 0, bytes_total = transformer_bytes)
        bf16 = root / MODEL_BF16
        payload = merge_shards(
            [local[name] for name in shard_names],
            bf16,
            cancel = self._cancel,
            on_bytes = lambda n: self._update(bytes_done = n),
        )
        self._update(phase = "vae", bytes_done = None, bytes_total = None)
        convert_vae(local[VAE_FILE], root / VAE_OUT, cancel = self._cancel)
        model_file = bf16
        if weights == "q8_0":
            self._check()
            self._update(phase = "quantize")
            q8 = root / MODEL_Q8
            (self._quantize or _quantize_q8)(bf16, q8, self._cancel)
            bf16.unlink(missing_ok = True)
            model_file = q8
        record = {
            "version": RECORD_VERSION,
            "repo": MODEL_ID,
            "revision": revision,
            "weights": weights,
            "model": {"file": model_file.name, "bytes": model_file.stat().st_size},
            "vae": {"file": VAE_OUT, "bytes": (root / VAE_OUT).stat().st_size},
            "transformer_payload_bytes": payload,
        }
        tmp = root / (RECORD_NAME + ".partial")
        tmp.write_text(json.dumps(record, indent = 2), encoding = "utf-8")
        os.replace(tmp, root / RECORD_NAME)

    def _check_disk(
        self, blobs: Optional[Path], download_left: int, root: Path, out_bytes: int
    ) -> None:
        need: dict[Any, list[Any]] = {}
        for path, nbytes in ((blobs or root, max(0, download_left)), (root, out_bytes)):
            free, dev = _free_bytes(path)
            if free is None:
                continue
            slot = need.setdefault(dev, [free, 0, path])
            slot[1] += nbytes
        for free, nbytes, path in need.values():
            if nbytes + _DISK_MARGIN > free:
                raise SetupError(
                    f"Restyle setup needs about {(nbytes + _DISK_MARGIN) / 1e9:.0f} GB free on the disk holding "
                    f"{path}, but only {free / 1e9:.0f} GB is free."
                )


def _repo_files(repo_id: str, hf_token: Optional[str]) -> tuple[str, list[dict[str, Any]]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata = True, token = hf_token)
    files = []
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha is None and isinstance(lfs, dict):
            sha = lfs.get("sha256")
        files.append({"name": s.rfilename, "size": s.size, "sha256": sha})
    return str(info.sha), files


def _download(
    repo_id: str,
    filename: str,
    hf_token: Optional[str],
    *,
    revision: Optional[str],
    cancel_event: threading.Event,
) -> str:
    from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

    path = hf_hub_download_with_xet_fallback(
        repo_id,
        filename,
        hf_token,
        cancel_event = cancel_event,
        revision = revision,
        reuse_other_cache_root = True,
    )
    if cancel_event.is_set():
        raise SetupCancelled("cancelled")
    return path


def install_accelerator() -> str:
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.diffusion_engine_router import image_install_accelerator
    from core.inference.sd_cpp_backend import preferred_accelerator

    return preferred_accelerator(
        image_install_accelerator(getattr(resolve_diffusion_device_target(), "backend", "cpu"))
    )


def _ensure_engine(cancel: threading.Event) -> str:
    """The pinned prebuilt through Studio's own installer (release checksum verified before extraction)."""
    from core.inference.sd_cpp_backend import _install_allowed, ensure_sd_server_binary
    from core.inference.video_restyle import binary_supports_cosmos3

    if not _install_allowed():
        raise SetupError(
            "Installing stable-diffusion.cpp is turned off (UNSLOTH_DIFFUSION_SD_CPP_INSTALL). Unset it, or "
            "point SD_SERVER_PATH at a build with Cosmos3 support."
        )
    binary = ensure_sd_server_binary(allow_install = True, accelerator = install_accelerator())
    if not binary:
        raise SetupError(
            "Could not install the stable-diffusion.cpp server. If an image is generating, try again "
            "once it finishes; otherwise check the network and the Studio log."
        )
    if not binary_supports_cosmos3(binary):
        raise SetupError(
            f"The stable-diffusion.cpp server at {binary} has no Cosmos3 support. Remove that build "
            "(or unset SD_SERVER_PATH) so Studio can install its own."
        )
    return binary


def _quantize_q8(src: Path, out: Path, cancel: threading.Event) -> None:
    from core.inference.sd_cpp_backend import _tree_reader
    from core.inference.sd_cpp_engine import SdCppCancelled, SdCppEngine, find_sd_cpp_binary

    cli = find_sd_cpp_binary()
    if not cli:
        raise SetupError(
            "sd-cli is missing from the stable-diffusion.cpp install, so the 8-bit copy cannot be made."
        )
    work = out.parent / (out.name + ".partial")
    shutil.rmtree(work, ignore_errors = True)
    work.mkdir(parents = True)
    target = work / out.name
    try:
        with _tree_reader(cli, cancel, "cancelled"):
            SdCppEngine(cli)._run(
                [cli, "-M", "convert", "-m", str(src), "-o", str(target), "--type", "q8_0"],
                str(target),
                timeout = None,
                env = None,
                on_log = None,
                cancel_event = cancel,
            )
        os.replace(target, out)
    except SdCppCancelled as exc:
        raise SetupCancelled("cancelled") from exc
    finally:
        shutil.rmtree(work, ignore_errors = True)


setup_runner = RestyleSetup()

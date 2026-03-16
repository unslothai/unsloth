# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions
through its OpenAI-compatible /v1/chat/completions endpoint.
"""

import atexit
import json
import struct
import structlog
from loggers import get_logger
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import httpx

logger = get_logger(__name__)


class LlamaCppBackend:
    """
    Manages a llama-server subprocess for GGUF model inference.

    Lifecycle:
        1. load_model()  — starts llama-server with the GGUF file
        2. generate_chat_completion() — proxies to /v1/chat/completions, streams back
        3. unload_model() — terminates llama-server subprocess
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_identifier: Optional[str] = None
        self._gguf_path: Optional[str] = None
        self._hf_repo: Optional[str] = None
        self._hf_variant: Optional[str] = None
        self._is_vision: bool = False
        self._healthy = False
        self._context_length: Optional[int] = None
        self._chat_template: Optional[str] = None
        self._lock = threading.Lock()
        self._stdout_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        self._kill_orphaned_servers()
        atexit.register(self._cleanup)

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._healthy

    @property
    def is_active(self) -> bool:
        """True if a llama-server process exists (loading or loaded)."""
        return self._process is not None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        return self._is_vision

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    @property
    def context_length(self) -> Optional[int]:
        return self._context_length

    @property
    def chat_template(self) -> Optional[str]:
        return self._chat_template

    # ── Binary discovery ──────────────────────────────────────────

    @staticmethod
    def _find_llama_server_binary() -> Optional[str]:
        """
        Locate the llama-server binary.

        Search order:
        1.  LLAMA_SERVER_PATH environment variable (direct path to binary)
        1b. UNSLOTH_LLAMA_CPP_PATH env var (custom llama.cpp install dir)
        2.  ~/.unsloth/llama.cpp/llama-server        (make build, root dir)
        3.  ~/.unsloth/llama.cpp/build/bin/llama-server  (cmake build, Linux)
        4.  ~/.unsloth/llama.cpp/build/bin/Release/llama-server.exe  (cmake build, Windows)
        5.  ./llama.cpp/llama-server                 (legacy: make build, root dir)
        6.  ./llama.cpp/build/bin/llama-server        (legacy: cmake in-tree build)
        7.  llama-server on PATH                     (system install)
        8.  ./bin/llama-server                       (legacy: extracted binary)
        """
        import os
        import sys

        binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        # 1. Env var — direct path to binary
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path and Path(env_path).is_file():
            return env_path

        # 1b. UNSLOTH_LLAMA_CPP_PATH — custom llama.cpp install directory
        custom_llama_cpp = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
        if custom_llama_cpp:
            custom_dir = Path(custom_llama_cpp)
            # Root dir (make builds)
            root_bin = custom_dir / binary_name
            if root_bin.is_file():
                return str(root_bin)
            # build/bin/ (cmake builds on Linux)
            cmake_bin = custom_dir / "build" / "bin" / binary_name
            if cmake_bin.is_file():
                return str(cmake_bin)
            # build/bin/Release/ (cmake builds on Windows)
            if sys.platform == "win32":
                win_bin = custom_dir / "build" / "bin" / "Release" / binary_name
                if win_bin.is_file():
                    return str(win_bin)

        # 2–4. ~/.unsloth/llama.cpp (primary — setup.sh / setup.ps1 build here)
        unsloth_home = Path.home() / ".unsloth" / "llama.cpp"
        # Root dir (make builds copy binaries here)
        home_root = unsloth_home / binary_name
        if home_root.is_file():
            return str(home_root)
        # build/bin/ (cmake builds on Linux)
        home_linux = unsloth_home / "build" / "bin" / binary_name
        if home_linux.is_file():
            return str(home_linux)

        # 3. Windows MSVC build has Release subdir
        if sys.platform == "win32":
            home_win = unsloth_home / "build" / "bin" / "Release" / binary_name
            if home_win.is_file():
                return str(home_win)

        # 5–6. Legacy: in-tree build (older setup.sh / setup.ps1 versions)
        project_root = Path(__file__).resolve().parents[4]
        # Root dir (make builds)
        root_path = project_root / "llama.cpp" / binary_name
        if root_path.is_file():
            return str(root_path)
        # build/bin/ (cmake builds)
        build_path = project_root / "llama.cpp" / "build" / "bin" / binary_name
        if build_path.is_file():
            return str(build_path)
        if sys.platform == "win32":
            win_path = (
                project_root / "llama.cpp" / "build" / "bin" / "Release" / binary_name
            )
            if win_path.is_file():
                return str(win_path)

        # 7. System PATH
        system_path = shutil.which("llama-server")
        if system_path:
            return system_path

        # 8. Legacy: extracted to bin/
        bin_path = project_root / "bin" / binary_name
        if bin_path.is_file():
            return str(bin_path)

        return None

    # ── GPU allocation ────────────────────────────────────────────

    @staticmethod
    def _get_gguf_size_bytes(model_path: str) -> int:
        """Get total GGUF size in bytes, including split shards."""
        import re

        main = Path(model_path)
        total = main.stat().st_size

        # Check for split shards (e.g., model-00001-of-00003.gguf)
        shard_pat = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
        m = shard_pat.match(main.name)
        if m:
            prefix, _, num_total = m.group(1), m.group(2), m.group(3)
            sibling_pat = re.compile(
                r"^"
                + re.escape(prefix)
                + r"-\d{5}-of-"
                + re.escape(num_total)
                + r"\.gguf$"
            )
            for sibling in main.parent.iterdir():
                if sibling != main and sibling_pat.match(sibling.name):
                    total += sibling.stat().st_size

        return total

    @staticmethod
    def _get_gpu_free_memory() -> list[tuple[int, int]]:
        """Query free memory per GPU via nvidia-smi.

        Returns list of (gpu_index, free_mib) sorted by index.
        Respects CUDA_VISIBLE_DEVICES if set.
        Returns empty list if nvidia-smi is not available.
        """
        import os

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                timeout = 10,
            )
            if result.returncode != 0:
                return []

            # Parse which GPUs are allowed by existing CUDA_VISIBLE_DEVICES
            allowed = None
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cvd is not None and cvd.strip():
                try:
                    allowed = set(int(x.strip()) for x in cvd.split(","))
                except ValueError:
                    pass  # Non-numeric (e.g., "GPU-uuid"), ignore filter

            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) == 2:
                    idx = int(parts[0].strip())
                    free_mib = int(parts[1].strip())
                    if allowed is not None and idx not in allowed:
                        continue
                    gpus.append((idx, free_mib))
            return gpus
        except Exception:
            return []

    @staticmethod
    def _select_gpus(
        model_size_bytes: int,
        gpus: list[tuple[int, int]],
    ) -> tuple[Optional[list[int]], bool]:
        """Pick GPU(s) for a model based on file size and free memory.

        Uses GGUF file size as a rough proxy for VRAM usage (actual usage
        is higher due to KV cache and compute buffers, but 70% threshold
        accounts for that).

        Returns (gpu_indices, use_fit):
          - ([1], False)       model fits on 1 GPU at 70% of free
          - ([1, 2], False)    model needs 2 GPUs
          - (None, True)       model too large, let --fit handle it
        """
        if not gpus:
            return None, True

        model_size_mib = model_size_bytes / (1024 * 1024)

        # Sort GPUs by free memory descending
        ranked = sorted(gpus, key = lambda g: g[1], reverse = True)

        # Try fitting on 1 GPU (70% of free memory threshold)
        if ranked[0][1] * 0.70 >= model_size_mib:
            return [ranked[0][0]], False

        # Try fitting on N GPUs (accumulate free memory from most-free)
        cumulative = 0
        selected = []
        for idx, free_mib in ranked:
            selected.append(idx)
            cumulative += free_mib * 0.70
            if cumulative >= model_size_mib:
                return sorted(selected), False

        # Model is too large even for all GPUs, let --fit handle it
        return None, True

    # ── Variant fallback ────────────────────────────────────────────

    @staticmethod
    def _find_smallest_fitting_variant(
        hf_repo: str,
        free_bytes: int,
        hf_token: Optional[str] = None,
    ) -> Optional[tuple[str, int]]:
        """Find the smallest GGUF variant (including all shards) that fits.

        Groups split shards by variant prefix and sums their sizes.
        For example, UD-Q4_K_XL with 9 shards of 50 GB each = 450 GB total.

        Returns (first_shard_filename, total_size_bytes) or None if nothing fits.
        """
        import re

        try:
            from huggingface_hub import get_paths_info, list_repo_files

            files = list_repo_files(hf_repo, token = hf_token)
            gguf_files = [
                f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()
            ]
            if not gguf_files:
                return None

            # Get sizes for all GGUF files
            path_infos = list(get_paths_info(hf_repo, gguf_files, token = hf_token))
            size_map = {p.path: (p.size or 0) for p in path_infos}

            # Group files by variant: shards share a prefix before -NNNNN-of-NNNNN
            shard_pat = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$")
            variants: dict[str, list[str]] = {}
            for f in gguf_files:
                m = shard_pat.match(f)
                key = m.group(1) if m else f
                variants.setdefault(key, []).append(f)

            # Sum shard sizes per variant, track the first shard (for download)
            variant_sizes: list[tuple[str, int, list[str]]] = []
            for key, shard_files in variants.items():
                total = sum(size_map.get(f, 0) for f in shard_files)
                first = sorted(shard_files)[0]
                variant_sizes.append((first, total, shard_files))

            # Sort by total size ascending and pick the smallest that fits
            variant_sizes.sort(key = lambda x: x[1])
            for first_file, total_size, _ in variant_sizes:
                if total_size > 0 and total_size <= free_bytes:
                    return first_file, total_size

            return None
        except Exception:
            return None

    # ── Port allocation ───────────────────────────────────────────

    @staticmethod
    def _find_free_port() -> int:
        """Find an available TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    # ── Stdout drain (prevents pipe deadlock on Windows) ─────────

    def _drain_stdout(self):
        """
        Read lines from the subprocess stdout in a background thread.

        This prevents a pipe-buffer deadlock on Windows where the default
        pipe buffer is only ~4 KB.  Without draining, llama-server blocks
        on writes and never becomes healthy.
        """
        try:
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    self._stdout_lines.append(line)
                    logger.info(f"[llama-server] {line}")
        except (ValueError, OSError):
            # Pipe closed — process is terminating
            pass

    # GGUF KV type sizes for fast skipping
    _GGUF_TYPE_SIZE = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    @staticmethod
    def _gguf_skip_value(f, vtype: int) -> None:
        """Skip a GGUF KV value without reading it."""
        sz = LlamaCppBackend._GGUF_TYPE_SIZE.get(vtype)
        if sz is not None:
            f.seek(sz, 1)
        elif vtype == 8:  # STRING
            slen = struct.unpack("<Q", f.read(8))[0]
            f.seek(slen, 1)
        elif vtype == 9:  # ARRAY
            atype = struct.unpack("<I", f.read(4))[0]
            alen = struct.unpack("<Q", f.read(8))[0]
            elem_sz = LlamaCppBackend._GGUF_TYPE_SIZE.get(atype)
            if elem_sz is not None:
                f.seek(elem_sz * alen, 1)
            elif atype == 8:
                for _ in range(alen):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    f.seek(slen, 1)
            else:
                for _ in range(alen):
                    LlamaCppBackend._gguf_skip_value(f, atype)

    def _read_gguf_metadata(self, gguf_path: str) -> None:
        """Read context_length and chat_template from a GGUF file's KV header.

        Parses only the KV pairs we need (~30ms even for multi-GB files).
        For split GGUFs, metadata is always in shard 1.
        """
        try:
            WANTED = {"general.architecture", "tokenizer.chat_template"}
            arch = None
            ctx_key = None

            with open(gguf_path, "rb") as f:
                magic = struct.unpack("<I", f.read(4))[0]
                if magic != 0x46554747:  # b"GGUF" as little-endian u32
                    return
                _version = struct.unpack("<I", f.read(4))[0]
                _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

                for _ in range(kv_count):
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    key = f.read(key_len).decode("utf-8")
                    vtype = struct.unpack("<I", f.read(4))[0]

                    if key in WANTED or (ctx_key and key == ctx_key):
                        # Read this value
                        if vtype == 8:  # STRING
                            slen = struct.unpack("<Q", f.read(8))[0]
                            val_s = f.read(slen).decode("utf-8")
                            if key == "general.architecture":
                                arch = val_s
                                ctx_key = f"{arch}.context_length"
                            elif key == "tokenizer.chat_template":
                                self._chat_template = val_s
                        elif vtype == 4:  # UINT32
                            val_i = struct.unpack("<I", f.read(4))[0]
                            if ctx_key and key == ctx_key:
                                self._context_length = val_i
                        elif vtype == 10:  # UINT64
                            val_i = struct.unpack("<Q", f.read(8))[0]
                            if ctx_key and key == ctx_key:
                                self._context_length = val_i
                        else:
                            self._gguf_skip_value(f, vtype)
                    else:
                        self._gguf_skip_value(f, vtype)

            if self._context_length:
                logger.info(f"GGUF metadata: context_length={self._context_length}")
            if self._chat_template:
                logger.info(
                    f"GGUF metadata: chat_template={len(self._chat_template)} chars"
                )
        except Exception as e:
            logger.warning(f"Failed to read GGUF metadata: {e}")

    # ── HF download (no lock held) ───────────────────────────────

    def _download_gguf(
        self,
        *,
        hf_repo: str,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> str:
        """Download GGUF file(s) from HuggingFace. Returns local path.

        Runs WITHOUT self._lock so that unload_model() can set
        _cancel_event at any time. Checks _cancel_event between
        each shard download.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for HF model loading. "
                "Install it with: pip install huggingface_hub"
            )

        # Determine the filename from the variant
        gguf_filename = None
        gguf_extra_shards: list[str] = []
        if hf_variant:
            try:
                import re
                from huggingface_hub import list_repo_files

                files = list_repo_files(hf_repo, token = hf_token)
                variant_lower = hf_variant.lower()
                boundary = re.compile(
                    r"(?<![a-zA-Z0-9])" + re.escape(variant_lower) + r"(?![a-zA-Z0-9])"
                )
                gguf_files = sorted(
                    f
                    for f in files
                    if f.endswith(".gguf") and boundary.search(f.lower())
                )
                if gguf_files:
                    gguf_filename = gguf_files[0]
                    shard_pat = re.compile(r"^(.*)-\d{5}-of-(\d{5})\.gguf$")
                    m = shard_pat.match(gguf_filename)
                    if m:
                        prefix = m.group(1)
                        total = m.group(2)
                        sibling_pat = re.compile(
                            r"^"
                            + re.escape(prefix)
                            + r"-\d{5}-of-"
                            + re.escape(total)
                            + r"\.gguf$"
                        )
                        gguf_extra_shards = [
                            f for f in gguf_files[1:] if sibling_pat.match(f)
                        ]
            except Exception as e:
                logger.warning(f"Could not list repo files: {e}")

            if not gguf_filename:
                repo_name = hf_repo.split("/")[-1].replace("-GGUF", "")
                gguf_filename = f"{repo_name}-{hf_variant}.gguf"

        # Check disk space and fall back to a smaller variant if needed
        all_gguf_files = [gguf_filename] + gguf_extra_shards
        try:
            import os

            from huggingface_hub import get_paths_info

            path_infos = list(get_paths_info(hf_repo, all_gguf_files, token = hf_token))
            total_download_bytes = sum((p.size or 0) for p in path_infos)

            if total_download_bytes > 0:
                cache_dir = os.environ.get(
                    "HF_HUB_CACHE",
                    str(Path.home() / ".cache" / "huggingface" / "hub"),
                )
                Path(cache_dir).mkdir(parents = True, exist_ok = True)
                free_bytes = shutil.disk_usage(cache_dir).free

                total_gb = total_download_bytes / (1024**3)
                free_gb = free_bytes / (1024**3)

                logger.info(
                    f"GGUF download: {total_gb:.1f} GB needed, "
                    f"{free_gb:.1f} GB free on disk"
                )

                if total_download_bytes > free_bytes:
                    smaller = self._find_smallest_fitting_variant(
                        hf_repo,
                        free_bytes,
                        hf_token,
                    )
                    if smaller:
                        fallback_file, fallback_size = smaller
                        logger.info(
                            f"Selected variant too large ({total_gb:.1f} GB), "
                            f"falling back to {fallback_file} ({fallback_size / (1024**3):.1f} GB)"
                        )
                        gguf_filename = fallback_file
                        import re as _re

                        _shard_pat = _re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$")
                        _m = _shard_pat.match(gguf_filename)
                        _prefix = _m.group(1) if _m else None
                        if _prefix:
                            gguf_extra_shards = sorted(
                                f
                                for f in all_gguf_files
                                if f.startswith(_prefix)
                                and f != gguf_filename
                                and "mmproj" not in f.lower()
                            )
                        else:
                            gguf_extra_shards = []
                    else:
                        raise RuntimeError(
                            f"Not enough disk space to download any variant. "
                            f"Only {free_gb:.1f} GB free in {cache_dir}"
                        )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        logger.info(
            f"Downloading GGUF: {hf_repo}/{gguf_filename}"
            + (f" (+{len(gguf_extra_shards)} shards)" if gguf_extra_shards else "")
        )
        try:
            if self._cancel_event.is_set():
                raise RuntimeError("Cancelled")
            local_path = hf_hub_download(
                repo_id = hf_repo,
                filename = gguf_filename,
                token = hf_token,
            )
            for shard in gguf_extra_shards:
                if self._cancel_event.is_set():
                    raise RuntimeError("Cancelled")
                logger.info(f"Downloading GGUF shard: {shard}")
                hf_hub_download(
                    repo_id = hf_repo,
                    filename = shard,
                    token = hf_token,
                )
        except RuntimeError as e:
            if "Cancelled" in str(e):
                raise
            raise RuntimeError(
                f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
            )

        logger.info(f"GGUF downloaded to: {local_path}")
        return local_path

    def _download_mmproj(
        self,
        *,
        hf_repo: str,
        hf_token: Optional[str] = None,
    ) -> Optional[str]:
        """Download the mmproj (vision projection) file from a GGUF repo.

        Prefers mmproj-F16.gguf, falls back to any mmproj*.gguf file.
        Returns the local path, or None if no mmproj file exists.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            files = list_repo_files(hf_repo, token = hf_token)
            mmproj_files = sorted(
                f for f in files if f.endswith(".gguf") and "mmproj" in f.lower()
            )
            if not mmproj_files:
                return None

            # Prefer F16 variant
            target = None
            for f in mmproj_files:
                if "f16" in f.lower():
                    target = f
                    break
            if target is None:
                target = mmproj_files[0]

            logger.info(f"Downloading mmproj: {hf_repo}/{target}")
            local_path = hf_hub_download(
                repo_id = hf_repo,
                filename = target,
                token = hf_token,
            )
            return local_path
        except Exception as e:
            logger.warning(f"Could not download mmproj: {e}")
            return None

    # ── Lifecycle ─────────────────────────────────────────────────

    def load_model(
        self,
        *,
        # Local mode: pass a path to a .gguf file
        gguf_path: Optional[str] = None,
        # Vision projection (mmproj) for local vision models
        mmproj_path: Optional[str] = None,
        # HF mode: let llama-server download via -hf "repo:quant"
        hf_repo: Optional[str] = None,
        hf_variant: Optional[str] = None,
        hf_token: Optional[str] = None,
        # Common
        model_identifier: str,
        is_vision: bool = False,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,  # Accepted for caller compat, unused
    ) -> bool:
        """
        Start llama-server with a GGUF model.

        Two modes:
        - Local: ``gguf_path="/path/to/model.gguf"`` → uses ``-m``
        - HF:    ``hf_repo="unsloth/gemma-3-4b-it-GGUF", hf_variant="Q4_K_M"`` → uses ``-hf``

        In HF mode, llama-server handles downloading, caching, and
        auto-loading mmproj files for vision models.

        Returns True if server started and health check passed.
        """
        self._cancel_event.clear()

        # ── Phase 1: kill old process (under lock, fast) ──────────
        with self._lock:
            self._kill_process()

        binary = self._find_llama_server_binary()
        if not binary:
            raise RuntimeError(
                "llama-server binary not found. "
                "Run setup.sh to build it, install llama.cpp, "
                "or set LLAMA_SERVER_PATH environment variable."
            )

        # ── Phase 2: download (NO lock held, so cancel can proceed) ──
        if hf_repo:
            model_path = self._download_gguf(
                hf_repo = hf_repo,
                hf_variant = hf_variant,
                hf_token = hf_token,
            )
            # Auto-download mmproj for vision models
            if is_vision and not mmproj_path:
                mmproj_path = self._download_mmproj(
                    hf_repo = hf_repo,
                    hf_token = hf_token,
                )
        elif gguf_path:
            if not Path(gguf_path).is_file():
                raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
            model_path = gguf_path
        else:
            raise ValueError("Either gguf_path or hf_repo must be provided")

        # Read GGUF metadata (context_length, chat_template) -- fast, header only
        self._read_gguf_metadata(model_path)

        # Check cancel after download
        if self._cancel_event.is_set():
            logger.info("Load cancelled after download phase")
            return False

        # ── Phase 3: start llama-server (under lock) ──────────────
        with self._lock:
            # Re-check cancel inside lock
            if self._cancel_event.is_set():
                logger.info("Load cancelled before server start")
                return False

            self._port = self._find_free_port()

            # Select GPU(s) based on model size and free memory
            try:
                model_size = self._get_gguf_size_bytes(model_path)
                gpus = self._get_gpu_free_memory()
                gpu_indices, use_fit = self._select_gpus(model_size, gpus)
                logger.info(
                    f"GGUF size: {model_size / (1024**3):.1f} GB, "
                    f"GPUs free: {gpus}, selected: {gpu_indices}, fit: {use_fit}"
                )
            except Exception as e:
                logger.warning(f"GPU selection failed ({e}), using --fit on")
                gpu_indices, use_fit = None, True

            cmd = [
                binary,
                "-m",
                model_path,
                "--port",
                str(self._port),
                "-c",
                "0",  # 0 = use model's native context size
                "--parallel",
                "1",  # Single-user studio, saves VRAM
                "--flash-attn",
                "on",  # Force flash attention for speed
            ]

            if use_fit:
                cmd.extend(["--fit", "on"])

            if n_threads is not None:
                cmd.extend(["--threads", str(n_threads)])

            if mmproj_path:
                if not Path(mmproj_path).is_file():
                    logger.warning(f"mmproj file not found: {mmproj_path}")
                else:
                    cmd.extend(["--mmproj", mmproj_path])
                    logger.info(f"Using mmproj for vision: {mmproj_path}")

            logger.info(f"Starting llama-server: {' '.join(cmd)}")

            # Set library paths so llama-server can find its shared libs and CUDA DLLs
            import os
            import sys

            env = os.environ.copy()
            binary_dir = str(Path(binary).parent)

            if sys.platform == "win32":
                # On Windows, CUDA DLLs (cublas64_12.dll, cudart64_12.dll, etc.)
                # must be on PATH. Add CUDA_PATH\bin if available.
                path_dirs = [binary_dir]
                cuda_path = os.environ.get("CUDA_PATH", "")
                if cuda_path:
                    cuda_bin = os.path.join(cuda_path, "bin")
                    if os.path.isdir(cuda_bin):
                        path_dirs.append(cuda_bin)
                    # Some CUDA installs put DLLs in bin\x64
                    cuda_bin_x64 = os.path.join(cuda_path, "bin", "x64")
                    if os.path.isdir(cuda_bin_x64):
                        path_dirs.append(cuda_bin_x64)
                existing_path = env.get("PATH", "")
                env["PATH"] = ";".join(path_dirs) + ";" + existing_path
            else:
                # Linux: set LD_LIBRARY_PATH for shared libs next to the binary
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{binary_dir}:{existing_ld}" if existing_ld else binary_dir
                )

            # Pin to selected GPU(s) via CUDA_VISIBLE_DEVICES
            if gpu_indices is not None:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

            self._stdout_lines = []
            self._process = subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                env = env,
            )

            # Start background thread to drain stdout and prevent pipe deadlock
            self._stdout_thread = threading.Thread(
                target = self._drain_stdout, daemon = True, name = "llama-stdout"
            )
            self._stdout_thread.start()

            self._gguf_path = gguf_path
            self._hf_repo = hf_repo
            self._hf_variant = hf_variant
            self._is_vision = is_vision
            self._model_identifier = model_identifier

            # Wait for llama-server to become healthy
            if not self._wait_for_health(timeout = 120.0):
                self._kill_process()
                raise RuntimeError(
                    "llama-server failed to start. "
                    "Check that the GGUF file is valid and you have enough memory."
                )

            self._healthy = True

            logger.info(
                f"llama-server ready on port {self._port} "
                f"for model '{model_identifier}'"
            )
            return True

    def unload_model(self) -> bool:
        """Terminate the llama-server subprocess and cancel any in-flight download."""
        self._cancel_event.set()
        with self._lock:
            self._kill_process()
            logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._hf_repo = None
            self._hf_variant = None
            self._is_vision = False
            self._is_audio = False
            self._audio_type = None
            self._port = None
            self._healthy = False
            self._context_length = None
            self._chat_template = None
            # Free audio codec GPU memory
            if LlamaCppBackend._codec_mgr is not None:
                LlamaCppBackend._codec_mgr.unload()
                LlamaCppBackend._codec_mgr = None
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return True

    def _kill_process(self):
        """Terminate the subprocess if running."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit on SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout = 5)
        except Exception as e:
            logger.warning(f"Error killing llama-server process: {e}")
        finally:
            self._process = None
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout = 2)
                self._stdout_thread = None

    @staticmethod
    def _kill_orphaned_servers():
        """Kill orphaned llama-server processes started by studio.

        Only kills processes whose binary lives under ~/.unsloth/llama.cpp/
        to avoid terminating unrelated llama-server instances on the machine.
        """
        import os
        import signal

        try:
            # Use pgrep with full command match to identify studio-managed servers
            result = subprocess.run(
                ["pgrep", "-a", "-f", "llama-server"],
                capture_output = True,
                text = True,
                timeout = 5,
            )
            if result.returncode != 0:
                return
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                pid = int(parts[0])
                cmdline = parts[1]
                if pid == os.getpid():
                    continue
                # Only kill if it's a studio-managed server (lives under .unsloth/)
                if ".unsloth/" not in cmdline and "unsloth" not in cmdline.lower():
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Killed orphaned llama-server process (pid={pid})")
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
        except Exception:
            pass

    def _cleanup(self):
        """atexit handler to ensure llama-server is terminated."""
        self._kill_process()

    def _wait_for_health(self, timeout: float = 120.0, interval: float = 0.5) -> bool:
        """
        Poll llama-server's /health endpoint until it responds 200.

        Also monitors subprocess for early exit/crash.
        """
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self._port}/health"

        while time.monotonic() < deadline:
            # Check if process crashed
            if self._process.poll() is not None:
                # Give the drain thread a moment to collect final output
                if self._stdout_thread is not None:
                    self._stdout_thread.join(timeout = 2)
                output = "\n".join(self._stdout_lines[-50:])
                logger.error(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"Output: {output[:2000]}"
                )
                return False

            try:
                resp = httpx.get(url, timeout = 2.0)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(interval)

        logger.error(f"llama-server health check timed out after {timeout}s")
        return False

    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _build_openai_messages(
        messages: list[dict],
        image_b64: Optional[str] = None,
    ) -> list[dict]:
        """
        Build OpenAI-format messages, optionally injecting an image_url
        content part into the last user message for vision models.

        If no image is provided, returns messages as-is.
        """
        if not image_b64:
            return messages

        # Find the last user message and convert to multimodal content parts
        result = [msg.copy() for msg in messages]
        last_user_idx = None
        for i, msg in enumerate(result):
            if msg["role"] == "user":
                last_user_idx = i

        if last_user_idx is not None:
            text_content = result[last_user_idx].get("content", "")
            result[last_user_idx]["content"] = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]

        return result

    # ── Generation (proxy to llama-server) ────────────────────────

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """
        Send a chat completion request to llama-server and stream tokens back.

        Uses /v1/chat/completions — llama-server handles chat template
        application and vision (multimodal image_url parts) natively.

        Yields cumulative text (matching InferenceBackend's convention).
        """
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        openai_messages = self._build_openai_messages(messages, image_b64)

        payload = {
            "messages": openai_messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False

        try:
            with httpx.Client(timeout = None) as client:
                with client.stream("POST", url, json = payload) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    for raw_chunk in response.iter_text():
                        if cancel_event is not None and cancel_event.is_set():
                            break

                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    cumulative += "</think>"
                                    yield cumulative
                                return
                            if not line.startswith("data: "):
                                continue

                            try:
                                data = json.loads(line[6:])
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    # Handle reasoning/thinking tokens
                                    # llama-server sends these as "reasoning_content"
                                    # Wrap in <think> tags for the frontend parser
                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        if not in_thinking:
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        yield cumulative

                                    token = delta.get("content", "")
                                    if token:
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                        cumulative += token
                                        yield cumulative
                            except json.JSONDecodeError:
                                logger.debug(
                                    f"Skipping malformed SSE line: {line[:100]}"
                                )

        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise

    # ── TTS support ────────────────────────────────────────────

    def detect_audio_type(self) -> Optional[str]:
        """Detect audio/TTS codec by probing the loaded model's vocabulary."""
        if not self.is_loaded:
            return None
        try:
            with httpx.Client(timeout = 10) as client:

                def _detok(tid: int) -> str:
                    r = client.post(
                        f"{self.base_url}/detokenize", json = {"tokens": [tid]}
                    )
                    return r.json().get("content", "") if r.status_code == 200 else ""

                def _tok(text: str) -> list[int]:
                    r = client.post(
                        f"{self.base_url}/tokenize",
                        json = {"content": text, "add_special": False},
                    )
                    return r.json().get("tokens", []) if r.status_code == 200 else []

                # Check codec-specific tokens (not generic ones that may exist in non-audio models)
                if "<custom_token_" in _detok(128258) and "<custom_token_" in _detok(
                    128259
                ):
                    return "snac"
                if len(_tok("<|AUDIO|>")) == 1 and len(_tok("<|audio_eos|>")) == 1:
                    return "csm"
                if len(_tok("<|startoftranscript|>")) == 1:
                    return "whisper"
                if (
                    len(_tok("<|bicodec_semantic_0|>")) == 1
                    and len(_tok("<|bicodec_global_0|>")) == 1
                ):
                    return "bicodec"
                if len(_tok("<|c1_0|>")) == 1 and len(_tok("<|c2_0|>")) == 1:
                    return "dac"
        except Exception as e:
            logger.debug(f"Audio type detection failed: {e}")
        return None

    # Prompt format per codec: (template, stop_tokens, needs_token_ids)
    # Matches prompts in InferenceBackend._generate_snac/bicodec/dac
    _TTS_PROMPTS = {
        "snac": (
            "<custom_token_3>{text}<|eot_id|><custom_token_4>",
            ["<custom_token_2>"],
            True,
        ),
        "bicodec": (
            "<|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>",
            ["<|im_end|>", "</s>"],
            False,
        ),
        "dac": (
            "<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|><|global_features_start|>\n",
            ["<|im_end|>", "<|audio_end|>"],
            False,
        ),
    }

    _codec_mgr = None  # Shared AudioCodecManager instance

    def init_audio_codec(self, audio_type: str) -> None:
        """Load the audio codec at model load time (mirrors non-GGUF path)."""
        import torch
        from core.inference.audio_codecs import AudioCodecManager

        if LlamaCppBackend._codec_mgr is None:
            LlamaCppBackend._codec_mgr = AudioCodecManager()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_repo_path = None

        # BiCodec needs a repo with BiCodec/ weights — download canonical SparkTTS
        if audio_type == "bicodec":
            from huggingface_hub import snapshot_download
            import os

            repo_path = snapshot_download(
                "unsloth/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B"
            )
            model_repo_path = os.path.abspath(repo_path)

        LlamaCppBackend._codec_mgr.load_codec(
            audio_type, device, model_repo_path = model_repo_path
        )
        logger.info(f"Loaded audio codec for GGUF TTS: {audio_type}")

    def generate_audio_response(
        self,
        text: str,
        audio_type: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
    ) -> tuple:
        """
        Generate TTS audio via llama-server /completion + codec decoding.
        Returns (wav_bytes, sample_rate).
        """
        if audio_type not in self._TTS_PROMPTS:
            raise RuntimeError(f"GGUF TTS does not support '{audio_type}' codec.")

        tpl, stop, need_ids = self._TTS_PROMPTS[audio_type]

        payload: dict = {
            "prompt": tpl.format(text = text),
            "stream": False,
            "n_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
        }
        if stop:
            payload["stop"] = stop
        if need_ids:
            payload["n_probs"] = 1

        with httpx.Client(timeout = httpx.Timeout(300, connect = 10)) as client:
            resp = client.post(f"{self.base_url}/completion", json = payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"llama-server returned {resp.status_code}: {resp.text}"
                )

        data = resp.json()
        token_ids = (
            [p["id"] for p in data.get("completion_probabilities", []) if "id" in p]
            if need_ids
            else None
        )

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LlamaCppBackend._codec_mgr.decode(
            audio_type, device, token_ids = token_ids, text = data.get("content", "")
        )

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions
through its OpenAI-compatible /v1/chat/completions endpoint.
"""

import atexit
import contextlib
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
        self._supports_reasoning: bool = False
        self._supports_tools: bool = False
        self._cache_type_kv: Optional[str] = None
        self._reasoning_default: bool = True
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

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_default(self) -> bool:
        return self._reasoning_default

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    @property
    def cache_type_kv(self) -> Optional[str]:
        return self._cache_type_kv

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
                    logger.debug(f"[llama-server] {line}")
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
        # Reset metadata from any previously loaded model so stale flags
        # (eg _supports_reasoning) do not carry over when switching models.
        self._context_length = None
        self._chat_template = None
        self._supports_reasoning = False
        self._supports_tools = False

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
                # Detect thinking/reasoning support from chat template
                tpl = self._chat_template
                if "enable_thinking" in tpl:
                    self._supports_reasoning = True
                    logger.info(
                        "GGUF metadata: model supports reasoning (enable_thinking)"
                    )
                elif "thinking" in tpl:
                    # DeepSeek uses 'thinking' instead of 'enable_thinking'
                    normalized_id = (self._model_identifier or "").lower()
                    if "deepseek" in normalized_id:
                        self._supports_reasoning = True
                        logger.info(
                            "GGUF metadata: model supports reasoning (DeepSeek thinking)"
                        )
                # Detect tool calling support from chat template
                tool_markers = [
                    "{%- if tools %}",
                    "{% if tools %}",
                    '"role" == "tool"',
                    "'role' == 'tool'",
                    'message.role == "tool"',
                    "message.role == 'tool'",
                ]
                if any(marker in tpl for marker in tool_markers):
                    self._supports_tools = True
                    logger.info("GGUF metadata: model supports tool calling")
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

        gguf_label = f"{hf_repo}/{gguf_filename}" + (
            f" (+{len(gguf_extra_shards)} shards)" if gguf_extra_shards else ""
        )
        logger.info(f"Resolving GGUF: {gguf_label}")
        try:
            if self._cancel_event.is_set():
                raise RuntimeError("Cancelled")
            dl_start = time.monotonic()
            local_path = hf_hub_download(
                repo_id = hf_repo,
                filename = gguf_filename,
                token = hf_token,
            )
            for shard in gguf_extra_shards:
                if self._cancel_event.is_set():
                    raise RuntimeError("Cancelled")
                logger.info(f"Resolving GGUF shard: {shard}")
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

        dl_elapsed = time.monotonic() - dl_start
        if dl_elapsed < 2.0:
            logger.info(f"GGUF resolved from cache: {local_path}")
        else:
            logger.info(f"GGUF downloaded in {dl_elapsed:.1f}s: {local_path}")
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
        chat_template_override: Optional[str] = None,
        cache_type_kv: Optional[str] = None,
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

        # Set identifier early so _read_gguf_metadata can use it for DeepSeek detection
        self._model_identifier = model_identifier

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

            # Always enable Jinja chat template rendering for proper template support
            cmd.extend(["--jinja"])

            # KV cache data type
            _valid_cache_types = {
                "f16",
                "bf16",
                "q8_0",
                "q4_0",
                "q4_1",
                "q5_0",
                "q5_1",
                "iq4_nl",
                "f32",
            }
            if cache_type_kv and cache_type_kv in _valid_cache_types:
                cmd.extend(
                    ["--cache-type-k", cache_type_kv, "--cache-type-v", cache_type_kv]
                )
                self._cache_type_kv = cache_type_kv
                logger.info(f"KV cache type: {cache_type_kv}")
            else:
                self._cache_type_kv = None

            # Apply custom chat template override if provided
            if chat_template_override:
                import tempfile

                self._chat_template_file = tempfile.NamedTemporaryFile(
                    mode = "w",
                    suffix = ".jinja",
                    delete = False,
                    prefix = "unsloth_chat_template_",
                )
                self._chat_template_file.write(chat_template_override)
                self._chat_template_file.close()
                cmd.extend(["--chat-template-file", self._chat_template_file.name])
                logger.info(
                    f"Using custom chat template file: {self._chat_template_file.name}"
                )

            # For reasoning models, set default thinking mode.
            # Qwen3.5 models below 9B (0.8B, 2B, 4B) disable thinking by default.
            # Only 9B and larger enable thinking.
            if self._supports_reasoning:
                import re

                thinking_default = True
                mid = (model_identifier or "").lower()
                if "qwen3.5" in mid:
                    # Extract size like "0.8b", "4b", "35b" etc.
                    size_match = re.search(r"(\d+\.?\d*)\s*b", mid)
                    if size_match:
                        size_val = float(size_match.group(1))
                        if size_val < 9:
                            thinking_default = False
                self._reasoning_default = thinking_default
                cmd.extend(
                    [
                        "--chat-template-kwargs",
                        json.dumps({"enable_thinking": thinking_default}),
                    ]
                )
                logger.info(
                    f"Reasoning model: enable_thinking={thinking_default} by default"
                )

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
                # and CUDA runtime libs (libcudart, libcublas, etc.)
                import platform

                lib_dirs = [binary_dir]
                _arch = platform.machine()  # x86_64, aarch64, etc.
                for cuda_lib in [
                    "/usr/local/cuda/lib64",
                    f"/usr/local/cuda/targets/{_arch}-linux/lib",
                    # Fallback CUDA compat paths (e.g. binary built with
                    # CUDA 12 on a system where default /usr/local/cuda
                    # points to CUDA 13+).
                    "/usr/local/cuda-12/lib64",
                    "/usr/local/cuda-12.8/lib64",
                    f"/usr/local/cuda-12/targets/{_arch}-linux/lib",
                    f"/usr/local/cuda-12.8/targets/{_arch}-linux/lib",
                ]:
                    if os.path.isdir(cuda_lib):
                        lib_dirs.append(cuda_lib)
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                new_ld = ":".join(lib_dirs)
                env["LD_LIBRARY_PATH"] = (
                    f"{new_ld}:{existing_ld}" if existing_ld else new_ld
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
            self._supports_reasoning = False
            self._supports_tools = False
            self._cache_type_kv = None
            # Clean up temp chat template file
            if hasattr(self, "_chat_template_file") and self._chat_template_file:
                try:
                    import os

                    os.unlink(self._chat_template_file.name)
                except Exception:
                    pass
                self._chat_template_file = None
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
    def _parse_tool_calls_from_text(content: str) -> list[dict]:
        """
        Parse tool calls from XML markup in content text.

        Handles formats like:
          <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
          <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
        Closing tags (</tool_call>, </function>, </parameter>) are all optional
        since models frequently omit them.
        """
        import re

        tool_calls = []

        # Pattern 1: JSON inside <tool_call> tags.
        # Use balanced-brace extraction that skips braces inside JSON strings.
        for m in re.finditer(r"<tool_call>\s*\{", content):
            brace_start = m.end() - 1  # position of the opening {
            depth, i = 0, brace_start
            in_string = False
            while i < len(content):
                ch = content[i]
                if in_string:
                    if ch == "\\" and i + 1 < len(content):
                        i += 2  # skip escaped character
                        continue
                    if ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                json_str = content[brace_start : i + 1]
                try:
                    obj = json.loads(json_str)
                    tc = {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": obj.get("name", ""),
                            "arguments": obj.get("arguments", {}),
                        },
                    }
                    if isinstance(tc["function"]["arguments"], dict):
                        tc["function"]["arguments"] = json.dumps(
                            tc["function"]["arguments"]
                        )
                    tool_calls.append(tc)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Pattern 2: XML-style <function=name><parameter=key>value</parameter></function>
        # All closing tags optional -- models frequently omit </parameter>,
        # </function>, and/or </tool_call>.
        if not tool_calls:
            # Step 1: Find all <function=name> positions and extract their bodies.
            # Body boundary: use only </tool_call> or next <function= as hard
            # boundaries.  We avoid using </function> as a boundary because
            # code parameter values can contain that literal string.
            # After extracting, we trim a trailing </function> if present.
            func_starts = list(re.finditer(r"<function=(\w+)>\s*", content))
            for idx, fm in enumerate(func_starts):
                func_name = fm.group(1)
                body_start = fm.end()
                # Hard boundaries: next <function= tag or </tool_call>
                next_func = (
                    func_starts[idx + 1].start()
                    if idx + 1 < len(func_starts)
                    else len(content)
                )
                end_tag = re.search(r"</tool_call>", content[body_start:])
                if end_tag:
                    body_end = body_start + end_tag.start()
                else:
                    body_end = len(content)
                body_end = min(body_end, next_func)
                body = content[body_start:body_end]
                # Trim trailing </function> if present (it's the real closing tag)
                body = re.sub(r"\s*</function>\s*$", "", body)

                # Step 2: Extract parameters from body.
                # For single-parameter functions (the common case: code, command,
                # query), use body end as the only boundary to avoid false matches
                # on </parameter> inside code strings.
                arguments = {}
                param_starts = list(re.finditer(r"<parameter=(\w+)>\s*", body))
                if len(param_starts) == 1:
                    # Single parameter: value is everything from after the tag
                    # to end of body, trimming any trailing </parameter>.
                    pm = param_starts[0]
                    val = body[pm.end() :]
                    val = re.sub(r"\s*</parameter>\s*$", "", val)
                    arguments[pm.group(1)] = val.strip()
                else:
                    for pidx, pm in enumerate(param_starts):
                        param_name = pm.group(1)
                        val_start = pm.end()
                        # Value ends at next <parameter= or end of body
                        next_param = (
                            param_starts[pidx + 1].start()
                            if pidx + 1 < len(param_starts)
                            else len(body)
                        )
                        val = body[val_start:next_param]
                        # Trim trailing </parameter> if present
                        val = re.sub(r"\s*</parameter>\s*$", "", val)
                        arguments[param_name] = val.strip()

                tc = {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments),
                    },
                }
                tool_calls.append(tc)

        return tool_calls

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

    @staticmethod
    def _iter_text_cancellable(
        response: "httpx.Response",
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Iterate over an httpx streaming response with cancel support.

        Checks cancel_event between chunks and on ReadTimeout.  The
        cancel watcher in _stream_with_retry also calls response.close()
        on cancel, which unblocks iter_text() once the response exists.
        During normal streaming llama-server sends tokens frequently,
        so the cancel check between chunks is the primary mechanism.
        """
        text_iter = response.iter_text()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            try:
                chunk = next(text_iter)
                yield chunk
            except StopIteration:
                return
            except httpx.ReadTimeout:
                # No data within the timeout window -- just loop back
                # and re-check cancel_event.
                continue

    @staticmethod
    @contextlib.contextmanager
    def _stream_with_retry(
        client: "httpx.Client",
        url: str,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Open an httpx streaming POST with cancel support.

        Sends the request once with a long read timeout (120 s) so
        prompt processing (prefill) can finish without triggering a
        retry storm.  The previous 0.5 s timeout caused duplicate POST
        requests every half second, forcing llama-server to restart
        processing each time.

        A background watcher thread provides cancel by closing the
        response when cancel_event is set.  Limitation: httpx does not
        allow interrupting a blocked read from another thread before
        the response object exists, so cancel during the initial
        header wait (prefill phase) only takes effect once headers
        arrive.  After that, response.close() unblocks reads promptly.
        In practice llama-server prefill is 1-5 s for typical prompts,
        during which cancel is deferred -- still much better than the
        old retry storm which made prefill slower.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise GeneratorExit

        # Background watcher: close the response if cancel is requested.
        # Only effective after response headers arrive (httpx limitation).
        _cancel_closed = threading.Event()
        _response_ref: list = [None]

        def _cancel_watcher():
            while not _cancel_closed.is_set():
                if cancel_event.wait(timeout = 0.3):
                    # Cancel requested. Keep polling until the response object
                    # exists so we can close it, or until the main thread
                    # finishes on its own (_cancel_closed is set in finally).
                    while not _cancel_closed.is_set():
                        r = _response_ref[0]
                        if r is not None:
                            try:
                                r.close()
                                return
                            except Exception as e:
                                logger.debug(
                                    f"Error closing response in cancel watcher: {e}"
                                )
                        # Response not created yet -- wait briefly and retry
                        _cancel_closed.wait(timeout = 0.1)
                    return

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, daemon = True, name = "prefill-cancel"
            )
            watcher.start()

        try:
            # Long read timeout so prefill (prompt processing) can finish
            # without triggering a retry storm.  Cancel during both
            # prefill and streaming is handled by the watcher thread
            # which closes the response, unblocking any httpx read.
            prefill_timeout = httpx.Timeout(
                connect = 30,
                read = 120.0,
                write = 10,
                pool = 10,
            )
            with client.stream(
                "POST", url, json = payload, timeout = prefill_timeout
            ) as response:
                _response_ref[0] = response
                if cancel_event is not None and cancel_event.is_set():
                    raise GeneratorExit
                yield response
                return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.CloseError):
            # Response was closed by the cancel watcher
            if cancel_event is not None and cancel_event.is_set():
                raise GeneratorExit
            raise
        finally:
            _cancel_closed.set()

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
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
            "presence_penalty": presence_penalty,
        }
        # Pass enable_thinking per-request for reasoning models
        if self._supports_reasoning and enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False

        try:
            # _stream_with_retry uses a 120 s read timeout so prefill
            # can finish.  Cancel during streaming is handled by the
            # watcher thread (closes the response on cancel_event).
            stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
            with httpx.Client(timeout = stream_timeout) as client:
                with self._stream_with_retry(
                    client, url, payload, cancel_event
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    has_content_tokens = False
                    reasoning_text = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response, cancel_event
                    ):
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if has_content_tokens:
                                        # Real thinking + content: close the tag
                                        cumulative += "</think>"
                                        yield cumulative
                                    else:
                                        # Only reasoning_content, no content tokens:
                                        # the model put its entire reply in reasoning
                                        # (e.g. Qwen3 always-think mode). Show it
                                        # as the main response, not as a thinking block.
                                        cumulative = reasoning_text
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
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        yield cumulative

                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
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

    # ── Tool-calling agentic loop ──────────────────────────────

    def generate_chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        max_tool_iterations: int = 10,
        auto_heal_tool_calls: bool = True,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Agentic loop: let the model call tools, execute them, and continue.

        Yields dicts with:
          {"type": "status", "text": "Searching: ..."}   -- tool status updates
          {"type": "content", "text": "token"}            -- streamed content tokens (cumulative)
          {"type": "reasoning", "text": "token"}          -- streamed reasoning tokens (cumulative)
        """
        from core.inference.tools import execute_tool

        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        conversation = list(messages)
        url = f"{self.base_url}/v1/chat/completions"

        for iteration in range(max_tool_iterations):
            if cancel_event is not None and cancel_event.is_set():
                return

            # Build payload for non-streaming tool detection pass
            payload = {
                "messages": conversation,
                "stream": False,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if top_k >= 0 else 0,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "tools": tools,
                "tool_choice": "auto",
            }
            if self._supports_reasoning and enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if stop:
                payload["stop"] = stop

            try:
                with httpx.Client(timeout = None) as client:
                    resp = client.post(url, json = payload)
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"llama-server returned {resp.status_code}: {resp.text}"
                        )
                    data = resp.json()
            except httpx.ConnectError:
                raise RuntimeError("Lost connection to llama-server")

            choices = data.get("choices", [])
            if not choices:
                return

            choice = choices[0]
            finish_reason = choice.get("finish_reason", "")
            message = choice.get("message", {})

            # If model wants to call tools
            tool_calls = message.get("tool_calls")

            # Fallback: detect tool calls embedded as XML/text in content
            # Some models output <tool_call> XML instead of structured tool_calls,
            # or bare <function=...> tags without <tool_call> wrapper.
            content_text = message.get("content", "") or ""
            if (
                auto_heal_tool_calls
                and not tool_calls
                and ("<tool_call>" in content_text or "<function=" in content_text)
            ):
                tool_calls = self._parse_tool_calls_from_text(content_text)
                if tool_calls:
                    logger.info(
                        f"Parsed {len(tool_calls)} tool call(s) from content text"
                    )

            # Always strip tool-call XML from content_text when any tool
            # calls are present.  llama-server may return structured
            # tool_calls AND also leave <tool_call> XML in the content
            # field, which would leak into the chat UI and conversation.
            if (
                auto_heal_tool_calls
                and tool_calls
                and ("<tool_call>" in content_text or "<function=" in content_text)
            ):
                import re

                content_text = re.sub(
                    r"<tool_call>.*?</tool_call>",
                    "",
                    content_text,
                    flags = re.DOTALL,
                )
                content_text = re.sub(
                    r"<tool_call>.*$",
                    "",
                    content_text,
                    flags = re.DOTALL,
                )
                content_text = re.sub(
                    r"<function=\w+>.*?</function>",
                    "",
                    content_text,
                    flags = re.DOTALL,
                )
                content_text = re.sub(
                    r"<function=\w+>.*$",
                    "",
                    content_text,
                    flags = re.DOTALL,
                ).strip()

            if finish_reason == "tool_calls" or (tool_calls and len(tool_calls) > 0):
                # Append the assistant message with tool_calls to conversation
                assistant_msg = {"role": "assistant", "content": content_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                conversation.append(assistant_msg)

                # Execute each tool call
                for tc in tool_calls or []:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    raw_args = func.get("arguments", {})

                    # Handle arguments as either string or dict
                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            if auto_heal_tool_calls:
                                arguments = {"query": raw_args}
                            else:
                                arguments = {"raw": raw_args}
                    else:
                        arguments = raw_args

                    # Yield status update
                    if tool_name == "web_search":
                        status_text = f"Searching: {arguments.get('query', '')}"
                    elif tool_name == "python":
                        preview = (
                            (arguments.get("code") or "").strip().split("\n")[0][:60]
                        )
                        status_text = (
                            f"Running Python: {preview}"
                            if preview
                            else "Running Python..."
                        )
                    elif tool_name == "terminal":
                        cmd_preview = (arguments.get("command") or "")[:60]
                        status_text = (
                            f"Running: {cmd_preview}"
                            if cmd_preview
                            else "Running command..."
                        )
                    else:
                        status_text = f"Calling: {tool_name}"
                    yield {"type": "status", "text": status_text}

                    # Emit tool_start so the frontend can record inputs
                    yield {
                        "type": "tool_start",
                        "tool_name": tool_name,
                        "tool_call_id": tc.get("id", ""),
                        "arguments": arguments,
                    }

                    # Execute the tool
                    _effective_timeout = (
                        None if tool_call_timeout >= 9999 else tool_call_timeout
                    )
                    result = execute_tool(
                        tool_name,
                        arguments,
                        cancel_event = cancel_event,
                        timeout = _effective_timeout,
                        session_id = session_id,
                    )

                    # Emit tool_end so the frontend can record outputs
                    yield {
                        "type": "tool_end",
                        "tool_name": tool_name,
                        "tool_call_id": tc.get("id", ""),
                        "result": result,
                    }

                    # Append tool result to conversation
                    tool_msg = {
                        "role": "tool",
                        "name": tool_name,
                        "content": result,
                    }
                    tool_call_id = tc.get("id")
                    if tool_call_id:
                        tool_msg["tool_call_id"] = tool_call_id
                    conversation.append(tool_msg)

                # Continue the loop to let model respond with context
                continue

            # No tool calls -- model answered directly.
            # If no tools were executed at all, just yield the content
            # from this response instead of making a redundant second request.
            if iteration == 0 and content_text:
                yield {"type": "status", "text": ""}
                yield {"type": "content", "text": content_text}
                return

            # Tools were called in previous iterations; do a final
            # streaming pass so the model can synthesize a response
            # incorporating the tool results.
            break

        # Clear status
        yield {"type": "status", "text": ""}

        # Final streaming pass with the full conversation context.
        # Add stop sequences so the model cannot emit tool-call XML --
        # the non-streaming loop above already handled all tool
        # iterations.  If the model tries to call tools here it will
        # simply stop, and we yield whatever text came before.
        stream_payload = {
            "messages": conversation,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        if self._supports_reasoning and enable_thinking is not None:
            stream_payload["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }
        if max_tokens is not None:
            stream_payload["max_tokens"] = max_tokens
        _stop = list(stop) if stop else []
        if auto_heal_tool_calls:
            _stop += ["<tool_call>", "<function="]
        stream_payload["stop"] = _stop

        import re as _re_final

        _TOOL_PATTERNS = [
            _re_final.compile(r"<tool_call>.*?</tool_call>", _re_final.DOTALL),
            _re_final.compile(r"<function=\w+>.*?</function>", _re_final.DOTALL),
            _re_final.compile(r"<tool_call>.*$", _re_final.DOTALL),
            _re_final.compile(r"<function=\w+>.*$", _re_final.DOTALL),
        ]

        def _strip_tool_markup(text: str, *, final: bool = False) -> str:
            if not auto_heal_tool_calls:
                return text
            for pat in _TOOL_PATTERNS:
                text = pat.sub("", text)
            return text.strip() if final else text

        cumulative = ""
        _last_emitted = ""
        in_thinking = False
        has_content_tokens = False
        reasoning_text = ""

        try:
            stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
            with httpx.Client(timeout = stream_timeout) as client:
                with self._stream_with_retry(
                    client, url, stream_payload, cancel_event
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response, cancel_event
                    ):
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if has_content_tokens:
                                        cumulative += "</think>"
                                        yield {
                                            "type": "content",
                                            "text": _strip_tool_markup(
                                                cumulative, final = True
                                            ),
                                        }
                                    else:
                                        cumulative = reasoning_text
                                        yield {"type": "content", "text": cumulative}
                                return
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        yield {"type": "content", "text": cumulative}

                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                        cumulative += token
                                        cleaned = _strip_tool_markup(cumulative)
                                        # Only emit when cleaned text grows (monotonic).
                                        if len(cleaned) > len(_last_emitted):
                                            _last_emitted = cleaned
                                            yield {"type": "content", "text": cleaned}
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

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions
through its OpenAI-compatible /v1/chat/completions endpoint.
"""

import atexit
import json
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
        self._lock = threading.Lock()
        self._stdout_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None

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
        Only returns GPUs that are allowed by CUDA_VISIBLE_DEVICES
        (if set). Returns empty list if nvidia-smi is not available.
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
            gguf_files = [f for f in files if f.endswith(".gguf")]
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
        with self._lock:
            self._kill_process()

            binary = self._find_llama_server_binary()
            if not binary:
                raise RuntimeError(
                    "llama-server binary not found. "
                    "Run setup.sh to build it, install llama.cpp, "
                    "or set LLAMA_SERVER_PATH environment variable."
                )

            self._port = self._find_free_port()

            # Build command based on mode
            if hf_repo:
                # Download the GGUF file ourselves using huggingface_hub
                # (llama-server's -hf flag requires HTTPS/curl which may not
                #  be available, e.g. Windows builds with -DLLAMA_CURL=OFF)
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError:
                    raise RuntimeError(
                        "huggingface_hub is required for HF model loading. "
                        "Install it with: pip install huggingface_hub"
                    )

                # Determine the filename from the variant (e.g., "Q4_K_M" -> find matching file)
                # For split GGUFs (e.g., *-00001-of-00003.gguf) we must download ALL shards.
                gguf_filename = None
                gguf_extra_shards: list[str] = []
                if hf_variant:
                    # Try common naming patterns
                    try:
                        import re
                        from huggingface_hub import list_repo_files

                        files = list_repo_files(hf_repo, token = hf_token)
                        variant_lower = hf_variant.lower()
                        # Use word-boundary matching so "Q8_0" doesn't also
                        # match "IQ8_0" or other superset variant names.
                        boundary = re.compile(
                            r"(?<![a-zA-Z0-9])"
                            + re.escape(variant_lower)
                            + r"(?![a-zA-Z0-9])"
                        )
                        gguf_files = sorted(
                            f
                            for f in files
                            if f.endswith(".gguf") and boundary.search(f.lower())
                        )
                        if gguf_files:
                            gguf_filename = gguf_files[0]
                            # For split GGUFs (e.g. model-Q8_0-00001-of-00003.gguf)
                            # discover siblings by exact basename + total match
                            # so "model-Q8_0-v2-*" isn't pulled in as a sibling.
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
                        # Fallback: construct common filename pattern
                        # e.g., "unsloth/gemma-3-4b-it-GGUF" + "Q4_K_M" -> try model name
                        repo_name = hf_repo.split("/")[-1].replace("-GGUF", "")
                        gguf_filename = f"{repo_name}-{hf_variant}.gguf"

                # Check disk space and fall back to a smaller variant if needed
                all_gguf_files = [gguf_filename] + gguf_extra_shards
                try:
                    import os

                    from huggingface_hub import get_paths_info

                    path_infos = list(
                        get_paths_info(hf_repo, all_gguf_files, token = hf_token)
                    )
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
                            # Try to find a smaller variant that fits
                            smaller = self._find_smallest_fitting_variant(
                                hf_repo,
                                free_bytes,
                                hf_token,
                            )
                            if smaller:
                                logger.info(
                                    f"Selected variant too large ({total_gb:.1f} GB), "
                                    f"falling back to {smaller[0]} ({smaller[1] / (1024**3):.1f} GB)"
                                )
                                gguf_filename = smaller[0]
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
                    + (
                        f" (+{len(gguf_extra_shards)} shards)"
                        if gguf_extra_shards
                        else ""
                    )
                )
                try:
                    local_path = hf_hub_download(
                        repo_id = hf_repo,
                        filename = gguf_filename,
                        token = hf_token,
                    )
                    # Download remaining shards for split GGUFs — llama-server
                    # auto-discovers them when they are in the same directory.
                    for shard in gguf_extra_shards:
                        logger.info(f"Downloading GGUF shard: {shard}")
                        hf_hub_download(
                            repo_id = hf_repo,
                            filename = shard,
                            token = hf_token,
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download GGUF file '{gguf_filename}' from {hf_repo}: {e}"
                    )

                logger.info(f"GGUF downloaded to: {local_path}")
                model_path = local_path
            elif gguf_path:
                if not Path(gguf_path).is_file():
                    raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
                model_path = gguf_path
            else:
                raise ValueError("Either gguf_path or hf_repo must be provided")

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
                str(n_ctx),
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
        """Terminate the llama-server subprocess and clean up state."""
        with self._lock:
            self._kill_process()
            logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._hf_repo = None
            self._hf_variant = None
            self._is_vision = False
            self._port = None
            self._healthy = False
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
        """Kill any orphaned llama-server processes from previous studio runs."""
        import os
        import signal

        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output = True,
                text = True,
                timeout = 5,
            )
            if result.returncode != 0:
                return
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                if pid == os.getpid():
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
        repetition_penalty: float = 1.1,
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

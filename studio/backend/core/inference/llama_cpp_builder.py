# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Background llama.cpp compilation.

On server startup, if the llama-server binary is not found, a background
thread clones and builds llama.cpp from source. Features that need
llama.cpp (GGUF chat, export, AI Assist) can await `wait_for_ready()`
which blocks until compilation finishes or fails.

Cross-platform: Linux, macOS, Windows.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import structlog
from loggers import get_logger

logger = get_logger(__name__)

_IS_WIN = sys.platform == "win32"
_IS_MAC = sys.platform == "darwin"


class LlamaCppBuilder:
    """Manages background llama.cpp compilation."""

    def __init__(self):
        self._ready = threading.Event()
        self._building = False
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._binary_path: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def is_building(self) -> bool:
        return self._building

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def binary_path(self) -> Optional[str]:
        return self._binary_path

    def check_and_build(self) -> None:
        """Check if llama-server exists. If not, start background build."""
        from core.inference.llama_cpp import LlamaCppBackend

        binary = LlamaCppBackend._find_llama_server_binary()
        if binary:
            self._binary_path = binary
            self._ready.set()
            logger.info(f"llama-server binary found: {binary}")
            return

        logger.info("llama-server binary not found, starting background build...")
        self._building = True
        self._thread = threading.Thread(
            target = self._build, daemon = True, name = "llama-cpp-build"
        )
        self._thread.start()

    def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until llama.cpp is ready. Returns True if ready, False on timeout."""
        return self._ready.wait(timeout = timeout)

    # ── Platform helpers ─────────────────────────────────────────

    @staticmethod
    def _find_nvcc() -> Optional[str]:
        """Find nvcc across platforms."""
        nvcc = shutil.which("nvcc")
        if nvcc:
            return nvcc

        if _IS_WIN:
            # Windows: check CUDA_PATH env, then standard install dirs
            cuda_path = os.environ.get("CUDA_PATH", "")
            if cuda_path:
                candidate = Path(cuda_path) / "bin" / "nvcc.exe"
                if candidate.is_file():
                    return str(candidate)
            toolkit_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
            if toolkit_base.is_dir():
                for d in sorted(toolkit_base.iterdir(), reverse = True):
                    candidate = d / "bin" / "nvcc.exe"
                    if candidate.is_file():
                        return str(candidate)
        else:
            # Linux: check standard locations
            if Path("/usr/local/cuda/bin/nvcc").is_file():
                return "/usr/local/cuda/bin/nvcc"
            for cuda_dir in sorted(
                Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse = True
            ):
                return str(cuda_dir)

        return None

    @staticmethod
    def _detect_cuda_architectures() -> Optional[str]:
        """Detect GPU compute capabilities via nvidia-smi."""
        nvidia_smi = "nvidia-smi"
        if _IS_WIN:
            # nvidia-smi is typically in System32 on Windows
            win_path = Path(r"C:\Windows\System32\nvidia-smi.exe")
            if win_path.is_file():
                nvidia_smi = str(win_path)

        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output = True,
                text = True,
                timeout = 10,
            )
            if result.returncode == 0:
                caps = set()
                for line in result.stdout.strip().split("\n"):
                    cap = line.strip().replace(".", "")
                    if cap:
                        caps.add(cap)
                if caps:
                    return ";".join(sorted(caps))
        except Exception:
            pass
        return None

    # ── Build ────────────────────────────────────────────────────

    def _build(self) -> None:
        """Clone and build llama.cpp in the background."""
        try:
            llama_dir = Path.home() / ".unsloth" / "llama.cpp"
            binary_name = "llama-server.exe" if _IS_WIN else "llama-server"

            # Don't rebuild if binary appeared while we were starting
            from core.inference.llama_cpp import LlamaCppBackend

            binary = LlamaCppBackend._find_llama_server_binary()
            if binary:
                self._binary_path = binary
                self._building = False
                self._ready.set()
                return

            # Check prerequisites
            if not shutil.which("cmake"):
                self._error = "cmake not found. Install cmake to enable GGUF inference."
                self._building = False
                logger.warning(self._error)
                return
            if not shutil.which("git"):
                self._error = "git not found. Install git to enable GGUF inference."
                self._building = False
                logger.warning(self._error)
                return

            # Clone (don't delete if already exists -- might be a partial build)
            if not (llama_dir / "CMakeLists.txt").is_file():
                if llama_dir.exists():
                    shutil.rmtree(llama_dir, ignore_errors = True)

                logger.info("Cloning llama.cpp...")
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/ggml-org/llama.cpp.git",
                        str(llama_dir),
                    ],
                    check = True,
                    capture_output = True,
                )
            else:
                logger.info("llama.cpp source already present, building...")

            # ── CMake arguments ──────────────────────────────────
            cmake_args = [
                "-DBUILD_SHARED_LIBS=OFF",  # Self-contained binary
                "-DGGML_NATIVE=ON",  # Native CPU optimizations
                "-DLLAMA_BUILD_TESTS=OFF",  # Skip tests
                "-DLLAMA_BUILD_EXAMPLES=OFF",  # Skip examples
                "-DLLAMA_BUILD_SERVER=ON",  # Ensure server target
            ]

            # ccache (27x faster rebuilds when available)
            ccache = shutil.which("ccache")
            if ccache:
                cmake_args.extend(
                    [
                        f"-DCMAKE_C_COMPILER_LAUNCHER={ccache}",
                        f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache}",
                        f"-DCMAKE_CUDA_COMPILER_LAUNCHER={ccache}",
                    ]
                )
                logger.info("Using ccache for faster compilation")

            # ── GPU backend ──────────────────────────────────────
            if _IS_MAC:
                # macOS: Metal is enabled by default in llama.cpp, no extra flags needed
                logger.info("macOS detected, Metal backend enabled by default")
            else:
                nvcc = self._find_nvcc()
                if nvcc:
                    logger.info(f"CUDA detected: {nvcc}")
                    cmake_args.append("-DGGML_CUDA=ON")
                    # Build only for the detected GPU architecture(s)
                    archs = self._detect_cuda_architectures()
                    if archs:
                        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={archs}")
                    # Multi-threaded CUDA compilation
                    cmake_args.append("-DCMAKE_CUDA_FLAGS=--threads=0")
                else:
                    logger.info("No CUDA detected, building CPU-only llama.cpp")

            # ── Generator ────────────────────────────────────────
            generator_args = []
            if _IS_WIN:
                # Windows: prefer Ninja, fall back to VS generator
                if shutil.which("ninja"):
                    generator_args = ["-G", "Ninja"]
                # else: cmake will use default VS generator
            else:
                # Linux/Mac: prefer Ninja
                if shutil.which("ninja"):
                    generator_args = ["-G", "Ninja"]

            # ── Configure ────────────────────────────────────────
            build_dir = llama_dir / "build"
            logger.info("Configuring llama.cpp build...")
            subprocess.run(
                ["cmake", "-B", str(build_dir), "-S", str(llama_dir)]
                + generator_args
                + cmake_args,
                check = True,
                capture_output = True,
            )

            # ── Build ────────────────────────────────────────────
            logger.info("Building llama.cpp (this may take a few minutes)...")
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--config",
                    "Release",
                    "--target",
                    "llama-server",
                    "llama-quantize",
                    "-j",
                ],
                check = True,
                capture_output = True,
            )

            # ── Verify binaries ──────────────────────────────────
            # Windows VS generator puts binaries in build/bin/Release/
            if _IS_WIN:
                bin_dir = build_dir / "bin" / "Release"
                if not (bin_dir / binary_name).is_file():
                    bin_dir = build_dir / "bin"  # Ninja puts them here
            else:
                bin_dir = build_dir / "bin"

            server_bin = bin_dir / binary_name
            quantize_name = "llama-quantize.exe" if _IS_WIN else "llama-quantize"
            quantize_bin = bin_dir / quantize_name

            if server_bin.is_file():
                self._binary_path = str(server_bin)
                logger.info(f"llama-server built: {self._binary_path}")
            else:
                self._error = (
                    f"Build completed but llama-server not found at {server_bin}"
                )
                logger.error(self._error)

            if quantize_bin.is_file():
                logger.info(f"llama-quantize built: {quantize_bin}")
                # Create symlink/copy for unsloth-zoo's check_llama_cpp()
                quantize_link = llama_dir / quantize_name
                if not quantize_link.exists():
                    try:
                        if _IS_WIN:
                            shutil.copy2(str(quantize_bin), str(quantize_link))
                        else:
                            quantize_link.symlink_to(quantize_bin)
                    except Exception:
                        pass
            else:
                logger.warning(f"llama-quantize not found at {quantize_bin}")

        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            ) or ""
            self._error = f"llama.cpp build failed: {stderr[-500:]}"
            logger.error(self._error)
        except Exception as e:
            self._error = f"llama.cpp build error: {e}"
            logger.error(self._error)
        finally:
            self._building = False
            self._ready.set()  # Unblock waiters even on failure


# ── Singleton ────────────────────────────────────────────────────

_builder: Optional[LlamaCppBuilder] = None


def get_llama_cpp_builder() -> LlamaCppBuilder:
    global _builder
    if _builder is None:
        _builder = LlamaCppBuilder()
    return _builder

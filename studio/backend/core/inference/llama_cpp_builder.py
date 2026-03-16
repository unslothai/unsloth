# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Background llama.cpp compilation.

On server startup, if the llama-server binary is not found, a background
thread clones and builds llama.cpp from source. Features that need
llama.cpp (GGUF chat, export, AI Assist) can await `wait_for_ready()`
which blocks until compilation finishes or fails.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import structlog
from loggers import get_logger

logger = get_logger(__name__)


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

    def _build(self) -> None:
        """Clone and build llama.cpp in the background."""
        try:
            import sys

            llama_dir = Path.home() / ".unsloth" / "llama.cpp"
            binary_name = (
                "llama-server.exe" if sys.platform == "win32" else "llama-server"
            )

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

            # Clone
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

            # Build configuration: static binary, only needed targets, max parallelism
            cmake_args = [
                "-DBUILD_SHARED_LIBS=OFF",  # Self-contained binary, no LD_LIBRARY_PATH needed
                "-DGGML_NATIVE=ON",  # Native CPU optimizations
                "-DLLAMA_BUILD_TESTS=OFF",  # Skip tests
                "-DLLAMA_BUILD_EXAMPLES=OFF",  # Skip examples (we build server explicitly)
                "-DLLAMA_BUILD_SERVER=ON",  # Ensure server target is available
            ]

            # Use ccache if available (27x faster rebuilds)
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

            # Detect CUDA
            nvcc = shutil.which("nvcc")
            if not nvcc:
                for cuda_dir in sorted(
                    Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse = True
                ):
                    nvcc = str(cuda_dir)
                    break
            if not nvcc and Path("/usr/local/cuda/bin/nvcc").is_file():
                nvcc = "/usr/local/cuda/bin/nvcc"

            if nvcc:
                logger.info(f"CUDA detected: {nvcc}")
                cmake_args.append("-DGGML_CUDA=ON")
                # Detect compute capabilities (build only for this GPU, not all)
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=compute_cap",
                            "--format=csv,noheader",
                        ],
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
                            cmake_args.append(
                                f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(sorted(caps))}"
                            )
                except Exception:
                    pass
                # Multi-threaded CUDA compilation
                cmake_args.append("-DCMAKE_CUDA_FLAGS=--threads=0")
            else:
                logger.info("No CUDA detected, building CPU-only llama.cpp")

            # Use Ninja if available (faster than Make)
            generator_args = []
            if shutil.which("ninja"):
                generator_args = ["-G", "Ninja"]

            # Configure
            build_dir = llama_dir / "build"
            logger.info("Configuring llama.cpp build...")
            subprocess.run(
                [
                    "cmake",
                    "-B",
                    str(build_dir),
                    "-S",
                    str(llama_dir),
                ]
                + generator_args
                + cmake_args,
                check = True,
                capture_output = True,
            )

            # Build only the targets we need (llama-server + llama-quantize)
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

            # Symlink llama-quantize for unsloth-zoo's check_llama_cpp()
            quantize_src = build_dir / "bin" / "llama-quantize"
            quantize_link = llama_dir / "llama-quantize"
            if quantize_src.is_file() and not quantize_link.exists():
                try:
                    quantize_link.symlink_to(quantize_src)
                except Exception:
                    pass

            # Verify
            expected = build_dir / "bin" / binary_name
            if expected.is_file():
                self._binary_path = str(expected)
                logger.info(f"llama.cpp build complete: {self._binary_path}")
            else:
                self._error = f"Build completed but binary not found at {expected}"
                logger.error(self._error)

        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else ""
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

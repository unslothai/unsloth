#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build beellama.cpp from source for Unsloth Studio.

Companion to ``install_llama_prebuilt.py``. Instead of downloading a prebuilt
llama.cpp bundle, this detects the host GPU backend (CUDA / ROCm / Metal / CPU)
and compiles the beellama.cpp fork (the turboquant build that understands the
``turbo4`` KV-cache type) at a pinned ref with cmake.

Hardware detection is reused verbatim from ``install_llama_prebuilt.py`` so the
two installers agree on what the host is. Binaries land in exactly the layout
the inference backend searches (see ``_find_llama_server_binary`` in
``core/inference/llama_cpp.py``) and that ``install_llama_prebuilt.py`` targets
via ``normalize_install_layout``::

    <install-dir>/build/bin/llama-server             (Linux / macOS, cmake)
    <install-dir>/build/bin/Release/llama-server.exe (Windows, multi-config)

On Linux / macOS ``llama-quantize`` (and the optional DiffusionGemma visual
server) are also symlinked into the install-dir root, mirroring ``setup.sh``.

The compile recipe mirrors the source-build branch of ``setup.sh`` /
``setup.ps1`` so a beellama build picks the same CMake flags Unsloth already
uses for upstream llama.cpp.

Usage::

    python install_beellama_source.py --install-dir ~/.unsloth/llama.cpp
    python install_beellama_source.py --install-dir ~/.unsloth/llama.cpp --backend cpu
    python install_beellama_source.py --install-dir ~/.unsloth/llama.cpp --ref v0.3.2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Reuse the prebuilt installer's host detection so the two agree on the host
# profile (CUDA visibility, ROCm gfx target, compute caps, ...). Kept as an
# import rather than a copy so a detection fix lands in both at once. The script
# lives next to install_llama_prebuilt.py, so add its own dir to sys.path for
# the standalone-invocation case.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from install_llama_prebuilt import (  # noqa: E402
        HostInfo,
        detect_host,
        normalize_compute_caps,
        run_capture,
    )
except Exception as exc:  # pragma: no cover - defensive: upstream refactor
    print(
        "install_beellama_source.py: could not import host detection from "
        f"install_llama_prebuilt.py ({exc}). Run this script from the studio/ "
        "directory next to install_llama_prebuilt.py.",
        file = sys.stderr,
    )
    raise SystemExit(2)


# ── Constants ────────────────────────────────────────────────────────

DEFAULT_REPO = os.environ.get("UNSLOTH_BEELLAMA_REPO", "https://github.com/Anbeeld/beellama.cpp")
DEFAULT_REF = os.environ.get("UNSLOTH_BEELLAMA_REF", "v0.3.2") # Experimental? I believe it's the only branch supporting kvarn, but I may be wrong. Switch to main if it causes issues.

# Matches setup.sh / setup.ps1 so the rest of Studio treats this install-dir as
# owned (safe to replace on a re-run, never an unrelated user directory).
STUDIO_OWNED_MARKER = ".unsloth-studio-owned"

# Informational marker recording what this script built. Not the prebuilt
# metadata file -- a source build deliberately does not masquerade as a
# published prebuilt.
METADATA_FILENAME = "beellama-source.json"

# llama.cpp cmake targets. Only the server is required; the rest are best effort
# (a fork at a given ref may not ship every example target).
REQUIRED_TARGET = "llama-server"
OPTIONAL_TARGETS = ("llama-quantize", "llama-diffusion-gemma-visual-server")

# Pin a low macOS deployment target so the build also loads on older macOS
# (mirrors setup.sh). Overridable via the same env var setup.sh reads.
MACOS_DEPLOYMENT_TARGET = os.environ.get("UNSLOTH_MACOS_DEPLOYMENT_TARGET", "13.3")

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


class BuildError(RuntimeError):
    """Fatal, user-facing build failure."""


def log(message: str) -> None:
    print(f"[beellama] {message}", flush = True)


# ── Small subprocess helpers ─────────────────────────────────────────

def which(name: str) -> str | None:
    return shutil.which(name)


def run_streaming(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Run a command, streaming its output to our stdout. Returns the exit code.

    Used for the long-running clone/cmake steps so the user sees live progress
    (unlike ``run_capture``, which buffers for short detection probes).
    """
    log("$ " + " ".join(command))
    completed = subprocess.run(command, cwd = str(cwd) if cwd else None, env = env)
    return completed.returncode


def _rmtree_robust(path: Path, *, best_effort: bool = False) -> None:
    """Remove a directory tree, retrying read-only files.

    Git clones mark packed objects read-only, which makes a plain
    ``shutil.rmtree`` raise ``PermissionError`` on Windows. The handler clears
    the read-only bit and retries. With ``best_effort`` a residual failure is
    swallowed (cleanup path); otherwise it propagates (the swap must succeed).
    """
    if not path.exists():
        return

    def _on_error(func, fpath, _exc):
        try:
            os.chmod(fpath, 0o700)
            func(fpath)
        except OSError:
            if not best_effort:
                raise

    # Python 3.12+ renamed the hook to onexc; keep onerror for older runtimes.
    try:
        shutil.rmtree(path, onexc = lambda f, p, e: _on_error(f, p, e))  # type: ignore[call-arg]
    except TypeError:
        shutil.rmtree(path, onerror = lambda f, p, e: _on_error(f, p, e))
    except OSError:
        if not best_effort:
            raise


def capture(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Thin wrapper over the shared run_capture for short probes."""
    return run_capture(command, timeout = timeout, check = False)


# ── Toolchain discovery (mirrors setup.sh's nvcc/hipcc search) ───────

def find_nvcc() -> str | None:
    """Locate nvcc on PATH, then the conventional CUDA install dirs."""
    found = which("nvcc")
    if found:
        return found
    candidates = ["/usr/local/cuda/bin/nvcc"]
    # Newest /usr/local/cuda-XX.Y/bin/nvcc, if any.
    cuda_root = Path("/usr/local")
    if cuda_root.is_dir():
        versioned = sorted(
            (p for p in cuda_root.glob("cuda-*/bin/nvcc") if p.is_file()),
            key = lambda p: p.parent.parent.name,
        )
        candidates.extend(str(p) for p in reversed(versioned))
    if IS_WINDOWS:
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            candidates.append(str(Path(cuda_path) / "bin" / "nvcc.exe"))
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def find_hipcc() -> str | None:
    """Locate the HIP compiler driver on PATH, then conventional ROCm dirs."""
    found = which("hipcc")
    if found:
        return found
    candidates = ["/opt/rocm/bin/hipcc"]
    rocm_root = Path("/opt")
    if rocm_root.is_dir():
        versioned = sorted(
            (p for p in rocm_root.glob("rocm-*/bin/hipcc") if p.is_file()),
            key = lambda p: p.parent.parent.name,
        )
        candidates.extend(str(p) for p in reversed(versioned))
    for env_var in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(env_var)
        if root:
            suffix = "hipcc.exe" if IS_WINDOWS else "hipcc"
            candidates.append(str(Path(root) / "bin" / suffix))
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


# ── Backend selection ────────────────────────────────────────────────

def resolve_backend(host: HostInfo, override: str) -> str:
    """Decide the GPU backend to build. NVIDIA > AMD, Metal on Apple Silicon.

    ``override`` is one of auto/cuda/rocm/metal/cpu. ``auto`` mirrors setup.sh:
    a usable NVIDIA GPU + nvcc -> cuda; else an AMD GPU + hipcc -> rocm; else
    Metal on Apple Silicon; else cpu. A GPU present without its toolkit falls
    back to cpu with a warning (the build would compile but fail at runtime).
    """
    if override != "auto":
        return override

    if host.is_macos and host.is_arm64:
        return "metal"

    if host.has_usable_nvidia or host.has_physical_nvidia:
        if find_nvcc():
            return "cuda"
        log(
            "NVIDIA GPU detected but nvcc (CUDA Toolkit) not found; building CPU. "
            "Install the CUDA Toolkit and re-run for a GPU build."
        )
        return "cpu"

    if host.has_rocm:
        if find_hipcc():
            return "rocm"
        log(
            "AMD GPU detected but hipcc (ROCm) not found; building CPU. "
            "Install ROCm and re-run for a GPU build."
        )
        return "cpu"

    return "cpu"


# ── CMake argument assembly ──────────────────────────────────────────

def base_cmake_args() -> list[str]:
    # Release explicitly (llama.cpp only defaults to it on non-MSVC/Xcode).
    return [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_NATIVE=ON",
    ]


def generator_args() -> list[str]:
    """Pick a CMake generator. Ninja on Unix when present (faster, single-config);
    on Windows let CMake pick the installed Visual Studio (multi-config) unless
    --generator overrides it."""
    if not IS_WINDOWS and which("ninja"):
        return ["-G", "Ninja"]
    return []


def cuda_arch_args(host: HostInfo) -> list[str]:
    archs = normalize_compute_caps(host.compute_caps)
    if not archs:
        return []
    return [f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(archs)}"]


def gpu_fa_quant_args() -> list[str]:
    """TurboQuant FlashAttention kernels for the CUDA / HIP build.

    Default beellama GPU builds compile only the standard q / KVarN-fallback
    FA cache pairs, so the turbo4 KV-cache type fails at *runtime* with
    "CUDA FlashAttention cache pair K=turbo4 V=turbo4 is not compiled in this
    build". GGML_CUDA_FA_HALF_QUANTS=ON compiles the TurboQuant/TCQ pairs
    (K >= V, V at most two tier groups below K) -- the minimal set turbo4 needs.

    The flag is CUDA-named but the HIP backend (GGML_HIP=ON) is built from the
    same hipified ggml-cuda sources, so it applies there too; an unknown -D var
    is only a cmake warning (not an error), so passing it is safe even if a
    given beellama ref ignores it. Metal/CPU FA lives in shaders/CPU kernels
    with no equivalent matrix flag -- see the per-backend note in
    cmake_args_for_backend / install_beellama.

    Controlled by UNSLOTH_BEELLAMA_FA_QUANTS:
      'half' (default) -> -DGGML_CUDA_FA_HALF_QUANTS=ON
      'all'            -> -DGGML_CUDA_FA_ALL_QUANTS=ON  (full K/V matrix, e.g.
                          for asymmetric V-cache overrides HALF can't cover --
                          V higher precision than K, or >2 tiers below K;
                          much longer build, larger binary)
      'off'/'none'/'0' -> no flag (standard build; turbo4 will NOT run)
    """
    mode = os.environ.get("UNSLOTH_BEELLAMA_FA_QUANTS", "half").strip().lower()
    if mode == "all":
        return ["-DGGML_CUDA_FA_ALL_QUANTS=ON"]
    if mode in ("off", "none", "0", "false"):
        return []
    return ["-DGGML_CUDA_FA_HALF_QUANTS=ON"]


def rocm_rocwmma_fattn_args() -> list[str]:
    """Opt-in rocWMMA-accelerated FlashAttention for the HIP build.

    HIP FlashAttention (and thus the turbo4 FA path) may require rocWMMA on AMD.
    But GGML_HIP_ROCWMMA_FATTN=ON needs the rocWMMA library installed AND a
    supported arch (CDNA gfx90a/gfx942, RDNA3 gfx110x, RDNA4 gfx120x); enabling
    it elsewhere breaks the build. So it stays OFF unless the user opts in with
    UNSLOTH_BEELLAMA_HIP_ROCWMMA_FATTN=1.
    """
    if os.environ.get("UNSLOTH_BEELLAMA_HIP_ROCWMMA_FATTN", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return ["-DGGML_HIP_ROCWMMA_FATTN=ON"]
    return []


def fa_quant_mode_from_args(cmake_args: list[str]) -> str | None:
    """Read back the FA-quant mode emitted into ``cmake_args`` (for metadata)."""
    if "-DGGML_CUDA_FA_ALL_QUANTS=ON" in cmake_args:
        return "all"
    if "-DGGML_CUDA_FA_HALF_QUANTS=ON" in cmake_args:
        return "half"
    return None


def rocm_gfx_args(host: HostInfo) -> list[str]:
    if host.rocm_gfx_target:
        return [f"-DGPU_TARGETS={host.rocm_gfx_target}"]
    return []


def configure_env(backend: str) -> dict[str, str]:
    """Environment for the configure/build steps, by backend (mirrors setup.sh)."""
    env = dict(os.environ)
    if backend == "cuda":
        # Allow a host gcc/clang newer than nvcc's whitelist (else a fresh
        # toolkit aborts with "unsupported GNU version").
        prepend = env.get("NVCC_PREPEND_FLAGS", "")
        flag = "-allow-unsupported-compiler"
        env["NVCC_PREPEND_FLAGS"] = f"{prepend} {flag}".strip() if prepend else flag
    elif backend == "rocm":
        hipcc = find_hipcc()
        rocm_root = ""
        hipconfig = which("hipconfig")
        if hipconfig:
            result = capture([hipconfig, "-R"])
            if result.returncode == 0 and result.stdout.strip():
                rocm_root = result.stdout.strip()
        if not rocm_root and hipcc:
            # <root>/bin/hipcc -> <root>
            rocm_root = str(Path(hipcc).resolve().parent.parent)
        if rocm_root:
            env["ROCM_PATH"] = rocm_root
            env["HIP_PATH"] = rocm_root
        if hipconfig:
            result = capture([hipconfig, "-l"])
            if result.returncode == 0 and result.stdout.strip():
                env.setdefault("HIPCXX", str(Path(result.stdout.strip()) / "clang"))
    elif backend == "metal":
        env.setdefault("MACOSX_DEPLOYMENT_TARGET", MACOS_DEPLOYMENT_TARGET)
    return env


def cmake_args_for_backend(backend: str, host: HostInfo) -> list[str]:
    """Full configure args for a backend. Used as the GPU attempt; the CPU
    fallback rebuilds with :func:`cpu_cmake_args`."""
    args = base_cmake_args()
    if which("ccache"):
        args += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ]
    if host.is_macos:
        args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MACOS_DEPLOYMENT_TARGET}")

    if backend == "cuda":
        args += ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_FLAGS=--threads=0"]
        args += cuda_arch_args(host)
        # Compile the TurboQuant FA kernels so turbo4 KV cache works at runtime.
        args += gpu_fa_quant_args()
    elif backend == "rocm":
        args += ["-DGGML_HIP=ON"]
        args += rocm_gfx_args(host)
        # Same TurboQuant FA kernels on AMD (HIP reuses the ggml-cuda sources).
        args += gpu_fa_quant_args()
        args += rocm_rocwmma_fattn_args()
    elif backend == "metal":
        # NOTE: Metal FA lives in compiled-in Metal shaders; there is no cmake
        # FA-quant matrix flag, so turbo4 cannot be enabled here at build time.
        # It works only if beellama's Metal shaders implement it.
        args += [
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_METAL_USE_BF16=ON",
            "-DCMAKE_INSTALL_RPATH=@loader_path",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        ]
    # backend == "cpu": no extra flags. turbo4 is a GPU FlashAttention feature;
    # the CPU backend has no equivalent, so turbo4 is unavailable on CPU builds.
    return args


def cpu_cmake_args(host: HostInfo) -> list[str]:
    """Configure args for a plain CPU build (the fallback when a GPU configure
    or build fails). Metal is explicitly disabled."""
    args = base_cmake_args()
    if which("ccache"):
        args += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
    if host.is_macos:
        args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MACOS_DEPLOYMENT_TARGET}")
        args.append("-DGGML_METAL=OFF")
    return args


# ── Build orchestration ──────────────────────────────────────────────

def cpu_count() -> int:
    return os.cpu_count() or 4


def clone_source(repo: str, ref: str, dest: Path) -> None:
    url = repo if repo.endswith(".git") else f"{repo}.git"
    cmd = ["git", "clone", "--depth", "1"]
    if ref and ref != "latest":
        cmd += ["--branch", ref]
    cmd += [url, str(dest)]
    if run_streaming(cmd) != 0:
        raise BuildError(
            f"git clone of {url} (ref {ref!r}) failed. Check the repo/ref and your network."
        )


def source_commit(src: Path) -> str | None:
    result = capture(["git", "-C", str(src), "rev-parse", "HEAD"])
    if result.returncode == 0:
        return result.stdout.strip() or None
    return None


def configure(
    src: Path,
    build_dir: Path,
    args: list[str],
    *,
    gen_args: list[str],
    generator: str | None,
    env: dict[str, str],
) -> bool:
    _rmtree_robust(build_dir, best_effort = True)
    cmd = ["cmake", "-S", str(src), "-B", str(build_dir)]
    if generator:
        cmd += ["-G", generator]
    else:
        cmd += gen_args
    cmd += ["-Wno-dev", *args]
    return run_streaming(cmd, env = env) == 0


def build_target(build_dir: Path, target: str, *, jobs: int, env: dict[str, str]) -> bool:
    cmd = ["cmake", "--build", str(build_dir), "--config", "Release", "--target", target, "-j", str(jobs)]
    return run_streaming(cmd, env = env) == 0


def build_in(
    src: Path,
    build_dir: Path,
    backend: str,
    host: HostInfo,
    *,
    generator: str | None,
    jobs: int,
) -> tuple[str, list[str]]:
    """Configure + build llama-server in ``build_dir``. Returns
    ``(built_backend, cmake_args)`` -- the backend that actually built
    (``backend`` or ``"cpu"`` after a GPU fallback) and the configure flags used
    for it (recorded in the build metadata). Raises BuildError if even the CPU
    build fails."""
    gen_args = generator_args()
    gpu_args = cmake_args_for_backend(backend, host)
    gpu_env = configure_env(backend)

    def cpu_fallback(reason: str) -> tuple[str, list[str]]:
        log(f"{backend.upper()} {reason}; retrying CPU build...")
        cpu_args = cpu_cmake_args(host)
        cpu_env = configure_env("cpu")
        if not configure(src, build_dir, cpu_args, gen_args = gen_args, generator = generator, env = cpu_env):
            raise BuildError("CPU cmake configure failed.")
        if not build_target(build_dir, REQUIRED_TARGET, jobs = jobs, env = cpu_env):
            raise BuildError("CPU build of llama-server failed.")
        return "cpu", cpu_args

    log(f"Configuring ({backend}) ...")
    if not configure(src, build_dir, gpu_args, gen_args = gen_args, generator = generator, env = gpu_env):
        if backend != "cpu":
            return cpu_fallback("configure failed")
        raise BuildError("cmake configure failed.")

    log(f"Building {REQUIRED_TARGET} ({backend}) with -j{jobs} ...")
    if not build_target(build_dir, REQUIRED_TARGET, jobs = jobs, env = gpu_env):
        if backend != "cpu":
            return cpu_fallback("build failed")
        raise BuildError(f"build of {REQUIRED_TARGET} failed.")

    # Best-effort extras: a fork at a given ref may not ship every target.
    for target in OPTIONAL_TARGETS:
        if build_target(build_dir, target, jobs = jobs, env = gpu_env):
            log(f"built optional target {target}")
        else:
            log(f"optional target {target} not built (skipped)")
    return backend, gpu_args


# ── Install layout / swap ────────────────────────────────────────────

def server_binary_candidates(root: Path) -> list[Path]:
    """Where llama-server may land for a cmake build under ``root``.

    Windows multi-config generators (Visual Studio) emit build/bin/Release/;
    single-config (Ninja) emits build/bin/. The inference backend searches both,
    so accept both here too.
    """
    build_bin = root / "build" / "bin"
    if IS_WINDOWS:
        return [build_bin / "Release" / f"{REQUIRED_TARGET}.exe", build_bin / f"{REQUIRED_TARGET}.exe"]
    return [build_bin / REQUIRED_TARGET]


def find_built_server(root: Path) -> Path | None:
    return next((p for p in server_binary_candidates(root) if p.exists()), None)


def looks_like_llama_install(path: Path) -> bool:
    """Heuristic: is ``path`` an existing (Studio or hand-built) llama.cpp dir we
    may replace? Guards the rm -rf so we never delete an unrelated directory."""
    if (path / STUDIO_OWNED_MARKER).exists():
        return True
    markers = [
        path / ".git",
        path / "CMakeLists.txt",
        path / "build" / "bin",
        path / REQUIRED_TARGET,
        path / f"{REQUIRED_TARGET}.exe",
        path / "convert_hf_to_gguf.py",
    ]
    return any(m.exists() for m in markers)


def assert_replaceable(install_dir: Path) -> None:
    if not install_dir.exists():
        return
    if any(install_dir.iterdir()) and not looks_like_llama_install(install_dir):
        raise BuildError(
            f"refusing to replace {install_dir}: it is non-empty and does not look "
            "like a llama.cpp install (no .unsloth-studio-owned marker, .git, "
            "CMakeLists.txt, or build/bin). Move it aside and re-run."
        )


def finalize_install(staging: Path, install_dir: Path, *, host: HostInfo) -> None:
    """Atomically swap the freshly built ``staging`` tree into ``install_dir``.

    Only runs after a successful build, so a failed build never disturbs an
    existing install. Mirrors setup.sh: replace the dir, drop the owned marker,
    and symlink llama-quantize (+ diffusion server) into the root on Unix.
    """
    assert_replaceable(install_dir)
    install_dir.parent.mkdir(parents = True, exist_ok = True)

    if install_dir.exists():
        try:
            _rmtree_robust(install_dir)
        except OSError as exc:
            raise BuildError(
                f"could not remove existing {install_dir} ({exc}). A running "
                "llama-server may be holding files open -- stop it and re-run."
            ) from exc

    # staging is a sibling of install_dir (same filesystem) -> atomic rename.
    os.replace(staging, install_dir)

    # Mark as Studio-owned so future installer runs can safely replace it.
    try:
        (install_dir / STUDIO_OWNED_MARKER).touch()
    except OSError:
        pass

    if not host.is_windows:
        _link_into_root(install_dir, "llama-quantize")
        _link_into_root(install_dir, "llama-diffusion-gemma-visual-server")


def _link_into_root(install_dir: Path, name: str) -> None:
    """Symlink build/bin/<name> to <install_dir>/<name> (Unix), mirroring setup.sh.
    No-op when the binary was not built."""
    target = install_dir / "build" / "bin" / name
    if not target.exists():
        return
    link = install_dir / name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(os.path.join("build", "bin", name), link)
    except OSError as exc:
        log(f"could not symlink {name} into install root ({exc}); leaving it in build/bin")


def write_metadata(
    install_dir: Path,
    *,
    repo: str,
    ref: str,
    commit: str | None,
    backend: str,
    cmake_args: list[str],
) -> None:
    """Record what was built so a later "why doesn't turbo4 work?" is one file
    read away. ``fa_quants`` is the decisive field: None means the TurboQuant FA
    kernels were NOT compiled (turbo4 will fail at runtime on a GPU build)."""
    payload = {
        "source": "beellama.cpp",
        "repo": repo,
        "ref": ref,
        "commit": commit,
        "backend": backend,
        # half / all / None. None on a cuda/rocm build => turbo4 won't run; on
        # metal/cpu turbo4 is unavailable regardless (no FA-quant build flag).
        "fa_quants": fa_quant_mode_from_args(cmake_args),
        "turbo4_supported": backend in ("cuda", "rocm") and fa_quant_mode_from_args(cmake_args) is not None,
        "cmake_args": cmake_args,
        "built_unix": int(time.time()),
        "platform": platform.platform(),
    }
    try:
        (install_dir / METADATA_FILENAME).write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")
    except OSError as exc:
        log(f"could not write {METADATA_FILENAME} ({exc})")


# ── Prerequisites ────────────────────────────────────────────────────

def check_prerequisites() -> None:
    missing = [tool for tool in ("git", "cmake") if not which(tool)]
    if missing:
        raise BuildError(
            f"missing required build tool(s): {', '.join(missing)}. Install them and re-run. "
            + (
                "On Windows, also install Visual Studio 2022 with the "
                '"Desktop development with C++" workload.'
                if IS_WINDOWS
                else "Also ensure a C/C++ compiler (gcc/clang) is installed."
            )
        )


# ── Entry point ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Detect host hardware and build beellama.cpp from source for Unsloth Studio."
    )
    parser.add_argument(
        "--install-dir",
        required = True,
        help = "Target ~/.unsloth/llama.cpp directory (same dir install_llama_prebuilt.py installs into).",
    )
    parser.add_argument("--repo", default = DEFAULT_REPO, help = "beellama.cpp git repository URL.")
    parser.add_argument("--ref", default = DEFAULT_REF, help = "Git branch/tag/commit to build (default v0.3.2).")
    parser.add_argument(
        "--backend",
        choices = ["auto", "cuda", "rocm", "metal", "cpu"],
        default = "auto",
        help = "GPU backend. 'auto' detects from the host (NVIDIA > AMD > Metal > CPU).",
    )
    parser.add_argument("--jobs", type = int, default = cpu_count(), help = "Parallel build jobs (default: CPU count).")
    parser.add_argument(
        "--generator",
        default = None,
        help = "Override the CMake generator (e.g. 'Visual Studio 17 2022', 'Ninja').",
    )
    parser.add_argument(
        "--keep-build-on-failure",
        action = "store_true",
        help = "Keep the temporary build directory if the build fails (for debugging).",
    )
    return parser.parse_args()


def install_beellama(
    *,
    install_dir: Path,
    repo: str,
    ref: str,
    backend_override: str,
    jobs: int,
    generator: str | None,
    keep_build_on_failure: bool,
) -> None:
    check_prerequisites()

    host = detect_host()
    backend = resolve_backend(host, backend_override)
    log(
        f"host: {host.system}/{host.machine} | backend: {backend} | repo: {repo} | ref: {ref}"
    )
    # turbo4/TurboQuant is a CUDA/HIP FlashAttention feature; warn up front that
    # a non-GPU build cannot provide it no matter what cache type is selected.
    if backend in ("metal", "cpu"):
        log(
            f"note: the turbo4 / TurboQuant KV cache is a CUDA/HIP feature; the "
            f"{backend} build cannot provide it (use f16/bf16/q8_0/... instead)."
        )

    # Fail fast on an un-replaceable target before a long compile (re-checked at
    # swap time for TOCTOU).
    assert_replaceable(install_dir)

    staging = install_dir.parent / f"{install_dir.name}.build.{os.getpid()}"
    _rmtree_robust(staging, best_effort = True)

    built_backend = backend
    try:
        clone_source(repo, ref, staging)
        commit = source_commit(staging)
        built_backend, used_cmake_args = build_in(
            staging,
            staging / "build",
            backend,
            host,
            generator = generator,
            jobs = jobs,
        )

        if find_built_server(staging) is None:
            raise BuildError(
                f"build reported success but {REQUIRED_TARGET} not found under "
                f"{staging / 'build' / 'bin'}."
            )

        finalize_install(staging, install_dir, host = host)
        write_metadata(
            install_dir,
            repo = repo,
            ref = ref,
            commit = commit,
            backend = built_backend,
            cmake_args = used_cmake_args,
        )
    except BaseException:
        if not keep_build_on_failure:
            _rmtree_robust(staging, best_effort = True)
        else:
            log(f"keeping build dir for debugging: {staging}")
        raise

    final_server = find_built_server(install_dir) or server_binary_candidates(install_dir)[0]
    log(f"done: {built_backend} build of beellama.cpp ({ref})")
    log(f"llama-server: {final_server}")
    if built_backend in ("cuda", "rocm"):
        fa_mode = fa_quant_mode_from_args(used_cmake_args)
        if fa_mode:
            log(f"turbo4: enabled (GGML_CUDA_FA_{fa_mode.upper()}_QUANTS)")
        else:
            log("turbo4: NOT compiled -- set UNSLOTH_BEELLAMA_FA_QUANTS=half and rebuild")


def main() -> int:
    args = parse_args()
    install_beellama(
        install_dir = Path(args.install_dir).expanduser().resolve(),
        repo = args.repo,
        ref = args.ref,
        backend_override = args.backend,
        jobs = max(1, args.jobs),
        generator = args.generator,
        keep_build_on_failure = args.keep_build_on_failure,
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BuildError as exc:
        log(f"error: {exc}")
        raise SystemExit(EXIT_ERROR)
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(EXIT_ERROR)
    except Exception as exc:  # pragma: no cover - last-resort diagnostics
        log(f"unexpected error: {exc}")
        raise SystemExit(EXIT_ERROR)

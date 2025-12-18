from __future__ import annotations

import contextlib
import ctypes
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from setuptools.command.install import install

_FORCE_RUNTIME_ENV = "UNSLOTH_FORCE_RUNTIME"
_FORCE_RUNTIME_VERSION_ENV = "UNSLOTH_FORCE_RUNTIME_VERSION"
_FORCE_ROCM_HOME_ENV = "UNSLOTH_FORCE_ROCM_HOME"
_FORCE_CUDA_HOME_ENV = "UNSLOTH_FORCE_CUDA_HOME"
_ROCM_BOOTSTRAP_ENV = "UNSLOTH_BOOTSTRAP_ROCM"
_ROCM_TORCH_ARGS_ENV = "UNSLOTH_BOOTSTRAP_TORCH_ARGS"
_BOOTSTRAP_COMPLETED = False
_INSTINCT_ARCH = ("gfx942", "gfx90a")
_RADEON_ARCH = ("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")
_REQUIRED_TRANSFORMERS_FOR_AMD = "4.56.2"

__all__ = [
    "RuntimeInfo",
    "detect_runtime",
    "get_rocm_version",
    "get_nvcc_cuda_version",
    "compute_version_string",
    "RocmExtraInstallCommand",
]


@dataclass
class RuntimeInfo:
    has_torch: bool = False
    has_cuda: bool = False
    has_hip: bool = False
    torch_module: Any = None
    cuda_home: Optional[str] = None
    rocm_home: Optional[str] = None


def _log(message: str) -> None:
    print(f"Unsloth: {message}")


def _env_var_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


def _get_bootstrap_python() -> str:
    python_exec = os.getenv("UNSLOTH_BOOTSTRAP_PYTHON") or sys.executable
    if "pip-build-env-" in python_exec:
        _log(
            "Bootstrapping inside pip build isolation. Set UNSLOTH_BOOTSTRAP_PYTHON "
            "to your target interpreter or rerun pip with --no-build-isolation so "
            "flash-attention/bitsandbytes install into the desired environment."
        )
    else:
        _log(f"Using Python executable {python_exec} for ROCm bootstrap installs.")
    return python_exec


def _pip_install(
    python_exec: str, args: Sequence[str], env: Optional[dict[str, str]] = None
) -> None:
    subprocess.check_call([python_exec, "-m", "pip", *args], env = env)


def detect_runtime() -> RuntimeInfo:
    runtime = RuntimeInfo()
    forced_runtime = os.getenv(_FORCE_RUNTIME_ENV)
    if forced_runtime:
        forced_runtime = forced_runtime.lower()
        runtime.has_cuda = forced_runtime in {"cuda", "nvidia"}
        runtime.has_hip = forced_runtime in {"hip", "rocm", "amd"}
        runtime.cuda_home = os.getenv(_FORCE_CUDA_HOME_ENV)
        runtime.rocm_home = os.getenv(_FORCE_ROCM_HOME_ENV)
        _log(
            "Forcing runtime to "
            f"{'CUDA' if runtime.has_cuda else 'ROCm'} via {_FORCE_RUNTIME_ENV}."
        )
        _try_populate_torch(runtime)
        return runtime
    if not _try_populate_torch(runtime):
        runtime.has_cuda = True  # default to CUDA builds when torch is absent
        _log(
            "Torch not found during build; assuming CUDA runtime. "
            f"Set {_FORCE_RUNTIME_ENV}=hip to override."
        )
    return runtime


def _try_populate_torch(runtime: RuntimeInfo) -> bool:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
    except Exception as exc:  # pragma: no cover - defensive logging
        _log(f"Importing torch failed with error = {exc}.")
        return False
    runtime.has_torch = True
    runtime.torch_module = torch
    torch_cuda = bool(getattr(torch.version, "cuda", None))
    torch_hip = bool(getattr(torch.version, "hip", None))
    runtime.has_cuda = runtime.has_cuda or torch_cuda
    runtime.has_hip = runtime.has_hip or torch_hip
    if runtime.has_cuda and not runtime.cuda_home:
        runtime.cuda_home = CUDA_HOME
    if runtime.has_hip and not runtime.rocm_home:
        runtime.rocm_home = ROCM_HOME
    _log(f"Imported torch with CUDA_HOME = {CUDA_HOME}.")
    _log(f"Imported torch with ROCM_HOME = {ROCM_HOME}.")
    return True


def get_nvcc_cuda_version() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        output = subprocess.check_output([nvcc, "--version"], text = True)
    except Exception:
        return None
    match = re.search(r"release (\d+\.\d+)", output)
    if not match:
        return None
    return match.group(1)


def get_rocm_version(runtime: Optional[RuntimeInfo] = None) -> Optional[str]:
    runtime = runtime or detect_runtime()
    if not runtime.rocm_home:
        return None
    try:
        librocm_core_file = Path(runtime.rocm_home) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(str(librocm_core_file))
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = ctypes.c_uint32
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()
        status = get_rocm_core_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        if status == 0:
            version = f"{major.value}.{minor.value}.{patch.value}"
            _log(f"Detected ROCm version from librocm-core.so: {version}")
            return version
    except Exception:  # pragma: no cover - best effort detection
        return None
    return None


def _normalize_cuda_string(cuda_version: Optional[str]) -> Optional[str]:
    if not cuda_version:
        return None
    digits = cuda_version.replace(".", "")[:3]
    return digits if digits else None


def _normalize_rocm_string(rocm_version: Optional[str]) -> Optional[str]:
    if not rocm_version:
        return None
    digits = rocm_version.replace(".", "")[:3]
    return digits if digits else None


def _detect_rocm_arch(runtime: RuntimeInfo) -> Optional[str]:
    rocm_arch = os.environ.get("ROCM_ARCH") or os.environ.get("PYTORCH_ROCM_ARCH")
    if rocm_arch:
        _log(f"Using ROCm arch from environment: {rocm_arch}")
        return rocm_arch
    if not runtime.has_hip:
        return None
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output = True, text = True, check = False
        )
    except Exception as exc:
        _log(f"Could not detect ROCm GPU architecture: {exc}")
        return None
    match = re.search(r"Name:\s+gfx([a-zA-Z\d]+)", result.stdout)
    if not match:
        _log("rocminfo did not report a GFX name; set ROCM_ARCH manually if needed")
        return None
    rocm_arch = f"gfx{match.group(1)}"
    _log(f"Detected ROCm arch via rocminfo: {rocm_arch}")
    return rocm_arch


def _log_installed_package(package: str) -> None:
    candidates = [package]
    if "_" in package:
        candidates.append(package.replace("_", "-"))
    if "-" in package:
        candidates.append(package.replace("-", "_"))
    for dist_name in candidates:
        try:
            version = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
        else:
            _log(f"Detected pre-installed {package} version {version}.")
            return
    if importlib.util.find_spec(package) is None:
        _log(f"Package {package} not found prior to bootstrap.")
        return
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        _log(f"Detected pre-installed {package} version {version}.")
    except Exception as exc:  # pragma: no cover - defensive logging
        _log(f"Detected {package} but failed to read version: {exc}")


def _ensure_transformers_for_amd(runtime: RuntimeInfo, python_exec: str) -> None:
    if not runtime.has_hip:
        return
    try:
        import transformers

        current = transformers.__version__
    except Exception:
        current = None
    if current and current == _REQUIRED_TRANSFORMERS_FOR_AMD:
        return

    skip_env = os.getenv("UNSLOTH_SKIP_AMD_TRANSFORMERS_PIN", "0") == "1"
    if skip_env:
        raise RuntimeError(
            "AMD/ROCm support requires transformers==4.56.2. "
            'Set UNSLOTH_SKIP_AMD_TRANSFORMERS_PIN=0 or install manually via `pip install -U "transformers==4.56.2"`.'
        )

    _log(
        "Installing transformers==4.56.2 for AMD/ROCm compatibility "
        "(use UNSLOTH_SKIP_AMD_TRANSFORMERS_PIN=1 to skip)."
    )
    _pip_install(
        python_exec,
        [
            "install",
            "-U",
            f"transformers=={_REQUIRED_TRANSFORMERS_FOR_AMD}",
        ],
    )
    importlib.invalidate_caches()
    if "transformers" in sys.modules:
        del sys.modules["transformers"]
    import transformers  # type: ignore

    if transformers.__version__ != _REQUIRED_TRANSFORMERS_FOR_AMD:
        raise RuntimeError(
            "Failed to enforce transformers==4.56.2 automatically. "
            "Please install it manually."
        )


def _log_rocm_summary(runtime: RuntimeInfo) -> None:
    _log(
        "Runtime summary: "
        f"torch_present={runtime.has_torch} "
        f"has_cuda={runtime.has_cuda} has_hip={runtime.has_hip}"
    )
    if runtime.rocm_home:
        _log(f"ROCM_HOME set to: {runtime.rocm_home}")
    rocm_version = get_rocm_version(runtime)
    if rocm_version:
        _log(f"ROCm version detected: {rocm_version}")
    arch = _detect_rocm_arch(runtime)
    if arch:
        _log(f"Active ROCm arch: {arch}")
    _log_installed_package("flash_attn")
    _log_installed_package("bitsandbytes")


def _ensure_rocm_bootstrap(runtime: RuntimeInfo) -> bool:
    global _BOOTSTRAP_COMPLETED
    if not _env_var_truthy(_ROCM_BOOTSTRAP_ENV):
        return False
    if _BOOTSTRAP_COMPLETED:
        return True
    _BOOTSTRAP_COMPLETED = True
    _perform_rocm_bootstrap(runtime)
    return True


def _perform_rocm_bootstrap(runtime: RuntimeInfo) -> None:
    python_exec = _get_bootstrap_python()
    _ensure_transformers_for_amd(runtime, python_exec)
    thirdparty_dir = Path("thirdparties")
    if thirdparty_dir.exists():
        shutil.rmtree(thirdparty_dir)
    thirdparty_dir.mkdir()
    with _pushd(thirdparty_dir):
        _install_build_prereqs(python_exec)
        _ensure_rocm_torch(runtime, python_exec)
        rocm_arch = _detect_rocm_arch(runtime)
        if not rocm_arch:
            _log("Skipping ROCm extra install, unable to infer architecture.")
            return
        _maybe_install_flash_attention(rocm_arch, python_exec)
        _maybe_install_bitsandbytes(python_exec)


def _install_build_prereqs(python_exec: str) -> None:
    _pip_install(python_exec, ["install", "cmake>=3.26"])
    _pip_install(python_exec, ["install", "ninja"])


def _ensure_rocm_torch(runtime: RuntimeInfo, python_exec: str) -> None:
    if runtime.has_torch and runtime.has_hip:
        torch_version = getattr(
            getattr(runtime.torch_module, "version", None), "__version__", None
        )
        if torch_version:
            _log(
                f"Detected ROCm-enabled torch {torch_version}, skipping torch bootstrap."
            )
        else:
            _log("Detected ROCm-enabled torch, skipping torch bootstrap.")
        return
    torch_args_raw = os.getenv(_ROCM_TORCH_ARGS_ENV)
    if not torch_args_raw:
        raise RuntimeError(
            "flash-attention bootstrap requires torch, but no ROCm torch installation was detected. "
            "Install a ROCm-enabled torch wheel beforehand or set "
            f'{_ROCM_TORCH_ARGS_ENV} to the pip arguments needed to install it (for example, "torch==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1").'
        )
    pip_args = shlex.split(torch_args_raw)
    if not pip_args:
        raise RuntimeError(
            f"{_ROCM_TORCH_ARGS_ENV} was provided but empty after parsing."
        )
    _log("Installing ROCm-enabled torch via pip arguments: " + " ".join(pip_args))
    _pip_install(python_exec, ["install", *pip_args])
    refreshed = detect_runtime()
    if not refreshed.has_torch or not refreshed.has_hip:
        raise RuntimeError(
            "Torch installation via ROCm bootstrap completed but torch still does not report ROCm support. "
            "Double-check the pip arguments passed through UNSLOTH_BOOTSTRAP_TORCH_ARGS."
        )
    runtime.has_torch = refreshed.has_torch
    runtime.has_hip = refreshed.has_hip
    runtime.torch_module = refreshed.torch_module
    runtime.rocm_home = refreshed.rocm_home


"""
def _maybe_install_flash_attention(rocm_arch: Optional[str], python_exec: str) -> None:
    if importlib.util.find_spec("flash_attn") is not None:
        _log("flash-attention already present, skipping bootstrap clone.")
        return
    _log("Installing flash-attention...")
    subprocess.check_call([
        "git",
        "clone",
        "--recursive",
        "https://github.com/ROCm/flash-attention.git",
        "flash-attention",
    ])
    with _pushd(Path("flash-attention")):
        if rocm_arch in _RADEON_ARCH:
            subprocess.check_call(["git", "checkout", "main_perf"])
            env = os.environ.copy()
            env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            _log(f"flash-attention commit {commit} (main_perf branch)")
            subprocess.check_call([python_exec, "setup.py", "install"], env=env)
            return
        jobs = max((os.cpu_count() or 2) - 1, 1)
        env = os.environ.copy()
        env["MAX_JOBS"] = str(jobs)
        env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        _log(f"flash-attention commit {commit} (default branch)")
        _pip_install(python_exec, ["install", "-v", "--no-build-isolation", "."], env=env)
"""
"""
def _maybe_install_bitsandbytes(python_exec: str) -> None:
    if importlib.util.find_spec("bitsandbytes") is not None:
        _log("bitsandbytes already present, skipping bootstrap clone.")
        return
    _log("Installing bitsandbytes...")
    subprocess.check_call([
        "git",
        "clone",
        "--recurse-submodules",
        "https://github.com/ROCm/bitsandbytes",
        "bitsandbytes",
    ])
    with _pushd(Path("bitsandbytes")):
        subprocess.check_call(["git", "checkout", "rocm_enabled_multi_backend"])
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        _log(f"bitsandbytes commit {commit} (rocm_enabled_multi_backend branch)")
        _pip_install(python_exec, ["install", "-r", "requirements-dev.txt"])
        subprocess.check_call(["cmake", "-DCOMPUTE_BACKEND=hip", "-S", "."])
        subprocess.check_call(["make"])
        env = os.environ.copy()
        env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
        _pip_install(python_exec, ["install", "--no-build-isolation", "."], env=env)
"""


def _maybe_install_flash_attention(rocm_arch: Optional[str], python_exec: str) -> None:
    # 注意：这里建议不要在前面就用 find_spec 判定“已经装好”
    # 因为你之前环境里就有一个坏的 CUDA 版 flash_attn，find_spec 会误判。
    # 可以换成：如果已经有 *且能 import* 并通过简单自检，再跳过安装。

    try:
        spec = importlib.util.find_spec("flash_attn")
        if spec is not None:
            import flash_attn
            from importlib import reload

            reload(flash_attn)  # 避免残留
            # 可以在这里加一些简单自检，比如检查 __version__ 或 backend
            return
    except Exception:
        # 有 flash_attn 但 import 失败，当成“没装好”，继续重新安装
        pass

    _log("Installing flash-attention...")
    subprocess.check_call(
        [
            "git",
            "clone",
            "--recursive",
            "https://github.com/ROCm/flash-attention.git",
            "flash-attention",
        ]
    )

    with _pushd(Path("flash-attention")):
        env = os.environ.copy()
        if rocm_arch in _RADEON_ARCH:
            subprocess.check_call(["git", "checkout", "main_perf"])
            env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text = True
            ).strip()
            _log(f"flash-attention commit {commit} (main_perf branch)")
            # 这里用 pip，和你手动一致
            _pip_install(
                python_exec,
                ["install", "-v", "--no-build-isolation", "."],
                env = env,
            )
            return

        # 非 RADEON_ARCH，保持原逻辑
        jobs = max((os.cpu_count() or 2) - 1, 1)
        env["MAX_JOBS"] = str(jobs)
        env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text = True
        ).strip()
        _log(f"flash-attention commit {commit} (default branch)")
        _pip_install(
            python_exec,
            ["install", "-v", "--no-build-isolation", "."],
            env = env,
        )


def _maybe_install_bitsandbytes(python_exec: str) -> None:
    if importlib.util.find_spec("bitsandbytes") is not None:
        _log("bitsandbytes already present, skipping bootstrap clone.")
        return

    _log("Installing bitsandbytes 0.48.2 from official repository...")

    subprocess.check_call(
        [
            "git",
            "clone",
            "--recurse-submodules",
            "https://github.com/bitsandbytes-foundation/bitsandbytes.git",
            "bitsandbytes",
        ]
    )

    with _pushd(Path("bitsandbytes")):
        subprocess.check_call(["git", "checkout", "main"])

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text = True,
        ).strip()
        _log(f"bitsandbytes commit {commit} (tag 0.48.2)")

        # 可选：开发依赖
        if Path("requirements-dev.txt").exists():
            _log("Installing bitsandbytes dev requirements...")
            _pip_install(python_exec, ["install", "-r", "requirements-dev.txt"])

        # 关键：安装构建后端 scikit-build-core
        _log(
            "Ensuring scikit-build-core is installed for bitsandbytes build backend..."
        )
        _pip_install(python_exec, ["install", "scikit-build-core"])

        _log("Configuring bitsandbytes with CMake (ROCm / HIP backend)...")
        subprocess.check_call(
            ["cmake", "-DCOMPUTE_BACKEND=hip", "-DBNB_ROCM_ARCH=gfx1201", "-S", "."]
        )

        _log("Building bitsandbytes via make...")
        subprocess.check_call(["make"])

        _log("Installing bitsandbytes 0.48.2 into Python environment (editable)...")
        env = os.environ.copy()
        # 既然用 PEP517 后端，这里可以不强制关闭 build isolation；保守起见也可以保留
        # env.setdefault("PIP_NO_BUILD_ISOLATION", "1")

        _pip_install(
            python_exec,
            ["install", "-e", "."],  # 不再加 --no-build-isolation
            env = env,
        )


def compute_version_string(base_version: str) -> str:
    runtime = detect_runtime()
    version = base_version
    sep = "+" if "+" not in version else "."
    forced_version = os.getenv(_FORCE_RUNTIME_VERSION_ENV)
    if runtime.has_cuda:
        cuda_version = forced_version or get_nvcc_cuda_version()
        if not cuda_version and runtime.torch_module is not None:
            cuda_version = getattr(runtime.torch_module.version, "cuda", None)
        cuda_digits = _normalize_cuda_string(cuda_version)
        if cuda_digits:
            version = f"{version}{sep}cu{cuda_digits}"
            _log(f"Computed package version suffix: cu{cuda_digits}")
        return version
    if runtime.has_hip:
        rocm_version = forced_version or get_rocm_version(runtime)
        if not rocm_version and runtime.torch_module is not None:
            rocm_version = getattr(runtime.torch_module.version, "hip", None)
            if rocm_version:
                _log(f"Using torch.version.hip={rocm_version} for version tagging")
        _log_rocm_summary(runtime)
        _ensure_rocm_bootstrap(runtime)
        rocm_digits = _normalize_rocm_string(rocm_version)
        if rocm_digits:
            version = f"{version}{sep}rocm{rocm_digits}"
            _log(f"Computed package version suffix: rocm{rocm_digits}")
        return version
    _log(
        "Unknown runtime environment detected, defaulting to base version. "
        f"Provide {_FORCE_RUNTIME_ENV} and {_FORCE_RUNTIME_VERSION_ENV} to override."
    )
    return version


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RocmExtraInstallCommand(install):  # type: ignore[misc]
    def run(self):
        runtime = detect_runtime()
        try:
            ran_bootstrap = _ensure_rocm_bootstrap(runtime)
        except Exception as exc:  # pragma: no cover - bootstrap best effort
            _log(f"ROCm bootstrap failed: {exc}")
        else:
            if not ran_bootstrap:
                _log(
                    "UNSLOTH_BOOTSTRAP_ROCM not set; skipping ROCm dependency bootstrap during install."
                )
        super().run()

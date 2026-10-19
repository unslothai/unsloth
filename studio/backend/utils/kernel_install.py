# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kernel install pieces shared by Studio setup, the training worker, the SSM runtime and
`unsloth install-kernels [names]`.

Imports only stdlib, wheel_utils and child_stdio so setup and torch-less hosts can import it.
The CLI is wheel-only (never a source build) and uses --no-deps so torch is never touched.
"""

from __future__ import annotations

import argparse
import errno
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from utils.child_stdio import utf8_child_env
from utils.wheel_utils import (
    CAUSAL_CONV1D_PACKAGE_VERSION,
    CAUSAL_CONV1D_RELEASE_BASE_URL,
    CAUSAL_CONV1D_RELEASE_TAG,
    MAMBA_SSM_PACKAGE_VERSION,
    MAMBA_SSM_RELEASE_BASE_URL,
    MAMBA_SSM_RELEASE_TAG,
    direct_wheel_url,
    flash_attn_wheel_url,
    install_wheel,
    package_installs_offline_env,
    probe_torch_wheel_env,
    redact_url_credentials,
    url_exists,
    xformers_wheel_url,
)


@dataclass(frozen = True)
class PinnedKernel:
    """A kernel installed from a pinned release: prebuilt wheel first, else `pypi_name==version`."""

    import_name: str
    display_name: str
    pypi_name: str
    package_version: str
    release_tag: str
    release_base_url: str

    def wheel_url(self, env: dict[str, str] | None) -> str | None:
        return direct_wheel_url(
            filename_prefix = self.import_name,
            package_version = self.package_version,
            release_tag = self.release_tag,
            release_base_url = self.release_base_url,
            env = env,
        )


CAUSAL_CONV1D = PinnedKernel(
    import_name = "causal_conv1d",
    display_name = "causal-conv1d",
    pypi_name = "causal-conv1d",
    package_version = CAUSAL_CONV1D_PACKAGE_VERSION,
    release_tag = CAUSAL_CONV1D_RELEASE_TAG,
    release_base_url = CAUSAL_CONV1D_RELEASE_BASE_URL,
)
MAMBA_SSM = PinnedKernel(
    import_name = "mamba_ssm",
    display_name = "mamba-ssm",
    pypi_name = "mamba-ssm",
    package_version = MAMBA_SSM_PACKAGE_VERSION,
    release_tag = MAMBA_SSM_RELEASE_TAG,
    release_base_url = MAMBA_SSM_RELEASE_BASE_URL,
)


def install_prebuilt(
    wheel_url: str,
    *,
    install: Callable[..., list[tuple[str, Any]]],
    verify: Callable[[], bool],
    on_failed: Callable[[str, Any], None],
    **install_kwargs: Any,
) -> str:
    """Install and verify a prebuilt wheel: "installed", "rejected" (installed, `verify()` failed)
    or "failed". `install_kwargs` are forwarded as given so each caller keeps its own installer flags."""
    for installer, result in install(wheel_url, python_executable = sys.executable, **install_kwargs):
        if getattr(result, "returncode", 1) == 0:
            # A wheel can install yet not load (CUDA/ABI or arch mismatch), so the exit code is no proof.
            return "installed" if verify() else "rejected"
        on_failed(installer, result)
    return "failed"


def hipcc_gcc_install_dir() -> str | None:
    """Highest-numbered ``/usr/lib/gcc/x86_64-linux-gnu/<N>`` that has BOTH the gcc runtime dir AND
    ``/usr/include/c++/<N>`` headers, or None. Ubuntu 24.04 ships gcc-14 runtime but not
    ``/usr/include/c++/14``; ROCm clang-20 picks the highest runtime dir, finds no ``<cstdlib>``,
    and the HIP build fails, hence ``--gcc-install-dir``. Mirrors studio/setup.sh (PR #5301)."""
    if not sys.platform.startswith("linux") or platform.machine().lower() != "x86_64":
        return None
    for ver in (14, 13, 12, 11):
        if os.path.isdir(f"/usr/lib/gcc/x86_64-linux-gnu/{ver}/include") and os.path.isdir(
            f"/usr/include/c++/{ver}"
        ):
            return f"/usr/lib/gcc/x86_64-linux-gnu/{ver}"
    return None


def source_build_command(spec: str, *, use_uv: bool, is_hip: bool, reinstall: bool) -> list[str]:
    """`--no-build-isolation --no-deps` install of *spec* against the resident torch; the cache is
    skipped on HIP (uv) or always (pip) so stale partial build artifacts are never reused."""
    if use_uv:
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-build-isolation",
            "--no-deps",
        ]
        if reinstall:
            cmd.append("--reinstall")
        if is_hip:
            cmd.append("--no-cache")
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
        ]
        if reinstall:
            cmd.append("--force-reinstall")
    cmd.append(spec)
    return cmd


# Overall deadline for a ROCm / HIP source build (10-30 min).
SOURCE_BUILD_TIMEOUT = 1800


def _cuda_source_build_timeout() -> int:
    """UNSLOTH_SOURCE_BUILD_TIMEOUT (seconds, > 0) when set, else 7200."""
    raw = os.environ.get("UNSLOTH_SOURCE_BUILD_TIMEOUT", "").strip()
    try:
        value = int(float(raw)) if raw else 0
    except ValueError:
        value = 0
    return value if value > 0 else 7200


# Overall deadline for a CUDA source build. Far looser than HIP's: before the shared deadline a
# CUDA build had none, and a slow one (causal-conv1d on aarch64 can compile for over an hour) must
# not be aborted; this only stops a build that is truly stuck. Read once at import, so the lock
# wait below covers it.
SOURCE_BUILD_TIMEOUT_CUDA = _cuda_source_build_timeout()
# The real runner, captured at import: a caller handing in anything else (a test double, or
# subprocess.run patched over) gets exactly that called.
_SUBPROCESS_RUN = subprocess.run


def run_process_tree(
    cmd: list[str],
    *,
    run: Callable[..., Any] = _SUBPROCESS_RUN,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """``run(cmd, **kwargs)``, but when *run* is the real ``subprocess.run`` a ``timeout`` stops the
    whole process tree (pip / uv and the nvcc / hipcc / ninja they start), not just its leader,
    which ``subprocess.run`` alone would leave compiling. Raises ``TimeoutExpired`` as run does."""
    timeout = kwargs.pop("timeout", None)
    if run is not _SUBPROCESS_RUN or timeout is None:
        if timeout is not None:
            kwargs["timeout"] = timeout
        return run(cmd, **kwargs)
    if os.name == "nt":
        kwargs.setdefault("creationflags", getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    with subprocess.Popen(cmd, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout = timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            process.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout)
        except BaseException:
            _kill_process_tree(process)
            raise
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def _kill_process_tree(process: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 60,
            )
        else:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass


# Long enough for one source build (the longer of the HIP and CUDA deadlines) under another
# process's lock, then give up rather than block.
KERNEL_LOCK_WAIT = max(SOURCE_BUILD_TIMEOUT, SOURCE_BUILD_TIMEOUT_CUDA) + 600
_LOCK_CONTENDED = frozenset(
    code
    for code in (
        getattr(errno, "EACCES", None),
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EWOULDBLOCK", None),
        getattr(errno, "EDEADLK", None),
        getattr(errno, "EDEADLOCK", None),
    )
    if code is not None
)


# How long an optional acceleration (causal-conv1d, an optional mamba-ssm wheel, flash-attn) waits
# for another process's install of it before the load goes on without it.
OPTIONAL_KERNEL_LOCK_WAIT = 120


def _kernel_lock_path(name: str) -> str:
    """The one lock file for installing *name* into this environment, shared by every entry point
    that installs or removes it (Studio workers, setup, ``unsloth install-kernels``): beside the
    environment when it is writable, else one per environment in the temp folder."""
    import hashlib
    import tempfile

    override = os.environ.get("UNSLOTH_KERNEL_LOCK_DIR")
    if override:
        os.makedirs(override, exist_ok = True)
        return os.path.join(override, f"{name}.lock")
    folder = os.path.join(sys.prefix, ".unsloth-kernel-locks")
    try:
        os.makedirs(folder, exist_ok = True)
        if os.access(folder, os.W_OK):
            return os.path.join(folder, f"{name}.lock")
    except OSError:
        pass
    env = hashlib.sha256(os.path.realpath(sys.prefix).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(tempfile.gettempdir(), "unsloth-kernel-locks")
    os.makedirs(folder, exist_ok = True)
    return os.path.join(folder, f"{env}-{name}.lock")


@contextmanager
def kernel_install_lock(
    name: str,
    *,
    on_wait: Callable[[], None] | None = None,
    wait_seconds: float = KERNEL_LOCK_WAIT,
    beat_seconds: float = 60.0,
) -> Iterator[bool]:
    """Hold the cross-process lock for installing kernel *name* into this environment, covering the
    whole precheck / install / verify / cleanup transaction. Yields True when held, False when it
    is not: another process kept it past *wait_seconds*, or the lock cannot be opened or taken at
    all. Either way the caller installs nothing rather than race a peer unguarded. *on_wait* is
    called when the wait starts and every *beat_seconds* while it lasts."""
    import time

    try:
        handle = open(_kernel_lock_path(name), "a+b")
    except OSError as exc:
        logger_message = f"Unsloth: cannot open the {name} install lock ({exc}); not installing it."
        print(logger_message, file = sys.stderr)
        yield False
        return
    held = False
    try:
        deadline = time.monotonic() + wait_seconds
        next_beat = 0.0
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except (OSError, ImportError) as exc:
                if not isinstance(exc, OSError) or exc.errno not in _LOCK_CONTENDED:
                    print(
                        f"Unsloth: cannot take the {name} install lock ({exc}); not installing it.",
                        file = sys.stderr,
                    )
                    break
                now = time.monotonic()
                if now >= deadline:
                    break
                if on_wait is not None and now >= next_beat:
                    next_beat = now + beat_seconds
                    try:
                        on_wait()
                    except Exception:
                        pass
                time.sleep(0.5)
        yield held
    finally:
        if held:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, ImportError):
                pass
        if not handle.closed:
            handle.close()


def source_build_run_kwargs(
    *, is_hip: bool, gcc_install_dir: Callable[[], str | None]
) -> tuple[dict[str, Any], str | None]:
    """subprocess.run kwargs for a source build, and the gcc dir added to HIPCC_COMPILE_FLAGS_APPEND
    (None when nothing was added). ROCm builds take 10-30 min and are bounded at
    SOURCE_BUILD_TIMEOUT (1800 s); CUDA builds at SOURCE_BUILD_TIMEOUT_CUDA (7200 s, or
    UNSLOTH_SOURCE_BUILD_TIMEOUT), loose enough for a slow aarch64 causal-conv1d build."""
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        # pip and the compilers it drives write UTF-8 down this pipe; the Windows ANSI codepage would mojibake or raise over a fine install.
        "encoding": "utf-8",
        "errors": "replace",
        "env": utf8_child_env(),
    }
    # An overall deadline for every source build (run it through run_process_tree so a timeout
    # also stops the compilers pip / uv started): HIP's, or the far looser CUDA one.
    kwargs["timeout"] = SOURCE_BUILD_TIMEOUT if is_hip else SOURCE_BUILD_TIMEOUT_CUDA
    if not is_hip:
        return kwargs, None
    existing = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "")
    if "--gcc-install-dir" in existing:
        return kwargs, None
    gcc_dir = gcc_install_dir()
    if not gcc_dir:
        return kwargs, None
    env = dict(kwargs["env"])
    env["HIPCC_COMPILE_FLAGS_APPEND"] = f"{existing} --gcc-install-dir={gcc_dir}".strip()
    kwargs["env"] = env
    return kwargs, gcc_dir


def uninstall_command(
    distribution: str,
    *,
    use_uv: bool,
    uv_needs_system: bool = False,
) -> list[str]:
    """Uninstall *distribution* from ``sys.executable``. uv gets --python as well as --system:
    --system alone would remove from the system Python and leave the venv copy behind."""
    if use_uv:
        system = ["--system"] if uv_needs_system else []
        return ["uv", "pip", "uninstall", *system, "--python", sys.executable, distribution]
    return [sys.executable, "-m", "pip", "uninstall", "-y", distribution]


# `unsloth install-kernels` from here on.

# name -> (pip distribution, check that loads the compiled extension). The package imports are no
# proof: `import causal_conv1d` never loads it, and `import mamba_ssm` fails on einops under --no-deps.
# Install order: mamba_ssm's fast path imports causal_conv1d.
KERNELS = {
    "xformers": (
        "xformers",
        "from xformers._cpp_lib import _register_extensions; _register_extensions()",
    ),
    "flash_attn": ("flash-attn", "import torch, flash_attn_2_cuda"),
    "causal_conv1d": ("causal-conv1d", "import torch, causal_conv1d_cuda"),
    "mamba_ssm": ("mamba-ssm", "import torch, selective_scan_cuda"),
}

_PINNED = {"causal_conv1d": CAUSAL_CONV1D, "mamba_ssm": MAMBA_SSM}
_WITHOUT_GPU_ENV = "UNSLOTH_INSTALL_KERNELS_WITHOUT_GPU"


def resolve_wheel_url(name: str, env: dict[str, str] | None) -> str | None:
    if env is None:
        return None
    if name == "xformers":
        return xformers_wheel_url(env)
    if not str(env.get("platform_tag") or "").startswith(("linux", "win_")):
        return None
    if name == "flash_attn":
        return flash_attn_wheel_url(env)
    return _PINNED[name].wheel_url(env)


# FlashAttention 2 needs sm80. mamba_ssm's Triton kernels also run on sm75 with Triton 3.4+
# (torch 2.8+), the same rule unsloth_zoo's patch_mamba_ssm_pre_ampere_fallback applies, which
# turns the fast path off anywhere else, so the wheel would go unused there.
_GPU_GATED = ("flash_attn", "mamba_ssm")
_MAMBA_SM75_MIN_TRITON = (3, 4)
_CAPABILITY: dict = {}


_TRITON: dict = {}


def _triton_version(run: Callable[..., subprocess.CompletedProcess]) -> tuple[int, int] | None:
    """The `triton` module's version, as unsloth_zoo reads it. Probed once per runner.

    Asked of the module, not the `triton` distribution: pytorch-triton and other
    providers ship the same import under another name.
    """
    if run not in _TRITON:
        try:
            result = run(
                [sys.executable, "-c", "import triton; print(triton.__version__)"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 120,
            )
            major, minor = result.stdout.strip().split(".")[:2]
            _TRITON[run] = (int(major), int(minor))
        except Exception:
            _TRITON[run] = None
    return _TRITON[run]


def _pre_ampere_skip_reason(
    name: str, capability: tuple[int, int], run: Callable[..., subprocess.CompletedProcess]
) -> str | None:
    """Why a kernel stays off a pre-sm80 GPU, or None when unsloth_zoo uses it there."""
    if name != "mamba_ssm":
        return "needs sm80 or newer"
    forced = os.environ.get("UNSLOTH_MAMBA_PRE_AMPERE_FAST", "").strip()
    if forced in ("0", "1"):
        return None if forced == "1" else "UNSLOTH_MAMBA_PRE_AMPERE_FAST=0 turns off"
    if capability < (7, 5):
        return "needs sm80, or sm75 with Triton 3.4+"
    triton = _triton_version(run)
    if triton is not None and triton >= _MAMBA_SM75_MIN_TRITON:
        return None
    found = "missing" if triton is None else "%d.%d" % triton
    return f"needs sm80, or sm75 with Triton 3.4+ (torch 2.8+), and Triton is {found}"


def _gpu_capability(run: Callable[..., subprocess.CompletedProcess]) -> tuple[int, int] | None:
    """The best compute capability across the visible GPUs, or None. Probed once per runner."""
    if run not in _CAPABILITY:
        _CAPABILITY[run] = _probe_gpu_capability(run)
    return _CAPABILITY[run]


def gpu_below_flash_attn_floor(
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Whether the best visible CUDA GPU is below sm80, where Unsloth leaves FlashAttention 2 off
    (a T4 is sm75), so a flash-attn download would go unused. No visible GPU answers False: an
    image build installs ahead of the GPU it will run on."""
    capability = _gpu_capability(run)
    # ROCm reports a gfx version, not an sm, as its capability: the floor is NVIDIA's.
    return capability is not None and capability < (8, 0) and not _torch_is_hip(run)


_HIP: dict = {}


def _torch_is_hip(run: Callable[..., subprocess.CompletedProcess]) -> bool:
    """Whether torch is a ROCm build. Probed once per runner, and only below sm80."""
    if run not in _HIP:
        try:
            result = run(
                [sys.executable, "-c", "import torch; print(bool(torch.version.hip))"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 120,
            )
            _HIP[run] = result.stdout.strip() == "True"
        except Exception:
            _HIP[run] = False
    return _HIP[run]


def _probe_gpu_capability(
    run: Callable[..., subprocess.CompletedProcess],
) -> tuple[int, int] | None:
    check = (
        "import torch; n = torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "print(*max(torch.cuda.get_device_capability(i) for i in range(n))) if n else None"
    )
    try:
        result = run(
            [sys.executable, "-c", check],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 120,
        )
    except subprocess.TimeoutExpired:
        return None
    try:
        major, minor = (int(part) for part in result.stdout.split())
    except ValueError:
        return None
    return major, minor


def _loads(check: str, run: Callable[..., subprocess.CompletedProcess]) -> bool:
    result = run(
        [sys.executable, "-c", check], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL
    )
    return result.returncode == 0


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return None


def _outside_venv() -> bool:
    # Colab and other system interpreters: uv refuses them without --system.
    return sys.prefix == sys.base_prefix


def _uninstall(distribution: str, run: Callable[..., subprocess.CompletedProcess]) -> bool:
    cmd = uninstall_command(
        distribution, use_uv = bool(shutil.which("uv")), uv_needs_system = _outside_venv()
    )
    return run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL).returncode == 0


def install_kernel(
    name: str,
    env: dict[str, str] | None,
    *,
    dry_run: bool = False,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    exists: Callable[[str], bool | None] = url_exists,
    install_without_gpu: bool = False,
    skipped: list[str] | None = None,
) -> int:
    # install_without_gpu: no visible GPU yet (an image build for the GPU it will run on) still
    # installs. Each kernel left uninstalled for this machine is appended to *skipped*.
    distribution, check = KERNELS[name]

    def skip(message: str) -> int:
        print(message)
        if skipped is not None:
            skipped.append(name)
        return 0

    # A copy that already loads (a ROCm source build, a GPU-hidden image builder's earlier install)
    # is success whatever the wheel, GPU and network checks below would say.
    if not dry_run and _installed_version(distribution) is not None and _loads(check, run):
        print(f"Unsloth: {name} already installed and loads.")
        return 0
    # A dry run only prints the URL it would install, so it runs offline too.
    if not dry_run and package_installs_offline_env():
        return skip(
            f"Unsloth: not installing {name} while offline "
            "(UV_OFFLINE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE)."
        )
    url = resolve_wheel_url(name, env)
    if name in _GPU_GATED and url is not None:
        capability = _gpu_capability(run)
        if capability is None and not install_without_gpu:
            return skip(
                f"Unsloth: skipping {name}, which needs sm80 or newer (no CUDA GPU is visible; "
                "pass --without-gpu to install it for the GPU this image will run on)."
            )
        if capability is not None and capability < (8, 0):
            reason = _pre_ampere_skip_reason(name, capability, run)
            if reason is not None:
                return skip(
                    f"Unsloth: skipping {name}, which {reason} "
                    f"(the best GPU is sm{capability[0]}{capability[1]})."
                )
    torch_desc = f"torch {env.get('torch_version')}" if env else "this environment"
    # Only a 404 proves nothing is published; an unreachable check falls through to the install.
    if url is None or exists(url) is False:
        return skip(f"Unsloth: no prebuilt {name} for {torch_desc}; using the torch fallback.")
    # mamba_ssm imports triton at load; on Windows that is triton-windows, which its metadata cannot name.
    if name == "mamba_ssm" and sys.platform == "win32" and not _loads("import triton", run):
        return skip(
            "Unsloth: skipping mamba_ssm, which needs Triton on Windows; "
            "install triton-windows first (Studio setup does), then re-run."
        )
    # UNSLOTH_PYTORCH_MIRROR may carry credentials, and notebook output gets shared.
    shown = redact_url_credentials(url)
    if dry_run:
        print(shown)
        return 0
    # The lock Studio's workers take for the same kernel: never install or remove it under a
    # running load's precheck / verify.
    with kernel_install_lock(
        name, on_wait = lambda: print(f"Unsloth: waiting for another process installing {name}...")
    ) as held:
        if not held:
            return skip(f"Unsloth: could not take the {name} install lock; not installing it.")
        if _loads(check, run):
            print(f"Unsloth: {name} already installed and loads.")
            return 0
        failures: list[subprocess.CompletedProcess] = []
        outcome = install_prebuilt(
            url,
            install = install_wheel,
            verify = lambda: _loads(check, run),
            on_failed = lambda _installer, result: failures.append(result),
            use_uv = True,
            uv_needs_system = _outside_venv(),
            reinstall = True,
            run = run,
        )
        if outcome == "failed":
            print(
                f"Unsloth: installing {name} from {shown} failed; using the torch fallback.\n"
                + failures[-1].stdout[-2000:].replace(url, shown)
            )
            return 1
        if outcome == "rejected":
            # A wheel that imports but whose extension cannot load would be picked up and crash later.
            removed = (
                "removed it"
                if _uninstall(distribution, run)
                else f"run `pip uninstall {distribution}`"
            )
            print(
                f"Unsloth: {name} from {shown} does not load with {torch_desc}; {removed}, using the torch fallback."
            )
            return 1
        print(f"Unsloth: installed {name} ({shown.rsplit('/', 1)[-1]}).")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog = "unsloth install-kernels",
        description = "Install prebuilt kernel wheels matching the installed torch. Never builds from source or changes torch.",
    )
    parser.add_argument(
        "kernels",
        nargs = "*",
        choices = list(KERNELS),
        help = "kernels to install (default: all, each skipped when no wheel matches)",
    )
    parser.add_argument(
        "--dry-run", action = "store_true", help = "print the resolved wheel URL and install nothing"
    )
    parser.add_argument(
        "--without-gpu",
        action = "store_true",
        help = (
            "install the GPU-gated kernels (flash_attn, mamba_ssm) even though no CUDA GPU is "
            f"visible, e.g. in an image build; also {_WITHOUT_GPU_ENV}=1"
        ),
    )
    parser.add_argument(
        "--strict",
        action = "store_true",
        help = "exit 2 when a requested kernel was skipped (no GPU, no wheel, offline)",
    )
    args = parser.parse_args(argv)
    without_gpu = args.without_gpu or os.environ.get(_WITHOUT_GPU_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    env = probe_torch_wheel_env(timeout = 120, include_windows = True)
    status = 0
    skipped: list[str] = []
    for name in [name for name in KERNELS if not args.kernels or name in args.kernels]:
        status = max(
            status,
            install_kernel(
                name,
                env,
                dry_run = args.dry_run,
                install_without_gpu = without_gpu,
                skipped = skipped,
            ),
        )
    if skipped:
        print(f"Unsloth: not installed on this machine: {', '.join(skipped)}.")
        # A skip stays exit 0, as scripts running under `set -e` expect; --strict opts in.
        if args.strict and status == 0:
            status = 2
    return status


if __name__ == "__main__":
    sys.exit(main())

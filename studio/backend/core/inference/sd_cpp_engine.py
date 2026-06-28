# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native stable-diffusion.cpp (``sd-cli``) engine for the diffusion backend.

This mirrors the chat backend's llama.cpp shell-out: locate a prebuilt / built
binary, then run it as a one-shot subprocess. It is the CPU / Apple-Silicon
(and low-VRAM) tier of the two-engine strategy -- the diffusers path stays the
default on CUDA / ROCm / XPU, this covers the hardware diffusers serves poorly.

Kept deliberately thin:
  * ``find_sd_cpp_binary()`` -- env override -> ``~/.unsloth/stable-diffusion.cpp``
    build layouts -> in-tree build -> PATH. Same precedence as the llama finder.
  * ``SdCppEngine`` -- ``is_available`` / ``version`` probe + a ``generate`` that
    builds the argv (``sd_cpp_args``), runs ``sd-cli``, and returns the PNG path.
  * ``select_diffusion_engine(...)`` -- the routing decision (which backend gets
    diffusers vs. sd.cpp), a pure function so the device layer can call it.

Everything heavy (the subprocess, the binary) is reached only inside ``generate``
/ ``version`` so importing this module is free and unit tests stay hermetic.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    SdCppUpscaleParams,
    build_sd_cpp_command,
    build_sd_cpp_upscale_command,
    native_speed_flags,
)

logger = logging.getLogger(__name__)

# Binary name: sd-cli (sd-cli.exe on Windows). The stable-diffusion.cpp CMake
# target is ``sd-cli``; older builds shipped ``sd`` -- both are probed on PATH.
_BINARY_STEM = "sd-cli"
_LEGACY_STEM = "sd"


class SdCppCancelled(RuntimeError):
    """A generation was cancelled via its ``cancel_event`` (unload / superseding load
    / arbiter eviction). Distinct from a generation *failure* so the caller can keep
    cancellation semantics (no diffusers fallback, no error surfaced as a crash)."""


def _terminate(proc: "subprocess.Popen") -> None:
    """Hard-stop an sd-cli process (and any children). On POSIX the process is its
    own session leader (``start_new_session``), so kill the whole group; otherwise
    fall back to killing just the process."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001 -- killpg can miss (no pgid / already gone); fall back
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass


def _binary_name(stem: str) -> str:
    return f"{stem}.exe" if sys.platform == "win32" else stem


def _lib_path_var() -> str:
    """The platform's shared-library search env var."""
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    if sys.platform == "win32":
        return "PATH"
    return "LD_LIBRARY_PATH"


def runtime_env(binary: str, base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Environment that lets ``binary`` find its bundled shared libraries.

    The prebuilt archives ship ``libstable-diffusion.so`` (/ ``.dylib`` / DLLs)
    next to ``sd-cli``, so prepend the binary's own directory to the platform
    library path. A locally-built binary that is already linked finds its libs
    regardless, so this is harmless there.
    """
    env = dict(os.environ if base_env is None else base_env)
    var = _lib_path_var()
    bindir = str(Path(binary).resolve().parent)
    existing = env.get(var, "")
    env[var] = bindir + (os.pathsep + existing if existing else "")
    return env


def _layout_candidates(root: Path) -> list[Path]:
    """sd-cli locations under a stable-diffusion.cpp checkout/install ``root``,
    highest priority first: the cmake ``build/bin`` tree, then a Windows Release
    subdir, then the root itself."""
    name = _binary_name(_BINARY_STEM)
    cands = [
        root / "build" / "bin" / name,
        root / "build" / "bin" / "Release" / name,
        root / "bin" / name,
        root / name,
    ]
    return cands


def find_sd_cpp_binary() -> Optional[str]:
    """Locate the ``sd-cli`` binary, or None.

    Search order (mirrors the llama.cpp finder so a Studio install lands where
    both engines look):
      1. ``SD_CLI_PATH`` env -- a direct path to the binary.
      2. ``UNSLOTH_SD_CPP_PATH`` env -- a stable-diffusion.cpp install dir.
      3. ``~/.unsloth/stable-diffusion.cpp`` build layouts (the installer target).
      4. ``./stable-diffusion.cpp`` in-tree build (developer checkout).
      5. ``sd-cli`` (then legacy ``sd``) on PATH.
    """

    def _first_file(paths: list[Path]) -> Optional[str]:
        for p in paths:
            try:
                if p.is_file():
                    return str(p)
            except OSError:
                continue
        return None

    # 1. Direct binary path.
    env_bin = os.environ.get("SD_CLI_PATH")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    # 2. Custom install dir.
    custom = os.environ.get("UNSLOTH_SD_CPP_PATH")
    if custom:
        hit = _first_file(_layout_candidates(Path(custom)))
        if hit:
            return hit

    # 3. Default install root (sibling of ~/.unsloth/llama.cpp).
    hit = _first_file(_layout_candidates(Path.home() / ".unsloth" / "stable-diffusion.cpp"))
    if hit:
        return hit

    # 4. In-tree developer build: <repo_root>/stable-diffusion.cpp.
    try:
        project_root = Path(__file__).resolve().parents[4]
        hit = _first_file(_layout_candidates(project_root / "stable-diffusion.cpp"))
        if hit:
            return hit
    except (OSError, IndexError):
        pass

    # 5. PATH.
    for stem in (_BINARY_STEM, _LEGACY_STEM):
        on_path = shutil.which(stem)
        if on_path:
            return on_path
    return None


class SdCppEngine:
    """A thin handle over a located ``sd-cli`` binary.

    Construct it (cheap -- just resolves the path), check ``is_available``, then
    call ``generate``. Holds no process: each generation is an independent
    one-shot ``sd-cli`` run, so there is nothing to leak or clean up.
    """

    def __init__(self, binary: Optional[str] = None) -> None:
        self.binary = binary or find_sd_cpp_binary()
        self._version: Optional[str] = None

    def is_available(self) -> bool:
        return bool(self.binary) and Path(self.binary).is_file()

    def version(self, *, timeout: float = 10.0) -> Optional[str]:
        """First line of ``sd-cli --version``, cached. None if it can't run."""
        if not self.is_available():
            return None
        if self._version is not None:
            return self._version
        try:
            res = subprocess.run(
                [self.binary, "--version"],
                capture_output = True,
                text = True,
                errors = "replace",
                timeout = timeout,
                check = False,
                env = runtime_env(self.binary),
            )
            text = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
            self._version = text.splitlines()[0] if text else ""
        except (OSError, subprocess.SubprocessError):
            self._version = ""
        return self._version

    def generate(
        self,
        files: SdCppModelFiles,
        params: SdCppGenParams,
        *,
        output_path: str,
        offload: Optional[list[str]] = None,
        native_speed: Optional[str] = None,
        threads: Optional[int] = None,
        verbose: bool = False,
        extra_args: Optional[list[str]] = None,
        timeout: Optional[float] = 1800.0,
        env: Optional[dict[str, str]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Run one ``sd-cli`` generation; return the written image path.

        ``native_speed`` ("default"/"max") adds sd.cpp's own speed flags
        (``--diffusion-fa`` etc.), de-duplicated against the offload flags that may
        already include them. Raises ``RuntimeError`` if the binary is missing, the
        process exits nonzero, or no output file is produced. ``on_log`` (if given)
        receives each line of sd-cli's progress output as it arrives. ``cancel_event``
        (if given) is polled while the child runs; when set, the process tree is
        killed and ``SdCppCancelled`` is raised.
        """
        offload = list(offload or [])
        speed = [f for f in native_speed_flags(native_speed) if f not in offload]
        merged_extra = speed + list(extra_args or [])
        cmd = build_sd_cpp_command(
            self._require_binary(),
            files,
            params,
            output_path = str(self._prepare_out(output_path)),
            offload = offload,
            threads = threads,
            verbose = verbose,
            extra_args = merged_extra,
        )
        return self._run(
            cmd, output_path, timeout = timeout, env = env, on_log = on_log,
            cancel_event = cancel_event,
        )

    def upscale(
        self,
        params: "SdCppUpscaleParams",
        *,
        output_path: str,
        verbose: bool = False,
        extra_args: Optional[list[str]] = None,
        timeout: Optional[float] = 1800.0,
        env: Optional[dict[str, str]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Upscale an image with an ESRGAN model; return the written path."""
        cmd = build_sd_cpp_upscale_command(
            self._require_binary(),
            params,
            output_path = str(self._prepare_out(output_path)),
            verbose = verbose,
            extra_args = extra_args,
        )
        return self._run(
            cmd, output_path, timeout = timeout, env = env, on_log = on_log,
            cancel_event = cancel_event,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _require_binary(self) -> str:
        if not self.is_available():
            raise RuntimeError(
                "sd-cli (stable-diffusion.cpp) binary not found. Build it or set "
                "SD_CLI_PATH / UNSLOTH_SD_CPP_PATH."
            )
        return self.binary  # type: ignore[return-value]

    @staticmethod
    def _prepare_out(output_path: str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents = True, exist_ok = True)
        return out

    def _run(
        self,
        cmd: list[str],
        output_path: str,
        *,
        timeout: Optional[float],
        env: Optional[dict[str, str]],
        on_log: Optional[Callable[[str], None]],
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Run an sd-cli argv, stream output, and return the produced image path.

        Raises ``RuntimeError`` on nonzero exit, timeout, or a missing output, and
        ``SdCppCancelled`` when ``cancel_event`` fires. Shared by ``generate`` and
        ``upscale``.
        """
        out = Path(output_path)
        base = dict(os.environ)
        if env:
            base.update(env)
        run_env = runtime_env(self._require_binary(), base)
        logger.info("sd-cli run: %s", " ".join(cmd))

        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            errors = "replace",
            env = run_env,
            # Own session/process group so cancellation/timeout can kill the whole
            # tree, not just the parent (POSIX only; harmless flag elsewhere).
            start_new_session = (os.name == "posix"),
        )
        # Drain stdout on a reader thread so the timeout is enforced even when the
        # child hangs WITHOUT printing (e.g. stuck in model load / GPU init): a plain
        # `for line in proc.stdout` blocks until EOF, so proc.wait(timeout) would
        # never be reached. The reader pushes lines (then a None sentinel at EOF) to a
        # queue the main loop polls against a wall-clock deadline.
        tail: list[str] = []
        line_q: "queue.Queue[Optional[str]]" = queue.Queue()

        def _drain() -> None:
            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line_q.put(raw.rstrip("\n"))
            finally:
                line_q.put(None)

        reader = threading.Thread(target = _drain, daemon = True)
        reader.start()

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        stdout_done = False
        try:
            while True:
                # Cancellation (unload / superseding load / arbiter eviction): kill the
                # process tree and signal the caller it was cancelled, not a failure.
                if cancel_event is not None and cancel_event.is_set() and proc.poll() is None:
                    _terminate(proc)
                    raise SdCppCancelled("sd-cli generation was cancelled.")
                if deadline is not None and time.monotonic() >= deadline and proc.poll() is None:
                    _terminate(proc)
                    raise RuntimeError(f"sd-cli timed out after {timeout}s")
                try:
                    line = line_q.get(timeout = 0.1)
                except queue.Empty:
                    if proc.poll() is not None and stdout_done:
                        break
                    continue
                if line is None:
                    stdout_done = True
                    if proc.poll() is not None:
                        break
                    continue
                tail.append(line)
                if len(tail) > 40:
                    tail.pop(0)
                if on_log is not None:
                    on_log(line)
            ret = proc.wait(timeout = 5.0)
        finally:
            if proc.poll() is None:
                _terminate(proc)

        if ret != 0:
            raise RuntimeError(f"sd-cli exited {ret}. Last output:\n" + "\n".join(tail[-12:]))
        if not out.is_file():
            raise RuntimeError(
                f"sd-cli reported success but no image at {out}. Last output:\n"
                + "\n".join(tail[-12:])
            )
        logger.info("sd-cli run ok in %.1fs -> %s", time.time() - t0, out)
        return out


# ── engine routing ──────────────────────────────────────────────────────────

ENGINE_DIFFUSERS = "diffusers"
ENGINE_SD_CPP = "sd_cpp"

# Backends diffusers serves well with GPU acceleration. Everything else (CPU,
# Apple MPS) is where the native engine earns its place.
_GPU_BACKENDS = frozenset({"cuda", "rocm", "xpu"})


def select_diffusion_engine(
    backend: str,
    *,
    native_available: bool,
    prefer_native: bool = False,
) -> str:
    """Choose the engine for a resolved device ``backend``.

    - ``prefer_native`` + an available binary always wins (a user can force the
      native engine even on a CUDA box, e.g. to fit a tiny VRAM budget).
    - CPU / MPS route to sd.cpp when the binary is available, else fall back to
      diffusers (which still runs there, just slowly).
    - CUDA / ROCm / XPU stay on diffusers.
    """
    if prefer_native and native_available:
        return ENGINE_SD_CPP
    if backend not in _GPU_BACKENDS and native_available:
        return ENGINE_SD_CPP
    return ENGINE_DIFFUSERS

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd-cli engine for CPU, Apple Silicon and low-VRAM; diffusers stays default on CUDA/ROCm/XPU;
heavy work only in generate/version so import is free"""

from __future__ import annotations

import codecs
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator, Optional

from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
)
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs,
    windows_loader_error_boxes_suppressed,
)
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    SdCppUpscaleParams,
    SdCppVideoGenParams,
    build_sd_cpp_command,
    build_sd_cpp_upscale_command,
    build_sd_cpp_video_command,
    native_speed_flags,
    speed_flags_without,
    sd_cli_output_paths,
)

logger = logging.getLogger(__name__)

# Lines worth keeping from a failed native process's output whatever their position.
_DIAGNOSTIC_MARKERS = (
    "error",
    "abort",
    "assert",
    "unsupported",
    "not implemented",
    "out of memory",
    "failed",
    "exception",
)


def _diagnostic_tail(
    lines,
    *,
    keep: int = 20,
    limit: int = 1500,
) -> str:
    """Marked diagnostic lines come first, since a native abort prints its reason before a long
    backtrace."""
    captured = list(lines)
    marked = [line for line in captured if any(m in line.lower() for m in _DIAGNOSTIC_MARKERS)]
    chosen: list[str] = []
    for line in marked[-keep:] + captured[-max(keep // 2, 4) :]:
        if line not in chosen:
            chosen.append(line)
    return "\n".join(chosen)[:limit]


# sd-cli (sd-cli.exe on Windows); older builds shipped ``sd`` -- both probed on PATH.
_BINARY_STEM = "sd-cli"
_LEGACY_STEM = "sd"
# Together these tell old stable-diffusion.cpp help text apart from unrelated tools named ``sd``.
_LEGACY_HELP_MARKERS = ("--negative-prompt", "--cfg-scale", "--steps")
# The persistent HTTP server target, shipped next to sd-cli in both prebuilt and cmake builds.
_SERVER_STEM = "sd-server"

# Ownership marker; setup.sh / uninstall.sh / uninstall.ps1 require it before deleting a tree.
OWNER_MARKER = ".unsloth-studio-owned"

# Only stops a WEDGED process: CPU-only runs took up to ~25 min. Matches the Images page SETTLE_MAX_MS.
NATIVE_GENERATION_TIMEOUT_S = 6 * 60 * 60.0

_PRIVATE_TEXT_OPTIONS = frozenset({"--prompt", "--negative-prompt", "-p", "-n"})


def _verbose_native_logs() -> bool:
    """Return whether Studio verbose logging is enabled."""
    try:
        from loggers.config import verbose_logging_requested
        return verbose_logging_requested()
    except Exception:
        return False


def _last_option_value(cmd: list[str], option: str) -> Optional[str]:
    """Return an option's effective last value, including ``--option=value`` form."""
    value = None
    prefix = f"{option}="
    for index, token in enumerate(cmd):
        if token == option and index + 1 < len(cmd):
            value = str(cmd[index + 1])
        elif str(token).startswith(prefix):
            value = str(token)[len(prefix) :]
    return value


def _sd_cpp_command_for_log(cmd: list[str]) -> list[str]:
    """Redact prompt text from argv before logging."""
    out: list[str] = []
    index = 0
    while index < len(cmd):
        token = str(cmd[index])
        if token in _PRIVATE_TEXT_OPTIONS:
            out.append(token)
            if index + 1 < len(cmd):
                out.append("<redacted>")
                index += 2
                continue
            index += 1
            continue
        replaced = False
        for option in _PRIVATE_TEXT_OPTIONS:
            if token.startswith(f"{option}="):
                out.append(f"{option}=<redacted>")
                replaced = True
                break
        if not replaced:
            out.append(token)
        index += 1
    return out


def _compact_log_value(value: str, limit: int = 48) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _sd_cpp_command_summary(cmd: list[str], *, default_mode: str = "img_gen") -> str:
    """Summarize argv without paths or prompt text."""
    # Prompt values can resemble options; redact them before extracting settings.
    cmd = _sd_cpp_command_for_log(cmd)
    mode = _compact_log_value(_last_option_value(cmd, "--mode") or default_mode)
    fields = [f"mode={mode}"]
    model = _last_option_value(cmd, "--diffusion-model")
    if model:
        model_name = str(model).replace("\\", "/").rsplit("/", 1)[-1]
        fields.append(f"model={_compact_log_value(model_name)}")
    width = _last_option_value(cmd, "--width")
    height = _last_option_value(cmd, "--height")
    if width is not None and height is not None:
        fields.append(f"size={_compact_log_value(width)}x{_compact_log_value(height)}")
    for label, option in (
        ("steps", "--steps"),
        ("frames", "--video-frames"),
        ("fps", "--fps"),
        ("seed", "--seed"),
        ("port", "--listen-port"),
        ("threads", "--threads"),
    ):
        value = _last_option_value(cmd, option)
        if value is not None:
            fields.append(f"{label}={_compact_log_value(value)}")
    return " ".join(fields)


# sd-cli redraws progress in place: "\r<bar> <step>/<steps> - <speed>\033[K", so \r LEADS a record
# and the erase-to-EOL CLOSES it.
_ANSI_ERASE = "\x1b[K"
# Escapes mid-line corrupt on_log and the error tail.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# one read1() per redraw in practice; large enough that a burst of finished lines costs one read
_READ_CHUNK = 4096


def strip_ansi(text: str) -> str:
    """Drop CSI escape sequences (notably the ``\\033[K`` that closes every progress redraw)."""
    return _ANSI_CSI_RE.sub("", text)


def split_progress_records(buf: str) -> tuple[list[str], str]:
    """Splits raw sd-cli stdout on CR, LF or a trailing erase-to-EOL, since in-place redraws carry no newline; returns records plus the unterminated remainder."""
    records: list[str] = []
    start = 0
    i = 0
    n = len(buf)
    while i < n:
        ch = buf[i]
        if ch == "\r" or ch == "\n":
            records.append(buf[start:i])
            # CRLF is one terminator, not two (Windows sd-cli builds).
            if ch == "\r" and i + 1 < n and buf[i + 1] == "\n":
                i += 1
            i += 1
            start = i
            continue
        if buf.startswith(_ANSI_ERASE, i):
            i += len(_ANSI_ERASE)
            records.append(buf[start:i])
            start = i
            continue
        i += 1
    return records, buf[start:]


def iter_sd_cpp_records(stream) -> Iterator[str]:
    """Reads the raw pipe with read1 so a newline-less progress redraw is not stuck behind readline."""
    raw = getattr(stream, "buffer", None)
    if raw is None or not hasattr(raw, "read1"):
        for line in stream:
            records, rest = split_progress_records(line)
            for rec in records:
                yield strip_ansi(rec)
            if rest:
                yield strip_ansi(rest)
        return
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    while True:
        chunk = raw.read1(_READ_CHUNK)
        if not chunk:
            pending += decoder.decode(b"", final = True)
            break
        pending += decoder.decode(chunk)
        records, pending = split_progress_records(pending)
        for rec in records:
            yield strip_ansi(rec)
    # EOF: hand over any final line the child left unterminated.
    if pending:
        yield strip_ansi(pending)


class SdCppCancelled(RuntimeError):
    """A generation cancelled via its ``cancel_event`` (unload / superseding load / arbiter
    eviction). Distinct from a *failure* so the caller keeps cancellation semantics."""


def _terminate(proc: "subprocess.Popen") -> None:
    """Hard-stop sd-cli and its children: POSIX kills the session group, else just the process."""
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
    # reap the killed child: callers raise right after _terminate
    try:
        proc.wait(timeout = 5)
    except Exception:  # noqa: BLE001 -- best-effort reap; never block teardown
        pass


def binary_accelerator(binary: Optional[str]) -> Optional[str]:
    """Accelerator class from the install record, else a cached probe listing (no subprocess); None if unknown."""
    if not binary:
        return None
    try:
        from core.inference import sd_cpp_backend  # noqa: PLC0415 - the backend imports this module
        return (
            sd_cpp_backend._installed_accelerator_of(binary)
            or sd_cpp_backend.sd_cpp_listed_device(binary, probe = False)[0]
        )
    except Exception:  # noqa: BLE001 - unknown keeps the default speed flags
        return None


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
    """Child env with the binary dir on the library path for bundled libs; also strips the native-path lease secret so the child cannot mint grants."""
    env = child_env_without_native_path_secret(os.environ if base_env is None else base_env)
    var = _lib_path_var()
    bindir = str(Path(binary).resolve().parent)
    dirs = [bindir]
    if _needs_host_cuda_runtime(bindir):
        dirs.extend(d for d in host_cuda_runtime_dirs() if d not in dirs)
    existing = env.get(var, "")
    # After the caller's PATH, so an SDK the user put on PATH still wins.
    env[var] = os.pathsep.join(
        [*dirs, *([existing] if existing else []), *host_hip_sdk_dirs(bindir)]
    )
    return env


def host_hip_sdk_dirs(bindir: str) -> list[str]:
    try:
        from core.inference import sd_cpp_backend  # noqa: PLC0415 - the backend imports this module
        return sd_cpp_backend._installer_module().hip_sdk_dirs_for_bundle(bindir)
    except Exception:  # noqa: BLE001 - without the installer the child keeps the caller's PATH
        return []


def _needs_host_cuda_runtime(bindir: str) -> bool:
    try:
        names = [n.lower() for n in os.listdir(bindir)]
    except OSError:
        return False
    cuda = any(n == "ggml-cuda.dll" or n.startswith("libggml-cuda.so") for n in names)
    runtime = any(n.startswith(("cudart64_", "libcudart.so")) for n in names)
    return cuda and not runtime


def _llama_server_binary() -> Optional[str]:
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._find_llama_server_binary()
    except Exception:  # noqa: BLE001 -- no llama.cpp install: nothing to share
        return None


def host_cuda_runtime_dirs() -> list[str]:
    """llama-server's CUDA runtime dirs. Its build dir is dropped off Windows: its libggml*.so share
    sonames with the bundle's (Windows searches the executable's dir first)."""
    binary = _llama_server_binary()
    if not binary:
        return []
    try:
        from core.inference.llama_cpp import LlamaCppBackend, _llama_lib_dir

        var = _lib_path_var()
        inherited = {d for d in os.environ.get(var, "").split(os.pathsep) if d}
        llama_dir = os.path.normcase(str(_llama_lib_dir(binary)))
        out: list[str] = []
        for d in (
            LlamaCppBackend._llama_server_env_for_binary(binary).get(var, "").split(os.pathsep)
        ):
            if not d or d in inherited or d in out:
                continue
            if sys.platform != "win32" and os.path.normcase(d) == llama_dir:
                continue
            out.append(d)
        return out
    except Exception as exc:  # noqa: BLE001 -- a launch must not fail on a path lookup
        logger.debug("sd.cpp: CUDA runtime dirs from llama.cpp unavailable: %s", exc)
        return []


def host_cuda_runtime_major() -> Optional[str]:
    """CUDA major the installed llama.cpp links, else torch's."""
    binary = _llama_server_binary()
    if binary:
        try:
            from utils.llama_cpp_freshness import read_install_marker

            line = (read_install_marker(binary) or {}).get("runtime_line")
            match = re.fullmatch(r"cuda(\d+)", line if isinstance(line, str) else "")
            if match:
                return match.group(1)
        except Exception:  # noqa: BLE001
            pass
    version = getattr(getattr(sys.modules.get("torch"), "version", None), "cuda", None)
    return version.split(".")[0] if isinstance(version, str) and version else None


def _layout_candidates(root: Path, stem: str = _BINARY_STEM) -> list[Path]:
    """``stem`` locations under a stable-diffusion.cpp ``root``, highest priority first: the cmake
    ``build/bin`` tree, a Windows Release subdir, the root, then the prebuilt's versioned subdir."""
    name = _binary_name(stem)
    cands = [
        root / "build" / "bin" / name,
        root / "build" / "bin" / "Release" / name,
        root / "bin" / name,
        root / name,
    ]
    try:
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        subdirs.sort(key = lambda p: p.stat().st_mtime, reverse = True)
        for sub in subdirs:
            cands.append(sub / name)
            cands.append(sub / "bin" / name)
    except OSError:
        pass
    return cands


def _first_file(paths: list[Path]) -> Optional[str]:
    for p in paths:
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


# Keyed by file stat, so an in-place replace re-probes while a rebuild elsewhere on PATH is unaffected.
_IDENTITY_MEMO: dict[tuple[str, int, int, int], tuple[bool, float]] = {}
_IDENTITY_MEMO_LOCK = threading.Lock()
_IDENTITY_MEMO_MAX = 32
# No stat tuple is a content hash (Windows st_ctime survives overwrite), so verdicts also expire.
_IDENTITY_MEMO_TTL_S = 60.0


def _identity_key(binary: str) -> Optional[tuple[str, int, int, int]]:
    """Keyed on st_mtime and st_ctime, since restored mtimes hide same-size replacements on POSIX; None when unreadable, so that candidate is never memoized."""
    try:
        st = os.stat(binary)
    except OSError:
        return None
    return (str(Path(binary).resolve(strict = False)), st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def help_text_identifies_sd_cpp(help_text: str) -> bool:
    """Identity, not capability; pre-banner builds (shipped as sd) match on their legacy option
    signature."""
    return "stable-diffusion.cpp" in help_text.lower() or all(
        marker in help_text for marker in _LEGACY_HELP_MARKERS
    )


def sd_cpp_binary_identifies(binary: str) -> bool:
    """Fails closed; only decisive verdicts are memoized, so a timeout cannot blacklist a real build."""
    key = _identity_key(binary)
    if key is not None:
        now = time.monotonic()
        with _IDENTITY_MEMO_LOCK:
            cached = _IDENTITY_MEMO.get(key)
            if cached is not None and now - cached[1] > _IDENTITY_MEMO_TTL_S:
                _IDENTITY_MEMO.pop(key, None)
                cached = None
        if cached is not None:
            return cached[0]
    returncode = None
    try:
        with windows_loader_error_boxes_suppressed():
            result = subprocess.run(
                [binary, "--help"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
                env = runtime_env(binary),
                **windows_hidden_subprocess_kwargs(),
            )
        help_text = (result.stdout or "") + "\n" + (result.stderr or "")
        returncode = result.returncode
    except (OSError, subprocess.SubprocessError):
        help_text = ""
    identified = help_text_identifies_sd_cpp(help_text)
    # Old builds print usage and exit 1; rc 127 from the dynamic loader is not evidence.
    decisive = identified or returncode == 0
    if key is not None and decisive:
        with _IDENTITY_MEMO_LOCK:
            if len(_IDENTITY_MEMO) >= _IDENTITY_MEMO_MAX:
                _IDENTITY_MEMO.clear()
            _IDENTITY_MEMO[key] = (identified, time.monotonic())
    return identified


def _is_legacy_sd_cpp_binary(binary: str) -> bool:
    """Bare sd on PATH is accepted only if --help identifies it; Debian/Ubuntu ship an unrelated sd."""
    identified = sd_cpp_binary_identifies(binary)
    if not identified:
        logger.warning(
            "ignoring PATH executable %s named sd because its --help output does not identify "
            "stable-diffusion.cpp",
            binary,
        )
    return identified


def managed_install_root() -> Path:
    """Installer-owned root: only copies here may be reinstalled over; it follows the studio home, and legacy ~/.unsloth paths stay found for existing installs."""
    return _studio_component_root("stable-diffusion.cpp")


def _studio_component_root(name: str) -> Path:
    """Home is made absolute, so a relative UNSLOTH_STUDIO_HOME cannot move the tree when cwd changes."""
    home = (os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or "").strip()
    legacy = Path.home() / ".unsloth" / name
    if not home:
        return legacy
    root = Path(home).expanduser()
    legacy_studio = Path.home() / ".unsloth" / "studio"
    try:
        root = root.resolve()
        is_legacy = root == legacy_studio.resolve()
    except (OSError, ValueError):
        root = root.absolute()
        is_legacy = root == legacy_studio
    return legacy if is_legacy else root / name


def legacy_sibling_install_root() -> Optional[Path]:
    """Pre-fix tree beside the home, only with the ownership marker; lexical parent tried first, since resolving symlinks first looks in the wrong place."""
    home = (os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or "").strip()
    if not home:
        return None
    current = managed_install_root()
    for base in _legacy_sibling_bases(home):
        root = base / "stable-diffusion.cpp"
        try:
            if root != current and (root / OWNER_MARKER).is_file():
                return root
        except OSError:
            continue
    return None


def _legacy_sibling_bases(home: str) -> list[Path]:
    """The directories an older build could have taken as ``<studio home>/..``, lexical first."""
    bases: list[Path] = []
    for candidate in (lambda p: p.absolute(), lambda p: p.resolve()):
        try:
            base = candidate(Path(home).expanduser()).parent
        except (OSError, ValueError):
            continue
        if base not in bases:
            bases.append(base)
    return bases


def in_tree_install_root() -> Optional[Path]:
    """Developer in-tree build root; a named function so tests can point it at an empty location."""
    try:
        return Path(__file__).resolve().parents[4] / "stable-diffusion.cpp"
    except (OSError, IndexError):
        return None


def is_managed_binary(binary: Optional[str]) -> bool:
    """Path alone is not enough: a git clone has the same name, so the ownership marker decides."""
    return owning_managed_root(binary) is not None


def owning_managed_root(binary: Optional[str]) -> Optional[Path]:
    """Checks both roots, current first; per-install state must be read from the root the binary is in, or an installed bundle gets re-downloaded."""
    if not binary:
        return None
    roots = [managed_install_root()]
    legacy = legacy_sibling_install_root()
    if legacy is not None:
        roots.append(legacy)
    for root in roots:
        try:
            Path(binary).resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        try:
            if (root / OWNER_MARKER).is_file():
                return root
        except OSError:
            continue
    return None


def _find_binary(
    *, direct_env: str, path_stems: tuple[str, ...], layout_stem: str
) -> Optional[str]:
    """Lookup order: direct env binary, UNSLOTH_SD_CPP_PATH dir, install root, in-tree build, then PATH."""
    env_bin = os.environ.get(direct_env)
    if env_bin and Path(env_bin).is_file():
        return env_bin

    custom = os.environ.get("UNSLOTH_SD_CPP_PATH")
    if custom:
        hit = _first_file(_layout_candidates(Path(custom), layout_stem))
        if hit:
            return hit

    # 3. Default install root (honors UNSLOTH_STUDIO_HOME / STUDIO_HOME like the installer).
    default_root = managed_install_root()
    hit = _first_file(_layout_candidates(default_root, layout_stem))
    if hit:
        return hit

    # 3b. Marker-gated legacy tree beside the Unsloth home.
    legacy_root = legacy_sibling_install_root()
    if legacy_root is not None:
        hit = _first_file(_layout_candidates(legacy_root, layout_stem))
        if hit:
            return hit

    # 4. In-tree developer build: <repo_root>/stable-diffusion.cpp.
    in_tree = in_tree_install_root()
    if in_tree is not None:
        hit = _first_file(_layout_candidates(in_tree, layout_stem))
        if hit:
            return hit

    for stem in path_stems:
        on_path = shutil.which(stem)
        if on_path and (stem != _LEGACY_STEM or _is_legacy_sd_cpp_binary(on_path)):
            return on_path
    return None


def find_sd_cpp_binary() -> Optional[str]:
    """Locate the one-shot ``sd-cli`` binary (env ``SD_CLI_PATH``), or None. Probes ``sd-cli`` then
    legacy ``sd``. The fallback engine once ``sd-server`` exists; also backs ESRGAN upscale."""
    return _find_binary(
        direct_env = "SD_CLI_PATH",
        path_stems = (_BINARY_STEM, _LEGACY_STEM),
        layout_stem = _BINARY_STEM,
    )


def find_sd_server_binary() -> Optional[str]:
    """Locate the persistent ``sd-server`` binary (env ``SD_SERVER_PATH``), or None. Same precedence
    as ``find_sd_cpp_binary`` keyed to the ``sd-server`` stem. Preferred over the one-shot CLI: it
    loads the model once and serves many generations without reloading."""
    return _find_binary(
        direct_env = "SD_SERVER_PATH",
        path_stems = (_SERVER_STEM,),
        layout_stem = _SERVER_STEM,
    )


class SdCppEngine:
    """A thin handle over a located ``sd-cli`` binary. Holds no process: each generation is an
    independent one-shot run, so there is nothing to leak or clean up."""

    def __init__(self, binary: Optional[str] = None) -> None:
        self.binary = binary or find_sd_cpp_binary()
        self._version: Optional[str] = None

    def is_available(self) -> bool:
        return bool(self.binary) and Path(self.binary).is_file()

    def version(self, *, timeout: float = 10.0) -> Optional[str]:
        """First line of ``sd-cli --version``, cached on success. ``None`` when the binary is absent
        OR present-but-unrunnable (missing libs / bad permissions), so callers fail a load early
        instead of a "ready" state that crashes on first generation."""
        if not self.is_available():
            return None
        if self._version is not None:
            return self._version
        try:
            with windows_loader_error_boxes_suppressed():
                res = subprocess.run(
                    [self.binary, "--version"],
                    capture_output = True,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
                    timeout = timeout,
                    check = False,
                    env = runtime_env(self.binary),
                    **windows_hidden_subprocess_kwargs(),
                )
        except (OSError, subprocess.SubprocessError):
            return None
        if res.returncode != 0:
            return None
        text = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
        self._version = text.splitlines()[0] if text else ""
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
        timeout: Optional[float] = NATIVE_GENERATION_TIMEOUT_S,
        env: Optional[dict[str, str]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Speed flags skip any offload flag; a set cancel_event kills the sd-cli process tree."""
        offload = list(offload or [])
        binary = self._require_binary()
        speed = speed_flags_without(
            native_speed_flags(
                native_speed, [*offload, *(extra_args or ())], binary_accelerator(binary), binary
            ),
            offload,
        )
        merged_extra = speed + list(extra_args or [])
        cmd = build_sd_cpp_command(
            binary,
            files,
            params,
            output_path = str(self._prepare_out(output_path)),
            offload = offload,
            threads = threads,
            verbose = verbose,
            extra_args = merged_extra,
        )
        # A layered run writes numbered files, never output_path itself.
        expected = output_path
        if params.qwen_image_layers is not None:
            expected = sd_cli_output_paths(str(output_path), int(params.qwen_image_layers) + 1)[0]
        return self._run(
            cmd,
            expected,
            timeout = timeout,
            env = env,
            on_log = on_log,
            cancel_event = cancel_event,
        )

    def upscale(
        self,
        params: "SdCppUpscaleParams",
        *,
        output_path: str,
        verbose: bool = False,
        extra_args: Optional[list[str]] = None,
        timeout: Optional[float] = NATIVE_GENERATION_TIMEOUT_S,
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
            cmd,
            output_path,
            timeout = timeout,
            env = env,
            on_log = on_log,
            cancel_event = cancel_event,
        )

    def generate_video(
        self,
        files: SdCppModelFiles,
        params: SdCppVideoGenParams,
        *,
        output_path: str,
        offload: Optional[list[str]] = None,
        threads: Optional[int] = None,
        verbose: bool = False,
        extra_args: Optional[list[str]] = None,
        timeout: Optional[float] = NATIVE_GENERATION_TIMEOUT_S,
        env: Optional[dict[str, str]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Run one ``vid_gen`` generation and return its video container."""
        cmd = build_sd_cpp_video_command(
            self._require_binary(),
            files,
            params,
            output_path = str(self._prepare_out(output_path)),
            offload = offload,
            threads = threads,
            verbose = verbose,
            extra_args = extra_args,
        )
        return self._run(
            cmd,
            output_path,
            timeout = timeout,
            env = env,
            on_log = on_log,
            cancel_event = cancel_event,
        )

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
        # drop a stale file so the post-run is_file() check proves THIS run produced the image
        out.unlink(missing_ok = True)
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
        """Run an sd-cli argv, stream output, return the image path. Raises ``RuntimeError`` on
        nonzero exit / timeout / missing output, ``SdCppCancelled`` on cancel. Shared by generate/upscale.
        """
        out = Path(output_path)
        base = dict(os.environ)
        if env:
            base.update(env)
        run_env = runtime_env(self._require_binary(), base)
        summary = _sd_cpp_command_summary(cmd)
        verbose_logs = _verbose_native_logs()
        if verbose_logs:
            logger.info("sd-cli run: %s", " ".join(_sd_cpp_command_for_log(cmd)))
        else:
            logger.info("sd-cli run started: %s", summary)

        t0 = time.time()
        # A generation admitted before quit can reach Popen after the shutdown sweep snapshot; sd-cli would then run unreaped, holding VRAM past exit.
        if is_process_shutting_down():
            raise SdCppCancelled("Unsloth is shutting down; not starting sd-cli.")
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = run_env,
            # Own session/process group so cancellation/timeout kills the whole tree (POSIX).
            start_new_session = (os.name == "posix"),
            # Bind sd-cli to the parent's lifetime (PR_SET_PDEATHSIG) so a crash cannot orphan it holding VRAM.
            **child_popen_kwargs(),
        )
        # child_popen_kwargs is empty on macOS, so record the pid for the shutdown sweep too.
        adopt_pid(proc.pid)
        # Recheck after recording: covers the window between the gate above and adopt_pid.
        if is_process_shutting_down():
            logger.info("shutdown began during the spawn; killing the new sd-cli")
            _terminate(proc)
            raise SdCppCancelled("Unsloth is shutting down; not starting sd-cli.")
        # Drains stdout on a thread so the timeout holds even if the child hangs without printing.
        tail: deque[str] = deque(maxlen = 200)
        line_q: "queue.Queue[Optional[str]]" = queue.Queue()

        def _drain() -> None:
            try:
                assert proc.stdout is not None
                for rec in iter_sd_cpp_records(proc.stdout):
                    line_q.put(rec)
            finally:
                line_q.put(None)

        reader = threading.Thread(target = _drain, daemon = True)
        reader.start()

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        stdout_done = False
        try:
            while True:
                # Cancellation: kill the process tree and signal cancelled, not failure.
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
                if on_log is not None:
                    on_log(line)
            ret = proc.wait(timeout = 5.0)
        finally:
            if proc.poll() is None:
                _terminate(proc)
            if proc.poll() is not None:
                forget_pid(proc.pid)

        if ret != 0:
            raise RuntimeError(f"sd-cli exited {ret}. Last output:\n" + _diagnostic_tail(tail))
        if not out.is_file():
            raise RuntimeError(
                f"sd-cli reported success but no image at {out}. Last output:\n"
                + _diagnostic_tail(tail)
            )
        elapsed = time.time() - t0
        if verbose_logs:
            logger.info("sd-cli run ok in %.1fs -> %s", elapsed, out)
        else:
            logger.info("sd-cli run completed: %s elapsed=%.1fs", summary, elapsed)
        return out


ENGINE_DIFFUSERS = "diffusers"
ENGINE_SD_CPP = "sd_cpp"
# Opt-in Apple Silicon MLX engine (core.inference.mlx_diffusion_backend), chosen only by the router, never here.
ENGINE_MLX = "mlx"

# Backends diffusers serves well with GPU acceleration; everything else is native-engine territory.
_GPU_BACKENDS = frozenset({"cuda", "rocm", "xpu"})


def select_diffusion_engine(
    backend: str,
    *,
    native_available: bool,
    prefer_native: bool = False,
) -> str:
    """CUDA, ROCm and XPU stay on diffusers unless native is preferred and the binary is available."""
    if prefer_native and native_available:
        return ENGINE_SD_CPP
    if backend not in _GPU_BACKENDS and native_available:
        return ENGINE_SD_CPP
    return ENGINE_DIFFUSERS

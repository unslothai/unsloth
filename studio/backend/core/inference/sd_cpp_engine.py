# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native stable-diffusion.cpp (``sd-cli``) engine for the diffusion backend.

Mirrors the chat backend's llama.cpp shell-out: locate a prebuilt / built binary, run it as a
one-shot subprocess. The CPU / Apple-Silicon / low-VRAM tier; diffusers stays the default on
CUDA / ROCm / XPU. Thin: ``find_sd_cpp_binary()`` (env -> install layouts -> in-tree -> PATH),
``SdCppEngine`` (``is_available`` / ``version`` + a ``generate`` that builds argv, runs sd-cli,
returns the PNG path), and the pure ``select_diffusion_engine(...)`` routing decision. Everything
heavy is reached only inside ``generate`` / ``version``, so import is free and tests stay hermetic.
"""

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
from pathlib import Path
from typing import Callable, Iterator, Optional

from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    SdCppUpscaleParams,
    SdCppVideoGenParams,
    build_sd_cpp_command,
    build_sd_cpp_upscale_command,
    build_sd_cpp_video_command,
    native_speed_flags,
)

logger = logging.getLogger(__name__)

# sd-cli (sd-cli.exe on Windows); older builds shipped ``sd`` -- both probed on PATH.
_BINARY_STEM = "sd-cli"
_LEGACY_STEM = "sd"
# The first stable-diffusion.cpp release already exposed all three. Together they distinguish its
# oldest help text (before it printed the project name) from unrelated tools also called ``sd``.
_LEGACY_HELP_MARKERS = ("--negative-prompt", "--cfg-scale", "--steps")
# The persistent HTTP server target, shipped next to sd-cli in both prebuilt and cmake builds.
_SERVER_STEM = "sd-server"

# Ownership marker written by install_sd_cpp_prebuilt.install and required by setup.sh / uninstall.sh / uninstall.ps1 before they delete a tree.
OWNER_MARKER = ".unsloth-studio-owned"

# Ceiling for one native run. The native engine exists FOR slow CPU hosts: on GPU-less CI runners a 512x512 4-step Q2_K generation took 900 s on Linux and 1465 s on Windows, so a 30-minute cap killed jobs that were still progressing.
# It matches the Images page's own SETTLE_MAX_MS (6 h), so it only stops a WEDGED process from holding the lock forever; cancel_event is the user-facing abort.
NATIVE_GENERATION_TIMEOUT_S = 6 * 60 * 60.0


# sd-cli redraws its progress bar IN PLACE. Each redraw is one printf + fflush shaped
# "\r<bar> <step>/<steps> - <speed>\033[K", with a trailing newline only on the final step of a
# phase. So the carriage return LEADS the record and the erase-to-end-of-line CLOSES it.
_ANSI_ERASE = "\x1b[K"
# Any CSI escape (the erase above, plus colour runs some builds emit), stripped before a record
# reaches on_log / the error tail: an escape in the middle of a line corrupts both.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# One read1() per redraw in practice; large enough that a burst of finished lines costs one read.
_READ_CHUNK = 4096


def strip_ansi(text: str) -> str:
    """Drop CSI escape sequences (notably the ``\\033[K`` that closes every progress redraw)."""
    return _ANSI_CSI_RE.sub("", text)


def split_progress_records(buf: str) -> tuple[list[str], str]:
    """Split raw sd-cli stdout into complete records plus the still-unterminated remainder.

    A record ends at a carriage return, a newline, OR a trailing erase-to-end-of-line. That last
    terminator is the point: an in-place redraw carries no newline, and its carriage return is at
    the FRONT of the *next* redraw, so keying only on CR/LF delivers every sampling step one step
    late (and the final one only once sampling is over). ``\\033[K`` closes the record sd-cli has
    already flushed, so the step is delivered when it happens.

    Returns records in order (still containing their escapes; call ``strip_ansi``) and whatever
    trailing text is not yet terminated, which the caller carries into the next chunk.
    """
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
    """Yield cleaned sd-cli output records from ``stream`` as soon as each is flushed.

    Reads the undecoded pipe via ``buffer.read1`` so a redraw that never sends a newline is not
    stuck behind a blocking readline, decoding incrementally so a multi-byte character split
    across two reads survives. Streams without a raw ``.buffer`` (test doubles, non-pipes) fall
    back to line iteration, which still splits on CR under universal newlines and so still
    reports progress, just one redraw behind.
    """
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
    """Hard-stop an sd-cli process (and children). On POSIX it's a session leader, so kill the
    whole group; else kill just the process."""
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
    # Reap the killed child so it does not linger as a zombie: callers raise right after _terminate, so a burst of cancellations would leak process-table entries.
    try:
        proc.wait(timeout = 5)
    except Exception:  # noqa: BLE001 -- best-effort reap; never block teardown
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

    The prebuilt archives ship the shared libs next to ``sd-cli``, so prepend the binary's own dir
    to the platform library path (harmless for an already-linked local build). Every sd-cli launch
    funnels through here, so it's also the chokepoint that strips the native-path lease secret from
    the child env (an external process must not mint/verify native-path grants).
    """
    env = child_env_without_native_path_secret(os.environ if base_env is None else base_env)
    var = _lib_path_var()
    bindir = str(Path(binary).resolve().parent)
    existing = env.get(var, "")
    env[var] = bindir + (os.pathsep + existing if existing else "")
    return env


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
    # Newest install first, by mtime (tag strings don't sort numerically).
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


# Identity verdicts, keyed by the file itself rather than by the path alone, so replacing a binary
# in place re-probes it while a rebuild elsewhere on PATH is unaffected. Bounded: an Unsloth session
# sees a handful of candidates, and a runaway key set would only ever come from a path being
# rewritten under us, which is exactly the case that must not be served from here.
_IDENTITY_MEMO: dict[tuple[str, int, int, int], tuple[bool, float]] = {}
_IDENTITY_MEMO_LOCK = threading.Lock()
_IDENTITY_MEMO_MAX = 32
# How long a verdict may answer for. The key catches the replacements it can SEE, but no stat tuple
# is a content hash: on Windows ``st_ctime`` is the CREATION time, which an in-place overwrite
# preserves, so a same-sized write that also restores mtime is invisible to it. Hashing the file on
# every lookup would trade the exec this memo exists to avoid for a read of the whole binary, on a
# path walked for every load. A short life is the cheaper guarantee and it is not platform-specific:
# whatever the key misses, and whatever nobody has thought of, expires within a minute. Long enough
# for its actual job, which is the several resolutions inside one load sequence.
_IDENTITY_MEMO_TTL_S = 60.0


def _identity_key(binary: str) -> Optional[tuple[str, int, int, int]]:
    """A cache key that changes whenever ``binary``'s content is SEEN to change, or None when it
    cannot be read -- an unreadable candidate is never memoized, so a file that appears later is
    probed.

    ``st_ctime`` as well as ``st_mtime``: metadata-preserving copies (``cp -p``, ``shutil.copy2``,
    an archive carrying source timestamps) restore the modification time of the file they replace,
    so on POSIX a same-sized replacement is otherwise indistinguishable from the binary it
    overwrote, and the inode change time is not restorable that way. It is NOT a content revision
    on Windows, where the field is the creation time and survives an in-place overwrite -- hence
    the TTL above, which is what actually bounds a stale verdict."""
    try:
        st = os.stat(binary)
    except OSError:
        return None
    return (str(Path(binary).resolve(strict = False)), st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def help_text_identifies_sd_cpp(help_text: str) -> bool:
    """Whether ``--help`` output belongs to stable-diffusion.cpp.

    Identity, NOT capability: "is this the right program at all", which is a different question
    from ``sd_cpp_supports_minimax_h3``'s "does this build carry the H3 options". Accepts the
    project banner (current upstream's ``print_usage`` prints ``stable-diffusion.cpp version ...``
    first) or the full legacy option signature, which is what the pre-banner builds -- the ones
    that shipped the binary as ``sd`` -- print instead.

    Pure, so a caller that has already paid for the ``--help`` output can reuse it rather than
    spawning the binary a second time.
    """
    return "stable-diffusion.cpp" in help_text.lower() or all(
        marker in help_text for marker in _LEGACY_HELP_MARKERS
    )


def sd_cpp_binary_identifies(binary: str) -> bool:
    """``help_text_identifies_sd_cpp`` against a live ``binary``.

    Fails CLOSED: every caller is deciding whether to trust an ambiguously named executable, and a
    probe that cannot be read is no evidence that it is the one we want. Memoized per file
    revision -- discovery runs on every load and ``ensure_sd_cpp_binary`` alone resolves twice, so
    without this a candidate that hangs costs its full timeout again on each one.

    Only a DECISIVE verdict is memoized, because the key cannot see the difference. A timeout, a
    failed spawn, or a non-zero exit with nothing identifying in the output are all "could not
    tell", and none of them touches the file, so its key is unchanged -- caching that "no" would
    blacklist a genuine build for the life of the process over one slow ``--help`` under memory
    pressure, or over a missing shared library the user then installs. Same rule as
    ``utils.node_runtime``, which memoizes only an adequate result so a runtime installed after the
    first probe is still picked up.

    A clean exit that simply is not stable-diffusion.cpp IS decisive, which is the case that
    matters: Debian/Ubuntu's ``sd`` answers ``--help`` with rc 0, so the candidate this exists to
    stop re-executing is still probed exactly once.
    """
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
    # Identifying output settles it whatever the exit code (old builds print usage and exit 1).
    # Otherwise only a clean exit is evidence of anything; rc 127 from the dynamic loader is not.
    decisive = identified or returncode == 0
    if key is not None and decisive:
        with _IDENTITY_MEMO_LOCK:
            if len(_IDENTITY_MEMO) >= _IDENTITY_MEMO_MAX:
                _IDENTITY_MEMO.clear()
            _IDENTITY_MEMO[key] = (identified, time.monotonic())
    return identified


def _is_legacy_sd_cpp_binary(binary: str) -> bool:
    """Whether an ambiguous PATH executable named ``sd`` identifies as stable-diffusion.cpp.

    The PATH fallback is the one hop that picks a candidate purely by filename, and ``sd`` is a
    name Debian and Ubuntu already ship an unrelated find-and-replace utility under, so accepting
    it on the name alone pointed native diffusion at the wrong program AND suppressed the managed
    install (#8507). Rejecting one is read-only: the unrelated command is left exactly as it is.
    """
    identified = sd_cpp_binary_identifies(binary)
    if not identified:
        logger.warning(
            "ignoring PATH executable %s named sd because its --help output does not identify "
            "stable-diffusion.cpp",
            binary,
        )
    return identified


def managed_install_root() -> Path:
    """The directory the prebuilt installer owns, so callers can tell an Unsloth-managed binary
    from a user-supplied one (SD_CLI_PATH / UNSLOTH_SD_CPP_PATH / PATH / an in-tree build).

    Only a copy under this root may be reinstalled over: replacing anything else would delete
    a build the user chose. Honors UNSLOTH_STUDIO_HOME / STUDIO_HOME like the installer, so
    side-by-side Unsloth instances stay isolated.

    ``<studio home>/stable-diffusion.cpp``, which is where every other managed component lives
    (``default_managed_llama_dir``, ``managed_whisper_dir``, ``managed_node_dir`` all place their
    tree *under* the Unsloth home). The legacy default home ``~/.unsloth/studio`` keeps mapping to
    ``~/.unsloth/stable-diffusion.cpp`` so existing installs are still found."""
    return _studio_component_root("stable-diffusion.cpp")


def _studio_component_root(name: str) -> Path:
    """``<studio home>/<name>``, or the legacy ``~/.unsloth/<name>`` when no custom home is set
    (or the home *is* the legacy ``~/.unsloth/studio``). The home is expanded and made absolute
    first: a relative ``UNSLOTH_STUDIO_HOME`` must not leave the root relative, because the
    process' working directory can change and would silently move the managed tree."""
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
    """The pre-fix managed root, ``<studio home>/../stable-diffusion.cpp``, or None.

    Older builds derived the sd.cpp root from the *parent* of the Unsloth home, which put the tree
    outside the Unsloth home entirely. Two problems: a relative ``UNSLOTH_STUDIO_HOME`` collapsed
    that parent to the working directory, so an unrelated ``stable-diffusion.cpp`` checkout sitting
    there became "the managed install" and the installer refused to run; and it disagreed with
    every other component, which install under the home.

    Kept only so a tree an older build really did install still resolves. Returned solely when it
    carries the ownership marker, so a checkout that merely happens to sit next to the Unsloth home
    is never adopted.

    The LEXICAL parent first, because that is the one the old code took: ``Path(home).parent`` does
    not resolve symlinks, so for a home under a symlinked directory the tree an older build really
    created sits next to the link, not next to its target. Resolving first looked in the wrong
    place, re-downloaded the bundle and left the old install orphaned from uninstall as well. The
    resolved parent is still tried after it, for a home reached through a link the other way
    around."""
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
    """``<repo_root>/stable-diffusion.cpp``, the developer build the finder falls back to, or None
    when the layout above this file is not what we expect. Named rather than inlined so tests can
    point it somewhere empty: a developer with a real in-tree build must not change what the
    discovery tests assert."""
    try:
        return Path(__file__).resolve().parents[4] / "stable-diffusion.cpp"
    except (OSError, IndexError):
        return None


def is_managed_binary(binary: Optional[str]) -> bool:
    """True when ``binary`` is a copy the installer may replace: under the installer-owned root
    (see managed_install_root) AND that root carries the installer's ownership marker.

    The marker is the SAME definition of "ours" the installer and the uninstaller use --
    ``install_sd_cpp_prebuilt.install`` writes it before any extraction and refuses a pre-existing
    non-empty target without it, and uninstall.sh/.ps1 keep an unmarked directory. Path alone is not
    enough, because "stable-diffusion.cpp" is exactly what a ``git clone`` of leejet's repo produces,
    so a user may keep their own build (or point UNSLOTH_SD_CPP_PATH) at the default path.
    Deleting out of an unmarked root would take a file we are then refused permission to reinstall:
    the repair unlinks sd-server, install() rejects the now still-non-empty unmarked directory, and
    the user is left with no binary at all and no way back."""
    return owning_managed_root(binary) is not None


def owning_managed_root(binary: Optional[str]) -> Optional[Path]:
    """The installer-owned root ``binary`` lives under, or None when it is not ours.

    Both locations are checked, current first, because a tree an older build installed beside the
    Unsloth home is still discovered by the finder. Callers that read per-install state (the
    accelerator record) must read it from the root the binary is actually in: reading the current
    root while the binary came from the legacy one reports "unrecorded", which a GPU target treats
    as a mismatch and answers by re-downloading a bundle that is already installed."""
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
    """Shared finder for the stable-diffusion.cpp binaries (mirrors the llama.cpp finder).

    Order: (1) ``direct_env`` binary path; (2) ``UNSLOTH_SD_CPP_PATH`` install dir; (3) the default
    install root (honors ``UNSLOTH_STUDIO_HOME`` / ``STUDIO_HOME``, else ``~/.unsloth/...``);
    (4) ``./stable-diffusion.cpp`` in-tree build; (5) ``path_stems`` on PATH.
    """
    # 1. Direct binary path.
    env_bin = os.environ.get(direct_env)
    if env_bin and Path(env_bin).is_file():
        return env_bin

    # 2. Custom install dir.
    custom = os.environ.get("UNSLOTH_SD_CPP_PATH")
    if custom:
        hit = _first_file(_layout_candidates(Path(custom), layout_stem))
        if hit:
            return hit

    # 3. Default install root: <studio home>/stable-diffusion.cpp (honors UNSLOTH_STUDIO_HOME / STUDIO_HOME like the installer so side-by-side Unsloth instances stay isolated), else ~/.unsloth/....
    default_root = managed_install_root()
    hit = _first_file(_layout_candidates(default_root, layout_stem))
    if hit:
        return hit

    # 3b. A tree an older build installed beside the Unsloth home. Marker-gated (see legacy_sibling_install_root), so only a real previous install is picked up here.
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

    # 5. PATH.
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
            res = subprocess.run(
                [self.binary, "--version"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = timeout,
                check = False,
                env = runtime_env(self.binary),
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
        """Run one ``sd-cli`` generation; return the written image path.

        ``native_speed`` ("default"/"max") adds sd.cpp's speed flags, de-duplicated against the
        offload flags. Raises ``RuntimeError`` on a missing binary, nonzero exit, or no output.
        ``on_log`` receives each progress line; ``cancel_event`` is polled and, when set, kills the
        process tree and raises ``SdCppCancelled``.
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
            cmd,
            output_path,
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
        # Drop a stale file so the post-run is_file() check proves THIS run produced the image.
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
        logger.info("sd-cli run: %s", " ".join(cmd))

        t0 = time.time()
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
            # Bind the child to the parent's lifetime (PR_SET_PDEATHSIG) so a parent crash cannot orphan sd-cli holding VRAM/RAM.
            **child_popen_kwargs(),
        )
        # The kwargs above are empty on macOS, so record it too: a crash mid-generation
        # would otherwise leave sd-cli holding VRAM with nothing able to find it.
        adopt_pid(proc.pid)
        # Drain stdout on a reader thread so the timeout holds even when the child hangs WITHOUT printing (a plain `for line in proc.stdout` blocks until EOF). Lines, then a None sentinel, go to a queue the main loop polls against a wall-clock deadline.
        # iter_sd_cpp_records also splits sd-cli's in-place progress redraws, which carry no newline of their own, so sampling progress reaches on_log while sampling is still running.
        tail: list[str] = []
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
                if len(tail) > 40:
                    tail.pop(0)
                if on_log is not None:
                    on_log(line)
            ret = proc.wait(timeout = 5.0)
        finally:
            if proc.poll() is None:
                _terminate(proc)
            # Only once it has actually exited: a pid still running has to stay
            # recorded, or the next startup has no handle on it.
            if proc.poll() is not None:
                forget_pid(proc.pid)

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

# Backends diffusers serves well with GPU acceleration; everything else is native-engine territory.
_GPU_BACKENDS = frozenset({"cuda", "rocm", "xpu"})


def select_diffusion_engine(
    backend: str,
    *,
    native_available: bool,
    prefer_native: bool = False,
) -> str:
    """Choose the engine for a resolved device ``backend``.

    ``prefer_native`` + an available binary always wins (force native even on CUDA). CPU / MPS route
    to sd.cpp when the binary is available, else diffusers. CUDA / ROCm / XPU stay on diffusers.
    """
    if prefer_native and native_available:
        return ENGINE_SD_CPP
    if backend not in _GPU_BACKENDS and native_available:
        return ENGINE_SD_CPP
    return ENGINE_DIFFUSERS

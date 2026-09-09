# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Warm the uv cache for the next update, without touching the managed venv.

The desktop runs `unsloth studio prefetch-update` in the background while the user
keeps working. It resolves what the next update would install, downloads those
wheels into a throwaway `--target` directory, and leaves them in the shared uv
cache. The swap is the ordinary `studio update` at restart: it finds every wheel
already cached and does no network work of its own.

Nothing here mutates the live environment, so this module never takes the runtime
gate, never runs the idle scan, and is never wrapped in the Windows launcher
transaction. Killing it at any point leaves cache entries and an owned directory
without a marker, which the next prefetch wipes.

stdlib only, so it stays importable from a CLI started with `python -I`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PREFETCH_DIR_NAME = ".update-prefetch"
MARKER_NAME = "PREFETCHED.json"
# Same sentinel the installer writes over any tree it may later delete, so a
# directory somebody else put at this path is refused rather than removed.
OWNED_MARKER = ".unsloth-studio-owned"
LOCK_NAME = ".prefetch.lock"
SITE_DIR_NAME = "site"
MARKER_SCHEMA = 1
# The wheels are ~52 MB today; the floor is for the unpack, the cache copy and
# whatever the requirement files pull in, not for the two core distributions.
MIN_FREE_BYTES = 1024 * 1024 * 1024
# Distinct from 1 so the desktop can tell "a prefetch is already running" from a
# prefetch that failed, and offer nothing instead of an error.
EXIT_BUSY = 3

SUBPROCESS_TIMEOUT_SECONDS = 1800
# Wall clock for the whole run, well under update.rs's two-hour cap on the child.
# Without it, eighteen uv calls at the per-call timeout add up to nine hours, and the
# desktop shows "Preparing update" with no button for every one of them: the pill only
# offers Restart once it is ready, and the settings row is disabled while it prepares.
# Reaching this is not a failure -- the core packages are already cached by then, and
# whatever is left is what the update downloads, as it always did.
BUDGET_SECONDS = 20 * 60

# Mirror of studio/install_manifest.py:TRACKED_REQUIREMENT_FILES. Duplicated
# rather than imported: install_manifest lives in the INSTALLED (old) tree and
# would digest the old files, and the point of this record is the NEW wheel's.
TRACKED_REQUIREMENT_FILES: Tuple[str, ...] = (
    "studio.txt",
    "base.txt",
    "extras.txt",
    "extras-no-deps.txt",
    "no-torch-runtime.txt",
    "single-env/data-designer-deps.txt",
    "single-env/data-designer.txt",
)

# The dependency pass in studio/install_python_stack.py, in its own order, with
# the `--no-deps` it uses for each. triton-kernels.txt is deliberately absent: it
# names a git revision, not a wheel, so a dry-run of it is a clone.
REQUIREMENT_PASS: Tuple[Tuple[str, bool], ...] = (
    ("base.txt", False),
    ("no-torch-runtime.txt", True),
    ("extras.txt", False),
    ("extras-no-deps.txt", True),
    ("studio.txt", False),
    ("single-env/data-designer-deps.txt", False),
    ("single-env/data-designer.txt", True),
    ("diffusers-pin.txt", False),
)

# install_manifest.NO_TORCH_MARKER, for the same reason as the tuple above.
NO_TORCH_MARKER = ".unsloth-no-torch"

# Mirrors of install_python_stack.py:NO_TORCH_SKIP_PACKAGES and
# WINDOWS_SKIP_PACKAGES. The installer drops these lines from a requirement file
# before it installs it (`_filter_requirements`), so resolving the unfiltered file
# describes an install nobody performs: on a GGUF-only machine openai-whisper and
# librosa drag the whole CUDA stack into the plan, and prefetching that is gigabytes
# of downloads the update will never use.
NO_TORCH_SKIP_PACKAGES = frozenset(
    {
        "torch-stoi",
        "timm",
        "torchcodec",
        "torch-c-dlpack-ext",
        "openai-whisper",
        "librosa",
    }
)
WINDOWS_SKIP_PACKAGES = frozenset({"triton_kernels"})

# Mirror of install_python_stack.py:SDIST_ONLY_PACKAGES: requirements with no wheel
# on PyPI at any version, which the installer builds from source
# (`_sdist_only_build_args`). The prefetch fetches with `--only-binary :all:`, and uv
# refuses the WHOLE command when one pin has no wheel, so leaving these in the pin
# list loses the entire file rather than the one package. Building them here would be
# a compiler run in the background for a wheel the update builds anyway, so they are
# dropped and left to swap time.
SDIST_ONLY_PACKAGES = frozenset(
    {
        "openai-whisper",
        "argbind",
        "randomname",
        "antlr4-python3-runtime",
        # _extras_sdist_only_packages adds this on macOS cp314+; naming it everywhere
        # costs a pin the update would build in either case.
        "mecab",
    }
)

VENV_NAME = "unsloth_studio"

_UV_TRUE = ("1", "true", "yes", "on")


class PrefetchError(RuntimeError):
    """The prefetch could not finish. The classic update still works."""


class PrefetchBusy(RuntimeError):
    """Another prefetch holds the lock."""


class PrefetchSkipped(RuntimeError):
    """Nothing to prepare on this install. Exit 0 and write no marker."""


# ── Layout ──


def prefetch_root(studio_home: Path) -> Path:
    return studio_home / PREFETCH_DIR_NAME


def marker_path(studio_home: Path) -> Path:
    return prefetch_root(studio_home) / MARKER_NAME


def site_dir(studio_home: Path) -> Path:
    return prefetch_root(studio_home) / SITE_DIR_NAME


def managed_venv(studio_home: Path) -> Path:
    return studio_home / VENV_NAME


# ── Lock ──


@contextlib.contextmanager
def prefetch_lock(studio_home: Path) -> Iterator[None]:
    """Refuse a second prefetch rather than queue it.

    Non-blocking on purpose: the caller is a background task with nothing to
    wait for, and a queued second run would only redo the first one's work.
    """
    studio_home.mkdir(parents = True, exist_ok = True)
    handle = (studio_home / LOCK_NAME).open("a+b")
    try:
        if sys.platform != "win32":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise PrefetchBusy(str(handle.name)) from exc
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return

        import msvcrt

        # Byte 0 of a file nobody reads: msvcrt has no whole-file lock, and a
        # single byte is enough for mutual exclusion between two prefetches.
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise PrefetchBusy(str(handle.name)) from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


# ── uv commands ──


def _uv_safe_path(path: Path) -> str:
    """install_python_stack.py:_uv_safe_path's rule, for the same reason.

    uv truncates a path at the first space in some positions (astral-sh/uv#6503),
    so a Studio root under `C:\\Program Files` needs the short form there.
    """
    text = str(path)
    if " " not in text or platform.system() != "Windows":
        return text
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(len(text) + 260)
        length = ctypes.windll.kernel32.GetShortPathNameW(text, buffer, len(buffer))
        if length and length < len(buffer):
            return buffer.value
    except Exception:
        pass
    return text


def _torch_backend_argument(cmd: Sequence[str]) -> List[str]:
    """_build_uv_cmd's rule: never on a pinned index, it would defeat the pin."""
    backend = os.environ.get("UV_TORCH_BACKEND", "")
    if not backend:
        return []
    if any(arg in ("--index-url", "--default-index") for arg in cmd):
        return []
    return [f"--torch-backend={backend}"]


def core_dry_run_command(
    python: Path,
    *,
    floor: str,
    constraints: Optional[Path],
    no_torch: bool = False,
    use_system: bool = False,
    uv: str = "uv",
) -> List[str]:
    """The core step's argv from install_python_stack.py, plus `--dry-run`.

    Same shape, same order, same flags: the plan this returns has to be the plan
    the update will act on, and any drift between the two makes the prefetch warm
    the wrong wheels. `--no-cache-dir` is absent because
    `_translate_pip_args_for_uv` drops it on the uv path.

    `no_torch` is the installer's own branch, and it is load bearing rather than
    cosmetic: PyPI metadata makes torch a hard dependency of unsloth, so without
    `--no-deps` the resolver plans torch and every nvidia wheel behind it against
    a venv that deliberately has none of them. Measured: a 24-package plan and
    2.7 GB downloaded on a GGUF-only install whose update installs two wheels.
    """
    cmd = [uv, "pip", "install"]
    if use_system:
        cmd.append("--system")
    cmd.extend(["--python", str(python)])
    cmd.append("--dry-run")
    if no_torch:
        cmd.append("--no-deps")
    spec = f"unsloth>={floor}" if floor else "unsloth"
    cmd.extend(
        [
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            spec,
            "unsloth-zoo",
        ]
    )
    cmd.extend(_torch_backend_argument(cmd))
    if constraints is not None and constraints.is_file():
        cmd.extend(["-c", _uv_safe_path(constraints)])
    return cmd


# A requirement line's distribution name: everything before the first marker,
# extra, comparison or comment character.
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def effective_requirements(requirement: Path, skip: Iterable[str], work_dir: Path) -> Path:
    """`_filter_requirements`'s output, or the file itself when nothing is skipped.

    Mirrors install_python_stack.py:_filter_requirements, including the rule that a
    `-r`/`-c` include or an option line is copied through untouched.
    """
    skipped = {canonical_name(name) for name in skip}
    if not skipped:
        return requirement
    try:
        lines = requirement.read_text(encoding = "utf-8").splitlines(keepends = True)
    except OSError:
        return requirement
    kept: List[str] = []
    dropped = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "-")):
            match = _REQUIREMENT_NAME.match(stripped)
            if match and canonical_name(match.group(1)) in skipped:
                dropped = True
                continue
        kept.append(line)
    if not dropped:
        return requirement
    # Beside the source, as _filter_requirements does and for the same reason: a
    # relative `-r`/`-c` include is copied through and resolves against the file's own
    # directory. `work_dir` is the fallback for a tree this process cannot write to.
    filtered = requirement.with_name(f".{requirement.stem}-filtered.txt")
    try:
        filtered.write_text("".join(kept), encoding = "utf-8")
        return filtered
    except OSError:
        pass
    work_dir.mkdir(parents = True, exist_ok = True)
    filtered = work_dir / requirement.name
    try:
        filtered.write_text("".join(kept), encoding = "utf-8")
    except OSError:
        # Resolving the unfiltered file is still better than not preparing at all;
        # the pins it plans are recorded and whatever cannot be fetched is skipped.
        return requirement
    return filtered


def requirement_dry_run_command(
    python: Path,
    requirement: Path,
    *,
    constraints: Optional[Path],
    no_deps: bool = False,
    use_system: bool = False,
    uv: str = "uv",
) -> List[str]:
    cmd = [uv, "pip", "install"]
    if use_system:
        cmd.append("--system")
    cmd.extend(["--python", str(python)])
    cmd.append("--dry-run")
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(_torch_backend_argument(cmd))
    if constraints is not None and constraints.is_file():
        cmd.extend(["-c", _uv_safe_path(constraints)])
    cmd.extend(["-r", _uv_safe_path(requirement)])
    return cmd


def fetch_command(
    python: Path,
    target: Path,
    pins: Sequence[str],
    *,
    only_binary: bool = False,
    use_system: bool = False,
    uv: str = "uv",
) -> List[str]:
    """Download `pins` into `target`, warming the shared cache on the way.

    `--python` so uv picks the same wheel tags the update will, `--target` so not
    one byte reaches the venv, `--no-deps` because the plan is already complete.
    """
    cmd = [uv, "pip", "install"]
    if use_system:
        cmd.append("--system")
    cmd.extend(["--python", str(python)])
    cmd.extend(["--target", _uv_safe_path(target)])
    cmd.append("--no-deps")
    if only_binary:
        cmd.extend(["--only-binary", ":all:"])
    cmd.extend(_torch_backend_argument(cmd))
    cmd.extend(pins)
    return cmd


# ── Plan parsing ──

# uv writes its plan on stderr, one entry per line, as ` + name==version` for an
# install and ` - name==version` for the removal it replaces. A direct URL adds a
# trailing ` (from ...)`, so only the first token is the pin.
_PLAN_LINE = re.compile(r"^\s*\+\s+(?P<pin>\S+)\s*(?:\(.*\))?\s*$")
# uv prints this immediately above the plan. It is the only way to tell "nothing to
# do" from "the plan came out in a shape this parser does not know": both leave the
# dict empty, and one of them is a prefetch that reports a warm cache and warmed
# nothing.
_PLAN_COUNT = re.compile(r"^\s*Would install (?P<count>\d+) packages?\s*$", re.M)


def planned_install_count(output: str) -> Optional[int]:
    """How many installs uv said it would do, or None when it did not say."""
    match = _PLAN_COUNT.search(output.replace("\r\n", "\n").replace("\r", "\n"))
    return int(match.group("count")) if match else None


def plan_is_readable(output: str, planned: Dict[str, str]) -> bool:
    """False when uv announced installs and not one line parsed as a pin at all.

    Deliberately measured against the lines, not against `planned`: dropping every
    local-tag pin can legitimately empty the dict for a plan that was read perfectly
    (a torch-only plan is one line, and it carries `+cu128`).
    """
    if not planned_install_count(output):
        return True
    if planned:
        return True
    return any(
        _PLAN_LINE.match(line) and "==" in line
        for line in output.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )


def parse_dry_run_plan(output: str) -> Dict[str, str]:
    """Planned installs as {name: version}, in the order uv reported them.

    `-` lines are removals, not installs. A version carrying a local tag
    (`2.9.0+cu128`) is dropped: it comes from a pinned index or a torch backend
    that this command cannot reproduce as a bare `name==version`, and asking for
    one anyway resolves to a different wheel or to nothing.
    """
    planned: Dict[str, str] = {}
    for raw in output.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _PLAN_LINE.match(raw)
        if not match:
            continue
        pin = match.group("pin").split()[0]
        if "==" not in pin:
            continue
        name, _, version = pin.partition("==")
        name = name.strip()
        version = version.strip()
        if not name or not version or "+" in version:
            continue
        planned.setdefault(canonical_name(name), version)
    return planned


def canonical_name(name: str) -> str:
    """PEP 503 normalisation, so unsloth_zoo and unsloth-zoo compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pins_from_plan(planned: Dict[str, str], *, only_binary: bool = False) -> List[str]:
    """The plan as `name==version` pins, minus what cannot be fetched as a wheel."""
    return [
        f"{name}=={version}"
        for name, version in planned.items()
        if not (only_binary and name in SDIST_ONLY_PACKAGES)
    ]


# ── Version comparison (PEP 440 release segment only) ──


def _release_tuple(version: str) -> Tuple[int, ...]:
    head = version.split("+", 1)[0]
    parts: List[int] = []
    for chunk in head.split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def version_meets_floor(version: str, floor: str) -> bool:
    """True when `version` is at least `floor` on the release segment.

    Deliberately coarse: the floor the desktop passes is a release version, and a
    full PEP 440 comparison here would need packaging, which this module cannot
    import. A `.postN` of the floor compares equal and passes, which is right.
    """
    if not floor:
        return True
    left = _release_tuple(version)
    right = _release_tuple(floor)
    if not left or not right:
        return True
    return left >= right


# ── Marker ──


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> Optional[str]:
    try:
        return _digest_bytes(path.read_bytes())
    except OSError:
        return None


def requirement_digests(req_root: Path) -> Dict[str, str]:
    digests: Dict[str, str] = {}
    for name in TRACKED_REQUIREMENT_FILES:
        digest = digest_file(req_root / name)
        if digest is not None:
            digests[name] = digest
    return digests


def core_record_digests(target: Path, names: Iterable[str]) -> Dict[str, str]:
    """sha256 of each fetched distribution's RECORD.

    The swap does not read these; they are what lets a later reader say whether
    the directory it found is the one this marker describes, without unpacking it.
    """
    wanted = {canonical_name(name) for name in names}
    digests: Dict[str, str] = {}
    try:
        entries = sorted(target.iterdir())
    except OSError:
        return digests
    for entry in entries:
        if not entry.name.endswith(".dist-info") or not entry.is_dir():
            continue
        distribution = canonical_name(entry.name[: -len(".dist-info")].rsplit("-", 1)[0])
        if distribution not in wanted:
            continue
        digest = digest_file(entry / "RECORD")
        if digest is not None:
            digests[entry.name] = digest
    return digests


def write_marker(studio_home: Path, payload: dict) -> None:
    """Atomic: a reader never sees a half-written plan.

    Written last, after every wheel is in the cache, so the marker's presence is
    the statement that the cache is warm.
    """
    marker = marker_path(studio_home)
    marker.parent.mkdir(parents = True, exist_ok = True)
    temporary = marker.with_name(marker.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")
    os.replace(temporary, marker)


def read_marker(studio_home: Path) -> Optional[dict]:
    try:
        data = json.loads(marker_path(studio_home).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# The environment variable through which `unsloth studio update` names the core pins a
# current prefetch cached. install_python_stack.py reads it when the index cannot be
# reached during the core step and installs those pins from the uv cache with --offline,
# instead of falling through to a pip resolution that needs the index it has not got.
CORE_PINS_ENV = "UNSLOTH_PREFETCHED_CORE_PINS"


def prefetched_core_pins(marker: Optional[dict]) -> list:
    """`name==version` for every core package a marker planned, in plan order.

    Empty for a marker that planned nothing (`noop`) or one that is not a marker at all.
    Callers pair this with marker_is_current: a plan is only worth naming when the
    cache it was fetched into is the cache the update is about to read.
    """
    plan = (marker or {}).get("core_plan") if isinstance(marker, dict) else None
    if not isinstance(plan, dict):
        return []
    pins = []
    for name, version in plan.items():
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        name, version = name.strip(), version.strip()
        if not name or not version or any(ch.isspace() for ch in name + version):
            continue
        pins.append(f"{name}=={version}")
    return pins


def marker_is_current(
    marker: Optional[dict],
    *,
    floor: str = "",
    python: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    """Whether a prefetch on disk still describes the update about to run."""
    if not isinstance(marker, dict):
        return False
    if marker.get("schema") != MARKER_SCHEMA:
        return False
    if marker.get("state") not in ("ready", "partial", "noop"):
        return False
    if python is not None and marker.get("python") != python:
        return False
    if cache_dir is not None and marker.get("cache_dir") != cache_dir:
        return False
    if floor:
        backend = marker.get("backend_version")
        if marker.get("state") == "noop":
            # Nothing was planned, so the floor has to be met by what is installed.
            backend = marker.get("installed_backend_version") or backend
        if not isinstance(backend, str) or not version_meets_floor(backend, floor):
            return False
    return True


# ── Discard ──


def _is_owned(root: Path) -> bool:
    return (root / OWNED_MARKER).is_file()


def discard(studio_home: Path) -> bool:
    """Remove the prefetch directory, but only one this code wrote.

    A directory without the owned marker is somebody else's; refusing it is the
    same rule `_assert_studio_owned_or_absent` applies in the installer.
    """
    root = prefetch_root(studio_home)
    if not root.exists():
        return False
    if not _is_owned(root):
        return False
    shutil.rmtree(root, ignore_errors = True)
    return not root.exists()


def discard_after_update(studio_home: Path) -> bool:
    """Called once an update has succeeded: the cache is warm, the copy is spent.

    Never raises. A prefetch left behind costs disk, not correctness, and the
    next prefetch wipes it anyway.
    """
    try:
        return discard(studio_home)
    except Exception:
        return False


# ── Preflight ──


def _installed_version(name: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _is_editable_install(name: str = "unsloth") -> bool:
    """An editable or local checkout has nothing to prefetch from an index."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return False
    except Exception:
        return False
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return False
    if not raw:
        return False
    try:
        record = json.loads(raw)
    except ValueError:
        return False
    info = record.get("dir_info")
    return isinstance(info, dict) and bool(info.get("editable"))


def uv_no_cache_requested() -> bool:
    return (os.environ.get("UV_NO_CACHE") or "").strip().lower() in _UV_TRUE


def _free_bytes(path: Path) -> Optional[int]:
    probe = path
    for _ in range(8):
        if probe.exists():
            break
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError:
        return None


# ── Runner ──


def _run(cmd: Sequence[str], env: Optional[dict]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        timeout = SUBPROCESS_TIMEOUT_SECONDS,
    )


def _combined(result: subprocess.CompletedProcess) -> str:
    """uv reports the plan on stderr; read both so a future uv cannot move it."""
    return f"{result.stdout or ''}\n{result.stderr or ''}"


# ── Finding uv ──


# The installer scripts run `uv` as a bare token, and get away with it because
# they prepend the directory they installed it into to PATH before they start
# python (setup.ps1 `$env:PATH = "$destDir;$env:PATH"`, setup.sh
# `export PATH="$_siup_dest:$PATH"`). The prefetch is spawned straight from the
# desktop shell instead, and on Windows that shell can be holding a PATH from
# before the install wrote HKCU\Environment\Path, so the token resolves to
# nothing and the whole prefetch skips on a machine that has uv.
#
# So look where the installer put it. This is astral's destination priority,
# which install.ps1 and install.sh both compute the same way and which
# install.ps1 re-probes verbatim when a fresh uv hides behind an older one.
# Nothing installs uv into the Studio root or the managed venv, and nothing in
# this codebase falls back to `python -m uv`, so there is no Studio-owned copy to
# prefer: these directories are the whole search.
def _uv_search_dirs(env: dict) -> List[Path]:
    dirs: List[Path] = []

    def add(value: Optional[str]) -> None:
        if not value:
            return
        path = Path(value)
        if path not in dirs:
            dirs.append(path)

    for name in ("UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL", "XDG_BIN_HOME"):
        add(env.get(name))
    data_home = env.get("XDG_DATA_HOME")
    if data_home:
        # `$XDG_DATA_HOME/../bin`, as both installers spell it.
        add(str(Path(data_home).parent / "bin"))
    home = env.get("USERPROFILE") if platform.system() == "Windows" else None
    home = home or env.get("HOME") or os.path.expanduser("~")
    add(str(Path(home) / ".local" / "bin"))
    local_app_data = env.get("LOCALAPPDATA")
    if platform.system() == "Windows" and local_app_data:
        # The winget shim, which install.ps1 names in the same scan.
        add(str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links"))
    return dirs


def _uv_runs(candidate: Path, env: Optional[dict]) -> bool:
    """install.ps1's Test-UvCandidateVersion, minus the floor.

    A file at the right path that cannot answer `--version` is not a uv the
    update will use either, and running it is the only way to know.
    """
    try:
        return _run([str(candidate), "--version"], env).returncode == 0
    except Exception:
        return False


def locate_uv(env: Optional[dict] = None) -> Tuple[Optional[str], List[Path]]:
    """The uv the update's core step will run, and every place this looked.

    PATH first, because a uv on PATH is exactly the one `install_python_stack.py`
    resolves; the directories are the fallback for a shell that never saw the
    installer's PATH.
    """
    environment = dict(env) if env is not None else dict(os.environ)
    searched = _uv_search_dirs(environment)
    # Bare, one argument, as install_python_stack.py:_bootstrap_uv calls it.
    found = shutil.which("uv")
    if found:
        return found, searched
    name = "uv.exe" if platform.system() == "Windows" else "uv"
    for directory in searched:
        candidate = directory / name
        if candidate.is_file() and _uv_runs(candidate, environment):
            return str(candidate), searched
    return None, searched


def _uv_version(uv: str, env: Optional[dict]) -> Optional[str]:
    try:
        result = _run([uv, "--version"], env)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def _no_torch(venv: Path) -> bool:
    return (venv / NO_TORCH_MARKER).is_file()


def _installed_studio_root(venv: Path) -> Optional[Path]:
    """`studio/` inside the LIVE venv, which holds the constraints the plan uses."""
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for site in sorted(venv.glob(pattern)):
            candidate = site / "studio"
            if candidate.is_dir():
                return candidate
    return None


def _fetched_studio_root(target: Path) -> Optional[Path]:
    candidate = target / "studio"
    return candidate if candidate.is_dir() else None


def run(
    *,
    studio_home: Path,
    floor: str = "",
    shell_version: Optional[str] = None,
    env: Optional[dict] = None,
    echo: Callable[[str], None] = print,
    python: Optional[Path] = None,
) -> dict:
    """Prepare the next update in the background. Returns the marker payload.

    Raises `PrefetchSkipped` for an install that has nothing to prepare (exit 0,
    no marker), `PrefetchBusy` when another prefetch holds the lock, and
    `PrefetchError` for a real failure.
    """
    interpreter = Path(python) if python is not None else Path(sys.executable)
    venv = Path(sys.prefix)

    def step(text: str) -> None:
        echo(f"[TAURI:STEP] {text}")

    # 1. Preflight. Every branch here is "this install prepares nothing", not a
    #    failure: the classic update still works on all of them.
    if not (venv / "pyvenv.cfg").is_file():
        raise PrefetchSkipped("not running from a managed environment")
    expected = managed_venv(studio_home)
    try:
        same_venv = venv.resolve() == expected.resolve()
    except OSError:
        same_venv = venv == expected
    if not same_venv:
        raise PrefetchSkipped("not running from the managed Studio environment")
    if _is_editable_install():
        raise PrefetchSkipped("unsloth is installed from a local checkout")
    if (os.environ.get("STUDIO_LOCAL_INSTALL") or "").strip() == "1":
        raise PrefetchSkipped("a local install has no index to prepare from")
    if uv_no_cache_requested():
        raise PrefetchSkipped("UV_NO_CACHE leaves nothing behind to prepare")
    # Built here rather than after the disk check: the uv search reads the same
    # environment the child will run under, so it has to exist by now.
    child_env = dict(env) if env is not None else dict(os.environ)
    uv, searched_for_uv = locate_uv(child_env)
    if uv is None:
        # Name the places, so a machine that has uv somewhere else says so in one
        # line instead of leaving the reader to guess what "not available" means.
        where = ", ".join(str(directory) for directory in searched_for_uv)
        raise PrefetchSkipped(f"uv is not available (looked on PATH and in {where})")

    root = prefetch_root(studio_home)
    free = _free_bytes(root)
    if free is not None and free < MIN_FREE_BYTES:
        raise PrefetchError(
            f"not enough free space to prepare an update ({free} bytes free, "
            f"{MIN_FREE_BYTES} needed)"
        )

    cache_dir = child_env.get("UV_CACHE_DIR")

    if root.exists() and not _is_owned(root):
        raise PrefetchError(f"{root} exists and was not created by Unsloth")
    shutil.rmtree(root, ignore_errors = True)
    root.mkdir(parents = True, exist_ok = True)
    # Written before anything else lands in it, so an interrupted run still
    # leaves a directory the next prefetch is allowed to remove.
    (root / OWNED_MARKER).write_text("", encoding = "utf-8")
    target = site_dir(studio_home)
    target.mkdir(parents = True, exist_ok = True)

    live_studio = _installed_studio_root(venv)
    live_constraints = (
        live_studio / "backend" / "requirements" / "single-env" / "constraints.txt"
        if live_studio is not None
        else None
    )
    if live_constraints is not None and not live_constraints.is_file():
        live_constraints = None

    # 2. Resolve. The plan is what the update's core step would do, asked of the
    #    live venv so anything already satisfied is absent from it.
    deadline = time.monotonic() + BUDGET_SECONDS
    # The installer's own branch, read from the venv the update will run against.
    no_torch = _no_torch(venv)
    step("prefetch resolving core packages")
    core_cmd = core_dry_run_command(
        interpreter,
        floor = floor,
        constraints = live_constraints,
        no_torch = no_torch,
        uv = uv,
    )
    try:
        resolved = _run(core_cmd, child_env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrefetchError(f"could not resolve the core packages: {exc}") from exc
    if resolved.returncode != 0:
        raise PrefetchError(
            "could not resolve the core packages: "
            + (_combined(resolved).strip()[-800:] or f"uv exited {resolved.returncode}")
        )
    core_output = _combined(resolved)
    planned = parse_dry_run_plan(core_output)
    if not plan_is_readable(core_output, planned):
        raise PrefetchError(
            "could not read the core plan uv printed: " + core_output.strip()[-800:]
        )

    installed_backend = _installed_version("unsloth")
    backend_version = planned.get("unsloth")
    zoo_version = planned.get("unsloth-zoo")

    created_at = int(time.time() * 1000)
    payload: dict = {
        "schema": MARKER_SCHEMA,
        "state": "noop",
        "backend_version": backend_version,
        "installed_backend_version": installed_backend,
        "zoo_version": zoo_version,
        "floor": floor or None,
        "shell_version": shell_version,
        "cache_dir": cache_dir,
        "python": str(interpreter),
        "python_version": platform.python_version(),
        "uv_version": _uv_version(uv, child_env),
        "core_plan": dict(planned),
        "requirements": {},
        "requirement_digests": {},
        "core_records": {},
        "created_at": created_at,
    }

    if not planned:
        # Nothing to fetch. The desktop still shows "ready": the update it would
        # run has no downloads left to do.
        #
        # Keyed off the whole plan rather than off unsloth alone: the two
        # distributions release independently, so a zoo-only bump is a real download
        # that a "noop" marker would tell the desktop it had already prepared.
        if floor and installed_backend and not version_meets_floor(installed_backend, floor):
            raise PrefetchError(
                f"the index offers no unsloth>={floor}; installed is {installed_backend}"
            )
        step("prefetch nothing to prepare")
        write_marker(studio_home, payload)
        return payload

    if floor and backend_version is not None and not version_meets_floor(backend_version, floor):
        raise PrefetchError(f"the resolved unsloth {backend_version} is below the required {floor}")

    # 3. Fetch the core plan. --target keeps every byte out of the venv; the
    #    download is what warms archive-v0 in the shared cache.
    pins = pins_from_plan(planned)
    step(f"prefetch downloading {len(pins)} core package(s)")
    fetch_cmd = fetch_command(interpreter, target, pins, uv = uv)
    try:
        fetched = _run(fetch_cmd, child_env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrefetchError(f"could not download the core packages: {exc}") from exc
    if fetched.returncode != 0:
        raise PrefetchError(
            "could not download the core packages: "
            + (_combined(fetched).strip()[-800:] or f"uv exited {fetched.returncode}")
        )
    payload["core_records"] = core_record_digests(target, planned)

    # 4. Requirement files, best effort. These come from the NEW wheel, resolved
    #    against the live venv: a file that will not resolve is recorded and left
    #    to swap time, which downloads it then. Never a failure of the prefetch.
    new_studio = _fetched_studio_root(target)
    state = "ready"
    if new_studio is None:
        payload["requirements"]["*"] = {"skipped_reason": "the new wheel ships no studio tree"}
        state = "partial"
    else:
        req_root = new_studio / "backend" / "requirements"
        payload["requirement_digests"] = requirement_digests(req_root)
        new_constraints = req_root / "single-env" / "constraints.txt"
        if not new_constraints.is_file():
            new_constraints = None
        skip = set(NO_TORCH_SKIP_PACKAGES) if no_torch else set()
        if platform.system() == "Windows":
            skip |= set(WINDOWS_SKIP_PACKAGES)
        work_dir = prefetch_root(studio_home) / "req"
        for name, no_deps in REQUIREMENT_PASS:
            if time.monotonic() >= deadline:
                # Recorded, not raised: the core packages are cached, and the files
                # left over are the ones the update would have downloaded anyway.
                payload["requirements"][name] = {"skipped_reason": "out of time"}
                state = "partial"
                continue
            if no_torch and name == "base.txt":
                continue
            if not no_torch and name == "no-torch-runtime.txt":
                continue
            requirement = req_root / name
            if not requirement.is_file():
                continue
            record = _prefetch_requirement_file(
                interpreter,
                effective_requirements(requirement, skip, work_dir),
                target = target,
                constraints = new_constraints,
                no_deps = no_deps,
                env = child_env,
                uv = uv,
                step = step,
                label = name,
            )
            payload["requirements"][name] = record
            if record.get("skipped_reason"):
                state = "partial"

    payload["state"] = state
    # 5. Last, so its presence means every wheel above is in the cache.
    write_marker(studio_home, payload)
    step(f"prefetch {state}")
    return payload


def _prefetch_requirement_file(
    interpreter: Path,
    requirement: Path,
    *,
    target: Path,
    constraints: Optional[Path],
    no_deps: bool,
    env: Optional[dict],
    uv: str,
    step: Callable[[str], None],
    label: str,
) -> dict:
    try:
        resolved = _run(
            requirement_dry_run_command(
                interpreter,
                requirement,
                constraints = constraints,
                no_deps = no_deps,
                uv = uv,
            ),
            env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"skipped_reason": f"resolve failed: {exc}"}
    if resolved.returncode != 0:
        return {
            "skipped_reason": "resolve failed: "
            + (_combined(resolved).strip()[-400:] or f"uv exited {resolved.returncode}")
        }
    output = _combined(resolved)
    planned = parse_dry_run_plan(output)
    if not plan_is_readable(output, planned):
        return {"skipped_reason": "could not read the plan uv printed"}
    if not planned:
        return {"pins": {}}
    pins = pins_from_plan(planned, only_binary = True)
    if not pins:
        # Everything this file plans is built from source; there is no wheel to warm.
        return {"pins": dict(planned)}
    step(f"prefetch downloading {len(pins)} package(s) for {label}")
    try:
        # --only-binary: a source distribution would be BUILT here, against the
        # live interpreter, which is real work in the background for a wheel the
        # update can just as well build itself.
        fetched = _run(
            fetch_command(interpreter, target, pins, only_binary = True, uv = uv),
            env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"pins": dict(planned), "skipped_reason": f"download failed: {exc}"}
    if fetched.returncode != 0:
        return {
            "pins": dict(planned),
            "skipped_reason": "download failed: "
            + (_combined(fetched).strip()[-400:] or f"uv exited {fetched.returncode}"),
        }
    return {"pins": dict(planned)}

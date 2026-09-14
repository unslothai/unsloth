# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Warm the uv cache for the next update without touching the managed venv.

Resolves what the next update would install and downloads it into a throwaway `--target`,
leaving the wheels in the shared uv cache for the ordinary `studio update` at restart.
Nothing here mutates the live environment, so no runtime gate, idle scan or launcher
transaction. stdlib only: importable from a CLI started with `python -I`.
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
OWNED_MARKER = ".unsloth-studio-owned"
LOCK_NAME = ".prefetch.lock"
SITE_DIR_NAME = "site"
MARKER_SCHEMA = 1
MIN_FREE_BYTES = 1024 * 1024 * 1024
# Distinct from 1 so the desktop can tell "already running" from a failure.
EXIT_BUSY = 3

SUBPROCESS_TIMEOUT_SECONDS = 1800
# Whole-run wall clock: per-call timeouts alone could hold "Preparing update" for hours.
BUDGET_SECONDS = 20 * 60

# install_manifest.TRACKED_REQUIREMENT_FILES, not imported: the installed one is the OLD tree's.
TRACKED_REQUIREMENT_FILES: Tuple[str, ...] = (
    "studio.txt",
    "base.txt",
    "extras.txt",
    "extras-no-deps.txt",
    "no-torch-runtime.txt",
    "single-env/data-designer-deps.txt",
    "single-env/data-designer.txt",
)

# install_python_stack.py's dependency pass and its --no-deps; no triton-kernels.txt (a git clone).
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

# install_python_stack.py's skip lists: unfiltered, a GGUF-only plan pulls in the CUDA stack.
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

# install_python_stack.py:SDIST_ONLY_PACKAGES: one wheel-less pin fails an --only-binary fetch.
SDIST_ONLY_PACKAGES = frozenset(
    {
        "openai-whisper",
        "argbind",
        "randomname",
        "antlr4-python3-runtime",
        "mecab",
    }
)

VENV_NAME = "unsloth_studio"

# commands/studio.py:_UV_TRUE, clap's boolish set: uv reads UV_NO_CACHE=t or y as true too.
_UV_TRUE = ("1", "y", "yes", "t", "true", "on")


class PrefetchError(RuntimeError):
    """The prefetch could not finish. The classic update still works."""


class PrefetchBusy(RuntimeError):
    """Another prefetch holds the lock."""


class PrefetchSkipped(RuntimeError):
    """Nothing to prepare on this install. Exit 0 and write no marker."""


def prefetch_root(studio_home: Path) -> Path:
    return studio_home / PREFETCH_DIR_NAME


def marker_path(studio_home: Path) -> Path:
    return prefetch_root(studio_home) / MARKER_NAME


def site_dir(studio_home: Path) -> Path:
    return prefetch_root(studio_home) / SITE_DIR_NAME


def managed_venv(studio_home: Path) -> Path:
    return studio_home / VENV_NAME


@contextlib.contextmanager
def prefetch_lock(studio_home: Path) -> Iterator[None]:
    """Refuse a second prefetch rather than queue it: a queued run would only redo the first."""
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


def _uv_safe_path(path: Path) -> str:
    """install_python_stack.py:_uv_safe_path: uv truncates at a space (astral-sh/uv#6503)."""
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
    """The installer's core step argv plus `--dry-run`; any drift warms the wrong wheels.

    `no_torch` adds --no-deps as the installer does: PyPI metadata makes torch a hard
    dependency of unsloth, so without it a GGUF-only install plans the whole CUDA stack.
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


_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def effective_requirements(requirement: Path, skip: Iterable[str], work_dir: Path) -> Path:
    """Mirror of install_python_stack.py:_filter_requirements; option lines pass through."""
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
    # Beside the source, so relative -r/-c includes still resolve; work_dir for an unwritable tree.
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
        # The unfiltered file still beats not preparing; what cannot be fetched is skipped.
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
    """Download `pins` into `target`, warming the cache; --python picks the update's wheel tags."""
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


# uv's plan: ` + name==version` per install (` - ` removal), a direct URL adds ` (from ...)`.
_PLAN_LINE = re.compile(r"^\s*\+\s+(?P<pin>\S+)\s*(?:\(.*\))?\s*$")
# Tells "nothing to do" apart from a plan shape this parser does not know.
_PLAN_COUNT = re.compile(r"^\s*Would install (?P<count>\d+) packages?\s*$", re.M)


def planned_install_count(output: str) -> Optional[int]:
    match = _PLAN_COUNT.search(output.replace("\r\n", "\n").replace("\r", "\n"))
    return int(match.group("count")) if match else None


def plan_is_readable(output: str, planned: Dict[str, str]) -> bool:
    """False when uv announced installs and no line parsed as a pin.

    Measured on the lines, not `planned`: dropping local-tag pins can empty a well-read plan.
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
    """Planned installs as {name: version} in uv's order.

    Local-tag versions (`+cu128`) are dropped: a bare `name==version` cannot reproduce them.
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
    return re.sub(r"[-_.]+", "-", name).lower()


def pins_from_plan(planned: Dict[str, str], *, only_binary: bool = False) -> List[str]:
    return [
        f"{name}=={version}"
        for name, version in planned.items()
        if not (only_binary and name in SDIST_ONLY_PACKAGES)
    ]


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


_VERSION_RE = re.compile(
    r"^v?(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)"
    r"(?:[._-]?(?P<pre_l>a|b|c|rc|alpha|beta|pre|preview)[._-]?(?P<pre_n>\d*))?"
    r"(?:(?:[._-]?(?:post|rev|r)[._-]?(?P<post_n>\d*))|(?:-(?P<post_implicit>\d+)))?"
    r"(?:[._-]?dev[._-]?(?P<dev_n>\d*))?"
    r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?$",
    re.IGNORECASE,
)
_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}


def _version_key(version: str) -> Optional[tuple]:
    """A PEP 440 ordering key, or None when unparseable (stdlib only, no packaging)."""
    match = _VERSION_RE.match(version.strip())
    if match is None:
        return None
    release = [int(part) for part in match.group("release").split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    pre_label = match.group("pre_l")
    pre = (_PRE_RANK[pre_label.lower()], int(match.group("pre_n") or 0)) if pre_label else None
    if match.group("post_n") is not None:
        post: Optional[int] = int(match.group("post_n") or 0)
    elif match.group("post_implicit") is not None:
        post = int(match.group("post_implicit"))
    else:
        post = None
    dev = int(match.group("dev_n") or 0) if match.group("dev_n") is not None else None
    # packaging's sentinels: a bare dev release sorts before any pre-release, a final after all.
    if pre is None and post is None and dev is not None:
        pre_key: tuple = (-1,)
    elif pre is None:
        pre_key = (1,)
    else:
        pre_key = (0, pre[0], pre[1])
    post_key = (-1,) if post is None else (0, post)
    dev_key = (1,) if dev is None else (0, dev)
    return (int(match.group("epoch") or 0), tuple(release), pre_key, post_key, dev_key)


def version_meets_floor(version: str, floor: str) -> bool:
    """True when `version` is at least `floor`; release tuples when either does not parse."""
    if not floor:
        return True
    left_key, right_key = _version_key(version), _version_key(floor)
    if left_key is not None and right_key is not None:
        return left_key >= right_key
    left = _release_tuple(version)
    right = _release_tuple(floor)
    if not left or not right:
        return True
    return left >= right


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
    """sha256 of each fetched distribution's RECORD, identifying the tree without unpacking it."""
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
    """Atomic, and written last: its presence states the cache is warm."""
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


# Pins a current prefetch cached; the installer installs them --offline when the index fails.
CORE_PINS_ENV = "UNSLOTH_PREFETCHED_CORE_PINS"


def _core_pin_source(marker: Optional[dict]) -> dict:
    """The pins a marker stands for: its plan, or for a `noop` the versions then installed."""
    if not isinstance(marker, dict):
        return {}
    plan = marker.get("core_plan")
    if isinstance(plan, dict) and plan:
        return plan
    if marker.get("state") == "noop":
        installed = marker.get("installed_core")
        if isinstance(installed, dict):
            return installed
    return {}


def prefetched_core_pins(marker: Optional[dict]) -> list:
    """`name==version` for every core package a marker stands for; pair with marker_is_current."""
    plan = _core_pin_source(marker)
    pins = []
    for name, version in plan.items():
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        name, version = name.strip(), version.strip()
        if not name or not version or any(ch.isspace() for ch in name + version):
            continue
        pins.append(f"{name}=={version}")
    return pins


def planned_core_names(marker: Optional[dict]) -> list:
    """Every name the plan pins: the offline retry installs each one, not just the core two."""
    plan = _core_pin_source(marker)
    names = []
    for name in plan:
        if isinstance(name, str) and name.strip():
            canonical = _canonical_name(name.strip())
            if canonical not in names:
                names.append(canonical)
    return names


# prefetch.rs MAX_AGE_MS.
MARKER_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000


def marker_is_current(
    marker: Optional[dict],
    *,
    floor: str = "",
    python: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> bool:
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
    created = marker.get("created_at")
    if isinstance(created, (int, float)) and (time.time() * 1000 - created) > MARKER_MAX_AGE_MS:
        return False
    if floor:
        backend = marker.get("backend_version")
        if not isinstance(backend, str) or not backend:
            # No unsloth in the plan (a zoo-only bump): the installed unsloth must meet the floor.
            backend = marker.get("installed_backend_version")
        if not isinstance(backend, str) or not version_meets_floor(backend, floor):
            return False
    return True


def plan_is_not_behind(marker: Optional[dict], installed: Dict[str, Optional[str]]) -> bool:
    """Whether every planned pin is at least what is installed: old pins offline would downgrade.

    Unknown installed versions do not count against the plan.
    """
    plan = _core_pin_source(marker)
    for name, version in plan.items():
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        current = installed.get(_canonical_name(name))
        if not current:
            continue
        planned, present = version.strip(), current.strip()
        planned_key, present_key = _version_key(planned), _version_key(present)
        if planned_key is not None and present_key is not None:
            if planned_key < present_key:
                return False
            continue
        if _release_tuple(planned) < _release_tuple(present):
            return False
        # Unparseable at the same release reads as behind: a wrong yes is a silent downgrade.
        if (
            _release_tuple(planned) == _release_tuple(present)
            and planned.lower() != present.lower()
        ):
            return False
    return True


def _canonical_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _is_owned(root: Path) -> bool:
    return (root / OWNED_MARKER).is_file()


def discard(studio_home: Path) -> bool:
    """Remove the prefetch directory, only when it carries the owned marker."""
    root = prefetch_root(studio_home)
    if not root.exists():
        return False
    if not _is_owned(root):
        return False
    shutil.rmtree(root, ignore_errors = True)
    return not root.exists()


def discard_after_update(studio_home: Path) -> bool:
    """Discard after a successful update. Never raises.

    Under the prefetch lock: removing a running prefetch's root makes it recreate the
    directory without the owned marker, which every later prefetch then refuses.
    """
    try:
        with prefetch_lock(studio_home):
            return discard(studio_home)
    except PrefetchBusy:
        return False
    except Exception:
        return False


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
    # Not stripped, as commands/studio.py:_uv_no_cache_requested: uv rejects a padded value.
    return (os.environ.get("UV_NO_CACHE") or "").lower() in _UV_TRUE


def _nearest_existing(path: Path) -> Path:
    probe = path
    for _ in range(8):
        if probe.exists():
            break
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe


def _free_bytes(path: Path) -> Optional[int]:
    try:
        return shutil.disk_usage(str(_nearest_existing(path))).free
    except OSError:
        return None


def _filesystem_id(path: Path) -> Optional[object]:
    try:
        return os.stat(_nearest_existing(path)).st_dev
    except OSError:
        return None


def resolved_cache_dir(cache_dir: Optional[str], cwd: Optional[Path] = None) -> Optional[str]:
    """UV_CACHE_DIR made absolute as uv resolves it (against uv's working directory), or None.

    Every reader comparing a live setting to the marker must resolve it this same way.
    """
    value = (cache_dir or "").strip()
    if not value:
        return None
    cache = Path(value)
    if not cache.is_absolute():
        cache = _uv_working_directory(cwd) / cache
    return os.path.normpath(str(cache))


def _uv_working_directory(cwd: Optional[Path] = None) -> Path:
    """Where uv anchors a relative path: its working directory, moved by UV_WORKING_DIR."""
    base = Path(cwd if cwd is not None else (_RUN_CWD or os.getcwd()))
    working = (os.environ.get("UV_WORKING_DIR") or "").strip()
    if working:
        base = base / working
    return base


def _volumes_to_check(root: Path, cache_dir: Optional[str]) -> list:
    """The prefetch root, plus the uv cache when on another filesystem: uv writes both."""
    volumes = [root]
    if cache_dir:
        cache = Path(cache_dir)
        if not cache.is_absolute():
            cache = _uv_working_directory() / cache
        root_id, cache_id = _filesystem_id(root), _filesystem_id(cache)
        if root_id is None or cache_id is None or root_id != cache_id:
            volumes.append(cache)
    return volumes


# Every uv call is bounded by what is LEFT of the run's budget, not its own timeout.
_RUN_DEADLINE: Optional[float] = None


@contextlib.contextmanager
def _within_budget(deadline: Optional[float]) -> Iterator[None]:
    global _RUN_DEADLINE
    previous = _RUN_DEADLINE
    _RUN_DEADLINE = deadline
    try:
        yield
    finally:
        _RUN_DEADLINE = previous


def _run_timeout(cmd: Sequence[str]) -> float:
    if _RUN_DEADLINE is None:
        return float(SUBPROCESS_TIMEOUT_SECONDS)
    remaining = _RUN_DEADLINE - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(list(cmd), 0)
    return min(float(SUBPROCESS_TIMEOUT_SECONDS), remaining)


# uv's cwd for every call, set by `run`: uv discovers uv.toml from it, as the update's uv does.
_RUN_CWD: Optional[str] = None


@contextlib.contextmanager
def _working_directory(cwd: Optional[Path]) -> Iterator[None]:
    global _RUN_CWD
    previous = _RUN_CWD
    _RUN_CWD = str(cwd) if cwd is not None else None
    try:
        yield
    finally:
        _RUN_CWD = previous


def _run(cmd: Sequence[str], env: Optional[dict]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        cwd = _RUN_CWD,
        timeout = _run_timeout(cmd),
    )


# The installer's _redact_install_output: failure text can embed index credentials.
_URL_USERINFO_RE = re.compile(r"(https?://)[^/@\s`]+@")
_URL_QUERY_VALUE_RE = re.compile(r"([?&][^=\s&`]+)=[^&#\s`]+")
_URL_FRAGMENT_RE = re.compile(r"(https?://[^\s`#]+)#[^\s`]+")


def _redact(text: str) -> str:
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _URL_QUERY_VALUE_RE.sub(r"\1=<redacted>", text)
    return _URL_FRAGMENT_RE.sub(r"\1#<redacted>", text)


def _failure_text(result: subprocess.CompletedProcess, limit: int) -> str:
    # Redacted before truncating: a cut through userinfo would lose the scheme the pattern needs.
    return _redact(_combined(result)).strip()[-limit:] or f"uv exited {result.returncode}"


def _timed_out(exc: BaseException) -> bool:
    return isinstance(exc, subprocess.TimeoutExpired)


def _combined(result: subprocess.CompletedProcess) -> str:
    return f"{result.stdout or ''}\n{result.stderr or ''}"


# The desktop shell's PATH can predate the install: also look where the installers put uv.
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
        add(str(Path(data_home).parent / "bin"))
    home = env.get("USERPROFILE") if platform.system() == "Windows" else None
    home = home or env.get("HOME") or os.path.expanduser("~")
    add(str(Path(home) / ".local" / "bin"))
    local_app_data = env.get("LOCALAPPDATA")
    if platform.system() == "Windows" and local_app_data:
        add(str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links"))
    return dirs


def _uv_runs(candidate: Path, env: Optional[dict]) -> bool:
    """install.ps1's Test-UvCandidateVersion, minus the floor."""
    try:
        return _run([str(candidate), "--version"], env).returncode == 0
    except Exception:
        return False


def locate_uv(env: Optional[dict] = None) -> Tuple[Optional[str], List[Path]]:
    """The uv the update's core step will run (PATH first), and every place searched."""
    environment = dict(env) if env is not None else dict(os.environ)
    searched = _uv_search_dirs(environment)
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


MANIFEST_NAME = "unsloth_install_manifest.json"
NO_TORCH_TRUTHY = ("1", "true", "yes", "on")


def no_torch_mode(venv: Path) -> bool:
    """install_python_stack._infer_no_torch: UNSLOTH_NO_TORCH, manifest, marker, then Intel Mac."""
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None and env.strip():
        return env.strip().lower() in NO_TORCH_TRUTHY
    try:
        manifest = json.loads((venv / MANIFEST_NAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        manifest = None
    if isinstance(manifest, dict):
        value = manifest.get("no_torch")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in NO_TORCH_TRUTHY
    if (venv / NO_TORCH_MARKER).is_file():
        return True
    return platform.system() == "Darwin" and platform.machine() == "x86_64"


def _no_torch(venv: Path) -> bool:
    return no_torch_mode(venv)


def _installed_studio_root(venv: Path) -> Optional[Path]:
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
    cwd: Optional[Path] = None,
) -> dict:
    """Prepare the next update; returns the marker payload. `cwd` is where the update's uv runs."""
    with _working_directory(cwd):
        return _run_unguarded(
            studio_home = studio_home,
            floor = floor,
            shell_version = shell_version,
            env = env,
            echo = echo,
            python = python,
        )


def _run_unguarded(
    *,
    studio_home: Path,
    floor: str = "",
    shell_version: Optional[str] = None,
    env: Optional[dict] = None,
    echo: Callable[[str], None] = print,
    python: Optional[Path] = None,
) -> dict:
    """Raises PrefetchSkipped (exit 0, no marker) or PrefetchError."""
    interpreter = Path(python) if python is not None else Path(sys.executable)
    venv = Path(sys.prefix)

    def step(text: str) -> None:
        echo(f"[TAURI:STEP] {text}")

    # Before the skips: another offer's marker would hand its pins to the offline retry.
    existing = read_marker(studio_home)
    if isinstance(existing, dict) and existing.get("shell_version") != shell_version:
        discard(studio_home)

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
    child_env = dict(env) if env is not None else dict(os.environ)
    # Before the uv search: a candidate hanging on --version spends the budget too.
    deadline = time.monotonic() + BUDGET_SECONDS
    with _within_budget(deadline):
        uv, searched_for_uv = locate_uv(child_env)
    if uv is None:
        where = ", ".join(str(directory) for directory in searched_for_uv)
        raise PrefetchSkipped(f"uv is not available (looked on PATH and in {where})")

    root = prefetch_root(studio_home)
    cache_dir = resolved_cache_dir(child_env.get("UV_CACHE_DIR"))
    for volume in _volumes_to_check(root, cache_dir):
        free = _free_bytes(volume)
        if free is not None and free < MIN_FREE_BYTES:
            raise PrefetchError(
                f"not enough free space to prepare an update ({free} bytes free at "
                f"{volume}, {MIN_FREE_BYTES} needed)"
            )

    if root.exists() and not _is_owned(root):
        raise PrefetchError(f"{root} exists and was not created by Unsloth")
    shutil.rmtree(root, ignore_errors = True)
    root.mkdir(parents = True, exist_ok = True)
    # First, so an interrupted run leaves a directory the next prefetch may remove.
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
    # The installer's macOS arm64 UV_OVERRIDE, from the live tree: without it MLX plans a downgrade.
    _applied_live_override = False
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and "UV_OVERRIDE" not in child_env
        and live_studio is not None
    ):
        overrides = (
            live_studio / "backend" / "requirements" / "single-env" / "overrides-darwin-arm64.txt"
        )
        if overrides.is_file():
            child_env["UV_OVERRIDE"] = str(overrides)
            _applied_live_override = True

    no_torch = _no_torch(venv)
    live_override = child_env.get("UV_OVERRIDE") if _applied_live_override else None
    step("prefetch resolving core packages")
    core_cmd = core_dry_run_command(
        interpreter,
        floor = floor,
        constraints = live_constraints,
        no_torch = no_torch,
        uv = uv,
    )
    try:
        with _within_budget(deadline):
            return _run_prefetch(
                studio_home = studio_home,
                core_cmd = core_cmd,
                child_env = child_env,
                floor = floor,
                shell_version = shell_version,
                cache_dir = cache_dir,
                interpreter = interpreter,
                uv = uv,
                no_torch = no_torch,
                target = target,
                live_constraints = live_constraints,
                deadline = deadline,
                step = step,
                live_override = live_override,
            )
    finally:
        # Scratch on every outcome: the swap reads the uv cache, and a failed fetch's partial tree
        # has no marker for any later cleanup to find.
        shutil.rmtree(target, ignore_errors = True)
        shutil.rmtree(root / "req", ignore_errors = True)


def _run_prefetch(
    *,
    studio_home: Path,
    core_cmd: Sequence[str],
    child_env: Optional[dict],
    floor: str,
    shell_version: Optional[str],
    cache_dir: Optional[str],
    interpreter: Path,
    uv: str,
    no_torch: bool,
    target: Path,
    live_constraints: Optional[Path],
    deadline: float,
    step: Callable[[str], None],
    live_override: Optional[str] = None,
) -> dict:
    try:
        resolved = _run(core_cmd, child_env)
    except (OSError, subprocess.SubprocessError) as exc:
        if _timed_out(exc):
            raise PrefetchError("out of time while resolving the core packages") from exc
        raise PrefetchError(f"could not resolve the core packages: {_redact(str(exc))}") from exc
    if resolved.returncode != 0:
        raise PrefetchError("could not resolve the core packages: " + _failure_text(resolved, 800))
    core_output = _combined(resolved)
    planned = parse_dry_run_plan(core_output)
    if not plan_is_readable(core_output, planned):
        raise PrefetchError(
            "could not read the core plan uv printed: " + _redact(core_output).strip()[-800:]
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
        # The mode the plan was made for; the consumer refuses a plan made for the other.
        "no_torch": no_torch,
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
        # Keyed off the whole plan: a zoo-only bump is a real download.
        if floor and installed_backend and not version_meets_floor(installed_backend, floor):
            raise PrefetchError(
                f"the index offers no unsloth>={floor}; installed is {installed_backend}"
            )
        payload["installed_core"] = {
            name: version
            for name, version in (
                ("unsloth", installed_backend),
                ("unsloth-zoo", _installed_version("unsloth-zoo")),
            )
            if isinstance(version, str) and version
        }
        step("prefetch nothing to prepare")
        write_marker(studio_home, payload)
        return payload

    if floor and backend_version is not None and not version_meets_floor(backend_version, floor):
        raise PrefetchError(f"the resolved unsloth {backend_version} is below the required {floor}")

    pins = pins_from_plan(planned)
    step(f"prefetch downloading {len(pins)} core package(s)")
    fetch_cmd = fetch_command(interpreter, target, pins, uv = uv)
    try:
        fetched = _run(fetch_cmd, child_env)
    except (OSError, subprocess.SubprocessError) as exc:
        if _timed_out(exc):
            raise PrefetchError("out of time while downloading the core packages") from exc
        raise PrefetchError(f"could not download the core packages: {_redact(str(exc))}") from exc
    if fetched.returncode != 0:
        raise PrefetchError("could not download the core packages: " + _failure_text(fetched, 800))
    payload["core_records"] = core_record_digests(target, planned)

    # Requirement files, best effort: the NEW wheel's, resolved against the live venv.
    new_studio = _fetched_studio_root(target)
    state = "ready"
    if new_studio is None:
        payload["requirements"]["*"] = {"skipped_reason": "the new wheel ships no studio tree"}
        state = "partial"
    else:
        req_root = new_studio / "backend" / "requirements"
        payload["requirement_digests"] = requirement_digests(req_root)
        # The dependency passes run under the NEW wheel's override; the core plan used the live one.
        requirement_env = child_env
        if live_override is not None:
            new_overrides = req_root / "single-env" / "overrides-darwin-arm64.txt"
            if new_overrides.is_file():
                requirement_env = {**(child_env or {}), "UV_OVERRIDE": str(new_overrides)}
        new_constraints = req_root / "single-env" / "constraints.txt"
        if not new_constraints.is_file():
            new_constraints = None
        skip = set(NO_TORCH_SKIP_PACKAGES) if no_torch else set()
        if platform.system() == "Windows":
            skip |= set(WINDOWS_SKIP_PACKAGES)
        work_dir = prefetch_root(studio_home) / "req"
        for name, no_deps in REQUIREMENT_PASS:
            if time.monotonic() >= deadline:
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
                env = requirement_env,
                uv = uv,
                step = step,
                label = name,
            )
            payload["requirements"][name] = record
            if record.get("skipped_reason"):
                state = "partial"

    payload["state"] = state
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
        if _timed_out(exc):
            return {"skipped_reason": "out of time"}
        return {"skipped_reason": f"resolve failed: {_redact(str(exc))}"}
    if resolved.returncode != 0:
        return {"skipped_reason": "resolve failed: " + _failure_text(resolved, 400)}
    output = _combined(resolved)
    planned = parse_dry_run_plan(output)
    if not plan_is_readable(output, planned):
        return {"skipped_reason": "could not read the plan uv printed"}
    if not planned:
        return {"pins": {}}
    pins = pins_from_plan(planned, only_binary = True)
    # Only fetched pins: the desktop reads a recorded pin missing from the cache as stale.
    wanted = set(pins)
    fetched_plan = {
        name: version for name, version in planned.items() if f"{name}=={version}" in wanted
    }
    source_only = sorted(name for name in planned if name not in fetched_plan)
    if not pins:
        return {"pins": {}, "source_only": source_only}
    step(f"prefetch downloading {len(pins)} package(s) for {label}")
    try:
        # --only-binary: never build an sdist here that the update builds anyway.
        fetched = _run(
            fetch_command(interpreter, target, pins, only_binary = True, uv = uv),
            env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if _timed_out(exc):
            return {"pins": dict(planned), "skipped_reason": "out of time"}
        return {"pins": dict(planned), "skipped_reason": f"download failed: {_redact(str(exc))}"}
    if fetched.returncode != 0:
        return {
            "pins": dict(planned),
            "skipped_reason": "download failed: " + _failure_text(fetched, 400),
        }
    record: dict = {"pins": fetched_plan}
    if source_only:
        record["source_only"] = source_only
    return record

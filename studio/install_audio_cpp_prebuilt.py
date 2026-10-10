# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install a prebuilt ``audiocpp_server`` (audio.cpp) for Studio's audio.cpp engine.

audio.cpp vendors its own patched ggml, so unlike whisper.cpp it cannot share the
llama.cpp runtime: every bundle is self-contained (the server links ggml statically).
Releases are named ``audio-<tag>-bin-<os>-<arch>-<backend>.{zip,tar.gz}``, the scheme
upstream uses and the Unsloth fork keeps, so one resolver serves both. Windows CUDA
bundles need the CUDA runtime DLLs, which upstream publishes as a separate
``audio-<tag>-cudart-windows-x64-<line>.zip``; that download is skipped when the
running Python's torch already ships the same CUDA runtime line.

The bundle is extracted into a staging directory beside the managed tree, started once
with ``--help`` so a bundle this host cannot load is refused, and swapped in only then,
so an interrupted or unusable install never replaces a working tree.

Every bundle and CUDA runtime archive must match a sha256 pinned in
``audio_cpp_prebuilt_pins.json`` unless the user picked the release
(``UNSLOTH_AUDIO_CPP_REPO`` / ``UNSLOTH_AUDIO_CPP_TAG``). A re-run whose install record
already names the pinned release makes no network call, and a lookup that cannot answer
keeps a complete install.

Usage:
    python studio/install_audio_cpp_prebuilt.py                 # auto-detect host
    python studio/install_audio_cpp_prebuilt.py --accelerator cpu
    python studio/install_audio_cpp_prebuilt.py --print-asset   # resolve only
    python studio/install_audio_cpp_prebuilt.py --write-pins    # regenerate the digest pins
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence

if __package__:
    from . import prebuilt_core as core
    from .install_sd_cpp_prebuilt import (
        GitHubRateLimited,
        _is_rate_limited,
        _rate_limit_message,
        _safe_extractall,
    )
else:
    _STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
    if _STUDIO_DIR not in sys.path:
        sys.path.insert(0, _STUDIO_DIR)
    import prebuilt_core as core  # type: ignore[no-redef]
    from install_sd_cpp_prebuilt import (  # type: ignore[no-redef]
        GitHubRateLimited,
        _is_rate_limited,
        _rate_limit_message,
        _safe_extractall,
    )

DEFAULT_REPO = "unslothai/audio.cpp"
UPSTREAM_FALLBACK_REPO = "0xShug0/audio.cpp"
# Pinned for reproducibility; UNSLOTH_AUDIO_CPP_TAG overrides ('' tracks latest).
DEFAULT_TAG = "v0.9.0-unsloth.1"
# The upstream release the fork tag is built from, tried when the fork cannot serve this host.
UPSTREAM_FALLBACK_TAG = "v0.9.0"

INSTALL_RECORD = "UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json"
OWNERSHIP_MARKER = ".unsloth-studio-owned"
SERVER_NAME = "audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server"
# Regenerate with `python studio/install_audio_cpp_prebuilt.py --write-pins` after bumping a tag.
PINS_PATH = Path(__file__).resolve().with_name("audio_cpp_prebuilt_pins.json")

ACCELERATORS = ("auto", "cpu", "cuda", "vulkan", "metal")
SMOKE_TIMEOUT_SECONDS = 30.0
USER_AGENT = "unsloth-audio-cpp-installer"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BUSY = 3

KEPT_EXISTING_LINE = "audio.cpp: keeping the existing complete install"


class ReleaseLookupUnavailable(RuntimeError):
    """No release lookup answered (offline, DNS, GitHub down): says nothing about the tree on disk."""


def log(message: str) -> None:
    print(f"audio.cpp: {message}", file = sys.stderr, flush = True)


_OPS = core.ModuleOps(globals())

_OS_TOKENS = {
    "windows": ("windows", "win"),
    "linux": ("ubuntu", "linux"),
    "darwin": ("macos", "darwin"),
}
_ARCH_TOKENS = {
    "x86_64": "x64",
    "amd64": "x64",
    "x64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
}
# "-cuda12.8" (upstream) or "-cuda12" (unslothai/audio.cpp: one lean bundle per CUDA major line).
_CUDA_LINE = re.compile(r"-cuda(\d+)(?:\.(\d+))?(?=[-.]|$)")


def _repo() -> str:
    return (os.environ.get("UNSLOTH_AUDIO_CPP_REPO") or "").strip() or DEFAULT_REPO


def _pinned_tag() -> Optional[str]:
    val = os.environ.get("UNSLOTH_AUDIO_CPP_TAG", DEFAULT_TAG).strip()
    return val or None


def _user_picked_release() -> bool:
    """The user chose the repo or tag, so a digest this repo never pinned is theirs to trust."""
    return (
        bool((os.environ.get("UNSLOTH_AUDIO_CPP_REPO") or "").strip())
        or "UNSLOTH_AUDIO_CPP_TAG" in os.environ
    )


_PINNED_LINUX_GLIBC_FLOOR = (2, 35)


def _glibc_below_pinned_floor() -> Optional[str]:
    """The host glibc when it is older than the pinned Linux bundles need, else ``None``. Without
    this the start check fails after a full download (~1 GB for CUDA) on every setup run."""
    if not sys.platform.startswith("linux") or _user_picked_release():
        return None
    name, version = platform.libc_ver()
    try:
        parsed = tuple(int(x) for x in version.split(".")[:2])
    except ValueError:
        return None
    if name != "glibc" or len(parsed) != 2 or parsed >= _PINNED_LINUX_GLIBC_FLOOR:
        return None
    return version


def _release_ladder() -> list[tuple[str, Optional[str]]]:
    """``(repo, tag)`` lookups in order: the pinned tag on the primary repo, then, when the user
    pinned neither repo nor tag, the upstream release the fork tag was cut from. ``None`` (latest)
    only when the user set ``UNSLOTH_AUDIO_CPP_TAG=''``; a pinned install never drifts to an
    untested latest. A user-chosen tag never reaches a repo the user did not choose."""
    primary = _repo()
    ladder = [(primary, _pinned_tag())]
    if not _user_picked_release() and primary != UPSTREAM_FALLBACK_REPO:
        ladder.append((UPSTREAM_FALLBACK_REPO, UPSTREAM_FALLBACK_TAG))
    return ladder


def _accelerator_from_env() -> str:
    """``UNSLOTH_AUDIO_CPP_ACCELERATOR`` (auto|cpu|cuda|vulkan|metal), for hosts whose GPU is not visible
    at install time (a Docker build). Set, it is an explicit request and never downgraded."""
    raw = (os.environ.get("UNSLOTH_AUDIO_CPP_ACCELERATOR") or "").strip().lower()
    if not raw:
        return "auto"
    if raw not in ACCELERATORS:
        print(
            f"audio.cpp: ignoring UNSLOTH_AUDIO_CPP_ACCELERATOR={raw!r} (expected one of {', '.join(ACCELERATORS)})",
            file = sys.stderr,
            flush = True,
        )
        return "auto"
    return raw


def load_pins(path: Optional[Path] = None) -> dict:
    try:
        with open(path or PINS_PATH, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    releases = data.get("releases") if isinstance(data, dict) else None
    return releases if isinstance(releases, dict) else {}


def pinned_sha256(repo: str, tag: Optional[str], asset: str) -> Optional[str]:
    if not tag:
        return None
    digest = ((load_pins().get(repo) or {}).get(tag) or {}).get(asset)
    return core.normalize_sha256_digest(digest)


def _expected_sha256(repo: str, tag: Optional[str], asset: dict) -> Optional[str]:
    """The sha256 an asset must hash to: the in-repo pin, which GitHub's own digest must agree with.
    Unpinned assets are refused unless the user picked the release, where GitHub's digest (if any)
    is the best available."""
    name = asset["name"]
    published = core.normalize_sha256_digest(asset.get("digest"))
    pinned = pinned_sha256(repo, tag, name)
    if pinned:
        if published and published != pinned:
            raise RuntimeError(
                f"{name}: GitHub reports sha256 {published} but {PINS_PATH.name} pins {pinned}; refusing it"
            )
        return pinned
    if _user_picked_release():
        return published
    raise RuntimeError(
        f"{name} from {repo} {tag or 'latest'} has no sha256 pinned in {PINS_PATH.name}; refusing an unpinned "
        "download (set UNSLOTH_AUDIO_CPP_REPO or UNSLOTH_AUDIO_CPP_TAG to install a release you chose)"
    )


def default_install_dir() -> Path:
    """``<UNSLOTH_HOME>/audio.cpp``, else ``<STUDIO_HOME>/audio.cpp``, else ``~/.unsloth/audio.cpp``.

    Kept in step with ``audio_cpp_server.managed_audio_cpp_dir``; separate because this
    script runs before the backend package is importable.
    """
    master = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if master:
        return Path(master).expanduser().resolve() / "audio.cpp"
    home = (os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or "").strip()
    legacy = Path.home() / ".unsloth" / "audio.cpp"
    if not home:
        return legacy
    root = Path(home).expanduser().resolve()
    return legacy if root == (Path.home() / ".unsloth" / "studio").resolve() else root / "audio.cpp"


def nvidia_driver_cuda_version() -> Optional[tuple[int, int]]:
    """The CUDA version the installed NVIDIA driver supports, from nvidia-smi, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output = True, text = True, timeout = 15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", out or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def detect_accelerator() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "metal"
    if nvidia_driver_cuda_version() is not None:
        return "cuda"
    return "cpu"


def torch_cuda_major() -> Optional[int]:
    """The CUDA major line of the torch importable here, without importing torch."""
    try:
        import importlib.metadata as md
        version = md.version("torch")
    except Exception:  # noqa: BLE001 - no torch
        return None
    match = re.search(r"\+cu(\d+)", version)
    if match:
        digits = match.group(1)
        return int(digits[:-1]) if len(digits) >= 3 else int(digits[:2])
    return None


def resolve_release_asset(
    asset_names: Sequence[str],
    *,
    system: str,
    machine: str,
    accelerator: str,
    driver_cuda: Optional[tuple[int, int]] = None,
    prefer_cuda_major: Optional[int] = None,
) -> Optional[str]:
    """Pick the bundle for a host, or None. Pure: the caller passes the release's asset names.

    CUDA picks the newest line the driver can run, preferring ``prefer_cuda_major`` (the
    torch line, whose runtime DLLs can then be reused). CPU prefers the portable build,
    which avoids host-ISA features a native build may assume.
    """
    system = system.lower()
    accel = (accelerator or "cpu").lower()
    os_tokens = _OS_TOKENS.get(system)
    arch = _ARCH_TOKENS.get(machine.lower())
    if not os_tokens or not arch:
        return None
    bundles = [
        n
        for n in asset_names
        if "-bin-" in n
        and (n.endswith(".zip") or n.endswith(".tar.gz"))
        and any(f"-bin-{tok}-{arch}-" in n for tok in os_tokens)
    ]

    def backend_of(name: str) -> str:
        stem = name[: -len(".tar.gz")] if name.endswith(".tar.gz") else name[: -len(".zip")]
        return stem.split(f"-{arch}-", 1)[1]

    if accel == "metal":
        pool = [n for n in bundles if backend_of(n).startswith("metal")]
        return pool[0] if pool else None
    if accel == "cuda":
        candidates = []
        for n in bundles:
            match = _CUDA_LINE.search("-" + backend_of(n))
            if not match:
                continue
            line = (int(match.group(1)), int(match.group(2) or 0))
            if driver_cuda is not None and line > driver_cuda:
                continue
            candidates.append((line, n))
        if not candidates:
            return None
        preferred = [
            c for c in candidates if prefer_cuda_major is not None and c[0][0] == prefer_cuda_major
        ]
        # At equal lines a general build beats upstream's Colab one, which carries sm_75 kernels only.
        return max(preferred or candidates, key = lambda c: (c[0], "colab" not in c[1]))[1]
    if accel == "vulkan":
        pool = sorted(
            (n for n in bundles if backend_of(n).startswith("vulkan")),
            key = lambda n: "portable" not in n,
        )
        return pool[0] if pool else None
    pool = sorted(
        (n for n in bundles if backend_of(n).startswith("cpu")), key = lambda n: "portable" not in n
    )
    return pool[0] if pool else None


def cudart_asset_for(asset_names: Sequence[str], bundle: str) -> Optional[str]:
    """The separate Windows CUDA runtime archive matching a CUDA bundle's line."""
    match = _CUDA_LINE.search(bundle)
    if not match or "-windows-" not in bundle:
        return None
    line = match.group(0)[1:]
    for name in asset_names:
        if "-cudart-windows-" in name and name.endswith(f"-{line}.zip"):
            return name
    return None


def _torch_lib_dir() -> Optional[Path]:
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
    except Exception:  # noqa: BLE001
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / "lib"


def torch_provides_cuda_runtime(cuda_major: int) -> bool:
    """Whether this Python's torch ships the CUDA runtime DLLs a line-``cuda_major`` bundle loads."""
    if sys.platform != "win32":
        return False
    lib = _torch_lib_dir()
    if lib is None:
        return False
    needed = (f"cudart64_{cuda_major}*.dll", f"cublas64_{cuda_major}*.dll", "cufft64_*.dll")
    return all(glob.glob(str(lib / pattern)) for pattern in needed)


def _fetch_release(
    repo: str,
    tag: Optional[str],
    token: Optional[str],
    timeout: float = 30.0,
) -> Optional[dict]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{repo}/releases/" + (f"tags/{tag}" if tag else "latest")
    req = urllib.request.Request(url, headers = headers)
    attempts = core.JSON_FETCH_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout = timeout) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if _is_rate_limited(exc):
                raise GitHubRateLimited(_rate_limit_message()) from exc
            if exc.code == 404:
                return None
            if attempt >= attempts or not core.is_retryable_url_error(exc):
                raise
            core.sleep_backoff(attempt, exc = exc)
        except Exception as exc:  # noqa: BLE001 - classified below
            if attempt >= attempts or not core.is_retryable_url_error(exc):
                raise
            core.sleep_backoff(attempt, exc = exc)
    return None  # pragma: no cover - the loop returns or raises


def resolve(accelerator: str, token: Optional[str]) -> tuple[str, Optional[dict], Optional[str]]:
    """``(repo, release, asset)`` for this host along ``_release_ladder()``. Raises
    ``ReleaseLookupUnavailable`` when no lookup answered, so a caller can keep what is installed."""
    primary = _repo()
    driver = nvidia_driver_cuda_version() if accelerator == "cuda" else None
    prefer = torch_cuda_major()
    answered = False
    failures = []
    for repo, want in _release_ladder():
        try:
            release = _fetch_release(repo, want, token)
        except GitHubRateLimited:
            raise
        except Exception as exc:  # noqa: BLE001 - network: try the next rung
            print(f"audio.cpp: {repo} release lookup failed ({exc})", file = sys.stderr, flush = True)
            failures.append(f"{repo}: {exc}")
            continue
        answered = True
        if not release:
            continue
        names = [a["name"] for a in release.get("assets") or []]
        chosen = resolve_release_asset(
            names,
            system = platform.system(),
            machine = platform.machine(),
            accelerator = accelerator,
            driver_cuda = driver,
            prefer_cuda_major = prefer,
        )
        if chosen:
            if repo != primary:
                print(f"audio.cpp: falling back to {repo}", file = sys.stderr, flush = True)
            return repo, release, chosen
    if not answered:
        raise ReleaseLookupUnavailable(
            "no audio.cpp release lookup answered (" + "; ".join(failures) + ")"
        )
    return primary, None, None


def resolve_for_request(
    requested: str, detected: Optional[str], token: Optional[str]
) -> tuple[str, str, Optional[dict], Optional[str]]:
    """``(accelerator, repo, release, asset)``. An auto-detected GPU the published bundles do not
    cover (an older driver, Linux arm64) falls back to the CPU build, which still runs every model;
    an explicit request is never downgraded."""
    accel = detected if requested == "auto" and detected else requested
    repo, release, chosen = resolve(accel, token)
    if requested == "auto" and accel == "cuda" and chosen:
        major = _cuda_major(chosen)
        if not _linux_cuda_runtime_available(major):
            # Linux CUDA bundles load cudart / cuBLAS from torch's nvidia-* wheels.
            print(
                f"audio.cpp: no CUDA {major} runtime (cudart, cuBLAS) for {chosen}; installing the CPU build",
                flush = True,
            )
            accel = "cpu"
            repo, release, chosen = resolve(accel, token)
    if (release is None or not chosen) and requested == "auto" and accel != "cpu":
        print(f"audio.cpp: no {accel} bundle for this host; installing the CPU build", flush = True)
        accel = "cpu"
        repo, release, chosen = resolve(accel, token)
    return accel, repo, release, chosen


def write_pins(path: Optional[Path] = None, token: Optional[str] = None) -> Path:
    """Regenerate the digest pins from GitHub for the default fork tag and its upstream fallback.
    Only runtime archives are pinned: host bundles (``-bin-``) and Windows CUDA runtimes (``-cudart-``)."""
    releases: dict = {}
    for repo, tag in ((DEFAULT_REPO, DEFAULT_TAG), (UPSTREAM_FALLBACK_REPO, UPSTREAM_FALLBACK_TAG)):
        release = _fetch_release(repo, tag, token)
        if not release:
            raise RuntimeError(f"{repo} has no release {tag}")
        digests = {}
        for asset in release.get("assets") or []:
            name = asset.get("name") or ""
            if "-bin-" not in name and "-cudart-" not in name:
                continue
            digest = core.normalize_sha256_digest(asset.get("digest"))
            if not digest:
                raise RuntimeError(f"{repo} {tag} publishes no sha256 digest for {name}")
            digests[name] = digest
        if not digests:
            raise RuntimeError(f"{repo} {tag} has no audio.cpp runtime assets")
        releases.setdefault(repo, {})[tag] = dict(sorted(digests.items()))
    payload = {
        "schema_version": 1,
        "comment": (
            "Trust anchor for install_audio_cpp_prebuilt.py: sha256 of each audio.cpp runtime archive, "
            "copied from the GitHub release asset digests. Generated; do not edit by hand. To bump: change "
            "DEFAULT_TAG / UPSTREAM_FALLBACK_TAG, then run `python studio/install_audio_cpp_prebuilt.py "
            "--write-pins` (the same digests `gh release view <tag> -R <repo> --json assets` lists)."
        ),
        "releases": releases,
    }
    target = path or PINS_PATH
    target.write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8", newline = "\n")
    return target


def _download(url: str, dest: Path) -> None:
    """Retried, progress-reporting download (prebuilt_core's), written atomically to ``dest``."""
    core.download_file(_OPS, url, dest)


def _verify(path: Path, expected_sha256: Optional[str]) -> str:
    """sha256 of ``path``; raises when an expected digest is known and differs."""
    got = core.sha256_file(path)
    if expected_sha256:
        if expected_sha256.lower() != got:
            raise RuntimeError(
                f"sha256 mismatch for {path.name}: expected {expected_sha256.lower()}, got {got}"
            )
    else:
        print(f"audio.cpp: WARNING no published digest for {path.name}", flush = True)
    return got


def _extract(archive: Path, target: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _safe_extractall(zf, target)
        return
    with tarfile.open(archive, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(target, filter = "data")
            return
        # Pythons without extraction filters: the same containment rules, checked by hand.
        base = target.resolve()
        for member in tf.getmembers():
            dest = (base / member.name).resolve()
            if dest != base and base not in dest.parents:
                raise RuntimeError(f"unsafe path in archive: {member.name!r}")
            if member.isdev() or member.isfifo():
                raise RuntimeError(f"special file in archive: {member.name!r}")
            if member.issym() or member.islnk():
                link = (
                    (dest.parent / member.linkname).resolve()
                    if member.issym()
                    else (base / member.linkname).resolve()
                )
                if link != base and base not in link.parents:
                    raise RuntimeError(f"unsafe link in archive: {member.name!r}")
        tf.extractall(target)


def _locate_server(root: Path) -> Optional[Path]:
    for p in sorted(root.rglob(SERVER_NAME), key = lambda p: len(p.parts)):
        if p.is_file():
            return p
    return None


def read_install_record(root: Path) -> dict:
    try:
        with open(root / INSTALL_RECORD, "r", encoding = "utf-8") as f:
            rec = json.load(f)
        return rec if isinstance(rec, dict) else {}
    except (OSError, ValueError):
        return {}


def _backend_of(asset: str) -> str:
    # The Intel macOS bundle keeps the "-metal" name but is built with Metal off (CPU only).
    if "-macos-x64-" in asset:
        return "cpu"
    if "-metal" in asset:
        return "metal"
    if "-cuda" in asset:
        return "cuda"
    if "-vulkan" in asset:
        return "vulkan"
    return "cpu"


# Leftovers of older Studio builds; a tree holding nothing else is Studio's, not the user's.
_STUDIO_LEFTOVERS = frozenset({".child_home"})


def _sweep_retired_trees(target: Path) -> None:
    """Remove ``<target>.old-*`` trees an earlier swap could not delete (Windows keeps a running
    server's files open). Best-effort: one still in use stays for the next run."""
    for old in target.parent.glob(target.name + ".old-*"):
        if old.is_dir() and (old / OWNERSHIP_MARKER).is_file():
            shutil.rmtree(old, ignore_errors = True)
    for staging in target.parent.glob(".audio.cpp-staging-*"):
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors = True)


def _server_in_use(server: Optional[Path]) -> bool:
    """Whether a running audiocpp_server holds this binary. Windows maps a running image
    without write sharing, so opening it for writing fails; elsewhere a replace is safe anyway."""
    if server is None or sys.platform != "win32":
        return False
    try:
        with open(server, "r+b"):
            return False
    except PermissionError:
        return True
    except OSError:
        return False


def _cuda_runtime_satisfied(record: dict) -> bool:
    """A CUDA install that skipped its cudart archive still has torch's runtime to lean on."""
    if (
        record.get("backend") != "cuda"
        or record.get("cudart_asset")
        or record.get("os") != "windows"
    ):
        return True
    match = _CUDA_LINE.search(str(record.get("asset") or ""))
    return bool(match) and torch_provides_cuda_runtime(int(match.group(1)))


def _intact_install(target: Path, record: dict) -> Optional[Path]:
    """The server of a complete, unaltered Studio install of ``record``, or None."""
    if not record or not (target / OWNERSHIP_MARKER).is_file():
        return None
    rel = record.get("server_relpath")
    want = core.normalize_sha256_digest(record.get("server_sha256"))
    if not isinstance(rel, str) or not rel or not want:
        return None
    server = target / rel
    try:
        if not server.is_file() or core.sha256_file(server) != want:
            return None
    except OSError:
        return None
    if record.get("espeak") is True and not (server.parent / "espeak-ng-data.bin").is_file():
        return None
    return server if _cuda_runtime_satisfied(record) else None


def _pinned_install_matches(
    target: Path, record: dict, requested: str, detected: Optional[str]
) -> Optional[Path]:
    """Answer "already matches" with no network call: the record names a (repo, tag) this run would
    look up, its archive digest is the pinned one, it was installed for the same accelerator request
    (and, for auto, the same detected host), and the server on disk is the one it installed."""
    ladder = _release_ladder()
    if not record or any(tag is None for _, tag in ladder):
        return None
    repo, tag = record.get("published_repo"), record.get("release_tag")
    # Only the first rung: a fallback install must ask again.
    if (repo, tag) != ladder[0]:
        return None
    pin = pinned_sha256(repo, tag, str(record.get("asset") or ""))
    if pin is None and not _user_picked_release():
        return None
    if pin is not None and pin != record.get("asset_sha256"):
        return None
    if record.get("accelerator_request") != requested:
        return None
    if requested == "auto" and record.get("detected_accelerator") != detected:
        return None
    if (
        requested == "auto"
        and detected == "cuda"
        and record.get("accelerator") == "cpu"
        and _linux_cuda_runtime_available(torch_cuda_major())
    ):
        return None
    return _intact_install(target, record)


_WINDOWS_LOADER_FAILURES = {
    0xC0000135: "a DLL it needs was not found",
    0xC000007B: "a DLL it loads is built for another architecture",
    0xC0000139: "a DLL it loads lacks an entry point it needs",
}


def _scrubbed_environ() -> dict:
    """``os.environ`` without secrets, through the backend's ``child_env.scrub_env`` so the staged
    server sees what Studio will later launch it with."""
    import importlib.util

    path = Path(__file__).resolve().parent / "backend" / "utils" / "prebuilt" / "child_env.py"
    try:
        spec = importlib.util.spec_from_file_location("_unsloth_audio_cpp_child_env", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.scrub_env(os.environ)
    except Exception:  # noqa: BLE001 - a partial checkout: drop the obvious secret names
        markers = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "PROXY")
        return {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in markers)}


def _loader_env(server: Path, extra_dirs: Sequence[Path] = ()) -> dict:
    """The environment with the server's own directory first on the loader path, as the backend's
    ``child_env`` launches it."""
    env = _scrubbed_environ()
    if sys.platform == "win32":
        var = "PATH"
    elif sys.platform == "darwin":
        var = "DYLD_LIBRARY_PATH"
    else:
        var = "LD_LIBRARY_PATH"
    existing = [p for p in env.get(var, "").split(os.pathsep) if p]
    env[var] = os.pathsep.join([str(server.parent), *(str(d) for d in extra_dirs), *existing])
    return env


def _cuda_runtime_dirs(backend: str) -> list[str]:
    """The CUDA libraries the backend's ``child_env`` puts after the bundle dir: the Linux CUDA server
    links cuBLAS, cudart, cuFFT and NCCL dynamically and finds them in torch's nvidia-* wheels, and a
    Windows bundle installed without its cudart archive uses torch's DLLs."""
    if backend != "cuda" or sys.platform == "darwin":
        return []
    try:
        if __package__:
            from .install_llama_prebuilt import python_runtime_dirs
        else:
            from install_llama_prebuilt import python_runtime_dirs  # type: ignore[no-redef]

        return list(python_runtime_dirs())
    except Exception:  # noqa: BLE001 - no wheel runtime to offer
        return []


def _cuda_major(asset: str) -> Optional[int]:
    match = _CUDA_LINE.search(asset)
    return int(match.group(1)) if match else None


def _ldconfig_libs() -> set[str]:
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output = True, text = True, timeout = 10).stdout
    except Exception:  # noqa: BLE001 - no ldconfig: only the wheel dirs count
        return set()
    return {line.split()[0] for line in out.splitlines()[1:] if line.strip()}


def _linux_cuda_runtime_available(major: Optional[int]) -> bool:
    """Whether the CUDA libraries a line-``major`` Linux bundle links but does not ship (cudart,
    cuBLAS, cuFFT) are in torch's nvidia-* wheels or on the system loader path. True off Linux,
    where the bundle (or its cudart archive) carries them."""
    if not sys.platform.startswith("linux"):
        return True
    if major is None:
        return False
    dirs = [Path(d) for d in _cuda_runtime_dirs("cuda")]
    system = _ldconfig_libs()

    def present(pattern: str) -> bool:
        if any(any(d.glob(pattern)) for d in dirs):
            return True
        return any(fnmatch.fnmatch(name, pattern) for name in system)

    return all(
        present(pattern)
        for pattern in (f"libcudart.so.{major}", f"libcublas.so.{major}", "libcufft.so.*")
    )


def _run_staged(server: Path, arg: str, env: dict, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(server), arg],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = timeout,
        env = env,
        cwd = str(server.parent),
        **core.windows_hidden_subprocess_kwargs(),
    )


def smoke_test_staged_server(
    server: Path,
    *,
    backend: str,
    extra_dirs: Sequence[Path] = (),
) -> Optional[str]:
    """Start the staged server once with ``--help``: a bundle this host cannot load (a newer glibc, a
    missing DLL) raises here, before it replaces a working tree. Returns its ``--version`` text
    (build and enabled backends) when it prints one."""
    env = _loader_env(server, extra_dirs)
    try:
        result = _run_staged(server, "--help", env, SMOKE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"staged {SERVER_NAME} --help did not exit within {SMOKE_TIMEOUT_SECONDS:.0f}s; keeping the existing install"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"staged {SERVER_NAME} could not be started ({exc}); keeping the existing install"
        ) from exc
    if result.returncode != 0:
        output = "\n".join(s.strip() for s in (result.stderr, result.stdout) if s and s.strip())
        if backend == "cuda" and "libcuda.so" in output and nvidia_driver_cuda_version() is None:
            # No NVIDIA driver (e.g. Docker image build): libcuda arrives with the driver at run time.
            print(
                "audio.cpp: no NVIDIA driver here to load the CUDA build; skipping its start check",
                flush = True,
            )
            return None
        hint = (
            _WINDOWS_LOADER_FAILURES.get(result.returncode & 0xFFFFFFFF)
            if sys.platform == "win32"
            else None
        )
        detail = "\n".join(output.splitlines()[-20:]) or hint or "no output"
        raise RuntimeError(
            f"staged {SERVER_NAME} --help exited {result.returncode}; this host cannot run the bundle, "
            f"keeping the existing install:\n{detail}"
        )
    try:
        version = _run_staged(server, "--version", env, 10.0)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (version.stdout or "").strip()
    return text[:2000] if version.returncode == 0 and text else None


def install(
    *,
    install_dir: Optional[Path] = None,
    accelerator: str = "auto",
    token: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Install (or keep) the bundle for ``accelerator``; ``"auto"`` detects the host here, so the
    CPU fallback for an uncovered GPU applies. Serialised on prebuilt_core's install lock."""
    if accelerator not in ACCELERATORS:
        raise RuntimeError(
            f"unknown accelerator {accelerator!r}; expected one of {', '.join(ACCELERATORS)}"
        )
    target = (install_dir or default_install_dir()).resolve()
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"audio.cpp install target is not a directory: {target}")
    if target.exists() and not (target / OWNERSHIP_MARKER).is_file():
        entries = {p.name for p in target.iterdir()}
        if entries - _STUDIO_LEFTOVERS:
            raise RuntimeError(
                f"{target} already exists and is not an Unsloth-managed directory; refusing to replace it. "
                "Move it away or pass a different --install-dir."
            )
    with core.install_lock(core.install_lock_path(target)):
        return _install_locked(target, accelerator, token, force)


def _install_locked(target: Path, requested: str, token: Optional[str], force: bool) -> Path:
    _sweep_retired_trees(target)
    detected = detect_accelerator() if requested == "auto" else None
    existing = read_install_record(target)
    if not force:
        server = _pinned_install_matches(target, existing, requested, detected)
        if server is not None:
            print(f"audio.cpp: already matches {existing.get('asset')}", flush = True)
            return server
    old_glibc = _glibc_below_pinned_floor()
    if old_glibc:
        raise RuntimeError(
            f"audio.cpp prebuilt bundles need glibc {'.'.join(map(str, _PINNED_LINUX_GLIBC_FLOOR))}+; "
            f"this host has {old_glibc}. Set UNSLOTH_AUDIO_CPP_PATH to a local build instead."
        )
    token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    try:
        accel, repo, release, chosen = resolve_for_request(requested, detected, token)
    except (ReleaseLookupUnavailable, GitHubRateLimited) as exc:
        # A failed lookup says nothing about the tree on disk; keep a complete one unless --force.
        kept = None if force else _intact_install(target, existing)
        if kept is None or (requested != "auto" and existing.get("accelerator") != requested):
            raise
        # "keeping the existing complete install" is the substring setup.sh and setup.ps1 grep for.
        print(KEPT_EXISTING_LINE, flush = True)
        print(
            f"audio.cpp: {existing.get('release_tag')} stays; release lookup unavailable ({exc})",
            file = sys.stderr,
            flush = True,
        )
        return kept
    if release is None or not chosen:
        raise RuntimeError(
            f"No prebuilt audiocpp_server for {platform.system()}/{platform.machine()} "
            f"(accelerator={accel}) from {repo}. Build it from source: https://github.com/{repo}"
        )
    assets = {a["name"]: a for a in release.get("assets") or []}
    asset = assets[chosen]
    tag = release.get("tag_name")
    expected = _expected_sha256(repo, tag, asset)
    if (
        not force
        and expected
        and existing.get("asset") == chosen
        and existing.get("asset_sha256") == expected
        and _intact_install(target, existing) is not None
    ):
        print(f"audio.cpp: already matches {chosen}", flush = True)
        return _intact_install(target, existing)

    cudart = cudart_asset_for(list(assets), chosen)
    cuda_match = _CUDA_LINE.search(chosen)
    cuda_major = int(cuda_match.group(1)) if cuda_match else None
    torch_runtime = bool(cudart and cuda_major and torch_provides_cuda_runtime(cuda_major))
    cudart_expected = (
        _expected_sha256(repo, tag, assets[cudart]) if cudart and not torch_runtime else None
    )

    if target.exists() and _server_in_use(_locate_server(target)):
        raise PermissionError(f"audiocpp_server in {target} is running")
    target.parent.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = ".audio.cpp-staging-", dir = str(target.parent)))
    try:
        archive = staging / chosen
        print(f"audio.cpp: downloading {chosen} from {repo} {tag}", flush = True)
        _download(asset["browser_download_url"], archive)
        digest = _verify(archive, expected)
        tree = staging / "tree"
        tree.mkdir()
        _extract(archive, tree)
        archive.unlink()
        server = _locate_server(tree)
        if server is None:
            raise RuntimeError(f"{chosen} contains no {SERVER_NAME}")
        cudart_used = None
        if cudart and not torch_runtime:
            print(f"audio.cpp: downloading CUDA runtime {cudart}", flush = True)
            rt_archive = staging / cudart
            _download(assets[cudart]["browser_download_url"], rt_archive)
            _verify(rt_archive, cudart_expected)
            with zipfile.ZipFile(rt_archive) as zf:
                _safe_extractall(zf, server.parent)
            rt_archive.unlink()
            cudart_used = cudart
        if sys.platform != "win32":
            server.chmod(server.stat().st_mode | 0o111)
        backend = _backend_of(chosen)
        version = smoke_test_staged_server(
            server, backend = backend, extra_dirs = _cuda_runtime_dirs(backend)
        )
        (tree / OWNERSHIP_MARKER).touch()
        record = {
            "schema_version": 1,
            "component": "audio.cpp",
            "published_repo": repo,
            "release_tag": tag,
            "asset": chosen,
            "asset_sha256": digest,
            "cudart_asset": cudart_used,
            "backend": backend,
            "accelerator": accel,
            "accelerator_request": requested,
            "detected_accelerator": detected,
            "espeak": (server.parent / "espeak-ng-data.bin").is_file(),
            "server_relpath": server.relative_to(tree).as_posix(),
            "server_sha256": core.sha256_file(server),
            "server_version": version,
            "os": platform.system().lower(),
            "arch": _ARCH_TOKENS.get(platform.machine().lower(), platform.machine().lower()),
            "installed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (tree / INSTALL_RECORD).write_text(json.dumps(record, indent = 2), encoding = "utf-8")
        # A running server would keep serving the old build; the caller stops it instead.
        if target.exists() and _server_in_use(_locate_server(target)):
            raise PermissionError(f"audiocpp_server in {target} is running")
        retired = None
        if target.exists():
            retired = target.with_name(target.name + f".old-{os.getpid()}")
            os.replace(target, retired)
        try:
            os.replace(tree, target)
        except OSError:
            if retired is not None:
                os.replace(retired, target)
            raise
        if retired is not None:
            shutil.rmtree(retired, ignore_errors = True)
    finally:
        shutil.rmtree(staging, ignore_errors = True)
    installed = target / record["server_relpath"]
    print(f"audio.cpp: installed {installed}", flush = True)
    return installed


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description = "Install a prebuilt audiocpp_server (audio.cpp).")
    p.add_argument(
        "--accelerator",
        default = _accelerator_from_env(),
        type = lambda value: value.strip().lower(),
        choices = list(ACCELERATORS),
        help = "default: UNSLOTH_AUDIO_CPP_ACCELERATOR, else auto (detect; CPU when no bundle covers the GPU)",
    )
    p.add_argument("--install-dir", default = None)
    p.add_argument("--force", action = "store_true", help = "reinstall even when the bundle matches")
    p.add_argument("--print-asset", action = "store_true", help = "resolve and print the asset only")
    p.add_argument(
        "--write-pins",
        action = "store_true",
        help = f"regenerate {PINS_PATH.name} from the pinned GitHub releases",
    )
    args = p.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    try:
        if args.write_pins:
            print(f"audio.cpp: wrote {write_pins(token = token)}")
            return EXIT_OK
        if args.print_asset:
            detected = detect_accelerator() if args.accelerator == "auto" else None
            _, repo, release, chosen = resolve_for_request(args.accelerator, detected, token)
            print(f"{repo} {release.get('tag_name') if release else '-'} {chosen or '(none)'}")
            return EXIT_OK if chosen else EXIT_FAILED
        install(
            install_dir = Path(args.install_dir).expanduser() if args.install_dir else None,
            accelerator = args.accelerator,
            force = args.force,
        )
    except GitHubRateLimited as exc:
        print(f"error: {exc}", file = sys.stderr)
        return EXIT_FAILED
    except core.BusyInstallConflict as exc:
        print(f"error: another audio.cpp install is running ({exc})", file = sys.stderr)
        return EXIT_BUSY
    except PermissionError as exc:
        print(
            f"error: the audio.cpp runtime is in use ({exc}); stop Studio and retry",
            file = sys.stderr,
        )
        return EXIT_BUSY
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file = sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

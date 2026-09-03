# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install a prebuilt ``sd-cli`` (stable-diffusion.cpp) for the native diffusion
engine.

The chat backend ships a prebuilt llama-server; this is the diffusion analogue,
kept deliberately small. stable-diffusion.cpp publishes per-platform release
zips (macOS-arm64/Metal, Linux x86_64 CPU, plus Vulkan / ROCm / Windows
variants), so on the Phase-4 targets (Apple Silicon and CPU) there is nothing to
compile: resolve the right asset, download, extract into
``~/.unsloth/stable-diffusion.cpp``, and the engine's finder picks it up.

``resolve_release_asset`` -- the host -> asset choice -- is a pure function so the
matching matrix is unit-tested without any network. CUDA / ROCm / XPU hosts stay
on diffusers and never need this; it exists for the engines diffusers serves
poorly.

Usage:
    python studio/install_sd_cpp_prebuilt.py            # auto-detect host
    python studio/install_sd_cpp_prebuilt.py --accelerator vulkan
    python studio/install_sd_cpp_prebuilt.py --print-asset   # resolve only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Sequence

# Default source: the Unsloth mirror's CPU/Apple prebuilts (override with UNSLOTH_SD_CPP_REPO). GPU hosts run diffusers, so only CPU/Apple assets are needed.
DEFAULT_REPO = "unslothai/stable-diffusion.cpp"
# Fallback when the mirror cannot serve this host (release missing, or a host we do not build).
UPSTREAM_FALLBACK_REPO = "leejet/stable-diffusion.cpp"
# Pinned for reproducibility; UNSLOTH_SD_CPP_TAG overrides (empty tracks latest). A missing tag falls back to latest.
#
# The -u<id> suffix means the mirror built upstream master-813-bfbef5b plus the patch set in its
# patches/ directory, and the id is the hash of that set. MiniMax-H3 needs it: an unpatched build
# aborts on the default --cfg-scale, aborts again on --vae-on-cpu, and quantizes H3's 1-D norms
# into an output uncorrelated with its own bf16 reference. All three fixes are open upstream
# (leejet/stable-diffusion.cpp#1861, #1862, #1863) and the patches are deleted once they ship
# there, at which point this pin goes back to a plain upstream tag.
#
# leejet has no release under this name, so the upstream fallback cannot match the pin and drops
# to leejet's latest, which is the documented behaviour for a mirror-only tag.
DEFAULT_TAG = "master-813-bfbef5b-u13b9d92"

# Back-compat alias (some callers/tests import REPO).
REPO = DEFAULT_REPO

# ``master-<n>-<sha>-u<id>``: the mirror's marker for "upstream tree plus our patches/".
_MIRROR_ONLY_TAG_RE = re.compile(r"^master-\d+-[0-9a-f]+-u[0-9a-f]+$")

# What the managed directory records about the install it holds, so a later ensure_* can tell a CPU
# bundle from a CUDA one instead of reusing whatever binary happens to be on disk.
INSTALL_RECORD = ".unsloth-sd-cpp-install.json"

# Marks the directory as one Unsloth created, so an uninstall never wipes a user's own tree.
OWNERSHIP_MARKER = ".unsloth-studio-owned"


def accelerator_class(accelerator: Optional[str]) -> str:
    """The accelerator an install actually serves. ``auto`` resolves to the plain build, so it and
    ``cpu`` are the same install and must not look like an upgrade to one another."""
    accel = (accelerator or "auto").strip().lower()
    return "cpu" if accel in ("auto", "cpu", "") else accel


def read_install_record(root: Path) -> dict:
    """The install record in ``root``, or ``{}`` when there is none (an install predating the
    record, or a directory that is not ours). Never raises."""
    try:
        with open(root / INSTALL_RECORD, "r", encoding = "utf-8") as f:
            rec = json.load(f)
        return rec if isinstance(rec, dict) else {}
    except (OSError, ValueError):
        return {}


def installed_accelerator(root: Path) -> Optional[str]:
    """The accelerator class the install in ``root`` was built for, or None when unrecorded.

    The memo wins over the file, but only while the file is exactly the one it could not replace.
    It exists for a record that stayed readable and stale (still naming the PREVIOUS accelerator)
    after a successful install, where trusting the file re-downloads the bundle on every selection.
    Once anything else rewrites that file -- the installer CLI, another Unsloth -- the content no
    longer matches the snapshot and the file is the newer answer again.

    Only a SUCCESSFUL read that disagrees retires the memo. A record we cannot read right now is
    not a newer answer: the memo's own reason for existing is a record this process could not
    write, and on Windows another writer holding that file open reads as a failure. Dropping the
    memo on it would hand the next selection the stale accelerator it was protecting against, and
    that is a multi-GB reinstall on every load."""
    memo = _INSTALLED_ACCELERATOR_MEMO.get(str(root))
    val = None
    if memo is not None:
        raw = _raw_install_record(root)
        if raw is None or raw == memo[1]:
            val = memo[0]
        else:
            _INSTALLED_ACCELERATOR_MEMO.pop(str(root), None)
    val = val or read_install_record(root).get("accelerator")
    return val if isinstance(val, str) and val else None


# Set ONLY when the on-disk record could not be written: install root -> (accelerator, the raw
# record bytes seen at that moment). An unwritable record (a read-only file, a directory in its
# place) otherwise means the accelerator reads as the PREVIOUS one, or as unknown, forever -- and
# either is a mismatch for a GPU target, so every later engine selection re-downloads the same
# multi-GB bundle. Keyed on the snapshot so it only speaks for the record it saw: once anything
# else updates that file (the installer CLI, another Unsloth), the file is newer and wins again.
_INSTALLED_ACCELERATOR_MEMO: dict[str, tuple[str, Optional[str]]] = {}


def _raw_install_record(root: Path) -> Optional[str]:
    """The record file's bytes as text, or None when it cannot be read. Never raises.

    None is "cannot tell", NOT empty content: an absent file, a directory in its place and a
    transient permission/sharing failure all land here, and none of them proves the record was
    rewritten. Callers must not treat it as a value that differs from a snapshot."""
    try:
        with open(root / INSTALL_RECORD, "r", encoding = "utf-8") as f:
            return f.read()
    except Exception:  # noqa: BLE001 -- absent / unreadable / a directory: all "cannot tell"
        return None


# The same, for the bundle's sd-server capability. Memoised alongside the accelerator or not at
# all: with only half of it remembered, an unwritable record leaves a serverless install looking
# server-capable, and the load that finds a mismatched legacy server keeps reinstalling.
_INSTALLED_SHIPS_SERVER_MEMO: dict[str, bool] = {}


def installed_ships_server(root: Path) -> Optional[bool]:
    """Whether the bundle installed in ``root`` carried an sd-server, or None when unrecorded.

    None is the honest answer for every install that predates this field, and callers must treat
    it as "unknown" rather than "serverless": a missing sd-server is otherwise indistinguishable
    from one a bundle never shipped, and suppressing the reinstall on a guess would strand a tree
    whose server was deleted (by hand, or by the runnability repair) on the one-shot CLI forever.

    The memo WINS over the file, for the same reason ``installed_accelerator``'s does: it is only
    set by an install that completed in this process, and the case it exists for is a record that
    could not be written, where the file is stale or absent."""
    memo = _INSTALLED_SHIPS_SERVER_MEMO.get(str(root))
    val = memo if memo is not None else read_install_record(root).get("ships_server")
    return val if isinstance(val, bool) else None


def _write_install_record(
    root: Path,
    *,
    accelerator: str,
    repo: str,
    tag: Optional[str],
    ships_server: Optional[bool] = None,
) -> None:
    """Record what this install is, so a later ensure_* can tell a CPU bundle from a GPU one.

    The write itself stays best-effort -- a metadata failure must not throw away binaries that
    extracted correctly -- but the answer is memoised either way, so this process never re-installs
    what it just installed."""
    klass = accelerator_class(accelerator)
    rec: dict = {"accelerator": klass, "repo": repo, "tag": tag}
    if ships_server is not None:
        rec["ships_server"] = ships_server
        _INSTALLED_SHIPS_SERVER_MEMO[str(root)] = ships_server
    else:
        # An install that did not report the capability must not leave an older memo standing in
        # for this one -- the tree is now whatever this bundle put there.
        _INSTALLED_SHIPS_SERVER_MEMO.pop(str(root), None)
    try:
        with open(root / INSTALL_RECORD, "w", encoding = "utf-8") as f:
            json.dump(rec, f)
    except OSError as exc:
        # Remember it, pinned to the record we could not replace.
        _INSTALLED_ACCELERATOR_MEMO[str(root)] = (klass, _raw_install_record(root))
        print(
            f"sd-cli: WARNING could not write the install record in {root}: {exc}; "
            f"remembering {klass} for this process only",
            flush = True,
        )
    else:
        # The file is authoritative again, so a memo from an earlier failed write must not outlive it.
        _INSTALLED_ACCELERATOR_MEMO.pop(str(root), None)


def _repo() -> str:
    return (os.environ.get("UNSLOTH_SD_CPP_REPO") or DEFAULT_REPO).strip() or DEFAULT_REPO


def _pinned_tag() -> Optional[str]:
    """The release tag to install: env override, else the pinned default; '' = latest."""
    val = os.environ.get("UNSLOTH_SD_CPP_TAG", DEFAULT_TAG).strip()
    return val or None


def is_mirror_only_tag(tag: Optional[str]) -> bool:
    """True for a ``<upstream tag>-u<id>`` tag, which only the Unsloth mirror can serve.

    The mirror publishes such a tag when it builds the upstream tree plus the patch set in its
    ``patches/`` directory, so by construction upstream has no release under that name."""
    return bool(tag) and bool(_MIRROR_ONLY_TAG_RE.match(tag))


# A mirror release built on top of an upstream one carries a "-u<short sha>" suffix naming the
# fork commit (master-813-bfbef5b-u13b9d92 is upstream master-813-bfbef5b plus fork commit 13b9d92).
_MIRROR_TAG_SUFFIX = re.compile(r"-u[0-9a-f]{7,}$")


def upstream_tag_for(tag: Optional[str]) -> Optional[str]:
    """The upstream release ``tag`` was built from: the same tag with the mirror's fork suffix
    dropped, or ``tag`` unchanged when it carries none.

    Without this, pinning a fork-only tag silently costs every host the mirror does not build
    (Linux Vulkan/ROCm, Windows GPU) its pinned install: the exact string 404s upstream and the
    fallback settles for upstream *latest*, which is any build published since."""
    if not tag:
        return tag
    return _MIRROR_TAG_SUFFIX.sub("", tag) or tag


# accelerator -> the token that must appear in a Linux/Windows asset name.
_LINUX_ACCEL_TOKEN = {"rocm": "rocm", "vulkan": "vulkan"}
_WINDOWS_ACCEL_TOKEN = {
    "cuda": "cuda12",
    "vulkan": "vulkan",
    "rocm": "rocm",
    "cpu": "avx2",
    "auto": "avx2",
}
# Tokens that mark an accelerator-specific Linux build; "auto"/"cpu" want none of them.
_LINUX_ACCEL_MARKERS = ("rocm", "vulkan", "cuda", "sycl", "musa")

_ARCH_TOKENS = {
    "x86_64": ("x86_64", "x64", "amd64"),
    "amd64": ("x86_64", "x64", "amd64"),
    "arm64": ("arm64", "aarch64"),
    "aarch64": ("arm64", "aarch64"),
}


def _arch_tokens(machine: str) -> tuple[str, ...]:
    return _ARCH_TOKENS.get(machine.lower(), (machine.lower(),))


def resolve_release_asset(
    asset_names: Sequence[str],
    *,
    system: str,
    machine: str,
    accelerator: str = "auto",
) -> Optional[str]:
    """Pick the best release asset for a host, or None if none matches.

    ``system`` / ``machine`` are ``platform.system()`` / ``platform.machine()``
    values; ``accelerator`` is ``auto`` (CPU/Metal default), ``vulkan``,
    ``rocm``, or ``cuda`` (Windows only). Pure -- the caller passes the release's
    asset name list.
    """
    system = system.lower()
    accel = accelerator.lower()
    arch = _arch_tokens(machine)
    zips = [
        a for a in asset_names if a.lower().endswith(".zip") and not a.lower().startswith("cudart")
    ]

    if system == "darwin":
        pool = [
            a
            for a in zips
            if ("darwin" in a.lower() or "macos" in a.lower()) and any(t in a.lower() for t in arch)
        ]
        return pool[0] if pool else None

    if system == "windows":
        # Filter by host arch: an arm64 host must not install an unrunnable x64 sd-cli. No match returns None so the caller falls back.
        pool = [a for a in zips if "bin-win" in a.lower() and any(t in a.lower() for t in arch)]
        token = _WINDOWS_ACCEL_TOKEN.get(accel, accel)
        sel = [a for a in pool if token in a.lower()]
        if sel:
            return sel[0]
        # An explicit GPU accelerator with no asset returns None, so the caller falls back instead of installing a CPU build.
        if accel in ("cuda", "vulkan", "rocm"):
            return None
        # auto / cpu -> a plain avx2 CPU build, else any windows build.
        cpu = [a for a in pool if "avx2" in a.lower()]
        return cpu[0] if cpu else (pool[0] if pool else None)

    # linux (and anything else unix-like)
    pool = [a for a in zips if "linux" in a.lower() and any(t in a.lower() for t in arch)]
    if accel in ("cuda", "vulkan", "rocm"):
        # Explicit GPU accelerator: require its marker, never hand back a plain CPU build.
        marker = _LINUX_ACCEL_TOKEN.get(accel, accel)
        sel = [a for a in pool if marker in a.lower()]
    else:  # auto / cpu -> the plain build with no accelerator marker
        sel = [a for a in pool if not any(m in a.lower() for m in _LINUX_ACCEL_MARKERS)]
    return sel[0] if sel else None


def _fetch_release(
    tag: Optional[str] = None,
    *,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = 30.0,
    allow_latest: bool = True,
) -> Optional[dict]:
    """GET a release JSON from GitHub. With ``tag`` set, fetch that exact release; otherwise
    fetch latest. ``token`` is optional and lifts the API rate limit.

    When the pinned ``tag`` is missing (404): if ``allow_latest`` fall back to that repo's
    latest, else return ``None`` so the caller can try the SAME pin on another repo before
    settling for any repo's unpinned latest."""
    repo = repo or _repo()
    token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

    def _get(url: str) -> dict:
        req = urllib.request.Request(url, headers = {"Accept": "application/vnd.github+json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout = timeout) as resp:  # noqa: S310 (fixed https host)
            return json.loads(resp.read().decode("utf-8"))

    base = f"https://api.github.com/repos/{repo}/releases"
    if tag:
        try:
            return _get(f"{base}/tags/{tag}")
        except urllib.error.HTTPError as exc:  # pinned tag removed -> maybe latest
            if exc.code != 404:
                raise
            if not allow_latest:
                return None
            print(
                f"sd-cli: pinned tag {tag} not found on {repo}; falling back to latest", flush = True
            )
    return _get(f"{base}/latest")


# Back-compat alias: the old name fetched latest.
def _fetch_latest_release(*, token: Optional[str] = None, timeout: float = 30.0) -> dict:
    return _fetch_release(None, token = token, timeout = timeout)


def _verify_sha256(path: Path, expected_digest: Optional[str]) -> None:
    """Verify ``path`` against a GitHub asset ``digest`` ('sha256:<hex>'). Integrity check
    against a corrupted/tampered download before we extract + execute the binary. When the
    release publishes no digest (older releases), warn and proceed rather than hard-fail."""
    if not expected_digest:
        print(f"sd-cli: WARNING no digest for {path.name}; cannot verify integrity", flush = True)
        return
    algo, _, want = expected_digest.partition(":")
    if algo.lower() != "sha256" or not want:
        print(
            f"sd-cli: WARNING unrecognised digest {expected_digest!r}; skipping check", flush = True
        )
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != want.lower():
        raise RuntimeError(f"sha256 mismatch for {path.name}: expected {want.lower()}, got {got}")


def default_install_dir() -> Path:
    """``<UNSLOTH_STUDIO_HOME>/stable-diffusion.cpp``, else the legacy
    ``~/.unsloth/stable-diffusion.cpp``.

    The same placement ``install_llama_prebuilt.default_managed_llama_dir`` uses for
    llama.cpp, and the whisper.cpp / node installs use for theirs: the tree goes
    *under* the Unsloth home, so side-by-side Unsloth instances stay isolated and nothing outside
    the home is ever claimed. The legacy default home ``~/.unsloth/studio`` still maps
    to ``~/.unsloth/stable-diffusion.cpp`` so an existing install is reused.

    Kept byte-identical in meaning to ``sd_cpp_engine.managed_install_root``; the two
    are separate because this script must run standalone, before the backend package is
    importable.

    Derived from an absolute home: a relative ``UNSLOTH_STUDIO_HOME`` must not leave the
    install dir relative to whatever the working directory happens to be."""
    home = (os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or "").strip()
    legacy = Path.home() / ".unsloth" / "stable-diffusion.cpp"
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
    return legacy if is_legacy else root / "stable-diffusion.cpp"


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _locate_sd_cli(root: Path) -> Optional[Path]:
    name = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


class SupersededBinaryError(RuntimeError):
    """An install failed PART WAY through replacing the managed tree.

    Distinct from every other install failure because the tree is now a mixture of two bundles.
    ``ensure_*`` must not record it as a failed accelerator upgrade: that memo suppresses the
    mismatch for the whole process, so the mixed tree would be accepted forever instead of being
    retried, which is the opposite of what withholding the install record is for."""


def _binary_names() -> tuple[str, ...]:
    """The executables the managed tree may hold, in both platform spellings."""
    suffix = ".exe" if sys.platform == "win32" else ""
    return (f"sd-cli{suffix}", f"sd-server{suffix}")


def _binary_key(path: Path) -> Path:
    """How a binary is compared: parents resolved, final component left alone.

    Both halves matter now that the tree holds symlinks. Resolving the leaf would turn a bundle
    shipping ``sd-cli -> sd-cli-1.2`` into a name no member has; leaving the parents lexical
    would spell a member under a symlinked directory differently from ``rglob``, and the sweep
    would delete the binary this bundle just supplied."""
    p = Path(os.path.abspath(path))
    return Path(os.path.realpath(p.parent)) / p.name


def _archive_binary_paths(zf: zipfile.ZipFile, target: Path) -> set[Path]:
    """Where this archive puts its executables, as absolute LEXICAL paths under ``target``.

    Read from the MEMBER LIST, never from the extracted tree: a leftover binary from an earlier
    install looks identical on disk once extraction has run, which is the whole confusion here.

    Unresolved on purpose: extraction can change what the parents mean, so resolving here would
    key the binary off a layout that no longer exists by the time the sweep runs."""
    names = _binary_names()
    out: set[Path] = set()
    for member in zf.namelist():
        if member.rsplit("/", 1)[-1] in names:
            out.add(Path(os.path.abspath(target / member)))
    return out


def _tree_has_binaries(root: Path) -> bool:
    """True when ``root`` already holds an sd-cli / sd-server from an earlier bundle.

    Asked BEFORE extraction, because that is the only moment the answer is about the previous
    bundle alone."""
    for name in _binary_names():
        for found in root.rglob(name):
            if found.is_file():
                return True
    return False


def _discard_superseded_binaries(root: Path, supplied: set[Path]) -> None:
    """Remove managed sd-cli / sd-server copies this bundle did NOT write.

    Extraction MERGES into the tree, so a bundle whose layout differs from the previous one (or
    that ships no server at all) leaves the old accelerator's executables behind -- and
    ``_layout_candidates`` prefers ``build/bin`` over the prebuilt's versioned subdirectory, so the
    stale one keeps winning. Nothing downstream repairs that: a leftover binary is still RUNNABLE,
    so ``_usable_or_discard_managed`` keeps it, and once the record below names the new accelerator
    ``_accelerator_changed`` trusts the tree and serves the old build forever.

    Raises when a copy cannot go, which withholds the record and makes the next load retry."""
    names = _binary_names()
    # Resolved HERE, not when ``supplied`` was built: the tree is final only now, and extraction
    # may have replaced a directory symlink a member path was spelled through.
    keys = {_binary_key(p) for p in supplied}
    for name in names:
        for found in sorted(root.rglob(name)):
            if not found.is_file() or _binary_key(found) in keys:
                continue
            try:
                found.unlink()
            except OSError as exc:
                raise SupersededBinaryError(
                    f"could not remove the superseded binary {found}: {exc}. It was not written by "
                    f"this bundle, and leaving it would keep serving the previous accelerator."
                ) from exc
            print(f"removed a superseded binary -> {found}", flush = True)


def _locate_sd_server(root: Path) -> Optional[Path]:
    """The persistent ``sd-server`` binary in the extracted tree, if the archive ships
    one (modern stable-diffusion.cpp releases do). Best-effort: the native backend
    falls back to one-shot ``sd-cli`` when it is absent."""
    name = "sd-server.exe" if sys.platform == "win32" else "sd-server"
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


def _download(
    url: str,
    dest: Path,
    *,
    timeout: float = 300.0,
) -> None:
    """Stream ``url`` to ``dest`` with an explicit timeout. ``urlretrieve`` takes no
    timeout and can hang forever on a stalled socket. A User-Agent is set because the
    GitHub asset CDN can reject header-less requests; the API fetch carries any token."""
    import shutil

    req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-sd-cpp-installer"})
    with urllib.request.urlopen(req, timeout = timeout) as resp, open(dest, "wb") as f:  # noqa: S310
        shutil.copyfileobj(resp, f)


# PATH_MAX: a link payload is a pathname and ``zf.read`` holds it in memory, so anything larger
# is a decompression bomb rather than a library name.
_MAX_LINK_TARGET_BYTES = 4096

# What Linux allows before ELOOP, so a deeper layout is one the loader could not read anyway.
_MAX_LINK_DEPTH = 40


# Creators whose ``external_attr`` high bits are a Unix ``st_mode``: 3 (Info-ZIP, CPython) and
# 19 (Apple's ditto, same layout). FAT and NTFS keep DOS attribute flags there, so reading a mode
# out of one would invent symlinks the archive never described.
_UNIX_CREATORS = (3, 19)


def _is_symlink_member(member: zipfile.ZipInfo) -> bool:
    """``external_attr``'s high bits are a Unix mode only when a Unix host wrote the entry."""
    return member.create_system in _UNIX_CREATORS and stat.S_ISLNK(member.external_attr >> 16)


def _checked_link_target(
    zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest: Path, base: Path
) -> str:
    """A symlink member's payload, refused unless it is relative and stays inside ``base``.

    Rejects absolute (including Windows drive-relative, which Win32 resolves against that drive's
    cwd), empty, NUL-bearing, self-referential and escaping targets."""
    if member.file_size > _MAX_LINK_TARGET_BYTES:
        raise RuntimeError(f"oversized symlink target in archive: {member.filename!r}")
    link_target = zf.read(member).decode("utf-8", "surrogateescape")
    unsafe = (
        # Shape first: resolve() stats the path, and a NUL in it raises ValueError.
        not link_target
        or "\x00" in link_target
        or PurePosixPath(link_target).is_absolute()
        or bool(PureWindowsPath(link_target).drive)
    )
    if not unsafe:
        resolved = (dest.parent / link_target).resolve()
        # Self-check lexically: dest may already be a correct link to this very target.
        unsafe = os.path.normpath(dest.parent / link_target) == os.path.normpath(dest) or (
            resolved != base and base not in resolved.parents
        )
    if unsafe:
        raise RuntimeError(f"unsafe symlink in archive: {member.filename!r} -> {link_target!r}")
    return link_target


def _plan_resolve(
    path: Path,
    base: Path,
    replaced: set[str],
    archive: dict,
    depth: int = 0,
) -> Path:
    """``path`` with every component resolved as the tree WILL have it after extraction.

    Links this archive ships are followed from the archive (not on disk yet); links it replaces
    are not followed (gone before the first write); anything else is a previous bundle's. Two
    members can point through each other's directories, so a runaway recursion is that cycle."""
    if path == base or base not in path.parents:
        return path
    if depth > _MAX_LINK_DEPTH:
        raise RuntimeError(f"symlink cycle in archive: {str(path.relative_to(base))!r}")
    cur = base
    for part in path.relative_to(base).parts:
        cur = cur / part
        target = archive.get(str(cur))
        if target is not None:
            cur = _plan_resolve(
                Path(os.path.normpath(cur.parent / target)), base, replaced, archive, depth + 1
            )
        elif str(cur) not in replaced and cur.is_symlink():
            cur = Path(os.path.realpath(cur))
    return cur


def _plan_key(dest: Path, base: Path, replaced: set[str], archive: dict) -> Path:
    """Where ``dest`` will really land. Only the parents are resolved: the final component is the
    link about to be created, and following it would compare the wrong thing."""
    if dest == base or base not in dest.parents:
        return dest
    return _plan_resolve(dest.parent, base, replaced, archive) / dest.name


def _safe_extractall(zf: zipfile.ZipFile, target: Path) -> None:
    """``extractall`` with a per-member containment check, so an archive carrying an absolute
    path or a ``..`` entry can't write outside ``target`` (Zip-Slip).

    Symlink members are RECREATED rather than extracted: CPython's ``zipfile`` writes a symlink's
    payload (the link target text) as a regular file, which flattens the ``lib*.so`` links
    upstream sd.cpp releases ship and leaves ``sd-cli`` with ``file too short`` libraries (#9268).

    Everything is decided BEFORE the first write, so a refused archive leaves the install it was
    refused over exactly as it was."""
    base = target.resolve()
    # The installer writes these itself. A link at one makes _write_install_record follow it and
    # overwrite the target while the record still reads back, so a broken install reports success.
    reserved = {base / INSTALL_RECORD, base / OWNERSHIP_MARKER}
    links: list[tuple[Path, str, zipfile.ZipInfo]] = []
    plain: list[zipfile.ZipInfo] = []
    written: list[tuple[Path, str]] = []
    for member in zf.infolist():
        # extractall DROPS ".." instead of cancelling the component before it, so "a/.." is "a" to
        # it and normalising here would check a path it never writes. No release ships one.
        if ".." in PurePosixPath(member.filename).parts:
            raise RuntimeError(f"unsafe path in archive: {member.filename!r}")
        # Lexical, never resolve()d: extraction MERGES, and resolving would follow the previous
        # install's link and leave the real library replaced by a link to itself.
        dest = Path(os.path.normpath(base / member.filename))
        checked = dest.resolve()
        if (dest != base and base not in dest.parents) or (
            checked != base and base not in checked.parents
        ):
            raise RuntimeError(f"unsafe path in archive: {member.filename!r}")
        written.append((dest, member.filename))
        if _is_symlink_member(member):
            links.append((dest, _checked_link_target(zf, member, dest, base), member))
        else:
            plain.append(member)
    # No member may sit under a directory this archive turns into a link: extraction would write
    # through it, and catching that at creation time means part of the tree is already replaced.
    link_dests = {d for d, _, _ in links}
    for dest, filename in written:
        for parent in dest.parents:
            if parent == base:
                break
            if parent in link_dests:
                raise RuntimeError(
                    f"unsafe path in archive: {filename!r} is under a symlink member"
                )
    # Every destination this archive writes becomes its own member, so a stale link at one must
    # not be followed.
    replaced = {str(d) for d, _ in written}
    archive = {str(d): t for d, t, _ in links}
    replaced |= {str(_plan_key(d, base, replaced, archive)) for d, _ in written}
    # A member path is not where the link lands: a previous bundle's alias -> . puts alias/<record>
    # on the record itself, which a lexical compare misses. Same for a directory already there.
    keys = {str(d): _plan_key(d, base, replaced, archive) for d, _, _ in links}
    for dest, link_target, member in links:
        key = keys[str(dest)]
        if key in reserved:
            raise RuntimeError(f"symlink at a reserved installer path: {member.filename!r}")
        # Nor may one POINT at them: the marker is already there on a root Unsloth owns, so
        # sd-cli -> marker leaves _locate_sd_cli reporting an empty file as the executable.
        landing = _plan_resolve(
            Path(os.path.normpath(key.parent / link_target)), base, replaced, archive
        )
        if landing in reserved:
            raise RuntimeError(f"symlink onto a reserved installer path: {member.filename!r}")
        if key.is_dir() and not key.is_symlink():
            raise RuntimeError(f"symlink member collides with a directory: {member.filename!r}")
    # Chains are normal (libwebp.so -> .so.7 -> .so.7.2.0) but must terminate: a cycle installs a
    # library nothing can read, so every load reinstalls it. Walk the graph the tree WILL have,
    # keyed by landing point so alias/a and real/b count as one cycle.
    by_dest = {str(keys[str(d)]): t for d, t, _ in links}
    for dest, _, member in links:
        seen, cur, hops = set(), str(keys[str(dest)]), 0
        while cur not in seen:
            seen.add(cur)
            if cur in by_dest:
                nxt = by_dest[cur]
            elif cur not in replaced and os.path.islink(cur):
                nxt = os.readlink(cur)
            else:
                break
            # Counted per hop FOLLOWED, not per node seen: the file a chain ends at is not a
            # traversal. A chain the kernel cannot walk is unreadable for the same reason a
            # cycle is, and terminating does not save it, so it is refused rather than installed.
            hops += 1
            if hops > _MAX_LINK_DEPTH:
                raise RuntimeError(f"symlink chain too deep in archive: {member.filename!r}")
            nxt = Path(os.path.normpath(os.path.join(os.path.dirname(cur), nxt)))
            # a -> a/x never reaches a second node, so repetition never fires, yet resolving a
            # walks a again. Anything under the link is a loop.
            if Path(cur) in nxt.parents:
                raise RuntimeError(f"symlink cycle in archive: {member.filename!r}")
            cur = str(_plan_key(nxt, base, replaced, archive))
        else:
            raise RuntimeError(f"symlink cycle in archive: {member.filename!r}")
    # Last thing decided before the first write: can this filesystem hold links at all? Some mounts
    # (exFAT, SMB without unix extensions) refuse, and learning that at creation time means
    # extractall has already put the new binary over the working one.
    if links:
        # extractall would create the tree itself, so the probe must not be what needs it first.
        base.mkdir(parents = True, exist_ok = True)
        # A probe a killed install left behind must not answer for this one: symlink_to raises
        # EEXIST on an existing path, which reads below as "no symlink support", and a restarted
        # container reuses the pid while the directory persists. Sweep stragglers, and take a
        # unique name so a concurrent install cannot collide either.
        for stale in base.glob(".unsloth-symlink-probe-*"):
            try:
                stale.unlink()
            except OSError:
                pass
        probe = base / f".unsloth-symlink-probe-{os.getpid()}-{os.urandom(4).hex()}"
        try:
            probe.symlink_to(".")
        except OSError as exc:
            # Windows outside developer mode is the one place flattening is right; see below.
            if sys.platform != "win32":
                raise RuntimeError(f"this filesystem cannot store symlinks: {exc}") from exc
        else:
            # missing_ok: a concurrent install's sweep may have taken this one already.
            probe.unlink(missing_ok = True)
    # extractall opens each destination "wb", which FOLLOWS a link a previous bundle left and
    # writes the member into its target. Drop stale links first: a name one bundle ships as a link
    # the next can ship as a file (the mirror ships copies where upstream ships links).
    for dest, _ in written:
        if dest.is_symlink():
            dest.unlink()
    zf.extractall(target, members = plain)
    for dest, link_target, member in links:
        # Re-resolved HERE: the pass above ran before any link existed, so an earlier member
        # (a -> .) can turn a later member's parent into a link and send this outside base. Both
        # ends are checked for containment, not for being link-free, so a tree that legitimately
        # symlinks its own subdirectory still installs.
        parent = Path(os.path.realpath(dest.parent))
        resolved = (dest.parent / link_target).resolve()
        if (parent != base and base not in parent.parents) or (
            resolved != base and base not in resolved.parents
        ):
            raise RuntimeError(f"unsafe symlink in archive: {member.filename!r} -> {link_target!r}")
        if dest.is_dir() and not dest.is_symlink():
            raise RuntimeError(f"symlink member collides with a directory: {member.filename!r}")
        dest.parent.mkdir(parents = True, exist_ok = True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        try:
            dest.symlink_to(link_target)
        except OSError as exc:
            # Windows outside developer mode cannot create a link, and every Windows asset ships
            # plain files, so flattening there costs nothing and keeps an install that used to
            # finish finishing. Anywhere else a refusal means the filesystem cannot hold the layout
            # sd-cli needs, and writing the link text back is the "file too short" install of #9268.
            if sys.platform != "win32":
                raise RuntimeError(
                    f"could not restore the symlink {member.filename!r}: {exc}"
                ) from exc
            # Load-bearing assumption: no published Windows asset ships a symlink member, so
            # this writes the link text back only for an archive that never reaches a user.
            # If that ever changes, this branch produces the #9268 install silently while the
            # one above it raises, and it should be revisited rather than left as a fallback.
            zf.extract(member, target)


def _maybe_fetch_windows_cudart(release: dict, chosen: str, target: Path) -> None:
    """On Windows + a CUDA build, also fetch the separate CUDA-runtime DLL archive.

    Upstream ships the runtime as ``cudart-sd-...-win-cu12-...zip`` (which
    ``resolve_release_asset`` filters out); without those DLLs ``sd-cli.exe`` cannot start
    on a machine that does not already have the CUDA runtime installed."""
    if platform.system().lower() != "windows" or "cuda" not in chosen.lower():
        return
    cudart = next(
        (
            a
            for a in release.get("assets", [])
            if a["name"].lower().startswith("cudart") and "win" in a["name"].lower()
        ),
        None,
    )
    if cudart is None:
        return
    dest = target / cudart["name"]
    print(f"downloading CUDA runtime {cudart['name']} ...", flush = True)
    try:
        _download(cudart["browser_download_url"], dest)
        # Verify integrity BEFORE extracting: these DLLs load into sd-cli.exe, so a tampered archive must be rejected.
        _verify_sha256(dest, cudart.get("digest"))
        with zipfile.ZipFile(dest) as zf:
            _safe_extractall(zf, target)
    finally:
        dest.unlink(missing_ok = True)


def _resolve_repo_asset(
    repo: str,
    tag: Optional[str],
    accelerator: str,
    token: Optional[str],
    *,
    allow_latest: bool = True,
) -> tuple[Optional[dict], Optional[str]]:
    """Fetch ``repo``'s release and pick the asset for this host. Returns
    ``(release, asset_name)`` or ``(None, None)`` when the repo has no usable release
    (fetch failed, or the pinned tag is missing and ``allow_latest`` is False) or no
    asset for this host, so the caller can fall back."""
    try:
        release = _fetch_release(tag, repo = repo, token = token, allow_latest = allow_latest)
    except Exception as exc:  # noqa: BLE001 - network / rate limit -> fall back
        print(f"sd-cli: {repo} release fetch failed ({exc})", flush = True)
        return None, None
    if release is None:  # pinned tag missing and the latest fallback was withheld
        return None, None
    names = [a["name"] for a in (release.get("assets") or [])]
    chosen = resolve_release_asset(
        names,
        system = platform.system(),
        machine = platform.machine(),
        accelerator = accelerator,
    )
    return release, chosen


def _resolve_with_fallback(
    accelerator: str, token: Optional[str]
) -> tuple[str, Optional[dict], Optional[str]]:
    """Resolve ``(used_repo, release, asset_name)`` for this host across the primary repo
    and -- only when the built-in default is in use and the user did not pin a repo -- the
    upstream fallback.

    Ordering guarantees reproducibility: a pinned tag is tried EXACTLY on every candidate
    repo before any repo's unpinned latest, so a mirror that is missing the pinned release
    prefers the pinned upstream build over an unpinned mirror-latest. Returns
    ``(primary, None, None)`` when nothing serves this host. Shared by ``install`` and
    ``--print-asset`` so both honour the same fallback."""
    tag = _pinned_tag()
    primary = _repo()
    # Only substitute upstream when no UNSLOTH_SD_CPP_REPO is pinned: an explicit repo gets exactly that repo.
    repo_pinned = bool((os.environ.get("UNSLOTH_SD_CPP_REPO") or "").strip())
    allow_upstream = (
        not repo_pinned and primary == DEFAULT_REPO and DEFAULT_REPO != UPSTREAM_FALLBACK_REPO
    )
    mirror_only = is_mirror_only_tag(tag)

    # (repo, tag_to_fetch, allow_latest): with a pin, try the exact pin on every repo first, then each repo's latest.
    attempts: list[tuple[str, Optional[str], bool]] = []
    if tag:
        attempts.append((primary, tag, False))
        if allow_upstream:
            # The mirror's own tag does not exist upstream, so the pin has to be translated back
            # to the upstream release it was built from -- otherwise this attempt always 404s and
            # the pin degrades to upstream latest for every host the mirror does not build.
            # This supersedes simply skipping the attempt for a mirror-only tag: skipping kept the
            # round trip cheap but dropped the pin entirely on those hosts.
            attempts.append((UPSTREAM_FALLBACK_REPO, upstream_tag_for(tag), False))
        attempts.append((primary, None, True))
        if allow_upstream:
            attempts.append((UPSTREAM_FALLBACK_REPO, None, True))
    else:
        attempts.append((primary, None, True))
        if allow_upstream:
            attempts.append((UPSTREAM_FALLBACK_REPO, None, True))

    for repo, want_tag, allow_latest in attempts:
        release, chosen = _resolve_repo_asset(
            repo, want_tag, accelerator, token, allow_latest = allow_latest
        )
        if release is not None and chosen:
            if repo != primary:
                # stderr, not stdout: --print-asset documents its stdout as the asset name only.
                print(
                    f"falling back to {repo} for {platform.system()}/{platform.machine()}",
                    file = sys.stderr,
                    flush = True,
                )
                # A mirror-only pin means the shipped default carries fixes that upstream has
                # not released. Falling back is still better than no native engine at all for
                # every other model, but the H3 failures are SILENT (it renders, just wrongly),
                # so this has to be said out loud rather than left to the generic line above.
                if mirror_only:
                    print(
                        f"warning: {repo} has no {tag}; this build lacks the MiniMax-H3 fixes, "
                        "so H3 will abort on the default cfg-scale and on --vae-on-cpu, and a "
                        "blanket --type will quantize its 1-D norms into a broken render. Other "
                        "models are unaffected.",
                        file = sys.stderr,
                        flush = True,
                    )
            return repo, release, chosen
    return primary, None, None


def install(
    *,
    install_dir: Optional[Path] = None,
    accelerator: str = "auto",
    token: Optional[str] = None,
) -> Path:
    """Download + extract the prebuilt for this host. Returns the sd-cli path.

    Resolves against the Unsloth mirror (``DEFAULT_REPO``) first; if the mirror can't
    serve this host (release missing, or a host we don't build) AND the default repo is
    in use, falls back to leejet upstream so native install still works. Raises
    ``RuntimeError`` only when neither source has an asset for the host, or the archive
    has no ``sd-cli``.
    """
    target = install_dir or default_install_dir()
    # Claim ownership of `target` only if we created it, it was empty, or it is already marked: adopting a user's non-empty dir would let a later uninstall wipe it.
    marker = target / OWNERSHIP_MARKER
    _may_own = True
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"sd.cpp install target is not a directory: {target}")
        try:
            _pre_existing_entries = any(target.iterdir())
        except OSError:
            _pre_existing_entries = True
        # Empty dir, or one we already own, may be (re)claimed; a non-empty unowned dir may not.
        _may_own = (not _pre_existing_entries) or marker.is_file()
    # Refuse to extract into a pre-existing non-empty dir we do not own: merging would overwrite the user's files.
    if not _may_own:
        raise RuntimeError(
            f"sd.cpp install target already exists and is not an Unsloth-managed directory: {target}. "
            f"Refusing to extract prebuilt binaries into it to avoid overwriting or mixing them "
            f"into your files. Remove or move that directory, or install into a different, empty "
            f"location (pass a different --install-dir / set the Unsloth sd.cpp install dir)."
        )
    used_repo, release, chosen = _resolve_with_fallback(accelerator, token)

    if release is None or not chosen:
        raise RuntimeError(
            f"No prebuilt sd-cli for {platform.system()}/{platform.machine()} "
            f"(accelerator={accelerator}) from {used_repo}. Build from source: "
            f"https://github.com/{used_repo}"
        )
    print(f"sd-cli: source {used_repo} release {release.get('tag_name', '?')}", flush = True)
    asset = next(a for a in release["assets"] if a["name"] == chosen)
    url = asset["browser_download_url"]
    target.mkdir(parents = True, exist_ok = True)
    # Claim ownership BEFORE any partial write: an interrupted extract leaves the target non-empty, and without
    # the marker the next install would trip the refusal above. Only set when _may_own, so it never adopts user files.
    if _may_own:
        try:
            marker.touch()
        except OSError:
            pass
    archive = target / chosen
    # Set the moment the tree stops being purely the OLD bundle. From then on it is a mixture of
    # two bundles and every later failure has to be reported as an incomplete replacement, not as
    # "this accelerator is unavailable".
    replacing = False
    print(f"downloading {chosen} -> {archive}", flush = True)
    try:
        _download(url, archive)
        # Verify integrity BEFORE extracting + executing.
        _verify_sha256(archive, asset.get("digest"))
        print("extracting ...", flush = True)
        with zipfile.ZipFile(archive) as zf:
            supplied = _archive_binary_paths(zf, target)
            # The boundary opens HERE, not at the sweep, whenever an existing bundle is about
            # to gain a SECOND copy of a binary. Same path is the obvious case: zipfile rewrites
            # each member in place, so an interrupted extract leaves the old sd-cli truncated.
            # A different layout is the same problem one step removed, because the new copy WINS
            # the next lookup -- _layout_candidates prefers build/bin, then the newest
            # subdirectory by mtime -- while still having had neither the sweep, nor
            # _make_executable, nor (on a Windows CUDA upgrade) the cudart DLLs the fetch below
            # can still fail to get. Either way, calling that an ordinary failure makes ensure_*
            # memoise the accelerator and serve the half-finished copy for the rest of the
            # process. A first install has nothing to compete with, so it stays ordinary and the
            # accelerator really is the thing that was unavailable.
            replacing = bool(supplied) and _tree_has_binaries(target)
            _safe_extractall(zf, target)
        # Nothing is swept until this bundle is known to have supplied the one binary the install
        # cannot do without. A malformed archive is caught below either way, but only AFTER the
        # sweep would have deleted the working sd-cli that ensure_sd_cpp_binary keeps precisely so
        # a failed upgrade still leaves something to generate with.
        cli_name = _binary_names()[0]
        if not any(p.name == cli_name for p in supplied):
            raise RuntimeError(f"archive {chosen} contained no sd-cli binary")
        # Windows CUDA builds need the separately-published cudart runtime DLLs. Still before the
        # sweep, so a failure here costs nothing extra when the old copies are still in place;
        # when the extract above already overwrote them, ``replacing`` is what makes this report
        # as an incomplete replacement instead of a missing accelerator. Nothing it writes is an
        # sd-cli or an sd-server, so the sweep below cannot take it.
        _maybe_fetch_windows_cudart(release, chosen, target)
        # Extraction merges, so anything the previous bundle put somewhere this one does not write
        # survives -- and it outranks the new copy whenever its path sorts higher. Drop what this
        # bundle did not supply, so the tree and the record written below agree.
        #
        # LAST, and past it the tree is mixed whatever the layout was: the caller has to retry the
        # sweep rather than memoise the accelerator as unavailable.
        if _may_own:
            # Set BEFORE the call, not after: the sweep removes copies one at a time, so a failure
            # inside it has already changed the tree.
            replacing = True
            _discard_superseded_binaries(target, supplied)
    except SupersededBinaryError:
        raise
    except Exception as exc:  # noqa: BLE001 -- past the boundary, every failure is a mixed tree
        if not replacing:
            raise
        raise SupersededBinaryError(
            f"the managed tree was left part way through a replacement: {exc}"
        ) from exc
    finally:
        # Always drop the archive: a corrupt or partial one must not linger and defeat a later retry.
        # Inside the boundary too: an unlink failure after the sweep is still a mixed tree.
        try:
            archive.unlink(missing_ok = True)
        except OSError as exc:
            if replacing:
                raise SupersededBinaryError(
                    f"the managed tree was left part way through a replacement: {exc}"
                ) from exc
            raise
    # EVERYTHING from the sweep on is finalisation of a tree that is now a mixture of two bundles:
    # locating the new binaries, chmod-ing them, and the record. Reported as an ordinary failure,
    # any of these makes ensure_* memoise the accelerator as unavailable and hand back a
    # pre-install path the sweep may already have removed. Reported as an incomplete replacement,
    # the caller re-finds what survived and the next load retries.
    try:
        sd_cli = _locate_sd_cli(target)
        if not sd_cli:
            raise RuntimeError(f"archive {chosen} contained no sd-cli binary")
        if sys.platform != "win32":
            _make_executable(sd_cli)
        print(f"installed sd-cli -> {sd_cli}", flush = True)
        # The same archive ships the persistent sd-server; make it runnable so the native backend
        # can prefer it.
        sd_server = _locate_sd_server(target)
        if sd_server is not None and sys.platform != "win32":
            _make_executable(sd_server)
        if sd_server is not None:
            print(f"installed sd-server -> {sd_server}", flush = True)
    except SupersededBinaryError:
        raise
    except Exception as exc:  # noqa: BLE001 -- past the sweep, every failure is a mixed tree
        if not replacing:
            raise
        raise SupersededBinaryError(
            f"the managed tree was left part way through a replacement: {exc}"
        ) from exc
    # Written only now, on a complete install: a record naming an accelerator whose binaries never
    # finished extracting would suppress the very reinstall that repairs it. Only for a directory we
    # own -- an unowned one is the user's build, which we never claim to have installed.
    if _may_own:
        _write_install_record(
            target,
            accelerator = accelerator,
            repo = used_repo,
            tag = release.get("tag_name"),
            # Read off the archive's MEMBER LIST, so "this bundle is serverless" is recorded fact
            # rather than something a later load has to infer from an sd-server not being there.
            # A leftover server from an earlier bundle is indistinguishable on disk, which is the
            # whole confusion the record exists to settle.
            ships_server = any(p.name == _binary_names()[1] for p in supplied),
        )
    # The ownership marker was written before extraction, so a crashed partial install is still recognised as ours.
    return sd_cli


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description = "Install a prebuilt sd-cli (stable-diffusion.cpp).")
    p.add_argument(
        "--accelerator", default = "auto", choices = ["auto", "cpu", "vulkan", "rocm", "cuda"]
    )
    p.add_argument("--install-dir", default = None)
    p.add_argument(
        "--print-asset", action = "store_true", help = "resolve + print the asset, don't download"
    )
    args = p.parse_args(argv)

    if args.print_asset:
        # Same primary/fallback resolution as install(), so a host the mirror skips reports the upstream asset, not a false miss.
        _used, _release, chosen = _resolve_with_fallback(args.accelerator, None)
        print(chosen or "(no matching prebuilt; build from source)")
        return 0 if chosen else 2

    try:
        install(
            install_dir = Path(args.install_dir).expanduser() if args.install_dir else None,
            accelerator = args.accelerator,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file = sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence

# Default source: the Unsloth-built mirror, whose CPU/Apple prebuilts are published by
# unslothai/stable-diffusion.cpp (like unslothai/llama.cpp ships its prebuilts). Override with
# UNSLOTH_SD_CPP_REPO to point elsewhere (e.g. back to leejet). GPU hosts never reach here (they
# run diffusers), so only CPU/Apple assets are needed.
DEFAULT_REPO = "unslothai/stable-diffusion.cpp"
# Upstream we fall back to if the mirror can't serve this host (mirror release missing, or
# a host we don't yet build): resolve against leejet so native install still works.
UPSTREAM_FALLBACK_REPO = "leejet/stable-diffusion.cpp"
# Pinned release tag for REPRODUCIBILITY: "releases/latest" silently swaps the binary under
# users on every push. Override with UNSLOTH_SD_CPP_TAG; set it empty to track latest. If the
# pinned tag is gone, install falls back to that repo's latest.
DEFAULT_TAG = "master-741-484baa4"

# Back-compat alias (some callers/tests import REPO).
REPO = DEFAULT_REPO


def _repo() -> str:
    return (os.environ.get("UNSLOTH_SD_CPP_REPO") or DEFAULT_REPO).strip() or DEFAULT_REPO


def _pinned_tag() -> Optional[str]:
    """The release tag to install: env override, else the pinned default; '' = latest."""
    val = os.environ.get("UNSLOTH_SD_CPP_TAG", DEFAULT_TAG).strip()
    return val or None


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
        pool = [a for a in zips if "bin-win" in a.lower()]
        token = _WINDOWS_ACCEL_TOKEN.get(accel, accel)
        sel = [a for a in pool if token in a.lower()]
        if sel:
            return sel[0]
        # An EXPLICIT GPU accelerator with no matching asset is a real miss, not a CPU request:
        # return None so the caller can fall back to a repo that builds it, rather than installing
        # a CPU build for a --accelerator cuda request.
        if accel in ("cuda", "vulkan", "rocm"):
            return None
        # auto / cpu -> a plain avx2 CPU build, else any windows build.
        cpu = [a for a in pool if "avx2" in a.lower()]
        return cpu[0] if cpu else (pool[0] if pool else None)

    # linux (and anything else unix-like)
    pool = [a for a in zips if "linux" in a.lower() and any(t in a.lower() for t in arch)]
    if accel in ("cuda", "vulkan", "rocm"):
        # Explicit GPU accelerator: require its marker, else no match (let the caller
        # fall back) -- never hand back a plain CPU build for a GPU request.
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
    """``~/.unsloth/stable-diffusion.cpp`` (or under ``UNSLOTH_STUDIO_HOME`` /
    ``STUDIO_HOME`` if set), the sibling of the llama.cpp install the finder
    probes."""
    home = os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME")
    base = Path(home).parent if home else Path.home() / ".unsloth"
    return base / "stable-diffusion.cpp"


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _locate_sd_cli(root: Path) -> Optional[Path]:
    name = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


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


def _safe_extractall(zf: zipfile.ZipFile, target: Path) -> None:
    """``extractall`` with a per-member containment check, so an archive carrying an
    absolute path or a ``..`` entry can't write outside ``target`` (Zip-Slip)."""
    base = target.resolve()
    for member in zf.infolist():
        dest = (base / member.filename).resolve()
        if dest != base and base not in dest.parents:
            raise RuntimeError(f"unsafe path in archive: {member.filename!r}")
    zf.extractall(target)


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
        # Verify integrity BEFORE extracting, like the main sd-cli archive: these DLLs load into
        # sd-cli.exe at runtime, so a corrupt/tampered runtime archive must be rejected rather than
        # extracted next to the binary.
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
    # Fall back to upstream ONLY when the user didn't pin a repo (env unset) and the built-in
    # default is in use: an explicit UNSLOTH_SD_CPP_REPO (even one equal to the default) gets
    # exactly that repo, with no surprise upstream substitution.
    repo_pinned = bool((os.environ.get("UNSLOTH_SD_CPP_REPO") or "").strip())
    allow_upstream = (
        not repo_pinned and primary == DEFAULT_REPO and DEFAULT_REPO != UPSTREAM_FALLBACK_REPO
    )

    # (repo, tag_to_fetch, allow_latest). With a pin set, try the exact pin on every repo
    # first (allow_latest = False), then each repo's latest (tag = None).
    attempts: list[tuple[str, Optional[str], bool]] = []
    if tag:
        attempts.append((primary, tag, False))
        if allow_upstream:
            attempts.append((UPSTREAM_FALLBACK_REPO, tag, False))
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
                # Diagnostic goes to stderr, not stdout: --print-asset documents its stdout as the
                # asset name only, so a caller parsing it as a single line must not see this log.
                print(
                    f"falling back to {repo} for {platform.system()}/{platform.machine()}",
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
    # Only claim ownership of ``target`` (marking it for the uninstaller's recursive delete) when
    # this install created it or it was empty, or it already carries our marker -- never adopt a
    # pre-existing non-empty dir, else a later uninstall could wipe a user's own checkout.
    marker = target / ".unsloth-studio-owned"
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
    # Refuse to extract into a pre-existing, non-empty directory we do not own: merging the release
    # in would overwrite or mix our binaries into the user's own files. Fail so they point us at a
    # fresh/empty location.
    if not _may_own:
        raise RuntimeError(
            f"sd.cpp install target already exists and is not a Studio-managed directory: {target}. "
            f"Refusing to extract prebuilt binaries into it to avoid overwriting or mixing them "
            f"into your files. Remove or move that directory, or install into a different, empty "
            f"location (pass a different --install-dir / set the Studio sd.cpp install dir)."
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
    # Claim ownership BEFORE any partial write (download/extract/cudart fetch below). An
    # interrupted extraction (disk full, killed process, a raising _maybe_fetch_windows_cudart)
    # leaves the target non-empty; without the marker the next lazy install would see it as
    # non-empty-and-unowned and trip the refusal guard above, wedging native install until the
    # user manually deletes the directory. Writing the marker first makes the retry treat this
    # partial install as reclaimable (_may_own) and re-extract over it. Written only when
    # _may_own (empty/new or already ours), so it never adopts a user's pre-existing files.
    if _may_own:
        try:
            marker.touch()
        except OSError:
            pass
    archive = target / chosen
    print(f"downloading {chosen} -> {archive}", flush = True)
    try:
        _download(url, archive)
        # Verify integrity BEFORE extracting + executing.
        _verify_sha256(archive, asset.get("digest"))
        print("extracting ...", flush = True)
        with zipfile.ZipFile(archive) as zf:
            _safe_extractall(zf, target)
        # Windows CUDA builds need the separately-published cudart runtime DLLs.
        _maybe_fetch_windows_cudart(release, chosen, target)
    finally:
        # Always drop the archive: on a sha256 mismatch / corrupt zip / network error it
        # must not linger (and a stale partial would defeat a later retry).
        archive.unlink(missing_ok = True)
    sd_cli = _locate_sd_cli(target)
    if not sd_cli:
        raise RuntimeError(f"archive {chosen} contained no sd-cli binary")
    if sys.platform != "win32":
        _make_executable(sd_cli)
    print(f"installed sd-cli -> {sd_cli}", flush = True)
    # The same archive ships the persistent sd-server; make it runnable too so the
    # native backend can prefer it (load once, serve many) over one-shot sd-cli.
    sd_server = _locate_sd_server(target)
    if sd_server is not None and sys.platform != "win32":
        _make_executable(sd_server)
    if sd_server is not None:
        print(f"installed sd-server -> {sd_server}", flush = True)
    # The ownership marker (the same one setup.sh/_is_studio_root use, so the uninstaller deletes
    # only Studio-installed sd.cpp) was already written above, before extraction, so a crashed
    # partial install is still recognised as ours on a retry.
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
        # Route through the same primary/fallback resolution as install(), so a host the mirror
        # doesn't build (e.g. a Linux Vulkan request) reports the upstream asset it would actually
        # download instead of a false "no matching prebuilt".
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

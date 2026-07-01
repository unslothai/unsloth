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

# Default source: the Unsloth-built mirror, whose CPU/Apple prebuilts are compiled and
# published by unslothai/stable-diffusion.cpp (the same way unslothai/llama.cpp ships its
# prebuilts). Override with UNSLOTH_SD_CPP_REPO to point elsewhere (e.g. back to leejet).
# GPU hosts never reach here -- they run diffusers -- so only CPU/Apple assets are needed.
DEFAULT_REPO = "unslothai/stable-diffusion.cpp"
# Upstream we fall back to if the mirror can't serve this host (mirror release missing, or
# a host we don't yet build): resolve against leejet so native install still works.
UPSTREAM_FALLBACK_REPO = "leejet/stable-diffusion.cpp"
# Pinned release tag for REPRODUCIBILITY: "releases/latest" silently swaps the binary
# under users on every push. Override with UNSLOTH_SD_CPP_TAG; set it empty to track
# latest. If the pinned tag is gone, install falls back to that repo's latest.
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
        if not sel:  # fall back to a plain avx2 CPU build
            sel = [a for a in pool if "avx2" in a.lower()]
        return sel[0] if sel else (pool[0] if pool else None)

    # linux (and anything else unix-like)
    pool = [a for a in zips if "linux" in a.lower() and any(t in a.lower() for t in arch)]
    if accel in _LINUX_ACCEL_TOKEN:
        sel = [a for a in pool if _LINUX_ACCEL_TOKEN[accel] in a.lower()]
    else:  # auto / cpu -> the plain build with no accelerator marker
        sel = [a for a in pool if not any(m in a.lower() for m in _LINUX_ACCEL_MARKERS)]
    return sel[0] if sel else None


def _fetch_release(
    tag: Optional[str] = None,
    *,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """GET a release JSON from GitHub. With ``tag`` set, fetch that exact release (and fall
    back to latest if the tag is gone upstream); otherwise fetch latest. ``token`` is
    optional and lifts the API rate limit."""
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
        except urllib.error.HTTPError as exc:  # pinned tag removed upstream -> latest
            if exc.code != 404:
                raise
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
        with zipfile.ZipFile(dest) as zf:
            _safe_extractall(zf, target)
    finally:
        dest.unlink(missing_ok = True)


def _resolve_repo_asset(
    repo: str, tag: Optional[str], accelerator: str, token: Optional[str]
) -> tuple[Optional[dict], Optional[str]]:
    """Fetch ``repo``'s release and pick the asset for this host. Returns
    ``(release, asset_name)`` or ``(None, None)`` when the repo has no usable
    release (fetch failed) or no asset for this host, so the caller can fall back."""
    try:
        release = _fetch_release(tag, repo = repo, token = token)
    except Exception as exc:  # noqa: BLE001 - network / 404-no-latest / rate limit -> fall back
        print(f"sd-cli: {repo} release fetch failed ({exc}); ", end = "", flush = True)
        return None, None
    names = [a["name"] for a in release.get("assets", [])]
    chosen = resolve_release_asset(
        names,
        system = platform.system(),
        machine = platform.machine(),
        accelerator = accelerator,
    )
    return release, chosen


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
    tag = _pinned_tag()
    primary = _repo()
    release, chosen = _resolve_repo_asset(primary, tag, accelerator, token)
    used_repo = primary

    # Fall back to upstream ONLY when the built-in default repo is in use (a user who set
    # UNSLOTH_SD_CPP_REPO gets exactly that repo, no surprise substitution). Covers a
    # missing mirror release or a host the mirror doesn't build.
    if (
        (release is None or not chosen)
        and primary == DEFAULT_REPO
        and DEFAULT_REPO != UPSTREAM_FALLBACK_REPO
    ):
        print(
            f"falling back to {UPSTREAM_FALLBACK_REPO} for "
            f"{platform.system()}/{platform.machine()}",
            flush = True,
        )
        release, chosen = _resolve_repo_asset(UPSTREAM_FALLBACK_REPO, tag, accelerator, token)
        used_repo = UPSTREAM_FALLBACK_REPO

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
        release = _fetch_release(_pinned_tag())
        names = [a["name"] for a in release.get("assets", [])]
        chosen = resolve_release_asset(
            names,
            system = platform.system(),
            machine = platform.machine(),
            accelerator = args.accelerator,
        )
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

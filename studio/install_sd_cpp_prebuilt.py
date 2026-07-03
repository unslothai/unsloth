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
import json
import os
import platform
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Sequence

REPO = "leejet/stable-diffusion.cpp"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"

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


def _fetch_latest_release(*, token: Optional[str] = None, timeout: float = 30.0) -> dict:
    """GET the latest-release JSON from GitHub (token optional, lifts rate limit)."""
    req = urllib.request.Request(RELEASES_API, headers = {"Accept": "application/vnd.github+json"})
    token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout = timeout) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read().decode("utf-8"))


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
    timeout and can hang forever on a stalled socket."""
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


def install(
    *,
    install_dir: Optional[Path] = None,
    accelerator: str = "auto",
    token: Optional[str] = None,
) -> Path:
    """Download + extract the prebuilt for this host. Returns the sd-cli path.

    Raises ``RuntimeError`` if no asset matches the host (the caller should then
    build from source) or the archive has no ``sd-cli``.
    """
    target = install_dir or default_install_dir()
    release = _fetch_latest_release(token = token)
    names = [a["name"] for a in release.get("assets", [])]
    chosen = resolve_release_asset(
        names,
        system = platform.system(),
        machine = platform.machine(),
        accelerator = accelerator,
    )
    if not chosen:
        raise RuntimeError(
            f"No prebuilt sd-cli for {platform.system()}/{platform.machine()} "
            f"(accelerator={accelerator}). Build from source: "
            f"https://github.com/{REPO}"
        )
    url = next(a["browser_download_url"] for a in release["assets"] if a["name"] == chosen)
    target.mkdir(parents = True, exist_ok = True)
    archive = target / chosen
    print(f"downloading {chosen} -> {archive}", flush = True)
    _download(url, archive)
    print("extracting ...", flush = True)
    with zipfile.ZipFile(archive) as zf:
        _safe_extractall(zf, target)
    archive.unlink(missing_ok = True)
    # Windows CUDA builds need the separately-published cudart runtime DLLs.
    _maybe_fetch_windows_cudart(release, chosen, target)
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
        release = _fetch_latest_release()
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

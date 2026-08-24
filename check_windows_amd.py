# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""TEMPORARY: manual validation of PR #9270 on a Windows AMD host. Delete before merge.

Every automated run of this PR so far has been on Linux. Two branches the PR adds are
therefore proven only against a monkeypatched ``symlink_to``:

  * Windows outside developer mode, where creating a link is refused and the installer
    must flatten and finish, exactly as it did before this PR.
  * A filesystem that refuses links (exFAT, SMB without unix extensions), where it must
    stop with a clear error rather than write a tree whose libraries cannot load.

Worth knowing before you start: no published Windows asset ships symlink members, so a
real Windows install never reaches the new code at all. That is checked here (section 4)
and is why the fallback has to be exercised with a synthetic archive instead.

Run:
    python check_windows_amd.py
    python check_windows_amd.py --install-dir E:\\sdcpp     # an exFAT stick or SMB share
    python check_windows_amd.py --skip-network             # no asset check

Prints a markdown report; paste the whole thing into the PR.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "studio"))

REAL = b"\x7fELF" + b"real-library-payload" * 32
ASSETS = {
    "win-rocm": "sd-master-bfbef5b-bin-win-rocm-7.14.0-x64.zip",
    "win-cpu": "sd-master-bfbef5b-bin-win-cpu-x64.zip",
}
BASE_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-813-bfbef5b"

RESULTS: list[tuple[str, str, str]] = []


def record(
    name: str,
    outcome: str,
    detail: str = "",
) -> None:
    RESULTS.append((name, outcome, detail))
    print(f"  [{outcome}] {name}" + (f" - {detail}" if detail else ""), flush = True)


def link_member(zf: zipfile.ZipFile, name: str, target: str) -> None:
    """The symlink member CPython's zipfile writes: Unix link mode, target as the data."""
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    zf.writestr(info, target)


# --------------------------------------------------------------------------- 1. host
def volume_filesystem(path: Path) -> str:
    if sys.platform != "win32":
        return "n/a (not Windows)"
    try:
        drive = os.path.splitdrive(str(path.resolve()))[0] + "\\"
        buf = ctypes.create_unicode_buffer(256)
        name = ctypes.create_unicode_buffer(256)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(drive), name, 256, None, None, None, buf, 256
        )
        return f"{buf.value} (drive {drive})" if ok else f"unknown (drive {drive})"
    except Exception as exc:  # noqa: BLE001 -- a probe, never fatal
        return f"unknown: {type(exc).__name__}"


def gpus() -> str:
    if sys.platform != "win32":
        return "n/a (not Windows)"
    for cmd in (
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController).Name -join '; '",
        ],
        ["wmic", "path", "win32_VideoController", "get", "name"],
    ):
        try:
            out = subprocess.run(cmd, capture_output = True, text = True, timeout = 60)
            got = " ".join(
                l.strip()
                for l in out.stdout.splitlines()
                if l.strip() and l.strip().lower() != "name"
            )
            if got:
                return got
        except Exception:  # noqa: BLE001 -- try the next probe
            continue
    return "unknown"


def can_symlink(root: Path) -> tuple[bool, str]:
    probe = root / ".symlink-capability-probe"
    try:
        probe.symlink_to(".")
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    try:
        probe.unlink()
    except OSError:
        pass
    return True, "symlink created and removed"


# ------------------------------------------------------------------- 2/3. the branches
def scenario_privileged(mod, root: Path, allowed: bool) -> None:
    """With link privilege the archive's links must be restored and readable."""
    if not allowed:
        record(
            "2. links restored when privilege is available",
            "SKIP",
            "this host cannot create symlinks, so section 3 is the live path",
        )
        return
    work = root / "restore"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents = True)
    archive = root / "restore.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("build/bin/libwebp.so.7.2.0", REAL)
        link_member(zf, "build/bin/libwebp.so.7", "libwebp.so.7.2.0")
        link_member(zf, "build/bin/libwebp.so", "libwebp.so.7")
    try:
        with zipfile.ZipFile(archive) as zf:
            mod._safe_extractall(zf, work)
    except Exception as exc:  # noqa: BLE001 -- the outcome under test
        record(
            "2. links restored when privilege is available",
            "FAIL",
            f"raised {type(exc).__name__}: {exc}",
        )
        return
    chained = work / "build" / "bin" / "libwebp.so"
    ok = chained.is_symlink() and chained.read_bytes() == REAL
    record(
        "2. links restored when privilege is available",
        "PASS" if ok else "FAIL",
        f"chained link is_symlink={chained.is_symlink()}, "
        f"reads back the real payload={chained.exists() and chained.read_bytes() == REAL}",
    )


def scenario_no_privilege(mod, root: Path) -> None:
    """A refused link means different, deliberate things per platform.

    On Windows it must flatten and finish, because every Windows asset ships plain files
    and an install that used to complete must keep completing. Anywhere else a refusal
    means the filesystem cannot hold the layout sd-cli needs, so it must stop with a clear
    error rather than write the link text back as a file, which is the "file too short"
    install of #9268. Both outcomes are checked, whichever this host is.
    """
    work = root / "fallback"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents = True)
    archive = root / "fallback.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("build/bin/sd-cli.exe", REAL)
        zf.writestr("build/bin/libwebp.so.7.2.0", REAL)
        link_member(zf, "build/bin/libwebp.so.7", "libwebp.so.7.2.0")

    real_symlink = Path.symlink_to

    def denied(self, *a, **k):
        raise OSError(1314, "A required privilege is not held by the client")

    Path.symlink_to = denied
    try:
        with zipfile.ZipFile(archive) as zf:
            mod._safe_extractall(zf, work)
        raised = None
    except Exception as exc:  # noqa: BLE001 -- the outcome under test
        raised = f"{type(exc).__name__}: {exc}"
    finally:
        Path.symlink_to = real_symlink

    cli = work / "build" / "bin" / "sd-cli.exe"
    flat = work / "build" / "bin" / "libwebp.so.7"
    if sys.platform == "win32":
        ok = raised is None and cli.is_file() and cli.read_bytes() == REAL and flat.is_file()
        record(
            "3. Windows flattens and finishes when links are refused",
            "PASS" if ok else "FAIL",
            f"raised={raised}, sd-cli.exe present={cli.is_file()}, "
            f"link member flattened to a regular file={flat.is_file()}",
        )
    else:
        ok = raised is not None and "cannot store symlinks" in raised
        record(
            "3. non-Windows refuses with a clear error when links are refused",
            "PASS" if ok else "FAIL",
            f"raised={raised}; nothing partially written={not cli.is_file()}",
        )


# ------------------------------------------------------------------ 4. published assets
class _RemoteFile:
    """Seekable file over HTTP range requests: enough for zipfile's central directory."""

    def __init__(self, url: str):
        self.url = url
        _, headers = self._get(0, 0)
        rng = headers.get("Content-Range")
        self.size = int(rng.split("/")[-1]) if rng else int(headers["Content-Length"])
        self.pos = 0

    def _get(self, start, end):
        req = urllib.request.Request(
            self.url,
            headers = {"User-Agent": "unsloth-sd-cpp-installer", "Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(req, timeout = 120) as r:
            return r.read(), r.headers

    def seek(
        self,
        off,
        whence = 0,
    ):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        pass

    def read(self, n = -1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        data, _ = self._get(self.pos, min(self.pos + n, self.size) - 1)
        self.pos += len(data)
        return data


def scenario_real_assets() -> None:
    """No published Windows asset ships links, so the new code is unreachable from one."""
    for label, name in ASSETS.items():
        try:
            zf = zipfile.ZipFile(_RemoteFile(f"{BASE_URL}/{name}"))
            infos = zf.infolist()
        except Exception as exc:  # noqa: BLE001 -- network is optional here
            record(
                f"4. {label} asset carries no symlink members",
                "SKIP",
                f"could not read the central directory: {type(exc).__name__}: {exc}",
            )
            continue
        links = [
            i for i in infos if i.create_system in (3, 19) and stat.S_ISLNK(i.external_attr >> 16)
        ]
        record(
            f"4. {label} asset carries no symlink members",
            "PASS" if not links else "UNEXPECTED",
            f"{len(infos)} members, {len(links)} symlink members",
        )


# ----------------------------------------------------------------------- 5. the suite
def scenario_suite() -> None:
    backend = HERE / "studio" / "backend"
    if not (backend / "tests" / "test_sd_cpp_install.py").is_file():
        record("5. extraction test suite", "SKIP", "test file not found; run from the repo root")
        return
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_sd_cpp_install.py",
        "-k",
        "safe_extractall",
        "-q",
        "--noconftest",
        "-p",
        "no:cacheprovider",
    ]
    try:
        p = subprocess.run(cmd, cwd = backend, capture_output = True, text = True, timeout = 900)
    except Exception as exc:  # noqa: BLE001
        record("5. extraction test suite", "SKIP", f"{type(exc).__name__}: {exc}")
        return
    tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()]
    record(
        "5. extraction test suite",
        "PASS" if p.returncode == 0 else "FAIL",
        tail[-1][:160] if tail else f"rc={p.returncode}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--install-dir",
        default = None,
        help = "where to run the file tests; point at an exFAT or SMB path to cover those",
    )
    ap.add_argument("--skip-network", action = "store_true")
    args = ap.parse_args()

    root = Path(args.install_dir) if args.install_dir else Path(tempfile.mkdtemp(prefix = "pr9270_"))
    root.mkdir(parents = True, exist_ok = True)

    print("PR 9270 Windows AMD check\n")
    allowed, why = can_symlink(root)

    try:
        import install_sd_cpp_prebuilt as mod
    except Exception:  # noqa: BLE001 -- report it rather than dying
        print("could not import studio/install_sd_cpp_prebuilt.py; run from the repo root")
        traceback.print_exc()
        return 2

    scenario_privileged(mod, root, allowed)
    scenario_no_privilege(mod, root)
    if not args.skip_network:
        scenario_real_assets()
    scenario_suite()

    print("\n\n===== paste everything below into the PR =====\n")
    print("### Host\n")
    print(f"- OS: `{platform.platform()}`")
    print(f"- Python: `{sys.version.split()[0]}`, machine `{platform.machine()}`")
    print(f"- GPU: `{gpus()}`")
    print(f"- Install dir: `{root}`, filesystem `{volume_filesystem(root)}`")
    print(f"- Can create symlinks: **{allowed}** ({why})")
    print("\n### Results\n")
    print("| check | outcome | detail |")
    print("|---|---|---|")
    for name, outcome, detail in RESULTS:
        print(f"| {name} | **{outcome}** | {detail} |")
    bad = [r for r in RESULTS if r[1] in ("FAIL", "UNEXPECTED")]
    print(f"\n{'All checks passed.' if not bad else f'{len(bad)} check(s) need attention.'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

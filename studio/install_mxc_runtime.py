# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install the pinned MXC sandbox executor for Windows tool isolation.

Run at Studio setup time, never during a tool call. A tool call that could
download its own sandbox executor would be choosing its own boundary.

MXC publishes signed native binaries two ways: a 358MB `mxc-release-binaries.zip`
on the GitHub release, and the same per-arch binaries inside the npm tarball at
`package/bin/<arch>/`. This uses the npm tarball, which is 26MB, because only
`wxc-exec.exe` and its helpers are wanted and the registry serves the tarball
over plain HTTPS with an integrity digest. Node is NOT required: the tarball is
a gzipped tar and is unpacked with the standard library.

Usage:
    python studio/install_mxc_runtime.py [--dest DIR] [--verify-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


# The pinned identity lives in one dependency-free module the backend reads
# too, so the installer and the launch-time trust check cannot disagree about
# what the binary is. Imported by path rather than as a package, because setup
# runs before the backend's own imports are necessarily satisfiable, and a
# missing pin must REFUSE rather than install something unverified.
def _load_pins():
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "backend",
        "core",
        "inference",
        "mxc_pins.py",
    )
    spec = importlib.util.spec_from_file_location("mxc_pins", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot read the MXC pins at {path}; refusing to install unverified")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pins = _load_pins()

MXC_VERSION = pins.MXC_VERSION
TARBALL_SHA256 = pins.TARBALL_SHA256
EXECUTOR_SHA256 = pins.EXECUTOR_SHA256
HOST_PREP_SHA256 = pins.HOST_PREP_SHA256
REGISTRY_URL = f"https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{MXC_VERSION}.tgz"

# What the Windows backend actually launches, plus the helpers MXC's own
# ProcessContainer path expects to find beside it.
REQUIRED = ("wxc-exec.exe",)
OPTIONAL = (
    "wxc-host-prep.exe",
    "winhttp-proxy-shim.exe",
    "plm.exe",
    "mxc-diagnostic-console.exe",
)


def arch_dir() -> str:
    machine = (os.environ.get("PROCESSOR_ARCHITECTURE") or "").lower()
    if "arm64" in machine or "arm64" in (sys.platform or ""):
        return "arm64"
    return "x64"


def default_dest() -> str:
    override = os.environ.get("UNSLOTH_MXC_DIR")
    if override:
        return override
    # The same resolution the backend uses, so the installer and
    # sandbox_windows.managed_mxc_dir never disagree about where the executor
    # lives. Degraded rather than raising: setup can run before the backend
    # is importable at all.
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
        from core.inference.sandbox_windows import managed_mxc_dir
        return managed_mxc_dir()
    except Exception:  # noqa: BLE001 - setup can run before the backend imports
        # The same custom-home fallback managed_mxc_dir() uses. Without it a
        # degraded install on a custom-home Studio writes to ~/.unsloth/mxc and
        # reports success, while the backend later looks in <custom-home>/mxc
        # and reports the executor as not installed.
        override = (
            os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
        ).strip()
        if override:
            return os.path.join(os.path.expanduser(override), "mxc")
        return os.path.join(os.path.expanduser("~"), ".unsloth", "mxc")


def download(url: str, into: str) -> str:
    target = os.path.join(into, "mxc-sdk.tgz")
    with urllib.request.urlopen(url, timeout = 300) as response:  # noqa: S310 - pinned https
        with open(target, "wb") as handle:
            shutil.copyfileobj(response, handle)
    return target


def digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def extract(tarball: str, arch: str, dest: str) -> list[str]:
    """Pull just the binaries for this architecture out of the tarball.

    Members are matched by exact name and written by basename, so a crafted
    tarball cannot traverse out of the destination.
    """
    wanted = {f"package/bin/{arch}/{name}": name for name in (*REQUIRED, *OPTIONAL)}
    written: list[str] = []
    os.makedirs(dest, exist_ok = True)
    with tarfile.open(tarball, "r:gz") as archive:
        for member in archive.getmembers():
            name = wanted.get(member.name)
            if name is None or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            out = os.path.join(dest, name)
            with source, open(out, "wb") as handle:
                shutil.copyfileobj(source, handle)
            written.append(name)
    return written


def verify(path: str, expected: str, what: str) -> str:
    """Refuse anything whose digest is not the pinned one."""
    actual = digest(path)
    if actual != expected:
        raise SystemExit(
            f"{what} does not match the pinned SHA-256 for MXC {MXC_VERSION}.\n"
            f"  expected {expected}\n"
            f"  got      {actual}\n"
            "Windows tool isolation was NOT installed. This is what a tampered "
            "mirror or a republished artifact looks like; if the version was "
            "bumped deliberately, record the new digests in install_mxc_runtime.py."
        )
    return actual


def install(dest: str) -> dict:
    arch = arch_dir()
    expected_executor = EXECUTOR_SHA256.get(arch)
    if expected_executor is None:
        raise SystemExit(f"no pinned MXC executor digest for {arch}")
    with tempfile.TemporaryDirectory() as scratch:
        tarball = download(REGISTRY_URL, scratch)
        # Before extraction, so nothing from an unrecognised tarball is ever
        # written into the destination.
        tarball_sha = verify(tarball, TARBALL_SHA256, "the MXC tarball")
        written = extract(tarball, arch, dest)

    missing = [name for name in REQUIRED if name not in written]
    if missing:
        raise SystemExit(
            f"the MXC tarball did not contain {', '.join(missing)} for {arch}. "
            "Windows tool isolation cannot be installed."
        )

    executor = os.path.join(dest, "wxc-exec.exe")
    # Checked again after extraction: the tarball digest covers what was
    # downloaded, this covers what is now on disk and about to be run.
    executor_sha = verify(executor, expected_executor, f"the installed {arch} wxc-exec.exe")
    return {
        "version": MXC_VERSION,
        "arch": arch,
        "dest": dest,
        "executor": executor,
        "executor_sha256": executor_sha,
        "tarball_sha256": tarball_sha,
        "installed": sorted(written),
    }


def prepare_host(dest: str, timeout: float = 120.0) -> dict:
    """Run MXC's one-time elevated host preparation.

    Tier 3 (AppContainer plus DACL) is the tier every shipping Windows build
    lands on, and it needs minimum-rights ACEs on the system drive root and
    MXC's managed security descriptor on \\Device\\Null before a sandboxed
    workload runs reliably. `wxc-host-prep.exe` has requireAdministrator in its
    manifest, so this prompts for UAC and exits 65 if it is refused.

    Deliberately separate from install(): fetching the binary needs no
    privilege, and Studio should ask before changing host security state rather
    than doing it silently as part of a download.
    """
    prep = os.path.join(dest, "wxc-host-prep.exe")
    if not os.path.isfile(prep):
        raise SystemExit(f"{prep} is missing; run this without --prepare-host first")
    # This one runs ELEVATED, and the managed directory is user-writable, so
    # the download-time check does not cover it: any same-user process that
    # replaced the helper between installation and this step would get
    # administrator execution the moment the user approves the UAC prompt.
    if not pins.matches_pin(prep, HOST_PREP_SHA256):
        raise SystemExit(
            f"{prep} is not the pinned MXC {MXC_VERSION} host-preparation helper. "
            "Nothing was run. Re-run this script without --prepare-host to reinstall it."
        )
    results = {}
    for subcommand in ("prepare-system-drive", "prepare-null-device"):
        # Re-checked before EVERY invocation, not once before the loop. The
        # first call can take up to the full timeout, and the pathname stays
        # user-writable throughout, so a same-user process could wait for it to
        # finish and swap the helper before the second elevated run.
        if not pins.matches_pin(prep, HOST_PREP_SHA256):
            raise SystemExit(
                f"{prep} changed and is no longer the pinned MXC {MXC_VERSION} "
                f"host-preparation helper. {subcommand} was NOT run."
            )
        try:
            completed = subprocess.run(
                [prep, subcommand],
                capture_output = True,
                timeout = timeout,
            )
        except subprocess.TimeoutExpired:
            # requireAdministrator in the manifest means the loader raises a UAC
            # consent dialog at process start. With no interactive desktop to
            # answer it the binary simply never returns, which is what happens
            # on a GitHub Windows runner: prepare-system-drive hung for the full
            # 300s. Report it as a state rather than hanging the caller, because
            # Studio setup has to be able to say "this needs your approval"
            # instead of appearing to freeze.
            results[subcommand] = {
                "exit": None,
                "timed_out_after": timeout,
                "stderr": (
                    "no response, which usually means the UAC consent prompt was "
                    "never answered. Run this from an elevated terminal."
                ),
            }
            continue
        results[subcommand] = {
            "exit": completed.returncode,
            "stderr": completed.stderr.decode(errors = "replace").strip()[:300],
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--dest", default = None, help = "where to install (default: Studio home)")
    parser.add_argument(
        "--prepare-host-timeout",
        type = float,
        default = 120.0,
        help = "seconds to wait for each elevated step before reporting no response",
    )
    parser.add_argument(
        "--prepare-host",
        action = "store_true",
        help = "run MXC's elevated host preparation (prompts for UAC)",
    )
    parser.add_argument(
        "--verify-only", action = "store_true", help = "report an existing install without downloading"
    )
    args = parser.parse_args()

    dest = args.dest or default_dest()
    if args.verify_only:
        executor = os.path.join(dest, "wxc-exec.exe")
        present = os.path.isfile(executor)
        actual = digest(executor) if present else None
        expected = EXECUTOR_SHA256.get(arch_dir())
        print(
            json.dumps(
                {
                    "present": present,
                    "executor": executor,
                    "executor_sha256": actual,
                    "expected_sha256": expected,
                    # Presence alone is not verification: an executor replaced
                    # after installation would still be "present".
                    "matches_pin": present and actual == expected,
                },
                indent = 2,
            )
        )
        return 0 if (present and actual == expected) else 1

    if args.prepare_host:
        results = prepare_host(dest, args.prepare_host_timeout)
        print(json.dumps(results, indent = 2))
        # Non-zero when the Tier 3 prerequisites were NOT applied. A refused
        # UAC prompt, a missing elevation or a failed prepare-null-device
        # otherwise looked like success to setup and to automation.
        if any(result.get("exit") != 0 for result in results.values()):
            return 1
        return 0

    print(json.dumps(install(dest), indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

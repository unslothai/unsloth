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

# Pinned. An upgrade is a deliberate change that re-runs the Windows
# qualification job, not something that drifts in from the registry.
MXC_VERSION = "0.8.0"
REGISTRY_URL = f"https://registry.npmjs.org/@microsoft/mxc-sdk/-/mxc-sdk-{MXC_VERSION}.tgz"

# wxc-exec.exe is the security boundary for every isolated Windows tool call,
# so what gets installed is pinned by digest and a mismatch REFUSES rather than
# warning. Without this a compromised registry response, mirror or republished
# artifact becomes the sandbox and the install still reports success.
#
# Recorded from the artifact the Windows qualification job ran against, and
# cross-checked against the registry's own integrity metadata for 0.8.0
# (sha512-pnf5QsASwp+qtRi5uth2GDjwuyG0rHWRpxCf3RbAjQ4wDTNfBX/9l0A+RVZspU2agpF3/11uWB1JisIS7WrNYg==).
# Bumping MXC_VERSION means recording these again from the new tarball.
TARBALL_SHA256 = "06bb2399d7e98ab1907acf851e12a4e44748dd467b79d3e53c2f2fbf569da14e"
EXECUTOR_SHA256 = {
    "x64": "6049c64723af1173c3739dc6cd6b2f33f6c021bb2832c4216233cba7f71aee9a",
    "arm64": "dde1c592270e9a659b01dccad70362da7b99fec114885fa4d625507aa775a503",
}

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
    results = {}
    for subcommand in ("prepare-system-drive", "prepare-null-device"):
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
        print(json.dumps(prepare_host(dest, args.prepare_host_timeout), indent = 2))
        return 0

    print(json.dumps(install(dest), indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

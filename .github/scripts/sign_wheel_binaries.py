# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticode-sign the native binaries inside a wheel and repack it.

A .whl cannot itself be Authenticode signed: it is a zip, and Windows has no subject
interface package for one. What CAN be signed are the PE images inside it -- the .pyd
extension modules and the .dll files they link -- which is what Windows actually loads and
what SmartScreen and an enterprise WDAC policy judge. So this unpacks the wheel, signs each
PE we built, and packs it back under the same filename.

RECORD is regenerated, because it has to be: signing appends a certificate table, so every
signed member's sha256 and length change, and a wheel whose RECORD disagrees with its
contents is one `pip check`, `uv pip install --strict` or reproducibility audit away from
being called corrupt.

Files that already carry a certificate are left alone. delvewheel vendors Microsoft's own
redistributables (msvcp140.dll and friends) into pyarrow, already signed by Microsoft;
re-signing those would replace their signature with ours and claim authorship of somebody
else's binary.

The signing itself is delegated, never reimplemented: --signer names the command, and CI
passes studio/src-tauri/windows/sign-with-trusted-signing.ps1, the same helper the desktop
release uses, so the endpoint, the identity and the Azure retry policy have one definition.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: IMAGE_FILE_MACHINE_ARM64. The wheels this signs are win_arm64.
MACHINE_ARM64 = 0xAA64
#: Data directory index 4 is the Certificate Table -- a non-zero size means "already signed".
CERTIFICATE_TABLE_INDEX = 4


def _pe_header_offset(blob: bytes) -> "int | None":
    """Offset of the PE signature, or None when this is not a PE image at all."""
    if len(blob) < 0x40 or blob[:2] != b"MZ":
        return None
    try:
        offset = struct.unpack_from("<I", blob, 0x3C)[0]
    except struct.error:
        return None
    if offset <= 0 or offset + 24 > len(blob) or blob[offset : offset + 4] != b"PE\0\0":
        return None
    return offset


def pe_machine(blob: bytes) -> "int | None":
    """The COFF machine type, or None when the bytes are not a PE image."""
    offset = _pe_header_offset(blob)
    if offset is None:
        return None
    return struct.unpack_from("<H", blob, offset + 4)[0]


def is_signed(blob: bytes) -> bool:
    """Does this PE image already carry a certificate table?

    Read from the optional header's data directory rather than by running a tool, so the
    answer does not depend on a Windows host or on the file being unpacked first.
    """
    offset = _pe_header_offset(blob)
    if offset is None:
        return False
    magic_at = offset + 24
    if magic_at + 2 > len(blob):
        return False
    magic = struct.unpack_from("<H", blob, magic_at)[0]
    # PE32+ has 8 more bytes of fixed fields before the data directory than PE32 does.
    directory_at = magic_at + (112 if magic == 0x20B else 96) + CERTIFICATE_TABLE_INDEX * 8
    if directory_at + 8 > len(blob):
        return False
    _, size = struct.unpack_from("<II", blob, directory_at)
    return size > 0


def record_line(name: str, blob: bytes) -> str:
    """One RECORD row: urlsafe base64 sha256 with the padding stripped, per PEP 376."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(blob)}"


def sign_file(path: Path, signer: "list[str]") -> None:
    """Run the signing command over one file, failing the build if it cannot sign."""
    result = subprocess.run([*signer, str(path)], text=True)
    if result.returncode != 0:
        raise SystemExit(f"signing failed for {path.name} (exit {result.returncode})")


def sign_wheel(
    wheel: Path,
    out_dir: Path,
    signer: "list[str]",
    *,
    require_machine: "int | None" = MACHINE_ARM64,
) -> "tuple[Path, list[str], list[str]]":
    """Sign every unsigned PE inside `wheel` and write the result into `out_dir`.

    Returns (written wheel, names signed, names skipped as already signed).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / wheel.name
    signed: list[str] = []
    skipped: list[str] = []

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            payload = {name: archive.read(name) for name in names}

        for name in names:
            blob = payload[name]
            machine = pe_machine(blob)
            if machine is None:
                continue
            if require_machine is not None and machine != require_machine:
                raise SystemExit(
                    f"{wheel.name}: {name} is machine {machine:#x}, not {require_machine:#x}"
                )
            if is_signed(blob):
                skipped.append(name)
                continue
            # Signed on disk under its own name: signtool and trusted-signing-cli both work
            # in place, and the suffix decides which subject interface package is used.
            staged = work / Path(name).name
            staged.write_bytes(blob)
            sign_file(staged, signer)
            resigned = staged.read_bytes()
            if not is_signed(resigned):
                raise SystemExit(f"{wheel.name}: {name} has no certificate after signing")
            payload[name] = resigned
            signed.append(name)
            staged.unlink()

        if not signed and not skipped:
            raise SystemExit(f"{wheel.name}: no PE images found -- nothing to sign")

        record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
        if record_name is None:
            raise SystemExit(f"{wheel.name}: no .dist-info/RECORD")

        # Member order preserved: the installer does not care, but a diff between the signed
        # and unsigned wheel should show the signatures and nothing else.
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as out:
            lines = []
            for name in names:
                if name == record_name:
                    continue
                out.writestr(name, payload[name])
                lines.append(record_line(name, payload[name]))
            lines.append(f"{record_name},,")
            out.writestr(record_name, "\n".join(lines) + "\n")

    return destination, signed, skipped


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--signer",
        required=True,
        help="signing command; the file to sign is appended as the last argument",
    )
    parser.add_argument(
        "--expect-machine",
        default="arm64",
        choices=("arm64", "any"),
        help="refuse a wheel carrying a PE for another architecture (default: arm64)",
    )
    args = parser.parse_args(argv)

    import shlex

    signer = shlex.split(args.signer, posix=False)
    require = MACHINE_ARM64 if args.expect_machine == "arm64" else None
    failures = 0
    for wheel in args.wheels:
        written, signed, skipped = sign_wheel(wheel, args.out, signer, require_machine=require)
        print(f"{written.name}: signed {len(signed)}, already signed {len(skipped)}")
        for name in signed:
            print(f"  signed  {name}")
        for name in skipped:
            print(f"  kept    {name} (already carries a certificate)")
        if not signed:
            print(f"::warning::{wheel.name} had nothing to sign")
            failures += 1
    return 1 if failures and len(args.wheels) == failures else 0


if __name__ == "__main__":
    sys.exit(main())

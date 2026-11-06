#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check every native extension in a wheel was built for the wheel's platform.

An ARM runner can still run an emulated x64 Python or link with an x64 toolchain, and the wheel
would be named for arm64 either way. This reads the machine field of each .so (ELF e_machine) and
.pyd (PE/COFF Machine). On Linux it also bounds the newest GLIBC symbol version referenced, because
a plain `linux_*` tag makes pip skip the glibc check a manylinux tag would have done.

Usage: prebuilt_wheels_check_binaries.py WHEEL PLATFORM [--max-glibc 2.35]
"""

from __future__ import annotations

import re
import struct
import sys
import zipfile

ELF_MACHINE = {"linux_x86_64": 62, "linux_aarch64": 183}  # EM_X86_64, EM_AARCH64
PE_MACHINE = {"win_amd64": 0x8664, "win_arm64": 0xAA64}
_GLIBC = re.compile(rb"GLIBC_(\d+)\.(\d+)(?:\.\d+)?\x00")


def elf_machine(data: bytes) -> int:
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    endian = "<" if data[5] == 1 else ">"
    return struct.unpack_from(f"{endian}H", data, 18)[0]


def pe_machine(data: bytes) -> int:
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[offset : offset + 4] != b"PE\x00\x00":
        raise ValueError("no PE signature")
    return struct.unpack_from("<H", data, offset + 4)[0]


def newest_glibc(data: bytes) -> tuple[int, int] | None:
    found = [(int(major), int(minor)) for major, minor in _GLIBC.findall(data)]
    return max(found) if found else None


def check(
    wheel: str,
    platform: str,
    max_glibc: tuple[int, int] = (2, 35),
) -> list[str]:
    """Problems found, one line each; empty when the wheel is consistent."""
    problems = []
    checked = 0
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if platform in PE_MACHINE and name.endswith(".pyd"):
                data = zf.read(name)
                got = pe_machine(data)
                if got != PE_MACHINE[platform]:
                    problems.append(f"{name}: PE machine {got:#x}, want {PE_MACHINE[platform]:#x}")
                checked += 1
            elif platform in ELF_MACHINE and re.search(r"\.so(\.|$)", name):
                data = zf.read(name)
                got = elf_machine(data)
                if got != ELF_MACHINE[platform]:
                    problems.append(f"{name}: ELF machine {got}, want {ELF_MACHINE[platform]}")
                glibc = newest_glibc(data)
                print(f"{name}: newest GLIBC {'.'.join(map(str, glibc)) if glibc else 'none'}")
                if glibc and glibc > max_glibc:
                    problems.append(
                        f"{name}: needs GLIBC {glibc[0]}.{glibc[1]}, floor is {max_glibc[0]}.{max_glibc[1]}"
                    )
                checked += 1
    if not checked:
        problems.append(f"no native extension for {platform} in {wheel}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 4) or (len(argv) == 4 and argv[2] != "--max-glibc"):
        print(__doc__, file = sys.stderr)
        return 2
    wheel, platform = argv[:2]
    if platform not in ELF_MACHINE and platform not in PE_MACHINE:
        print(f"unknown platform {platform}", file = sys.stderr)
        return 2
    max_glibc = tuple(int(part) for part in argv[3].split(".")) if len(argv) == 4 else (2, 35)
    problems = check(wheel, platform, max_glibc)
    for problem in problems:
        print(f"::error::{problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

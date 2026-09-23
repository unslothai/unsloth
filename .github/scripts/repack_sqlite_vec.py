# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Repack the PyPI sqlite-vec win_amd64 wheel with an ARM64 vec0.dll under the win_arm64 tag.

The wheel is pure Python plus one DLL, generated upstream by sqlite-dist: replacing the DLL,
the WHEEL tag and RECORD is the whole port. Usage: repack_sqlite_vec.py <amd64 wheel> <vec0.dll> <out dir>
"""

import base64, hashlib, struct, sys, zipfile
from pathlib import Path

src, dll, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert src.name.endswith("-py3-none-win_amd64.whl"), src.name
dst = out / src.name.replace("-win_amd64.whl", "-win_arm64.whl")

data = dll.read_bytes()
e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
assert data[e_lfanew : e_lfanew + 4] == b"PE\0\0", "not a PE file"
machine = struct.unpack_from("<H", data, e_lfanew + 4)[0]
assert machine == 0xAA64, f"vec0.dll machine is {machine:#x}, not ARM64 (0xaa64)"


def record_line(name: str, blob: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(blob)}"


out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    names = zin.namelist()
    record_name = next(n for n in names if n.endswith(".dist-info/RECORD"))
    lines = []
    for name in names:
        if name == record_name:
            continue
        blob = zin.read(name)
        if name.endswith("/vec0.dll"):
            blob = data
        elif name.endswith(".dist-info/WHEEL"):
            text = blob.decode()
            assert "Tag: py3-none-win_amd64" in text, text
            blob = text.replace("Tag: py3-none-win_amd64", "Tag: py3-none-win_arm64").encode()
        zout.writestr(name, blob)
        lines.append(record_line(name, blob))
    lines.append(f"{record_name},,")
    zout.writestr(record_name, "\n".join(lines) + "\n")
print(dst.name, dst.stat().st_size, "bytes")

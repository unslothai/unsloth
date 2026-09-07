# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Explicit offline wheel assembly from all reviewed ABI builds, never tool-time."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from core.inference.windows_sandbox.artifacts import (
    DISTRIBUTION,
    PACKAGE,
    VERSION,
    WHEEL_TAG,
    SHIMS,
    SOURCE_NAMES,
    canonical_json,
    validate_build,
    validate_manifest,
)
from core.inference.windows_sandbox.dependencies import PEFILE_VERSION, read_regular_file
from core.inference.windows_sandbox.profiles import ABI_ADAPTERS, PYTHON_PROFILE


def assemble(hosts, output):
    """Build a standard data-only wheel; no setup/entrypoint/.pth code is shipped."""
    if set(hosts) != {adapter.identity for adapter in ABI_ADAPTERS}:
        raise ValueError("A build for every declared ABI is required.")
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Wheel output must be outside the runtime sources.")
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_NAMES
    }
    contents, builds = {}, {}
    for adapter in ABI_ADAPTERS:
        binary = Path(hosts[adapter.identity])
        identity, data = read_regular_file(binary, limit = 16 * 1024 * 1024)
        _, evidence = read_regular_file(binary.with_suffix(".build.json"), limit = 65536)
        build = validate_build(json.loads(evidence), adapter)
        if build["sources"] != source_hashes or build["binary"] != {
            "sha256": identity.sha256,
            "size": identity.size,
        }:
            raise ValueError("Native helper bytes or sources do not match build evidence.")
        contents[f"bin/python_host-{adapter.identity}.exe"] = data
        builds[adapter.identity] = build
    backend = ROOT.parent / "backend/core/inference"
    for name in SHIMS:
        path = backend / (
            "windows_sandbox/policy.py" if name == "policy.py" else "sandbox_site/sitecustomize.py"
        )
        # Text assets avoid pip generating unlisted bytecode or importable hooks.
        _, contents[f"shims/{name}.txt"] = read_regular_file(path, limit = 1024 * 1024)
    _, contents["LICENSE.AGPL-3.0"] = read_regular_file(
        ROOT.parent / "LICENSE.AGPL-3.0", limit = 1024 * 1024
    )
    manifest = canonical_json(
        {
            "schema": 1,
            "distribution": DISTRIBUTION,
            "version": VERSION,
            "profile": PYTHON_PROFILE.digest,
            "protocol": PYTHON_PROFILE.protocol_version,
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                for name, data in contents.items()
            },
            "builds": builds,
        }
    )
    validate_manifest(manifest)
    contents["manifest.json"] = manifest
    archive = {f"{PACKAGE}/{name}": data for name, data in contents.items()}
    dist_info = f"{PACKAGE}-{VERSION}.dist-info"
    archive[f"{dist_info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: {VERSION}\n"
        "Summary: Native Windows runtime artifacts for Unsloth Studio tool isolation\n"
        "Requires-Python: >=3.11,<3.14\n"
        f"Requires-Dist: pefile=={PEFILE_VERSION}\nLicense: AGPL-3.0-only\n\n"
    ).encode()
    archive[f"{dist_info}/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: unsloth-runtime-build\nRoot-Is-Purelib: false\nTag: {WHEEL_TAG}\n".encode()
    )
    records = io.StringIO(newline = "")
    writer = csv.writer(records, lineterminator = "\n")
    for name, data in sorted(archive.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((f"{dist_info}/RECORD", "", ""))
    archive[f"{dist_info}/RECORD"] = records.getvalue().encode()
    output.mkdir(parents = True, exist_ok = True)
    target = output / f"{PACKAGE}-{VERSION}-{WHEEL_TAG}.whl"
    # Exclusive create: never overwrite an earlier build with different evidence.
    with target.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, data in sorted(archive.items()):
            entry = zipfile.ZipInfo(name, date_time = (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            wheel.writestr(entry, data)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--host", action = "append", required = True, help = "ABI=absolute helper path (one per ABI)"
    )
    parser.add_argument("--output", type = Path, required = True)
    options = parser.parse_args()
    pairs = [item.split("=", 1) for item in options.host]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        parser.error("Host arguments must be unique ABI=path pairs.")
    print(assemble(dict(pairs), options.output))

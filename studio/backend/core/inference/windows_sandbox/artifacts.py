# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Admit data from the exact companion package in Studio's selected environment.

The trusted Studio installation is the supply-chain root. Manifest hashes prove
integrity, not publisher authenticity. No package is imported, downloaded or
compiled here; tool/provider arguments cannot select an artifact source.
"""

from dataclasses import dataclass
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path

from .content import SnapshotFile
from .dependencies import (
    FileIdentity,
    PEFILE_VERSION,
    checked_path,
    inspect_native_image,
    read_regular_file,
)
from .profiles import ABI_ADAPTERS, PYTHON_PROFILE, WindowsRuntimeError

DISTRIBUTION = "unsloth-windows-sandbox-runtime"
PACKAGE = "unsloth_windows_sandbox_runtime"
VERSION = "0.1.0.dev1"
WHEEL_TAG = "py3-none-win_amd64"
MSVC_VERSION = "14.44.35207"
SDK_VERSION = "10.0.22621.0"
SHIMS = ("policy.py", "sitecustomize.py")
SOURCE_NAMES = (
    "src/gate.c",
    "src/gate.h",
    "src/host_config.c",
    "src/host_config.h",
    "src/python_host.c",
    "src/runtime.manifest.xml",
)


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_ARTIFACT_INVALID", message)


def canonical_json(value):
    return json.dumps(value, sort_keys = True, separators = (",", ":")).encode("utf-8")


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_build(value, adapter):
    if type(value) is not dict or set(value) != {
        "schema",
        "abi",
        "headers",
        "header_digest",
        "profile",
        "protocol",
        "compiler",
        "sdk",
        "sources",
        "binary",
    }:
        raise _invalid("Invalid native build provenance schema.")
    if (
        type(value["schema"]) is not int
        or value["schema"] != 1
        or value["abi"] != adapter.identity
        or value["profile"] != PYTHON_PROFILE.digest
        or type(value["protocol"]) is not int
        or value["protocol"] != PYTHON_PROFILE.protocol_version
        or type(value["headers"]) is not list
        or len(value["headers"]) != 3
        or any(type(part) is not int or part < 0 for part in value["headers"])
        or value["headers"][:2] != [adapter.major, adapter.minor]
        or not _hash(value["header_digest"])
        or type(value["compiler"]) is not dict
        or set(value["compiler"]) != {"version", "sha256"}
        or value["compiler"]["version"] != MSVC_VERSION
        or not _hash(value["compiler"]["sha256"])
        or value["sdk"] != SDK_VERSION
        or type(value["sources"]) is not dict
        or set(value["sources"]) != set(SOURCE_NAMES)
        or not all(_hash(digest) for digest in value["sources"].values())
    ):
        raise _invalid("Native build provenance does not match this ABI/profile/toolchain.")
    _file_record(value["binary"])
    return value


def _file_record(value):
    if (
        type(value) is not dict
        or set(value) != {"sha256", "size"}
        or not _hash(value["sha256"])
        or type(value["size"]) is not int
        or not 0 < value["size"] <= 16 * 1024 * 1024
    ):
        raise _invalid("Invalid companion file digest or size.")


def artifact_names():
    return (
        tuple(f"bin/python_host-{adapter.identity}.exe" for adapter in ABI_ADAPTERS)
        + tuple(f"shims/{name}.txt" for name in SHIMS)
        + ("LICENSE.AGPL-3.0",)
    )


def validate_manifest(data):
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise _invalid("Malformed companion manifest.") from error
    if type(value) is not dict or set(value) != {
        "schema",
        "distribution",
        "version",
        "profile",
        "protocol",
        "files",
        "builds",
    }:
        raise _invalid("Unknown companion manifest schema.")
    if (
        type(value["schema"]) is not int
        or value["schema"] != 1
        or value["distribution"] != DISTRIBUTION
        or value["version"] != VERSION
        or value["profile"] != PYTHON_PROFILE.digest
        or type(value["protocol"]) is not int
        or value["protocol"] != PYTHON_PROFILE.protocol_version
        or type(value["files"]) is not dict
        or set(value["files"]) != set(artifact_names())
        or type(value["builds"]) is not dict
        or set(value["builds"]) != {a.identity for a in ABI_ADAPTERS}
        or canonical_json(value) != data
    ):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PROTOCOL_MISMATCH",
            "Companion version, profile, protocol or file inventory is incompatible.",
        )
    for record in value["files"].values():
        _file_record(record)
    for adapter in ABI_ADAPTERS:
        build = validate_build(value["builds"][adapter.identity], adapter)
        if build["binary"] != value["files"][f"bin/python_host-{adapter.identity}.exe"]:
            raise _invalid("The native helper does not match its build record.")
    return value


@dataclass(frozen = True)
class AdmittedArtifacts:
    abi: str
    manifest: FileIdentity
    metadata: tuple[FileIdentity, ...]
    files: tuple[SnapshotFile, ...]

    @property
    def digest(self):
        # Installation path/inode changes must also change preparation identity.
        from dataclasses import asdict
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


def admit_installed_artifacts(prefix, adapter):
    """Backend-owned prefix comes from the running broker, not import search."""
    if adapter not in ABI_ADAPTERS:
        raise _invalid("No companion helper exists for this ABI.")
    site = checked_path(Path(prefix) / "Lib/site-packages")
    root = site / PACKAGE
    metadata_root = site / f"{PACKAGE}-{VERSION}.dist-info"
    if not root.exists() or not metadata_root.exists():
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_MISSING",
            f"Install the matching {DISTRIBUTION}=={VERSION} with Studio; no runtime is downloaded during tool execution.",
        )
    root, metadata_root = checked_path(root), checked_path(metadata_root)
    identity, data = read_regular_file(root / "manifest.json", limit = 65536)
    manifest = validate_manifest(data)
    metadata = []
    for name, expected in (
        (
            "METADATA",
            {
                "Name": DISTRIBUTION,
                "Version": VERSION,
                "Requires-Dist": f"pefile=={PEFILE_VERSION}",
            },
        ),
        ("WHEEL", {"Wheel-Version": "1.0", "Root-Is-Purelib": "false", "Tag": WHEEL_TAG}),
    ):
        file, data = read_regular_file(metadata_root / name, limit = 65536)
        parsed = BytesParser().parsebytes(data, headersonly = True)
        if any(parsed.get_all(key) != [value] for key, value in expected.items()):
            raise _invalid("The installed companion metadata is incompatible.")
        metadata.append(file)
    # Fixed-depth inventory, never recursive search through a Python environment.
    names = set()
    for child in root.iterdir():
        if child.name in ("bin", "shims") and checked_path(child).is_dir():
            for item in child.iterdir():
                if len(names) > 16:
                    raise _invalid("Companion inventory exceeded its bound.")
                names.add(f"{child.name}/{item.name}")
        else:
            names.add(child.name)
        if len(names) > 16:
            raise _invalid("Companion inventory exceeded its bound.")
    if names != set(artifact_names()) | {"manifest.json"}:
        raise _invalid("The companion package contains missing or unlisted files.")
    files = {}
    for name, expected in manifest["files"].items():
        file, _ = read_regular_file(root / name, limit = expected["size"])
        if (file.sha256, file.size) != (expected["sha256"], expected["size"]):
            raise _invalid("An installed companion file changed.")
        files[name] = file
    shim_root = Path(__file__).parent
    for name, path in (
        ("policy.py", shim_root / "policy.py"),
        ("sitecustomize.py", shim_root.parent / "sandbox_site/sitecustomize.py"),
    ):
        expected, _ = read_regular_file(path, limit = 1024 * 1024)
        actual = files[f"shims/{name}.txt"]
        if (expected.sha256, expected.size) != (actual.sha256, actual.size):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_PROTOCOL_MISMATCH",
                "The companion policy shim does not match this Studio installation.",
            )
    helper = files[f"bin/python_host-{adapter.identity}.exe"]
    image = inspect_native_image(helper.path)
    if (
        image.file != helper
        or image.architecture != adapter.architecture
        or any(name.startswith("python") for name in (*image.imports, *image.delay_imports))
    ):
        raise _invalid("The native host image is not the expected standalone ABI adapter.")
    return AdmittedArtifacts(
        adapter.identity,
        identity,
        tuple(metadata),
        (SnapshotFile(helper, "trusted/python_host.exe"),)
        + tuple(SnapshotFile(files[f"shims/{name}.txt"], f"trusted/{name}") for name in SHIMS),
    )

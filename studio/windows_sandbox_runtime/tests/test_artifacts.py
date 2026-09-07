# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Companion wheel integrity and actual offline installation, not qualification."""

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox import artifacts
from core.inference.windows_sandbox.profiles import ABI_ADAPTERS, WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows artifact lane")


@pytest.fixture(scope = "session")
def runtime_wheel():
    value = os.environ.get("UNSLOTH_TEST_RUNTIME_WHEEL")
    assert value, "The artifact lane requires its built runtime wheel; absence is not a skip"
    path = Path(value).resolve()
    with zipfile.ZipFile(path) as wheel:
        dist = f"{artifacts.PACKAGE}-{artifacts.VERSION}.dist-info"
        expected = {f"{artifacts.PACKAGE}/{name}" for name in artifacts.artifact_names()}
        expected |= {
            f"{artifacts.PACKAGE}/manifest.json",
            *(f"{dist}/{name}" for name in ("METADATA", "WHEEL", "RECORD")),
        }
        assert len(wheel.namelist()) == len(expected)
        assert set(wheel.namelist()) == expected
        for item in wheel.infolist():
            assert item.file_size <= 16 * 1024 * 1024
            assert (item.external_attr >> 16) & 0o170000 == 0o100000
        artifacts.validate_manifest(wheel.read(f"{artifacts.PACKAGE}/manifest.json"))
        records = list(csv.reader(io.StringIO(wheel.read(f"{dist}/RECORD").decode())))
        assert {row[0] for row in records} == expected
        assert len(records) == len(expected)
        for name, digest, size in records:
            if name.endswith("/RECORD"):
                assert digest == size == ""
            else:
                data = wheel.read(name)
                expected_hash = (
                    base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                )
                assert digest == "sha256=" + expected_hash and int(size) == len(data)
    return path


@pytest.fixture(scope = "session")
def installed_runtime(runtime_wheel, tmp_path_factory):
    prefix = tmp_path_factory.mktemp("installed-runtime") / "Studio environment λ"
    subprocess.run(
        [sys.executable, "-I", "-S", "-m", "venv", "--without-pip", str(prefix)],
        check = True,
        capture_output = True,
        timeout = 30,
    )
    parser_wheel = runtime_wheel.parent / f"pefile-{artifacts.PEFILE_VERSION}-py3-none-any.whl"
    assert parser_wheel.is_file(), "The offline installation lane requires its pinned parser wheel"
    logger_wheel = runtime_wheel.parent / "structlog-25.5.0-py3-none-any.whl"
    assert (
        logger_wheel.is_file()
    ), "The ordinary Studio broker fixture requires its logging dependency"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(prefix / "Scripts/python.exe"),
            "install",
            "--no-index",
            "--no-deps",
            "--no-cache-dir",
            "--disable-pip-version-check",
            str(runtime_wheel),
            str(parser_wheel),
            str(logger_wheel),
        ],
        capture_output = True,
        timeout = 30,
    )
    assert result.returncode == 0, result.stderr
    check = subprocess.run(
        [sys.executable, "-m", "pip", "--python", str(prefix / "Scripts/python.exe"), "check"],
        capture_output = True,
        timeout = 15,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    return prefix


@pytest.fixture
def artifact_prefix(installed_runtime, tmp_path):
    prefix = tmp_path / "isolated-installation"
    site = prefix / "Lib/site-packages"
    site.mkdir(parents = True)
    for name in (artifacts.PACKAGE, f"{artifacts.PACKAGE}-{artifacts.VERSION}.dist-info"):
        shutil.copytree(installed_runtime / "Lib/site-packages" / name, site / name)
    return prefix


def package_root(prefix):
    return prefix / "Lib/site-packages" / artifacts.PACKAGE


def manifest_edit(prefix, edit):
    path = package_root(prefix) / "manifest.json"
    value = json.loads(path.read_bytes())
    edit(value)
    path.write_bytes(artifacts.canonical_json(value))


@pytest.mark.parametrize("adapter", ABI_ADAPTERS, ids = lambda value: value.identity)
def test_offline_installed_wheel_admits_each_matching_helper(installed_runtime, adapter):
    value = artifacts.admit_installed_artifacts(str(installed_runtime), adapter)
    assert value.abi == adapter.identity
    assert len(value.files) == 3
    assert {f.relative_path for f in value.files} == {
        "trusted/python_host.exe",
        "trusted/policy.py",
        "trusted/sitecustomize.py",
    }
    assert not hasattr(value, "qualified") and not hasattr(value, "available")
    assert not list(package_root(installed_runtime).rglob("*.pyc"))
    assert not list(package_root(installed_runtime).rglob("*.pth"))


def test_missing_companion_is_actionable_and_does_not_search_import_paths(tmp_path, monkeypatch):
    (tmp_path / "Lib/site-packages").mkdir(parents = True)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "not-installed"))
    with pytest.raises(WindowsRuntimeError) as error:
        artifacts.admit_installed_artifacts(str(tmp_path), ABI_ADAPTERS[1])
    assert error.value.code == "WINDOWS_SANDBOX_RUNTIME_MISSING"


@pytest.mark.parametrize(
    "field,value",
    [("version", "999"), ("profile", "0" * 64), ("protocol", 999), ("protocol", True)],
)
def test_incompatible_manifest_fails_closed(artifact_prefix, field, value):
    manifest_edit(artifact_prefix, lambda item: item.update({field: value}))
    with pytest.raises(WindowsRuntimeError, match = "incompatible"):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])


@pytest.mark.parametrize(
    "name", ["shims/policy.py.txt", "bin/python_host-cpython-312-x64-release.exe"]
)
def test_same_size_same_mtime_replacement_is_detected(artifact_prefix, name):
    path = package_root(artifact_prefix) / name
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns = (stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(WindowsRuntimeError, match = "changed"):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])


def test_rehashing_a_different_shim_cannot_approve_it(artifact_prefix):
    name = "shims/policy.py.txt"
    path = package_root(artifact_prefix) / name
    path.write_bytes(b"raise AssertionError('not the Studio shim')\n")
    data = path.read_bytes()
    manifest_edit(
        artifact_prefix,
        lambda item: item["files"].update(
            {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}}
        ),
    )
    with pytest.raises(WindowsRuntimeError, match = "shim does not match"):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])


@pytest.mark.parametrize("name", ["__init__.py", "startup.pth", "shims/extra.py", "bin/extra.dll"])
def test_companion_code_or_unlisted_file_never_executes(artifact_prefix, name):
    sentinel = artifact_prefix / "executed"
    (package_root(artifact_prefix) / name).write_text(
        f"open({str(sentinel)!r}, 'w').write('bad')", encoding = "utf-8"
    )
    with pytest.raises(WindowsRuntimeError, match = "unlisted"):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])
    assert not sentinel.exists()


def test_hardlinked_helper_is_not_admitted(artifact_prefix):
    path = package_root(artifact_prefix) / "bin/python_host-cpython-312-x64-release.exe"
    os.link(path, artifact_prefix / "outside.exe")
    with pytest.raises(WindowsRuntimeError, match = "hardlinked"):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])


def test_junction_cannot_redirect_artifact_directory(artifact_prefix):
    import _winapi

    root = package_root(artifact_prefix)
    native = root / "bin"
    outside = artifact_prefix / "moved-bin"
    native.rename(outside)
    _winapi.CreateJunction(str(outside), str(native))
    try:
        with pytest.raises(WindowsRuntimeError, match = "reparse"):
            artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])
    finally:
        native.rmdir()
        outside.rename(native)


def test_duplicate_manifest_key_is_not_canonical(artifact_prefix):
    path = package_root(artifact_prefix) / "manifest.json"
    data = path.read_bytes()
    path.write_bytes(b'{"schema":1,' + data[1:])
    with pytest.raises(WindowsRuntimeError):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])


def test_metadata_duplicate_version_cannot_select_a_package(artifact_prefix):
    path = (
        artifact_prefix
        / "Lib/site-packages"
        / f"{artifacts.PACKAGE}-{artifacts.VERSION}.dist-info/METADATA"
    )
    path.write_bytes(b"Version: forged\n" + path.read_bytes())
    with pytest.raises(WindowsRuntimeError, match = "metadata"):
        artifacts.admit_installed_artifacts(str(artifact_prefix), ABI_ADAPTERS[1])


def test_assembler_refuses_incomplete_abi_inventory(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from package_runtime import assemble
    with pytest.raises(ValueError, match = "every declared ABI"):
        assemble({}, tmp_path)


def test_build_evidence_rejects_wrong_abi_or_toolchain(installed_runtime):
    manifest = json.loads((package_root(installed_runtime) / "manifest.json").read_bytes())
    build = manifest["builds"][ABI_ADAPTERS[1].identity]
    build["compiler"]["version"] = "unreviewed"
    with pytest.raises(WindowsRuntimeError, match = "toolchain"):
        artifacts.validate_build(build, ABI_ADAPTERS[1])


@pytest.mark.parametrize("change", ["binary", "source", "abi", "protocol"])
def test_assembler_rejects_mismatched_build_evidence(artifact_prefix, tmp_path, change):
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from package_runtime import assemble

    root = package_root(artifact_prefix)
    manifest = json.loads((root / "manifest.json").read_bytes())
    hosts = {}
    for adapter in ABI_ADAPTERS:
        binary = root / f"bin/python_host-{adapter.identity}.exe"
        build = manifest["builds"][adapter.identity]
        if adapter == ABI_ADAPTERS[1]:
            if change == "binary":
                data = bytearray(binary.read_bytes())
                data[-1] ^= 1
                binary.write_bytes(data)
            elif change == "source":
                build["sources"][artifacts.SOURCE_NAMES[0]] = "0" * 64
            elif change == "abi":
                build["abi"] = ABI_ADAPTERS[0].identity
            else:
                build["protocol"] += 1
        binary.with_suffix(".build.json").write_bytes(artifacts.canonical_json(build))
        hosts[adapter.identity] = str(binary)
    output = tmp_path / "unpublished-wheel"
    with pytest.raises((ValueError, WindowsRuntimeError), match = "evidence|provenance"):
        assemble(hosts, output)
    assert not output.exists()

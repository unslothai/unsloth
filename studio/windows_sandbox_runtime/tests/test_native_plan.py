# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static graphs and OS metadata are not live LPAC qualification evidence."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox import dependencies as pe, native_plan as graph, runtime
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    library = tmp_path / "selected λ runtime"
    library.mkdir()
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    (system_dir / "kernel32.dll").write_bytes(b"OS boundary fixture")
    system = graph.SystemLoaderPolicy(str(system_dir), ("kernel32.dll",), (10, 0, 26200), "x64")
    images = {}

    def add(
        name,
        imports = (),
        delay = (),
        directory = None,
        **changes,
    ):
        path = (directory or library) / name
        path.write_bytes(name.encode() + b"static fixture")
        identity, _ = pe.read_regular_file(path, limit = 1024)
        image = pe.NativeImage(identity, "x64", tuple(imports), tuple(delay), None, None, ())
        images[str(path)] = replace(image, **changes)
        return path

    def inspect(path):
        image = images[str(path)]
        identity, _ = pe.read_regular_file(path, limit = 1024)
        return replace(image, file = identity)

    monkeypatch.setattr(graph, "inspect_native_image", inspect)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: pytest.fail("graph launched a payload")
    )

    def build(
        *roots,
        directories = None,
        bounds = graph.ScanBounds(),
    ):
        return graph.build_dependency_plan(
            tuple(map(str, roots)),
            tuple(map(str, directories or (library,))),
            architecture = "x64",
            system = system,
            bounds = bounds,
        )

    return library, system_dir, images, add, build


def test_transitive_and_delay_dependencies_get_dependency_first_order(inventory):
    _, _, _, add, build = inventory
    leaf = add("leaf.dll", ["kernel32.dll"])
    delayed = add("delay.dll", ["leaf.dll"])
    root = add("root.pyd", ["leaf.dll"], ["delay.dll"])
    plan = build(root)
    assert plan.ordered_loads == tuple(map(str, (leaf, delayed, root)))
    assert any(edge.delayed and edge.target == str(delayed) for edge in plan.edges)
    assert any(edge.kind == "known_dll" for edge in plan.edges)
    assert len(plan.images) == 3
    assert plan.trust_classification == "payload_only"
    assert plan.dynamic_dependencies == "requires_isolated_probe"
    assert not hasattr(plan, "qualified")
    assert not hasattr(plan, "startup_actions")
    assert build(root).digest == plan.digest


def test_shared_dependency_is_scanned_once(inventory, monkeypatch):
    _, _, _, add, build = inventory
    shared = add("shared.dll")
    first = add("a.dll", ["shared.dll"])
    second = add("b.dll", ["shared.dll"])
    original = graph.inspect_native_image
    calls = []

    def inspect(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(graph, "inspect_native_image", inspect)
    plan = build(first, second)
    assert calls.count(str(shared)) == 1
    assert plan.ordered_loads.index(str(shared)) < plan.ordered_loads.index(str(first))


@pytest.mark.parametrize("delayed", [False, True])
def test_missing_dependency_never_searches_path(inventory, tmp_path, monkeypatch, delayed):
    _, _, _, add, build = inventory
    external = tmp_path / "on-path"
    external.mkdir()
    add("missing.dll", directory = external)
    monkeypatch.setenv("PATH", str(external))
    root = add("root.dll", () if delayed else ["missing.dll"], ["missing.dll"] if delayed else ())
    with pytest.raises(WindowsRuntimeError, match = "DEPENDENCY_MISSING"):
        build(root)


@pytest.mark.parametrize("target", ["application", "known"])
def test_same_name_collision_is_rejected(inventory, tmp_path, target):
    library, system_dir, _, add, build = inventory
    other = tmp_path / "another package"
    other.mkdir()
    name = "kernel32.dll" if target == "known" else "shared.dll"
    add(name)
    if target == "application":
        add(name, directory = other)
    root = add("root.dll", [name])
    with pytest.raises(WindowsRuntimeError, match = "DEPENDENCY_COLLISION"):
        build(root, directories = (library, other))


def test_single_app_dependency_precedes_system32_fallback(inventory):
    _, system_dir, _, add, build = inventory
    bundled = add("vcruntime140.dll")
    add("vcruntime140.dll", directory = system_dir)
    root = add("root.dll", ["vcruntime140.dll"])
    plan = build(root)
    assert plan.edges[0].kind == "application"
    assert plan.edges[0].target == str(bundled)


def test_system32_is_used_only_without_application_candidate(inventory):
    _, system_dir, _, add, build = inventory
    system_dll = add("vcruntime140.dll", directory = system_dir)
    root = add("root.dll", ["vcruntime140.dll"])
    plan = build(root)
    assert plan.edges[0].kind == "system"
    assert plan.edges[0].target == str(system_dll)


@pytest.mark.parametrize("delayed", [False, True])
def test_cycle_has_explicit_failure_not_alphabetical_load_order(inventory, delayed):
    _, _, _, add, build = inventory
    root = add("a.dll", ["b.dll"])
    add("b.dll", () if delayed else ["a.dll"], ["a.dll"] if delayed else ())
    with pytest.raises(WindowsRuntimeError, match = "DEPENDENCY_CYCLE"):
        build(root)


def test_api_set_uses_os_contract_query_not_path(inventory, monkeypatch):
    _, _, _, add, build = inventory
    name = "api-ms-win-core-memory-l1-1-0.dll"
    root = add("root.dll", [name])
    queries = []
    monkeypatch.setattr(graph, "api_set_implemented", lambda name: queries.append(name) or True)
    plan = build(root)
    assert queries == [name]
    assert plan.edges[0].kind == "api_set"
    assert plan.edges[0].target == name
    assert len(plan.images) == 1
    add(name)
    with pytest.raises(WindowsRuntimeError, match = "API-set shadow"):
        build(root)


def test_missing_api_set_is_not_assumed_present(inventory, monkeypatch):
    _, _, _, add, build = inventory
    monkeypatch.setattr(graph, "api_set_implemented", lambda _name: False)
    root = add("root.dll", ["ext-missing-contract-l1-1-0.dll"])
    with pytest.raises(WindowsRuntimeError, match = "API set unavailable"):
        build(root)


def test_transitive_wrong_architecture_is_rejected(inventory):
    _, _, _, add, build = inventory
    root = add("root.dll", ["wrong.dll"])
    add("wrong.dll", architecture = "arm64")
    with pytest.raises(WindowsRuntimeError, match = "architecture mismatch"):
        build(root)


@pytest.mark.parametrize(
    "suffix", [".local", ".manifest", ".1.manifest", ".2.manifest", ".3.manifest"]
)
def test_external_loader_redirection_cannot_change_plan(inventory, suffix):
    _, _, _, add, build = inventory
    root = add("root.dll")
    Path(str(root) + suffix).write_text("untrusted redirection")
    with pytest.raises(WindowsRuntimeError, match = "MANIFEST_UNSUPPORTED"):
        build(root)


def test_embedded_manifest_redirection_is_explicitly_unresolved(inventory):
    _, _, _, add, build = inventory
    root = add("root.dll", manifests = (pe.ImageManifest("a" * 64, ("dependency",)),))
    with pytest.raises(WindowsRuntimeError, match = "manifest redirection"):
        build(root)


@pytest.mark.parametrize("bound", ["entries", "images", "edges", "depth", "bytes"])
def test_graph_limits_fail_closed(inventory, bound):
    _, _, _, add, build = inventory
    root = add("root.dll", ["child.dll", "kernel32.dll"])
    add("child.dll")
    with pytest.raises(WindowsRuntimeError, match = "SCAN_LIMIT"):
        build(root, bounds = replace(graph.ScanBounds(), **{bound: 1}))


def test_deadline_failure_does_not_return_partial_plan(inventory, monkeypatch):
    _, _, _, add, build = inventory
    root = add("root.dll")
    ticks = iter((0, 31))
    monkeypatch.setattr(graph.time, "monotonic", lambda: next(ticks))
    with pytest.raises(WindowsRuntimeError, match = "timed out"):
        build(root)


def test_dependency_content_change_changes_plan_digest(inventory):
    _, _, _, add, build = inventory
    root = add("root.dll", ["child.dll"])
    child = add("child.dll")
    before = build(root)
    info = child.stat()
    child.write_bytes(b"x" * info.st_size)
    os.utime(child, ns = (info.st_atime_ns, info.st_mtime_ns))
    assert build(root).digest != before.digest


def test_hardlinked_dependency_is_not_admitted(inventory, tmp_path):
    _, _, _, add, build = inventory
    root = add("root.dll", ["child.dll"])
    child = add("child.dll")
    os.link(child, tmp_path / "outside-alias")
    with pytest.raises(WindowsRuntimeError, match = "hardlinked"):
        build(root)


def test_metadata_manifest_is_hashed_without_executing_it():
    content = b'<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0"><assemblyIdentity name="test" version="1.0.0.0"/></assembly>'
    result = pe.inspect_manifest(content)
    assert result.loader_directives == ()
    assert len(result.sha256) == 64


@pytest.mark.parametrize("modified", [False, True])
def test_common_controls_identity_is_exact_not_a_general_sxs_allowlist(modified):
    token = "0000000000000000" if modified else "6595b64144ccf1df"
    content = f"""<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
      <dependency><dependentAssembly><assemblyIdentity type="win32"
        name="Microsoft.Windows.Common-Controls" version="6.0.0.0"
        processorArchitecture="*" publicKeyToken="{token}" language="*"/>
      </dependentAssembly></dependency></assembly>""".encode()
    result = pe.inspect_manifest(content)
    if modified:
        assert result.loader_directives == ("dependency", "dependentAssembly")
        assert result.system_assemblies == ()
    else:
        assert result.loader_directives == ()
        assert len(result.system_assemblies) == 1
        assert "6595b64144ccf1df" in result.system_assemblies[0]


def test_privileged_manifest_and_file_redirection_remain_rejected():
    content = b"""<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
      <file name="malicious.dll"/>
      <trustInfo><requestedExecutionLevel level="requireAdministrator"/></trustInfo>
    </assembly>"""
    assert pe.inspect_manifest(content).loader_directives == ("file", "privileged_execution_level")


def test_manifest_node_limit_is_bounded():
    content = (
        '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">'
        + "<assemblyIdentity/>" * 513
        + "</assembly>"
    ).encode()
    with pytest.raises(WindowsRuntimeError, match = "PE_INVALID"):
        pe.inspect_manifest(content)


def test_unparsed_resource_directory_is_not_treated_as_no_manifest():
    image = SimpleNamespace(
        OPTIONAL_HEADER = SimpleNamespace(
            DATA_DIRECTORY = [None, None, SimpleNamespace(VirtualAddress = 1234)]
        )
    )
    with pytest.raises(WindowsRuntimeError, match = "Unparsed resource"):
        pe._image_manifests(image)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_manifest_entity_and_external_access_are_rejected(encoding):
    content = '<!DOCTYPE assembly [<!ENTITY x SYSTEM "file:///host-secret">]><assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">&x;</assembly>'
    with pytest.raises(WindowsRuntimeError, match = "PE_INVALID"):
        pe.inspect_manifest(content.encode(encoding))


@pytest.mark.parametrize(
    "content",
    [b"", b"x" * (pe.MAX_MANIFEST_BYTES + 1), b"<broken", b"<assembly/>"],
    ids = ["empty", "oversized", "truncated", "namespace"],
)
def test_malformed_manifest_is_rejected(content):
    with pytest.raises(WindowsRuntimeError, match = "PE_INVALID"):
        pe.inspect_manifest(content)


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows loader metadata inspection")
def test_actual_cpython_dependency_plan_does_not_execute_python(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("interpreter executed"))
    descriptor = runtime.discover_runtime(sys.executable)
    system = graph.windows_loader_policy()
    plan = graph.build_dependency_plan(
        (descriptor.runtime_dll.file.path,),
        (descriptor.base_prefix,),
        architecture = descriptor.architecture,
        system = system,
    )
    assert plan.ordered_loads[-1] == descriptor.runtime_dll.file.path
    assert any(edge.kind == "api_set" for edge in plan.edges)
    assert any(edge.kind == "known_dll" for edge in plan.edges)
    assert plan.trust_classification == "payload_only"
    assert plan.dynamic_dependencies == "requires_isolated_probe"


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows stdlib dependency metadata")
def test_actual_stdlib_native_features_resolve_without_importing_them(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("interpreter executed"))
    descriptor = runtime.discover_runtime(sys.executable)
    directory = Path(descriptor.base_prefix) / "DLLs"
    features = tuple(
        str(directory / f"{name}.pyd")
        for name in (
            "_ssl",
            "_socket",
            "_sqlite3",
            "_overlapped",
            "_ctypes",
            "_multiprocessing",
        )
    )
    plan = graph.build_dependency_plan(
        features,
        (descriptor.base_prefix, str(directory)),
        architecture = descriptor.architecture,
        system = graph.windows_loader_policy(),
    )
    assert all(feature in plan.ordered_loads for feature in features)
    assert descriptor.runtime_dll.file.path in plan.ordered_loads
    assert any(edge.name == "sqlite3.dll" and edge.kind == "application" for edge in plan.edges)
    assert any(edge.kind == "system_assembly" for edge in plan.edges)


@pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows system metadata")
def test_system_directory_and_api_sets_ignore_environment(monkeypatch, tmp_path):
    original = graph.windows_loader_policy()
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setenv("WINDIR", str(tmp_path))
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "ARM64")
    monkeypatch.setenv("PROCESSOR_ARCHITEW6432", "x86")
    assert graph.windows_loader_policy() == original
    assert graph.api_set_implemented("api-ms-win-core-memory-l1-1-0.dll")
    assert not graph.api_set_implemented("ext-unsloth-nonexistent-contract-l99-99-99.dll")


@pytest.mark.parametrize(
    "process_machine,native_machine,success,allowed",
    [
        (0, 0x8664, True, True),
        (0x14C, 0x8664, True, False),
        (0x8664, 0xAA64, True, False),
        (0, 0xAA64, True, False),
        (0, 0, True, False),
        (0, 0x8664, False, False),
    ],
)
def test_native_architecture_query_never_admits_emulation_or_failure(
    process_machine, native_machine, success, allowed
):
    import ctypes
    from ctypes import wintypes
    from types import SimpleNamespace

    def query(handle, process, native):
        assert handle.value == wintypes.HANDLE(-1).value
        ctypes.cast(process, ctypes.POINTER(wintypes.USHORT))[0] = process_machine
        ctypes.cast(native, ctypes.POINTER(wintypes.USHORT))[0] = native_machine
        return success

    kernel = SimpleNamespace(IsWow64Process2 = query)
    if allowed:
        graph._require_native_x64(kernel)
    else:
        with pytest.raises(WindowsRuntimeError, match = "native x64"):
            graph._require_native_x64(kernel)


def test_missing_native_architecture_api_fails_closed():
    from types import SimpleNamespace
    with pytest.raises(WindowsRuntimeError, match = "query unavailable"):
        graph._require_native_x64(SimpleNamespace())

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch preflight health probe across every platform, backend and shipped marker shape.

``installed_runtime_health`` is asked at desktop launch, and a ``(False, reason)`` marks the
install stale, which runs a repair that ends in ``_existing_install_runs`` deciding
keep-or-reinstall. Those two deciders are what this file holds together.

The load-bearing property is one directional inequality: a tree this probe calls broken must
be a tree the repair would not keep. Otherwise the repair leaves the tree identical, the next
launch rejects it again, and the user has an unbreakable loop with no error to act on.
Asserted as a property over the whole matrix, since the pair of deciders is what drifts.

The other half is backwards compatibility. ``UNSLOTH_PREBUILT_INFO.json`` is append-only
across twelve shapes with no version field, so a wrong ``(False, ...)`` on any of them is a
repair loop for every user carrying that shape today.

Platforms are simulated through ``HostInfo`` and passed with ``host=``, as in
``test_keep_install_backcompat_9979``, which this file borrows its fixtures and marker tables
from. WSL is not a fourth platform: ``platform.system()`` says ``Linux`` there and every
table is chosen off ``is_windows`` / ``is_macos``, so it is graded by the Linux rows. The
arm64 hosts matter for the install kind names, though the payload intersection is picked by
the platform prefix rather than the architecture.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

WINDOWS_HOST = os.name == "nt"
ROOT_USER = hasattr(os, "geteuid") and os.geteuid() == 0

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

HostInfo = ILP.HostInfo


def _host(**kw) -> HostInfo:
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    base.update(kw)
    return HostInfo(**base)


LINUX = _host()
LINUX_ARM64 = _host(machine = "aarch64", is_x86_64 = False, is_arm64 = True)
WINDOWS = _host(system = "Windows", machine = "AMD64", is_windows = True, is_linux = False)
MACOS_ARM64 = _host(
    system = "Darwin",
    machine = "arm64",
    is_windows = False,
    is_linux = False,
    is_macos = True,
    is_x86_64 = False,
    is_arm64 = True,
    macos_version = (15, 5),
)
MACOS_X64 = _host(
    system = "Darwin",
    machine = "x86_64",
    is_windows = False,
    is_linux = False,
    is_macos = True,
    is_x86_64 = True,
    is_arm64 = False,
    macos_version = (14, 6),
)

HOSTS = [
    ("linux-x64", LINUX),
    ("linux-arm64", LINUX_ARM64),
    ("windows-x64", WINDOWS),
    ("macos-arm64", MACOS_ARM64),
    ("macos-x64", MACOS_X64),
]


# The payload each platform's install kinds share. Literal file names rather than the module's
# globs, so these are an independent statement of the requirement.
_SHARED_PAYLOAD = {
    "linux": [
        "libllama-common.so",
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
        "libggml-cpu.so",
        "libmtmd.so",
        # The Linux half of the same impl split as llama-server-impl.dll below:
        # llama-server and llama-quantize load these by DT_NEEDED. Written for every
        # shape, like the Windows one, and taken back out by required_runtime_files
        # for the shapes that do not owe them.
        "libllama-server-impl.so",
        "libllama-quantize-impl.so",
    ],
    "windows": [
        "llama.dll",
        "llama-common.dll",
        "llama-server-impl.dll",
        "llama-quantize-impl.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "mtmd.dll",
    ],
    # The names the real macos-arm64 bundle ships, one per library the runtime
    # links against.
    "macos": [
        "libllama-common.dylib",
        "libllama.dylib",
        "libggml.dylib",
        "libggml-base.dylib",
        "libggml-cpu.dylib",
        "libmtmd.dylib",
    ],
}
_BACKEND_PAYLOAD = {
    ("linux", "cuda"): ["libggml-cuda.so"],
    ("linux", "rocm"): ["libggml-hip.so"],
    ("linux", "vulkan"): ["libggml-vulkan.so"],
    ("windows", "cuda"): ["ggml-cuda.dll"],
    ("windows", "rocm"): ["ggml-hip.dll"],
    ("windows", "vulkan"): ["ggml-vulkan.dll"],
}
_PUBLISHED_PAYLOAD = {
    "linux": ["llama-diffusion-gemma-visual-server"],
    "windows": ["llama-diffusion-gemma-visual-server.exe"],
    "macos": [],
}
_CUDART_TRIO = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")

# ggml-org/llama.cpp#23462 split the per-binary entry code out between b9279 and b9283, so an
# older Windows archive is monolithic and healthy without llama-server-impl.dll.
_IMPL_SPLIT_BUILD = 9283


def _platform_of(host: HostInfo) -> str:
    return "windows" if host.is_windows else "macos" if host.is_macos else "linux"


def _runtime_dir(install_dir: Path, host: HostInfo) -> Path:
    return (
        install_dir / "build" / "bin" / "Release"
        if host.is_windows
        else install_dir / "build" / "bin"
    )


# ---------------------------------------------------------------------------
# The shipped marker shapes, oldest first, trimmed to the keys the health path reads.

S1 = {  # 2026-03-25 #4562: no release_tag, no backend, no asset_sha256
    "requested_tag": "b6099",
    "tag": "b6099",
    "asset": "llama-b6099-bin-ubuntu-x64.tar.gz",
    "source": "upstream",
    "bundle_profile": "full",
    "runtime_line": None,
    "coverage_class": None,
    "prebuilt_fallback_used": False,
    "installed_at_utc": "2026-03-26T04:11:07Z",
}
S2 = {  # 2026-04-01 #4741: release_tag + fingerprint arrive
    **S1,
    "release_tag": "b6210",
    "published_repo": "unslothai/llama.cpp",
    "asset": "app-b6210-linux-x64-cuda12.tar.gz",
    "asset_sha256": "3f" * 32,
    "source": "published",
    "runtime_line": "cuda12",
    "install_fingerprint": "aa" * 32,
}
S3 = {**S2, "prebuilt_fallback_used": True}  # a fallback install, same keys
S4 = {**S2, "coverage_class": "older"}  # coverage_class starts being filled in
S5 = {**S2, "force_cpu": False}  # 2026-07-20 #7228
S6 = {
    **S5,
    "llama_backend": "vulkan",  # 2026-07-27 #7373
    "asset": "llama-b7001-bin-ubuntu-vulkan-x64.tar.gz",
    "runtime_line": None,
}
S7 = {**S5, "ggml_tree": "b7440"}  # 2026-08-04 #7817
S8 = {
    **S7,
    "rocm_gfx": "gfx1151",  # 2026-08-08 #8050
    "asset": "app-b9001-linux-x64-rocm-gfx110X.tar.gz",
    "runtime_line": None,
}
S9 = {**S7, "backend": "cuda", "backend_request": "auto"}  # 2026-08-13 #8520
S10 = {**S9, "gfx_target": None, "mapped_targets": []}  # 2026-08-13 #7670
S11 = {**S10, "supported_sms": ["80", "86", "89", "90"]}  # 2026-08-18 #8841 == main
S12 = {**S11, "runtime_asset": None}  # this PR

# A real marker, produced by running studio/install_llama_prebuilt.py. Its tag is past the
# impl split, so it is the one shape that owes llama-server-impl.dll on Windows.
S12_REAL = {
    "requested_tag": "latest",
    "tag": "b10698",
    "release_tag": "b10698-mix-67dfc8b",
    "published_repo": "unslothai/llama.cpp",
    "asset": "app-b10698-mix-67dfc8b-linux-x64-cuda13-newer.tar.gz",
    "force_cpu": False,
    "llama_backend": None,
    "backend": "cuda",
    "backend_request": "auto",
    "asset_sha256": "d4" * 32,
    "runtime_asset": None,
    "source": "published",
    "ggml_tree": "0034c6eb",
    "bundle_profile": "cuda13-newer",
    "runtime_line": "cuda13",
    "coverage_class": "newer",
    "gfx_target": None,
    "mapped_targets": [],
    "supported_sms": ["86", "89", "90", "100", "103", "120"],
    "install_fingerprint": "36" * 32,
    "prebuilt_fallback_used": False,
    "installed_at_utc": "2026-08-31T06:40:37Z",
}

SHAPES = [
    ("S1", S1),
    ("S2", S2),
    ("S3", S3),
    ("S4", S4),
    ("S5", S5),
    ("S6", S6),
    ("S7", S7),
    ("S8", S8),
    ("S9", S9),
    ("S10", S10),
    ("S11", S11),
    ("S12", S12),
    ("S12real", S12_REAL),
]

# "metal" only exists on macOS, but a tree can be carried between machines, so it is
# exercised on every platform.
BACKENDS = ["cpu", "cuda", "rocm", "vulkan", "metal"]

# An asset name per backend, so shapes predating the ``backend`` key still resolve one the way
# a real old install does, through backend_from_asset_name.
_ASSET_TOKEN = {
    "cpu": "app-b1-linux-x64-cpu.tar.gz",
    "cuda": "app-b1-linux-x64-cuda12.tar.gz",
    "rocm": "app-b1-linux-x64-rocm-gfx110X.tar.gz",
    "vulkan": "app-b1-linux-x64-vulkan.tar.gz",
    "metal": "app-b1-macos-arm64.tar.gz",
}


def shape_with_backend(shape: dict, backend: str) -> dict:
    """A shipped shape re-pointed at ``backend`` the way that shape would record it.

    Shapes older than #8520 have no ``backend`` key, and adding one would test a marker no
    install ever carried, so they get the asset name ``marker_backend`` falls back to.
    """
    marker = {**shape, "asset": _ASSET_TOKEN[backend]}
    if "backend" in shape:
        marker["backend"] = backend
    return marker


def required_runtime_files(platform: str, backend: str, marker: dict) -> list[str]:
    """The files a tree of this shape owes, stated independently of the module's tables.

    Only backends the platform builds contribute a library: ``cuda`` on macOS filters to no
    install kind, so the decider falls back to every kind the platform has, whose
    intersection is the shared payload alone.
    """
    source = marker.get("source")
    files = list(_SHARED_PAYLOAD[platform])
    if platform == "windows" and source not in {"published", "upstream"}:
        # setup.ps1 links statically and ships none of the rest.
        files = ["llama.dll"]
    if platform == "windows" and source in {"published", "upstream"}:
        build = ILP._release_build_number(marker.get("tag"))
        if build is not None and build < _IMPL_SPLIT_BUILD:
            files.remove("llama-server-impl.dll")
            files.remove("llama-quantize-impl.dll")
        files.append("llama-server.exe")
    # The same split, on the side that names the libraries lib<binary>-impl.so.
    # llama-server and llama-quantize load them by DT_NEEDED, so a Linux bundle from
    # a published or upstream release owes both; a source build and a pre-split
    # archive ship neither, and requiring one of those would reinstall forever.
    if platform == "linux":
        build = ILP._release_build_number(marker.get("tag"))
        owed = source in {"published", "upstream"} and (build is None or build >= _IMPL_SPLIT_BUILD)
        if not owed:
            files.remove("libllama-server-impl.so")
            files.remove("libllama-quantize-impl.so")
    files += _BACKEND_PAYLOAD.get((platform, backend), [])
    if backend == "vulkan" and source == "published":
        files += _PUBLISHED_PAYLOAD[platform]
    return files


def build_tree(
    root: Path,
    *,
    host: HostInfo = LINUX,
    marker = "default",
    backend: str = "cuda",
    payload: bool = True,
    cudart: bool = False,
    binaries: bool = True,
    runtime_dir: bool = True,
) -> Path:
    """Write a complete install tree at ``root``.

    Complete for ``_existing_install_runs`` as well as for the health probe, since the
    invariant test drives both against the same tree, so the root entrypoint copies,
    ``convert_hf_to_gguf.py`` and ``gguf-py`` are written and the binaries are runnable
    stubs. ``marker`` is the object to serialise, ``None`` writes none, ``str`` verbatim.
    """
    platform = _platform_of(host)
    ext = ".exe" if host.is_windows else ""
    runtime = _runtime_dir(root, host)
    if runtime_dir:
        runtime.mkdir(parents = True)
    else:
        root.mkdir(parents = True, exist_ok = True)

    if binaries:
        targets = [root / f"llama-server{ext}", root / f"llama-quantize{ext}"]
        if runtime_dir:
            targets += [runtime / f"llama-server{ext}", runtime / f"llama-quantize{ext}"]
        for path in targets:
            path.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
            os.chmod(path, 0o755)
    (root / "convert_hf_to_gguf.py").write_text("", encoding = "utf-8")
    (root / "gguf-py").mkdir(exist_ok = True)

    if marker == "default":
        marker = S12
    if marker is not None:
        (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
            marker if isinstance(marker, str) else json.dumps(marker) + "\n",
            encoding = "utf-8",
        )

    if payload and runtime_dir:
        names = list(_SHARED_PAYLOAD[platform])
        names += _BACKEND_PAYLOAD.get((platform, backend), [])
        names += _PUBLISHED_PAYLOAD[platform]
        if cudart:
            names += list(_CUDART_TRIO)
        for name in names:
            (runtime / name).write_text("x", encoding = "utf-8")
    return root


# Every (host, backend, shape) cell. The decision is driven by the platform booleans and the
# marker's backend only, so this is the full space the probe can distinguish.
CELLS = [
    (f"{host_id}-{backend}-{shape_id}", host, backend, shape)
    for host_id, host in HOSTS
    for backend in BACKENDS
    for shape_id, shape in SHAPES
]
CELL_IDS = [cell[0] for cell in CELLS]


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_a_complete_install_of_every_shipped_shape_is_healthy_everywhere(
    tmp_path, cell, host, backend, shape
):
    """The backwards-compatibility half: a wrong ``(False, ...)`` on any shape sends every user
    carrying it into a repair on next launch."""
    marker = shape_with_backend(shape, backend)
    root = build_tree(tmp_path / "llama.cpp", host = host, marker = marker, backend = backend)
    assert ILP.installed_runtime_health(root, host = host) == (True, ""), cell


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_removing_any_single_required_file_is_reported_broken(tmp_path, cell, host, backend, shape):
    """One file at a time, the shape quarantine leaves: the tree is otherwise whole, so a
    check that only looks at the directory or the marker would pass it."""
    marker = shape_with_backend(shape, backend)
    platform = _platform_of(host)
    ext = ".exe" if host.is_windows else ""
    # Deduplicated: llama-server.exe is both an entrypoint and a member of the Windows shared
    # payload. One tree is built per victim, under its own name.
    victims = required_runtime_files(platform, backend, marker)
    victims += [f"llama-server{ext}", f"llama-quantize{ext}"]
    for victim in dict.fromkeys(victims):
        root = build_tree(
            tmp_path / victim.replace("*", "_"),
            host = host,
            marker = marker,
            backend = backend,
        )
        (_runtime_dir(root, host) / victim).unlink()
        verdict = ILP.installed_runtime_health(root, host = host)
        assert verdict is not None, f"{cell}: {victim}"
        ok, reason = verdict
        assert ok is False, f"{cell}: removing {victim} was called healthy"
        assert reason in {
            "llama_runtime_payload_incomplete",
            "llama_runtime_binaries_missing",
        }, f"{cell}: {victim} -> {reason}"


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_a_broken_tree_is_never_one_the_repair_would_keep(tmp_path, cell, host, backend, shape):
    """THE INVARIANT, as a property over the matrix rather than as examples.

    Every damaged tree the probe rejects is put to ``_existing_install_runs``, which the repair
    ultimately consults. A tree rejected here and kept there is repaired, left unchanged, and
    rejected again next launch, with no actionable error. The trees carry runnable stubs and
    the full ``confirm_install_tree`` set so the keep path reaches its real decision.
    """
    marker = shape_with_backend(shape, backend)
    platform = _platform_of(host)
    ext = ".exe" if host.is_windows else ""

    damages: list[tuple[str, object]] = [("healthy", None)]
    for name in required_runtime_files(platform, backend, marker):
        damages.append((f"remove-{name}", name))
    damages.append(("remove-server", f"llama-server{ext}"))
    damages.append(("remove-quantize", f"llama-quantize{ext}"))

    for label, victim in damages:
        root = build_tree(
            tmp_path / label,
            host = host,
            marker = marker,
            backend = backend,
        )
        if victim is not None:
            (_runtime_dir(root, host) / str(victim)).unlink()
        verdict = ILP.installed_runtime_health(root, host = host)
        if label == "healthy":
            # The positive control, without which the implication below holds for free.
            assert verdict == (True, ""), cell
            assert ILP._existing_install_runs(root, host) is True, cell
            continue
        if verdict is None or verdict[0] is True:
            continue
        assert ILP._existing_install_runs(root, host) is False, (
            f"REPAIR LOOP: {cell} with {label} is rejected by installed_runtime_health "
            f"({verdict[1]}) but kept by _existing_install_runs"
        )


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_the_structural_damage_cases_are_broken_and_never_kept(
    tmp_path, cell, host, backend, shape
):
    """The three ways a tree loses more than a file, on every cell: the runtime directory is
    gone, the marker is gone, the marker is unreadable."""
    marker = shape_with_backend(shape, backend)

    without_dir = build_tree(
        tmp_path / "no-runtime-dir",
        host = host,
        marker = marker,
        backend = backend,
        runtime_dir = False,
    )
    assert ILP.installed_runtime_health(without_dir, host = host) == (
        False,
        "llama_runtime_dir_missing",
    ), cell
    assert ILP._existing_install_runs(without_dir, host) is False, cell

    # No marker is a source build this path never owned: not installed, not broken. Reporting
    # it broken would offer a repair for a runtime the user never installed here.
    without_marker = build_tree(
        tmp_path / "no-marker",
        host = host,
        marker = None,
        backend = backend,
    )
    assert ILP.installed_runtime_health(without_marker, host = host) is None, cell

    # A present but unreadable marker is graded on its tree, not short-circuited to "nothing
    # installed"; load_prebuilt_metadata cannot tell the two apart, so the file itself does.
    # A complete tree stays healthy, which keeps the repair from looping, since the keep path
    # keeps it too (confirm_install_tree only checks the marker file exists).
    corrupt = build_tree(
        tmp_path / "corrupt-marker",
        host = host,
        marker = "{not json",
        backend = backend,
    )
    assert ILP.installed_runtime_health(corrupt, host = host) == (True, ""), cell
    assert ILP._existing_install_runs(corrupt, host) is True, cell

    # Damaged as well as unreadable: the case the old behaviour missed, leaving preflight
    # Ready with a library gone.
    corrupt_and_gutted = build_tree(
        tmp_path / "corrupt-marker-gutted",
        host = host,
        marker = "{not json",
        backend = backend,
    )
    for path in sorted(_runtime_dir(corrupt_and_gutted, host).glob("*")):
        if path.is_file() and not path.name.startswith("llama-"):
            path.unlink()
    verdict = ILP.installed_runtime_health(corrupt_and_gutted, host = host)
    assert verdict is not None and verdict[0] is False, cell
    assert ILP._existing_install_runs(corrupt_and_gutted, host) is False, cell


def test_removing_a_file_this_install_kind_does_not_owe_stays_healthy(tmp_path):
    """Over-strictness is the loop direction, so the deliberately-not-required files are
    asserted too: the diffusion visual server outside a published Vulkan install, and
    llama-server-impl.dll on a Windows archive built before the upstream impl split."""
    cuda = shape_with_backend(S12, "cuda")
    root = build_tree(tmp_path / "cuda", host = LINUX, marker = cuda, backend = "cuda")
    (_runtime_dir(root, LINUX) / "llama-diffusion-gemma-visual-server").unlink()
    assert ILP.installed_runtime_health(root, host = LINUX) == (True, "")

    old_windows = shape_with_backend(S2, "cpu")  # tag b6099, before build 9283
    win = build_tree(tmp_path / "win", host = WINDOWS, marker = old_windows, backend = "cpu")
    (_runtime_dir(win, WINDOWS) / "llama-server-impl.dll").unlink()
    assert ILP.installed_runtime_health(win, host = WINDOWS) == (True, "")

    # S11 and older cannot record a cudart pairing, so demanding the trio would reject every
    # Windows CUDA install that exists today, on every launch.
    unpaired = build_tree(
        tmp_path / "unpaired",
        host = WINDOWS,
        marker = shape_with_backend(S11, "cuda"),
        backend = "cuda",
        cudart = False,
    )
    assert ILP.installed_runtime_health(unpaired, host = WINDOWS) == (True, "")


# ---------------------------------------------------------------------------
# GPU independence. The probe runs with platform_only_host(), which probes no hardware, so a
# verdict that moved with the GPU would make the same tree healthy or broken depending on
# which detection the caller paid for.

GPU_HOSTS = [
    ("no-gpu", {}),
    ("nvidia", {"has_physical_nvidia": True, "has_usable_nvidia": True, "compute_caps": ["8.9"]}),
    ("nvidia-unusable", {"has_physical_nvidia": True, "has_usable_nvidia": False}),
    ("rocm", {"has_rocm": True, "rocm_gfx_target": "gfx1151"}),
    ("amd-no-rocm", {"has_amd_gpu_without_rocm": True}),
    ("intel", {"has_intel_gpu": True}),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("host_id", "host"),
    HOSTS,
    ids = [h[0] for h in HOSTS],
)
def test_the_verdict_does_not_move_with_the_detected_gpu(tmp_path, host_id, host, backend):
    """Same tree, six hardware stories, one verdict. Asserted healthy and gutted, so a check
    that silently relaxed on, say, an Intel host is caught."""
    marker = shape_with_backend(S12, backend)
    healthy = build_tree(tmp_path / "healthy", host = host, marker = marker, backend = backend)
    gutted = build_tree(tmp_path / "gutted", host = host, marker = marker, backend = backend)
    (_runtime_dir(gutted, host) / _SHARED_PAYLOAD[_platform_of(host)][0]).unlink()

    healthy_verdicts = set()
    gutted_verdicts = set()
    for gpu_id, fields in GPU_HOSTS:
        variant = _host(**{**host.__dict__, **fields})
        healthy_verdicts.add(ILP.installed_runtime_health(healthy, host = variant))
        gutted_verdicts.add(ILP.installed_runtime_health(gutted, host = variant))
    assert healthy_verdicts == {(True, "")}, f"{host_id}-{backend}: {healthy_verdicts}"
    assert gutted_verdicts == {
        (False, "llama_runtime_payload_incomplete")
    }, f"{host_id}-{backend}: {gutted_verdicts}"


# ---------------------------------------------------------------------------
# Both entrypoint binaries, since the payload groups name libraries only, so on Linux and
# macOS a quarantined llama-server would otherwise read as a complete install.


@pytest.mark.parametrize(
    ("host_id", "host"),
    HOSTS,
    ids = [h[0] for h in HOSTS],
)
@pytest.mark.parametrize(
    "missing",
    [("server",), ("quantize",), ("server", "quantize")],
    ids = ["server", "quantize", "both"],
)
def test_a_missing_entrypoint_is_caught_with_a_complete_library_payload(
    tmp_path, host_id, host, missing
):
    marker = shape_with_backend(S12, "cuda")
    root = build_tree(tmp_path / "tree", host = host, marker = marker, backend = "cuda")
    ext = ".exe" if host.is_windows else ""
    for name in missing:
        (_runtime_dir(root, host) / f"llama-{name}{ext}").unlink()
    ok, reason = ILP.installed_runtime_health(root, host = host)
    assert ok is False
    if host.is_windows and "server" in missing:
        # llama-server.exe is itself in the Windows shared payload, so the payload check
        # answers first.
        assert reason == "llama_runtime_payload_incomplete"
    else:
        assert reason == "llama_runtime_binaries_missing"


# ---------------------------------------------------------------------------
# Robustness. Every case is a real disk state a user can arrive in, and the failure to avoid
# is the same throughout: an exception or a wrong verdict on the launch path.


def test_an_install_dir_reached_through_a_symlink_is_judged_the_same(tmp_path):
    """Installs move, and a symlink at the old path is how users keep the launcher working."""
    real = build_tree(tmp_path / "real", host = LINUX, marker = S12, backend = "cuda")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory = True)
    assert ILP.installed_runtime_health(link, host = LINUX) == (True, "")


def test_a_runtime_path_that_is_a_file_is_broken_not_an_exception(tmp_path):
    """A truncated extract can leave build/bin as a regular file, and globbing a
    non-directory must not raise on the launch path."""
    root = tmp_path / "llama.cpp"
    (root / "build").mkdir(parents = True)
    (root / "build" / "bin").write_text("not a directory", encoding = "utf-8")
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(S12), encoding = "utf-8")
    assert ILP.installed_runtime_health(root, host = LINUX) == (False, "llama_runtime_dir_missing")
    assert ILP._existing_install_runs(root, LINUX) is False


@pytest.mark.skipif(WINDOWS_HOST, reason = "chmod cannot clear read permission on Windows")
@pytest.mark.skipif(ROOT_USER, reason = "root reads a 000 file regardless of its mode")
def test_a_marker_the_process_cannot_read_is_graded_on_its_tree(tmp_path):
    """Permission denied is one more way a marker stops parsing, so it lands in the same arm as
    a truncated one: the tree is real and is graded rather than called not installed."""
    root = build_tree(tmp_path / "llama.cpp", host = LINUX, marker = S12, backend = "cuda")
    marker_path = root / "UNSLOTH_PREBUILT_INFO.json"
    os.chmod(marker_path, 0o000)
    try:
        if os.access(marker_path, os.R_OK):
            pytest.skip("running as a user that ignores the mode, so nothing is denied")
        assert ILP.installed_runtime_health(root, host = LINUX) == (True, "")
    finally:
        os.chmod(marker_path, 0o644)


@pytest.mark.parametrize(
    "body",
    ["[]", '["cuda"]', "123", '"cuda"', "null", "", "   ", "{}", "﻿{}", '{"tag": "b1"'],
    ids = [
        "array",
        "array-of-strings",
        "number",
        "string",
        "null",
        "empty",
        "whitespace",
        "empty-object",
        "bom",
        "truncated",
    ],
)
def test_a_marker_that_is_not_an_object_never_raises_and_never_loops(tmp_path, body):
    """Valid JSON that is not an object and invalid JSON both read as "no marker" through
    load_prebuilt_metadata. The empty object is a marker, and it names no backend, so it owes
    the payload every install kind on the platform shares."""
    root = build_tree(tmp_path / "llama.cpp", host = LINUX, marker = body, backend = "cuda")
    verdict = ILP.installed_runtime_health(root, host = LINUX)
    assert verdict in (None, (True, ""))
    if verdict is not None and verdict[0] is False:
        assert ILP._existing_install_runs(root, LINUX) is False


def test_a_tree_full_of_unrelated_files_is_still_answered_fast(tmp_path):
    """On the launch path, so timed as well as checked: the payload check globs the runtime
    directory once per group, and a user's also holds every model shard they dropped in it."""
    root = build_tree(tmp_path / "llama.cpp", host = LINUX, marker = S12, backend = "cuda")
    runtime = _runtime_dir(root, LINUX)
    for index in range(2000):
        (runtime / f"unrelated-{index}.bin").write_text("", encoding = "utf-8")

    start = time.perf_counter()
    verdict = ILP.installed_runtime_health(root, host = LINUX)
    elapsed = time.perf_counter() - start
    assert verdict == (True, "")
    assert elapsed < 1.0, f"took {elapsed:.3f}s with 2000 extra files"


@pytest.mark.parametrize(
    "name",
    ["with spaces", "unslöth ünicode", "日本語のパス", "trailing.dot."],
    ids = ["spaces", "unicode-latin", "unicode-cjk", "trailing-dot"],
)
def test_an_awkward_install_path_is_judged_normally(tmp_path, name):
    """Default install roots sit under the user's home, so a path handled as anything other
    than a Path would fail at launch for those users only."""
    root = build_tree(tmp_path / name / "llama.cpp", host = LINUX, marker = S12, backend = "cuda")
    assert ILP.installed_runtime_health(root, host = LINUX) == (True, "")
    (_runtime_dir(root, LINUX) / "libggml.so").unlink()
    assert ILP.installed_runtime_health(root, host = LINUX) == (
        False,
        "llama_runtime_payload_incomplete",
    )


def test_a_very_deep_install_path_is_judged_normally(tmp_path):
    """A deep root is the cheapest way to keep paths off string arithmetic; on Windows it is
    also where a non-extended path stops resolving."""
    deep = tmp_path
    for index in range(40):
        deep = deep / f"level{index}"
    root = build_tree(deep / "llama.cpp", host = LINUX, marker = S12, backend = "cuda")
    assert ILP.installed_runtime_health(root, host = LINUX) == (True, "")


def test_the_probe_never_executes_anything_it_finds(tmp_path, monkeypatch):
    """Preflight runs at every launch and the setup scripts own the exec probes, so this looks
    and does not run. Also why the invariant is one-directional: this call can only be as
    strict as, or looser than, the keep path."""
    root = build_tree(tmp_path / "llama.cpp", host = LINUX, marker = S12, backend = "cuda")

    def refuse(*args, **kwargs):
        raise AssertionError("installed_runtime_health must not spawn a process")

    monkeypatch.setattr(ILP.subprocess, "run", refuse)
    monkeypatch.setattr(ILP.shutil, "which", refuse)
    assert ILP.installed_runtime_health(root, host = LINUX) == (True, "")

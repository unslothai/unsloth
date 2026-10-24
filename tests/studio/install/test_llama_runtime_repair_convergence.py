# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launch cycle as a state machine over real trees: does probe, repair, probe terminate?"""

import hashlib
import os
import shutil
import sys
import urllib.error
from pathlib import Path

import pytest

# Take ILP from the fixture module so the patched and probed module are the same object.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_installed_runtime_health_matrix as MF  # noqa: E402

ILP = MF.ILP

# arm64 rows walk the same rows as their x64 twins, and each step is a real install_prebuilt.
HOSTS = [
    ("linux", MF.LINUX),
    ("windows", MF.WINDOWS),
    ("macos", MF.MACOS_ARM64),
]
BACKENDS = ["cpu", "cuda", "rocm", "vulkan"]
# Shapes that differ in what the deciders read, not all thirteen.
SHAPES = [
    ("S1", MF.S1),
    ("S2", MF.S2),
    ("S6", MF.S6),
    ("S8", MF.S8),
    ("S11", MF.S11),
    ("S12", MF.S12),
    ("S12real", MF.S12_REAL),
]

CELLS = [
    (f"{host_id}-{backend}-{shape_id}", host, backend, shape)
    for host_id, host in HOSTS
    for backend in BACKENDS
    for shape_id, shape in SHAPES
]
CELL_IDS = [cell[0] for cell in CELLS]

# macOS publishes one universal Metal bundle, so the plan is Metal whatever the marker says.
_INSTALL_KIND = {
    ("linux", "cpu"): "linux-cpu",
    ("linux", "cuda"): "linux-cuda",
    ("linux", "rocm"): "linux-rocm",
    ("linux", "vulkan"): "linux-vulkan",
    ("windows", "cpu"): "windows-cpu",
    ("windows", "cuda"): "windows-cuda",
    ("windows", "rocm"): "windows-rocm",
    ("windows", "vulkan"): "windows-vulkan",
}


def fingerprint(root: Path) -> str:
    """Digest of names, sizes, modes and marker bytes; directory sizes are excluded (ext4 grows them)."""
    digest = hashlib.sha256()
    if not root.exists():
        return "absent"
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        size = "dir" if path.is_dir() else str(stat.st_size)
        digest.update(f"{path.is_dir()}:{size}:{stat.st_mode & 0o777}".encode("utf-8"))
    marker = root / "UNSLOTH_PREBUILT_INFO.json"
    if marker.is_file():
        digest.update(marker.read_bytes())
    return digest.hexdigest()


def probe(root: Path, host) -> tuple[bool, str] | None:
    return ILP.installed_runtime_health(root, host = host)


def _platform_of(host) -> str:
    return "windows" if host.is_windows else "macos" if host.is_macos else "linux"


def current_marker(host, backend: str) -> dict:
    """Fresh-install marker stamped with the keep check's fingerprint, so the next cycle keeps it."""
    platform = _platform_of(host)
    kind = "macos-arm64" if platform == "macos" else _INSTALL_KIND[(platform, backend)]
    marker = {
        **MF.S12,
        "requested_tag": "latest",
        "tag": "b10698",
        "release_tag": "b10698-mix-67dfc8b",
        "published_repo": "unslothai/llama.cpp",
        "asset": MF._ASSET_TOKEN[backend],
        "asset_sha256": "d4" * 32,
        "backend": ILP.backend_for_install_kind(kind),
        "backend_request": "auto",
        "source": "published",
        "runtime_asset": None,
        "runtime_line": None,
        "bundle_profile": None,
        "coverage_class": None,
        "llama_backend": None,
    }
    marker["install_fingerprint"] = ILP.expected_install_fingerprint(
        llama_tag = marker["tag"],
        release_tag = marker["release_tag"],
        choice = _choice(marker, kind),
        approved_checksums = _checksums(marker),
    )
    return marker


def _choice(marker: dict, install_kind: str):
    return ILP.AssetChoice(
        repo = marker["published_repo"],
        tag = marker["tag"],
        name = marker["asset"],
        url = f"https://example.invalid/{marker['asset']}",
        source_label = marker["source"],
        install_kind = install_kind,
        bundle_profile = marker["bundle_profile"],
        runtime_line = marker["runtime_line"],
        coverage_class = marker["coverage_class"],
        expected_sha256 = marker["asset_sha256"],
        runtime_name = marker["runtime_asset"],
    )


def _checksums(marker: dict):
    return ILP.ApprovedReleaseChecksums(
        repo = marker["published_repo"],
        release_tag = marker["release_tag"],
        upstream_tag = marker["tag"],
    )


def current_plan(host, backend: str) -> tuple[object, dict]:
    """``(plan, marker)`` for the release a reachable listing would offer this cell."""
    platform = _platform_of(host)
    kind = "macos-arm64" if platform == "macos" else _INSTALL_KIND[(platform, backend)]
    marker = current_marker(host, backend)
    plan = ILP.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = marker["tag"],
        release_tag = marker["release_tag"],
        attempts = [_choice(marker, kind)],
        approved_checksums = _checksums(marker),
    )
    return plan, marker


def install_fresh_prebuilt(root: Path, host, backend: str) -> None:
    """Replace ``root`` with what ``activate_install_tree`` leaves behind for this cell."""
    if root.exists():
        shutil.rmtree(root)
    MF.build_tree(root, host = host, marker = current_marker(host, backend), backend = backend)


def install_fresh_source_build(root: Path, host) -> None:
    """Markerless: setup.sh's rm -rf then mv drops the marker, so the probe reads nothing installed."""
    if root.exists():
        shutil.rmtree(root)
    MF.build_tree(root, host = host, marker = None, backend = "cpu")


@pytest.fixture
def offline(monkeypatch):
    """Release listing fails with a URLError, as a dropped connection would, and output is muted."""

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(ILP, "_fork_manifest_release_plans", boom)
    monkeypatch.setattr(ILP, "collect_system_report", lambda *a, **k: "report")
    monkeypatch.setattr(ILP, "log", lambda *a, **k: None)
    monkeypatch.setattr(ILP, "log_lines", lambda *a, **k: None)
    return monkeypatch


def offline_repair(root: Path, host, monkeypatch, *, shell_stage: bool) -> str:
    """Runs the offline branch for real: returning means kept; EXIT_FALLBACK means source fallback."""
    monkeypatch.setattr(ILP, "detect_host", lambda *a, **k: host)
    try:
        ILP.install_prebuilt(root, "latest", "unslothai/llama.cpp", "")
        return "python-kept"
    except SystemExit as exc:
        code = exc.code
    if code != ILP.EXIT_FALLBACK:
        # Exit 1/3/5 reach setup_fail: the user gets an error to act on, which is not a loop.
        return f"aborted-exit{code}"
    if not shell_stage:
        install_fresh_source_build(root, host)
        return "source-rebuilt"
    # Both shells have a reuse shortcut; each must ask reusable_existing_install /
    # Test-LlamaTreeStillHealthy first, or a tree missing only a library is kept.
    if host.is_windows:
        entrypoints = [root / "build" / "bin" / "Release" / "llama-server.exe"]
        reusable = ILP.reusable_existing_install(root, host) and all(
            path.is_file() for path in entrypoints
        )
    else:
        reusable = ILP.reusable_existing_install(root, host) and all(
            os.access(root / "build" / "bin" / f"llama-{name}", os.X_OK)
            for name in ("server", "quantize")
        )
    if reusable:
        return "shell-kept"
    install_fresh_source_build(root, host)
    return "source-rebuilt"


def online_repair(root: Path, host, backend: str) -> str:
    """The reachable-plan branch: keep and re-record the selection, or reinstall."""
    plan, _ = current_plan(host, backend)
    if ILP.existing_install_matches_plan(root, host, plan):
        # The keep rewrites the marker, an input to the next probe, so it can hand over a rejected tree.
        ILP.sync_marker_selection(
            root,
            choice = plan.attempts[0],
            backend_request = "auto",
            persist_force_cpu = False,
            persist_llama_backend = None,
            ggml_tree = None,
            rocm_gfx = None,
        )
        return "kept"
    install_fresh_prebuilt(root, host, backend)
    return "reinstalled"


MAX_CYCLES = 6


def run_cycle(
    root: Path,
    host,
    repair,
    *,
    max_cycles: int = MAX_CYCLES,
):
    """Iterates probe then repair; classifies converged, aborted, loop (identical tree) or diverged."""
    trail: list[str] = []
    for cycles in range(max_cycles + 1):
        verdict = probe(root, host)
        if verdict is None or verdict[0]:
            return "converged", cycles, trail
        if cycles == max_cycles:
            return "diverged", cycles, trail
        before = fingerprint(root)
        action = repair(root)
        trail.append(action)
        if action.startswith("aborted"):
            return "aborted", cycles + 1, trail
        if fingerprint(root) == before:
            return "loop", cycles + 1, trail
    raise AssertionError("unreachable")


def damage_modes(host, backend: str, marker: dict) -> list[tuple[str, object]]:
    """Damage cases: a str removes that file, a tuple several, and @ constants a structural loss."""
    platform = _platform_of(host)
    ext = ".exe" if host.is_windows else ""
    server, quantize = f"llama-server{ext}", f"llama-quantize{ext}"
    libraries = [
        name
        for name in MF.required_runtime_files(platform, backend, marker)
        if name not in {server, quantize}
    ]
    modes: list[tuple[str, object]] = [(f"remove-{name}", name) for name in libraries]
    modes += [
        ("remove-server", server),
        ("remove-quantize", quantize),
        ("remove-both-entrypoints", (server, quantize)),
        ("remove-server-and-library", (server, libraries[0])),
        ("remove-first-and-last-library", (libraries[0], libraries[-1])),
        ("remove-runtime-dir", "@runtime-dir"),
        ("remove-tree", "@tree"),
        ("remove-marker", "@marker"),
        ("corrupt-marker", "@corrupt-marker"),
        ("corrupt-marker-and-library", "@corrupt-marker+library"),
    ]
    return modes


# Damages that already decide the marker's fate; pairing them with a corrupt marker is a no-op.
_MARKER_DAMAGES = {"@tree", "@marker", "@corrupt-marker", "@corrupt-marker+library"}


def build_damaged(
    root: Path,
    host,
    backend: str,
    marker: dict,
    damage,
    *,
    corrupt = False,
) -> Path:
    """Builds a complete tree, applies the damage, and with corrupt also writes an unparsable marker."""
    MF.build_tree(root, host = host, marker = marker, backend = backend)
    runtime = MF._runtime_dir(root, host)
    if corrupt:
        (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{not json", encoding = "utf-8")
    if damage == "@tree":
        shutil.rmtree(root)
    elif damage == "@runtime-dir":
        shutil.rmtree(runtime)
    elif damage == "@marker":
        (root / "UNSLOTH_PREBUILT_INFO.json").unlink()
    elif damage == "@corrupt-marker":
        (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{not json", encoding = "utf-8")
    elif damage == "@corrupt-marker+library":
        (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{not json", encoding = "utf-8")
        platform = _platform_of(host)
        (runtime / MF._SHARED_PAYLOAD[platform][0]).unlink()
    elif isinstance(damage, tuple):
        for name in damage:
            (runtime / name).unlink()
    else:
        (runtime / str(damage)).unlink()
    return root


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_the_online_repair_reaches_a_fixed_point_from_every_damaged_tree(
    tmp_path, cell, host, backend, shape
):
    """Reachable listing: one reinstall, and its result is a fixed point. A second cycle would
    mean the installer writes a tree its own probe rejects, a permanent loop for every user.
    """
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        root = build_damaged(tmp_path / label, host, backend, marker, damage)
        outcome, cycles, trail = run_cycle(
            root, host, lambda tree: online_repair(tree, host, backend)
        )
        assert outcome == "converged", f"{cell}/{label}: {outcome} after {cycles} ({trail})"
        assert cycles <= 1, f"{cell}/{label}: took {cycles} repairs ({trail})"


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_the_online_keep_never_rewrites_the_marker_into_a_tree_it_then_rejects(
    tmp_path, cell, host, backend, shape
):
    """Keeps must not stamp runtime_asset onto a pair-less tree: it makes the cudart trio required."""
    del shape
    root = tmp_path / "healthy"
    install_fresh_prebuilt(root, host, backend)
    assert probe(root, host) == (True, ""), cell
    for _ in range(3):
        assert online_repair(root, host, backend) == "kept", cell
        assert probe(root, host) == (True, ""), f"{cell}: a keep made the tree unhealthy"


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_the_offline_python_repair_never_keeps_a_tree_the_probe_rejects(
    tmp_path, offline, cell, host, backend, shape
):
    """Offline, a probe-rejected tree must not be python-kept, since nothing downstream repairs it."""
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        root = build_damaged(tmp_path / label, host, backend, marker, damage)
        verdict = probe(root, host)
        if verdict is None or verdict[0]:
            continue
        action = offline_repair(root, host, offline, shell_stage = False)
        assert action != "python-kept", (
            f"REPAIR LOOP: {cell}/{label} is rejected by installed_runtime_health "
            f"({verdict[1]}) and kept unchanged by the offline branch of install_prebuilt"
        )


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_the_offline_repair_terminates_once_the_source_build_actually_runs(
    tmp_path, offline, cell, host, backend, shape
):
    """Offline, with the shell's rebuild-skip left out of the model: a markerless build ends the cycle."""
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        root = build_damaged(tmp_path / label, host, backend, marker, damage)
        outcome, cycles, trail = run_cycle(
            root,
            host,
            lambda tree: offline_repair(tree, host, offline, shell_stage = False),
        )
        assert outcome in {
            "converged",
            "aborted",
        }, f"{cell}/{label}: {outcome} after {cycles} repairs ({trail})"
        assert cycles <= 1, f"{cell}/{label}: took {cycles} repairs ({trail})"


def test_both_shells_gate_their_reuse_shortcut_on_the_same_check():
    """setup.sh and setup.ps1 must gate their build-reuse shortcut on the same health check."""
    root = Path(__file__).resolve().parents[3]

    shell = (root / "studio" / "setup.sh").read_text(encoding = "utf-8")
    assert "--check-existing-install" in shell
    assert "_LLAMA_REUSE_EXISTING" in shell, "setup.sh lost its reuse gate"

    ps1 = (root / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    assert "function Test-LlamaTreeStillHealthy" in ps1, "setup.ps1 lost its reuse gate"
    assert (
        "--check-existing-install" in ps1
    ), "the PowerShell gate must ask install_llama_prebuilt, not reimplement healthy"
    # The gate must be inside the reuse predicate itself: an elseif reaching 'already built' is the bug.
    plan = ps1[ps1.index("$CanReuseLlamaBuild = ") :]
    plan = plan[: plan.index("$WillBuildLlamaFromSource = ")]
    assert "Test-LlamaTreeStillHealthy" in plan, "the reuse shortcut skips the health gate again"
    shortcut = ps1[ps1.index("} elseif ($CanReuseLlamaBuild) {") :]
    assert "already built" in shortcut[: shortcut.index("} elseif", 1)]


def test_the_windows_build_plan_asks_the_same_question_as_its_reuse_shortcut():
    """$WillBuildLlamaFromSource uses $CanReuseLlamaBuild, so a refused tree still gets its build tools."""
    ps1 = (Path(__file__).resolve().parents[3] / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    assert ps1.index("$CanReuseLlamaBuild = ") < ps1.index("$WillBuildLlamaFromSource = ")
    assert (
        "$WillBuildLlamaFromSource = $NeedLlamaSourceBuild -and -not $CanReuseLlamaBuild" in ps1
    ), "the build plan must derive from the same predicate the shortcut reads"
    # Once, so its 'incomplete' line is not printed twice.
    assert ps1.count("Test-LlamaTreeStillHealthy $LlamaCppDir") == 1, ps1.count(
        "Test-LlamaTreeStillHealthy $LlamaCppDir"
    )


def test_the_offline_repair_terminates_with_the_shell_rebuild_skip_in_place(tmp_path, offline):
    """Offline repair terminates with setup.sh's rebuild skip in place, over a quarantined CUDA library."""
    host, backend = MF.LINUX, "cuda"
    marker = MF.shape_with_backend(MF.S12, backend)
    root = build_damaged(tmp_path / "quarantined", host, backend, marker, "libggml-cuda.so")
    assert probe(root, host) == (False, "llama_runtime_payload_incomplete")
    outcome, cycles, trail = run_cycle(
        root,
        host,
        lambda tree: offline_repair(tree, host, offline, shell_stage = True),
    )
    assert outcome == "converged", f"{outcome} after {cycles} repairs ({trail})"


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_no_damaged_tree_survives_the_shell_rebuild_skip(
    tmp_path, offline, cell, host, backend, shape
):
    """No damaged tree survives the shell's rebuild skip, which now asks reusable_existing_install first."""
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        root = build_damaged(tmp_path / label, host, backend, marker, damage)
        verdict = probe(root, host)
        if verdict is None or verdict[0]:
            continue
        outcome, cycles, trail = run_cycle(
            root,
            host,
            lambda tree: offline_repair(tree, host, offline, shell_stage = True),
        )
        assert outcome in {"converged", "aborted"}, f"{cell}/{label}: {outcome} ({trail})"
        assert cycles <= 1, f"{cell}/{label}: {cycles} repairs ({trail})"


@pytest.mark.parametrize(("host_id", "host"), HOSTS, ids = [h[0] for h in HOSTS])
def test_a_runtime_quarantined_again_after_every_repair_is_not_a_code_loop(tmp_path, host_id, host):
    """Antivirus that re-quarantines after each repair never converges, and must not read as
    the bug above. The distinguisher is progress, not convergence: every repair here replaces
    the tree, whereas the shell shortcut returned an identical one."""
    backend = "cuda"
    root = tmp_path / "quarantined"
    install_fresh_prebuilt(root, host, backend)
    seen: list[str] = []
    for _ in range(4):
        # The quarantine strikes between the install and the launch, every time.
        (MF._runtime_dir(root, host) / MF._SHARED_PAYLOAD[_platform_of(host)][0]).unlink()
        verdict = probe(root, host)
        assert verdict is not None and verdict[0] is False, host_id
        before = fingerprint(root)
        assert online_repair(root, host, backend) == "reinstalled", host_id
        after = fingerprint(root)
        assert after != before, f"{host_id}: the repair did not change the tree"
        seen.append(after)
    assert len(set(seen)) == 1, "each repair should rebuild the same healthy tree"


def test_a_corrupt_marker_over_a_damaged_payload_is_graded_and_not_called_uninstalled(tmp_path):
    """A corrupt marker over a damaged payload is graded, not treated as no install."""
    host, backend = MF.LINUX, "cuda"
    gutted = build_damaged(
        tmp_path / "corrupt-gutted",
        host,
        backend,
        MF.shape_with_backend(MF.S12, backend),
        "@corrupt-marker+library",
    )
    assert probe(gutted, host) == (False, "llama_runtime_payload_incomplete")
    assert ILP._existing_install_runs(gutted, host) is False


def test_the_two_kinds_of_missing_marker_stay_apart(tmp_path):
    """An absent marker means not installed (None); an unparsable one still grades the payload."""
    host, backend = MF.LINUX, "cuda"
    shape = MF.shape_with_backend(MF.S12, backend)

    absent = build_damaged(tmp_path / "absent", host, backend, shape, "@marker")
    assert probe(absent, host) is None
    # confirm_install_tree requires the marker file, so the keep path refuses this outright.
    assert ILP._existing_install_runs(absent, host) is False

    corrupt = build_damaged(tmp_path / "corrupt", host, backend, shape, "@corrupt-marker")
    assert probe(corrupt, host) == (True, "")
    assert ILP._existing_install_runs(corrupt, host) is True


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_grading_an_unparseable_marker_keeps_the_no_stricter_invariant(
    tmp_path, cell, host, backend, shape
):
    """The static invariant holds for every damage crossed with an unparsable marker."""
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        if damage in _MARKER_DAMAGES:
            continue
        root = build_damaged(tmp_path / label, host, backend, marker, damage, corrupt = True)
        verdict = probe(root, host)
        if verdict is None or verdict[0]:
            continue
        assert ILP._existing_install_runs(root, host) is False, (
            f"REPAIR LOOP: {cell}/{label} with an unparseable marker is rejected by the "
            f"pending probe ({verdict[1]}) but kept by _existing_install_runs"
        )


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_grading_an_unparseable_marker_still_terminates_on_every_cell(
    tmp_path, offline, cell, host, backend, shape
):
    """Unparsable markers over every damage still reach a fixed point under both repair models."""
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        if damage in _MARKER_DAMAGES:
            continue
        for model, repair in (
            ("online", lambda tree: online_repair(tree, host, backend)),
            ("offline", lambda tree: offline_repair(tree, host, offline, shell_stage = False)),
        ):
            root = build_damaged(
                tmp_path / f"{model}-{label}",
                host,
                backend,
                marker,
                damage,
                corrupt = True,
            )
            outcome, cycles, trail = run_cycle(root, host, repair)
            assert outcome in {
                "converged",
                "aborted",
            }, f"{cell}/{model}/{label}: {outcome} after {cycles} repairs ({trail})"
            assert cycles <= 1, f"{cell}/{model}/{label}: {cycles} repairs ({trail})"


def test_the_freshly_installed_tree_is_a_fixed_point_on_every_cell(tmp_path):
    """The base case the whole argument rests on: whatever the repair installs is accepted. If
    this failed for any cell, every cycle above would be infinite whichever branch ran.
    """
    for host_id, host in HOSTS:
        for backend in BACKENDS:
            prebuilt = tmp_path / f"{host_id}-{backend}-prebuilt"
            install_fresh_prebuilt(prebuilt, host, backend)
            assert probe(prebuilt, host) == (True, ""), f"{host_id}-{backend}"

            source = tmp_path / f"{host_id}-{backend}-source"
            install_fresh_source_build(source, host)
            assert probe(source, host) is None, f"{host_id}-{backend}"

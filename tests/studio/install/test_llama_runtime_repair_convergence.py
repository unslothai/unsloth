# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does ``probe -> repair -> probe`` terminate? The dynamic half of the launch preflight.

``test_installed_runtime_health_matrix`` proves a STATIC implication: no damaged tree is
rejected by ``installed_runtime_health`` and kept by ``_existing_install_runs``. Necessary,
not sufficient. The repair is not ``_existing_install_runs`` but the whole chain ``unsloth
studio update`` -> ``setup.sh`` / ``setup.ps1`` -> ``install_llama_prebuilt.py``, and the
decision to leave the tree alone is taken in three places along it, so the implication can
hold while the cycle still makes no progress.

So this file simulates the launch cycle as a state machine over real trees on disk:

  probe    installed_runtime_health(tree, host = host), the real function.
  repair   one of three models, each named after the branch of the real chain it stands for.
  state    the tree itself, fingerprinted so "the repair changed nothing" is observable.

The three repair models:

  ONLINE       the listing is reachable, so ``install_prebuilt`` either records the reused
               selection (``sync_marker_selection``, which REWRITES the marker the next probe
               reads) or installs a fresh bundle. Modelled by calling the real
               ``existing_install_matches_plan`` and ``sync_marker_selection``.
  OFFLINE      the listing raised, so the ``PrebuiltFallback`` handler's
               ``_existing_install_runs`` gate KEEPS a complete tree. Not modelled: the real
               ``install_prebuilt`` is called with the listing broken and its exit code read.
  OFFLINE+SH   the same, continued into the shell: EXIT_FALLBACK sets
               ``_NEED_LLAMA_SOURCE_BUILD`` and the source-build stage may skip the rebuild.
               That guard is the one place in the chain where a tree the probe rejects can be
               left byte for byte identical.

The marker is an axis of its own rather than one more damage, since the probe reads it and
``sync_marker_selection`` writes it, so it is the one piece of state that could carry a cycle.
Every damage is therefore also run with a marker that does not parse: the probe grades those
trees, and grading them must not reject anything ``_existing_install_runs`` would keep.

Fixtures, marker shapes and payload tables come from
``test_installed_runtime_health_matrix`` rather than being restated, so the state machine runs
over exactly the trees the static invariant was proved over. WSL is not a fourth platform:
``platform.system()`` reports ``Linux`` there, so it is graded by the Linux rows.
"""

import hashlib
import os
import shutil
import sys
import urllib.error
from pathlib import Path

import pytest

# The fixture module execs studio/install_llama_prebuilt.py and caches it in sys.modules.
# Taking ILP from it keeps the monkeypatched and the probed module the same object.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_installed_runtime_health_matrix as MF  # noqa: E402

ILP = MF.ILP

# Three platforms. The arm64 rows are dropped because the payload intersection is picked by
# the platform prefix, so they walk the same rows as their x64 twins, and each step costs a
# real install_prebuilt call.
HOSTS = [
    ("linux", MF.LINUX),
    ("windows", MF.WINDOWS),
    ("macos", MF.MACOS_ARM64),
]
# "metal" is dropped for the same reason: a metal marker on Linux only re-tests the fall-open
# path the static file already pins.
BACKENDS = ["cpu", "cuda", "rocm", "vulkan"]
# A spread of shipped shapes rather than all thirteen: the ones differing in what the deciders
# read (no release_tag, no fingerprint, no backend key, a legacy vulkan override, a rocm arch,
# today's shape, and a marker a real run produced).
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

# The install kind a fresh install lands on per host and backend. macOS publishes one
# universal Metal bundle, so its backend axis is a marker carried in from elsewhere and the
# plan below is the Metal one whatever the marker says.
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


# ---------------------------------------------------------------------------
# State


def fingerprint(root: Path) -> str:
    """A digest of everything the probe and the keep path can see.

    Names, sizes and mode bits plus the marker's bytes, which is enough to tell "rewrote the
    marker" and "replaced the tree" apart from "did nothing".

    Directory sizes are excluded: on ext4 a directory's ``st_size`` grows with the entries it
    has ever held, so an identical tree rebuilt in a reused path would hash differently.
    """
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


# ---------------------------------------------------------------------------
# The plan a reachable release listing would produce, and the tree it installs


def _platform_of(host) -> str:
    return "windows" if host.is_windows else "macos" if host.is_macos else "linux"


def current_marker(host, backend: str) -> dict:
    """The marker a fresh install of today's release writes for this cell.

    Built off the shipped S12 shape so it stays real, then stamped with the fingerprint
    ``existing_install_matches_choice`` recomputes, which is what makes the freshly installed
    tree a fixed point rather than something the next cycle reinstalls.
    """
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
    """Replace ``root`` with what setup.sh's source-build swap leaves behind.

    Markerless, and load-bearing rather than a simplification: the swap is ``rm -rf`` then
    ``mv``, so ``UNSLOTH_PREBUILT_INFO.json`` cannot survive it and the next probe reads
    "nothing installed" instead of judging a source build by a prebuilt payload table. A swap
    that preserved the marker would be a loop of its own.
    """
    if root.exists():
        shutil.rmtree(root)
    MF.build_tree(root, host = host, marker = None, backend = "cpu")


# ---------------------------------------------------------------------------
# The three repair models


@pytest.fixture
def offline(monkeypatch):
    """Make the release listing fail the way a dropped connection does, and silence the log.

    Same technique as ``test_keep_install_backcompat_9979._transient_listing_failure``. The
    muting is not cosmetic: the cycle runs install_prebuilt thousands of times, each printing
    a paragraph.
    """

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(ILP, "_fork_manifest_release_plans", boom)
    monkeypatch.setattr(ILP, "collect_system_report", lambda *a, **k: "report")
    monkeypatch.setattr(ILP, "log", lambda *a, **k: None)
    monkeypatch.setattr(ILP, "log_lines", lambda *a, **k: None)
    return monkeypatch


def offline_repair(root: Path, host, monkeypatch, *, shell_stage: bool) -> str:
    """The OFFLINE branch, run for real, optionally continued into setup.sh.

    ``install_prebuilt`` returning is the keep; ``EXIT_FALLBACK`` is the source fallback
    setup.sh acts on; anything else stops the update with a message the user sees.
    """
    monkeypatch.setattr(ILP, "detect_host", lambda *a, **k: host)
    try:
        ILP.install_prebuilt(root, "latest", "unslothai/llama.cpp", "")
        return "python-kept"
    except SystemExit as exc:
        code = exc.code
    if code != ILP.EXIT_FALLBACK:
        # Exit 1/3/5 all reach setup_fail, so the update stops and says why. The tree stays
        # broken, but the user has an error to act on, which is not the loop being hunted.
        return f"aborted-exit{code}"
    if not shell_stage:
        install_fresh_source_build(root, host)
        return "source-rebuilt"
    # setup.sh's source-build stage keeps a reusable local build instead of rebuilding. The
    # executable test used to be the whole condition, which kept every tree that had lost only
    # a library; reusable_existing_install is the gate the shell now asks first. Defaults
    # assumed: no UNSLOTH_LLAMA_FORCE_COMPILE, no UNSLOTH_LLAMA_PR.
    #
    # Both shells, because both have a reuse shortcut. This comment used to say setup.ps1
    # had none and that a Windows update reaching the source stage always builds; that was
    # wrong, and the mistake is what let the Windows half of this loop survive (Codex
    # 3963478816). setup.ps1's shortcut is an elseif on Test-PathQuiet $LlamaServerBin, so
    # on Windows the entrypoint test is llama-server.exe under build/bin/Release, and the
    # gate it now asks first is Test-LlamaTreeStillHealthy, which is the same
    # --check-existing-install call the shell makes.
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
        # The keep is not a no-op: it rewrites the marker, which is an input to the next
        # probe, so a keep could hand the probe a tree it then rejects.
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


# ---------------------------------------------------------------------------
# The cycle


MAX_CYCLES = 6


def run_cycle(
    root: Path,
    host,
    repair,
    *,
    max_cycles: int = MAX_CYCLES,
):
    """Iterate ``probe -> repair`` and classify how it ends.

    ``converged``  the probe stopped rejecting: healthy, or nothing installed.
    ``aborted``    the repair refused and told the user why. Terminating, not silent.
    ``loop``       the probe rejects and the repair left the tree identical. The bug.
    ``diverged``   still rejecting after ``max_cycles`` repairs that each changed something.
    """
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
    """Every way the shipped code says a tree can rot, plus two damages at once.

    A ``str`` removes that file, a ``tuple`` removes several, and the ``@`` constants name a
    structural loss instead.
    """
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
        # Two at once, since quarantine takes whole signatures rather than one file.
        ("remove-server-and-library", (server, libraries[0])),
        ("remove-first-and-last-library", (libraries[0], libraries[-1])),
        ("remove-runtime-dir", "@runtime-dir"),
        ("remove-tree", "@tree"),
        ("remove-marker", "@marker"),
        ("corrupt-marker", "@corrupt-marker"),
        # A corrupt marker with an incomplete payload: does the damage under it still get
        # seen.
        ("corrupt-marker-and-library", "@corrupt-marker+library"),
    ]
    return modes


# The damages that already decide the marker's fate, so pairing them with "and the marker is
# corrupt too" is a no-op or a different damage under the same name.
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
    """A complete tree of this cell, then ``damage`` applied to it.

    ``corrupt`` also replaces the marker with bytes that do not parse, the axis worth crossing
    with every other damage rather than testing on its own.
    """
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


# ---------------------------------------------------------------------------
# The online model


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
    """``sync_marker_selection`` runs on every keep and the marker it writes is the next
    probe's input. ``runtime_asset`` is the dangerous key: it turns the Windows cudart trio
    from optional into required, so stamping it onto a pair-less tree would reject on the next
    launch. Cycled from the already-healthy tree, the state a keep is reached from."""
    del shape  # the keep path is only reachable from the marker the current plan describes
    root = tmp_path / "healthy"
    install_fresh_prebuilt(root, host, backend)
    assert probe(root, host) == (True, ""), cell
    for _ in range(3):
        assert online_repair(root, host, backend) == "kept", cell
        assert probe(root, host) == (True, ""), f"{cell}: a keep made the tree unhealthy"


# ---------------------------------------------------------------------------
# The offline model, install_prebuilt for real


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_the_offline_python_repair_never_keeps_a_tree_the_probe_rejects(
    tmp_path, offline, cell, host, backend, shape
):
    """The branch the loop would hide in, driven through the real ``install_prebuilt``.

    Unreachable listing, so the run reaches the ``PrebuiltFallback`` handler and its
    ``_existing_install_runs`` gate. ``python-kept`` on a rejected tree is the loop: nothing
    downstream runs, so the next launch rejects it again. Everything else hands the tree to
    the shell, which is the next test.
    """
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
    """Offline, with the shell's rebuild-skip shortcut out of the model.

    The cycle as intended: the probe rejects, no release is reachable, the keep path refuses
    the tree, and setup.sh source builds over it. The swap is markerless, so the next probe
    reports "nothing installed" and stops. A second repair would mean the source build
    produced a tree the probe rejects.
    """
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
    """Codex 3963478816, P1. The model above is only worth as much as its fidelity, and it
    was wrong about this: it recorded that setup.ps1 had no reuse step, when it has an
    ``elseif`` on ``Test-PathQuiet $LlamaServerBin`` that reports "already built". So the
    Windows half of the loop survived the fix that closed the POSIX half. A Windows repair
    that fell through to the source stage kept a tree with a quarantined DLL, returned it
    unchanged, and preflight offered the same repair on every launch.

    Read off both scripts, because the model cannot catch a shell losing its gate.
    """
    root = Path(__file__).resolve().parents[3]

    shell = (root / "studio" / "setup.sh").read_text(encoding = "utf-8")
    assert "--check-existing-install" in shell
    assert "_LLAMA_REUSE_EXISTING" in shell, "setup.sh lost its reuse gate"

    ps1 = (root / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    assert "function Test-LlamaTreeStillHealthy" in ps1, "setup.ps1 lost its reuse gate"
    assert (
        "--check-existing-install" in ps1
    ), "the PowerShell gate must ask install_llama_prebuilt, not reimplement healthy"
    # On the shortcut itself, not somewhere else in the file: an elseif that reaches
    # "already built" without it is the exact defect. The shortcut now reads one
    # predicate, so the gate has to be inside what computes it.
    plan = ps1[ps1.index("$CanReuseLlamaBuild = ") :]
    plan = plan[: plan.index("$WillBuildLlamaFromSource = ")]
    assert "Test-LlamaTreeStillHealthy" in plan, "the reuse shortcut skips the health gate again"
    shortcut = ps1[ps1.index("} elseif ($CanReuseLlamaBuild) {") :]
    assert "already built" in shortcut[: shortcut.index("} elseif", 1)]


def test_the_windows_build_plan_asks_the_same_question_as_its_reuse_shortcut():
    """Codex 3971674498, P2. ``$WillBuildLlamaFromSource`` gates the last-chance git install
    and ``Ensure-BuildToolsForLlamaSourceBuild``. With the health check read only by the
    shortcut, a tree the shortcut refused left that predicate false, so a prebuilt-only box
    reached the rebuild the refusal forces with no cmake and no Visual Studio toolchain.
    Both now read ``$CanReuseLlamaBuild``, and it is computed once, above the plan."""
    ps1 = (Path(__file__).resolve().parents[3] / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    assert ps1.index("$CanReuseLlamaBuild = ") < ps1.index("$WillBuildLlamaFromSource = ")
    assert (
        "$WillBuildLlamaFromSource = $NeedLlamaSourceBuild -and -not $CanReuseLlamaBuild" in ps1
    ), "the build plan must derive from the same predicate the shortcut reads"
    # Once, so the helper runs once and its "incomplete" line is not printed twice.
    assert ps1.count("Test-LlamaTreeStillHealthy $LlamaCppDir") == 1, ps1.count(
        "Test-LlamaTreeStillHealthy $LlamaCppDir"
    )


def test_the_offline_repair_terminates_with_the_shell_rebuild_skip_in_place(tmp_path, offline):
    """THE NON-TERMINATING CASE, stated as the assertion that should hold.

    Linux, CUDA, today's marker shape, one quarantined library. The probe says
    ``llama_runtime_payload_incomplete``; ``install_prebuilt`` cannot reach a release and exits
    ``EXIT_FALLBACK`` because ``_existing_install_runs`` refuses the tree, which is the static
    invariant working as designed; setup.sh then found two executable entrypoints, called it an
    existing source build, and skipped the rebuild, so every later launch got the same answer
    while the update reported success.
    """
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
    """The property the shell fix establishes, over every cell rather than one example.

    This used to pin the opposite: on setup.sh hosts every damage that left both ``build/bin``
    entrypoints executable was kept by the reuse shortcut, so the offline repair returned the
    tree unchanged. The shortcut now asks ``reusable_existing_install`` first, so a tree the
    prebuilt helper just refused is refused here too and the source build runs.

    Aborted stays allowed and is not a loop: a marker naming a concrete backend makes
    ``preserve_backend`` true, so the helper exits ``EXIT_BACKEND_UNAVAILABLE`` and setup.sh
    stops with an error the user can act on.
    """
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


# ---------------------------------------------------------------------------
# The cases the report has to keep apart from a loop


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
    """The gap answering ``None`` for an unparseable marker used to leave.

    An unparseable marker over a missing library is a real install with a real hole, and
    ``None`` reads to the desktop as "no managed runtime here": preflight Ready, no repair
    offered, and the failure resurfacing at model load. It is graded instead, and the keep path
    agrees the tree is broken, so the grading is not a loop either.
    """
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
    """No marker file means not installed and answers ``None``; a marker that will not parse is
    an install whose payload can still be graded. Collapsing them either way is a bug: one
    direction repairs a source build this path never owned, the other hides a quarantined
    library behind a marker the same crash corrupted."""
    host, backend = MF.LINUX, "cuda"
    shape = MF.shape_with_backend(MF.S12, backend)

    absent = build_damaged(tmp_path / "absent", host, backend, shape, "@marker")
    assert probe(absent, host) is None
    # confirm_install_tree requires the marker file, so the keep path refuses this tree
    # outright and there is nothing to loop on.
    assert ILP._existing_install_runs(absent, host) is False

    corrupt = build_damaged(tmp_path / "corrupt", host, backend, shape, "@corrupt-marker")
    assert probe(corrupt, host) == (True, "")
    assert ILP._existing_install_runs(corrupt, host) is True


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_grading_an_unparseable_marker_keeps_the_no_stricter_invariant(
    tmp_path, cell, host, backend, shape
):
    """The static half, re-proved for the trees an unparseable marker used to excuse.

    Every damage is crossed with "and the marker does not parse", the axis that moved when the
    probe stopped short-circuiting those trees to ``None``. A rejection must still be one
    ``_existing_install_runs`` shares, or grading them trades a missed detection for a loop.
    """
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
    """The dynamic half of the same question, under both repair models.

    An unparseable marker plus every damage, cycled to a fixed point. Online the reinstall
    replaces the marker; offline the source build removes it. Neither model gains a cycle from
    grading these trees. The shell rebuild-skip stage is left out, since it is not a marker
    question.
    """
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

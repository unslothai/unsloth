# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does ``probe -> repair -> probe`` terminate? The dynamic half of the launch preflight.

``test_installed_runtime_health_matrix`` proves a STATIC implication: no damaged tree is
rejected by ``installed_runtime_health`` and kept by ``_existing_install_runs``. That is
necessary and it is not sufficient. Termination is a property of the whole cycle, and a
static implication can hold while the cycle still fails to make progress, because the repair
is not ``_existing_install_runs``: it is ``unsloth studio update`` ->  ``studio/setup.sh`` /
``studio/setup.ps1`` -> ``install_llama_prebuilt.py``, and the decision to leave the tree
alone is taken in three different places along that chain.

So this file simulates the launch cycle as a state machine over real trees on disk:

  probe    installed_runtime_health(tree, host = host), the real function.
  repair   one of three models, each named after the branch of the real chain it stands for.
  state    the tree itself, fingerprinted so "the repair changed nothing" is observable.

The three repair models, and where each is modelled from in ``install_llama_prebuilt.py``:

  ONLINE       the release listing is reachable, so ``install_prebuilt`` reaches
               ``existing_install_matches_plan(install_dir, host, release_plans[0])`` at line
               8767 and either records the reused selection (``sync_marker_selection``, which
               REWRITES the marker the next probe reads) or downloads and activates a fresh
               bundle. Modelled by calling the real ``existing_install_matches_plan`` and the
               real ``sync_marker_selection`` against a plan built here.
  OFFLINE      the listing raised, so ``install_prebuilt`` reaches the ``PrebuiltFallback``
               handler at line 8901 whose last condition is ``_existing_install_runs`` at line
               8907, and KEEPS a complete tree rather than reinstalling. Not modelled at all:
               the real ``install_prebuilt`` is called with the listing broken, exactly as
               ``test_keep_install_backcompat_9979`` does it, and its exit code is read.
  OFFLINE+SH   the same, continued into the shell: exit 2 sets ``_NEED_LLAMA_SOURCE_BUILD``
               (setup.sh line 2901) and the source-build stage then SKIPS the rebuild when
               ``build/bin/llama-server`` and ``build/bin/llama-quantize`` are both executable
               (setup.sh line 2917). That guard is the one place in the chain where a tree the
               probe rejects is left byte for byte identical, and the cycle does not
               terminate. It is pinned below as a strict xfail rather than as a passing
               assertion, so fixing the shell turns this file red and the pin gets deleted.

The marker is an axis of its own rather than one more damage, because the probe reads it and
one repair (``sync_marker_selection``) writes it, so it is the one piece of state that could
carry a cycle. Every damage below is therefore also run with a marker that does not parse: the
probe grades those trees rather than short-circuiting them to "not installed", and grading
them must not reject anything ``_existing_install_runs`` would keep.

Fixtures, marker shapes and payload tables come from ``test_installed_runtime_health_matrix``
(which took them from ``test_keep_install_backcompat_9979``) rather than being restated, so
the state machine is driven over exactly the trees the static invariant was proved over.
WSL is not a fourth platform: ``platform.system()`` reports ``Linux`` there, ``HostInfo``
carries no WSL flag, and every path and payload decision below is taken off ``is_windows`` /
``is_macos``, so a WSL install is graded by the Linux rows.
"""

import hashlib
import os
import shutil
import sys
import urllib.error
from pathlib import Path

import pytest

# The fixture module execs studio/install_llama_prebuilt.py itself and caches it in
# sys.modules. Taking ILP from it rather than loading a second copy keeps the monkeypatched
# module and the probed module the same object.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_installed_runtime_health_matrix as MF  # noqa: E402

ILP = MF.ILP

# Three platforms, and the arm64 rows are dropped on purpose: the payload intersection is
# picked by the platform prefix, so an arm64 host walks the same rows as its x64 twin, and
# the cycle costs a real install_prebuilt call per step.
HOSTS = [
    ("linux", MF.LINUX),
    ("windows", MF.WINDOWS),
    ("macos", MF.MACOS_ARM64),
]
# "metal" is dropped from the health-matrix list for the same reason: it is not requestable,
# and a metal marker on Linux only re-tests the fall-open path the static file already pins.
BACKENDS = ["cpu", "cuda", "rocm", "vulkan"]
# A spread of shipped shapes rather than all thirteen: the ones that differ in what the
# deciders read (no release_tag, no fingerprint, no backend key, a legacy vulkan override, a
# rocm arch, today's shape, and a marker a real run produced).
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

# The install kind a fresh install would land on for a host and a backend. macOS publishes
# one universal Metal bundle, so its backend axis is a marker carried in from elsewhere
# rather than a choice, and the plan below is the Metal one whatever the marker says.
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

    Names, sizes and mode bits, plus the marker's bytes: enough that "the repair rewrote the
    marker" and "the repair replaced the tree" are both distinguishable from "the repair did
    nothing", which is the whole question this file asks.

    Directory sizes are excluded rather than hashed: on ext4 a directory's ``st_size`` grows
    with the entries it has ever held, so an identical tree rebuilt in a reused path hashes
    differently and a no-op repair would be indistinguishable from a real one by luck.
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

    Built off the shipped S12 shape so it stays a real shape rather than an invention, then
    stamped with the fingerprint ``existing_install_matches_choice`` recomputes, which is what
    makes the freshly installed tree a fixed point instead of something the next cycle
    reinstalls again.
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

    Markerless on purpose, and that is the load-bearing detail rather than a simplification:
    the swap is ``rm -rf "$LLAMA_CPP_DIR"`` then ``mv "$_BUILD_TMP"`` (setup.sh lines
    3454-3464), so ``UNSLOTH_PREBUILT_INFO.json`` cannot survive it and the next probe reads
    "nothing installed" rather than judging a source build by a prebuilt marker's payload
    table. A swap that preserved the marker would be a loop of its own.
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
    log muting is not cosmetic: the cycle runs install_prebuilt thousands of times and each
    call prints a paragraph.
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

    ``install_prebuilt`` returning is the keep at line 8893; ``EXIT_FALLBACK`` is the source
    fallback setup.sh acts on; anything else stops the update with a message the user sees.
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
    # setup.sh's source-build stage: _NEED_LLAMA_SOURCE_BUILD is true and a reusable local
    # source build is kept instead of rebuilt. The executable test alone used to be the whole
    # condition, which kept every tree that had lost only a library and made the offline
    # repair a no-op; reusable_existing_install is the gate the shell now asks first.
    # Defaults assumed: no UNSLOTH_LLAMA_FORCE_COMPILE, no UNSLOTH_LLAMA_PR.
    #
    # POSIX only, and that is the shell rather than this model: setup.ps1 has no counterpart
    # to the reuse step, so a Windows update that reaches the source stage always builds. The
    # paths checked are setup.sh's own, which is why they are build/bin and not the Windows
    # build/bin/Release the probe reads.
    if (
        not host.is_windows
        and ILP.reusable_existing_install(root, host)
        and all(
            os.access(root / "build" / "bin" / f"llama-{name}", os.X_OK)
            for name in ("server", "quantize")
        )
    ):
        return "shell-kept"
    install_fresh_source_build(root, host)
    return "source-rebuilt"


def online_repair(root: Path, host, backend: str) -> str:
    """The reachable-plan branch: keep and re-record the selection, or reinstall."""
    plan, _ = current_plan(host, backend)
    if ILP.existing_install_matches_plan(root, host, plan):
        # The keep is not a no-op: it rewrites the marker, and the marker is an input to the
        # next probe, so this is exactly where a keep could hand the probe a tree it then
        # rejects for a reason the keep created.
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

    A ``str`` removes that file from the runtime dir, a ``tuple`` removes several, and the
    four ``str`` constants name a structural loss instead.
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
        # Two damages at once, since quarantine takes whole signatures rather than one file:
        # a library plus an entrypoint, and the two ends of the library list together.
        ("remove-server-and-library", (server, libraries[0])),
        ("remove-first-and-last-library", (libraries[0], libraries[-1])),
        ("remove-runtime-dir", "@runtime-dir"),
        ("remove-tree", "@tree"),
        ("remove-marker", "@marker"),
        ("corrupt-marker", "@corrupt-marker"),
        # A corrupt marker with an incomplete payload: the marker alone is deliberately
        # reported as "nothing installed", so this asks whether the payload damage under it
        # is still seen, and whether the pair behaves when it is not.
        ("corrupt-marker-and-library", "@corrupt-marker+library"),
    ]
    return modes


# The damages that already decide the marker's fate, so pairing them with "and the marker is
# corrupt too" would either be a no-op or a different damage under the same name.
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

    ``corrupt`` additionally replaces the marker with bytes that do not parse, which is the
    axis the pending corrupt-marker change moves and therefore the one worth crossing with
    every other damage rather than testing on its own.
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
    """Reachable listing: one reinstall, and the tree the reinstall produced is a fixed point.

    A second cycle here would mean the installer writes a tree its own probe rejects, which is
    a permanent loop for every user rather than a conditional one.
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
    """``sync_marker_selection`` runs on every keep, and the marker it writes is the next
    probe's input. ``runtime_asset`` is the dangerous key: it turns the Windows cudart trio
    from optional into required, so a keep that stamped it onto a pair-less tree would reject
    on the very next launch. Asserted by cycling the already-healthy tree, which is the state
    a keep is reached from."""
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
    ``_existing_install_runs`` gate. ``python-kept`` on a tree the probe rejects is the loop:
    nothing downstream runs, so the tree is returned unchanged and the next launch rejects it
    again. Everything else hands the tree to the shell, which is the next test.
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
    """Offline, with the shell's rebuild-skip shortcut taken out of the model.

    This is the cycle as the code intends it: the probe rejects, the prebuilt update cannot
    reach a release, the tree is not one the keep path accepts, and setup.sh source builds
    over it. The swap is markerless, so the next probe reports "nothing installed" and stops.
    The bound is one repair, and a second would mean the source build produced a tree the
    probe rejects.
    """
    marker = MF.shape_with_backend(shape, backend)
    for label, damage in damage_modes(host, backend, marker):
        root = build_damaged(tmp_path / label, host, backend, marker, damage)
        outcome, cycles, trail = run_cycle(
            root,
            host,
            lambda tree: offline_repair(tree, host, offline, shell_stage = False),
        )
        assert outcome in {"converged", "aborted"}, (
            f"{cell}/{label}: {outcome} after {cycles} repairs ({trail})"
        )
        assert cycles <= 1, f"{cell}/{label}: took {cycles} repairs ({trail})"


def test_the_offline_repair_terminates_with_the_shell_rebuild_skip_in_place(tmp_path, offline):
    """THE NON-TERMINATING CASE, stated as the assertion that should hold.

    Linux, CUDA, today's marker shape, one quarantined library and nothing else wrong. The
    probe says ``llama_runtime_payload_incomplete``; ``install_prebuilt`` cannot reach a
    release and exits ``EXIT_FALLBACK`` because ``_existing_install_runs`` refuses the tree,
    which is the static invariant holding exactly as designed; setup.sh then finds two
    executable entrypoints, calls the tree an existing source build, and skips the rebuild.
    The tree is unchanged, so the launch that follows asks the same question and gets the same
    answer, forever, while the update reports success.

    Written as ``expected to converge`` rather than as ``expected to loop`` so that it is the
    fix, not the bug, that turns this file green.
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

    This test used to pin the opposite, because the opposite was true: on setup.sh hosts every
    damage that left both ``build/bin`` entrypoints executable was kept by the reuse shortcut,
    so the offline repair returned the tree unchanged and the next launch rejected it again.
    Windows never had the shortcut and always built. Now the shortcut asks
    ``reusable_existing_install`` first, so a tree the prebuilt helper just refused to keep is
    refused here too and the source build actually runs.

    Aborted stays allowed and is not a loop: a marker naming a concrete backend makes
    ``preserve_backend`` true, so the helper exits ``EXIT_BACKEND_UNAVAILABLE`` and setup.sh
    stops with an error the user can act on rather than reporting a repair that did nothing.
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
    """Antivirus that re-quarantines after each repair never converges, and must not be read
    as the bug above. The distinguisher is progress, not convergence: here every repair
    replaces the tree and the next probe is asking about a different install, whereas the
    shell shortcut returns the identical tree. Asserted as "the fingerprint changed every
    time", which is the property a real repair has and a no-op does not."""
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
    """The gap that answering ``None`` for an unparseable marker used to leave.

    A marker that does not parse and a library that is gone is a real install with a real
    hole, and ``None`` would be read by the desktop as "no managed runtime here": preflight
    Ready, repair never offered, and the failure resurfacing at model load as something
    unrelated, which is the case this preflight was added to remove. It is graded instead, and
    the keep path agrees the tree is broken, so the grading is not a loop either.
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
    """No marker file is genuinely not installed and answers ``None``; a marker file that will
    not parse is an install whose payload can still be graded. Both halves, because collapsing
    them either way is a bug: one direction offers a repair for a source build this path never
    owned, the other hides a quarantined library behind a marker the crash also corrupted."""
    host, backend = MF.LINUX, "cuda"
    shape = MF.shape_with_backend(MF.S12, backend)

    absent = build_damaged(tmp_path / "absent", host, backend, shape, "@marker")
    assert probe(absent, host) is None
    # confirm_install_tree requires the marker file, so the keep path refuses this tree
    # outright: there is nothing here for the probe to loop on either way.
    assert ILP._existing_install_runs(absent, host) is False

    corrupt = build_damaged(tmp_path / "corrupt", host, backend, shape, "@corrupt-marker")
    assert probe(corrupt, host) == (True, "")
    assert ILP._existing_install_runs(corrupt, host) is True


@pytest.mark.parametrize(("cell", "host", "backend", "shape"), CELLS, ids = CELL_IDS)
def test_grading_an_unparseable_marker_keeps_the_no_stricter_invariant(
    tmp_path, cell, host, backend, shape
):
    """The static half, re-proved for the trees an unparseable marker used to excuse.

    Every damage is crossed with "and the marker does not parse", which is the axis that moved
    when the probe stopped short-circuiting those trees to ``None``. A rejection there must
    still be one ``_existing_install_runs`` shares, or grading them trades a missed detection
    for a repair loop.
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
    replaces the marker, so the next probe reads a parseable one; offline the source build
    removes it entirely. Neither model gains a cycle from grading these trees, and the only
    trees that do not terminate are the shell rebuild-skip ones, which is not a marker
    question at all and is excluded here by leaving that stage out of the model.
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
            assert outcome in {"converged", "aborted"}, (
                f"{cell}/{model}/{label}: {outcome} after {cycles} repairs ({trail})"
            )
            assert cycles <= 1, f"{cell}/{model}/{label}: {cycles} repairs ({trail})"


def test_the_freshly_installed_tree_is_a_fixed_point_on_every_cell(tmp_path):
    """The base case the whole argument rests on: whatever the repair installs is accepted.

    If this failed for any cell, every cycle above would be infinite regardless of which
    branch the repair took, so it is asserted for itself rather than inferred.
    """
    for host_id, host in HOSTS:
        for backend in BACKENDS:
            prebuilt = tmp_path / f"{host_id}-{backend}-prebuilt"
            install_fresh_prebuilt(prebuilt, host, backend)
            assert probe(prebuilt, host) == (True, ""), f"{host_id}-{backend}"

            source = tmp_path / f"{host_id}-{backend}-source"
            install_fresh_source_build(source, host)
            assert probe(source, host) is None, f"{host_id}-{backend}"

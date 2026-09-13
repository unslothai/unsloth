# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The host half of the marker fast path, across every OS x accelerator cell (#10648).

``existing_install_current_without_plan`` answers "the install on disk is already what
this run would produce" without listing a release. ``backend_request`` is ``"auto"`` on
every automatic install and the release tag does not move when the hardware does, so the
only thing in that answer that can notice a hardware change is ``host_profile``: recorded
into the marker at install time and compared for whole-dict equality on every later run.

Without that comparison a box that gains a GPU keeps its CPU bundle, a box whose card or
driver is gone keeps a CUDA bundle that cannot load, and a box that changes vendor keeps
an unusable backend -- in each case until the fork happens to publish a new release. Each
of those is a named test below, per platform, one axis flipped at a time.

Two halves, and they matter equally. The rejection half is above; the acceptance half is
that every reachable cell must answer True for the UNCHANGED host, or the "rejections"
below would be proving nothing but a broken fixture. The markers here are therefore
written by the real ``write_prebuilt_metadata`` against the routed host, so their
``install_fingerprint`` reproduces; a hand-written marker fails the fingerprint guard and
would make every case pass for the wrong reason.

Platforms are simulated by constructing ``HostInfo`` (``os.name`` is never patched -- it
changes ``pathlib``), and the three probes ``host_profile`` reads that a ``HostInfo`` does
not carry (the CUDA runtime scan, the ROCm runtime version, the torch CUDA preference) are
pinned per test on the module object, so no test reads this machine's real hardware.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    # Same idiom as conftest.py: these test modules are not a package, so the directory
    # is what makes a sibling importable.
    sys.path.insert(0, str(TEST_DIR))

from _pr10648_helpers import llama_host, load_studio_module  # noqa: E402

# The install-tree writer the back-compat suite already maintains. Reused rather than
# re-implemented so a change to what a real tree contains reaches this file too.
from test_keep_install_backcompat_9979 import build_install  # noqa: E402

# A distinct sys.modules name: the sibling suites load this file under
# "studio_install_llama_prebuilt", and a shared module leaks monkeypatches across xdist tests.
ILP = load_studio_module(
    "studio_install_llama_prebuilt_pr10648_matrix", "install_llama_prebuilt.py"
)

HostInfo = ILP.HostInfo
host_profile = ILP.host_profile
existing_install_current_without_plan = ILP.existing_install_current_without_plan

# Every variable that can move routing or the fast path, cleared per test: CUDA_VISIBLE_DEVICES
# in particular is set on any GPU box, so the answer would depend on the calling shell.
_ENV_KEYS = (
    "UNSLOTH_PREBUILT_FULL_CHECK",
    "UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE",
    "UNSLOTH_ROCM_GFX_ARCH",
    "UNSLOTH_ROCM_GFX_REMEMBERED",
    "UNSLOTH_LLAMA_CPP_BACKEND",
    "UNSLOTH_FORCE_VULKAN",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "GGML_VK_VISIBLE_DEVICES",
)

RELEASE_TAG = "release-1"
UPSTREAM_TAG = "b9001"
PUBLISHED_REPO = "unslothai/llama.cpp"


def make_host(**overrides) -> HostInfo:
    return llama_host(HostInfo, **overrides)


# The matrix. Rows are operating systems, columns are accelerators.

# WSL reports itself as Linux and HostInfo carries no WSL flag, deliberately (see
# test_keep_install_backcompat_9979.test_wsl_is_treated_exactly_like_linux_by_the_keep_path).
# A row anyway, so "WSL is judged by the Linux tables" is stated rather than assumed.
ROWS = {
    "linux": dict(system = "Linux", machine = "x86_64"),
    "windows": dict(system = "Windows", machine = "AMD64"),
    "wsl": dict(system = "Linux", machine = "x86_64"),
    "macos-arm64": dict(system = "Darwin", machine = "arm64", macos_version = (15, 5)),
    "macos-x86_64": dict(system = "Darwin", machine = "x86_64", macos_version = (14, 6)),
}

# The rows that can carry a discrete accelerator, i.e. everything the hardware transitions
# below apply to. macOS is Metal or nothing.
GPU_ROWS = ("linux", "windows", "wsl")

NVIDIA_CUDA12 = dict(
    nvidia_smi = "nvidia-smi",
    has_physical_nvidia = True,
    has_usable_nvidia = True,
    driver_cuda_version = (12, 8),
    compute_caps = ["8.9"],
)
NVIDIA_CUDA13 = dict(
    nvidia_smi = "nvidia-smi",
    has_physical_nvidia = True,
    has_usable_nvidia = True,
    driver_cuda_version = (13, 0),
    compute_caps = ["8.9"],
)
# gfx1100 (Navi 31, discrete): _should_prefer_vulkan_for_amd_igpu routes the integrated archs
# to Vulkan, which would make the cell test the router instead of the profile.
AMD_ROCM = dict(has_rocm = True, rocm_gfx_target = "gfx1100", rocm_gfx_targets = ["gfx1100"])
AMD_NO_ROCM = dict(has_amd_gpu_without_rocm = True)
INTEL = dict(has_intel_gpu = True)

ACCELERATORS = {
    "nvidia-cuda12": NVIDIA_CUDA12,
    "nvidia-cuda13": NVIDIA_CUDA13,
    "amd-rocm": AMD_ROCM,
    "amd-no-rocm": AMD_NO_ROCM,
    "intel": INTEL,
    "cpu": {},
    "metal": {},
}

COLUMNS = tuple(ACCELERATORS)

# (install_kind, the payload_backend key build_install writes) per platform and column.
_KINDS = {
    "linux": {
        "nvidia-cuda12": ("linux-cuda", "cuda"),
        "nvidia-cuda13": ("linux-cuda", "cuda"),
        "amd-rocm": ("linux-rocm", "rocm"),
        "amd-no-rocm": ("linux-vulkan", "vulkan"),
        "intel": ("linux-vulkan", "vulkan"),
        "cpu": ("linux-cpu", None),
    },
    "windows": {
        "nvidia-cuda12": ("windows-cuda", "cuda"),
        "nvidia-cuda13": ("windows-cuda", "cuda"),
        "amd-rocm": ("windows-rocm", "rocm"),
        "amd-no-rocm": ("windows-vulkan", "vulkan"),
        "intel": ("windows-vulkan", "vulkan"),
        "cpu": ("windows-cpu", None),
    },
}

_MACOS_KIND = {"macos-arm64": "macos-arm64", "macos-x86_64": "macos-x64"}

# The CUDA runtimes the on-disk scan reports, per column. host_profile records them, and
# they are what moves when a box's CUDA toolkit does.
_CUDA_LINES = {"nvidia-cuda12": ("cuda12",), "nvidia-cuda13": ("cuda13",)}
# What _detect_host_rocm_version answers; upstream ROCm assets are chosen by it.
_ROCM_RUNTIME = (6, 2)


@dataclasses.dataclass(frozen = True)
class Cell:
    """One (OS, accelerator) box: the host, the bundle it would be given, and the probe
    answers ``host_profile`` reads that ``HostInfo`` does not carry."""

    cell_id: str
    row: str
    column: str
    host: HostInfo
    install_kind: str
    payload_backend: "str | None"
    runtime_line: "str | None"
    cuda_lines: "tuple[str, ...]"
    rocm_runtime: "tuple[int, int] | None"


def _skip_reason(row: str, column: str) -> "str | None":
    """Why an (OS, accelerator) pair does not exist, or None when it does."""
    is_macos = row.startswith("macos")
    if column == "metal" and not is_macos:
        return "Metal is the macOS backend; there is no Metal install kind off Darwin"
    if is_macos and column != "metal":
        return (
            "macOS publishes one universal Metal bundle and no CUDA, ROCm or Vulkan "
            "install kind, so this accelerator column has no cell on Darwin"
        )
    return None


def _cell(row: str, column: str) -> Cell:
    host = make_host(**ROWS[row], **ACCELERATORS[column])
    if row.startswith("macos"):
        install_kind, payload_backend = _MACOS_KIND[row], None
    else:
        install_kind, payload_backend = _KINDS["windows" if row == "windows" else "linux"][column]
    cuda_lines = _CUDA_LINES.get(column, ())
    return Cell(
        cell_id = f"{row}-{column}",
        row = row,
        column = column,
        host = host,
        install_kind = install_kind,
        payload_backend = payload_backend,
        runtime_line = cuda_lines[0] if cuda_lines else None,
        cuda_lines = cuda_lines,
        rocm_runtime = _ROCM_RUNTIME if column == "amd-rocm" else None,
    )


MATRIX = [(row, column) for row in ROWS for column in COLUMNS]
MATRIX_IDS = [f"{row}-{column}" for row, column in MATRIX]
REACHABLE = [(row, column) for row, column in MATRIX if _skip_reason(row, column) is None]
REACHABLE_IDS = [f"{row}-{column}" for row, column in REACHABLE]


class Probe:
    """The hardware answers this run gives, pinned on the loaded module.

    ``host_profile`` reads three things a ``HostInfo`` does not carry -- the CUDA runtime
    scan, the ROCm runtime version and (for the marker) the torch CUDA preference -- and
    all three would otherwise read this machine. ``set`` moves them together with the
    detected host, which is exactly what a hardware change does.
    """

    def __init__(self, monkeypatch):
        self.host: "HostInfo | None" = None
        self.cuda_lines: "tuple[str, ...]" = ()
        self.rocm_runtime: "tuple[int, int] | None" = None
        self.torch_line: "str | None" = None
        self.latest_release = RELEASE_TAG
        monkeypatch.setattr(ILP, "detect_host", lambda **_kwargs: self.host)
        monkeypatch.setattr(
            ILP, "detected_linux_runtime_lines", lambda: (list(self.cuda_lines), {})
        )
        monkeypatch.setattr(
            ILP, "detected_windows_runtime_lines", lambda: (list(self.cuda_lines), {})
        )
        monkeypatch.setattr(ILP, "_detect_host_rocm_version", lambda: self.rocm_runtime)
        # torch.cuda.is_available() is slow and beside the point here; the preference only has
        # to be STABLE, since a moved one is _runtime_preference_moved's subject.
        monkeypatch.setattr(
            ILP,
            "detect_torch_cuda_runtime_preference",
            lambda _host: SimpleNamespace(runtime_line = self.torch_line, selection_log = []),
        )
        monkeypatch.setattr(
            ILP, "_download_host_latest_release_tag", lambda _repo: self.latest_release
        )

    def set(
        self,
        host: HostInfo,
        *,
        cuda_lines: "tuple[str, ...]" = (),
        rocm_runtime: "tuple[int, int] | None" = None,
        torch_line: "str | None" = None,
    ) -> HostInfo:
        self.host = host
        self.cuda_lines = cuda_lines
        self.rocm_runtime = rocm_runtime
        self.torch_line = torch_line
        return host

    def set_from(
        self,
        cell: Cell,
        host: "HostInfo | None" = None,
        **overrides,
    ) -> HostInfo:
        """Point the probes at ``cell``'s answers, optionally for a different host."""
        answers = dict(
            cuda_lines = cell.cuda_lines,
            rocm_runtime = cell.rocm_runtime,
            torch_line = cell.runtime_line,
        )
        answers.update(overrides)
        return self.set(cell.host if host is None else host, **answers)


@pytest.fixture
def probe(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising = False)
    return Probe(monkeypatch)


def route_for(host: HostInfo, repo: str = PUBLISHED_REPO):
    """The routed host and repo, the same no-network routing the fast path re-derives.

    ``host`` is passed explicitly rather than left to the patched ``detect_host`` so a
    fixture can never route one box while the probes answer for another."""
    return ILP.route_backend_request(
        backend = "auto",
        published_repo = repo,
        published_release_tag = "",
        host = host,
    )


def install_cell(
    tmp_path: Path,
    probe: Probe,
    cell: Cell,
    *,
    name: str = "install",
) -> Path:
    """An install tree plus a marker genuinely written for ``cell``'s host.

    The marker goes through the real ``write_prebuilt_metadata`` against the ROUTED host,
    so its ``install_fingerprint``, ``runtime_files`` and ``host_profile`` are the ones a
    real install of this bundle on this box would carry. Anything hand-written here would
    fail the fingerprint guard and make a later "rejected" assertion meaningless.
    """
    probe.set_from(cell)
    route = route_for(cell.host)
    install_dir = build_install(
        tmp_path / name,
        host = route.host,
        marker = None,
        payload_backend = cell.payload_backend,
    )
    runtime_dir = ILP.install_runtime_dir(install_dir, route.host)
    # The fork bundles ship the DiffusionGemma visual server and the marker-only backfill
    # check looks for it; build_install writes it on Linux and Windows but not on macOS.
    visual_server = runtime_dir / (
        "llama-diffusion-gemma-visual-server" + (".exe" if route.host.is_windows else "")
    )
    if not visual_server.exists():
        visual_server.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    choice = ILP.AssetChoice(
        repo = route.published_repo or PUBLISHED_REPO,
        tag = RELEASE_TAG,
        name = f"llama-{UPSTREAM_TAG}-{cell.install_kind}.tar.gz",
        url = f"https://example.com/llama-{UPSTREAM_TAG}-{cell.install_kind}.tar.gz",
        source_label = "published",
        install_kind = cell.install_kind,
        expected_sha256 = "a" * 64,
        runtime_line = cell.runtime_line,
        coverage_class = "targeted" if cell.runtime_line else None,
        supported_sms = ["89"] if cell.runtime_line else None,
        bundle_profile = f"{cell.runtime_line}-newer" if cell.runtime_line else None,
        gfx_target = cell.host.rocm_gfx_target,
        mapped_targets = list(cell.host.rocm_gfx_targets or []),
    )
    checksums = _checksums(choice, repo = route.published_repo or PUBLISHED_REPO)
    ILP.write_prebuilt_metadata(
        install_dir,
        host = route.host,
        requested_tag = "latest",
        llama_tag = UPSTREAM_TAG,
        release_tag = RELEASE_TAG,
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        backend_request = "auto",
    )
    return install_dir


def _checksums(choice, *, repo: str):
    logical = ILP.source_archive_logical_name(UPSTREAM_TAG)
    artifacts = {
        logical: ILP.ApprovedArtifactHash(
            asset_name = logical,
            sha256 = "b" * 64,
            repo = "ggml-org/llama.cpp",
            kind = "upstream-source",
        ),
        choice.name: ILP.ApprovedArtifactHash(
            asset_name = choice.name,
            sha256 = choice.expected_sha256,
            repo = repo,
            kind = "prebuilt",
        ),
    }
    return ILP.ApprovedReleaseChecksums(
        repo = repo,
        release_tag = RELEASE_TAG,
        upstream_tag = UPSTREAM_TAG,
        source_commit = "deadbeef",
        artifacts = artifacts,
    )


def check(install_dir: Path, **overrides) -> bool:
    kwargs = dict(
        llama_tag = "latest",
        published_repo = PUBLISHED_REPO,
        published_release_tag = "",
        backend_request = "auto",
        force_cpu = False,
    )
    kwargs.update(overrides)
    return existing_install_current_without_plan(install_dir, **kwargs)


def marker_of(install_dir: Path) -> dict:
    return json.loads((install_dir / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8"))


def moved(tmp_path: Path, probe: Probe, cell: Cell, after: HostInfo, **answers) -> Path:
    """Install on ``cell``'s box, prove it is kept, then move the hardware under it."""
    install_dir = install_cell(tmp_path, probe, cell)
    assert check(install_dir) is True, f"fixture is not current on its own host: {cell.cell_id}"
    probe.set_from(cell, after, **answers)
    return install_dir


def assert_reinstall_forced(monkeypatch, install_dir: Path) -> None:
    """The fast path refuses, and the host profile is the guard that refused.

    A bare ``is False`` would also be satisfied by a broken tree, a moved release tag or a
    marker that does not add up -- so every hardware transition proves the attribution as
    well: with the profile comparison neutralised, and nothing else touched, the same call
    answers True again. That is the install the user would have been left holding.

    Patches last, so no assertion after this one is trustworthy; each caller ends here.
    """
    assert check(install_dir) is False
    recorded = marker_of(install_dir)["host_profile"]
    monkeypatch.setattr(ILP, "host_profile", lambda _host: recorded)
    assert check(install_dir) is True, (
        "the install was refused by some other guard, so this case does not show the "
        "host profile closing anything"
    )


# (A) the matrix itself: which cells exist.


@pytest.mark.parametrize(("row", "column"), MATRIX, ids = MATRIX_IDS)
def test_every_cell_of_the_matrix_is_reachable_or_explicitly_impossible(row, column):
    """The user population this fast path runs for: every OS crossed with every
    accelerator. A cell that does not exist is skipped with the reason, so a column
    quietly losing a platform shows up as a skip rather than as absence."""
    reason = _skip_reason(row, column)
    if reason is not None:
        pytest.skip(reason)
    cell = _cell(row, column)
    assert cell.install_kind in ILP.INSTALL_KIND_BACKENDS, cell.cell_id
    prefix = "windows-" if cell.host.is_windows else "macos-" if cell.host.is_macos else "linux-"
    assert cell.install_kind.startswith(prefix), cell.cell_id


# (B) the acceptance half: an unchanged box keeps its install, in every reachable cell.


@pytest.mark.parametrize(("row", "column"), REACHABLE, ids = REACHABLE_IDS)
def test_an_unchanged_box_keeps_its_install_in_every_cell(tmp_path, probe, row, column):
    """Nothing happened to the user's machine since the install: no download, no listing,
    no re-validation. This is the half the fast path exists for, and every rejection test
    below is only meaningful because this one passes with the same fixtures."""
    cell = _cell(row, column)
    install_dir = install_cell(tmp_path, probe, cell)
    assert check(install_dir) is True, cell.cell_id
    # And the recorded profile really is this box, not a default that would match anything.
    assert marker_of(install_dir)["host_profile"] == host_profile(route_for(cell.host).host)


@pytest.mark.parametrize(("row", "column"), REACHABLE, ids = REACHABLE_IDS)
def test_a_second_update_on_the_same_box_is_still_current(tmp_path, probe, row, column):
    """Updates are not one-shot: the user runs setup again next week with nothing changed.
    A profile that failed to compare equal to itself would be a silent permanent
    regression -- every update back to the full path, forever, with no error to report."""
    cell = _cell(row, column)
    install_dir = install_cell(tmp_path, probe, cell)
    assert check(install_dir) is True, cell.cell_id
    assert check(install_dir) is True, cell.cell_id
    assert check(install_dir) is True, cell.cell_id


# (C) the rejection half: one axis flipped at a time, each its own user scenario.


def _cpu_cell(row: str) -> Cell:
    return _cell(row, "cpu")


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_gpu_added_to_a_cpu_only_box_is_not_current(tmp_path, probe, monkeypatch, row):
    """The user drops an NVIDIA card into a CPU-only machine and re-runs setup. The
    request recorded is "auto" and the release tag has not moved, so only the host
    profile can notice; without it the box keeps its CPU bundle and stays slow."""
    cell = _cpu_cell(row)
    after = make_host(**ROWS[row], **NVIDIA_CUDA12)
    install_dir = moved(tmp_path, probe, cell, after, cuda_lines = ("cuda12",), torch_line = None)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_gpu_removed_since_the_install_is_not_current(tmp_path, probe, monkeypatch, row):
    """The card was pulled, or the driver uninstalled. A CUDA bundle on a box with no
    CUDA does not load at all, so keeping it is worse than a reinstall."""
    cell = _cell(row, "nvidia-cuda12")
    after = make_host(**ROWS[row])
    install_dir = moved(tmp_path, probe, cell, after, cuda_lines = (), torch_line = None)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_cuda_12_to_cuda_13_move_is_not_current(tmp_path, probe, monkeypatch, row):
    """The user upgrades the NVIDIA driver and the CUDA toolkit. The card is the same and
    the release tag has not moved; what moved is the driver's CUDA version and the CUDA
    runtimes on disk, which are what orders the CUDA bundles."""
    cell = _cell(row, "nvidia-cuda12")
    after = make_host(**ROWS[row], **NVIDIA_CUDA13)
    install_dir = moved(
        tmp_path, probe, cell, after, cuda_lines = ("cuda13",), torch_line = cell.runtime_line
    )
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_only_the_cuda_runtimes_on_disk_moving_is_not_current(tmp_path, probe, monkeypatch, row):
    """A CUDA 13 toolkit installed beside the existing one, driver untouched. The HostInfo
    is byte-identical; cuda_runtime_lines is the only field that can carry it."""
    cell = _cell(row, "nvidia-cuda12")
    install_dir = moved(
        tmp_path,
        probe,
        cell,
        cell.host,
        cuda_lines = ("cuda12", "cuda13"),
        torch_line = cell.runtime_line,
    )
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_swapping_an_nvidia_card_for_an_amd_one_is_not_current(tmp_path, probe, monkeypatch, row):
    """The user replaces the NVIDIA card with a Radeon and installs ROCm. A CUDA bundle
    on an AMD box is an unusable backend, not a slow one."""
    cell = _cell(row, "nvidia-cuda12")
    after = make_host(**ROWS[row], **AMD_ROCM)
    install_dir = moved(tmp_path, probe, cell, after, cuda_lines = (), rocm_runtime = _ROCM_RUNTIME)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_swapping_an_amd_card_for_an_nvidia_one_is_not_current(tmp_path, probe, monkeypatch, row):
    """The mirror image: a ROCm/HIP bundle left on a box that now has an NVIDIA card."""
    cell = _cell(row, "amd-rocm")
    after = make_host(**ROWS[row], **NVIDIA_CUDA12)
    install_dir = moved(tmp_path, probe, cell, after, cuda_lines = ("cuda12",), rocm_runtime = None)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_rocm_runtime_version_bump_is_not_current(tmp_path, probe, monkeypatch, row):
    """The user upgrades ROCm 6.2 to 6.4 with the same card. Upstream ROCm assets are
    chosen by the runtime's major.minor, which no GPU field carries -- so the bundle this
    run would pick has moved even though every other hardware fact is identical."""
    cell = _cell(row, "amd-rocm")
    install_dir = moved(tmp_path, probe, cell, cell.host, rocm_runtime = (6, 4))
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_rocm_gfx_target_change_is_not_current(tmp_path, probe, monkeypatch, row):
    """A different AMD card: gfx1100 out, gfx1030 in. A HIP bundle carries kernels for the
    archs it was built for and nothing else, so the install starts and then cannot run a
    model."""
    cell = _cell(row, "amd-rocm")
    after = make_host(
        **ROWS[row],
        has_rocm = True,
        rocm_gfx_target = "gfx1030",
        rocm_gfx_targets = ["gfx1030"],
    )
    install_dir = moved(tmp_path, probe, cell, after, rocm_runtime = _ROCM_RUNTIME)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_second_amd_card_the_bundle_has_no_kernels_for_is_not_current(
    tmp_path, probe, monkeypatch, row
):
    """The active gfx is unchanged, but a second Radeon appeared. rocm_gfx_targets is the
    inventory the runtime device gate reads, so it has to be part of the comparison."""
    cell = _cell(row, "amd-rocm")
    after = make_host(
        **ROWS[row],
        has_rocm = True,
        rocm_gfx_target = "gfx1100",
        rocm_gfx_targets = ["gfx1100", "gfx1030"],
    )
    install_dir = moved(tmp_path, probe, cell, after, rocm_runtime = _ROCM_RUNTIME)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_an_intel_gpu_that_appeared_is_not_current(tmp_path, probe, monkeypatch, row):
    """The user enables the integrated Intel GPU (a BIOS change, or a new machine restored
    from the same home directory). An Intel GPU with no NVIDIA and no ROCm is the whole
    purpose of the has_intel_gpu probe: it routes the install to Vulkan."""
    cell = _cpu_cell(row)
    after = make_host(**ROWS[row], **INTEL)
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_an_intel_gpu_that_disappeared_is_not_current(tmp_path, probe, monkeypatch, row):
    """The mirror: the Vulkan bundle chosen for an Intel GPU, on a box that no longer
    reports one."""
    cell = _cell(row, "intel")
    after = make_host(**ROWS[row])
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_physical_nvidia_disappearing_behind_an_intel_gpu_is_not_current(
    tmp_path, probe, monkeypatch, row
):
    """The Vulkan-route gate. The box has an Intel GPU and an NVIDIA card hidden by
    CUDA_VISIBLE_DEVICES, so has_usable_nvidia is already False and the Intel auto-route is
    held off by has_physical_nvidia alone. Pull the card and the same run routes to Vulkan
    -- a transition invisible to every other field in the profile."""
    cell = dataclasses.replace(
        _cpu_cell(row),
        host = make_host(
            **ROWS[row],
            **INTEL,
            has_physical_nvidia = True,
            has_usable_nvidia = False,
            nvidia_smi = "nvidia-smi",
            visible_cuda_devices = "",
        ),
    )
    probe.set_from(cell)
    before_profile = host_profile(route_for(cell.host).host)
    assert before_profile["has_physical_nvidia"] is True
    assert before_profile["has_usable_nvidia"] is False
    assert before_profile["has_intel_gpu"] is True
    after = make_host(**ROWS[row], **INTEL)
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_home_directory_carried_to_another_cpu_architecture_is_not_current(
    tmp_path, probe, monkeypatch, row
):
    """A home directory rsynced from an x86_64 box onto an arm64 one (or opened under
    Rosetta). Every GPU field is equal and the release tag is the same; only machine
    differs, and the bundle on disk cannot be executed at all."""
    cell = _cpu_cell(row)
    arm = "ARM64" if row == "windows" else "aarch64"
    after = make_host(**{**ROWS[row], "machine": arm})
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_new_card_with_a_different_sm_is_not_current(tmp_path, probe, monkeypatch, row):
    """An RTX 4090 (sm_89) swapped for a Blackwell part (sm_120), same driver. A bundle
    carries kernels for a fixed supported_sms set, so an install outside it starts and
    then fails on the first model load."""
    cell = _cell(row, "nvidia-cuda12")
    after = make_host(**ROWS[row], **{**NVIDIA_CUDA12, "compute_caps": ["12.0"]})
    install_dir = moved(
        tmp_path, probe, cell, after, cuda_lines = cell.cuda_lines, torch_line = cell.runtime_line
    )
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", GPU_ROWS)
def test_the_same_cards_reported_in_another_order_is_not_a_hardware_change(tmp_path, probe, row):
    """The other direction, and the reason compute_caps is sorted and de-duplicated: two
    identical cards enumerated in either order, with whitespace, are the same box. Calling
    that a change would send every multi-GPU user down the full path on every update."""
    cell = _cell(row, "nvidia-cuda12")
    after = make_host(**ROWS[row], **{**NVIDIA_CUDA12, "compute_caps": [" 8.9 ", "8.9"]})
    install_dir = moved(
        tmp_path, probe, cell, after, cuda_lines = cell.cuda_lines, torch_line = cell.runtime_line
    )
    assert check(install_dir) is True


MACOS_ROWS = ("macos-arm64", "macos-x86_64")


@pytest.mark.parametrize("row", MACOS_ROWS)
def test_a_macos_upgrade_across_the_min_os_floor_is_not_current(tmp_path, probe, monkeypatch, row):
    """The user upgrades macOS past the floor the newest bundle needs. The bundle the
    planner would pick changes with the OS version (walk-back), so the recorded profile
    has to carry macos_version or a Mac would keep a walked-back release forever."""
    cell = _cell(row, "metal")
    before = cell.host.macos_version
    assert before is not None and before < ILP._PINNED_MACOS_LATEST_FLOOR
    after = make_host(**{**ROWS[row], "macos_version": ILP._PINNED_MACOS_LATEST_FLOOR})
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


@pytest.mark.parametrize("row", MACOS_ROWS)
def test_a_macos_point_release_is_not_current(tmp_path, probe, monkeypatch, row):
    """Not only the floor: any macOS version move re-decides the bundle, because the
    minimum-OS preflight is what accepted this one."""
    cell = _cell(row, "metal")
    major, minor = cell.host.macos_version
    after = make_host(**{**ROWS[row], "macos_version": (major, minor + 1)})
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


def test_a_mac_home_directory_restored_onto_apple_silicon_is_not_current(
    tmp_path, probe, monkeypatch
):
    """The Intel Mac's home directory restored onto an Apple Silicon Mac, or the same one
    opened under Rosetta. macos_version and every GPU field are equal; machine is not."""
    cell = _cell("macos-x86_64", "metal")
    after = make_host(**{**ROWS["macos-x86_64"], "machine": "arm64"})
    install_dir = moved(tmp_path, probe, cell, after)
    assert_reinstall_forced(monkeypatch, install_dir)


# (D) the guard is load-bearing, not incidental.


def _assert_everything_but_the_host_still_matches(install_dir: Path, after: HostInfo) -> dict:
    """Everything the fast path checks APART from the host profile still holds.

    Without this, a "rejected" assertion could be passing because the tree was broken or
    the release moved, and the host guard could be removed with the tests still green.
    """
    marker = marker_of(install_dir)
    route = route_for(after)
    assert marker["release_tag"] == RELEASE_TAG
    assert marker["published_repo"] == (route.published_repo or ILP.DEFAULT_PUBLISHED_REPO)
    assert marker["backend_request"] == "auto"
    assert marker["prebuilt_fallback_used"] is False
    # The marker is intact and self-consistent: the fingerprint recomputes from its own
    # fields, so nothing has been hand-edited.
    assert ILP._marker_install_fingerprint(marker) == marker["install_fingerprint"]
    # The release this run would ask for is the release that is installed.
    assert (
        ILP._expected_release_tag_without_plan(
            marker, "latest", PUBLISHED_REPO, "", host = route.host
        )
        == RELEASE_TAG
    )
    # The tree is intact, executable, and the recorded bytes are still on disk.
    assert ILP._marker_backend_fits_host(marker, route.host) is True
    assert ILP._install_tree_is_usable(install_dir, route.host) is True
    assert ILP._kept_install_payload_is_healthy(install_dir, route.host) is True
    assert ILP._runtime_files_match(install_dir, route.host, marker) is True
    assert ILP._diffusion_visual_server_missing_for_marker(install_dir, route.host, marker) is False
    # And torch has not moved either, so the CUDA-preference guard is not the one talking.
    assert ILP._runtime_preference_moved(marker, route.host) is False
    # The one thing that HAS moved.
    assert marker["host_profile"] != host_profile(route.host)
    return marker


@pytest.mark.parametrize("row", GPU_ROWS)
def test_adding_a_gpu_is_rejected_by_the_host_profile_and_nothing_else(
    tmp_path, probe, monkeypatch, row
):
    """The CPU-to-NVIDIA case again, but proving WHICH guard rejected it. Same release,
    same fingerprint, intact tree -- and with the profile comparison neutralised the fast
    path keeps the CPU bundle, which is the bug the guard exists to prevent."""
    cell = _cpu_cell(row)
    after = make_host(**ROWS[row], **NVIDIA_CUDA12)
    install_dir = moved(tmp_path, probe, cell, after, cuda_lines = ("cuda12",), torch_line = None)
    assert check(install_dir) is False
    marker = _assert_everything_but_the_host_still_matches(install_dir, after)
    # Neutralise only the profile comparison: everything else is untouched.
    monkeypatch.setattr(ILP, "host_profile", lambda _host: marker["host_profile"])
    assert (
        check(install_dir) is True
    ), "without the host_profile guard this box keeps its CPU bundle after gaining a GPU"


@pytest.mark.parametrize("row", GPU_ROWS)
def test_a_cuda_line_move_is_rejected_by_the_host_profile_and_nothing_else(
    tmp_path, probe, monkeypatch, row
):
    """The cuda12-to-cuda13 case, same proof. The release did not move, the marker is
    whole and the bundle is intact, so the profile is the only thing that can say the
    installed CUDA bundle is no longer the one this run would pick."""
    cell = _cell(row, "nvidia-cuda12")
    after = make_host(**ROWS[row], **NVIDIA_CUDA13)
    install_dir = moved(
        tmp_path, probe, cell, after, cuda_lines = ("cuda13",), torch_line = cell.runtime_line
    )
    assert check(install_dir) is False
    marker = _assert_everything_but_the_host_still_matches(install_dir, after)
    monkeypatch.setattr(ILP, "host_profile", lambda _host: marker["host_profile"])
    assert (
        check(install_dir) is True
    ), "without the host_profile guard this box keeps its cuda12 bundle after moving to cuda13"


# (E) the JSON round trip. A profile that never equals itself takes the full path forever.


def _tuples_in(value) -> bool:
    if isinstance(value, tuple):
        return True
    if isinstance(value, dict):
        return any(_tuples_in(item) for item in value.values())
    if isinstance(value, list):
        return any(_tuples_in(item) for item in value)
    return False


@pytest.mark.parametrize(("row", "column"), REACHABLE, ids = REACHABLE_IDS)
def test_a_profile_written_to_a_marker_reads_back_equal_to_a_fresh_one(
    tmp_path, probe, row, column
):
    """Written by one run, compared by the next through json.loads. Whole-dict equality is
    what the fast path asks, so a single field that does not survive serialisation would
    send this cell down the full path on every update forever -- with no error, just a
    permanently slow setup."""
    cell = _cell(row, column)
    install_dir = install_cell(tmp_path, probe, cell)
    recorded = marker_of(install_dir)["host_profile"]
    fresh = host_profile(route_for(cell.host).host)
    assert recorded == fresh, cell.cell_id
    # The hazard named in the docstring at host_profile: a tuple reads back as a list.
    assert not _tuples_in(fresh), f"{cell.cell_id}: a tuple would never read back equal"
    assert json.loads(json.dumps(fresh)) == fresh, cell.cell_id


@pytest.mark.parametrize(("row", "column"), REACHABLE, ids = REACHABLE_IDS)
def test_every_profile_field_is_present_in_every_cell(tmp_path, probe, row, column):
    """Absent and null are different answers to the comparison. Every field the routing
    branches on must be written in every cell, or two different boxes could record the
    same profile."""
    cell = _cell(row, column)
    probe.set_from(cell)
    profile = host_profile(route_for(cell.host).host)
    assert set(profile) == {
        "machine",
        "has_usable_nvidia",
        "has_physical_nvidia",
        "driver_cuda_version",
        "compute_caps",
        "has_rocm",
        "rocm_runtime",
        "rocm_gfx_target",
        "rocm_gfx_targets",
        "has_intel_gpu",
        "has_amd_gpu_without_rocm",
        "macos_version",
        "cuda_runtime_lines",
    }, cell.cell_id


def test_the_tuple_valued_fields_survive_the_marker_round_trip(tmp_path, probe):
    """driver_cuda_version, macos_version, compute_caps, rocm_gfx_targets and
    cuda_runtime_lines are the fields built from tuples or sets. Each is checked on a cell
    that actually populates it, against the value read back out of the marker file."""
    cuda = _cell("linux", "nvidia-cuda12")
    cuda_marker = marker_of(install_cell(tmp_path, probe, cuda, name = "cuda"))["host_profile"]
    assert cuda_marker["driver_cuda_version"] == [12, 8]
    assert cuda_marker["compute_caps"] == ["8.9"]
    assert cuda_marker["cuda_runtime_lines"] == ["cuda12"]

    rocm = _cell("linux", "amd-rocm")
    rocm_marker = marker_of(install_cell(tmp_path, probe, rocm, name = "rocm"))["host_profile"]
    assert rocm_marker["rocm_runtime"] == [6, 2]
    assert rocm_marker["rocm_gfx_target"] == "gfx1100"
    assert rocm_marker["rocm_gfx_targets"] == ["gfx1100"]

    mac = _cell("macos-arm64", "metal")
    mac_marker = marker_of(install_cell(tmp_path, probe, mac, name = "mac"))["host_profile"]
    assert mac_marker["macos_version"] == [15, 5]
    assert mac_marker["machine"] == "arm64"

    # The round trip is what turns each of those tuples into a list; a profile computed
    # fresh from the same host must still compare equal to the parsed one.
    for cell, parsed in ((cuda, cuda_marker), (rocm, rocm_marker), (mac, mac_marker)):
        probe.set_from(cell)
        assert parsed == host_profile(route_for(cell.host).host), cell.cell_id


def test_a_marker_with_no_host_profile_cannot_answer_and_takes_the_full_path(tmp_path, probe):
    """Every marker written before #10648 has no profile. "Cannot tell" must not read as
    "unchanged", or the very installs this guard was added for would skip it."""
    cell = _cell("linux", "nvidia-cuda12")
    install_dir = install_cell(tmp_path, probe, cell)
    assert check(install_dir) is True
    marker = marker_of(install_dir)
    marker.pop("host_profile")
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(marker, indent = 2) + "\n", encoding = "utf-8"
    )
    assert check(install_dir) is False


def test_two_different_boxes_never_record_the_same_profile(probe):
    """The comparison is only worth running if the cells it separates are distinguishable.
    Every reachable cell's profile must differ from every other's, or a hardware change
    between those two boxes would be invisible."""
    owner_of_profile: dict[str, str] = {}
    profile_of_cell: dict[str, str] = {}
    for row, column in REACHABLE:
        cell = _cell(row, column)
        probe.set_from(cell)
        key = json.dumps(host_profile(route_for(cell.host).host), sort_keys = True)
        profile_of_cell[cell.cell_id] = key
        # Rows that are the same box by construction: WSL reports itself as Linux, and
        # HostInfo has no WSL flag, so linux-<column> and wsl-<column> are one profile.
        if row == "wsl":
            assert key == profile_of_cell[f"linux-{column}"], "WSL must read exactly as Linux"
            continue
        clash = owner_of_profile.get(key)
        assert clash is None, f"{cell.cell_id} records the same profile as {clash}"
        owner_of_profile[key] = cell.cell_id

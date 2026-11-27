# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Old markers must take the full path exactly once; new keys must not make old readers refuse."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    # Prepend import mode may not put this dir on sys.path before the module body runs.
    sys.path.insert(0, str(_TEST_DIR))

from _pr10648_helpers import PACKAGE_ROOT, git, llama_host  # noqa: E402
from _pr10648_helpers import load_studio_module as _load  # noqa: E402

import test_keep_install_backcompat_9979 as CORPUS  # noqa: E402

ILP = _load("studio_install_llama_prebuilt_pr10648_legacy", "install_llama_prebuilt.py")
WSP = _load("studio_install_whisper_prebuilt_pr10648_legacy", "install_whisper_prebuilt.py")
NDP = _load("studio_install_node_prebuilt_pr10648_legacy", "install_node_prebuilt.py")
# The core instance whisper actually calls into, not a second copy of it.
CORE = WSP.core


LEGACY_TAGS = (
    "v0.1.800-beta",
    "v0.1.802-beta",
    "v0.1.804-beta",
    "v0.1.806-beta",
    "v0.1.808-beta",
)

_LEGACY_MODULES = (
    "install_llama_prebuilt.py",
    "install_whisper_prebuilt.py",
    "install_node_prebuilt.py",
    "prebuilt_core.py",
)

_SENTINEL = "@@PR10648-RESULT@@"

# Runs inside the legacy subprocess; introspects the tag's own dataclasses.
_LEGACY_DRIVER = r'''
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

SENTINEL = "@@PR10648-RESULT@@"


def fill(cls, wanted):
    """Build *cls* from the subset of *wanted* this tag's dataclass actually declares."""
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in wanted:
            kwargs[field.name] = wanted[field.name]
        elif (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ):
            kwargs[field.name] = None
    return cls(**kwargs)


def read(path):
    text = Path(path).read_text(encoding="utf-8")
    return {"text": text, "marker": json.loads(text)}


def selection_class(whisper, core):
    return getattr(whisper, "InstallSelection", None) or core.InstallSelection


def write_markers(spec):
    import install_llama_prebuilt as llama
    import install_node_prebuilt as node
    import install_whisper_prebuilt as whisper
    import prebuilt_core as core

    out = {}
    root = Path(tempfile.mkdtemp())

    llama_dir = root / "llama.cpp"
    llama_dir.mkdir()
    llama.write_prebuilt_metadata(
        llama_dir,
        requested_tag=spec["llama"]["requested_tag"],
        llama_tag=spec["llama"]["upstream_tag"],
        release_tag=spec["llama"]["release_tag"],
        choice=fill(llama.AssetChoice, spec["llama"]["choice"]),
        approved_checksums=fill(llama.ApprovedReleaseChecksums, spec["llama"]["checksums"]),
        prebuilt_fallback_used=False,
        backend_request=spec["llama"]["backend_request"],
    )
    out["llama"] = read(llama_dir / "UNSLOTH_PREBUILT_INFO.json")

    whisper_dir = root / "whisper.cpp"
    whisper_dir.mkdir()
    whisper.write_prebuilt_metadata(
        whisper_dir, fill(selection_class(whisper, core), spec["whisper"]["selection"])
    )
    out["whisper"] = read(whisper.metadata_path(whisper_dir))

    node_dir = root / "node"
    node_dir.mkdir()
    node.write_metadata(node_dir, **spec["node"]["metadata"])
    out["node"] = read(node.metadata_path(node_dir))
    return out


def read_markers(spec):
    import install_llama_prebuilt as llama
    import install_node_prebuilt as node
    import install_whisper_prebuilt as whisper
    import prebuilt_core as core

    out = {"readers": []}

    llama_dir = Path(spec["llama"]["install_dir"])
    llama_host = fill(llama.HostInfo, spec["llama"]["host"])
    llama_marker = llama.load_prebuilt_metadata(llama_dir)
    out["llama_marker_keys"] = sorted(llama_marker)
    out["llama_backend_request"] = llama.persisted_backend_request(llama_dir)
    out["readers"].append("llama.persisted_backend_request")
    for name in ("marker_backend",):
        reader = getattr(llama, name, None)
        if reader is not None:
            out["llama_backend"] = reader(llama_marker)
            out["readers"].append("llama." + name)
    for name in ("_install_tree_is_usable", "_kept_install_payload_is_healthy"):
        reader = getattr(llama, name, None)
        if reader is not None:
            out["llama" + name] = reader(llama_dir, llama_host)
            out["readers"].append("llama." + name)

    whisper_dir = Path(spec["whisper"]["install_dir"])
    whisper_host = fill(whisper.HostInfo, spec["whisper"]["host"])
    out["whisper_marker_keys"] = sorted(whisper.load_prebuilt_metadata(whisper_dir))
    out["whisper_matches"] = whisper.existing_install_matches(
        whisper_dir,
        whisper_host,
        fill(selection_class(whisper, core), spec["whisper"]["selection"]),
    )
    out["readers"].append("whisper.existing_install_matches")

    node_dir = Path(spec["node"]["install_dir"])
    node_host = fill(node.HostInfo, spec["node"]["host"])
    # The two spawns an old installer would do; stubbed so this stays offline and
    # hardware-free. What is under test is the marker reading either side of them.
    node.installed_node_version = lambda *a, **k: spec["node"]["version"]
    node.installed_npm_major = lambda *a, **k: spec["node"]["npm_major"]
    out["node_marker_keys"] = sorted(node.load_metadata(node_dir))
    out["node_matches"] = node.existing_install_matches(
        node_dir,
        node_host,
        version=spec["node"]["version"],
        expected_sha=spec["node"]["sha256"],
    )
    out["readers"].append("node.existing_install_matches")
    return out


def main():
    request = json.loads(sys.argv[1])
    handler = {"write_markers": write_markers, "read_markers": read_markers}[request["op"]]
    sys.stdout.write(SENTINEL + json.dumps(handler(request["spec"]), default=str) + "\n")


main()
'''


def _extract_legacy_tree(tag: str, destination: Path) -> "Path | None":
    """Extracts a released tag's installer modules and prebuilt package via git show; None if absent."""
    destination.mkdir(parents = True, exist_ok = True)
    wanted = [f"studio/{name}" for name in _LEGACY_MODULES]
    wanted += ["studio/backend/__init__.py", "studio/backend/utils/__init__.py"]
    # PYTHONPATH below is REPLACED; ls-tree so tags predating auth_safe.py resolve.
    optional = git("ls-tree", "-r", "--name-only", tag, "studio/backend/utils/auth_safe.py")
    if optional.returncode == 0:
        wanted += [
            line
            for line in optional.stdout.decode("utf-8", "replace").split()
            if line.endswith(".py")
        ]
    listing = git("ls-tree", "-r", "--name-only", tag, "studio/backend/utils/prebuilt")
    if listing.returncode != 0:
        return None
    wanted += [
        line for line in listing.stdout.decode("utf-8", "replace").split() if line.endswith(".py")
    ]
    for path in wanted:
        blob = git("show", f"{tag}:{path}")
        if blob.returncode != 0:
            return None
        target = destination / Path(path).relative_to("studio")
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(blob.stdout)
    return destination


class LegacyRunFailed(RuntimeError):
    pass


def _run_legacy(tree: Path, op: str, spec: dict) -> dict:
    """Run the driver against one legacy tree, with only that tree importable."""
    environment = dict(os.environ)
    # Replaced, not prepended, or legacy modules would import today's prebuilt_core.
    environment["PYTHONPATH"] = str(tree)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _LEGACY_DRIVER, json.dumps({"op": op, "spec": spec})],
        cwd = str(tree),
        env = environment,
        capture_output = True,
        text = True,
        timeout = 600,
        check = False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL) :])
    raise LegacyRunFailed(
        f"legacy {op} failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


LLAMA_REPO = "unslothai/llama.cpp"
LLAMA_UPSTREAM_TAG = "b10698"
LLAMA_RELEASE_TAG = "b10698-mix-67dfc8b"
LLAMA_ASSET = "app-b10698-mix-67dfc8b-linux-x64-vulkan.tar.gz"
LLAMA_GGML_TREE = "0034c6eb"
# GPU-less Vulkan box: host_profile is a pure function of this HostInfo.
LLAMA_CHOICE = {
    "repo": LLAMA_REPO,
    "tag": LLAMA_UPSTREAM_TAG,
    "name": LLAMA_ASSET,
    "url": f"https://example.invalid/{LLAMA_ASSET}",
    "source_label": "published",
    "expected_sha256": "d4" * 32,
    "install_kind": "linux-vulkan",
    "bundle_profile": "vulkan",
    "runtime_line": None,
    "coverage_class": None,
    "is_ready_bundle": True,
    "supported_sms": [],
    "mapped_targets": [],
    "runtime_name": None,
    "runtime_url": None,
    "runtime_sha256": None,
    "gfx_target": None,
}
LLAMA_CHECKSUMS = {
    "repo": LLAMA_REPO,
    "release_tag": LLAMA_RELEASE_TAG,
    "upstream_tag": LLAMA_UPSTREAM_TAG,
    "source_commit": "0" * 40,
    "ggml_tree": LLAMA_GGML_TREE,
    "artifacts": {},
}

WHISPER_REPO = "unslothai/whisper.cpp"
WHISPER_RELEASE_TAG = "v1.9.1-unsloth.17"
WHISPER_UPSTREAM_TAG = "v1.9.1"
WHISPER_ASSET = "whisper-v1.9.1-linux-x64-cpu.tar.gz"
WHISPER_SELECTION = {
    "published_repo": WHISPER_REPO,
    "release_tag": WHISPER_RELEASE_TAG,
    "upstream_tag": WHISPER_UPSTREAM_TAG,
    "source_commit": "1" * 40,
    "asset": WHISPER_ASSET,
    "asset_sha256": "ab" * 32,
    "backend": "cpu",
    "runtime_line": None,
    "coverage": {},
    "studio_protocol": "inference/multipart-v1",
}
# platform_os/platform_arch are new here; a legacy tag's InstallSelection drops them.
CURRENT_WHISPER_SELECTION = {**WHISPER_SELECTION, "platform_os": "linux", "platform_arch": "x64"}

NODE_VERSION = "22.20.0"
NODE_ASSET = "node-v22.20.0-linux-x64.tar.xz"
NODE_SHA256 = "ef" * 32


def _fill(cls, wanted: dict):
    """Construct *cls* from the fields it declares, as the legacy driver does."""
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in wanted:
            kwargs[field.name] = wanted[field.name]
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            kwargs[field.name] = None
    return cls(**kwargs)


LINUX = llama_host(ILP.HostInfo)
# The legacy subprocess builds its own HostInfo by field name from this dict.
LLAMA_HOST_KWARGS = {
    "system": "Linux",
    "machine": "x86_64",
    "is_windows": False,
    "is_linux": True,
    "is_macos": False,
    "is_x86_64": True,
    "is_arm64": False,
    "nvidia_smi": None,
    "driver_cuda_version": None,
    "compute_caps": [],
    "visible_cuda_devices": None,
    "has_physical_nvidia": False,
    "has_usable_nvidia": False,
}
WHISPER_HOST_KWARGS = {
    "system": "Linux",
    "machine": "x86_64",
    "whisper_os": "linux",
    "whisper_arch": "x64",
    "archive_ext": ".tar.gz",
    "is_windows": False,
    "is_macos": False,
    "is_apple_silicon": False,
}
NODE_HOST_KWARGS = {
    "system": "Linux",
    "machine": "x86_64",
    "node_os": "linux",
    "node_arch": "x64",
    "archive_ext": ".tar.gz",
    "is_windows": False,
}
WHISPER_HOST = _fill(WSP.HostInfo, WHISPER_HOST_KWARGS)
NODE_HOST = _fill(NDP.HostInfo, NODE_HOST_KWARGS)

MARKER_SPEC = {
    "llama": {
        "requested_tag": "latest",
        "upstream_tag": LLAMA_UPSTREAM_TAG,
        "release_tag": LLAMA_RELEASE_TAG,
        "backend_request": "auto",
        "choice": LLAMA_CHOICE,
        "checksums": LLAMA_CHECKSUMS,
    },
    "whisper": {"selection": WHISPER_SELECTION},
    "node": {"metadata": {"version": NODE_VERSION, "asset": NODE_ASSET, "sha256": NODE_SHA256}},
}


@pytest.fixture(scope = "session")
def legacy_markers(tmp_path_factory) -> dict:
    """One marker per component, per released tag, written by that tag's own code."""
    if git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout, so the released tags cannot be read")
    root = tmp_path_factory.mktemp("pr10648-legacy")
    produced: dict = {}
    for tag in LEGACY_TAGS:
        tree = _extract_legacy_tree(tag, root / tag)
        if tree is None:
            produced[tag] = {"error": f"{tag} does not carry the installer modules"}
            continue
        try:
            produced[tag] = {
                "tree": tree,
                "markers": _run_legacy(tree, "write_markers", MARKER_SPEC),
            }
        except (LegacyRunFailed, subprocess.SubprocessError) as exc:
            produced[tag] = {"error": str(exc)}
    if not any("markers" in entry for entry in produced.values()):
        pytest.skip("no released tag could be loaded standalone; see the per-tag skips")
    return produced


def _legacy(legacy_markers: dict, tag: str) -> dict:
    entry = legacy_markers[tag]
    if "markers" not in entry:
        pytest.skip(f"{tag} could not be loaded standalone: {entry['error']}")
    return entry


@pytest.fixture(autouse = True)
def _offline(monkeypatch):
    """Forbids network lookups and clears the full-check hatch, which would force every fast path False."""

    def refuse(*args, **kwargs):
        raise AssertionError("the marker fast path must not reach the network")

    monkeypatch.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising = False)
    for module in (ILP, WSP.llama, CORE):
        for name in (
            "_download_host_latest_release_tag",
            "_api_newest_release_tag",
            "github_releases",
            "fetch_json",
            "urlopen",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, refuse)


def _llama_install(root: Path, marker):
    """A healthy published Vulkan tree, with *marker* written verbatim."""
    return CORPUS.build_install(root, host = LINUX, marker = marker, payload_backend = "vulkan")


def _whisper_install(root: Path, marker_text: "str | None"):
    install_dir = root / "whisper.cpp"
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    server = bin_dir / "whisper-server"
    server.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    os.chmod(server, 0o755)
    if marker_text is not None:
        WSP.metadata_path(install_dir).write_text(marker_text, encoding = "utf-8")
    return install_dir


def _node_install(root: Path, marker_text: "str | None"):
    install_dir = root / "node"
    node_binary = NDP.node_binary_path(install_dir, NODE_HOST)
    node_binary.parent.mkdir(parents = True)
    node_binary.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    os.chmod(node_binary, 0o755)
    npm_cli = NDP.npm_cli_path(install_dir, NODE_HOST)
    npm_cli.parent.mkdir(parents = True)
    npm_cli.write_text("// npm-cli\n", encoding = "utf-8")
    if marker_text is not None:
        NDP.metadata_path(install_dir).write_text(marker_text, encoding = "utf-8")
    return install_dir


def _llama_route(
    host = LINUX,
    published_repo = LLAMA_REPO,
    release_tag = LLAMA_RELEASE_TAG,
):
    return ILP.BackendRoute(
        backend = "auto",
        host = host,
        published_repo = published_repo,
        published_release_tag = release_tag,
        persist_llama_backend = None,
        persist_rocm_gfx = None,
    )


def _llama_fast_path(
    install_dir: Path,
    host = LINUX,
    backend_request = "auto",
    published_repo = LLAMA_REPO,
    release_tag = LLAMA_RELEASE_TAG,
) -> bool:
    """Pre-check with the release pinned, so no release lookup runs and only marker and tree decide."""
    return ILP.existing_install_current_without_plan(
        install_dir,
        llama_tag = "latest",
        published_repo = published_repo,
        published_release_tag = release_tag,
        backend_request = backend_request,
        force_cpu = False,
        route = _llama_route(host, published_repo, release_tag),
    )


def _current_llama_install(root: Path):
    """A tree whose marker this very code wrote: the fast path's own best case."""
    install_dir = _llama_install(root, marker = None)
    ILP.write_prebuilt_metadata(
        install_dir,
        host = LINUX,
        requested_tag = "latest",
        llama_tag = LLAMA_UPSTREAM_TAG,
        release_tag = LLAMA_RELEASE_TAG,
        choice = _llama_choice(),
        approved_checksums = _fill(ILP.ApprovedReleaseChecksums, LLAMA_CHECKSUMS),
        prebuilt_fallback_used = False,
        backend_request = "auto",
    )
    return install_dir


def _rewrite_marker(marker_path: Path, marker: dict) -> None:
    marker_path.write_text(json.dumps(marker, indent = 2) + "\n", encoding = "utf-8")


def _whisper_fast_path(install_dir: Path) -> bool:
    return WSP.existing_install_current_without_plan(
        install_dir,
        WHISPER_HOST,
        whisper_tag = "latest",
        published_repo = WHISPER_REPO,
        published_release_tag = WHISPER_RELEASE_TAG,
        requested_backend = "cpu",
    )


def _llama_choice():
    return _fill(ILP.AssetChoice, LLAMA_CHOICE)


def _whisper_selection():
    return _fill(CORE.InstallSelection, CURRENT_WHISPER_SELECTION)


@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_llama_marker_from_a_released_unsloth_never_takes_the_fast_path(
    tmp_path, legacy_markers, tag
):
    """Old llama markers lack host_profile, runtime_files and runtime_sha256; absence is not agreement."""
    marker = _legacy(legacy_markers, tag)["markers"]["llama"]
    assert "host_profile" not in marker["marker"], tag
    assert "runtime_files" not in marker["marker"], tag
    install_dir = _llama_install(tmp_path, marker["text"])
    assert _llama_fast_path(install_dir) is False


@pytest.mark.parametrize(
    ("name", "marker", "backend"),
    CORPUS.ALL_SHAPES,
    ids = [shape[0] for shape in CORPUS.ALL_SHAPES],
)
def test_no_shipped_llama_marker_shape_reaches_the_fast_path(tmp_path, name, marker, backend):
    """Shipped llama marker shapes are healthy but lack the new keys, so the fast path must decline them."""
    install_dir = CORPUS.build_install(tmp_path, host = LINUX, marker = marker, payload_backend = backend)
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True, name
    # Same release and repo, so the verdict turns on missing evidence, not a release mismatch.
    assert (
        _llama_fast_path(
            install_dir,
            published_repo = marker.get("published_repo") or LLAMA_REPO,
            release_tag = marker.get("release_tag") or LLAMA_RELEASE_TAG,
        )
        is False
    ), name


@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_whisper_marker_from_a_released_unsloth_never_takes_the_fast_path(
    tmp_path, legacy_markers, tag
):
    """Old whisper markers lack fingerprint_coverage, so they recompute to None and take the full path."""
    marker = _legacy(legacy_markers, tag)["markers"]["whisper"]
    assert "fingerprint_coverage" not in marker["marker"], tag
    assert CORE.marker_install_fingerprint(marker["marker"]) is None, tag
    install_dir = _whisper_install(tmp_path, marker["text"])
    assert _whisper_fast_path(install_dir) is False


@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_node_marker_from_a_released_unsloth_never_skips_the_version_probe(
    tmp_path, legacy_markers, tag
):
    """Old node markers lack node_version_checked, so the fast path declines them and node is spawned."""
    marker = _legacy(legacy_markers, tag)["markers"]["node"]
    assert "node_version_checked" not in marker["marker"], tag
    install_dir = _node_install(tmp_path, marker["text"])
    meta = NDP.load_metadata(install_dir)
    assert NDP._recorded_runtime_matches(install_dir, NODE_HOST, meta, NODE_VERSION) is False


@pytest.mark.parametrize("key", ["host_profile", "runtime_files", "runtime_sha256"])
def test_a_llama_marker_missing_one_new_key_is_not_read_as_agreement(tmp_path, key):
    """Each new llama marker key, missing alone, must force the full path rather than read as agreement."""
    install_dir = _current_llama_install(tmp_path)
    assert _llama_fast_path(install_dir) is True

    marker = ILP.load_prebuilt_metadata(install_dir)
    assert key in marker
    marker.pop(key)
    _rewrite_marker(install_dir / "UNSLOTH_PREBUILT_INFO.json", marker)
    assert _llama_fast_path(install_dir) is False


def test_a_whisper_marker_missing_fingerprint_coverage_is_not_read_as_agreement(tmp_path):
    """Without fingerprint_coverage the fingerprint cannot be recomputed, so an edited release_tag
    passes."""
    install_dir = _whisper_install(tmp_path, marker_text = None)
    WSP.write_prebuilt_metadata(install_dir, _whisper_selection())
    assert _whisper_fast_path(install_dir) is True

    marker = WSP.load_prebuilt_metadata(install_dir)
    assert isinstance(marker.pop("fingerprint_coverage"), dict)
    _rewrite_marker(WSP.metadata_path(install_dir), marker)
    assert _whisper_fast_path(install_dir) is False


def test_a_node_marker_missing_node_version_checked_is_not_read_as_agreement(tmp_path):
    """node_binary and npm_cli present does not imply node_version_checked; the spawn still has to
    happen."""
    install_dir = _node_install(tmp_path, marker_text = None)
    NDP.write_metadata(install_dir, version = NODE_VERSION, asset = NODE_ASSET, sha256 = NODE_SHA256)
    NDP.record_runtime_verification(
        install_dir, NODE_HOST, version = NODE_VERSION, npm_major = NDP.NPM_MIN_MAJOR
    )
    marker = NDP.load_metadata(install_dir)
    assert NDP._recorded_runtime_matches(install_dir, NODE_HOST, marker, NODE_VERSION) is True

    marker.pop("node_version_checked")
    assert NDP._recorded_runtime_matches(install_dir, NODE_HOST, marker, NODE_VERSION) is False


def test_the_full_check_env_var_puts_a_current_install_back_on_the_slow_path(tmp_path, monkeypatch):
    """Full-check variable sends llama and whisper to the slow path, so support gives one instruction."""
    llama_dir = _current_llama_install(tmp_path / "llama")
    whisper_dir = _whisper_install(tmp_path / "whisper", marker_text = None)
    WSP.write_prebuilt_metadata(whisper_dir, _whisper_selection())
    assert _llama_fast_path(llama_dir) is True
    assert _whisper_fast_path(whisper_dir) is True

    monkeypatch.setenv("UNSLOTH_PREBUILT_FULL_CHECK", "1")
    assert _llama_fast_path(llama_dir) is False
    assert _whisper_fast_path(whisper_dir) is False


def _newest_loadable_tag(legacy_markers: dict) -> str:
    for tag in reversed(LEGACY_TAGS):
        if "markers" in legacy_markers[tag]:
            return tag
    pytest.skip("no released tag could be loaded standalone")


def test_an_old_llama_install_pays_the_full_path_once_and_is_fast_afterwards(
    tmp_path, legacy_markers
):
    """Full path backfills via sync_marker_selection, so old installs are fast after one update."""
    tag = _newest_loadable_tag(legacy_markers)
    marker = _legacy(legacy_markers, tag)["markers"]["llama"]
    install_dir = _llama_install(tmp_path, marker["text"])

    assert _llama_fast_path(install_dir) is False

    ILP.sync_marker_selection(
        install_dir,
        choice = _llama_choice(),
        backend_request = "auto",
        ggml_tree = LLAMA_GGML_TREE,
        host = LINUX,
        prebuilt_fallback_used = False,
    )
    backfilled = ILP.load_prebuilt_metadata(install_dir)
    assert isinstance(backfilled.get("host_profile"), dict)
    assert backfilled.get("runtime_files")
    # The released fingerprint must still recompute, or the backfill has rewritten history.
    assert backfilled["install_fingerprint"] == marker["marker"]["install_fingerprint"]
    assert ILP._marker_install_fingerprint(backfilled) == backfilled["install_fingerprint"]

    assert _llama_fast_path(install_dir) is True


def test_the_backfilled_llama_marker_still_refuses_a_tree_whose_bytes_moved(
    tmp_path, legacy_markers
):
    """A backfilled runtime_files record must still fail closed when a recorded binary's bytes change."""
    tag = _newest_loadable_tag(legacy_markers)
    marker = _legacy(legacy_markers, tag)["markers"]["llama"]
    install_dir = _llama_install(tmp_path, marker["text"])
    ILP.sync_marker_selection(
        install_dir,
        choice = _llama_choice(),
        backend_request = "auto",
        ggml_tree = LLAMA_GGML_TREE,
        host = LINUX,
        prebuilt_fallback_used = False,
    )
    assert _llama_fast_path(install_dir) is True

    server = install_dir / "build" / "bin" / "llama-server"
    server.write_text("#!/bin/sh\nexit 0\n# a different build\n", encoding = "utf-8")
    os.chmod(server, 0o755)
    assert _llama_fast_path(install_dir) is False


def test_an_old_whisper_install_pays_the_full_path_once_and_is_fast_afterwards(
    tmp_path, legacy_markers
):
    """Old whisper markers take the full path once, which writes fingerprint_coverage and os/arch tokens."""
    tag = _newest_loadable_tag(legacy_markers)
    marker = _legacy(legacy_markers, tag)["markers"]["whisper"]
    install_dir = _whisper_install(tmp_path, marker["text"])
    selection = _whisper_selection()
    # If the formula moved, the settle silently declines and old installs stay on the full path.
    assert selection.fingerprint() == marker["marker"]["install_fingerprint"]

    assert _whisper_fast_path(install_dir) is False

    CORE._settle_kept_install(WSP._OPS, install_dir, WHISPER_HOST, selection, locked = True)
    settled = WSP.load_prebuilt_metadata(install_dir)
    assert isinstance(settled.get("fingerprint_coverage"), dict)
    assert settled["install_fingerprint"] == marker["marker"]["install_fingerprint"]

    assert _whisper_fast_path(install_dir) is True


def test_an_old_node_install_spawns_node_once_and_never_again(
    tmp_path, legacy_markers, monkeypatch
):
    """Node's version is probed once after upgrade and recorded; the npm probe still runs on every
    update."""
    tag = _newest_loadable_tag(legacy_markers)
    marker = _legacy(legacy_markers, tag)["markers"]["node"]
    install_dir = _node_install(tmp_path, marker["text"])

    spawns: list[str] = []

    def counted_version(*args, **kwargs):
        spawns.append("node -v")
        return NODE_VERSION

    monkeypatch.setattr(NDP, "installed_node_version", counted_version)
    monkeypatch.setattr(NDP, "installed_npm_major", lambda *a, **k: NDP.NPM_MIN_MAJOR)

    assert (
        NDP.existing_install_matches(
            install_dir, NODE_HOST, version = NODE_VERSION, expected_sha = NODE_SHA256
        )
        is True
    )
    assert spawns == ["node -v"]

    recorded = NDP.load_metadata(install_dir)
    assert recorded.get("node_version_checked") == NODE_VERSION

    assert (
        NDP.existing_install_matches(
            install_dir, NODE_HOST, version = NODE_VERSION, expected_sha = NODE_SHA256
        )
        is True
    )
    assert spawns == ["node -v"]


def test_a_replaced_node_binary_is_probed_again_after_the_record_was_written(
    tmp_path, legacy_markers, monkeypatch
):
    """A replaced node binary must be re-probed; a recorded version must not outlive its bytes."""
    tag = _newest_loadable_tag(legacy_markers)
    marker = _legacy(legacy_markers, tag)["markers"]["node"]
    install_dir = _node_install(tmp_path, marker["text"])
    monkeypatch.setattr(NDP, "installed_node_version", lambda *a, **k: NODE_VERSION)
    monkeypatch.setattr(NDP, "installed_npm_major", lambda *a, **k: NDP.NPM_MIN_MAJOR)
    assert (
        NDP.existing_install_matches(
            install_dir, NODE_HOST, version = NODE_VERSION, expected_sha = NODE_SHA256
        )
        is True
    )

    node_binary = NDP.node_binary_path(install_dir, NODE_HOST)
    node_binary.write_text("#!/bin/sh\nexit 0\n# another build\n", encoding = "utf-8")
    meta = NDP.load_metadata(install_dir)
    assert NDP._recorded_runtime_matches(install_dir, NODE_HOST, meta, NODE_VERSION) is False


@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_marker_written_today_is_still_read_by_every_released_unsloth(
    tmp_path, legacy_markers, tag
):
    """Released installers must ignore the new additive marker keys and keep the install, not re-
    download."""
    entry = _legacy(legacy_markers, tag)

    llama_dir = _llama_install(tmp_path / "current", marker = None)
    ILP.write_prebuilt_metadata(
        llama_dir,
        host = LINUX,
        requested_tag = "latest",
        llama_tag = LLAMA_UPSTREAM_TAG,
        release_tag = LLAMA_RELEASE_TAG,
        choice = _llama_choice(),
        approved_checksums = _fill(ILP.ApprovedReleaseChecksums, LLAMA_CHECKSUMS),
        prebuilt_fallback_used = False,
        backend_request = "auto",
    )

    whisper_dir = _whisper_install(tmp_path / "current", marker_text = None)
    WSP.write_prebuilt_metadata(whisper_dir, _whisper_selection())

    node_dir = _node_install(tmp_path / "current", marker_text = None)
    NDP.write_metadata(node_dir, version = NODE_VERSION, asset = NODE_ASSET, sha256 = NODE_SHA256)
    NDP.record_runtime_verification(
        node_dir, NODE_HOST, version = NODE_VERSION, npm_major = NDP.NPM_MIN_MAJOR
    )

    # The keys that did not exist when these tags shipped, so the run below is a real test.
    assert "host_profile" in ILP.load_prebuilt_metadata(llama_dir)
    assert "fingerprint_coverage" in WSP.load_prebuilt_metadata(whisper_dir)
    assert "node_version_checked" in NDP.load_metadata(node_dir)

    result = _run_legacy(
        entry["tree"],
        "read_markers",
        {
            "llama": {"install_dir": str(llama_dir), "host": LLAMA_HOST_KWARGS},
            "whisper": {
                "install_dir": str(whisper_dir),
                "host": WHISPER_HOST_KWARGS,
                "selection": WHISPER_SELECTION,
            },
            "node": {
                "install_dir": str(node_dir),
                "host": NODE_HOST_KWARGS,
                "version": NODE_VERSION,
                "sha256": NODE_SHA256,
                "npm_major": NDP.NPM_MIN_MAJOR,
            },
        },
    )

    assert result["llama_backend_request"] == "auto"
    assert result.get("llama_backend", "vulkan") == "vulkan"
    # Only the tags that HAVE these readers are asked; the older ones report neither.
    assert result.get("llama_kept_install_payload_is_healthy", True) is True
    assert result.get("llama_install_tree_is_usable", True) is True
    assert result["whisper_matches"] is True
    assert result["node_matches"] is True

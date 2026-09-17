# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Old installs meeting the marker fast paths, and new markers meeting old installers.

``studio update`` now skips re-validating a prebuilt llama.cpp / whisper.cpp / Node tree
when the marker on disk carries enough evidence to say the install is already the one this
run would produce. Every user upgrading into that code has a marker written by an Unsloth
that recorded none of it, so the two directions this pins are:

  * BACKWARDS. A marker written by a released, pre-fast-path Unsloth must ALWAYS fall
    through to the full path -- the fast path may not read an absent key as agreement --
    and it must fall through exactly ONCE, because the full path backfills the missing
    evidence onto the marker it kept. An old install that never backfills pays the full
    re-validation (13-63 s on macOS) on every update forever, which is the bug the fast
    path exists to fix; an old install that is WRONGLY trusted keeps a tree this run
    would have replaced, which is worse.
  * FORWARDS. A user who downgrades, or who runs an older Studio out of another
    checkout, hands a released installer a marker full of keys it has never heard of.
    The new keys are additive, so the old reader must ignore them and still keep the
    install rather than refuse and re-download it.

The old markers here are not hand-written shapes. The released tags in this repository
are checked out read-only with ``git show``, loaded in a subprocess whose ``PYTHONPATH``
holds nothing but that tag's own modules, and asked to write a marker with their OWN
writer -- so what the current readers are fed is the exact bytes that shipped. The
hand-written corpus is still covered: ``ALL_SHAPES`` from
``test_keep_install_backcompat_9979`` is imported rather than re-derived, so the twelve
historical shapes and one real captured install are asserted here too.

Linux-only in what it touches: platforms are simulated through ``HostInfo`` (never
through ``os.name``, which changes pathlib underneath the install trees), no network,
no GPU. The subprocesses are read-only against the repository and write only under
``tmp_path``.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    # The canonical corpus lives beside this file; pytest's prepend import mode puts this
    # directory on sys.path too, but not necessarily before this module body runs.
    sys.path.insert(0, str(_TEST_DIR))

from _pr10648_helpers import PACKAGE_ROOT, git, llama_host  # noqa: E402
from _pr10648_helpers import load_studio_module as _load  # noqa: E402

import test_keep_install_backcompat_9979 as CORPUS  # noqa: E402

ILP = _load("studio_install_llama_prebuilt_pr10648_legacy", "install_llama_prebuilt.py")
WSP = _load("studio_install_whisper_prebuilt_pr10648_legacy", "install_whisper_prebuilt.py")
NDP = _load("studio_install_node_prebuilt_pr10648_legacy", "install_node_prebuilt.py")
# The core instance whisper actually calls into, not a second copy of it.
CORE = WSP.core


# ── The released tags whose markers are replayed here ──
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

# Runs inside the legacy subprocess. Everything it needs is introspected from the tag's own
# dataclasses, so a tag whose fields differ still gets a marker its own writer produced.
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
    """``git show`` a released tag's installers into *destination*, or None if it is absent.

    The four installer modules plus the ``backend.utils.prebuilt`` package they import at
    module scope -- roughly 600 KB, against 46 MB for the whole ``studio`` tree.
    """
    destination.mkdir(parents = True, exist_ok = True)
    wanted = [f"studio/{name}" for name in _LEGACY_MODULES]
    wanted += ["studio/backend/__init__.py", "studio/backend/utils/__init__.py"]
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
    # Replaced, not prepended: the caller's PYTHONPATH points at the CURRENT studio/, and
    # inheriting it would let the legacy modules import today's prebuilt_core.
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


# ── The install this scenario describes, in each component's dialect ──
LLAMA_REPO = "unslothai/llama.cpp"
LLAMA_UPSTREAM_TAG = "b10698"
LLAMA_RELEASE_TAG = "b10698-mix-67dfc8b"
LLAMA_ASSET = "app-b10698-mix-67dfc8b-linux-x64-vulkan.tar.gz"
LLAMA_GGML_TREE = "0034c6eb"
# Vulkan on a GPU-less Linux box: no CUDA runtime scan, no ROCm probe and no torch
# preference read, so host_profile is a pure function of the HostInfo constructed here.
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
# Spelled out again as a plain dict: the legacy subprocess builds its own tag's HostInfo by
# field name, so this is data crossing a process boundary rather than a second factory.
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
    """No fast-path assertion here may be bought with a network call.

    Every scenario pins the release tag, which is the shape of the check that resolves
    without any lookup; anything reaching a resolver is a test that stopped testing the
    fast path. Also clears the full-check escape hatch, which would turn every fast path
    below into an unconditional False and make the suite pass vacuously.
    """

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
    """The pre-check as ``install_prebuilt`` calls it, with the release pinned.

    A pinned release is what ``_expected_release_tag_without_plan`` answers outright, so
    nothing here resolves anything: the verdict is decided purely by the marker and the
    tree, which is what these tests are about.
    """
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


# ── Backwards compatibility: an old marker is never trusted ──
@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_llama_marker_from_a_released_unsloth_never_takes_the_fast_path(
    tmp_path, legacy_markers, tag
):
    """Upgrading into this code with a llama.cpp install from any shipped release.

    The marker was written by that release's own ``write_prebuilt_metadata`` and records
    no ``host_profile``, no ``runtime_files`` and no ``runtime_sha256``. Reading any of
    those absences as agreement would keep a tree this run never verified.
    """
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
    """The twelve marker shapes that have shipped, plus one captured from a real install.

    ``UNSLOTH_PREBUILT_INFO.json`` is append-only with no version field, so "the key is
    missing" is the only signal an old shape gives. Each of these describes a tree that is
    perfectly healthy -- the full path keeps it -- so the fast path refusing them is the
    ONLY thing forcing the one re-validation that backfills the new evidence.
    """
    install_dir = CORPUS.build_install(tmp_path, host = LINUX, marker = marker, payload_backend = backend)
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True, name
    # The shape's own release and repo, so the tag comparison AGREES and the verdict turns
    # on the evidence the shape does not carry rather than on a release mismatch.
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
    """Upgrading into this code with a whisper.cpp install from any shipped release.

    ``fingerprint_coverage`` is what lets a no-network check recompute the fingerprint and
    tell a whole marker from a ``release_tag`` edited over an old binary. No released
    marker carries it, so all of them must recompute to None and take the full path.
    """
    marker = _legacy(legacy_markers, tag)["markers"]["whisper"]
    assert "fingerprint_coverage" not in marker["marker"], tag
    assert CORE.marker_install_fingerprint(marker["marker"]) is None, tag
    install_dir = _whisper_install(tmp_path, marker["text"])
    assert _whisper_fast_path(install_dir) is False


@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_node_marker_from_a_released_unsloth_never_skips_the_version_probe(
    tmp_path, legacy_markers, tag
):
    """Upgrading into this code with a managed Node runtime from any shipped release.

    ``node_version_checked`` is the record that stands in for spawning a 110 MB
    interpreter. No released marker has one, so the recorded-runtime fast path must
    decline every one of them and let the spawn happen.
    """
    marker = _legacy(legacy_markers, tag)["markers"]["node"]
    assert "node_version_checked" not in marker["marker"], tag
    install_dir = _node_install(tmp_path, marker["text"])
    meta = NDP.load_metadata(install_dir)
    assert NDP._recorded_runtime_matches(install_dir, NODE_HOST, meta, NODE_VERSION) is False


# ── Each new key on its own is what an old marker is missing ──
@pytest.mark.parametrize("key", ["host_profile", "runtime_files", "runtime_sha256"])
def test_a_llama_marker_missing_one_new_key_is_not_read_as_agreement(tmp_path, key):
    """Isolates the guard the released markers above rely on, one key at a time.

    Those markers are missing all three at once, so on their own they cannot show WHICH
    absence is doing the work -- and a fast path that only refused them because of, say,
    the fingerprint would silently start trusting a hand-edited marker the day a backfill
    added that one key. Start from a marker this code wrote, which the fast path accepts,
    and remove exactly one key: each must be enough on its own to send the run back to
    the full path.
    """
    install_dir = _current_llama_install(tmp_path)
    assert _llama_fast_path(install_dir) is True

    marker = ILP.load_prebuilt_metadata(install_dir)
    assert key in marker
    marker.pop(key)
    _rewrite_marker(install_dir / "UNSLOTH_PREBUILT_INFO.json", marker)
    assert _llama_fast_path(install_dir) is False


def test_a_whisper_marker_missing_fingerprint_coverage_is_not_read_as_agreement(tmp_path):
    """The one key that decides whisper's fast path, isolated the same way.

    Without it the fingerprint cannot be recomputed, so a ``release_tag`` edited over an
    old binary would read as current. Every pre-PR whisper marker lacks it.
    """
    install_dir = _whisper_install(tmp_path, marker_text = None)
    WSP.write_prebuilt_metadata(install_dir, _whisper_selection())
    assert _whisper_fast_path(install_dir) is True

    marker = WSP.load_prebuilt_metadata(install_dir)
    assert isinstance(marker.pop("fingerprint_coverage"), dict)
    _rewrite_marker(WSP.metadata_path(install_dir), marker)
    assert _whisper_fast_path(install_dir) is False


def test_a_node_marker_missing_node_version_checked_is_not_read_as_agreement(tmp_path):
    """The record that stands in for the node spawn must not be inferable from anything else.

    ``node_binary`` and ``npm_cli`` can be present (an install this code wrote) while the
    version record is not, and the spawn still has to happen.
    """
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
    """The workaround a user needs when a skip is wrong for reasons nothing on disk shows.

    Both components gate on the same variable, so a support answer is one instruction and
    not two. Asserted for llama and whisper together for that reason.
    """
    llama_dir = _current_llama_install(tmp_path / "llama")
    whisper_dir = _whisper_install(tmp_path / "whisper", marker_text = None)
    WSP.write_prebuilt_metadata(whisper_dir, _whisper_selection())
    assert _llama_fast_path(llama_dir) is True
    assert _whisper_fast_path(whisper_dir) is True

    monkeypatch.setenv("UNSLOTH_PREBUILT_FULL_CHECK", "1")
    assert _llama_fast_path(llama_dir) is False
    assert _whisper_fast_path(whisper_dir) is False


# ── Backwards compatibility: the full path is paid exactly once ──
def _newest_loadable_tag(legacy_markers: dict) -> str:
    for tag in reversed(LEGACY_TAGS):
        if "markers" in legacy_markers[tag]:
            return tag
    pytest.skip("no released tag could be loaded standalone")


def test_an_old_llama_install_pays_the_full_path_once_and_is_fast_afterwards(
    tmp_path, legacy_markers
):
    """The property every existing user gets: update once slowly, then quickly forever.

    The full path reuses the bundle it found and calls ``sync_marker_selection``, which is
    the only place that catches an old marker up to the evidence the no-network check
    needs. Without the backfill an old install would re-validate on every single update --
    the very cost the fast path was added to remove -- and nothing else in the tree would
    look wrong.
    """
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
    # The fingerprint the released installer wrote must still be the one this code
    # recomputes from the marker's own fields, or the backfill has rewritten history.
    assert backfilled["install_fingerprint"] == marker["marker"]["install_fingerprint"]
    assert ILP._marker_install_fingerprint(backfilled) == backfilled["install_fingerprint"]

    assert _llama_fast_path(install_dir) is True


def test_the_backfilled_llama_marker_still_refuses_a_tree_whose_bytes_moved(
    tmp_path, legacy_markers
):
    """The fast path that an old install earns must still be a real check.

    ``runtime_files`` replaces actually starting llama-server, so a backfilled marker that
    answered True for any tree would have swapped a 5 s probe for no probe at all. Rewrite
    one recorded binary and the same call has to fail closed.
    """
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
    """The same once-only cost for dictation: whisper.cpp installs from any shipped release.

    The full path keeps the bundle and settles the marker, which is where
    ``fingerprint_coverage`` (and the os/arch tokens) are written onto a marker that
    predates them.
    """
    tag = _newest_loadable_tag(legacy_markers)
    marker = _legacy(legacy_markers, tag)["markers"]["whisper"]
    install_dir = _whisper_install(tmp_path, marker["text"])
    selection = _whisper_selection()
    # If this ever fails the fingerprint formula itself moved, and the settle below would
    # silently decline -- leaving every old install on the full path forever.
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
    """``node -v`` on a 110 MB runtime, on every update, to re-derive a constant.

    The first call after upgrading has to pay it, because the released marker records
    nothing about the binary; the second must not. The npm probe is deliberately still
    paid each time -- npm-cli.js only bootstraps thousands of other files -- so only the
    node spawn is counted here.
    """
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
    """The record earned above must not outlive the bytes it describes.

    A repaired or hand-swapped node in an existing install has to be re-probed, or the
    fast path would keep reporting a version nothing on disk still reports.
    """
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


# ── Forwards compatibility: a new marker read by a released installer ──
@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_a_marker_written_today_is_still_read_by_every_released_unsloth(
    tmp_path, legacy_markers, tag
):
    """Downgrading, or running an older Studio from a second checkout, over one install.

    The fast path's keys are additive, so a released installer must ignore them and still
    keep the install. If an old reader instead refuses, a user who ever opens an older
    Studio gets a full re-download of llama.cpp, whisper.cpp and Node -- and gets it back
    again the next time they open the new one.
    """
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
